"""Synthetic data-boundary checks; no source graph access or model training."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import womd_pilot_data as pilot


def rows_fixture():
    return [dict(status='BUILT', scenario_id=f'scene-{shard:02d}-{scene:02d}',
                 source_shard=f'training_20s.tfrecord-{shard:05d}-of-00010',
                 eligible_class_0=str(scene), eligible_class_1=str(10 - scene))
            for shard in range(10) for scene in range(10)]


def part_fixture(sid='fixture', value=0.):
    y = np.asarray(([0, 1] * 8) + [-1] * 4, dtype=np.int64)
    count = len(y)
    return dict(scenario_id=sid, source_shard='fixture-shard',
                x=np.full((count, 784), value, dtype=np.float32), y=y,
                eligible=y >= 0, edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
                canonical_pairs=np.column_stack((np.arange(count), np.arange(count) + 1)),
                canonical_pair_track_ids=np.asarray([[str(i), str(i + 1)] for i in range(count)]),
                motion_regime=np.arange(count) % 2,
                message_passing_pair_mask=np.ones(count, dtype=bool))


def archive_bytes(arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def write_provenance(root, **plan_overrides):
    plan = dict(source_dataset_split='training_20s', dataset='Waymo Open Motion Dataset',
                test_arrays_loaded=False, **plan_overrides)
    (root / 'plan.json').write_text(json.dumps(plan))
    plan_hash = pilot.sha256(root / 'plan.json')
    (root / 'registration.json').write_text(json.dumps(dict(plan_sha256=plan_hash)))
    summary = dict(plan_sha256=plan_hash, test_arrays_loaded=False,
                   manifest_sha256=pilot.sha256(root / 'manifest.csv'))
    (root / 'summary.json').write_text(json.dumps(summary))
    completion = dict(plan_sha256=plan_hash, summary_sha256=pilot.sha256(root / 'summary.json'),
                      status='GRAPH_BUILD_COMPLETE_SPLIT_REVIEW_REQUIRED')
    (root / 'completion.json').write_text(json.dumps(completion))


class WOMDPilotDataTests(unittest.TestCase):
    def test_provenance_requires_registered_hash_chain_and_training_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.csv').write_text('scenario_id,status\n')
            write_provenance(root)
            original = {path.name: path.read_bytes() for path in root.iterdir()}
            self.assertEqual(set(pilot.validate_source(root)),
                             {'plan.json', 'registration.json', 'completion.json',
                              'summary.json', 'manifest.csv'})
            for filename in ('plan.json', 'summary.json', 'manifest.csv'):
                with self.subTest(tampered=filename):
                    path = root / filename
                    path.write_bytes(original[filename] + b' ')
                    with self.assertRaisesRegex(ValueError, 'Unverified WOMD'):
                        pilot.validate_source(root)
                    path.write_bytes(original[filename])
            for filename, field in (('registration.json', 'plan_sha256'),
                                    ('completion.json', 'summary_sha256')):
                changed = json.loads(original[filename])
                changed[field] = '0' * 64
                (root / filename).write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'Unverified WOMD'):
                    pilot.validate_source(root)
                (root / filename).write_bytes(original[filename])
            plan = json.loads((root / 'plan.json').read_text())
            plan['source_dataset_split'] = 'validation'
            (root / 'plan.json').write_text(json.dumps(plan))
            new_hash = pilot.sha256(root / 'plan.json')
            for filename in ('registration.json', 'summary.json', 'completion.json'):
                value = json.loads((root / filename).read_text())
                value['plan_sha256'] = new_hash
                if filename == 'completion.json':
                    value['summary_sha256'] = pilot.sha256(root / 'summary.json')
                (root / filename).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Unverified WOMD'):
                pilot.validate_source(root)

    def test_scene_split_is_label_independent_order_stable_and_shard_disjoint(self):
        rows = rows_fixture()
        sizes = dict(train=8, dev=3, audit=4)
        chosen = pilot.select_rows(rows, 17, sizes)
        changed = deepcopy(rows[::-1])
        for row in changed:
            row['eligible_class_0'], row['eligible_class_1'] = '999999', '0'
            row['prediction_score'] = '1.0'
        repeated = pilot.select_rows(changed, 17, sizes)
        for role, count in sizes.items():
            self.assertEqual(len(chosen[role]), count)
            self.assertEqual([r['scenario_id'] for r in chosen[role]],
                             [r['scenario_id'] for r in repeated[role]])
        shards = {role: {r['source_shard'] for r in pool} for role, pool in chosen.items()}
        for a, b in (('train', 'dev'), ('train', 'audit'), ('dev', 'audit')):
            self.assertFalse(shards[a] & shards[b])

    def test_selection_excludes_unbuilt_rows_and_rejects_insufficient_pool(self):
        rows = rows_fixture()
        rows += [dict(status='EXCLUDED', scenario_id='excluded', source_shard='outside')]
        chosen = pilot.select_rows(rows, 17, dict(train=2, dev=2, audit=2))
        self.assertNotIn('excluded', [r['scenario_id'] for pool in chosen.values() for r in pool])
        with self.assertRaisesRegex(ValueError, 'Insufficient source shards'):
            pilot.select_rows(rows[:90], 17, dict(train=1, dev=1, audit=1))
        with self.assertRaisesRegex(ValueError, 'Insufficient scenes'):
            pilot.select_rows(rows, 17, dict(train=1000, dev=1, audit=1))

    def test_normalizer_uses_pooled_pair_rows_with_population_scale(self):
        a, b = part_fixture(value=1.), part_fixture(value=7.)
        a['x'] = a['x'][:3]
        b['x'] = b['x'][:7]
        a['x'][:, 0] = [1., 3., 8.]
        b['x'][:, 0] = np.arange(7)
        original = a['x'].copy()
        normalizer = pilot.fit_normalizer([a, dict(x=np.empty((0, 784))), b])
        pooled = np.concatenate([a['x'], b['x']]).astype(np.float64)
        np.testing.assert_allclose(normalizer['mean'], pooled.mean(axis=0))
        np.testing.assert_allclose(normalizer['scale'], np.maximum(pooled.std(axis=0), 1e-6))
        np.testing.assert_array_equal(a['x'], original)
        self.assertEqual(normalizer['fitted_rows'], 10)
        with self.assertRaisesRegex(ValueError, 'Empty TRAIN'):
            pilot.fit_normalizer([dict(x=np.empty((0, 784)))])

    def test_target_uses_identity_and_eligible_negative_without_model_score(self):
        part = part_fixture()
        index = pilot.target_index(part, 17)
        self.assertEqual(int(part['y'][index]), 0)
        chosen_identity = part['canonical_pair_track_ids'][index].tolist()
        order = np.random.RandomState(4).permutation(len(part['y']))
        reordered = dict(part)
        for name in ('eligible', 'y', 'canonical_pair_track_ids'):
            reordered[name] = part[name][order]
        other_index = pilot.target_index(reordered, 17)
        self.assertEqual(reordered['canonical_pair_track_ids'][other_index].tolist(), chosen_identity)
        part['y'][part['y'] == 0] = 1
        self.assertIsNone(pilot.target_index(part, 17))

    def run_overlap(self, first, second):
        rows = {
            'train': [dict(scenario_id='a', payload_sha256='payload-a',
                           map_fingerprint_sha256='map-a')],
            'audit': [dict(scenario_id='b', payload_sha256='payload-b',
                           map_fingerprint_sha256='map-b')],
        }
        archives = {'a': archive_bytes(first), 'b': archive_bytes(second)}
        np_load = np.load
        with patch.object(pilot, '_asset', side_effect=lambda root, row, kind: row['scenario_id']), \
             patch.object(pilot.np, 'load', side_effect=lambda path, **kwargs:
                          np_load(io.BytesIO(archives[path]), **kwargs)):
            return pilot.overlap_audit(Path('/synthetic-only'), rows)

    def test_overlap_catches_shifted_moving_sdc_fragments(self):
        pos = np.column_stack([np.arange(24), np.zeros(24)]).astype(float)
        result = self.run_overlap(
            dict(source_sdc_position=pos[:20], source_sdc_valid=np.ones(20, bool)),
            dict(source_sdc_position=pos[4:], source_sdc_valid=np.ones(20, bool)))
        self.assertTrue(result['cross_role_conflicts'])
        self.assertTrue(all(c['kind'] == 'sdc_window' for c in result['cross_role_conflicts']))
        self.assertFalse(result['recording_independence_proven'])

    def test_stationary_and_invalid_windows_do_not_assert_independence(self):
        stationary = dict(source_sdc_position=np.zeros((12, 2)),
                          source_sdc_valid=np.ones(12, bool))
        result = self.run_overlap(stationary, stationary)
        self.assertEqual(result['stationary_windows_skipped'], 6)
        self.assertEqual(result['cross_role_conflicts'], [])
        self.assertFalse(result['recording_independence_proven'])
        invalid = dict(source_sdc_position=np.column_stack([np.arange(10), np.zeros(10)]),
                       source_sdc_valid=np.zeros(10, bool))
        result = self.run_overlap(invalid, invalid)
        self.assertEqual(result['sdc_windows'], 0)
        self.assertEqual(result['cross_role_conflicts'], [])

    def test_asset_path_and_digest_are_checked_before_reading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            asset = root / 'synthetic.bin'
            asset.write_bytes(b'original')
            row = dict(pair_graph_path=asset.name,
                       pair_graph_sha256=hashlib.sha256(b'original').hexdigest())
            self.assertEqual(pilot._asset(root, row, 'pair_graph'), asset)
            asset.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed or invalid'):
                pilot._asset(root, row, 'pair_graph')
            row['pair_graph_path'] = '../outside.npz'
            with self.assertRaisesRegex(ValueError, 'Changed or invalid'):
                pilot._asset(root, row, 'pair_graph')

    def test_pair_reader_checks_identity_and_does_not_load_extra_members(self):
        part = part_fixture()
        arrays = {key: value for key, value in part.items() if isinstance(value, np.ndarray)}
        arrays['scenario_id'] = np.asarray(part['scenario_id'])
        arrays['future_secret'] = np.asarray([object()], dtype=object)
        blob = archive_bytes(arrays)
        np_load = np.load
        with patch.object(pilot, '_asset', return_value='synthetic-memory'), \
             patch.object(pilot.np, 'load', side_effect=lambda path, **kwargs:
                          np_load(io.BytesIO(blob), **kwargs)):
            loaded = pilot.read_part(Path('/synthetic-only'), part)
            self.assertNotIn('future_secret', loaded)
            np.testing.assert_array_equal(loaded['x'], part['x'])
            with self.assertRaisesRegex(ValueError, 'Scenario identity mismatch'):
                pilot.read_part(Path('/synthetic-only'), dict(part, scenario_id='different'))

    def test_pair_reader_rejects_bad_labels_indices_masks_and_identities(self):
        part = part_fixture()
        cases = [
            ('bad_label', 'y', np.full(20, 2, dtype=np.int64), 'Eligibility/label'),
            ('float_index', 'edge_index', np.asarray([[0., .5], [1., 0.]]), 'adjacency'),
            ('out_of_range', 'edge_index', np.asarray([[0, 20], [1, 0]]), 'adjacency'),
            ('wrong_mask', 'eligible', np.ones(20, np.int64), 'Eligibility/label'),
            ('duplicate_ids', 'canonical_pair_track_ids', np.asarray([['0', '1']] * 20), 'identity'),
            ('short_motion', 'motion_regime', np.zeros(19, np.int64), 'identity'),
            ('bad_message', 'message_passing_pair_mask', np.ones(20, np.int64), 'identity'),
        ]
        np_load = np.load
        for name, key, changed, message in cases:
            with self.subTest(case=name):
                arrays = {key: value for key, value in part.items() if isinstance(value, np.ndarray)}
                arrays['scenario_id'] = np.asarray(part['scenario_id'])
                arrays[key] = changed
                if name == 'bad_label':
                    arrays['eligible'] = np.ones(20, dtype=bool)
                blob = archive_bytes(arrays)
                with patch.object(pilot, '_asset', return_value='synthetic-memory'), \
                     patch.object(pilot.np, 'load', side_effect=lambda path, **kwargs:
                                  np_load(io.BytesIO(blob), **kwargs)):
                    with self.assertRaisesRegex(ValueError, message):
                        pilot.read_part(Path('/synthetic-only'), part)

    def test_prepare_fits_only_train_and_indexes_are_exact_partition(self):
        rows = rows_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'source'
            root.mkdir()
            with (root / 'manifest.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_provenance(root)
            sizes = dict(train=4, dev=2, audit=2)
            selected = pilot.select_rows(rows, 17, sizes)
            role_for = {r['scenario_id']: role for role, pool in selected.items() for r in pool}
            source_parts = {sid: part_fixture(sid, {'train': 2., 'dev': 1000., 'audit': -1000.}[role])
                            for sid, role in role_for.items()}
            audit = dict(cross_role_conflicts=[], recording_independence_proven=False)
            destination = Path(temp) / 'prepared'
            with patch.dict(pilot.ROLE_SIZES, sizes, clear=True), \
                 patch.object(pilot, 'overlap_audit', return_value=audit), \
                 patch.object(pilot, 'read_part', side_effect=lambda root, row:
                              deepcopy(source_parts[row['scenario_id']])):
                summary = pilot.prepare(root, destination, seed=17)
            with np.load(destination / 'normalizer.npz') as norm:
                np.testing.assert_array_equal(norm['mean'], 2.)
                self.assertEqual(int(norm['fitted_rows']), 80)
            self.assertEqual(summary['normalizer_fitted_rows'], 80)
            parts, train = pilot.load_role(destination, 'train')
            self.assertTrue(all(p['split_group_id'] == 'source_shard:' + p['source_shard']
                                for p in parts))
            np.testing.assert_array_equal(train['x'], 0.)
            indices = torch.load(destination / 'indices.pt', weights_only=False)
            clean, unlabeled = indices['idx_train'].numpy(), indices['idx_unlabeled'].numpy()
            self.assertTrue(train['eligible'][clean].all())
            self.assertEqual(len(clean), int(.2 * int(train['eligible'].sum())))
            self.assertEqual(len(clean), len(np.unique(clean)))
            self.assertEqual(set(train['y'][clean].tolist()), {0, 1})
            self.assertFalse(np.intersect1d(clean, unlabeled).size)
            np.testing.assert_array_equal(np.sort(np.concatenate([clean, unlabeled])), np.arange(80))
            self.assertEqual(train['offsets'], [0, 20, 40, 60])
            np.testing.assert_array_equal(train['edge_index'][:, 2:4], [[20, 21], [21, 20]])
            _, dev = pilot.load_role(destination, 'dev')
            np.testing.assert_array_equal(dev['x'], 10.)
            registration = json.loads((destination / 'registration.json').read_text())
            self.assertFalse(registration['recording_independence_proven'])
            with (destination / 'dev.pt').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'Changed cached data'):
                pilot.load_role(destination, 'dev')


if __name__ == '__main__':
    unittest.main()
