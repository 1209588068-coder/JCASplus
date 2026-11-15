"""Synthetic lossless WOMD cache conversion and provenance checks; no training."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from spear_current import av2_formal_data as shared
from spear_current import womd_aligned_data as data
from spear_current.runtime import sha256
from spear_current.womd_pilot_data import validate_source


def write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True))


def fixture(root):
    """A registered synthetic source chain with existing normalized arrays."""
    source_run, graphs, ref = root / 'source', root / 'graphs', root / 'reference'
    cache = source_run / 'data'
    for folder in (cache, graphs, ref / 'data'):
        folder.mkdir(parents=True)
    seed = 20261104
    selected, parts_by_role, summary = {}, {}, {}
    for role, count in [('train', 3), ('dev', 2), ('audit', 1)]:
        rows, parts = [], []
        for index in range(count):
            n = 30
            sid = f'{role}-{index}'
            row = dict(scenario_id=sid, source_shard=f'{role}-shard-{index}', status='BUILT')
            rows.append(row)
            y = np.asarray([0, 1] * 13 + [-1] * 4, dtype=np.int64)
            x = np.random.RandomState(index + len(role)).normal(0, .2, (n, 784)).astype(np.float32)
            part = dict(scenario_id=sid, source_shard=row['source_shard'], x=x,
                y=y, eligible=y >= 0, edge_index=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
                canonical_pairs=np.column_stack((np.arange(n), np.arange(n) + 1)),
                canonical_pair_track_ids=np.array([[str(i), str(i+1)] for i in range(n)]),
                motion_regime=(np.arange(n) % 3 == 0).astype(np.int8),
                message_passing_pair_mask=np.ones(n, dtype=bool), target_index=2,
                custom_preserved_array=np.array([index, 100 + index], dtype=np.int32))
            parts.append(part)
        selected[role], parts_by_role[role] = rows, parts
        torch.save(parts, cache / f'{role}.pt')
        summary[role] = dict(scenes=count, pair_nodes=30*count, pair_edges=3*count,
            eligible=26*count, positive=13*count, targets=count, moving_targets=0)
    with (graphs / 'manifest.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scenario_id', 'source_shard', 'status'])
        writer.writeheader()
        writer.writerows(row for rows in selected.values() for row in rows)
    write(graphs / 'plan.json', dict(dataset='Waymo Open Motion Dataset',
                                    source_dataset_split='training_20s', test_arrays_loaded=False))
    graph_plan_hash = sha256(graphs / 'plan.json')
    write(graphs / 'registration.json', dict(plan_sha256=graph_plan_hash))
    write(graphs / 'summary.json', dict(plan_sha256=graph_plan_hash,
        manifest_sha256=sha256(graphs / 'manifest.csv'), test_arrays_loaded=False))
    write(graphs / 'completion.json', dict(plan_sha256=graph_plan_hash,
        summary_sha256=sha256(graphs / 'summary.json'), status='GRAPH_BUILD_COMPLETE_SPLIT_REVIEW_REQUIRED'))
    write(cache / 'registration.json', dict(seed=seed, graph_root=str(graphs),
        manifest_sha256=sha256(graphs / 'manifest.csv'), roles=selected,
        role_sizes={r: len(rows) for r, rows in selected.items()}, source_split='training_20s',
        recording_independence_proven=False, test_arrays_loaded=False))
    write(cache / 'overlap_audit.json', dict(cross_role_conflicts=[], recording_independence_proven=False))
    np.savez(cache / 'normalizer.npz', mean=np.arange(784, dtype=float),
        scale=np.full(784, 3.), clip=10., minimum_scale=1e-6, feature_dimension=784, fitted_rows=90)
    eligible = np.flatnonzero(np.concatenate([p['eligible'] for p in parts_by_role['train']]))
    clean = np.sort(np.random.RandomState(seed).permutation(eligible)[:int(.2 * len(eligible))])
    indices = dict(idx_train=torch.from_numpy(clean),
                   idx_unlabeled=torch.from_numpy(np.setdiff1d(np.arange(90), clean)))
    torch.save(indices, cache / 'indices.pt')
    summary.update(seed=seed, status='PREPARED', test_arrays_loaded=False,
        recording_independence_proven=False, clean_supervised_pairs=len(clean), normalizer_fitted_rows=90)
    summary['asset_sha256'] = {p.name: sha256(p) for p in cache.iterdir()}
    write(cache / 'summary.json', summary)
    source_plan = dict(seed=seed, graph_root=str(graphs), graph_provenance_sha256=validate_source(graphs),
        data_summary_sha256=sha256(cache / 'summary.json'), test_arrays_loaded=False,
        source_selection=dict(attachment_count=26))
    write(source_run / 'plan.json', source_plan)
    write(source_run / 'registration.json', dict(plan_sha256=sha256(source_run / 'plan.json')))
    write(source_run / 'completion.json', dict(plan_sha256=sha256(source_run / 'plan.json'),
        status='PILOT_COMPLETE_REVIEW_REQUIRED', test_arrays_loaded=False))
    ref_summary = dict(train=dict(eligible=2188291), original_test_arrays_loaded=False)
    write(ref / 'data/summary.json', ref_summary)
    write(ref / 'plan.json', dict(data_dir=str(ref / 'data'), data_summary=ref_summary,
        data_summary_sha256=sha256(ref / 'data/summary.json'), poison_budget=dict(count=5865),
        source_selection=dict(attachment_count=5865),
        clean_supervision='20 percent of all eligible TRAIN pairs; sampled anew by each main seed'))
    write(ref / 'registration.json', dict(plan_sha256=sha256(ref / 'plan.json')))
    return source_run, ref / 'plan.json', parts_by_role, indices


def rebind_source(source_run):
    """Keep a synthetic source's outer hash chain valid after deliberate edits."""
    cache = source_run / 'data'
    summary = json.loads((cache / 'summary.json').read_text())
    summary['asset_sha256'] = {name: sha256(cache / name) for name in summary['asset_sha256']}
    write(cache / 'summary.json', summary)
    plan = json.loads((source_run / 'plan.json').read_text())
    plan['data_summary_sha256'] = sha256(cache / 'summary.json')
    write(source_run / 'plan.json', plan)
    for name in ('registration.json', 'completion.json'):
        record = json.loads((source_run / name).read_text())
        record['plan_sha256'] = sha256(source_run / 'plan.json')
        write(source_run / name, record)


class WOMDAlignedDataTests(unittest.TestCase):
    def test_budget_matches_active_reference_fraction(self):
        budget = data.aligned_attachment_budget(11514, 5865, 437658)
        self.assertEqual(budget['count'], 154)
        self.assertEqual(budget['candidates'], [154, 155])
        self.assertEqual(budget['denominator_count'], 11668)
        self.assertEqual(budget['original_pilot_attachment_count'], 26)
        self.assertAlmostEqual(budget['realized_fraction'], 154 / 11668)
        self.assertEqual(data.aligned_attachment_budget(10, 2, 10)['count'], 2)
        with self.assertRaises(ValueError):
            data.aligned_attachment_budget(0, 2, 10)

    def test_repack_preserves_every_payload_and_shared_reader_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, ref, originals, indices = fixture(root)
            hashes = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
            destination = root / 'new'
            summary = data.prepare(source, destination, ref, max_nodes=45)
            self.assertEqual(summary['dataset_id'], 'WOMD')
            self.assertEqual(summary['source_dataset_split'], 'training_20s')
            self.assertEqual(summary['new_statistics_fitted'], 0)
            self.assertFalse(summary['test_arrays_loaded'])
            self.assertFalse(summary['recording_independence_proven'])
            self.assertEqual(sha256(source / 'data/normalizer.npz'), sha256(destination / 'normalizer.npz'))
            self.assertEqual(sha256(source / 'data/indices.pt'), sha256(destination / 'indices.pt'))
            for role in shared.ROLES:
                restored = list(shared.iter_role_parts(destination, role))
                self.assertEqual(data.payload_sha256(originals[role]), data.payload_sha256(restored))
                self.assertEqual(summary[role]['source_payload_sha256'], summary[role]['repacked_payload_sha256'])
                for old, new in zip(originals[role], restored):
                    for key, value in old.items():
                        if isinstance(value, np.ndarray):
                            self.assertEqual(value.dtype, new[key].dtype)
                            self.assertEqual(value.tobytes(), new[key].tobytes())
                        else:
                            self.assertEqual(value, new[key])
                    self.assertEqual(new['split'], 'training_20s')
                    self.assertEqual(new['split_group_id'], 'source_shard:' + old['source_shard'])
                batches = list(shared.iter_role_batches(destination, role, max_nodes=60))
                self.assertEqual(sum(len(b['x']) for b in batches), summary[role]['pair_nodes'])
                for batch in batches:
                    owner = np.concatenate([np.full(len(p['x']), i) for i, p in enumerate(batch['parts'])])
                    edges = batch['edge_index'].numpy()
                    np.testing.assert_array_equal(owner[edges[0]], owner[edges[1]])
                pop = shared.load_role_arrays(destination, role)
                np.testing.assert_array_equal(pop['y'], np.concatenate([p['y'] for p in originals[role]]))
            regenerated = shared.make_indices(destination, 20261104)
            for key in indices:
                self.assertTrue(torch.equal(regenerated[key], indices[key]))
            self.assertEqual(hashes, {str(p): sha256(p) for p in source.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                data.prepare(source, destination, ref)

    def test_oversize_scene_is_never_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, ref, _, _ = fixture(root)
            summary = data.prepare(source, root / 'new', ref, max_nodes=20)
            self.assertEqual([s['scenes'] for s in summary['train']['shards']], [1, 1, 1])
            self.assertEqual([s['nodes'] for s in summary['train']['shards']], [30, 30, 30])

    def test_rejects_changed_source_before_writing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, ref, _, _ = fixture(root)
            with (source / 'data/train.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed registered asset'):
                data.prepare(source, root / 'new', ref)
            self.assertFalse((root / 'new').exists())

    def test_rejects_reordered_scenes_even_with_rebound_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, ref, originals, _ = fixture(root)
            torch.save(originals['train'][::-1], source / 'data/train.pt')
            rebind_source(source)
            with self.assertRaisesRegex(ValueError, 'identity/order'):
                data.prepare(source, root / 'new', ref)
            self.assertFalse((root / 'new/summary.json').exists())
            self.assertEqual(json.loads((root / 'new/status.json').read_text())['stage'], 'FAILED')

    def test_rejects_changed_clean_indices_even_with_rebound_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, ref, _, indices = fixture(root)
            indices['idx_train'] = indices['idx_train'].flip(0)
            torch.save(indices, source / 'data/indices.pt')
            rebind_source(source)
            with self.assertRaisesRegex(ValueError, 'Original clean indices'):
                data.prepare(source, root / 'new', ref)

    def test_rejects_changed_reference_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, ref, _, _ = fixture(root)
            plan = json.loads(ref.read_text())
            plan['poison_budget']['count'] = 26
            write(ref, plan)
            with self.assertRaisesRegex(ValueError, 'Changed registered asset'):
                data.prepare(source, root / 'new', ref)


if __name__ == '__main__':
    unittest.main()
