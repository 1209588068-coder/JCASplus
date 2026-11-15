"""Synthetic full-data boundary and streaming-cache integrity checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import av2_formal_data as data


def row(sid, group, split='train', role='', status='exists'):
    return dict(scenario_id=sid, split_group_id=group, graph_path=f'graphs/{sid}.npz',
                graph_sha256='1' * 64, split=split, role=role, status=status, city='city',
                evaluation_subgroup='')


def part(record, value):
    n = 20
    y = np.asarray([0, 1] * 8 + [-1] * 4, dtype=np.int64)
    return dict(scenario_id=record['scenario_id'], split_group_id=record['split_group_id'],
        split='train', role=record['role'], evaluation_subgroup=record['evaluation_subgroup'],
        x=np.full((n, 784), value, dtype=np.float32), y=y, eligible=y >= 0,
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        canonical_pairs=np.column_stack((np.arange(n), np.arange(n) + 1)),
        canonical_pair_track_ids=np.asarray([[str(i), str(i + 1)] for i in range(n)]),
        motion_regime=np.arange(n, dtype=np.int8) % 2,
        message_passing_pair_mask=np.ones(n, dtype=bool))


def make_tiny_cache(destination):
    """Synthetic all-stage CUDA smoke input; never a formal AV2 run/result.

    TRAIN has 640 nodes and 128 labeled rows, sufficient for the registered
    64-row attribution sample without inheriting any real data or selection.
    """
    selected = {role: [row(f'{role}-{i}', f'{role}-group-{i}', role=role)
                       for i in range(count)]
                for role, count in [('train', 10), ('dev', 2), ('audit', 2)]}
    contract = dict(role_counts={r: dict(groups=len(rows)) for r, rows in selected.items()},
                    label_config={}, sources={}, attachment_budget=dict(count=6),
                    scope='synthetic engineering smoke only; not AV2 experiment results')

    def read(root, record, config):
        n = 64 if record['role'] == 'train' else 16
        seed = 1700 + int(record['scenario_id'].split('-')[-1])
        seed += {'train': 0, 'dev': 100, 'audit': 200}[record['role']]
        rng = np.random.RandomState(seed)
        x = rng.normal(0, .35, (n, 784)).astype(np.float32)
        y = (np.arange(n) % 2).astype(np.int64)
        x[:, 0] += y * .5
        edges = np.stack((np.arange(n - 1), np.arange(1, n)))
        return dict(scenario_id=record['scenario_id'], split_group_id=record['split_group_id'],
            split='train', role=record['role'], evaluation_subgroup='synthetic_smoke',
            x=x, y=y, eligible=np.ones(n, dtype=bool),
            edge_index=np.concatenate((edges, edges[::-1]), axis=1).astype(np.int64),
            canonical_pairs=np.column_stack((np.arange(n), np.arange(n) + 1)),
            canonical_pair_track_ids=np.asarray([[str(i), str(i + 1)] for i in range(n)]),
            motion_regime=(np.arange(n) % 3 == 0).astype(np.int8),
            message_passing_pair_mask=np.ones(n, dtype=bool))

    with patch.object(data, 'build_contract', return_value=(contract, selected)), \
         patch.object(data, 'read_part', side_effect=read):
        return data.prepare('/synthetic-only', destination, seed=17, max_nodes=128)


class AV2FormalDataTests(unittest.TestCase):
    def test_selection_excludes_entire_heldout_groups_and_val_test(self):
        rows = [row('a', 'g1'), row('b', 'g2'), row('c', 'g3'), row('d', 'g4'),
                row('same-dev-group', 'g2'), row('same-confirmation-group', 'g4'),
                row('val', 'g5', 'val'), row('test', 'g6', 'test'),
                row('unbuilt', 'g7', status='failed')]
        old = dict(dev=[rows[1]], audit=[rows[2]])
        selected = data.select_rows(rows, old, [rows[3]])
        self.assertEqual([r['scenario_id'] for r in selected['train']], ['a'])
        self.assertEqual(len(selected['audit']), 2)
        self.assertEqual(selected['audit'][1]['evaluation_subgroup'], 'historical_confirmation1024')
        changed = deepcopy(rows[::-1])
        for r in changed:
            r['num_risk_positive_edges'] = '999999'
            r['score'] = '0.99'
        repeated = data.select_rows(changed, old, [rows[3]])
        self.assertEqual(selected, repeated)

    def test_selection_rejects_cross_role_groups_and_changed_historical_hash(self):
        rows = [row('train', 'a'), row('dev', 'b'), row('audit', 'c'), row('confirm', 'd')]
        old = dict(dev=[rows[1]], audit=[rows[2]])
        changed = deepcopy(old)
        changed['dev'][0]['graph_sha256'] = '2' * 64
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            data.select_rows(rows, changed, [rows[3]])
        rows[3]['split_group_id'] = 'b'
        with self.assertRaisesRegex(ValueError, 'groups overlap'):
            data.select_rows(rows, old, [rows[3]])

    def test_forbidden_source_split_rejected_before_deserialization(self):
        with patch.object(data.np, 'load') as load:
            for split in ('val', 'test'):
                with self.assertRaisesRegex(ValueError, 'original TRAIN'):
                    data.read_part('/no-access', row('scene', 'group', split, 'audit'))
            load.assert_not_called()

    def prepare_fixture(self, destination):
        selected = {role: [row(f'{role}-{i}', f'{role}-group-{i}', role=role)
                           for i in range(count)]
                    for role, count in [('train', 3), ('dev', 2), ('audit', 2)]}
        contract = dict(role_counts={r: dict(groups=len(rows)) for r, rows in selected.items()},
                        label_config={}, sources={}, attachment_budget=dict(count=1))
        def read(root, record, config):
            value = {'train': float(record['scenario_id'][-1]), 'dev': 1000., 'audit': -1000.}[record['role']]
            return part(record, value)
        with patch.object(data, 'build_contract', return_value=(contract, selected)), \
             patch.object(data, 'read_part', side_effect=read):
            summary = data.prepare('/synthetic-only', destination, seed=17, max_nodes=25)
        return summary

    def test_streaming_prepare_fits_only_full_train_and_preserves_global_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'data'
            summary = self.prepare_fixture(destination)
            self.assertEqual(summary['train']['pair_nodes'], 60)
            self.assertEqual(summary['train']['pair_edges'], 6)
            self.assertEqual(summary['normalizer_fitted_rows'], 60)
            with np.load(destination / 'normalizer.npz', allow_pickle=False) as norm:
                np.testing.assert_allclose(norm['mean'], 1.)
                np.testing.assert_allclose(norm['scale'], np.sqrt(2 / 3))
            batches = list(data.iter_role_batches(destination, 'train', max_nodes=45))
            self.assertEqual([(b['global_start'], b['global_stop']) for b in batches], [(0, 40), (40, 60)])
            self.assertEqual(batches[0]['offsets'], [0, 20])
            np.testing.assert_array_equal(batches[0]['edge_index'][:, 2:], [[20, 21], [21, 20]])
            _, dev = data.load_role(destination, 'dev')
            np.testing.assert_array_equal(dev['x'], 10.)
            _, audit = data.load_role(destination, 'audit')
            np.testing.assert_array_equal(audit['x'], -10.)
            arrays = data.load_role_arrays(destination, 'train')
            np.testing.assert_array_equal(arrays['scene_offsets'], [0, 20, 40, 60])
            self.assertFalse(summary['original_test_arrays_loaded'])
            self.assertFalse(summary['blind_evaluation'])

    def test_each_seed_indices_partition_only_eligible_supervision(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'data'
            self.prepare_fixture(destination)
            a, b = data.make_indices(destination, 17), data.make_indices(destination, 18)
            again = data.make_indices(destination, 17)
            torch.testing.assert_close(a['idx_train'], again['idx_train'])
            self.assertFalse(torch.equal(a['idx_train'], b['idx_train']))
            population = data.load_role_arrays(destination, 'train')
            self.assertEqual(len(a['idx_train']), int(.2 * population['eligible'].sum()))
            self.assertTrue(population['eligible'][a['idx_train']].all())
            combined = torch.cat((a['idx_train'], a['idx_unlabeled'])).sort().values
            torch.testing.assert_close(combined, torch.arange(60))

    def test_corrupt_cache_is_rejected_and_oversized_scene_is_not_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'data'
            self.prepare_fixture(destination)
            batches = list(data.iter_role_batches(destination, 'train', max_nodes=5))
            self.assertEqual([len(b['x']) for b in batches], [20, 20, 20])
            with (destination / 'train/00000.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed cached scene'):
                list(data.iter_role_parts(destination, 'train'))
            with (destination / 'train_population.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed cached scene shard or population'):
                data.load_role_arrays(destination, 'train')


if __name__ == '__main__':
    unittest.main()
