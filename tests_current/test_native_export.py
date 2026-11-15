"""Synthetic-only checks for the portable clean-data export."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from author_native_runs.export_clean import DataAssets, assemble, validate


def fixture():
    def part(name):
        return dict(x=np.zeros((3, 784), dtype=np.float32),
            y=np.array([1, 0, -1]), eligible=np.array([True, True, False]),
            motion_regime=np.array([0, 1, 0]),
            edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
            canonical_pair_track_ids=np.array([[name+'a', name+'b'], [name+'a', name+'c'], [name+'b', name+'c']]))
    parts = {key: [part(key)] for key in ('train', 'dev')}
    rows = {key: [dict(role=key, scenario_id=key, split_group_id=key+'group')] for key in parts}
    indices = dict(idx_train=torch.tensor([0]), idx_attach=torch.tensor([1]),
        idx_unlabeled=torch.tensor([], dtype=torch.long), dev_indices=torch.tensor([3, 4]))
    return parts, rows, indices


class NativeExportTests(unittest.TestCase):
    def test_clean_alignment_and_no_attack_fields(self):
        parts, rows, indices = fixture()
        before = deepcopy(parts)
        out = assemble(parts, rows, indices)
        self.assertEqual(out['idx_atk'].tolist(), [4])
        self.assertEqual(out['candidates'].tolist(), [1, 2])
        self.assertEqual(out['scene_ptr'].tolist(), [0, 3, 6])
        self.assertTrue(torch.equal(out['y'], torch.tensor([1, 0, -1, 1, 0, -1])))
        self.assertFalse(any('generator' in key or 'poisoned' in key for key in out))
        for role in parts:
            np.testing.assert_array_equal(before[role][0]['x'], parts[role][0]['x'])

    def test_cross_scene_edge_rejected(self):
        out = assemble(*fixture())
        out['edge_index'][1, 0] = 3
        with self.assertRaisesRegex(ValueError, 'Cross-scene'):
            validate(out)

    def test_background_never_supervised(self):
        parts, rows, indices = fixture()
        indices['idx_train'] = torch.tensor([2])
        with self.assertRaisesRegex(ValueError, 'supervision'):
            assemble(parts, rows, indices)

    def test_dev_cannot_enter_candidates(self):
        out = assemble(*fixture())
        out['candidates'] = torch.tensor([1, 4])
        with self.assertRaisesRegex(ValueError, 'role'):
            validate(out)

    def test_old_attachment_and_reservoir_are_not_read(self):
        parts, rows, indices = fixture()
        del indices['idx_attach'], indices['idx_unlabeled']
        out = assemble(parts, rows, indices)
        self.assertEqual(out['candidates'].tolist(), [1, 2])

    def test_duplicate_recording_rejected(self):
        parts, rows, indices = fixture()
        rows['dev'][0]['split_group_id'] = rows['train'][0]['split_group_id']
        with self.assertRaisesRegex(ValueError, 'recording'):
            assemble(parts, rows, indices)

    def test_checkpoint_asset_access_blocked(self):
        assets = object.__new__(DataAssets)
        with self.assertRaisesRegex(ValueError, 'Only frozen clean'):
            assets.asset_path('clean_reference')


if __name__ == '__main__':
    unittest.main()
