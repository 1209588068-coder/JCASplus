"""Synthetic reverse-sampler checks; no dataset, model calls or fitting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current.reverse_selection import select_reverse_populations


def inputs():
    labels = torch.tensor([0, 1, 0, 0, 1, -1, 1, 1, 1, 1, 0, -1, 1, 1, 1])
    eligible = torch.ones(len(labels), dtype=torch.bool)
    eligible[[5, 6, 11, 12]] = False
    return (dict(y=labels, eligible=eligible),
            dict(idx_train=torch.tensor([0, 1]), idx_attach=torch.tensor([2, 3]),
                 idx_unlabeled=torch.tensor([14, 2, 7, 4, 8, 5, 1, 0, 9, 6, 11, 12, 10, 13]),
                 train_labels=labels.clone()))


class RequiredFieldsOnly(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        if key not in self:
            raise AssertionError('Forbidden field access: ' + key)
        self.reads.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key]


class ReverseSelectionTests(unittest.TestCase):
    def test_one_exact_ordered_draw_partition_and_exclusion_accounting(self):
        train, indices = inputs()
        result = select_reverse_populations(train, indices, attachment_count=2, extra_count=3)
        candidates = torch.tensor([14, 7, 4, 8, 9, 13])
        expected = candidates[np.random.RandomState(20260621).choice(6, 5, replace=False)]
        self.assertTrue(torch.equal(result['candidates'], candidates))
        self.assertTrue(torch.equal(result['outer'], expected))
        self.assertTrue(torch.equal(result['idx_attach'], expected[:2]))
        self.assertTrue(torch.equal(result['extra'], expected[2:]))
        self.assertTrue(bool(train['eligible'][result['outer']].all()))
        self.assertTrue(bool((train['y'][result['outer']] == 1).all()))
        self.assertFalse(bool(torch.isin(result['outer'], indices['idx_train']).any()))
        self.assertFalse(bool(torch.isin(result['outer'], indices['idx_attach']).any()))
        self.assertFalse(bool(torch.isin(result['idx_attach'], result['extra']).any()))
        self.assertEqual(len(result['outer'].unique()), len(result['outer']))
        self.assertEqual(result['selection']['counts'], dict(
            train_rows=15, idx_train=2, original_idx_attach=2, idx_unlabeled=14,
            eligible_positive_before_exclusions=7, excluded_supervised_overlap=2,
            excluded_original_attachment_overlap=1, excluded_unknown_label=2,
            excluded_negative_label=3, excluded_ineligible=4, excluded_total=8,
            candidate_count=6, attachment_count=2, extra_count=3, outer_count=5))
        self.assertEqual(json.loads(json.dumps(result['selection'])), result['selection'])
        for key in ('idx_attach', 'candidates', 'extra', 'outer'):
            expected_hash = hashlib.sha256(result[key].numpy().astype('<i8').tobytes()).hexdigest()
            self.assertEqual(result['selection']['sha256'][key], expected_hash)
        old_hash = hashlib.sha256(indices['idx_attach'].numpy().astype('<i8').tobytes()).hexdigest()
        self.assertEqual(result['selection']['sha256']['original_idx_attach'], old_hash)
        self.assertNotEqual(old_hash, result['selection']['sha256']['idx_attach'])

    def test_default_production_counts_are_new26_plus512_not_old_attachments(self):
        labels = torch.ones(700, dtype=torch.long)
        labels[50:76] = 0
        train = dict(y=labels, eligible=torch.ones(700, dtype=torch.bool))
        indices = dict(idx_train=torch.arange(50), idx_attach=torch.arange(50, 76),
                       idx_unlabeled=torch.arange(699, 75, -1), train_labels=labels.clone())
        result = select_reverse_populations(train, indices)
        expected = indices['idx_unlabeled'][np.random.RandomState(20260621).choice(624, 538, replace=False)]
        self.assertTrue(torch.equal(result['outer'], expected))
        self.assertTrue(torch.equal(result['idx_attach'], expected[:26]))
        self.assertTrue(torch.equal(result['extra'], expected[26:]))
        self.assertEqual((len(result['idx_attach']), len(result['extra']), len(result['outer'])), (26, 512, 538))
        self.assertTrue(torch.equal(indices['idx_attach'], torch.arange(50, 76)))

    def test_local_rng_is_constructed_once_and_choice_called_once(self):
        train, indices = inputs()
        real = np.random.RandomState(17)
        with patch.object(np.random, 'RandomState') as constructor:
            constructor.return_value.choice.side_effect = real.choice
            select_reverse_populations(train, indices, seed=17, attachment_count=2, extra_count=3)
        constructor.assert_called_once_with(17)
        constructor.return_value.choice.assert_called_once_with(6, 5, replace=False)

    def test_mixed_numpy_readonly_inputs_no_mutation_and_all_outputs_independent(self):
        train, indices = inputs()
        train['eligible'] = train['eligible'].numpy()
        indices['idx_unlabeled'] = indices['idx_unlabeled'].numpy().astype(np.int32)
        indices['idx_unlabeled'].setflags(write=False)
        before = deepcopy((train, indices))
        first = select_reverse_populations(train, indices, attachment_count=np.int64(2), extra_count=np.int32(3))
        second = select_reverse_populations(train, indices, attachment_count=2, extra_count=3)
        self.assertEqual(first['selection'], second['selection'])
        keys = ('idx_attach', 'outer', 'extra', 'candidates')
        for key in keys:
            self.assertTrue(torch.equal(first[key], second[key]))
            self.assertEqual(first[key].dtype, torch.long)
            self.assertEqual(first[key].device.type, 'cpu')
            self.assertFalse(first[key].requires_grad)
        for key in keys:
            fresh = select_reverse_populations(train, indices, attachment_count=2, extra_count=3)
            fresh[key].fill_(0)
            for other in keys:
                if other != key:
                    self.assertTrue(torch.equal(fresh[other], second[other]), (key, other))
        for actual_group, original_group in zip((train, indices), before):
            for key in actual_group:
                np.testing.assert_array_equal(actual_group[key], original_group[key])
        self.assertFalse(indices['idx_unlabeled'].flags.writeable)

    def test_global_numpy_torch_python_rng_and_files_untouched(self):
        train, indices = inputs()
        numpy_before, torch_before, python_before = (
            np.random.get_state(), torch.random.get_rng_state().clone(), random.getstate())
        with patch.object(np.random, 'choice', side_effect=AssertionError('global choice')), \
                patch('builtins.open', side_effect=AssertionError('file access')), \
                patch.object(Path, 'open', side_effect=AssertionError('path access')), \
                patch.object(torch, 'load', side_effect=AssertionError('asset load')):
            select_reverse_populations(train, indices, attachment_count=2, extra_count=3)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        self.assertEqual(python_before, random.getstate())

    def test_no_features_models_motion_future_or_unlisted_fields_are_read(self):
        train, indices = inputs()
        expected = select_reverse_populations(train, indices, attachment_count=2, extra_count=3)
        train, indices = RequiredFieldsOnly(train), RequiredFieldsOnly(indices)
        actual = select_reverse_populations(train, indices, attachment_count=2, extra_count=3)
        self.assertEqual(set(train.reads), {'y', 'eligible'})
        self.assertEqual(set(indices.reads), {'idx_train', 'idx_attach', 'idx_unlabeled', 'train_labels'})
        self.assertEqual(actual['selection'], expected['selection'])
        train.update(motion_regime=object(), future_labels=object(), model_outputs=object(),
                     x=object(), edge_index=object(), clean_predictions=object())
        changed = select_reverse_populations(train, indices, attachment_count=2, extra_count=3)
        self.assertEqual(changed['selection'], expected['selection'])

    def test_positive_rows_outside_original_unlabeled_never_added(self):
        train, indices = inputs()
        indices['idx_unlabeled'] = torch.tensor([14, 10, 5, 6])
        result = select_reverse_populations(train, indices, attachment_count=1, extra_count=0)
        self.assertEqual(result['candidates'].tolist(), [14])
        self.assertEqual(result['idx_attach'].tolist(), [14])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_reverse_populations(train, indices, attachment_count=1, extra_count=1)

    def test_insufficient_candidates_raise_before_rng_without_retry(self):
        train, indices = inputs()
        with patch.object(np.random, 'RandomState', side_effect=AssertionError('unexpected draw')):
            with self.assertRaisesRegex(ValueError, '6 available, 7 required'):
                select_reverse_populations(train, indices, attachment_count=3, extra_count=4)
        indices['idx_unlabeled'] = torch.tensor([10, 5, 6])
        with self.assertRaisesRegex(ValueError, '0 available, 1 required'):
            select_reverse_populations(train, indices, attachment_count=1, extra_count=0)

    def test_zero_counts_support_synthetic_empty_and_one_sided_partitions(self):
        train, indices = inputs()
        for attachment_count, extra_count in ((0, 0), (0, 2), (2, 0)):
            result = select_reverse_populations(train, indices, attachment_count=attachment_count,
                                                extra_count=extra_count)
            self.assertEqual(len(result['idx_attach']), attachment_count)
            self.assertEqual(len(result['extra']), extra_count)
            self.assertTrue(torch.equal(result['outer'], torch.cat([result['idx_attach'], result['extra']])))
        indices['idx_unlabeled'] = torch.empty(0, dtype=torch.long)
        result = select_reverse_populations(train, indices, attachment_count=0, extra_count=0)
        for key in ('idx_attach', 'extra', 'outer', 'candidates'):
            self.assertEqual(result[key].tolist(), [])
            self.assertEqual(result[key].dtype, torch.long)

    def test_duplicate_and_out_of_range_indices_rejected_for_every_domain(self):
        for key in ('idx_train', 'idx_attach', 'idx_unlabeled'):
            for bad, message in ((torch.tensor([2, 2]), 'duplicate'),
                                 (torch.tensor([-1]), 'out-of-range'),
                                 (torch.tensor([15]), 'out-of-range'),
                                 (np.array([2 ** 64 - 1], dtype=np.uint64), 'out-of-range')):
                train, indices = inputs()
                indices[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, message):
                    select_reverse_populations(train, indices, attachment_count=1, extra_count=1)

    def test_supervised_overlap_and_nonnegative_original_attachments_rejected(self):
        for attachment in ([0], [4], [5], [6]):
            train, indices = inputs()
            indices['idx_attach'] = torch.tensor(attachment)
            with self.subTest(attachment=attachment), self.assertRaisesRegex(ValueError, 'disjoint|original negative'):
                select_reverse_populations(train, indices, attachment_count=1, extra_count=1)
        for supervised in ([5], [6]):
            train, indices = inputs()
            indices['idx_train'] = torch.tensor(supervised)
            with self.subTest(supervised=supervised), self.assertRaisesRegex(ValueError, 'eligible supervised labels'):
                select_reverse_populations(train, indices, attachment_count=1, extra_count=1)

    def test_original_label_identity_and_eligibility_validation(self):
        invalid = [('y', torch.zeros(15)), ('y', torch.full((15,), 2)),
                   ('y', torch.zeros((15, 1), dtype=torch.long)),
                   ('eligible', torch.ones(15, dtype=torch.long)),
                   ('eligible', torch.ones(14, dtype=torch.bool))]
        for key, value in invalid:
            train, indices = inputs()
            train[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                select_reverse_populations(train, indices, attachment_count=1, extra_count=1)
        train, indices = inputs()
        train['y'][14] = 0
        with self.assertRaisesRegex(ValueError, 'differs from original'):
            select_reverse_populations(train, indices, attachment_count=1, extra_count=1)
        train, indices = inputs()
        indices['train_labels'] = indices['train_labels'][:-1]
        with self.assertRaisesRegex(ValueError, 'differs from original'):
            select_reverse_populations(train, indices, attachment_count=1, extra_count=1)

    def test_index_dtype_shape_device_and_parameters_rejected(self):
        for value in ([7, 8], torch.tensor([7., 8.]), torch.tensor([True, False]),
                      torch.tensor([[7, 8]]), torch.empty(2, device='meta', dtype=torch.long),
                      torch.tensor([0, 1]).to_sparse()):
            train, indices = inputs()
            indices['idx_unlabeled'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_reverse_populations(train, indices, attachment_count=1, extra_count=1)
        for key in ('attachment_count', 'extra_count'):
            for value in (-1, True, np.bool_(False), 1.5, '2', None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    select_reverse_populations(*inputs(), **{key: value})
        for seed in (-1, 2 ** 32, True, np.bool_(False), 1.5, '17', None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                select_reverse_populations(*inputs(), seed=seed, attachment_count=1, extra_count=1)

    def test_protocol_explicitly_disclaims_entropy_selection_and_direction_symmetry(self):
        selection = select_reverse_populations(*inputs(), attachment_count=2, extra_count=3)['selection']
        self.assertIn('NEW uniform positive', selection['protocol_note'])
        self.assertIn('not original entropy', selection['protocol_note'])
        self.assertIn('no direction-symmetry claim', selection['protocol_note'])
        self.assertIsNone(selection['motion_quota'])
        self.assertFalse(selection['model_predictions_used'])
        self.assertFalse(selection['original_labels_changed'])
        self.assertFalse(selection['original_indices_changed'])


if __name__ == '__main__':
    unittest.main()
