"""Synthetic frozen-sampler checks; no graph/model assets or real fits."""
from copy import deepcopy
import hashlib
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current.outer_selection import select_strict_negative_outer


def inputs():
    labels = torch.tensor([0, 1, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0])
    eligible = torch.ones(len(labels), dtype=torch.bool)
    eligible[[5, 6, 11, 12]] = False
    train = dict(y=labels, eligible=eligible)
    indices = dict(idx_train=torch.tensor([0, 1]), idx_attach=torch.tensor([2, 3]),
        idx_unlabeled=torch.tensor([14, 2, 7, 4, 8, 5, 0, 9, 6, 11, 12, 10, 13]),
        train_labels=labels.clone())
    return train, indices


class RequiredFieldsOnly(dict):
    """Fail if selection inspects any field outside the frozen input contract."""

    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        if key not in self:
            raise AssertionError(f'Forbidden field access: {key}')
        self.reads.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key]


class OuterSelectionTests(unittest.TestCase):
    def test_exact_ordered_domain_draw_and_exclusion_accounting(self):
        train, indices = inputs()
        result = select_strict_negative_outer(train, indices, extra_count=3)
        candidates = torch.tensor([14, 7, 8, 9, 13])
        expected = candidates[np.random.RandomState(20260621).choice(5, 3, replace=False)]
        self.assertTrue(torch.equal(result['candidates'], candidates))
        self.assertTrue(torch.equal(result['extra'], expected))
        self.assertTrue(torch.equal(result['outer'], torch.cat([indices['idx_attach'], expected])))
        self.assertTrue(bool(train['eligible'][result['outer']].all()))
        self.assertTrue(bool((train['y'][result['outer']] == 0).all()))
        self.assertFalse(bool(torch.isin(result['outer'], indices['idx_train']).any()))
        self.assertEqual(len(result['outer'].unique()), len(result['outer']))
        self.assertEqual(result['selection']['counts'], dict(
            train_rows=15, idx_train=2, idx_attach=2, idx_unlabeled=13,
            eligible_negative_before_exclusions=7, excluded_supervised_overlap=1,
            excluded_attachment_overlap=1, excluded_unknown_label=2,
            excluded_positive_label=2, excluded_ineligible=4, excluded_total=8,
            candidate_count=5, extra_count=3, outer_count=5))
        self.assertEqual(json.loads(json.dumps(result['selection'])), result['selection'])
        for key in ('candidates', 'extra', 'outer'):
            expected_hash = hashlib.sha256(result[key].numpy().astype('<i8').tobytes()).hexdigest()
            self.assertEqual(result['selection']['sha256'][key], expected_hash)

    def test_frozen_default_draw_is_512_extras_after_26_attachments(self):
        labels = torch.zeros(602, dtype=torch.long)
        train = dict(y=labels, eligible=torch.ones(602, dtype=torch.bool))
        indices = dict(idx_train=torch.arange(50), idx_attach=torch.arange(50, 76),
                       idx_unlabeled=torch.arange(601, 75, -1), train_labels=labels.clone())
        result = select_strict_negative_outer(train, indices)
        expected = indices['idx_unlabeled'][np.random.RandomState(20260621).choice(526, 512, replace=False)]
        self.assertTrue(torch.equal(result['extra'], expected))
        self.assertTrue(torch.equal(result['outer'][:26], indices['idx_attach']))
        self.assertEqual((len(result['outer']), len(result['extra'])), (538, 512))

    def test_deterministic_mixed_numpy_tensors_without_input_mutation_or_aliasing(self):
        train, indices = inputs()
        train['eligible'] = train['eligible'].numpy()
        indices['idx_unlabeled'] = indices['idx_unlabeled'].numpy().astype(np.int32)
        indices['idx_unlabeled'].setflags(write=False)
        before = deepcopy((train, indices))
        first = select_strict_negative_outer(train, indices, extra_count=np.int64(3))
        second = select_strict_negative_outer(train, indices, extra_count=3)
        self.assertEqual(first['selection'], second['selection'])
        for field in ('outer', 'extra', 'candidates'):
            self.assertTrue(torch.equal(first[field], second[field]))
            self.assertEqual(first[field].dtype, torch.long)
            self.assertEqual(first[field].device.type, 'cpu')
        first['outer'].fill_(0)
        first['extra'].fill_(1)
        self.assertTrue(torch.equal(first['candidates'], second['candidates']))
        first['candidates'].fill_(2)
        for actual_group, original_group in zip((train, indices), before):
            for key in actual_group:
                np.testing.assert_array_equal(actual_group[key], original_group[key])
        self.assertFalse(indices['idx_unlabeled'].flags.writeable)

    def test_global_numpy_torch_and_python_rng_are_unchanged(self):
        train, indices = inputs()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        python_before = random.getstate()
        with patch.object(np.random, 'choice', side_effect=AssertionError('global choice')):
            select_strict_negative_outer(train, indices, extra_count=3)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        self.assertEqual(python_before, random.getstate())

    def test_only_original_labels_eligibility_and_indices_are_read(self):
        train, indices = inputs()
        expected = select_strict_negative_outer(train, indices, extra_count=3)
        train = RequiredFieldsOnly(train)
        indices = RequiredFieldsOnly(indices)
        actual = select_strict_negative_outer(train, indices, extra_count=3)
        self.assertEqual(set(train.reads), {'y', 'eligible'})
        self.assertEqual(set(indices.reads), {'idx_train', 'idx_attach', 'idx_unlabeled', 'train_labels'})
        self.assertTrue(torch.equal(actual['outer'], expected['outer']))
        # Unrelated motion, future and model fields cannot affect the result.
        train.update(motion_regime=object(), future_labels=object(), model_outputs=object(),
                     x=object(), edge_index=object(), clean_predictions=object())
        changed = select_strict_negative_outer(train, indices, extra_count=3)
        self.assertEqual(changed['selection'], expected['selection'])

    def test_strict_negative_rows_outside_original_unlabeled_are_never_added(self):
        train, indices = inputs()
        indices['idx_unlabeled'] = torch.tensor([14, 4, 5, 6])
        result = select_strict_negative_outer(train, indices, extra_count=1)
        self.assertEqual(result['candidates'].tolist(), [14])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_strict_negative_outer(train, indices, extra_count=2)

    def test_insufficiency_raises_before_random_state_construction_without_retry(self):
        train, indices = inputs()
        with patch.object(np.random, 'RandomState', side_effect=AssertionError('unexpected draw')):
            with self.assertRaisesRegex(ValueError, '5 available, 6 required'):
                select_strict_negative_outer(train, indices, extra_count=6)
        indices['idx_unlabeled'] = torch.tensor([4, 5, 6])
        with self.assertRaisesRegex(ValueError, '0 available, 1 required'):
            select_strict_negative_outer(train, indices, extra_count=1)
        result = select_strict_negative_outer(train, indices, extra_count=0)
        self.assertEqual(result['extra'].tolist(), [])
        self.assertEqual(result['outer'].tolist(), indices['idx_attach'].tolist())

    def test_duplicate_and_out_of_range_indices_rejected_for_each_domain(self):
        for key in ('idx_train', 'idx_attach', 'idx_unlabeled'):
            for bad, message in ((torch.tensor([2, 2]), 'duplicate'),
                                 (torch.tensor([-1]), 'out-of-range'),
                                 (torch.tensor([15]), 'out-of-range'),
                                 (np.array([2 ** 64 - 1], dtype=np.uint64), 'out-of-range')):
                with self.subTest(key=key, bad=bad):
                    train, indices = inputs()
                    indices[key] = bad
                    with self.assertRaisesRegex(ValueError, message):
                        select_strict_negative_outer(train, indices, extra_count=1)

    def test_supervised_attachment_overlap_and_invalid_attachments_rejected(self):
        for attachment in ([0], [4], [5], [6]):
            with self.subTest(attachment=attachment):
                train, indices = inputs()
                indices['idx_attach'] = torch.tensor(attachment)
                with self.assertRaisesRegex(ValueError, 'disjoint|eligible original negative'):
                    select_strict_negative_outer(train, indices, extra_count=1)
        train, indices = inputs()
        indices['idx_train'] = torch.tensor([5])
        with self.assertRaisesRegex(ValueError, 'eligible supervised labels'):
            select_strict_negative_outer(train, indices, extra_count=1)

    def test_label_identity_and_label_eligibility_validation(self):
        invalid = [('y', torch.zeros(15)), ('y', torch.full((15,), 2)),
                   ('y', torch.zeros((15, 1), dtype=torch.long)),
                   ('eligible', torch.ones(15, dtype=torch.long)),
                   ('eligible', torch.ones(14, dtype=torch.bool))]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                train, indices = inputs()
                train[key] = value
                with self.assertRaises(ValueError):
                    select_strict_negative_outer(train, indices, extra_count=1)
        train, indices = inputs()
        train['y'][14] = 1
        with self.assertRaisesRegex(ValueError, 'differs from original'):
            select_strict_negative_outer(train, indices, extra_count=1)

    def test_index_dtype_shape_device_and_parameter_validation(self):
        for value in ([7, 8], torch.tensor([7., 8.]), torch.tensor([True, False]),
                      torch.tensor([[7, 8]]), torch.empty(2, device='meta', dtype=torch.long)):
            with self.subTest(value=value):
                train, indices = inputs()
                indices['idx_unlabeled'] = value
                with self.assertRaises(ValueError):
                    select_strict_negative_outer(train, indices, extra_count=1)
        for kwargs in (dict(extra_count=-1), dict(extra_count=True), dict(extra_count=1.5),
                       dict(seed=-1), dict(seed=2 ** 32), dict(seed=True), dict(seed=1.5)):
            with self.subTest(kwargs=kwargs):
                train, indices = inputs()
                with self.assertRaises(ValueError):
                    select_strict_negative_outer(train, indices, **kwargs)


if __name__ == '__main__':
    unittest.main()
