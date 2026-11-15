"""Pure protocol checks: no graph/model deserialization or training."""
import json
import random
import unittest

from spear_current.extension_protocol import case_matrix
from spear_current.support_schema import (
    EXCLUDED_COORDINATES, LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups,
)


def source_mask():
    # Five triplets + nine singleton coordinates = a feasible exact24 support.
    triplets = [g for g in support_groups('semantic') if len(g) == 3]
    singletons = [g for g in support_groups('semantic') if len(g) == 1]
    selected = {c for group in triplets[:5] + singletons[:9] for c in group}
    return [c in selected for c in SELECTED_COORDINATES]


class ExtensionProtocolTests(unittest.TestCase):
    def test_matrix_counts_and_json(self):
        cases = case_matrix(source_mask())
        self.assertEqual(len(cases), 14)
        self.assertEqual(json.loads(json.dumps(cases)), cases)
        self.assertEqual([c['poison_count'] for c in cases if c['axis'] == 'poison'], [0, 4, 8, 16])
        self.assertEqual([c['active_coordinate_count'] for c in cases if c['axis'] == 'features'], [4, 8, 16])
        self.assertEqual(cases[0]['active_coordinate_count'], 24)

    def test_every_active_support_is_source_subset_and_groups_are_whole(self):
        mask = source_mask()
        source = {c for c, enabled in zip(SELECTED_COORDINATES, mask) if enabled}
        cases = case_matrix(mask)
        self.assertEqual(set(cases[0]['active_coordinates']), source)
        for case in cases:
            self.assertTrue(set(case['active_coordinates']) <= source)
            for key in ('active_coordinates', 'missing_coordinates', 'immutable_coordinates'):
                selected = set(case[key])
                self.assertFalse(selected & set(EXCLUDED_COORDINATES))
                self.assertTrue(selected <= set(LEGAL_COORDINATES))
                for group in support_groups('semantic'):
                    self.assertTrue(not selected.intersection(group) or set(group) <= selected)

    def test_missing_and_immutable_are_distinct_exact_cost_interventions(self):
        cases = case_matrix(source_mask())
        missing = [c for c in cases if c['axis'] == 'missing']
        immutable = [c for c in cases if c['axis'] == 'immutable']
        self.assertEqual([c['actual_missing_coordinate_count'] for c in missing], [7, 14, 21])
        self.assertEqual([c['actual_immutable_coordinate_count'] for c in immutable], [14, 21, 27])
        self.assertEqual(immutable[-1]['nominal_rate'], .95)
        self.assertEqual(immutable[-1]['actual_immutable_rate'], 27 / 28)
        for case in missing:
            self.assertFalse(case['immutable_coordinates'])
            self.assertFalse(set(case['active_coordinates']) & set(case['missing_coordinates']))
        for case in immutable:
            self.assertFalse(case['missing_coordinates'])
            self.assertFalse(set(case['active_coordinates']) & set(case['immutable_coordinates']))

    def test_no_global_random_change_or_feedback(self):
        before = random.getstate()
        first = case_matrix(source_mask())
        self.assertEqual(before, random.getstate())
        try:
            random.seed(123)
            self.assertEqual(first, case_matrix(source_mask()))
        finally:
            random.setstate(before)
        for case in first:
            self.assertFalse(case['generator_retrained'])
            self.assertFalse(case['selector_retrained'])
            self.assertFalse(case['support_uses_evaluation_predictions'])

    def test_rates_have_explicit_denominators(self):
        for case in case_matrix(source_mask()):
            count = case['poison_count']
            self.assertEqual(case['poison_pair_rate'], count / 22005)
            self.assertEqual(case['poison_fitting_row_rate'], count / (1876 + count))

    def test_reject_metadata_and_partial_groups(self):
        mask = source_mask()
        metadata = SELECTED_COORDINATES.index(EXCLUDED_COORDINATES[0])
        enabled = next(i for i, value in enumerate(mask) if value)
        invalid = mask.copy()
        invalid[metadata], invalid[enabled] = True, False
        with self.assertRaises(ValueError):
            case_matrix(invalid)
        group = next(g for g in support_groups('semantic') if len(g) == 3 and mask[SELECTED_COORDINATES.index(g[0])])
        omitted = next(c for c in LEGAL_COORDINATES if not mask[SELECTED_COORDINATES.index(c)])
        invalid = mask.copy()
        invalid[SELECTED_COORDINATES.index(group[0])] = False
        invalid[SELECTED_COORDINATES.index(omitted)] = True
        with self.assertRaises(ValueError):
            case_matrix(invalid)

    def test_reject_wrong_mask_contract(self):
        for invalid in ([], [True] * 30, [0] * 30, [False] * 30, 'mask'):
            with self.assertRaises(ValueError):
                case_matrix(invalid)


if __name__ == '__main__':
    unittest.main()
