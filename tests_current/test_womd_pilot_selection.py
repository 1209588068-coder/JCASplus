"""Synthetic selector-contract tests; no graph assets or real training loaded."""
import unittest

import torch

from spear_current.support_generator import HardConcreteSupport, exact_cost_support
from spear_current.support_schema import SELECTED_COORDINATES, support_groups
from spear_current.womd_pilot_selection import CandidateHardConcrete, candidate_groups, exact_support


class CandidateSupportTests(unittest.TestCase):
    def test_historical_groups_and_projection_match_exactly(self):
        coordinates = torch.tensor(SELECTED_COORDINATES)
        groups, _ = candidate_groups(coordinates)
        self.assertEqual(groups, support_groups('semantic'))
        scores = torch.linspace(.1, .9, len(groups))
        self.assertTrue(torch.equal(exact_support(coordinates, groups, scores),
                                    exact_cost_support(groups, scores, 24)))

    def test_hard_concrete_matches_existing_equations_and_rng(self):
        new = CandidateHardConcrete(torch.tensor(SELECTED_COORDINATES), seed=17)
        old = HardConcreteSupport(seed=17)
        for _ in range(3):
            self.assertTrue(torch.equal(new(), old()))
        self.assertTrue(torch.equal(new.expected_coordinate_count(), old.expected_coordinate_count()))

    def test_partial_copy_groups_do_not_expand_permissions(self):
        coordinates = torch.tensor([350, 742] + list(range(28)))
        groups, _ = candidate_groups(coordinates)
        self.assertEqual(groups[0], (350, 742))
        self.assertNotIn(777, {c for g in groups for c in g})
        support = exact_support(coordinates, groups, torch.ones(len(groups)))
        self.assertEqual(int(support.sum()), 24)
        self.assertEqual(bool(support[0]), bool(support[1]))

    def test_focal_is_protected_but_no_silent_other_filter(self):
        coordinates = torch.tensor([357, 715, 300, 356] + list(range(26)))
        groups, metadata = candidate_groups(coordinates)
        legal = {c for g in groups for c in g}
        self.assertNotIn(357, legal)
        self.assertNotIn(715, legal)
        self.assertIn(300, legal)
        self.assertIn(356, legal)
        self.assertEqual(metadata[2]['quantity'], 'observed_valid')
        self.assertEqual(metadata[3]['quantity'], 'observed_fraction')

    def test_budget_cannot_be_silently_underfilled(self):
        coordinates = torch.arange(30)
        groups = (tuple(range(15)), tuple(range(15, 30)))
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            exact_support(coordinates, groups, [1., 1.], budget=24)

    def test_non_candidate_group_rejected(self):
        coordinates = torch.arange(30)
        groups = tuple((i,) for i in range(29)) + ((777,),)
        with self.assertRaises(ValueError):
            exact_support(coordinates, groups, torch.ones(30))


if __name__ == '__main__':
    unittest.main()
