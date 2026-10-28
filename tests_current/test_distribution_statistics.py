# Source: tests/test_spear_distribution_audit_statistics.py; synthetic tests migrated with current-package imports.
"""Synthetic, asset-free checks of frozen distribution diagnostics."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist, pdist

from spear_current import distribution_statistics as stats


def fixture():
    train = np.linspace(-2, 2, 8)[:, None] + np.arange(24)[None, :] / 100
    conditions = np.linspace(-1, 1, 8)[:, None] + np.arange(6)[None, :] / 100
    moving = np.array([False] * 6 + [True] * 2)
    rows = np.array([0, 2, 5, 7])
    clean, audit_conditions, audit_moving = train[rows].copy(), conditions[rows].copy(), moving[rows].copy()
    arms = {arm: clean + offset for arm, offset in zip(stats.ARMS, (0.05, 0.1, 0.2))}
    return train, conditions, moving, clean, audit_conditions, audit_moving, arms


def direct_mmd(left, right, sigma):
    def kernel(x, y):
        return np.exp(-cdist(x, y, metric='sqeuclidean') / (2 * sigma ** 2))
    return float(kernel(left, left).mean() + kernel(right, right).mean() - 2 * kernel(left, right).mean())


class DistributionAuditStatisticsTests(unittest.TestCase):
    def setUp(self):
        (self.train, self.conditions, self.moving, self.clean, self.audit_conditions,
         self.audit_moving, self.arms) = fixture()

    def reference(self):
        return stats.fit_reference(self.train, self.conditions, self.moving)

    def evaluate(self, reference=None, arms=None):
        return stats.evaluate(reference or self.reference(), self.clean, self.audit_conditions,
                              self.audit_moving, self.arms if arms is None else arms)

    def test_all_train_bandwidths_shared_by_strata_and_dimension_balanced(self):
        train = np.array([[0.] * 24, [1.] * 24])
        conditions = np.array([[0.] * 6, [1.] * 6])
        reference = stats.fit_reference(train, conditions, np.array([False, True]))
        self.assertAlmostEqual(reference.selected_bandwidth.sigma, 0.1)
        self.assertAlmostEqual(reference.joint_bandwidth.sigma, np.sqrt(2) * 0.1)
        metadata = reference.to_metadata()
        self.assertTrue(metadata['bandwidths_shared_across_all_strata_and_arms'])
        self.assertEqual(metadata['joint_block_weights'], 'equal')
        self.assertFalse(metadata['second_standardization'])
        self.assertEqual(metadata['selected_bandwidth']['positive_pair_count'], 1)
        self.assertEqual(metadata['strata']['slow']['count'], 1)
        self.assertEqual(metadata['strata']['moving']['count'], 1)
        # With singleton strata there is no stratum-specific median distance;
        # the global scales still define every comparison.
        arms = {arm: train + .1 for arm in stats.ARMS}
        output = stats.evaluate(reference, train, conditions, np.array([False, True]), arms)
        self.assertTrue(all(output['strata'][name]['status'] == 'AVAILABLE' for name in stats.STRATA))

    def test_mmd_matches_dense_biased_v_statistic_with_diagonals(self):
        reference = self.reference()
        output = self.evaluate(reference)
        train_sel = self.train / 10 / np.sqrt(24)
        clean_sel = self.clean / 10 / np.sqrt(24)
        train_joint = np.concatenate((train_sel, self.conditions / 10 / np.sqrt(6)), axis=1)
        clean_joint = np.concatenate((clean_sel, self.audit_conditions / 10 / np.sqrt(6)), axis=1)
        expected_sigma = float(np.median(pdist(train_sel)[pdist(train_sel) > 0]))
        self.assertAlmostEqual(reference.selected_bandwidth.sigma, expected_sigma)
        baseline = output['strata']['all']['clean_audit']['versus_train']
        self.assertAlmostEqual(baseline['rbf_mmd2_biased_selected'], direct_mmd(train_sel, clean_sel, expected_sigma), places=14)
        self.assertAlmostEqual(baseline['rbf_mmd2_biased_joint'], direct_mmd(train_joint, clean_joint, reference.joint_bandwidth.sigma), places=14)
        # Moving uses its matching two TRAIN rows but the same global kernel.
        expected_moving = direct_mmd(train_sel[self.moving], clean_sel[self.audit_moving], expected_sigma)
        actual = output['strata']['moving']['clean_audit']['versus_train']['rbf_mmd2_biased_selected']
        self.assertAlmostEqual(actual, expected_moving, places=14)

    def test_identical_populations_and_no_modification_are_exact_zero(self):
        reference = self.reference()
        output = stats.evaluate(reference, self.train, self.conditions, self.moving,
                                {arm: self.train.copy() for arm in stats.ARMS})
        for part in output['strata'].values():
            for value in part['clean_audit']['versus_train'].values():
                self.assertAlmostEqual(value, 0, places=14)
            for arm in part['arms'].values():
                for comparison in ('versus_train', 'versus_clean_audit'):
                    for value in arm[comparison].values():
                        self.assertAlmostEqual(value, 0, places=14)
                self.assertTrue(all(value == 0 for value in arm['selected_modification'].values()))

    def test_wasserstein_equal_coordinate_weight_and_norms_keep_original_units(self):
        train, conditions = np.zeros((2, 24)), np.zeros((2, 6))
        moving = np.array([False, True])
        changed = train.copy()
        changed[:, 0] = 1
        reference = stats.fit_reference(train, conditions, moving)
        output = stats.evaluate(reference, train, conditions, moving, {arm: changed for arm in stats.ARMS})
        result = output['strata']['all']['arms']['continuation']
        self.assertAlmostEqual(result['versus_train']['wasserstein_1d_mean'], 1 / 24)
        self.assertAlmostEqual(result['versus_clean_audit']['wasserstein_1d_mean'], 1 / 24)
        self.assertTrue(all(value == 1 for value in result['selected_modification'].values()))
        self.assertTrue(reference.selected_bandwidth.fallback_used)
        self.assertEqual(reference.selected_bandwidth.sigma, 1)

    def test_joint_detects_reassociation_when_selected_marginals_are_identical(self):
        selected = np.array([[-1.] * 24, [1.] * 24])
        conditions = np.array([[-1.] * 6, [1.] * 6])
        moving = np.array([False, True])
        reference = stats.fit_reference(selected, conditions, moving)
        output = stats.evaluate(reference, selected, conditions, moving,
                                {arm: selected[::-1] for arm in stats.ARMS})
        result = output['strata']['all']['arms']['conditional']['versus_clean_audit']
        self.assertEqual(result['wasserstein_1d_mean'], 0)
        self.assertAlmostEqual(result['rbf_mmd2_biased_selected'], 0)
        self.assertGreater(result['rbf_mmd2_biased_joint'], 0)
        self.assertFalse(output['interpretation']['joint_discrepancy_proves_conditional_distribution'])

    def test_quantiles_are_train_only_stratum_specific_and_strict(self):
        reference = self.reference()
        metadata = reference.to_metadata()
        for name, mask in (('all', np.ones(8, dtype=bool)), ('slow', ~self.moving), ('moving', self.moving)):
            expected = np.quantile(self.train[mask], [.01, .99], axis=0, method='linear')
            np.testing.assert_array_equal(metadata['strata'][name]['lower_quantile'], expected[0])
            np.testing.assert_array_equal(metadata['strata'][name]['upper_quantile'], expected[1])
        thresholds = np.array([metadata['strata']['all']['lower_quantile'], metadata['strata']['all']['upper_quantile']])
        conditions = np.zeros((2, 6))
        moving = np.array([False, True])
        arms = {arm: thresholds.copy() for arm in stats.ARMS}
        arms['conditional'][0, 0] -= 1e-8
        output = stats.evaluate(reference, thresholds, conditions, moving, arms)
        self.assertEqual(output['strata']['all']['clean_audit']['train_quantile_outliers']['coordinate_count'], 0)
        self.assertEqual(output['strata']['all']['arms']['conditional']['train_quantile_outliers']['coordinate_count'], 1)

    def test_hard_clamp_boundary_uses_locked_absolute_tolerance(self):
        train, conditions = np.zeros((2, 24)), np.zeros((2, 6))
        moving = np.array([False, True])
        changed = train.copy()
        changed[0, :2] = [10., 10. - .5e-6]
        changed[1, :2] = [-10. + .5e-6, 10. - 2e-6]
        reference = stats.fit_reference(train, conditions, moving)
        output = stats.evaluate(reference, train, conditions, moving, {arm: changed for arm in stats.ARMS})
        result = output['strata']['all']['arms']['conditional']['hard_clamp_boundary']
        self.assertEqual(result['coordinate_count'], 3)
        self.assertEqual(result['row_count'], 2)
        self.assertEqual(result['coordinate_fraction'], 3 / 48)
        self.assertEqual(result['row_fraction'], 1)
        self.assertEqual(reference.to_metadata()['boundary_atol'], 1e-6)

    def test_evaluation_does_not_refit_or_change_configuration_for_audit_extremes(self):
        reference = self.reference()
        before = reference.to_metadata()
        extreme_arms = {arm: self.clean + 5 for arm in stats.ARMS}
        with patch.object(stats, '_bandwidth', side_effect=AssertionError('audit bandwidth fit forbidden')), \
             patch.object(stats.np, 'quantile', side_effect=AssertionError('audit quantile fit forbidden')):
            self.evaluate(reference)
            self.evaluate(reference, extreme_arms)
        self.assertEqual(reference.to_metadata(), before)

    def test_reference_owns_immutable_data_and_metadata_is_independent(self):
        reference = self.reference()
        before = reference.to_metadata()
        self.train[:] = 100
        self.conditions[:] = 100
        self.moving[:] = False
        self.assertEqual(reference.to_metadata(), before)
        for part in reference.strata:
            for value in (part.selected, part.conditions, part.selected_embedding, part.joint_embedding):
                with self.assertRaises(ValueError): value.setflags(write=True)
            if part.lower_quantile is not None:
                with self.assertRaises(ValueError): part.lower_quantile[0] = 0
        before['strata']['all']['lower_quantile'][0] = 12345
        self.assertNotEqual(reference.to_metadata()['strata']['all']['lower_quantile'][0], 12345)
        with self.assertRaises(AttributeError): reference.selected_bandwidth = None

    def test_evaluation_preserves_all_inputs_rng_and_serializes_without_nan(self):
        arrays = (self.train, self.conditions, self.moving, self.clean, self.audit_conditions,
                  self.audit_moving, *self.arms.values())
        before = [value.copy() for value in arrays]
        rng = np.random.get_state()
        reference = self.reference()
        output = self.evaluate(reference)
        for actual, expected in zip(arrays, before): np.testing.assert_array_equal(actual, expected)
        after = np.random.get_state()
        self.assertEqual(rng[0], after[0])
        np.testing.assert_array_equal(rng[1], after[1])
        self.assertEqual(rng[2:], after[2:])
        json.dumps(output, allow_nan=False)
        self.assertEqual(output['strata']['all']['audit_count'], 4)
        self.assertEqual(output['strata']['slow']['audit_count'], 3)
        self.assertEqual(output['strata']['moving']['audit_count'], 1)

    def test_empty_strata_explicitly_unavailable_without_fallback_reference(self):
        reference = stats.fit_reference(self.train, self.conditions, np.zeros(8, dtype=bool))
        output = self.evaluate(reference)
        missing = output['strata']['moving']
        self.assertEqual(missing['status'], 'NOT_AVAILABLE')
        self.assertEqual(missing['reason'], 'empty_train_reference_stratum')
        self.assertIsNone(missing['clean_audit'])
        self.assertTrue(all(value is None for value in missing['arms'].values()))
        output = stats.evaluate(self.reference(), self.clean, self.audit_conditions,
                                np.zeros(4, dtype=bool), self.arms)
        self.assertEqual(output['strata']['moving']['reason'], 'empty_audit_stratum')
        json.dumps(output, allow_nan=False)

    def test_nonfinite_misaligned_wrong_arm_or_wrong_regime_inputs_fail(self):
        for invalid in (self.train[:, :-1], self.train[:0], self.train.astype(int),
                        np.full_like(self.train, np.nan), np.full_like(self.train, np.inf)):
            with self.assertRaises(ValueError): stats.fit_reference(invalid, self.conditions, self.moving)
        for regimes in (self.moving.astype(int), self.moving[:-1], self.moving[:, None]):
            with self.assertRaises(ValueError): stats.fit_reference(self.train, self.conditions, regimes)
        reference = self.reference()
        for mutation in ('missing_arm', 'extra_arm', 'wrong_rows', 'nonfinite'):
            arms = deepcopy(self.arms)
            if mutation == 'missing_arm': arms.pop('conditional')
            elif mutation == 'extra_arm': arms['extra'] = self.clean
            elif mutation == 'wrong_rows': arms['conditional'] = self.clean[:-1]
            else: arms['conditional'][0, 0] = np.nan
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.evaluate(reference, arms)
        with self.assertRaises(ValueError):
            stats.evaluate(reference, self.clean, self.audit_conditions[:-1], self.audit_moving, self.arms)


if __name__ == '__main__':
    unittest.main()
