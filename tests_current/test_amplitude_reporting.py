# Source: tests/test_spear_amplitude_control_reporting.py; synthetic tests migrated with current-package imports.
"""Synthetic-only amplitude review and fixed-seed distribution summaries."""
from copy import deepcopy
import json
import math
import unittest

import numpy as np

from spear_current import amplitude_reporting as r
from spear_current import distribution_statistics as statistics
from tests_current.fixtures_metrics import fixture, normal_metrics, target_metrics, collateral


def evaluations():
    _, original = fixture()
    return {arm: deepcopy(original) for arm in r.ARMS}, original


def distribution(value):
    def point(offset):
        v = value + offset
        return dict(versus_train=dict(rbf_mmd2_biased_selected=v, rbf_mmd2_biased_joint=2 * v,
            wasserstein_1d_mean=3 * v), train_quantile_outliers=dict(coordinate_fraction=v, row_fraction=v),
            hard_clamp_boundary=dict(coordinate_fraction=v),
            selected_modification={key: v for key in ('l1_mean', 'l2_mean', 'linf_mean', 'l1_max', 'l2_max', 'linf_max')})
    return dict(strata={name: dict(status='AVAILABLE', arms={arm: point(index * .01)
        for index, arm in enumerate(r.ARMS)}) for name in r.STRATA})


def replicates():
    return [dict(seed=seed, review=r.review_seed(*evaluations()), distribution=distribution((index + 1) * .1))
        for index, seed in enumerate(r.SEEDS)]


class AmplitudeReviewTests(unittest.TestCase):
    def test_metrics_comparisons_and_no_mutation(self):
        reports, original = evaluations()
        reports['amplitude_scaled']['metrics']['victim'].update(normal_metrics(230, 120, .899))
        before = deepcopy((reports, original))
        result = r.review_seed(reports, original)
        self.assertEqual((reports, original), before)
        self.assertEqual(set(result['metrics']), set(r.ARMS))
        self.assertTrue(all(gate['passed'] for gate in result['gates'].values()))
        delta = result['comparisons']['amplitude_scaled']['versus_matched_continuation']['victim']
        self.assertAlmostEqual(delta['clean_pair_auc_delta'], .899 - original['metrics']['victim']['clean_pair_auc'])
        self.assertEqual(delta, result['comparisons']['amplitude_scaled']['versus_unconditional']['victim'])
        self.assertTrue(result['interpretation']['posthoc_development_control'])
        self.assertFalse(result['interpretation']['new_independent_end_to_end_seeds'])

    def test_eighty_percent_and_strict_fp_boundaries(self):
        reports, original = evaluations()
        victim = reports['amplitude_scaled']['metrics']['victim']
        victim['targets'] = target_metrics(132, 48)
        victim['collateral'] = collateral(60)
        self.assertTrue(r.review_seed(reports, original)['gates']['amplitude_scaled']['passed'])
        for slow, moving, adjacent, failed in (
                (131, 48, 60, 'overall_incremental_asr_at_least_80pct'),
                (140, 47, 60, 'moving_incremental_asr_at_least_80pct'),
                (132, 48, 61, 'adjacent_incremental_fp_strictly_below_5pct')):
            victim['targets'] = target_metrics(slow, moving)
            victim['collateral'] = collateral(adjacent)
            with self.subTest(failed=failed):
                self.assertIn(failed, r.review_seed(reports, original)['gates']['amplitude_scaled']['failed_checks'])

    def test_quality_against_original_and_same_seed_continuation(self):
        reports, original = evaluations()
        reports['continuation']['metrics']['victim'].update(normal_metrics(255, 119, .910))
        result = r.review_seed(reports, original)
        for arm in ('unconditional', 'amplitude_scaled'):
            checks = result['gates'][arm]['checks']
            self.assertTrue(checks['original_auc_drop_at_most_0p005'])
            self.assertTrue(checks['original_recall_drop_at_most_0p05'])
            self.assertFalse(checks['matched_continuation_auc_drop_at_most_0p005'])
            self.assertFalse(checks['matched_continuation_recall_drop_at_most_0p05'])
        self.assertEqual(len(result['gates']['continuation']['checks']), 6)

    def test_auc_boundary_and_all_failures_retained(self):
        reports, original = evaluations()
        victim = reports['amplitude_scaled']['metrics']['victim']
        victim.update(normal_metrics(204, 119, original['metrics']['victim']['clean_pair_auc'] - .005))
        self.assertTrue(r.review_seed(reports, original)['gates']['amplitude_scaled']['passed'])
        victim.update(normal_metrics(0, 0, .8))
        victim['targets'] = target_metrics(1, 1)
        victim['collateral'] = collateral(61)
        result = r.review_seed(reports, original)
        gate = result['gates']['amplitude_scaled']
        self.assertFalse(gate['passed'])
        self.assertEqual(set(gate['failed_checks']), set(gate['checks']))
        self.assertEqual(len(gate['failed_checks']), 8)
        self.assertEqual(set(result['gates']), set(r.ARMS))

    def test_schema_frozen_reference_and_nonfinite_rejected(self):
        for kind in ('arm', 'alias', 'population', 'holdout', 'reference', 'missing', 'nan', 'infinity'):
            reports, original = evaluations()
            victim = reports['amplitude_scaled']['metrics']['victim']
            if kind == 'arm': del reports['amplitude_scaled']
            elif kind == 'alias': reports['conditional'] = reports.pop('amplitude_scaled')
            elif kind == 'population': reports['amplitude_scaled']['audit_counts']['targets'] = 223
            elif kind == 'holdout': reports['amplitude_scaled']['original_test_arrays_loaded'] = True
            elif kind == 'reference': reports['amplitude_scaled']['metrics']['clean_reference'].update(normal_metrics(207))
            elif kind == 'missing': del victim['clean_pair_auc']
            else: victim['clean_pair_auc'] = math.nan if kind == 'nan' else math.inf
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                r.review_seed(reports, original)


class AmplitudeDistributionTests(unittest.TestCase):
    def test_exact_fixed_diagnostics_without_alias_or_mutation(self):
        clean = np.arange(96, dtype=float).reshape(4, 24) / 100
        conditions = np.arange(24, dtype=float).reshape(4, 6) / 100
        moving = np.array([False, True, False, True])
        reference = statistics.fit_reference(clean, conditions, moving)
        reference_before = reference.to_metadata()
        arrays = {arm: clean + offset for arm, offset in zip(r.ARMS, (.2, .1, .05))}
        before = {arm: values.copy() for arm, values in arrays.items()}
        result = r.evaluate_distribution(reference, clean, conditions, moving, arrays)
        expected = statistics.evaluate(reference, clean, conditions, moving,
            dict(continuation=arrays['continuation'], unconditional=arrays['unconditional'], conditional=arrays['amplitude_scaled']))
        self.assertEqual(statistics.ARMS, ('continuation', 'unconditional', 'conditional'))
        self.assertEqual(reference.to_metadata(), reference_before)
        self.assertEqual(result['reference'], {**reference_before, 'arms': list(r.ARMS)})
        for arm in r.ARMS:
            np.testing.assert_array_equal(arrays[arm], before[arm])
        for stratum in r.STRATA:
            self.assertEqual(set(result['strata'][stratum]['arms']), set(r.ARMS))
            self.assertEqual(result['strata'][stratum]['arms']['amplitude_scaled'],
                expected['strata'][stratum]['arms']['conditional'])
        self.assertNotIn('"conditional"', json.dumps(result))
        arrays['amplitude_scaled'][0, 0] = math.nan
        with self.assertRaises(ValueError):
            r.evaluate_distribution(reference, clean, conditions, moving, arrays)

    def test_unavailable_stratum_renamed_and_missing_arm_rejected(self):
        clean, conditions, moving = np.zeros((2, 24)), np.zeros((2, 6)), np.zeros(2, dtype=bool)
        reference = statistics.fit_reference(clean, conditions, moving)
        arrays = {arm: clean for arm in r.ARMS}
        result = r.evaluate_distribution(reference, clean, conditions, moving, arrays)
        self.assertEqual(result['strata']['moving']['status'], 'NOT_AVAILABLE')
        self.assertEqual(result['strata']['moving']['arms'], dict.fromkeys(r.ARMS))
        del arrays['amplitude_scaled']
        with self.assertRaises(ValueError):
            r.evaluate_distribution(reference, clean, conditions, moving, arrays)


class AmplitudeAggregateTests(unittest.TestCase):
    def test_all_seed_summary_directions_interpretation_and_no_mutation(self):
        values = replicates(); before = deepcopy(values)
        result = r.aggregate(values)
        self.assertEqual(values, before)
        self.assertEqual(result, r.aggregate(list(reversed(values))))
        metric = result['summary']['distribution']['all']['continuation']['selected_mmd2']
        self.assertAlmostEqual(metric['mean'], .2)
        self.assertAlmostEqual(metric['sd_ddof1'], .1)
        self.assertEqual(metric['n'], 3)
        self.assertTrue(result['complete_metric_coverage'])
        self.assertFalse(result['protocol']['pooled_target_count_used'])
        self.assertTrue(result['interpretation']['posthoc_development_control'])
        self.assertFalse(result['interpretation']['new_independent_end_to_end_seeds'])
        self.assertEqual(result['metric_directions']['performance']['adjacent_incremental_fp'], 'lower')
        self.assertEqual(result['metric_directions']['performance']['clean_pair_auc'], 'higher')
        self.assertNotIn('"conditional"', json.dumps(result, allow_nan=False))

    def test_paired_subtraction_precedes_sample_sd(self):
        values = replicates()
        for index, (u, scaled) in enumerate(zip((.9, .8, .7), (.8, .85, .6))):
            values[index]['review']['metrics']['unconditional']['victim']['clean_pair_auc'] = u
            values[index]['review']['metrics']['amplitude_scaled']['victim']['clean_pair_auc'] = scaled
        result = r.aggregate(values)
        metric = result['paired_differences']['unconditional_minus_amplitude_scaled']['performance']['clean_pair_auc']
        for seed, expected in zip(r.SEEDS, (.1, -.05, .1)):
            self.assertAlmostEqual(metric['values_by_seed'][str(seed)], expected)
        self.assertAlmostEqual(metric['mean'], .05)
        self.assertAlmostEqual(metric['sd_ddof1'], math.sqrt(.0075))
        other = result['paired_differences']['amplitude_scaled_minus_continuation']['distribution']['moving']['l1_mean']
        self.assertAlmostEqual(other['mean'], .02)
        self.assertAlmostEqual(other['sd_ddof1'], 0)

    def test_missing_duplicate_and_unregistered_seeds_rejected(self):
        values = replicates()
        cases = [values[:2], values + [values[0]], [values[0], values[1], values[1]]]
        for seed in (20260624, True, '20260623'):
            changed = deepcopy(values); changed[2]['seed'] = seed; cases.append(changed)
        for case in cases:
            with self.subTest(seeds=[item['seed'] for item in case]), self.assertRaises(ValueError):
                r.aggregate(case)

    def test_missing_nonfinite_and_non_numeric_metrics_have_no_partial_mean(self):
        for value in (None, math.nan, math.inf, -math.inf, True, '.5'):
            values = replicates()
            values[1]['review']['metrics']['amplitude_scaled']['victim']['clean_pair_auc'] = value
            result = r.aggregate(values)
            item = result['summary']['performance']['amplitude_scaled']['clean_pair_auc']
            self.assertEqual(item['missing_seeds'], [r.SEEDS[1]])
            self.assertEqual(item['n'], 2)
            self.assertIsNone(item['mean']); self.assertIsNone(item['sd_ddof1'])
            self.assertIsNone(result['paired_differences']['unconditional_minus_amplitude_scaled']['performance']['clean_pair_auc']['mean'])
            self.assertIsNone(result['primary_gate_summary']['amplitude_scaled']['all_three_seeds_pass'])
            json.dumps(result, allow_nan=False)

    def test_missing_arm_and_empty_stratum_preserve_fixed_seed_slots(self):
        values = replicates()
        del values[1]['distribution']['strata']['all']['arms']['amplitude_scaled']
        values[2]['distribution']['strata']['moving']['status'] = 'NOT_AVAILABLE'
        result = r.aggregate(values)
        for metric in r.DISTRIBUTION_PATHS:
            self.assertIsNone(result['summary']['distribution']['all']['amplitude_scaled'][metric]['mean'])
            for arm in r.ARMS:
                self.assertEqual(result['summary']['distribution']['moving'][arm][metric]['missing_seeds'], [r.SEEDS[2]])

    def test_failed_and_missing_gate_evidence_is_retained(self):
        values = replicates()
        reports, original = evaluations()
        reports['amplitude_scaled']['metrics']['victim']['targets'] = target_metrics(70, 20)
        values[1]['review'] = r.review_seed(reports, original)
        result = r.aggregate(values)
        self.assertEqual(result['primary_gate_summary']['amplitude_scaled']['pass_count'], 2)
        self.assertEqual(result['summary']['performance']['amplitude_scaled']['all_incremental_asr']['n'], 3)
        self.assertEqual(len(result['per_seed'][str(r.SEEDS[1])]['primary_gates']['amplitude_scaled']['failed_checks']), 2)
        del values[2]['review']['gates']['amplitude_scaled']['checks']['original_auc_drop_at_most_0p005']
        self.assertIsNone(r.aggregate(values)['primary_gate_summary']['amplitude_scaled']['all_three_seeds_pass'])


if __name__ == '__main__':
    unittest.main()
