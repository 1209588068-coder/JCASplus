# Source: tests/test_spear_distribution_replication_reporting.py; synthetic tests migrated with current-package imports.
"""Synthetic-only fixed-seed summaries; no checkpoints, graphs or jobs."""
from copy import deepcopy
import json
import math
import unittest

from spear_current import replication_reporting as r
from spear_current import distribution_reporting as prior
from tests_current.fixtures_metrics import fixture, normal_metrics, target_metrics


def evaluations():
    _, original = fixture()
    return {arm: deepcopy(original) for arm in r.ARMS}, original, deepcopy(original)


def distribution(value):
    def point(offset):
        v = value + offset
        return dict(versus_train=dict(rbf_mmd2_biased_selected=v, rbf_mmd2_biased_joint=2 * v,
                                     wasserstein_1d_mean=3 * v),
            train_quantile_outliers=dict(coordinate_fraction=v, row_fraction=v),
            hard_clamp_boundary=dict(coordinate_fraction=v),
            selected_modification={name: factor * v for name, factor in
                (('l1_mean', 1), ('l2_mean', 2), ('linf_mean', 3), ('l1_max', 4), ('l2_max', 5), ('linf_max', 6))})
    return dict(strata={stratum: dict(status='AVAILABLE', reference_count=447, audit_count=224,
        arms={arm: point(i * .01) for i, arm in enumerate(r.ARMS)}) for stratum in r.STRATA})


def replicates():
    values = []
    for index, seed in enumerate(r.SEEDS):
        reports, original, projected = evaluations()
        # Deliberately retain the old 85% review for the already observed seed.
        review = (prior if index == 0 else r).review_evaluations(reports, original, projected)
        values.append(dict(seed=seed, review=review, distribution=distribution((index + 1) * .1)))
    return values


class ReplicationReviewTests(unittest.TestCase):
    def test_primary_eighty_historical_eightyfive_and_no_mutation(self):
        reports, original, projected = evaluations()
        reports['conditional']['metrics']['victim']['targets'] = target_metrics(140, 48)
        before = deepcopy((reports, original, projected))
        result = r.review_evaluations(reports, original, projected)
        self.assertEqual((reports, original, projected), before)
        self.assertTrue(result['gates']['conditional']['passed'])
        self.assertFalse(result['historical_comparison']['gates']['conditional']['passed'])
        self.assertEqual(result['gate_contract']['overall_incremental_asr_min'], .8)
        self.assertEqual(result['historical_comparison']['gate_contract']['overall_incremental_asr_min'], .85)
        self.assertNotIn('overall_incremental_asr_at_least_85pct', result['gates']['conditional']['checks'])
        self.assertIn('one fixed seed', result['gate_contract']['quality_reference_interpretation'])
        self.assertEqual(result['metrics'], prior.review_evaluations(reports, original, projected)['metrics'])

    def test_primary_eighty_boundary_and_quality_not_relaxed(self):
        reports, original, projected = evaluations()
        reports['conditional']['metrics']['victim']['targets'] = target_metrics(140, 48)
        self.assertTrue(r.review_evaluations(reports, original, projected)['gates']['conditional']['passed'])
        reports['conditional']['metrics']['victim']['targets'] = target_metrics(140, 47)
        self.assertFalse(r.review_evaluations(reports, original, projected)['gates']['conditional']['passed'])
        reports['conditional']['metrics']['victim']['targets'] = target_metrics(140, 48)
        reports['continuation']['metrics']['victim'].update(normal_metrics(255, 119, .910))
        checks = r.review_evaluations(reports, original, projected)['gates']['conditional']['checks']
        self.assertFalse(checks['matched_continuation_auc_drop_at_most_0p005'])

    def test_original_validation_still_rejects_changed_population_or_holdout(self):
        for kind in ('arm', 'count', 'test', 'clean_reference'):
            reports, original, projected = evaluations()
            if kind == 'arm': del reports['conditional']
            elif kind == 'count': reports['conditional']['audit_counts']['targets'] = 223
            elif kind == 'test': reports['conditional']['original_test_arrays_loaded'] = True
            else: projected['metrics']['clean_reference'].update(normal_metrics(207))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                r.review_evaluations(reports, original, projected)


class ReplicationAggregateTests(unittest.TestCase):
    def test_complete_seed_mean_sample_sd_no_pooling_or_mutation(self):
        values = replicates(); before = deepcopy(values)
        result = r.aggregate(values)
        self.assertEqual(values, before)
        metric = result['summary']['distribution']['all']['continuation']['selected_mmd2']
        self.assertAlmostEqual(metric['mean'], .2)
        self.assertAlmostEqual(metric['sd_ddof1'], .1)
        self.assertEqual(metric['n'], 3)
        self.assertEqual(metric['ddof'], 1)
        self.assertTrue(result['complete_metric_coverage'])
        self.assertFalse(result['protocol']['pooled_target_count_used'])
        self.assertEqual(result['protocol']['retrospective_seeds'], [20260621])
        self.assertEqual(result['protocol']['prospective_seeds'], [20260622, 20260623])
        self.assertEqual(result, r.aggregate(list(reversed(values))))
        json.dumps(result, allow_nan=False)

    def test_pair_before_summarizing_not_difference_of_standard_deviations(self):
        values = replicates()
        left, right = (.9, .8, .7), (.8, .85, .6)
        for i, record in enumerate(values):
            record['review']['metrics']['conditional']['victim']['clean_pair_auc'] = left[i]
            record['review']['metrics']['unconditional']['victim']['clean_pair_auc'] = right[i]
        result = r.aggregate(values)
        metric = result['paired_differences']['conditional_minus_unconditional']['performance']['clean_pair_auc']
        expected = [.1, -.05, .1]
        for seed, value in zip(r.SEEDS, expected):
            self.assertAlmostEqual(metric['values_by_seed'][str(seed)], value)
        self.assertAlmostEqual(metric['mean'], .05)
        self.assertAlmostEqual(metric['sd_ddof1'], math.sqrt(.0075))
        for name, _, _ in r.PAIRS:
            self.assertIn(name, result['paired_differences'])
        dist = result['paired_differences']['conditional_minus_continuation']['distribution']['moving']['l1_mean']
        self.assertAlmostEqual(dist['mean'], .02)
        self.assertAlmostEqual(dist['sd_ddof1'], 0.)

    def test_observed_old_gate_kept_and_separate_eighty_view_computed(self):
        values = replicates()
        reports, original, projected = evaluations()
        reports['conditional']['metrics']['victim']['targets'] = target_metrics(140, 48)
        values[0]['review'] = prior.review_evaluations(reports, original, projected)
        before = deepcopy(values[0]['review'])
        result = r.aggregate(values)
        point = result['per_seed']['20260621']
        self.assertFalse(point['source_gates']['conditional']['passed'])
        self.assertTrue(point['primary_gates']['conditional']['passed'])
        self.assertEqual(point['design_role'], 'retrospective_observed')
        self.assertEqual(values[0]['review'], before)

    def test_missing_seed_duplicate_extra_seed_and_bool_rejected(self):
        base = replicates()
        cases = [base[:2], base + [deepcopy(base[0])], [base[0], base[1], deepcopy(base[1])]]
        for seed in (20260624, True, '20260623'):
            current = deepcopy(base); current[2]['seed'] = seed; cases.append(current)
        for current in cases:
            with self.subTest(seeds=[x['seed'] for x in current]), self.assertRaises(ValueError):
                r.aggregate(current)

    def test_failed_quality_seed_is_retained_in_mean_and_gate_counts(self):
        values = replicates()
        reports, original, projected = evaluations()
        reports['conditional']['metrics']['victim']['targets'] = target_metrics(70, 20)
        values[1]['review'] = r.review_evaluations(reports, original, projected)
        result = r.aggregate(values)
        metric = result['summary']['performance']['conditional']['all_incremental_asr']
        self.assertEqual(metric['n'], 3)
        self.assertAlmostEqual(metric['values_by_seed']['20260622'], 90 / 224)
        self.assertAlmostEqual(metric['mean'], (207 + 90 + 207) / 224 / 3)
        self.assertFalse(result['primary_gate_summary']['conditional']['all_three_seeds_pass'])
        self.assertEqual(result['primary_gate_summary']['conditional']['pass_count'], 2)

    def test_missing_scalar_not_zero_imputed_or_partial_seed_mean(self):
        values = replicates()
        del values[1]['review']['metrics']['conditional']['victim']['clean_pair_auc']
        result = r.aggregate(values)
        metric = result['summary']['performance']['conditional']['clean_pair_auc']
        self.assertEqual(metric['status'], 'NOT_AVAILABLE')
        self.assertEqual(metric['missing_seeds'], [20260622])
        self.assertEqual(metric['n'], 2)
        self.assertIsNone(metric['mean']); self.assertIsNone(metric['sd_ddof1'])
        self.assertIsNone(metric['values_by_seed']['20260622'])
        paired = result['paired_differences']['conditional_minus_unconditional']['performance']['clean_pair_auc']
        self.assertIsNone(paired['mean'])
        self.assertFalse(result['complete_metric_coverage'])
        self.assertEqual(result['summary']['performance']['unconditional']['clean_pair_auc']['status'], 'AVAILABLE')

    def test_missing_arm_and_unavailable_stratum_retain_all_fixed_slots(self):
        values = replicates()
        del values[1]['distribution']['strata']['all']['arms']['conditional']
        values[2]['distribution']['strata']['moving']['status'] = 'NOT_AVAILABLE'
        result = r.aggregate(values)
        for metric in r.DISTRIBUTION_PATHS:
            self.assertIsNone(result['summary']['distribution']['all']['conditional'][metric]['mean'])
            for arm in r.ARMS:
                item = result['summary']['distribution']['moving'][arm][metric]
                self.assertEqual(item['missing_seeds'], [20260623])
        self.assertEqual(set(result['per_seed']), {str(seed) for seed in r.SEEDS})

    def test_nonfinite_boolean_and_string_metrics_are_unavailable(self):
        for value in (math.nan, math.inf, -math.inf, True, '.5'):
            values = replicates()
            values[1]['distribution']['strata']['slow']['arms']['conditional']['versus_train']['wasserstein_1d_mean'] = value
            result = r.aggregate(values)
            metric = result['summary']['distribution']['slow']['conditional']['marginal_w1']
            self.assertEqual(metric['status'], 'NOT_AVAILABLE')
            json.dumps(result, allow_nan=False)

    def test_missing_quality_check_does_not_become_pass(self):
        values = replicates()
        del values[2]['review']['gates']['conditional']['checks']['original_auc_drop_at_most_0p005']
        result = r.aggregate(values)
        point = result['per_seed']['20260623']['primary_gates']['conditional']
        self.assertIsNone(point['passed'])
        self.assertEqual(point['status'], 'NOT_AVAILABLE')
        self.assertIsNone(result['primary_gate_summary']['conditional']['all_three_seeds_pass'])

    def test_no_independent_confirmation_or_significance_claim(self):
        result = r.aggregate(replicates())
        for key in ('independent_data_confirmation', 'statistical_significance_claim', 'independence_assumed',
                    'physical_validity_claim', 'conditional_distribution_matching_proven',
                    'distribution_preservation_claim', 'automatic_promotion', 'automatic_training'):
            self.assertFalse(result['interpretation'][key])
        self.assertIn('within-seed', result['protocol']['quality_reference_interpretation'])
        self.assertEqual(result['metric_directions']['performance']['adjacent_incremental_fp'], 'lower')
        self.assertEqual(result['metric_directions']['performance']['clean_pair_auc'], 'higher')


if __name__ == '__main__':
    unittest.main()
