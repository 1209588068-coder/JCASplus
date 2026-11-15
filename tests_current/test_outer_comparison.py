"""Synthetic development-summary comparisons; no data/model reads or fits."""
from copy import deepcopy
import json
import unittest

from spear_current.outer_comparison import GATES, compare_dev_summaries


def model(*, slow=8, moving=7, auc=.9, tp=18, tn=79, fp=1, fn=2, adjacent_flips=0):
    return dict(normal_pairs=100, normal_positive_pairs=20, normal_negative_pairs=80,
        clean_pair_auc=auc, clean_pair_positive_recall=tp / 20,
        clean_confusion_matrix=dict(true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp),
        clean_prediction_distribution=dict(class_0=tn + fn, class_1=fp + tp),
        targets={key: dict(targets=n, incremental_flips=k, incremental_asr=k / n)
                 for key, n, k in (('all', 20, slow + moving), ('slow', 10, slow), ('moving', 10, moving))},
        collateral=dict(adj_negative=40, adj_flip=adjacent_flips,
            adjacent_incremental_fp=adjacent_flips / 40, nonincident_negative=20, adj_positive=10))


def summary(*, slow=8, moving=7, **kwargs):
    return dict(status='DEV_EVALUATION_COMPLETE',
        population_identity_sha256={key: char * 64 for key, char in
            zip(('scenes', 'eligible_clean_pairs', 'targets', 'triggered_pairs'), 'abcd')},
        target_identity_sha256='c' * 64,
        dev_counts=dict(registered_scenes=24, processed_scenes=24,
            scenes_without_eligible_negative_target=4, all_pair_nodes=120, supervised_pairs=100,
            targets=20, triggered_supervised_pair_rows=95),
        metrics=dict(victim=model(slow=slow, moving=moving, **kwargs), surrogate=model(), clean_reference=model()),
        raw_targets=dict(rows=20), raw_clean_scores=dict(rows=100), raw_triggered_pair_scores=dict(rows=95))


def set_target_counts(report, *, slow_count, moving_count, slow_flips, moving_flips):
    for metric in report['metrics'].values():
        for key, count, flips in (('all', slow_count + moving_count, slow_flips + moving_flips),
                                  ('slow', slow_count, slow_flips), ('moving', moving_count, moving_flips)):
            metric['targets'][key] = dict(targets=count, incremental_flips=flips,
                                         incremental_asr=flips / count if count else None)


class OuterComparisonTests(unittest.TestCase):
    def test_all_gates_and_exploratory_gain_pass_with_no_input_mutation(self):
        baseline, candidate = summary(), summary(moving=8)
        originals = deepcopy((baseline, candidate))
        result = compare_dev_summaries(baseline, candidate)
        self.assertEqual((baseline, candidate), originals)
        self.assertEqual(result['status'], 'PILOT_COMPLETE_REVIEW_REQUIRED')
        self.assertTrue(result['acceptable_performance'])
        self.assertTrue(result['exploratory_improvement'])
        self.assertTrue(result['recommend_follow_up_review'])
        self.assertFalse(result['automatic_jobs_started'])
        self.assertFalse(result['statistical_significance_claimed'])
        self.assertFalse(result['multiseed_evidence'])
        self.assertEqual(result['incremental_changes']['moving']['incremental_count_gain'], 1)
        self.assertEqual(result['performance_gates']['overall_incremental_asr']['numerator'], 16)
        self.assertEqual(result['performance_gates']['overall_incremental_asr']['denominator'], 20)
        self.assertEqual(result['performance_gates']['moving_incremental_asr']['denominator'], 10)
        self.assertEqual(len(result['performance_gates']), 8)
        self.assertEqual(result['preregistered_gates'], GATES)
        self.assertEqual(json.loads(json.dumps(result)), result)
        result['measurements']['candidate']['clean']['prediction_counts']['class_0'] = 0
        self.assertEqual((baseline, candidate), originals)

    def test_performance_and_improvement_are_separate(self):
        result = compare_dev_summaries(summary(moving=8), summary(moving=8))
        self.assertTrue(result['acceptable_performance'])
        self.assertFalse(result['exploratory_improvement'])
        self.assertFalse(result['recommend_follow_up_review'])
        result = compare_dev_summaries(summary(moving=5), summary(moving=6))
        self.assertFalse(result['acceptable_performance'])
        self.assertTrue(result['exploratory_improvement'])
        self.assertFalse(result['recommend_follow_up_review'])
        self.assertEqual(result['status'], 'PILOT_COMPLETE_REVIEW_REQUIRED')

    def test_moving_gain_requires_overall_not_worse(self):
        result = compare_dev_summaries(summary(slow=10, moving=7), summary(slow=8, moving=8))
        self.assertTrue(result['acceptable_performance'])
        self.assertTrue(result['exploratory_improvement_gates']['moving_gain_at_least_one']['passed'])
        self.assertFalse(result['exploratory_improvement_gates']['overall_not_worse']['passed'])
        self.assertFalse(result['recommend_follow_up_review'])

    def test_incremental_asr_threshold_inclusive_and_adjacent_fp_strict(self):
        result = compare_dev_summaries(summary(), summary(moving=8, adjacent_flips=1))
        self.assertTrue(result['performance_gates']['overall_incremental_asr']['passed'])
        self.assertTrue(result['performance_gates']['moving_incremental_asr']['passed'])
        self.assertTrue(result['performance_gates']['adjacent_incremental_fp']['passed'])
        result = compare_dev_summaries(summary(), summary(moving=8, adjacent_flips=2))
        gate = result['performance_gates']['adjacent_incremental_fp']
        self.assertEqual((gate['numerator'], gate['denominator'], gate['value']), (2, 40, .05))
        self.assertFalse(gate['passed'])
        result = compare_dev_summaries(summary(), summary(slow=7, moving=8))
        self.assertFalse(result['performance_gates']['overall_incremental_asr']['passed'])

    def test_auc_and_recall_drop_thresholds_inclusive(self):
        result = compare_dev_summaries(summary(), summary(moving=8, auc=.895, tp=17, fn=3))
        for key in ('clean_auc_vs_baseline', 'clean_auc_vs_reference',
                    'positive_recall_vs_baseline', 'positive_recall_vs_reference'):
            self.assertTrue(result['performance_gates'][key]['passed'], key)
        result = compare_dev_summaries(summary(), summary(moving=8, auc=.8949, tp=16, fn=4))
        for key in ('clean_auc_vs_baseline', 'clean_auc_vs_reference',
                    'positive_recall_vs_baseline', 'positive_recall_vs_reference'):
            self.assertFalse(result['performance_gates'][key]['passed'], key)

    def test_both_baseline_and_clean_reference_quality_gates_apply(self):
        result = compare_dev_summaries(summary(auc=.88, tp=16, fn=4),
                                       summary(moving=8, auc=.88, tp=16, fn=4))
        for metric in ('clean_auc', 'positive_recall'):
            self.assertTrue(result['performance_gates'][metric + '_vs_baseline']['passed'])
            self.assertFalse(result['performance_gates'][metric + '_vs_reference']['passed'])
        result = compare_dev_summaries(summary(auc=.92, tp=20, fn=0), summary(moving=8))
        for metric in ('clean_auc', 'positive_recall'):
            self.assertFalse(result['performance_gates'][metric + '_vs_baseline']['passed'])
            self.assertTrue(result['performance_gates'][metric + '_vs_reference']['passed'])

    def test_all_negative_and_all_positive_prediction_collapse_fail(self):
        for kwargs in (dict(tn=80, fp=0, fn=20, tp=0), dict(tn=0, fp=80, fn=0, tp=20)):
            with self.subTest(kwargs=kwargs):
                result = compare_dev_summaries(summary(), summary(moving=8, **kwargs))
                self.assertFalse(result['performance_gates']['noncollapsed_clean_predictions']['passed'])
                self.assertFalse(result['acceptable_performance'])

    def test_zero_moving_and_adjacent_denominators_fail_without_fallback(self):
        baseline, candidate = summary(), summary()
        for report in (baseline, candidate):
            set_target_counts(report, slow_count=20, moving_count=0, slow_flips=18, moving_flips=0)
            for metric in report['metrics'].values():
                metric['collateral'].update(adj_negative=0, adjacent_incremental_fp=None)
        result = compare_dev_summaries(baseline, candidate)
        for key in ('moving_incremental_asr', 'adjacent_incremental_fp'):
            gate = result['performance_gates'][key]
            self.assertEqual(gate['denominator'], 0)
            self.assertIsNone(gate['value'])
            self.assertFalse(gate['passed'])
        self.assertFalse(result['recommend_follow_up_review'])

    def test_undefined_auc_is_an_explicit_failed_quality_gate(self):
        result = compare_dev_summaries(summary(), summary(moving=8, auc=None))
        self.assertFalse(result['performance_gates']['clean_auc_vs_baseline']['passed'])
        self.assertEqual(result['performance_gates']['clean_auc_vs_baseline']['reason'], 'undefined_metric')

    def test_population_hash_mismatches_are_rejected_for_every_pool(self):
        for key in ('scenes', 'eligible_clean_pairs', 'targets', 'triggered_pairs'):
            with self.subTest(key=key):
                baseline, candidate = summary(), summary(moving=8)
                candidate['population_identity_sha256'][key] = 'e' * 64
                if key == 'targets':
                    candidate['target_identity_sha256'] = 'e' * 64
                with self.assertRaisesRegex(ValueError, 'pool identities differ'):
                    compare_dev_summaries(baseline, candidate)

    def test_pool_count_and_stratum_changes_rejected_even_with_same_hash(self):
        baseline, candidate = summary(), summary(moving=8)
        candidate['dev_counts']['all_pair_nodes'] += 1
        with self.assertRaisesRegex(ValueError, 'pool counts differ'):
            compare_dev_summaries(baseline, candidate)
        candidate = summary(moving=8)
        set_target_counts(candidate, slow_count=9, moving_count=11, slow_flips=8, moving_flips=9)
        with self.assertRaisesRegex(ValueError, 'pool counts differ'):
            compare_dev_summaries(baseline, candidate)
        candidate = summary(moving=8)
        for metric in candidate['metrics'].values():
            metric['collateral']['adj_negative'] += 1
        with self.assertRaisesRegex(ValueError, 'pool counts differ'):
            compare_dev_summaries(baseline, candidate)

    def test_mismatched_model_denominators_and_changed_reference_rejected(self):
        baseline, candidate = summary(), summary(moving=8)
        candidate['metrics']['surrogate']['collateral']['adj_positive'] += 1
        with self.assertRaisesRegex(ValueError, 'different evaluation pool counts'):
            compare_dev_summaries(baseline, candidate)
        candidate = summary(moving=8)
        candidate['metrics']['clean_reference']['clean_pair_auc'] = .91
        with self.assertRaisesRegex(ValueError, 'Frozen clean-reference'):
            compare_dev_summaries(baseline, candidate)

    def test_reference_trigger_response_can_change_without_changing_clean_reference(self):
        baseline, candidate = summary(), summary(moving=8)
        candidate['metrics']['clean_reference'] = model(slow=2, moving=1, adjacent_flips=1)
        result = compare_dev_summaries(baseline, candidate)
        self.assertTrue(result['acceptable_performance'])
        self.assertTrue(result['exploratory_improvement'])

    def test_inconsistent_rates_counts_and_missing_metadata_rejected(self):
        mutations = (
            lambda report: report['metrics']['victim']['targets']['moving'].update(incremental_asr=.9),
            lambda report: report['metrics']['victim']['collateral'].update(adjacent_incremental_fp=.01),
            lambda report: report['metrics']['victim'].update(clean_pair_positive_recall=.1),
            lambda report: report['metrics']['victim']['clean_prediction_distribution'].update(class_0=0),
            lambda report: report['metrics']['victim']['clean_prediction_distribution'].update(class_2=1),
            lambda report: report['metrics']['victim']['targets']['moving'].update(incremental_flips=True),
            lambda report: report['metrics']['victim'].update(clean_pair_auc=float('nan')),
            lambda report: report['dev_counts'].update(processed_scenes=23),
            lambda report: report['raw_targets'].update(rows=19),
            lambda report: report.update(status='RUNNING'),
            lambda report: report.pop('population_identity_sha256'))
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                baseline, candidate = summary(), summary(moving=8)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    compare_dev_summaries(baseline, candidate)


if __name__ == '__main__':
    unittest.main()
