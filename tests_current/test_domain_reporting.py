# Source: tests/test_spear_domain_training_reporting.py; synthetic tests migrated with current-package imports.
"""Synthetic aggregate-only review tests: no datasets, models or file I/O."""
from copy import deepcopy
import math
import unittest

from spear_current import domain_reporting as r


def target_metrics(slow=154, moving=53, slow_negative=156, moving_negative=55, *, reference=False):
    out = {}
    for regime, n, flips, negative in (
        ('all', 224, slow + moving, slow_negative + moving_negative),
        ('slow', 164, slow, slow_negative), ('moving', 60, moving, moving_negative),
    ):
        positive = 0 if reference else n - negative
        out[regime] = dict(targets=n, absolute_asr=(flips + positive) / n,
            triggered_target_class_count=flips + positive, incremental_asr=flips / n, incremental_flips=flips,
            conditional_flip=flips / negative if negative else None, clean_negative_targets=negative,
            clean_activation_rate=(n - negative) / n, target_probability_delta_mean=.2)
    return out


def normal_metrics(tp=228, fp=119, auc=.8978394238463592):
    tn, fn = 3910 - fp, 489 - tp
    return dict(normal_pairs=4399, normal_positive_pairs=489, normal_negative_pairs=3910,
        clean_pair_auc=auc, clean_pair_accuracy=(tn + tp) / 4399, clean_pair_positive_recall=tp / 489,
        clean_prediction_distribution=dict(class_0=tn + fn, class_1=tp + fp),
        clean_confusion_matrix=dict(true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp),
        all_clean_predictions_negative=(tp + fp == 0),
        clean_class_one_probability_min=.01, clean_class_one_probability_max=.99)


def collateral(adj=11):
    return dict(adj_negative=1202, adj_flip=adj, nonincident_negative=2484, nonincident_flip=9,
        adj_positive=199, adj_suppressed=7, adjacent_incremental_fp=adj / 1202,
        nonincident_incremental_fp=9 / 2484, adjacent_positive_suppression=7 / 199)


def fixture():
    original = dict(audit_counts={key: r.EXPECTED_COUNTS[key] for key in r.AUDIT_COUNT_KEYS},
        metrics=dict(
            victim=dict(**normal_metrics(), targets=target_metrics(), collateral=collateral()),
            clean_reference=dict(**normal_metrics(206, 119, .8958012332700485),
                targets=target_metrics(0, 0, 159, 57, reference=True), collateral=collateral(27))),
        original_val_arrays_loaded=False, original_test_arrays_loaded=False)
    return {arm: deepcopy(original) for arm in r.ARMS}, original


class DomainTrainingReportingTests(unittest.TestCase):
    def test_success_no_mutation_and_explicit_scope(self):
        reports, original = fixture(); before = deepcopy((reports, original))
        result = r.review_evaluations(reports, original)
        self.assertEqual((reports, original), before)
        self.assertEqual(result['status'], 'DOMAIN_TRAINING_REVIEW_COMPLETE')
        self.assertTrue(all(g['passed'] for g in result['gates'].values()))
        self.assertFalse(result['automatic_promotion'])
        self.assertFalse(result['independent_confirmation_claim'])
        self.assertFalse(result['new_method_claim'])

    def test_fresh_victim_normal_and_clean_target_predictions_may_change(self):
        reports, original = fixture()
        victim = reports['domain_projection']['metrics']['victim']
        victim.update(normal_metrics(230, 120, .899))
        victim['targets'] = target_metrics(150, 52, 160, 57)
        result = r.review_evaluations(reports, original)
        self.assertTrue(result['gates']['domain_projection']['passed'])
        self.assertFalse(result['integrity']['victim_normal_prediction_equality_required'])
        self.assertGreater(result['comparisons']['domain_projection']['versus_original']['victim']['clean_pair_auc_delta'], 0)

    def test_frozen_reference_triggered_response_may_change(self):
        reports, original = fixture()
        reports['domain_projection']['metrics']['clean_reference']['targets'] = target_metrics(1, 1, 159, 57, reference=True)
        reports['domain_projection']['metrics']['clean_reference']['collateral'] = collateral(10)
        r.review_evaluations(reports, original)

    def test_frozen_reference_normal_or_clean_target_changes_are_rejected(self):
        for kind in ('normal', 'target'):
            reports, original = fixture()
            ref = reports['identity']['metrics']['clean_reference']
            if kind == 'normal': ref.update(normal_metrics(207, 119, .896))
            else: ref['targets'] = target_metrics(0, 0, 160, 57, reference=True)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'clean-reference'):
                r.review_evaluations(reports, original)

    def test_performance_failure_reports_every_arm_without_raising(self):
        reports, original = fixture()
        reports['domain_projection']['metrics']['victim']['targets'] = target_metrics(113, 41)
        result = r.review_evaluations(reports, original)
        self.assertEqual(set(result['gates']), set(r.ARMS))
        self.assertTrue(result['gates']['identity']['passed'])
        self.assertFalse(result['gates']['domain_projection']['passed'])
        self.assertFalse(result['gates']['domain_projection']['checks']['moving_incremental_asr_at_least_85pct'])

    def test_quality_compared_to_original_and_new_identity(self):
        reports, original = fixture()
        reports['identity']['metrics']['victim'].update(normal_metrics(255, 119, .910))
        result = r.review_evaluations(reports, original)
        checks = result['gates']['domain_projection']['checks']
        self.assertTrue(checks['original_auc_drop_at_most_0p005'])
        self.assertTrue(checks['original_recall_drop_at_most_0p05'])
        self.assertFalse(checks['matched_identity_auc_drop_at_most_0p005'])
        self.assertFalse(checks['matched_identity_recall_drop_at_most_0p05'])
        self.assertFalse(result['gates']['domain_projection']['passed'])

    def test_auc_drop_boundary_and_actual_recall_counts(self):
        reports, original = fixture()
        lower = original['metrics']['victim']['clean_pair_auc'] - .005
        reports['domain_projection']['metrics']['victim'].update(normal_metrics(204, 119, lower))
        self.assertTrue(r.review_evaluations(reports, original)['gates']['domain_projection']['passed'])
        reports['domain_projection']['metrics']['victim'].update(normal_metrics(203, 119, lower - 1e-6))
        checks = r.review_evaluations(reports, original)['gates']['domain_projection']['checks']
        self.assertFalse(checks['original_auc_drop_at_most_0p005'])
        self.assertFalse(checks['original_recall_drop_at_most_0p05'])

    def test_five_percent_cutoff_and_moving_85pct_boundary(self):
        reports, original = fixture()
        victim = reports['domain_projection']['metrics']['victim']
        victim['targets'] = target_metrics(154, 51)  # 51/60 is exactly 85%.
        victim['collateral'] = collateral(60)  # 60/1202 < 5%.
        self.assertTrue(r.review_evaluations(reports, original)['gates']['domain_projection']['passed'])
        victim['collateral'] = collateral(61)
        self.assertFalse(r.review_evaluations(reports, original)['gates']['domain_projection']['passed'])

    def test_all_negative_victim_is_a_performance_failure(self):
        reports, original = fixture()
        reports['identity']['metrics']['victim'].update(normal_metrics(0, 0, .8))
        result = r.review_evaluations(reports, original)
        self.assertFalse(result['gates']['identity']['checks']['victim_not_all_negative'])
        self.assertFalse(result['gates']['identity']['passed'])

    def test_wrong_arms_models_population_or_holdout_access_rejected(self):
        for mutation in ('arm', 'model', 'population', 'val', 'test', 'original'):
            reports, original = fixture()
            if mutation == 'arm': reports['extra'] = deepcopy(reports['identity'])
            elif mutation == 'model': del reports['identity']['metrics']['clean_reference']
            elif mutation == 'population': reports['identity']['audit_counts']['targets'] = 223
            elif mutation in ('val', 'test'): reports['identity']['original_' + mutation + '_arrays_loaded'] = True
            else: del original['metrics']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                r.review_evaluations(reports, original)

    def test_invalid_counts_auc_recall_and_target_rates_rejected(self):
        for mutation in ('auc', 'recall', 'confusion', 'rate', 'stratum', 'delta', 'collateral', 'labels'):
            reports, original = fixture(); victim = reports['identity']['metrics']['victim']
            if mutation == 'auc': victim['clean_pair_auc'] = math.nan
            elif mutation == 'recall': victim['clean_pair_positive_recall'] = .9
            elif mutation == 'confusion': victim['clean_confusion_matrix']['true_positive'] = 229
            elif mutation == 'rate': victim['targets']['all']['incremental_asr'] = .99
            elif mutation == 'stratum': victim['targets']['moving']['targets'] = 59
            elif mutation == 'delta': victim['targets']['all']['target_probability_delta_mean'] = .3
            elif mutation == 'collateral': victim['collateral'].update(adj_negative=1203, adjacent_incremental_fp=11/1203)
            else:
                victim.update(normal_negative_pairs=3909, normal_positive_pairs=490,
                    clean_pair_positive_recall=228/490,
                    clean_confusion_matrix=dict(true_negative=3790, false_positive=119, false_negative=262, true_positive=228),
                    clean_pair_accuracy=(3790+228)/4399)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                r.review_evaluations(reports, original)


if __name__ == '__main__':
    unittest.main()
