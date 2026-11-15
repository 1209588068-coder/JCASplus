# Source: tests/test_spear_domain_audit_reporting.py; synthetic tests migrated with current-package imports.
"""Synthetic reporting tests; no arrays, models, checkpoints or file I/O."""
from copy import deepcopy
import math
import unittest

from spear_current import domain_audit_reporting as r


def counts():
    return dict(registered_scenes=5, processed_scenes=5, scenes_without_eligible_negative_target=1,
                all_pair_nodes=40, supervised_pairs=32, targets=4, triggered_supervised_pair_rows=31,
                slow_targets=2, moving_targets=2)


def target_metrics(slow_flips, moving_flips):
    result = {}
    for regime, n, flips in (('all', 4, slow_flips + moving_flips), ('slow', 2, slow_flips), ('moving', 2, moving_flips)):
        result[regime] = dict(targets=n, absolute_asr=flips / n, triggered_target_class_count=flips,
            incremental_asr=flips / n, incremental_flips=flips, conditional_flip=flips / n,
            clean_negative_targets=n, clean_activation_rate=0., target_probability_delta_mean=.2)
    return result


def normal_metrics():
    return dict(normal_pairs=32, normal_positive_pairs=3, normal_negative_pairs=29,
        clean_pair_auc=1., clean_pair_accuracy=1., clean_pair_positive_recall=1.,
        clean_prediction_distribution=dict(class_0=29, class_1=3),
        clean_confusion_matrix=dict(true_negative=29, false_positive=0, false_negative=0, true_positive=3),
        all_clean_predictions_negative=False, clean_class_one_probability_min=.1, clean_class_one_probability_max=.9)


def collateral():
    return dict(adj_negative=20, adj_flip=0, nonincident_negative=5, nonincident_flip=0,
                adj_positive=1, adj_suppressed=0, adjacent_incremental_fp=0.,
                nonincident_incremental_fp=0., adjacent_positive_suppression=0.)


def fixture():
    expected = counts()
    reports, budgets = {}, {}
    # These are exactly the changes for clean=[9,0], generated=[100,1].
    # The clipped output has L1=2; the radial control remains outside [-10,10].
    for mode, slow, moving in (('identity', 2, 2), ('domain_projection', 2, 1), ('l1_matched_scale', 1, 1)):
        reports[mode] = dict(audit_counts={key: expected[key] for key in r.AUDIT_COUNT_KEYS},
            metrics={role: dict(**normal_metrics(), targets=target_metrics(slow, moving), collateral=collateral())
                     for role in r.ROLES}, original_val_arrays_loaded=False, original_test_arrays_loaded=False)
        budgets[mode] = []
        for index in range(4):
            original = dict(original_outside_coordinate_count=1, original_l1=92., original_linf=91.,
                projected_l1=2., projected_linf=1., original_changed_coordinate_count=2,
                projected_changed_coordinate_count=2, alpha=2 / 92)
            actual = (dict(actual_l1=92., actual_linf=91., actual_changed_coordinate_count=2,
                           actual_outside_coordinate_count=1) if mode == 'identity' else
                      dict(actual_l1=2., actual_linf=1. if mode == 'domain_projection' else 91 * (2 / 92),
                           actual_changed_coordinate_count=2,
                           actual_outside_coordinate_count=0 if mode == 'domain_projection' else 1))
            budgets[mode].append(dict(scenario_id=f'scene-{index}', motion_regime='slow' if index < 2 else 'moving',
                                      **original, **actual))
    return reports, budgets, expected


class DomainAuditReportingTests(unittest.TestCase):
    def review(self, reports, budgets, expected):
        return r.review_reports(reports, budgets, expected_counts=expected)

    def test_valid_synthetic_report_and_no_mutation(self):
        reports, budgets, expected = fixture()
        original_reports, original_budgets = deepcopy(reports), deepcopy(budgets)
        result = self.review(reports, budgets, expected)
        self.assertEqual(result['status'], 'FROZEN_DOMAIN_AUDIT_REVIEW_COMPLETE')
        self.assertTrue(result['gates']['identity']['passed'])
        self.assertFalse(result['gates']['domain_projection']['passed'])
        self.assertFalse(result['gates']['l1_matched_scale']['passed'])
        self.assertEqual(result['comparison']['domain_projection']['victim']['incremental_asr_delta'],
                         {'all': -.25, 'slow': 0., 'moving': -.5})
        self.assertTrue(result['comparison']['domain_projection']['victim']['normal_metrics_exactly_equal'])
        self.assertEqual(result['modification_statistics']['domain_projection']['all']['targets_outside_domain'], 0)
        self.assertEqual(result['modification_statistics']['l1_matched_scale']['all']['targets_outside_domain'], 4)
        self.assertEqual(result['modification_statistics']['identity']['moving']['targets'], 2)
        self.assertFalse(result['automatic_promotion'])
        self.assertFalse(result['new_method_claim'])
        self.assertEqual(reports, original_reports)
        self.assertEqual(budgets, original_budgets)

    def test_default_population_cannot_silently_accept_a_small_fixture(self):
        reports, budgets, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'population'):
            r.review_reports(reports, budgets)
        self.assertEqual(r.EXPECTED_COUNTS['targets'], 224)
        self.assertEqual(r.EXPECTED_COUNTS['slow_targets'], 164)
        self.assertEqual(r.EXPECTED_COUNTS['moving_targets'], 60)
        self.assertEqual(r.EXPECTED_COUNTS['supervised_pairs'], 4399)
        self.assertEqual(r.EXPECTED_COUNTS['triggered_supervised_pair_rows'], 4394)

    def test_adjacent_boundary_five_percent_is_not_a_pass(self):
        reports, budgets, expected = fixture()
        entry = reports['identity']['metrics']['victim']['collateral']
        entry.update(adj_flip=1, adjacent_incremental_fp=.05)
        result = self.review(reports, budgets, expected)
        self.assertFalse(result['gates']['identity']['passed'])
        self.assertFalse(result['gates']['identity']['checks']['adjacent_incremental_fp_strictly_below_5pct'])

    def test_normal_auc_or_any_normal_metric_change_is_not_called_improvement(self):
        for key, value in (('clean_pair_auc', .999), ('clean_class_one_probability_max', .89)):
            reports, budgets, expected = fixture()
            reports['domain_projection']['metrics']['victim'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'normal metric block changed'):
                self.review(reports, budgets, expected)

    def test_missing_mode_model_or_original_test_access_fails(self):
        for mutation in ('mode', 'model', 'test', 'val', 'counts', 'empty_targets'):
            reports, budgets, expected = fixture()
            if mutation == 'mode': del reports['identity']
            elif mutation == 'model': del reports['identity']['metrics']['clean_reference']
            elif mutation == 'test': reports['identity']['original_test_arrays_loaded'] = True
            elif mutation == 'val': reports['identity']['original_val_arrays_loaded'] = True
            elif mutation == 'counts': reports['identity']['audit_counts']['supervised_pairs'] -= 1
            else: expected['moving_targets'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.review(reports, budgets, expected)

    def test_target_rate_counts_and_strata_must_agree(self):
        for mutation in ('wrong_rate', 'stratum_count', 'nan_probability_delta', 'nonpooled_mean', 'impossible_transition'):
            reports, budgets, expected = fixture()
            targets = reports['domain_projection']['metrics']['victim']['targets']
            if mutation == 'wrong_rate': targets['all']['incremental_asr'] = .99
            elif mutation == 'stratum_count': targets['moving']['targets'] = 1
            elif mutation == 'nan_probability_delta': targets['all']['target_probability_delta_mean'] = math.nan
            elif mutation == 'nonpooled_mean': targets['all']['target_probability_delta_mean'] = .1
            else: targets['all']['triggered_target_class_count'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.review(reports, budgets, expected)

    def test_collateral_denominators_cannot_disappear_or_change(self):
        for mutation in ('empty', 'changed', 'wrong_ratio', 'model_pool'):
            reports, budgets, expected = fixture()
            coll = reports['domain_projection']['metrics']['victim']['collateral']
            if mutation == 'empty': coll.update(adj_negative=0, adjacent_incremental_fp=None)
            elif mutation == 'changed': coll['adj_negative'] += 1
            elif mutation == 'wrong_ratio': coll['adjacent_incremental_fp'] = .001
            else:
                for mode in r.MODES:
                    reports[mode]['metrics']['clean_reference']['collateral']['adj_positive'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.review(reports, budgets, expected)

    def test_budget_scene_pool_unique_matched_and_stratified(self):
        for mutation in ('missing', 'duplicate', 'different', 'regime', 'unknown_regime'):
            reports, budgets, expected = fixture()
            if mutation == 'missing': budgets['domain_projection'].pop()
            elif mutation == 'duplicate': budgets['domain_projection'][1]['scenario_id'] = 'scene-0'
            elif mutation == 'different': budgets['domain_projection'][0]['scenario_id'] = 'other-scene'
            elif mutation == 'regime':
                budgets['domain_projection'][0]['motion_regime'] = 'moving'
                budgets['domain_projection'][2]['motion_regime'] = 'slow'
            else: budgets['domain_projection'][0]['motion_regime'] = 'fast'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.review(reports, budgets, expected)

    def test_source_generator_budget_cannot_change_across_modes(self):
        reports, budgets, expected = fixture()
        row = budgets['l1_matched_scale'][0]
        row.update(original_l1=93., alpha=2 / 93, actual_linf=91 * 2 / 93)
        with self.assertRaisesRegex(ValueError, 'Original generated values'):
            self.review(reports, budgets, expected)

    def test_projection_requires_zero_outside_and_nonexpansion(self):
        for mutation in ('outside', 'expansion', 'changed_count', 'projection_norm'):
            reports, budgets, expected = fixture()
            row = budgets['domain_projection'][0]
            if mutation == 'outside': row['actual_outside_coordinate_count'] = 1
            elif mutation == 'expansion': row.update(projected_l1=93., projected_linf=92., alpha=93 / 92)
            elif mutation == 'changed_count': row['actual_changed_coordinate_count'] = 1
            else: row['actual_l1'] = 1.5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.review(reports, budgets, expected)

    def test_radial_control_matches_l1_not_domain_or_projected_linf(self):
        reports, budgets, expected = fixture()
        result = self.review(reports, budgets, expected)
        projected = result['modification_statistics']['domain_projection']['all']
        scaled = result['modification_statistics']['l1_matched_scale']['all']
        self.assertEqual(projected['actual_l1']['mean'], scaled['actual_l1']['mean'])
        self.assertNotEqual(projected['actual_linf']['maximum'], scaled['actual_linf']['maximum'])
        self.assertGreater(scaled['coordinates_outside_domain'], 0)
        for mutation in ('l1', 'linf', 'alpha'):
            bad = deepcopy(budgets)
            if mutation == 'l1': bad['l1_matched_scale'][0]['actual_l1'] = 2.1
            elif mutation == 'linf': bad['l1_matched_scale'][0]['actual_linf'] = 1.5
            else: bad['l1_matched_scale'][0]['alpha'] = .5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.review(reports, bad, expected)

    def test_tolerance_accepts_float32_roundoff_not_another_amplitude(self):
        reports, budgets, expected = fixture()
        budgets['l1_matched_scale'][0]['actual_l1'] += 1e-6
        self.review(reports, budgets, expected)
        budgets['l1_matched_scale'][0]['actual_l1'] += 1e-3
        with self.assertRaises(ValueError):
            self.review(reports, budgets, expected)

    def test_count_bounds_nonfinite_and_identity_exactness(self):
        for key, value in (('original_changed_coordinate_count', 25), ('actual_outside_coordinate_count', -1),
                           ('actual_changed_coordinate_count', 1.5), ('actual_l1', math.inf),
                           ('actual_linf', math.nan), ('alpha', True), ('actual_l1', 91.99999)):
            reports, budgets, expected = fixture()
            budgets['identity'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.review(reports, budgets, expected)

    def test_zero_displacement_is_valid_and_has_zero_alpha(self):
        reports, budgets, expected = fixture()
        for mode in r.MODES:
            for key in r.COUNT_BUDGET_KEYS + r.FLOAT_BUDGET_KEYS:
                budgets[mode][0][key] = 0
        result = self.review(reports, budgets, expected)
        self.assertEqual(result['modification_statistics']['identity']['all']['actual_l1']['minimum'], 0.)
        budgets['identity'][0]['alpha'] = 1
        with self.assertRaises(ValueError):
            self.review(reports, budgets, expected)

    def test_reordered_budget_rows_are_paired_by_scene_not_position(self):
        reports, budgets, expected = fixture()
        before = self.review(reports, budgets, expected)
        budgets['domain_projection'].reverse()
        after = self.review(reports, budgets, expected)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
