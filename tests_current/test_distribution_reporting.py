# Source: tests/test_spear_distribution_training_reporting.py; synthetic tests migrated with current-package imports.
"""Synthetic review checks; no real inputs or optimization."""
from copy import deepcopy
import unittest
from tests_current.fixtures_metrics import fixture, normal_metrics, target_metrics
from spear_current import distribution_reporting as r


class DistributionReviewTests(unittest.TestCase):
    def fixture(self):
        _, original = fixture()
        return {a: deepcopy(original) for a in r.ARMS}, original, deepcopy(original)

    def test_all_arms_no_mutation_or_claims(self):
        values = self.fixture(); before = deepcopy(values)
        report = r.review_evaluations(*values)
        self.assertEqual(values, before)
        self.assertTrue(all(g['passed'] for g in report['gates'].values()))
        for key in ('distribution_preservation_claim', 'new_method_claim', 'automatic_followup',
                    'independent_confirmation_claim', 'physical_validity_claim'):
            self.assertFalse(report[key])

    def test_all_arms_required(self):
        reports, original, projected = self.fixture()
        del reports['unconditional']
        with self.assertRaisesRegex(ValueError, 'three'):
            r.review_evaluations(reports, original, projected)

    def test_report_failure_without_skipping_arms(self):
        reports, original, projected = self.fixture()
        reports['conditional']['metrics']['victim']['targets'] = target_metrics(154, 49)
        result = r.review_evaluations(reports, original, projected)
        self.assertFalse(result['gates']['conditional']['passed'])
        self.assertTrue(result['gates']['unconditional']['passed'])
        self.assertEqual(set(result['metrics']), set(r.ARMS))

    def test_matched_quality_not_only_old_source(self):
        reports, original, projected = self.fixture()
        reports['continuation']['metrics']['victim'].update(normal_metrics(255, 119, .910))
        result = r.review_evaluations(reports, original, projected)
        self.assertFalse(result['gates']['conditional']['checks']['matched_continuation_auc_drop_at_most_0p005'])

    def test_frozen_reference_and_holdout_provenance(self):
        for bad in ('reference', 'test'):
            reports, original, projected = self.fixture()
            if bad == 'test': reports['conditional']['original_test_arrays_loaded'] = True
            else: projected['metrics']['clean_reference'].update(normal_metrics(208))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                r.review_evaluations(reports, original, projected)


if __name__ == '__main__':
    unittest.main()
