"""Synthetic-only fixed-checkpoint metric checks; no real data or models."""

from copy import deepcopy
import json
import math
import unittest

from spear_current.development_metrics import (
    compare_models, population_summary, response_metrics,
)


def make_row(pair_index, *, motion="slow", label=0, eligible=True,
             surrogate=(0, 1), victim=(0, 1), attachment=False, outer=True):
    row = {
        "scenario_id": "synthetic-scenario",
        "pair_index": pair_index,
        "motion_regime": motion,
        "true_label": label,
        "eligible": eligible,
        "is_attachment": attachment,
        "is_outer": outer,
    }
    for model, predictions in (("surrogate", surrogate), ("victim", victim)):
        for state, prediction in zip(("clean", "trigger"), predictions):
            row[f"{model}_{state}_pred"] = prediction
            row[f"{model}_{state}_p"] = 0.8 if prediction else 0.2
    return row


def fixture():
    return [
        make_row(0, surrogate=(0, 1), victim=(0, 1), attachment=True, outer=False),
        make_row(1, surrogate=(0, 1), victim=(0, 0)),
        make_row(2, motion="moving", surrogate=(1, 1), victim=(0, 1)),
        make_row(3, motion="moving", surrogate=(1, 0), victim=(1, 1)),
        make_row(4, motion="moving", surrogate=(0, 0), victim=(0, 0)),
        # Outer rows remain visible but cannot enter the strict-negative ASRs.
        make_row(5, label=1, eligible=True),
        make_row(6, label=-1, eligible=False),
        make_row(7, motion="moving", label=0, eligible=False),
    ]


class DevelopmentPopulationTests(unittest.TestCase):
    def test_population_marginal_counts_and_motion_strata(self):
        summary = population_summary(fixture())
        self.assertEqual(summary["all"], {
            "count": 8, "strict_negative": 5, "positive": 1, "unknown": 1,
            "ineligible": 2, "eligible": 6, "attachment": 1, "outer": 7,
        })
        self.assertEqual(summary["slow"], {
            "count": 4, "strict_negative": 2, "positive": 1, "unknown": 1,
            "ineligible": 1, "eligible": 3, "attachment": 1, "outer": 3,
        })
        self.assertEqual(summary["moving"], {
            "count": 4, "strict_negative": 3, "positive": 0, "unknown": 0,
            "ineligible": 1, "eligible": 3, "attachment": 0, "outer": 4,
        })

    def test_identity_includes_scenario_and_pair_index(self):
        rows = [make_row(0), make_row(0)]
        rows[1]["scenario_id"] = "second-synthetic-scenario"
        self.assertEqual(population_summary(rows)["all"]["count"], 2)

    def test_metadata_and_duplicate_validation_in_all_public_helpers(self):
        invalid = (
            ("scenario_id", ""), ("scenario_id", None),
            ("pair_index", -1), ("pair_index", 1.0), ("pair_index", True),
            ("motion_regime", "parked"), ("motion_regime", None),
            ("true_label", 2), ("true_label", 0.0), ("true_label", False),
            ("eligible", 1), ("is_attachment", "false"), ("is_outer", None),
            ("true_label", -1),  # unknown while eligible=True
        )
        helpers = (population_summary, lambda rows: response_metrics(rows, "surrogate"),
                   compare_models)
        for helper in helpers:
            for field, value in invalid:
                row = make_row(0)
                row[field] = value
                with self.subTest(helper=helper, field=field, value=value):
                    with self.assertRaises(ValueError):
                        helper([row])
            for field in make_row(0):
                if field.startswith(("surrogate_", "victim_")):
                    continue
                row = make_row(0)
                del row[field]
                with self.subTest(helper=helper, missing=field):
                    with self.assertRaisesRegex(ValueError, "missing metadata"):
                        helper([row])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                helper([make_row(0), make_row(0)])
            with self.assertRaisesRegex(ValueError, "mapping"):
                helper([None])


class DevelopmentResponseTests(unittest.TestCase):
    def test_incremental_absolute_and_conditional_have_exact_denominators(self):
        summary = response_metrics(fixture(), "surrogate")
        self.assertEqual(summary["all"]["strict_negative_response"], {
            "count": 5, "incremental_flips": 2, "incremental_asr": 2 / 5,
            "incremental_asr_denominator": 5,
            "absolute_asr_numerator": 3, "absolute_asr_denominator": 5,
            "absolute_asr": 3 / 5,
            "clean_activation_numerator": 2, "clean_activation_denominator": 5,
            "clean_activation": 2 / 5,
            "conditional_flip_numerator": 2, "conditional_flip_denominator": 3,
            "conditional_flip": 2 / 3,
        })
        slow = summary["slow"]["strict_negative_response"]
        self.assertEqual((slow["count"], slow["incremental_flips"]), (2, 2))
        self.assertEqual(slow["incremental_asr"], 1.0)
        moving = summary["moving"]["strict_negative_response"]
        self.assertEqual((moving["count"], moving["incremental_flips"]), (3, 0))
        self.assertEqual(moving["absolute_asr"], 1 / 3)
        self.assertEqual(moving["conditional_flip_denominator"], 1)

    def test_all_rows_include_unknown_positive_ineligible_outer(self):
        rows = fixture()
        rows[0]["surrogate_target_nll"] = 0.25
        rows[6]["surrogate_target_nll"] = 0.75
        response = response_metrics(rows, "surrogate")["all"]["all_row_response"]
        self.assertEqual(response["count"], 8)
        self.assertAlmostEqual(response["mean_clean_p"], 0.35)
        self.assertAlmostEqual(response["mean_trigger_p"], 0.65)
        self.assertAlmostEqual(response["mean_delta_p"], 0.3)
        self.assertEqual(response["clean_activation_count"], 2)
        self.assertEqual(response["trigger_activation_count"], 6)
        self.assertEqual(response["activation_denominator"], 8)
        self.assertEqual(response["clean_activation"], 2 / 8)
        self.assertEqual(response["trigger_activation"], 6 / 8)
        self.assertEqual(response["target_nll"], 0.5)
        self.assertEqual(response["target_nll_count"], 2)

    def test_no_prediction_based_eligibility_and_no_conditional_denominator(self):
        rows = [make_row(0, surrogate=(1, 1)), make_row(1, surrogate=(1, 0))]
        strict = response_metrics(rows, "surrogate")["all"]["strict_negative_response"]
        self.assertEqual(strict["count"], 2)
        self.assertEqual(strict["incremental_asr_denominator"], 2)
        self.assertEqual(strict["incremental_asr"], 0.0)
        self.assertEqual(strict["absolute_asr"], 0.5)
        self.assertEqual(strict["clean_activation"], 1.0)
        self.assertEqual(strict["conditional_flip_denominator"], 0)
        self.assertIsNone(strict["conditional_flip"])

    def test_rows_without_strict_negatives_still_have_all_row_responses(self):
        rows = [make_row(0, label=1), make_row(1, label=-1, eligible=False)]
        response = response_metrics(rows, "surrogate")["all"]
        self.assertEqual(response["all_row_response"]["trigger_activation"], 1.0)
        self.assertEqual(response["strict_negative_response"]["count"], 0)
        for field in ("incremental_asr", "absolute_asr", "clean_activation", "conditional_flip"):
            self.assertIsNone(response["strict_negative_response"][field])

    def test_missing_optional_nll_and_large_finite_nll(self):
        rows = [make_row(0), make_row(1)]
        result = response_metrics(rows, "surrogate")["all"]["all_row_response"]
        self.assertIsNone(result["target_nll"])
        self.assertEqual(result["target_nll_count"], 0)
        for row in rows:
            row["surrogate_target_nll"] = 1e308
        result = response_metrics(rows, "surrogate")["all"]["all_row_response"]
        self.assertEqual(result["target_nll"], 1e308)

    def test_invalid_or_missing_model_values(self):
        fields = ("surrogate_clean_p", "surrogate_trigger_p",
                  "surrogate_clean_pred", "surrogate_trigger_pred")
        invalid_probabilities = (math.nan, math.inf, -math.inf, -0.1, 1.1, True, "0.2", None)
        invalid_predictions = (-1, 2, 0.0, True, math.nan, "1", None)
        for field in fields:
            bad_values = invalid_probabilities if field.endswith("_p") else invalid_predictions
            for value in bad_values:
                row = make_row(0)
                row[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        response_metrics([row], "surrogate")
            row = make_row(0)
            del row[field]
            with self.subTest(missing=field):
                with self.assertRaisesRegex(ValueError, "missing model fields"):
                    response_metrics([row], "surrogate")
        for value in (math.nan, math.inf, -1.0, True, "0.2", None):
            row = make_row(0)
            row["surrogate_target_nll"] = value
            with self.subTest(nll=value), self.assertRaises(ValueError):
                response_metrics([row], "surrogate")
        for model in ("", " ", None, 1):
            with self.subTest(model=model), self.assertRaises(ValueError):
                response_metrics([], model)


class DevelopmentComparisonTests(unittest.TestCase):
    def test_same_strict_row_flip_cross_tabulation(self):
        summary = compare_models(fixture())
        self.assertEqual(summary["all"], {
            "count": 5, "both": 1, "left_only": 1, "right_only": 1, "neither": 2,
        })
        self.assertEqual(summary["slow"], {
            "count": 2, "both": 1, "left_only": 1, "right_only": 0, "neither": 0,
        })
        self.assertEqual(summary["moving"], {
            "count": 3, "both": 0, "left_only": 0, "right_only": 1, "neither": 2,
        })

    def test_custom_models_and_order(self):
        rows = [make_row(0, surrogate=(0, 1), victim=(1, 1))]
        swapped = compare_models(rows, left="victim", right="surrogate")["all"]
        self.assertEqual(swapped["left_only"], 0)
        self.assertEqual(swapped["right_only"], 1)
        renamed = [{key.replace("victim_", "reference_"): value
                    for key, value in rows[0].items()}]
        self.assertEqual(compare_models(renamed, right="reference")["all"]["left_only"], 1)

    def test_unpaired_rows_and_invalid_model_responses_are_rejected(self):
        for model in ("surrogate", "victim"):
            for suffix in ("clean_p", "trigger_p", "clean_pred", "trigger_pred"):
                rows = fixture()
                # Even an ineligible row must have paired valid model outputs.
                del rows[-1][f"{model}_{suffix}"]
                with self.subTest(model=model, suffix=suffix):
                    with self.assertRaisesRegex(ValueError, "missing model fields"):
                        compare_models(rows)
            rows = fixture()
            rows[0][f"{model}_trigger_pred"] = -1
            with self.assertRaises(ValueError):
                compare_models(rows)
            rows[0][f"{model}_trigger_pred"] = 1
            rows[0][f"{model}_clean_p"] = math.inf
            with self.assertRaises(ValueError):
                compare_models(rows)
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare_models([], left="victim", right="victim")


class DevelopmentEmptyAndPurityTests(unittest.TestCase):
    def test_empty_populations_and_motion_strata_are_serializable(self):
        for rows in ([], [make_row(0)]):
            population = population_summary(rows)
            response = response_metrics(rows, "surrogate")
            comparison = compare_models(rows)
            empty_strata = ("all", "slow", "moving") if not rows else ("moving",)
            for stratum in empty_strata:
                self.assertTrue(all(value == 0 for value in population[stratum].values()))
                self.assertTrue(all(value == 0 for value in comparison[stratum].values()))
                for group in response[stratum].values():
                    self.assertTrue(all(value is None or value == 0 for value in group.values()))
                self.assertIsNone(response[stratum]["all_row_response"]["mean_delta_p"])
                self.assertIsNone(response[stratum]["strict_negative_response"]["incremental_asr"])
            json.dumps([population, response, comparison], allow_nan=False)

    def test_pure_helpers_accept_generators_and_do_not_mutate(self):
        rows = fixture()
        before = deepcopy(rows)
        outputs = [population_summary(row for row in rows),
                   response_metrics((row for row in rows), "surrogate"),
                   compare_models(row for row in rows)]
        self.assertEqual(rows, before)
        self.assertEqual(outputs, [population_summary(rows), response_metrics(rows, "surrogate"),
                                   compare_models(rows)])
        json.dumps(outputs, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
