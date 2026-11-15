"""Synthetic table tests only; no data/model IO, inference, or training."""
from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from spear_current.reverse_metrics import reverse_response_metrics


COUNTS = ("n", "clean_detected", "baseline_misses", "induced_misses_1to0",
          "corrected_misses_0to1", "triggered_misses")
RATES = ("baseline_fnr", "absolute_suppression", "incremental_asr",
         "conditional_suppression", "net_fnr_change", "clean_recall_ceiling")


def row(index=0, predictions=(1, 0), *, motion="slow", model="victim"):
    result = dict(scenario_id="synthetic-scene", pair_index=index, motion_regime=motion,
                  true_label=1, eligible=True, is_attachment=False, is_outer=False)
    for state, prediction in zip(("clean", "trigger"), predictions):
        result[f"{model}_{state}_pred"] = prediction
        result[f"{model}_{state}_p"] = .8 if prediction else .2
    return result


def transitions():
    return [row(0, (1, 0)), row(1, (1, 1)), row(2, (0, 1)),
            row(3, (1, 0), motion="moving"), row(4, (0, 0), motion="moving"),
            row(5, (1, 1), motion="moving")]


class ReverseResponseTests(unittest.TestCase):
    def test_exact_counts_strata_and_miss_repair_decomposition(self):
        result = reverse_response_metrics(transitions(), "victim")
        self.assertEqual(result["all"], dict(n=6, clean_detected=4, baseline_misses=2,
            induced_misses_1to0=2, corrected_misses_0to1=1, triggered_misses=3,
            baseline_fnr=2 / 6, absolute_suppression=3 / 6, incremental_asr=2 / 6,
            conditional_suppression=2 / 4, net_fnr_change=1 / 6, clean_recall_ceiling=4 / 6))
        self.assertEqual(result["slow"], dict(n=3, clean_detected=2, baseline_misses=1,
            induced_misses_1to0=1, corrected_misses_0to1=1, triggered_misses=1,
            baseline_fnr=1 / 3, absolute_suppression=1 / 3, incremental_asr=1 / 3,
            conditional_suppression=1 / 2, net_fnr_change=0, clean_recall_ceiling=2 / 3))
        self.assertEqual(result["moving"], dict(n=3, clean_detected=2, baseline_misses=1,
            induced_misses_1to0=1, corrected_misses_0to1=0, triggered_misses=2,
            baseline_fnr=1 / 3, absolute_suppression=2 / 3, incremental_asr=1 / 3,
            conditional_suppression=1 / 2, net_fnr_change=1 / 3, clean_recall_ceiling=2 / 3))
        for group in result.values():
            self.assertEqual(group["triggered_misses"], group["baseline_misses"]
                + group["induced_misses_1to0"] - group["corrected_misses_0to1"])
            self.assertAlmostEqual(group["absolute_suppression"] - group["baseline_fnr"],
                                   group["net_fnr_change"])
        for field in COUNTS:
            self.assertEqual(result["all"][field], result["slow"][field] + result["moving"][field])

    def test_clean_all_missed_is_not_new_attack_success_and_can_be_repaired(self):
        for predictions, absolute, change in (((0, 0), 1, 0), ((0, 1), 0, -1)):
            with self.subTest(predictions=predictions):
                result = reverse_response_metrics([row(predictions=predictions)], "victim")["all"]
                self.assertEqual(result["n"], 1)
                self.assertEqual(result["baseline_misses"], 1)
                self.assertEqual(result["baseline_fnr"], 1)
                self.assertEqual(result["incremental_asr"], 0)
                self.assertEqual(result["clean_recall_ceiling"], 0)
                self.assertIsNone(result["conditional_suppression"])
                self.assertEqual(result["absolute_suppression"], absolute)
                self.assertEqual(result["net_fnr_change"], change)

    def test_fixed_denominator_and_clean_recall_ceiling_even_if_all_detections_flip(self):
        rows = [row(0, (1, 0)), row(1, (0, 0)), row(2, (0, 1)), row(3, (1, 0))]
        metric = reverse_response_metrics(rows, "victim")["all"]
        self.assertEqual(metric["n"], 4)
        self.assertEqual(metric["incremental_asr"], .5)
        self.assertEqual(metric["clean_recall_ceiling"], .5)
        self.assertEqual(metric["conditional_suppression"], 1)
        self.assertEqual(metric["absolute_suppression"], .75)
        self.assertEqual(metric["net_fnr_change"], .25)

    def test_empty_population_and_empty_stratum_have_zero_counts_undefined_rates(self):
        for rows, empty in (([], ("all", "slow", "moving")), ([row()], ("moving",))):
            result = reverse_response_metrics(rows, "victim")
            self.assertEqual(tuple(result), ("all", "slow", "moving"))
            for stratum in empty:
                self.assertEqual({key: result[stratum][key] for key in COUNTS}, dict.fromkeys(COUNTS, 0))
                self.assertEqual({key: result[stratum][key] for key in RATES}, dict.fromkeys(RATES))
            json.dumps(result, allow_nan=False)

    def test_all_rows_require_strict_positive_ground_truth_and_valid_metadata(self):
        invalid = {
            "true_label": (-1, 0, 2, 1.0, True, "1", None),
            "eligible": (False, 1, "True", None),
            "scenario_id": ("", " ", 1, None),
            "pair_index": (-1, 1.0, True, "0", None),
            "motion_regime": ("stationary", 0, "", None),
            "is_attachment": (1, "False", None),
            "is_outer": (0, "False", None),
        }
        for field, values in invalid.items():
            for value in values:
                bad = row()
                bad[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    reverse_response_metrics([bad], "victim")
        for field in ("scenario_id", "pair_index", "motion_regime", "true_label",
                      "eligible", "is_attachment", "is_outer"):
            bad = row()
            del bad[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "missing metadata fields"):
                reverse_response_metrics([bad], "victim")

    def test_duplicate_pair_keys_rejected_but_same_pair_index_in_other_scene_allowed(self):
        with self.assertRaisesRegex(ValueError, "duplicate pair identity"):
            reverse_response_metrics([row(), row()], "victim")
        second = row()
        second["scenario_id"] = "second-scene"
        self.assertEqual(reverse_response_metrics([row(), second], "victim")["all"]["n"], 2)
        for rows in (None, 3, [None], ["row"], [3]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                reverse_response_metrics(rows, "victim")

    def test_missing_and_invalid_prediction_and_probability_fields_rejected(self):
        for suffix in ("clean_pred", "trigger_pred", "clean_p", "trigger_p"):
            field = "victim_" + suffix
            values = ((-1, 2, 0.0, True, "0", None, math.nan) if suffix.endswith("pred") else
                      (-.1, 1.1, math.nan, math.inf, -math.inf, True, "0.2", None))
            for value in values:
                bad = row()
                bad[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    reverse_response_metrics([bad], "victim")
            bad = row()
            del bad[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "missing model fields"):
                reverse_response_metrics([bad], "victim")

    def test_valid_probability_boundaries_and_predictions_are_not_rethresholded(self):
        for probability in (0, .5, 1):
            item = row(predictions=(1, 0))
            item["victim_clean_p"] = item["victim_trigger_p"] = probability
            # The input's paired argmax decisions are authoritative; probabilities
            # are validated, but this metric helper introduces no new threshold.
            with self.subTest(probability=probability):
                self.assertEqual(reverse_response_metrics([item], "victim")["all"]["induced_misses_1to0"], 1)

    def test_model_names_are_explicit_and_custom_prefixes_are_supported(self):
        for model in ("", " ", None, 1, True):
            with self.subTest(model=model), self.assertRaises(ValueError):
                reverse_response_metrics([], model)
        for model in ("victim", "surrogate", "clean_reference", "custom_model"):
            with self.subTest(model=model):
                self.assertEqual(reverse_response_metrics([row(model=model)], model)["all"]["incremental_asr"], 1)
        with self.assertRaisesRegex(ValueError, "missing model fields"):
            reverse_response_metrics([row(model="surrogate")], "victim")

    def test_pure_iterable_helper_leaves_inputs_unchanged_and_does_no_file_io(self):
        rows = transitions()
        rows[0].update(split_group_id="synthetic-group", src_track_id="track-a", dst_track_id="track-b")
        rows[0]["is_attachment"] = True
        rows[1]["is_outer"] = True
        before = deepcopy(rows)
        with patch("builtins.open", side_effect=AssertionError("file IO forbidden")):
            result = reverse_response_metrics((item for item in rows), "victim")
            self.assertEqual(result, reverse_response_metrics(rows, "victim"))
        self.assertEqual(rows, before)
        self.assertEqual(result["all"]["n"], 6)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
