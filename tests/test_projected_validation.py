import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from jcas.core.poison import sha256_file
from jcas.workflows.evaluator import (
    load_expected_target_manifest, training_access_context, validate_public_target_pool,
)
from jcas.workflows import projected_counterfactual_validation as workflow
from jcas.workflows.projected_counterfactual_pilot import evaluate_gate
from tests.test_projected_counterfactual import gate_fixture


def projected_result():
    return {
        "config": {"presentation": "explicit_counterfactual", "checkpoint_metric": "val_loss", "evaluate_test": False},
        "training_protocol": {
            "objective": "balanced_counterfactual_with_clean_reference_gradient_projection_v3",
            "gradient_projection": {"enabled": True}, "checkpoint_data": "clean_validation_only",
        },
    }


class TrainingAccessTests(unittest.TestCase):
    def test_projected_training_and_validation_reuse_are_disclosed(self):
        metadata = training_access_context(projected_result(), split="val", model_role="victim")
        self.assertEqual(metadata["threat_model"], "training_procedure_control")
        self.assertTrue(metadata["victim_structure_parameters_or_gradients_used_by_attacker"])
        self.assertTrue(metadata["clean_validation_used_for_checkpoint_selection"])
        self.assertFalse(metadata["evaluation_split_not_used_for_training_checkpoint_selection"])
        self.assertFalse(metadata["training_asr_checkpoint_selection"])

    def test_ordinary_bce_retains_data_only_semantics(self):
        result = projected_result()
        result["config"].pop("presentation")
        result["training_protocol"] = {
            "objective": "ordinary_binary_cross_entropy_symmetric_pair_labels",
            "checkpoint_data": "clean_validation_only",
        }
        metadata = training_access_context(result, split="test", model_role="victim")
        self.assertEqual(metadata["threat_model"], "data_only_poisoning")
        self.assertFalse(metadata["victim_structure_parameters_or_gradients_used_by_attacker"])
        self.assertTrue(metadata["evaluation_split_not_used_for_training_checkpoint_selection"])

    def test_reference_optimization_is_not_attacker_access(self):
        metadata = training_access_context(projected_result(), split="val", model_role="clean_reference")
        self.assertEqual(metadata["threat_model"], "clean_reference_response_baseline")
        self.assertFalse(metadata["victim_structure_parameters_or_gradients_used_by_attacker"])

    def test_missing_provenance_is_unknown(self):
        metadata = training_access_context({}, split="val", model_role="victim")
        self.assertIsNone(metadata["victim_structure_parameters_or_gradients_used_by_attacker"])
        self.assertIsNone(metadata["evaluation_split_not_used_for_training_checkpoint_selection"])


def pool_fixture():
    return pd.DataFrame({
        "scenario_id": ["scene-1", "scene-2", "scene-3"],
        "edge_id": [0, 2, 4], "reverse_edge_id": [1, 3, 5],
        "allocation_policy": [workflow.ALLOCATION] * 3,
        "allocation_alpha": [0.5] * 3,
        "motion_regime": ["moving_ge_0p5", "moving_ge_0p5", "slow_lt_0p5"],
        "src_track_id": ["001", "002", "003"],
        "dst_track_id": ["101", "102", "103"],
        "min_endpoint_speed_mps": [1.1, 2.2, 0.1],
    })


class PublicPoolTests(unittest.TestCase):
    def test_order_invariant_pool_and_float_roundtrip(self):
        pool = pool_fixture()
        generated = pool.iloc[::-1].copy()
        generated["min_endpoint_speed_mps"] += 1e-15
        validate_public_target_pool(generated, pool)

    def test_equal_counts_with_wrong_identity_regime_or_allocation_fail(self):
        for column, value in [("edge_id", 7), ("motion_regime", "slow_lt_0p5"), ("allocation_alpha", 0.4)]:
            pool = pool_fixture()
            generated = pool.copy()
            generated.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_public_target_pool(generated, pool)

    def test_truncated_duplicate_or_new_column_fail(self):
        pool = pool_fixture()
        for generated in [pool.iloc[:2], pd.concat([pool.iloc[:2], pool.iloc[:1]]), pool.assign(new_column=1)]:
            with self.assertRaises(ValueError):
                validate_public_target_pool(generated, pool)

    def test_nonfinite_float_diagnostics_fail(self):
        pool = pool_fixture()
        pool.loc[0, "min_endpoint_speed_mps"] = float("inf")
        with self.assertRaises(ValueError):
            validate_public_target_pool(pool, pool)

    def test_hash_binding_and_track_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.csv"
            pool_fixture().to_csv(path, index=False)
            loaded = load_expected_target_manifest(path, sha256_file(path))
            self.assertEqual(loaded.src_track_id.iloc[0], "001")
            with self.assertRaises(ValueError):
                load_expected_target_manifest(path, "0" * 64)


def metric_fixture():
    pool = pool_fixture()
    scores = pool.copy()
    scores["clean_pair_probability"] = [0.1, 0.7, 0.2]
    scores["triggered_pair_probability"] = [0.8, 0.9, 0.1]
    scores["common_threshold"] = 0.5
    records = []
    for index, scenario in enumerate(pool.scenario_id):
        for subset in ["adjacent_negative", "nonincident_negative", "adjacent_positive"]:
            total = [1, 9, 10][index]
            n01 = int(index == 0 and subset != "adjacent_positive")
            records.append({
                "scenario_id": scenario, "threshold_kind": "common", "threshold": 0.5,
                "subset": subset, "n00": total - n01, "n01": n01,
                "n10": 0, "n11": 0, "total": total,
            })
    return scores, pd.DataFrame(records), pool


class ValidationMetricsTests(unittest.TestCase):
    def test_subgroup_target_and_edge_weighted_collateral(self):
        summary = workflow.summarize_common_scores(*metric_fixture(), threshold=0.5)
        self.assertAlmostEqual(summary["all"]["incremental_flip_rate_all_targets"], 1 / 3)
        moving = summary["moving_ge_0p5"]
        self.assertAlmostEqual(moving["incremental_flip_rate_all_targets"], 0.5)
        self.assertAlmostEqual(moving["conditional_flip_rate"], 1.0)
        self.assertAlmostEqual(moving["collateral"]["adjacent_negative"]["incremental_positive_rate"], 0.1)

    def test_duplicate_targets_wrong_threshold_and_missing_collateral_fail(self):
        scores, transitions, pool = metric_fixture()
        wrong_threshold = transitions.assign(threshold=0.6)
        for s, t in [(pd.concat([scores.iloc[:2], scores.iloc[:1]]), transitions),
                     (scores, wrong_threshold), (scores, transitions.iloc[:-1])]:
            with self.assertRaises(ValueError):
                workflow.summarize_common_scores(s, t, pool, 0.5)

    def test_noninteger_or_inconsistent_transition_counts_fail(self):
        for column, value in [("n01", 0.5), ("total", 300)]:
            scores, transitions, pool = metric_fixture()
            transitions[column] = transitions[column].astype(float)
            transitions.loc[0, column] = value
            with self.assertRaises(ValueError):
                workflow.summarize_common_scores(scores, transitions, pool, 0.5)


def protocol_fixture():
    return {
        "evaluation": {"graph_dir": "/synthetic/graphs"},
        "assets": {
            "candidate_checkpoint": {"path": "/synthetic/victim.pt"},
            "reference_checkpoint": {"path": "/synthetic/reference.pt"},
            "graph_manifest": {"path": "/synthetic/manifest.csv"},
            "reference_result": {"path": "/synthetic/reference.json"},
            "public_targets": {"path": "/synthetic/targets.csv", "sha256": "a" * 64},
        },
    }


class ValidationWorkflowTests(unittest.TestCase):
    def test_gate_must_pass_recomputation_and_exact_thresholds(self):
        result, audit = gate_fixture()
        gate = evaluate_gate(result, audit)
        workflow.verify_passing_gate(gate, result, audit)
        changed = copy.deepcopy(gate)
        changed["checks"]["moving_incremental_asr"]["threshold"] = 0.1
        with self.assertRaises(ValueError):
            workflow.verify_passing_gate(changed, result, audit)
        result["val_pair_metrics"]["auc"] = 0.978
        with self.assertRaises(ValueError):
            workflow.verify_passing_gate(gate, result, audit)

    def test_both_commands_use_common_pool_threshold_and_val_only(self):
        protocol = protocol_fixture()
        for role in ("clean_reference", "victim"):
            command = workflow.evaluation_command(protocol, Path("out.json"), role, "cpu")
            self.assertEqual(command[2], "jcas.workflows.evaluator")
            self.assertEqual(command[command.index("--split") + 1], "val")
            self.assertEqual(command[command.index("--expected-target-manifest-sha256") + 1], "a" * 64)
            self.assertEqual(command[command.index("--clean-reference-result") + 1], "/synthetic/reference.json")
            for flag in ("--force", "--max-targets", "--max-graphs", "--common-threshold"):
                self.assertNotIn(flag, command)

    def test_check_only_does_not_run_evaluation_or_finalize(self):
        with patch.object(workflow, "preflight", return_value=protocol_fixture()), \
                patch.object(workflow, "run_command") as run, \
                patch.object(workflow, "finalize_results") as finalize, \
                patch("sys.argv", ["validation", "--check-only"]), \
                contextlib.redirect_stdout(io.StringIO()):
            workflow.main()
        run.assert_not_called()
        finalize.assert_not_called()

    def test_preflight_failure_prevents_stage_execution(self):
        with patch.object(workflow, "preflight", side_effect=ValueError("gate STOP")), \
                patch.object(workflow, "run_command") as run, \
                patch("sys.argv", ["validation"]):
            with self.assertRaises(ValueError):
                workflow.main()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
