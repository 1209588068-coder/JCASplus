import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jcas.workflows import random_pair_control as pilot
from tests.test_projected_validation import metric_fixture


def candidate(scene, src, dst, speed):
    regime = pilot.selection.motion_regime(speed)
    return {"scenario_id": scene, "src": src, "dst": dst,
            "src_track_id": f"track-{src}", "dst_track_id": f"track-{dst}",
            "orientation_policy": "lower_destination_mean_speed_v1",
            "orientation_destination_mean_speed_mps": speed,
            "orientation_candidate_count": 2, "src_endpoint_speed_mps": speed + 1,
            "dst_endpoint_speed_mps": speed, "min_endpoint_speed_mps": speed,
            "motion_regime": regime,
            "selection_priority_sha256": pilot.selection.pair_priority(pilot.SEED, scene, src, dst, regime)}


def selection_fixture():
    all_candidates, originals, selected, graphs = [], [], [], []
    for index in range(4):
        scene = f"scene-{index}"
        choices = [candidate(scene, 0, 1, 0.1), candidate(scene, 0, 2, 2.0), candidate(scene, 1, 2, 1.0)]
        choices = [{**row, "graph_sha256": "a" * 64, "split": "train"} for row in choices]
        slow = pilot.selection.choose_motion_candidate(choices, preferred_regime=pilot.selection.SLOW_MOTION_REGIME)
        chosen = pilot.choose_random_candidate(choices)
        constant = {"displacement_m": 0.2, "perturb_window": 10, "ramp_style": "minimum_jerk",
                    "velocity_mode": "residual", "allocation_alpha": 0.5, "allocation_policy": "fixed_symmetric_biend_v1",
                    "poison_label": 1, "seed": pilot.SEED, "eligible_pair_count": len(choices),
                    "base_src": 0, "base_dst": 1, "base_motion_regime": "slow_lt_0p5",
                    "label_config_hash": pilot.selection.label_config_hash(pilot.RiskLabelConfig()),
                    "label_mode": "dynamic_risk", "require_strict_label": True, "label_unit": pilot.selection.PAIR_LABEL_UNIT}
        original = {**slow, **constant, "preferred_motion_regime": "slow_lt_0p5", "preferred_pair_available": True}
        originals.append(original)
        selected.append({**chosen, **constant, "slow_reference_src": slow["src"], "slow_reference_dst": slow["dst"],
                         "pair_changed_from_slow": pilot.pair_identity(chosen) != pilot.pair_identity(slow),
                         "pair_changed_from_v6": pilot.pair_identity(chosen)[1:] != (0, 1)})
        all_candidates.extend(choices)
        graphs.append({"scenario_id": scene, "graph_sha256": "a" * 64, "graph_path": f"{scene}.npz", "split": "train"})
    context = {"slow": pd.DataFrame(originals), "graph_rows": pd.DataFrame(graphs),
               "slow_metadata": {}, "assets": {"graph_manifest": {"path": "/dummy/manifest.csv"}},
               "sources": {}, "threshold": 0.5}
    return pd.DataFrame(selected), pd.DataFrame(all_candidates), context


def passing_groups():
    groups = pilot.summarize_common_scores(*metric_fixture(), threshold=0.5)
    for group in groups.values():
        group["clean_pair_metrics"] = {"auc": 0.986}
        group["collateral"]["adjacent_negative"]["incremental_positive_rate"] = 0.01
    groups["all"]["incremental_flip_rate_all_targets"] = 0.62
    groups["slow_lt_0p5"]["incremental_flip_rate_all_targets"] = 0.82
    groups["slow_lt_0p5"]["clean_activation_rate"] = 0.02
    return groups


class SelectionTests(unittest.TestCase):
    def test_hash_minimum_is_order_invariant_and_uses_no_global_rng(self):
        _, candidates, _ = selection_fixture()
        rows = candidates[candidates.scenario_id.eq("scene-0")].to_dict("records")
        state = np.random.get_state()
        chosen = pilot.choose_random_candidate(rows)
        self.assertEqual(chosen, pilot.choose_random_candidate(list(reversed(rows))))
        self.assertEqual(chosen["selection_priority_sha256"], min(r["selection_priority_sha256"] for r in rows))
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_all_eligible_random_can_select_moving_when_slow_is_available(self):
        _, candidates, context = selection_fixture()
        observed = []
        for scene, rows in candidates.groupby("scenario_id"):
            observed.append(pilot.choose_random_candidate(rows.to_dict("records"))["motion_regime"])
        self.assertIn("moving_ge_0p5", observed)
        self.assertTrue(context["slow"].motion_regime.eq("slow_lt_0p5").all())

    def test_only_slow_pool_is_exactly_same_as_original_policy(self):
        rows = [candidate("scene", 0, 1, 0.1), candidate("scene", 1, 2, 0.2)]
        self.assertEqual(pilot.choose_random_candidate(rows), pilot.selection.choose_motion_candidate(rows, preferred_regime="slow_lt_0p5"))

    def test_duplicate_reverse_empty_mixed_and_wrong_hash_rejected(self):
        row = candidate("scene", 0, 1, 1.0)
        for rows in [[], [row, row], [row, {**row, "src": 1, "dst": 0}],
                     [row, candidate("other", 1, 2, 1.0)], [{**row, "selection_priority_sha256": "0" * 64}],
                     [{**row, "motion_regime": "slow_lt_0p5"}], [{**row, "min_endpoint_speed_mps": float("nan")}]]:
            with self.assertRaises(ValueError):
                pilot.choose_random_candidate(rows)

    def test_selection_audit_preserves_scenes_and_exact_parent_hash(self):
        frame, candidates, context = selection_fixture()
        pilot.check_selection(frame, candidates, context)
        pilot.check_selection(frame.iloc[::-1], candidates.iloc[::-1], context)
        self.assertEqual(pilot.selection_summary(frame, candidates)["scenarios"], 4)

    def test_tampered_candidates_trigger_split_or_selected_pair_rejected(self):
        frame, candidates, context = selection_fixture()
        for bad in [frame.assign(displacement_m=0.3), frame.assign(split="val"), frame.iloc[:-1], frame.assign(src=99)]:
            with self.assertRaises(ValueError):
                pilot.check_selection(bad, candidates, context)
        for bad in [candidates.iloc[:-1], candidates.assign(split="test"), candidates.assign(graph_sha256="b" * 64)]:
            with self.assertRaises(ValueError):
                pilot.check_selection(frame, bad, context)

    def test_generation_reads_only_train_and_recomputes_allocation(self):
        _, candidates, context = selection_fixture()
        context["graph_rows"] = pd.concat([context["graph_rows"], pd.DataFrame([{
            "scenario_id": "forbidden", "split": "test", "graph_sha256": "b" * 64, "graph_path": "forbidden.npz"}])])
        allocations = {key: 7 for key in ["allocation_total_feature_energy", "allocation_incident_edge_energy",
                                         "allocation_endpoint_node_energy", "allocation_candidate_count", "allocation_non_target_incident_edges"]}
        loaded = []
        def resolve(row, directory):
            self.assertEqual(row.split, "train")
            loaded.append(row.scenario_id)
            return Path(row.graph_path)
        def enumerate_candidates(graph, groups, *, scenario_id, selection_seed):
            self.assertEqual(selection_seed, pilot.SEED)
            return candidates[candidates.scenario_id.eq(scenario_id)].to_dict("records")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pilot.selection, "resolve_verified_graph_path", side_effect=resolve))
            stack.enter_context(patch.object(pilot.np, "load", side_effect=lambda *a, **k: contextlib.nullcontext({})))
            stack.enter_context(patch.object(pilot.selection, "labels_for_graph", return_value={}))
            eligible = stack.enter_context(patch.object(pilot.selection, "eligible_negative_pair_groups", return_value=[]))
            stack.enter_context(patch.object(pilot.selection, "_scenario_candidates", side_effect=enumerate_candidates))
            stack.enter_context(patch.object(pilot.selection, "fixed_symmetric_bi_endpoint_allocation", return_value=(0.5, allocations)))
            generated, audit = pilot.generate_tables(context)
        self.assertEqual(set(loaded), set(context["slow"].scenario_id))
        self.assertTrue(generated.allocation_total_feature_energy.eq(7).all())
        self.assertNotIn("preferred_motion_regime", generated)
        for call in eligible.call_args_list:
            self.assertTrue(call.kwargs["require_bi_endpoint_contiguous"])
            self.assertTrue(call.kwargs["require_strict_label"])
            self.assertEqual(call.kwargs["perturb_window"], 10)
        pilot.check_selection(generated, audit, context)


class ContractTests(unittest.TestCase):
    def test_plan_prepare_and_reload_and_tampered_gate_rejection(self):
        frame, candidates, context = selection_fixture()
        context["slow_metadata"] = {"label_config_hash": pilot.selection.label_config_hash(pilot.RiskLabelConfig()),
                                    "orientation_policy": "lower_destination_mean_speed_v1", "allocation_policy": "fixed_symmetric_biend_v1",
                                    "graph_manifest_sha256": "c" * 64, "training_manifest_eligible": True,
                                    "requested_poison_scenario_rate": 0.05,
                                    "model_outputs_used": False, "original_test_used": False, "original_validation_used": False}
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "ROOT", Path(directory)):
            context["assets"] = {"slow_manifest": {"path": str(Path(directory) / "slow.csv"), "sha256": "d" * 64},
                                  "graph_manifest": {"path": str(Path(directory) / "graphs.csv"), "sha256": "c" * 64}}
            with patch.object(pilot, "generate_tables", return_value=(frame, candidates)), patch.object(pilot, "verify_bound"):
                path = pilot.prepare(context, Path(directory) / "plan")
            loaded = pilot.checked_plan(path, context)
            self.assertEqual(loaded["selection_objective"], pilot.OBJECTIVE)
            self.assertEqual(loaded["training"]["epochs"], 50)
            loaded["reference_gates"]["slow_incremental_asr"][1] = 0.1
            with patch.object(pilot, "read_json", side_effect=lambda p: loaded if Path(p) == path else pilot_json(p)):
                with self.assertRaises(ValueError):
                    pilot.checked_plan(path, context)

    def test_no_overwrite_or_frozen_or_external_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "ROOT", Path(directory)):
            allowed = Path(directory) / "record/reproduction/new"
            self.assertEqual(pilot.new_directory(allowed), allowed)
            for bad in [Path(directory) / "record/v6/run", Path(directory), Path(directory).parent]:
                with self.assertRaises(ValueError):
                    pilot.new_directory(bad)
            with self.assertRaises(FileExistsError):
                pilot.new_directory(allowed)
            with self.assertRaises(ValueError):
                pilot.asset_path(Path(directory) / "record/v6/test/result.json")

    def test_commands_ordinary_bce_and_exact_public_validation_only(self):
        context = {"assets": {"graph_manifest": {"path": "/x/manifest.csv"}, "reference_result": {"path": "/x/clean.json"},
                               "baseline_target_manifest": {"path": "/x/public.csv", "sha256": "a" * 64}}}
        train = pilot.training_command(context, Path("/x/plan.json"), Path("/x/output"), "cpu")
        self.assertIn("jcas.workflows.trainer", train)
        self.assertIn("relative_motion_residual_dct8_v1", train)
        for forbidden in ["--evaluate-test", "--target-weight", "--gradient-projection", "--max-train-graphs", "--require-strict-crossfit-manifest"]:
            self.assertNotIn(forbidden, train)
        evaluate = pilot.evaluation_command(context, {"path": "/x/best_model.pt"}, Path("/x/victim.json"), "cpu")
        self.assertEqual(evaluate[evaluate.index("--split") + 1], "val")
        self.assertEqual(evaluate[evaluate.index("--expected-target-manifest-sha256") + 1], "a" * 64)
        self.assertNotIn("--max-targets", evaluate)

    def test_check_only_never_writes_loads_graphs_or_runs_jobs(self):
        with patch.object(pilot, "preflight", return_value={"slow": range(5776)}), \
                patch.object(pilot, "new_directory", side_effect=AssertionError), \
                patch.object(pilot, "run_command", side_effect=AssertionError), \
                patch.object(pilot.np, "load", side_effect=AssertionError), contextlib.redirect_stdout(io.StringIO()) as output:
            pilot.main(["--check-only"])
        self.assertIn("PREFLIGHT_PASS", output.getvalue())

    def test_invalid_resume_fails_before_preflight(self):
        with patch.object(pilot, "preflight", side_effect=AssertionError), contextlib.redirect_stderr(io.StringIO()):
            for args in [["--result", "x"], ["--prepare-only", "--check-only"], ["--prepare-only", "--plan", "x"]]:
                with self.assertRaises(SystemExit):
                    pilot.main(args)

    def test_csv_roundtrip_preserves_common_threshold_exactly(self):
        value = 0.4962041974067688
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            pd.DataFrame({"common_threshold": [value], "src_track_id": ["001"]}).to_csv(path, index=False)
            result = pilot.read_table(path)
        self.assertEqual(result.common_threshold.iloc[0], value)
        self.assertEqual(result.src_track_id.iloc[0], "001")

    def test_training_result_binding_rejects_old_manifests_incomplete_or_modified_objectives(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "ROOT", Path(directory)):
            root = Path(directory)
            plan_path = root / "plan.json"
            for name in ("poison.csv", "poison.csv.metadata.json", "best_model.pt"):
                pilot.write_json(root / name, {})
            context = {"assets": {"graph_manifest": {"sha256": "a" * 64}}}
            result = {"config": {**pilot.TRAINING, "poison_manifest": str(root / "poison.csv")},
                      "training_protocol": {"objective": "ordinary_binary_cross_entropy_symmetric_pair_labels", "initialization": "random"},
                      "history": [{"epoch": n} for n in range(1, 51)], "graph_manifest": {"sha256": "a" * 64},
                      "poison_manifest": {"sha256": pilot.sha256_file(root / "poison.csv"),
                                          "metadata_sha256": pilot.sha256_file(root / "poison.csv.metadata.json"),
                                          "rows": 5776, "applied_train_rows": 5776, "selection_objective": pilot.OBJECTIVE},
                      "train_stats": {"poisoned_graphs": 5776}, "val_stats": {"poisoned_graphs": 0},
                      "checkpoint": {"sha256": pilot.sha256_file(root / "best_model.pt")}}
            with patch.object(pilot, "read_json", return_value=result):
                pilot.validate_training(root / "result.json", context, plan_path)
            variants = []
            for section, key, value in [("config", "poison_manifest", str(root / "old.csv")),
                                        ("config", "evaluate_test", True), ("config", "max_train_graphs", 5),
                                        ("training_protocol", "objective", "counterfactual"),
                                        ("training_protocol", "initialization", "warmstart"),
                                        ("poison_manifest", "applied_train_rows", 1), ("checkpoint", "sha256", "0" * 64)]:
                bad = copy.deepcopy(result)
                bad[section][key] = value
                variants.append(bad)
            bad = copy.deepcopy(result)
            bad["history"] = bad["history"][:-1]
            variants.append(bad)
            for bad in variants:
                with patch.object(pilot, "read_json", return_value=bad), self.assertRaises(ValueError):
                    pilot.validate_training(root / "result.json", context, plan_path)

    def test_wrong_resume_result_rejected_before_creating_output(self):
        with patch.object(pilot, "preflight", return_value={}), patch.object(pilot, "checked_plan"), \
                patch.object(pilot, "validate_training", side_effect=ValueError("wrong result")), \
                patch.object(pilot, "new_directory", side_effect=AssertionError):
            with self.assertRaises(ValueError):
                pilot.main(["--plan", str(pilot.ROOT / "plan.json"), "--result", str(pilot.ROOT / "result.json")])

    def test_stop_exit_two_and_no_automatic_followup(self):
        with patch.object(pilot, "preflight", return_value={}), patch.object(pilot, "checked_plan"), \
                patch.object(pilot, "new_directory", return_value=Path("/dummy")), \
                patch.object(pilot, "execute", return_value="STOP") as execute:
            with self.assertRaises(SystemExit) as error:
                pilot.main(["--plan", str(pilot.ROOT / "plan.json"), "--device", "cpu"])
        self.assertEqual(error.exception.code, 2)
        execute.assert_called_once()


pilot_json = pilot.read_json


class ReportingTests(unittest.TestCase):
    def test_reference_gates_keep_original_low_speed_thresholds_and_fail_closed(self):
        groups = passing_groups()
        self.assertEqual(pilot.reference_gate(groups)["status"], "PASS")
        for key, regime, field, value in [
            ("slow_incremental_asr", "slow_lt_0p5", "incremental_flip_rate_all_targets", 0.799),
            ("overall_incremental_asr", "all", "incremental_flip_rate_all_targets", 0.599),
            ("slow_clean_activation", "slow_lt_0p5", "clean_activation_rate", 0.026),
            ("slow_incremental_asr", "slow_lt_0p5", "incremental_flip_rate_all_targets", float("nan")),
        ]:
            changed = copy.deepcopy(groups)
            changed[regime][field] = value
            gate = pilot.reference_gate(changed)
            self.assertEqual(gate["status"], "STOP")
            self.assertFalse(gate["checks"][key]["passed"])
        for threshold, expected in [(0.9849, "PASS"), (0.9848, "STOP")]:
            groups["all"]["clean_pair_metrics"]["auc"] = threshold
            self.assertEqual(pilot.reference_gate(groups)["status"], expected)
        groups = passing_groups()
        groups["all"]["collateral"]["adjacent_negative"]["incremental_positive_rate"] = 0.015001
        self.assertEqual(pilot.reference_gate(groups)["status"], "STOP")

    def test_same_seed_table_keeps_overall_and_low_speed_separate(self):
        groups = passing_groups()
        rows = pilot.comparison_rows(groups, groups)
        self.assertEqual(len(rows), 6)
        self.assertEqual(set(rows.seed), {20260621})
        current = rows[rows.model_role.eq("all_eligible_random")].set_index("regime")
        self.assertEqual(current.loc["all", "incremental_asr"], 0.62)
        self.assertEqual(current.loc["slow_lt_0p5", "incremental_asr"], 0.82)

    def test_e2e_orchestration_and_resume_stop_after_validation_even_when_gate_fails(self):
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as directory, patch.object(pilot, "ROOT", Path(directory)):
                root = Path(directory)
                plan_path = root / "plan.json"
                pilot.write_json(plan_path, {})
                output = root / "output"
                output.mkdir()
                result_path = root / "completed/result.json"
                result_path.parent.mkdir()
                pilot.write_json(result_path, {})
                checkpoint_path = result_path.parent / "best_model.pt"
                pilot.write_json(checkpoint_path, {})
                checkpoint = pilot.bind(checkpoint_path)
                context = {"assets": {"baseline_validation": {"path": str(root / "old.json")},
                                       "baseline_result": {}, "baseline_checkpoint": {}}, "threshold": 0.5}
                groups = passing_groups()
                groups["all"]["incremental_flip_rate_all_targets"] = 0.1
                commands = []
                def execute_job(command, log):
                    commands.append(command)
                    if command[0] == "training":
                        path = output / "train/run/result.json"
                        path.parent.mkdir(parents=True)
                        pilot.write_json(path, {})
                    else:
                        pilot.write_json(output / "validation/victim.json", {})
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(pilot, "checked_plan", return_value={"selection": {}}))
                    stack.enter_context(patch.object(pilot, "validate_training", return_value=({"history": [{"epoch": 42, "val_loss": 0.1}]}, checkpoint)))
                    stack.enter_context(patch.object(pilot, "verify_bound"))
                    stack.enter_context(patch.object(pilot, "training_command", return_value=["training"]))
                    stack.enter_context(patch.object(pilot, "evaluation_command", return_value=["validation"]))
                    stack.enter_context(patch.object(pilot, "run_command", side_effect=execute_job))
                    stack.enter_context(patch.object(pilot, "summarize_evaluation", return_value={"groups": groups, "assets": {}}))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    status = pilot.execute(context, plan_path, output, "cpu", result_path if resume else None)
                self.assertEqual(status, "STOP")
                self.assertEqual(commands, [["validation"]] if resume else [["training"], ["validation"]])
                summary = pilot.read_json(output / "summary.json")
                self.assertEqual(summary["status"], "VALIDATION_COMPLETE_REVIEW_REQUIRED")
                self.assertFalse(summary["scope"]["test_accessed"])
                self.assertFalse(summary["scope"]["automatic_replication"])
                self.assertEqual(len(pd.read_csv(output / "comparison.csv")), 6)


if __name__ == "__main__":
    unittest.main()
