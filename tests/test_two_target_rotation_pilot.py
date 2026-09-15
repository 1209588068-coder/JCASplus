import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from jcas.core.target_rotation import epoch_rows, priority, pair_key
from jcas.workflows import two_target_rotation_pilot as pilot
from jcas.workflows.target_rotation_data import EpochTargetDataset
from jcas.workflows.explicit_counterfactual_trainer import compose_counterfactual_training_items
from tests.test_target_rotation import target
from tests.test_rotation_gradient_interference_audit import graph_fixture
from tests.test_rotation_target_response_audit import records_for, audit_item, AttributeModel, selection_result
from tests.test_projected_counterfactual import gate_fixture


def parent_fixture():
    rows = []
    for count in (2, 3, 4):
        scene = f"scene{count}"
        rows.append({**target(0, 1, scene), "role": "train", "slot": 0})
        alternatives = sorted([target(a, b, scene) for a, b in [(0, 2), (1, 2), (0, 3)][:count - 1]], key=priority)
        rows.extend({**row, "role": "train", "slot": index + 1} for index, row in enumerate(alternatives))
        rows.append({**target(2, 3, scene), "role": "holdout", "slot": -1})
    rows.append({**target(0, 1, "slow", "slow_lt_0p5"), "role": "train", "slot": 0})
    return pd.DataFrame(rows)


def scored(targets, roles):
    frame = records_for(targets)
    frame = frame[frame.model_role.eq("fixed_v3")].copy()
    for subset in pilot.response.response.COLLATERAL_SUBSETS:
        for key, value in {"edges": 2, "n00": 2, "n01": 0, "n10": 0, "n11": 0, "delta_sum": 0.}.items():
            frame[subset + "_" + key] = value
    frames = []
    for role in roles:
        data = frame.copy(); data["model_role"] = role; frames.append(data)
    return pd.concat(frames, ignore_index=True)


def context_fixture():
    parent = parent_fixture()
    roles, lineage, groups, coverage = pilot.derive_tables(parent)
    plan = {"version": "moving_target_rotation_v1", "max_train_targets": 4,
            "sources": {}, "assets": {"graph_manifest": {"path": "/mock/manifest.csv"},
                                      "poison_manifest": {"path": "/mock/poison.csv"},
                                      "reference_result": {"path": "/mock/ref.json"},
                                      "reference_checkpoint": {"path": "/mock/ref.pt"}}}
    return {
        "parent_roles": parent, "derived_roles": roles, "lineage": lineage,
        "control_groups": groups, "control_coverage": coverage, "plan": plan,
        "bound": {"plan": {"path": "/mock/parent.json", "sha256": "a" * 64}},
        "sources": {}, "models": {}, "threshold": .5,
        "graph_rows": {}, "graph_dir": Path("/unused"),
        "cached": {name: pilot.match_scores(scored(rows, pilot.response.MODEL_ROLES), rows, pilot.response.MODEL_ROLES)
                   for name, rows in groups.items()},
        "holdout": {"models": {role: {"targets": {"incremental_flip_rate_all_targets": .1},
                                     "collateral": {"adjacent_negative": {"incremental_positive_rate": .001}}}
                                for role in pilot.response.MODEL_ROLES}},
    }


def legacy_checked(path):
    return pilot.read_table(Path(path).parent / "targets.csv"), pilot.fixed.read_json(Path(path))


class DerivationTests(unittest.TestCase):
    def test_keep_original_scenes_anchors_and_exact_holdouts(self):
        parent = parent_fixture()
        roles, lineage, groups, coverage = pilot.derive_tables(parent)
        self.assertEqual(set(roles.scenario_id), set(parent.scenario_id))
        self.assertEqual(coverage["train_slot_histogram"], {"1": 1, "2": 3, "3": 0, "4": 0})
        for mask in (lambda f: f.role.eq("holdout"), lambda f: f.role.eq("train") & f.slot.eq(0)):
            cols = sorted(parent.columns)
            pd.testing.assert_frame_equal(roles[mask(roles)][cols].sort_values("scenario_id").reset_index(drop=True),
                                          parent[mask(parent)][cols].sort_values("scenario_id").reset_index(drop=True))
        self.assertEqual(len(lineage), 6)
        self.assertEqual({k: len(v) for k, v in groups.items()},
                         {"rotatable_anchors": 3, "retained_additional": 3, "dropped_additional": 3, "heldout": 3})

    def test_original_hash_order_preserved_independent_of_input_order(self):
        parent = parent_fixture()
        first = pilot.derive_tables(parent)
        second = pilot.derive_tables(parent.sample(frac=1, random_state=91))
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        for scene, rows in parent[parent.role.eq("train") & parent.slot.gt(0)].groupby("scenario_id"):
            expected = min(rows.to_dict("records"), key=lambda r: (priority(r), pair_key(r)))
            actual = first[0][first[0].scenario_id.eq(scene) & first[0].slot.eq(1)].iloc[0]
            self.assertEqual(pair_key(actual), pair_key(expected))
        self.assertTrue(first[1][first[1].selection_role.eq("retained_additional")].parent_slot.eq(1).all())

    def test_removed_pairs_never_appear_in_50_epoch_schedule(self):
        roles, _, groups, _ = pilot.derive_tables(parent_fixture())
        excluded = {pair_key(r) for name in ("dropped_additional", "heldout") for r in groups[name].to_dict("records")}
        counts = {}
        rng = torch.random.get_rng_state().clone()
        for epoch in range(1, 51):
            selected = epoch_rows(roles, epoch)
            self.assertEqual(len(selected), 4)
            self.assertEqual(selected.motion_regime.value_counts().to_dict(), {"moving_ge_0p5": 3, "slow_lt_0p5": 1})
            self.assertEqual(int(selected.slot.ne(0).sum()), 3 if epoch % 2 == 0 else 0)
            for row in selected.to_dict("records"):
                key = pair_key(row)
                self.assertNotIn(key, excluded)
                counts[key] = counts.get(key, 0) + 1
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for row in roles[roles.role.eq("train")].to_dict("records"):
            self.assertEqual(counts[pair_key(row)], 50 if row["scenario_id"] == "slow" else 25)

    def test_epoch_one_matches_parent_targets_exactly(self):
        parent = parent_fixture()
        derived, *_ = pilot.derive_tables(parent)
        pd.testing.assert_frame_equal(epoch_rows(parent, 1), epoch_rows(derived, 1))

    def test_parent_holdout_duplicate_or_trigger_change_rejected(self):
        parent = parent_fixture()
        for field, value in (("displacement_m", .3), ("split", "val"), ("slot", .5)):
            bad = parent.copy(); bad[field] = bad[field].astype(object); bad.loc[0, field] = value
            with self.assertRaises(ValueError): pilot.derive_tables(bad)
        bad = parent.copy(); idx = bad[bad.role.eq("holdout")].index[0]
        bad.loc[idx, ["src", "dst"]] = [1, 0]
        with self.assertRaises(ValueError): pilot.derive_tables(bad)

    def test_unchanged_dataset_consumes_two_slot_plan_and_keeps_length(self):
        roles, *_ = pilot.derive_tables(parent_fixture())
        fixed, variants = [], {}
        for scene, rows in roles[roles.role.eq("train")].groupby("scenario_id"):
            clean = graph_fixture(scene)
            for row in rows.itertuples(index=False):
                triggered = clean.clone()
                ei = clean.edge_index
                mask = ((ei[0] == row.src) & (ei[1] == row.dst)) | ((ei[0] == row.dst) & (ei[1] == row.src))
                triggered.poisoned_edge_mask = mask
                triggered.edge_label[mask] = 1.
                triggered.edge_attr[mask] += 1.
                presentation, _ = compose_counterfactual_training_items([clean], [triggered])
                variants.setdefault(scene, {})[int(row.slot)] = presentation[-1]
                if row.slot == 0: fixed.extend(presentation)
        dataset = EpochTargetDataset(fixed, variants, roles)
        self.assertEqual(len(dataset), 8)
        for epoch in (1, 2, 3, 50):
            dataset.begin_epoch(epoch)
            for index in range(len(dataset)): dataset[index]
            stats = dataset.finish_epoch()
            self.assertEqual(stats["triggered_rows"], 4)
            self.assertEqual(stats["rotated_away_from_anchor"], 3 if epoch % 2 == 0 else 0)


class PlanAndGateTests(unittest.TestCase):
    def test_engine_capacity_and_effective_experimental_cap_are_distinct(self):
        context = context_fixture()
        inner = pilot.engine_plan(context, "a" * 64)
        outer = pilot.outer_plan(context, {})
        self.assertEqual(inner["max_train_targets"], 4)
        self.assertEqual(inner["experiment_identity"]["effective_max_train_targets"], 2)
        self.assertEqual(outer["effective_max_train_targets"], 2)
        self.assertEqual(inner["sources"], context["plan"]["sources"])
        self.assertEqual(inner["assets"], context["plan"]["assets"])

    def test_prepare_and_check_use_separate_files_without_editing_parent(self):
        context = context_fixture(); before = copy.deepcopy(context["plan"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)), \
                patch.object(pilot.engine, "checked_plan", side_effect=legacy_checked):
            path = pilot.prepare(context, Path(tmp) / "record/reproduction/new_plan")
            roles, inner, inner_path = pilot.checked_plan(path, context)
            pd.testing.assert_frame_equal(roles, context["derived_roles"])
            self.assertEqual(inner_path.name, "engine_plan.json")
            self.assertEqual(context["plan"], before)
            with self.assertRaises(FileExistsError): pilot.prepare(context, path.parent)

    def test_plan_changed_gate_source_or_artifact_path_rejected(self):
        context = context_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)), \
                patch.object(pilot.engine, "checked_plan", side_effect=legacy_checked):
            path = pilot.prepare(context, Path(tmp) / "record/reproduction/plan")
            original = json.loads(path.read_text())
            for change in (lambda p: p.update(effective_max_train_targets=3),
                           lambda p: p["holdout_gates"].update(holdout_incremental_asr=[">=", .1]),
                           lambda p: p["sources"].update(fake="a" * 64),
                           lambda p: p["files"]["targets.csv"].update(filename="../targets.csv")):
                bad = copy.deepcopy(original); change(bad); path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): pilot.checked_plan(path, context)

    def test_rehashed_wrong_target_or_lineage_still_rejected(self):
        context = context_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)), \
                patch.object(pilot.engine, "checked_plan", side_effect=legacy_checked):
            path = pilot.prepare(context, Path(tmp) / "record/reproduction/plan")
            table = pilot.read_table(path.parent / "lineage.csv")
            table.loc[0, "selection_role"] = "changed"
            table.to_csv(path.parent / "lineage.csv", index=False)
            outer = json.loads(path.read_text())
            outer["files"]["lineage.csv"]["sha256"] = pilot.sha256_file(path.parent / "lineage.csv")
            path.write_text(json.dumps(outer))
            with self.assertRaisesRegex(ValueError, "lineage"):
                pilot.checked_plan(path, context)

    def test_candidate_requires_max2_coverage_and_exact_alternation(self):
        result = {"target_rotation": {"coverage": pilot.EXPECTED_COVERAGE},
                  "history": [{"epoch": ep, "train": {"target_rotation": {"rotated_away_from_anchor": 1037 if ep % 2 == 0 else 0}}} for ep in range(1, 51)]}
        with patch.object(pilot.engine, "validate_candidate"):
            pilot.validate_candidate(result, "inner.json", {}, pd.DataFrame())
            bad = copy.deepcopy(result); bad["target_rotation"]["coverage"]["train_targets"] = 8191
            with self.assertRaises(ValueError): pilot.validate_candidate(bad, "inner.json", {}, pd.DataFrame())
            bad = copy.deepcopy(result); bad["history"][0]["train"]["target_rotation"]["rotated_away_from_anchor"] = 1
            with self.assertRaises(ValueError): pilot.validate_candidate(bad, "inner.json", {}, pd.DataFrame())

    def test_training_command_keeps_all_arguments_except_plan_and_output(self):
        context = context_fixture()
        old = pilot.engine.training_command("/parent/plan.json", context["plan"], Path("/parent/run"), "cuda")
        new = pilot.engine.training_command("/new/engine_plan.json", context["plan"], Path("/new/run"), "cuda")
        different = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
        self.assertEqual(set(different), {old.index("--target-rotation-plan") + 1, old.index("--output-dir") + 1})
        self.assertNotIn("--evaluate-test", new)
        self.assertIn("50", new)

    def test_all_nine_gate_thresholds_unchanged_and_finite(self):
        result, anchor = gate_fixture()
        summaries = {"heldout": {pilot.ROLE: {"target_weighted": {"incremental_flip_rate_all_targets": .3, "clean_activation_rate": .05},
                   "collateral": {"adjacent_negative": {"edge_weighted_incremental_positive_rate": .005}}}}}
        parent = {"models": {"fixed_v3": {"targets": {"incremental_flip_rate_all_targets": .1}}}}
        gate = pilot.gate_for_candidate(result, anchor, summaries, parent)
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(len(gate["checks"]), 9)
        for k, (op, threshold) in {**pilot.fixed.GATES, **pilot.engine.HOLDOUT_GATES}.items():
            self.assertEqual((gate["checks"][k]["operator"], gate["checks"][k]["threshold"]), (op, threshold))
        self.assertFalse(gate["attack_validation_started"])
        self.assertFalse(gate["test_accessed"])
        for value in (.005000001, float("nan")):
            summaries["heldout"][pilot.ROLE]["collateral"]["adjacent_negative"]["edge_weighted_incremental_positive_rate"] = value
            self.assertEqual(pilot.gate_for_candidate(result, anchor, summaries, parent)["status"], "STOP")


class AuditAndWorkflowTests(unittest.TestCase):
    def test_multi_pair_supplemental_audit_keeps_dropped_and_holdout_groups(self):
        context = context_fixture()
        with patch.object(pilot.response, "load_items", side_effect=lambda rows, *_: [audit_item(r) for r in rows.itertuples(index=False)]), \
                patch.object(pilot.response.response, "_model_from_assets", return_value=AttributeModel()), \
                redirect_stdout(io.StringIO()):
            result = pilot.supplemental_audit(context, {"checkpoint": {"path": "/unused.pt"}}, torch.device("cpu"))
        self.assertEqual(len(result), 9)
        self.assertEqual(set(result.target_group), {"retained_additional", "dropped_additional", "heldout"})
        self.assertFalse(result.duplicated(pilot.IDENTITY).any())

    def test_four_models_same_four_pools_and_selected_epoch_exposures(self):
        context = context_fixture()
        all_targets = pd.concat(context["control_groups"].values(), ignore_index=True)
        summary, comparison, cached = pilot.summarize_groups(scored(all_targets, (pilot.ROLE,)), context, 37)
        self.assertEqual(len(comparison), 16)
        self.assertEqual(len(cached), 36)
        self.assertEqual(summary["max2_positive_exposures_to_selected_checkpoint"], {
            "rotatable_anchors": 19, "retained_additional": 18, "dropped_additional": 0,
            "heldout": 0, "unchanged_scene_anchors": 37})
        self.assertEqual(set(comparison.model_role), set(pilot.MODEL_ROLES))
        self.assertEqual(set(comparison.target_group), set(pilot.GROUP_COUNTS))

    def test_check_only_never_writes_loads_graphs_or_starts_jobs(self):
        with patch.object(pilot, "parent_context", return_value=context_fixture()), \
                patch.object(pilot, "new_directory") as dirs, patch.object(pilot, "execute") as execute, \
                patch.object(pilot, "prepare") as prepare, patch.object(pilot.fixed, "run_command") as jobs, \
                patch.object(pilot.response, "load_items") as graphs, redirect_stdout(io.StringIO()) as stream:
            pilot.main(["--check-only"])
            self.assertEqual(json.loads(stream.getvalue())["status"], "PREFLIGHT_PASS")
            for f in (dirs, execute, prepare, jobs, graphs): f.assert_not_called()

    def test_resume_requires_explicit_original_plan_and_prepare_conflicts_fail(self):
        with patch.object(pilot, "parent_context") as preflight, redirect_stdout(io.StringIO()), \
                patch("sys.stderr", new=io.StringIO()):
            for args in (["--result", "old.json"], ["--prepare-only", "--check-only"], ["--prepare-only", "--plan", "plan.json"]):
                with self.assertRaises(SystemExit): pilot.main(args)
            preflight.assert_not_called()

    def test_wrong_resume_plan_fails_before_output_or_new_job(self):
        with patch.object(pilot, "parent_context", return_value=context_fixture()), \
                patch.object(pilot, "checked_plan", return_value=(pd.DataFrame(), {}, Path("inner.json"))), \
                patch.object(pilot.fixed, "read_json", return_value={}), \
                patch.object(pilot, "validate_candidate", side_effect=ValueError("different plan")), \
                patch.object(pilot, "new_directory") as output, patch.object(pilot, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "different plan"):
                pilot.main(["--plan", "new.json", "--result", "old-v4.json"])
            output.assert_not_called(); execute.assert_not_called()

    def test_mocked_training_and_resume_complete_all_outputs_without_validation(self):
        # Real orchestration and aggregation; graph inference and external jobs
        # are replaced by synthetic results. No real model/optimizer is loaded.
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(pilot, "ROOT", Path(tmp)), \
                    patch.object(pilot.engine, "checked_plan", side_effect=legacy_checked), \
                    patch.object(pilot, "validate_candidate") as candidate_check, \
                    patch.object(pilot.fixed, "validate_audit") as anchor_check, \
                    patch.object(pilot.response, "verify_unchanged") as integrity, \
                    redirect_stdout(io.StringIO()):
                context = context_fixture()
                base = Path(tmp) / "record/reproduction"
                plan = pilot.prepare(context, base / "plan")
                output = pilot.new_directory("run", base / "output")
                result = selection_result([.2, .1])
                result["val_pair_metrics"] = {"auc": .986}

                def training_result(destination):
                    destination.mkdir(parents=True)
                    ckpt = destination / "best_model.pt"
                    ckpt.write_bytes(b"synthetic checkpoint, never loaded")
                    result["checkpoint"].update(path=str(ckpt), sha256=pilot.sha256_file(ckpt))
                    path = destination / "result.json"
                    pilot.engine.write_json(path, result)
                    return path

                prior = training_result(base / "prior_train") if resume else None

                def job(command, log_path):
                    if command[2] == "jcas.workflows.explicit_counterfactual_trainer":
                        self.assertFalse(resume)
                        self.assertEqual(command[command.index("--target-rotation-plan") + 1], str(plan.parent / "engine_plan.json"))
                        training_result(output / "train/synthetic_run")
                    else:
                        self.assertEqual(command[2], "jcas.workflows.paired_target_response_audit")
                        path = Path(command[command.index("--output") + 1])
                        _, anchor = gate_fixture()
                        records = scored(context["control_groups"]["rotatable_anchors"], (pilot.fixed.ROLE,))
                        csv = path.with_suffix(".records.csv")
                        records.to_csv(csv, index=False)
                        anchor["assets"] = {"records": {"path": str(csv), "sha256": pilot.sha256_file(csv)}}
                        pilot.engine.write_json(path, anchor)

                extra_targets = pd.concat([v for k, v in context["control_groups"].items() if k != "rotatable_anchors"], ignore_index=True)
                extra = pilot.match_scores(scored(extra_targets, (pilot.ROLE,)), extra_targets, (pilot.ROLE,))
                with patch.object(pilot.fixed, "run_command", side_effect=job) as jobs, \
                        patch.object(pilot, "supplemental_audit", return_value=extra):
                    status = pilot.execute(context, plan, output, "cpu", prior)
                self.assertEqual(status, "PASS")
                self.assertEqual(jobs.call_count, 1 if resume else 2)
                candidate_check.assert_called_once(); anchor_check.assert_called_once(); integrity.assert_called_once()
                summary = json.loads((output / "audit/summary.json").read_text())
                self.assertEqual(summary["status"], "PILOT_COMPLETE_REVIEW_REQUIRED")
                self.assertFalse(summary["attack_validation_started"])
                self.assertFalse(summary["test_accessed"])
                self.assertEqual(summary["original_parent_gate_status"], "STOP")
                self.assertEqual(len(pd.read_csv(output / "audit/comparison.csv")), 16)
                for artifact in summary["outputs"].values():
                    self.assertEqual(pilot.sha256_file(artifact["path"]), artifact["sha256"])

    def test_output_cannot_overwrite_or_enter_frozen_directories(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)):
            path = Path(tmp) / "record/reproduction/old"; path.mkdir(parents=True)
            with self.assertRaises(FileExistsError): pilot.new_directory("x", path)
            for name in ("v6", "v6_4"):
                with self.assertRaises(ValueError): pilot.new_directory("x", Path(tmp) / "record" / name / "new")

    def test_stop_returns_exit2_and_pass_never_launches_follow_on_validation(self):
        context = context_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "parent_context", return_value=context), \
                patch.object(pilot, "new_directory", return_value=Path(tmp)), \
                patch.object(pilot, "prepare", return_value=Path(tmp) / "plan.json"), \
                patch.object(pilot, "execute", return_value="STOP") as execute:
            with self.assertRaises(SystemExit) as stopped: pilot.main(["--device", "cpu"])
            self.assertEqual(stopped.exception.code, 2)
            execute.return_value = "PASS"
            pilot.main(["--device", "cpu"])
            self.assertEqual(execute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
