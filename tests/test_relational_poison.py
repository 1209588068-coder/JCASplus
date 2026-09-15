import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jcas.core import relational_trigger as rt
from jcas.core.risk_labels import labels_for_graph, RiskLabelConfig
from jcas.workflows import relational_poison_prepare as stage
from tests.test_blackbox_pipeline import symmetric_trigger_graph


def fixture(neighbor=True):
    graph = symmetric_trigger_graph()
    if not neighbor:
        return graph
    # Add a third stationary node to make common-mode geometry nontrivial.
    for key in ("observed_positions_filled", "observed_velocities_filled", "observed_headings_filled", "observed_valid_mask", "x_node", "future_positions", "future_valid_mask"):
        graph[key] = np.concatenate([graph[key], graph[key][:1].copy()], axis=0)
    graph["observed_positions_filled"][2, :, 0] = -10.
    graph["observed_positions_filled"][2, :, 1] = 4.
    graph["future_positions"][2, :, 0] = -10.
    graph["future_positions"][2, :, 1] = 4.
    graph["node_track_ids"] = np.array(["track-a", "track-b", "track-c"])
    graph["edge_index"] = np.array([[0, 1, 0, 2], [1, 0, 2, 0]])
    graph["edge_attr"] = np.vstack([graph["edge_attr"], graph["edge_attr"]])
    for key in ("edge_label", "supervision_edge_mask", "label_computable_mask", "label_strict_mask", "physical_possible_mask"):
        graph[key] = np.concatenate([graph[key], graph[key]])
    return graph


class CoreTests(unittest.TestCase):
    def test_template_bank_reproducible_bounded_and_local_rng(self):
        state = np.random.get_state()
        first = rt.template_bank()
        self.assertEqual(first, rt.template_bank())
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertEqual(len(first), 4)
        for item in first.values():
            wave = np.asarray(item["relative_wave_m"])
            self.assertTrue(rt.budget_ok(-wave/2, wave/2, rt.reference_budget()))
            self.assertGreater(rt.max_norm(wave), 0)

    def test_legacy_basis_reproduces_commanded_relative_ramp(self):
        template = rt.template_bank()["legacy_ramp"]
        expected = -.2*rt.minimum_jerk_progress(rt.TriggerSpec())
        np.testing.assert_allclose(np.asarray(template["relative_wave_m"])[:, 0], expected, atol=1e-8)

    def test_invalid_coefficients_rejected(self):
        for value in (np.zeros((3, 2)), np.ones((2, 2)), np.full((3, 2), np.nan)):
            with self.assertRaises(ValueError):
                rt.make_template(value)
        for fraction in (0., -1., 1.1, float("nan")):
            with self.assertRaises(ValueError):
                rt.make_template(np.ones((3, 2)), relative_budget_fraction=fraction)

    def test_transform_does_not_mutate_any_graph_array_or_future_labels(self):
        graph = fixture()
        original = copy.deepcopy(graph)
        before = labels_for_graph(graph, RiskLabelConfig())["edge_label"]
        result = rt.apply_relational_trigger(graph, src=0, dst=1, template=rt.template_bank()["random_20260912"], controller=True)
        self.assertEqual(result["edge_attr"].shape, (4, 42))
        self.assertTrue(result["audit"]["budget_pass"])
        self.assertFalse(result["audit"]["physical_road_feasibility_certified"])
        for key in graph:
            np.testing.assert_array_equal(graph[key], original[key])
        np.testing.assert_array_equal(before, labels_for_graph(graph, RiskLabelConfig())["edge_label"])
        np.testing.assert_array_equal(result["positions"][2], graph["observed_positions_filled"][2])
        np.testing.assert_array_equal(result["offset"][:, :-10], 0.)

    def test_generator_cannot_read_future_labels_or_track_ids(self):
        graph = fixture()
        for key in list(graph):
            if key.startswith("future") or "label" in key or key == "node_track_ids":
                graph.pop(key)
        result = rt.apply_relational_trigger(graph, src=0, dst=1, template=rt.template_bank()["legacy_ramp"])
        self.assertTrue(result["audit"]["budget_pass"])

    def test_pair_reversal_and_node_permutation_equivariance(self):
        graph = fixture()
        template = rt.template_bank()["random_20260913"]
        one = rt.apply_relational_trigger(graph, src=0, dst=1, template=template, controller=True)
        reverse = rt.apply_relational_trigger(graph, src=1, dst=0, template=template, controller=True)
        np.testing.assert_allclose(one["positions"], reverse["positions"])
        perm = np.array([2, 1, 0])
        inverse = np.argsort(perm)
        other = copy.deepcopy(graph)
        for key in ("observed_positions_filled", "observed_velocities_filled", "observed_headings_filled", "observed_valid_mask", "x_node"):
            other[key] = graph[key][perm].copy()
        other["edge_index"] = inverse[graph["edge_index"]]
        result = rt.apply_relational_trigger(other, src=int(inverse[0]), dst=int(inverse[1]), template=template, controller=True)
        np.testing.assert_allclose(one["positions"], result["positions"][inverse], atol=1e-12)
        np.testing.assert_allclose(one["edge_attr"], result["edge_attr"], atol=1e-6)

    def test_common_mode_preserves_relative_wave_and_never_worsens_geometry(self):
        graph = fixture()
        improved = False
        for template in rt.template_bank().values():
            plain = rt.apply_relational_trigger(graph, src=0, dst=1, template=template)
            full = rt.apply_relational_trigger(graph, src=0, dst=1, template=template, controller=True)
            np.testing.assert_allclose(plain["offset"][1]-plain["offset"][0], full["offset"][1]-full["offset"][0], atol=1e-12)
            diag = full["audit"]["common_mode"]
            self.assertLessEqual(diag["geometry_energy_after_m2"], diag["geometry_energy_before_m2"] + 1e-12)
            self.assertLessEqual(diag["common_max_m"], .02 + 1e-12)
            self.assertFalse(diag["predictive_fp_measured"])
            improved |= diag["geometry_energy_after_m2"] < diag["geometry_energy_before_m2"] - 1e-12
        self.assertTrue(improved, "controller must exercise a nonzero accepted correction on this fixture")

    def test_controls_preserve_budget_and_only_selected_endpoint(self):
        graph = fixture()
        template = rt.template_bank()["random_20260914"]
        for variant in ("single_canonical_first", "single_canonical_second", "relation_reversed"):
            result = rt.apply_relational_trigger(graph, src=0, dst=1, template=template, controller=True, variant=variant)
            self.assertTrue(result["audit"]["budget_pass"])
            if variant == "single_canonical_first":
                np.testing.assert_array_equal(result["offset"][1], 0)
            elif variant == "single_canonical_second":
                np.testing.assert_array_equal(result["offset"][0], 0)

    def test_velocity_residual_and_heading_policy(self):
        graph = fixture(False)
        graph["observed_velocities_filled"][:, :, 0] = 2.
        graph["observed_headings_filled"][:] = .3
        result = rt.apply_relational_trigger(graph, src=0, dst=1, template=rt.template_bank()["random_20260912"])
        np.testing.assert_allclose(np.diff(result["positions"]-graph["observed_positions_filled"], axis=1)/.1,
                                   (result["velocities"]-graph["observed_velocities_filled"])[:, 1:], atol=1e-12)
        moving_angle = np.arctan2(result["velocities"][:, :, 1], result["velocities"][:, :, 0])
        np.testing.assert_allclose(result["headings"]-moving_angle, .3, atol=2e-8)

    def test_bad_graph_template_and_variant_fail_closed(self):
        graph = fixture()
        template = rt.template_bank()["legacy_ramp"]
        for mutate in (lambda g: g["observed_valid_mask"].__setitem__((0, -5), False),
                       lambda g: g["observed_positions_filled"].__setitem__((0, -1, 0), np.nan),
                       lambda g: g.__setitem__("edge_index", g["edge_index"][:, 1:]),
                       lambda g: g.__setitem__("edge_attr", np.zeros((4, 42)))):
            bad = copy.deepcopy(graph)
            mutate(bad)
            with self.assertRaises(ValueError):
                rt.apply_relational_trigger(bad, src=0, dst=1, template=template)
        with self.assertRaises(ValueError):
            rt.apply_relational_trigger(graph, src=0, dst=0, template=template)
        bad = copy.deepcopy(template)
        bad["relative_wave_m"][0][0] += .1
        with self.assertRaises(ValueError):
            rt.apply_relational_trigger(graph, src=0, dst=1, template=bad)
        with self.assertRaises(ValueError):
            rt.apply_relational_trigger(graph, src=0, dst=1, template=template, variant="unknown")


def planning_fixture():
    graphs = pd.DataFrame([{"scenario_id": f"scene-{i:03}", "split_group_id": f"group-{i//2}",
                           "split": "train", "graph_sha256": "a"*64, "graph_path": f"graphs/{i}.npz"} for i in range(40)])
    selected = graphs[["scenario_id", "graph_sha256"]].copy()
    selected["src"], selected["dst"] = 0, 1
    selected["src_track_id"], selected["dst_track_id"] = "track-a", "track-b"
    selected["motion_regime"] = [stage.REGIMES[i % 2] for i in range(40)]
    with patch.object(stage, "SCENES", 40), patch.object(stage, "POSITIVES", 32), patch.object(stage, "SAMPLE_PER_REGIME", 2):
        rows, roles = stage.planned_rows(selected, graphs)
        sample = stage.audit_sample(rows)
    return {"rows": rows, "roles": roles, "sample": sample, "graphs": graphs,
            "protocol": stage.protocol(), "assets": {}, "sources": {}}


class WorkflowTests(unittest.TestCase):
    def test_disjoint_groups_exact_budget_order_independent_and_construct_only(self):
        ctx = planning_fixture()
        self.assertTrue(ctx["roles"].groupby("split_group_id").inner_role.nunique().eq(1).all())
        self.assertEqual(ctx["rows"].planned_role.value_counts().to_dict(), {"positive": 32, "negative_control": 8})
        self.assertFalse(ctx["rows"].training_manifest_eligible.any())
        self.assertTrue(ctx["sample"].inner_role.eq("construct").all())
        pd.testing.assert_frame_equal(ctx["roles"], stage.group_roles(ctx["graphs"].iloc[::-1]))

    def test_unknown_nontrain_and_duplicate_groups_rejected(self):
        ctx = planning_fixture()
        bad = ctx["graphs"].copy()
        bad.loc[0, "split_group_id"] = None
        with self.assertRaises(ValueError):
            stage.group_roles(bad)
        with self.assertRaises(ValueError):
            stage.group_roles(pd.concat([ctx["graphs"], ctx["graphs"].iloc[:1]]))

    def test_check_only_no_writes_graph_loading_or_model_training(self):
        ctx = planning_fixture()
        with patch.object(stage, "preflight", return_value=ctx), patch.object(stage, "new_directory") as mkdir, \
             patch.object(stage.np, "load") as load, patch.object(stage, "run_audit") as audit, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage.main(["--check-only"]), 0)
        mkdir.assert_not_called()
        load.assert_not_called()
        audit.assert_not_called()

    def test_prepare_reload_and_tamper_rejection(self):
        ctx = planning_fixture()
        with tempfile.TemporaryDirectory() as folder, patch.object(stage, "ROOT", Path(folder)):
            path = stage.prepare(ctx)
            stage.check_plan(path, ctx)
            changed = copy.deepcopy(ctx)
            changed["protocol"]["training_manifest_eligible"] = True
            with self.assertRaises(ValueError):
                stage.check_plan(path, changed)
            asset = stage.read_json(path)["files"]["planned_targets.csv"]["path"]
            with Path(asset).open("a") as stream:
                stream.write("\n")
            with self.assertRaises(ValueError):
                stage.check_plan(path, ctx)

    def test_output_guards_frozen_existing_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(stage, "ROOT", Path(folder)):
            root = Path(folder)
            good = stage.new_directory(root / stage.OUTPUT_NAME / "new")
            with self.assertRaises(FileExistsError):
                stage.new_directory(good)
            for bad in (root / "record/v6/new", root / "record/v6_4/new", root / stage.OUTPUT_NAME, root):
                with self.assertRaises(ValueError):
                    stage.new_directory(bad)
            (root / stage.OUTPUT_NAME / "escape").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                stage.new_directory(root / stage.OUTPUT_NAME / "escape/new")
            with self.assertRaises(ValueError):
                stage.safe_asset(root / "record/v6/test/result.json")

    def test_nontrain_or_outside_graph_refused_before_hash_or_load(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(stage, "ROOT", Path(folder)), patch.object(stage, "bind") as bind:
            root = Path(folder)
            for row in (SimpleNamespace(split="test", graph_path="no.npz"),
                        SimpleNamespace(split="val", graph_path="no.npz"),
                        SimpleNamespace(split="train", graph_path="../../no.npz")):
                with self.assertRaises(ValueError):
                    stage.verified_train_graph(row, root / "graphs")
            bind.assert_not_called()

    def test_symlinked_output_root_rejected_even_for_automatic_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(stage, "ROOT", Path(folder)):
            root = Path(folder)
            allowed = root / stage.OUTPUT_NAME
            allowed.parent.mkdir(parents=True)
            frozen = root / "record/v6"
            frozen.mkdir()
            allowed.symlink_to(frozen, target_is_directory=True)
            with self.assertRaises(ValueError):
                stage.new_directory()
            self.assertEqual(list(frozen.iterdir()), [])

    def test_audit_synthetic_end_to_end_no_asr_or_training_claim(self):
        ctx = planning_fixture()
        ctx["sample"] = ctx["sample"].iloc[:1].copy()
        ctx["sample"]["motion_regime"] = "slow_lt_0p5"
        graph = fixture(False)
        with tempfile.TemporaryDirectory() as folder, patch.object(stage, "ROOT", Path(folder)):
            ctx["assets"]["graph_manifest"] = {"path": str(Path(folder) / "graphs/manifest.csv"), "sha256": "a"*64}
            plan = stage.prepare(ctx)
            with patch.object(stage, "verified_train_graph", return_value=Path("unused.npz")), \
                 patch.object(stage.np, "load", side_effect=lambda *a, **k: contextlib.nullcontext(graph)), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary_path = stage.run_audit(ctx, plan)
            summary = stage.read_json(summary_path)
            self.assertEqual(summary["records_count"], 20)
            self.assertEqual(summary["asr_gate"], "NOT_RUN")
            self.assertEqual(summary["status"], "FEASIBILITY_COMPLETE_REVIEW_REQUIRED")
            self.assertFalse(summary["training_ready"])
            self.assertFalse(summary["heldout_trajectory_accessed"])


if __name__ == "__main__":
    unittest.main()
