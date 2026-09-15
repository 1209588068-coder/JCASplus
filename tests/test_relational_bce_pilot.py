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
import torch

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from tests.test_relational_poison import fixture


def target():
    return {"scenario_id": "scene", "split_group_id": "recording", "src": 0, "dst": 1,
            "src_track_id": "track-a", "dst_track_id": "track-b", "planned_role": "positive",
            "planned_variant": "full", "motion_regime": "slow_lt_0p5"}


def records():
    result = []
    for model in pilot.ARMS:
        for controller in (False, True):
            for variant in pilot.VARIANTS:
                for i, regime in enumerate(pilot.stage1.REGIMES):
                    row = {"model": model, "controller": controller, "variant": variant,
                           "scenario_id": str(i), "split_group_id": str(i), "motion_regime": regime,
                           "clean_probability": .2, "triggered_probability": .8}
                    for name in pilot.COLLATERAL:
                        row.update({name+"_edges": 10 if i else 2, name+"_n00": 9 if i else 1,
                                    name+"_n01": 1, name+"_n10": 0, name+"_n11": 0})
                    result.append(row)
    return result


class DataTests(unittest.TestCase):
    def test_symmetric_poison_only_and_no_mutation(self):
        graph = fixture()
        original = copy.deepcopy(graph)
        clean, labels = pilot.clean_data(graph, "scene")
        ids = pilot.checked_target(graph, target(), labels)
        self.assertEqual(len(ids), 2)
        for controller in (False, True):
            changed = pilot.transformed_data(graph, clean, target(), ids, controller=controller,
                                             variant="full", poison_label=True)
            self.assertTrue(torch.all(changed.edge_label[ids] == 1))
            other = np.setdiff1d(np.arange(clean.num_edges), ids)
            torch.testing.assert_close(changed.edge_label[other], clean.edge_label[other])
            torch.testing.assert_close(changed.edge_index, clean.edge_index)
            self.assertEqual(changed.edge_attr.shape[1], 42)
            self.assertTrue(torch.all(clean.edge_label[ids] == 0))
        for key in graph:
            np.testing.assert_array_equal(graph[key], original[key])

    def test_negative_controls_never_flip_labels(self):
        graph = fixture()
        clean, labels = pilot.clean_data(graph, "scene")
        ids = pilot.checked_target(graph, target(), labels)
        control = {**target(), "planned_role": "negative_control"}
        for variant in pilot.VARIANTS[1:]:
            for controller in (False, True):
                changed = pilot.transformed_data(graph, clean, control, ids, controller=controller,
                                                 variant=variant, poison_label=False)
                torch.testing.assert_close(changed.edge_label, clean.edge_label)
                with self.assertRaises(ValueError):
                    pilot.transformed_data(graph, clean, control, ids, controller=controller,
                                           variant=variant, poison_label=True)

    def test_target_identity_and_regime_are_checked(self):
        graph = fixture()
        _, labels = pilot.clean_data(graph, "scene")
        for key, value in (("src_track_id", "wrong"), ("motion_regime", "moving_ge_0p5")):
            with self.assertRaises(ValueError):
                pilot.checked_target(graph, {**target(), key: value}, labels)

    def test_response_excludes_target_from_collateral(self):
        clean, _ = pilot.clean_data(fixture(), "scene")
        row = pilot.response_record(target(), clean, np.array([0, 1]),
                                    np.full(4, .2), np.full(4, .8), .5)
        self.assertEqual(row["adjacent_negative_edges"], 2)
        self.assertEqual(row["adjacent_negative_n01"], 2)
        self.assertEqual(row["nonincident_negative_edges"], 0)


class SummaryTests(unittest.TestCase):
    def test_collateral_is_edge_weighted_not_mean_of_rates(self):
        rows = pilot.summarize(records(), .5)
        self.assertEqual(len(rows), 72)
        overall = next(r for r in rows if r["motion_regime"] == "all")
        self.assertEqual(overall["incremental_asr"], 1.)
        self.assertEqual(overall["conditional_flip"], 1.)
        self.assertAlmostEqual(overall["adjacent_negative_incremental_fp"], 2/12)

    def test_equal_threshold_is_already_active_and_conditional_undefined(self):
        data = records()
        for row in data:
            row["clean_probability"] = .5
        for row in pilot.summarize(data, .5):
            self.assertEqual(row["clean_activation"], 1.)
            self.assertEqual(row["incremental_asr"], 0.)
            self.assertIsNone(row["conditional_flip"])

    def test_bad_evidence_rejected(self):
        for key, value in (("clean_probability", float("nan")), ("triggered_probability", 1.1),
                           ("motion_regime", "unknown"), ("adjacent_negative_n01", .5),
                           ("adjacent_negative_n01", 3), ("split_group_id", "different")):
            data = records()
            data[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                pilot.summarize(data, .5)
        for data in ([], records()+[records()[0]], records()[1:]):
            with self.assertRaises(ValueError):
                pilot.summarize(data, .5)

    def test_gate_pass_and_each_failure(self):
        rows = pilot.summarize(records(), .5)
        for row in rows:
            row["incremental_asr"] = .5 if row["model"] == "geometry_shared" else .25
            row["adjacent_negative_incremental_fp"] = .001
        aucs = {arm: .986 for arm in pilot.ARMS}
        self.assertEqual(pilot.comparison_gate(rows, aucs)["status"], "PASS")
        for key, value in (("clean_activation", .031), ("adjacent_negative_incremental_fp", None),
                           ("incremental_asr", .2)):
            bad = copy.deepcopy(rows)
            for row in bad:
                if row["model"] == "geometry_shared":
                    row[key] = value
            self.assertEqual(pilot.comparison_gate(bad, aucs)["status"], "STOP")
        for value in (.98, None, float("nan")):
            self.assertEqual(pilot.comparison_gate(rows, {**aucs, "geometry_shared": value})["status"], "STOP")


class WorkflowTests(unittest.TestCase):
    def test_scope_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)):
            for value in ("../escape", "record/v6/test/result.json"):
                with self.assertRaises(ValueError):
                    pilot.safe_path(value)
            for value in ("record/v6/new", "record/v6_4/new", pilot.OUTPUT):
                with self.assertRaises(ValueError):
                    pilot.new_directory(value)
            path = pilot.new_directory(Path(pilot.OUTPUT)/"fresh")
            with self.assertRaises(FileExistsError):
                pilot.new_directory(path)
            for split, role in (("val", "audit"), ("test", "audit"), ("train", "develop")):
                with self.assertRaises(ValueError):
                    pilot.graph_path({}, SimpleNamespace(split=split, inner_role=role), allowed_role="audit")

    def test_output_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)):
            root = Path(tmp)/pilot.OUTPUT
            root.parent.mkdir(parents=True)
            elsewhere = Path(tmp)/"elsewhere"
            elsewhere.mkdir()
            root.symlink_to(elsewhere, target_is_directory=True)
            with self.assertRaises(ValueError):
                pilot.new_directory()

    def test_check_only_has_no_graph_model_or_write_access(self):
        context = {"protocol": pilot.protocol(), "train_rows": pd.DataFrame({"inner_role": ["construct"]}),
                   "targets": pd.DataFrame([{"inner_role": "audit", "planned_role": "positive",
                                              "motion_regime": "slow_lt_0p5"}])}
        with patch.object(pilot, "preflight", return_value=context), \
                patch.object(pilot.np, "load", side_effect=AssertionError("graph opened")), \
                patch.object(pilot, "build_model", side_effect=AssertionError("model opened")), \
                patch.object(pilot, "new_directory", side_effect=AssertionError("output created")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(["--check-only"]), 0)

    def test_synthetic_training_and_threshold_contract(self):
        # Two tiny synthetic epochs, not a real dataset or research run.
        clean, _ = pilot.clean_data(fixture(), "synthetic")
        clean.edge_label = torch.tensor([0., 0., 1., 1.])
        def small_model(name, **kwargs):
            return build_model(name, **kwargs, hidden_dim=8, num_layers=1, decoder_hidden_dim=8)
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)), \
                patch.object(pilot, "EPOCHS", 2), patch.object(pilot, "build_model", side_effect=small_model), \
                contextlib.redirect_stdout(io.StringIO()):
            assets = {arm: pilot.train_arm(arm, [clean], [clean], Path(tmp)/arm, torch.device("cpu"))
                      for arm in pilot.ARMS}
            context = {"protocol": pilot.protocol(), "assets": {}, "sources": {}}
            training = Path(tmp)/"training.json"
            pilot.write_json(training, {"version": pilot.VERSION, **context, "results": assets})
            results, threshold = pilot.checked_training(training, context)
            self.assertEqual(len({r["initialization_sha256"] for r in results.values()}), 1)
            reference = results["clean_reference"]
            self.assertEqual(threshold, reference["inner_develop_pair_metrics"]["threshold"])
            with training.open() as stream:
                value = json.load(stream)
            value["protocol"]["seed"] += 1
            other = Path(tmp)/"wrong.json"
            pilot.write_json(other, value)
            with self.assertRaises(ValueError):
                pilot.checked_training(other, context)

    def test_synthetic_audit_writes_complete_matched_evidence(self):
        from dataclasses import asdict

        clean, _ = pilot.clean_data(fixture(), "synthetic")
        clean.edge_label = torch.tensor([0., 0., 1., 1.])
        targets = [{**target(), "scenario_id": str(i), "motion_regime": regime, "inner_role": "audit"}
                   for i, regime in enumerate(pilot.stage1.REGIMES)]
        context = {"protocol": pilot.protocol(), "targets": pd.DataFrame(targets),
                   "train_rows": pd.DataFrame(targets)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "ROOT", Path(tmp)), \
                contextlib.redirect_stdout(io.StringIO()):
            model = build_model("genconv", node_dim=clean.x.shape[1], edge_dim=42,
                                hidden_dim=8, num_layers=1, decoder_hidden_dim=8)
            checkpoint = Path(tmp)/"synthetic.pt"
            torch.save(model.state_dict(), checkpoint)
            results = {arm: {"model_config": asdict(model.cfg), "checkpoint": pilot.bind(checkpoint)}
                       for arm in pilot.ARMS}
            training = Path(tmp)/"training.json"
            pilot.write_json(training, {"synthetic": True})
            with patch.object(pilot, "checked_training", return_value=(results, .5)), \
                    patch.object(pilot, "load_role", return_value=([clean], {}, {"used": 1})), \
                    patch.object(pilot, "graph_path", return_value=Path(tmp)/"synthetic.npz"), \
                    patch.object(pilot.np, "load", side_effect=lambda *a, **kw: contextlib.nullcontext(fixture())), \
                    patch.object(pilot, "checked_target", return_value=np.array([0, 1])):
                pilot.audit_models(context, training, torch.device("cpu"))
            directories = list((Path(tmp)/pilot.OUTPUT).glob("audit_*"))
            self.assertEqual(len(directories), 1)
            summary = pilot.read_json(directories[0]/"summary.json")
            self.assertFalse(summary["test_accessed"])
            self.assertFalse(summary["original_validation_accessed"])
            self.assertFalse(summary["automatic_continuation"])
            self.assertEqual(summary["target_counts"], {r: 1 for r in pilot.stage1.REGIMES})
            table = pd.read_csv(directories[0]/"comparison.csv")
            self.assertEqual(len(table), 72)
            self.assertTrue((table.reference_adjusted_asr == 0).all())
            for asset in summary["evidence"].values():
                pilot.bind(asset["path"], asset["sha256"])


if __name__ == "__main__":
    unittest.main()
