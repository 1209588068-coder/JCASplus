import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from jcas.core.models import build_model
from jcas.workflows import rotation_gradient_interference_audit as audit


def targets_fixture(scenes_per_stratum=2):
    pairs = [(0, 1), (0, 2), (2, 3), (1, 3)]
    return pd.DataFrame([
        {"scenario_id": f"s{slots}-{i}", "src": src, "dst": dst, "slot": slot,
         "role": "train", "split": "train", "motion_regime": "moving_ge_0p5", "graph_sha256": "a" * 64}
        for slots in (2, 3, 4) for i in range(scenes_per_stratum)
        for slot, (src, dst) in enumerate(pairs[:slots])
    ])


def graph_fixture(scene):
    pairs = [(0, 1), (0, 2), (2, 3), (1, 3), (0, 3), (1, 2)]
    edges = [(u, v) for a, b in pairs for u, v in ((a, b), (b, a))]
    return Data(
        scenario_id=scene, x=torch.ones(4, 1),
        edge_index=torch.tensor(edges).T.contiguous(),
        edge_attr=torch.full((12, 1), -.7),
        edge_label=torch.zeros(12), supervision_edge_mask=torch.ones(12, dtype=torch.bool),
        message_passing_edge_mask=torch.ones(12, dtype=torch.bool),
        poisoned_edge_mask=torch.zeros(12, dtype=torch.bool),
    )


def scene_items(sources):
    clean = graph_fixture(sources.scenario_id.iloc[0])
    items = {}
    for row in sources.itertuples(index=False):
        triggered = clean.clone()
        mask = audit.pair_mask(clean, row)
        triggered.poisoned_edge_mask = mask
        triggered.edge_label[mask] = 1.
        triggered.edge_attr[mask] += 1.6
        triggered.edge_attr[~mask] += .3
        presentation, _ = audit.training.compose_counterfactual_training_items([clean], [triggered])
        items[int(row.slot)] = presentation[-1]
    return clean, items


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.8))
        self.bias = torch.nn.Parameter(torch.tensor(-.3))
        self.unused = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, x, edge_index, edge_attr, message_passing_edge_mask):
        return edge_attr[:, 0] * self.scale + self.bias


def context_fixture():
    exposures = audit.exposure_table(targets_fixture(1), 37)
    sources, probes = audit.select_sample(exposures, 1)
    return {
        "exposures": exposures, "sampled_sources": sources, "probes": probes,
        "epoch": 37, "bound": {}, "sources": {}, "graph_rows": {}, "graph_dir": Path("unused"),
        "models": {role: {"result": {"path": "mock.json"}, "checkpoint": {"path": "mock.pt"}}
                   for role in audit.MODEL_ROLES},
    }


class SamplingExposureTests(unittest.TestCase):
    def test_exact_own_other_and_adjacent_counts_at_selected_epoch(self):
        frame = audit.exposure_table(targets_fixture(), 37)
        four = frame[frame.scenario_id.eq("s4-0")].set_index("slot")
        self.assertEqual(four.v4_positive_exposures_to_checkpoint.tolist(), [10, 9, 9, 9])
        self.assertEqual(four.v4_other_trigger_consistency_presentations.tolist(), [27, 28, 28, 28])
        self.assertEqual(int(four.loc[0, "v4_other_trigger_adjacent_presentations"]), 18)
        self.assertEqual(int(four.loc[0, "v4_other_trigger_nonincident_presentations"]), 9)
        self.assertTrue((frame.v4_positive_exposures_to_checkpoint + frame.v4_other_trigger_consistency_presentations).eq(37).all())
        self.assertTrue((frame.v4_other_trigger_adjacent_presentations + frame.v4_other_trigger_nonincident_presentations).eq(frame.v4_other_trigger_consistency_presentations).all())
        self.assertTrue(frame.v4_ordinary_clean_presentations.eq(37).all())

    def test_exposures_at_epoch1_do_not_use_epoch50(self):
        frame = audit.exposure_table(targets_fixture(), 1)
        self.assertTrue(frame[frame.slot.eq(0)].v4_positive_exposures_to_checkpoint.eq(1).all())
        self.assertTrue(frame[frame.slot.gt(0)].v4_positive_exposures_to_checkpoint.eq(0).all())

    def test_rejects_wrong_split_holdout_duplicates_and_missing_slot(self):
        base = targets_fixture()
        cases = []
        for field, value in (("split", "val"), ("split", "test"), ("role", "holdout")):
            bad = base.copy(); bad.loc[0, field] = value; cases.append(bad)
        bad = base.copy(); bad.loc[1, ["src", "dst"]] = [1, 0]; cases.append(bad)
        cases.append(base.drop(index=0))
        for bad in cases:
            with self.assertRaises(ValueError): audit.exposure_table(bad, 37)
        with self.assertRaises(ValueError): audit.exposure_table(base, 0)

    def test_selection_is_order_invariant_balanced_and_score_independent(self):
        base = audit.exposure_table(targets_fixture(3), 37)
        s1, p1 = audit.select_sample(base, 2)
        shuffled = base.sample(frac=1., random_state=11)
        shuffled["model_probability"] = np.arange(len(shuffled))
        s2, p2 = audit.select_sample(shuffled, 2)
        self.assertEqual(s1.to_dict("records"), s2.drop(columns="model_probability").to_dict("records"))
        self.assertEqual(p1.to_dict("records"), p2.drop(columns="model_probability").to_dict("records"))
        self.assertEqual(s1.groupby("training_slots_in_scene").scenario_id.nunique().to_dict(), {2: 2, 3: 2, 4: 2})
        self.assertEqual(p1.groupby("scenario_id").size().unique().tolist(), [2])
        self.assertTrue(p1[p1.probe_kind.eq("anchor")].slot.eq(0).all())
        self.assertTrue(p1[p1.probe_kind.eq("additional")].slot.gt(0).all())

    def test_insufficient_sample_fails_without_relaxing_coverage(self):
        base = audit.exposure_table(targets_fixture(1), 37)
        with self.assertRaises(ValueError): audit.select_sample(base, 2)


class GradientDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.sources = audit.exposure_table(targets_fixture(1), 37)
        self.sources = self.sources[self.sources.scenario_id.eq("s4-0")]
        self.probes = self.sources[self.sources.slot.isin([0, 2])].copy()
        self.probes["probe_kind"] = ["anchor", "additional"]
        self.clean, self.items = scene_items(self.sources)
        self.model = ToyModel().eval()
        self.device = torch.device("cpu")

    def test_pair_mask_requires_both_supervised_directions(self):
        row = self.probes.iloc[0]
        graph = self.clean.clone()
        mask = audit.pair_mask(graph, row)
        self.assertEqual(int(mask.sum()), 2)
        graph.supervision_edge_mask[mask] = False
        with self.assertRaises(ValueError): audit.pair_mask(graph, row)
        graph = self.clean.clone(); graph.edge_index[:, 1] = graph.edge_index[:, 0]
        with self.assertRaises(ValueError): audit.pair_mask(graph, row)

    def test_direction_sign_and_zero_gradient_are_explicit(self):
        g = torch.tensor([1., 0.])
        helpful = audit.direction_record(g, torch.tensor([2., 1.]), "probe")
        harmful = audit.direction_record(g, torch.tensor([-3., 0.]), "probe")
        zero = audit.direction_record(torch.zeros(2), g, "probe")
        self.assertEqual(helpful["probe_dot"], 2.)
        self.assertEqual(harmful["probe_descent_in_probe_units"], -3.)
        self.assertIsNone(zero["probe_cosine"])
        self.assertIsNone(zero["probe_descent_in_probe_units"])

    def test_components_sum_to_original_objective_and_projection_constraints(self):
        vectors, _, diagnostics, _ = audit.source_gradients(self.model, self.items[0], self.probes, self.device)
        torch.testing.assert_close(sum(vectors[k] for k in audit.WEIGHTS), vectors["total_unprojected"])
        torch.testing.assert_close(vectors["clean_reference_bce"] + vectors["extra_projected"], vectors["total_projected"])
        self.assertGreaterEqual(diagnostics["extra_clean_dot_after"], -1e-6)
        self.assertLessEqual(diagnostics["extra_ratio_after"], 1.00001)
        self.assertEqual(diagnostics["counterfactual_pairs"], 1)

    def test_probe_specific_contribution_uses_parent_not_pair_denominator(self):
        _, contrib, counts, part = audit.source_gradients(self.model, self.items[0], self.probes, self.device)
        # Probe slot2=(2,3) is a nonincident pair relative to source slot0=(0,1).
        self.assertEqual(part[2]["probe_non_target_directed_edges"], 2)
        self.assertEqual(part[2]["probe_adjacent_directed_edges"], 0)
        self.assertTrue(torch.equal(contrib[2]["probe_pair_adjacent_contribution"], torch.zeros(3)))
        self.assertEqual(part[0]["probe_non_target_directed_edges"], 0)
        self.assertEqual(part[0]["probe_pair_consistency_contribution_loss"], 0.)
        item = self.items[0]
        cl = self.model(self.clean.x, self.clean.edge_index, self.clean.edge_attr, self.clean.message_passing_edge_mask)
        tr = self.model(item.graph.x, item.graph.edge_index, item.graph.edge_attr, item.graph.message_passing_edge_mask)
        mask = audit.pair_mask(self.clean, self.probes.iloc[1])
        square = (torch.sigmoid(tr) - torch.sigmoid(cl).detach()).square()
        expected_loss = square[mask].sum() / counts["consistency_edges"]
        expected_gradient = audit.flat_gradient(expected_loss, list(self.model.parameters()))
        self.assertAlmostEqual(part[2]["probe_pair_consistency_contribution_loss"], float(expected_loss.detach()))
        torch.testing.assert_close(contrib[2]["probe_pair_consistency_contribution"], expected_gradient)
        self.assertNotAlmostEqual(float(expected_loss.detach()), float(square[mask].mean().detach()))

    def test_partitioned_pair_contributions_reconstruct_parent_terms(self):
        # Include every physical pair solely for this mathematical denominator test.
        pairs = [(0, 1), (0, 2), (2, 3), (1, 3), (0, 3), (1, 2)]
        probes = pd.DataFrame([{"src": a, "dst": b, "slot": i} for i, (a, b) in enumerate(pairs)])
        vectors, contrib, _, _ = audit.source_gradients(self.model, self.items[0], probes, self.device)
        torch.testing.assert_close(sum(c["probe_pair_consistency_contribution"] for c in contrib.values()), vectors["non_target_consistency"])
        torch.testing.assert_close(sum(c["probe_pair_adjacent_contribution"] for c in contrib.values()), vectors["adjacent_negative_consistency"])

    def test_probe_gradients_match_analytic_toy_bce_derivative(self):
        gradients, _ = audit.probe_gradients(self.model, self.items, self.probes, self.device)
        # Trigger target feature=.9, logit=.8*.9-.3=.42.
        derivative = torch.sigmoid(torch.tensor(.42)) - 1
        torch.testing.assert_close(gradients[0]["trigger"], torch.stack([derivative * .9, derivative, torch.tensor(0.)]))

    def test_audit_preserves_state_rng_and_empty_grad_buffers(self):
        before = audit.snapshot(self.model)
        rng = torch.random.get_rng_state().clone()
        with patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")):
            records, events = audit.diagnose_scene(self.model, self.items, self.sources, self.probes,
                                                   role="rotating_v4", device=self.device)
        audit.check_state(self.model, before)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(len(records), 4 * 2 * len(audit.COMPONENTS))
        self.assertEqual(len(events), 4)
        self.assertEqual({r["relation"] for r in records}, {"self", "other_shared_endpoint", "other_disjoint"})

    def test_training_mode_and_no_grad_are_rejected(self):
        self.model.train()
        with self.assertRaises(ValueError):
            audit.diagnose_scene(self.model, self.items, self.sources, self.probes, role="rotating_v4", device=self.device)
        self.model.eval()
        with torch.no_grad(), self.assertRaises(ValueError):
            audit.diagnose_scene(self.model, self.items, self.sources, self.probes, role="rotating_v4", device=self.device)

    def test_state_guard_catches_parameter_and_grad_mutation(self):
        before = audit.snapshot(self.model)
        self.model.scale.grad = torch.ones_like(self.model.scale)
        with self.assertRaises(RuntimeError): audit.check_state(self.model, before)
        self.model.scale.grad = None
        with torch.no_grad(): self.model.scale.add_(1.)
        with self.assertRaises(RuntimeError): audit.check_state(self.model, before)

    def test_synthetic_genconv_gradients_without_optimizer(self):
        model = build_model(model_name="genconv", node_dim=1, edge_dim=1, hidden_dim=8,
                            num_layers=1, dropout=.1, norm="layer", decoder_hidden_dim=8).eval()
        before = audit.snapshot(model)
        records, events = audit.diagnose_scene(model, self.items, self.sources, self.probes,
                                                role="rotating_v4", device=self.device)
        audit.check_state(model, before)
        self.assertEqual(len(events), 4)
        self.assertTrue(np.isfinite([r["trigger_dot"] for r in records]).all())


class BoundariesAndOutputTests(unittest.TestCase):
    def test_nontrain_missing_graph_and_sha_mismatch_rejected_before_loading(self):
        rows = audit.exposure_table(targets_fixture(1), 37)
        rows = rows[rows.scenario_id.eq("s2-0")]
        gr = SimpleNamespace(split="train", graph_sha256="a" * 64)
        with patch.object(audit, "load_split_graphs") as loader:
            for graph_rows in ({}, {"s2-0": SimpleNamespace(split="test", graph_sha256="a" * 64)},
                               {"s2-0": SimpleNamespace(split="train", graph_sha256="b" * 64)}):
                with self.assertRaises(ValueError): audit.load_scene(rows, graph_rows, Path("unused"))
            bad = rows.copy(); bad.loc[bad.index[0], "role"] = "holdout"
            with self.assertRaises(ValueError): audit.load_scene(bad, {"s2-0": gr}, Path("unused"))
            loader.assert_not_called()

    def test_check_only_cannot_load_models_graphs_or_create_output(self):
        with patch.object(audit, "preflight", return_value=context_fixture()), \
                patch.object(audit, "run") as run, patch.object(audit, "load_scene") as graphs, \
                patch.object(audit, "new_output") as output, \
                patch.object(audit.response.response, "_model_from_assets") as models, \
                redirect_stdout(io.StringIO()) as stream:
            audit.main(["--check-only"])
            self.assertEqual(json.loads(stream.getvalue())["status"], "PREFLIGHT_PASS")
            for blocked in (run, graphs, output, models): blocked.assert_not_called()

    def test_summary_averages_exposure_weighted_sources_within_scene_first(self):
        context = context_fixture()
        rows = []
        for scene, values, counts in (("a", [-1., 2.], [3, 1]), ("b", [2.], [4])):
            for source, (dot, count) in enumerate(zip(values, counts)):
                rows.append({"model_role": "rotating_v4", "probe_kind": "additional", "training_slots_in_scene": 3,
                             "component": "ranking", "relation": "other_shared_endpoint", "scenario_id": scene,
                             "probe_slot": 1, "source_slot": source, "v4_source_presentations": count,
                             "trigger_dot": dot, "clean_negative_dot": -dot})
        summary, table = audit.summarize(pd.DataFrame(rows), pd.DataFrame(), context)
        self.assertAlmostEqual(table.iloc[0].trigger_mean_schedule_weighted_dot, .875)
        self.assertEqual(table.iloc[0].trigger_negative_dot_scene_fraction, .5)
        self.assertEqual(summary["original_training_gate_status"], "STOP")
        self.assertIsNone(summary["new_metric_gate"])

    def test_new_output_refuses_existing_and_frozen_directories(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.response, "ROOT", Path(temp)):
            existing = Path(temp) / "record/reproduction/old"; existing.mkdir(parents=True)
            with self.assertRaises(FileExistsError): audit.new_output(existing)
            for name in ("v6", "v6_4"):
                with self.assertRaises(ValueError): audit.new_output(Path(temp) / "record" / name / "new")

    def test_synthetic_end_to_end_output_integrity_and_completeness_checks(self):
        context = context_fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.response, "ROOT", Path(temp)), \
                patch.object(audit, "load_scene", side_effect=lambda rows, *_: scene_items(rows)), \
                patch.object(audit.response.fixed, "read_json", return_value={}), \
                patch.object(audit.response.response, "_model_from_assets", side_effect=lambda *a, **k: ToyModel().eval()), \
                patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")), \
                redirect_stdout(io.StringIO()):
            output = audit.run(context, device_name="cpu", output_dir=Path(temp) / "record/reproduction/diagnostic")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "AUDIT_COMPLETE_REVIEW_REQUIRED")
            self.assertFalse(summary["scope"]["historical_training_replay"])
            for asset in summary["outputs"].values(): self.assertEqual(audit.sha256_file(asset["path"]), asset["sha256"])
            frame = pd.read_csv(output / "gradient.records.csv")
            events = pd.read_csv(output / "source_events.csv")
            audit.validate_records(frame, events, context["sampled_sources"], context["probes"])
            with self.assertRaises(ValueError):
                audit.validate_records(frame.iloc[1:], events, context["sampled_sources"], context["probes"])
            with self.assertRaises(ValueError):
                audit.validate_records(pd.concat([frame, frame.iloc[[0]]]), events, context["sampled_sources"], context["probes"])
            with self.assertRaises(ValueError):
                audit.validate_records(frame, events.iloc[1:], context["sampled_sources"], context["probes"])

    def test_integrity_failure_never_publishes_completion(self):
        context = context_fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.response, "ROOT", Path(temp)), \
                patch.object(audit, "load_scene", side_effect=lambda rows, *_: scene_items(rows)), \
                patch.object(audit.response.fixed, "read_json", return_value={}), \
                patch.object(audit.response.response, "_model_from_assets", side_effect=lambda *a, **k: ToyModel().eval()), \
                patch.object(audit.response, "verify_unchanged", side_effect=ValueError("changed asset")), \
                redirect_stdout(io.StringIO()):
            output = Path(temp) / "record/reproduction/failed"
            with self.assertRaisesRegex(ValueError, "changed asset"):
                audit.run(context, device_name="cpu", output_dir=output)
            self.assertTrue((output / "protocol.json").exists())
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
