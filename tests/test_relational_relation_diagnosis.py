import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_relation_diagnosis as diagnosis
from tests.test_relational_bce_pilot import target
from tests.test_relational_poison import fixture


def planning_rows():
    rows = []
    for role in ("construct", "develop", "audit"):
        for regime in pilot.stage1.REGIMES:
            for index in range(160):
                scene = f"{role}-{regime}-{index}"
                rows.append({"scenario_id": scene, "split_group_id": scene, "inner_role": role,
                             "motion_regime": regime, "planned_role": "positive", "planned_variant": "full"})
            if role == "construct":
                for index in range(3):
                    scene = f"negative-{regime}-{index}"
                    rows.append({"scenario_id": scene, "split_group_id": scene, "inner_role": role,
                                 "motion_regime": regime, "planned_role": "negative_control", "planned_variant": "relation_reversed"})
    return pd.DataFrame(rows)


class DiagnosisTests(unittest.TestCase):
    def test_sample_fixed_independent_of_input_order_and_excludes_audit(self):
        rows = planning_rows()
        sample = diagnosis.sample_targets(rows)
        pd.testing.assert_frame_equal(sample, diagnosis.sample_targets(rows.sample(frac=1, random_state=8)))
        self.assertEqual(sample.cohort.value_counts().to_dict(),
                         {"construct_positive": 256, "develop": 128, "construct_reversed": 6})
        self.assertNotIn("audit", set(sample.inner_role))

    def test_duplicate_and_recording_leakage_rejected(self):
        rows = planning_rows()
        with self.assertRaises(ValueError):
            diagnosis.sample_targets(pd.concat([rows, rows.iloc[:1]]))
        rows["split_group_id"] = "same-recording"
        with self.assertRaises(ValueError):
            diagnosis.sample_targets(rows)

    def test_graph_scope_fails_before_file_access(self):
        with patch.object(pilot, "graph_path", side_effect=AssertionError("file checked")):
            for role in ("audit", "val", "test"):
                with self.assertRaises(ValueError):
                    diagnosis.graph_path({}, pd.Series({"inner_role": role}))

    def test_pooled_is_pair_order_invariant(self):
        values = np.array([[1., 3.], [4., 2.]])
        np.testing.assert_array_equal(diagnosis.pooled(values), diagnosis.pooled(values[::-1]))
        for invalid in (np.zeros((3, 2)), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                diagnosis.pooled(invalid)

    def test_hooks_preserve_forward_weights_and_are_removed(self):
        graph = fixture()
        clean, labels = pilot.clean_data(graph, "synthetic")
        ids = pilot.checked_target(graph, target(), labels)
        data = [clean]+[pilot.transformed_data(graph, clean, target(), ids, controller=True,
                         variant=v, poison_label=False) for v in diagnosis.CONDITIONS[1:]]
        model = build_model("genconv", node_dim=clean.x.shape[1], edge_dim=42,
                            hidden_dim=8, num_layers=1, decoder_hidden_dim=8)
        model.eval()
        before = pilot.initialization_sha(model)
        views, scores = diagnosis.extract(model, data, target(), ids, torch.device("cpu"))
        reference = [v[ids].mean() for v in pilot.predict(model, data, torch.device("cpu"))]
        np.testing.assert_allclose(scores, reference, atol=1e-7)
        self.assertEqual(before, pilot.initialization_sha(model))
        self.assertEqual(set(views), set(diagnosis.VIEWS))
        self.assertTrue(all(len(v) == 3 for v in views.values()))
        self.assertEqual(len(model.edge_encoder._forward_hooks), 0)
        self.assertEqual(len(model.decoder.mlp._forward_pre_hooks), 0)
        with patch.object(model, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                diagnosis.extract(model, data, target(), ids, torch.device("cpu"))
        self.assertEqual(len(model.edge_encoder._forward_hooks), 0)
        self.assertEqual(len(model.decoder.mlp._forward_pre_hooks), 0)

    def test_probe_distinguishes_signed_signal_without_gnn_training(self):
        sample = pd.DataFrame({"scenario_id": [str(i) for i in range(20)],
                               "split_group_id": [str(i) for i in range(20)],
                               "inner_role": ["construct"]*12+["develop"]*8,
                               "motion_regime": ["slow_lt_0p5"]*20})
        values = np.zeros((20, 3, 2), dtype=np.float32)
        values[:, 1, 0] = 1
        values[:, 2, 0] = -1
        original = values.copy()
        for relative in (False, True):
            row, scores = diagnosis.linear_probe(values, sample, "slow_lt_0p5",
                                                  task="full_vs_reversed", clean_relative=relative)
            self.assertEqual(row["auc"], 1.)
            self.assertEqual(row["balanced_accuracy"], 1.)
            self.assertEqual(row["within_scene_full_rank"], 1.)
            self.assertEqual(len(scores), 16)
        np.testing.assert_array_equal(values, original)
        sample["split_group_id"] = "leak"
        with self.assertRaises(ValueError):
            diagnosis.linear_probe(values, sample, "slow_lt_0p5", task="full_vs_reversed")

    def test_seen_negative_activation_includes_already_active_targets(self):
        frame = pd.DataFrame([{"model": "geometry_shared", "cohort": "construct_reversed",
                               "motion_regime": "slow_lt_0p5", "clean": .8, "full": .9, "relation_reversed": .8},
                              {"model": "geometry_shared", "cohort": "construct_reversed",
                               "motion_regime": "slow_lt_0p5", "clean": .1, "full": .9, "relation_reversed": .1}])
        row = diagnosis.response_summary(frame, .5)[0]
        self.assertEqual(row["reversed_activation"], .5)
        self.assertEqual(row["reversed_incremental_asr"], 0.)
        self.assertEqual(row["full_only_activation"], .5)


if __name__ == "__main__":
    unittest.main()
