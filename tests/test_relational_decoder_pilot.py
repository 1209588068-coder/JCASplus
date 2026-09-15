import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_decoder_pilot as decoder
from tests.test_relational_poison import fixture


class DecoderPilotTests(unittest.TestCase):
    def test_matched_labels_and_clean_control_same_shapes(self):
        values = np.arange(5*3*2*4, dtype=np.float32).reshape(5, 3, 2, 4)
        original = values.copy()
        ids = np.array([1, 3])
        paired, y = decoder.auxiliary_batch(values, ids, "matched_relation")
        control, cy = decoder.auxiliary_batch(values, ids, "clean_replay")
        self.assertEqual(paired.shape, control.shape)
        np.testing.assert_array_equal(y.reshape(2, 3, 2), [[[0, 0], [1, 1], [0, 0]]]*2)
        np.testing.assert_array_equal(cy, 0.)
        for condition in range(3):
            np.testing.assert_array_equal(control.reshape(2, 3, 2, 4)[:, condition], values[ids, 0])
        np.testing.assert_array_equal(values, original)
        with self.assertRaises(ValueError):
            decoder.auxiliary_batch(values, ids, "bad")

    def test_clean_selection_is_stable_disjoint_and_role_limited(self):
        rows = pd.DataFrame([{"scenario_id": str(i), "inner_role": "construct" if i < 10 else "develop", "split": "train"}
                             for i in range(20)])
        targets = rows.iloc[[0, 10]]
        chosen = decoder.select_clean(rows, targets, "construct", 5)
        pd.testing.assert_frame_equal(chosen, decoder.select_clean(rows.sample(frac=1, random_state=7), targets, "construct", 5))
        self.assertFalse(set(chosen.scenario_id) & set(targets.scenario_id))
        self.assertEqual(set(chosen.inner_role), {"construct"})
        for role in ("audit", "val", "test"):
            with self.assertRaises(ValueError):
                decoder.select_clean(rows, targets, role, 1)

    def test_cache_equals_forward_and_hook_is_removed(self):
        clean, _ = pilot.clean_data(fixture(), "synthetic")
        model = build_model("genconv", node_dim=clean.x.shape[1], edge_dim=42,
                            hidden_dim=8, num_layers=1, decoder_hidden_dim=8)
        before = pilot.initialization_sha(model)
        z = decoder.capture_decoder(model, [clean, clean.clone()], torch.device("cpu"))
        self.assertEqual(len(z), 2)
        np.testing.assert_array_equal(z[0], z[1])
        self.assertEqual(before, pilot.initialization_sha(model))
        self.assertFalse(model.decoder.mlp._forward_pre_hooks)
        with patch.object(model, "forward", side_effect=RuntimeError("synthetic")):
            with self.assertRaises(RuntimeError):
                decoder.capture_decoder(model, [clean], torch.device("cpu"))
        self.assertFalse(model.decoder.mlp._forward_pre_hooks)

    def test_directed_pair_grouping_and_contradiction(self):
        clean, _ = pilot.clean_data(fixture(), "synthetic")
        ids, pairs = decoder.supervised_pairs(clean)
        np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        np.testing.assert_array_equal(pairs, [[0, 1], [2, 3]])
        clean.edge_label[0] = 1.
        with self.assertRaises(ValueError):
            decoder.supervised_pairs(clean)

    def test_small_update_changes_only_decoder(self):
        clean, _ = pilot.clean_data(fixture(), "synthetic")
        model = build_model("genconv", node_dim=clean.x.shape[1], edge_dim=42,
                            hidden_dim=8, num_layers=1, decoder_hidden_dim=8)
        z = decoder.capture_decoder(model, [clean], torch.device("cpu"))[0]
        features = np.stack([np.stack([z[:2], z[:2]+.01, z[:2]-.01])]*2)
        replay = {"features": z, "labels": np.array([0, 0, 1, 1], dtype=np.float32)}
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp, patch.object(decoder, "EPOCHS", 1), \
                patch.object(decoder, "TARGET_BATCH", 2), contextlib.redirect_stdout(io.StringIO()):
            history = decoder.train_readout(model, replay, features, "matched_relation", Path(tmp), torch.device("cpu"))
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith("decoder.mlp.") for k in changed))
        self.assertEqual(history[0]["optimizer_steps"], 1)
        self.assertEqual(history[0]["directed_rows"], 4+2*3*2)

    def test_gate_needs_full_reverse_and_clean_retention(self):
        def result(full=.7, reversed=.05, auc=.986):
            return {"responses": [{"cohort": "develop", "motion_regime": "slow_lt_0p5",
                "full_incremental_asr": full, "reversed_activation": reversed, "full_only_activation": .6}],
                "clean_guard": {"auc": auc}}
        values = {arm: result() for arm in ("initial", *decoder.ARMS)}
        self.assertEqual(decoder.gate(values)["status"], "PASS")
        for bad in (result(full=.59), result(reversed=.11), result(auc=.98), result(auc=None)):
            self.assertEqual(decoder.gate({**values, "matched_relation": bad})["status"], "STOP")


if __name__ == "__main__":
    unittest.main()
