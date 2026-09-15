import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_motion_branch_pilot as branch
from tests.test_relational_poison import fixture


class MotionBranchTests(unittest.TestCase):
    def test_zero_initialization_exactly_preserves_base_logits(self):
        fit = np.random.default_rng(7).normal(size=(20, 8))
        values = torch.tensor(np.random.default_rng(8).normal(size=(12, 9)), dtype=torch.float32)
        for magnitude in (False, True):
            model = branch.MotionResidual(fit, magnitude_only=magnitude)
            torch.testing.assert_close(model(values), values[:, :1], atol=0, rtol=0)

    def test_abs_control_discards_only_sign_in_adapter(self):
        fit = np.random.default_rng(7).normal(size=(20, 8))
        x = torch.ones((4, 9))
        y = x.clone()
        y[:, 1:] *= -1
        for magnitude in (False, True):
            model = branch.MotionResidual(fit, magnitude_only=magnitude)
            with torch.no_grad():
                model.net[0].weight.fill_(.1)
                model.net[0].bias.zero_()
                model.net[-1].weight.fill_(.1)
            if magnitude:
                torch.testing.assert_close(model(x), model(y), atol=0, rtol=0)
            else:
                self.assertFalse(torch.allclose(model(x), model(y)))

    def test_finite_normalization_and_correction_bound(self):
        model = branch.MotionResidual(np.zeros((4, 8)))
        with torch.no_grad():
            model.net[-1].bias.fill_(100.)
        x = torch.full((2, 9), 1000.)
        self.assertTrue(torch.isfinite(model(x)).all())
        self.assertTrue(((model(x)-x[:, :1]).abs() <= 8.).all())
        for bad in (np.zeros((0, 8)), np.zeros((3, 7)), np.full((3, 8), np.nan)):
            with self.assertRaises(ValueError):
                branch.MotionResidual(bad)

    def test_full_graph_forward_matches_zero_branch_and_no_base_grad(self):
        clean, _ = pilot.clean_data(fixture(), "synthetic")
        base = build_model("genconv", node_dim=clean.x.shape[1], edge_dim=42,
                           hidden_dim=8, num_layers=1, decoder_hidden_dim=8)
        base.eval()
        batch = Batch.from_data_list([clean])
        model = branch.MotionResidual(clean.edge_attr.numpy()[:, 34:42])
        before = pilot.initialization_sha(base)
        logits = branch.augmented_forward(base, model, batch)
        reference = base(batch.x, batch.edge_index, batch.edge_attr, batch.message_passing_edge_mask)
        torch.testing.assert_close(logits, reference, atol=0, rtol=0)
        logits.sum().backward()
        self.assertTrue(all(v.grad is None for v in base.parameters()))
        self.assertEqual(before, pilot.initialization_sha(base))

    def test_tiny_branch_training_and_checkpoint(self):
        rng = np.random.default_rng(9)
        replay = {"features": rng.normal(size=(12, 9)).astype(np.float32),
                  "labels": np.tile(np.array([0., 1.], dtype=np.float32), 6)}
        target = rng.normal(size=(4, 3, 2, 9)).astype(np.float32)
        model = branch.Readout(branch.MotionResidual(replay["features"][:, 1:]))
        mean = model.decoder.mlp.mean.clone()
        with tempfile.TemporaryDirectory() as tmp, patch.object(branch, "EPOCHS", 2), \
                patch.object(branch.previous, "TARGET_BATCH", 2), contextlib.redirect_stdout(io.StringIO()):
            branch.fit(model, replay, target, Path(tmp), "signed_motion", torch.device("cpu"))
            saved = torch.load(Path(tmp)/"signed_motion.last_branch.pt", weights_only=True)
            self.assertFalse(saved["magnitude_only"])
        self.assertTrue(torch.count_nonzero(model.decoder.mlp.net[-1].weight))
        torch.testing.assert_close(model.decoder.mlp.mean, mean, atol=0, rtol=0)

    def test_gate_retains_all_three_limits(self):
        def result(full=.7, reverse=.05, auc=.986):
            return {"clean_guard": {"auc": auc}, "responses": [{"cohort": "develop", "motion_regime": "slow_lt_0p5",
                    "full_incremental_asr": full, "reversed_activation": reverse, "full_only_activation": .6}]}
        values = {arm: result() for arm in ("initial", *branch.ARMS)}
        self.assertEqual(branch.gate(values)["status"], "PASS")
        for bad in (result(full=.59), result(reverse=.11), result(auc=.98), result(auc=None)):
            self.assertEqual(branch.gate({**values, "signed_motion": bad})["status"], "STOP")


if __name__ == "__main__":
    unittest.main()
