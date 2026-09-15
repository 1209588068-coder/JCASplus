import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcas.workflows import relational_motion_branch_pilot as branch
from jcas.workflows import relational_motion_branch_anchored as anchor


class MotionAnchorTests(unittest.TestCase):
    def test_anchor_only_clean_rows(self):
        features = torch.zeros((10, 9))
        logits = torch.tensor([1., 1., 1., 1., 2., 2., 10., 10., -10., -10.], requires_grad=True)
        labels = torch.zeros(10)
        total, bce, penalty = anchor.anchored_loss(logits, features, labels, 4)
        self.assertAlmostEqual(float(penalty.detach()), 2.)
        penalty.backward()
        torch.testing.assert_close(logits.grad[6:], torch.zeros(4), atol=0, rtol=0)
        self.assertTrue(torch.all(logits.grad[:6] > 0))
        torch.testing.assert_close(total, bce+penalty)

    def test_zero_correction_zero_penalty_and_bad_layout(self):
        features = torch.ones((8, 9))
        _, _, penalty = anchor.anchored_loss(torch.ones(8), features, torch.zeros(8), 2)
        self.assertEqual(float(penalty), 0.)
        with self.assertRaises(ValueError):
            anchor.anchored_loss(torch.ones(8), features, torch.zeros(8), 3)

    def test_tiny_training_does_not_update_normalizer(self):
        rng = np.random.default_rng(21)
        replay = {"features": rng.normal(size=(12, 9)).astype(np.float32),
                  "labels": np.tile(np.array([0., 1.], dtype=np.float32), 6)}
        targets = rng.normal(size=(3, 3, 2, 9)).astype(np.float32)
        model = branch.Readout(branch.MotionResidual(replay["features"][:, 1:]))
        before = model.decoder.mlp.mean.clone()
        with tempfile.TemporaryDirectory() as tmp, patch.object(branch, "EPOCHS", 2), \
                patch.object(anchor.decoder, "TARGET_BATCH", 2), contextlib.redirect_stdout(io.StringIO()):
            anchor.fit(model, replay, targets, Path(tmp), "signed_motion", torch.device("cpu"))
            saved = torch.load(Path(tmp)/"signed_motion.last_branch.pt", weights_only=True)
            self.assertEqual(saved["version"], anchor.VERSION)
        torch.testing.assert_close(before, model.decoder.mlp.mean, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
