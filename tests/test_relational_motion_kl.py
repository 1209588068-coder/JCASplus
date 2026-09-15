import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcas.workflows import relational_motion_branch_pilot as branch
from jcas.workflows import relational_motion_branch_kl as kl


class MotionKLTests(unittest.TestCase):
    def test_matches_bernoulli_kl_and_direction(self):
        teacher = torch.tensor([-8., -2., 0., 1., 5.], dtype=torch.float64)
        student = torch.tensor([2., -1., 1., -3., 4.], dtype=torch.float64)
        expected = torch.distributions.kl_divergence(torch.distributions.Bernoulli(logits=teacher),
                                                   torch.distributions.Bernoulli(logits=student))
        torch.testing.assert_close(kl.bernoulli_kl(teacher, student), expected, atol=1e-12, rtol=1e-12)
        self.assertFalse(torch.allclose(kl.bernoulli_kl(teacher, student), kl.bernoulli_kl(student, teacher)))

    def test_identical_and_extreme_logits_finite_teacher_detached(self):
        teacher = torch.tensor([-1000., -80., 0., 80., 1000.], requires_grad=True)
        student = teacher.detach().clone().requires_grad_(True)
        identical = kl.bernoulli_kl(teacher, student)
        torch.testing.assert_close(identical, torch.zeros(5, dtype=torch.float64), rtol=0, atol=0)
        value = kl.bernoulli_kl(teacher, -student).sum()
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_anchor_only_replay_and_auxiliary_clean(self):
        features = torch.zeros((10, 9), requires_grad=True)
        logits = torch.tensor([1., 1., 1., 1., 2., 2., 9., 9., -9., -9.], requires_grad=True)
        labels = torch.zeros(10)
        total, bce, penalty = kl.kl_loss(logits, features, labels, 4)
        expected = kl.bernoulli_kl(torch.zeros(6), logits[:6]).mean()
        torch.testing.assert_close(penalty, expected)
        torch.testing.assert_close(total, bce+penalty)
        penalty.backward()
        self.assertIsNone(features.grad)
        torch.testing.assert_close(logits.grad[6:], torch.zeros(4), atol=0, rtol=0)
        self.assertTrue((logits.grad[:6] > 0).all())

    def test_invalid_layout(self):
        with self.assertRaises(ValueError):
            kl.kl_loss(torch.zeros(8), torch.zeros((8, 9)), torch.zeros(8), 3)
        with self.assertRaises(ValueError):
            kl.bernoulli_kl(torch.zeros(2), torch.zeros(3))

    def test_high_confidence_same_logit_shift_less_penalized(self):
        penalty = kl.bernoulli_kl(torch.tensor([-8., 0.]), torch.tensor([-7., 1.]))
        self.assertLess(float(penalty[0]), float(penalty[1]))

    def test_tiny_training_preserves_normalizer_and_serializes(self):
        rng = np.random.default_rng(21)
        replay = {"features": rng.normal(size=(12, 9)).astype(np.float32),
                  "labels": np.tile(np.array([0., 1.], dtype=np.float32), 6)}
        targets = rng.normal(size=(3, 3, 2, 9)).astype(np.float32)
        model = branch.Readout(branch.MotionResidual(replay["features"][:, 1:]))
        before = model.decoder.mlp.mean.clone()
        with tempfile.TemporaryDirectory() as tmp, patch.object(branch, "EPOCHS", 2), \
                patch.object(kl.decoder, "TARGET_BATCH", 2), contextlib.redirect_stdout(io.StringIO()):
            kl.fit(model, replay, targets, Path(tmp), "signed_motion", torch.device("cpu"))
            saved = torch.load(Path(tmp)/"signed_motion.last_branch.pt", weights_only=True)
            self.assertEqual(saved["version"], kl.VERSION)
        torch.testing.assert_close(before, model.decoder.mlp.mean, atol=0, rtol=0)
        self.assertTrue(torch.count_nonzero(model.decoder.mlp.net[-1].weight))


if __name__ == "__main__":
    unittest.main()
