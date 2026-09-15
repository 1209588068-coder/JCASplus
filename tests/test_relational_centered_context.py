import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcas.workflows import relational_centered_context_pilot as centered


class CenteredContextTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(21).normal(size=(24, 9)).astype(np.float32)

    def head(self, enabled):
        torch.manual_seed(21)
        return centered.CenteredContextResidual(self.x, centered=enabled)

    def test_matched_initial_parameters_and_zero_output(self):
        a, b = self.head(False), self.head(True)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        for model in (a, b):
            torch.testing.assert_close(model(torch.from_numpy(self.x)), torch.from_numpy(self.x[:, :1]), atol=0, rtol=0)

    def test_control_exactly_matches_previous(self):
        a = self.head(False)
        torch.manual_seed(21)
        b = centered.previous.ContextResidual(self.x, context_enabled=True)
        with torch.no_grad():
            a.net[-1].weight.fill_(.7)
            b.net[-1].weight.fill_(.7)
        torch.testing.assert_close(a(torch.from_numpy(self.x)), b(torch.from_numpy(self.x)), atol=0, rtol=0)

    def test_reference_is_identity_with_zero_parameter_gradients(self):
        head = self.head(True)
        with torch.no_grad():
            for p in head.parameters():
                p.uniform_(-2., 2.)
        x = torch.from_numpy(self.x.copy())
        x[:, 1:] = head.mean
        out = head(x)
        torch.testing.assert_close(out, x[:, :1], atol=0, rtol=0)
        out.sum().backward()
        for p in head.parameters():
            torch.testing.assert_close(p.grad, torch.zeros_like(p), atol=0, rtol=0)

    def test_pure_context_mapping_cancels(self):
        head = self.head(True)
        with torch.no_grad():
            head.net[0].weight[:, :8].zero_()
            head.net[0].weight[:, 8].fill_(3.)
            head.net[-1].weight.fill_(2.)
            head.net[-1].bias.fill_(4.)
        torch.testing.assert_close(head(torch.from_numpy(self.x)), torch.from_numpy(self.x[:, :1]), atol=0, rtol=0)

    def test_bound_and_base_detach(self):
        head = self.head(True)
        with torch.no_grad():
            head.net[-1].weight.fill_(50.)
        x = torch.tensor(self.x, requires_grad=True)
        out = head(x)
        self.assertTrue(((out-x[:, :1]).abs() <= 8.000001).all())
        out.sum().backward()
        torch.testing.assert_close(x.grad[:, 0], torch.zeros(len(x)), atol=0, rtol=0)

    def test_synthetic_training_and_restore(self):
        rng = np.random.default_rng(22)
        targets = rng.normal(size=(3, 3, 2, 9)).astype(np.float32)
        replay = {"features": self.x, "labels": np.tile([0., 1.], 12).astype(np.float32)}
        model = centered.branch.Readout(self.head(True))
        with tempfile.TemporaryDirectory() as tmp, patch.object(centered, 'EPOCHS', 2), \
                patch.object(centered.decoder, 'TARGET_BATCH', 2), contextlib.redirect_stdout(io.StringIO()):
            centered.fit(model, replay, targets, Path(tmp), 'candidate', torch.device('cpu'), {})
            restored = centered.load_branch(Path(tmp)/'candidate.last_branch.pt')
            torch.testing.assert_close(restored(torch.from_numpy(self.x)), model.decoder.mlp(torch.from_numpy(self.x)), atol=0, rtol=0)
            reference = torch.from_numpy(self.x.copy())
            reference[:, 1:] = restored.mean
            torch.testing.assert_close(restored(reference), reference[:, :1], atol=0, rtol=0)
            self.assertTrue(torch.count_nonzero(restored.net[-1].weight))

    def test_gate_preserves_limits(self):
        def result(auc, reverse=.09):
            return {'clean_guard': {'auc': auc}, 'responses': [{'cohort': 'develop', 'motion_regime': 'slow_lt_0p5',
                'full_incremental_asr': .7, 'reversed_activation': reverse, 'full_only_activation': .6}]}
        results = {'initial': result(.987), 'uncentered_context': result(.984), 'centered_context': result(.986)}
        self.assertEqual(centered.gate(results)['status'], 'PASS')
        results['centered_context'] = result(.986, .11)
        self.assertEqual(centered.gate(results)['status'], 'STOP')
        self.assertNotIn('signed_full_only_lift_over_magnitude', centered.gate(results))


if __name__ == '__main__':
    unittest.main()
