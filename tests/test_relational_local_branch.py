import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jcas.workflows import relational_local_branch_pilot as local
from jcas.workflows import relational_motion_branch_pilot as branch


class LocalBranchTests(unittest.TestCase):
    def setUp(self):
        self.fit = np.random.default_rng(5).normal(size=(30, 8))

    def test_zero_identity_and_far_field_no_correction(self):
        with patch.object(local, "CENTERS", 4):
            model = local.LocalResidual(self.fit, self.fit)
        x = torch.ones((3, 9))
        torch.testing.assert_close(model(x), x[:, :1], rtol=0, atol=0)
        with torch.no_grad():
            model.readout.weight.fill_(2.)
        x[:, 1:] = 1e6
        torch.testing.assert_close(model(x), x[:, :1], rtol=0, atol=0)

    def test_basis_reproducible_and_abs_invariant(self):
        with patch.object(local, "CENTERS", 4):
            a = local.LocalResidual(self.fit, self.fit, magnitude_only=True)
            b = local.LocalResidual(self.fit, self.fit, magnitude_only=True)
        torch.testing.assert_close(a.centers, b.centers, atol=0, rtol=0)
        self.assertTrue(torch.all(a.width > 0))
        with torch.no_grad():
            a.readout.weight.fill_(1.)
        x = torch.tensor(np.column_stack([np.zeros(3), self.fit[:3]]), dtype=torch.float32)
        y = x.clone(); y[:, 1:] *= -1
        torch.testing.assert_close(a(x), a(y), atol=0, rtol=0)

    def test_tiny_fit_only_changes_readout(self):
        rng = np.random.default_rng(9)
        replay = {"features": rng.normal(size=(12, 9)).astype(np.float32),
                  "labels": np.tile(np.array([0., 1.], dtype=np.float32), 6)}
        targets = rng.normal(size=(3, 3, 2, 9)).astype(np.float32)
        with patch.object(local, "CENTERS", 4):
            model = branch.Readout(local.LocalResidual(self.fit, self.fit))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp, patch.object(branch, "EPOCHS", 2), \
                patch.object(local.decoder, "TARGET_BATCH", 2), contextlib.redirect_stdout(io.StringIO()):
            local.fit(model, replay, targets, Path(tmp), "signed_motion", torch.device("cpu"))
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
        self.assertEqual(changed, ["decoder.mlp.readout.weight"])


if __name__ == "__main__":
    unittest.main()
