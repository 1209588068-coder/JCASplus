import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from jcas.workflows import relational_context_branch_pilot as context
from jcas.workflows import relational_motion_branch_pilot as branch


class ContextBranchTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(21)
        self.x = self.rng.normal(size=(24, 9)).astype(np.float32)

    def test_matched_initialization_and_zero_correction(self):
        heads = []
        for enabled in (False, True):
            torch.manual_seed(21)
            head = context.ContextResidual(self.x, context_enabled=enabled)
            torch.testing.assert_close(head(torch.from_numpy(self.x)), torch.from_numpy(self.x[:, :1]), atol=0, rtol=0)
            heads.append(head)
        for a, b in zip(heads[0].parameters(), heads[1].parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertTrue(torch.count_nonzero(heads[0].net[0].weight[:, 8]) == 0)

    def test_control_matches_original_motion_function_and_gradients(self):
        torch.manual_seed(21)
        old = branch.MotionResidual(self.x[:, 1:])
        torch.manual_seed(21)
        new = context.ContextResidual(self.x, context_enabled=False)
        with torch.no_grad():
            old.net[-1].weight.fill_(.1)
            new.net[-1].weight.copy_(old.net[-1].weight)
        x = torch.from_numpy(self.x)
        torch.testing.assert_close(old(x), new(x), atol=1e-6, rtol=1e-6)
        old(x).sum().backward()
        new(x).sum().backward()
        torch.testing.assert_close(old.net[0].weight.grad, new.net[0].weight.grad[:, :8], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new.net[0].weight.grad[:, 8], torch.zeros(32), atol=0, rtol=0)

    def test_context_can_change_correction_but_control_cannot(self):
        x = torch.zeros((2, 9))
        x[:, 0] = torch.tensor([-2., 2.])
        for enabled in (False, True):
            head = context.ContextResidual(self.x, context_enabled=enabled)
            with torch.no_grad():
                head.net[0].weight.zero_()
                head.net[0].bias.zero_()
                head.net[0].weight[0, 8] = 1.
                head.net[-1].weight.zero_()
                head.net[-1].weight[0, 0] = 1.
            correction = head(x)-x[:, :1]
            self.assertEqual(bool(correction[0] != correction[1]), enabled)

    def test_base_logit_detached_and_correction_bounded(self):
        head = context.ContextResidual(self.x, context_enabled=True)
        with torch.no_grad():
            head.net[-1].weight.fill_(100.)
        x = torch.tensor(self.x, requires_grad=True)
        output = head(x)
        output.sum().backward()
        torch.testing.assert_close(x.grad[:, 0], torch.zeros(len(x)), atol=0, rtol=0)
        self.assertTrue(((output.detach()-x.detach()[:, :1]).abs() <= 8.000001).all())

    def test_normalizer_excludes_development_targets(self):
        sample = pd.DataFrame({"scenario_id": ["a", "b"], "split_group_id": ["ga", "gb"],
                               "inner_role": ["construct", "develop"], "outer_split": ["train", "train"],
                               "motion_regime": ["slow_lt_0p5", "slow_lt_0p5"]})
        targets = self.rng.normal(size=(2, 3, 2, 9)).astype(np.float32)
        fit, mask = context.construct_fit_features({"features": self.x}, targets, sample)
        targets[1] = 1e9
        second, _ = context.construct_fit_features({"features": self.x}, targets, sample)
        np.testing.assert_array_equal(fit, second)
        np.testing.assert_array_equal(mask, [True, False])
        with self.assertRaises(ValueError):
            context.construct_fit_features({"features": self.x}, targets, sample.assign(outer_split="val"))

    def test_synthetic_fit_round_trip_and_context_learning(self):
        replay = {"features": self.x, "labels": np.tile([0., 1.], 12).astype(np.float32)}
        targets = self.rng.normal(size=(3, 3, 2, 9)).astype(np.float32)
        for enabled in (False, True):
            torch.manual_seed(21)
            model = branch.Readout(context.ContextResidual(self.x, context_enabled=enabled))
            before = model.decoder.mlp.context_mean.clone()
            with tempfile.TemporaryDirectory() as tmp, patch.object(context, "EPOCHS", 2), \
                    patch.object(context.decoder, "TARGET_BATCH", 2), contextlib.redirect_stdout(io.StringIO()):
                context.fit(model, replay, targets, Path(tmp), "arm", torch.device("cpu"), {"path": "synthetic"})
                saved = context.load_branch(Path(tmp)/"arm.last_branch.pt")
                self.assertEqual(saved.context_enabled, enabled)
                torch.testing.assert_close(saved(torch.from_numpy(self.x)), model.decoder.mlp(torch.from_numpy(self.x)), atol=0, rtol=0)
                with self.assertRaises(FileExistsError):
                    context.save_checkpoint(saved, Path(tmp)/"arm.last_branch.pt", {})
            torch.testing.assert_close(model.decoder.mlp.context_mean, before, atol=0, rtol=0)
            self.assertEqual(bool(torch.count_nonzero(model.decoder.mlp.net[0].weight[:, 8])), enabled)

    def test_gate_unchanged_and_control_not_mislabeled(self):
        def result(full=.7, reverse=.09, auc=.986):
            return {"clean_guard": {"auc": auc}, "responses": [{"cohort": "develop", "motion_regime": "slow_lt_0p5",
                "full_incremental_asr": full, "reversed_activation": reverse, "full_only_activation": .6}]}
        results = {"initial": result(auc=.987), "motion_only": result(), "motion_with_context": result()}
        gate = context.gate(results)
        self.assertEqual(gate["status"], "PASS")
        self.assertNotIn("signed_full_only_lift_over_magnitude", gate)
        self.assertFalse(gate["automatic_continuation"])
        for bad in (result(full=.59), result(reverse=.11), result(auc=.98), result(auc=float("nan"))):
            results["motion_with_context"] = bad
            self.assertEqual(context.gate(results)["status"], "STOP")

    def test_check_only_never_loads_arrays_or_starts_training(self):
        rows = pd.DataFrame({"scenario_id": ["a"], "split_group_id": ["g"],
                             "inner_role": ["construct"], "split": ["train"]})
        original = ({}, {}, .4, rows, rows)
        with patch.object(context, "preflight", return_value=(original, {}, {}, {})), \
                patch.object(context.np, "load", side_effect=AssertionError("array access")), \
                patch.object(context, "fit", side_effect=AssertionError("training")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(context.main(["--check-only"]), 0)


if __name__ == "__main__":
    unittest.main()
