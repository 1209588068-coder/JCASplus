import unittest

import numpy as np
import pandas as pd
import torch

from jcas.workflows import relational_construct_conflict_audit as audit
from jcas.workflows import relational_motion_branch_pilot as branch


class ConstructConflictTests(unittest.TestCase):
    def test_partition_fail_closed(self):
        rows = pd.DataFrame({"scenario_id": ["a"], "split_group_id": ["g"],
                             "inner_role": ["construct"], "outer_split": ["train"]})
        audit.require_construct(rows)
        for role in ("develop", "audit"):
            with self.assertRaises(ValueError):
                audit.require_construct(rows.assign(inner_role=role))
        with self.assertRaises(ValueError):
            audit.require_construct(rows.assign(outer_split="test"))
        with self.assertRaises(ValueError):
            audit.require_construct(pd.concat([rows, rows]))

    def test_neighbors_exclude_entire_recording_and_stable_ties(self):
        q = np.array([[0., 0.]])
        ref = np.array([[0., 0.], [1., 0.], [-1., 0.], [2., 0.]])
        d, ids = audit.nearest_other_recording(q, ref, ["a"], ["a", "b", "c", "d"], 2)
        np.testing.assert_array_equal(ids, [[1, 2]])
        np.testing.assert_array_equal(d, [[1., 1.]])
        with self.assertRaises(ValueError):
            audit.nearest_other_recording(q, ref, ["a"], ["a"]*4, 1)

    def test_gradient_sign_and_no_mutation(self):
        torch.manual_seed(21)
        x = np.ones((2, 9), dtype=np.float32)
        head = branch.MotionResidual(np.zeros((1, 8)))
        before = {k: v.clone() for k, v in head.state_dict().items()}
        positive = audit.gradient(head, x, np.ones(2))
        negative = audit.gradient(head, x, np.zeros(2))
        self.assertAlmostEqual(audit.cosine(positive, negative), -1., places=6)
        self.assertIsNone(audit.cosine(positive, np.zeros_like(positive)))
        np.testing.assert_allclose(audit.gradient(head, x), 0, atol=1e-7)
        for k, v in head.state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_probe_groups_stay_together_and_reproducible(self):
        rng = np.random.default_rng(21)
        groups = np.repeat(np.arange(20).astype(str), 4)
        y = np.tile([0., 0., 1., 1.], 20)
        x = rng.normal(size=(80, 2))
        x[:, 0] += y*3
        _, scores, folds = audit.probe_cv(x, y, groups)
        self.assertTrue(np.isfinite(scores).all())
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups == group])), 1)
        _, second, second_folds = audit.probe_cv(x, y, groups)
        np.testing.assert_array_equal(scores, second)
        np.testing.assert_array_equal(folds, second_folds)


if __name__ == "__main__":
    unittest.main()
