import unittest

import numpy as np
import pandas as pd

from jcas.workflows.trigger_learnability_audit import (
    GradientCoherenceAccumulator,
    grouped_linear_probe,
    pair_feature_views,
    stable_subset,
)


class TriggerLearnabilityAuditTests(unittest.TestCase):
    def test_stable_subset_is_order_invariant(self) -> None:
        frame = pd.DataFrame(
            {"scenario_id": [f"scenario-{value}" for value in range(20)]}
        )
        first = stable_subset(frame, 7, seed=42)
        second = stable_subset(
            frame.sample(frac=1.0, random_state=3), 7, seed=42
        )
        self.assertEqual(
            first["scenario_id"].tolist(), second["scenario_id"].tolist()
        )

    def test_pair_feature_views_have_frozen_dimensions(self) -> None:
        x_node = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
        edge_attr = np.arange(4 * 42, dtype=np.float32).reshape(4, 42)
        views = pair_feature_views(
            x_node,
            edge_attr,
            src=0,
            dst=1,
            forward_edge_id=2,
            reverse_edge_id=3,
        )
        self.assertEqual(views["node_pair"].shape, (16,))
        self.assertEqual(views["edge_base34_pair"].shape, (68,))
        self.assertEqual(views["edge_motion8_pair"].shape, (16,))
        self.assertEqual(views["edge_augmented42_pair"].shape, (84,))
        self.assertEqual(views["all_pair_inputs"].shape, (100,))

    def test_gradient_coherence_reports_aligned_and_opposed_vectors(self) -> None:
        aligned = GradientCoherenceAccumulator()
        aligned.add(np.asarray([1.0, 0.0]))
        aligned.add(np.asarray([2.0, 0.0]))
        aligned_summary = aligned.summary()
        self.assertAlmostEqual(
            aligned_summary["mean_pairwise_cosine"], 1.0
        )
        self.assertAlmostEqual(
            aligned_summary["mean_unit_resultant_length"], 1.0
        )

        opposed = GradientCoherenceAccumulator()
        opposed.add(np.asarray([1.0, 0.0]))
        opposed.add(np.asarray([-1.0, 0.0]))
        opposed_summary = opposed.summary()
        self.assertAlmostEqual(
            opposed_summary["mean_pairwise_cosine"], -1.0
        )
        self.assertAlmostEqual(
            opposed_summary["mean_unit_resultant_length"], 0.0
        )

    def test_grouped_probe_detects_unseen_group_trigger_offset(self) -> None:
        rng = np.random.default_rng(7)
        values = []
        labels = []
        groups = []
        for group_id in range(40):
            baseline = rng.normal(0.0, 0.2, size=5)
            for label in (0, 1):
                row = baseline.copy()
                row[0] += 2.0 * label
                values.append(row)
                labels.append(label)
                groups.append(f"group-{group_id}")
        result = grouped_linear_probe(
            np.asarray(values),
            np.asarray(labels),
            groups,
            folds=5,
            seed=17,
            regularization_c=0.1,
        )
        self.assertGreater(result["oof_roc_auc"], 0.99)


if __name__ == "__main__":
    unittest.main()
