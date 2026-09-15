import unittest

import pandas as pd
import torch

from jcas.workflows.counterfactual_gradient_conflict_audit import (
    REGIMES,
    combine_gradients,
    compare_gradients,
    cpu_gradient_copy,
    parameter_block,
    stable_regime_sample,
    subtract_gradients,
    summarize_gradient_records,
)


class ParameterBlockTests(unittest.TestCase):
    def test_known_genconv_blocks(self) -> None:
        self.assertEqual(parameter_block("node_encoder.net.0.weight"), "node_encoder")
        self.assertEqual(parameter_block("edge_encoder.net.0.weight"), "edge_encoder")
        self.assertEqual(parameter_block("convs.0.mlp.0.weight"), "message_passing")
        self.assertEqual(parameter_block("norms.2.bias"), "message_passing")
        self.assertEqual(parameter_block("decoder.mlp.net.0.weight"), "decoder")

    def test_unknown_parameter_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            parameter_block("mystery.weight")


class StableSampleTests(unittest.TestCase):
    def test_sample_is_order_invariant_and_balanced(self) -> None:
        rows = []
        for regime in REGIMES:
            for index in range(8):
                rows.append(
                    {"scenario_id": f"{regime}-{index}", "motion_regime": regime}
                )
        frame = pd.DataFrame(rows)
        one = stable_regime_sample(frame, samples_per_regime=4, audit_seed=11)
        two = stable_regime_sample(
            frame.sample(frac=1.0, random_state=7),
            samples_per_regime=4,
            audit_seed=11,
        )
        self.assertEqual(one.to_dict("records"), two.to_dict("records"))
        self.assertEqual(one.groupby("motion_regime").size().to_dict(), {
            regime: 4 for regime in REGIMES
        })


class GradientMathTests(unittest.TestCase):
    def test_aligned_opposed_and_orthogonal_directions(self) -> None:
        base = (torch.tensor([1.0, 0.0]),)
        aligned = compare_gradients(base, (torch.tensor([2.0, 0.0]),), [0])
        opposed = compare_gradients(base, (torch.tensor([-3.0, 0.0]),), [0])
        orthogonal = compare_gradients(base, (torch.tensor([0.0, 4.0]),), [0])
        self.assertAlmostEqual(aligned["cosine"], 1.0)
        self.assertFalse(aligned["conflict"])
        self.assertAlmostEqual(opposed["cosine"], -1.0)
        self.assertTrue(opposed["conflict"])
        self.assertAlmostEqual(orthogonal["cosine"], 0.0)
        self.assertFalse(orthogonal["conflict"])

    def test_combination_and_subtraction(self) -> None:
        gradients = {
            "a": (torch.tensor([1.0, 2.0]), None),
            "b": (torch.tensor([3.0, 4.0]), torch.tensor([2.0])),
        }
        combined = combine_gradients(gradients, {"a": 0.5, "b": 2.0})
        self.assertTrue(torch.equal(combined[0], torch.tensor([6.5, 9.0])))
        self.assertTrue(torch.equal(combined[1], torch.tensor([4.0])))
        difference = subtract_gradients(combined, gradients["b"])
        self.assertTrue(torch.equal(difference[0], torch.tensor([3.5, 5.0])))
        self.assertTrue(torch.equal(difference[1], torch.tensor([2.0])))

    def test_cpu_copy_is_detached_and_independent(self) -> None:
        source = torch.tensor([1.0], requires_grad=True)
        copied = cpu_gradient_copy((source, None))
        self.assertFalse(copied[0].requires_grad)
        source.detach().add_(2.0)
        self.assertEqual(float(copied[0].item()), 1.0)
        self.assertIsNone(copied[1])

    def test_summary_reports_conflict_fraction(self) -> None:
        rows = []
        for index, cosine in enumerate((-0.5, 0.25)):
            for block in ("all", "node_encoder", "edge_encoder", "message_passing", "decoder"):
                rows.append({
                    "motion_regime": "slow_lt_0p5",
                    "regime_batch_index": index,
                    "pairs": 8,
                    "component": "ranking",
                    "parameter_block": block,
                    "cosine": cosine,
                    "conflict": cosine < 0.0,
                    "candidate_to_base_norm_ratio": 2.0,
                    "effective_candidate_to_base_norm_ratio": 0.2,
                    "candidate_projection_in_base_units": cosine * 2.0,
                    "base_gradient_norm": 1.0,
                    "candidate_gradient_norm": 2.0,
                })
        summary = summarize_gradient_records(rows)
        all_block = summary["slow_lt_0p5"]["ranking"]["all"]
        self.assertEqual(all_block["batches"], 2)
        self.assertAlmostEqual(all_block["conflict_fraction"], 0.5)
        self.assertAlmostEqual(all_block["cosine"]["mean"], -0.125)


if __name__ == "__main__":
    unittest.main()
