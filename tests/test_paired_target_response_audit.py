import unittest

import numpy as np
import pandas as pd

from jcas.workflows.paired_target_response_audit import (
    CollateralAccumulator,
    _distribution,
    summarize_target_records,
)


class PairedTargetResponseAuditTests(unittest.TestCase):
    def test_distribution_reports_expected_quantities(self) -> None:
        result = _distribution([0.0, 1.0, 2.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["median"], 1.0)
        self.assertAlmostEqual(result["sample_std"], 1.0)

    def test_target_summary_uses_incremental_flip_definition(self) -> None:
        frame = pd.DataFrame(
            {
                "clean_pair_probability": [0.1, 0.7, 0.2],
                "triggered_pair_probability": [0.8, 0.9, 0.3],
                "clean_pair_mean_logit": [-2.0, 1.0, -1.0],
                "triggered_pair_mean_logit": [2.0, 2.0, -0.5],
                "clean_negative_bce": [0.1, 1.0, 0.3],
                "triggered_positive_bce": [0.1, 0.2, 1.0],
            }
        )
        result = summarize_target_records(frame, threshold=0.5)
        self.assertEqual(result["incremental_flips"], 1)
        self.assertAlmostEqual(
            result["incremental_flip_rate_all_targets"], 1.0 / 3.0
        )
        self.assertAlmostEqual(result["conditional_flip_rate"], 0.5)
        self.assertEqual(result["logit_margin_ge_0p5_fraction"], 1.0)

    def test_collateral_accumulator_counts_transitions(self) -> None:
        accumulator = CollateralAccumulator()
        accumulator.add(
            np.asarray([0.1, 0.2, 0.8, 0.9]),
            np.asarray([0.2, 0.7, 0.3, 0.9]),
            np.asarray([True, True, True, True]),
            threshold=0.5,
        )
        result = accumulator.summary()
        self.assertEqual(result["edges"], 4)
        self.assertEqual(
            result["transitions"], {"n00": 1, "n01": 1, "n10": 1, "n11": 1}
        )
        self.assertAlmostEqual(result["incremental_positive_rate"], 0.25)
        self.assertAlmostEqual(result["suppression_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
