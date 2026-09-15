import unittest

import torch
from torch_geometric.data import Data

from jcas.workflows.paired_counterfactual_trainer import (
    compose_training_graphs,
)


def _graph(scenario_id: str, *, poisoned: bool = False) -> Data:
    return Data(
        scenario_id=scenario_id,
        poisoned_edge_mask=torch.tensor(
            [poisoned, poisoned], dtype=torch.bool
        ),
    )


class PairedCounterfactualTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clean = [_graph(f"scenario-{index}") for index in range(10)]
        self.triggered = [
            _graph("scenario-1", poisoned=True),
            _graph("scenario-3", poisoned=True),
        ]

    def test_paired_keeps_clean_rows_and_adds_triggered_counterparts(self) -> None:
        rows, audit = compose_training_graphs(
            self.clean,
            self.triggered,
            presentation="paired",
            control_seed=17,
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(audit["clean_counterpart_rows"], 2)
        self.assertEqual(audit["unrelated_clean_control_duplicates"], 0)
        scenario_one = [row for row in rows if row.scenario_id == "scenario-1"]
        self.assertEqual(len(scenario_one), 2)
        self.assertEqual(
            sorted(int(row.poisoned_edge_mask.sum()) for row in scenario_one),
            [0, 2],
        )

    def test_size_control_replaces_poison_and_adds_unrelated_clean_rows(self) -> None:
        rows, audit = compose_training_graphs(
            self.clean,
            self.triggered,
            presentation="size_control",
            control_seed=17,
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(audit["clean_counterpart_rows"], 0)
        self.assertEqual(audit["unrelated_clean_control_duplicates"], 2)
        scenario_one = [row for row in rows if row.scenario_id == "scenario-1"]
        self.assertEqual(len(scenario_one), 1)
        self.assertEqual(int(scenario_one[0].poisoned_edge_mask.sum()), 2)

    def test_size_control_selection_is_deterministic(self) -> None:
        _, first = compose_training_graphs(
            self.clean,
            self.triggered,
            presentation="size_control",
            control_seed=23,
        )
        _, second = compose_training_graphs(
            list(reversed(self.clean)),
            list(reversed(self.triggered)),
            presentation="size_control",
            control_seed=23,
        )
        self.assertEqual(
            first["control_duplicate_scenario_sha256"],
            second["control_duplicate_scenario_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
