import unittest

import torch
from torch_geometric.data import Data

from jcas.workflows.explicit_counterfactual_trainer import (
    CounterfactualLossConfig,
    _paired_logits_with_shared_dropout,
    adjacent_negative_mask_for_pair,
    collate_counterfactual_items,
    compose_counterfactual_training_items,
    counterfactual_loss_components,
    objective_profile,
)


def _graph(
    scenario_id: str,
    *,
    triggered: bool = False,
    target_logit: float = 0.0,
    non_target_logit: float = 0.0,
) -> Data:
    target = torch.tensor([triggered, triggered, False, False])
    labels = torch.tensor(
        [float(triggered), float(triggered), 0.0, 1.0]
    )
    edge_attr = torch.tensor(
        [
            [target_logit],
            [target_logit],
            [non_target_logit],
            [non_target_logit],
        ],
        dtype=torch.float32,
    )
    return Data(
        scenario_id=scenario_id,
        x=torch.zeros((2, 1), dtype=torch.float32),
        edge_index=torch.tensor(
            [[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long
        ),
        edge_attr=edge_attr,
        edge_label=labels,
        supervision_edge_mask=torch.ones(4, dtype=torch.bool),
        message_passing_edge_mask=torch.ones(4, dtype=torch.bool),
        poisoned_edge_mask=target,
    )


class EdgeAttributeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index, edge_attr, message_mask):
        del x, edge_index, message_mask
        return edge_attr[:, 0] * self.scale


class DropoutEdgeAttributeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_attr, message_mask):
        del x, edge_index, message_mask
        return self.dropout(edge_attr[:, 0])


class ExplicitCounterfactualTrainerTests(unittest.TestCase):
    def test_composition_binds_each_triggered_row_to_clean_counterpart(self) -> None:
        clean = [_graph(f"scenario-{index}") for index in range(4)]
        triggered = [
            _graph("scenario-1", triggered=True, target_logit=2.0),
            _graph("scenario-3", triggered=True, target_logit=2.0),
        ]
        items, audit = compose_counterfactual_training_items(clean, triggered)
        self.assertEqual(len(items), 6)
        self.assertEqual(audit["counterfactual_pairs_per_epoch"], 2)
        paired = [item for item in items if item.has_counterfactual_pair]
        self.assertEqual(len(paired), 2)
        self.assertEqual(
            [item.graph.scenario_id for item in paired],
            ["scenario-1", "scenario-3"],
        )

    def test_collate_keeps_only_triggered_rows_in_auxiliary_pair_batch(self) -> None:
        clean = [_graph("clean"), _graph("poison")]
        triggered = [_graph("poison", triggered=True, target_logit=2.0)]
        items, _ = compose_counterfactual_training_items(clean, triggered)
        batch = collate_counterfactual_items(items)
        self.assertEqual(batch.base.num_graphs, len(items))
        self.assertEqual(batch.pair_count, 1)
        self.assertIsNotNone(batch.clean_counterparts)
        self.assertIsNotNone(batch.triggered_counterparts)

    def test_adjacent_negative_mask_excludes_target_and_positive_edges(self) -> None:
        clean = _graph("poison")
        triggered = _graph("poison", triggered=True, target_logit=2.0)
        mask = adjacent_negative_mask_for_pair(clean, triggered)
        self.assertEqual(mask.tolist(), [False, False, True, False])

    def test_explicit_losses_use_target_margin_and_ignore_target_consistency(self) -> None:
        clean = [_graph("poison", target_logit=0.0, non_target_logit=0.0)]
        triggered = [
            _graph(
                "poison",
                triggered=True,
                target_logit=2.0,
                non_target_logit=0.0,
            )
        ]
        items, _ = compose_counterfactual_training_items(clean, triggered)
        batch = collate_counterfactual_items(items)
        config = CounterfactualLossConfig(
            target_weight=0.25,
            ranking_weight=0.25,
            consistency_weight=1.0,
            ranking_margin=1.0,
            max_gradient_norm=5.0,
        )
        model = EdgeAttributeModel()
        losses, counts = counterfactual_loss_components(
            model,
            batch,
            config=config,
            device=torch.device("cpu"),
        )
        expected_target = 0.5 * (
            torch.nn.functional.softplus(torch.tensor(0.0))
            + torch.nn.functional.softplus(torch.tensor(-2.0))
        )
        expected_ranking = torch.nn.functional.softplus(torch.tensor(-1.0))
        self.assertTrue(torch.allclose(losses["target_pair_bce"], expected_target))
        self.assertTrue(torch.allclose(losses["ranking"], expected_ranking))
        self.assertEqual(
            float(losses["non_target_consistency"].detach()), 0.0
        )
        self.assertEqual(counts["counterfactual_pairs"], 1)
        self.assertEqual(counts["target_edges"], 2)
        self.assertEqual(counts["consistency_edges"], 2)
        self.assertEqual(counts["adjacent_negative_edges"], 1)
        expected_total = (
            losses["base_bce"]
            + 0.25 * expected_target
            + 0.25 * expected_ranking
        )
        self.assertTrue(torch.allclose(losses["total"], expected_total))
        losses["total"].backward()
        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(bool(torch.isfinite(model.scale.grad)))

    def test_non_target_consistency_detects_counterfactual_drift(self) -> None:
        clean = [_graph("poison", target_logit=0.0, non_target_logit=0.0)]
        triggered = [
            _graph(
                "poison",
                triggered=True,
                target_logit=2.0,
                non_target_logit=1.0,
            )
        ]
        items, _ = compose_counterfactual_training_items(clean, triggered)
        model = EdgeAttributeModel()
        batch = collate_counterfactual_items(items)
        losses, _ = counterfactual_loss_components(
            model,
            batch,
            config=CounterfactualLossConfig(),
            device=torch.device("cpu"),
        )
        expected = (torch.sigmoid(torch.tensor(1.0)) - 0.5).square()
        self.assertTrue(
            torch.allclose(losses["non_target_consistency"], expected)
        )
        self.assertTrue(
            torch.allclose(
                losses["adjacent_negative_consistency"], expected
            )
        )
        focused_losses, _ = counterfactual_loss_components(
            model,
            batch,
            config=CounterfactualLossConfig(
                adjacent_negative_consistency_weight=5.0
            ),
            device=torch.device("cpu"),
        )
        self.assertTrue(
            torch.allclose(
                focused_losses["total"] - losses["total"],
                5.0 * expected,
            )
        )

    def test_paired_forward_reuses_dropout_stream(self) -> None:
        graph = _graph("same", target_logit=1.0, non_target_logit=1.0)
        batch = collate_counterfactual_items(
            compose_counterfactual_training_items(
                [graph],
                [
                    _graph(
                        "same",
                        triggered=True,
                        target_logit=1.0,
                        non_target_logit=1.0,
                    )
                ],
            )[0]
        )
        model = DropoutEdgeAttributeModel()
        model.train()
        clean, triggered = _paired_logits_with_shared_dropout(
            model,
            batch.clean_counterparts,
            batch.triggered_counterparts,
        )
        self.assertTrue(torch.equal(clean, triggered))

    def test_loss_config_rejects_disabled_target_objective(self) -> None:
        with self.assertRaises(ValueError):
            CounterfactualLossConfig(
                target_weight=0.0,
                ranking_weight=0.0,
            ).validate()
        with self.assertRaises(ValueError):
            CounterfactualLossConfig(
                adjacent_negative_consistency_weight=-1.0
            ).validate()

    def test_balanced_v2_profile_is_exactly_bound(self) -> None:
        balanced = CounterfactualLossConfig(
            target_weight=0.10,
            ranking_weight=0.10,
            consistency_weight=1.0,
            adjacent_negative_consistency_weight=5.0,
            ranking_margin=1.0,
            max_gradient_norm=5.0,
        )
        self.assertEqual(
            objective_profile(balanced),
            "balanced_v2_t0p1_r0p1_c1_adj5_m1",
        )
        self.assertEqual(
            objective_profile(
                CounterfactualLossConfig(
                    target_weight=0.1001,
                    ranking_weight=0.10,
                    consistency_weight=1.0,
                    adjacent_negative_consistency_weight=5.0,
                )
            ),
            "custom_explicit_counterfactual_v2",
        )


if __name__ == "__main__":
    unittest.main()
