import copy
import unittest

import torch
import torch.nn.functional as F

from jcas.core.counterfactual_gradients import (
    CleanProjectionConfig,
    assign_flat_gradient,
    project_extra_gradient,
)
from jcas.core.models import build_model
from jcas.workflows.explicit_counterfactual_trainer import (
    CounterfactualLossConfig,
    CounterfactualPresentationItem,
    collate_counterfactual_items,
    compose_counterfactual_training_items,
    counterfactual_loss_components,
    mixed_clean_reference_gradient,
    train_counterfactual_epoch,
)
from jcas.workflows.projected_counterfactual_pilot import (
    ROLE, evaluate_gate, validate_full_validation_counts,
)
from jcas.workflows.trainer import build_optimizer
from tests.test_explicit_counterfactual_trainer import _graph, EdgeAttributeModel


BALANCED = CounterfactualLossConfig(
    target_weight=0.1, ranking_weight=0.1,
    consistency_weight=1.0, adjacent_negative_consistency_weight=5.0,
)


class ProjectionMathTests(unittest.TestCase):
    def test_conflict_removed_before_cap(self):
        update, stats = project_extra_gradient(
            torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 4.0]),
            config=CleanProjectionConfig(),
        )
        torch.testing.assert_close(update, torch.tensor([1.0, 1.0]))
        self.assertTrue(stats["projected"])
        self.assertTrue(stats["capped"])
        self.assertEqual(stats["extra_clean_dot_after"], 0.0)

    def test_aligned_small_extra_is_unchanged(self):
        clean = torch.tensor([2.0, 0.0])
        total = torch.tensor([2.5, 0.5])
        update, stats = project_extra_gradient(clean, total, config=CleanProjectionConfig())
        torch.testing.assert_close(update, total)
        self.assertFalse(stats["projected"])
        self.assertFalse(stats["capped"])

    def test_zero_and_tiny_clean_drop_extra(self):
        for clean in (torch.zeros(2), torch.tensor([1e-14, 0.0])):
            update, stats = project_extra_gradient(
                clean, torch.tensor([5.0, -3.0]), config=CleanProjectionConfig()
            )
            torch.testing.assert_close(update, clean)
            self.assertTrue(stats["clean_norm_floor_hit"])

    def test_nonfinite_gradient_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(FloatingPointError):
                project_extra_gradient(
                    torch.ones(2), torch.tensor([value, 1.0]), config=CleanProjectionConfig()
                )

    def test_random_gradients_obey_halfspace_and_norm_constraint(self):
        generator = torch.Generator().manual_seed(20260907)
        for _ in range(30):
            clean = torch.randn(100, generator=generator)
            total = 20 * torch.randn(100, generator=generator)
            update, _ = project_extra_gradient(clean, total, config=CleanProjectionConfig())
            extra = update - clean
            self.assertGreaterEqual(float(torch.dot(clean, extra)), -1e-5)
            self.assertLessEqual(float(extra.norm()), float(clean.norm()) * 1.000001)

    def test_unused_parameter_keeps_none_gradient(self):
        params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))]
        assign_flat_gradient(torch.ones(3), params, used=[True, False])
        self.assertIsNotNone(params[0].grad)
        self.assertIsNone(params[1].grad)


def mixed_items():
    clean = [_graph("ordinary", target_logit=-3), _graph("paired", target_logit=-0.5)]
    triggered = [_graph("paired", triggered=True, target_logit=1.2, non_target_logit=0.8)]
    items, _ = compose_counterfactual_training_items(clean, triggered)
    # Ordinary and triggered rows, excluding the separately presented paired clean row.
    return [items[0], items[-1]]


class DropoutScaleModel(EdgeAttributeModel):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, x, edge_index, edge_attr, message_mask):
        return self.dropout(edge_attr[:, 0]) * self.scale


class MixedProjectionTests(unittest.TestCase):
    def test_reference_uses_entire_mixed_batch_and_edge_mean(self):
        batch = collate_counterfactual_items(mixed_items(), include_clean_base=True)
        self.assertEqual(batch.clean_base.scenario_id, batch.base.scenario_id)
        self.assertEqual(batch.clean_base.num_graphs, 2)
        self.assertEqual(batch.clean_counterparts.num_graphs, 1)
        self.assertFalse(bool(batch.clean_base.poisoned_edge_mask.any()))
        model = EdgeAttributeModel()
        gradients, value = mixed_clean_reference_gradient(
            model, batch, list(model.parameters()), device=torch.device("cpu")
        )
        clean = batch.clean_base
        logits = clean.edge_attr[:, 0] * model.scale
        expected = F.binary_cross_entropy_with_logits(logits, clean.edge_label)
        expected_gradient, = torch.autograd.grad(expected, [model.scale])
        torch.testing.assert_close(gradients[0], expected_gradient)
        self.assertAlmostEqual(value, float(expected.detach()))

    def test_reference_forward_restores_rng_and_original_loss_gradient(self):
        batch = collate_counterfactual_items(mixed_items(), include_clean_base=True)
        model = DropoutScaleModel().train()
        torch.manual_seed(7)
        before_rng = torch.random.get_rng_state().clone()
        expected, _ = counterfactual_loss_components(model, batch, config=BALANCED, device=torch.device("cpu"))
        expected_grad, = torch.autograd.grad(expected["total"], [model.scale])
        after_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(before_rng)
        mixed_clean_reference_gradient(model, batch, [model.scale], device=torch.device("cpu"))
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        observed, _ = counterfactual_loss_components(model, batch, config=BALANCED, device=torch.device("cpu"))
        observed_grad, = torch.autograd.grad(observed["total"], [model.scale])
        self.assertTrue(torch.equal(after_rng, torch.random.get_rng_state()))
        torch.testing.assert_close(expected_grad, observed_grad, rtol=0, atol=0)

    def test_clean_only_batches_match_ordinary_updates(self):
        model = DropoutScaleModel()
        projected_model = copy.deepcopy(model)
        batch = collate_counterfactual_items([
            CounterfactualPresentationItem(_graph("clean", target_logit=-2, non_target_logit=1))
        ], include_clean_base=True)
        self.assertIsNone(batch.clean_base)
        ordinary_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        projected_optimizer = torch.optim.AdamW(projected_model.parameters(), lr=0.001)
        torch.manual_seed(11)
        train_counterfactual_epoch(model, [batch, batch], ordinary_optimizer, config=BALANCED, device=torch.device("cpu"))
        torch.manual_seed(11)
        stats = train_counterfactual_epoch(
            projected_model, [batch, batch], projected_optimizer, config=BALANCED,
            device=torch.device("cpu"), projection=CleanProjectionConfig(),
        )
        torch.testing.assert_close(model.scale, projected_model.scale, rtol=0, atol=0)
        self.assertEqual(stats["gradient_projection"]["paired_steps"], 0)

    def _genconv_step(self, device):
        torch.manual_seed(13)
        model = build_model(
            model_name="genconv", node_dim=1, edge_dim=1, hidden_dim=8,
            num_layers=1, dropout=0.1, norm="layer", decoder_hidden_dim=8,
        ).to(device)
        optimizer, _ = build_optimizer(model, lr=0.001, weight_decay=0.0001)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        batch = collate_counterfactual_items(mixed_items(), include_clean_base=True)
        stats = train_counterfactual_epoch(
            model, [batch], optimizer, config=BALANCED, device=device,
            projection=CleanProjectionConfig(),
        )
        gs = stats["gradient_projection"]
        self.assertEqual(gs["paired_steps"], 1)
        self.assertEqual(gs["constraint_violations"], 0)
        self.assertLessEqual(gs["per_step"][0]["extra_ratio_after"], 1.00001)
        self.assertIsNotNone(gs["per_step"][0]["adamw_descent_clean_cosine"])
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))

    def test_synthetic_genconv_adamw_step_records_projection_and_actual_step(self):
        self._genconv_step(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_synthetic_genconv_cuda_step(self):
        self._genconv_step(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_reference_forward_restores_cuda_rng(self):
        batch = collate_counterfactual_items(mixed_items(), include_clean_base=True)
        model = DropoutScaleModel().to("cuda").train()
        torch.manual_seed(7)
        before_cpu = torch.random.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state().clone()
        mixed_clean_reference_gradient(model, batch, [model.scale], device=torch.device("cuda"))
        self.assertTrue(torch.equal(before_cpu, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state()))


def gate_fixture():
    result = {"val_pair_metrics": {"auc": 0.985}}
    audit = {
        "target_response": {ROLE: {"moving_ge_0p5": {
            "incremental_flip_rate_all_targets": 0.30,
            "logit_delta": {"mean": 1.0},
            "clean_activation_rate": 0.05,
        }}},
        "collateral_response": {ROLE: {"moving_ge_0p5": {
            "adjacent_negative": {"incremental_positive_rate": 0.005},
        }}},
    }
    return result, audit


class PilotGateTests(unittest.TestCase):
    def test_full_validation_allows_graphs_without_supervision(self):
        stats = {
            "graphs_considered": 29071, "graph_sha256_verified": 29071,
            "graphs_used": 26458, "graphs_without_supervision": 2613,
        }
        validate_full_validation_counts(stats, 29071)
        stats["graphs_used"] -= 1
        with self.assertRaises(ValueError):
            validate_full_validation_counts(stats, 29071)

    def test_exact_gate_boundaries_pass(self):
        result, audit = gate_fixture()
        gate = evaluate_gate(result, audit)
        self.assertEqual(gate["status"], "PASS")
        self.assertFalse(gate["attack_validation_started"])
        self.assertFalse(gate["test_accessed"])

    def test_auc_failure_and_nonfinite_fail_closed(self):
        for auc in (0.97823721, float("nan"), float("inf"), None):
            result, audit = gate_fixture()
            result["val_pair_metrics"]["auc"] = auc
            self.assertEqual(evaluate_gate(result, audit)["status"], "STOP")

    def test_slight_clean_activation_excess_is_not_rounded_away(self):
        result, audit = gate_fixture()
        audit["target_response"][ROLE]["moving_ge_0p5"]["clean_activation_rate"] = 0.05000001
        self.assertEqual(evaluate_gate(result, audit)["status"], "STOP")


if __name__ == "__main__":
    unittest.main()
