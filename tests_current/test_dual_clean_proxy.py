# Source: tests/test_spear_dual_clean_proxy.py; synthetic tests migrated with current-package imports.
"""Synthetic contracts for the fixed-total-weight mean clean-proxy penalty."""

from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spear_current.clean_response import FrozenCleanResponsePenalty
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty


class PointwiseProxy(nn.Module):
    def __init__(self, sign=1.0):
        super().__init__()
        self.weight = nn.Parameter(sign * torch.tensor([
            [1.0, -0.4, 0.2], [-0.3, 0.7, -0.6],
        ]))
        self.register_buffer('offset', torch.tensor([0.05, -0.05]))
        self.dropout = nn.Dropout(0.5)
        self.forward_calls = 0

    def forward(self, x, edge_index, edge_weight):
        self.forward_calls += 1
        return F.log_softmax(self.dropout(x) @ self.weight.T + self.offset, dim=1)


class MeanCleanResponsePenaltyTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[0.2, 0.4, -0.1], [0.5, -0.2, 0.3], [-0.4, 0.2, 0.6]])
        self.edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.edge_weight = torch.ones(4)
        self.idx = torch.tensor([0, 2])

    def changed(self):
        x = self.x.clone()
        x[0, 0] += 2.0
        x[2, 1] -= 1.0
        return x.requires_grad_()

    def penalty(self, models=None, weight=1.0):
        return FrozenMeanCleanResponsePenalty(
            models if models is not None else [PointwiseProxy(), PointwiseProxy(-0.6)],
            self.x, self.edge, self.edge_weight, weight,
        )

    def test_average_member_kl_not_kl_of_average_probabilities(self):
        models = [PointwiseProxy(), PointwiseProxy(-1.3)]
        penalty = self.penalty(models, weight=0.7)
        changed = self.changed()
        loss = penalty(changed, self.edge, self.edge_weight, self.idx)
        raw = [F.kl_div(model(changed, self.edge, self.edge_weight)[self.idx],
                        member.q[self.idx], reduction='batchmean')
               for model, member in zip(models, penalty.members)]
        expected = 0.7 * torch.stack(raw).mean()
        self.assertTrue(torch.equal(loss, expected))
        mixed_q = torch.stack([member.q for member in penalty.members]).mean(0)
        mixed_p = torch.stack([model(changed, self.edge, self.edge_weight).exp()
                               for model in models]).mean(0)
        mixed_kl = 0.7 * F.kl_div(mixed_p[self.idx].log(), mixed_q[self.idx], reduction='batchmean')
        self.assertGreater(float((loss - mixed_kl).detach()), 0.01)
        self.assertEqual([member.weight for member in penalty.members], [1.0, 1.0])
        self.assertEqual(penalty.calls, 1)
        row = penalty.history[0]
        self.assertEqual(row['member_raw_kl'], [float(value.detach()) for value in raw])
        self.assertEqual(row['idx_count'], 2)
        self.assertEqual(row['member_count'], 2)
        self.assertEqual(row['raw_kl'], row['mean_kl'])
        self.assertEqual(row['weighted_kl'], float(loss.detach()))
        self.assertIs(penalty.history, penalty.stats)

    def test_proxy_states_frozen_but_changed_input_has_finite_nonzero_gradient(self):
        models = [PointwiseProxy(), PointwiseProxy(-0.6)]
        for model in models:
            model.weight.grad = torch.ones_like(model.weight)
        before = [deepcopy(model.state_dict()) for model in models]
        penalty = self.penalty(models)
        changed = self.changed()
        loss = penalty(changed, self.edge, self.edge_weight, self.idx)
        loss.backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        self.assertGreater(float(changed.grad[self.idx].abs().sum()), 0)
        self.assertEqual(float(changed.grad[1].abs().sum()), 0)
        for model, state, member in zip(models, before, penalty.members):
            self.assertFalse(model.training)
            self.assertFalse(member.q.requires_grad)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            for key, value in state.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_single_member_bitwise_matches_old_loss_and_input_gradient(self):
        for weight in (0.0, 0.1, 1.0, 3.0):
            with self.subTest(weight=weight):
                old_model, new_model = PointwiseProxy(), PointwiseProxy()
                old = FrozenCleanResponsePenalty(old_model, self.x, self.edge, self.edge_weight, weight)
                new = self.penalty([new_model], weight)
                a, b = self.changed(), self.changed()
                expected = old(a, self.edge, self.edge_weight, self.idx)
                actual = new(b, self.edge, self.edge_weight, self.idx)
                expected.backward()
                actual.backward()
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(a.grad, b.grad))
                self.assertEqual(old_model.forward_calls, new_model.forward_calls)
                self.assertEqual(new.history[0]['member_raw_kl'], [old.history[0]['raw_kl']])

    def test_duplicate_independent_checkpoints_do_not_double_total_weight(self):
        old_model, first, second = PointwiseProxy(), PointwiseProxy(), PointwiseProxy()
        old = FrozenCleanResponsePenalty(old_model, self.x, self.edge, self.edge_weight, 0.6)
        new = self.penalty([first, second], 0.6)
        a, b = self.changed(), self.changed()
        expected = old(a, self.edge, self.edge_weight, self.idx)
        actual = new(b, self.edge, self.edge_weight, self.idx)
        expected.backward()
        actual.backward()
        self.assertTrue(torch.equal(expected, actual))
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_constructor_and_calls_do_not_consume_torch_or_numpy_rng(self):
        models = [PointwiseProxy(), PointwiseProxy(-0.7)]
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        penalty = self.penalty(models)
        changed = self.changed()
        penalty(changed, self.edge, self.edge_weight, self.idx).backward()
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_empty_indices_and_zero_weight_skip_proxy_forward_preserving_autograd(self):
        for weight, indices in ((1.0, torch.empty(0, dtype=torch.long)), (0.0, self.idx)):
            with self.subTest(weight=weight):
                models = [PointwiseProxy(), PointwiseProxy(-0.7)]
                penalty = self.penalty(models, weight)
                calls = [model.forward_calls for model in models]
                changed = self.changed()
                loss = penalty(changed, self.edge, self.edge_weight, indices)
                loss.backward()
                self.assertEqual(float(loss.detach()), 0)
                self.assertTrue(torch.equal(changed.grad, torch.zeros_like(changed)))
                self.assertEqual(calls, [model.forward_calls for model in models])
                self.assertEqual(penalty.history[-1]['member_raw_kl'], [0., 0.])

    def test_invalid_weights_and_model_collections_rejected(self):
        for weight in (-1, float('nan'), float('inf'), -float('inf')):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                self.penalty(weight=weight)
        for models in ([], [object()]):
            with self.assertRaises(ValueError):
                self.penalty(models)
        model = PointwiseProxy()
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.penalty([model, model])
        self.assertTrue(model.weight.requires_grad)

    def test_shared_parameter_objects_storage_views_and_buffers_rejected(self):
        for sharing in ('parameter', 'storage', 'buffer'):
            with self.subTest(sharing=sharing):
                a, b = PointwiseProxy(), PointwiseProxy()
                if sharing == 'parameter':
                    b.weight = a.weight
                elif sharing == 'storage':
                    b.weight = nn.Parameter(a.weight.detach().view_as(a.weight))
                else:
                    b.offset = a.offset
                with self.assertRaisesRegex(ValueError, 'share'):
                    self.penalty([a, b])
                self.assertTrue(a.weight.requires_grad)
                self.assertTrue(b.weight.requires_grad)

    def test_shape_finite_index_and_bound_topology_checks(self):
        penalty = self.penalty()
        bad = self.x.clone()
        bad[0, 0] = float('nan')
        for changed, edge, weight, indices in (
            (self.x[:2], self.edge, self.edge_weight, self.idx),
            (bad, self.edge, self.edge_weight, self.idx),
            (self.x, self.edge.flip(0), self.edge_weight, self.idx),
            (self.x, self.edge, self.edge_weight * 2, self.idx),
            (self.x, self.edge, None, self.idx),
            (self.x, self.edge, self.edge_weight, torch.tensor([-1])),
            (self.x, self.edge, self.edge_weight, torch.tensor([3])),
            (self.x, self.edge, self.edge_weight, torch.tensor([0.])),
            (self.x, self.edge, self.edge_weight, torch.tensor([[0]])),
        ):
            with self.assertRaises(ValueError):
                penalty(changed, edge, weight, indices)
        self.assertEqual(penalty.calls, 0)

    def test_original_features_must_be_finite_and_two_dimensional(self):
        bad = self.x.clone()
        bad[0, 0] = float('inf')
        for x in (bad, self.x[0]):
            with self.assertRaises(ValueError):
                FrozenMeanCleanResponsePenalty([PointwiseProxy()], x, self.edge, self.edge_weight)

    def test_models_must_return_normalized_finite_log_probabilities(self):
        for kind in ('unnormalized', 'nonfinite', 'wrong_count'):
            with self.subTest(kind=kind):
                model = PointwiseProxy()
                original_forward = model.forward

                def invalid(x, edge, weights):
                    output = original_forward(x, edge, weights)
                    if kind == 'unnormalized':
                        return output + 1
                    if kind == 'nonfinite':
                        return output * float('nan')
                    return output[:1]

                model.forward = invalid
                with self.assertRaises(ValueError):
                    self.penalty([model])


if __name__ == '__main__':
    unittest.main()
