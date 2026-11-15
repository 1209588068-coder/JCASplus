"""Independent synthetic support-write/gradient checks, with no optimizer steps."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from spear_current.compact_ablation_training import CompactAblationGenerator
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.full_ablation_evaluation import _inject
from spear_current.full_ablation_training import _freeze, _state_hash
from spear_current.models import GCN, GraphTrojanNet
from spear_current.support_generator import apply_gates
from spear_current.womd_pilot_selection import CandidateHardConcrete, candidate_groups, exact_support


class TinyProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(784, 2)

    def forward(self, x, edge, weights=None):
        return F.log_softmax(self.head(x), dim=1)


class WOMDPilotInjectionReviewTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.rng = torch.get_rng_state()
        torch.manual_seed(17)
        self.coordinates = torch.tensor(
            [350, 742, 777, 351, 744, 779, 353, 746, 781, 354, 748, 783,
             357, 715] + list(range(16)), dtype=torch.long)
        self.legal = (self.coordinates != 357) & (self.coordinates != 715)
        base = GraphTrojanNet(torch.device('cpu'), 80, 30)
        with torch.no_grad():
            for value in base.parameters():
                value.zero_()
        shadow = GCN(784, 80, 2, dropout=0., device=torch.device('cpu'))
        self.generator = CompactAblationGenerator(base, shadow, self.coordinates,
                                                  self.legal, variant='full')
        self.generator.set_phase('support_surrogate')
        _freeze(self.generator)
        self.x = torch.linspace(1., 2., 4 * 784).reshape(4, 784)
        self.edge = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        self.weights = torch.ones(self.edge.shape[1])
        self.targets = torch.tensor([1, 3])

    def tearDown(self):
        torch.set_rng_state(self.rng)
        torch.set_num_threads(self.threads)

    def test_final_support_writes_exactly_24_coordinates_only_on_targets(self):
        groups, _ = candidate_groups(self.coordinates)
        support = exact_support(self.coordinates, groups, torch.arange(len(groups)), 24)
        self.generator.set_support(support)
        before = _state_hash(self.generator)
        original = self.x.clone()
        changed, edges, weights = _inject(self.generator, self.x, self.edge,
                                          self.targets, self.weights, 24)
        writable = torch.zeros_like(self.x, dtype=torch.bool)
        writable[self.targets[:, None], self.coordinates[support]] = True
        torch.testing.assert_close(changed != self.x, writable, rtol=0, atol=0)
        self.assertEqual(writable.sum().item(), 48)
        self.assertFalse(bool(writable[:, [357, 715]].any()))
        torch.testing.assert_close(edges, self.edge, rtol=0, atol=0)
        torch.testing.assert_close(weights, self.weights, rtol=0, atol=0)
        torch.testing.assert_close(self.x, original, rtol=0, atol=0)
        self.assertEqual(_state_hash(self.generator), before)
        # Returning selector support alone is insufficient: the caller must
        # explicitly install it; evaluation enforces the final coordinate count.
        self.generator.set_support(self.legal)
        with self.assertRaisesRegex(ValueError, 'support capacity'):
            _inject(self.generator, self.x, self.edge, self.targets, self.weights, 24)

    def test_gate_loss_reaches_selector_but_not_frozen_anchor_or_clean_proxies(self):
        selector = CandidateHardConcrete(self.coordinates, seed=17)
        proxy_a, proxy_b, surrogate = TinyProxy(), TinyProxy(), TinyProxy()
        penalty = FrozenMeanCleanResponsePenalty([proxy_a, proxy_b], self.x,
                                                 self.edge, self.weights, weight=1.)
        _freeze(surrogate)
        context = SimpleNamespace(edge_index=self.edge)
        anchor_before = _state_hash(self.generator)
        proxy_before = [_state_hash(model) for model in (proxy_a, proxy_b)]
        with torch.no_grad():
            generated = self.generator(self.x, context, self.targets).detach()
        original = self.x[self.targets[:, None], self.coordinates]
        changed = self.x.clone()
        changed[self.targets[:, None], self.coordinates] = apply_gates(original, generated, selector())
        target = F.nll_loss(surrogate(changed, self.edge)[self.targets], torch.ones(2, dtype=torch.long))
        cosine = 1 - F.cosine_similarity(changed, self.x, dim=1).mean()
        loss = target + .1 * cosine + penalty(changed, self.edge, self.weights, self.targets)
        loss.backward()
        self.assertIsNotNone(selector.log_alpha.grad)
        self.assertTrue(bool(torch.isfinite(selector.log_alpha.grad).all()))
        self.assertGreater(selector.log_alpha.grad.abs().sum().item(), 0.)
        for model in (self.generator, proxy_a, proxy_b, surrogate):
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(_state_hash(self.generator), anchor_before)
        self.assertEqual([_state_hash(model) for model in (proxy_a, proxy_b)], proxy_before)


if __name__ == '__main__':
    unittest.main()
