"""Disconnected-scene numerical acceptance; no real datasets are read."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from spear_current.av2_formal_generator import (fit_generator, fit_support,
    outer_loss_terms, sample_gate_uniform, gates_from_uniform)
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.full_ablation_training import _freeze, _state_hash
from spear_current.models import GCN, GraphSage
from spear_current.womd_pilot_selection import CandidateHardConcrete


class StreamingGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.rng = torch.get_rng_state(); torch.manual_seed(19)
        self.x = torch.randn(12, 784) * .1
        self.y = torch.tensor([0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
        self.cuts = (0, 4, 9, 12)
        edges = []
        for start, stop in zip(self.cuts[:-1], self.cuts[1:]):
            source = torch.arange(start, stop - 1)
            edge = torch.stack((source, source + 1)); edges.extend((edge, edge.flip(0)))
        self.edge = torch.cat(edges, 1)
        self.indices = dict(idx_train=torch.tensor([0, 1, 5, 7, 9, 10]),
            idx_attach=torch.tensor([2, 6]), fixed_outer=torch.tensor([2, 6, 8]))
        self.coordinates = torch.tensor([357, 715] + list(range(28)))
        self.proxies = [GraphSage(784, 12, 2, dropout=0., device='cpu') for _ in range(2)]
        for model in self.proxies:
            _freeze(model)

    def tearDown(self):
        torch.set_rng_state(self.rng); torch.set_num_threads(self.threads)

    def batches(self, data_dir, role, max_nodes, device):
        self.assertEqual(role, 'train')
        cuts = (0, 12) if max_nodes >= 12 else self.cuts
        for start, stop in zip(cuts[:-1], cuts[1:]):
            mask = (self.edge[0] >= start) & (self.edge[0] < stop)
            yield dict(x=self.x[start:stop].to(device), y=self.y[start:stop].to(device),
                eligible=torch.ones(stop - start, dtype=torch.bool, device=device),
                edge_index=(self.edge[:, mask] - start).to(device), global_start=start, global_stop=stop)

    def test_outer_loss_and_input_gradient_match_whole_graph(self):
        surrogate = GCN(784, 12, 2, dropout=0., device='cpu'); _freeze(surrogate)
        outer = self.indices['fixed_outer']; clean = self.indices['idx_train']
        supervised = torch.cat((clean, outer))
        changed = self.x.clone(); changed[outer, :28] += .2
        changed.requires_grad_()
        labels = self.y.clone(); labels[outer] = 1
        penalty = FrozenMeanCleanResponsePenalty(self.proxies, self.x, self.edge,
                                                  torch.ones(self.edge.shape[1]), weight=1.)
        expected = (F.nll_loss(surrogate(changed, self.edge)[supervised], labels[supervised])
            + .1 * (1. - F.cosine_similarity(changed, self.x, dim=1).mean())
            + penalty(changed, self.edge, torch.ones(self.edge.shape[1]), outer))
        gradient = torch.autograd.grad(expected, changed)[0]
        actual = 0.; partial_gradient = torch.zeros_like(self.x)
        for start, stop in zip(self.cuts[:-1], self.cuts[1:]):
            mask = (self.edge[0] >= start) & (self.edge[0] < stop)
            local_outer = outer[(outer >= start) & (outer < stop)] - start
            local_sup = supervised[(supervised >= start) & (supervised < stop)] - start
            modified = changed[start:stop].detach().clone().requires_grad_()
            terms = outer_loss_terms(surrogate, self.proxies, self.x[start:stop], modified,
                self.edge[:, mask] - start, labels[start:stop], local_sup, local_outer,
                global_supervised=len(supervised), global_nodes=12, global_outer=len(outer))
            loss = terms[0] + .1 * terms[1] + terms[2]
            actual += float(loss.detach())
            partial_gradient[start:stop] = torch.autograd.grad(loss, modified)[0]
        self.assertAlmostEqual(actual, float(expected.detach()), places=6)
        torch.testing.assert_close(partial_gradient, gradient, rtol=3e-5, atol=2e-7)

    def test_shared_uniform_matches_original_gate_rng_and_gradient(self):
        original = CandidateHardConcrete(self.coordinates, seed=31)
        split = CandidateHardConcrete(self.coordinates, seed=31)
        for _ in range(4):
            expected = original()
            uniform = sample_gate_uniform(split)
            actual = gates_from_uniform(split, uniform)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(split.rng_state, original.rng_state, rtol=0, atol=0)
            original.zero_grad(); split.zero_grad()
            expected.square().sum().backward()
            (gates_from_uniform(split, uniform).square().sum() / 3).backward()
            (gates_from_uniform(split, uniform).square().sum() * 2 / 3).backward()
            torch.testing.assert_close(split.log_alpha.grad, original.log_alpha.grad)

    def test_complete_pipeline_updates_focal_protection_and_partition_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'summary.json').write_text(json.dumps(dict(train=dict(pair_nodes=12))))
            with patch('spear_current.av2_formal_generator.batches', self.batches):
                whole = fit_generator(root, self.indices, self.proxies, self.coordinates,
                    seed=17, output_dir=root / 'whole', base_epochs=2, residual_epochs=2,
                    max_nodes=12, device='cpu', allow_cpu_smoke=True)
                split = fit_generator(root, self.indices, self.proxies, self.coordinates,
                    seed=17, output_dir=root / 'split', base_epochs=2, residual_epochs=2,
                    max_nodes=5, device='cpu', allow_cpu_smoke=True)
                for name, expected in whole['generator'].state_dict().items():
                    torch.testing.assert_close(split['generator'].state_dict()[name], expected,
                                               rtol=1e-3, atol=3e-5, msg=name)
                for result in (whole, split):
                    self.assertEqual([r['generator_steps'] for r in result['summary']['stages']], [2, 2])
                    self.assertEqual([r['surrogate_steps'] for r in result['summary']['stages']], [2, 2])
                    self.assertFalse(bool(result['generator'].hard_support[:2].any()))
                support, summary = fit_support(split['generator'], split['surrogate'], root,
                    self.indices, self.proxies, self.coordinates, seed=23, output_dir=root / 'support',
                    epochs=2, max_nodes=5, device='cpu', allow_cpu_smoke=True)
                self.assertEqual(int(support.sum()), 24); self.assertFalse(bool(support[:2].any()))
                self.assertEqual(summary['selector_updates'], 2)
                self.assertEqual(summary['surrogate_updates'], 2)
                full_support, _ = fit_support(split['generator'], split['surrogate'], root,
                    self.indices, self.proxies, self.coordinates, seed=23, output_dir=root / 'full_support',
                    epochs=2, max_nodes=12, device='cpu', allow_cpu_smoke=True)
                torch.testing.assert_close(support, full_support, rtol=0, atol=0)
                partial = torch.load(root / 'support/selector.pt', weights_only=True)
                full = torch.load(root / 'full_support/selector.pt', weights_only=True)
                torch.testing.assert_close(partial['log_alpha'], full['log_alpha'], rtol=1e-5, atol=1e-6)
                split['generator'].set_support(support)
                original_hash = _state_hash(split['generator'])
                changed = split['generator'].inject(self.x, SimpleNamespace(edge_index=self.edge),
                                                    self.indices['idx_attach']).features
                allowed = torch.zeros_like(self.x, dtype=torch.bool)
                allowed[self.indices['idx_attach'][:, None], self.coordinates[support]] = True
                self.assertFalse(bool((changed[~allowed] != self.x[~allowed]).any()))
                self.assertEqual(_state_hash(split['generator']), original_hash)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for real-device regression')
    def test_cuda_generator_to_semantic_support(self):
        device = torch.device('cuda:0')
        proxies = [deepcopy(p).to(device) for p in self.proxies]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'summary.json').write_text(json.dumps(dict(train=dict(pair_nodes=12))))
            with patch('spear_current.av2_formal_generator.batches', self.batches):
                result = fit_generator(root, self.indices, proxies, self.coordinates,
                    seed=17, output_dir=root / 'anchor', base_epochs=2, residual_epochs=2,
                    max_nodes=5, device=device)
                support, summary = fit_support(result['generator'], result['surrogate'], root,
                    self.indices, proxies, self.coordinates, seed=23, output_dir=root / 'support',
                    epochs=2, max_nodes=5, device=device)
                self.assertEqual(int(support.sum()), 24)
                self.assertEqual(summary['selector_updates'], 2)
                self.assertTrue(all(p.grad is None for model in proxies for p in model.parameters()))
                self.assertTrue(all(p.grad is None for p in result['generator'].parameters()))
                torch.cuda.synchronize(device)

    def test_real_api_refuses_cpu_without_explicit_synthetic_flag(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            fit_generator('unused', self.indices, self.proxies, self.coordinates,
                          seed=17, output_dir='unused', device='cpu')


if __name__ == '__main__':
    unittest.main()
