"""Small CPU-only contract tests for paper-guided CGBA; no dataset access."""
from dataclasses import replace
import json
import unittest

import numpy as np
import torch

from comparison_methods.cgba import (
    CGBAConfig, CGBAFixedFeatureTrigger, DEFAULT_CONFIG, concealment_losses,
    fit_cgba, graphlime_coefficients, select_attachment_indices,
    validate_config,
)


def fixture():
    rng = torch.Generator().manual_seed(91)
    x = torch.randn(10, 6, generator=rng)
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, -1, -1])
    x[:4, 0] += 3
    x[4:8, 0] -= 3
    a = torch.arange(10)
    b = (a + 1) % 10
    edges = torch.stack((torch.cat((a, b)), torch.cat((b, a))))
    return x, edges, y, torch.arange(8), torch.arange(4)


def config(**overrides):
    return replace(CGBAConfig(epochs=3, select_pretrain_epochs=3, hidden=8,
                              poison_count=2, feature_count=2, dropout=0), **overrides)


class CGBATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_public_config_is_complete_json_safe_and_defaults_target_one(self):
        default = validate_config()
        self.assertEqual(default, DEFAULT_CONFIG)
        self.assertEqual(default['target'], 1)
        self.assertEqual(default['epochs'], 200)
        self.assertEqual(default, json.loads(json.dumps(default, allow_nan=False)))
        custom = {'feature_count': 2, 'allowed_feature_indices': (2, 5)}
        expanded = validate_config(custom)
        self.assertEqual(expanded['allowed_feature_indices'], [2, 5])
        self.assertEqual(expanded, validate_config(expanded))
        expanded['allowed_feature_indices'].append(8)
        self.assertEqual(custom['allowed_feature_indices'], (2, 5))
        self.assertEqual(validate_config(config())['hidden'], 8)
        for bad in ({'target': True}, {'target': 1.0}, {'target': 3},
                    {'feature_count': 2, 'allowed_feature_indices': [1, 1]},
                    {'feature_count': 2, 'allowed_feature_indices': [1]},
                    {'feature_count': 1, 'allowed_feature_indices': [-1]},
                    {'epochs': 0}, {'lr': float('nan')}, {'lr': '0.01'},
                    {'unexpected': 1}, [1], 'config'):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_config(bad)

    def test_selector_matches_equation_and_stable_ties(self):
        p = torch.tensor([.1, .1, .2, .8])
        d = torch.tensor([20., 0., 0., 0.])
        selected, scores = select_attachment_indices(p, d, torch.tensor([3, 2, 1, 0]), 2, .01)
        self.assertEqual(selected.tolist(), [1, 2])
        torch.testing.assert_close(scores, -(p + .01 * d))
        selected, _ = select_attachment_indices(p, d * 0, torch.tensor([1, 0]), 2, .01)
        self.assertEqual(selected.tolist(), [0, 1])

    def test_graphlime_is_nonlinear_hsic_not_gradient(self):
        rng = np.random.default_rng(18)
        signal = np.linspace(-2, 2, 32)
        x = np.c_[signal, rng.normal(size=32), np.ones(32)]
        prob = 1 / (1 + np.exp(-4 * signal))
        coefficients = graphlime_coefficients(x, np.c_[1 - prob, prob], config(graphlime_rho=.01))
        self.assertEqual(int(coefficients.argmax()), 0)
        self.assertGreater(coefficients[0], 0)
        self.assertEqual(coefficients[2], 0)
        self.assertTrue(np.isfinite(coefficients).all())
        np.testing.assert_array_equal(graphlime_coefficients(x[:1], np.c_[1-prob, prob][:1], config()), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'memory limit'):
            graphlime_coefficients(x, np.c_[1-prob, prob], config(graphlime_max_entries=1))

    def test_fit_preserves_topology_labels_and_support(self):
        inputs = fixture()
        snapshots = [a.clone() for a in inputs]
        callbacks = []
        out = fit_cgba(*inputs, seed=123, config=config(), progress=callbacks.append)
        for got, old in zip(inputs, snapshots):
            self.assertTrue(torch.equal(got, old))
        self.assertTrue(torch.equal(out['poisoned_y'], inputs[2]))
        self.assertTrue(torch.equal(out['poisoned_edge_index'], inputs[1]))
        self.assertTrue(torch.equal(out['fit_indices'], inputs[3]))
        self.assertEqual(out['attachment_indices'].numel(), 2)
        self.assertTrue(bool((inputs[2][out['attachment_indices']] == 1).all()))
        generator = out['generator']
        mask = torch.zeros_like(inputs[0], dtype=torch.bool)
        mask[out['attachment_indices'][:, None], generator.feature_indices[None]] = True
        self.assertTrue(torch.equal(out['poisoned_x'][~mask], inputs[0][~mask]))
        self.assertTrue(bool(torch.isfinite(out['poisoned_x']).all()))
        self.assertTrue(all(not p.requires_grad for p in generator.parameters()))
        record = out['record']
        self.assertFalse(record['author_code'])
        self.assertFalse(record['exact_author_reproduction'])
        self.assertFalse(record['evaluation_accessed'])
        self.assertEqual(len(record['history']), 3)
        self.assertEqual(len(record['pretrain_loss']), 3)
        self.assertEqual(record['feature_domain'], 'all_input_coordinates')
        self.assertEqual(len(record['aif']), 6)
        self.assertEqual({v['stage'] for v in callbacks},
                         {'cgba_surrogate', 'cgba_graphlime', 'cgba_trigger'})
        for row in record['history']:
            self.assertAlmostEqual(row['total'], row['classification_sum'] + row['self_sum'] + row['neighbor_sum'], places=5)
        triggered, edges, weights = generator.inject(inputs[0], inputs[1], torch.tensor([4, 5]))
        self.assertTrue(torch.equal(edges, inputs[1]))
        self.assertTrue(torch.equal(weights, torch.ones(len(weights))))
        self.assertTrue(torch.equal(triggered[4, generator.feature_indices], triggered[5, generator.feature_indices]))
        untouched, _, _ = generator.inject(inputs[0], inputs[1], torch.empty(0, dtype=torch.long))
        self.assertTrue(torch.equal(untouched, inputs[0]))

    def test_reproducibility_and_rng_isolation(self):
        torch.manual_seed(444)
        state = torch.get_rng_state().clone()
        a = fit_cgba(*fixture(), seed=21, config=config())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        b = fit_cgba(*fixture(), seed=21, config=config())
        self.assertEqual({k: v for k, v in a['record'].items() if k != 'elapsed_seconds'},
                         {k: v for k, v in b['record'].items() if k != 'elapsed_seconds'})
        for key in a['generator'].state_dict():
            self.assertTrue(torch.equal(a['generator'].state_dict()[key], b['generator'].state_dict()[key]))

    def test_unlabeled_ground_truth_not_used(self):
        args = list(fixture())
        a = fit_cgba(*args, seed=21, config=config())
        args[2] = args[2].clone()
        args[2][8:] = torch.tensor([1, 0])
        b = fit_cgba(*args, seed=21, config=config())
        self.assertTrue(torch.equal(a['poisoned_x'], b['poisoned_x']))
        self.assertTrue(torch.equal(a['attachment_indices'], b['attachment_indices']))
        self.assertEqual(a['record']['aif'], b['record']['aif'])

    def test_explicit_restricted_feature_domain(self):
        out = fit_cgba(*fixture(), seed=17,
                       config=config(allowed_feature_indices=(2, 4)))
        self.assertEqual(set(out['generator'].feature_indices.tolist()), {2, 4})
        self.assertEqual(out['record']['feature_domain'], 'explicit_restricted_domain')
        self.assertEqual(out['record']['config']['allowed_feature_indices'], (2, 4))

    def test_reject_dirty_label_candidates_and_budget(self):
        args = list(fixture())
        args[4] = torch.tensor([0, 4])
        with self.assertRaisesRegex(ValueError, 'no relabeling'):
            fit_cgba(*args, seed=1, config=config())
        args[4] = torch.tensor([0, 8])
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            fit_cgba(*args, seed=1, config=config())
        with self.assertRaisesRegex(ValueError, 'Not enough'):
            fit_cgba(*fixture(), seed=1, config=config(poison_count=5))
        with self.assertRaisesRegex(ValueError, 'eligible domain'):
            fit_cgba(*fixture(), seed=1, config=config(feature_count=7))
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            fit_cgba(*fixture(), seed=1, config=config(epochs=0))
        with self.assertRaisesRegex(ValueError, 'Unknown CGBA'):
            fit_cgba(*fixture(), seed=1, config={'surprise': 1})
        self.assertEqual(DEFAULT_CONFIG['feature_count'], 24)

    def test_concealment_is_literal_sum_without_hinge(self):
        x = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        selected = torch.tensor([0])
        neighbors = torch.tensor([[0, 0], [1, 2]])
        own, neighbor = concealment_losses(x, x, selected, neighbors, config())
        self.assertAlmostEqual(float(own), -.5)
        self.assertAlmostEqual(float(neighbor), 0)
        generated = x.clone().requires_grad_()
        own, neighbor = concealment_losses(generated, x, selected,
                                           torch.empty((2, 0), dtype=torch.long), config())
        (own + neighbor).backward()
        self.assertTrue(bool(torch.isfinite(generated.grad).all()))
        self.assertEqual(float(neighbor.detach()), 0)

    def test_unfitted_generator_rejected(self):
        generator = CGBAFixedFeatureTrigger(6, 8, torch.tensor([0, 1]), device='cpu')
        with self.assertRaisesRegex(RuntimeError, 'not fitted'):
            generator.inject(fixture()[0], fixture()[1], torch.tensor([4]))


if __name__ == '__main__':
    unittest.main()
