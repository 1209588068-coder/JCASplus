"""Synthetic clean-objective ablations; no graph assets or experiment launches."""
import unittest

import torch

from spear_current.clean_response import FrozenCleanResponsePenalty
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.kl_ablation_penalty import FixedProxyAblationPenalty, MODES
from spear_current.role_training import _check_constraint, _state_hash
from spear_current.distribution_training import fit_distribution_residual
from tests_current.test_distribution_training import inputs
from tests_current.test_dual_clean_proxy import PointwiseProxy


class FixedProxyAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.tensor([[.2, .4, -.1], [.5, -.2, .3], [-.4, .2, .6]])
        self.edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.weights = torch.ones(4)
        self.targets = torch.tensor([0, 2])

    def changed(self):
        x = self.x.clone()
        x[0, 0] += 2.
        x[2, 1] -= 1.
        return x.requires_grad_()

    def penalty(self, mode='dual'):
        return FixedProxyAblationPenalty([PointwiseProxy(), PointwiseProxy(-.6)],
            self.x, self.edges, self.weights, mode=mode)

    def test_dual_bitwise_matches_original_loss_and_gradient(self):
        old = FrozenMeanCleanResponsePenalty([PointwiseProxy(), PointwiseProxy(-.6)],
            self.x, self.edges, self.weights, weight=1.)
        new = self.penalty()
        a, b = self.changed(), self.changed()
        expected = old(a, self.edges, self.weights, self.targets)
        actual = new(b, self.edges, self.weights, self.targets)
        expected.backward(); actual.backward()
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(a.grad, b.grad))
        for key, value in old.history[-1].items():
            self.assertEqual(value, new.history[-1][key])

    def test_single_matches_first_proxy_loss_and_gradient(self):
        old = FrozenCleanResponsePenalty(PointwiseProxy(), self.x, self.edges, self.weights, 1.)
        new = self.penalty('single_proxy1')
        a, b = self.changed(), self.changed()
        expected = old(a, self.edges, self.weights, self.targets)
        actual = new(b, self.edges, self.weights, self.targets)
        expected.backward(); actual.backward()
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(a.grad, b.grad))
        self.assertEqual(new.history[-1]['coefficients'], [1., 0.])

    def test_disabled_records_nonzero_raw_kl_but_has_exact_zero_gradient(self):
        penalty = self.penalty('disabled')
        x = self.changed()
        loss = penalty(x, self.edges, self.weights, self.targets)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
        self.assertGreater(penalty.history[-1]['raw_kl'], 0.)
        self.assertTrue(all(value > 0 for value in penalty.history[-1]['member_raw_kl']))
        self.assertEqual(penalty.history[-1]['weighted_kl'], 0.)
        self.assertEqual(penalty.history[-1]['effective_kl'], 0.)

    def test_both_proxies_called_once_and_frozen_in_every_mode(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                penalty = self.penalty(mode)
                before = [_state_hash(m.model) for m in penalty.members]
                calls = [m.model.forward_calls for m in penalty.members]
                rng = torch.get_rng_state().clone()
                x = self.changed()
                penalty(x, self.edges, self.weights, self.targets).backward()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(before, [_state_hash(m.model) for m in penalty.members])
                self.assertEqual([n + 1 for n in calls], [m.model.forward_calls for m in penalty.members])
                self.assertEqual([m.calls for m in penalty.members], [1, 1])
                self.assertEqual(penalty.calls, 1)
                for member in penalty.members:
                    self.assertTrue(all(not p.requires_grad and p.grad is None
                                        for p in member.model.parameters()))
                    self.assertTrue(all(not child.training for child in member.model.modules()))
                penalty.assert_contract()

    def test_empty_targets_keep_autograd_and_record_zero_without_proxy_forwards(self):
        for mode in MODES:
            penalty = self.penalty(mode)
            counts = [m.model.forward_calls for m in penalty.members]
            x = self.changed()
            loss = penalty(x, self.edges, self.weights, torch.empty(0, dtype=torch.long))
            loss.backward()
            self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
            self.assertEqual(counts, [m.model.forward_calls for m in penalty.members])
            self.assertEqual(penalty.history[-1]['member_raw_kl'], [0., 0.])

    def test_invalid_modes_and_proxy_count_rejected_before_freezing(self):
        for mode in ('', None, [], 'single_proxy2', 'none', 0):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.penalty(mode)
        for count in (0, 1, 3):
            models = [PointwiseProxy() for _ in range(count)]
            with self.assertRaises(ValueError):
                FixedProxyAblationPenalty(models, self.x, self.edges, self.weights)
            self.assertTrue(all(p.requires_grad for m in models for p in m.parameters()))

    def test_mode_coefficients_members_and_weights_cannot_change_silently(self):
        mutations = (
            lambda p: setattr(p, '_mode', 'disabled'),
            lambda p: setattr(p, '_coefficients', (1., 0.)),
            lambda p: setattr(p, 'weight', 0.),
            lambda p: setattr(p.members[0], 'weight', 0.),
            lambda p: setattr(p, 'members', p.members[::-1]),
            lambda p: p.members[0].model.train(),
            lambda p: setattr(p, 'stats', []),
        )
        for mutate in mutations:
            penalty = self.penalty()
            mutate(penalty)
            with self.assertRaises(ValueError):
                penalty(self.changed(), self.edges, self.weights, self.targets)
            self.assertEqual(penalty.calls, 0)
        penalty = self.penalty()
        for name, value in (('mode', 'disabled'), ('coefficients', (0., 0.))):
            with self.assertRaises(AttributeError):
                setattr(penalty, name, value)

    def test_existing_topology_input_guards_remain(self):
        for bad_x, bad_edge, bad_weight, bad_idx in (
            (self.x[:2], self.edges, self.weights, self.targets),
            (self.x, self.edges.flip(0), self.weights, self.targets),
            (self.x, self.edges, self.weights * 2, self.targets),
            (self.x, self.edges, self.weights, torch.tensor([-1])),
        ):
            penalty = self.penalty()
            with self.assertRaises(ValueError):
                penalty(bad_x, bad_edge, bad_weight, bad_idx)

    def test_all_modes_pass_original_constraint_check_and_short_training(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                data = inputs('unconditional', epochs=2)
                old = data['clean_constraint']
                new = FixedProxyAblationPenalty([m.model for m in old.members], data['features'],
                    data['context'].edge_index, data['features'].new_ones(data['context'].edge_index.shape[1]),
                    mode=mode)
                data['clean_constraint'] = new
                _check_constraint(new, data['features'], data['context'], new.members[0].edge_weights)
                result = fit_distribution_residual(**data)
                self.assertEqual(result['residual_steps'], 2)
                self.assertEqual(result['surrogate_steps'], 2)
                self.assertEqual(result['discriminator_steps'], 2)
                self.assertEqual(result['clean_constraint_calls'], 2)
                self.assertEqual([m.calls for m in new.members], [2, 2])
                new.assert_contract()

    def test_dual_short_training_exact_replay_including_rng_and_states(self):
        original = inputs('unconditional', epochs=2)
        # A second deterministic construction preserves discriminator ownership;
        # that object deliberately rejects a generic deepcopy of its guards.
        changed = inputs('unconditional', epochs=2)
        old = changed['clean_constraint']
        changed['clean_constraint'] = FixedProxyAblationPenalty(
            [m.model for m in old.members], changed['features'], changed['context'].edge_index,
            old.members[0].edge_weights, mode='dual')
        initial_rng = torch.get_rng_state().clone()
        expected = fit_distribution_residual(**original)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        actual = fit_distribution_residual(**changed)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for key in expected:
            if key in ('generator', 'surrogate'):
                self.assertEqual(_state_hash(expected[key]), _state_hash(actual[key]))
            elif isinstance(expected[key], torch.Tensor):
                self.assertTrue(torch.equal(expected[key], actual[key]), key)
            else:
                self.assertEqual(expected[key], actual[key], key)
        self.assertEqual(_state_hash(original['distribution'].discriminator),
                         _state_hash(changed['distribution'].discriminator))


if __name__ == '__main__':
    unittest.main()
