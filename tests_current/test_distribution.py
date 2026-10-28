# Source: tests/test_spear_distribution_training_module.py; synthetic tests migrated with current-package imports.
"""Synthetic distribution-loss contracts; no AV2 data or model assets loaded."""
from copy import deepcopy
import random
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from spear_current.distribution import (
    ARMS, CONDITION_COORDINATES, DISTRIBUTION_WEIGHT, DistributionRegularizer,
)
from spear_current.support_schema import LEGAL_COORDINATES


def fixture():
    x = torch.linspace(-2, 2, 7 * 784).reshape(7, 784)
    targets = torch.tensor([5, 1, 4])
    coordinates = torch.tensor(LEGAL_COORDINATES[:24])
    return x, targets, coordinates


def inject(x, targets, coordinates):
    values = (x[targets[:, None], coordinates] + 0.25).detach().requires_grad_(True)
    injected = x.clone()
    injected[targets[:, None], coordinates] = values
    return injected, values


def state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_state_equal(test, left, right):
    test.assertEqual(set(left), set(right))
    for key in left:
        test.assertTrue(torch.equal(left[key], right[key]), key)


class DistributionTrainingModuleTests(unittest.TestCase):
    def setUp(self):
        self.x, self.targets, self.coordinates = fixture()
        self.provenance = dict(original_split='train', role='train', population='synthetic_outer')

    def regularizer(self, arm='conditional', **kwargs):
        return DistributionRegularizer(arm, self.x, self.targets, self.coordinates,
                                       provenance=self.provenance, **kwargs)

    def test_same_initial_parameters_architecture_and_rng_for_all_arms(self):
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        models = [self.regularizer(arm) for arm in ARMS]
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_state[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_state[1], numpy_after[1]))
        self.assertEqual(numpy_state[2:], numpy_after[2:])
        for model in models:
            self.assertEqual(model.discriminator_hash(), models[0].discriminator_hash())
            self.assertEqual([type(m).__name__ for m in model.discriminator],
                             ['Linear', 'LeakyReLU', 'Linear', 'LeakyReLU', 'Linear'])
            self.assertEqual([(m.in_features, m.out_features) for m in model.discriminator if isinstance(m, torch.nn.Linear)],
                             [(30, 64), (64, 64), (64, 1)])
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.discriminator.parameters()))
            model.assert_contract()
        self.assertIsNone(models[0].optimizer)

    def test_inputs_use_only_selected24_and_owned_original_conditions(self):
        original = self.x.clone()
        d = self.regularizer()
        injected, values = inject(self.x, self.targets, self.coordinates)
        real_input, fake_input = d.discriminator_inputs(injected)
        self.assertTrue(torch.equal(real_input[:, :24], self.x[self.targets[:, None], self.coordinates] / 10))
        self.assertTrue(torch.equal(fake_input[:, :24], values / 10))
        expected_condition = self.x[self.targets[:, None], torch.tensor(CONDITION_COORDINATES)] / 10
        self.assertTrue(torch.equal(real_input[:, 24:], expected_condition))
        self.assertTrue(torch.equal(fake_input[:, 24:], expected_condition))
        self.assertFalse(real_input.requires_grad)
        self.assertTrue(fake_input.requires_grad)
        self.assertTrue(torch.equal(self.x, original))
        changed = injected.clone()
        changed[0, 0] += 7  # Outside observed rows and selected coordinates.
        self.assertTrue(torch.equal(d.discriminator_inputs(changed)[1], fake_input))
        # The cached reference owns storage; a later caller mutation cannot rewrite it.
        self.x[self.targets[0], self.coordinates[0]] += 1
        self.assertTrue(torch.equal(d.discriminator_inputs(injected)[0], real_input))
        with self.assertRaises(ValueError):
            d.validate_binding(self.x, self.targets, self.coordinates)

    def test_conditional_zero_matches_unconditional_without_parameter_count_change(self):
        self.x[:, torch.tensor(CONDITION_COORDINATES)] = 0
        conditional, unconditional = self.regularizer(), self.regularizer('unconditional')
        injected, _ = inject(self.x, self.targets, self.coordinates)
        for a, b in zip(conditional.discriminator_inputs(injected), unconditional.discriminator_inputs(injected)):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(conditional.discriminator_step(injected), unconditional.discriminator_step(injected))
        assert_state_equal(self, state(conditional.discriminator), state(unconditional.discriminator))
        self.assertTrue(torch.equal(conditional.generator_loss(injected), unconditional.generator_loss(injected)))

    def test_unconditional_zeros_both_condition_blocks_but_retains_real_reference(self):
        d = self.regularizer('unconditional')
        injected, _ = inject(self.x, self.targets, self.coordinates)
        real, fake = d.discriminator_inputs(injected)
        self.assertTrue(bool((real[:, 24:] == 0).all()))
        self.assertTrue(bool((fake[:, 24:] == 0).all()))
        self.assertTrue(torch.equal(real[:, :24], self.x[self.targets[:, None], self.coordinates] / 10))
        self.assertEqual(len(d.snapshot()['condition_provenance']['fields']), 6)

    def test_balanced_discriminator_step_detaches_generator_and_updates_only_discriminator(self):
        d = self.regularizer()
        injected, values = inject(self.x, self.targets, self.coordinates)
        before_features, before_d = injected.detach().clone(), state(d.discriminator)
        real, fake = d.discriminator_inputs(injected)
        with torch.no_grad():
            real_logits, fake_logits = d.discriminator(real), d.discriminator(fake)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            expected = float((real_loss + fake_loss) / 2)
        rng = torch.random.get_rng_state().clone()
        metrics = d.discriminator_step(injected)
        self.assertAlmostEqual(metrics['d_loss'], expected)
        self.assertAlmostEqual(metrics['d_loss'], (metrics['d_real_loss'] + metrics['d_fake_loss']) / 2)
        self.assertEqual(metrics['d_steps'], 1)
        self.assertIsNone(values.grad)
        self.assertTrue(torch.equal(before_features, injected.detach()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(any(not torch.equal(before_d[k], v) for k, v in d.discriminator.state_dict().items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in d.discriminator.parameters()))
        d.assert_contract()

    def test_generator_loss_is_weighted_nonsaturating_with_only_selected_input_gradients(self):
        d = self.regularizer()
        injected = self.x.clone().requires_grad_(True)
        before = state(d.discriminator)
        _, fake = d.discriminator_inputs(injected)
        with torch.no_grad():
            logits = d.discriminator(fake)
            expected = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss = d.generator_loss(injected)
        self.assertAlmostEqual(float(loss.detach()), float(expected) * DISTRIBUTION_WEIGHT, places=7)
        self.assertAlmostEqual(d.last_generator_raw_loss, float(expected), places=7)
        loss.backward()
        allowed = torch.zeros_like(self.x, dtype=torch.bool)
        allowed[self.targets[:, None], self.coordinates] = True
        self.assertGreater(float(injected.grad[allowed].abs().sum()), 0)
        self.assertTrue(bool((injected.grad[~allowed] == 0).all()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in d.discriminator.parameters()))
        assert_state_equal(self, state(d.discriminator), before)
        self.assertEqual(d.generator_calls, 1)
        self.assertEqual(d.steps, 0)

    def test_continuation_has_no_distribution_optimization_but_keeps_zero_loss_graph(self):
        d = self.regularizer('continuation')
        before = state(d.discriminator)
        injected, values = inject(self.x, self.targets, self.coordinates)
        self.assertEqual(d.discriminator_step(injected)['d_loss'], 0)
        loss = d.generator_loss(injected)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(bool((values.grad == 0).all()))
        self.assertEqual(d.steps, 0)
        self.assertEqual(d.generator_calls, 1)
        self.assertIsNone(d.optimizer)
        assert_state_equal(self, state(d.discriminator), before)

    def test_no_update_smoke_is_finite_nonzero_and_preserves_all_state_counters_rng_gradients(self):
        d = self.regularizer()
        injected, values = inject(self.x, self.targets, self.coordinates)
        before = d.snapshot()
        rng = torch.random.get_rng_state().clone()
        result = d.no_update_smoke(injected)
        self.assertGreater(result['d_gradient_parameters'], 0)
        self.assertGreater(result['generator_input_gradient_l1'], 0)
        self.assertGreater(result['generator_raw_loss'], 0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertIsNone(values.grad)
        self.assertEqual(d.steps, 0)
        self.assertEqual(d.generator_calls, 0)
        self.assertEqual(d.snapshot()['optimizer'], before['optimizer'])
        assert_state_equal(self, state(d.discriminator), before['discriminator'])
        self.assertTrue(all(p.grad is None for p in d.discriminator.parameters()))

    def test_binding_frozen_conditions_support_and_input_validation(self):
        d = self.regularizer()
        d.validate_binding(self.x, self.targets, self.coordinates)
        for kind in ('condition', 'nan', 'inf', 'range', 'shape', 'dtype'):
            injected = self.x.clone()
            if kind == 'condition': injected[self.targets[0], CONDITION_COORDINATES[0]] += 1
            elif kind == 'nan': injected[0, 0] = float('nan')
            elif kind == 'inf': injected[0, 0] = float('inf')
            elif kind == 'range': injected[self.targets[0], self.coordinates[0]] = 10.01
            elif kind == 'shape': injected = injected[:-1]
            else: injected = injected.double()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                d.generator_loss(injected)
        self.assertEqual(d.generator_calls, 0)
        with self.assertRaises(ValueError): d.validate_binding(self.x, self.targets.flip(0), self.coordinates)
        with self.assertRaises(ValueError): d.validate_binding(self.x, self.targets, self.coordinates.flip(0))
        for change in ('reference', 'condition', 'parameter_grad', 'training', 'lr', 'activation'):
            fresh = self.regularizer()
            if change == 'reference': fresh._real.data.add_(1)
            elif change == 'condition': fresh._clean_conditions.add_(1)
            elif change == 'parameter_grad': next(fresh.discriminator.parameters()).grad = torch.zeros_like(next(fresh.discriminator.parameters()))
            elif change == 'training': fresh.discriminator.train()
            elif change == 'lr': fresh.optimizer.param_groups[0]['lr'] *= 2
            else: fresh.discriminator[1].negative_slope = 0.3
            with self.subTest(change=change), self.assertRaises(ValueError): fresh.assert_contract()

    def test_constructor_rejects_invalid_contract_without_mutating_inputs(self):
        for kwargs in ({'lr': 0.001}, {'weight': 0.2}, {'seed': 1}, {'weight_decay': 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.regularizer(**kwargs)
        for coordinates in (self.coordinates[:23], self.coordinates.float(), self.coordinates.repeat(2),
                            torch.tensor([CONDITION_COORDINATES[0], *self.coordinates[1:].tolist()])):
            with self.assertRaises(ValueError):
                DistributionRegularizer('conditional', self.x, self.targets, coordinates)
        for targets in (self.targets[:0], self.targets.float(), torch.tensor([1, 1]), torch.tensor([7])):
            with self.assertRaises(ValueError):
                DistributionRegularizer('conditional', self.x, targets, self.coordinates)
        with self.assertRaises(ValueError):
            DistributionRegularizer('conditional', self.x.clone().requires_grad_(True), self.targets, self.coordinates)

    def test_snapshot_has_condition_and_real_provenance_and_roundtrips_owned_state(self):
        d = self.regularizer()
        injected, _ = inject(self.x, self.targets, self.coordinates)
        d.discriminator_step(injected)
        d.generator_loss(injected)
        saved = d.snapshot()
        self.assertEqual(saved['provenance'], self.provenance)
        self.assertEqual(saved['condition_provenance']['coordinates'], list(CONDITION_COORDINATES))
        self.assertFalse(saved['condition_provenance']['contains_labels'])
        self.assertFalse(saved['condition_provenance']['contains_future'])
        self.assertFalse(saved['condition_provenance']['contains_ids'])
        self.assertEqual(saved['real_reference_provenance']['source'], 'same_original_clean_outer_rows')
        other = self.regularizer()
        rng = torch.random.get_rng_state().clone()
        other.load_snapshot(saved)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(other.discriminator_hash(), d.discriminator_hash())
        self.assertEqual((other.steps, other.generator_calls), (1, 1))
        self.assertEqual(other.discriminator_step(injected), d.discriminator_step(injected))
        assert_state_equal(self, state(other.discriminator), state(d.discriminator))
        key = next(iter(saved['discriminator']))
        saved['discriminator'][key].zero_()
        self.assertFalse(torch.equal(saved['discriminator'][key], d.discriminator.state_dict()[key]))
        for change in ('arm', 'binding', 'tensor', 'hash', 'optimizer_shape'):
            corrupt = d.snapshot()
            if change == 'arm': corrupt['config']['arm'] = 'unconditional'
            elif change == 'binding': corrupt['reference']['target_count'] += 1
            elif change == 'tensor': corrupt['discriminator'][key].fill_(float('nan'))
            elif change == 'hash': corrupt['discriminator_sha256'] = '0' * 64
            else:
                first = next(iter(corrupt['optimizer']['state'].values()))
                first['exp_avg'] = torch.zeros(1)
            before = state(d.discriminator)
            with self.subTest(change=change), self.assertRaises(ValueError): d.load_snapshot(corrupt)
            assert_state_equal(self, before, state(d.discriminator))


if __name__ == '__main__':
    unittest.main()
