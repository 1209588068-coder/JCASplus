# Source: tests/test_spear_distribution_training_training.py; synthetic tests migrated with current-package imports.
"""Synthetic three-arm training checks; no graph/model assets or real fits."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from tests_current.reference_content_training import fit_content_residual
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.distribution import DistributionRegularizer
from spear_current.distribution_training import (
    ARMS, ResidualTrainingConfig, fit_distribution_residual,
)
from tests_current.fixtures_tensors import setup, classifier


ADDITIONAL_HISTORY = {'d_loss', 'd_real_loss', 'd_fake_loss', 'd_steps',
                      'distribution_raw_g_loss', 'distribution_weighted_g_loss'}
ADDITIONAL_RESULT = {'distribution_arm', 'discriminator_steps', 'distribution_generator_calls',
                     'distribution_initial_state_sha256', 'distribution_final_state_sha256'}


def inputs(arm='continuation', *, epochs=2, inner_steps=1):
    with torch.random.fork_rng():
        torch.manual_seed(27)
        anchor, mask, x, context, _ = setup()
        generator = DomainConstrainedGenerator(anchor, mask, output_mode='domain_projection')
        surrogate = classifier()
        weights = x.new_ones(context.edge_index.shape[1])
        penalty = FrozenMeanCleanResponsePenalty([classifier(), classifier()], x,
                                                 context.edge_index, weights, weight=1.)
        outer = torch.tensor([2, 3, 4])
        distribution = DistributionRegularizer(arm, x, target_indices=outer,
            selected_coordinates=generator.selected_coordinates[generator.hard_support],
            lr=.0002, seed=20261621, provenance={'scope': 'synthetic_train'})
    return dict(generator=generator, surrogate=surrogate, features=x, context=context,
        labels=torch.tensor([0, 1, 0, -1, 1, 0]), idx_train=torch.tensor([0, 1]),
        idx_attach=torch.tensor([2]), fixed_outer=outer, clean_constraint=penalty,
        config=ResidualTrainingConfig(epochs=epochs, inner_steps=inner_steps,
                                      shadow_lr=.0002, residual_lr=.0002),
        distribution=distribution)


class DistributionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_continuation_control_exactly_replays_old_loop_and_rng(self):
        data = inputs(inner_steps=2)
        old_data = deepcopy({k: v for k, v in data.items() if k != 'distribution'})
        initial_rng = torch.random.get_rng_state().clone()
        with patch.object(data['distribution'], 'discriminator_step',
                          side_effect=AssertionError('continuation discriminator update')):
            result = fit_distribution_residual(**data)
        final_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(initial_rng)
        previous = fit_content_residual(**old_data)
        self.assertTrue(torch.equal(final_rng, torch.random.get_rng_state()))
        self.assertEqual(set(result) - set(previous), ADDITIONAL_RESULT)
        for key in previous:
            if key in ('generator', 'surrogate', 'history'):
                continue
            if isinstance(previous[key], torch.Tensor):
                self.assertTrue(torch.equal(result[key], previous[key]), key)
            else:
                self.assertEqual(result[key], previous[key], key)
        stripped = [{k: v for k, v in row.items() if k not in ADDITIONAL_HISTORY}
                    for row in result['history']]
        self.assertEqual(stripped, previous['history'])
        self.assertEqual(result['discriminator_steps'], 0)
        self.assertEqual(result['distribution_generator_calls'], 2)
        self.assertEqual(result['distribution_initial_state_sha256'],
                         result['distribution_final_state_sha256'])
        for row in result['history']:
            self.assertTrue(all(row[key] == 0 for key in ADDITIONAL_HISTORY))

    def test_both_adversarial_arms_update_once_per_epoch_in_registered_order(self):
        for arm in ('unconditional', 'conditional'):
            data = inputs(arm, inner_steps=2)
            regularizer = data['distribution']
            events, d_inputs, g_inputs = [], [], []
            original_step = torch.optim.Adam.step
            original_d, original_g = regularizer.discriminator_step, regularizer.generator_loss
            residual_parameters = {id(p) for p in data['generator'].residual.parameters()}

            def step(optimizer, *args, **kwargs):
                if optimizer is regularizer.optimizer:
                    events.append('d')
                elif id(optimizer.param_groups[0]['params'][0]) in residual_parameters:
                    events.append('residual')
                else:
                    events.append('surrogate')
                return original_step(optimizer, *args, **kwargs)

            def d_step(x):
                d_inputs.append(x.detach().clone())
                self.assertFalse(x.requires_grad)
                self.assertTrue(all(p.grad is None for p in data['generator'].parameters()))
                return original_d(x)

            def g_loss(x):
                events.append('g_loss')
                g_inputs.append(x.detach().clone())
                self.assertTrue(x.requires_grad)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in regularizer.discriminator.parameters()))
                self.assertTrue(all(not m.training for m in regularizer.discriminator.modules()))
                return original_g(x)

            with self.subTest(arm=arm), patch.object(torch.optim.Adam, 'step', step), \
                    patch.object(regularizer, 'discriminator_step', d_step), \
                    patch.object(regularizer, 'generator_loss', g_loss):
                result = fit_distribution_residual(**data)
            self.assertEqual(events, ['surrogate', 'surrogate', 'd', 'g_loss', 'residual'] * 2)
            self.assertEqual(result['surrogate_steps'], 4)
            self.assertEqual(result['residual_steps'], 2)
            self.assertEqual(result['discriminator_steps'], 2)
            self.assertEqual(result['distribution_generator_calls'], 2)
            self.assertEqual(result['clean_constraint_calls'], 2)
            self.assertNotEqual(result['distribution_initial_state_sha256'],
                                result['distribution_final_state_sha256'])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(d_inputs, g_inputs)))
            for row in result['history']:
                self.assertAlmostEqual(row['distribution_weighted_g_loss'],
                                       .1 * row['distribution_raw_g_loss'], places=6)
                self.assertGreater(row['d_loss'], 0)
                self.assertAlmostEqual(row['outer_loss'], row['target_loss'] + row['similarity_loss']
                    + row['clean_kl_loss'] + row['distribution_weighted_g_loss'], places=6)
            self.assertTrue(all(p.grad is None and not p.requires_grad
                                for p in regularizer.discriminator.parameters()))

    def test_all_arms_preserve_target_support_domain_topology_and_context_labels(self):
        for arm in ARMS:
            data = inputs(arm)
            x = data['features'].clone()
            y = data['labels'].clone()
            permutation = data['context'].score_permutation.clone()
            support = data['generator'].hard_support.clone()
            result = fit_distribution_residual(**data)
            allowed = torch.zeros_like(x, dtype=torch.bool)
            coordinates = data['generator'].selected_coordinates[support]
            allowed[data['idx_attach'][:, None], coordinates] = True
            with self.subTest(arm=arm):
                self.assertTrue(torch.equal(result['poisoned_features'][~allowed], x[~allowed]))
                self.assertTrue(bool((result['poisoned_features'][allowed].abs() <= 10).all()))
                self.assertTrue(torch.equal(data['features'], x))
                self.assertTrue(torch.equal(data['labels'], y))
                self.assertTrue(torch.equal(data['context'].score_permutation, permutation))
                self.assertTrue(torch.equal(data['generator'].hard_support, support))
                self.assertTrue(torch.equal(result['edge_index'], data['context'].edge_index))
                self.assertTrue(bool((result['edge_weight'] == 1).all()))
                self.assertEqual(result['poisoned_labels'][3].item(), -1)
                expected_labels = y.clone(); expected_labels[data['idx_attach']] = 1
                self.assertTrue(torch.equal(result['poisoned_labels'], expected_labels))
                self.assertTrue(result['frozen_models_unchanged'])
                self.assertTrue(result['caller_surrogate_unchanged'])

    def test_wrong_output_mode_is_rejected_before_optimizer_creation(self):
        data = inputs()
        anchor, support, *_ = setup()
        data['generator'] = DomainConstrainedGenerator(anchor, support, output_mode='identity')
        with patch('torch.optim.Adam', side_effect=AssertionError('optimizer created')):
            with self.assertRaisesRegex(ValueError, 'projected DomainConstrainedGenerator'):
                fit_distribution_residual(**data)

    def test_binding_mismatch_is_rejected_before_optimizer_creation(self):
        for kind in ('targets', 'features'):
            data = inputs('conditional')
            if kind == 'targets':
                data['fixed_outer'] = torch.tensor([2, 4, 3])
            else:
                data['features'] = data['features'].clone()
                data['features'][2, data['generator'].selected_coordinates[0]] += .01
            with self.subTest(kind=kind), patch('torch.optim.Adam',
                    side_effect=AssertionError('optimizer created')):
                with self.assertRaises(ValueError):
                    fit_distribution_residual(**data)

    def test_used_regularizer_is_rejected_before_any_optimizer_update(self):
        data = inputs('conditional')
        data['distribution'].generator_loss(data['features'])
        with patch.object(torch.optim.Adam, 'step') as step:
            with self.assertRaisesRegex(ValueError, 'zero update/loss counts'):
                fit_distribution_residual(**data)
            step.assert_not_called()

    def test_discriminator_cannot_consume_global_rng(self):
        data = inputs('unconditional', epochs=1)
        original = data['distribution'].discriminator_step

        def d_step(x):
            record = original(x)
            torch.rand(1)
            return record

        with torch.random.fork_rng(), patch.object(data['distribution'], 'discriminator_step', d_step):
            with self.assertRaisesRegex(ValueError, 'shared training RNG'):
                fit_distribution_residual(**data)

    def test_discriminator_cannot_change_generator_parameters(self):
        data = inputs('conditional', epochs=1)
        original = data['distribution'].discriminator_step

        def d_step(x):
            record = original(x)
            data['generator'].residual.head.bias.data.add_(1.)
            return record

        with patch.object(data['distribution'], 'discriminator_step', d_step):
            with self.assertRaisesRegex(ValueError, 'changed generator or surrogate parameters'):
                fit_distribution_residual(**data)

    def test_nonfinite_generator_distribution_loss_stops_before_residual_update(self):
        data = inputs('conditional', epochs=1)
        original = data['distribution'].generator_loss

        def g_loss(x):
            return original(x) * float('nan')

        with patch.object(data['distribution'], 'generator_loss', g_loss):
            with self.assertRaisesRegex(ValueError, 'Nonfinite or malformed distribution'):
                fit_distribution_residual(**data)
        self.assertEqual(data['distribution'].steps, 1)
        self.assertTrue(all(p.grad is None for p in data['generator'].parameters()))

    def test_progress_cannot_change_support_or_run_extra_distribution_calls(self):
        for kind in ('support', 'calls'):
            data = inputs('conditional', epochs=1)

            def progress(record):
                if kind == 'support':
                    mask = data['generator'].hard_support
                    selected, closed = torch.nonzero(mask).flatten()[0], torch.nonzero(~mask).flatten()[0]
                    mask.data[selected] = False; mask.data[closed] = True
                else:
                    data['distribution'].generator_loss(data['features'])

            with self.subTest(kind=kind), self.assertRaises(ValueError):
                fit_distribution_residual(**data, progress=progress)

    def test_progress_receives_detached_copy_of_loss_records(self):
        data = inputs('conditional', epochs=1)

        def progress(record):
            record['d_loss'] = -999
            record['inner_nll'].clear()

        result = fit_distribution_residual(**data, progress=progress)
        self.assertGreater(result['history'][0]['d_loss'], 0)
        self.assertEqual(len(result['history'][0]['inner_nll']), 1)


if __name__ == '__main__':
    unittest.main()
