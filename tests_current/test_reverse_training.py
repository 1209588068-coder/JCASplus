"""Synthetic reverse-training checks; no graph/model assets or real fits."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from spear_current.distribution import DistributionRegularizer
from spear_current.distribution_training import fit_distribution_residual
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.reverse_training import ReverseTrainingConfig, fit_reverse_distribution_residual
from spear_current.role_training import ResidualTrainingConfig, _state_hash
from tests_current.fixtures_tensors import classifier, setup


def inputs(*, epochs=2, inner_steps=1, arm='unconditional', target_weight=1.):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        anchor, support, features, context, _ = setup()
        generator = DomainConstrainedGenerator(anchor, support, output_mode='domain_projection')
        surrogate = classifier()
        outer = torch.tensor([2, 4, 5])
        weights = torch.ones(context.edge_index.shape[1])
        constraint = FrozenMeanCleanResponsePenalty([classifier(), classifier()], features,
                                                    context.edge_index, weights, weight=1.)
        distribution = DistributionRegularizer(arm, features, outer,
            generator.selected_coordinates[generator.hard_support], provenance={'scope': 'synthetic_train'})
    return dict(generator=generator, surrogate=surrogate, features=features, context=context,
        labels=torch.tensor([0, 1, 1, -1, 1, 1]),
        eligible=torch.tensor([True, True, True, False, True, True]),
        idx_train=torch.tensor([0, 1]), idx_attach=torch.tensor([2]), fixed_outer=outer,
        clean_constraint=constraint,
        config=ReverseTrainingConfig(epochs=epochs, inner_steps=inner_steps, target_weight=target_weight),
        distribution=distribution)


class ReverseTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_independent_reverse_config_preserves_all_other_forward_defaults(self):
        reverse, forward = asdict(ReverseTrainingConfig()), asdict(ResidualTrainingConfig())
        self.assertEqual(reverse.pop('target_class'), 0)
        self.assertEqual(forward.pop('target_class'), 1)
        self.assertEqual(reverse, forward)
        with self.assertRaises(FrozenInstanceError):
            ReverseTrainingConfig().target_class = 1
        for kwargs in (dict(target_class=1), dict(target_class=-1), dict(target_class=False),
                       dict(epochs=0), dict(inner_steps=True), dict(residual_lr=0),
                       dict(shadow_lr=-1), dict(target_weight=0), dict(similarity_weight=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReverseTrainingConfig(**kwargs)

    def test_two_steps_preserve_inputs_frozen_models_and_only_poison_attachment_rows(self):
        data = inputs(epochs=2, inner_steps=2)
        original = {key: data[key].clone() for key in
                    ('features', 'labels', 'eligible', 'idx_train', 'idx_attach', 'fixed_outer')}
        old_surrogate = _state_hash(data['surrogate'])
        old_generator = _state_hash(data['generator'])
        base_hashes = {name: _state_hash(getattr(data['generator'], name))
                       for name in ('base_generator', 'base_shadow')}
        proxy_hashes = [_state_hash(member.model) for member in data['clean_constraint'].members]
        permutation = data['context'].score_permutation.clone()
        progress = []
        result = fit_reverse_distribution_residual(**data, progress=progress.append)
        for key, value in original.items():
            self.assertTrue(torch.equal(data[key], value), key)
        expected_labels = original['labels'].clone()
        expected_labels[original['idx_attach']] = 0
        self.assertTrue(torch.equal(result['poisoned_labels'], expected_labels))
        self.assertEqual(result['poisoned_labels'][3].item(), -1)
        self.assertEqual(result['poisoned_labels'][[4, 5]].tolist(), [1, 1])
        allowed = torch.zeros_like(original['features'], dtype=torch.bool)
        coordinates = data['generator'].selected_coordinates[data['generator'].hard_support]
        allowed[data['idx_attach'][:, None], coordinates] = True
        self.assertTrue(torch.equal(result['poisoned_features'][~allowed], original['features'][~allowed]))
        self.assertTrue(bool((result['poisoned_features'][allowed].abs() <= 10).all()))
        self.assertFalse(torch.equal(result['poisoned_features'][allowed], original['features'][allowed]))
        self.assertTrue(torch.equal(result['edge_index'], data['context'].edge_index))
        self.assertTrue(bool((result['edge_weight'] == 1).all()))
        self.assertTrue(torch.equal(data['context'].score_permutation, permutation))
        self.assertEqual(_state_hash(data['surrogate']), old_surrogate)
        self.assertIsNot(result['surrogate'], data['surrogate'])
        self.assertNotEqual(_state_hash(result['surrogate']), old_surrogate)
        self.assertNotEqual(_state_hash(data['generator']), old_generator)
        for name, digest in base_hashes.items():
            model = getattr(data['generator'], name)
            self.assertEqual(_state_hash(model), digest)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertEqual([_state_hash(member.model) for member in data['clean_constraint'].members], proxy_hashes)
        self.assertTrue(result['original_inputs_unchanged'])
        self.assertTrue(result['frozen_models_unchanged'])
        self.assertTrue(result['caller_surrogate_unchanged'])
        self.assertEqual((result['surrogate_steps'], result['residual_steps'], result['discriminator_steps']), (4, 2, 2))
        self.assertEqual((result['distribution_generator_calls'], result['clean_constraint_calls']), (2, 2))
        self.assertEqual((result['selection_policy'], result['selected_epoch']), ('final_epoch', 1))
        self.assertEqual(result['config']['target_class'], 0)
        self.assertEqual(result['distribution_arm'], 'unconditional')
        self.assertEqual(len(progress), 2)
        progress[0]['outer_nll'] = -1
        self.assertNotEqual(result['history'][0]['outer_nll'], -1)
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in data['distribution'].discriminator.parameters()))

    def test_inner_and_outer_nll_target_zero_with_original_supervised_labels(self):
        data = inputs(epochs=2, target_weight=1.3)
        result = fit_reverse_distribution_residual(**data)
        calls = result['surrogate'].calls
        inner_calls = [row for row in calls if row['training']]
        outer_calls = [row for row in calls if row['input_requires_grad']]
        self.assertEqual((len(inner_calls), len(outer_calls)), (2, 2))
        inner_indices = torch.cat([data['idx_train'], data['idx_attach']])
        outer_indices = torch.cat([data['idx_train'], data['fixed_outer']])
        inner_labels, outer_labels = data['labels'].clone(), data['labels'].clone()
        inner_labels[data['idx_attach']] = 0
        outer_labels[data['fixed_outer']] = 0
        for record, inner, outer in zip(result['history'], inner_calls, outer_calls):
            inner_nll = float(F.nll_loss(inner['output'][inner_indices], inner_labels[inner_indices]))
            outer_nll = float(F.nll_loss(outer['output'][outer_indices], outer_labels[outer_indices]))
            self.assertAlmostEqual(record['inner_nll'][0], inner_nll, places=7)
            self.assertAlmostEqual(record['outer_nll'], outer_nll, places=7)
            self.assertAlmostEqual(record['target_loss'], 1.3 * outer_nll, places=6)
            self.assertAlmostEqual(record['outer_loss'], record['target_loss'] + record['similarity_loss']
                + record['clean_kl_loss'] + record['distribution_weighted_g_loss'], places=6)
            self.assertAlmostEqual(record['distribution_weighted_g_loss'], .1 * record['distribution_raw_g_loss'], places=6)
            self.assertFalse(outer['parameters_require_grad'])
            self.assertFalse(outer['training'])

    def test_class_swapped_forward_problem_matches_the_reverse_numerical_updates(self):
        reverse_data, forward_data = inputs(epochs=2), inputs(epochs=2)
        forward_data.pop('eligible')
        forward_data['labels'] = torch.where(forward_data['labels'] >= 0,
            1 - forward_data['labels'], forward_data['labels'])
        forward_data['config'] = ResidualTrainingConfig(epochs=2)
        proxies = [member.model for member in forward_data['clean_constraint'].members]
        with torch.no_grad():
            for model in [forward_data['surrogate'], *proxies]:
                model.linear.weight.copy_(model.linear.weight.flip(0))
                model.linear.bias.copy_(model.linear.bias.flip(0))
        forward_data['clean_constraint'] = FrozenMeanCleanResponsePenalty(proxies,
            forward_data['features'], forward_data['context'].edge_index,
            torch.ones(forward_data['context'].edge_index.shape[1]), weight=1.)
        starting_rng = torch.random.get_rng_state().clone()
        reverse = fit_reverse_distribution_residual(**reverse_data)
        reverse_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(starting_rng)
        forward = fit_distribution_residual(**forward_data)
        self.assertTrue(torch.equal(reverse_rng, torch.random.get_rng_state()))
        self.assertEqual(set(reverse), set(forward))
        for left, right in zip(reverse['history'], forward['history']):
            for key in left:
                if key == 'inner_nll':
                    for a, b in zip(left[key], right[key]):
                        self.assertAlmostEqual(a, b, places=6)
                else:
                    self.assertAlmostEqual(left[key], right[key], places=6, msg=key)
        torch.testing.assert_close(reverse['poisoned_features'], forward['poisoned_features'], atol=1e-6, rtol=1e-6)
        expected_labels = torch.where(forward['poisoned_labels'] >= 0,
            1 - forward['poisoned_labels'], forward['poisoned_labels'])
        self.assertTrue(torch.equal(reverse['poisoned_labels'], expected_labels))

    def test_every_attachment_and_outer_row_requires_eligible_original_positive(self):
        for row in (2, 4, 5):
            for kind in ('negative', 'unknown', 'ineligible'):
                with self.subTest(row=row, kind=kind):
                    data = inputs(epochs=1)
                    if kind == 'ineligible':
                        data['eligible'][row] = False
                    else:
                        data['labels'][row] = 0 if kind == 'negative' else -1
                    with patch('torch.optim.Adam', side_effect=AssertionError('optimizer created')):
                        with self.assertRaisesRegex(ValueError, 'eligible original true-positive'):
                            fit_reverse_distribution_residual(**data)

    def test_eligibility_shape_dtype_and_supervision_are_checked(self):
        for value in (torch.ones(6), torch.ones(5, dtype=torch.bool),
                      torch.ones((6, 1), dtype=torch.bool), [True] * 6):
            data = inputs(epochs=1)
            data['eligible'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Eligibility'):
                fit_reverse_distribution_residual(**data)
        data = inputs(epochs=1)
        data['eligible'][0] = False
        with self.assertRaisesRegex(ValueError, 'eligible computable binary'):
            fit_reverse_distribution_residual(**data)

    def test_overlaps_duplicate_indices_and_attachment_prefix_are_rejected(self):
        for key, value, message in (
            ('idx_train', torch.tensor([0, 4]), 'disjoint'),
            ('idx_attach', torch.tensor([2, 2]), 'unique'),
            ('idx_attach', torch.tensor([4]), 'attachment order'),
            ('idx_train', torch.tensor([6]), 'in-range')):
            data = inputs(epochs=1)
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                fit_reverse_distribution_residual(**data)

    def test_only_fresh_unconditional_distribution_and_projected_generator_allowed(self):
        for arm in ('conditional', 'continuation'):
            data = inputs(epochs=1, arm=arm)
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, 'unconditional'):
                fit_reverse_distribution_residual(**data)
        data = inputs(epochs=1)
        data['distribution'].generator_loss(data['features'])
        with self.assertRaisesRegex(ValueError, 'zero update/loss counts'):
            fit_reverse_distribution_residual(**data)
        data = inputs(epochs=1)
        anchor, support, *_ = setup()
        data['generator'] = DomainConstrainedGenerator(anchor, support, output_mode='identity')
        with self.assertRaisesRegex(ValueError, 'projected DomainConstrainedGenerator'):
            fit_reverse_distribution_residual(**data)

    def test_forward_configuration_and_negative_attachment_guards_remain_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'target_class=1'):
            ResidualTrainingConfig(target_class=0)
        data = inputs(epochs=1)
        data['config'] = ResidualTrainingConfig(epochs=1)
        with self.assertRaisesRegex(ValueError, 'ReverseTrainingConfig'):
            fit_reverse_distribution_residual(**data)
        data.pop('eligible')
        with self.assertRaisesRegex(ValueError, 'original true-negative'):
            fit_distribution_residual(**data)

    def test_progress_cannot_mutate_eligibility_or_frozen_score_permutation(self):
        for kind in ('eligible', 'score_permutation'):
            data = inputs(epochs=2)
            def progress(record):
                if kind == 'eligible':
                    data['eligible'][2] = False
                else:
                    data['context'].score_permutation.add_(0)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Immutable'):
                fit_reverse_distribution_residual(**data, progress=progress)


if __name__ == '__main__':
    unittest.main()
