"""Synthetic copy-consistency wiring; no saved assets, graphs, or real fits."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import train
from spear_current.copy_consistency import CopyConsistentGenerator
from spear_current.distribution import DistributionRegularizer
from spear_current.distribution_training import ResidualTrainingConfig, fit_distribution_residual
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.support_schema import SELECTED_COORDINATES
from tests_current.fixtures_tensors import classifier, snapshot
from tests_current.test_domain_generator import setup


# Explicit independent oracle: the general Domain fixture uses a random mask
# that omits one triplet, so it is NOT a valid current-SAGE24 copy fixture.
GROUPS = (
    (350, 742, 777), (708, 743, 776), (351, 744, 779),
    (709, 745, 778), (353, 746, 781), (712, 749, 782),
)
ACTIVE = (
    776, 743, 778, 708, 709, 742, 777, 745, 744, 779, 351, 782,
    350, 729, 132, 712, 781, 353, 749, 746, 134, 724, 135, 760,
)


def fixture():
    content, _, x, context, targets = setup()
    mask = torch.tensor([coordinate in ACTIVE for coordinate in SELECTED_COORDINATES])
    x = x.clone()
    for group in GROUPS:
        x[:, list(group)] = x[:, group[0], None].expand(-1, 3)
    anchor = DomainConstrainedGenerator(content, mask, output_mode='domain_projection')
    anchor.eval().requires_grad_(False)
    normalizer = dict(mean=np.zeros(784, dtype=np.float64), scale=np.ones(784, dtype=np.float64))
    return anchor, normalizer, x, context, targets


def training_inputs(mode='pair_mean', *, epochs=2):
    with torch.random.fork_rng():
        torch.manual_seed(27)
        anchor, normalizer, x, context, _ = fixture()
        generator = (train.new_generator(anchor) if mode is None else
                     CopyConsistentGenerator(anchor, normalizer, mode=mode))
        surrogate = classifier()
        weights = x.new_ones(context.edge_index.shape[1])
        penalty = FrozenMeanCleanResponsePenalty([classifier(), classifier()], x,
                                                 context.edge_index, weights, weight=1.)
        outer = torch.tensor([2, 3, 4])
        distribution = DistributionRegularizer('unconditional', x, outer,
            generator.selected_coordinates[generator.hard_support],
            provenance={'scope': 'synthetic_train_only'})
    return dict(generator=generator, surrogate=surrogate, features=x, context=context,
        labels=torch.tensor([0, 1, 0, -1, 1, 0]), idx_train=torch.tensor([0, 1]),
        idx_attach=torch.tensor([2]), fixed_outer=outer, clean_constraint=penalty,
        config=ResidualTrainingConfig(epochs=epochs), distribution=distribution)


class CopyConsistencyTrainingTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state().clone())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(random.setstate, random.getstate())
        self.enterContext(patch('torch.load', side_effect=AssertionError('No checkpoint reads')))
        self.enterContext(patch('numpy.load', side_effect=AssertionError('No array-file reads')))

    def assert_equal_state(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), name)

    def assert_copies(self, values, coordinates=None):
        coordinates = list(range(784)) if coordinates is None else list(coordinates)
        for group in GROUPS:
            positions = [coordinates.index(c) for c in group]
            selected = values[:, positions]
            self.assertTrue(torch.equal(selected, selected[:, :1].expand_as(selected)), group)

    def assert_scope(self, before, after, targets, generator):
        allowed = torch.zeros_like(before, dtype=torch.bool)
        allowed[targets[:, None], generator.selected_coordinates[generator.hard_support]] = True
        self.assertTrue(torch.equal(after[~allowed], before[~allowed]))
        self.assertTrue(bool((after[allowed].abs() <= 10).all()))

    def test_identity_replay_and_paired_initialization_preserve_weights_and_rng(self):
        anchor, normalizer, x, context, targets = fixture()
        before = snapshot(anchor)
        clean_before = x.clone()
        rng, np_rng, py_rng = torch.random.get_rng_state().clone(), np.random.get_state(), random.getstate()
        identity = CopyConsistentGenerator(anchor, normalizer, mode='identity')
        treatment = CopyConsistentGenerator(anchor, normalizer, mode='pair_mean')
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(random.getstate(), py_rng)
        after_np = np.random.get_state()
        self.assertEqual(np_rng[0], after_np[0])
        self.assertTrue(np.array_equal(np_rng[1], after_np[1]))
        self.assertEqual(np_rng[2:], after_np[2:])
        self.assert_equal_state(snapshot(anchor), before)
        for candidate in (identity, treatment):
            self.assertEqual(candidate.output_mode, 'domain_projection')
            self.assertFalse(candidate.frozen_postprocess)
            self.assert_equal_state({k: v for k, v in candidate.state_dict().items() if k in before}, before)
            for name, parameter in candidate.named_parameters():
                source = dict(anchor.named_parameters())[name]
                self.assertTrue(torch.equal(parameter, source), name)
                self.assertNotEqual(parameter.data_ptr(), source.data_ptr(), name)
        self.assert_equal_state(dict(identity.named_parameters()), dict(treatment.named_parameters()))
        with torch.no_grad():
            expected = anchor(x, context, targets)
            actual = identity(x, context, targets)
            projected = treatment(x, context, targets)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(projected, expected))  # Nontrivial fixture.
        self.assert_copies(projected, SELECTED_COORDINATES)
        self.assertTrue(torch.equal(x, clean_before))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_identity_two_step_training_is_bitwise_equal_to_existing_domain_path(self):
        baseline, identity = training_inputs(None), training_inputs('identity')
        rng = torch.random.get_rng_state().clone()
        old = fit_distribution_residual(**baseline)
        final_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng)
        new = fit_distribution_residual(**identity)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), final_rng))
        self.assertEqual(new['history'], old['history'])
        for key in ('poisoned_features', 'poisoned_labels', 'edge_index', 'edge_weight',
                    'idx_train', 'idx_attach', 'fixed_outer'):
            self.assertTrue(torch.equal(new[key], old[key]), key)
        for key in ('residual_steps', 'surrogate_steps', 'discriminator_steps',
                    'clean_constraint_calls', 'distribution_generator_calls', 'selected_epoch'):
            self.assertEqual(new[key], old[key], key)
        old_state = snapshot(old['generator'])
        self.assert_equal_state({k: v for k, v in new['generator'].state_dict().items() if k in old_state}, old_state)
        self.assert_equal_state(snapshot(new['surrogate']), snapshot(old['surrogate']))
        self.assert_equal_state(snapshot(identity['distribution'].discriminator),
                                snapshot(baseline['distribution'].discriminator))

    def test_pair_mean_reaches_inner_discriminator_outer_kl_and_final_poison(self):
        data = training_inputs()
        generator, regularizer = data['generator'], data['distribution']
        original_x, original_y = data['features'].clone(), data['labels'].clone()
        original_surrogate = snapshot(data['surrogate'])
        frozen = {name: snapshot(getattr(generator, name)) for name in ('base_generator', 'base_shadow')}
        proxies = [snapshot(member.model) for member in data['clean_constraint'].members]
        residual_before = snapshot(generator.residual)
        injections, discriminator_inputs = [], []
        original_inject, original_inputs = generator.inject, regularizer.discriminator_inputs

        def inject(features, context, targets, weights=None, *, detach=False):
            result = original_inject(features, context, targets, weights, detach=detach)
            self.assert_copies(result.features)
            self.assert_scope(original_x, result.features, targets, generator)
            injections.append(dict(targets=targets.clone(), detach=detach,
                grad_enabled=torch.is_grad_enabled(), requires_grad=result.features.requires_grad,
                features=result.features.detach().clone()))
            return result

        def discriminator_input(features):
            real, fake = original_inputs(features)
            selected = generator.selected_coordinates[generator.hard_support].tolist()
            self.assert_copies(real[:, :24], selected)
            self.assert_copies(fake[:, :24], selected)
            self.assertTrue(bool((real[:, 24:] == 0).all()))
            self.assertTrue(bool((fake[:, 24:] == 0).all()))
            self.assertTrue(torch.equal(fake[:, :24],
                features[data['fixed_outer'][:, None], generator.selected_coordinates[generator.hard_support]] / 10))
            discriminator_inputs.append(dict(features=features.detach().clone(), grad=fake.requires_grad))
            return real, fake

        with patch.object(generator, 'inject', inject), \
                patch.object(regularizer, 'discriminator_inputs', discriminator_input):
            result = fit_distribution_residual(**data)

        self.assertEqual(len(injections), 7)  # Inner, D, outer per epoch; final attachments.
        self.assertEqual(len(discriminator_inputs), 4)
        surrogate_calls = result['surrogate'].calls
        self.assertEqual(len(surrogate_calls), 5)  # One clean check; two inner/outer pairs.
        for epoch in range(2):
            inner, d_phase, outer = injections[epoch * 3:epoch * 3 + 3]
            self.assertTrue(torch.equal(inner['targets'], data['idx_attach']))
            self.assertTrue(torch.equal(d_phase['targets'], data['fixed_outer']))
            self.assertTrue(torch.equal(outer['targets'], data['fixed_outer']))
            self.assertEqual([v['detach'] for v in (inner, d_phase, outer)], [True, True, False])
            self.assertEqual([v['grad_enabled'] for v in (inner, d_phase, outer)], [False, False, True])
            self.assertEqual([v['requires_grad'] for v in (inner, d_phase, outer)], [False, False, True])
            self.assertTrue(torch.equal(d_phase['features'], outer['features']))
            for call, injection in zip(surrogate_calls[1 + 2 * epoch:3 + 2 * epoch], (inner, outer)):
                self.assertTrue(torch.equal(call['features'], injection['features']))
            self.assertTrue(surrogate_calls[1 + 2 * epoch]['training'])
            self.assertFalse(surrogate_calls[2 + 2 * epoch]['training'])
            for record, injection in zip(discriminator_inputs[2 * epoch:2 * epoch + 2], (d_phase, outer)):
                self.assertTrue(torch.equal(record['features'], injection['features']))
            self.assertEqual([r['grad'] for r in discriminator_inputs[2 * epoch:2 * epoch + 2]], [False, True])
            for member in data['clean_constraint'].members:
                kl_calls = [call for call in member.model.calls if call['input_requires_grad']]
                self.assertEqual(len(kl_calls), 2)
                self.assertTrue(torch.equal(kl_calls[epoch]['features'], outer['features']))
                self.assertFalse(kl_calls[epoch]['parameters_require_grad'])
        self.assertTrue(torch.equal(injections[-1]['targets'], data['idx_attach']))
        self.assertTrue(torch.equal(result['poisoned_features'], injections[-1]['features']))
        self.assert_copies(result['poisoned_features'])
        self.assert_scope(original_x, result['poisoned_features'], data['idx_attach'], generator)
        self.assertTrue(torch.equal(original_x, data['features']))
        self.assertTrue(torch.equal(original_y, data['labels']))
        expected_y = original_y.clone(); expected_y[data['idx_attach']] = 1
        self.assertTrue(torch.equal(result['poisoned_labels'], expected_y))
        self.assert_equal_state(snapshot(data['surrogate']), original_surrogate)
        self.assertTrue(any(not torch.equal(v, residual_before[k]) for k, v in generator.residual.state_dict().items()))
        for name, before in frozen.items():
            self.assert_equal_state(snapshot(getattr(generator, name)), before)
        for member, before in zip(data['clean_constraint'].members, proxies):
            self.assert_equal_state(snapshot(member.model), before)
        for count in ('residual_steps', 'surrogate_steps', 'discriminator_steps',
                      'distribution_generator_calls', 'clean_constraint_calls'):
            self.assertEqual(result[count], 2)
        self.assertEqual(result['selected_epoch'], 1)

    def test_projected_poison_is_passed_unchanged_to_fresh_victim_entry(self):
        small = training_inputs(epochs=1)
        result = fit_distribution_residual(**small)
        # Only exercise the entry's handoff: its fixed 1902-row budget is built
        # from toy rows, and BOTH victim construction and fitting are mocked.
        clean = small['features'][0:1].repeat(1902, 1)
        clean[1] = small['features'][1]
        clean[1876:] = small['features'][2]
        poisoned = clean.clone()
        poisoned[1876:] = result['poisoned_features'][2]
        y = torch.zeros(1902, dtype=torch.long); y[1] = 1
        poison_y = y.clone(); poison_y[1876:] = 1
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        dev = dict(x=small['features'][:2].clone(), y=torch.tensor([0, 1]),
                   eligible=torch.ones(2, dtype=torch.bool), edge_index=empty_edges)
        inputs = dict(train=dict(x=clean, y=y, edge_index=empty_edges), dev=dev,
            indices=dict(idx_train=torch.arange(1876), idx_attach=torch.arange(1876, 1902),
                         dev_indices=torch.tensor([1902, 1903])))
        handoff = dict(poisoned_features=poisoned, poisoned_labels=poison_y)
        victim = classifier()
        captured = []

        def fake_fit(features, edges, weights, labels, fit_indices, dev_indices, **kwargs):
            captured.append(features.detach().clone())
            self.assertTrue(torch.equal(features[:1902], poisoned))
            self.assertTrue(torch.equal(features[1902:], dev['x']))
            self.assertTrue(torch.equal(labels, torch.cat((poison_y, dev['y']))))
            self.assert_copies(features)
            self.assertEqual(len(fit_indices), 1902)
            self.assertEqual(kwargs['train_iters'], 200)  # Observed argument, never executed.

        with patch('spear_current.runtime.new_victim', return_value=victim), \
                patch.object(victim, 'fit', fake_fit, create=True), \
                patch.object(train, 'AUCTrace') as trace_type, \
                patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No victim optimization')):
            trace = trace_type.return_value
            trace.observe.return_value = nullcontext()
            trace.finish.return_value = dict(selected_auc_epoch=0)
            trace.auc_state = deepcopy(victim.state_dict())
            trained, _ = train.fit_fresh_victim(SimpleNamespace(config={}), inputs, handoff,
                Path('synthetic-output-never-created'), seed=20260622)
        self.assertEqual(len(captured), 1)
        self.assertIs(trained, victim)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in trained.parameters()))
        self.assertTrue(torch.equal(handoff['poisoned_features'], poisoned))

    def test_mode_matched_no_update_preflight_keeps_optimizer_and_reference_frozen(self):
        data = training_inputs(epochs=1)
        anchor, normalizer, _, _, _ = fixture()
        reference = CopyConsistentGenerator(anchor, normalizer, mode='pair_mean', frozen=True)
        inputs = dict(train=dict(x=data['features'], y=data['labels']), context=data['context'],
                      indices=dict(idx_train=data['idx_train'], idx_attach=data['idx_attach']),
                      outer=data['fixed_outer'])
        models = [data['generator'], reference, data['surrogate'],
                  *(member.model for member in data['clean_constraint'].members)]
        before = [snapshot(model) for model in models]
        rng = torch.random.get_rng_state().clone()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No preflight updates')):
            result = train.no_update_preflight(data['generator'], reference, data['surrogate'],
                [member.model for member in data['clean_constraint'].members], inputs, 20260621)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertTrue(result['exact_projected_replay'])
        self.assertTrue(all(v > 0 for v in result['residual_gradient_l2'].values()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for model, state in zip(models, before):
            self.assert_equal_state(snapshot(model), state)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_frozen_postprocess_has_no_training_or_optimizer_path(self):
        anchor, normalizer, x, context, targets = fixture()
        frozen = CopyConsistentGenerator(anchor, normalizer, mode='pair_mean', frozen=True)
        before = snapshot(frozen)
        rng = torch.random.get_rng_state().clone()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No frozen updates')):
            with torch.no_grad():
                changed = frozen.inject(x, context, targets, detach=True)
            self.assert_copies(changed.features)
            self.assert_scope(x, changed.features, targets, frozen)
            with self.assertRaises(ValueError):
                frozen.train()
        self.assert_equal_state(snapshot(frozen), before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in frozen.parameters()))


if __name__ == '__main__':
    unittest.main()
