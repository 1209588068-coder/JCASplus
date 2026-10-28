# Source: tests/test_spear_domain_training_module.py; synthetic tests migrated with current-package imports.
"""Synthetic generator/gradient contracts; never load graph/model assets."""
from copy import deepcopy
import io
import unittest

import torch

from spear_current.attention_context import attention_context
from spear_current.attention_generator import ContentAttentionGenerator
from tests_current.reference_content_training import ResidualTrainingConfig, fit_content_residual
from spear_current.domain_generator import DomainConstrainedGenerator, OUTPUT_MODES
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.role_training import _finite_module, _state_hash
from spear_current.support_generator import SupportMaskedGenerator, fixed_support
from spear_current.support_schema import SELECTED_COORDINATES
from tests_current.fixtures_tensors import fixture, make_model, activate_branch, snapshot
from tests_current.fixtures_tensors import ToyClassifier


def setup():
    with torch.random.fork_rng():
        torch.manual_seed(20260621)
        _, roles, x = fixture(784)
        x = x / 200  # Synthetic inputs remain strictly inside the clean domain.
        context = attention_context(roles, ['synthetic-scene'], seed=17)
        anchor = make_model(input_dim=784, selected=torch.tensor(SELECTED_COORDINATES), mode='untyped')
        activate_branch(anchor)
        anchor = ContentAttentionGenerator(anchor, attention_mode='uniform', score_dim=4)
        anchor.eval().requires_grad_(False)
    mask = fixed_support('random_semantic', 24, seed=20260621)
    return anchor, mask, x, context, torch.tensor([4, 1, 5])


def classifier():
    model = ToyClassifier()
    model.linear = torch.nn.Linear(784, 2)
    return model


class DomainTrainingModuleTests(unittest.TestCase):
    def setUp(self):
        self.anchor, self.mask, self.x, self.context, self.targets = setup()

    def model(self, mode='identity'):
        return DomainConstrainedGenerator(self.anchor, self.mask, output_mode=mode)

    def test_constructor_preserves_all_completed_weights_and_rng(self):
        before = snapshot(self.anchor)
        rng = torch.random.get_rng_state().clone()
        for mode in OUTPUT_MODES:
            model = self.model(mode)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            self.assertIsInstance(model, ContentAttentionGenerator)
            state = model.state_dict()
            self.assertEqual(set(state) - set(before), {'hard_support', 'output_mode_code'})
            for key, value in before.items():
                self.assertTrue(torch.equal(state[key], value), key)
            self.assertGreater(float(model.residual.head.weight.detach().abs().sum()), 0.)
            self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
            _finite_module(model, 'synthetic domain generator')
            self.assertEqual(len(_state_hash(model)), 64)
        for key, value in self.anchor.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_identity_matches_frozen_support_wrapper_bitwise(self):
        reference = SupportMaskedGenerator(self.anchor, hard_support=self.mask).eval()
        model = self.model()
        expected = reference(self.x, self.context, self.targets)
        for mode in (False, True, False):
            model.train(mode)
            actual = model(self.x, self.context, self.targets)
            self.assertTrue(torch.equal(actual, expected))
            raw = ContentAttentionGenerator.forward(model, self.x, self.context, self.targets)
            clean = self.x[self.targets[:, None], model.selected_coordinates]
            self.assertTrue(torch.equal(actual, model.apply_output(clean, raw)))

    def test_owned_copies_preserve_caller_modes_gradients_and_storage(self):
        self.anchor.train()
        for parameter in self.anchor.residual.parameters():
            parameter.grad = torch.full_like(parameter, 7.)
        before = snapshot(self.anchor)
        model = self.model()
        for name in ('base_generator', 'base_shadow', 'residual'):
            old, new = getattr(self.anchor, name), getattr(model, name)
            self.assertIsNot(old, new)
            for a, b in zip(old.parameters(), new.parameters()):
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
        model(self.x, self.context, self.targets).sum().backward()
        self.assertTrue(self.anchor.training)
        for key, value in self.anchor.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        for parameter in self.anchor.residual.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, 7.)))
        for base in (model.base_generator, model.base_shadow):
            self.assertFalse(any(child.training for child in base.modules()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in base.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.residual.parameters()))

    def test_apply_output_identity_routes_gradients_only_to_selected_generated_values(self):
        model = self.model('identity')
        clean = torch.arange(30, dtype=torch.float32).reshape(1, 30) / 10
        raw = torch.full((1, 30), 50., requires_grad=True)
        result = model.apply_output(clean, raw)
        self.assertTrue(torch.equal(result[:, ~self.mask], clean[:, ~self.mask]))
        self.assertTrue(torch.equal(result[:, self.mask], raw[:, self.mask]))
        result.sum().backward()
        self.assertTrue(torch.equal(raw.grad, self.mask.float().unsqueeze(0)))

    def test_native_clamp_has_zero_saturated_derivative_without_ste(self):
        model = self.model('domain_projection')
        clean = torch.zeros((1, 30))
        raw = torch.zeros((1, 30), requires_grad=True)
        selected = torch.nonzero(self.mask).flatten()
        with torch.no_grad():
            raw[0, selected[0]] = 11.
            raw[0, selected[1]] = -11.
            raw[0, selected[2]] = 3.
            raw[0, ~self.mask] = 50.
        result = model.apply_output(clean, raw)
        self.assertTrue(bool((result.abs() <= 10).all()))
        result.sum().backward()
        expected = self.mask.float().unsqueeze(0)
        expected[0, selected[:2]] = 0.
        self.assertTrue(torch.equal(raw.grad, expected))
        self.assertEqual(float(result[0, selected[0]].detach()), 10.)
        self.assertEqual(float(result[0, selected[1]].detach()), -10.)

    def test_gradient_reaches_existing_residual_but_not_frozen_bases_or_masked_head_rows(self):
        model = self.model()
        before_mask = self.mask.clone()
        output = model(self.x, self.context, self.targets)
        output.sum().backward()
        gradients = [p.grad for p in model.residual.parameters()]
        self.assertTrue(any(g is not None and bool((g != 0).any()) for g in gradients))
        self.assertTrue(all(g is None or bool(torch.isfinite(g).all()) for g in gradients))
        self.assertTrue(bool((model.residual.head.bias.grad[~self.mask] == 0).all()))
        self.assertTrue(bool((model.residual.head.weight.grad[~self.mask] == 0).all()))
        self.assertTrue(bool((model.residual.head.bias.grad[self.mask] != 0).all()))
        self.assertTrue(torch.equal(model.hard_support, before_mask))
        for base in (model.base_generator, model.base_shadow):
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in base.parameters()))

    def test_inherited_inject_preserves_unselected_non_target_topology_and_weights(self):
        for mode in OUTPUT_MODES:
            model = self.model(mode)
            weights = torch.ones(self.context.edge_index.shape[1])
            before = self.x.clone(), self.context.edge_index.clone(), weights.clone()
            actual = model.inject(self.x, self.context, self.targets, weights)
            allowed = torch.zeros_like(self.x, dtype=torch.bool)
            allowed[self.targets[:, None], model.selected_coordinates[model.hard_support]] = True
            self.assertTrue(torch.equal(actual.features[~allowed], self.x[~allowed]))
            self.assertTrue(torch.equal(actual.edge_index, before[1]))
            self.assertTrue(torch.equal(actual.edge_weight, before[2]))
            self.assertTrue(torch.equal(self.x, before[0]))
            self.assertTrue(actual.features.requires_grad)
            detached = model.inject(self.x, self.context, self.targets, weights, detach=True)
            self.assertFalse(detached.features.requires_grad)
            self.assertTrue(torch.equal(detached.features, actual.features))
            if mode == 'domain_projection':
                self.assertTrue(bool((actual.features[allowed].abs() <= 10).all()))

    def test_output_mode_and_support_mutations_fail_in_forward_and_state(self):
        for mutation in ('mode', 'mask', 'mask_data', 'replace', 'restored', 'coordinates', 'attention'):
            model = self.model()
            if mutation == 'mode': model.output_mode_code.fill_(1)
            elif mutation == 'mask': model.hard_support.logical_not_()
            elif mutation == 'mask_data': model.hard_support.data.logical_not_()
            elif mutation == 'replace': model.hard_support = model.hard_support.clone()
            elif mutation == 'restored':
                old = model.hard_support.clone()
                model.hard_support.logical_not_(); model.hard_support.copy_(old)
            elif mutation == 'coordinates': model.selected_coordinates[0] = 0
            else: model.attention_mode = 'learned'
            with self.subTest(mutation=mutation):
                for action in (lambda: model(self.x, self.context, self.targets), model.state_dict, model.train):
                    with self.assertRaises(ValueError): action()
        with self.assertRaises(AttributeError):
            self.model().output_mode = 'domain_projection'

    def test_constructor_rejects_wrong_anchor_mask_and_modes(self):
        for mutation in ('mode', 'dtype', 'shape', 'count', 'metadata', 'anchor', 'nonfinite'):
            anchor, mask, output_mode = deepcopy(self.anchor), self.mask.clone(), 'identity'
            if mutation == 'mode': output_mode = 'l1_matched_scale'
            elif mutation == 'dtype': mask = mask.float()
            elif mutation == 'shape': mask = mask[:29]
            elif mutation == 'count': mask[torch.nonzero(mask)[0]] = False
            elif mutation == 'metadata':
                mask[0] = True; mask[torch.nonzero(mask)[-1]] = False
            elif mutation == 'anchor': anchor.attention_mode = 'learned'
            else:
                with torch.no_grad(): anchor.residual.head.bias[0] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                DomainConstrainedGenerator(anchor, mask, output_mode=output_mode)

    def test_checkpoint_restores_residual_and_rejects_cross_mode_mask_or_base_changes_before_loading(self):
        model = self.model('domain_projection')
        with torch.no_grad(): model.residual.head.bias.add_(.25)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream); stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = self.model('domain_projection')
        restored.load_state_dict(saved)
        self.assertTrue(torch.equal(model(self.x, self.context, self.targets), restored(self.x, self.context, self.targets)))
        self.assertEqual(_state_hash(restored), _state_hash(model))
        with self.assertRaisesRegex(ValueError, 'output_mode_code'):
            self.model('identity').load_state_dict(saved)
        for mutation in ('mask', 'base', 'missing', 'nonfinite', 'shape', 'assign', 'nonstrict'):
            target, bad = self.model('domain_projection'), deepcopy(saved)
            before = snapshot(target)
            if mutation == 'mask': bad['hard_support'].logical_not_()
            elif mutation == 'base':
                key = next(key for key in bad if key.startswith('base_generator.') and bad[key].is_floating_point())
                bad[key].add_(1)
            elif mutation == 'missing': del bad['hard_support']
            elif mutation == 'nonfinite': bad['residual.head.bias'][0] = float('inf')
            elif mutation == 'shape': bad['residual.head.bias'] = bad['residual.head.bias'][:1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                target.load_state_dict(bad, assign=mutation == 'assign', strict=mutation != 'nonstrict')
            for key, value in target.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)

    def test_dtype_transfer_preserves_contract_and_output_dtype(self):
        model = self.model('domain_projection').double()
        model.assert_contract()
        output = model(self.x.double(), self.context, self.targets)
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(model.hard_support.dtype, torch.bool)
        self.assertEqual(model.output_mode_code.dtype, torch.uint8)
        self.assertTrue(bool((output.abs() <= 10).all()))

    def test_deepcopy_preserves_owned_contract_without_rng_or_shared_storage(self):
        model = self.model('domain_projection')
        initial_rng = torch.random.get_rng_state().clone()
        copied = deepcopy(model)
        self.assertTrue(torch.equal(initial_rng, torch.random.get_rng_state()))
        copied.assert_contract()
        self.assertEqual(_state_hash(model), _state_hash(copied))
        for before, after in zip(model.parameters(), copied.parameters()):
            self.assertNotEqual(before.data_ptr(), after.data_ptr())
        self.assertNotEqual(model.hard_support.data_ptr(), copied.hard_support.data_ptr())
        self.assertTrue(torch.equal(model(self.x, self.context, self.targets),
                                    copied(self.x, self.context, self.targets)))

    def test_original_conditioning_inputs_and_clean_domain_are_checked(self):
        model = self.model()
        x = self.x.clone().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'must not require gradients'):
            model(x, self.context, self.targets)
        clean = torch.zeros((2, 30)); generated = torch.zeros_like(clean)
        clean[0, 0] = 11.
        with self.assertRaisesRegex(ValueError, 'clean selected'):
            model.apply_output(clean, generated)
        with self.assertRaises(ValueError):
            model.apply_output(torch.zeros((2, 29)), torch.zeros((2, 29)))

    def test_legacy_content_training_loop_accepts_subclass_and_tensor_only_state(self):
        for mode in OUTPUT_MODES:
            model = self.model(mode)
            initial_base = snapshot(model.base_generator), snapshot(model.base_shadow)
            initial_residual = snapshot(model.residual)
            surrogate = classifier()
            weights = torch.ones(self.context.edge_index.shape[1])
            constraint = FrozenMeanCleanResponsePenalty((classifier(), classifier()), self.x,
                                                        self.context.edge_index, weights, weight=1.)
            result = fit_content_residual(model, surrogate, self.x, self.context,
                torch.tensor([0, 1, 0, 0, 1, 0]), torch.tensor([0, 1]), torch.tensor([2]),
                torch.tensor([2, 3, 4]), constraint,
                ResidualTrainingConfig(epochs=1, inner_steps=1, shadow_lr=.001, residual_lr=.001))
            self.assertEqual((result['residual_steps'], result['surrogate_steps']), (1, 1))
            self.assertTrue(result['original_inputs_unchanged'])
            self.assertTrue(result['frozen_models_unchanged'])
            self.assertTrue(result['caller_surrogate_unchanged'])
            self.assertTrue(any(not torch.equal(value, initial_residual[key])
                                for key, value in model.residual.state_dict().items()))
            model.assert_contract()
            for base, before in zip((model.base_generator, model.base_shadow), initial_base):
                for key, value in base.state_dict().items(): self.assertTrue(torch.equal(value, before[key]))
            if mode == 'domain_projection':
                self.assertTrue(bool((result['poisoned_features'].abs() <= 10).all()))


if __name__ == '__main__':
    unittest.main()
