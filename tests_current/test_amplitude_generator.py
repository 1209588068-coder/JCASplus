# Source: tests/test_spear_amplitude_control_module.py; synthetic tests migrated with current-package imports.
"""Synthetic-only frozen amplitude and calibration contracts; no graph assets."""
from copy import deepcopy
import io
import json
import unittest

import torch
from torch import nn

from spear_current.amplitude_generator import (
    FrozenAmplitudeGenerator, calibrate_scale,
)
from tests_current.fixtures_tensors import assert_output_scope
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.role_generator import InjectedFeatures
from tests_current.fixtures_tensors import setup
from tests_current.fixtures_tensors import snapshot


class FrozenAmplitudeModuleTests(unittest.TestCase):
    def setUp(self):
        anchor, self.mask, self.x, self.context, self.targets = setup()
        self.anchor = DomainConstrainedGenerator(anchor, self.mask, output_mode='domain_projection')

    def model(self, scale=.3):
        return FrozenAmplitudeGenerator(self.anchor, scale)

    def assert_state_equal(self, module, before):
        self.assertEqual(set(module.state_dict()), set(before))
        for name, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_construction_owns_frozen_copies_preserves_source_and_rng(self):
        self.anchor.train()
        for parameter in self.anchor.residual.parameters():
            parameter.grad = torch.full_like(parameter, 3.)
        before = snapshot(self.anchor)
        rng = torch.random.get_rng_state().clone()
        model = self.model()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertIsInstance(model, nn.Module)
        self.assertNotIsInstance(model, DomainConstrainedGenerator)
        self.assertIsNot(model.anchor, self.anchor)
        self.assertFalse(any(child.training for child in model.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertEqual(set(model.state_dict()), {'scale_value'} | {'anchor.' + key for key in before})
        for old, new in zip(self.anchor.parameters(), model.parameters()):
            self.assertNotEqual(old.data_ptr(), new.data_ptr())
            self.assertTrue(torch.equal(old, new))
        self.assertTrue(self.anchor.training)
        self.assertTrue(all(p.requires_grad for p in self.anchor.residual.parameters()))
        self.assertTrue(all(torch.equal(p.grad, torch.full_like(p, 3.))
                            for p in self.anchor.residual.parameters()))
        self.assert_state_equal(self.anchor, before)
        model.assert_contract()

    def test_forward_matches_float64_formula_and_is_gradient_free(self):
        model = self.model(.123456789012345)
        clean = self.x[self.targets[:, None], model.selected_coordinates]
        with torch.no_grad():
            generated = self.anchor(self.x, self.context, self.targets)
        expected = (clean.double() + model.scale * (generated.double() - clean.double())).to(self.x.dtype)
        expected[:, ~self.mask] = clean[:, ~self.mask]
        actual = model(self.x, self.context, self.targets)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, self.x.dtype)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(bool((actual.abs() <= 10.).all()))
        self.assertEqual(model.output_mode, 'domain_projection')
        self.assertTrue(torch.equal(model.hard_support, self.mask))

    def test_alpha_one_is_exact_and_zero_deltas_and_off_support_preserve_bits(self):
        model = self.model(1.)
        generated = self.anchor(self.x, self.context, self.targets)
        self.assertTrue(torch.equal(model(self.x, self.context, self.targets), generated))
        clean = torch.linspace(-10., 10., 60).reshape(2, 30)
        clean[0, :] = -0.
        values = clean.clone()
        values[1, self.mask] = -values[1, self.mask]
        values[:, ~self.mask] = 9.
        for scale in (1., .5, 1e-15):
            actual = self.model(scale).apply_output(clean, values)
            self.assertTrue(torch.equal(actual[:, ~self.mask].view(torch.int32),
                                        clean[:, ~self.mask].contiguous().view(torch.int32)))
            self.assertTrue(torch.equal(actual[0].view(torch.int32), clean[0].view(torch.int32)))
            self.assertTrue(bool((actual.abs() <= 10.).all()))
        self.assertTrue(torch.equal(model.apply_output(clean, values)[:, self.mask], values[:, self.mask]))

    def test_injection_preserves_context_topology_weights_inputs_and_state(self):
        model = self.model()
        weights = torch.ones(self.context.edge_index.shape[1], dtype=torch.float64)
        inputs = [self.x, self.targets, weights, self.context.edge_index,
                  self.context.edge_type, self.context.pair_scene, self.context.score_permutation]
        originals = [(value.clone(), value._version) for value in inputs]
        before = snapshot(model)
        rng = torch.random.get_rng_state().clone()
        actual = model.inject(self.x, self.context, self.targets, weights)
        self.assertIsInstance(actual, InjectedFeatures)
        assert_output_scope(self.x, actual.features, self.targets, model)
        self.assertFalse(actual.features.requires_grad)
        self.assertTrue(torch.equal(actual.edge_index, self.context.edge_index))
        self.assertTrue(torch.equal(actual.edge_weight, weights))
        self.assertEqual(actual.edge_weight.dtype, weights.dtype)
        for output, original in ((actual.features, self.x), (actual.edge_index, self.context.edge_index),
                                 (actual.edge_weight, weights)):
            self.assertNotEqual(output.data_ptr(), original.data_ptr())
        for value, (original, version) in zip(inputs, originals):
            self.assertTrue(torch.equal(value, original))
            self.assertEqual(value._version, version)
        self.assert_state_equal(model, before)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(model.inject(self.x, self.context, self.targets, detach=True).features,
                                    actual.features))

    def test_empty_targets_keep_original_graph(self):
        model = self.model()
        empty = torch.empty(0, dtype=torch.long)
        self.assertEqual(model(self.x, self.context, empty).shape, (0, 30))
        self.assertTrue(torch.equal(model.inject(self.x, self.context, empty).features, self.x))

    def test_training_is_rejected_without_mode_or_state_changes(self):
        model = self.model()
        before = snapshot(model)
        for mode in (True, 1, None, 'False'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                model.train(mode)
        self.assertIs(model.train(False), model)
        self.assertIs(model.eval(), model)
        self.assert_state_equal(model, before)
        self.assertFalse(any(child.training for child in model.modules()))

    def test_scalar_and_owned_state_tampering_is_rejected(self):
        for mutation in ('scalar', 'scalar_data', 'scalar_replace', 'scalar_restore', 'scalar_grad',
                         'weight', 'weight_data', 'grad', 'requires_grad', 'training',
                         'child_training', 'child_replace', 'anchor_replace', 'support', 'coordinates'):
            model = self.model()
            parameter = next(model.anchor.residual.parameters())
            if mutation == 'scalar': model.scale_value.fill_(.4)
            elif mutation == 'scalar_data': model.scale_value.data.fill_(.4)
            elif mutation == 'scalar_replace': model.scale_value = model.scale_value.clone()
            elif mutation == 'scalar_restore':
                model.scale_value.add_(.1); model.scale_value.fill_(.3)
            elif mutation == 'scalar_grad': model.scale_value.requires_grad_(True)
            elif mutation == 'weight': parameter.add_(.1)
            elif mutation == 'weight_data': parameter.data.add_(.1)
            elif mutation == 'grad': parameter.grad = torch.zeros_like(parameter)
            elif mutation == 'requires_grad': parameter.requires_grad_(True)
            elif mutation == 'training': model.training = True
            elif mutation == 'child_training': model.anchor.residual.train()
            elif mutation == 'child_replace': model.anchor.residual.head = deepcopy(model.anchor.residual.head)
            elif mutation == 'anchor_replace': model.anchor = deepcopy(model.anchor)
            elif mutation == 'support': model.hard_support.data.logical_not_()
            else: model.selected_coordinates.data[0] = 0
            for action in (model.assert_contract, model.state_dict,
                           lambda: model(self.x, self.context, self.targets)):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    action()

    def test_constructor_rejects_invalid_scale_anchor_and_nonfinite_weights(self):
        for scale in (0, -1, 1.00000001, float('nan'), float('inf'), True, torch.tensor(.3), '.3'):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.model(scale)
        anchor, mask, _, _, _ = setup()
        for bad in (anchor, DomainConstrainedGenerator(anchor, mask, output_mode='identity'), None):
            with self.assertRaises(ValueError): FrozenAmplitudeGenerator(bad, .3)
        with torch.no_grad(): self.anchor.residual.head.bias[0] = float('nan')
        with self.assertRaises(ValueError): self.model()

    def test_selected_tensor_validation_is_strict(self):
        model = self.model()
        for mutation in ('shape', 'mismatch', 'integer', 'dtype', 'nan', 'infinity', 'clean_bound', 'generated_bound'):
            clean, values = torch.zeros(2, 30), torch.ones(2, 30)
            if mutation == 'shape': clean, values = clean[:, :29], values[:, :29]
            elif mutation == 'mismatch': values = values[:1]
            elif mutation == 'integer': clean, values = clean.long(), values.long()
            elif mutation == 'dtype': values = values.double()
            elif mutation == 'nan': clean[0, 0] = float('nan')
            elif mutation == 'infinity': values[0, 0] = float('inf')
            elif mutation == 'clean_bound': clean[0, 0] = -10.0001
            else: values[0, 0] = 10.0001
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                model.apply_output(clean, values)

    def test_graph_and_target_validation_uses_original_injection_contract(self):
        model = self.model()
        for targets in (torch.tensor([1, 1]), torch.tensor([-1]), torch.tensor([6]),
                        torch.tensor([1.]), torch.tensor([[1]])):
            with self.assertRaises(ValueError): model.inject(self.x, self.context, targets)
        for weights in (torch.full((self.context.edge_index.shape[1],), .5),
                        torch.ones(1), torch.full((self.context.edge_index.shape[1],), float('nan'))):
            with self.assertRaises(ValueError):
                model.inject(self.x, self.context, self.targets, weights)
        with self.assertRaises(ValueError): model.inject(self.x, self.context, self.targets, detach=1)
        with self.assertRaises(ValueError): model(self.x.requires_grad_(), self.context, self.targets)

    def test_checkpoint_round_trip_and_preflight_rejection(self):
        model = self.model()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = self.model()
        restored.load_state_dict(saved)
        self.assert_state_equal(restored, saved)
        self.assertTrue(torch.equal(restored(self.x, self.context, self.targets),
                                    model(self.x, self.context, self.targets)))
        for mutation in ('scale', 'weight', 'support', 'missing', 'dtype', 'shape', 'nan', 'assign', 'nonstrict'):
            target, bad = self.model(), deepcopy(saved)
            before = snapshot(target)
            if mutation == 'scale': bad['scale_value'].fill_(.4)
            elif mutation == 'weight': bad['anchor.residual.head.bias'].add_(1.)
            elif mutation == 'support': bad['anchor.hard_support'].logical_not_()
            elif mutation == 'missing': del bad['scale_value']
            elif mutation == 'dtype': bad['scale_value'] = bad['scale_value'].float()
            elif mutation == 'shape': bad['scale_value'] = bad['scale_value'].reshape(1)
            elif mutation == 'nan': bad['anchor.residual.head.bias'][0] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                target.load_state_dict(bad, assign=mutation == 'assign', strict=mutation != 'nonstrict')
            self.assert_state_equal(target, before)

    def test_deepcopy_and_dtype_transfer_rebind_guards_and_keep_scalar_exact(self):
        model = self.model(.123456789012345)
        copied = deepcopy(model)
        self.assert_state_equal(copied, snapshot(model))
        for old, new in zip(model.parameters(), copied.parameters()):
            self.assertNotEqual(old.data_ptr(), new.data_ptr())
        self.assertNotEqual(model.scale_value.data_ptr(), copied.scale_value.data_ptr())
        for dtype in (torch.float64, torch.float32):
            copied.to(dtype=dtype)
            copied.assert_contract()
            self.assertEqual(copied.scale, model.scale)
            self.assertEqual(copied.scale_value.dtype, torch.float64)
            self.assertEqual(copied(self.x.to(dtype), self.context, self.targets).dtype, dtype)


class AmplitudeCalibrationTests(unittest.TestCase):
    def test_global_mean_per_row_l1_ratio_and_reconstruction(self):
        clean = torch.zeros(2, 24)
        continuation = torch.stack((torch.full((24,), 2.), torch.full((24,), 4.)))
        unconditional = torch.ones(2, 24)
        before = [value.clone() for value in (clean, continuation, unconditional)]
        scale, stats = calibrate_scale(clean, continuation, unconditional)
        self.assertEqual(scale, 1. / 3.)
        self.assertEqual(stats['continuation_mean_l1'], 72.)
        self.assertEqual(stats['unconditional_mean_l1'], 24.)
        self.assertAlmostEqual(stats['actual_mean_l1'], 24., places=5)
        self.assertEqual((stats['target_count'], stats['support_count']), (2, 24))
        self.assertTrue(stats['matched'])
        json.dumps(stats, allow_nan=False)
        for actual, original in zip((clean, continuation, unconditional), before):
            self.assertTrue(torch.equal(actual, original))

    def test_equal_amplitude_is_exact_one_and_zero_delta_rows_are_allowed(self):
        clean = torch.zeros(3, 24, dtype=torch.float64)
        continuation = clean.clone()
        continuation[1, :] = 10.
        continuation[2, :] = -10.
        scale, stats = calibrate_scale(clean, continuation, continuation.clone())
        self.assertEqual(scale, 1.)
        self.assertEqual(stats['actual_mean_l1'], 160.)
        self.assertEqual(stats['absolute_error'], 0.)

    def test_calibration_rejects_invalid_population_bounds_norms_or_expansion(self):
        for mutation in ('empty', 'width', 'rows', 'dtype', 'integer', 'nan', 'inf',
                         'clean_bound', 'continuation_bound', 'unconditional_bound',
                         'zero_c', 'zero_u', 'expansion'):
            clean = torch.zeros(2, 24)
            continuation, unconditional = torch.ones_like(clean) * 2., torch.ones_like(clean)
            if mutation == 'empty': clean, continuation, unconditional = clean[:0], continuation[:0], unconditional[:0]
            elif mutation == 'width': clean, continuation, unconditional = clean[:, :23], continuation[:, :23], unconditional[:, :23]
            elif mutation == 'rows': unconditional = unconditional[:1]
            elif mutation == 'dtype': unconditional = unconditional.double()
            elif mutation == 'integer': clean, continuation, unconditional = clean.long(), continuation.long(), unconditional.long()
            elif mutation == 'nan': clean[0, 0] = float('nan')
            elif mutation == 'inf': continuation[0, 0] = float('inf')
            elif mutation == 'clean_bound': clean[0, 0] = -10.0001
            elif mutation == 'continuation_bound': continuation[0, 0] = 10.0001
            elif mutation == 'unconditional_bound': unconditional[0, 0] = 10.0001
            elif mutation == 'zero_c': continuation.zero_()
            elif mutation == 'zero_u': unconditional.zero_()
            else: unconditional.fill_(3.)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                calibrate_scale(clean, continuation, unconditional)

    def test_actual_norm_is_measured_after_dtype_restoration(self):
        clean = torch.full((2, 24), 9., dtype=torch.float32)
        continuation = clean - torch.linspace(.01, 4., 48).reshape(2, 24)
        unconditional = clean - .02
        scale, stats = calibrate_scale(clean, continuation, unconditional)
        reconstructed = (clean.double() + scale * (continuation.double() - clean.double())).float()
        actual = float((reconstructed.double() - clean.double()).abs().sum(1).mean())
        self.assertEqual(stats['actual_mean_l1'], actual)
        self.assertLessEqual(stats['absolute_error'], 1e-5)

    def test_unrepresentable_dtype_restored_match_fails_instead_of_changing_scale(self):
        clean = torch.full((2, 24), 5., dtype=torch.float16)
        continuation = clean.clone()
        continuation[:, :8], continuation[:, 8:] = 6., 8.
        unconditional = torch.full_like(clean, 5.00390625)
        with self.assertRaisesRegex(ValueError, 'Dtype-restored mean'):
            calibrate_scale(clean, continuation, unconditional)


if __name__ == '__main__':
    unittest.main()
