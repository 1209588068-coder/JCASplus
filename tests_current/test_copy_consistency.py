"""Synthetic equality/projection contracts; no dataset or checkpoint reads."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from spear_current.copy_consistency import (
    COPY_GROUPS, COPY_POSITIONS, CopyConsistentGenerator, copy_consistency_statistics,
)
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.support_schema import SELECTED_COORDINATES
from tests_current.test_domain_generator import setup


ACTIVE = (776, 743, 778, 708, 709, 742, 777, 745, 744, 779, 351, 782,
          350, 729, 132, 712, 781, 353, 749, 746, 134, 724, 135, 760)


def fixture():
    content, _, x, context, targets = setup()
    mask = torch.tensor([c in ACTIVE for c in SELECTED_COORDINATES])
    anchor = DomainConstrainedGenerator(content, mask, output_mode='domain_projection').eval()
    for g in COPY_GROUPS:
        x[:, list(g)] = x[:, [g[0]]].expand(-1, 3)
    normalizer = dict(mean=np.zeros(784, dtype=np.float64), scale=np.ones(784, dtype=np.float64),
                      clip=10., feature_dimension=784, minimum_scale=1e-6)
    return anchor, normalizer, x, context, targets


class CopyConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.anchor, self.normalizer, self.x, self.context, self.targets = fixture()

    def model(self, mode='pair_mean', frozen=False):
        return CopyConsistentGenerator(self.anchor, self.normalizer, mode=mode, frozen=frozen)

    def test_constructor_owns_weights_preserves_rng_and_anchor(self):
        before = {k: v.clone() for k, v in self.anchor.state_dict().items()}
        rng = torch.get_rng_state().clone()
        models = [self.model(mode) for mode in ('identity', 'pair_mean')]
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for model in models:
            for k, value in before.items():
                self.assertTrue(torch.equal(model.state_dict()[k], value), k)
                self.assertTrue(torch.equal(self.anchor.state_dict()[k], value), k)
            for p, q in zip(model.parameters(), self.anchor.parameters()):
                self.assertNotEqual(p.data_ptr(), q.data_ptr())
            self.assertEqual(model.output_mode, 'domain_projection')
            self.assertEqual(model.contract()['allowed_write_count'], 24)
        self.assertEqual(models[0].contract()['independent_output_value_dof'], 24)
        self.assertEqual(models[1].contract()['independent_output_value_dof'], 12)

    def test_identity_replays_parent_bitwise(self):
        model = self.model('identity')
        for training in (False, True, False):
            model.train(training)
            self.assertTrue(torch.equal(model(self.x, self.context, self.targets),
                                        self.anchor(self.x, self.context, self.targets)))

    def test_mean_is_per_row_after_clamp_and_only_on_six_groups(self):
        model = self.model()
        clean = torch.zeros(2, 30)
        raw = torch.arange(60, dtype=torch.float32).reshape(2, 30) / 10 - 2
        raw[0, list(COPY_POSITIONS[0])] = torch.tensor([20., 0., 0.])
        raw[1, list(COPY_POSITIONS[0])] = torch.tensor([-9., 3., 0.])
        original = raw.clone()
        expected = self.anchor.apply_output(clean, raw)
        for positions in COPY_POSITIONS:
            group = expected[:, positions].double()
            expected[:, positions] = group.mean(1, keepdim=True).to(raw.dtype).expand(-1, 3)
        actual = model.apply_output(clean, raw)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(raw, original))
        self.assertAlmostEqual(float(actual[0, COPY_POSITIONS[0][0]]), 10/3, places=6)
        self.assertEqual(float(actual[1, COPY_POSITIONS[0][0]]), -2.)
        self.assertTrue(bool((actual.abs() <= 10).all()))
        other = [i for i in range(30) if all(i not in g for g in COPY_POSITIONS)]
        self.assertTrue(torch.equal(actual[:, other], self.anchor.apply_output(clean, raw)[:, other]))
        stats = copy_consistency_statistics(actual)
        self.assertEqual(stats['any_group_violations'], 0)
        self.assertTrue(all(row['max_range'] == 0 for row in stats['groups']))

    def test_exact_equal_double_and_idempotence(self):
        model = self.model().double()
        clean = torch.full((2, 30), .1, dtype=torch.float64)
        self.assertTrue(torch.equal(model.apply_output(clean, clean), clean))
        negative_zero = torch.full((2, 30), -0., dtype=torch.float64)
        self.assertTrue(torch.equal(model.apply_output(negative_zero, negative_zero).view(torch.int64),
                                    negative_zero.view(torch.int64)))
        raw = torch.linspace(-9, 9, 60, dtype=torch.float64).reshape(2, 30)
        once = model.apply_output(clean, raw)
        self.assertTrue(torch.equal(model.apply_output(clean, once), once))

    def test_true_mean_jacobian_and_native_clamp(self):
        model = self.model().double()
        clean = torch.zeros(1, 30, dtype=torch.float64)
        raw = torch.zeros_like(clean, requires_grad=True)
        p = COPY_POSITIONS[0]
        with torch.no_grad():
            raw[0, list(p)] = torch.tensor([11., 3., 6.], dtype=raw.dtype)
        model.apply_output(clean, raw)[0, p[0]].backward()
        expected = torch.zeros_like(raw)
        expected[0, list(p[1:])] = 1/3
        torch.testing.assert_close(raw.grad, expected, rtol=0, atol=1e-15)
        # At equality the derivative must still average, not select one copy.
        equal = torch.zeros_like(clean, requires_grad=True)
        model.apply_output(clean, equal)[0, p[0]].backward()
        expected[0, p[0]] = 1/3
        torch.testing.assert_close(equal.grad, expected, rtol=0, atol=1e-15)
        raw = torch.linspace(-2, 2, 30, dtype=torch.float64).reshape(1, 30).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda v: model.apply_output(clean, v), (raw,)))

    def test_copy_valid_clean_budget_norms_do_not_increase(self):
        model = self.model()
        clean = self.x[:, model.selected_coordinates]
        raw = torch.linspace(-13, 13, clean.numel()).reshape_as(clean)
        before = self.anchor.apply_output(clean, raw) - clean
        after = model.apply_output(clean, raw) - clean
        for p in (1, 2, float('inf')):
            self.assertTrue(bool((after.double().norm(p=p, dim=1) <=
                                  before.double().norm(p=p, dim=1) + 1e-5).all()))

    def test_injection_preserves_other_rows_fields_and_edges(self):
        model = self.model()
        old = self.x.clone()
        injected = model.inject(self.x, self.context, self.targets)
        allowed = torch.zeros_like(self.x, dtype=torch.bool)
        allowed[self.targets[:, None], model.selected_coordinates[model.hard_support]] = True
        self.assertTrue(torch.equal(injected.features[~allowed], self.x[~allowed]))
        self.assertTrue(torch.equal(self.x, old))
        self.assertTrue(torch.equal(injected.edge_index, self.context.edge_index))
        self.assertTrue(bool((injected.edge_weight == 1).all()))
        values = injected.features[self.targets[:, None], model.selected_coordinates]
        self.assertEqual(copy_consistency_statistics(values)['any_group_violations'], 0)
        injected.features.sum().backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.residual.parameters()))
        self.assertTrue(all(p.grad is None for base in (model.base_generator, model.base_shadow) for p in base.parameters()))

    def test_empty_targets_and_empty_statistics(self):
        model = self.model()
        empty = torch.empty(0, dtype=torch.long)
        output = model(self.x, self.context, empty)
        self.assertEqual(output.shape, (0, 30))
        self.assertTrue(torch.equal(model.inject(self.x, self.context, empty).features, self.x))
        stats = copy_consistency_statistics(output)
        self.assertEqual(stats['any_group_violations'], 0)
        self.assertIsNone(stats['any_group_violation_rate'])

    def test_reject_inconsistent_clean_in_both_modes(self):
        clean = torch.zeros(1, 30)
        clean[0, COPY_POSITIONS[0][0]] = 1e-7
        for mode in ('identity', 'pair_mean'):
            with self.assertRaisesRegex(ValueError, 'Clean input'):
                self.model(mode).apply_output(clean, torch.zeros_like(clean))

    def test_normalizer_guard_and_binding(self):
        for field, value in [('scale', 2.), ('mean', 1e-12)]:
            bad = deepcopy(self.normalizer)
            bad[field][350] = value
            with self.assertRaisesRegex(ValueError, 'Different normalization'):
                CopyConsistentGenerator(self.anchor, bad)
        for field, value in [('scale', np.zeros(784)), ('mean', np.full(784, np.nan)),
                             ('mean', np.zeros(783)), ('clip', 9.), ('feature_dimension', 785),
                             ('minimum_scale', 1e-5)]:
            bad = deepcopy(self.normalizer)
            bad[field] = value
            with self.assertRaises(ValueError):
                CopyConsistentGenerator(self.anchor, bad)
        model = self.model()
        before = model.contract()
        self.normalizer['mean'][350] = 9
        self.assertEqual(model.contract(), before)

    def test_missing_group_support_is_rejected(self):
        content, _, _, _, _ = setup()
        for remove in ([350], [350, 742, 777]):
            active = set(ACTIVE) - set(remove)
            for candidate in (352, 133, 503, 179):
                if len(active) < 24:
                    active.add(candidate)
            mask = torch.tensor([c in active for c in SELECTED_COORDINATES])
            domain = DomainConstrainedGenerator(content, mask, output_mode='domain_projection')
            with self.assertRaisesRegex(ValueError, 'All three'):
                CopyConsistentGenerator(domain, self.normalizer)

    def test_deepcopy_dtype_and_strict_restore(self):
        for frozen in (False, True):
            model = self.model(frozen=frozen)
            copied = deepcopy(model).double()
            copied.assert_contract()
            x = self.x.double()
            self.assertEqual(copy_consistency_statistics(copied(x, self.context, self.targets))['any_group_violations'], 0)
            restored = self.model(frozen=frozen)
            restored.load_state_dict(deepcopy(model.state_dict()))
            self.assertTrue(torch.equal(restored(self.x, self.context, self.targets),
                                        model(self.x, self.context, self.targets)))

    def test_checkpoint_mismatch_rejected_before_any_parameter_write(self):
        model = self.model()
        before = deepcopy(model.state_dict())
        for name in ('copy_mode_code', 'copy_frozen_code', 'copy_positions', 'copy_normalizer_digest'):
            bad = deepcopy(before)
            bad[name].view(-1)[0] += 1
            weight = next(k for k in bad if k.startswith('residual.') and bad[k].is_floating_point())
            bad[weight].add_(1.)
            with self.assertRaises(ValueError):
                model.load_state_dict(bad)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]), k)
        with self.assertRaises(ValueError):
            model.load_state_dict(self.model('identity').state_dict())
        with self.assertRaises(ValueError):
            model.load_state_dict(self.anchor.state_dict())

    def test_contract_tamper_and_frozen_rejection(self):
        for name in ('copy_mode_code', 'copy_frozen_code', 'copy_positions', 'copy_normalizer_digest'):
            model = self.model()
            getattr(model, name).data.view(-1)[0] += 1
            with self.assertRaises(ValueError):
                model.assert_contract()
        model = self.model(frozen=True)
        model.eval()
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):
            model.train()
        p = next(model.residual.parameters())
        p.requires_grad_(True)
        with self.assertRaises(ValueError):
            model.assert_contract()

    def test_frozen_buffers_and_nonfinite_anchor_are_rejected(self):
        model = self.model(frozen=True)
        model.base_generator.offset.add_(1.)
        with self.assertRaisesRegex(ValueError, 'Frozen postprocess tensor'):
            model.assert_contract()
        with torch.no_grad():
            self.anchor.residual.head.bias[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite tensor-only'):
            self.model()


if __name__ == '__main__':
    unittest.main()
