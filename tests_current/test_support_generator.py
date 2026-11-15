# Source: tests/test_spear_semantic_support_module.py; synthetic tests migrated with current-package imports.
"""Synthetic support contracts; no real graphs, assets, fitting, or file I/O."""
from copy import deepcopy
import io
import itertools
import random
import unittest

import torch

from spear_current.attention_context import attention_context
from spear_current.attention_generator import ContentAttentionGenerator
from spear_current.support_generator import (
    HardConcreteSupport, SupportMaskedGenerator, apply_gates, exact_cost_support,
    fixed_support, modification_statistics,
)
from spear_current.support_schema import (
    EXCLUDED_COORDINATES, LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups,
)
from tests_current.fixtures_tensors import activate_branch, fixture, make_model


def serialized_state(module):
    """Exercise the real weights-only format entirely in memory."""
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu', weights_only=True)


class StateAssertions:
    def assert_state_equal(self, module, expected):
        actual = module.state_dict()
        self.assertEqual(set(actual), set(expected))
        for name, value in actual.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, expected[name]), name)
            else:
                self.assertEqual(value, expected[name], name)

    def assert_grouped(self, mask, grouping):
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(tuple(mask.shape), (30,))
        for coordinate in EXCLUDED_COORDINATES:
            self.assertFalse(bool(mask[SELECTED_COORDINATES.index(coordinate)]))
        for group in support_groups(grouping):
            active = [bool(mask[SELECTED_COORDINATES.index(c)]) for c in group]
            self.assertTrue(all(active) or not any(active), group)


class SelectorTests(StateAssertions, unittest.TestCase):
    def test_probabilities_weighted_cost_and_shared_group_gates(self):
        for grouping in ('semantic', 'coordinate'):
            selector = HardConcreteSupport(grouping, initial_open_probability=.8)
            self.assertTrue(torch.allclose(selector.activation_probability(),
                                           torch.full((len(selector.groups),), .8)))
            self.assertAlmostEqual(float(selector.expected_coordinate_count().detach()), 28 * .8, places=4)
            gates = selector(sample=False)
            self.assertEqual(gates.shape, (30,))
            self.assertTrue(bool(((gates >= 0) & (gates <= 1)).all()))
            for coordinate in EXCLUDED_COORDINATES:
                self.assertEqual(float(gates[SELECTED_COORDINATES.index(coordinate)].detach()), 0.)
            for group in selector.groups:
                values = gates[[SELECTED_COORDINATES.index(c) for c in group]]
                self.assertTrue(torch.equal(values, values[0].expand_as(values)))
            selector.expected_coordinate_count().backward()
            self.assertTrue(bool((selector.log_alpha.grad > 0).all()))

    def test_projection_all_exact_budgets_including_zero_and_full_is_pure(self):
        for grouping in ('semantic', 'coordinate'):
            selector = HardConcreteSupport(grouping, seed=23)
            before = deepcopy(selector.state_dict())
            for budget in range(29):
                with self.subTest(grouping=grouping, budget=budget):
                    mask = selector.project(budget)
                    self.assertEqual(int(mask.sum()), budget)
                    self.assert_grouped(mask, grouping)
                    self.assertTrue(torch.equal(mask, selector.project(budget)))
            self.assert_state_equal(selector, before)
            self.assertTrue(torch.equal(selector.project(28), fixed_support('ranking', 28)))

    def test_projection_matches_exhaustive_weighted_optimum_and_stable_ties(self):
        groups = support_groups('semantic')
        scores = torch.arange(len(groups), dtype=torch.float64) / 16
        budget = 7
        feasible = []
        for length in range(len(groups) + 1):
            for chosen in itertools.combinations(range(len(groups)), length):
                if sum(len(groups[i]) for i in chosen) == budget:
                    objective = sum(len(groups[i]) * float(scores[i]) for i in chosen)
                    feasible.append((objective, chosen))
        expected = min(feasible, key=lambda value: (-value[0], value[1]))[1]
        mask = exact_cost_support(groups, scores, budget)
        actual = tuple(i for i, group in enumerate(groups)
                       if bool(mask[SELECTED_COORDINATES.index(group[0])]))
        self.assertEqual(actual, expected)
        coordinate_groups = support_groups('coordinate')
        tied = exact_cost_support(coordinate_groups, torch.zeros(28), 5)
        self.assertTrue(torch.equal(tied, fixed_support('ranking', 5)))

    def test_fixed_controls_all_budgets_reproducible_and_private_rng(self):
        python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
        for kind in ('ranking', 'random_coordinate', 'random_semantic'):
            for budget in range(29):
                with self.subTest(kind=kind, budget=budget):
                    mask = fixed_support(kind, budget, seed=17)
                    self.assertEqual(int(mask.sum()), budget)
                    self.assert_grouped(mask, 'semantic' if kind == 'random_semantic' else 'coordinate')
                    self.assertTrue(torch.equal(mask, fixed_support(kind, budget, seed=17)))
            self.assertGreater(len({tuple(fixed_support(kind, 7, seed=i).tolist())
                                    for i in range(8)}), 1 if kind != 'ranking' else 0)
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))
        ranking = fixed_support('ranking', 7)
        self.assertEqual([c for c, keep in zip(SELECTED_COORDINATES, ranking.tolist()) if keep],
                         list(LEGAL_COORDINATES[:7]))

    def test_sampling_rng_and_checkpoint_restore_are_isolated(self):
        python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
        selector = HardConcreteSupport('semantic', seed=11)
        first = selector()
        saved = serialized_state(selector)
        expected_next = selector()
        restored = HardConcreteSupport('semantic', seed=11)
        restored.load_state_dict(saved, strict=True)
        self.assertTrue(torch.equal(restored(), expected_next))
        self.assertFalse(torch.equal(first, expected_next))
        before = restored.rng_state.clone()
        self.assertTrue(torch.equal(restored(sample=False), restored(sample=False)))
        restored.project(9)
        self.assertTrue(torch.equal(before, restored.rng_state))
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))

    def test_shuffled_semantic_matches_capacity_and_freezes_seeded_group_contract(self):
        python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
        semantic = HardConcreteSupport('semantic', seed=11)
        shuffled = HardConcreteSupport('shuffled_semantic', seed=11, group_seed=41)
        self.assertEqual(shuffled.log_alpha.numel(), 16)
        self.assertTrue(torch.equal(shuffled.log_alpha, semantic.log_alpha))
        self.assertEqual(tuple(map(len, shuffled.groups)), tuple(map(len, semantic.groups)))
        self.assertNotEqual(shuffled.groups, semantic.groups)
        self.assertEqual(shuffled.groups, support_groups('shuffled_semantic', seed=41))
        other_gate_seed = HardConcreteSupport('shuffled_semantic', seed=12, group_seed=41)
        self.assertEqual(shuffled.groups, other_gate_seed.groups)
        other_group_seed = HardConcreteSupport('shuffled_semantic', seed=11, group_seed=42)
        self.assertNotEqual(shuffled.groups, other_group_seed.groups)
        self.assertTrue(torch.equal(shuffled.rng_state, other_group_seed.rng_state))
        groups = shuffled.groups
        shuffled()
        saved = serialized_state(shuffled)
        expected_next = shuffled()
        restored = HardConcreteSupport('shuffled_semantic', seed=11, group_seed=41)
        restored.load_state_dict(saved, strict=True)
        self.assertTrue(torch.equal(restored(), expected_next))
        rng_before = restored.rng_state.clone()
        for budget in range(29):
            with self.subTest(budget=budget):
                mask = restored.project(budget)
                self.assertEqual(int(mask.sum()), budget)
                self.assert_grouped(mask, 'coordinate')
                for group in groups:
                    active = mask[[SELECTED_COORDINATES.index(c) for c in group]]
                    self.assertTrue(bool(active.all()) or not bool(active.any()))
        self.assertTrue(torch.equal(restored.project(24), shuffled.project(24)))
        self.assertTrue(torch.equal(restored.rng_state, rng_before))
        self.assertEqual(restored.groups, groups)
        with self.assertRaises(ValueError):
            other_group_seed.load_state_dict(saved, strict=True)
        restored.groups = tuple(reversed(restored.groups))
        with self.assertRaises(ValueError):
            restored.project(24)
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))

    def test_exact_cost_rejects_invalid_partitions_and_infeasible_budgets(self):
        single_group = (LEGAL_COORDINATES,)
        self.assertEqual(int(exact_cost_support(single_group, [1.], 0).sum()), 0)
        self.assertTrue(torch.equal(exact_cost_support(single_group, [1.], 28),
                                    fixed_support('ranking', 28)))
        for budget in (1, 24, 27):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'infeasible'):
                exact_cost_support(single_group, [1.], budget)
        invalid = (
            [], (), [LEGAL_COORDINATES], (list(LEGAL_COORDINATES),),
            ((), LEGAL_COORDINATES), (LEGAL_COORDINATES[:-1],),
            ((LEGAL_COORDINATES[0],) + LEGAL_COORDINATES[:-1],),
            ((EXCLUDED_COORDINATES[0],) + LEGAL_COORDINATES[1:],),
            ((float(LEGAL_COORDINATES[0]),) + LEGAL_COORDINATES[1:],),
        )
        for partition in invalid:
            with self.subTest(partition=partition), self.assertRaises(ValueError):
                exact_cost_support(partition, torch.ones(len(partition)), 0)
        for scores in ([], [1., 2.], [float('nan')], [float('inf')]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                exact_cost_support(single_group, scores, 0)

    def test_invalid_selector_budgets_and_configuration_mismatches_rejected(self):
        for kwargs in (dict(grouping='unknown'), dict(seed=True), dict(temperature=0),
                       dict(lower=0), dict(upper=1), dict(initial_open_probability=1),
                       dict(initial_open_probability=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HardConcreteSupport(**kwargs)
        selector = HardConcreteSupport()
        for budget in (-1, 29, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                selector.project(budget)
            with self.assertRaises(ValueError):
                fixed_support('ranking', budget)
        with self.assertRaises(ValueError):
            fixed_support('unknown', 3)
        with self.assertRaises(ValueError):
            selector(sample=1)
        for kwargs in (dict(seed=2), dict(temperature=.5), dict(lower=-.2),
                       dict(initial_open_probability=.5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HardConcreteSupport(**kwargs).load_state_dict(deepcopy(selector.state_dict()), strict=True)
        with torch.no_grad():
            selector.log_alpha[0] = float('inf')
        with self.assertRaises(ValueError):
            selector()


class GateArithmeticTests(unittest.TestCase):
    def test_extreme_binary_endpoints_do_not_create_nan_gate_gradients(self):
        maximum = torch.finfo(torch.float32).max
        clean = torch.full((1, 2), maximum)
        generated = -clean
        gates = torch.tensor([0., 1.], requires_grad=True)
        output = apply_gates(clean, generated, gates)
        self.assertTrue(torch.equal(output, torch.tensor([[maximum, -maximum]])))
        output.sum().backward()
        self.assertTrue(torch.equal(gates.grad, torch.zeros(2)))
        with self.assertRaisesRegex(ValueError, 'Nonfinite interpolation delta'):
            apply_gates(clean, generated, torch.full((2,), .5))

    def test_binary_gates_preserve_bits_and_soft_gate_receives_gradient(self):
        clean = torch.tensor([[1e20, 1., -0., 8.], [-1e20, -1., 0., 3.]])
        generated = torch.tensor([[1., -0., 9., 12.], [-1., 0., -9., 7.]])
        gates = torch.tensor([1., 1., 0., .25], requires_grad=True)
        result = apply_gates(clean, generated, gates)
        self.assertTrue(torch.equal(result[:, :2].contiguous().view(torch.int32),
                                    generated[:, :2].contiguous().view(torch.int32)))
        self.assertTrue(torch.equal(result[:, 2].contiguous().view(torch.int32),
                                    clean[:, 2].contiguous().view(torch.int32)))
        self.assertTrue(torch.equal(result[:, 3], torch.tensor([9., 4.])))
        result.sum().backward()
        self.assertEqual(float(gates.grad[3]), 8.)
        self.assertTrue(torch.equal(apply_gates(clean, generated, torch.zeros(4)).view(torch.int32),
                                    clean.view(torch.int32)))
        self.assertTrue(torch.equal(apply_gates(clean, generated, torch.ones(4)).view(torch.int32),
                                    generated.view(torch.int32)))

    def test_invalid_gate_values_shapes_and_dtypes_rejected(self):
        clean, generated = torch.zeros(2, 3), torch.ones(2, 3)
        for gates in (torch.ones(2), torch.ones(3, dtype=torch.float64),
                      torch.tensor([0., -.1, 1.]), torch.tensor([0., 1.1, 1.]),
                      torch.tensor([0., float('nan'), 1.])):
            with self.subTest(gates=gates), self.assertRaises(ValueError):
                apply_gates(clean, generated, gates)
        with self.assertRaises(ValueError):
            apply_gates(clean, generated[:, :2], torch.ones(3))
        generated[0, 0] = float('inf')
        with self.assertRaises(ValueError):
            apply_gates(clean, generated, torch.ones(3))


class MaskedGeneratorTests(StateAssertions, unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            anchor = make_model(input_dim=784, hidden_dim=4,
                                selected=torch.tensor(SELECTED_COORDINATES), mode='untyped')
            activate_branch(anchor)
            self.anchor = ContentAttentionGenerator(anchor, attention_mode='uniform', score_dim=4).eval()
        _, role_context, self.features = fixture(input_dim=784)
        self.context = attention_context(role_context, ['synthetic-scene'], seed=17)
        self.targets = torch.tensor([4, 1, 5])

    def learned(self, grouping='semantic'):
        return SupportMaskedGenerator(self.anchor, HardConcreteSupport(grouping, seed=11))

    def test_owned_copies_preserve_caller_states_modes_and_gradients(self):
        self.anchor.train()
        for parameter in self.anchor.residual.parameters():
            parameter.grad = torch.full_like(parameter, 7.)
        caller_state = deepcopy(self.anchor.state_dict())
        selector = HardConcreteSupport(seed=11)
        selector_state = deepcopy(selector.state_dict())
        model = SupportMaskedGenerator(self.anchor, selector)
        self.assertIsNot(model.anchor, self.anchor)
        self.assertIsNot(model.selector, selector)
        self.assertTrue(self.anchor.training)
        self.assertTrue(all(torch.equal(p.grad, torch.full_like(p, 7.))
                            for p in self.anchor.residual.parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.anchor.parameters()))
        self.assertTrue(all(not child.training for child in model.anchor.modules()))
        model(self.features, self.context, self.targets)
        self.assert_state_equal(self.anchor, caller_state)
        self.assert_state_equal(selector, selector_state)
        self.assertNotEqual(model.anchor.selected_coordinates.data_ptr(), self.anchor.selected_coordinates.data_ptr())

    def test_training_mode_keeps_entire_anchor_frozen_and_eval(self):
        model = self.learned()
        before = deepcopy(model.anchor.state_dict())
        model.anchor.requires_grad_(True).train()
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.selector.training)
        self.assertTrue(all(not child.training for child in model.anchor.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.anchor.parameters()))
        model(self.features, self.context, self.targets)
        self.assert_state_equal(model.anchor, before)

    def test_only_explicit_selector_receives_gradients_and_detach_is_respected(self):
        model = self.learned()
        anchor_state = deepcopy(model.anchor.state_dict())
        original = self.features.clone()
        output = model.inject(self.features, self.context, self.targets)
        output.features.square().mean().backward()
        self.assertIsNotNone(model.selector.log_alpha.grad)
        self.assertTrue(bool(torch.isfinite(model.selector.log_alpha.grad).all()))
        self.assertGreater(int(torch.count_nonzero(model.selector.log_alpha.grad)), 0)
        self.assertTrue(all(p.grad is None for p in model.anchor.parameters()))
        self.assertIsNone(self.features.grad)
        self.assertTrue(torch.equal(self.features, original))
        self.assert_state_equal(model.anchor, anchor_state)
        self.assertFalse(model.inject(self.features, self.context, self.targets, detach=True).features.requires_grad)

    def test_hard_support_changes_only_allowed_rows_columns_and_keeps_anchor_values(self):
        mask = fixed_support('random_semantic', 7, seed=31)
        model = SupportMaskedGenerator(self.anchor, hard_support=mask).eval()
        mask.zero_()  # The constructor must own the supplied support.
        before = self.features.clone()
        result = model.inject(self.features, self.context, self.targets)
        with torch.no_grad():
            raw = self.anchor(self.features, self.context, self.targets)
        values = result.features[self.targets[:, None], model.selected_coordinates]
        clean_values = before[self.targets[:, None], model.selected_coordinates]
        self.assertTrue(torch.equal(values[:, model.hard_support], raw[:, model.hard_support]))
        self.assertTrue(torch.equal(values[:, ~model.hard_support], clean_values[:, ~model.hard_support]))
        allowed = torch.zeros_like(before, dtype=torch.bool)
        allowed[self.targets[:, None], model.selected_coordinates[model.hard_support]] = True
        self.assertTrue(torch.equal(result.features[~allowed], before[~allowed]))
        self.assertTrue(torch.equal(result.features[:, list(EXCLUDED_COORDINATES)], before[:, list(EXCLUDED_COORDINATES)]))
        self.assertTrue(torch.equal(result.edge_index, self.context.edge_index))
        self.assertTrue(torch.equal(result.edge_weight, torch.ones(self.context.edge_index.shape[1])))
        self.assertTrue(torch.equal(self.features, before))
        self.assertNotEqual(result.features.data_ptr(), self.features.data_ptr())
        self.assertNotEqual(result.edge_index.data_ptr(), self.context.edge_index.data_ptr())

    def test_zero_and_full_legal_support_restore_expected_values_exactly(self):
        zero = SupportMaskedGenerator(self.anchor, hard_support=fixed_support('ranking', 0)).eval()
        self.assertTrue(torch.equal(zero.inject(self.features, self.context, self.targets).features, self.features))
        full = SupportMaskedGenerator(self.anchor, hard_support=fixed_support('ranking', 28)).eval()
        with torch.no_grad():
            expected = self.anchor.inject(self.features, self.context, self.targets).features
        expected[self.targets[:, None], torch.tensor(EXCLUDED_COORDINATES)] = self.features[
            self.targets[:, None], torch.tensor(EXCLUDED_COORDINATES)]
        self.assertTrue(torch.equal(full.inject(self.features, self.context, self.targets).features, expected))

    def test_eval_requires_hardening_and_hardening_is_one_way(self):
        model = self.learned().eval()
        rng_before = model.selector.rng_state.clone()
        with self.assertRaisesRegex(ValueError, 'frozen binary support'):
            model(self.features, self.context, self.targets)
        self.assertTrue(torch.equal(rng_before, model.selector.rng_state))
        model.harden(7)
        first = model.inject(self.features, self.context, self.targets).features
        second = model.inject(self.features, self.context, self.targets).features
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(rng_before, model.selector.rng_state))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        for action in (lambda: model.harden(8), lambda: model.freeze_support(fixed_support('ranking', 7))):
            with self.assertRaises(ValueError):
                action()

    def test_fixed_controls_have_no_selector_and_no_trainable_parameters(self):
        for kind in ('ranking', 'random_coordinate', 'random_semantic'):
            model = SupportMaskedGenerator(self.anchor, hard_support=fixed_support(kind, 7)).train()
            self.assertIsNone(model.selector)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            result = model.inject(self.features, self.context, self.targets)
            self.assertFalse(result.features.requires_grad)
            self.assertEqual(int(model.hard_support.sum()), 7)
            with self.assertRaises(ValueError):
                model.harden(7)

    def test_hardened_checkpoint_roundtrip_restores_freeze_and_exact_inference(self):
        model = self.learned()
        model(self.features, self.context, self.targets)
        model.harden(7)
        model.eval()
        expected = model.inject(self.features, self.context, self.targets).features
        saved = serialized_state(model)
        restored = self.learned()
        restored.load_state_dict(saved, strict=True)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        restored.eval()
        self.assertTrue(torch.equal(restored.inject(self.features, self.context, self.targets).features, expected))
        self.assert_state_equal(restored, saved)
        fixed = SupportMaskedGenerator(self.anchor, hard_support=fixed_support('ranking', 2))
        fixed_restored = SupportMaskedGenerator(self.anchor, hard_support=fixed_support('ranking', 0))
        fixed_restored.load_state_dict(serialized_state(fixed), strict=True)
        self.assertTrue(torch.equal(fixed_restored(self.features, self.context, self.targets),
                                    fixed(self.features, self.context, self.targets)))

    def test_checkpoint_rejects_wrong_configuration_and_tampered_mask(self):
        model = self.learned()
        model.harden(7)
        saved = deepcopy(model.state_dict())
        wrong = deepcopy(saved)
        wrong['_extra_state']['anchor_mode'] = 'learned'
        with self.assertRaises(ValueError):
            self.learned().load_state_dict(wrong, strict=True)
        with self.assertRaises(ValueError):
            SupportMaskedGenerator(self.anchor, HardConcreteSupport(seed=17)).load_state_dict(saved, strict=True)
        tampered = deepcopy(saved)
        slot = SELECTED_COORDINATES.index(LEGAL_COORDINATES[0])
        tampered['hard_support'][slot] = ~tampered['hard_support'][slot]
        restored = self.learned()
        with self.assertRaises(ValueError):
            restored.load_state_dict(tampered, strict=True)
            restored.eval()(self.features, self.context, self.targets)

    def test_mutated_frozen_mask_mode_coordinates_and_freeze_flag_rejected(self):
        for mutation in ('mask', 'mode', 'coordinates', 'freeze_flag'):
            model = self.learned()
            model.harden(7)
            if mutation == 'mask':
                model.hard_support[2] = ~model.hard_support[2]
            elif mutation == 'mode':
                model.anchor.attention_mode = 'learned'
            elif mutation == 'coordinates':
                model.selected_coordinates[2], model.selected_coordinates[3] = (
                    model.selected_coordinates[3].clone(), model.selected_coordinates[2].clone())
            else:
                model.support_frozen.fill_(False)
                with self.assertRaises(ValueError):
                    model.freeze_support(fixed_support('ranking', 8))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                model.eval()(self.features, self.context, self.targets)

    def test_empty_targets_and_statistics_distinguish_capacity_from_actual_changes(self):
        mask = fixed_support('ranking', 7)
        model = SupportMaskedGenerator(self.anchor, hard_support=mask).eval()
        empty = torch.empty(0, dtype=torch.long)
        result = model.inject(self.features, self.context, empty)
        self.assertTrue(torch.equal(result.features, self.features))
        self.assertEqual(model(self.features, self.context, empty).shape, (0, 30))
        stats = modification_statistics(self.features, result.features, empty, mask)
        self.assertEqual(stats['target_count'], 0)
        self.assertEqual(stats['support_capacity'], 7)
        for name in ('actual_changed_coordinates_per_target', 'normalized_l1_per_target',
                     'normalized_linf_per_target'):
            self.assertEqual(stats[name], [])
        identity = modification_statistics(self.features, self.features.clone(), self.targets, mask)
        self.assertEqual(identity['actual_changed_coordinates_per_target'], [0, 0, 0])
        modified = self.features.clone()
        modified[self.targets[0], LEGAL_COORDINATES[0]] += 2
        measured = modification_statistics(self.features, modified, self.targets, mask)
        self.assertEqual(measured['actual_changed_coordinates_per_target'], [1, 0, 0])
        self.assertEqual(measured['normalized_l1_per_target'], [2., 0., 0.])
        self.assertEqual(measured['normalized_linf_per_target'], [2., 0., 0.])
        modified[0, 0] += 1
        with self.assertRaises(ValueError):
            modification_statistics(self.features, modified, self.targets, mask)

    def test_invalid_anchor_masks_features_and_edge_weights_fail_closed(self):
        with self.assertRaises(ValueError):
            SupportMaskedGenerator(self.anchor)
        with self.assertRaises(ValueError):
            SupportMaskedGenerator(self.anchor, HardConcreteSupport(), hard_support=fixed_support('ranking', 7))
        for mask in (torch.ones(30, dtype=torch.bool), torch.ones(30), torch.zeros(29, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                SupportMaskedGenerator(self.anchor, hard_support=mask)
        changed = deepcopy(self.anchor)
        changed.attention_mode = 'learned'
        with self.assertRaises(ValueError):
            SupportMaskedGenerator(changed, HardConcreteSupport())
        model = SupportMaskedGenerator(self.anchor, hard_support=fixed_support('ranking', 7))
        with self.assertRaises(ValueError):
            model(self.features.clone().requires_grad_(), self.context, self.targets)
        with self.assertRaises(ValueError):
            model.inject(self.features, self.context, self.targets, detach=1)
        for weights in (torch.zeros(self.context.edge_index.shape[1]),
                        torch.ones(self.context.edge_index.shape[1], dtype=torch.long),
                        torch.ones(self.context.edge_index.shape[1] + 1)):
            with self.assertRaises(ValueError):
                model.inject(self.features, self.context, self.targets, weights)


if __name__ == '__main__':
    unittest.main()
