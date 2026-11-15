"""Deletion permissions and delegation checks; no real graphs or model fitting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from spear_current import strict_deletion_evaluation as strict
from spear_current import full_ablation_evaluation as original
from spear_current.support_schema import EXCLUDED_COORDINATES, SELECTED_COORDINATES


class ToyGenerator(nn.Module):
    input_dim = 784

    def __init__(self, variant='full'):
        super().__init__()
        self.register_buffer('selected_coordinates', torch.tensor(SELECTED_COORDINATES))
        legal = ~torch.isin(self.selected_coordinates, torch.tensor(EXCLUDED_COORDINATES))
        support = legal.clone()
        if variant != 'no_support_mask':
            support[torch.nonzero(legal).flatten()[-4:]] = False
        self.register_buffer('hard_support', support)
        self.expected_support_count = 28 if variant == 'no_support_mask' else 24
        self.output_mode = 'identity' if variant == 'no_domain' else 'domain_projection'
        self.base_generator = nn.Linear(2, 2).requires_grad_(False)
        self.base_shadow = nn.Linear(2, 2).requires_grad_(False)
        self.residual = None if variant == 'no_residual' else nn.Linear(2, 2)
        if self.residual is not None:
            self.residual.mode = 'uniform'

    def inject(self, x, context, targets, edge_weight=None, detach=True):
        changed = x.clone()
        changed[targets[:, None], self.selected_coordinates[self.hard_support]] = 2.
        return SimpleNamespace(features=changed, edge_index=context.edge_index,
                               edge_weight=edge_weight)


class StrictEvaluationTests(unittest.TestCase):
    def test_four_declared_contracts_keep_their_actual_permissions(self):
        frozen = ToyGenerator().hard_support
        for variant in strict.VARIANTS:
            with self.subTest(variant=variant):
                contract = strict.validate_contract(ToyGenerator(variant), variant,
                                                    expected_support=frozen)
                self.assertEqual(contract['support_capacity'], 28 if variant == 'no_support_mask' else 24)
                self.assertEqual(contract['changes_write_budget'], variant == 'no_support_mask')
                self.assertFalse(contract['no_generator_or_no_attack_control'])

    def test_28_coordinates_cannot_be_hidden_under_a_24_coordinate_arm(self):
        generator = ToyGenerator('no_support_mask')
        with self.assertRaisesRegex(ValueError, 'capacity'):
            strict.validate_contract(generator, 'full')
        generator = ToyGenerator()
        wrong = generator.hard_support.clone()
        active = torch.nonzero(wrong).flatten()
        inactive = torch.nonzero(~wrong & ~torch.isin(generator.selected_coordinates,
                                                     torch.tensor(EXCLUDED_COORDINATES))).flatten()
        wrong[active[-1]], wrong[inactive[0]] = False, True
        with self.assertRaisesRegex(ValueError, 'also changed'):
            strict.validate_contract(generator, 'full', expected_support=wrong)

    def test_metadata_and_output_projection_stay_protected(self):
        generator = ToyGenerator()
        generator.hard_support[0], generator.hard_support[2] = True, False
        with self.assertRaisesRegex(ValueError, 'Focal'):
            strict.validate_contract(generator, 'full')
        generator = ToyGenerator('no_domain')
        generator.output_mode = 'domain_projection'
        with self.assertRaisesRegex(ValueError, 'projection'):
            strict.validate_contract(generator, 'no_domain')

    def test_no_residual_cannot_retain_a_module_or_train_the_base(self):
        generator = ToyGenerator('no_residual')
        generator.residual = nn.Identity()
        with self.assertRaisesRegex(ValueError, 'must not retain'):
            strict.validate_contract(generator, 'no_residual')
        generator.residual = None
        generator.base_generator.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'remain frozen'):
            strict.validate_contract(generator, 'no_residual')

    def test_actual_original_injection_guard_accepts_only_explicit_28(self):
        generator = ToyGenerator('no_support_mask')
        x = torch.zeros(3, 784)
        changed = x.clone()
        targets = torch.tensor([1])
        allowed = generator.selected_coordinates[generator.hard_support]
        changed[1, allowed] = 2.
        edges, weights = torch.tensor([[0, 1], [1, 0]]), torch.ones(2)
        with self.assertRaises(ValueError):
            original._check_injection(x, changed, targets, generator,
                                      edges, edges, weights, weights)
        actual = original._check_injection(x, changed, targets, generator,
            edges, edges, weights, weights, support_capacity=28)
        self.assertEqual(len(actual), 28)
        changed[1, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'Injection changed'):
            original._check_injection(x, changed, targets, generator,
                edges, edges, weights, weights, support_capacity=28)

    def test_domain_deletion_keeps_finite_and_scope_guards_without_clipping(self):
        generator = ToyGenerator('no_domain')
        x = torch.zeros(2, 784)
        changed = x.clone()
        changed[0, generator.selected_coordinates[generator.hard_support]] = 25.
        targets = torch.tensor([0])
        edges, weights = torch.empty((2, 0), dtype=torch.long), torch.empty(0)
        original._check_injection(x, changed, targets, generator,
                                  edges, edges, weights, weights, support_capacity=24)
        generator.output_mode = 'domain_projection'
        with self.assertRaisesRegex(ValueError, 'escaped'):
            original._check_injection(x, changed, targets, generator,
                                      edges, edges, weights, weights, support_capacity=24)
        generator.output_mode = 'identity'
        changed[0, generator.selected_coordinates[2]] = float('inf')
        with self.assertRaisesRegex(ValueError, 'Injection changed'):
            original._check_injection(x, changed, targets, generator,
                                      edges, edges, weights, weights, support_capacity=24)

    def test_wrappers_delegate_the_explicit_budget_and_write_contracts(self):
        generator = ToyGenerator('no_support_mask')
        bundle = SimpleNamespace(manifest={'hard_support': ToyGenerator().hard_support.tolist()})
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(original, 'fit_victim', return_value=('model', {'epochs': 2})) as fit:
                model, record = strict.fit_victim(generator, {}, {}, bundle,
                    variant='no_support_mask', seed=21, output_dir=folder, epochs=2)
                self.assertEqual(fit.call_args.kwargs['support_capacity'], 28)
                self.assertEqual(model, 'model')
                self.assertTrue(record['strict_deletion_contract']['changes_write_budget'])
            self.assertEqual(json.loads((Path(folder) / 'strict_victim_contract.json').read_text())
                             ['support_capacity'], 28)
            with patch.object(original, 'evaluate_case', return_value={'unchanged': True}) as evaluate:
                result = strict.evaluate_case(generator, object(), object(), {},
                    variant='no_support_mask', output_dir=folder)
                self.assertEqual(evaluate.call_args.kwargs['support_capacity'], 28)
                self.assertTrue(result['unchanged'])
            self.assertTrue((Path(folder) / 'strict_evaluation_contract.json').is_file())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA numerical path only')
    def test_28_coordinate_arm_reaches_original_cuda_fit_and_scoring(self):
        from tests_current.test_full_ablation_evaluation import prepared
        inputs, selection, bundle = prepared()
        generator = ToyGenerator('no_support_mask').requires_grad_(False)
        bundle.manifest = {'hard_support': ToyGenerator().hard_support.tolist()}
        with tempfile.TemporaryDirectory() as folder:
            victim, record = strict.fit_victim(generator, inputs, selection, bundle,
                variant='no_support_mask', seed=21, output_dir=Path(folder) / 'victim', epochs=2)
            self.assertEqual(record['optimizer_steps'], 2)
            self.assertEqual(record['strict_deletion_contract']['support_capacity'], 28)
            reference = type(victim)(784, 80, 2, dropout=.5, lr=.01,
                weight_decay=.0005, device=torch.device('cpu')).eval().requires_grad_(False)
            report = strict.evaluate_case(generator, victim, reference, inputs,
                variant='no_support_mask', output_dir=Path(folder) / 'evaluation')
            self.assertEqual(report['feature_changes']['support_coordinates'], 28)
            self.assertEqual(report['feature_changes']['maximum_changed_coordinates_per_target'], 28)
            self.assertEqual(report['dev_counts']['targets'], 2)
            self.assertEqual(report['models']['victim']['targets']['moving']['targets'], 1)
            self.assertTrue(report['rng_unchanged'])


if __name__ == '__main__':
    unittest.main()
