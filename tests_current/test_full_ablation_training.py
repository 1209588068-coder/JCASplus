"""Synthetic fresh-start ablation checks; no AV2 arrays or saved attack state."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from spear_current.full_ablation_training import (
    AblationGenerator, CONFIG_VARIANTS, STAGES, UniformResidual, fit_case,
)
from spear_current.models import GCN, GraphTrojanNet
from spear_current.role_generator import RoleResidualBlock
from spear_current.role_context import RoleContext
from spear_current.support_schema import SELECTED_COORDINATES


SEED = 20260621


def tensors(device='cpu'):
    with torch.random.fork_rng():
        torch.manual_seed(44)
        x = torch.randn(8, 784).clamp(-3, 3).to(device)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7],
                          [1, 0, 2, 1, 4, 3, 5, 4, 7, 6]], device=device)
    y = torch.tensor([0, 1, 0, 0, 0, 1, 0, 1], device=device)
    coordinates = torch.tensor(SELECTED_COORDINATES, device=device)
    support = torch.zeros(30, dtype=torch.bool, device=device)
    support[2:26] = True
    context = SimpleNamespace(edge_index=edges)
    return x, y, context, coordinates, support


def generator(mode='uniform', output='identity', device='cpu'):
    x, y, context, coordinates, support = tensors(device)
    with torch.random.fork_rng():
        torch.manual_seed(51)
        shadow = GCN(784, 80, 2, dropout=0., device=torch.device(device)).to(device)
        base = GraphTrojanNet(torch.device(device), 80, 30).to(device)
        result = AblationGenerator(base, shadow, coordinates, support,
                                   residual_mode=mode, output_mode=output)
    return result, x, context


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_uniform_matches_existing_block_and_zero_head_identity(self):
        torch.manual_seed(19)
        old = RoleResidualBlock(784, 80, 30, mode='untyped')
        new = UniformResidual()
        new.load_state_dict(old.state_dict())
        # Nonzero head checks the actual aggregation, beyond zero-head equality.
        with torch.no_grad():
            old.head.weight.normal_(); old.head.bias.normal_()
        new.load_state_dict(old.state_dict())
        x, _, context, _, _ = tensors()
        role = RoleContext(context.edge_index,
                           torch.zeros(context.edge_index.shape[1], dtype=torch.long),
                           torch.zeros(len(x), dtype=torch.long), len(x))
        self.assertTrue(torch.equal(new(x, context), old(x, role)))
        gen, x, context = generator()
        targets = torch.tensor([2, 3])
        with torch.no_grad():
            expected = gen.base_generator(gen.base_shadow.get_h(x, context.edge_index)[targets], .5)
            values = gen(x, context, targets)
        self.assertTrue(torch.equal(values[:, gen.hard_support], expected[:, gen.hard_support]))

    def test_self_only_is_independent_of_neighbor_values(self):
        gen, x, context = generator('self_only')
        with torch.no_grad():
            gen.residual.head.weight.normal_()
        altered = x.clone(); altered[1] += 20
        # Only the residual is claimed independent: base SPEAR has GCN context.
        first = gen.residual(x, context)[0]
        second = gen.residual(altered, context)[0]
        self.assertTrue(torch.equal(first, second))

    def test_projection_scope_and_gradient_reach_only_active_parameters(self):
        gen, x, context = generator(output='domain_projection')
        targets = torch.tensor([2, 3])
        with torch.no_grad():
            gen.base_generator.feat.bias.fill_(20.)
        injected = gen.inject(x, context, targets, detach=False)
        legal = torch.zeros_like(x, dtype=torch.bool)
        legal[targets[:, None], gen.selected_coordinates[gen.hard_support]] = True
        self.assertTrue(torch.equal(injected.features[~legal], x[~legal]))
        self.assertLessEqual(float(injected.features[legal].detach().abs().max()), 10.)
        self.assertTrue(torch.equal(injected.edge_index, context.edge_index))
        self.assertTrue(bool((injected.edge_weight == 1).all()))
        injected.features.sum().backward()
        self.assertTrue(all(p.grad is None for p in gen.base_generator.parameters()))
        self.assertTrue(all(p.grad is None for p in gen.base_shadow.parameters()))
        self.assertIsNotNone(gen.residual.head.weight.grad)

    def test_focal_metadata_rejected_and_base_only_has_no_residual(self):
        gen, x, context = generator('base_only')
        self.assertIsNone(gen.residual)
        self.assertTrue(any(p.requires_grad for p in gen.base_generator.parameters()))
        support = gen.hard_support.clone(); support[0] = True
        with self.assertRaisesRegex(ValueError, 'Focal metadata'):
            gen.set_support(support)
        gen.set_phase('support_surrogate')
        self.assertFalse(any(p.requires_grad for p in gen.parameters()))

    def test_cpu_fit_is_rejected_before_any_update(self):
        x, y, context, coordinates, support = tensors()
        indices = {key: torch.tensor(value) for key, value in dict(
            idx_train=[0, 1], idx_attach=[2], fixed_outer=[2, 3, 4]).items()}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'requires CUDA'):
                fit_case(dict(x=x, y=y, edge_index=context.edge_index), indices, [],
                         coordinates, support, seed=SEED, variant='full',
                         output_dir=directory, config={'epochs': 1})
            self.assertFalse(list(Path(directory).iterdir()))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA synthetic checks require GPU')
class CudaTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_no_distribution_no_domain_accepts_outputs_above_ten_without_constructing_D(self):
        device = torch.device('cuda:0')
        x, y, context, coordinates, support = tensors(device)
        indices = {key: torch.tensor(value, device=device) for key, value in dict(
            idx_train=[0, 1], idx_attach=[2], fixed_outer=[2, 3, 4]).items()}
        proxies = [GCN(784, 80, 2, dropout=0., device=device).to(device) for _ in range(2)]
        for model in proxies:
            model.requires_grad_(False).eval()
        original = UniformResidual.forward

        def large_values(residual, features, bound_context):
            return original(residual, features, bound_context) + 25.

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(UniformResidual, 'forward', large_values), patch(
                'spear_current.full_ablation_training.SeededDistributionRegularizer',
                side_effect=AssertionError('Disabled distribution must not construct or validate D')):
                result = fit_case(dict(x=x, y=y, edge_index=context.edge_index, context=context),
                    indices, proxies, coordinates, support, seed=SEED,
                    variant='no_distribution_no_domain', output_dir=directory, config={'epochs': 1})
                injected = result['generator'].inject(x, context, indices['fixed_outer'])
                generated = injected.features[indices['fixed_outer'][:, None], coordinates[support]]
                self.assertGreater(float(generated.abs().max()), 10.)
            self.assertEqual(result['discriminator_steps'], 0)
            self.assertTrue(all(row['distribution_loss'] == 0. and row['distribution_weight'] == 0.
                                for row in result['records']))
            self.assertTrue(all(row['discriminator_initial_sha256'] is None for row in result['stages']))

    def test_all_nine_variants_complete_registered_stages_from_fresh_same_seed(self):
        device = torch.device('cuda:0')
        x, y, context, coordinates, support = tensors(device)
        indices = {key: torch.tensor(value, device=device) for key, value in dict(
            idx_train=[0, 1], idx_attach=[2], fixed_outer=[2, 3, 4]).items()}
        proxies = [GCN(784, 80, 2, dropout=0., device=device).to(device) for _ in range(2)]
        for model in proxies:
            model.requires_grad_(False).eval()
        initializations = []
        with tempfile.TemporaryDirectory() as directory:
            for variant in CONFIG_VARIANTS:
                with self.subTest(variant=variant):
                    result = fit_case(dict(x=x, y=y, edge_index=context.edge_index, context=context),
                        indices, proxies, coordinates, support, seed=SEED, variant=variant,
                        output_dir=Path(directory) / variant,
                        config=dict(stage_epochs={stage: 1 for stage in STAGES}))
                    initializations.append(result['initialization'])
                    self.assertEqual(result['generator_steps'], 4)
                    self.assertEqual(result['surrogate_steps'], 5)
                    self.assertEqual([row['stage'] for row in result['records']], list(STAGES))
                    self.assertEqual([row['support_count'] for row in result['records']], [28, 28, 24, 24, 24])
                    self.assertEqual(result['stages'][2]['generator_steps'], 0)
                    self.assertEqual(result['stages'][2]['initial_state_sha256']['generator'],
                                     result['stages'][2]['final_state_sha256']['generator'])
                    self.assertTrue(Path(result['final_checkpoint']).is_file())
                    self.assertEqual(result['discriminator_steps'],
                        0 if variant in ('no_distribution', 'no_distribution_no_domain') else 1)
                    self.assertEqual(result['generator'].output_mode,
                        'identity' if variant == 'no_distribution_no_domain' else 'domain_projection')
                    if variant == 'no_kl':
                        self.assertTrue(all(row['clean_kl_loss'] == 0. for row in result['records']))
                    if variant == 'single_kl':
                        self.assertTrue(all(row['clean_kl_members'] == 1 for row in result['records']))
                    # Every active stage must actually update its own tensors.
                    for stage in result['stages']:
                        if stage['generator_steps']:
                            self.assertNotEqual(stage['initial_state_sha256']['active_generator']['sha256'],
                                                stage['final_state_sha256']['active_generator']['sha256'])
                        self.assertNotEqual(stage['initial_state_sha256']['surrogate'],
                                            stage['final_state_sha256']['surrogate'])
            self.assertTrue(all(row['fresh_model_sha256'] == initializations[0]['fresh_model_sha256']
                                for row in initializations))
            self.assertTrue(all(row['initialization_rng_sha256'] == initializations[0]['initialization_rng_sha256']
                                for row in initializations))


if __name__ == '__main__':
    unittest.main()
