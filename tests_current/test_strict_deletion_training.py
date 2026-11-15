"""Mechanism checks for strict deletions; synthetic inputs only."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from spear_current.full_ablation_training import AblationGenerator, _cpu_state, _tensor_hash
from spear_current.models import GCN, GraphTrojanNet
from spear_current.strict_deletion_training import (
    STAGES, VARIANTS, StrictDeletionGenerator, UnboundedGeneratedRegularizer,
    fit_from_base, restore_base_prefix,
)
from spear_current.support_schema import SELECTED_COORDINATES, EXCLUDED_COORDINATES


SEED = 20260621


def fixture(device='cpu'):
    device = torch.device(device)
    with torch.random.fork_rng():
        torch.manual_seed(71)
        x = torch.randn(8, 784).clamp(-3., 3.).to(device)
    edge = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7],
                         [1, 0, 2, 1, 4, 3, 5, 4, 7, 6]], device=device)
    y = torch.tensor([0, 1, 0, 0, 0, 1, 0, 1], device=device)
    coordinates = torch.tensor(SELECTED_COORDINATES, device=device)
    support = torch.zeros(30, dtype=torch.bool, device=device)
    support[2:26] = True
    context = SimpleNamespace(edge_index=edge)
    indices = {key: torch.tensor(value, device=device) for key, value in
               dict(idx_train=[0, 1], idx_attach=[2], fixed_outer=[2, 3, 4]).items()}
    train = dict(x=x, y=y, edge_index=edge, context=context)
    return train, indices, coordinates, support


def write_prefix(path, train, indices, coordinates, support):
    # A tiny synthetic prefix stands in for a completed base stage. It is not
    # used as a real-run checkpoint and carries exact same-input provenance.
    with torch.random.fork_rng():
        torch.manual_seed(19)
        device = train['x'].device
        base = GraphTrojanNet(device, 80, 30).to(device)
        shadow = GCN(784, 80, 2, dropout=0., device=device).to(device)
        legal = ~torch.isin(coordinates, torch.tensor(EXCLUDED_COORDINATES, device=device))
        generator = AblationGenerator(base, shadow, coordinates, legal)
    input_tensors = dict(x=train['x'], y=train['y'], edge_index=train['edge_index'], **indices)
    value = dict(stage='base', seed=SEED, variant='full', epoch=0,
        generator=_cpu_state(generator), surrogate=_cpu_state(generator.base_shadow),
        stage_record=dict(epochs=1, generator_steps=1, surrogate_steps=1,
                          support_count=28, output_mode='identity', discriminator_steps=0),
        initialization=dict(input_sha256={k:_tensor_hash(v) for k,v in input_tensors.items()}),
        distribution=None)
    torch.save(value, path)
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def proxies(device):
    result = [GCN(784, 80, 2, dropout=0., device=device).to(device) for _ in range(2)]
    for model in result:
        model.requires_grad_(False).eval()
    return result


class CpuMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_no_residual_has_no_parameters_or_forward_in_removed_branch(self):
        train, indices, coordinates, support = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'prefix.pt'
            original, digest = write_prefix(path, train, indices, coordinates, support)
            generator, shadow, _ = restore_base_prefix(path, digest, seed=SEED,
                variant='no_residual', coordinates=coordinates, support=support,
                device='cpu', expected_base_epochs=1)
            self.assertIsNone(generator.residual)
            self.assertFalse(any(k.startswith('residual.') for k in generator.state_dict()))
            for phase in STAGES:
                generator.set_phase(phase)
                self.assertFalse(any(p.requires_grad for p in generator.parameters()))
                self.assertFalse(generator._use_residual)
            for key, value in generator.base_generator.state_dict().items():
                self.assertTrue(torch.equal(value, original['generator']['base_generator.'+key]))

    def test_rejects_wrong_prefix_hash_stage_and_prior_learned_residual(self):
        train, indices, coordinates, support = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'prefix.pt'
            original, digest = write_prefix(path, train, indices, coordinates, support)
            arguments = dict(seed=SEED, variant='full', coordinates=coordinates,
                             support=support, device='cpu', expected_base_epochs=1)
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                restore_base_prefix(path, '0'*64, **arguments)
            for field in ('stage', 'learned_residual'):
                altered = deepcopy(original)
                if field == 'stage':
                    altered['stage'] = 'distribution'
                else:
                    altered['generator']['residual.head.bias'][0] = 1.
                torch.save(altered, path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    restore_base_prefix(path, sha, **arguments)

    def test_unbounded_regularizer_keeps_D_and_fake_values_without_clipping(self):
        train, indices, coordinates, support = fixture()
        d = UnboundedGeneratedRegularizer('unconditional', train['x'], indices['fixed_outer'],
            coordinates[support], seed=SEED+1000)
        changed = train['x'].clone()
        changed[indices['fixed_outer'][:,None], coordinates[support]] = 25.
        changed.requires_grad_()
        real, fake = d.discriminator_inputs(changed)
        self.assertEqual(fake.shape, (3, 30))
        self.assertTrue(torch.equal(fake[:, :24], torch.full((3,24), 2.5)))
        self.assertEqual(d.discriminator_step(changed)['d_steps'], 1)
        loss = d.generator_loss(changed)
        gradient, = torch.autograd.grad(loss, changed)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)
        broken = changed.detach().clone()
        broken[indices['fixed_outer'][0], d._conditions[0]] += 1
        with self.assertRaisesRegex(ValueError, 'conditions'):
            d.discriminator_inputs(broken)

    def test_cpu_training_rejected(self):
        train, indices, coordinates, support = fixture()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'requires CUDA'):
                fit_from_base(train, indices, [], coordinates, support,
                    prefix_path='unused.pt', prefix_sha256='0'*64, seed=SEED,
                    variant='no_residual', output_dir=folder)
            self.assertFalse(list(Path(folder).iterdir()))


@unittest.skipUnless(torch.cuda.is_available(), 'Explicit synthetic CUDA checks require GPU')
class CudaMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_four_variants_continue_same_prefix_and_record_strict_deletions(self):
        train, indices, coordinates, support = fixture('cuda:0')
        clean_proxies = proxies(train['x'].device)
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder)/'prefix.pt'
            original, digest = write_prefix(prefix, train, indices, coordinates, support)
            results = {}
            for variant in VARIANTS:
                result = fit_from_base(train, indices, clean_proxies, coordinates, support,
                    prefix_path=prefix, prefix_sha256=digest, seed=SEED, variant=variant,
                    output_dir=Path(folder)/variant,
                    config=dict(expected_base_epochs=1, stage_epochs={s:1 for s in STAGES}))
                results[variant] = result
                self.assertEqual(result['surrogate_steps'], 4)
                self.assertEqual(result['discriminator_steps'], 1)
                self.assertEqual(result['initialization']['prefix']['sha256'], digest)
                self.assertTrue(result['initialization']['attack_checkpoint_loaded'])
                self.assertFalse(result['initialization']['fresh_attack_initialization'])
                self.assertEqual(result['discriminator_observed_count'], 24)
                for key, value in result['generator'].base_generator.state_dict().items():
                    self.assertTrue(torch.equal(value.cpu(), original['generator']['base_generator.'+key]))
                self.assertEqual(result['expected_support_count'], 28 if variant=='no_support_mask' else 24)
            removed = results['no_residual']
            self.assertEqual(removed['generator_steps'], 0)
            self.assertIsNone(removed['generator'].residual)
            self.assertTrue(all(row['active_generator']=='none' for row in removed['records']))
            self.assertTrue(all(not s['active_generator_parameters'] for s in removed['stages']))
            self.assertTrue(all(s['generator_update_opportunities']==0 for s in removed['stages']))
            self.assertEqual(results['full']['generator_steps'], 3)
            self.assertNotEqual(results['full']['stages'][0]['initial_state_sha256']['active_generator']['sha256'],
                                results['full']['stages'][0]['final_state_sha256']['active_generator']['sha256'])
            self.assertTrue(all(s['support_count']==28 for s in results['no_support_mask']['stages']))
            self.assertTrue(all(not bool(results[v]['generator'].hard_support[:2].any()) for v in VARIANTS))
            self.assertEqual(results['no_domain']['generator'].output_mode, 'identity')
            self.assertTrue(all(s['output_mode']=='identity' for s in results['no_domain']['stages']))

    def test_no_domain_retains_D_when_generator_outputs_above_ten(self):
        train, indices, coordinates, support = fixture('cuda:0')
        clean_proxies = proxies(train['x'].device)
        original_forward = StrictDeletionGenerator.forward

        def large_generated(generator, features, context, targets):
            return original_forward(generator, features, context, targets) + generator.hard_support * 25.

        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder)/'prefix.pt'
            _, digest = write_prefix(prefix, train, indices, coordinates, support)
            with patch.object(StrictDeletionGenerator, 'forward', large_generated):
                result = fit_from_base(train, indices, clean_proxies, coordinates, support,
                    prefix_path=prefix, prefix_sha256=digest, seed=SEED, variant='no_domain',
                    output_dir=Path(folder)/'fit',
                    config=dict(expected_base_epochs=1, stage_epochs={s:1 for s in STAGES}))
                injection = result['generator'].inject(train['x'], train['context'], indices['fixed_outer'])
                actual = injection.features[indices['fixed_outer'][:,None], coordinates[support]]
                self.assertGreater(float(actual.abs().max()), 10.)
            self.assertEqual(result['discriminator_steps'], 1)
            self.assertGreater(result['records'][-1]['distribution_loss'], 0.)


if __name__ == '__main__':
    unittest.main()
