"""Self-contained training-entry contracts; no AV2 reads or real model fitting."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from spear_current import train
from spear_current.attention_context import attention_context
from spear_current.attention_generator import ContentAttentionGenerator
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.role_context import build_role_context
from spear_current.role_generator import PairRoleResidualGenerator
from spear_current.support_generator import fixed_support
from spear_current.support_schema import SELECTED_COORDINATES


class Shadow(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 4)

    def get_h(self, x, edges):
        return self.linear(x)


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 30)

    def forward(self, x, threshold):
        return self.linear(x) + threshold


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 2)

    def forward(self, x, edges, weights):
        return self.linear(x).log_softmax(1)


def fixture():
    with torch.random.fork_rng():
        torch.manual_seed(19)
        pairs = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [0, 3], [2, 3]])
        edges = torch.tensor([(i, j) for i in range(6) for j in range(6)
            if i != j and len(set(pairs[i].tolist()) & set(pairs[j].tolist())) == 1]).T
        context = attention_context(build_role_context(pairs, edges), ['synthetic'], seed=17)
        x = torch.arange(6 * 784, dtype=torch.float32).reshape(6, 784) / 10000
        role = PairRoleResidualGenerator(Base(), Shadow(), torch.tensor(SELECTED_COORDINATES),
            input_dim=784, hidden_dim=4, mode='untyped', threshold=.37)
        with torch.no_grad():
            role.residual.relation_weight.fill_(.01)
            role.residual.self_projection.weight.fill_(.02)
            role.residual.self_projection.bias.fill_(.3)
            role.residual.head.weight.fill_(.1)
            role.residual.head.bias.fill_(.05)
        anchor = ContentAttentionGenerator(role, attention_mode='uniform', score_dim=4)
        mask = fixed_support('random_semantic', 24, seed=20260621)
        reference = DomainConstrainedGenerator(anchor, mask, output_mode='domain_projection')
        reference.eval().requires_grad_(False)
        surrogate = Classifier().eval().requires_grad_(False)
        proxies = [Classifier().eval().requires_grad_(False) for _ in range(2)]
    inputs = dict(train=dict(x=x, y=torch.tensor([0, 1, 0, 0, 1, 0])), context=context,
        indices=dict(idx_train=torch.tensor([0, 1]), idx_attach=torch.tensor([2])),
        outer=torch.tensor([2, 3, 4]))
    return reference, surrogate, proxies, inputs


class TrainingEntryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_seeds_preserve_original_data_seed_and_registered_budgets(self):
        for seed in train.SEEDS:
            self.assertEqual(train.training_seeds(seed), dict(
                continuation=seed + 3000, discriminator=seed + 1000, victim=seed))
        for seed in (True, 20260624, 20260621., None):
            with self.assertRaises(ValueError):
                train.training_seeds(seed)
        self.assertEqual(train.TRAINING_CONFIG['epochs'], 200)
        self.assertEqual(train.TRAINING_CONFIG['inner_steps'], 1)

    def test_seed_adapter_historical_parity_and_rng_isolation(self):
        reference, _, _, inputs = fixture()
        x, outer = inputs['train']['x'], inputs['outer']
        coordinates = reference.selected_coordinates[reference.hard_support]
        rng = torch.get_rng_state().clone()
        original = train.DistributionRegularizer('unconditional', x, outer, coordinates)
        adapted = train.SeededDistributionRegularizer('unconditional', x, outer, coordinates)
        self.assertEqual(original.discriminator_hash(), adapted.discriminator_hash())
        for seed in train.SEEDS[1:]:
            other = train.SeededDistributionRegularizer('unconditional', x, outer, coordinates,
                                                        seed=seed + 1000)
            self.assertNotEqual(other.discriminator_hash(), original.discriminator_hash())
            self.assertEqual(other.steps, 0)
            self.assertFalse(other.optimizer.state)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_no_update_preflight_preserves_frozen_inputs_models_rng(self):
        reference, surrogate, proxies, inputs = fixture()
        generator = train.new_generator(reference)
        before = train._state_hash(reference)
        x = inputs['train']['x'].clone()
        with patch.object(torch.optim.Adam, 'step') as step:
            result = train.no_update_preflight(generator, reference, surrogate, proxies,
                                               inputs, 20260621)
            step.assert_not_called()
        self.assertEqual(result['status'], 'NO_UPDATE_PREFLIGHT_PASS')
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertTrue(result['exact_projected_replay'])
        self.assertEqual(train._state_hash(generator), before)
        self.assertEqual(train._state_hash(reference), before)
        self.assertTrue(torch.equal(x, inputs['train']['x']))
        self.assertTrue(all(value > 0 for value in result['residual_gradient_l2'].values()))

    def test_tensor_entry_rejects_changed_attachment_budget_before_updates(self):
        reference, surrogate, proxies, inputs = fixture()
        with patch.object(train, 'fit_distribution_residual') as fit:
            with self.assertRaisesRegex(ValueError, '26 attachments'):
                train.fit_unconditional(train.new_generator(reference), surrogate, proxies, inputs)
            fit.assert_not_called()

    def test_auc_earliest_tie_from_existing_forwards_and_restores_forward(self):
        class Scores(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.eye(2))

            def forward(self):
                return self.weight.log_softmax(1)

        model = Scores()
        truth = torch.tensor([0, 1])
        trace = train.AUCTrace(truth, torch.arange(2), epochs=3)
        original = model.forward
        rng = torch.get_rng_state().clone()
        accuracy_state = None
        with trace.observe(model):
            for epoch in range(3):
                # Distinct weights but the same perfect AUC: earliest wins.
                with torch.no_grad():
                    model.weight.copy_(torch.eye(2) * (epoch + 1))
                model.train()
                model()
                model.eval()
                model()
                if epoch == 0:
                    accuracy_state = deepcopy(model.state_dict())
            model.load_state_dict(accuracy_state)
        self.assertEqual(model.forward, original)
        result = trace.finish(model)
        self.assertEqual(result['selected_auc_epoch'], 0)
        self.assertEqual(result['training_forwards'], 3)
        self.assertEqual(result['eval_forwards'], 3)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(trace.auc_state['weight'], torch.eye(2)))

    def test_auc_forward_override_is_restored_after_failure(self):
        model = nn.Linear(2, 2)
        original = model.forward
        trace = train.AUCTrace(torch.tensor([0, 1]), torch.arange(2), epochs=1)
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with trace.observe(model):
                raise RuntimeError('stop')
        self.assertEqual(model.forward, original)

    def test_default_cli_checks_assets_without_loading_graphs_or_training(self):
        bundle = SimpleNamespace(config=dict(roles=dict(train=[None] * 512, dev=[None] * 128)),
                                 asset_path=lambda key: Path(key))
        with patch('spear_current.runtime.load_bundle', return_value=bundle), \
                patch.object(train, 'validate_training_contract'), \
                patch.object(train, 'run') as run, redirect_stdout(io.StringIO()):
            result = train.main([])
        run.assert_not_called()
        self.assertFalse(result['training_started'])
        self.assertFalse(result['graph_arrays_loaded'])

    def test_original_outer_reconstruction_rejects_resampling_and_supervision_edits(self):
        indices = dict(idx_train=torch.arange(1876), idx_attach=torch.arange(1876, 1902),
            idx_unlabeled=torch.arange(1902, 2500), train_labels=torch.zeros(2500, dtype=torch.long))
        values = dict(y=indices['train_labels'].clone(), eligible=torch.ones(2500, dtype=torch.bool))
        draw = indices['idx_unlabeled'][np.random.RandomState(20260621).choice(598, 512, replace=False)]
        outer = torch.cat([indices['idx_attach'], draw])
        coordinates = torch.tensor(SELECTED_COORDINATES)
        expected = dict(outer_indices=outer.tolist(), selected_coordinates=coordinates.tolist(),
                        attachment_indices=indices['idx_attach'].tolist())
        self.assertTrue(torch.equal(train.check_outer(values, indices, expected, coordinates), outer))
        edited = dict(expected, outer_indices=torch.cat([indices['idx_attach'], draw.flip(0)]).tolist())
        with self.assertRaisesRegex(ValueError, 'outer draw'):
            train.check_outer(values, indices, edited, coordinates)
        values['eligible'][0] = False
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            train.check_outer(values, indices, expected, coordinates)

    def test_run_refuses_existing_or_frozen_output_before_graph_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = SimpleNamespace(manifest_path=root / 'artifacts' / 'manifest.json')
            with patch.object(train, 'OUTPUT_ROOT', root / 'results'), \
                    patch.object(train, 'restore_training_inputs') as restore:
                (root / 'results' / 'existing').mkdir(parents=True)
                for output in (root, root / 'results', root / 'results' / 'existing',
                               root / 'artifacts' / 'new', root / 'data' / 'new',
                               root / 'graphs' / 'new', root / 'spear_current' / 'new'):
                    with self.assertRaisesRegex(ValueError, 'NEW output'):
                        train.run(bundle, output)
                restore.assert_not_called()

    def test_run_accepts_only_new_results_child_and_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'results'
            results.mkdir()
            (root / 'graphs').mkdir()
            (results / 'escape').symlink_to(root / 'graphs', target_is_directory=True)
            bundle = SimpleNamespace(manifest_path=root / 'artifacts' / 'manifest.json')
            with patch.object(train, 'OUTPUT_ROOT', results), \
                    patch.object(train, 'validate_training_contract'), \
                    patch.object(train, 'restore_training_inputs', side_effect=RuntimeError('read sentinel')) as restore:
                with self.assertRaisesRegex(ValueError, 'NEW output'):
                    train.run(bundle, results / 'escape' / 'new')
                restore.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'read sentinel'):
                    train.run(bundle, results / 'new_run')
                restore.assert_called_once_with(bundle)
                self.assertTrue((results / 'new_run' / 'started.json').is_file())

    def test_exclusive_checkpoint_write_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            train._save_checkpoint(path, {'x': torch.tensor([1.])})
            with self.assertRaises(FileExistsError):
                train._save_checkpoint(path, {'x': torch.tensor([2.])})
            self.assertEqual(torch.load(path, weights_only=True)['x'].item(), 1.)


if __name__ == '__main__':
    unittest.main()
