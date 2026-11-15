"""Tiny synthetic-only checks; no AV2 graph/checkpoint loads or full training."""
from copy import deepcopy
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current.data import combine_parts
from spear_current.development_diagnostic import choose_target, row_metadata
from spear_current import full_ablation_evaluation as full


class ToyGenerator(torch.nn.Module):
    def __init__(self, value=1., output_mode='identity', corrupt=False):
        super().__init__()
        self.input_dim, self.output_mode, self.corrupt = 784, output_mode, corrupt
        self.register_buffer('selected_coordinates', torch.arange(30))
        self.register_buffer('hard_support', torch.arange(30) < 24)
        self.register_buffer('value', torch.tensor(value))
        self.calls = []

    def inject(self, x, context, targets, edge_weight=None, detach=True):
        self.calls.append(dict(x=x.detach().clone(), targets=targets.detach().clone(),
                               device=str(x.device)))
        changed = x.clone()
        changed[targets, 0] = self.value
        if self.corrupt:
            changed[targets, 70] += 1
        return SimpleNamespace(features=changed, edge_index=context.edge_index,
                               edge_weight=edge_weight)


class ToyClassifier(torch.nn.Module):
    def __init__(self, trigger_scale=1.):
        super().__init__()
        self.register_buffer('trigger_scale', torch.tensor(trigger_scale))
        self.calls = []

    def forward(self, x, edge_index, edge_weight=None):
        self.calls.append((str(x.device), x.dtype, self.training, torch.is_grad_enabled()))
        logit = x[:, 50] + self.trigger_scale * x[:, 0].sum() * x[:, 51]
        return torch.stack((torch.zeros_like(logit), logit), 1).log_softmax(1)


def scene(moving=False, no_target=False, prefix='synthetic-full'):
    pairs = np.asarray([[0, 1], [0, 2], [3, 4], [1, 2]])
    part = dict(scenario_id=prefix, x=np.zeros((4, 784), dtype=np.float32),
        y=np.asarray([0, 1, 0, 1], dtype=np.int64), eligible=np.ones(4, dtype=bool),
        # Omit one shared-vehicle neighbor to test the identity definition.
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        canonical_pairs=pairs, canonical_pair_track_ids=np.asarray([[str(a), str(b)] for a, b in pairs]),
        motion_regime=np.full(4, moving, dtype=bool))
    for index in range(1000):
        part['scenario_id'] = f'{prefix}-{int(moving)}-{index}'
        if choose_target(part) == 0:
            break
    part['x'][:, 50] = [-2, 2, -2, 2]
    part['x'][:, 51] = [4, -4, 4, 0]
    if no_target:
        part['scenario_id'] += '-no-target'
        part['y'][:] = 1
        part['x'][:, 50] = 2
    return part


def prepared(parts=None):
    parts = parts or [scene(), scene(moving=True), scene(no_target=True)]
    train_parts = [scene(prefix='synthetic-training')]
    result = dict(parts=dict(train=train_parts, dev=parts), combined={}, metadata={}, targets={})
    for role, group in result['parts'].items():
        combined = result['combined'][role] = combine_parts(group)
        result['metadata'][role] = []
        for part, offset in zip(group, combined['offsets']):
            source = dict(scenario_id=part['scenario_id'], split_group_id=part['scenario_id'] + '-group')
            result['metadata'][role].extend(row_metadata(part, source, offset, role, set(), set()))
        result['targets'][role] = [choose_target(p) for p in group]
    result['indices'] = dict(idx_train=torch.tensor([1, 3]), idx_attach=torch.tensor([0]),
                             candidates=torch.tensor([0, 2]))
    selection = dict(idx_train=torch.tensor([1, 3]), idx_attach=torch.tensor([0]), fixed_outer=torch.tensor([0, 2]))
    bundle = SimpleNamespace(config=dict(arguments=dict(hidden=80, dropout=.5, train_lr=.01, weight_decay=.0005)))
    return result, selection, bundle


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


class FullAblationEvaluationTests(unittest.TestCase):
    def test_cpu_fallback_is_rejected_before_work(self):
        with self.assertRaisesRegex(RuntimeError, 'CPU fallback'):
            full._cuda_device('cpu')

    def test_scope_and_domain_helpers_allow_unbounded_identity_only(self):
        x = torch.zeros(3, 784)
        edges = torch.tensor([[0, 1], [1, 0]])
        weights = torch.ones(2)
        targets = torch.tensor([0])
        generator = ToyGenerator(value=12.)
        injected = generator.inject(x, SimpleNamespace(edge_index=edges), targets, weights)
        support = full._check_injection(x, injected.features, targets, generator,
            edges, edges, weights, weights)
        self.assertEqual(len(support), 24)
        self.assertEqual(full._feature_row(x, injected.features, 0, support)['out_of_range_coordinates'], 1)
        generator.output_mode = 'domain_projection'
        with self.assertRaisesRegex(ValueError, 'escaped'):
            full._check_injection(x, injected.features, targets, generator, edges, edges, weights, weights)
        generator.output_mode = 'identity'
        injected.features[1, 0] = 1
        with self.assertRaisesRegex(ValueError, 'frozen support'):
            full._check_injection(x, injected.features, targets, generator, edges, edges, weights, weights)

    def test_final_support_must_be_exactly24(self):
        generator = ToyGenerator()
        generator.hard_support[:] = True
        with self.assertRaisesRegex(ValueError, 'support'):
            full._support(generator, torch.device('cpu'))

    def test_scene_edges_cannot_cross_components(self):
        inputs, _, _ = prepared()
        graph = inputs['combined']['dev']
        graph['edge_index'][0, 0] = 4
        with self.assertRaisesRegex(ValueError, 'cross scene'):
            full._role_graph(inputs, 'dev')

    def test_random_nodes_uses_current_attachments_with_fixed_clean_supervision(self):
        inputs, selection, _ = prepared()
        selection['idx_attach'] = torch.tensor([2])
        selection['fixed_outer'] = torch.tensor([2, 0])
        chosen, fit = full._selection(inputs, selection, inputs['combined']['train'])
        self.assertEqual(chosen['idx_attach'].tolist(), [2])
        self.assertEqual(fit.tolist(), [1, 3, 2])
        selection['idx_train'] = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, 'clean supervision'):
            full._selection(inputs, selection, inputs['combined']['train'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA-only numerical path')
    def test_dev_outputs_are_complete_gpu_and_identity_based_collateral(self):
        inputs, _, _ = prepared()
        generator = ToyGenerator()
        victim, clean = ToyClassifier(), ToyClassifier(trigger_scale=0.)
        with TemporaryDirectory() as temp:
            output = Path(temp)
            summary = full.evaluate_case(generator, victim, clean, inputs, output_dir=output)
            metric = summary['models']['victim']
            self.assertEqual(metric['targets']['all']['incremental_asr'], 1.)
            self.assertEqual(metric['targets']['moving']['targets'], 1)
            self.assertEqual(metric['clean_pair_auc'], 1.)
            self.assertEqual(metric['clean_pair_ap'], 1.)
            self.assertEqual(metric['collateral']['adjacent']['positive_pairs'], 4)
            self.assertEqual(metric['collateral']['adjacent']['positive_suppression'], .5)
            self.assertEqual(metric['collateral']['nonincident']['incremental_fp'], 1.)
            self.assertEqual(summary['models']['clean_reference']['targets']['all']['incremental_asr'], 0.)
            self.assertEqual(summary['dev_counts']['supervised_pairs'], 12)
            self.assertEqual(summary['dev_counts']['triggered_supervised_pair_rows'], 12)
            self.assertEqual(len(read_csv(output / 'targets.csv')), 2)
            triggered = read_csv(output / 'triggered_pair_scores.csv')
            self.assertEqual(sum(r['relation'] == 'no_target' for r in triggered), 4)
            self.assertTrue(all('moving' in r['motion_regime'] or 'slow' in r['motion_regime'] for r in triggered))
            self.assertTrue(summary['rng_unchanged'])
        self.assertEqual(len(generator.calls), 2)
        self.assertTrue(all(c['device'] == 'cuda:0' for c in generator.calls))
        self.assertTrue(all(c[0] == 'cuda:0' and c[1] == torch.float32 and not c[2] and not c[3]
                            for model in (victim, clean) for c in model.calls))
        self.assertTrue(all(not bool(c['x'][:, 0].any()) for c in generator.calls))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA-only numerical path')
    def test_identity_out_of_range_is_reported_without_clamping(self):
        inputs, _, _ = prepared([scene()])
        with TemporaryDirectory() as temp:
            summary = full.evaluate_case(ToyGenerator(value=12.), ToyClassifier(), ToyClassifier(),
                                          inputs, output_dir=temp)
            self.assertEqual(summary['feature_changes']['out_of_range_coordinates'], 1)
            self.assertEqual(summary['feature_changes']['delta_linf_max'], 12.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA-only numerical path')
    def test_tiny_fit_retains_same_seed_initialization_and_earliest_auc_tie(self):
        inputs, selection, bundle = prepared([scene()])
        records, models = [], []
        # Constant valid AUC observations make the checkpoint tie rule explicit.
        with TemporaryDirectory() as temp, patch.object(full, 'roc_auc_score', return_value=.5):
            for name in ('one', 'two'):
                model, record = full.fit_victim(ToyGenerator(), deepcopy(inputs), selection, bundle,
                    seed=123, output_dir=Path(temp) / name, epochs=2)
                models.append(model)
                records.append(record)
                self.assertEqual(record['selected_epoch'], 0)
                self.assertEqual(record['optimizer_steps'], 2)
                self.assertTrue(record['parameters_changed'])
                self.assertEqual(len(read_csv(Path(temp) / name / 'victim_dev_trace.csv')), 2)
                checkpoint = torch.load(Path(temp) / name / 'victim.pt', weights_only=True)
                self.assertEqual(checkpoint['selected_epoch'], 0)
                self.assertTrue(all(not p.requires_grad and p.device.type == 'cuda' for p in model.parameters()))
        self.assertEqual(records[0]['initial_state_sha256'], records[1]['initial_state_sha256'])
        self.assertEqual(records[0]['rng']['before_fit'], records[1]['rng']['before_fit'])
        self.assertEqual(records[0]['selected_state_sha256'], records[1]['selected_state_sha256'])


if __name__ == '__main__':
    unittest.main()
