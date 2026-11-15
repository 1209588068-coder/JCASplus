"""Synthetic-only comparison evidence/queue contracts; never loads graph assets."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from comparison_methods import evidence as e
from comparison_methods import runner as r
from spear_current.runtime import sha256


class Scorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.))

    def forward(self, x, edges, weights=None):
        return torch.stack((-x[:, 0], x[:, 0]), 1).log_softmax(1)


class Trigger(torch.nn.Module):
    def __init__(self, structural=False):
        super().__init__()
        self.register_buffer('signal', torch.tensor(5.))
        self.structural = structural

    def inject(self, x, edges, targets):
        fx, fe = x.clone(), edges.clone()
        if self.structural:
            start = len(x)
            fx = torch.cat([fx, x.new_zeros((len(targets) * 3, x.shape[1]))])
            extra = []
            for i, t in enumerate(targets.tolist()):
                extra.extend([(t, start + i * 3), (start + i * 3, t)])
            fe = torch.cat([fe, torch.tensor(extra).T], 1)
        else:
            fx[targets, 0] = self.signal
        return fx, fe, x.new_ones(fe.shape[1])


def scene():
    x = np.zeros((4, 784), np.float32)
    x[:, 0] = [-1, 1, -2, 2]
    return dict(scenario_id='scene_a', x=x, edge_index=np.array([[0, 1, 2, 3], [1, 0, 3, 2]]),
        y=np.array([0, 1, 0, 1]), eligible=np.ones(4, dtype=bool),
        canonical_pair_track_ids=np.array([['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]),
        motion_regime=np.array([0, 1, 1, 0]))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feature_and_structure_permissions(self):
        p = scene()
        x, edges = torch.from_numpy(p['x']), torch.from_numpy(p['edge_index'])
        target = torch.tensor([0])
        for structural in (False, True):
            out = Trigger(structural).inject(x, edges, target)
            e.validate_injection(x, edges, target, out, structural=structural)
            bad = (out[0], out[1], out[2] * 0)
            with self.assertRaisesRegex(ValueError, 'zero-weight'):
                e.validate_injection(x, edges, target, bad, structural=structural)
        out = list(Trigger(True).inject(x, edges, target))
        out[1][1, -2] = 2
        with self.assertRaises(ValueError):
            e.validate_injection(x, edges, target, out, structural=True)
        out = list(Trigger(False).inject(x, edges, target))
        out[0][1, 0] = 8
        with self.assertRaisesRegex(ValueError, 'non-target'):
            e.validate_injection(x, edges, target, out, structural=False)

    def test_evaluation_denominators_and_csv_recompute(self):
        source = dict(scenario_id='scene_a', split_group_id='recording_a')
        for structural, method in [(False, 'spear'), (True, 'ugba')]:
            with tempfile.TemporaryDirectory() as tmp:
                result = e.evaluate_dev([scene()], [source], Trigger(structural),
                    dict(victim=Scorer(), clean_reference=Scorer()), Path(tmp) / 'dev',
                    method=method, config={})
                self.assertEqual(result['targets'], 1)
                self.assertEqual(result['normal_pairs'], 4)
                self.assertEqual(result['triggered_pair_rows'], 4)
                self.assertFalse(result['synthetic_nodes_in_metric_denominators'])
                self.assertFalse(result['synthetic_nodes_supervised'])
                self.assertEqual(result['metrics']['victim']['targets']['all']['incremental_asr'],
                                 0 if structural else 1)
                self.assertFalse(result['clean_reference_is_blind_confirmation'])

    def test_fresh_classifier_matches_initialization_and_ignores_synthetic_labels(self):
        p = scene()
        x, y, edges = (torch.from_numpy(p[k]) for k in ('x', 'y', 'edge_index'))
        dev = dict(x=x, y=y, edge_index=edges, eligible=torch.ones(4, dtype=torch.bool))
        with tempfile.TemporaryDirectory() as tmp:
            clean, a = e.fit_classifier(x, edges, y, torch.arange(4), dev,
                seed=7, output=Path(tmp) / 'clean.csv', epochs=2)
            fx, fe, _ = Trigger(True).inject(x, edges, torch.tensor([0]))
            victim, b = e.fit_classifier(fx, fe, y, torch.arange(4), dev,
                seed=7, output=Path(tmp) / 'victim.csv', epochs=2)
            self.assertEqual(a['initialization_sha256'], b['initialization_sha256'])
            self.assertEqual(a['fit_rng_sha256'], b['fit_rng_sha256'])
            self.assertEqual(a['supervision_count'], 4)
            self.assertEqual(b['supervision_count'], 4)
            self.assertEqual(a['selected_dev_auc'], max(row['dev_auc'] for row in a['trace']))
            self.assertEqual(a['selected_epoch'], next(row['epoch'] for row in a['trace'] if row['dev_auc'] == a['selected_dev_auc']))

    def test_cgba_candidate_population_does_not_relabel_negatives(self):
        inputs = dict(combined={'train': dict(y=torch.tensor([0, 1, 1, 0, 0, 1]),
            eligible=torch.tensor([True, True, False, True, True, True]))},
            indices=dict(idx_train=torch.tensor([0, 1, 2]), idx_attach=torch.tensor([3]),
                         idx_unlabeled=torch.tensor([4, 5])))
        self.assertEqual(r.candidate_indices(inputs, 'cgba').tolist(), [1])
        self.assertEqual(r.candidate_indices(inputs, 'spear').tolist(), [3, 4])

    def test_spear_progress_keyword_adapter(self):
        class Module:
            @staticmethod
            def fit_spear(*args, **kwargs):
                kwargs['progress'](phase='selector', epoch=0)
                return 'ok'
        events = []
        with patch.object(r, 'method_module', return_value=Module):
            got = r.fit_method('spear', dict(x=None, edge_index=None, y=None),
                               None, None, 1, {}, events.append)
        self.assertEqual(got, 'ok')
        self.assertEqual(events, [dict(phase='selector', epoch=0)])

    def test_all_configs_are_json_roundtrippable(self):
        for name, config in r.configurations().items():
            self.assertEqual(config, json.loads(json.dumps(config)))
            self.assertEqual(config['poison_count'], 26)

    def test_method_fit_uses_common_validation_contract(self):
        # Exercise actual public method entrypoints on synthetic tensors only.
        # Native UGBA selector is separately covered; mock only its selection
        # here to make this interface test independent of arbitrary cluster IDs.
        torch.manual_seed(15)
        x = torch.randn(20, 784)
        edges = torch.tensor([list(range(19)) + list(range(1, 20)),
                              list(range(1, 20)) + list(range(19))])
        y = torch.arange(20) % 2
        train = dict(x=x, y=y, edge_index=edges, eligible=torch.ones(20, dtype=torch.bool))
        indices = torch.arange(10)
        for method in r.METHODS:
            cfg = r.configurations()[method]
            cfg.update(epochs=1, hidden=4, poison_count=2, select_pretrain_epochs=1)
            if method == 'spear':
                cfg.update(feature_count=2, feature_pretrain_epochs=1, sage_permutations=2,
                           sage_background=2, sage_batch_size=2, outer_count=2)
            elif method == 'cgba':
                cfg.update(feature_count=2)
            else:
                cfg.update(outer_count=2)
            candidates = torch.tensor([1, 3, 5, 7, 9]) if method == 'cgba' else torch.tensor([10, 12, 14, 16, 18])
            events = []
            if method == 'ugba':
                with patch('comparison_methods.subgraph._select', return_value=(
                        candidates[:2], dict(pretrain_steps=0, method='synthetic_interface_only'))):
                    result = r.fit_method(method, train, indices, candidates, 3, cfg, events.append)
            else:
                result = r.fit_method(method, train, indices, candidates, 3, cfg, events.append)
            e.validate_fit_result(result, train, indices, candidates, method=method, config=cfg)
            self.assertTrue(events)
            self.assertTrue(all(isinstance(event, dict) for event in events))
            self.assertTrue(torch.equal(train['y'], y))


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        e.write_json(self.path / 'plan.json', {'fixed': 1})
        self.binding = dict(directory=str(self.path), files={'plan.json': sha256(self.path / 'plan.json')},
                            process=dict(pid=123, start_ticks=4, boot_id='boot'))

    def finish(self):
        e.write_json(self.path / 'summary.json', {})
        e.write_json(self.path / 'completion.json', dict(status=r.PARENT_STATUS,
            completed=[dict(seed=s, arm=a) for s in r.SEEDS for a in ('dual', 'single_proxy1', 'disabled')],
            summary_sha256=sha256(self.path / 'summary.json'), original_validation_arrays_loaded=False,
            original_test_arrays_loaded=False, audit_arrays_loaded=False, confirmation_arrays_loaded=False))

    def test_must_wait_for_process_exit_even_with_completion(self):
        with patch.object(r, 'process_identity', return_value=self.binding['process']):
            self.assertEqual(r.predecessor_state(self.binding), 'WAITING_FOR_PREDECESSOR')
            self.finish()
            self.assertEqual(r.predecessor_state(self.binding), 'WAITING_FOR_PROCESS_EXIT')
        with patch.object(r, 'process_identity', return_value=None):
            self.assertEqual(r.predecessor_state(self.binding), 'READY')

    def test_exit_without_completion_does_not_start(self):
        with patch.object(r, 'process_identity', return_value=None):
            with self.assertRaisesRegex(ValueError, 'without successful'):
                r.predecessor_state(self.binding)

    def test_live_parent_can_be_midway_through_completion_write(self):
        (self.path / 'completion.json').write_text('{"status":')
        with patch.object(r, 'process_identity', return_value=self.binding['process']):
            self.assertEqual(r.predecessor_state(self.binding), 'WAITING_FOR_PROCESS_EXIT')
        with patch.object(r, 'process_identity', return_value=None):
            with self.assertRaises(ValueError):
                r.predecessor_state(self.binding)

    def test_pid_reuse_and_parent_failure(self):
        with patch.object(r, 'process_identity', return_value=dict(pid=123, start_ticks=5, boot_id='boot')):
            with self.assertRaises(ValueError):
                r.predecessor_state(self.binding)
        e.write_json(self.path / 'failure.json', {})
        with patch.object(r, 'process_identity', return_value=self.binding['process']):
            with self.assertRaisesRegex(ValueError, 'failed'):
                r.predecessor_state(self.binding)

    def test_completion_summary_tampering_is_rejected(self):
        self.finish()
        # Test-owned fixture, deliberately corrupt bytes.
        (self.path / 'summary.json').write_text('{"changed":true}')
        with patch.object(r, 'process_identity', return_value=None):
            with self.assertRaises(ValueError):
                r.predecessor_state(self.binding)

    def test_cancel_or_failed_plan_cannot_run_directly(self):
        for flag in ('CANCEL', 'cancelled.json', 'failure.json', 'completion.json'):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / flag).touch()
                with patch.object(r, 'validate_plan', return_value={}), patch.object(r, 'read_inputs') as reader:
                    with self.assertRaisesRegex(ValueError, 'cannot start'):
                        r.run(None, Path(tmp) / 'plan.json')
                    reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
