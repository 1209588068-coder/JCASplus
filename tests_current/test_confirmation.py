"""Synthetic confirmation orchestration tests; no real graphs or checkpoints."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import confirmation as current
from spear_current.evaluation import evaluate


class Model(torch.nn.Module):
    def forward(self, x, edges, weights):
        return torch.stack((-x[:, 0], x[:, 0]), dim=1).log_softmax(1)


class Attack:
    args = SimpleNamespace(target_class=1)
    device = 'cpu'

    def inject_trigger(self, indices, x, edges, weights, device):
        result = x.clone()
        result[indices, 0] = 2
        return result, edges, weights


def component():
    x = np.zeros((3, 784), dtype=np.float32)
    x[:, 0] = [-1, 1, -2]
    return dict(x=x, y=np.array([0, 1, 0]), eligible=np.ones(3, dtype=bool),
        canonical_pair_track_ids=np.array([['a', 'b'], ['a', 'c'], ['b', 'c']]),
        motion_regime=np.array([0, 1, 1]),
        edge_index=np.array([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=np.int64))


def quality_metrics():
    model = dict(targets={regime:dict(targets=2, incremental_asr=.85, absolute_asr=.9,
        conditional_flip=.87, clean_activation_rate=.05) for regime in ('all', 'moving', 'slow')},
        collateral=dict(adjacent_incremental_fp=.01, nonincident_incremental_fp=.001,
                        adjacent_positive_suppression=.005),
        clean_pair_auc=.9, clean_pair_positive_recall=.5,
        clean_prediction_distribution={'class_0': 4, 'class_1': 2})
    return dict(victim=deepcopy(model), clean_reference=deepcopy(model))


class ConfirmationTests(unittest.TestCase):
    def test_new_role_keeps_identical_prediction_and_target_semantics(self):
        row = dict(scenario_id='s1', split_group_id='g1', split='train')
        models = dict(victim=Model(), clean_reference=Model())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            outputs = []
            for role in ('audit', 'confirmation'):
                states = []
                plan = dict(seed=20260621, roles={role:[row], 'train':[
                    dict(scenario_id='train1', split_group_id='train-group', split='train')]})
                out = evaluate(plan, folder / role, models, Attack(), lambda _:component(),
                               lambda **value:states.append(value), role=role)
                self.assertEqual(states[0]['stage'], role + '_original_TRAIN_groups')
                outputs.append(out)
            self.assertEqual(outputs[0]['metrics'], outputs[1]['metrics'])
            self.assertEqual(outputs[0]['audit_counts'], outputs[1]['audit_counts'])
            for name in ('targets.csv', 'clean_pair_scores.csv', 'triggered_pair_scores.csv'):
                self.assertEqual((folder/'audit'/name).read_bytes(), (folder/'confirmation'/name).read_bytes())

    def test_role_overlap_and_nontrain_rejected_before_read(self):
        for split, duplicate in [('val',False), ('test',False), ('train',True)]:
            row = dict(scenario_id='s', split_group_id='g', split=split)
            plan = dict(seed=1, roles={'confirmation':[row], 'train':[
                dict(scenario_id='t', split_group_id='g' if duplicate else 'h', split='train')]})
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    evaluate(plan, directory, dict(victim=Model(), clean_reference=Model()), Attack(),
                             lambda _:self.fail('Reader must not be called'), lambda **_:None, role='confirmation')

    def test_gate_does_not_hide_one_moving_failure(self):
        metrics = quality_metrics()
        self.assertTrue(current.checks_for(metrics)['passed'])
        metrics['victim']['targets']['moving']['incremental_asr'] = .7833
        result = current.checks_for(metrics)
        self.assertFalse(result['passed'])
        self.assertEqual(result['failed_checks'], ['moving_incremental_asr_at_least_80pct'])

    def test_quality_reference_and_missing_metrics(self):
        metrics = quality_metrics()
        metrics['victim']['clean_pair_auc'] = .89
        self.assertIn('same_pool_clean_reference_auc_drop_at_most_0p005',
                      current.checks_for(metrics)['failed_checks'])
        metrics['victim']['targets']['moving']['incremental_asr'] = None
        metrics['victim']['collateral']['adjacent_incremental_fp'] = .05
        result = current.checks_for(metrics)
        self.assertFalse(result['checks']['moving_incremental_asr_at_least_80pct'])
        self.assertFalse(result['checks']['adjacent_incremental_fp_below_5pct'])
        self.assertIsNone(current.statistics([None, .8, .9])['mean'])

    def test_aggregate_keeps_all_seeds_and_sample_sd(self):
        reports = {str(seed):dict(metrics=quality_metrics(), audit_counts={'targets':2},
                    common_prediction_hashes={'targets.csv':'same'}, gate={'passed':True})
                   for seed in current.SEEDS}
        for index, value in enumerate((.8, .9, 1.)):
            reports[str(current.SEEDS[index])]['metrics']['victim']['targets']['all']['incremental_asr'] = value
        reports[str(current.SEEDS[1])]['gate']['passed'] = False
        result = current.aggregate(reports)
        self.assertFalse(result['all_seeds_pass'])
        self.assertAlmostEqual(result['metrics']['victim']['all']['incremental_asr']['mean'], .9)
        self.assertAlmostEqual(result['metrics']['victim']['all']['incremental_asr']['sample_sd'], .1)
        reports.pop(str(current.SEEDS[1]))
        with self.assertRaises(ValueError):
            current.aggregate(reports)

    def test_common_identity_digest_ignores_victim_not_population(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            def put(scene='s', victim='0.1', clean='0.2', trigger='0.3'):
                for name in ('targets.csv', 'clean_pair_scores.csv'):
                    with (folder/name).open('w') as stream:
                        row = dict(scenario_id=scene, victim_p=victim, clean_reference_p=clean)
                        if name == 'targets.csv': row['clean_reference_trigger_p'] = trigger
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader(); writer.writerow(row)
            put(); reference = current.common_prediction_hashes(folder)
            put(victim='0.9', trigger='0.7')
            self.assertEqual(reference, current.common_prediction_hashes(folder))
            put(scene='wrong')
            self.assertNotEqual(reference, current.common_prediction_hashes(folder))
            put(clean='0.9')
            self.assertNotEqual(reference, current.common_prediction_hashes(folder))

    def test_output_boundary_and_state_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(current, 'RESULT_ROOT', folder):
                self.assertEqual(current.check_output(folder/'new', new=True), folder/'new')
                for candidate in (folder, folder.parent/'escape'):
                    with self.assertRaises(ValueError): current.check_output(candidate, new=True)
                (folder/'exists').mkdir()
                with self.assertRaises(FileExistsError): current.check_output(folder/'exists', new=True)
            current.update_state(folder, dict(at='fixed', status='COMPLETE'))
            self.assertEqual(json.loads((folder/'state.json').read_text())['at'], 'fixed')

    def test_initialization_failure_after_claim_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'plan.json').write_text('{}')
            with patch.object(current, 'load_plan', return_value=(folder, {'protocol':{}}, object())), \
                    patch.object(current, 'load_normalizer', side_effect=ValueError('mock failed read')):
                with self.assertRaisesRegex(ValueError, 'mock failed read'):
                    current.run(folder/'plan.json')
            self.assertTrue((folder/'execution_claim.json').is_file())
            self.assertTrue((folder/'failure.json').is_file())
            state = json.loads((folder/'state.json').read_text())
            self.assertEqual(state['status'], 'CONFIRMATION_FAILED_INTEGRITY_OR_RUNTIME')
            self.assertFalse(state['automatic_retry'])


if __name__ == '__main__':
    unittest.main()
