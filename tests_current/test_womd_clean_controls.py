"""Synthetic checks for clean-only loss normalization and reporting contracts."""
import copy
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from spear_current.womd_clean_controls import (
    class_weights,
    fit_arm,
    global_weighted_loss,
    make_model,
    normal_metrics,
    state_hash,
    status_record,
    wait_for_preflight,
)


class CleanControlLossTests(unittest.TestCase):
    def test_unequal_batches_match_full_weighted_gradient(self):
        """A global denominator must not turn unequal batches into equal weights."""
        torch.manual_seed(73)
        whole = torch.nn.Linear(4, 2).double()
        streamed = copy.deepcopy(whole)
        x = torch.randn(11, 4, dtype=torch.float64)
        y = torch.tensor([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
        weights = torch.tensor([0.23, 1.77], dtype=torch.float64)
        denominator = float(weights[y].sum())
        expected = F.nll_loss(F.log_softmax(whole(x), dim=1), y,
                              weight=weights, reduction='mean')
        expected.backward()
        actual = 0.0
        for lo, hi in ((0, 1), (1, 8), (8, 11)):
            loss = global_weighted_loss(
                F.log_softmax(streamed(x[lo:hi]), dim=1), y[lo:hi],
                weights, denominator)
            loss.backward()
            actual += float(loss.detach())
        self.assertAlmostEqual(actual, float(expected.detach()), places=12)
        for first, second in zip(whole.parameters(), streamed.parameters()):
            torch.testing.assert_close(first.grad, second.grad,
                                       rtol=1e-12, atol=1e-12)

    def test_none_beta_is_ordinary_loss(self):
        weights = class_weights([8, 3], beta=None).double()
        torch.testing.assert_close(weights, torch.ones(2, dtype=torch.float64))
        logits = torch.tensor([[1.2, -0.4], [-0.3, 2.1], [0.5, 0.5]],
                              dtype=torch.float64)
        labels = torch.tensor([0, 1, 0])
        actual = global_weighted_loss(logits.log_softmax(1), labels,
                                      weights, len(labels))
        expected = F.cross_entropy(logits, labels)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_equal_class_counts_have_equal_weights(self):
        torch.testing.assert_close(class_weights([37, 37], .9999), torch.ones(2))

    def test_registered_counts_effective_number_weights(self):
        actual = class_weights([10836, 678], .9999)
        expected = torch.tensor([0.18029679805148088, 1.819703201948519])
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(float(actual.sum()), 2.0, places=6)

    def test_invalid_beta_is_rejected(self):
        for beta in (-.5, 0., 1., 1.01, math.nan, math.inf, -math.inf):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                class_weights([8, 3], beta)

    def test_invalid_counts_are_rejected(self):
        for counts in ([0, 3], [-1, 3], [3], [1, 2, 3],
                       [math.nan, 3], [math.inf, 3]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                class_weights(counts, .9999)

    def test_invalid_global_denominator_is_rejected(self):
        p = torch.tensor([[.25, .75]]).log()
        y = torch.tensor([1])
        for denominator in (0., -1., math.nan, math.inf):
            with self.subTest(denominator=denominator), self.assertRaises(ValueError):
                global_weighted_loss(p, y, torch.ones(2), denominator)

    def test_same_model_seed_pairs_initialization(self):
        first = make_model(20364104, torch.device('cpu'))
        # Intervening random work must not change the next arm's initialization.
        torch.randn(117)
        second = make_model(20364104, torch.device('cpu'))
        self.assertEqual(state_hash(first), state_hash(second))
        self.assertEqual(
            state_hash(first),
            '68ab2d1ae9d2d7293232b306ff3b18f48cc173ff5a653de565427c6b2a8bde0b')


class CleanControlMetricTests(unittest.TestCase):
    def test_confusion_matrix_uses_class_specific_denominators(self):
        result = normal_metrics(
            [0, 0, 0, 1, 1], [.1, .6, .2, .8, .5], [0, 1, 0, 1, 0])
        self.assertEqual(result['pairs'], 5)
        self.assertEqual(result['positives'], 2)
        self.assertEqual(result['negatives'], 3)
        self.assertEqual(result['true_positive'], 1)
        self.assertEqual(result['false_positive'], 1)
        self.assertEqual(result['false_negative'], 1)
        self.assertEqual(result['true_negative'], 2)
        self.assertEqual(result['recall'], .5)
        self.assertAlmostEqual(result['fpr'], 1/3)
        self.assertAlmostEqual(result['auc'], 5/6)
        self.assertAlmostEqual(result['ap'], 5/6)
        self.assertAlmostEqual(result['brier'], (.01+.36+.04+.04+.25)/5)

    def test_empty_group_has_no_fabricated_scores(self):
        result = normal_metrics([], [], [])
        for field in ('auc', 'ap', 'recall', 'fpr', 'brier'):
            self.assertIsNone(result[field])
        self.assertEqual(result['pairs'], 0)

    def test_no_positive_group(self):
        result = normal_metrics([0, 0], [.1, .8], [0, 1])
        self.assertIsNone(result['auc'])
        self.assertIsNone(result['ap'])
        self.assertIsNone(result['recall'])
        self.assertEqual(result['fpr'], .5)

    def test_no_negative_group(self):
        result = normal_metrics([1, 1], [.1, .8], [0, 1])
        self.assertIsNone(result['auc'])
        self.assertIsNone(result['fpr'])
        self.assertEqual(result['ap'], 1.)
        self.assertEqual(result['recall'], .5)

    def test_invalid_metric_input_is_rejected(self):
        for y, p, pred in (([0], [.1, .2], [0]),
                           ([0], [np.nan], [0]),
                           ([0], [np.inf], [0]),
                           ([2], [.1], [0]),
                           ([0], [.1], [2]),
                           ([0], [-.1], [0]),
                           ([0], [1.1], [1])):
            with self.subTest(y=y, p=p, pred=pred), self.assertRaises(ValueError):
                normal_metrics(y, p, pred)


class CleanControlExecutionTests(unittest.TestCase):
    def test_resource_wait_only_calls_preflight_when_memory_is_available(self):
        gib = 1024**3
        events = []

        def emit(status, **details):
            events.append(dict(status=status, **details))

        receipt = dict(status='PASSED', finite_gradients=True)
        with mock.patch('torch.cuda.mem_get_info',
                        side_effect=[(gib, 8*gib), (3*gib, 8*gib)]) as memory, \
             mock.patch('spear_current.womd_clean_controls.preflight',
                        return_value=receipt) as check, \
             mock.patch('spear_current.womd_clean_controls.time.sleep') as sleep, \
             mock.patch('spear_current.womd_clean_controls.fit_arm') as training:
            actual = wait_for_preflight(dict(device='cuda:0'), emit)
        self.assertEqual(actual, receipt)
        self.assertEqual(memory.call_count, 2)
        check.assert_called_once_with(dict(device='cuda:0'))
        sleep.assert_called_once_with(30)
        training.assert_not_called()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['status'], 'WAITING_FOR_GPU_MEMORY')
        self.assertEqual(events[0]['free_cuda_bytes'], gib)
        self.assertEqual(events[0]['required_free_cuda_bytes'], 2*gib)
        self.assertFalse(events[0]['training_started'])

    def test_resource_wait_does_not_swallow_unrelated_runtime_errors(self):
        gib = 1024**3
        for origin in ('memory_query', 'preflight'):
            with self.subTest(origin=origin):
                error = RuntimeError('unsupported sparse operation')
                memory_effect = error if origin == 'memory_query' else None
                check_effect = error if origin == 'preflight' else None
                with mock.patch('torch.cuda.mem_get_info',
                                side_effect=memory_effect, return_value=(3*gib, 8*gib)), \
                     mock.patch('spear_current.womd_clean_controls.preflight',
                                side_effect=check_effect) as check, \
                     mock.patch('spear_current.womd_clean_controls.time.sleep') as sleep, \
                     mock.patch('spear_current.womd_clean_controls.fit_arm') as training:
                    with self.assertRaisesRegex(RuntimeError, 'unsupported sparse operation'):
                        wait_for_preflight(dict(device='cuda:0'), mock.Mock())
                sleep.assert_not_called()
                training.assert_not_called()
                self.assertEqual(check.call_count, int(origin == 'preflight'))

    def test_preflight_out_of_memory_retries_resource_check_only(self):
        gib = 1024**3
        emit = mock.Mock()
        with mock.patch('torch.cuda.mem_get_info', return_value=(3*gib, 8*gib)), \
             mock.patch('spear_current.womd_clean_controls.preflight',
                        side_effect=[RuntimeError('CUDA error: out of memory'),
                                     {'status': 'PASSED'}]) as check, \
             mock.patch('spear_current.womd_clean_controls.time.sleep') as sleep, \
             mock.patch('spear_current.womd_clean_controls.fit_arm') as training:
            result = wait_for_preflight(dict(device='cuda:0'), emit)
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual(check.call_count, 2)
        sleep.assert_called_once_with(30)
        training.assert_not_called()
        self.assertEqual(emit.call_args.args, ('WAITING_FOR_GPU_MEMORY',))
        self.assertFalse(emit.call_args.kwargs['training_started'])

    def test_status_separates_total_and_arm_elapsed_time(self):
        # The epoch row already has elapsed_seconds. Passing it through emit
        # must neither raise duplicate-key errors nor replace run elapsed time.
        with mock.patch('spear_current.womd_clean_controls.time.monotonic',
                        return_value=125.0):
            result = status_record('TRAINING', 100.0, 0, arm='A_ce_512',
                                   epoch=1, elapsed_seconds=7.25)
        self.assertEqual(result['elapsed_seconds'], 25.0)
        self.assertEqual(result['arm_elapsed_seconds'], 7.25)
        self.assertEqual(result['epoch'], 1)
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['status'], 'TRAINING')
        json.dumps(result, allow_nan=False)

    def test_two_epoch_fit_writes_checkpoints_logs_and_statuses(self):
        """Exercise the real fit, Adam, callbacks and checkpoint lifecycle.

        Only input loading and CUDA resource/RNG queries are replaced. All
        computation uses the same GraphSAGE on a six-node CPU synthetic graph.
        """
        rng = torch.Generator().manual_seed(812)
        labels = torch.tensor([0, 0, 1, 0, 1, 0])
        batch = dict(x=torch.randn(6, 784, generator=rng), y=labels,
                     eligible=torch.ones(6, dtype=torch.bool),
                     edge_index=torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                                              [1, 0, 2, 1, 4, 3, 5, 4]]),
                     global_start=0, global_stop=6)
        population = dict(y=labels, eligible=batch['eligible'])
        accesses, events = [], []

        def fake_batches(data_dir, role, *, max_nodes, device):
            accesses.append(role)
            self.assertEqual(torch.device(device).type, 'cpu')
            yield batch

        def fake_population(data_dir, role):
            accesses.append(role)
            return population

        started = time.monotonic()

        def emit(status, **details):
            event = status_record(status, started, 0, **details)
            json.dumps(event, allow_nan=False)
            events.append(event)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            indices = root/'indices.pt'
            torch.save(dict(idx_train=torch.arange(6)), indices)
            initial = state_hash(make_model(20364104, torch.device('cpu')))
            plan = dict(device='cpu', output_dir=str(root), model_seed=20364104,
                        expected_initial_state_sha256=initial, epochs=2,
                        max_nodes=16384)
            arm = dict(name='synthetic_ce', data_dir=str(root/'synthetic_data'),
                       indices_path=str(indices), beta=None, train_scenes=2)
            with mock.patch('spear_current.womd_clean_controls.iter_role_batches',
                            side_effect=fake_batches), \
                 mock.patch('spear_current.womd_clean_controls.load_role_arrays',
                            side_effect=fake_population), \
                 mock.patch('torch.cuda.reset_peak_memory_stats'), \
                 mock.patch('torch.cuda.max_memory_allocated', return_value=0), \
                 mock.patch('torch.cuda.empty_cache'), \
                 mock.patch('torch.cuda.get_rng_state',
                            side_effect=lambda device: torch.get_rng_state()):
                result = fit_arm(plan, arm, emit)

            folder = root/arm['name']
            rows = [json.loads(line) for line in
                    (folder/'epochs.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows], [1, 2])
            self.assertEqual([r['optimizer_steps'] for r in rows], [1, 2])
            self.assertTrue(all(math.isfinite(r['train_loss']) for r in rows))
            latest = torch.load(folder/'latest.pt', weights_only=True)
            self.assertEqual(latest['next_epoch'], 2)
            self.assertTrue(latest['optimizer_state']['state'])
            for state in latest['optimizer_state']['state'].values():
                self.assertEqual(float(state['step']), 2.)
            selected = torch.load(folder/'model.pt', weights_only=True)
            expected_epoch = int(np.argmax([r['dev_auc'] for r in rows]))
            self.assertEqual(selected['epoch'], expected_epoch)
            self.assertEqual(result['selected_epoch'], expected_epoch+1)
            summary = json.loads((folder/'training_summary.json').read_text())
            self.assertEqual(summary, result)
            self.assertEqual(result['status'], 'TRAINED_PENDING_EVALUATION')
            self.assertTrue(result['clean_only'])
            self.assertFalse(result['attack_training'])
            self.assertFalse(result['asr_evaluated'])
            self.assertEqual(result['initial_state_sha256'], initial)
            epoch_events = [event for event in events if 'optimizer_steps' in event]
            self.assertEqual(len(epoch_events), 2)
            self.assertEqual([e['optimizer_steps'] for e in epoch_events], [1, 2])
            self.assertTrue(all('arm_elapsed_seconds' in e for e in epoch_events))
            self.assertTrue(all(e['elapsed_seconds'] >= 0 for e in events))
            self.assertTrue(set(accesses) <= {'train', 'dev'})
            self.assertNotIn('audit', accesses)


if __name__ == '__main__':
    unittest.main()
