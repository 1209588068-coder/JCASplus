"""Data/queue tests, not reimplementations of author algorithms."""
from contextlib import ExitStack, contextmanager
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import torch

from author_comparisons.data import assemble, validate_author_injection, validate_fit
from author_comparisons import runner


class AuthorDataTests(unittest.TestCase):
    def fixture(self):
        train = dict(x=torch.zeros(6, 784), y=torch.tensor([0, 1, -1, 0, 0, 1]),
                     eligible=torch.tensor([True, True, False, True, True, True]),
                     edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]))
        dev = dict(x=torch.ones(4, 784), y=torch.tensor([0, -1, 1, -1]),
                   eligible=torch.tensor([True, False, True, False]),
                   edge_index=torch.tensor([[0, 1], [1, 0]]))
        indices = dict(idx_train=torch.tensor([0, 1]), idx_attach=torch.tensor([3]),
                       idx_unlabeled=torch.tensor([4, 5]), dev_indices=torch.tensor([6, 8]))
        return train, dev, indices

    def test_disjoint_union_retains_invalid_context(self):
        train, dev, indices = self.fixture()
        out = assemble(train, dev, indices)
        self.assertEqual(out['x'].shape, (10, 784))
        self.assertEqual(out['y'].tolist().count(-1), 3)
        self.assertEqual(out['candidates'].tolist(), [3, 4])
        self.assertEqual(out['idx_val'].tolist(), [6, 8])
        self.assertTrue(torch.equal(out['edge_index'][:, -2:], dev['edge_index'] + 6))

    def test_reject_wrong_role_indices_and_invalid_supervision(self):
        for key, value in [('idx_unlabeled', torch.tensor([6])),
                           ('idx_train', torch.tensor([2])), ('dev_indices', torch.tensor([6, 7]))]:
            train, dev, indices = self.fixture()
            indices[key] = value
            with self.assertRaises(ValueError):
                assemble(train, dev, indices)

    def test_raw_author_zero_edges_are_checked_not_removed(self):
        x = torch.zeros(3, 5)
        edges = torch.tensor([[0, 1], [1, 0]])
        fx = torch.cat((x, torch.ones(3, 5)))
        fe = torch.cat((edges, torch.tensor([[0, 3, 3, 4], [3, 0, 4, 3]])), dim=1)
        weights = torch.tensor([1., 1., 1., 1., 0., 0.])
        original_edges, original_weights = fe.clone(), weights.clone()
        validate_author_injection(x, edges, torch.tensor([0]), (fx, fe, weights), structural=True)
        self.assertTrue(torch.equal(fe, original_edges))
        self.assertTrue(torch.equal(weights, original_weights))
        fe[0, -1] = 2
        with self.assertRaises(ValueError):
            validate_author_injection(x, edges, torch.tensor([0]), (fx, fe, weights), structural=True)

    def test_nonbinary_weights_are_not_silently_accepted(self):
        x = torch.zeros(2, 5)
        edges = torch.tensor([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            validate_author_injection(x, edges, torch.tensor([0]),
                (x.clone(), edges.clone(), torch.tensor([1., .5])), structural=False)

    def test_fit_validates_native_labels_not_synthetic_supervision(self):
        x = torch.zeros(30, 784)
        y = torch.zeros(30, dtype=torch.long)
        y[1] = 1
        y[-1] = -1
        inputs = dict(x=x, y=y, edge_index=torch.empty(2, 0, dtype=torch.long),
            idx_train=torch.tensor([0, 1]), candidates=torch.arange(2, 28))
        fitted = dict(attachment_indices=inputs['candidates'], poisoned_x=x.clone(),
            poisoned_y=y.clone(), poisoned_edge_index=inputs['edge_index'].clone(),
            poisoned_edge_weight=torch.empty(0), fit_indices=torch.arange(28))
        fitted['poisoned_y'][inputs['candidates']] = 1
        validate_fit(fitted, inputs, 'spear', dict(alpha_int=30))
        fitted['poisoned_y'][-1] = 0
        with self.assertRaises(ValueError):
            validate_fit(fitted, inputs, 'spear', dict(alpha_int=30))


class AuthorQueueTests(unittest.TestCase):
    @staticmethod
    def resources(parallel=2):
        return dict(max_parallel=parallel, exclusive_methods=['dpgba'],
                    hardware=dict(device='cuda:0', gpu_name='synthetic GPU', torch_version='test'),
                    minimum_host_available_bytes=8 * 1024 ** 3)

    @contextmanager
    def mocked_queue(self, directory, workers, sleep=None):
        """Real temporary bookkeeping, but no subprocess, signal, GPU or data IO."""
        path = Path(directory) / 'plan.json'
        path.write_text('{}')
        resources = self.resources()
        plan = dict(protocol=dict(predecessor=None, resources=resources))
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'validate_plan', return_value=plan))
            stack.enter_context(patch.object(runner, 'gpu_identity', return_value=resources['hardware']))
            stack.enter_context(patch.object(runner, 'check_host_memory'))
            stack.enter_context(patch.object(runner, 'process_identity', return_value='synthetic-parent'))
            stack.enter_context(patch.object(runner, 'SEEDS', (20260621,)))
            stack.enter_context(patch.object(runner.signal, 'signal'))
            stack.enter_context(patch.object(runner.time, 'sleep', side_effect=sleep))
            jobs = stack.enter_context(patch.object(runner.subprocess, 'Popen', side_effect=workers))
            reader = stack.enter_context(patch.object(runner, 'load_inputs'))
            yield path, jobs, reader

    def test_native_classifier_initializations_must_be_recorded(self):
        for field in ('initial_state_sha256', 'initialization_digest', 'initialization_sha256'):
            self.assertEqual(runner.initialization_hash({field: 'abc'}), 'abc')
        with self.assertRaises(ValueError):
            runner.initialization_hash({})

    def test_only_verified_task_compatible_methods_are_selected(self):
        self.assertEqual(runner.METHODS, ('spear', 'ugba', 'dpgba'))
        self.assertIn('cgba', runner.EXCLUDED)
        self.assertIn('gta', runner.EXCLUDED)
        for name in runner.EXCLUDED:
            with self.assertRaises(ValueError):
                runner.module(name)

    def test_terminal_queue_cannot_load_data(self):
        for flag in ('CANCEL', 'cancelled.json', 'failure.json', 'completion.json'):
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                (output / flag).touch()
                with patch.object(runner, 'validate_plan', return_value={}), patch.object(runner, 'load_inputs') as reader:
                    with self.assertRaises(ValueError):
                        runner.run(None, output / 'plan.json')
                    reader.assert_not_called()

    def test_independent_queue_is_ready_without_reading_cpu_predecessor(self):
        with patch.object(runner, 'predecessor_state') as predecessor:
            self.assertEqual(runner.ready(None), 'READY')
            predecessor.assert_not_called()
            predecessor.return_value = 'WAITING_FOR_PREDECESSOR'
            self.assertEqual(runner.ready({'directory': 'synthetic'}), 'WAITING_FOR_PREDECESSOR')
            predecessor.assert_called_once_with({'directory': 'synthetic'})

    def test_gpu_parallel_limits_and_dpgba_exclusion(self):
        for limit in (1, 2):
            resources = self.resources(limit)
            for name in runner.METHODS:
                self.assertTrue(runner.can_start(name, [], resources))
                self.assertFalse(runner.can_start(name, ['dpgba'], resources))
                self.assertFalse(runner.can_start(name, ['spear'] * limit, resources))
            self.assertFalse(runner.can_start('dpgba', ['spear'], resources))
            self.assertFalse(runner.can_start('dpgba', ['ugba'], resources))
            self.assertEqual(runner.can_start('ugba', ['spear'], resources), limit == 2)
            self.assertEqual(runner.can_start('spear', ['ugba'], resources), limit == 2)

    def test_prepare_requires_resource_acceptance_before_source_or_data_access(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'not_created'
            with patch.object(runner, 'bind_resources') as resources, \
                    patch.object(runner, 'specification') as spec, \
                    patch.object(runner, 'load_inputs') as reader:
                with self.assertRaisesRegex(ValueError, 'benchmark required'):
                    runner.prepare(None, output)
                resources.assert_not_called()
                spec.assert_not_called()
                reader.assert_not_called()
                self.assertFalse(output.exists())

    def test_gpu_precision_flags_and_autocast_rejection_without_gpu_calls(self):
        resources = self.resources()
        with patch.object(runner, 'gpu_identity', return_value=resources['hardware']), \
                patch.object(runner, 'torch') as backend:
            backend.is_autocast_enabled.return_value = False
            runner.configure_gpu(resources)
            backend.cuda.set_device.assert_called_once_with(0)
            backend.set_default_dtype.assert_called_once_with(backend.float32)
            backend.set_float32_matmul_precision.assert_called_once_with('highest')
            self.assertIs(backend.backends.cuda.matmul.allow_tf32, False)
            self.assertIs(backend.backends.cudnn.allow_tf32, False)
            self.assertIs(backend.backends.cudnn.benchmark, False)
            backend.cuda.reset_peak_memory_stats.assert_called_once_with()
            backend.is_autocast_enabled.return_value = True
            with self.assertRaisesRegex(ValueError, 'autocast'):
                runner.configure_gpu(resources)

    def test_cancelled_waiter_never_starts_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / 'CANCEL').touch()
            with patch.object(runner, 'validate_plan', return_value={}), \
                    patch.object(runner, 'run') as run, \
                    patch.object(runner, 'load_inputs') as reader, \
                    patch.object(runner.subprocess, 'Popen') as jobs:
                runner.wait_and_run(None, output / 'plan.json')
                run.assert_not_called()
                reader.assert_not_called()
                jobs.assert_not_called()
            self.assertEqual(json.loads((output / 'cancelled.json').read_text())['status'],
                             'CANCELLED_BEFORE_TRAINING')

    def test_worker_failure_stops_only_its_owned_active_peer(self):
        failed, peer, unrelated = (MagicMock() for _ in range(3))
        failed.pid, peer.pid, unrelated.pid = 101, 102, 999
        failed.poll.return_value = 7
        peer.poll.return_value = None
        peer.wait.return_value = 0
        with tempfile.TemporaryDirectory() as temp, self.mocked_queue(temp, [failed, peer]) as (path, jobs, reader):
            with self.assertRaisesRegex(RuntimeError, 'failed with exit code 7'):
                runner.run(None, path)
            self.assertEqual(jobs.call_count, 2)  # DPGBA never starts after peer failure.
            peer.terminate.assert_called_once_with()
            peer.wait.assert_called_once()
            peer.kill.assert_not_called()
            failed.terminate.assert_not_called()
            unrelated.terminate.assert_not_called()
            unrelated.kill.assert_not_called()
            reader.assert_not_called()
            for call in jobs.call_args_list:
                self.assertTrue(call.kwargs['stdout'].closed)
            failure = json.loads((path.parent / 'failure.json').read_text())
            self.assertEqual(failure['status'], 'AUTHOR_QUEUE_FAILED_NO_FURTHER_JOBS')
            self.assertFalse((path.parent / 'completion.json').exists())

    def test_cancellation_during_run_stops_workers_and_launches_no_more(self):
        workers = [MagicMock(pid=101), MagicMock(pid=102)]
        for process in workers:
            process.poll.return_value = None
            process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as temp:
            def cancel(_seconds):
                (Path(temp) / 'CANCEL').touch()
            with self.mocked_queue(temp, workers, sleep=cancel) as (path, jobs, reader):
                with self.assertRaisesRegex(ValueError, 'Cancelled'):
                    runner.run(None, path)
                self.assertEqual(jobs.call_count, 2)
                for process in workers:
                    process.terminate.assert_called_once_with()
                    process.wait.assert_called_once()
                reader.assert_not_called()

    def test_memory_pressure_stops_owned_workers_without_more_launches(self):
        workers = [MagicMock(pid=101), MagicMock(pid=102)]
        for process in workers:
            process.poll.return_value = None
            process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as temp, self.mocked_queue(temp, workers) as (path, jobs, reader):
            with patch.object(runner, 'check_host_memory',
                              side_effect=[None, RuntimeError('synthetic memory pressure')]):
                with self.assertRaisesRegex(RuntimeError, 'synthetic memory pressure'):
                    runner.run(None, path)
            self.assertEqual(jobs.call_count, 2)
            for process in workers:
                process.terminate.assert_called_once_with()
            reader.assert_not_called()

    def test_gpu_enqueue_preserves_visibility_and_never_launches_cpu_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            path = output / 'plan.json'
            path.write_text('{}')
            resources = self.resources()
            plan = dict(protocol=dict(predecessor=None, resources=resources))
            child = MagicMock(pid=777)
            with patch.object(runner, 'validate_plan', return_value=plan), \
                    patch.object(runner, 'gpu_identity', return_value=resources['hardware']), \
                    patch.object(runner, 'process_identity', return_value='synthetic-worker'), \
                    patch.object(runner, 'predecessor_state') as predecessor, \
                    patch.object(runner.subprocess, 'Popen', return_value=child) as jobs, \
                    patch.object(runner, 'load_inputs') as reader, \
                    patch.dict(runner.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
                result = runner.enqueue(None, path)
                jobs.assert_called_once()
                predecessor.assert_not_called()
                reader.assert_not_called()
            call = jobs.call_args
            self.assertIn('--wait-and-run', call.args[0])
            self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(call.kwargs['env']['OMP_NUM_THREADS'], '1')
            self.assertEqual(call.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')
            self.assertTrue(call.kwargs['start_new_session'])
            self.assertTrue(call.kwargs['stdout'].closed)
            self.assertEqual(result['status'], 'AUTHOR_GPU_QUEUE_LAUNCHED')
            self.assertEqual(result['max_parallel'], 2)
            self.assertEqual(result['exclusive_methods'], ['dpgba'])
            self.assertFalse((output / 'execution_claim.json').exists())

    def test_owned_shutdown_kills_only_timed_out_child(self):
        slow, done = MagicMock(pid=101), MagicMock(pid=102)
        slow.poll.return_value = None
        slow.wait.side_effect = [runner.subprocess.TimeoutExpired('owned synthetic child', 10), 0]
        done.poll.return_value = 0
        logs = [io.BytesIO(), io.BytesIO()]
        runner.stop_owned_children([dict(process=p, log=log) for p, log in zip((slow, done), logs)])
        slow.terminate.assert_called_once_with()
        slow.kill.assert_called_once_with()
        done.terminate.assert_not_called()
        done.kill.assert_not_called()
        self.assertTrue(all(log.closed for log in logs))


if __name__ == '__main__':
    unittest.main()
