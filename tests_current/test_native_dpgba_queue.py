"""Scheduling-only synthetic tests; no CUDA context or dataset training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from author_native_runs import dpgba_queue as queue


POLICY = dict(poll_seconds=30, consecutive_idle_checks=2,
              min_host_available_bytes=16 * 1024**3, min_cuda_free_bytes=8 * 1024**3,
              max_gpu_utilization_percent=5)
IDLE = dict(ok=True, gpu_compute_pids=[], host_available_bytes=32 * 1024**3,
            gpu_utilization_percent=0)
PROBE = dict(ok=True, free_bytes=24 * 1024**3, total_bytes=120 * 1024**3)


class IdleQueueTests(unittest.TestCase):
    def test_parallel_allows_active_gpu_but_still_requires_memory(self):
        policy = {**POLICY, 'require_no_gpu_compute_processes': False,
                  'max_gpu_utilization_percent': 100}
        busy = {**IDLE, 'gpu_compute_pids': [88], 'gpu_utilization_percent': 95}
        self.assertTrue(queue.resources_idle(busy, policy))
        self.assertFalse(queue.resources_idle({**busy, 'host_available_bytes': 1024}, policy))
        self.assertFalse(queue.resources_idle(dict(ok=False), policy))

    def test_parallel_does_not_wait_for_predecessor_terminal(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(queue.time, 'sleep') as sleep:
            root = Path(temp)
            plan = dict(schedule_mode='parallel', predecessor=str(root / 'active_gta'),
                        predecessor_plan_sha256='registered_hash')
            verify = Mock()
            queue.wait_for_predecessor(root, plan, verify)
            verify.assert_called_once()
            sleep.assert_not_called()
            self.assertTrue((root / 'parallel_predecessor.json').is_file())
            self.assertFalse((root / 'predecessor_terminal.json').exists())

    def test_parallel_still_requires_successful_cuda_probe(self):
        policy = {**POLICY, 'require_no_gpu_compute_processes': False,
                  'max_gpu_utilization_percent': 100}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(queue, 'resource_snapshot', return_value={**IDLE, 'gpu_compute_pids': [88]}), \
                patch.object(queue, 'cuda_probe', return_value=dict(ok=False, error='OOM')), \
                patch.object(queue.time, 'sleep'), \
                patch.object(queue.time, 'monotonic', side_effect=[0, 1, 101]):
            with self.assertRaises(TimeoutError):
                queue.wait_for_idle(Path(temp), dict(resource_policy=policy, max_wait_seconds=100), Mock(), 21)
            self.assertFalse((Path(temp) / 'seed21_resource_ready.json').exists())

    def test_gate_rejects_busy_unknown_and_low_memory(self):
        self.assertTrue(queue.resources_idle(IDLE, POLICY))
        for sample in [dict(ok=False), {**IDLE, 'gpu_compute_pids': [123]},
                       {**IDLE, 'host_available_bytes': 1024},
                       {**IDLE, 'gpu_utilization_percent': 20}]:
            self.assertFalse(queue.resources_idle(sample, POLICY))

    def test_two_stable_checks_then_probe_and_final_recheck(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(queue, 'resource_snapshot', return_value=IDLE) as snapshot, \
                patch.object(queue, 'cuda_probe', return_value=PROBE) as probe, \
                patch.object(queue.time, 'sleep') as sleep:
            root = Path(temp)
            verify = Mock()
            queue.wait_for_idle(root, dict(resource_policy=POLICY, max_wait_seconds=100), verify, 21)
            self.assertEqual(snapshot.call_count, 3)
            probe.assert_called_once()
            sleep.assert_called_once_with(30)
            self.assertTrue((root / 'seed21_resource_ready.json').is_file())
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'RESOURCE_READY')

    def test_busy_gpu_never_probes_or_launches(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(queue, 'resource_snapshot', return_value={**IDLE, 'gpu_compute_pids': [88]}), \
                patch.object(queue, 'cuda_probe') as probe, \
                patch.object(queue.subprocess, 'Popen') as launch, \
                patch.object(queue.time, 'monotonic', side_effect=[0, 101]):
            with self.assertRaises(TimeoutError):
                queue.wait_for_idle(Path(temp), dict(resource_policy=POLICY, max_wait_seconds=100), Mock(), 21)
            probe.assert_not_called()
            launch.assert_not_called()
            self.assertFalse((Path(temp) / 'seed21_resource_ready.json').exists())

    def test_failed_cuda_probe_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(queue, 'resource_snapshot', return_value=IDLE), \
                patch.object(queue, 'cuda_probe', return_value=dict(ok=False, error='OOM')), \
                patch.object(queue.time, 'sleep'), \
                patch.object(queue.time, 'monotonic', side_effect=[0, 1, 101]):
            with self.assertRaises(TimeoutError):
                queue.wait_for_idle(Path(temp), dict(resource_policy=POLICY, max_wait_seconds=100), Mock(), 21)
            self.assertFalse((Path(temp) / 'seed21_resource_ready.json').exists())

    def test_new_gpu_job_after_probe_prevents_launch(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(queue, 'resource_snapshot', side_effect=[IDLE, IDLE, {**IDLE, 'gpu_compute_pids': [99]}]), \
                patch.object(queue, 'cuda_probe', return_value=PROBE), \
                patch.object(queue.time, 'sleep'), \
                patch.object(queue.time, 'monotonic', side_effect=[0, 1, 101]):
            with self.assertRaises(TimeoutError):
                queue.wait_for_idle(Path(temp), dict(resource_policy=POLICY, max_wait_seconds=100), Mock(), 21)
            self.assertFalse((Path(temp) / 'seed21_resource_ready.json').exists())

    def test_cancel_checked_before_resource_query(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(queue, 'resource_snapshot') as query:
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                queue.wait_for_idle(Path(temp), dict(resource_policy=POLICY, max_wait_seconds=100),
                                    Mock(side_effect=RuntimeError('cancelled')), 21)
            query.assert_not_called()

    def test_nvidia_query_na_fails_closed(self):
        with patch.object(queue.subprocess, 'check_output', return_value='[N/A]\n'):
            sample = queue.resource_snapshot()
            self.assertFalse(sample['ok'])


if __name__ == '__main__':
    unittest.main()
