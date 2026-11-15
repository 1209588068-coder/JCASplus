"""Scheduling contract tests; no dataset or GPU training."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from spear_current import full_ablation as queue


class QueueTests(unittest.TestCase):
    def plan(self):
        return dict(manifest='manifest', manifest_sha256='hash',
            prepared_inputs='inputs', inputs_sha256='hash', clean_assets={}, sources={},
            cases=[dict(seed=s, variant=v) for s in queue.SEEDS for v in queue.VARIANTS])

    def test_exact_registered_matrix_required(self):
        plan = self.plan()
        with patch.object(queue, 'sha', return_value='hash'), patch.object(queue, 'read', return_value=plan):
            self.assertEqual(len(queue.verify(Path('/tmp/plan.json'), 'hash')['cases']), 27)
            plan['cases'].pop()
            with self.assertRaisesRegex(ValueError, 'matrix'):
                queue.verify(Path('/tmp/plan.json'), 'hash')

    def test_bound_input_change_prevents_execution(self):
        with patch.object(queue, 'sha', side_effect=['hash', 'changed']), \
                patch.object(queue, 'read', return_value=self.plan()):
            with self.assertRaisesRegex(ValueError, 'input changed'):
                queue.verify(Path('/tmp/plan.json'), 'hash')

    def test_failed_preflight_never_launches(self):
        with patch.object(queue, 'sha', return_value='hash'), \
                patch.object(queue, 'verify'), \
                patch.object(queue, 'read', return_value=dict(status='FAILED', plan_sha256='hash')), \
                patch.object(queue.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(ValueError, 'preflight'):
                queue.launch(Path('/tmp/plan.json'))
            start.assert_not_called()

    def test_valid_preflight_launch_is_detached_and_records_27(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(queue, 'sha', return_value='hash'), \
                patch.object(queue, 'verify'), \
                patch.object(queue, 'read', return_value=dict(status='ENGINEERING_PREFLIGHT_PASS', plan_sha256='hash')), \
                patch.object(queue.subprocess, 'Popen', return_value=Mock(pid=123)) as start:
            root = Path(temp)
            queue.launch(root / 'plan.json')
            self.assertTrue(start.call_args.kwargs['start_new_session'])
            import json
            receipt = json.loads((root / 'launch.json').read_text())
            self.assertEqual(receipt['total_cases'], 27)
            self.assertTrue(receipt['concurrent_with_author_queues'])
            with self.assertRaisesRegex(ValueError, 'only once'):
                queue.launch(root / 'plan.json')


if __name__ == '__main__':
    unittest.main()
