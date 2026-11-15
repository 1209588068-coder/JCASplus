"""Strict deletion scheduling/integrity guards; no dataset or model training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from spear_current import strict_deletion as queue


class QueueTests(unittest.TestCase):
    def plan(self):
        return dict(version='strict_module_deletion_A_v1', cases=queue.matrix(),
                    seeds=list(queue.SEEDS), variants=list(queue.VARIANTS),
                    bound_files={}, sources={})

    def test_exact_matrix_not_an_asr_selected_subset(self):
        plan = self.plan()
        with patch.object(queue, 'sha', return_value='registered'), \
                patch.object(queue, 'read', return_value=plan):
            self.assertEqual(len(queue.verify('/tmp/plan.json', 'registered')['cases']), 12)
            plan['cases'].pop()
            with self.assertRaisesRegex(ValueError, 'matrix'):
                queue.verify('/tmp/plan.json', 'registered')

    def test_prefix_change_prevents_any_case(self):
        plan = self.plan()
        plan['bound_files']['/tmp/prefix.pt'] = 'expected'
        with patch.object(queue, 'sha', side_effect=['registered', 'changed']), \
                patch.object(queue, 'read', return_value=plan):
            with self.assertRaisesRegex(ValueError, 'prefix or input changed'):
                queue.verify('/tmp/plan.json', 'registered')

    def test_all_preflight_arms_are_required(self):
        with patch.object(queue, 'verify'), \
                patch.object(queue, 'read', return_value=dict(status='FAILED', plan_sha256='registered')), \
                patch.object(queue.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'four real-input preflights'):
                queue.launch('/tmp/plan.json', 'registered')
            spawn.assert_not_called()

    def test_detached_launch_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(queue, 'verify'), \
                patch.object(queue, 'read', return_value=dict(
                    status='STRICT_DELETION_PREFLIGHT_PASS', plan_sha256='registered')), \
                patch.object(queue.subprocess, 'Popen', return_value=Mock(pid=22)) as spawn:
            plan = Path(folder) / 'plan.json'
            queue.launch(plan, 'registered')
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            receipt = json.loads((plan.parent / 'launch.json').read_text())
            self.assertEqual(receipt['cases'], 12)
            self.assertTrue(receipt['concurrent_with_gta'])
            with self.assertRaisesRegex(ValueError, 'only once'):
                queue.launch(plan, 'registered')


if __name__ == '__main__':
    unittest.main()
