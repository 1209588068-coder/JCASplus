"""Exact recovery of Adam/dropout state and cumulative registered fit budgets.

CPU execution is limited to synthetic unit tests via explicit CUDA mocks; the
production fit_model still requires CUDA and has no CPU training fallback.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import av2_formal_classifiers as fitting


class StoppedAfterCheckpoint(Exception):
    pass


class ResumeTests(unittest.TestCase):
    DEVICE = 'cpu'

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='av2_resume_test_')
        self.root = Path(self.temporary.name)
        (self.root/'summary.json').write_text('{"status":"SYNTHETIC"}')
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(77)
        self.graphs = [dict(x=torch.randn(8, 4, generator=rng).to(self.DEVICE),
            y=(torch.arange(8) % 2).to(self.DEVICE), eligible=torch.ones(8, dtype=torch.bool, device=self.DEVICE),
            global_start=start, global_stop=start+8) for start in (0, 8)]
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(fitting, 'batches', side_effect=lambda *a, **kw: iter(self.graphs)))
        if self.DEVICE == 'cpu':
            self.stack.enter_context(patch.object(fitting, '_cuda_device', return_value=torch.device('cpu')))
            self.stack.enter_context(patch('torch.cuda.get_rng_state', side_effect=lambda *a: torch.get_rng_state()))
            self.stack.enter_context(patch('torch.cuda.set_rng_state', side_effect=lambda state, *a: torch.set_rng_state(state)))

    def tearDown(self):
        self.stack.close()
        torch.set_num_threads(self.previous_threads)
        self.temporary.cleanup()

    def fit(self, name, **kwargs):
        torch.manual_seed(11)
        np.random.seed(11)
        model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.ReLU(),
                                    torch.nn.Dropout(.5), torch.nn.Linear(7, 2))
        return fitting.fit_model(model, self.root, torch.arange(16), seed=11,
            output_dir=self.root/name, role='fresh_author_MLP', model_kind='mlp',
            epochs=4, max_nodes=8, device=self.DEVICE, checkpoint_rule='dev_accuracy', **kwargs)

    def interrupt(self, name='partial'):
        def stop(event):
            if event.get('optimizer_steps') == 2 and 'train_loss' in event:
                raise StoppedAfterCheckpoint()
        with self.assertRaises(StoppedAfterCheckpoint):
            self.fit(name, progress=stop)

    def test_partial_resume_matches_uninterrupted_dropout_adam_and_budget(self):
        baseline = self.fit('baseline')
        self.interrupt()
        old_bytes = (self.root/'partial/latest.pt').read_bytes()
        restored = self.fit('restored', resume_from=self.root/'partial')
        for name, value in baseline['model'].state_dict().items():
            torch.testing.assert_close(value, restored['model'].state_dict()[name], rtol=0, atol=0)
        full = torch.load(self.root/'baseline/latest.pt', weights_only=True)
        resumed = torch.load(self.root/'restored/latest.pt', weights_only=True)
        for name, value in full['model_state'].items():
            torch.testing.assert_close(value, resumed['model_state'][name], rtol=0, atol=0)
        for param_id, state in full['optimizer_state']['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, resumed['optimizer_state']['state'][param_id][key], rtol=0, atol=0)
        torch.testing.assert_close(full['rng_cpu'], resumed['rng_cpu'], rtol=0, atol=0)
        torch.testing.assert_close(full['rng_cuda'], resumed['rng_cuda'], rtol=0, atol=0)
        self.assertEqual(restored['summary']['optimizer_steps'], 4)
        self.assertEqual(restored['summary']['restored_optimizer_steps'], 2)
        self.assertEqual(restored['summary']['optimizer_steps_this_attempt'], 2)
        self.assertEqual((self.root/'partial/latest.pt').read_bytes(), old_bytes)
        rows = [json.loads(s) for s in (self.root/'restored/epochs.jsonl').read_text().splitlines()]
        self.assertEqual([r['epoch'] for r in rows], [0, 1, 2, 3])

    def test_completed_resume_reuses_selected_model_without_optimizer_work(self):
        baseline = self.fit('complete')
        with patch.object(fitting, 'accumulate_supervised_gradients', side_effect=AssertionError('must not train')):
            resumed = self.fit('restored', resume_from=self.root/'complete')
        self.assertEqual(resumed['summary']['optimizer_steps_this_attempt'], 0)
        self.assertEqual(resumed['summary']['restored_optimizer_steps'], 4)
        self.assertEqual(resumed['summary']['selected_epoch'], baseline['summary']['selected_epoch'])
        self.assertEqual(resumed['summary']['selected_state_sha256'], baseline['summary']['selected_state_sha256'])

    def test_missing_one_log_row_is_recorded_without_repeating_its_step(self):
        self.interrupt()
        path = self.root/'partial/epochs.jsonl'
        path.write_text(path.read_text().splitlines()[0]+'\n')
        result = self.fit('restored', resume_from=self.root/'partial')
        self.assertTrue(result['summary']['recovery']['reconstructed_missing_log_row'])
        rows = [json.loads(s) for s in (self.root/'restored/epochs.jsonl').read_text().splitlines()]
        self.assertTrue(rows[1]['epoch_metrics_unavailable'])
        self.assertEqual(result['summary']['optimizer_steps_this_attempt'], 2)

    def test_changed_data_schedule_optimizer_and_nonfinite_state_fail_closed(self):
        self.interrupt()
        path = self.root/'partial/latest.pt'
        original = torch.load(path, weights_only=True)
        for name, mutate in [
            ('data', lambda d: d.update(data_summary_sha256='changed')),
            ('seed', lambda d: d['metadata'].update(seed=12)),
            ('steps', lambda d: d['metadata'].update(optimizer_steps=3)),
            ('lr', lambda d: d['optimizer_state']['param_groups'][0].update(lr=.02)),
            ('adam_step', lambda d: next(iter(d['optimizer_state']['state'].values()))['step'].fill_(3)),
            ('nan', lambda d: next(iter(d['model_state'].values())).fill_(float('nan'))),
        ]:
            with self.subTest(name=name):
                from copy import deepcopy
                value = deepcopy(original); mutate(value); torch.save(value, path)
                with self.assertRaises((ValueError, FloatingPointError)):
                    self.fit('invalid_'+name, resume_from=self.root/'partial')
        torch.save(original, path)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for real optimizer/RNG recovery')
class CudaResumeTests(ResumeTests):
    """Run the same interrupted/resumed comparisons with actual CUDA dropout."""
    DEVICE = 'cuda:0'


if __name__ == '__main__':
    unittest.main()
