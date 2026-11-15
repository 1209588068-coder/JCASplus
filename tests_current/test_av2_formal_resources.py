"""Synthetic coverage of the exact zero-update CUDA resource probe."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from spear_current.av2_formal_data import iter_role_batches
from spear_current.av2_formal_resources import probe_batch
from tests_current.test_av2_formal_data import make_tiny_cache


class ResourceProbeTests(unittest.TestCase):
    def test_cpu_fallback_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            probe_batch({}, '/unused-av2-probe.json', device='cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for actual resource probe')
    def test_all_six_backward_paths_without_any_optimizer_step(self):
        with tempfile.TemporaryDirectory(prefix='av2_resource_test_') as temporary:
            root = Path(temporary)
            with redirect_stdout(io.StringIO()):
                make_tiny_cache(root/'data')
            batch = next(iter_role_batches(root/'data', 'train', max_nodes=128))
            original = {k:batch[k].clone() for k in ('x', 'y', 'edge_index', 'eligible')}
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state()
            with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer step allowed')), \
                 patch.object(torch.optim.SGD, 'step', side_effect=AssertionError('No optimizer step allowed')):
                result = probe_batch(batch, root/'probe.json')
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['optimizer_steps'], 0)
            self.assertEqual(result['outer_targets'], 32)
            self.assertEqual(result['nodes'], 128)
            self.assertEqual(len(result['kernels']), 6)
            for checked in result['kernels'].values():
                self.assertTrue(checked['forward_backward_finite'])
                self.assertTrue(checked['parameters_unchanged'])
                self.assertEqual(checked['optimizer_steps'], 0)
                self.assertGreater(checked['peak_allocated_bytes'], 0)
            for key in original:
                torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)
            torch.testing.assert_close(cpu_rng, torch.get_rng_state(), rtol=0, atol=0)
            torch.testing.assert_close(cuda_rng, torch.cuda.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(json.loads((root/'probe.json').read_text()), result)


if __name__ == '__main__':
    unittest.main()
