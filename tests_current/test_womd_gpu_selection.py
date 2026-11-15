"""Exercise fresh author GCN -> MLP -> SAGE source selection on CUDA.

Only tiny synthetic graphs are created. A child process isolates the author's
generic import names and per-run working directory from other tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _run_selection(output_dir):
    from spear_current.womd_pilot_selection import fit_source_selection

    device = torch.device('cuda:0')
    rng = torch.Generator().manual_seed(17)

    def graph(count):
        labels = torch.arange(count, dtype=torch.long) % 2
        features = torch.randn(count, 784, generator=rng) * .1
        features[:, :4] += (labels.float() * 2 - 1)[:, None]
        rows = torch.arange(count)
        edges = torch.stack((rows, (rows + 1) % count))
        edges = torch.cat((edges, edges.flip(0)), dim=1)
        return dict(x=features.to(device), y=labels.to(device),
                    eligible=torch.ones(count, dtype=torch.bool, device=device),
                    edge_index=edges.to(device))

    train, dev = graph(48), graph(16)
    train['eligible'][-2:] = False
    train['y'][-2:] = -1
    original = {name: value.clone() for name, value in train.items()}
    clean = torch.arange(16, dtype=torch.long, device=device)
    stages = []
    result = fit_source_selection(
        train, dev, seed=17, output_dir=output_dir,
        indices=dict(idx_train=clean), attachment_count=2, outer_extra_count=4,
        model_epochs=2, sage_n_permutations=2, sage_batch_size=2,
        explanation_count=8, background_count=4,
        progress=lambda event: stages.append(event['stage']))
    torch.cuda.synchronize(device)

    assert stages == ['fresh_author_GCN_ranking', 'fresh_author_MLP',
                      'fresh_author_SAGE'], stages
    assert result['summary']['status'] == 'FRESH_SELECTION_COMPLETE'
    coordinates = result['coordinates']
    assert coordinates.device == device and coordinates.dtype == torch.long
    assert coordinates.shape == (30,) and coordinates.unique().numel() == 30
    assert bool(((coordinates >= 0) & (coordinates < 784)).all())
    indices = result['indices']
    assert all(value.device == device for value in indices.values())
    assert torch.equal(indices['idx_train'], clean)
    assert indices['idx_attach'].shape == (2,)
    outer = indices['fixed_outer']
    assert outer.shape == (6,) and outer.unique().numel() == 6
    assert torch.equal(outer[:2], indices['idx_attach'])
    assert not bool(torch.isin(clean, outer).any())
    assert bool(train['eligible'][outer].all())
    assert bool((train['y'][outer] == 0).all())
    assert all(torch.equal(train[name], value) for name, value in original.items())

    directory = Path(output_dir)
    with np.load(directory / 'attribution.npz') as attribution:
        assert attribution['values'].shape == (784,)
        assert attribution['std'].shape == (784,)
        assert np.isfinite(attribution['values']).all()
        assert np.isfinite(attribution['std']).all()
        assert np.isin(attribution['explanation'], clean.cpu().numpy()).all()
        assert np.isin(attribution['background'], clean.cpu().numpy()).all()
        assert len(attribution['explanation']) == 8
        assert len(attribution['background']) == 4
        expected = np.argsort(attribution['values'])[::-1][:30].copy()
        assert np.array_equal(coordinates.cpu().numpy(), expected)
    for checkpoint in ('ranking_gcn.pt', 'attribution_mlp.pt'):
        state = torch.load(directory / checkpoint, map_location='cpu', weights_only=True)
        assert state and all(bool(torch.isfinite(value).all()) for value in state.values())
    saved_indices = torch.load(directory / 'indices.pt', weights_only=True)
    assert all(value.device.type == 'cpu' for value in saved_indices.values())
    assert all(torch.equal(saved_indices[name], value.cpu())
               for name, value in indices.items())
    binding = json.loads((directory / 'author_runtime/runtime_binding.json').read_text())
    assert binding['algorithm_changes'] is False
    assert binding['existing_selection_artifacts_copied'] is False
    print(json.dumps(dict(status='CUDA_SOURCE_SELECTION_REGRESSION_PASSED',
                          stages=stages, coordinates=len(coordinates),
                          outer_rows=len(outer), device=str(device))))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for this regression')
class CudaSourceSelectionTests(unittest.TestCase):
    def test_fresh_source_selection_runs_all_author_stages_on_cuda(self):
        with tempfile.TemporaryDirectory(prefix='womd_gpu_selection_') as temporary:
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            completed = subprocess.run(
                [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                 str(Path(temporary) / 'selection')],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn('CUDA_SOURCE_SELECTION_REGRESSION_PASSED', completed.stdout)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        _run_selection(sys.argv[2])
    else:
        unittest.main()
