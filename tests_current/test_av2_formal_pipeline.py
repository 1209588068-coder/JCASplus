"""Exercise the same runner core used by real jobs, on synthetic CUDA data."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from spear_current.av2_formal import run_pipeline, write_json, sha
from tests_current.test_av2_formal_data import make_tiny_cache


def run_smoke(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data = destination/'data'
    make_tiny_cache(data)
    run = destination/'run'
    run.mkdir()
    plan = dict(data_dir=str(data), device='cuda:0', max_nodes=128,
        seed_offsets=dict(selection=0, clean_proxy_1=101000, clean_proxy_2=102000,
                          clean_reference=103000, anchor=201000, support=301000, victim=401000),
        source_selection=dict(attachment_count=6, outer_extra_count=16, model_epochs=2,
                              sage_n_permutations=2, sage_batch_size=2,
                              explanation_count=16, background_count=8),
        training=dict(clean_proxy_epochs=2, clean_reference_epochs=2, base_epochs=2,
                      residual_epochs=2, support_epochs=2, victim_epochs=2, inner_steps=1))
    events = []
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    generator_hash, results = run_pipeline(plan, 17, run, events.append)
    torch.cuda.synchronize()
    assert len(generator_hash) == 64
    checkpoint = torch.load(run/'generator.pt', map_location='cpu', weights_only=True)
    assert int(checkpoint['hard_support'].sum()) == 24
    active = checkpoint['coordinates'][checkpoint['hard_support']].tolist()
    assert 357 not in active and 715 not in active
    for role in ('dev', 'audit'):
        assert results[role]['status'] == 'EVALUATION_COMPLETE'
        assert results[role]['dataset'] == 'AV2'
        assert (run/f'evaluation_{role}/targets.csv').exists()
    for role in ('clean_proxy_1','clean_proxy_2','victim','clean_reference'):
        record = json.loads((run/role/'summary.json').read_text())
        assert record['epochs'] == record['optimizer_steps'] == 2
        assert record['fresh_initialization']
    support = json.loads((run/'support/support.json').read_text())
    assert support['coordinate_budget'] == 24
    selection = json.loads((run/'selection/selection.json').read_text())
    assert selection['attachment_count'] == 6
    receipt = dict(status='PASS', scope='synthetic engineering smoke, not AV2 experimental results',
        stages=sorted({e['stage'] for e in events}), device='cuda:0',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        generator_state_sha256=generator_hash, generator_file_sha256=sha(run/'generator.pt'),
        actual_runner_core='spear_current.av2_formal.run_pipeline', support_coordinates=24,
        evaluation_complete=['dev','audit'])
    write_json(destination/'receipt.json', receipt)
    return receipt


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AV2FormalPipelineTests(unittest.TestCase):
    def test_new_runner_executes_all_stages_on_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_smoke(Path(temporary)/'pipeline')
            self.assertEqual(result['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
