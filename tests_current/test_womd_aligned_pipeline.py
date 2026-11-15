"""Exercise the AV2-v2 runner core with a losslessly repacked WOMD fixture.

All arrays are synthetic. These checks are engineering acceptance, never WOMD
experimental results, and do not read the real AV2 or WOMD graph populations.
"""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from spear_current.av2_formal import run_pipeline, write_json, sha
from spear_current.womd_aligned_data import prepare, payload_sha256
from spear_current.av2_formal_data import iter_role_parts
from tests_current.test_womd_aligned_data import fixture


def run_smoke(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source, reference, original_parts, original_indices = fixture(destination)
    source_hashes = {str(p): sha(p) for p in source.rglob('*') if p.is_file()}
    data = destination/'data'
    summary = prepare(source, data, reference, max_nodes=35)
    run = destination/'run'
    run.mkdir()
    seed = 20261104
    plan = dict(data_dir=str(data), device='cuda:0', max_nodes=35,
        seed_offsets=dict(selection=0, clean_proxy_1=101000, clean_proxy_2=102000,
                          clean_reference=103000, anchor=201000, support=301000, victim=401000),
        source_selection=dict(attachment_count=3, outer_extra_count=4, model_epochs=2,
                              sage_n_permutations=2, sage_batch_size=2,
                              explanation_count=8, background_count=4),
        training=dict(clean_proxy_epochs=2, clean_reference_epochs=2, base_epochs=2,
                      residual_epochs=2, support_epochs=2, victim_epochs=2, inner_steps=1))
    events = []
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    generator_hash, results = run_pipeline(plan, seed, run, events.append)
    torch.cuda.synchronize()
    assert len(generator_hash) == 64
    assert source_hashes == {str(p): sha(p) for p in source.rglob('*') if p.is_file()}
    saved_indices = torch.load(run/'initial_indices.pt', map_location='cpu', weights_only=True)
    assert saved_indices.keys() == original_indices.keys()
    assert all(torch.equal(saved_indices[k], original_indices[k]) for k in original_indices)
    for role in ('train', 'dev', 'audit'):
        assert payload_sha256(original_parts[role]) == payload_sha256(iter_role_parts(data, role))
    checkpoint = torch.load(run/'generator.pt', map_location='cpu', weights_only=True)
    assert int(checkpoint['hard_support'].sum()) == 24
    active = checkpoint['coordinates'][checkpoint['hard_support']].tolist()
    assert 357 not in active and 715 not in active
    selection = json.loads((run/'selection/selection.json').read_text())
    assert selection['dataset'] == 'WOMD' and selection['source_dataset_split'] == 'training_20s'
    assert selection['attachment_count'] == 3
    assert selection['historical_checkpoint_loaded'] is False
    assert selection['historical_learned_support_loaded'] is False
    assert all('WOMD' in m['normalized_units'] for m in selection['metadata'])
    selected_indices = torch.load(run/'selection/indices.pt', map_location='cpu', weights_only=True)
    assert torch.equal(selected_indices['idx_train'], original_indices['idx_train'])
    assert len(selected_indices['idx_attach']) == 3 and len(selected_indices['fixed_outer']) == 7
    assert not bool(torch.isin(selected_indices['idx_train'], selected_indices['fixed_outer']).any())
    support = json.loads((run/'support/support.json').read_text())
    assert support['coordinate_budget'] == 24
    assert support['surrogate_updates'] == support['selector_updates'] == 2
    assert all('WOMD' in m['normalized_units'] for m in support['metadata'])
    anchor = json.loads((run/'anchor/summary.json').read_text())
    assert [(s['stage'],s['generator_steps'],s['surrogate_steps']) for s in anchor['stages']] == [
        ('base',2,2),('residual',2,2)]
    assert anchor['clean_kl_population'] == 'outer nodes only'
    assert anchor['output_mode'] == 'identity'
    initial_hashes = []
    for role in ('clean_proxy_1','clean_proxy_2','victim','clean_reference'):
        record = json.loads((run/role/'summary.json').read_text())
        assert record['dataset'] == 'WOMD' and record['source_dataset_split'] == 'training_20s'
        assert record['epochs'] == record['optimizer_steps'] == 2
        assert record['fresh_initialization'] and record['historical_checkpoint_loaded'] is False
        assert record['seed'] == seed + plan['seed_offsets'][role]
        initial_hashes.append(record['initial_state_sha256'])
    assert len(set(initial_hashes)) == 4
    for role in ('dev','audit'):
        record = results[role]
        assert record['status'] == 'EVALUATION_COMPLETE'
        assert record['dataset'] == 'WOMD' and record['source_dataset_split'] == 'training_20s'
        assert record['recording_independence_proven'] is False
        assert record['test_arrays_loaded'] is False
        assert record['targets'] == summary[role]['targets']
        with (run/f'evaluation_{role}/targets.csv').open() as stream:
            targets = list(csv.DictReader(stream))
        expected = [(p['scenario_id'],p['target_index'],
                     str(p['canonical_pair_track_ids'][p['target_index'],0]),
                     str(p['canonical_pair_track_ids'][p['target_index'],1]))
                    for p in original_parts[role] if p['target_index'] is not None]
        actual = [(r['scenario_id'],int(r['pair_index']),r['src_track_id'],r['dst_track_id']) for r in targets]
        assert actual == expected
        assert all(r['split_group_id'].startswith('source_shard:') for r in targets)
    stages = {e['stage'] for e in events}
    required = {'fresh_author_GCN_ranking','fresh_author_MLP','fresh_author_SAGE',
                'clean_proxy_1','clean_proxy_2','base','residual','semantic_support',
                'victim','clean_reference','evaluate_dev','evaluate_audit'}
    assert required <= stages
    receipt = dict(status='PASS', scope='synthetic WOMD engineering acceptance only; not experiment results',
        dataset='WOMD', source_dataset_split='training_20s', stages=sorted(stages), seed=seed,
        device='cuda:0', original_indices_preserved=True, original_payloads_preserved=True,
        original_targets_preserved=True, original_source_assets_unchanged=True,
        fresh_classifier_initializations=4, support_coordinates=24,
        protected_coordinates=[357,715], actual_runner_core='spear_current.av2_formal.run_pipeline',
        generator_state_sha256=generator_hash, generator_file_sha256=sha(run/'generator.pt'),
        evaluation_complete=['dev','audit'], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(destination/'receipt.json',receipt)
    return receipt


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class WOMDAlignedPipelineTests(unittest.TestCase):
    def test_womd_metadata_and_original_pools_survive_full_cuda_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = run_smoke(Path(temporary)/'pipeline')
            self.assertEqual(receipt['status'],'PASS')


if __name__ == '__main__':
    unittest.main()
