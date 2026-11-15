"""Register and supervise ten fresh full-TRAIN AV2 robustness runs on CUDA.

This is offline research on public data and researcher-owned models. Historical
development/confirmation pools remain disclosed development evaluation pools.
No validation/test arrays or historical attack weights are loaded here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'results/av2_formal_main_10seed_20261106_v2'
SEEDS = list(range(20260621, 20260631))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def recovery_registration(prior_plan_path, data, *, max_nodes, attachment_count):
    """Bind stopped prior ranking fits to the same data, seeds and budget."""
    prior_plan_path = Path(prior_plan_path).resolve()
    prior = verify(prior_plan_path)
    previous = Path(prior['output_dir'])
    if (sha(Path(data)/'summary.json') != prior['data_summary_sha256']
            or max_nodes != prior['max_nodes']
            or attachment_count != prior['source_selection']['attachment_count']):
        raise ValueError('Recovery may not change data, batching or poison budget')
    with (previous/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for status_path in [previous/'status.json', *previous.glob('seed*/status.json')]:
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        for key in ('pid', 'worker_pid'):
            pid = status.get(key)
            cmdline = Path(f'/proc/{pid}/cmdline')
            if isinstance(pid, int) and cmdline.exists():
                command = cmdline.read_bytes().replace(b'\0', b' ')
                if (b'spear_current.av2_formal' in command
                        and str(prior_plan_path).encode() in command):
                    raise RuntimeError('Stop the prior queue and worker before registering recovery')
    fits, files = {}, {str(prior_plan_path): sha(prior_plan_path),
                      str(previous/'registration.json'): sha(previous/'registration.json')}
    for seed in SEEDS:
        folder = previous/f'seed{seed}'
        if any((folder/p).exists() for p in ('completion.json', 'anchor', 'generator.pt',
                  'selection/attribution_fit', 'selection/ranking.pt')):
            raise ValueError('This recovery supports only interrupted ranking stages')
        fit = folder/'selection/ranking_fit'
        if not (fit/'latest.pt').exists():
            continue
        import torch
        saved = torch.load(fit/'latest.pt', map_location='cpu', weights_only=True)
        metadata = saved['metadata']
        if (any(type(metadata[k]) is not int for k in ('seed', 'epoch', 'next_epoch', 'epochs', 'optimizer_steps'))
                or metadata['seed'] != seed + prior['seed_offsets']['selection']
                or metadata['role'] != 'fresh_author_GCN_ranking'
                or metadata['epochs'] != prior['source_selection']['model_epochs']
                or metadata['epoch'] + 1 != metadata['next_epoch']
                or metadata['optimizer_steps'] != metadata['next_epoch']
                or not 0 < metadata['next_epoch'] <= metadata['epochs']):
            raise ValueError('Prior ranking checkpoint does not match the registered seed')
        for name in ('latest.pt', 'epochs.jsonl', 'model.pt', 'summary.json'):
            path = fit/name
            if path.exists():
                files[str(path)] = sha(path)
        initial = folder/'initial_indices.pt'
        if not initial.exists():
            raise ValueError('Missing original per-seed indices')
        files[str(initial)] = sha(initial)
        fits[str(seed)] = dict(directory=str(fit), completed_epochs=metadata['next_epoch'],
                              total_epochs=metadata['epochs'])
    if not fits:
        raise ValueError('No saved ranking checkpoints to recover')
    return dict(prior_plan=str(prior_plan_path), prior_plan_sha256=sha(prior_plan_path),
        asset_sha256=files, ranking_fits=fits,
        reason='Exact zero entropy terms use their continuous limit; resume same-seed ranking fits',
        score_policy='Unchanged author scores on ordinary rows; only zero terms extended to zero',
        same_seed_continuation=True, additional_training_budget=0,
        historical_failures_retained=True)


def register(output, *, acceptance, attachment_count, max_nodes=16384, recovery_from=None):
    output = Path(output).resolve()
    if (output / 'plan.json').exists():
        raise FileExistsError('Existing immutable plan; do not overwrite')
    data = output / 'data'
    summary = json.loads((data / 'summary.json').read_text())
    if (summary.get('status') != 'PREPARED' or summary.get('original_test_arrays_loaded', True)
            or any(summary[r]['scenes'] < 1 for r in ('train', 'dev', 'audit'))):
        raise ValueError('All frozen data roles must be prepared without test access')
    expected_count = math.ceil(26 * summary['train']['scenes'] / 512)
    if attachment_count != expected_count or attachment_count != summary['attachment_budget']['count']:
        raise ValueError('Attachment count differs from the explicit full-TRAIN budget')
    if summary['max_nodes_per_shard'] != max_nodes:
        raise ValueError('Execution batching differs from prepared batch registration')
    accepted = json.loads(Path(acceptance).read_text())
    if accepted.get('status') != 'PASS':
        raise ValueError('Full new CUDA pipeline acceptance required')
    for relative, digest in accepted['sources'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Source changed after acceptance: ' + relative)
    recovery = (recovery_registration(recovery_from, data, max_nodes=max_nodes,
                attachment_count=attachment_count) if recovery_from is not None else None)
    snapshot = output / 'source_snapshot'
    if snapshot.exists():
        raise FileExistsError('Snapshot already exists')
    plan = dict(
        schema='av2_full_train_fresh_ten_seed_v1', registered_at=now(),
        output_dir=str(output), data_dir=str(data), seeds=SEEDS,
        source_snapshot=str(snapshot),
        source_sha256={},
        data_summary_sha256=sha(data / 'summary.json'), data_summary=summary,
        acceptance=dict(path=str(Path(acceptance).resolve()), sha256=sha(acceptance)),
        prior_protocol=dict(path=str(ROOT / 'FORMAL_REPETITION_PROTOCOL_20261104.json'),
                            sha256=sha(ROOT / 'FORMAL_REPETITION_PROTOCOL_20261104.json')),
        method_identity=dict(path=str(ROOT / 'FINAL_METHOD_SELECTION_20261103.json'),
                             sha256=sha(ROOT / 'FINAL_METHOD_SELECTION_20261103.json')),
        method='fresh ranking GCN + author MLP/SAGE30 + semantic24 + uniform residual + dual clean KL',
        device='cuda:0', dtype='float32', amp=False, tf32=False, max_nodes=max_nodes,
        source_selection=dict(attachment_count=attachment_count, outer_extra_count=512,
                              model_epochs=200, sage_n_permutations=2048,
                              sage_batch_size=64, explanation_count=128, background_count=32),
        training=dict(clean_proxy_epochs=200, clean_reference_epochs=200,
                      base_epochs=200, residual_epochs=200, support_epochs=200,
                      victim_epochs=200, inner_steps=1),
        seed_offsets=dict(selection=0, clean_proxy_1=101000, clean_proxy_2=102000,
                          clean_reference=103000, anchor=201000, support=301000, victim=401000),
        stage_seed_policy='main_seed + disjoint registered stage offset; no stage/seed initialization collisions',
        clean_supervision='20 percent of all eligible TRAIN pairs; sampled anew by each main seed',
        poison_budget=dict(count=attachment_count, rule='ceil(26 * registered_train_scene_count / 512)',
                           user_choice='Explicitly confirmed scaled budget of 5865 pairs in this session',
                           interpretation='absolute pair budget scaled by scene count, not a 5 percent poison rate',
                           allow_multiple_pairs_per_scene=True,
                           denominator='report attach/(clean_supervised+attach) per seed'),
        training_execution=dict(
            batching='fixed scene order; whole disconnected scenes; maximum node target, no scene split',
            optimizer='one Adam step per full epoch after globally normalized gradient accumulation',
            dropout='independent seeded masks per batch; RNG layout differs from historical monolithic runs',
            no_neighbor_sampling=True, no_training_scene_subsampling=True,
            numerical_equivalence='same additive objective; FP32 summation order can differ'),
        checkpoint_selection=dict(generator='fixed last epoch each stage; no ASR selection',
                                  ranking_and_mlp='author earliest maximum clean DEV accuracy',
                                  graph_sage='earliest maximum clean DEV AUC'),
        support=dict(candidates='fresh author top30 per seed', protected=[357, 715],
                     grouping='copy identity intersections without candidate completion',
                     final_budget=24, infeasible='record failure; never refill candidates'),
        output_mode='identity', domain_projection=False, distribution_discriminator=False,
        pair_mean=False, historical_attack_checkpoint_loaded=False,
        historical_support_loaded=False, old_seeds_counted=False,
        evaluation=dict(roles=['dev', 'audit'], blind=False,
                        description='previously exposed DEV128 and AUDIT256+confirmation1024; full TRAIN excludes their entire recording groups',
                        targets='fixed one eligible negative per scene, independent of model',
                        reference='separate fresh clean GraphSAGE; never used in trigger optimization'),
        full_pipeline_repetition=True, frozen_test_arrays_loaded=False,
        minimum_cuda_free_bytes=8 * 1024**3, gpu_poll_seconds=30,
        queue_policy=dict(sequential=True, continue_independent_seeds_after_failure=True,
                          overwrite_partial_runs=False, completion_requires_all_ten=True,
                          automatic_retries=0, report_failures=True),
        scope='AV2 main method only; core baselines and five-seed extensions are separate batches',
        research_scope='offline authorized public-data model robustness research with researcher-owned GNNs')
    if recovery is not None:
        prior = json.loads(Path(recovery['prior_plan']).read_text())
        for key in ('seeds', 'device', 'dtype', 'amp', 'tf32', 'max_nodes', 'source_selection',
                    'training', 'seed_offsets', 'training_execution', 'checkpoint_selection',
                    'support', 'output_mode', 'evaluation', 'minimum_cuda_free_bytes'):
            if plan[key] != prior[key]:
                raise ValueError('Recovery changed the prior experiment contract: '+key)
        plan['recovery'] = recovery
    for folder in ('spear_current', 'third_party/author_baselines/SPEAR', 'tests_current'):
        pattern = 'test_av2*.py' if folder == 'tests_current' else '*.py'
        for source in sorted((ROOT / folder).rglob(pattern)):
            if '__pycache__' in source.parts:
                continue
            target = snapshot / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    plan['source_sha256'] = {str(p.relative_to(snapshot)): sha(p)
                            for p in sorted(snapshot.rglob('*.py'))}
    write_json(output / 'plan.json', plan)
    write_json(output / 'registration.json', dict(plan_sha256=sha(output / 'plan.json'), at=now()))
    write_json(output / 'status.json', dict(status='REGISTERED', at=now(), completed=0, total=10,
                                          training_started=False))
    return output / 'plan.json'


def verify(plan_path, *, snapshot_required=False):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan['output_dir'])
    if plan['seeds'] != SEEDS or sha(plan_path) != json.loads((output/'registration.json').read_text())['plan_sha256']:
        raise ValueError('Plan or seed list changed')
    if sha(Path(plan['data_dir'])/'summary.json') != plan['data_summary_sha256']:
        raise ValueError('Prepared data summary changed')
    for relative, digest in plan['data_summary']['asset_sha256'].items():
        if sha(Path(plan['data_dir'])/relative) != digest:
            raise ValueError('Prepared data registration or population changed: '+relative)
    if sha(plan['acceptance']['path']) != plan['acceptance']['sha256']:
        raise ValueError('Acceptance receipt changed')
    snapshot = Path(plan['source_snapshot'])
    for relative, digest in plan['source_sha256'].items():
        if sha(snapshot/relative) != digest:
            raise ValueError('Frozen source changed: '+relative)
    for path, digest in plan.get('recovery', {}).get('asset_sha256', {}).items():
        if sha(path) != digest:
            raise ValueError('Recovery checkpoint or prior registration changed: '+path)
    if snapshot_required and Path(__file__).resolve() != snapshot/'spear_current/av2_formal.py':
        raise ValueError('Execute only the registered source snapshot')
    return plan


def _scalars(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(_scalars(child, prefix+'.'+key if prefix else key))
    elif isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        result[prefix] = value
    return result


def aggregate(output, seeds=SEEDS):
    output = Path(output)
    completed, failed, metrics, paired = [], [], {}, {}
    plan_hash = sha(output/'plan.json')
    for seed in seeds:
        folder = output / f'seed{seed}'
        receipt = folder / 'completion.json'
        if not receipt.exists():
            if (folder/'failure.json').exists():
                failed.append(dict(seed=seed, failure=str(folder/'failure.json')))
            continue
        try:
            done = json.loads(receipt.read_text())
            if ((folder/'failure.json').exists() or done['seed'] != seed
                    or done['status'] != 'SEED_COMPLETE_REVIEW_REQUIRED'
                    or done['plan_sha256'] != plan_hash
                    or done['generator_file_sha256'] != sha(folder/'generator.pt')):
                raise ValueError('Contradictory or invalid completion receipt')
            records = {}
            for role in ('dev', 'audit'):
                summary_path = folder/f'evaluation_{role}/summary.json'
                record = json.loads(summary_path.read_text())
                if (record['status'] != 'EVALUATION_COMPLETE'
                        or done['result_sha256'][role] != sha(summary_path)):
                    raise ValueError('Missing, changed or incomplete evaluation: '+role)
                for model in ('victim', 'clean_reference'):
                    for population in ('all', 'slow', 'moving'):
                        normal = record['metrics'][model]['normal'][population]
                        targets = record['metrics'][model]['targets'][population]
                        count = targets['targets']
                        if type(count) is not int or count < 0:
                            raise ValueError('Malformed target denominator')
                        values = [normal[k] for k in ('clean_pair_auc', 'clean_pair_ap',
                                  'clean_pair_positive_recall')] + [targets['incremental_asr']]
                        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v))
                               for v in values):
                            raise ValueError('Malformed or nonfinite evaluation metric')
                records[role] = record
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failed.append(dict(seed=seed, validation_error=str(exc), receipt=str(receipt)))
            continue
        completed.append(seed)
        for role in ('dev', 'audit'):
            record = records[role]
            for name, value in _scalars(record['metrics']).items():
                metrics.setdefault(role+'.'+name, []).append(dict(seed=seed, value=value))
            for population in ('all', 'slow', 'moving'):
                victim = record['metrics']['victim']['normal'][population]
                reference = record['metrics']['clean_reference']['normal'][population]
                for name in ('clean_pair_auc', 'clean_pair_ap', 'clean_pair_positive_recall'):
                    a, b = victim.get(name), reference.get(name)
                    if a is not None and b is not None:
                        paired.setdefault(f'{role}.{population}.{name}.victim_minus_clean_reference', []).append(
                            dict(seed=seed, value=a-b))

    def reduce(rows):
        values = [r['value'] for r in rows]
        return dict(per_seed=rows, n=len(values), mean=statistics.mean(values),
                    sample_sd=statistics.stdev(values) if len(values)>1 else None)

    record = dict(status='ALL_TEN_COMPLETE' if len(completed)==len(seeds) else 'PARTIAL', at=now(),
                  expected_seeds=seeds, completed_seeds=completed, failed_seeds=failed,
                  completed=len(completed), total=len(seeds),
                  metrics={k:reduce(v) for k,v in metrics.items()},
                  paired_differences={k:reduce(v) for k,v in paired.items()},
                  independent_unit='fresh full pipeline seed; repeated targets are not new independent targets',
                  sample_sd_ddof=1, failure_is_zero_asr=False, blind_evaluation=False)
    write_json(output/'aggregate.json', record)
    return record


def run_pipeline(plan, seed, folder, progress):
    import numpy as np
    import torch
    from .av2_formal_data import make_indices
    from .av2_formal_selection import fit_source_selection
    from .av2_formal_classifiers import fit_classifier, fit_victim, evaluate
    from .av2_formal_generator import fit_generator, fit_support
    from .full_ablation_training import _cpu_state, _state_hash
    from .full_ablation_evaluation import _fp32
    from .womd_pilot_training import cuda_preflight

    np.random.seed(seed); torch.manual_seed(seed)
    data = Path(plan['data_dir'])
    indices = make_indices(data, seed)
    torch.save(indices, folder/'initial_indices.pt')
    offsets, config = plan['seed_offsets'], plan['training']
    shared = dict(device=plan['device'], max_nodes=plan['max_nodes'], progress=progress)
    ranking_fit = plan.get('recovery', {}).get('ranking_fits', {}).get(str(seed))
    recovery_args = ({'ranking_resume_from': ranking_fit['directory']} if ranking_fit else {})
    with _fp32():
        selected = fit_source_selection(data, seed=seed+offsets['selection'], indices=indices,
            output_dir=folder/'selection', **plan['source_selection'], **shared, **recovery_args)
        indices, coordinates = selected['indices'], selected['coordinates']
        proxies = []
        for role in ('clean_proxy_1', 'clean_proxy_2'):
            result = fit_classifier(data, indices['idx_train'], seed=seed+offsets[role],
                role=role, output_dir=folder/role, epochs=config['clean_proxy_epochs'], **shared)
            proxies.append(result['model'])
        anchor = fit_generator(data, indices, proxies, coordinates, seed=seed+offsets['anchor'],
            output_dir=folder/'anchor', base_epochs=config['base_epochs'],
            residual_epochs=config['residual_epochs'], **shared)
        generator = anchor['generator']
        support, support_summary = fit_support(generator, anchor['surrogate'], data, indices,
            proxies, coordinates, seed=seed+offsets['support'], output_dir=folder/'support',
            epochs=config['support_epochs'], **shared)
        generator.set_support(support)
        generator.set_phase('support_surrogate').requires_grad_(False).eval()
        torch.save(dict(state_dict=_cpu_state(generator), coordinates=coordinates.cpu(),
                   hard_support=support.cpu(), metadata=dict(seed=seed, output_mode='identity')),
                   folder/'generator.pt')
        generator_hash = _state_hash(generator)
        del proxies, anchor, result
        gc.collect(); torch.cuda.empty_cache()
        victim = fit_victim(generator, data, indices, seed=seed+offsets['victim'],
            output_dir=folder/'victim', epochs=config['victim_epochs'], **shared)
        reference = fit_classifier(data, indices['idx_train'], seed=seed+offsets['clean_reference'],
            output_dir=folder/'clean_reference', role='clean_reference',
            epochs=config['clean_reference_epochs'], **shared)
        results = {}
        for role in ('dev', 'audit'):
            results[role] = evaluate(generator, victim['model'], reference['model'], data,
                output_dir=folder/f'evaluation_{role}', role=role,
                device=plan['device'], progress=progress)
    return generator_hash, results


def worker(plan_path, seed):
    import numpy as np
    import torch
    from .av2_formal_data import make_indices
    from .av2_formal_selection import fit_source_selection
    from .av2_formal_classifiers import fit_classifier, fit_victim, evaluate
    from .av2_formal_generator import fit_generator, fit_support
    from .full_ablation_training import _cpu_state, _state_hash
    from .full_ablation_evaluation import _fp32
    from .womd_pilot_training import cuda_preflight

    plan = verify(plan_path, snapshot_required=True)
    if seed not in plan['seeds']:
        raise ValueError('Unregistered seed')
    folder = Path(plan['output_dir'])/f'seed{seed}'
    folder.mkdir(exist_ok=True)
    if any((folder/n).exists() for n in ('completion.json','selection','anchor','failure.json')):
        raise FileExistsError('Never overwrite a completed or partial seed')
    start = time.monotonic()
    training_started = False

    def status(state, **extra):
        write_json(folder/'status.json', dict(status=state, at=now(), pid=os.getpid(), seed=seed,
                  elapsed_seconds=time.monotonic()-start, training_started=training_started, **extra))

    last_logged = [0.]
    def progress(row):
        status('RUNNING', progress=row)
        if time.monotonic()-last_logged[0] > 30 or not any(k in row for k in ('batch', 'batches_done')):
            print('[av2] '+json.dumps(dict(seed=seed, **row), allow_nan=False), flush=True)
            last_logged[0] = time.monotonic()

    try:
        torch.set_num_threads(1)
        while True:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; CPU training fallback is forbidden')
            free, _ = torch.cuda.mem_get_info(plan['device'])
            if free >= plan['minimum_cuda_free_bytes']:
                break
            status('WAITING_FOR_GPU', free_bytes=free)
            time.sleep(plan['gpu_poll_seconds'])
        write_json(folder/'cuda_preflight.json', dict(at=now(), **cuda_preflight(plan['device'])))
        training_started = True
        generator_hash, results = run_pipeline(plan, seed, folder, progress)
        verify(plan_path, snapshot_required=True)
        write_json(folder/'completion.json', dict(status='SEED_COMPLETE_REVIEW_REQUIRED', seed=seed,
                   completed_at=now(), elapsed_seconds=time.monotonic()-start,
                   plan_sha256=sha(plan_path), generator_state_sha256=generator_hash,
                   generator_file_sha256=sha(folder/'generator.pt'), full_pipeline_repetition=True,
                   frozen_test_arrays_loaded=False, blind_evaluation=False,
                   result_sha256={r:sha(folder/f'evaluation_{r}/summary.json') for r in results},
                   result_paths={r:str(folder/f'evaluation_{r}/summary.json') for r in results},
                   ranking_recovery=plan.get('recovery', {}).get('ranking_fits', {}).get(str(seed))))
        status('SEED_COMPLETE_REVIEW_REQUIRED')
        print(f'[av2] seed {seed} complete', flush=True)
    except BaseException as exc:
        failure = dict(status='FAILED_REVIEW_REQUIRED', seed=seed, at=now(),
                       type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write_json(folder/'failure.json', failure)
        status('FAILED_REVIEW_REQUIRED', failure=failure)
        raise


def queue(plan_path):
    plan = verify(plan_path, snapshot_required=True)
    output = Path(plan['output_dir'])
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in plan['seeds']:
            folder = output/f'seed{seed}'
            if (folder/'completion.json').exists() or (folder/'failure.json').exists():
                continue
            folder.mkdir(exist_ok=True)
            if (folder/'run.log').exists():
                previous = json.loads((folder/'launch.json').read_text()) if (folder/'launch.json').exists() else {}
                old_pid = previous.get('pid')
                if isinstance(old_pid, int):
                    try:
                        os.kill(old_pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError('Existing worker may still be alive; refusing another GPU worker')
                write_json(folder/'failure.json', dict(status='INTERRUPTED_PARTIAL_SEED_REVIEW_REQUIRED',
                    seed=seed, at=now(), reason='Existing run log without completion; retained, never overwritten'))
                continue
            command = [sys.executable, '-B', '-m', 'spear_current.av2_formal', '--worker',
                       str(plan_path), '--seed', str(seed)]
            validated = aggregate(output, plan['seeds'])
            with (folder/'run.log').open('x') as log:
                child = subprocess.Popen(command, cwd=plan['source_snapshot'], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                write_json(folder/'launch.json', dict(pid=child.pid, at=now(), command=command))
                while True:
                    returncode = child.poll()
                    current = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                    state = ('WAITING_FOR_GPU' if current.get('status') == 'WAITING_FOR_GPU'
                             else 'RUNNING' if current.get('training_started') else 'STARTING_WORKER')
                    write_json(output/'status.json', dict(status=state, at=now(), pid=os.getpid(),
                        active_seed=seed, worker_pid=child.pid, completed=validated['completed'],
                        failed=len(validated['failed_seeds']),
                        total=len(plan['seeds']), worker_status=current,
                        training_started=current.get('training_started', False)))
                    if returncode is not None:
                        break
                    time.sleep(10)
            if returncode != 0 or not (folder/'completion.json').exists():
                if not (folder/'failure.json').exists():
                    write_json(folder/'failure.json', dict(status='WORKER_EXITED_WITHOUT_COMPLETION',
                        at=now(), seed=seed, returncode=returncode, log=str(folder/'run.log')))
                print(f'[av2-queue] seed {seed} FAILED; retained records; continuing remaining registered seeds', flush=True)
            aggregate(output, plan['seeds'])
        result = aggregate(output, plan['seeds'])
        state = 'TEN_SEEDS_COMPLETE_REVIEW_REQUIRED' if result['completed']==10 else 'FINISHED_WITH_FAILURES_REVIEW_REQUIRED'
        final = dict(status=state, at=now(), completed=result['completed'], total=10,
                     failed=len(result['failed_seeds']), aggregate=str(output/'aggregate.json'),
                     plan_sha256=sha(plan_path), pid=os.getpid())
        if result['completed']==10:
            write_json(output/'completion.json', final)
        write_json(output/'status.json', final)
        print(json.dumps(final), flush=True)


def launch(plan_path):
    plan = verify(plan_path)
    output = Path(plan['output_dir'])
    if (output/'launch.json').exists():
        raise FileExistsError('Launch already recorded; do not duplicate queue')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, '-B', '-m', 'spear_current.av2_formal', '--run', str(Path(plan_path).resolve())]
    with (output/'queue.log').open('x') as log:
        child = subprocess.Popen(command, cwd=plan['source_snapshot'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid, at=now(), command=command, source_snapshot=plan['source_snapshot'])
    write_json(output/'launch.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--launch', type=Path)
    actions.add_argument('--run', type=Path)
    actions.add_argument('--worker', type=Path)
    actions.add_argument('--aggregate', type=Path)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--acceptance', type=Path)
    p.add_argument('--attachment-count', type=int)
    p.add_argument('--max-nodes', type=int, default=16384)
    p.add_argument('--seed', type=int)
    p.add_argument('--recover-from', type=Path, help='Stopped prior plan; preserve data and recover ranking fits')
    args = p.parse_args()
    if args.prepare:
        if args.acceptance is None or args.attachment_count is None:
            p.error('--prepare requires --acceptance and --attachment-count')
        print(register(args.output_dir, acceptance=args.acceptance,
                       attachment_count=args.attachment_count, max_nodes=args.max_nodes,
                       recovery_from=args.recover_from))
    elif args.launch:
        print(json.dumps(launch(args.launch), indent=2))
    elif args.run:
        queue(args.run)
    elif args.worker:
        worker(args.worker, args.seed)
    else:
        print(json.dumps(aggregate(args.aggregate), indent=2))


if __name__ == '__main__':
    main()
