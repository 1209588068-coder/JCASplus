"""Launch one WOMD pilot matched to the registered AV2 repair pipeline.

Only public data and researcher-owned models are used, offline. Existing WOMD
scenes, normalization, supervision and evaluation targets are held fixed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import difflib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .av2_formal import now, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'results/womd_av2_aligned_pilot_20261106_v1'
DEFAULT_REFERENCE = ROOT/'results/av2_formal_main_10seed_20261106_v2/plan.json'
OVERLAYS = ('av2_formal_data.py', 'av2_formal_selection.py',
            'av2_formal_classifiers.py', 'av2_formal_generator.py')
COPIED_CONTRACT = ('training', 'seed_offsets', 'training_execution', 'checkpoint_selection',
                   'support', 'output_mode', 'domain_projection',
                   'distribution_discriminator', 'pair_mean', 'device', 'max_nodes',
                   'dtype', 'amp', 'tf32')


def read(path):
    return json.loads(Path(path).read_text())


def reference_contract(path):
    path = Path(path).resolve()
    reference = read(path)
    if sha(path) != read(path.parent/'registration.json')['plan_sha256']:
        raise ValueError('Changed AV2 reference plan')
    if reference.get('recovery', {}).get('score_policy') != (
            'Unchanged author scores on ordinary rows; only zero terms extended to zero'):
        raise ValueError('Expected the registered AV2 exact-zero entropy repair')
    for relative, digest in reference['source_sha256'].items():
        if sha(Path(reference['source_snapshot'])/relative) != digest:
            raise ValueError('Changed frozen reference code: '+relative)
    return reference


def validate_matched_contract(plan, reference, data):
    for key in COPIED_CONTRACT:
        if plan[key] != reference[key]:
            raise ValueError('WOMD contract differs from AV2: '+key)
    expected = dict(reference['source_selection'], attachment_count=data['attachment_budget']['count'])
    if plan['source_selection'] != expected or plan['poison_budget'] != data['attachment_budget']:
        raise ValueError('Selection or poisoning budget is not matched')
    if (plan['seed'] != data['seed'] or data['dataset_id'] != 'WOMD'
            or data['source_dataset_split'] != 'training_20s'
            or data['status'] != 'PREPARED' or data['test_arrays_loaded']
            or data['recording_independence_proven']
            or data['new_statistics_fitted'] != 0
            or not data['clean_indices']['exact_original_indices_preserved']
            or data['max_nodes_per_shard'] != plan['max_nodes']):
        raise ValueError('Existing WOMD population/supervision contract changed')
    if (plan.get('recovery') or plan.get('historical_attack_checkpoint_loaded')
            or plan.get('historical_support_loaded')):
        raise ValueError('The matched WOMD run must freshly train every model')
    for role in ('train', 'dev', 'audit'):
        if data[role]['source_payload_sha256'] != data[role]['repacked_payload_sha256']:
            raise ValueError('WOMD scene payload changed: '+role)


def register(output, reference_plan, acceptance):
    output, reference_plan, acceptance = map(lambda p: Path(p).resolve(),
                                            (output, reference_plan, acceptance))
    if (output/'plan.json').exists() or (output/'source_snapshot').exists():
        raise FileExistsError('Existing matched registration is never overwritten')
    reference = reference_contract(reference_plan)
    data = read(output/'data/summary.json')
    accepted = read(acceptance)
    if (accepted.get('status') != 'PASS'
            or accepted['reference_plan_sha256'] != sha(reference_plan)
            or accepted['data_summary_sha256'] != sha(output/'data/summary.json')):
        raise ValueError('Matching full CUDA/data/resource acceptance is required')
    for relative, digest in accepted['sources'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Source changed after acceptance: '+relative)
    allowed = {'spear_current/'+name for name in OVERLAYS}
    for relative, digest in reference['source_sha256'].items():
        if (relative.startswith(('spear_current/','third_party/'))
                and relative not in allowed and sha(ROOT/relative) != digest):
            raise ValueError('Unreviewed change to shared AV2 computation: '+relative)
    plan = {key:deepcopy(reference[key]) for key in COPIED_CONTRACT}
    plan.update(schema='womd_av2_repair_aligned_pilot_v1', registered_at=now(),
        output_dir=str(output), data_dir=str(output/'data'), source_snapshot=str(output/'source_snapshot'),
        seed=data['seed'], expected_runs=1,
        reference_plan=dict(path=str(reference_plan), sha256=sha(reference_plan)),
        data_summary_sha256=sha(output/'data/summary.json'), data_summary=data,
        source_selection=dict(reference['source_selection'], attachment_count=data['attachment_budget']['count']),
        poison_budget=data['attachment_budget'],
        method=reference['method'], historical_attack_checkpoint_loaded=False,
        historical_support_loaded=False, fresh_full_pipeline=True, full_dataset_repetition=False,
        acceptance=dict(path=str(acceptance), sha256=sha(acceptance)),
        scope='one matched WOMD internal development pilot on the existing 512/128/256 scenes',
        evaluation=dict(roles=['dev','audit'], threshold='binary argmax; ties class0',
            target_policy='byte-identical old target indices and vehicle pair identities; no model filtering',
            normal_and_response_metrics='same shared evaluator as active AV2 v2',
            clean_reference='fresh independently seeded clean GraphSAGE, absent from trigger gradients',
            blind=False, recording_independence_proven=False),
        loss_contract=dict(outer='global clean+outer NLL + 0.1 global node cosine + mean of two outer-node clean KL',
            support='same outer loss plus ((expected_coordinate_cost-24)/legal_capacity)^2 once per epoch',
            generator_lr=.0002, surrogate_lr=.0002, generator_weight_decay=.0005,
            selector_lr=.01, selector_weight_decay=0., classifier_lr=.01, classifier_weight_decay=.0005),
        interpretation='derived feature-space robustness, not proof of physically realizable trajectories',
        test_arrays_loaded=False, automatic_formal_runs_started=False,
        minimum_cuda_free_bytes=reference['minimum_cuda_free_bytes'], gpu_poll_seconds=30,
        resource_policy='CUDA only; keep all scenes and edges; no CPU fallback; do not stop existing AV2 jobs',
        reuse_policy='only deterministic WOMD data cache and existing clean indices; no prior model state',
        queue_policy=dict(automatic_retries=0, preserve_failures_and_checkpoints=True,
                          completion_requires_both_evaluations=True),
        research_scope='offline public-data model robustness research with researcher-owned GNNs')
    validate_matched_contract(plan, reference, data)
    # Use the running reference snapshot as the base. Only the reviewed dataset
    # identity/metadata adapters and this new launch wrapper are overlaid.
    snapshot = Path(plan['source_snapshot'])
    snapshot.mkdir()
    for relative, digest in reference['source_sha256'].items():
        source = Path(reference['source_snapshot'])/relative
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    diffs, adaptations = [], {}
    for name in OVERLAYS:
        relative = 'spear_current/'+name
        old = Path(reference['source_snapshot'])/relative
        source = ROOT/relative
        diffs.extend(difflib.unified_diff(old.read_text().splitlines(True), source.read_text().splitlines(True),
                     fromfile='AV2_v2/'+relative, tofile='WOMD_matched/'+relative))
        shutil.copyfile(source, snapshot/relative)
        adaptations[relative] = dict(reference_sha256=sha(old), matched_sha256=sha(source),
            purpose='explicit WOMD training_20s identity and dataset/normalization provenance; training arithmetic retained')
    for name in ('womd_aligned.py', 'womd_aligned_data.py'):
        shutil.copyfile(ROOT/'spear_current'/name, snapshot/'spear_current'/name)
    for source in (ROOT/'tests_current').glob('test_womd_aligned*.py'):
        target = snapshot/'tests_current'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (output/'code_alignment.diff').write_text(''.join(diffs))
    plan['implementation_adaptations'] = adaptations
    plan['alignment_diff_sha256'] = sha(output/'code_alignment.diff')
    plan['source_sha256'] = {str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*.py'))}
    plan['shared_pipeline'] = dict(module='spear_current.av2_formal.run_pipeline',
        file_sha256=sha(snapshot/'spear_current/av2_formal.py'), identical_to_active_av2_v2=True)
    write_json(output/'plan.json', plan)
    write_json(output/'registration.json', dict(plan_sha256=sha(output/'plan.json'), at=now()))
    write_json(output/'status.json', dict(status='REGISTERED', at=now(), training_started=False,
                                         completed=0, total=1, seed=plan['seed']))
    return output/'plan.json'


def verify(plan_path, *, frozen=False):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    output = Path(plan['output_dir'])
    if sha(plan_path) != read(output/'registration.json')['plan_sha256']:
        raise ValueError('Matched plan changed')
    if sha(plan['reference_plan']['path']) != plan['reference_plan']['sha256']:
        raise ValueError('Bound AV2 reference changed')
    reference = reference_contract(plan['reference_plan']['path'])
    data = read(Path(plan['data_dir'])/'summary.json')
    if sha(Path(plan['data_dir'])/'summary.json') != plan['data_summary_sha256'] or data != plan['data_summary']:
        raise ValueError('WOMD population summary changed')
    validate_matched_contract(plan, reference, data)
    for name, digest in data['asset_sha256'].items():
        if sha(Path(plan['data_dir'])/name) != digest:
            raise ValueError('WOMD preserved cache registration changed: '+name)
    if (sha(plan['acceptance']['path']) != plan['acceptance']['sha256']
            or sha(output/'code_alignment.diff') != plan['alignment_diff_sha256']):
        raise ValueError('Acceptance or alignment record changed')
    for relative, digest in plan['source_sha256'].items():
        if sha(Path(plan['source_snapshot'])/relative) != digest:
            raise ValueError('Matched frozen source changed: '+relative)
    if frozen and Path(__file__).resolve() != Path(plan['source_snapshot'])/'spear_current/womd_aligned.py':
        raise ValueError('Run only from the registered matched snapshot')
    return plan


def validate_completion(plan):
    from .av2_formal_data import iter_role_parts
    folder = Path(plan['output_dir'])/f"seed{plan['seed']}"
    done = read(folder/'completion.json')
    if ((folder/'failure.json').exists() or done['status'] != 'MATCHED_PILOT_COMPLETE_REVIEW_REQUIRED'
            or done['seed'] != plan['seed']
            or done['plan_sha256'] != sha(Path(plan['output_dir'])/'plan.json')
            or sha(folder/'generator.pt') != done['generator_file_sha256']):
        raise ValueError('Incomplete or contradictory matched pilot receipt')
    for role in ('dev','audit'):
        path = folder/f'evaluation_{role}/summary.json'
        result = read(path)
        if (sha(path) != done['result_sha256'][role] or result['status'] != 'EVALUATION_COMPLETE'
                or result['dataset'] != 'WOMD' or result['source_dataset_split'] != 'training_20s'
                or result['targets'] != plan['data_summary'][role]['targets']
                or result['scenes'] != plan['data_summary'][role]['scenes']):
            raise ValueError('Matched evaluation failed integrity checks: '+role)
        for name in ('targets.csv','clean_pair_scores.csv','triggered_pair_scores.csv'):
            if sha(path.parent/name) != done['evaluation_csv_sha256'][role][name]:
                raise ValueError('Changed evaluation CSV: '+role+'/'+name)
        if sha(path.parent/'targets.csv') != done['target_csv_sha256'][role]:
            raise ValueError('Changed target CSV: '+role)
        expected = []
        for part in iter_role_parts(plan['data_dir'],role):
            target = part['target_index']
            if target is not None:
                ids = part['canonical_pair_track_ids'][target]
                expected.append((str(part['scenario_id']),int(target),str(ids[0]),str(ids[1])))
        with (path.parent/'targets.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        actual = [(r['scenario_id'],int(r['pair_index']),r['src_track_id'],r['dst_track_id']) for r in rows]
        if (actual != expected or len(actual) != len(set(actual))
                or any(int(r['target_pair_index']) != int(r['pair_index']) for r in rows)):
            raise ValueError('Evaluation target identities differ from preserved WOMD pool: '+role)
    return done


def worker(plan_path):
    import torch
    from .av2_formal import run_pipeline
    from .av2_formal_data import make_indices
    from .womd_pilot_training import cuda_preflight
    plan = verify(plan_path, frozen=True)
    output = Path(plan['output_dir'])
    folder = output/f"seed{plan['seed']}"
    folder.mkdir(exist_ok=True)
    if any((folder/name).exists() for name in ('selection','anchor','failure.json','completion.json')):
        raise FileExistsError('Do not overwrite partial WOMD training')
    started, training_started = time.monotonic(), False
    last_log = [0.]

    def status(state, **extra):
        write_json(folder/'status.json', dict(status=state, at=now(), pid=os.getpid(),
            seed=plan['seed'], elapsed_seconds=time.monotonic()-started,
            training_started=training_started, **extra))

    def progress(row):
        status('RUNNING', progress=row)
        if (time.monotonic()-last_log[0] > 30
                or not any(k in row for k in ('batch','batches_done'))):
            print('[womd-aligned] '+json.dumps(row, allow_nan=False), flush=True)
            last_log[0] = time.monotonic()

    try:
        torch.set_num_threads(1)
        while True:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
            free, _ = torch.cuda.mem_get_info(plan['device'])
            if free >= plan['minimum_cuda_free_bytes']:
                break
            status('WAITING_FOR_GPU', free_bytes=free)
            time.sleep(plan['gpu_poll_seconds'])
        write_json(folder/'cuda_preflight.json', dict(at=now(), **cuda_preflight(plan['device'])))
        expected = torch.load(Path(plan['data_dir'])/'indices.pt', map_location='cpu', weights_only=True)
        actual = make_indices(plan['data_dir'], plan['seed'])
        if set(actual) != set(expected) or any(not torch.equal(actual[k],expected[k]) for k in expected):
            raise ValueError('Shared pipeline would change existing WOMD clean supervision')
        training_started = True
        state_hash, results = run_pipeline(plan, plan['seed'], folder, progress)
        verify(plan_path, frozen=True)
        done = dict(status='MATCHED_PILOT_COMPLETE_REVIEW_REQUIRED', completed_at=now(),
            seed=plan['seed'], elapsed_seconds=time.monotonic()-started,
            plan_sha256=sha(plan_path), generator_state_sha256=state_hash,
            generator_file_sha256=sha(folder/'generator.pt'),
            result_sha256={r:sha(folder/f'evaluation_{r}/summary.json') for r in results},
            target_csv_sha256={r:sha(folder/f'evaluation_{r}/targets.csv') for r in results},
            evaluation_csv_sha256={r:{name:sha(folder/f'evaluation_{r}'/name)
                for name in ('targets.csv','clean_pair_scores.csv','triggered_pair_scores.csv')} for r in results},
            result_paths={r:str(folder/f'evaluation_{r}/summary.json') for r in results},
            reference_plan=plan['reference_plan'], poison_budget=plan['poison_budget'],
            fixed_scene_and_target_pool=True, test_arrays_loaded=False,
            recording_independence_proven=False, automatic_formal_runs_started=False)
        write_json(folder/'completion.json', done)
        validate_completion(plan)
        status('MATCHED_PILOT_COMPLETE_REVIEW_REQUIRED')
        print('[womd-aligned] complete', flush=True)
    except BaseException as exc:
        failure = dict(status='FAILED_REVIEW_REQUIRED', at=now(), seed=plan['seed'],
                       error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write_json(folder/'failure.json', failure)
        status('FAILED_REVIEW_REQUIRED', failure=failure)
        raise


def supervise(plan_path):
    plan = verify(plan_path, frozen=True)
    output = Path(plan['output_dir'])
    folder = output/f"seed{plan['seed']}"
    folder.mkdir(exist_ok=True)
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((folder/name).exists() for name in ('run.log','completion.json','failure.json')):
            raise FileExistsError('Existing worker artifacts; never duplicate or overwrite')
        command = [sys.executable,'-B','-m','spear_current.womd_aligned','--worker',str(plan_path)]
        with (folder/'run.log').open('x') as log:
            child = subprocess.Popen(command, cwd=plan['source_snapshot'], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            write_json(folder/'launch.json', dict(pid=child.pid, at=now(), command=command))
            while True:
                code = child.poll()
                current = read(folder/'status.json') if (folder/'status.json').exists() else {}
                state = ('WAITING_FOR_GPU' if current.get('status')=='WAITING_FOR_GPU'
                         else 'RUNNING' if current.get('training_started') else 'STARTING_WORKER')
                write_json(output/'status.json', dict(status=state, at=now(), pid=os.getpid(),
                    worker_pid=child.pid, seed=plan['seed'], training_started=current.get('training_started',False),
                    completed=0, total=1, worker_status=current))
                if code is not None:
                    break
                time.sleep(10)
        try:
            if code != 0:
                raise RuntimeError('Matched worker exited with code '+str(code))
            done = validate_completion(plan)
        except Exception as exc:
            failure = dict(status='FAILED_REVIEW_REQUIRED', at=now(), returncode=code, error=str(exc),
                           completed=0,total=1,worker_log=str(folder/'run.log'))
            write_json(output/'failure.json', failure)
            write_json(output/'status.json', failure)
            raise
        done = dict(done, completed=1,total=1,worker_completion=str(folder/'completion.json'))
        write_json(output/'completion.json', done)
        write_json(output/'status.json', dict(status=done['status'],at=now(),completed=1,total=1,
                                              completion=str(output/'completion.json')))


def launch(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = verify(plan_path)
    output = Path(plan['output_dir'])
    if (output/'launch.json').exists():
        raise FileExistsError('Matched pilot already launched')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command = [sys.executable,'-B','-m','spear_current.womd_aligned','--run',str(plan_path)]
    with (output/'run.log').open('x') as log:
        child = subprocess.Popen(command,cwd=plan['source_snapshot'],env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record = dict(pid=child.pid,at=now(),command=command,source_snapshot=plan['source_snapshot'])
    write_json(output/'launch.json',record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare',action='store_true')
    actions.add_argument('--launch',type=Path)
    actions.add_argument('--run',type=Path)
    actions.add_argument('--worker',type=Path)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--reference-plan',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--acceptance',type=Path)
    args=parser.parse_args()
    if args.prepare:
        if args.acceptance is None:
            parser.error('--prepare requires --acceptance')
        print(register(args.output_dir,args.reference_plan,args.acceptance))
    elif args.launch:
        print(json.dumps(launch(args.launch),indent=2))
    elif args.run:
        supervise(args.run)
    else:
        worker(args.worker)


if __name__=='__main__':
    main()
