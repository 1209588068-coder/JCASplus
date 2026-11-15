"""Fresh CUDA paired trial: historical attachments vs motion-matched random.

Offline AV2/self-owned GNN robustness research. The frozen historical attack
generator is never restored. Selection uses only the original negative TRAIN
candidate pool and observation motion strata. Existing results are preserved.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from .full_ablation import read, write, sha, state, event, configure_cuda
from .support_replication import now


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'fresh_unranked_pair_pilot_v1'
SEED = 20260621
ARMS = ('historical', 'stratified_random')
TRAINING_VARIANT = 'no_distribution_no_domain'
STAGES = ('base', 'residual', 'support_surrogate', 'domain', 'distribution')
STAGE_EPOCHS = {stage: 200 for stage in STAGES}
OPTIMIZER_STEPS = dict(generator=800, surrogate=1000, discriminator=0, victim=200, selector=0)


def prepare(output):
    from .support_replication import protocol, _cache
    from .runtime import load_bundle
    from .unranked_selection import select_pair_indices
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'results') or output == ROOT/'results':
        raise ValueError('Require a new child of results')
    parent = protocol()
    inputs = _cache(parent)  # parent protocol verified our trusted cache bytes.
    selections = {arm:select_pair_indices(inputs, arm, SEED)['record'] for arm in ARMS}
    validate_selections(selections)
    bundle = load_bundle(parent['manifest'], verify_assets=False)
    bound = dict(parent['bound_files'])
    for key in ('clean_proxy1', 'clean_proxy2'):
        path = bundle.asset_path(key)
        bound[str(path)] = sha(path)
    sources = {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'spear_current').glob('*.py'))}
    sources['author_native_runs/dpgba_queue.py'] = sha(ROOT/'author_native_runs/dpgba_queue.py')
    plan = dict(version=VERSION, created_utc=now(), manifest=parent['manifest'],
        prepared_inputs=parent['prepared_inputs'], inputs_sha256=parent['inputs_sha256'],
        bound_files=bound, sources=sources, seed=SEED, arms=list(ARMS), selections=selections,
        source_mask=parent['runtime_bindings']['supports']['semantic']['hard_support'],
        source_support=parent['runtime_bindings']['supports']['semantic'],
        changed_variable='26 TRAIN attachment identities only; matched11Moving/15Slow and common512 outer background',
        hypothesis='Test the effect of removing prediction-based attachment ranking under fixed candidate and motion budgets',
        initialization='identical per-seed random GCN shadow and base MLP; no poisoned model checkpoint loaded',
        inherited='same clean-only proxies, normalizer, and historically learned semantic24 mask in both arms',
        generator_checkpoint='last epoch of every fixed stage; no ASR/DEV generator checkpoint selection',
        training_variant=TRAINING_VARIANT, stage_epochs=dict(STAGE_EPOCHS), victim_epochs=200,
        optimizer_steps=dict(OPTIMIZER_STEPS),
        training=dict(shadow_lr=.0002, generator_lr=.0002, weight_decay=.0005,
            target_weight=1., similarity_weight=.1, kl_weight=1., clean_proxies=2,
            residual='uniform neighbor mean; not learned attention',
            support_schedule='legal28 in base/residual, fixed semantic24 thereafter',
            output_mapping='identity in every stage; no clipping or domain projection',
            distribution_regularizer=False,
            stage_names_note='last two stage names inherited from tested trainer; here both are identity residual continuations, with D absent'),
        protocol_identity='prospective paired fresh reconstruction; not literal replay of historical96pct full30/multi-ancestor training',
        original_list_provenance='historical GCN-ranked list reused; new selection constructs no classifier and reads no prediction scores',
        common_background='TRAIN minus clean supervision minus union of both attachment lists; one fixed512 draw',
        clean_reference='same historical clean-only development model; not blind',
        evaluation=parent['evaluation'], historical_gates=parent['gates'],
        prospective_usability_screen=dict(overall_asr_min=.80, moving_asr_min=.80, adjacent_fp_max_exclusive=.05,
            note='user-accepted80pct exploratory screen; historical85pct checks still reported without revision; no automatic promotion'),
        performance_policy='complete and report both arms even if performance fails; no re-draws, no additional seeds or retuning',
        stop_policy='source, identity, permissions, nonfinite state, or child failure stops new launches; no automatic retry or CPU fallback',
        full_pipeline_replication=False, feature_selector_retrained=False,
        no_original_graph_reads=True, original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
        execution=dict(device='cuda:0', dtype='float32', amp=False, tf32=False, max_concurrent=2,
            no_cpu_fallback=True, bitwise_cuda_reproducibility_claim=False),
        resource_policy={**parent['resource_policy'], 'min_cuda_free_bytes':8*1024**3},
        preflight=dict(arms=list(ARMS), stage_epochs=2, victim_epochs=2, asr_evaluation=False, weights_reused=False))
    validate_contract(plan)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'plan.json', plan)
    with tarfile.open(output/'source_snapshot.tar', 'x') as archive:
        for relative in sources:
            archive.add(ROOT/relative, arcname=relative)
    write(output/'registration.json', dict(plan_sha256=sha(output/'plan.json'),
        source_snapshot_sha256=sha(output/'source_snapshot.tar'),utc=now(),
        trusted_train_dev_cache_loaded=True, original_graph_arrays_loaded=False,
        model_checkpoints_deserialized=False, training_jobs_started=0))
    state(output, 'REGISTERED_GPU_PREFLIGHT_REQUIRED', cases=2)
    print(str(output/'plan.json'), flush=True)
    return output/'plan.json'


def validate_selections(records):
    if set(records) != set(ARMS):
        raise ValueError('Require exactly the two paired selection arms')
    backgrounds, candidate_hashes = set(), set()
    for arm in ARMS:
        row = records[arm]
        if (row['arm'] != arm or row['attachment_count'] != 26
                or row['attachment_motion_counts'] != dict(moving=11, slow=15)
                or row['candidate_count'] != 6549 or len(row['attachment_indices']) != 26
                or len(set(row['attachment_indices'])) != 26
                or row['selection_uses_model_predictions'] is not False
                or row['selection_uses_dev'] is not False
                or row['selection_uses_audit_or_test'] is not False
                or row['common_background_shared_across_arms'] is not True
                or len(row['outer_background_indices']) != 512):
            raise ValueError('Candidate/attachment/motion/common-background selection contract changed')
        backgrounds.add(tuple(row['outer_background_indices']))
        candidate_hashes.add(row['candidate_pool_sha256'])
    if len(backgrounds) != 1 or len(candidate_hashes) != 1:
        raise ValueError('Two arms must share candidate pool and exact512 background')
    background = next(iter(backgrounds))
    union = set(records['historical']['attachment_indices']) | set(records['stratified_random']['attachment_indices'])
    if union.intersection(background):
        raise ValueError('Shared background contains paired attachments')


def validate_contract(plan):
    from .support_replication_runtime import _mask
    if (plan['version'] != VERSION or plan['seed'] != SEED or plan['arms'] != list(ARMS)
            or plan['training_variant'] != TRAINING_VARIANT or plan['stage_epochs'] != STAGE_EPOCHS
            or plan['victim_epochs'] != 200 or plan['optimizer_steps'] != OPTIMIZER_STEPS
            or plan['execution']['max_concurrent'] != 2
            or plan['preflight'] != dict(arms=list(ARMS), stage_epochs=2,
                victim_epochs=2, asr_evaluation=False, weights_reused=False)):
        raise ValueError('Paired training configuration changed')
    _mask(plan['source_mask'])
    validate_selections(plan['selections'])


def verify(path, expected=None):
    path = Path(path).resolve()
    if path.name != 'plan.json' or not path.parent.is_relative_to(ROOT/'results'):
        raise ValueError('Require registered results/plan.json')
    receipt = read(path.parent/'registration.json')
    if (sha(path) != receipt['plan_sha256'] or (expected is not None and sha(path) != expected)
            or sha(path.parent/'source_snapshot.tar') != receipt['source_snapshot_sha256']):
        raise ValueError('Plan/source snapshot changed')
    plan = read(path)
    validate_contract(plan)
    for filename, digest in plan['bound_files'].items():
        if sha(filename) != digest:
            raise ValueError('Bound clean asset/cache/provenance changed: '+filename)
    for relative, digest in plan['sources'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Registered source changed: '+relative)
    return plan


def run_case(path, arm, *, short=False, expected=None):
    import torch
    from .support_replication import _cache, acceptability, _verify_prediction_artifacts
    from .unranked_selection import select_pair_indices
    from .runtime import load_bundle, restore_proxy, restore_clean_reference
    from .full_ablation_training import fit_case
    from .full_ablation_evaluation import fit_victim, evaluate_case
    from .support_schema import SELECTED_COORDINATES
    from .development_diagnostic import _input_hashes
    path = Path(path).resolve()
    plan = verify(path, expected)
    expected, start = sha(path), time.monotonic()
    if arm not in ARMS:
        raise ValueError('Unregistered paired arm')
    output = path.parent/('preflight' if short else 'cases')/arm
    output.mkdir(parents=True, exist_ok=False)
    try:
        state(output, 'LOADING_VERIFIED_TRAIN_DEV', arm=arm, short=short)
        device = configure_cuda(SEED)
        inputs = _cache(plan)
        before = _input_hashes(inputs)
        selection = select_pair_indices(inputs, arm, SEED)
        if selection['record'] != plan['selections'][arm]:
            raise ValueError('Selection differs from preregistered draw')
        write(output/'selection.json', selection['record'])
        bundle = load_bundle(plan['manifest'], verify_assets=False)
        proxies = [restore_proxy(bundle,i).to(device).eval().requires_grad_(False) for i in (1,2)]
        train = {k:v.to(device) if torch.is_tensor(v) else v for k,v in inputs['combined']['train'].items()}
        train['context'] = inputs['context'].to(device)
        indices = {k:selection[k].to(device) for k in ('idx_train','idx_attach','fixed_outer')}
        coordinates = torch.tensor(SELECTED_COORDINATES, dtype=torch.long, device=device)
        support = torch.tensor(plan['source_mask'], dtype=torch.bool, device=device)
        config = dict(stage_epochs={s:2 if short else 200 for s in STAGES},
            generator_lr=.0002, shadow_lr=.0002, weight_decay=.0005,
            target_weight=1., similarity_weight=.1, inner_steps=1)

        def progress(row):
            if row['epoch'] in (0,1,49,99,149,199):
                event(output, phase='attack_training', record=row)

        state(output, 'CUDA_FRESH_GENERATOR_TRAINING', arm=arm, stage_epochs=config['stage_epochs'])
        trained = fit_case(train, indices, proxies, coordinates, support, seed=SEED,
            variant=TRAINING_VARIANT, output_dir=output/'training', config=config, progress=progress)
        expected_g, expected_s = (8,10) if short else (800,1000)
        if (trained['generator_steps'] != expected_g or trained['surrogate_steps'] != expected_s
                or trained['discriminator_steps'] != 0
                or any(s['output_mode'] != 'identity' or s['distribution_weight'] != 0 for s in trained['stages'])):
            raise ValueError('Actual optimization updates/mapping differ from registered route')
        generator = trained['generator'].eval().requires_grad_(False)
        state(output, 'CUDA_FRESH_VICTIM_TRAINING', arm=arm)
        victim, record = fit_victim(generator, inputs, selection, bundle, seed=SEED,
            epochs=2 if short else 200, device=str(device), output_dir=output/'victim')
        if short:
            report = None
        else:
            state(output, 'CUDA_DEVELOPMENT_EVALUATION', arm=arm)
            reference = restore_clean_reference(bundle).to(device).eval().requires_grad_(False)
            report = evaluate_case(generator, victim, reference, inputs, device=str(device), output_dir=output/'development')
            _verify_prediction_artifacts(report)
        if _input_hashes(inputs) != before:
            raise ValueError('Paired fitting/evaluation mutated original cache')
        verify(path, expected)
        result = dict(status='PREFLIGHT_PASS' if short else 'CASE_COMPLETE', arm=arm, seed=SEED,
            metrics=report, acceptability=None if short else acceptability(report),
            victim_record=record, initialization=trained['initialization'],
            stages=trained['stages'], selection=selection['record'],
            optimizer_steps=dict(generator=expected_g,surrogate=expected_s,discriminator=0,
                victim=2 if short else 200,selector=0),
            no_attack_checkpoint_loaded=True, clean_prefix_shared=True,
            feature_selector_retrained=False, historical_full30_replay=False,
            output_mapping='identity', device=str(device),
            original_test_arrays_loaded=False, original_validation_arrays_loaded=False,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            elapsed_seconds=time.monotonic()-start)
        write(output/'result.json', result)
        write(output/'completion.json', dict(status=result['status'], arm=arm,
            plan_sha256=expected,result_sha256=sha(output/'result.json'),
            asr_evaluated=not short, short_weights_reused=False, utc=now()))
        state(output, result['status'], arm=arm)
    except BaseException as error:
        write(output/'failure.json', dict(error=repr(error),traceback=traceback.format_exc(),utc=now()))
        state(output,'FAILED',arm=arm,error=repr(error))
        raise


def aggregate(path, expected):
    from .support_replication import metric_values, acceptability, _verify_prediction_artifacts
    path = Path(path).resolve()
    plan = verify(path, expected)
    root, rows, results = path.parent, [], {}
    identities, reference_hash = None, None
    for arm in ARMS:
        folder = root/'cases'/arm
        receipt, result = read(folder/'completion.json'), read(folder/'result.json')
        if (receipt['status'] != 'CASE_COMPLETE' or receipt['plan_sha256'] != expected
                or receipt['result_sha256'] != sha(folder/'result.json')
                or result['optimizer_steps'] != OPTIMIZER_STEPS
                or result['selection'] != plan['selections'][arm]
                or result['acceptability'] != acceptability(result['metrics'])):
            raise ValueError('Missing or inconsistent paired completion')
        current = _verify_prediction_artifacts(result['metrics'])
        current_ref = result['metrics']['model_state_sha256']['clean_reference']
        if identities is None:
            identities, reference_hash = current, current_ref
        if current != identities or current_ref != reference_hash:
            raise ValueError('Paired evaluation identity/reference changed')
        results[arm] = result
        rows.append(dict(arm=arm, **metric_values(result['metrics']),
            historical_gates_pass=result['acceptability']['passed'],
            elapsed_seconds=result['elapsed_seconds'],gpu_peak_allocated_bytes=result['gpu_peak_allocated_bytes']))
    old,new = (results[a] for a in ARMS)
    for key in ('fresh_model_sha256','initialization_rng_sha256','post_initialization_rng_sha256','clean_proxy_sha256'):
        if old['initialization'][key] != new['initialization'][key]:
            raise ValueError('Fresh attack initialization/clean proxies differ: '+key)
    for key in ('initialization_digest','initial_state_sha256','fit_rng_digest'):
        if old['victim_record'][key] != new['victim_record'][key]:
            raise ValueError('Fresh victim initialization/RNG differs: '+key)
    if old['selection']['outer_background_sha256'] != new['selection']['outer_background_sha256']:
        raise ValueError('Nonattachment outer background changed between arms')
    differences = {k:metric_values(new['metrics'])[k]-metric_values(old['metrics'])[k]
                   for k in metric_values(new['metrics'])}
    with (root/'case_metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = dict(status='UNRANKED_PAIR_PILOT_COMPLETE_REVIEW_REQUIRED',plan_sha256=expected,
        cases=rows,stratified_random_minus_historical=differences,
        matched_attack_and_victim_initializations=True, common_outer_background=True,
        original_cache_unchanged=True, single_seed=True, reused_development=True,
        given_fixed_historical_semantic24=True, historical96pct_replay=False,
        automatic_promotion=False, utc=now())
    write(root/'aggregate.json',summary)
    write(root/'completion.json',dict(status=summary['status'],plan_sha256=expected,
        aggregate_sha256=sha(root/'aggregate.json'),completed_cases=2,utc=now()))
    state(root,summary['status'],completed_cases=2)
    return summary


def launch(path):
    from .support_replication import _env
    path=Path(path).resolve();verify(path);expected=sha(path);root=path.parent
    write(root/'launch_claim.json',dict(pid=os.getpid(),plan_sha256=expected,utc=now()))
    command=[sys.executable,'-B','-u','-m','spear_current.unranked_pair_pilot',
             '--queue','--plan',str(path),'--expected-sha256',expected]
    with (root/'queue.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=_env(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,command=command,plan_sha256=expected,utc=now())
    write(root/'launch.json',receipt);print(receipt,flush=True);return receipt


def queue(path, expected):
    import torch
    from .support_replication import _env,_resources
    path=Path(path).resolve();plan=verify(path,expected);root=path.parent
    write(root/'execution_claim.json',dict(pid=os.getpid(),plan_sha256=expected,utc=now()))
    active=[]

    def interrupt(signum,frame):
        raise RuntimeError('Paired queue interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)

    def wait_resources():
        start=time.monotonic()
        while True:
            verify(path,expected)
            if (root/'CANCEL').exists():raise RuntimeError('Paired queue cancelled')
            sample=_resources(plan)
            if sample['ready']:return sample
            state(root,'WAITING_FOR_GPU_HEADROOM',resources=sample)
            if time.monotonic()-start>plan['resource_policy']['max_wait_seconds']:
                raise TimeoutError('No GPU headroom; no CPU fallback')
            time.sleep(5)
    try:
        for arm in ARMS:
            wait_resources();state(root,'GPU_ENGINEERING_PREFLIGHT',arm=arm)
            run_case(path,arm,short=True,expected=expected)
            gc.collect();torch.cuda.empty_cache()
        write(root/'preflight/completion.json',dict(status='PREFLIGHT_PASS',arms=list(ARMS),
            plan_sha256=expected,asr_evaluation=False,weights_reused=False,utc=now()))
        pending=list(ARMS);completed=[]
        while pending or active:
            verify(path,expected)
            if (root/'CANCEL').exists():raise RuntimeError('Paired queue cancelled')
            for job in list(active):
                code=job['process'].poll()
                if code is not None:
                    job['log'].close();active.remove(job)
                    if code:raise RuntimeError(f"Paired arm {job['arm']} exited {code}")
                    completed.append(job['arm'])
            if pending and len(active)<2:
                resource=wait_resources();arm=pending.pop(0)
                command=[sys.executable,'-B','-u','-m','spear_current.unranked_pair_pilot',
                    '--case',arm,'--plan',str(path),'--expected-sha256',expected]
                log=(root/(arm+'.log')).open('x')
                try:
                    child=subprocess.Popen(command,cwd=ROOT,env=_env(),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:
                    log.close();raise
                active.append(dict(arm=arm,process=child,log=log))
                write(root/(arm+'_launch.json'),dict(pid=child.pid,command=command,resources=resource,utc=now()))
            state(root,'CUDA_PAIRED_SELECTION_RUNNING',completed_cases=len(completed),
                pending=pending,active=[dict(arm=j['arm'],pid=j['process'].pid) for j in active])
            if pending or active:time.sleep(3)
        aggregate(path,expected)
    except BaseException as error:
        for job in active:
            if job['process'].poll() is None:os.killpg(job['process'].pid,signal.SIGTERM)
            job['log'].close()
        write(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            utc=now(),own_children_only_terminated=True))
        state(root,'FAILED',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',type=Path);mode.add_argument('--launch',action='store_true')
    mode.add_argument('--queue',action='store_true');mode.add_argument('--case',choices=ARMS)
    parser.add_argument('--plan',type=Path);parser.add_argument('--expected-sha256')
    args=parser.parse_args()
    if args.prepare is not None:prepare(args.prepare)
    elif args.plan is None:parser.error('Require --plan')
    elif args.launch:launch(args.plan)
    elif args.queue:queue(args.plan,args.expected_sha256)
    else:run_case(args.plan,args.case,expected=args.expected_sha256)


if __name__=='__main__':main()
