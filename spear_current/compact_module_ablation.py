"""Registered CUDA ablations of the current uniform/semantic-support route.

Offline AV2/self-owned models. Reads the bound TRAIN/internal-DEV cache only.
Old results and defaults remain intact. This is a fresh paired reconstruction
with historically fixed feature lists, not a replay of the old 96% checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

from .full_ablation import read, write, sha, state, event, configure_cuda
from .support_replication import now


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'compact_module_ablation_v1'
SEEDS = (20260621, 20260622, 20260623)
VARIANTS = ('full', 'matched_random_pairs', 'random_features', 'no_generator_neighborhood',
    'no_residual', 'self_only', 'no_kl')
STAGES = ('base', 'residual', 'support_surrogate', 'continuation_a', 'continuation_b')
EPOCHS = {stage: 200 for stage in STAGES}


def matrix():
    return [dict(seed=seed, variant=variant) for seed in SEEDS for variant in VARIANTS]


def expected_steps(variant, stage_epochs=200, victim_epochs=200):
    if variant not in VARIANTS:
        raise ValueError('Unknown current-module ablation')
    return dict(generator=(1 if variant == 'no_residual' else 4)*stage_epochs,
        surrogate=5*stage_epochs, discriminator=0, victim=victim_epochs, selector=0)


def retained_sources(plan):
    """Reuse only exact, completed conditions; never choose by ASR."""
    from . import current_module_ablation as parent_runtime
    from .support_replication import _verify_prediction_artifacts
    roots=(ROOT/'results/current_module_ablation_20261103_v1',
           ROOT/'results/current_module_ablation_20261103_recovery_v1')
    reusable={'full','matched_random_pairs','no_residual','self_only','no_kl'}
    retained=[];bindings={}
    for item in matrix():
        if item['variant'] not in reusable:continue
        for root in roots:
            folder=case_dir(root,**item)
            if not (folder/'completion.json').exists():continue
            parent_plan=root/'plan.json'
            parent_runtime.verify(parent_plan)
            receipt=read(folder/'completion.json');result=read(folder/'result.json')
            parent_sha=sha(parent_plan)
            if (receipt['status']!='CASE_COMPLETE' or result.get('status')!='CASE_COMPLETE'
                    or receipt['plan_sha256']!=parent_sha
                    or receipt.get('seed')!=item['seed'] or receipt.get('variant')!=item['variant']
                    or result.get('seed')!=item['seed'] or result.get('variant')!=item['variant']
                    or receipt['result_sha256']!=sha(folder/'result.json')
                    or result['optimizer_steps']!=expected_steps(item['variant'])
                    or result['selection']!=plan['policies'][str(item['seed'])][item['variant']]
                    or result['output_mapping']!='identity'):
                raise ValueError('Completed parent does not match compact condition')
            _verify_prediction_artifacts(result['metrics'])
            retained.append(dict(**item,source=str(folder.resolve()),parent_plan_sha256=parent_sha,
                result_sha256=sha(folder/'result.json'),original_training_not_repeated=True))
            for file in (parent_plan,folder/'completion.json',folder/'result.json'):
                bindings[str(file.resolve())]=sha(file)
            for key in ('raw_targets','raw_clean_scores','raw_triggered_pair_scores'):
                r=result['metrics'][key];bindings[r['path']]=r['sha256']
            break
    return retained,bindings


def prepare(output):
    import torch
    from .support_replication import protocol, _cache
    from .runtime import load_bundle
    from .compact_ablation_data import discover_supports, build_policy
    from .compact_ablation_training import VARIANTS as trainer_variants, STAGES as trainer_stages
    if tuple(trainer_variants) != VARIANTS or tuple(trainer_stages) != STAGES:
        raise ValueError('Data, trainer and workflow variant/stage contracts differ')
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'results') or output == ROOT/'results':
        raise ValueError('Require a new child of results')
    parent = protocol()
    inputs = _cache(parent)
    discovered = discover_supports()
    masks = {k:torch.as_tensor(v,dtype=torch.bool).tolist()
             for k,v in discovered['source_masks'].items()}
    policies = {str(seed):{variant:build_policy(inputs,variant,seed,masks)['record']
                           for variant in VARIANTS} for seed in SEEDS}
    bound = {**parent['bound_files'], **discovered['bound_files']}
    bundle = load_bundle(parent['manifest'], verify_assets=False)
    for key in ('clean_proxy1','clean_proxy2'):
        asset = bundle.asset_path(key);bound[str(asset)] = sha(asset)
    sources = {str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'spear_current').glob('*.py'))}
    sources['author_native_runs/dpgba_queue.py'] = sha(ROOT/'author_native_runs/dpgba_queue.py')
    plan = dict(version=VERSION, created_utc=now(), manifest=parent['manifest'],
        prepared_inputs=parent['prepared_inputs'], inputs_sha256=parent['inputs_sha256'],
        bound_files=bound, sources=sources, source_masks=masks,
        support_sources=discovered['records'], policies=policies,
        seeds=list(SEEDS),variants=list(VARIANTS),matrix=matrix(),stage_epochs=EPOCHS,
        victim_epochs=200, optimizer_steps={v:expected_steps(v) for v in VARIANTS},
        route='six ablations plus Full: RN/RF/NE, no new residual, self-only residual, NoKL',
        generator_neighborhood_ablation='shared-weight pointwise MLP conditioning and self-only residual; original classifier graph preserved',
        random_feature_ablation='single uniform24 draw from legal28 per seed; not random semantic groups',
        auxiliary_budget_control='existing base-continuation results retained as auxiliary only; no new primary run',
        causal_scope='current modules under fixed historical field lists; fresh G/shadow/victim; not historical96pct checkpoint replay',
        hypothesis='Separate attachment identities, fixed support, new residual, neighbor input, training opportunity and clean-response KL',
        support_selector_retrained=False, feature_lists_have_historical_attack_and_KL_origin=True,
        no_attack_checkpoint_loaded=True, original_list='reuse historical GCN-ranked26, not reranked',
        random_attachment_seed='equals training seed; one motion-stratified draw per seed, quota11Moving/15Slow',
        common_background='same512 within seed; exclude clean supervision and union of original/random attachments',
        no_residual_note='strict deletion: G200, S1000; removed updates not compensated',
        no_kl_note='KL0 before first new G update; fixed support itself has historical KL origin',
        dynamic_labels='fixed5m/1s/4mps2; task definition, not removed as a same-task ablation',
        training=dict(generator_lr=.0002,shadow_lr=.0002,weight_decay=.0005,
            target_weight=1.,similarity_weight=.1,inner_steps=1,
            output_mode='identity',distribution_weight=0.,
            support_schedule='legal28 base/residual; final fixed24 afterwards',
            generator_checkpoint='last epoch per fixed stage, no ASR selection',
            victim_checkpoint='earliest maximum normalDEV AUC',
            residual_initialization_rng='isolated constructor stream; shared base training RNG across architectures'),
        evaluation=parent['evaluation'],historical_gates=parent['gates'],
        screen=dict(overall_asr_min=.80,moving_asr_min=.80,adjacent_fp_max_exclusive=.05,
            diagnostic_only=True,no_automatic_promotion=True),
        execution=dict(device='cuda:0',dtype='float32',amp=False,tf32=False,max_concurrent=2,
            no_cpu_fallback=True,kill_external_jobs=False),
        resource_policy={**parent['resource_policy'],'min_cuda_free_bytes':12*1024**3,
            'second_min_cuda_free_bytes':16*1024**3,'stable_probes':2,'cold_start_gap_seconds':15,'wait_first_epoch':True},
        preflight=dict(seed=SEEDS[0],variants=list(VARIANTS),epochs_per_stage=2,
            victim_epochs=2,asr_evaluation=False,weights_reused=False),
        failure_policy='numerical or CUDA OOM: stop that case, retain and continue other registered cases; identity/source/permission errors stop queue; no auto-retry',
        performance_policy='complete all predeclared cases; preserve poor scores; no redraws, retuning, extra seeds or automatic deployment',
        original_validation_arrays_loaded=False,original_test_arrays_loaded=False,
        audit_arrays_loaded=False,confirmation_arrays_loaded=False)
    retained, retained_bindings = retained_sources(plan)
    plan['retained_cases'] = retained
    plan['bound_files'].update(retained_bindings)
    plan['primary_case_count'] = len(matrix())
    plan['primary_ablations'] = list(VARIANTS[1:])
    validate(plan)
    output.mkdir(parents=True,exist_ok=False)
    for row in retained:
        link = case_dir(output,row['seed'],row['variant'])
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(row['source'],target_is_directory=True)
    write(output/'plan.json',plan)
    with tarfile.open(output/'source_snapshot.tar','x') as archive:
        for relative in sources:archive.add(ROOT/relative,arcname=relative)
    write(output/'registration.json',dict(plan_sha256=sha(output/'plan.json'),
        source_snapshot_sha256=sha(output/'source_snapshot.tar'),utc=now(),
        training_jobs_started=0,model_checkpoints_deserialized=False))
    state(output,'REGISTERED_PREFLIGHT_REQUIRED',planned_cases=len(matrix()))
    print(output/'plan.json',flush=True)
    return output/'plan.json'


def validate(plan):
    import torch
    from .support_schema import SELECTED_COORDINATES, EXCLUDED_COORDINATES
    from .support_replication_runtime import _mask
    if (plan['version'] != VERSION or plan['seeds'] != list(SEEDS)
            or plan['variants'] != list(VARIANTS) or plan['matrix'] != matrix()
            or plan['stage_epochs'] != EPOCHS or plan['victim_epochs'] != 200
            or plan['optimizer_steps'] != {v:expected_steps(v) for v in VARIANTS}
            or plan['execution'] != dict(device='cuda:0',dtype='float32',amp=False,tf32=False,
                max_concurrent=2,no_cpu_fallback=True,kill_external_jobs=False)
            or plan['preflight'] != dict(seed=SEEDS[0],variants=list(VARIANTS),epochs_per_stage=2,
                victim_epochs=2,asr_evaluation=False,weights_reused=False)
            or plan['training']['output_mode'] != 'identity'
            or plan['training']['distribution_weight'] != 0.):
        raise ValueError('Current module training contract changed')
    _mask(plan['source_masks']['semantic'])
    retained_keys=[(r['seed'],r['variant']) for r in plan.get('retained_cases',[])]
    reusable={'full','matched_random_pairs','no_residual','self_only','no_kl'}
    if (len(set(retained_keys))!=len(retained_keys)
            or any(seed not in SEEDS or variant not in reusable for seed,variant in retained_keys)):
        raise ValueError('Duplicate or ineligible retained case')
    if set(plan['policies']) != {str(s) for s in SEEDS}:
        raise ValueError('Missing paired seed policies')
    for seed in SEEDS:
        policies=plan['policies'][str(seed)]
        if set(policies) != set(VARIANTS):raise ValueError('Missing paired variant policies')
        backgrounds=set();clean=set();candidates=set();attachments={}
        for variant,row in policies.items():
            # The data module validates identities, support groups and hashes;
            # execution recomputes the full record from the immutable cache.
            mask=torch.as_tensor(row['hard_support'],dtype=torch.bool)
            cap=28 if variant=='no_support28' else 24
            if (row['variant'] != variant or mask.shape != (30,) or int(mask.sum()) != cap
                    or bool(mask[torch.tensor([c in EXCLUDED_COORDINATES for c in SELECTED_COORDINATES])].any())
                    or row['attachment_count'] != 26 or row['candidate_count'] != 6549
                    or row['attachment_motion_counts'] != dict(moving=11,slow=15)
                    or len(set(row['attachment_indices'])) != 26
                    or len(set(row['outer_background_indices'])) != 512
                    or row['selection_uses_dev'] or row['selection_uses_audit_or_test']
                    or row['selection_uses_model_predictions']
                    or row['feature_selector_retrained']):
                raise ValueError('Candidate/support/prediction scope changed')
            backgrounds.add(tuple(row['outer_background_indices']))
            clean.add(row['idx_train_sha256']);candidates.add(row['candidate_pool_sha256'])
            attachments[variant]=tuple(row['attachment_indices'])
        if len(backgrounds)!=1 or len(clean)!=1 or len(candidates)!=1:
            raise ValueError('Within-seed background or candidate pool changed')
        original=attachments['full']
        if any(a!=original for v,a in attachments.items() if v!='matched_random_pairs'):
            raise ValueError('Unexpected attachment change in a module ablation')
        union=set(original)|set(attachments['matched_random_pairs'])
        if union.intersection(next(iter(backgrounds))):raise ValueError('Background leaks paired attachments')


def verify(path,expected=None):
    path=Path(path).resolve()
    if path.name!='plan.json' or not path.parent.is_relative_to(ROOT/'results'):
        raise ValueError('Require a registered child results/plan.json')
    receipt=read(path.parent/'registration.json')
    if (sha(path)!=receipt['plan_sha256'] or expected is not None and sha(path)!=expected
            or sha(path.parent/'source_snapshot.tar')!=receipt['source_snapshot_sha256']):
        raise ValueError('Registered plan/source snapshot changed')
    plan=read(path);validate(plan)
    for file,digest in plan['bound_files'].items():
        if sha(file)!=digest:raise ValueError('Bound cache/clean asset/support changed: '+file)
    for file,digest in plan['sources'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Registered runtime changed: '+file)
    for row in plan.get('retained_cases',[]):
        link=case_dir(path.parent,row['seed'],row['variant'])
        if not link.is_symlink() or str(link.resolve())!=row['source']:
            raise ValueError('Retained primary case link changed')
    return plan


def case_dir(root,seed,variant,short=False):
    return Path(root)/('preflight' if short else f'seed{seed}')/variant


def root_error_type(error):
    """Keep numeric/OOM identity through a trainer's contextual RuntimeError."""
    seen=set()
    while error.__cause__ is not None and id(error) not in seen:
        seen.add(id(error));error=error.__cause__
    return type(error).__name__


def run_case(path,seed,variant,*,short=False,expected=None):
    import torch
    from .support_replication import _cache,acceptability,_verify_prediction_artifacts
    from .compact_ablation_data import build_policy
    from .compact_ablation_training import fit_case
    from .full_ablation_evaluation import fit_victim,evaluate_case
    from .runtime import load_bundle,restore_proxy,restore_clean_reference
    from .development_diagnostic import _input_hashes
    from .support_schema import SELECTED_COORDINATES
    path=Path(path).resolve();plan=verify(path,expected);expected=sha(path)
    if seed not in SEEDS or variant not in VARIANTS:raise ValueError('Unregistered case')
    if short and seed!=SEEDS[0]:raise ValueError('Short tests use the registered first seed only')
    output=case_dir(path.parent,seed,variant,short);output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    try:
        device=configure_cuda(seed);state(output,'LOADING_BOUND_CACHE',seed=seed,variant=variant)
        inputs=_cache(plan);before=_input_hashes(inputs)
        selection=build_policy(inputs,variant,seed,plan['source_masks'])
        if selection['record']!=plan['policies'][str(seed)][variant]:
            raise ValueError('Actual field/attachment policy differs from registration')
        write(output/'selection.json',selection['record'])
        cap=28 if variant=='no_support28' else 24
        bundle=load_bundle(plan['manifest'],verify_assets=False)
        proxies=[restore_proxy(bundle,i).to(device).eval().requires_grad_(False) for i in (1,2)]
        train={k:v.to(device) if torch.is_tensor(v) else v for k,v in inputs['combined']['train'].items()}
        train['context']=inputs['context'].to(device)
        indices={k:selection[k].to(device) for k in ('idx_train','idx_attach','fixed_outer')}
        config={**plan['training'],'stage_epochs':{s:2 if short else 200 for s in STAGES}}
        def progress(row):
            if row['epoch'] in (0,1,49,99,149,199):event(output,phase='generator_training',record=row)
        state(output,'CUDA_FRESH_MODULE_TRAINING',seed=seed,variant=variant)
        trained=fit_case(train,indices,proxies,torch.tensor(SELECTED_COORDINATES,device=device),
            selection['support'].to(device),seed=seed,variant=variant,output_dir=output/'training',
            config=config,progress=progress)
        steps=expected_steps(variant,2 if short else 200,2 if short else 200)
        if (trained['generator_steps']!=steps['generator'] or trained['surrogate_steps']!=steps['surrogate']
                or trained['discriminator_steps']!=0
                or any(s['output_mode']!='identity' or s['distribution_weight']!=0 for s in trained['stages'])):
            raise ValueError('Actual optimizer updates or output mapping differ')
        generator=trained['generator'].eval().requires_grad_(False)
        state(output,'CUDA_FRESH_VICTIM_TRAINING',seed=seed,variant=variant)
        victim,victim_record=fit_victim(generator,inputs,selection,bundle,seed=seed,
            output_dir=output/'victim',epochs=steps['victim'],device=str(device),support_capacity=cap)
        if short:report=None
        else:
            state(output,'CUDA_FIXED_DEVELOPMENT_EVALUATION',seed=seed,variant=variant)
            reference=restore_clean_reference(bundle).to(device).eval().requires_grad_(False)
            report=evaluate_case(generator,victim,reference,inputs,device=str(device),
                output_dir=output/'development',support_capacity=cap)
            _verify_prediction_artifacts(report)
        if _input_hashes(inputs)!=before:raise ValueError('Training/evaluation mutated original inputs')
        verify(path,expected)
        result=dict(status='PREFLIGHT_PASS' if short else 'CASE_COMPLETE',seed=seed,variant=variant,
            metrics=report,acceptability=None if short else acceptability(report),
            selection=selection['record'],support_capacity=cap,initialization=trained['initialization'],
            stages=trained['stages'],optimizer_steps=steps,victim_record=victim_record,
            no_attack_checkpoint_loaded=True,support_selector_retrained=False,
            independent_full_pipeline_retraining=False,output_mapping='identity',distribution_weight=0.,
            source_scope=plan['causal_scope'],device=str(device),dtype='float32',
            original_validation_arrays_loaded=False,original_test_arrays_loaded=False,
            elapsed_seconds=time.monotonic()-began,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(output/'result.json',result)
        write(output/'completion.json',dict(status=result['status'],seed=seed,variant=variant,
            plan_sha256=expected,result_sha256=sha(output/'result.json'),asr_evaluated=not short,
            short_weights_reused=False,utc=now()))
        state(output,result['status'],seed=seed,variant=variant)
        return result
    except BaseException as error:
        write(output/'failure.json',dict(error_type=type(error).__name__,
            root_error_type=root_error_type(error),error=repr(error),
            traceback=traceback.format_exc(),seed=seed,variant=variant,short=short,utc=now()))
        state(output,'FAILED',seed=seed,variant=variant,error=repr(error));raise


def paired_checks(results):
    """Weights/RNG are paired; losses and resulting trained states may differ."""
    baseline=results['full']
    for variant,row in results.items():
        for key in ('fresh_model_sha256','initialization_rng_sha256',
                    'post_initialization_rng_sha256','clean_proxy_sha256'):
            if row['initialization'][key]!=baseline['initialization'][key]:
                raise ValueError('Unpaired base initialization/RNG/proxy: '+variant+' / '+key)
        for key in ('initialization_digest','initial_state_sha256','fit_rng_digest'):
            if row['victim_record'][key]!=baseline['victim_record'][key]:
                raise ValueError('Unpaired victim initialization/RNG: '+variant+' / '+key)
        if row['selection']['outer_background_sha256']!=baseline['selection']['outer_background_sha256']:
            raise ValueError('Within-seed background differs')


def aggregate(path,expected):
    from .support_replication import metric_values,acceptability,_verify_prediction_artifacts
    path=Path(path).resolve();plan=verify(path,expected);root=path.parent
    rows=[];failed=[];by_seed={s:{} for s in SEEDS};identities=None;reference=None
    for item in matrix():
        folder=case_dir(root,**item)
        if (folder/'failure.json').exists() and not (folder/'completion.json').exists():
            failed.append(read(folder/'failure.json'));continue
        receipt,result=read(folder/'completion.json'),read(folder/'result.json')
        origin = next((r for r in plan['retained_cases'] if r['seed']==item['seed'] and r['variant']==item['variant']), None)
        receipt_plan = origin['parent_plan_sha256'] if origin else expected
        if (receipt['status']!='CASE_COMPLETE' or result.get('status')!='CASE_COMPLETE'
                or receipt['plan_sha256']!=receipt_plan
                or receipt.get('seed')!=item['seed'] or receipt.get('variant')!=item['variant']
                or result.get('seed')!=item['seed'] or result.get('variant')!=item['variant']
                or receipt['result_sha256']!=sha(folder/'result.json')
                or origin is not None and sha(folder/'result.json')!=origin['result_sha256']
                or result['optimizer_steps']!=expected_steps(item['variant'])
                or result['selection']!=plan['policies'][str(item['seed'])][item['variant']]
                or result['acceptability']!=acceptability(result['metrics'])):
            raise ValueError('Case completion contract failed: '+str(item))
        ids=_verify_prediction_artifacts(result['metrics'])
        ref=result['metrics']['model_state_sha256']['clean_reference']
        if identities is None:identities,reference=ids,ref
        if ids!=identities or ref!=reference:raise ValueError('Evaluation target/reference changed')
        by_seed[item['seed']][item['variant']]=result
        victim=result['metrics']['models']['victim']
        rows.append(dict(**item,**metric_values(result['metrics']),
            absolute_asr=victim['targets']['all']['absolute_asr'],
            conditional_flip=victim['targets']['all']['conditional_flip'],
            clean_activation=victim['targets']['all']['clean_activation_rate'],
            clean_pair_ap=victim['clean_pair_ap'],passed=result['acceptability']['passed'],
            generator_steps=result['optimizer_steps']['generator'],
            surrogate_steps=result['optimizer_steps']['surrogate'],support_capacity=result['support_capacity'],
            elapsed_seconds=result['elapsed_seconds'],source=str(folder/'result.json')))
    paired_verified_seeds=[];paired_unavailable_seeds=[]
    for seed,results in by_seed.items():
        if 'full' in results:
            paired_checks(results);paired_verified_seeds.append(seed)
        else:
            paired_unavailable_seeds.append(seed)
    summaries=[];differences=[]
    metrics=list(metric_values(next(iter(next(v for v in by_seed.values() if v).values()))['metrics'])) if rows else []
    for variant in VARIANTS:
        selected=[r for r in rows if r['variant']==variant]
        summaries.append(dict(variant=variant,completed=len(selected),planned=len(SEEDS),
            metrics={k:dict(mean=statistics.mean(r[k] for r in selected),
                sd=statistics.stdev(r[k] for r in selected) if len(selected)>1 else None)
                for k in metrics} if selected else {}))
        for seed,results in by_seed.items():
            if variant!='full' and variant in results and 'full' in results:
                a,b=(metric_values(results[v]['metrics']) for v in (variant,'full'))
                differences.append(dict(seed=seed,variant=variant,**{k:a[k]-b[k] for k in metrics}))
    if rows:
        with (root/'case_metrics.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if differences:
        with (root/'paired_differences.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(differences[0]));w.writeheader();w.writerows(differences)
    status='COMPACT_ABLATION_COMPLETE_REVIEW_REQUIRED' if not failed else 'COMPACT_ABLATION_COMPLETE_WITH_FAILURES_REVIEW_REQUIRED'
    report=dict(status=status,plan_sha256=expected,planned_cases=len(matrix()),completed_cases=len(rows),
        failed_cases=failed,per_variant=summaries,paired_differences=differences,
        paired_initialization_checks=not paired_unavailable_seeds,
        paired_verified_seeds=paired_verified_seeds,paired_unavailable_seeds=paired_unavailable_seeds,
        scope=plan['causal_scope'],support_selector_retrained=False,
        retained_completed=len(plan['retained_cases']),new_cases=len(matrix())-len(plan['retained_cases']),
        no_automatic_promotion=True,utc=now())
    write(root/'aggregate.json',report)
    write(root/'completion.json',dict(status=status,plan_sha256=expected,
        aggregate_sha256=sha(root/'aggregate.json'),completed_cases=len(rows),failed_cases=len(failed),utc=now()))
    state(root,status,completed_cases=len(rows),failed_cases=len(failed))
    return report


def launch(path):
    from .support_replication import _env
    path=Path(path).resolve();verify(path);expected=sha(path);root=path.parent
    write(root/'launch_claim.json',dict(pid=os.getpid(),plan_sha256=expected,utc=now()))
    command=[sys.executable,'-B','-u','-m','spear_current.compact_module_ablation',
             '--queue','--plan',str(path),'--expected-sha256',expected]
    with (root/'queue.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=_env(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,command=command,plan_sha256=expected,utc=now())
    write(root/'launch.json',record);print(record,flush=True);return record


def queue(path, expected):
    import torch
    from .support_replication import _env,_resources
    from .current_ablation_recovery import is_resource_oom,warm_started
    path=Path(path).resolve();plan=verify(path,expected);root=path.parent
    write(root/'execution_claim.json',dict(pid=os.getpid(),plan_sha256=expected,utc=now()))
    active=[];completed=[{k:r[k] for k in ('seed','variant')} for r in plan['retained_cases']]
    failed=[];stable=0;last_launch=-float('inf');began=time.monotonic()
    def interrupt(signum,frame):raise RuntimeError('Compact queue interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def guard():
        verify(path,expected)
        if (root/'CANCEL').exists():raise RuntimeError('Compact ablation cancelled')
    try:
        preflight={}
        for variant in VARIANTS:
            while True:
                guard();resource=_resources(plan)
                if resource['ready']:break
                state(root,'WAITING_FOR_GPU_PREFLIGHT',resources=resource,variant=variant)
                if time.monotonic()-began>72*3600:raise TimeoutError('No GPU headroom')
                time.sleep(5)
            state(root,'CUDA_ENGINEERING_PREFLIGHT',variant=variant)
            preflight[variant]=run_case(path,SEEDS[0],variant,short=True,expected=expected)
            gc.collect();torch.cuda.empty_cache()
        paired_checks(preflight)
        write(root/'preflight/completion.json',dict(status='PREFLIGHT_PASS',variants=list(VARIANTS),
            plan_sha256=expected,asr_evaluation=False,weights_reused=False,paired_initializations=True,utc=now()))
        (root/'logs').mkdir()
        done={(r['seed'],r['variant']) for r in completed}
        pending=[item for item in matrix() if (item['seed'],item['variant']) not in done]
        while pending or active:
            guard()
            for job in list(active):
                code=job['process'].poll()
                if code is None:continue
                job['log'].close();active.remove(job)
                if code:
                    file=case_dir(root,job['seed'],job['variant'])/'failure.json'
                    failure=read(file) if file.exists() else {}
                    if is_resource_oom(failure) or failure.get('root_error_type')=='FloatingPointError':
                        failed.append(dict(seed=job['seed'],variant=job['variant'],exit_code=code));continue
                    raise RuntimeError('Compact case integrity/process failure '+str(job['variant'])+f' exit={code}')
                completed.append(dict(seed=job['seed'],variant=job['variant']))
            resources=None
            warm=all(warm_started(case_dir(root,j['seed'],j['variant'])) for j in active)
            if pending and len(active)<2 and warm and time.monotonic()-last_launch>=15:
                minimum=16*1024**3 if active else 12*1024**3
                probe_plan={**plan,'resource_policy':{**plan['resource_policy'],'min_cuda_free_bytes':minimum}}
                resources=_resources(probe_plan);stable=stable+1 if resources['ready'] else 0
                if stable>=2:
                    item=pending.pop(0);stable=0
                    command=[sys.executable,'-B','-u','-m','spear_current.compact_module_ablation',
                        '--case',item['variant'],'--seed',str(item['seed']),'--plan',str(path),'--expected-sha256',expected]
                    log=(root/'logs'/f"seed{item['seed']}_{item['variant']}.log").open('x')
                    try:child=subprocess.Popen(command,cwd=ROOT,env=_env(),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:log.close();raise
                    active.append(dict(**item,process=child,log=log));last_launch=time.monotonic()
                    write(root/'logs'/f"seed{item['seed']}_{item['variant']}_launch.json",
                        dict(**item,pid=child.pid,resources=resources,minimum_free_bytes=minimum,utc=now()))
            else:stable=0
            if time.monotonic()-began>72*3600:raise TimeoutError('Compact resource/runtime timeout')
            state(root,'CUDA_COMPACT_ABLATIONS_RUNNING' if active else 'WAITING_FOR_GPU_HEADROOM',
                planned_cases=len(matrix()),retained_cases=len(plan['retained_cases']),completed_cases=len(completed),
                failed_cases=failed,pending_cases=len(pending),
                active=[{k:j[k] for k in ('seed','variant')}|{'pid':j['process'].pid} for j in active],resources=resources)
            if pending or active:time.sleep(5)
        aggregate(path,expected)
    except BaseException as error:
        for job in active:
            if job['process'].poll() is None:os.killpg(job['process'].pid,signal.SIGTERM)
            job['log'].close()
        write(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            own_children_only_terminated=True,utc=now()));state(root,'FAILED',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',type=Path);mode.add_argument('--launch',action='store_true')
    mode.add_argument('--queue',action='store_true');mode.add_argument('--case',choices=VARIANTS)
    parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--plan',type=Path)
    parser.add_argument('--expected-sha256');args=parser.parse_args()
    if args.prepare is not None:prepare(args.prepare)
    elif args.plan is None:parser.error('Require --plan')
    elif args.launch:launch(args.plan)
    elif args.queue:queue(args.plan,args.expected_sha256)
    elif args.seed is None:parser.error('Require --seed for a case')
    else:run_case(args.plan,args.seed,args.case,expected=args.expected_sha256)


if __name__=='__main__':main()
