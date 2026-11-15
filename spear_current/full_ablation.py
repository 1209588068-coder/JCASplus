"""Registered GPU internal ablations; fresh attack starts, shared clean pretraining."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260621, 20260622, 20260623)
VARIANTS = ('full', 'random_nodes', 'random_features', 'self_only', 'base_only',
            'no_kl', 'single_kl', 'no_distribution', 'no_distribution_no_domain')
STAGES = ('base', 'residual', 'support_surrogate', 'domain', 'distribution')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, exclusive=True):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        temporary.replace(path)


def state(output, phase, **values):
    write(output / 'status.json', dict(status=phase, utc=utc(), pid=os.getpid(), **values), exclusive=False)


def event(output, **values):
    row = dict(utc=utc(), **values)
    with (output / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def configure_cuda(seed):
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Full ablation requires CUDA; no CPU fallback')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    return torch.device('cuda:0')


def prepare(output, manifest):
    import torch
    from .runtime import load_bundle
    from .full_ablation_data import prepare_inputs
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'results') or output == ROOT / 'results':
        raise ValueError('New output must be a child of JCAS/results')
    output.mkdir(parents=True, exist_ok=False)
    state(output, 'PREPARING_TRAIN_DEV_INPUTS')
    bundle = load_bundle(manifest, verify_assets=False)
    inputs = prepare_inputs(bundle)
    torch.save(inputs, output / 'prepared_inputs.pt')
    write(output / 'graph_access_ledger.json', inputs['ledger'])
    clean_keys = ('normalizer', 'indices', 'input_contract', 'support', 'feature_importance',
                  'context_binding', 'score_permutation', 'clean_proxy1', 'clean_proxy2', 'clean_reference')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'spear_current').glob('*.py'))}
    plan = dict(version='full_gpu_module_ablation_v1', created_utc=utc(),
        manifest=str(bundle.manifest_path), manifest_sha256=sha(bundle.manifest_path),
        clean_assets={k: dict(path=str(bundle.asset_path(k)), sha256=sha(bundle.asset_path(k))) for k in clean_keys},
        prepared_inputs=str(output / 'prepared_inputs.pt'), inputs_sha256=sha(output / 'prepared_inputs.pt'),
        sources=sources, seeds=list(SEEDS), variants=list(VARIANTS),
        cases=[dict(seed=s, variant=v) for s in SEEDS for v in VARIANTS], victim='GraphSage',
        stage_epochs={stage: 200 for stage in STAGES}, victim_epochs=200,
        clean_prefix='Retained clean-only proxies/reference; CPU-pretrained, GPU forwards; shared across seeds',
        attack_initialization='fresh per-seed GCN shadow and base generator; no retained attack checkpoints',
        support_policy='pure-clean SAGE24 moved before attack training; random grouped legal28 exact24',
        stage_policy='legal28 base/residual;24 from support_surrogate; domain projection from domain; D only distribution',
        kl_policy='dual mean total1 from base; single fixed proxy1 total1; no_kl disabled throughout',
        generator_checkpoint='final epoch at every stage; no ASR selection',
        victim_checkpoint='maximum normal DEV AUC strict improvement earliest tie',
        training_updates=dict(base_G=200, base_S=200, residual_G=200, residual_S=200,
            support_G=0, support_S=200, domain_G=200, domain_S=200,
            distribution_G=200, distribution_S=200, D=200, victim=200),
        discriminator_updates_by_variant={v: 0 if v in ('no_distribution', 'no_distribution_no_domain') else 200 for v in VARIANTS},
        hyperparameters=dict(generator_lr=.0002, shadow_lr=.0002, weight_decay=.0005,
            target_weight=1., similarity_weight=.1, kl_weight=1., distribution_weight=.1,
            discriminator_lr=.0002, discriminator_seed='seed+1000'),
        base_only_policy='base G trainable instead of residual in every post-base G opportunity',
        preprocessing='fixed TRAIN column normalizer; no_domain removes output projection only',
        execution=dict(device='cuda:0', precision='float32', amp=False, tf32=False,
            bitwise_reproducibility_claim=False, max_active_ablation_cases=1, concurrent_author_queues=True),
        resource_policy=dict(min_host_available_bytes=16 * 1024**3, min_cuda_free_bytes=8 * 1024**3,
                             allow_other_gpu_processes=True, max_wait_seconds=72 * 3600),
        evaluation=dict(train_scenes=512, dev_scenes=128, targets=112, moving_targets=33,
            reused_development=True, original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
            audit_arrays_loaded=False, blind_confirmation=False),
        historical_full_replay_claim=False,
        implementation_note='Prospective controlled staged reconstruction; unsuccessful historical branches/tuning not replayed',
        stop_policy='nonfinite state, source/identity mismatch or child failure stops queue; no automatic parameter retry',
        low_asr_policy='record and continue; no discarding low-effect configurations')
    write(output / 'plan.json', plan)
    state(output, 'PREREGISTERED_PREFLIGHT_REQUIRED', cases=27, plan_sha256=sha(output / 'plan.json'))
    print(json.dumps(dict(plan=str(output / 'plan.json'), sha256=sha(output / 'plan.json'))), flush=True)


def verify(plan_path, expected=None):
    plan_path = Path(plan_path).resolve()
    if expected is not None and sha(plan_path) != expected:
        raise ValueError('Registered plan changed')
    plan = read(plan_path)
    for path, digest in ((plan['manifest'], plan['manifest_sha256']),
                         (plan['prepared_inputs'], plan['inputs_sha256'])):
        if sha(path) != digest:
            raise ValueError('Registered input changed: ' + path)
    for item in plan['clean_assets'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Clean-only asset changed')
    for relative, digest in plan['sources'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Registered source changed: ' + relative)
    if plan['cases'] != [dict(seed=s, variant=v) for s in SEEDS for v in VARIANTS]:
        raise ValueError('Nine-by-three matrix changed')
    return plan


def load_case(plan, seed, variant, device):
    import torch
    from .runtime import load_bundle, restore_proxy
    from .full_ablation_data import select_indices, select_support
    # Own locally generated cache, verified by SHA before pickle deserialization.
    inputs = torch.load(plan['prepared_inputs'], map_location='cpu', weights_only=False)
    bundle = load_bundle(plan['manifest'], verify_assets=False)
    selection = select_indices(inputs, variant, seed)
    coordinates, support, support_record = select_support(bundle, variant, seed)
    train = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs['combined']['train'].items()}
    train['context'] = inputs['context'].to(device)
    indices = {k: selection[k].to(device) for k in ('idx_train', 'idx_attach', 'fixed_outer')}
    proxies = [restore_proxy(bundle, i).to(device).eval().requires_grad_(False) for i in (1, 2)]
    return inputs, bundle, selection, coordinates, support, support_record, train, indices, proxies


def case(plan_path, expected, seed, variant, *, preflight=False):
    import torch
    from .runtime import restore_clean_reference
    from .full_ablation_training import fit_case
    from .full_ablation_evaluation import fit_victim, evaluate_case
    plan_path = Path(plan_path).resolve()
    plan = verify(plan_path, expected)
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError('Unregistered case')
    output = plan_path.parent / ('engineering_preflight' if preflight else f'seed{seed}/{variant}')
    output.mkdir(parents=True, exist_ok=False)
    state(output, 'STARTING_CUDA', seed=seed, variant=variant, preflight=preflight)
    began = time.monotonic()
    try:
        device = configure_cuda(seed)
        inputs, bundle, selection, coordinates, support, support_record, train, indices, proxies = load_case(plan, seed, variant, device)
        write(output / 'selection.json', selection['record'])
        write(output / 'support.json', support_record)
        config = dict(stage_epochs={stage: 1 if preflight else plan['stage_epochs'][stage] for stage in STAGES},
            discriminator_seed=seed + 1000, shadow_lr=.0002, residual_lr=.0002,
            weight_decay=.0005, target_weight=1., similarity_weight=.1)
        state(output, 'GPU_ATTACK_TRAINING', seed=seed, variant=variant, preflight=preflight)
        trained = fit_case(train, indices, proxies, coordinates, support, seed=seed, variant=variant,
            output_dir=output / 'training', config=config,
            progress=lambda row: event(output, phase='training', record=row))
        generator = trained['generator'].eval().requires_grad_(False)
        state(output, 'GPU_FRESH_VICTIM_TRAINING', seed=seed, variant=variant, preflight=preflight)
        victim, victim_record = fit_victim(generator, inputs, selection, bundle, seed=seed,
            output_dir=output / 'victim', epochs=2 if preflight else plan['victim_epochs'], device=str(device),
            progress=lambda row: event(output, phase='victim', record=row))
        if preflight:
            write(output / 'completion.json', dict(status='ENGINEERING_PREFLIGHT_PASS',
                plan_sha256=expected, utc=utc(), stage_epochs=config['stage_epochs'], victim_epochs=2,
                final_attack_state_reused=False, asr_evaluation=False,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                elapsed_seconds=time.monotonic() - began))
        else:
            state(output, 'GPU_DEVELOPMENT_EVALUATION', seed=seed, variant=variant)
            reference = restore_clean_reference(bundle).to(device).eval().requires_grad_(False)
            report = evaluate_case(generator, victim, reference, inputs, device=str(device), output_dir=output / 'development')
            write(output / 'result.json', dict(seed=seed, variant=variant, status='CASE_COMPLETE',
                metrics=report, device=str(device), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                elapsed_seconds=time.monotonic() - began, clean_prefix_shared=True,
                no_attack_checkpoint_loaded=True, original_test_arrays_loaded=False))
            write(output / 'completion.json', dict(status='CASE_COMPLETE', seed=seed, variant=variant,
                plan_sha256=expected, result_sha256=sha(output / 'result.json'), utc=utc()))
        state(output, 'ENGINEERING_PREFLIGHT_PASS' if preflight else 'CASE_COMPLETE', seed=seed, variant=variant)
    except BaseException as error:
        write(output / 'failure.json', dict(error=repr(error), utc=utc(), seed=seed, variant=variant,
              elapsed_seconds=time.monotonic() - began))
        state(output, 'FAILED', error=repr(error), seed=seed, variant=variant)
        raise


def wait_headroom(output, plan, expected):
    from author_native_runs.dpgba_queue import resource_snapshot, cuda_probe
    began = time.monotonic()
    while True:
        verify(output / 'plan.json', expected)
        if (output / 'CANCEL').exists():
            raise RuntimeError('Ablation queue cancelled')
        sample = resource_snapshot()
        policy = plan['resource_policy']
        probe = cuda_probe() if sample.get('ok') and sample['host_available_bytes'] >= policy['min_host_available_bytes'] else None
        if probe and probe.get('ok') and probe['free_bytes'] >= policy['min_cuda_free_bytes']:
            return dict(snapshot=sample, cuda_probe=probe, utc=utc())
        state(output, 'WAITING_FOR_SHARED_GPU_HEADROOM', resources=sample, probe=probe)
        if time.monotonic() - began > policy['max_wait_seconds']:
            raise TimeoutError('Shared GPU headroom timeout')
        time.sleep(30)


def run_queue(plan_path, expected):
    plan_path = Path(plan_path).resolve()
    plan, output = verify(plan_path, expected), plan_path.parent
    write(output / 'execution_claim.json', dict(pid=os.getpid(), utc=utc()))
    child = None

    def cancel(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise RuntimeError(f'Ablation queue interrupted: {signum}')

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    completed = []
    try:
        for item in plan['cases']:
            ready = wait_headroom(output, plan, expected)
            seed, variant = item['seed'], item['variant']
            command = [sys.executable, '-B', '-u', '-m', 'spear_current.full_ablation', '--case',
                       '--plan', str(plan_path), '--plan-sha256', expected, '--seed', str(seed), '--variant', variant]
            write(output / f'seed{seed}_{variant}_launch.json', dict(command=command, resource_ready=ready, utc=utc()))
            with (output / f'seed{seed}_{variant}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                state(output, 'GPU_ABLATION_RUNNING', seed=seed, variant=variant, child_pid=child.pid,
                      completed_cases=len(completed), total_cases=len(plan['cases']))
                code = child.wait()
            folder = output / f'seed{seed}/{variant}'
            if code:
                raise RuntimeError(f'{seed}/{variant} exited {code}; no automatic retry')
            receipt = read(folder / 'completion.json')
            if receipt['status'] != 'CASE_COMPLETE' or receipt['result_sha256'] != sha(folder / 'result.json'):
                raise ValueError('Case completion binding mismatch')
            completed.append(dict(**item, result=str(folder / 'result.json'), result_sha256=receipt['result_sha256']))
            write(output / 'completed_cases.json', completed, exclusive=False)
        write(output / 'completion.json', dict(status='FULL_GPU_ABLATION_27_CASES_COMPLETE_REVIEW_REQUIRED',
              cases=completed, utc=utc(), plan_sha256=expected))
        state(output, 'FULL_GPU_ABLATION_27_CASES_COMPLETE_REVIEW_REQUIRED', completed_cases=len(completed))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=20)
        write(output / 'failure.json', dict(error=repr(error), utc=utc(), completed_cases=completed))
        state(output, 'STOPPED', error=repr(error), completed_cases=len(completed))
        raise


def launch(plan_path):
    plan_path = Path(plan_path).resolve()
    digest = sha(plan_path)
    verify(plan_path, digest)
    output = plan_path.parent
    preflight = read(output / 'engineering_preflight/completion.json')
    if preflight['status'] != 'ENGINEERING_PREFLIGHT_PASS' or preflight['plan_sha256'] != digest:
        raise ValueError('Real-input CUDA engineering preflight required')
    if any((output / name).exists() for name in ('launch.json', 'execution_claim.json', 'failure.json')):
        raise ValueError('Queue can be launched only once')
    with (output / 'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'spear_current.full_ablation',
            '--run', '--plan', str(plan_path), '--plan-sha256', digest], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '0'})
    receipt = dict(status='GPU_ABLATION_LAUNCHED', pid=process.pid, utc=utc(), plan_sha256=digest,
                   total_cases=27, concurrent_with_author_queues=True)
    write(output / 'launch.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'preflight', 'launch', 'run', 'case'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/current/manifest.json')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if args.prepare:
        if args.output_dir is None:
            parser.error('--output-dir required')
        prepare(args.output_dir, args.manifest)
    else:
        if args.plan is None:
            parser.error('--plan required')
        expected = args.plan_sha256 or sha(args.plan)
        if args.launch:
            launch(args.plan)
        elif args.run:
            run_queue(args.plan, expected)
        elif args.preflight:
            case(args.plan, expected, SEEDS[0], 'full', preflight=True)
        else:
            if args.seed is None or args.variant is None:
                parser.error('--seed and --variant required')
            case(args.plan, expected, args.seed, args.variant)


if __name__ == '__main__':
    main()
