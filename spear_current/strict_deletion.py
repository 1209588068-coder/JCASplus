"""Registered GPU supplements: remove one named mechanism without compensation.

Only original-TRAIN TRAIN/dev inputs are reused. The common checkpoint is the
saved base stage before residual, final support mask and domain were introduced.
No archived author comparison or original validation/test data enters this path.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .full_ablation import ROOT, configure_cuda, event, read, sha, state, utc, write


SEEDS = (20260621, 20260622, 20260623)
VARIANTS = ('full', 'no_residual', 'no_support_mask', 'no_domain')
STAGES = ('residual', 'support_surrogate', 'domain', 'distribution')
DEFAULT_PARENT = ROOT / 'results/full_gpu_module_ablation_20261030_v1'


def matrix():
    return [dict(seed=seed, variant=variant) for seed in SEEDS for variant in VARIANTS]


def prepare(output, parent=DEFAULT_PARENT, *, min_cuda_free_gib=8):
    if type(min_cuda_free_gib) is not int or min_cuda_free_gib not in (3, 8):
        raise ValueError('CUDA gate must be the default 8GiB or registered parallel 3GiB policy')
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output == ROOT / 'results' or not output.is_relative_to(ROOT / 'results'):
        raise ValueError('Output must be a new child of JCAS/results')
    original = read(parent / 'plan.json')
    completion = read(parent / 'completion.json')
    if (completion['status'] != 'FULL_GPU_ABLATION_27_CASES_COMPLETE_REVIEW_REQUIRED'
            or len(completion['cases']) != 27 or original['seeds'] != list(SEEDS)):
        raise ValueError('Expected the completed original 27-case parent experiment')
    for row in completion['cases']:
        if sha(row['result']) != row['result_sha256']:
            raise ValueError('Original result changed; do not start a supplement')
    snapshot = ROOT / 'artifacts/source_snapshots/full_gpu_module_ablation_v1_20261031'
    if sha(snapshot / 'original_plan.json') != sha(parent / 'plan.json'):
        raise ValueError('Original source snapshot plan mismatch')
    bound = {str(parent / 'plan.json'): sha(parent / 'plan.json'),
             str(parent / 'completion.json'): sha(parent / 'completion.json'),
             original['prepared_inputs']: original['inputs_sha256'],
             original['manifest']: original['manifest_sha256']}
    for relative, expected in original['sources'].items():
        path = snapshot / relative
        if sha(path) != expected:
            raise ValueError('Original registered source snapshot changed: ' + relative)
        bound[str(path)] = expected
    for item in original['clean_assets'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen clean asset changed')
        bound[item['path']] = item['sha256']
    prefixes = {}
    for seed in SEEDS:
        folder = parent / f'seed{seed}/full'
        path = folder / 'training/base_final.pt'
        prefixes[str(seed)] = dict(path=str(path), sha256=sha(path))
        for name in ('selection.json', 'support.json', 'result.json'):
            bound[str(folder / name)] = sha(folder / name)
        bound[str(path)] = sha(path)
    if any(sha(path) != expected for path, expected in bound.items()):
        raise ValueError('Bound parent input changed')
    sources = {str(path.relative_to(ROOT)): sha(path)
               for path in sorted((ROOT / 'spear_current').glob('*.py'))}
    sources['author_native_runs/dpgba_queue.py'] = sha(ROOT / 'author_native_runs/dpgba_queue.py')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(version='strict_module_deletion_A_v1', created_utc=utc(), parent=str(parent),
        bound_files=bound, sources=sources, manifest=original['manifest'],
        prepared_inputs=original['prepared_inputs'], prefixes=prefixes,
        seeds=list(SEEDS), variants=list(VARIANTS), cases=matrix(),
        stage_epochs={stage: 200 for stage in STAGES}, victim_epochs=200,
        prefix_identity='TRAIN attack base200; residual untrained; no final support/domain applied',
        attack_checkpoint_inherited=True, end_to_end_from_scratch=False,
        no_residual_policy='no residual parameters or forward; base G frozen; post-base G updates zero; no compensation',
        no_support_mask_policy='legal28 writes; original SPEAR30/focal exclusions retained; D still observes original SAGE24',
        no_domain_policy='no output clamp; keep D objective/architecture/reference and finite/shape checks',
        inherited_generator_updates=200, inherited_surrogate_updates=200,
        post_generator_updates={'full': 600, 'no_residual': 0, 'no_support_mask': 600, 'no_domain': 600},
        post_surrogate_updates=800, post_discriminator_updates=200,
        discriminator_observed_coordinates=24,
        write_coordinates={'full': 24, 'no_residual': 24, 'no_support_mask': 28, 'no_domain': 24},
        victim='fresh GraphSAGE 784/80/2; 200 epochs; earliest maximum normal DEV AUC',
        evaluation=dict(train_scenes=512, dev_scenes=128, targets=112, moving_targets=33,
            reused_development=True, original_validation_arrays_loaded=False,
            original_test_arrays_loaded=False, blind_confirmation=False),
        resource_policy=dict(min_host_available_bytes=16 * 1024 ** 3,
            min_cuda_free_bytes=min_cuda_free_gib * 1024 ** 3, max_wait_seconds=72 * 3600,
            allow_other_gpu_processes=True,
            scheduling_authorization=('User explicitly requested immediate concurrent execution with GTA'
                                      if min_cuda_free_gib == 3 else 'Original conservative headroom policy'),
            measured_parent_preflight_peak_allocated_bytes=2012600320,
            headroom_is_not_an_oom_guarantee=True),
        device='cuda:0', precision='float32', amp=False, tf32=False,
        acceptance='correct mechanism removal, finite states, bound identities; no required ASR drop',
        no_automatic_retry=True, overwrite_existing_results=False)
    write(output / 'plan.json', plan)
    state(output, 'REGISTERED_PREFLIGHT_REQUIRED', cases=12, plan_sha256=sha(output / 'plan.json'))
    print(str(output / 'plan.json'), flush=True)


def verify(plan_path, expected):
    plan_path = Path(plan_path).resolve()
    if sha(plan_path) != expected:
        raise ValueError('Registered strict-deletion plan changed')
    plan = read(plan_path)
    if (plan.get('version') != 'strict_module_deletion_A_v1'
            or plan.get('cases') != matrix() or plan.get('seeds') != list(SEEDS)
            or plan.get('variants') != list(VARIANTS)):
        raise ValueError('Strict-deletion registered matrix changed')
    for path, digest in plan['bound_files'].items():
        if sha(path) != digest:
            raise ValueError('Bound common prefix or input changed: ' + path)
    for relative, digest in plan['sources'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Registered source changed: ' + relative)
    return plan


def case(plan_path, expected, seed, variant, *, preflight=False):
    import torch
    from .runtime import load_bundle, restore_proxy, restore_clean_reference
    from .full_ablation_data import select_indices, select_support
    from .strict_deletion_training import fit_from_base
    from .strict_deletion_evaluation import fit_victim, evaluate_case
    plan = verify(plan_path, expected)
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError('Unregistered strict-deletion case')
    output = Path(plan_path).resolve().parent / (
        f'engineering_preflight/{variant}' if preflight else f'seed{seed}/{variant}')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state(output, 'LOADING_REGISTERED_PREFIX', seed=seed, variant=variant, preflight=preflight)
    try:
        device = configure_cuda(seed)
        # SHA checked above; this is our own previously generated input cache.
        inputs = torch.load(plan['prepared_inputs'], map_location='cpu', weights_only=False)
        if set(inputs['combined']) != {'train', 'dev'}:
            raise ValueError('Only original-TRAIN TRAIN/dev arrays are permitted')
        bundle = load_bundle(plan['manifest'], verify_assets=False)
        selection = select_indices(inputs, 'full', seed)
        coordinates, support, support_record = select_support(bundle, 'full', seed)
        write(output / 'selection.json', selection['record'])
        write(output / 'support.json', dict(original_sage24=support_record,
            treatment=variant, final_write_count=plan['write_coordinates'][variant],
            discriminator_observed_count=24))
        train = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in inputs['combined']['train'].items()}
        train['context'] = inputs['context'].to(device)
        indices = {key: selection[key].to(device) for key in ('idx_train', 'idx_attach', 'fixed_outer')}
        proxies = [restore_proxy(bundle, i).to(device).eval().requires_grad_(False) for i in (1, 2)]
        config = dict(stage_epochs={stage: 1 if preflight else plan['stage_epochs'][stage]
                                   for stage in STAGES}, discriminator_seed=seed + 1000)
        prefix = plan['prefixes'][str(seed)]
        state(output, 'GPU_POST_BASE_TRAINING', seed=seed, variant=variant, preflight=preflight)
        trained = fit_from_base(train, indices, proxies, coordinates, support,
            prefix_path=prefix['path'], prefix_sha256=prefix['sha256'], seed=seed,
            variant=variant, output_dir=output / 'training', config=config,
            progress=lambda row: event(output, phase='training', record=row))
        generator = trained['generator'].eval().requires_grad_(False)
        state(output, 'GPU_FRESH_VICTIM_TRAINING', seed=seed, variant=variant, preflight=preflight)
        victim, victim_record = fit_victim(generator, inputs, selection, bundle, variant=variant,
            seed=seed, output_dir=output / 'victim', epochs=2 if preflight else plan['victim_epochs'],
            device=str(device), progress=lambda row: event(output, phase='victim', record=row))
        if preflight:
            write(output / 'completion.json', dict(status='STRICT_CASE_PREFLIGHT_PASS',
                plan_sha256=expected, variant=variant, stage_epochs=config['stage_epochs'],
                victim_epochs=2, asr_evaluation=False, reuse_short_fit_weights=False,
                elapsed_seconds=time.monotonic() - started,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        else:
            reference = restore_clean_reference(bundle).to(device).eval().requires_grad_(False)
            report = evaluate_case(generator, victim, reference, inputs, variant=variant,
                expected_support=support, device=str(device), output_dir=output / 'development')
            result = dict(status='CASE_COMPLETE', seed=seed, variant=variant, metrics=report,
                prefix=prefix, inherited_attack_base=True, post_generator_steps=trained['generator_steps'],
                historical_results_replaced=False, device=str(device),
                original_test_arrays_loaded=False, original_validation_arrays_loaded=False,
                elapsed_seconds=time.monotonic() - started,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write(output / 'result.json', result)
            write(output / 'completion.json', dict(status='CASE_COMPLETE', seed=seed, variant=variant,
                plan_sha256=expected, result_sha256=sha(output / 'result.json'), utc=utc()))
        state(output, 'PREFLIGHT_PASS' if preflight else 'CASE_COMPLETE', seed=seed, variant=variant)
    except BaseException as error:
        write(output / 'failure.json', dict(error=repr(error), utc=utc(), seed=seed, variant=variant))
        state(output, 'FAILED', seed=seed, variant=variant, error=repr(error))
        raise


def headroom(plan_path, expected):
    from author_native_runs.dpgba_queue import resource_snapshot, cuda_probe
    output = Path(plan_path).resolve().parent
    started = time.monotonic()
    while True:
        plan = verify(plan_path, expected)
        if (output / 'CANCEL').exists():
            raise RuntimeError('Strict-deletion queue cancelled')
        sample = resource_snapshot()
        policy = plan['resource_policy']
        probe = (cuda_probe() if sample.get('ok') and
                 sample['host_available_bytes'] >= policy['min_host_available_bytes'] else None)
        if probe and probe.get('ok') and probe['free_bytes'] >= policy['min_cuda_free_bytes']:
            return dict(resources=sample, cuda_probe=probe, utc=utc())
        state(output, 'WAITING_FOR_GPU_HEADROOM', resources=sample, probe=probe)
        if time.monotonic() - started > policy['max_wait_seconds']:
            raise TimeoutError('GPU headroom wait timeout; no CPU fallback')
        time.sleep(30)


def preflight(plan_path, expected):
    import gc
    import torch
    plan = verify(plan_path, expected)
    folder = Path(plan_path).resolve().parent / 'engineering_preflight'
    for variant in VARIANTS:
        # Release this supervisor's unused cache before probing free memory.
        # It must not wait on memory reserved by its own completed short case.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        headroom(plan_path, expected)
        case(plan_path, expected, SEEDS[0], variant, preflight=True)
    receipts = [read(folder / variant / 'completion.json') for variant in VARIANTS]
    if any(r['status'] != 'STRICT_CASE_PREFLIGHT_PASS' or r['plan_sha256'] != expected for r in receipts):
        raise ValueError('Strict-deletion real-input preflight incomplete')
    write(folder / 'completion.json', dict(status='STRICT_DELETION_PREFLIGHT_PASS',
        plan_sha256=expected, variants=list(VARIANTS), asr_evaluation=False, utc=utc()))
    state(Path(plan_path).resolve().parent, 'PREFLIGHT_COMPLETE_READY_TO_LAUNCH', cases=len(plan['cases']))


def run(plan_path, expected):
    plan = verify(plan_path, expected)
    output = Path(plan_path).resolve().parent
    write(output / 'execution_claim.json', dict(pid=os.getpid(), utc=utc()))
    child = None

    def cancel(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise RuntimeError('Strict-deletion queue interrupted: ' + str(signum))

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    completed = []
    try:
        for item in plan['cases']:
            resources = headroom(plan_path, expected)
            seed, variant = item['seed'], item['variant']
            command = [sys.executable, '-B', '-u', '-m', 'spear_current.strict_deletion',
                       '--case', '--plan', str(plan_path), '--plan-sha256', expected,
                       '--seed', str(seed), '--variant', variant]
            write(output / f'seed{seed}_{variant}_launch.json', dict(command=command, resources=resources, utc=utc()))
            with (output / f'seed{seed}_{variant}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                state(output, 'GPU_STRICT_DELETION_RUNNING', seed=seed, variant=variant,
                      child_pid=child.pid, completed_cases=len(completed), total_cases=12)
                code = child.wait()
            folder = output / f'seed{seed}/{variant}'
            if code:
                raise RuntimeError(f'{seed}/{variant} exited {code}; no automatic retry')
            receipt = read(folder / 'completion.json')
            if receipt['status'] != 'CASE_COMPLETE' or receipt['result_sha256'] != sha(folder / 'result.json'):
                raise ValueError('Strict-deletion case completion mismatch')
            completed.append(dict(**item, result=str(folder / 'result.json'),
                                  result_sha256=receipt['result_sha256']))
            write(output / 'completed_cases.json', completed, exclusive=False)
        verify(plan_path, expected)
        write(output / 'completion.json', dict(status='STRICT_DELETION_12_CASES_COMPLETE_REVIEW_REQUIRED',
            cases=completed, plan_sha256=expected, utc=utc()))
        state(output, 'STRICT_DELETION_12_CASES_COMPLETE_REVIEW_REQUIRED', completed_cases=12)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=20)
        write(output / 'failure.json', dict(error=repr(error), completed_cases=completed, utc=utc()))
        state(output, 'STOPPED', error=repr(error), completed_cases=len(completed))
        raise


def launch(plan_path, expected):
    verify(plan_path, expected)
    output = Path(plan_path).resolve().parent
    receipt = read(output / 'engineering_preflight/completion.json')
    if receipt['status'] != 'STRICT_DELETION_PREFLIGHT_PASS' or receipt['plan_sha256'] != expected:
        raise ValueError('All four real-input preflights are required')
    if any((output / key).exists() for key in ('launch.json', 'execution_claim.json', 'failure.json')):
        raise ValueError('Strict-deletion queue can be launched only once')
    command = [sys.executable, '-B', '-u', '-m', 'spear_current.strict_deletion',
               '--run', '--plan', str(plan_path), '--plan-sha256', expected]
    with (output / 'queue.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '0'})
    write(output / 'launch.json', dict(pid=process.pid, command=command, utc=utc(),
        cases=12, plan_sha256=expected, concurrent_with_gta=True))
    print('Strict-deletion queue PID ' + str(process.pid), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'preflight', 'launch', 'run', 'case'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--min-cuda-free-gib', type=int, choices=(3, 8), default=8)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if args.prepare:
        if args.output_dir is None:
            parser.error('--prepare requires --output-dir')
        prepare(args.output_dir, args.parent, min_cuda_free_gib=args.min_cuda_free_gib)
        return
    if args.plan is None:
        parser.error('--plan is required')
    expected = args.plan_sha256 or sha(args.plan)
    if args.case:
        if args.seed is None or args.variant is None:
            parser.error('--case requires --seed and --variant')
        case(args.plan, expected, args.seed, args.variant)
    else:
        function = preflight if args.preflight else launch if args.launch else run
        function(args.plan, expected)


if __name__ == '__main__':
    main()
