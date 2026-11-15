"""Resource-gated GPU queue for native author implementations.

This file orchestrates data and artifacts only. It does not implement a
selector, feature attribution algorithm, generator, loss, or optimizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

import torch

from spear_current.runtime import load_bundle, sha256
from spear_current.development_diagnostic import registered_rows
from comparison_methods.evidence import write_json, save_state, evaluate_dev
from comparison_methods.runner import dependency_binding, predecessor_state, process_identity
from .data import load_inputs, input_hashes, validate_fit, validate_author_injection
from .source_registry import assert_primary_sources, audit_all, primary_methods


ROOT = Path(__file__).resolve().parents[1]
METHODS = primary_methods()
# Five preregistered comparison seeds, separate from the retained AV2 main-run
# seeds.  Existing three-seed records are historical and cannot be mixed into
# the new five-seed aggregate.
SEEDS = (20261201, 20261202, 20261203, 20261204, 20261205)
VERSION = 'verified_author_source_av2_gpu_v3_source_registry_five_seed'
EXCLUDED = dict(cgba='AUTHOR_CODE_NOT_OBTAINED: no local reimplementation substituted',
                gta='AUTHOR_GRAPH_CLASSIFICATION_MODEL_POISONING_NOT_DATA_ONLY_PAIR_NODE_ADAPTER')


def now():
    return datetime.now(timezone.utc).isoformat()


def module(method):
    if method not in METHODS:
        raise ValueError('Unsupported author method')
    return importlib.import_module('author_comparisons.' + method)


def directory(path):
    path = Path(path).resolve()
    if path == ROOT / 'results' or not path.is_relative_to(ROOT / 'results'):
        raise ValueError('Require a new child of JCAS/results')
    return path


def code_hashes():
    paths = list((ROOT / 'author_comparisons').glob('*.py'))
    paths += list((ROOT / 'spear_current').glob('*.py'))
    paths += [ROOT / 'comparison_methods' / f for f in ('__init__.py', 'evidence.py', 'runner.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def ready(predecessor):
    """An independent GPU queue need not wait for the unchanged CPU study."""
    return 'READY' if predecessor is None else predecessor_state(predecessor)


def gpu_identity(device='cuda:0'):
    if device != 'cuda:0' or not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA device unavailable; no CPU fallback')
    properties = torch.cuda.get_device_properties(0)
    return dict(device=device, gpu_name=properties.name, torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, compute_capability=list(torch.cuda.get_device_capability(0)))


def bind_resources(path):
    from .gpu_benchmark import validate_report
    path = Path(path).resolve()
    directory(path.parent)
    if path.name != 'summary.json':
        raise ValueError('Require resource summary.json')
    report = validate_report(path)
    identity = gpu_identity()
    if (report.get('status') != 'GPU_RESOURCE_BENCHMARK_COMPLETE'
            or report.get('precision') != 'float32_no_amp_no_tf32'
            or any(report.get(k) != identity[k] for k in ('device', 'gpu_name', 'torch_version'))
            or type(report.get('recommended_max_parallel')) is not int
            or report['recommended_max_parallel'] not in (1, 2)):
        raise ValueError('Successful resource acceptance on this GPU is required')
    if report['recommended_max_parallel'] == 2 and report.get('parallel_speedup', 0) < 1.10:
        raise ValueError('Two-way throughput gate did not pass')
    return dict(path=str(path), sha256=sha256(path), hardware=identity,
                max_parallel=report['recommended_max_parallel'], exclusive_methods=['dpgba'],
                precision='float32_no_amp_no_tf32',
                numerical_reproducibility='seeded GPU repetitions, not CPU bitwise replay',
                minimum_host_available_bytes=8 * 1024 ** 3,
                scope='classifier throughput only; generator peak memory remains monitored')


def configure_gpu(resources):
    if gpu_identity(resources['hardware']['device']) != resources['hardware']:
        raise ValueError('GPU runtime identity changed')
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    # No autocast/GradScaler is used anywhere in this queue.
    if torch.is_autocast_enabled('cuda'):
        raise ValueError('GPU queue must not run inside autocast')
    torch.cuda.reset_peak_memory_stats()


def specification(bundle, predecessor, resources):
    source_audit = assert_primary_sources()
    return dict(version=VERSION, methods=list(METHODS), seeds=list(SEEDS), excluded=EXCLUDED,
        configs={name: module(name).validate_config() for name in METHODS},
        author_sources={name: module(name).verify_source() for name in METHODS},
        uploaded_source_registry=source_audit,
        local_code=code_hashes(), predecessor=predecessor, resources=resources, sources=registered_rows(bundle),
        manifest=dict(path=str(bundle.manifest_path), sha256=sha256(bundle.manifest_path)),
        common=dict(input='retained AV2 pair-node 784; TRAIN-only normalization',
            train_scenes=512, dev_scenes=128, target_class=1, attachment_count=26,
            candidate_pool='eligible TRAIN negatives from original unsupervised reservoir',
            context_labels='retain -1; never supervise; explicit binary task metadata',
            victim='author GraphSage implementation and author .fit; normal DEV accuracy checkpoint',
            victim_seed='case seed + 9000; clean and victim matched initialization seed',
            author_training='native algorithm and defaults; compatibility and data patches separately recorded',
            generator_checkpoint='actual native behavior, even where it differs from comments or paper',
            defense='author supported none option; common no-defense condition, not every CLI default',
            target_pool='same public DEV112, Moving33, no model-based target exclusion',
            evaluation='original node prefix only, raw author edge weights passed unchanged',
            original_test_arrays_allowed=False, original_validation_arrays_allowed=False,
            audit_arrays_allowed=False, confirmation_arrays_allowed=False),
        limits=['not all authors use identical feature/node budget, hidden width, loss or computation',
            'native zero-weight/GraphSAGE behavior is not silently corrected',
            'retained multistage proposed model is not a fresh equal-total-budget comparator',
            'DEV has development history; this is not a blind final test',
            'SPEAR may be bound to the user-supplied author archive if remote clone is unavailable',
            'author unseeded randomness, if present, is recorded rather than replaced'],
        failure_policy='stop on failure, preserve artifacts, no parameter relaxation or retries',
        poll_seconds=30, max_wait_seconds=72 * 3600, torch_threads=2,
        automatic_promotion=False, automatic_additional_training=False)


def prepare(bundle, output, after=None, benchmark=None):
    if benchmark is None:
        raise ValueError('GPU resource benchmark required before registration')
    dependency = dependency_binding(after) if after is not None else None
    ready(dependency)
    resources = bind_resources(benchmark)
    spec = specification(bundle, dependency, resources)
    output = directory(output)
    if dependency is not None and output.is_relative_to(Path(dependency['directory'])):
        raise ValueError('Do not write into predecessor results')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', dict(created_utc=now(), protocol=spec))
    with tarfile.open(output / 'adapter_source_snapshot.tar', 'x') as archive:
        for name in spec['local_code']:
            archive.add(ROOT / name, arcname=name)
    if spec != specification(bundle, dependency, resources):
        raise ValueError('Sources changed during registration')
    write_json(output / 'registration.json', dict(plan_sha256=sha256(output / 'plan.json'),
        adapter_snapshot_sha256=sha256(output / 'adapter_source_snapshot.tar'),
        graph_arrays_loaded=False, checkpoints_loaded=False, training_started=False))
    return output / 'plan.json'


def validate_plan(bundle, path):
    path = Path(path).resolve()
    directory(path.parent)
    if path.name != 'plan.json':
        raise ValueError('Require registered plan.json')
    registration = json.loads((path.parent / 'registration.json').read_text())
    if (registration['plan_sha256'] != sha256(path)
            or registration['adapter_snapshot_sha256'] != sha256(path.parent / 'adapter_source_snapshot.tar')):
        raise ValueError('Changed registration')
    plan = json.loads(path.read_text())
    resources = plan['protocol']['resources']
    if sha256(resources['path']) != resources['sha256']:
        raise ValueError('Changed resource acceptance')
    if plan['protocol'] != specification(bundle, plan['protocol']['predecessor'], resources):
        raise ValueError('Registered code, data or author source changed')
    return plan


def check_not_terminal(output):
    if any((output / name).exists() for name in ('CANCEL', 'cancelled.json', 'failure.json', 'completion.json')):
        raise ValueError('Cancelled, failed or completed queue cannot run')


def initialization_hash(record):
    for name in ('initial_state_sha256', 'initialization_digest', 'initialization_sha256'):
        if name in record:
            return record[name]
    raise ValueError('Author classifier adapter did not record initialization')


def case(bundle, path, name, seed):
    plan = validate_plan(bundle, path)
    output = Path(path).resolve().parent
    check_not_terminal(output)
    claim = json.loads((output / 'execution_claim.json').read_text())
    if (seed not in SEEDS or name not in METHODS or os.getppid() != claim['pid']
            or process_identity(claim['pid']) != claim['process']
            or ready(plan['protocol']['predecessor']) != 'READY'):
        raise ValueError('Case must be started by its registered queue with ready dependencies')
    child = output / f'seed{seed}' / name
    child.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(plan['protocol']['torch_threads'])
    configure_gpu(plan['protocol']['resources'])
    device = plan['protocol']['resources']['hardware']['device']
    ledger = []
    with (child / 'progress.jsonl').open('x', buffering=1) as stream:
        def progress(*args, **kwargs):
            record = dict(args[0]) if args else {}
            record.update(kwargs)
            record.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                          gpu_reserved_bytes=torch.cuda.memory_reserved(),
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            stream.write(json.dumps(dict(record, utc=now()), allow_nan=False) + '\n')
        try:
            data = load_inputs(bundle, ledger)
            write_json(child / 'graph_access_ledger.json', ledger)
            before = input_hashes(data)
            write_json(child / 'input_hashes.json', before)
            api = module(name)
            config = plan['protocol']['configs'][name]
            work = child / 'author_work'
            progress(stage='author_method_started', method=name, seed=seed, device=device)
            fitted = getattr(api, 'fit_' + name)(data['x'], data['edge_index'], data['y'],
                data['idx_train'], data['idx_val'], data['candidates'],
                seed=seed, work_dir=work, config=config, progress=progress, device=device)
            if input_hashes(data) != before:
                raise ValueError('Author adapter mutated shared clean inputs')
            validate_fit(fitted, data, name, config)
            generator = fitted['generator'].eval().requires_grad_(False)
            generator.zero_grad(set_to_none=True)
            write_json(child / 'method_record.json', fitted['record'])
            save_state(child / 'generator.pt', generator.state_dict())
            save_state(child / 'poisoned_training.pt', {key: fitted[key] for key in
                ('poisoned_x', 'poisoned_edge_index', 'poisoned_edge_weight', 'poisoned_y',
                 'attachment_indices', 'fit_indices')})
            models, initializations = {}, []
            for label in ('clean_reference', 'victim'):
                progress(stage='author_classifier_started', model=label)
                fx, fe, fw, fy = ((data['x'], data['edge_index'],
                    torch.ones(data['edge_index'].shape[1]), data['y']) if label == 'clean_reference'
                    else (fitted['poisoned_x'], fitted['poisoned_edge_index'],
                          fitted['poisoned_edge_weight'], fitted['poisoned_y']))
                fit = api.fit_author_classifier(fx, fe, fw, fy, fitted['fit_indices'], data['idx_val'],
                    seed=seed + 9000, work_dir=work, config=config, progress=progress, device=device)
                models[label] = fit['model'].eval().requires_grad_(False)
                models[label].zero_grad(set_to_none=True)
                initializations.append(initialization_hash(fit['record']))
                save_state(child / (label + '.pt'), models[label].state_dict())
                write_json(child / (label + '_record.json'), fit['record'])
            if len(set(initializations)) != 1:
                raise ValueError('Matched author clean/victim initializations differ')
            report = evaluate_dev(data['parts']['dev'], data['sources']['dev'], generator, models,
                child / 'development', method=name,
                config=dict(feature_count=config.get('alpha_int', 30), trigger_size=config.get('trigger_size', 3)),
                progress=progress, injection_validator=validate_author_injection)
            if (report['targets'] != 112 or report['metrics']['victim']['targets']['moving']['targets'] != 33
                    or input_hashes(data) != before):
                raise ValueError('Public targets or original inputs changed')
            validate_plan(bundle, path)
            write_json(child / 'completion.json', dict(status='AUTHOR_CASE_COMPLETE', method=name, seed=seed,
                elapsed_seconds=time.monotonic() - started, metrics=report['metrics'],
                population_identity_sha256=report['population_identity_sha256'],
                summary_sha256=sha256(child / 'development/summary.json'),
                original_test_arrays_loaded=False, audit_arrays_loaded=False,
                original_validation_arrays_loaded=False, confirmation_arrays_loaded=False,
                author_algorithm_reimplemented=False, execution_device=device,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                precision='float32_no_amp_no_tf32'))
        except BaseException as error:
            write_json(child / 'failure.json', dict(status='AUTHOR_CASE_FAILED', error=str(error),
                type=type(error).__name__, traceback=traceback.format_exc(), utc=now()))
            raise


def can_start(name, active_methods, resources):
    exclusive = set(resources['exclusive_methods'])
    return (len(active_methods) < resources['max_parallel']
            and not any(m in exclusive for m in active_methods)
            and (not active_methods or name not in exclusive))


def check_host_memory(resources):
    # GB10 uses shared system memory. Never consume the last system reserve;
    # only this queue's children may be stopped on pressure, not other jobs.
    lines = Path('/proc/meminfo').read_text().splitlines()
    available = next(int(line.split()[1]) * 1024 for line in lines if line.startswith('MemAvailable:'))
    if available < resources['minimum_host_available_bytes']:
        raise RuntimeError('Host/unified-memory reserve below registered minimum')


def interrupted(signum, frame):
    raise RuntimeError(f'Queue interrupted by signal {signum}; stopping owned workers')


def stop_owned_children(active):
    """Only processes started by this queue, never pre-existing GPU jobs."""
    for item in active:
        if item['process'].poll() is None:
            item['process'].terminate()
    deadline = time.monotonic() + 10
    for item in active:
        process = item['process']
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        item['log'].close()


def run(bundle, path):
    plan = validate_plan(bundle, path)
    output = Path(path).resolve().parent
    check_not_terminal(output)
    if ready(plan['protocol']['predecessor']) != 'READY':
        raise ValueError('Predecessor not ready')
    if gpu_identity() != plan['protocol']['resources']['hardware']:
        raise ValueError('GPU identity mismatch')
    write_json(output / 'execution_claim.json', dict(pid=os.getpid(), process=process_identity(os.getpid()),
        plan_sha256=sha256(path), utc=now()))
    completed, active = [], []
    pending = [(name, seed) for seed in SEEDS for name in METHODS]
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or active:
            check_not_terminal(output)
            validate_plan(bundle, path)
            check_host_memory(plan['protocol']['resources'])
            # Collect failures before launching anything else. All output paths
            # are disjoint; a failed worker stops the queue and its active peers.
            for item in list(active):
                result = item['process'].poll()
                if result is None:
                    continue
                item['log'].close()
                active.remove(item)
                name, seed = item['name'], item['seed']
                child = output / f'seed{seed}' / name
                if result != 0:
                    raise RuntimeError(f'{name} seed {seed} failed with exit code {result}; queue stopped')
                record = json.loads((child / 'completion.json').read_text())
                if (record['status'] != 'AUTHOR_CASE_COMPLETE'
                        or record['summary_sha256'] != sha256(child / 'development/summary.json')):
                    raise ValueError('Case completion mismatch')
                completed.append(record)
            while pending and can_start(pending[0][0], [a['name'] for a in active], plan['protocol']['resources']):
                name, seed = pending.pop(0)
                command = [sys.executable, '-B', '-m', 'author_comparisons.runner', '--case',
                           '--plan', str(Path(path).resolve()), '--method', name, '--seed', str(seed)]
                log = (output / f'{name}_{seed}.log').open('xb')
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active.append(dict(name=name, seed=seed, process=process, log=log))
                with (output / 'schedule.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(utc=now(), method=name, seed=seed, pid=process.pid,
                        active=len(active), device='cuda:0')) + '\n')
            if active:
                time.sleep(5)
        if len({row['population_identity_sha256'] for row in completed}) != 1:
            raise ValueError('Different public populations across cases')
        aggregate = {}
        for name in METHODS:
            values = [r['metrics']['victim']['targets']['all']['incremental_asr']
                      for r in sorted(completed, key=lambda r: r['seed']) if r['method'] == name]
            aggregate[name] = dict(per_seed=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values))
        write_json(output / 'summary.json', dict(cases=completed, overall_incremental_asr=aggregate,
            permissions_and_total_computation_not_identical=True, reused_development_data=True,
            excluded_methods=EXCLUDED, author_algorithm_reimplemented=False))
        write_json(output / 'completion.json', dict(status='AUTHOR_COMPARISON_COMPLETE_REVIEW_REQUIRED',
            utc=now(), summary_sha256=sha256(output / 'summary.json'), cases=len(completed),
            original_test_arrays_loaded=False, audit_arrays_loaded=False, automatic_followup=False))
    except BaseException as error:
        stop_owned_children(active)
        write_json(output / 'failure.json', dict(status='AUTHOR_QUEUE_FAILED_NO_FURTHER_JOBS', utc=now(),
            error=str(error), type=type(error).__name__, traceback=traceback.format_exc(),
            completed=[dict(method=r['method'], seed=r['seed']) for r in completed]))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def wait_and_run(bundle, path):
    plan = validate_plan(bundle, path)
    output, started = Path(path).resolve().parent, time.monotonic()
    write_json(output / 'watcher_claim.json', dict(pid=os.getpid(), utc=now()))
    try:
        with (output / 'wait.jsonl').open('x', buffering=1) as stream:
            while True:
                if (output / 'CANCEL').exists():
                    write_json(output / 'cancelled.json', dict(status='CANCELLED_BEFORE_TRAINING', utc=now()))
                    return
                validate_plan(bundle, path)
                state = ready(plan['protocol']['predecessor'])
                stream.write(json.dumps(dict(utc=now(), state=state, training_started=False)) + '\n')
                if state == 'READY':
                    break
                if time.monotonic() - started > plan['protocol']['max_wait_seconds']:
                    raise TimeoutError('Predecessor wait expired')
                time.sleep(plan['protocol']['poll_seconds'])
        run(bundle, path)
    except BaseException as error:
        if not (output / 'failure.json').exists():
            write_json(output / 'failure.json', dict(status='AUTHOR_WAIT_FAILED', error=str(error), utc=now()))
        raise


def enqueue(bundle, path):
    plan = validate_plan(bundle, path)
    output = Path(path).resolve().parent
    check_not_terminal(output)
    ready(plan['protocol']['predecessor'])
    if gpu_identity() != plan['protocol']['resources']['hardware']:
        raise ValueError('Registered GPU unavailable')
    if any((output / name).exists() for name in ('watcher_claim.json', 'execution_claim.json', 'launch_claim.json')):
        raise ValueError('Queue already launched')
    write_json(output / 'launch_claim.json', dict(utc=now(), plan_sha256=sha256(path)))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    command = [sys.executable, '-B', '-m', 'author_comparisons.runner', '--wait-and-run',
               '--plan', str(Path(path).resolve())]
    with (output / 'terminal.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(status='AUTHOR_GPU_QUEUE_LAUNCHED', pid=child.pid, process=process_identity(child.pid),
        utc=now(), methods=list(METHODS), seeds=list(SEEDS), cases=len(METHODS) * len(SEEDS), command=command,
        max_parallel=plan['protocol']['resources']['max_parallel'], exclusive_methods=['dpgba'])
    write_json(output / 'launch.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ('check-only', 'prepare', 'enqueue', 'wait-and-run', 'run', 'case'):
        actions.add_argument('--' + name, action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--benchmark', type=Path, help='Successful GPU resource summary.json')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    bundle = load_bundle()
    if args.check_only:
        print(json.dumps(dict(configs={m: module(m).validate_config() for m in METHODS},
            sources={m: module(m).verify_source() for m in METHODS}, graph_arrays_loaded=False)))
    elif args.prepare:
        if args.output_dir is None or args.benchmark is None:
            parser.error('--prepare needs --output-dir and --benchmark; --after is optional')
        print(prepare(bundle, args.output_dir, args.after, args.benchmark))
    else:
        if args.plan is None:
            parser.error('execution needs --plan')
        if args.case:
            if args.method is None or args.seed is None:
                parser.error('--case needs --method and --seed')
            case(bundle, args.plan, args.method, args.seed)
        elif args.enqueue:
            print(json.dumps(enqueue(bundle, args.plan)))
        elif args.wait_and_run:
            wait_and_run(bundle, args.plan)
        else:
            run(bundle, args.plan)


if __name__ == '__main__':
    main()
