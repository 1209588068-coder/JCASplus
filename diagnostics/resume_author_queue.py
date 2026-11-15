"""Resume only unstarted author cases, preserving the stopped queue verbatim.

The native --case implementation remains unchanged. This is scheduling and
provenance code, not an algorithm patch or a retry of a numerically failed fit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from author_comparisons import runner
from comparison_methods.evidence import write_json
from comparison_methods.runner import process_identity
from spear_current.runtime import load_bundle, sha256

PROTOCOL_FILES = ('plan.json', 'registration.json', 'adapter_source_snapshot.tar')


def read(path):
    return json.loads(Path(path).read_text())


def completed_record(root, method, seed):
    child = Path(root) / f'seed{seed}' / method
    record = read(child / 'completion.json')
    if ((child / 'failure.json').exists()
            or record.get('status') != 'AUTHOR_CASE_COMPLETE'
            or record.get('method') != method or record.get('seed') != seed
            or record.get('summary_sha256') != sha256(child / 'development/summary.json')
            or not isinstance(record.get('population_identity_sha256'), str)
            or len(record['population_identity_sha256']) != 64):
        raise ValueError(f'Invalid completed case: {child}')
    return record


def classify_cases(root):
    """Classify registered cases without loading any graph or checkpoint."""
    root = Path(root).resolve()
    result = dict(completed=[], failed=[], pending=[])
    for seed in runner.SEEDS:
        for method in runner.METHODS:
            child = root / f'seed{seed}' / method
            complete, failed = child / 'completion.json', child / 'failure.json'
            identity = dict(method=method, seed=seed)
            if complete.exists():
                record = completed_record(root, method, seed)
                result['completed'].append(dict(identity, completion_path=str(complete),
                    completion_sha256=sha256(complete), summary_sha256=record['summary_sha256'],
                    population_identity_sha256=record['population_identity_sha256']))
            elif failed.exists():
                if read(failed).get('status') != 'AUTHOR_CASE_FAILED':
                    raise ValueError(f'Invalid failed case: {child}')
                result['failed'].append(dict(identity, failure_path=str(failed),
                                              failure_sha256=sha256(failed)))
            elif child.exists():
                raise ValueError(f'Started case without a terminal record: {child}')
            else:
                result['pending'].append(identity)
    return result


def schedule_pending(pending):
    seen, rows = set(), []
    for row in pending:
        method, seed = row['method'], row['seed']
        if method not in runner.METHODS or seed not in runner.SEEDS or (method, seed) in seen:
            raise ValueError('Invalid or duplicate pending case')
        seen.add((method, seed))
        rows.append(dict(method=method, seed=seed))
    return sorted(rows, key=lambda r: (r['method'] == 'spear', r['seed'],
                                      runner.METHODS.index(r['method'])))


def source_stopped(root):
    root = Path(root).resolve()
    if read(root / 'failure.json').get('status') != 'AUTHOR_QUEUE_FAILED_NO_FURTHER_JOBS':
        raise ValueError('Require a failed source queue; not a concurrent or complete queue')
    claim = read(root / 'execution_claim.json')
    if process_identity(claim['pid']) == claim['process']:
        raise ValueError('Source queue is still alive')
    for line in (root / 'schedule.jsonl').read_text().splitlines():
        item = json.loads(line)
        pid = item['pid']
        if process_identity(pid) is None:
            continue
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (b'author_comparisons.runner' in command
                and str(root / 'plan.json').encode() in command):
            raise ValueError(f'Source case {pid} is still alive')


def source_binding(root):
    names = (*PROTOCOL_FILES, 'failure.json', 'execution_claim.json', 'schedule.jsonl')
    return {name: sha256(root / name) for name in names}


def prepare(bundle, source, output):
    source, output = runner.directory(source), runner.directory(output)
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Source and continuation directories must be disjoint')
    plan = runner.validate_plan(bundle, source / 'plan.json')
    source_stopped(source)
    states = classify_cases(source)
    pending = schedule_pending(states['pending'])
    if not pending:
        raise ValueError('No unstarted cases; failed cases require a separate decision')
    if len({r['population_identity_sha256'] for r in states['completed']}) != 1:
        raise ValueError('Require a common completed reference population')
    if runner.gpu_identity() != plan['protocol']['resources']['hardware']:
        raise ValueError('GPU runtime identity changed')
    runner.check_host_memory(plan['protocol']['resources'])
    binding = source_binding(source)
    output.mkdir(parents=True, exist_ok=False)
    for name in PROTOCOL_FILES:
        with (source / name).open('rb') as src, (output / name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    value = dict(version='author_unstarted_case_continuation_v1', created_utc=runner.now(),
        source_directory=str(source), output_directory=str(output), source_binding=binding,
        preserved_states=states, scheduled_cases=pending, max_parallel=1,
        wrapper_sha256=sha256(__file__),
        authorization='User requested recovery after numerical-stop diagnosis, 2026-10-29',
        scheduling_change='Continue unstarted cases only; SPEAR last; stop on any new failure',
        copied_protocol='Byte-identical historical protocol bindings; new scheduling registered here',
        algorithm_or_hyperparameter_changes=False, retry_failed_cases=False,
        failed_cases_are_zero_asr=False, original_test_arrays_allowed=False,
        original_validation_arrays_allowed=False, audit_arrays_allowed=False,
        confirmation_arrays_allowed=False)
    write_json(output / 'continuation.json', value)
    write_json(output / 'continuation_registration.json', dict(
        continuation_sha256=sha256(output / 'continuation.json'),
        wrapper_sha256=sha256(__file__), copied_protocol={n: sha256(output / n) for n in PROTOCOL_FILES}))
    validate(bundle, output / 'continuation.json')
    return output / 'continuation.json'


def validate(bundle, path):
    path = Path(path).resolve()
    output = runner.directory(path.parent)
    if path.name != 'continuation.json':
        raise ValueError('Require registered continuation.json')
    value, registration = read(path), read(output / 'continuation_registration.json')
    if (registration['continuation_sha256'] != sha256(path)
            or registration['wrapper_sha256'] != sha256(__file__)
            or value['wrapper_sha256'] != sha256(__file__)
            or value['output_directory'] != str(output) or value['max_parallel'] != 1):
        raise ValueError('Changed continuation registration')
    source = runner.directory(value['source_directory'])
    if source_binding(source) != value['source_binding']:
        raise ValueError('Historical source queue changed')
    for name in PROTOCOL_FILES:
        if sha256(output / name) != registration['copied_protocol'][name] or sha256(output / name) != sha256(source / name):
            raise ValueError('Changed historical protocol copy')
    source_stopped(source)
    if classify_cases(source) != value['preserved_states']:
        raise ValueError('Historical completed/failed/unstarted partition changed')
    if value['scheduled_cases'] != schedule_pending(value['preserved_states']['pending']):
        raise ValueError('Continuation must contain exactly the unstarted cases')
    plan = runner.validate_plan(bundle, output / 'plan.json')
    if (runner.ready(plan['protocol']['predecessor']) != 'READY'
            or runner.gpu_identity() != plan['protocol']['resources']['hardware']):
        raise ValueError('Original runtime/dependency requirements no longer satisfied')
    return value, plan


def finalize(output, value, completed):
    if len({r['population_identity_sha256'] for r in completed}) != 1:
        raise ValueError('Different public populations across cases')
    aggregates = {}
    for name in runner.METHODS:
        rows = sorted((r for r in completed if r['method'] == name), key=lambda r: r['seed'])
        missing = sorted(set(runner.SEEDS) - {r['seed'] for r in rows})
        values = [r['metrics']['victim']['targets']['all']['incremental_asr'] for r in rows]
        aggregates[name] = dict(per_seed=[dict(seed=r['seed'], incremental_asr=v) for r, v in zip(rows, values)],
            missing_seeds=missing, complete_seed_set=not missing,
            mean=statistics.mean(values) if not missing else None,
            sample_sd=statistics.stdev(values) if not missing and len(values) > 1 else None)
    write_json(output / 'summary.json', dict(cases=completed, overall_incremental_asr=aggregates,
        preserved_failed_cases=value['preserved_states']['failed'], source_directory=value['source_directory'],
        reused_development_data=True, failed_cases_are_zero_asr=False,
        permissions_and_total_computation_not_identical=True, author_algorithm_reimplemented=False))
    write_json(output / 'completion.json', dict(status='AUTHOR_CONTINUATION_COMPLETE_WITH_PRESERVED_FAILURE',
        utc=runner.now(), completed_cases=len(completed), resumed_cases=len(value['scheduled_cases']),
        failed_cases=len(value['preserved_states']['failed']), summary_sha256=sha256(output / 'summary.json'),
        original_test_arrays_loaded=False, original_validation_arrays_loaded=False,
        audit_arrays_loaded=False, confirmation_arrays_loaded=False, automatic_followup=False))


def run(bundle, path):
    value, plan = validate(bundle, path)
    output = Path(path).resolve().parent
    runner.check_not_terminal(output)
    write_json(output / 'execution_claim.json', dict(pid=os.getpid(), process=process_identity(os.getpid()),
        utc=runner.now(), plan_sha256=sha256(output / 'plan.json'), continuation_sha256=sha256(path)))
    completed = [completed_record(value['source_directory'], r['method'], r['seed'])
                 for r in value['preserved_states']['completed']]
    active = []
    previous_term = signal.signal(signal.SIGTERM, runner.interrupted)
    try:
        for item in value['scheduled_cases']:
            runner.check_not_terminal(output)
            validate(bundle, path)
            runner.check_host_memory(plan['protocol']['resources'])
            name, seed = item['method'], item['seed']
            command = [sys.executable, '-B', '-m', 'author_comparisons.runner', '--case',
                       '--plan', str(output / 'plan.json'), '--method', name, '--seed', str(seed)]
            log = (output / f'{name}_{seed}.log').open('xb')
            try:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            active = [dict(process=process, log=log)]
            with (output / 'schedule.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(item, utc=runner.now(), pid=process.pid,
                    process=process_identity(process.pid), active=1, device='cuda:0')) + '\n')
            print(f"[author-resume] started {name} seed {seed}; pid={process.pid}", flush=True)
            while process.poll() is None:
                runner.check_not_terminal(output)
                runner.check_host_memory(plan['protocol']['resources'])
                time.sleep(5)
            log.close()
            active = []
            if process.returncode != 0:
                raise RuntimeError(f'{name} seed {seed} failed with exit code {process.returncode}; continuation stopped')
            record = completed_record(output, name, seed)
            if record['population_identity_sha256'] != completed[0]['population_identity_sha256']:
                raise ValueError('New result public target pool differs')
            completed.append(record)
            print(f"[author-resume] completed {name} seed {seed}", flush=True)
        runner.check_not_terminal(output)
        validate(bundle, path)
        finalize(output, value, completed)
    except BaseException as error:
        runner.stop_owned_children(active)
        write_json(output / 'failure.json', dict(status='AUTHOR_CONTINUATION_FAILED_NO_FURTHER_JOBS',
            utc=runner.now(), error=str(error), traceback=traceback.format_exc(),
            completed=[dict(method=r['method'], seed=r['seed']) for r in completed],
            historical_failures_preserved=value['preserved_states']['failed']))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def enqueue(bundle, path):
    value, plan = validate(bundle, path)
    output = Path(path).resolve().parent
    runner.check_not_terminal(output)
    if any((output / n).exists() for n in ('launch_claim.json', 'execution_claim.json', 'launch.json')):
        raise ValueError('Continuation already launched')
    runner.check_host_memory(plan['protocol']['resources'])
    write_json(output / 'launch_claim.json', dict(utc=runner.now(), continuation_sha256=sha256(path)))
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--run', '--continuation', str(Path(path).resolve())]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (output / 'terminal.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='AUTHOR_CONTINUATION_GPU_LAUNCHED', utc=runner.now(), pid=process.pid,
        process=process_identity(process.pid), command=command, max_parallel=1,
        scheduled_cases=value['scheduled_cases'], historical_failed_cases_preserved=True)
    write_json(output / 'launch.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'enqueue', 'run', 'check'):
        action.add_argument('--' + name, action='store_true')
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--continuation', type=Path)
    args = parser.parse_args()
    bundle = load_bundle()
    if args.prepare:
        if args.source_dir is None or args.output_dir is None:
            parser.error('prepare requires source-dir and output-dir')
        print(prepare(bundle, args.source_dir, args.output_dir))
    else:
        if args.continuation is None:
            parser.error('Require continuation.json')
        if args.run:
            run(bundle, args.continuation)
        elif args.enqueue:
            print(json.dumps(enqueue(bundle, args.continuation)))
        else:
            value, _ = validate(bundle, args.continuation)
            print(json.dumps(dict(status='CONTINUATION_VALID', scheduled_cases=value['scheduled_cases'],
                                  graph_arrays_loaded=False, checkpoints_loaded=False)))


if __name__ == '__main__':
    main()
