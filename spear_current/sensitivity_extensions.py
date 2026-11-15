"""GPU budget/availability sensitivity of the existing frozen AV2 uniform G.

This is a one-victim-seed development supplement, not full-method replication.
No original graph, original VAL/TEST, defense, or author-baseline execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from .full_ablation import configure_cuda, read, sha, state, write


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'results/frozen_uniform_support_replication_20261102_v1/plan.json'
VERSION = 'frozen_uniform_sensitivity_extensions_v1'
SEED = 20260621
PREFLIGHT = ('poison_0', 'features_4', 'missing_75', 'immutable_95')


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def prepare(output):
    from .support_replication import verify as parent_verify
    from .extension_protocol import case_matrix
    parent = parent_verify(PARENT)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'results') or output == ROOT/'results':
        raise ValueError('Require a new child of results')
    source = PARENT.parent/'seed20260621/semantic/result.json'
    receipt = read(source.parent/'completion.json')
    if (receipt['result_sha256'] != sha(source)
            or receipt['plan_sha256'] != sha(PARENT)):
        raise ValueError('Existing semantic result receipt changed')
    mask = parent['runtime_bindings']['supports']['semantic']['hard_support']
    cases = case_matrix(mask)
    bound = {**parent['bound_files'], str(PARENT): sha(PARENT),
             str(source): sha(source), str(source.parent/'completion.json'): sha(source.parent/'completion.json')}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'spear_current').glob('*.py'))}
    sources['author_native_runs/dpgba_queue.py'] = sha(ROOT/'author_native_runs/dpgba_queue.py')
    spec = dict(version=VERSION, created_utc=now(), parent_plan=str(PARENT),
        manifest=parent['manifest'], prepared_inputs=parent['prepared_inputs'],
        inputs_sha256=parent['inputs_sha256'], source_mask=mask, cases=cases,
        reference_result=dict(path=str(source), sha256=sha(source)),
        runtime_bindings=parent['runtime_bindings'], bound_files=bound, sources=sources,
        seed=SEED, victim_epochs=200, max_concurrent=2,
        resource_policy=parent['resource_policy'],
        hypothesis='Measure inherited high-effect route sensitivity to fewer attachments/active attributes and fixed availability restrictions',
        controls='One axis per case; fixed generator, target pool, risk labels, topology, optimizer and normal-DEV checkpoint rule',
        feature_ranking='original TRAIN-derived importance order within existing semantic24, whole-group exact knapsack; not re-learned',
        poison_ranking='original attachment order; no selection outside original26; G has already seen all26',
        missing_scope='candidate legal28 derived attributes only, not full784 or raw AV2 sensor/trajectory missingness',
        missing_rule='fixed group-consistent columns set to standardized0 in TRAIN/DEV and protected from trigger overwrite',
        immutable_rule='fixed group-consistent legal28 columns; original values preserved; only write permission changes',
        mask_replicates=1, independent_full_pipeline_seeds=False,
        source_support_envelope=24, actual_effective_support='case.active_coordinate_count; fitter envelope is an upper bound only',
        reference='same historical clean-only development model, including under missingness; not re-adapted to missing columns and not blind',
        evaluation=parent['evaluation'], no_original_graph_reads=True,
        efficiency='separate load, feature-view, victim training and DEV evaluation wall times; synchronized single-scene injection latency; no historical G/selector cost',
        stop_policy='Nonfinite, identity, permission, source or process failure stops new launches; retain all completed evidence',
        performance_policy='Report every predeclared point, including failed historical gates; no adaptive extra points or automatic method replacement',
        preflight=list(PREFLIGHT), execution=parent['execution'])
    output.mkdir(parents=True, exist_ok=False)
    write(output/'plan.json', spec)
    with tarfile.open(output/'source_snapshot.tar', 'x') as archive:
        for relative in sources:
            archive.add(ROOT/relative, arcname=relative)
    write(output/'registration.json', dict(plan_sha256=sha(output/'plan.json'),
        source_snapshot_sha256=sha(output/'source_snapshot.tar'), utc=now(),
        graph_arrays_loaded=False, training_jobs_started=0))
    state(output, 'REGISTERED_GPU_PREFLIGHT_REQUIRED', cases=len(cases), new_victim_fits=13)
    print(str(output/'plan.json'), flush=True)
    return output/'plan.json'


def verify(path, expected=None):
    from .extension_protocol import case_matrix
    path = Path(path).resolve()
    if path.name != 'plan.json' or not path.parent.is_relative_to(ROOT/'results'):
        raise ValueError('Require a registered results/plan.json')
    registration = read(path.parent/'registration.json')
    if (sha(path) != registration['plan_sha256']
            or (expected is not None and sha(path) != expected)
            or sha(path.parent/'source_snapshot.tar') != registration['source_snapshot_sha256']):
        raise ValueError('Extension plan/source snapshot changed')
    plan = read(path)
    if (plan['version'] != VERSION or plan['cases'] != case_matrix(plan['source_mask'])
            or plan['seed'] != SEED or plan['victim_epochs'] != 200
            or plan['max_concurrent'] != 2 or plan['preflight'] != list(PREFLIGHT)):
        raise ValueError('Registered experiment grid changed')
    for filename, digest in plan['bound_files'].items():
        if sha(filename) != digest:
            raise ValueError('Bound source/checkpoint/cache changed: '+filename)
    for relative, digest in plan['sources'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Runtime changed: '+relative)
    return plan


def run_case(path, name, *, short=False, expected=None):
    import torch
    from .support_replication import _cache, acceptability, _verify_prediction_artifacts
    from .support_replication_runtime import restore_uniform
    from .runtime import load_bundle, restore_clean_reference
    from .full_ablation_evaluation import fit_victim, evaluate_case
    from .extension_runtime import (RestrictedSupportGenerator, missing_view,
        budget_selection, check_effective_writes, benchmark, identity_digest)
    from .development_diagnostic import _input_hashes
    from .evaluate import state_hash
    path = Path(path).resolve()
    plan = verify(path, expected)
    expected = sha(path)
    spec = next((s for s in plan['cases'] if s['name'] == name), None)
    if spec is None or (short and name not in PREFLIGHT):
        raise ValueError('Unregistered case')
    output = path.parent/('preflight' if short else 'cases')/name
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    state(output, 'LOADING_REGISTERED_CACHE', case=name, short=short)
    try:
        # Existing reference metrics are reused, never disguised as a new fit.
        if name == 'reference':
            source = read(plan['reference_result']['path'])
            _verify_prediction_artifacts(source['metrics'])
            result = dict(status='CASE_COMPLETE', case=spec, seed=SEED,
                metrics=source['metrics'], acceptability=source['acceptability'],
                victim_record=source['victim_record'], reused=True,
                source=plan['reference_result'], new_optimizer_steps=0,
                timing=dict(new_wall_seconds=time.perf_counter()-start,
                    inherited_source_elapsed_seconds=source['elapsed_seconds']),
                gpu_peak_allocated_bytes=source['gpu_peak_allocated_bytes'])
        else:
            device = configure_cuda(SEED)
            original = _cache(plan)
            before = _input_hashes(original)
            loaded = time.perf_counter()
            inputs = missing_view(original, spec['missing_coordinates'])
            if identity_digest(inputs) != identity_digest(original):
                raise ValueError('Attribute intervention changed identities or labels')
            view_before = _input_hashes(inputs)
            transformed = time.perf_counter()
            bundle = load_bundle(plan['manifest'], verify_assets=False)
            anchor = restore_uniform(bundle, device=device)
            generator = RestrictedSupportGenerator(anchor, plan['source_mask'], spec['active_coordinates']).to(device).eval()
            del anchor
            generator_before = state_hash(generator)
            permission = check_effective_writes(generator, inputs, spec['active_coordinates'], device)
            selection = budget_selection(inputs, spec['poison_count'])
            write(output/'intervention.json', dict(**spec, permission_checks=permission,
                original_input_identity_digest=identity_digest(original),
                normal_features_sha256={r:view_before[r+'_x'] for r in ('train','dev')}))
            state(output, 'GPU_VICTIM_TRAINING', case=name, epochs=2 if short else 200)
            torch.cuda.synchronize(device)
            fitting = time.perf_counter()
            victim, record = fit_victim(generator, inputs, selection, bundle, seed=SEED,
                epochs=2 if short else 200, device=str(device), output_dir=output/'victim')
            torch.cuda.synchronize(device)
            trained = time.perf_counter()
            latency = benchmark(generator, inputs, device)
            if short:
                report = None
            else:
                state(output, 'GPU_FIXED_DEV_EVALUATION', case=name)
                reference = restore_clean_reference(bundle).to(device).eval().requires_grad_(False)
                report = evaluate_case(generator, victim, reference, inputs,
                    device=str(device), output_dir=output/'development')
                _verify_prediction_artifacts(report)
                if report['feature_changes']['maximum_changed_coordinates_per_target'] > spec['active_coordinate_count']:
                    raise ValueError('Actual target edits exceeded active write capacity')
                # The unchanged evaluator summarizes the source24 envelope.
                # Keep it intact, and disclose the stricter effective budget.
                report['effective_permission_metadata'] = dict(
                    source_envelope_coordinates=24, effective_write_coordinates=spec['active_coordinate_count'],
                    output_range_statistics_scope='all source24 envelope coordinates, including unchanged slots',
                    maximum_changed_coordinates_scope='actual changed coordinates; bounded by effective capacity')
            torch.cuda.synchronize(device)
            evaluated = time.perf_counter()
            if (_input_hashes(original) != before or _input_hashes(inputs) != view_before
                    or state_hash(generator) != generator_before):
                raise ValueError('Generator or original/intervened cache mutated')
            result = dict(status='PREFLIGHT_PASS' if short else 'CASE_COMPLETE', case=spec,
                seed=SEED, metrics=report,
                acceptability=None if short else acceptability(report), victim_record=record,
                reused=False, optimizer_steps=dict(generator=0, selector=0, victim=2 if short else 200),
                efficiency=latency, permission_checks=permission,
                source_support_envelope=24, effective_write_capacity=spec['active_coordinate_count'],
                timing=dict(load_and_hash_seconds=loaded-start,
                    feature_view_and_hash_seconds=transformed-loaded,
                    victim_training_seconds=trained-fitting,
                    post_training_benchmark_and_evaluation_seconds=evaluated-trained,
                    total_seconds=evaluated-start),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                device=str(device), generator_frozen=True, original_cache_unchanged=True,
                original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
                audit_arrays_loaded=False, independent_full_pipeline_retraining=False)
        verify(path, expected)
        write(output/'result.json', result)
        write(output/'completion.json', dict(status=result['status'], plan_sha256=expected,
            result_sha256=sha(output/'result.json'), utc=now(), case=name,
            asr_evaluated=not short, new_short_weights_reused=False))
        state(output, result['status'], case=name)
    except BaseException as error:
        write(output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
        state(output, 'FAILED', case=name, error=repr(error))
        raise


def aggregate(path, expected):
    from .support_replication import metric_values, _verify_prediction_artifacts
    plan = verify(path, expected)
    root = Path(path).resolve().parent
    rows, identities = [], None
    for spec in plan['cases']:
        folder = root/'cases'/spec['name']
        receipt, result = read(folder/'completion.json'), read(folder/'result.json')
        if (receipt['status'] != 'CASE_COMPLETE' or receipt['plan_sha256'] != expected
                or receipt['result_sha256'] != sha(folder/'result.json') or result['case'] != spec):
            raise ValueError('Missing or changed completed case')
        identity = _verify_prediction_artifacts(result['metrics'])
        if identities is None:
            identities = identity
        if identity != identities:
            raise ValueError('Public target population/order differs between cases')
        row = dict(case=spec['name'], axis=spec['axis'], poison_count=spec['poison_count'],
            active_coordinates=spec['active_coordinate_count'],
            missing_coordinates=spec['actual_missing_coordinate_count'],
            immutable_coordinates=spec['actual_immutable_coordinate_count'],
            **metric_values(result['metrics']), historical_gates_pass=result['acceptability']['passed'],
            absolute_asr=result['metrics']['models']['victim']['targets']['all']['absolute_asr'],
            conditional_flip=result['metrics']['models']['victim']['targets']['all']['conditional_flip'],
            clean_activation=result['metrics']['models']['victim']['targets']['all']['clean_activation_rate'],
            clean_pair_ap=result['metrics']['models']['victim']['clean_pair_ap'],
            victim_train_seconds=result.get('timing', {}).get('victim_training_seconds'),
            injection_median_ms=result.get('efficiency', {}).get('median_ms'),
            peak_allocated_bytes=result['gpu_peak_allocated_bytes'], reused=result['reused'])
        rows.append(row)
    reference = rows[0]
    for row in rows:
        for key in ('overall_incremental_asr','moving_incremental_asr','adjacent_incremental_fp','clean_pair_auc','clean_positive_recall'):
            row['delta_'+key] = row[key]-reference[key]
    with (root/'case_metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(status='SENSITIVITY_COMPLETE_REVIEW_REQUIRED', plan_sha256=expected,
        cases=rows, completed_cases=len(rows), new_victim_fits=13, reused_reference_cases=1,
        victim_seed=SEED, mask_replicates=1, targets=112, moving_targets=33,
        independent_confirmation=False, independent_full_pipeline_retraining=False,
        missing_scope=plan['missing_scope'], reference=plan['reference'],
        all_points_reported=True, method_auto_promoted=False,
        historical_generator_and_selector_training_cost_included=False, utc=now())
    write(root/'aggregate.json', summary)
    write(root/'completion.json', dict(status=summary['status'], plan_sha256=expected,
        aggregate_sha256=sha(root/'aggregate.json'), completed_cases=len(rows), utc=now()))
    state(root, summary['status'], completed_cases=len(rows))
    return summary


def launch(path):
    from .support_replication import _env
    path = Path(path).resolve()
    verify(path)
    expected, root = sha(path), path.parent
    with (root/'launch_claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), utc=now(), plan_sha256=expected), stream)
    command = [sys.executable, '-B', '-u', '-m', 'spear_current.sensitivity_extensions',
               '--queue', '--plan', str(path), '--expected-sha256', expected]
    with (root/'queue.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=_env(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, command=command, utc=now(), plan_sha256=expected)
    write(root/'launch.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def queue(path, expected):
    from .support_replication import _env, _resources
    path = Path(path).resolve()
    root = path.parent
    plan = verify(path, expected)
    write(root/'execution_claim.json', dict(pid=os.getpid(), utc=now(), plan_sha256=expected))
    active, done = [], []

    def interrupt(signum, frame):
        raise RuntimeError('Extension queue interrupted: '+str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def resources():
        began = time.monotonic()
        while True:
            verify(path, expected)
            if (root/'CANCEL').exists():
                raise RuntimeError('Extension cancelled')
            value = _resources(plan)
            if value['ready']:
                return value
            if time.monotonic()-began > plan['resource_policy']['max_wait_seconds']:
                raise TimeoutError('GPU resources unavailable; no CPU fallback')
            state(root, 'WAITING_FOR_GPU_HEADROOM', resources=value)
            time.sleep(5)

    try:
        # The detached queue first completes four real-input CUDA smoke tests.
        for name in PREFLIGHT:
            resources()
            state(root, 'GPU_ENGINEERING_PREFLIGHT', case=name)
            run_case(path, name, short=True, expected=expected)
        write(root/'preflight/completion.json', dict(status='PREFLIGHT_PASS',
            cases=list(PREFLIGHT), plan_sha256=expected, no_asr_selection=True, utc=now()))
        pending = [s['name'] for s in plan['cases']]
        waiting_since = time.monotonic()
        while pending or active:
            verify(path, expected)
            if (root/'CANCEL').exists():
                raise RuntimeError('Extension cancelled')
            for job in list(active):
                code = job['process'].poll()
                if code is not None:
                    job['log'].close(); active.remove(job)
                    if code:
                        raise RuntimeError(f"Case {job['name']} exited {code}")
                    done.append(job['name'])
            if pending and len(active) < plan['max_concurrent']:
                resource = _resources(plan)
                if resource['ready']:
                    name = pending.pop(0)
                    command = [sys.executable, '-B', '-u', '-m', 'spear_current.sensitivity_extensions',
                        '--case', name, '--plan', str(path), '--expected-sha256', expected]
                    log = (root/(name+'.log')).open('x')
                    process = subprocess.Popen(command, cwd=ROOT, env=_env(), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active.append(dict(name=name, process=process, log=log))
                    write(root/(name+'_launch.json'), dict(pid=process.pid, resources=resource, command=command, utc=now()))
                    waiting_since = time.monotonic()
                elif time.monotonic()-waiting_since > plan['resource_policy']['max_wait_seconds']:
                    raise TimeoutError('GPU resource wait exceeded limit; no CPU fallback')
            state(root, 'GPU_SENSITIVITY_RUNNING', completed_cases=len(done), pending_cases=len(pending),
                active=[dict(name=j['name'], pid=j['process'].pid) for j in active])
            if pending or active:
                time.sleep(3)
        aggregate(path, expected)
    except BaseException as error:
        for job in active:
            if job['process'].poll() is None:
                os.killpg(job['process'].pid, signal.SIGTERM)
            job['log'].close()
        write(root/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            completed=done, utc=now(), own_children_only_terminated=True))
        state(root, 'FAILED', error=repr(error), completed_cases=len(done))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', type=Path)
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--queue', action='store_true')
    modes.add_argument('--case')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if args.prepare is not None:
        prepare(args.prepare)
    elif args.plan is None:
        parser.error('Require --plan')
    elif args.launch:
        launch(args.plan)
    elif args.queue:
        queue(args.plan, args.expected_sha256)
    else:
        run_case(args.plan, args.case, expected=args.expected_sha256)


if __name__ == '__main__':
    main()
