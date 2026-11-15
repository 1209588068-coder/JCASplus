"""Registered external-method pilots, queued behind a specific successful run.

No original validation/test or internal audit arrays. No auto-retry, hidden
parameter search, performance-based pruning, or replacement of retained models.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import torch

from spear_current.development_diagnostic import registered_rows, read_inputs, _input_hashes
from spear_current.runtime import load_bundle, sha256
from .evidence import (evaluate_dev, fit_classifier, identity_hash, save_state,
                       tensor_hash, validate_fit_result, write_json)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('spear', 'ugba', 'cgba', 'gta')
DEFERRED = {'dpgba': 'NOT_IMPLEMENTED_NOT_QUEUED: author core unavailable; DOMINANT reference selection, discriminator graph construction and embedding loss details not verified'}
SEEDS = (20260621, 20260622, 20260623)
VERSION = 'external_method_av2_development_v1'
PARENT_STATUS = 'KL_ABLATION_COMPLETE_REVIEW_REQUIRED'


def now():
    return datetime.now(timezone.utc).isoformat()


def method_module(method):
    if method not in METHODS:
        raise ValueError('Unregistered method')
    return importlib.import_module('comparison_methods.' + ('subgraph' if method in ('ugba', 'dpgba', 'gta') else method))


def configurations():
    result = {}
    for method in METHODS:
        module = method_module(method)
        if method in ('ugba', 'dpgba', 'gta') and method not in module.SUPPORTED_METHODS:
            raise ValueError('Cannot register an unimplemented comparison method: ' + method)
        # All defaults and method-specific differences are expanded before
        # registration. Later changes to defaults invalidate the prepared plan.
        result[method] = module.validate_config(deepcopy(module.DEFAULT_CONFIG))
    return result


def result_directory(path):
    path = Path(path).resolve()
    if path == ROOT / 'results' or not path.is_relative_to(ROOT / 'results'):
        raise ValueError('Require a new child of JCAS/results')
    return path


def process_identity(pid):
    """PID plus kernel start ticks prevents mistaking a reused PID for a job."""
    try:
        raw = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=int(fields[19]),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def dependency_binding(directory):
    directory = result_directory(directory)
    paths = {name: directory / name for name in ('plan.json', 'registration.json', 'execution_claim.json', 'launch.json')}
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if (values['plan.json']['protocol']['version'] != 'stage_specific_clean_kl_ablation_v1'
            or values['registration.json']['plan_sha256'] != sha256(paths['plan.json'])
            or values['execution_claim.json']['plan_sha256'] != sha256(paths['plan.json'])
            or values['launch.json']['pid'] != values['execution_claim.json']['pid']):
        raise ValueError('Unexpected or inconsistent predecessor identity')
    return dict(directory=str(directory), files={k: sha256(v) for k, v in paths.items()},
                process=process_identity(values['launch.json']['pid']), required_status=PARENT_STATUS)


def predecessor_state(binding):
    directory = Path(binding['directory'])
    for name, digest in binding['files'].items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            raise ValueError('Predecessor registration changed')
    if (directory / 'failure.json').exists():
        raise ValueError('Predecessor failed; comparison must not start automatically')
    expected = binding['process']
    alive = expected is not None and process_identity(expected['pid']) == expected
    completion = directory / 'completion.json'
    if completion.exists():
        # The predecessor writes JSON directly. A live process may have just
        # created an incomplete file; validate it only after that exact process
        # exits, rather than converting a normal write race into queue failure.
        if alive:
            return 'WAITING_FOR_PROCESS_EXIT'
        value = json.loads(completion.read_text())
        expected_runs = [dict(seed=s, arm=a) for s in SEEDS for a in ('dual', 'single_proxy1', 'disabled')]
        if (value['status'] != PARENT_STATUS or value['completed'] != expected_runs
                or sha256(directory / 'summary.json') != value['summary_sha256']
                or any(value.get(k) is not False for k in ('original_validation_arrays_loaded',
                    'original_test_arrays_loaded', 'audit_arrays_loaded', 'confirmation_arrays_loaded'))):
            raise ValueError('Predecessor did not complete the registered nine valid fits')
        return 'READY'
    if not alive:
        raise ValueError('Predecessor exited without successful completion evidence')
    return 'WAITING_FOR_PREDECESSOR'


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p)
            for folder in ('comparison_methods', 'spear_current')
            for p in sorted((ROOT / folder).glob('*.py'))}


def specification(bundle, dependency):
    return dict(version=VERSION, methods=list(METHODS), deferred_methods=DEFERRED, seeds=list(SEEDS),
        order='seed-major, fixed method order, serial; performance never skips later cases',
        configs=configurations(), sources=registered_rows(bundle), predecessor=dependency,
        common=dict(victim='GraphSAGE', hidden=80, dropout=.5, lr=.01, weight_decay=.0005,
            victim_epochs=200, clean_reference_epochs=200, poison_count=26,
            feature_method_coordinate_budget=24, structural_trigger_nodes=3,
            normalizer='retained TRAIN-only 784-dimensional normalization',
            victim_checkpoint='normal DEV AUC, strict improvement, earliest tie',
            targets='existing identity-hash negative public DEV pool; 112 targets, 33 moving',
            dirty_candidate_policy='eligible true negatives in original idx_attach union idx_unlabeled reservoir',
            clean_label_candidate_policy='eligible true positives in original idx_train; labels unchanged',
            selection='method-specific train-only; NOT fixed historical 26 nodes or 24-coordinate support',
            clean_reference='fresh per method and seed, identical untriggered supervision and initial weights as victim'),
        research_question='External algorithms on the same AV2 pair-node task, with explicit original attack permissions and budgets',
        evidence_limits=[
            'Fixed-topology feature and structure-injection methods are separate permission families',
            'CGBA is clean-label; other methods use dirty target labels',
            'CGBA poisons existing supervised positives; dirty-label methods add 26 reservoir nodes to supervision; counts are reported, not claimed identical',
            'Synthetic pair-nodes have no asserted physical vehicle realization',
            'Native selector/pretraining/generator costs differ and are reported; not equal total FLOPs',
            'Retained current models have a multi-stage trained parent; old rows are not equal-total-budget fresh baselines',
            'Reused internal DEV is development evidence, not a blind held-out confirmation'],
        failure_policy='contract/numerical failure stops queue, preserves outputs; no retry or hyperparameter alteration',
        parent_gate_policy='successful completion required, not passing ASR thresholds',
        performance_policy='report all fixed cases including weak ASR; no automatic promotion',
        graph_read_policy='original TRAIN train512/dev128 only',
        original_validation_arrays_allowed=False, original_test_arrays_allowed=False,
        internal_audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        max_wait_seconds=72 * 3600, poll_seconds=30,
        manifest=dict(path=str(bundle.manifest_path), sha256=sha256(bundle.manifest_path)),
        code_sha256=source_hashes(),
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            sklearn=package_version('scikit-learn'), scipy=package_version('scipy'),
            torch_geometric=package_version('torch-geometric')),
        automatic_followup=False, automatic_promotion=False)


def prepare(bundle, output, after):
    dependency = dependency_binding(after)
    predecessor_state(dependency)
    spec = specification(bundle, dependency)
    output = result_directory(output)
    if output == Path(dependency['directory']) or output.is_relative_to(dependency['directory']):
        raise ValueError('Comparison output must be separate from predecessor')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', dict(created_utc=now(), protocol=spec))
    with tarfile.open(output / 'source_snapshot.tar', 'x') as archive:
        for name in spec['code_sha256']:
            archive.add(ROOT / name, arcname=name)
    if specification(bundle, dependency) != spec:
        raise ValueError('Code or protocol changed during preparation')
    write_json(output / 'registration.json', dict(plan_sha256=sha256(output / 'plan.json'),
        source_snapshot_sha256=sha256(output / 'source_snapshot.tar'),
        graph_arrays_loaded=False, model_checkpoints_loaded=False, training_started=False))
    return output / 'plan.json'


def validate_plan(bundle, path):
    path = Path(path).resolve()
    result_directory(path.parent)
    if path.name != 'plan.json':
        raise ValueError('Require registered plan.json')
    record = json.loads((path.parent / 'registration.json').read_text())
    if (sha256(path) != record['plan_sha256']
            or sha256(path.parent / 'source_snapshot.tar') != record['source_snapshot_sha256']):
        raise ValueError('Changed plan/source snapshot')
    plan = json.loads(path.read_text())
    if plan['protocol'] != specification(bundle, plan['protocol']['predecessor']):
        raise ValueError('Registered code, data, defaults or environment changed')
    return plan


def candidate_indices(inputs, method):
    train, idx = inputs['combined']['train'], inputs['indices']
    pool = idx['idx_train'] if method == 'cgba' else torch.unique(torch.cat([idx['idx_attach'], idx['idx_unlabeled']]))
    wanted = 1 if method == 'cgba' else 0
    return pool[train['eligible'][pool] & (train['y'][pool] == wanted)].sort().values


def fit_method(method, train, idx_train, candidates, seed, config, progress):
    module = method_module(method)
    arguments = (train['x'], train['edge_index'], train['y'], idx_train, candidates)
    callback = (lambda **event: progress(event)) if method == 'spear' else progress
    kwargs = dict(seed=seed, config=config, progress=callback)
    if method in ('ugba', 'dpgba', 'gta'):
        return module.fit_subgraph(method, *arguments, **kwargs)
    return getattr(module, 'fit_' + method)(*arguments, **kwargs)


def run_case(inputs, sources, method, seed, config, output, progress):
    output.mkdir(exist_ok=False)
    train, dev, indices = inputs['combined']['train'], inputs['combined']['dev'], inputs['indices']
    candidates = candidate_indices(inputs, method)
    if len(candidates) < config['poison_count']:
        raise ValueError('Insufficient candidates; do not relax labels or borrow another split')
    before = _input_hashes(inputs)
    write_json(output / 'case.json', dict(method=method, seed=seed, config=config,
        candidates=candidates.tolist(), candidate_sha256=tensor_hash(candidates),
        fresh_start=True, pretrained_attack_checkpoint_loaded=False,
        train_scenes=512, dev_scenes=128, started_utc=now()))
    start = time.monotonic()
    fitted = fit_method(method, train, indices['idx_train'], candidates, seed, config,
                        lambda r: progress(dict(r, stage='method_training', method_stage=r.get('stage'))))
    if before != _input_hashes(inputs):
        raise ValueError('Method mutated shared original inputs')
    validate_fit_result(fitted, train, indices['idx_train'], candidates, method=method, config=config)
    save_state(output / 'generator.pt', fitted['generator'].state_dict())
    save_state(output / 'poisoned_train.pt', {k: fitted[k].detach().cpu() for k in
        ('attachment_indices', 'poisoned_x', 'poisoned_edge_index', 'poisoned_y', 'fit_indices')})
    write_json(output / 'method_record.json', fitted['record'])
    write_json(output / 'training_binding.json', dict(
        attachment_indices=fitted['attachment_indices'].tolist(),
        label_changes=int((fitted['poisoned_y'] != train['y']).sum()),
        original_nodes=len(train['x']), poisoned_nodes=len(fitted['poisoned_x']),
        original_edges=train['edge_index'].shape[1], poisoned_edges=fitted['poisoned_edge_index'].shape[1],
        fit_indices=fitted['fit_indices'].tolist(),
        selector_generator_elapsed_seconds=time.monotonic() - start))
    classifier_seed = seed + 9000
    clean, clean_record = fit_classifier(train['x'], train['edge_index'], train['y'], fitted['fit_indices'],
        dev, seed=classifier_seed, output=output / 'clean_dev_trace.csv',
        progress=lambda r: progress(dict(r, stage='fit_clean_reference')))
    save_state(output / 'clean_reference.pt', clean.state_dict())
    write_json(output / 'clean_reference_record.json', clean_record)
    victim, victim_record = fit_classifier(fitted['poisoned_x'], fitted['poisoned_edge_index'],
        fitted['poisoned_y'], fitted['fit_indices'], dev, seed=classifier_seed,
        output=output / 'victim_dev_trace.csv', progress=lambda r: progress(dict(r, stage='fit_victim')))
    if any(clean_record[k] != victim_record[k] for k in
           ('initialization_sha256', 'fit_rng_sha256', 'supervision_indices_sha256')):
        raise ValueError('Matched fresh clean/victim initial state or supervision mismatch')
    save_state(output / 'victim.pt', victim.state_dict())
    write_json(output / 'victim_record.json', victim_record)
    summary = evaluate_dev(inputs['parts']['dev'], sources['dev'], fitted['generator'],
        dict(victim=victim, clean_reference=clean), output / 'development',
        method=method, config=config, progress=progress)
    if (summary['targets'] != 112 or summary['metrics']['victim']['targets']['moving']['targets'] != 33
            or before != _input_hashes(inputs)):
        raise ValueError('Changed shared inputs or public target population')
    completion = dict(status='COMPARISON_CASE_COMPLETE', seed=seed, method=method,
        elapsed_seconds=time.monotonic() - start, metrics=summary['metrics'],
        population_identity_sha256=summary['population_identity_sha256'],
        original_test_arrays_loaded=False, internal_audit_arrays_loaded=False,
        files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output / 'completion.json', completion)
    return completion


def aggregate(completed):
    if [(r['seed'], r['method']) for r in completed] != [(s, m) for s in SEEDS for m in METHODS]:
        raise ValueError('Missing, duplicate or reordered cases')
    if len({r['population_identity_sha256'] for r in completed}) != 1:
        raise ValueError('Methods/seeds used different target identities')
    metrics = {}
    for method in METHODS:
        rows = [r for r in completed if r['method'] == method]
        values = []
        for row in rows:
            v, c = row['metrics']['victim'], row['metrics']['clean_reference']
            values.append(dict(overall_incremental_asr=v['targets']['all']['incremental_asr'],
                moving_incremental_asr=v['targets']['moving']['incremental_asr'],
                slow_incremental_asr=v['targets']['slow']['incremental_asr'],
                clean_pair_auc=v['clean_pair_auc'], clean_pair_ap=v['clean_pair_ap'],
                clean_positive_recall=v['clean_pair_positive_recall'],
                clean_moving_recall=v['normal_by_pair_motion']['moving']['clean_pair_positive_recall'],
                adjacent_incremental_fp=v['collateral']['adjacent']['incremental_fp'],
                nonincident_incremental_fp=v['collateral']['nonincident']['incremental_fp'],
                adjacent_positive_suppression=v['collateral']['adjacent']['positive_suppression'],
                clean_reference_incremental_asr=c['targets']['all']['incremental_asr']))
        metrics[method] = {}
        for key in values[0]:
            items = [r[key] for r in values]
            metrics[method][key] = dict(per_seed=items,
                mean=statistics.mean(items) if all(i is not None for i in items) else None,
                sample_sd=statistics.stdev(items) if all(i is not None for i in items) else None)
    return dict(status='EXTERNAL_COMPARISON_COMPLETE_REVIEW_REQUIRED', metrics_mean_sd=metrics,
        distinct_targets=112, distinct_moving_targets=33, reused_development_data=True,
        repeated_seeds_are_not_independent_targets=True, permissions_not_identical=True,
        total_computation_not_equal=True, current_multistage_method_not_a_fresh_budget_matched_row=True,
        automatic_promotion=False, original_test_arrays_loaded=False, internal_audit_arrays_loaded=False,
        deferred_methods=DEFERRED, cases=completed)


def run(bundle, path):
    plan = validate_plan(bundle, path)
    output = Path(path).resolve().parent
    if any((output / name).exists() for name in ('CANCEL', 'cancelled.json', 'failure.json', 'completion.json')):
        raise ValueError('Cancelled, failed or completed plan cannot start training')
    if predecessor_state(plan['protocol']['predecessor']) != 'READY':
        raise ValueError('Predecessor must finish successfully before any comparison training')
    write_json(output / 'execution_claim.json', dict(pid=os.getpid(), at=now(), plan_sha256=sha256(path)))
    torch.set_num_threads(2)
    ledger, completed, started = [], [], time.monotonic()
    with (output / 'progress.jsonl').open('x', buffering=1) as stream:
        def progress(record):
            value = dict(record, utc=now(), elapsed_seconds=time.monotonic() - started)
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            if value.get('epoch', 0) % 20 == 0 or value.get('epoch') == 199:
                print(json.dumps(value, allow_nan=False), flush=True)
        try:
            inputs = read_inputs(bundle, plan['protocol']['sources'], ledger)
            write_json(output / 'input_graph_access_ledger.json', ledger)
            write_json(output / 'input_hashes.json', _input_hashes(inputs))
            for seed in SEEDS:
                child = output / ('seed' + str(seed))
                child.mkdir(exist_ok=False)
                for method in METHODS:
                    validate_plan(bundle, path)
                    progress(dict(seed=seed, method=method, stage='case_started'))
                    record = run_case(inputs, plan['protocol']['sources'], method, seed,
                        plan['protocol']['configs'][method], child / method,
                        lambda r: progress(dict(r, seed=seed, method=method)))
                    completed.append(record)
                    progress(dict(seed=seed, method=method, stage='case_complete'))
                    gc.collect()
            validate_plan(bundle, path)
            summary = aggregate(completed)
            write_json(output / 'summary.json', summary)
            write_json(output / 'completion.json', dict(status=summary['status'], completed_utc=now(),
                completed=[dict(seed=r['seed'], method=r['method']) for r in completed],
                summary_sha256=sha256(output / 'summary.json'), graph_reads=len(ledger),
                original_test_arrays_loaded=False, original_validation_arrays_loaded=False,
                internal_audit_arrays_loaded=False, confirmation_arrays_loaded=False,
                automatic_followup=False, automatic_promotion=False))
        except BaseException as error:
            write_json(output / 'failure.json', dict(status='FAILED_NO_FURTHER_JOBS', at=now(),
                error=str(error), type=type(error).__name__, traceback=traceback.format_exc(),
                completed=[dict(seed=r['seed'], method=r['method']) for r in completed]))
            raise


def wait_and_run(bundle, path):
    plan = validate_plan(bundle, path)
    output, started = Path(path).resolve().parent, time.monotonic()
    write_json(output / 'watcher_claim.json', dict(pid=os.getpid(), at=now(), plan_sha256=sha256(path)))
    try:
        with (output / 'wait.jsonl').open('x', buffering=1) as stream:
            while True:
                if (output / 'CANCEL').exists():
                    write_json(output / 'cancelled.json', dict(status='CANCELLED_BEFORE_TRAINING', at=now()))
                    return
                validate_plan(bundle, path)
                state = predecessor_state(plan['protocol']['predecessor'])
                stream.write(json.dumps(dict(at=now(), state=state, training_started=False)) + '\n')
                if state == 'READY':
                    break
                if time.monotonic() - started > plan['protocol']['max_wait_seconds']:
                    raise TimeoutError('Predecessor did not finish within registered wait bound')
                time.sleep(plan['protocol']['poll_seconds'])
        run(bundle, path)
    except BaseException as error:
        if not (output / 'failure.json').exists():
            write_json(output / 'failure.json', dict(status='WAIT_FAILED_NO_TRAINING', at=now(),
                error=str(error), type=type(error).__name__, traceback=traceback.format_exc()))
        raise


def enqueue(bundle, path):
    plan = validate_plan(bundle, path)
    path, output = Path(path).resolve(), Path(path).resolve().parent
    predecessor_state(plan['protocol']['predecessor'])
    if any((output / name).exists() for name in ('execution_claim.json', 'watcher_claim.json', 'failure.json')):
        raise ValueError('Plan already started or failed; no automatic retry')
    write_json(output / 'launch_claim.json', dict(at=now(), plan_sha256=sha256(path)))
    command = [sys.executable, '-B', '-m', 'comparison_methods.runner', '--wait-and-run', '--plan', str(path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (output / 'terminal.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(status='QUEUED_BEHIND_KL_ABLATION', at=now(), pid=process.pid,
        process=process_identity(process.pid), command=command,
        predecessor=plan['protocol']['predecessor']['directory'], cases=len(METHODS) * len(SEEDS),
        expected_state='waiting, not training; see wait.jsonl', cancel_file=str(output / 'CANCEL'))
    write_json(output / 'launch.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('check-only', 'prepare', 'enqueue', 'run', 'wait-and-run'):
        action.add_argument('--' + name, action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    bundle = load_bundle()
    if args.prepare:
        if args.output_dir is None or args.after is None or args.plan is not None:
            parser.error('--prepare requires --output-dir and --after only')
        print(prepare(bundle, args.output_dir, args.after), flush=True)
    elif args.check_only:
        if args.plan or args.output_dir or args.after:
            parser.error('--check-only accepts no other arguments')
        print(json.dumps(dict(status='EXTERNAL_COMPARISON_CONFIG_OK', configs=configurations(),
            graph_arrays_loaded=False, training_started=False)), flush=True)
    else:
        if args.plan is None or args.output_dir or args.after:
            parser.error('Execution requires --plan only')
        if args.enqueue:
            print(json.dumps(enqueue(bundle, args.plan)), flush=True)
        elif args.wait_and_run:
            wait_and_run(bundle, args.plan)
        else:
            run(bundle, args.plan)


if __name__ == '__main__':
    main()
