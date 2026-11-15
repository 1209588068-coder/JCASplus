"""GPU victim-seed replication of frozen uniform/24-coordinate supports.

Local AV2/self-owned GNN research. Four existing support lists share one
already-trained uniform generator; only fresh victims are optimized. This is
not independent selector/generator retraining or historical CPU replay.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

from .full_ablation import configure_cuda, event, read, sha, state, write


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_PLAN = ROOT / 'results/full_gpu_module_ablation_20261030_v1/plan.json'
VERSION = 'frozen_uniform_support_victim_replication_v1'
SEEDS = (20260621, 20260622, 20260623)
ARMS = ('semantic', 'random_semantic_s20260621', 'random_semantic_s20260622',
        'random_semantic_s20260623')
GATES = dict(overall_incremental_asr_min=.85, moving_incremental_asr_min=.85,
             adjacent_incremental_fp_max_exclusive=.05, clean_auc_drop_max=.005,
             clean_positive_recall_drop_max=.05, reject_clean_single_class=True)
METRICS = ('overall_incremental_asr', 'moving_incremental_asr', 'clean_pair_auc',
           'clean_positive_recall', 'adjacent_incremental_fp',
           'nonincident_incremental_fp', 'adjacent_positive_suppression',
           'clean_reference_incremental_asr')


def now():
    return datetime.now(timezone.utc).isoformat()


def case_matrix():
    return [dict(seed=seed, arm=arm) for seed in SEEDS for arm in ARMS]


matrix = case_matrix


def _directory(path):
    path = Path(path).resolve()
    if path == ROOT / 'results' or not path.is_relative_to(ROOT / 'results'):
        raise ValueError('Require a new child directory of JCAS/results')
    return path


def _binding_files(value):
    """Collect explicit path/SHA pairs, never the legacy original_path labels."""
    result = {}
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            path = str(Path(value['path']).resolve())
            if path in result and result[path] != value['sha256']:
                raise ValueError('Conflicting source binding')
            result[path] = value['sha256']
        for child in value.values():
            for path, digest in _binding_files(child).items():
                if path in result and result[path] != digest:
                    raise ValueError('Conflicting recursive source binding')
                result[path] = digest
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.update(_binding_files(child))
    return result


def _validate_supports(bindings):
    from .support_schema import SELECTED_COORDINATES, EXCLUDED_COORDINATES, support_groups
    supports = bindings.get('supports', {})
    if set(supports) != set(ARMS):
        raise ValueError('Require semantic and all three registered random supports')
    for arm, record in supports.items():
        coordinates, mask = record['selected_coordinates'], record['hard_support']
        if (coordinates != list(SELECTED_COORDINATES) or not isinstance(mask, list)
                or len(mask) != 30 or any(type(x) is not bool for x in mask)
                or sum(mask) != 24 or record['hard_support_capacity'] != 24):
            raise ValueError('Expected original-order, exact24 frozen support: ' + arm)
        selected = {c for c, enabled in zip(coordinates, mask) if enabled}
        if selected.intersection(EXCLUDED_COORDINATES):
            raise ValueError('Focal metadata cannot be written')
        for group in support_groups('semantic'):
            if selected.intersection(group) and not set(group) <= selected:
                raise ValueError('Registered semantic support splits a source group')


def protocol(manifest=None, *, parent_plan=DEFAULT_PARENT_PLAN):
    """File/metadata validation only: no checkpoint or graph deserialization."""
    from .runtime import load_bundle
    from .support_replication_runtime import source_bindings
    parent_plan = Path(parent_plan).resolve()
    parent = read(parent_plan)
    if (parent.get('version') != 'full_gpu_module_ablation_v1'
            or parent.get('seeds') != list(SEEDS)
            or parent.get('evaluation', {}).get('train_scenes') != 512
            or parent['evaluation'].get('dev_scenes') != 128
            or parent['evaluation'].get('targets') != 112
            or parent['evaluation'].get('moving_targets') != 33):
        raise ValueError('Unexpected original GPU cache parent protocol')
    manifest = Path(parent['manifest'] if manifest is None else manifest).resolve()
    if sha(manifest) != parent['manifest_sha256']:
        raise ValueError('Current bundle differs from the original cache manifest')
    cache = Path(parent['prepared_inputs']).resolve()
    if (not cache.is_relative_to(ROOT / 'results')
            or sha(cache) != parent['inputs_sha256']):
        raise ValueError('Original TRAIN/dev cache changed')
    bundle = load_bundle(manifest, verify_assets=False)
    bindings = source_bindings(bundle)
    _validate_supports(bindings)
    bound = {str(parent_plan): sha(parent_plan), str(manifest): sha(manifest),
             str(cache): parent['inputs_sha256'], **_binding_files(bindings)}
    required = ('normalizer', 'indices', 'input_contract', 'context_binding',
                'score_permutation', 'clean_reference')
    for key in required:
        path = bundle.asset_path(key)
        source = parent['clean_assets'][key]
        if sha(path) != source['sha256']:
            raise ValueError('Clean/cache asset differs from the parent: ' + key)
        bound[str(path)] = source['sha256']
    # Preserve and validate the original plan's source archive, not current
    # files changed later for independently registered experiments.
    snapshot = ROOT / 'artifacts/source_snapshots/full_gpu_module_ablation_v1_20261031'
    if sha(snapshot / 'original_plan.json') != sha(parent_plan):
        raise ValueError('Original GPU source snapshot does not match its plan')
    bound[str(snapshot / 'original_plan.json')] = sha(parent_plan)
    for relative, digest in parent['sources'].items():
        path = snapshot / relative
        if sha(path) != digest:
            raise ValueError('Original GPU registered source snapshot changed: ' + relative)
        bound[str(path)] = digest
    for path, digest in bound.items():
        if sha(path) != digest:
            raise ValueError('Missing/changed registered source: ' + path)
    sources = {str(p.relative_to(ROOT)): sha(p)
               for p in sorted((ROOT / 'spear_current').glob('*.py'))}
    sources['author_native_runs/dpgba_queue.py'] = sha(ROOT / 'author_native_runs/dpgba_queue.py')
    return dict(version=VERSION, manifest=str(manifest), parent_plan=str(parent_plan),
        prepared_inputs=str(cache), inputs_sha256=parent['inputs_sha256'],
        bound_files=bound, sources=sources, runtime_bindings=bindings,
        seeds=list(SEEDS), arms=list(ARMS), cases=case_matrix(), victim_epochs=200,
        hypothesis='Test fixed high-effect supports under fresh victim initialization; preserve all random controls',
        changed_variable='four previously registered frozen support lists, paired by victim seed',
        generator_start='same completed uniform trained with original full30; no new generator updates',
        inherited=['SPEAR base generator and conditioning GCN', 'trained uniform residual',
                   'dual clean-KL training history', 'historical full30 writable-field training'],
        support_policy='exact24 from legal28; focal357/715 protected; frozen source masks, no new drawing',
        output_mapping='identity; no generated-feature clipping and no distribution regularizer',
        optimizer_steps=dict(generator=0, selector=0, surrogate=0, discriminator=0, victim=200),
        omitted_historical_surrogate_updates=200,
        omission_reason='fixed G/support do not consume the subsequently updated surrogate in fresh victim fitting',
        victim=dict(name='GraphSAGE', input_dim=784, hidden=80, classes=2, layers=2,
                    dropout=.5, lr=.01, weight_decay=.0005,
                    checkpoint='earliest maximum normal DEV AUC; no ASR-based selection'),
        clean_reference='same pre-existing clean-only development reference in all cases; not a blind reference',
        evaluation=dict(train_scenes=512, dev_scenes=128, targets=112, moving_targets=33,
            normal_pairs=2223, poison_attachments=26, outer_rows=538,
            original_source_split='TRAIN', reused_development=True,
            audit_arrays_allowed=False, confirmation_arrays_allowed=False,
            original_validation_arrays_allowed=False, original_test_arrays_allowed=False),
        execution=dict(device='cuda:0', precision='float32', amp=False, tf32=False,
            torch_threads=1, max_active_cases=2, no_cpu_fallback=True,
            bitwise_reproducibility_claim=False),
        resource_policy=dict(min_host_available_bytes=16 * 1024**3,
            min_cuda_free_bytes=3 * 1024**3, allow_other_gpu_processes=True,
            max_active_cases=2, poll_seconds=5, max_wait_seconds=72 * 3600,
            no_external_process_termination=True, headroom_is_not_an_oom_guarantee=True),
        preflight=dict(arms=list(ARMS), seed=SEEDS[0], victim_epochs=2,
            asr_evaluation=False, short_weights_reused=False), gates=dict(GATES),
        aggregation='per-seed semantic minus mean of all three random supports; then mean and sample SD across three victim seeds',
        independent_full_pipeline_seeds=False, independent_selector_retraining=False,
        inherited_attack_checkpoint=True, historical_cpu_replay_claim=False,
        automatic_promotion=False, automatic_followup=False,
        failure_policy='source/identity/nonfinite/child error stops new launches and preserves evidence; no retry or parameter changes',
        performance_failure_policy='retain and report all twelve registered cases without retuning')


def prepare(output_dir, manifest=None, *, parent_plan=DEFAULT_PARENT_PLAN):
    output = _directory(output_dir)
    spec = protocol(manifest, parent_plan=parent_plan)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'plan.json', dict(created_utc=now(), **spec))
    with tarfile.open(output / 'source_snapshot.tar', 'x') as archive:
        for relative in spec['sources']:
            archive.add(ROOT / relative, arcname=relative)
    if spec != protocol(manifest, parent_plan=parent_plan):
        raise ValueError('Sources changed during metadata registration')
    write(output / 'registration.json', dict(plan_sha256=sha(output / 'plan.json'),
        source_snapshot_sha256=sha(output / 'source_snapshot.tar'), created_utc=now(),
        graph_arrays_loaded=False, checkpoint_tensors_loaded=False, training_jobs_started=0))
    state(output, 'REGISTERED_GPU_PREFLIGHT_REQUIRED', cases=12)
    print(str(output / 'plan.json'), flush=True)
    return output / 'plan.json'


def verify(plan_path, expected_sha256=None):
    path = Path(plan_path).resolve()
    if path.name != 'plan.json' or _directory(path.parent) != path.parent:
        raise ValueError('Require a registered plan.json below results/')
    registration = read(path.parent / 'registration.json')
    digest = sha(path)
    if (digest != registration['plan_sha256']
            or (expected_sha256 is not None and digest != expected_sha256)
            or sha(path.parent / 'source_snapshot.tar') != registration['source_snapshot_sha256']):
        raise ValueError('Plan or source snapshot changed')
    plan = read(path)
    if (plan.get('version') != VERSION or plan.get('cases') != case_matrix()
            or plan.get('seeds') != list(SEEDS) or plan.get('arms') != list(ARMS)
            or plan.get('gates') != GATES or plan.get('victim_epochs') != 200
            or plan.get('optimizer_steps') != dict(generator=0, selector=0, surrogate=0, discriminator=0, victim=200)
            or plan.get('execution', {}).get('max_active_cases') != 2):
        raise ValueError('Registered support replication contract changed')
    for filename, digest in plan['bound_files'].items():
        if sha(filename) != digest:
            raise ValueError('Bound input or checkpoint changed: ' + filename)
    for relative, digest in plan['sources'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Registered runtime source changed: ' + relative)
    _validate_supports(plan['runtime_bindings'])
    return plan


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Missing/nonfinite metric: ' + name)
    return float(value)


def metric_values(report):
    models = report['models']
    victim, clean = models['victim'], models['clean_reference']
    values = dict(overall_incremental_asr=victim['targets']['all']['incremental_asr'],
        moving_incremental_asr=victim['targets']['moving']['incremental_asr'],
        clean_pair_auc=victim['clean_pair_auc'],
        clean_positive_recall=victim['clean_pair_positive_recall'],
        adjacent_incremental_fp=victim['collateral']['adjacent']['incremental_fp'],
        nonincident_incremental_fp=victim['collateral']['nonincident']['incremental_fp'],
        adjacent_positive_suppression=victim['collateral']['adjacent']['positive_suppression'],
        clean_reference_incremental_asr=clean['targets']['all']['incremental_asr'])
    return {key: _finite(value, key) for key, value in values.items()}


def acceptability(report):
    """Prospective performance checks, separate from runtime integrity."""
    values = metric_values(report)
    victim, clean = report['models']['victim'], report['models']['clean_reference']
    normal_auc = _finite(clean['clean_pair_auc'], 'reference normal AUC')
    recall = _finite(clean['clean_pair_positive_recall'], 'reference normal recall')
    prediction = victim['clean_prediction_distribution']
    checks = dict(overall_incremental_asr_at_least_85pct=values['overall_incremental_asr'] >= .85,
        moving_incremental_asr_at_least_85pct=values['moving_incremental_asr'] >= .85,
        adjacent_incremental_fp_below_5pct=values['adjacent_incremental_fp'] < .05,
        same_pool_reference_auc_drop_at_most_0p005=normal_auc-values['clean_pair_auc'] <= .005+1e-12,
        same_pool_reference_recall_drop_at_most_0p05=recall-values['clean_positive_recall'] <= .05+1e-12,
        victim_predicts_both_clean_classes=prediction.get('class_0', 0) > 0 and prediction.get('class_1', 0) > 0)
    return dict(passed=all(checks.values()), checks=checks,
                failed_checks=[name for name, passed in checks.items() if not passed],
                gates=dict(GATES), clean_reference_incremental_response_is_diagnostic_only=True)


def _cache(plan):
    import torch
    from .full_ablation_data import _selection_contract
    # Verified own-cache pickle; no archived experiment code is imported.
    inputs = torch.load(plan['prepared_inputs'], map_location='cpu', weights_only=False)
    if (set(inputs['combined']) != {'train', 'dev'}
            or set(inputs['parts']) != {'train', 'dev'}
            or len(inputs['parts']['train']) != 512 or len(inputs['parts']['dev']) != 128
            or len(inputs['indices']['idx_attach']) != 26
            or len(inputs['indices']['idx_train']) != 1876
            or len(inputs['outer']) != 538):
        raise ValueError('Unexpected cached original-TRAIN TRAIN/dev scope')
    _selection_contract(inputs)
    population = inputs['populations']['dev_public_single']
    if (len(population) != 112 or sum(row['motion_regime']=='moving' for row in population) != 33
            or int(inputs['combined']['dev']['eligible'].sum()) != 2223):
        raise ValueError('Fixed DEV population changed')
    return inputs


def case(plan_path, seed, arm, *, preflight=False, expected_sha256=None):
    import torch
    from .runtime import load_bundle, restore_clean_reference
    from .support_replication_runtime import restore_uniform, frozen_support
    from .full_ablation_data import select_indices
    from .full_ablation_evaluation import fit_victim, evaluate_case
    from .evaluate import state_hash
    from .development_diagnostic import _input_hashes
    plan_path = Path(plan_path).resolve()
    plan = verify(plan_path, expected_sha256)
    expected = sha(plan_path)
    if seed not in SEEDS or arm not in ARMS or (preflight and seed != SEEDS[0]):
        raise ValueError('Unregistered fixed-support case')
    output = plan_path.parent / (f'engineering_preflight/{arm}' if preflight else f'seed{seed}/{arm}')
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    state(output, 'LOADING_FROZEN_UNIFORM', seed=seed, arm=arm, preflight=preflight)
    try:
        device = configure_cuda(seed)
        inputs = _cache(plan)
        inputs_before = _input_hashes(inputs)
        bundle = load_bundle(plan['manifest'], verify_assets=False)
        selection = select_indices(inputs, 'full', seed)
        support = plan['runtime_bindings']['supports'][arm]
        anchor = restore_uniform(bundle, device=device)
        generator = frozen_support(anchor, support['hard_support']).to(device).eval().requires_grad_(False)
        del anchor
        initial_generator = state_hash(generator)
        if any(p.requires_grad or p.grad is not None for p in generator.parameters()):
            raise ValueError('Fixed support generator must be entirely frozen')
        write(output / 'selection.json', selection['record'])
        write(output / 'support.json', dict(arm=arm, source=support,
            write_capacity=24, output_mode='identity', new_selector_updates=0,
            repeated_historical_full30_parent=True))
        state(output, 'GPU_FRESH_VICTIM_TRAINING', seed=seed, arm=arm)
        victim, victim_record = fit_victim(generator, inputs, selection, bundle, seed=seed,
            epochs=2 if preflight else 200, device=str(device), output_dir=output/'victim',
            progress=lambda row: event(output, phase='victim', record=row))
        if (state_hash(generator) != initial_generator or _input_hashes(inputs) != inputs_before):
            raise ValueError('Victim fitting changed frozen generator or cached data')
        if preflight:
            write(output/'completion.json', dict(status='SUPPORT_CASE_PREFLIGHT_PASS',
                plan_sha256=expected, seed=seed, arm=arm, victim_epochs=2,
                asr_evaluation=False, short_weights_reused=False,
                new_generator_steps=0, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                elapsed_seconds=time.monotonic()-began, utc=now()))
        else:
            state(output, 'GPU_DEVELOPMENT_EVALUATION', seed=seed, arm=arm)
            reference = restore_clean_reference(bundle).to(device).eval().requires_grad_(False)
            report = evaluate_case(generator, victim, reference, inputs, device=str(device),
                                   output_dir=output/'development')
            if (report['models']['victim']['targets']['all']['targets'] != 112
                    or report['models']['victim']['targets']['moving']['targets'] != 33
                    or report['models']['victim']['normal_pairs'] != 2223
                    or state_hash(generator) != initial_generator
                    or _input_hashes(inputs) != inputs_before):
                raise ValueError('Evaluation changed the registered population or frozen state')
            verify(plan_path, expected)
            result = dict(status='CASE_COMPLETE', seed=seed, arm=arm, metrics=report,
                acceptability=acceptability(report), victim_record=victim_record,
                optimizer_steps=dict(generator=0, selector=0, surrogate=0, discriminator=0, victim=200),
                parent=plan['runtime_bindings']['uniform'], generator_state_sha256=initial_generator,
                inherited_attack_checkpoint=True, independent_generator_retraining=False,
                original_test_arrays_loaded=False, original_validation_arrays_loaded=False,
                audit_arrays_loaded=False, confirmation_arrays_loaded=False,
                device=str(device), elapsed_seconds=time.monotonic()-began,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write(output/'result.json', result)
            write(output/'completion.json', dict(status='CASE_COMPLETE', seed=seed, arm=arm,
                plan_sha256=expected, result_sha256=sha(output/'result.json'), utc=now()))
        state(output, 'PREFLIGHT_PASS' if preflight else 'CASE_COMPLETE', seed=seed, arm=arm)
    except BaseException as error:
        write(output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            utc=now(), seed=seed, arm=arm, preflight=preflight))
        state(output, 'FAILED', seed=seed, arm=arm, error=repr(error))
        raise


def _resources(plan):
    from author_native_runs.dpgba_queue import resource_snapshot, cuda_probe
    snapshot = resource_snapshot()
    policy = plan['resource_policy']
    probe = (cuda_probe() if snapshot.get('ok') and
             snapshot['host_available_bytes'] >= policy['min_host_available_bytes'] else None)
    ready = bool(probe and probe.get('ok') and probe['free_bytes'] >= policy['min_cuda_free_bytes'])
    return dict(ready=ready, snapshot=snapshot, cuda_probe=probe, utc=now())


def _wait_resources(plan_path, expected):
    output, began = Path(plan_path).resolve().parent, time.monotonic()
    while True:
        plan = verify(plan_path, expected)
        if (output/'CANCEL').exists():
            raise RuntimeError('Fixed-support experiment cancelled')
        resources = _resources(plan)
        if resources['ready']:
            return resources
        state(output, 'WAITING_FOR_SHARED_GPU_HEADROOM', resources=resources)
        if time.monotonic()-began > plan['resource_policy']['max_wait_seconds']:
            raise TimeoutError('GPU resource wait timeout; no CPU fallback')
        time.sleep(plan['resource_policy']['poll_seconds'])


def preflight(plan_path, expected_sha256=None):
    import torch
    path = Path(plan_path).resolve()
    plan = verify(path, expected_sha256)
    expected = sha(path)
    folder = path.parent/'engineering_preflight'
    folder.mkdir(exist_ok=False)
    write(folder/'execution_claim.json', dict(pid=os.getpid(), utc=now(), plan_sha256=expected))
    try:
        for arm in ARMS:
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            _wait_resources(path, expected)
            case(path, SEEDS[0], arm, preflight=True, expected_sha256=expected)
        receipts = [read(folder/arm/'completion.json') for arm in ARMS]
        if any(r['status'] != 'SUPPORT_CASE_PREFLIGHT_PASS' or r['plan_sha256'] != expected
               or r['asr_evaluation'] is not False for r in receipts):
            raise ValueError('All four source-bound short tests must complete')
        write(folder/'completion.json', dict(status='SUPPORT_PREFLIGHT_PASS', plan_sha256=expected,
            arms=list(ARMS), victim_epochs=2, asr_evaluation=False, short_weights_reused=False,
            peaks=[r['gpu_peak_allocated_bytes'] for r in receipts], utc=now()))
        state(path.parent, 'PREFLIGHT_PASSED_READY_TO_LAUNCH', cases=len(plan['cases']))
    except BaseException as error:
        write(path.parent/'failure.json', dict(stage='preflight', error=repr(error),
            traceback=traceback.format_exc(), utc=now()))
        state(path.parent, 'FAILED_PREFLIGHT', error=repr(error))
        raise


def _check_preflight(path, expected):
    receipt = read(path.parent/'engineering_preflight/completion.json')
    if (receipt.get('status') != 'SUPPORT_PREFLIGHT_PASS' or receipt.get('plan_sha256') != expected
            or receipt.get('arms') != list(ARMS) or receipt.get('victim_epochs') != 2
            or receipt.get('asr_evaluation') is not False or receipt.get('short_weights_reused') is not False):
        raise ValueError('Launch requires four real-input short tests for this exact plan')
    for arm in ARMS:
        case_receipt = read(path.parent/'engineering_preflight'/arm/'completion.json')
        if (case_receipt.get('status') != 'SUPPORT_CASE_PREFLIGHT_PASS'
                or case_receipt.get('plan_sha256') != expected or case_receipt.get('arm') != arm
                or case_receipt.get('seed') != SEEDS[0]):
            raise ValueError('Missing/mismatched preflight arm: '+arm)


def _csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), values=values)


def _verify_prediction_artifacts(report):
    """Check actual prediction bytes and return the ordered public identities."""
    targets = None
    for key in ('raw_targets', 'raw_clean_scores', 'raw_triggered_pair_scores'):
        binding = report[key]
        path = Path(binding['path']).resolve()
        if sha(path) != binding['sha256']:
            raise ValueError('Recorded prediction artifact changed: ' + key)
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != binding['rows']:
            raise ValueError('Recorded prediction row count changed: ' + key)
        if key == 'raw_targets':
            fields = ('scenario_id', 'split_group_id', 'pair_index', 'src_track_id',
                      'dst_track_id', 'true_label', 'motion_regime')
            targets = tuple(tuple(row[field] for field in fields) for row in rows)
            if (len(targets) != 112 or len(set(targets)) != 112
                    or sum(row[-1] == 'moving' for row in targets) != 33
                    or any(row[-2] != '0' for row in targets)):
                raise ValueError('Prediction artifacts lost the public DEV112/Moving33 population')
        elif key == 'raw_clean_scores' and len(rows) != 2223:
            raise ValueError('Normal DEV prediction population changed')
    return targets


def aggregate(plan_path, expected_sha256=None):
    path = Path(plan_path).resolve()
    plan = verify(path, expected_sha256)
    expected, root = sha(path), path.parent
    records, rows, completed = {}, [], []
    target_identities, reference_state = None, None
    for item in case_matrix():
        seed, arm = item['seed'], item['arm']
        folder = root/f'seed{seed}'/arm
        receipt, result = read(folder/'completion.json'), read(folder/'result.json')
        if (receipt.get('status') != 'CASE_COMPLETE' or receipt.get('plan_sha256') != expected
                or receipt.get('seed') != seed or receipt.get('arm') != arm
                or receipt.get('result_sha256') != sha(folder/'result.json')
                or result.get('status') != 'CASE_COMPLETE' or result.get('seed') != seed
                or result.get('arm') != arm or result.get('optimizer_steps') != plan['optimizer_steps']):
            raise ValueError('Missing or mismatched formal case: '+str(item))
        values = metric_values(result['metrics'])
        checks = acceptability(result['metrics'])
        if checks != result['acceptability']:
            raise ValueError('Saved acceptability does not match metrics')
        actual_targets = _verify_prediction_artifacts(result['metrics'])
        actual_reference = result['metrics']['model_state_sha256']['clean_reference']
        if target_identities is None:
            target_identities, reference_state = actual_targets, actual_reference
        elif actual_targets != target_identities or actual_reference != reference_state:
            raise ValueError('Cases did not use the same public targets and frozen reference')
        records[(seed, arm)] = result
        rows.append(dict(seed=seed, arm=arm, **values, passed=checks['passed']))
        completed.append(dict(**item, result=str(folder/'result.json'), result_sha256=receipt['result_sha256']))
    initialization = {}
    for seed in SEEDS:
        states = {records[(seed, arm)]['victim_record']['initial_state_sha256'] for arm in ARMS}
        rng = {records[(seed, arm)]['victim_record']['fit_rng_digest'] for arm in ARMS}
        if len(states) != 1 or len(rng) != 1:
            raise ValueError('Within-seed victims did not share initialization and fit RNG')
        initialization[str(seed)] = next(iter(states))
    if len(set(initialization.values())) != len(SEEDS):
        raise ValueError('Different victim seeds unexpectedly reused initialization')
    per_arm = {arm: {key: _stats([metric_values(records[(seed, arm)]['metrics'])[key]
                                  for seed in SEEDS]) for key in METRICS} for arm in ARMS}
    differences, per_seed = [], {}
    for seed in SEEDS:
        semantic = metric_values(records[(seed, 'semantic')]['metrics'])
        random = {key: statistics.mean(metric_values(records[(seed, arm)]['metrics'])[key]
                  for arm in ARMS[1:]) for key in METRICS}
        difference = {key: semantic[key]-random[key] for key in METRICS}
        differences.append(dict(seed=seed, **difference))
        per_seed[str(seed)] = dict(semantic=semantic, random_family_mean=random,
                                   semantic_minus_random_family_mean=difference)
    summary = dict(status='SUPPORT_REPLICATION_COMPLETE_REVIEW_REQUIRED', plan_sha256=expected,
        seeds=list(SEEDS), arms=list(ARMS), cases=12, per_arm=per_arm,
        per_seed=per_seed, paired_differences={key:_stats([row[key] for row in differences]) for key in METRICS},
        per_case_acceptability={f'{seed}/{arm}': records[(seed,arm)]['acceptability']
                                for seed in SEEDS for arm in ARMS},
        all_cases_pass=all(row['passed'] for row in rows), paired_initializations=initialization,
        independent_full_pipeline_seeds=False, independent_selector_retraining=False,
        repeated_trained_generator=True, random_support_draws_are_not_independent_training_seeds=True,
        evaluation_scope='reused original-TRAIN DEV112/Moving33; no blind confirmation',
        automatic_promotion=False, automatic_followup=False,
        optimizer_steps_per_case=plan['optimizer_steps'], utc=now())
    _csv(root/'case_metrics.csv', rows)
    _csv(root/'paired_differences.csv', differences)
    write(root/'aggregate.json', summary)
    write(root/'completion.json', dict(status=summary['status'], plan_sha256=expected,
        cases=completed, aggregate_sha256=sha(root/'aggregate.json'), utc=now()))
    state(root, summary['status'], completed_cases=12, all_cases_pass=summary['all_cases_pass'])
    return summary


def _env():
    return {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'OPENBLAS_NUM_THREADS':'1',
            'OMP_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':'0'}


def launch(plan_path):
    path = Path(plan_path).resolve()
    verify(path)
    expected, output = sha(path), path.parent
    _check_preflight(path, expected)
    if (output/'failure.json').exists() or (output/'completion.json').exists():
        raise RuntimeError('Failed/completed experiments cannot be relaunched')
    write(output/'launch_claim.json', dict(plan_sha256=expected, utc=now()))
    command = [sys.executable, '-B', '-u', '-m', 'spear_current.support_replication',
               '--queue', '--plan', str(path), '--expected-sha256', expected]
    with (output/'queue.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=_env(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(status='SUPPORT_GPU_QUEUE_LAUNCHED', pid=child.pid,
                   command=command, max_active_cases=2, plan_sha256=expected, utc=now())
    write(output/'launch.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def queue(plan_path, expected_sha256):
    path = Path(plan_path).resolve()
    plan = verify(path, expected_sha256)
    _check_preflight(path, expected_sha256)
    output = path.parent
    write(output/'execution_claim.json', dict(pid=os.getpid(), utc=now(), plan_sha256=expected_sha256))
    pending, active, completed = list(case_matrix()), [], []
    waiting_since = time.monotonic()

    def stop(signum, frame):
        raise RuntimeError('Support queue interrupted: '+str(signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            verify(path, expected_sha256)
            if (output/'CANCEL').exists():
                raise RuntimeError('Fixed-support queue cancelled')
            for job in list(active):
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                active.remove(job)
                if code:
                    raise RuntimeError(f"Case failed: {job['seed']}/{job['arm']} exit={code}")
                folder = output/f"seed{job['seed']}"/job['arm']
                receipt = read(folder/'completion.json')
                if (receipt.get('status') != 'CASE_COMPLETE'
                        or receipt.get('plan_sha256') != expected_sha256
                        or receipt.get('result_sha256') != sha(folder/'result.json')):
                    raise ValueError('Case exited without matching completion evidence')
                completed.append(dict(seed=job['seed'],arm=job['arm'],result=str(folder/'result.json'),
                                      result_sha256=receipt['result_sha256']))
                write(output/'completed_cases.json', completed, exclusive=False)
            resources = None
            if pending and len(active) < plan['execution']['max_active_cases']:
                resources = _resources(plan)
                if resources['ready']:
                    item = pending.pop(0)
                    seed, arm = item['seed'], item['arm']
                    command = [sys.executable, '-B', '-u', '-m', 'spear_current.support_replication',
                        '--case', '--plan', str(path), '--expected-sha256', expected_sha256,
                        '--seed', str(seed), '--arm', arm]
                    write(output/f'seed{seed}_{arm}_launch.json', dict(**item,command=command,
                          resources=resources,utc=now()))
                    log = (output/f'seed{seed}_{arm}.log').open('x')
                    try:
                        child = subprocess.Popen(command,cwd=ROOT,env=_env(),stdin=subprocess.DEVNULL,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    active.append(dict(**item,process=child,log=log))
                    waiting_since = time.monotonic()
                elif time.monotonic()-waiting_since > plan['resource_policy']['max_wait_seconds']:
                    raise TimeoutError('GPU headroom wait exceeded registered limit')
            state(output, 'GPU_SUPPORT_REPLICATION_RUNNING' if active else 'WAITING_FOR_SHARED_GPU_HEADROOM',
                active=[dict(seed=j['seed'],arm=j['arm'],pid=j['process'].pid) for j in active],
                completed_cases=len(completed),pending_cases=len(pending),resources=resources)
            if pending or active:
                time.sleep(plan['resource_policy']['poll_seconds'])
        aggregate(path, expected_sha256)
    except BaseException as error:
        # Only children launched by this queue are terminated; unrelated GPU
        # jobs and all saved evidence are left alone. No child is retried.
        stopped = []
        for job in active:
            if job['process'].poll() is None:
                os.killpg(job['process'].pid, signal.SIGTERM)
                stopped.append(dict(seed=job['seed'],arm=job['arm'],pid=job['process'].pid))
            job['log'].close()
        write(output/'failure.json',dict(stage='queue',error=repr(error),traceback=traceback.format_exc(),
            completed_cases=len(completed),own_children_interrupted=stopped,
            pending_cases=pending,utc=now()))
        state(output,'FAILED',error=repr(error),completed_cases=len(completed))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','preflight','launch','case','queue','aggregate'):
        mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--parent-plan',type=Path,default=DEFAULT_PARENT_PLAN)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--expected-sha256','--plan-sha256',dest='expected_sha256')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.prepare:
        if args.output_dir is None:
            parser.error('--prepare requires --output-dir')
        prepare(args.output_dir,args.manifest,parent_plan=args.parent_plan)
        return
    if args.plan is None:
        parser.error('This mode requires --plan')
    if args.preflight:
        preflight(args.plan,args.expected_sha256)
    elif args.launch:
        launch(args.plan)
    elif args.case:
        if args.seed is None or args.arm is None:
            parser.error('--case requires --seed and --arm')
        case(args.plan,args.seed,args.arm,expected_sha256=args.expected_sha256)
    elif args.queue:
        if args.expected_sha256 is None:
            parser.error('--queue requires --expected-sha256')
        queue(args.plan,args.expected_sha256)
    else:
        aggregate(args.plan,args.expected_sha256)


if __name__=='__main__':
    main()
