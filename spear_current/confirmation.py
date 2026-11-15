"""Registered, inference-only confirmation on new original-TRAIN recording groups.

This is new-group confirmation for the retained SPEAR candidate, not a claim
that these data were never used by the whole JCAS project. No fitting, model
selection, repeated pool search, or original validation/test access is allowed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev

import torch

from .confirmation_protocol import prepare_protocol, validate_protocol, read_confirmation_part
from .data import load_normalizer
from .evaluate import SceneInjector, state_hash
from .evaluation import evaluate
from .runtime import load_bundle, restore_generator, restore_victim, restore_clean_reference, sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / 'results'
HISTORY = RESULT_ROOT / 'confirmation_preflight_20261027/history_reference_audit.json'
SEEDS = (20260621, 20260622, 20260623)
GROUP_COUNT = 1024
VERSION = 'spear_current_new_group_confirmation_v1'
EVALUATION_CONTRACT = dict(
    method='retained unconditional; no generator, proxy or victim updates',
    source_split='original TRAIN only', group_count=GROUP_COUNT, scenes_per_group=1,
    target_seed=20260621, seeds=list(SEEDS),
    target_rule='existing SHA identity rule; one eligible true-negative per scene; no prediction filter',
    empty_scene_policy='retain and count; no replacement',
    threshold='existing binary argmax; no threshold fitting',
    overall_incremental_asr_min=.80, moving_incremental_asr_min=.80,
    adjacent_incremental_fp_strict_max=.05,
    clean_auc_drop_max=.005, clean_positive_recall_drop_max=.05,
    quality_reference='same retained clean_reference on exactly the new confirmation scenes',
    quality_reference_difference='new confirmation comparison; NOT historical fixed SAGE24 audit anchor',
    quality_noncollapse='victim must predict both classes on eligible clean pairs',
    performance_failure_policy='report every registered seed; do not retune, resample or add jobs',
    integrity_failure_policy='stop immediately and retain failure evidence; no automatic retry',
    complete_success='all three seeds satisfy every registered check; means cannot hide a failed seed',
    seed_aggregation='equal-weight mean and sample SD; do not pool repeated targets across seeds',
    model_independence='shared fixed pretraining start and clean reference; not full-pipeline independent seeds',
    data_claim='absent from checked SPEAR use records; not never-seen-by-entire-JCAS-project data',
    model_claim='clean reference was previously used in development; not a newly blind reference',
    original_validation_arrays_allowed=False, original_test_arrays_allowed=False,
    automatic_promotion=False, automatic_training=False, automatic_followup=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def update_state(folder, value):
    destination = folder / 'state.json'
    temporary = folder / 'state.json.tmp'
    with temporary.open('w') as stream:
        json.dump({'at': now(), **value}, stream, indent=2, allow_nan=False)
    temporary.replace(destination)


def check_output(path, *, new):
    path = Path(path).resolve()
    if path == RESULT_ROOT or not path.is_relative_to(RESULT_ROOT):
        raise ValueError('Confirmation output must be a child of results/')
    if new and path.exists():
        raise FileExistsError('No overwrite or automatic retry: ' + str(path))
    return path


def source_catalog():
    return {str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / 'spear_current').glob('*.py'))}


def prepare(bundle, output, history=HISTORY):
    folder = check_output(output, new=True)
    protocol = prepare_protocol(bundle, Path(history), group_count=GROUP_COUNT)
    if protocol['seed_list'] != list(SEEDS) or protocol['target_seed'] != 20260621:
        raise ValueError('Unexpected seed contract')
    if len(protocol['rows']) != GROUP_COUNT:
        raise ValueError('Require the fixed group population; no smaller fallback')
    plan = dict(version=VERSION, created_at=now(), protocol=protocol,
                evaluation_contract=EVALUATION_CONTRACT, source_catalog=source_catalog())
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'plan.json', plan)
    write_json(folder / 'registration.json', dict(
        status='REGISTERED_BEFORE_GRAPH_ACCESS', plan_sha256=sha256(folder / 'plan.json'),
        at=now(), group_count=GROUP_COUNT, seeds=list(SEEDS),
        graph_arrays_loaded=False, model_inference_started=False, training_started=False))
    print(json.dumps(dict(status='CONFIRMATION_PREPARED_NO_JOBS_STARTED',
        plan=str(folder / 'plan.json'), plan_sha256=sha256(folder / 'plan.json')), indent=2), flush=True)
    return plan


def load_plan(path):
    path = Path(path).resolve()
    folder = check_output(path.parent, new=False)
    if path.name != 'plan.json':
        raise ValueError('Require the registered plan.json')
    plan = json.loads(path.read_text())
    registration = json.loads((folder / 'registration.json').read_text())
    if (registration['status'] != 'REGISTERED_BEFORE_GRAPH_ACCESS'
            or registration['plan_sha256'] != sha256(path)
            or plan['version'] != VERSION or plan['evaluation_contract'] != EVALUATION_CONTRACT
            or plan['source_catalog'] != source_catalog()):
        raise ValueError('Registered plan, evaluation rules, or implementation changed')
    protocol = plan['protocol']
    if (protocol['group_count'] != GROUP_COUNT or len(protocol['rows']) != GROUP_COUNT
            or protocol['seed_list'] != list(SEEDS) or protocol['target_seed'] != 20260621):
        raise ValueError('Registered population or seeds changed')
    bundle = load_bundle(protocol['references']['bundle']['path'])
    validate_protocol(bundle, protocol)
    return folder, plan, bundle


def common_prediction_hashes(folder):
    """Bind the public target/normal population and identical clean-reference predictions."""
    result = {}
    for name in ('targets.csv', 'clean_pair_scores.csv'):
        digest = hashlib.sha256()
        with (folder / name).open() as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                common = {key: value for key, value in row.items() if not key.startswith('victim_')}
                # Triggered clean-reference responses depend on each seed's G.
                if name == 'targets.csv':
                    common = {key: value for key, value in common.items()
                              if key not in ('clean_reference_trigger_p', 'clean_reference_trigger_pred')}
                digest.update(json.dumps(common, sort_keys=True, separators=(',', ':')).encode() + b'\n')
        result[name] = digest.hexdigest()
    return result


def checks_for(metrics):
    victim, clean = metrics['victim'], metrics['clean_reference']
    def at_least(value, minimum):
        return value is not None and value >= minimum
    def drop_ok(reference, value, maximum):
        return reference is not None and value is not None and reference - value <= maximum
    values = dict(
        overall_incremental_asr_at_least_80pct=at_least(victim['targets']['all']['incremental_asr'], .8),
        moving_incremental_asr_at_least_80pct=at_least(victim['targets']['moving']['incremental_asr'], .8),
        adjacent_incremental_fp_below_5pct=(victim['collateral']['adjacent_incremental_fp'] is not None
            and victim['collateral']['adjacent_incremental_fp'] < .05),
        same_pool_clean_reference_auc_drop_at_most_0p005=drop_ok(
            clean['clean_pair_auc'], victim['clean_pair_auc'], .005),
        same_pool_clean_reference_recall_drop_at_most_0p05=drop_ok(
            clean['clean_pair_positive_recall'], victim['clean_pair_positive_recall'], .05),
        victim_predicts_both_clean_classes=all(
            victim['clean_prediction_distribution'].get(key, 0) > 0 for key in ('class_0', 'class_1')))
    return dict(passed=all(values.values()), checks=values,
                failed_checks=[key for key, value in values.items() if not value])


def statistics(values):
    if any(value is None for value in values):
        return dict(values=values, mean=None, sample_sd=None, reason='one or more seeds not estimable')
    return dict(values=values, mean=mean(values), sample_sd=stdev(values), reason=None)


def aggregate(reports):
    if set(reports) != {str(seed) for seed in SEEDS}:
        raise ValueError('Require all registered seeds; no partial aggregate')
    ordered = [reports[str(seed)] for seed in SEEDS]
    if any(r['audit_counts'] != ordered[0]['audit_counts'] for r in ordered):
        raise ValueError('Seed populations differ')
    common = ordered[0]['common_prediction_hashes']
    if any(r['common_prediction_hashes'] != common for r in ordered):
        raise ValueError('Target identities or clean-reference normal predictions differ')
    values = {}
    for role in ('victim', 'clean_reference'):
        values[role] = {}
        for regime in ('all', 'slow', 'moving'):
            target_counts = [r['metrics'][role]['targets'][regime]['targets'] for r in ordered]
            if len(set(target_counts)) != 1:
                raise ValueError('Motion subgroup counts differ by seed')
            values[role][regime] = dict(targets_per_seed=target_counts[0], **{
                key: statistics([r['metrics'][role]['targets'][regime][key] for r in ordered])
                for key in ('incremental_asr', 'absolute_asr', 'conditional_flip', 'clean_activation_rate')})
        values[role]['normal'] = {key: statistics([r['metrics'][role][key] for r in ordered])
            for key in ('clean_pair_auc', 'clean_pair_positive_recall')}
        values[role]['collateral'] = {key: statistics([r['metrics'][role]['collateral'][key] for r in ordered])
            for key in ('adjacent_incremental_fp', 'nonincident_incremental_fp', 'adjacent_positive_suppression')}
    gates = {str(seed): reports[str(seed)]['gate'] for seed in SEEDS}
    return dict(status='GROUP_CONFIRMATION_COMPLETE_REVIEW_REQUIRED', seeds=list(SEEDS),
        metrics=values, per_seed_gates=gates, all_seeds_pass=all(g['passed'] for g in gates.values()),
        audit_counts_per_seed=ordered[0]['audit_counts'], common_prediction_hashes=common,
        interpretation=EVALUATION_CONTRACT, optimizer_steps=0,
        original_val_arrays_loaded=False, original_test_arrays_loaded=False,
        full_pipeline_independent_seeds=False, whole_project_never_seen_data=False)


def run(path):
    folder, plan, bundle = load_plan(path)
    write_json(folder / 'execution_claim.json', dict(at=now(), pid=os.getpid(),
        plan_sha256=sha256(folder / 'plan.json'), optimizer_steps=0))
    reports = {}
    try:
        torch.set_num_threads(2)
        protocol = plan['protocol']
        update_state(folder, dict(status='INITIALIZING', completed_seeds=[]))
        clean_normalizer = load_normalizer(bundle)
        for seed in SEEDS:
            validate_protocol(bundle, protocol)
            seed_folder = folder / f'seed{seed}'
            seed_folder.mkdir(exist_ok=False)
            models = dict(clean_reference=restore_clean_reference(bundle), victim=restore_victim(bundle, seed))
            generator = restore_generator(bundle, seed)
            modules = dict(generator=generator, **models)
            before = {key: state_hash(model) for key, model in modules.items()}
            attack = SceneInjector(generator, bundle.config['shuffle_seed'])
            ledger = []
            def reader(row):
                part = read_confirmation_part(bundle, protocol, row, clean_normalizer, ledger)
                attack.bind(part)
                return part
            def progress(**value):
                record = dict(status='RUNNING', seed=seed, **value)
                update_state(folder, record)
                print(json.dumps(record), flush=True)
            # Separate role; the original train/dev/audit population is retained.
            evaluation_plan = dict(seed=protocol['target_seed'],
                roles=dict(bundle.config['roles'], confirmation=protocol['rows']))
            try:
                with torch.no_grad():
                    result = evaluate(evaluation_plan, seed_folder, models, attack, reader, progress,
                                      role='confirmation')
            finally:
                write_json(seed_folder / 'graph_access_ledger.json', ledger)
            after = {key: state_hash(model) for key, model in modules.items()}
            if before != after or any(parameter.grad is not None for model in modules.values()
                                      for parameter in model.parameters()):
                raise ValueError('Inference changed weights or accumulated gradients')
            if (len(ledger) != GROUP_COUNT or any(entry['status'] != 'READ_COMPLETE'
                    or entry['split'] != 'train' or entry['role'] != 'confirmation' for entry in ledger)):
                raise ValueError('Graph access ledger differs from the registered role')
            if result['audit_counts']['processed_scenes'] != GROUP_COUNT:
                raise ValueError('Incomplete scene accounting')
            result.update(seed=seed, role='confirmation', method='unconditional',
                model_state_hashes=before, weights_unchanged=True, optimizer_steps=0,
                common_prediction_hashes=common_prediction_hashes(seed_folder),
                gate=checks_for(result['metrics']))
            if reports and result['common_prediction_hashes'] != next(iter(reports.values()))['common_prediction_hashes']:
                raise ValueError('Frozen target identities or clean predictions changed between seeds')
            write_json(seed_folder / 'summary.json', result)
            reports[str(seed)] = result
            del modules, models, generator, attack
        result = aggregate(reports)
        write_json(folder / 'aggregate.json', result)
        completion = dict(status=result['status'], at=now(), completed_seeds=list(SEEDS),
            plan_sha256=sha256(folder / 'plan.json'), aggregate_sha256=sha256(folder / 'aggregate.json'),
            all_seeds_pass=result['all_seeds_pass'], training_started=False,
            original_val_arrays_loaded=False, original_test_arrays_loaded=False,
            automatic_followup=False)
        write_json(folder / 'completion.json', completion)
        update_state(folder, completion)
        print(json.dumps(completion, indent=2), flush=True)
        return result
    except Exception as error:
        failure = dict(status='CONFIRMATION_FAILED_INTEGRITY_OR_RUNTIME', at=now(),
                       error=repr(error), completed_seeds=list(reports), automatic_retry=False)
        write_json(folder / 'failure.json', failure)
        update_state(folder, failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.output_dir is None or args.plan is not None:
            parser.error('--prepare requires only --output-dir')
        prepare(load_bundle(), args.output_dir)
    else:
        if args.plan is None or args.output_dir is not None:
            parser.error('--run requires only --plan')
        run(args.plan)


if __name__ == '__main__':
    main()
