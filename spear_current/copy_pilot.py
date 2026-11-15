"""Paired equal-budget within-pair consistency continuation on original TRAIN.

Offline AV2 / self-owned GNN research. Both arms inherit the SAME retained
final seed21 generator/surrogate, then receive 200 additional G/S/D updates
and a fresh 200-epoch GraphSAGE victim. This is not a from-scratch replication.
Only the six-copy output mapping differs. No audit/confirmation/VAL/TEST reads.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

import torch

from .copy_consistency import COPY_GROUPS, CopyConsistentGenerator, copy_consistency_statistics
from .copy_diagnostic import evaluate_arm
from .data import load_normalizer
from .development_diagnostic import registered_rows, read_inputs, _input_hashes
from .evaluate import state_hash
from .outer_evaluation import _dev_view, _validate, _rng_fingerprint
from .runtime import (load_bundle, sha256, restore_generator, restore_surrogate,
                      restore_proxy, restore_clean_reference)
from .support_schema import SELECTED_COORDINATES
from .train import (TRAINING_CONFIG, DISTRIBUTION_CONFIG, AUC_POLICY, training_seeds,
                    validate_training_contract, no_update_preflight, fit_unconditional,
                    fit_fresh_victim, _dev_quality, _write_json, _save_checkpoint)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260621
REPLICATION_SEEDS = (20260622, 20260623)
ARMS = ('identity', 'pair_mean')
VERSION = 'within_pair_copy_equal_budget_v1'
COPY_BUFFERS = {'copy_mode_code', 'copy_frozen_code', 'copy_positions', 'copy_normalizer_digest'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _directory(path):
    path = Path(path).resolve()
    if path == ROOT / 'results' or not path.is_relative_to(ROOT / 'results'):
        raise ValueError('Output must be a new child of results/')
    return path


def protocol(bundle, seed=SEED):
    """Metadata/file-byte checks only; no tensor deserialization or graph read."""
    from .copy_training_reporting import GATES
    if type(seed) is not int:
        raise ValueError('Require an integer training seed')
    seeds = training_seeds(seed)
    validate_training_contract(bundle)
    sources = registered_rows(bundle)
    coordinates, support = bundle.manifest['selected_coordinates'], bundle.manifest['hard_support']
    if (tuple(coordinates) != SELECTED_COORDINATES or len(support) != 30 or sum(support) != 24
            or any(not support[coordinates.index(c)] for group in COPY_GROUPS for c in group)):
        raise ValueError('Frozen original30/allowed24/six-copy support changed')
    keys = {name: bundle.manifest['seeds'][str(SEED)][name] for name in ('generator', 'surrogate')}
    required = set(keys.values()) | {'clean_reference', 'clean_proxy1', 'clean_proxy2',
        'normalizer', 'indices', 'support', 'input_contract', 'context_binding', 'score_permutation'}
    assets = {key: dict(path=str(bundle.asset_path(key)), sha256=bundle.manifest['assets'][key]['sha256'])
              for key in sorted(required)}
    return dict(version=VERSION, seed=seed, parent_seed=SEED, arms=list(ARMS), start_asset_keys=keys, assets=assets,
        hypothesis='Equal-budget training can adapt to the six exact within-pair copy equalities; '
                   'acceptable performance is tested, not presumed.',
        changed_variable='identity versus arithmetic-mean mapping after existing domain projection',
        starting_point='current retained final unconditional seed21 G and surrogate, not start_generator',
        optimizer_initialization='fresh Adam optimizers and same fresh unconditional discriminator in each arm',
        training_config=TRAINING_CONFIG, distribution_config=DISTRIBUTION_CONFIG,
        effective_distribution_config=dict(DISTRIBUTION_CONFIG, seed=seeds['discriminator']),
        seeds=seeds, generator_checkpoint='final_epoch_199',
        victim_epochs=200, victim_checkpoint=AUC_POLICY,
        train_scenes=512, dev_scenes=128, poison_attachments=26, outer_rows=538,
        normal_train_rows=1876, victim_fit_rows=1902,
        expected_dev_targets=112, expected_dev_moving=33, expected_dev_slow=79,
        target_rule='existing identity hash; eligible true-negative; no prediction filtering',
        selected_coordinates=coordinates, hard_support=support, copy_groups=[list(g) for g in COPY_GROUPS],
        gates=GATES, gates_scope='prospective single-seed DEV pilot; does not revise old confirmation failures',
        normal_metrics_population='all eligible DEV pairs, not the all-negative target pool',
        fixed=['original train/dev identities', '26 attachment identities', '538 outer identities',
               '24-coordinate support', 'normalizer', 'topology', 'dual clean proxies and KL',
               'unconditional distribution loss', 'starting G/S weights', 'initial victim and RNG',
               'training budgets', 'clean-DEV-AUC victim selection', 'common DEV targets and threshold'],
        scope='reused original-TRAIN development; shared pretrained start; no blind confirmation claim',
        warm_start_limitation='shared parent was previously trained with identity outputs; this is adaptation, not from-scratch comparison',
        automatic_followup=False, automatic_promotion=False, on_integrity_failure='stop and retain all evidence',
        on_performance_failure='report both completed arms; no automatic rerun or tuning',
        audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        original_validation_arrays_allowed=False, original_test_arrays_allowed=False,
        sources=sources, manifest=dict(path=str(bundle.manifest_path), sha256=sha256(bundle.manifest_path)),
        code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def prepare(bundle, output, seed=SEED):
    spec = protocol(bundle, seed)
    output = _directory(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'plan.json', dict(created_utc=_now(), protocol=spec))
    snapshot = output / 'source_snapshot.tar'
    with tarfile.open(snapshot, 'x') as archive:
        for name in spec['code_sha256']:
            archive.add(Path(__file__).parent / name, arcname='spear_current/' + name)
    if protocol(bundle, seed) != spec:
        raise ValueError('Sources changed during preparation')
    _write_json(output / 'registration.json', dict(created_utc=_now(),
        plan_sha256=sha256(output / 'plan.json'), source_snapshot_sha256=sha256(snapshot),
        graph_arrays_loaded=False, model_checkpoints_loaded=False, training_started=False))
    return output / 'plan.json'


def validate_plan(bundle, path):
    path = Path(path).resolve()
    if path.name != 'plan.json' or _directory(path.parent) != path.parent:
        raise ValueError('Require the prepared plan.json under results/')
    registration = json.loads((path.parent / 'registration.json').read_text())
    if (registration['plan_sha256'] != sha256(path)
            or registration['source_snapshot_sha256'] != sha256(path.parent / 'source_snapshot.tar')):
        raise ValueError('Prepared plan or source snapshot changed')
    value = json.loads(path.read_text())
    if value.get('protocol') != protocol(bundle, value.get('protocol', {}).get('seed')):
        raise ValueError('Prepared source, assets, budgets or rules changed')
    return value


def freeze(model):
    model.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False)


def make_generator(anchor, normalizer, mode):
    before, rng = state_hash(anchor), _rng_fingerprint()
    generator = CopyConsistentGenerator(anchor, normalizer, mode=mode, frozen=False)
    generator.eval()
    if (before != state_hash(anchor) or rng != _rng_fingerprint()
            or any(not torch.equal(generator.state_dict()[k], v) for k, v in anchor.state_dict().items())):
        raise ValueError('Generator construction changed shared weights or RNG')
    return generator


def evaluation_anchor(trained, anchor):
    """Export shared Domain tensors; same-mode frozen wrapper restores output.

    The saved training checkpoint includes four immutable copy buffers. They
    are not learned weights. The evaluator reconstructs those contract buffers
    with frozen=True; all inherited tensors remain exactly the trained values.
    """
    if type(trained) is not CopyConsistentGenerator or trained.frozen_postprocess:
        raise ValueError('Require the trainable-mode CopyConsistentGenerator')
    trained.assert_contract()
    state, keys = trained.state_dict(), set(anchor.state_dict())
    if set(state) != keys | COPY_BUFFERS:
        raise ValueError('Training checkpoint has missing or extra tensor keys')
    exported = deepcopy(anchor)
    exported.load_state_dict({key: state[key] for key in anchor.state_dict()}, strict=True)
    freeze(exported)
    if any(not torch.equal(exported.state_dict()[k], state[k]) for k in keys):
        raise ValueError('Evaluation export changed trained weights')
    return exported


def _training_inputs(diagnostic):
    return dict(train=diagnostic['combined']['train'], dev=diagnostic['combined']['dev'],
                indices=diagnostic['indices'], outer=diagnostic['outer'], context=diagnostic['context'])


def fit_one_arm(bundle, diagnostic, normalizer, anchor, start_surrogate, proxies, clean,
                mode, output, progress, *, seed=SEED):
    """One prescribed fit plus cached DEV evaluation; no checkpoint/arm search."""
    if mode not in ARMS:
        raise ValueError('Unknown consistency arm')
    training_seeds(seed)
    output = Path(output)
    output.mkdir(exist_ok=False)
    inputs = _training_inputs(diagnostic)
    before_inputs = _input_hashes(diagnostic)
    sources = [anchor, start_surrogate, *proxies, clean]
    before_sources = [state_hash(model) for model in sources]
    generator = make_generator(anchor, normalizer, mode)
    reference = CopyConsistentGenerator(anchor, normalizer, mode=mode, frozen=True)
    initial_state = dict(seed=seed, mode=mode, generator_parent_state_sha256=state_hash(anchor),
        surrogate_state_sha256=state_hash(start_surrogate),
        proxy_state_sha256=[state_hash(p) for p in proxies], clean_reference_state_sha256=state_hash(clean),
        training_generator_state_sha256=state_hash(generator), reference_generator_state_sha256=state_hash(reference),
        contract=generator.contract())
    _write_json(output / 'initial_state.json', initial_state)
    progress(arm=mode, stage='no_update_preflight')
    preflight = no_update_preflight(generator, reference, start_surrogate, proxies, inputs, seed)
    _write_json(output / 'preflight.json', preflight)
    del reference
    progress(arm=mode, stage='fit_generator_started')
    result, distribution, initial = fit_unconditional(generator, start_surrogate, proxies, inputs,
        seed=seed, progress=lambda row: progress(arm=mode, stage='fit_generator', **row))
    if (result['selected_epoch'] != 199 or result['selection_policy'] != 'final_epoch'
            or any(result[k] != 200 for k in ('residual_steps', 'surrogate_steps', 'discriminator_steps',
                                              'clean_constraint_calls', 'distribution_generator_calls'))):
        raise ValueError('Arm training budget or generator checkpoint rule changed')
    _save_checkpoint(output / 'generator.pt', generator.state_dict())
    _save_checkpoint(output / 'surrogate.pt', result['surrogate'].state_dict())
    _save_checkpoint(output / 'discriminator.pt', distribution.discriminator.state_dict())
    _save_checkpoint(output / 'distribution_initial.pt', initial)
    _save_checkpoint(output / 'distribution_final.pt', distribution.snapshot())
    excluded = {'generator', 'surrogate', 'poisoned_features', 'poisoned_labels', 'edge_index',
                'edge_weight', 'idx_train', 'idx_attach', 'fixed_outer'}
    _write_json(output / 'generator_record.json', {k: v for k, v in result.items() if k not in excluded})
    poison_values = result['poisoned_features'][inputs['indices']['idx_attach'][:, None], generator.selected_coordinates]
    copies = copy_consistency_statistics(poison_values)
    if mode == 'pair_mean' and copies['any_group_violations']:
        raise ValueError('Consistent training returned inconsistent final poison')
    _write_json(output / 'poison_consistency.json', copies)
    progress(arm=mode, stage='fit_fresh_victim_started')
    victim, victim_record = fit_fresh_victim(bundle, inputs, result, output, seed=seed,
        progress=lambda row: progress(arm=mode, **row))
    _save_checkpoint(output / 'victim_auc.pt', victim.state_dict())
    _write_json(output / 'victim_record.json', victim_record)
    freeze(generator)
    surrogate = freeze(result['surrogate'])
    freeze(victim)
    exported = evaluation_anchor(generator, anchor)
    # Check output parity only on already-loaded training attachments. This is
    # export correctness, not an additional DEV pass or model selection.
    eval_reference = CopyConsistentGenerator(exported, normalizer, mode=mode, frozen=True)
    train = inputs['train']
    with torch.no_grad():
        expected = generator(train['x'], inputs['context'], inputs['indices']['idx_attach'])
        actual = eval_reference(train['x'], inputs['context'], inputs['indices']['idx_attach'])
    if not torch.equal(expected, actual) or not torch.equal(actual, poison_values):
        raise ValueError('Evaluation restoration differs from trained/final-poison output')
    _write_json(output / 'evaluation_export.json', dict(bitwise_output_parity_on_attachments=True,
        training_contract=generator.contract(), evaluation_contract=eval_reference.contract(),
        training_state_sha256=state_hash(generator), exported_parent_state_sha256=state_hash(exported),
        note='Only immutable frozen flag differs in the reconstructed evaluation wrapper'))
    del eval_reference
    progress(arm=mode, stage='evaluate_dev_started')
    evaluate_arm(bundle, diagnostic, exported, normalizer,
        dict(surrogate=surrogate, victim=victim, clean_reference=clean), mode, output / 'development', seed=seed)
    if before_inputs != _input_hashes(diagnostic) or before_sources != [state_hash(m) for m in sources]:
        raise ValueError('Training changed shared source inputs or models')
    completion = dict(status='COPY_TRAINING_ARM_COMPLETE', arm=mode, seed=seed,
        residual_steps=200, surrogate_steps=200, discriminator_steps=200, victim_epochs=200,
        generator_selected_epoch=199, victim_selected_epoch=victim_record['trace']['selected_auc_epoch'],
        inputs_unchanged=True, frozen_sources_unchanged=True,
        initial_cpu_rng_sha256=result['initial_cpu_rng_sha256'],
        distribution_initial_state_sha256=result['distribution_initial_state_sha256'],
        victim_initialization_digest=victim_record['initialization_digest'],
        victim_fit_rng_digest=victim_record['fit_rng_digest'], automatic_followup=False)
    _write_json(output / 'completion.json', completion)
    progress(arm=mode, stage='arm_complete')
    return completion


def run(bundle, plan_path):
    from .copy_training_reporting import build_training_comparison
    plan = validate_plan(bundle, plan_path)
    seed = plan['protocol']['seed']
    output = Path(plan_path).resolve().parent
    _write_json(output / 'execution_claim.json', dict(started_utc=_now(), pid=os.getpid(),
        plan_sha256=sha256(plan_path), automatic_followup=False))
    start, ledger, completed = time.monotonic(), [], []
    torch.set_num_threads(2)
    with (output / 'progress.jsonl').open('x', buffering=1) as stream:
        def progress(**record):
            record.update(utc=_now(), elapsed_seconds=time.monotonic() - start)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if record.get('epoch', 0) % 20 == 0 or record.get('epoch') == 199:
                print(json.dumps(record, allow_nan=False), flush=True)
        try:
            progress(stage='read_original_train_dev')
            diagnostic = read_inputs(bundle, plan['protocol']['sources'], ledger)
            normalizer = load_normalizer(bundle)
            before_inputs = _input_hashes(diagnostic)
            _validate(_dev_view(bundle, diagnostic))
            targets = diagnostic['populations']['dev_public_single']
            if (len(targets) != 112 or sum(r['motion_regime'] == 'moving' for r in targets) != 33
                    or len(ledger) != 640 or sum(r['role'] == 'train' for r in ledger) != 512
                    or sum(r['role'] == 'dev' for r in ledger) != 128
                    or any(r['split'] != 'train' or r['status'] != 'READ_COMPLETE' for r in ledger)):
                raise ValueError('Registered target population or graph access scope changed')
            _write_json(output / 'populations.json', dict(dev_targets=targets,
                attachment_indices=diagnostic['indices']['idx_attach'].tolist(),
                outer_indices=diagnostic['outer'].tolist()))
            _write_json(output / 'frozen_input_hashes.json', before_inputs)
            _write_json(output / 'input_graph_access_ledger.json', ledger)
            anchor = restore_generator(bundle, SEED)
            start_surrogate = restore_surrogate(bundle, SEED)
            proxies = [restore_proxy(bundle, index=i) for i in (1, 2)]
            clean = restore_clean_reference(bundle)
            sources = [anchor, start_surrogate, *proxies, clean]
            before_sources = [state_hash(m) for m in sources]
            quality = [_dev_quality(model, diagnostic['combined']['dev']) for model in proxies]
            if any(row['gate'] != 'PASS_PROXY_QUALITY' for row in quality):
                raise ValueError('Frozen clean proxy failed existing normal-DEV quality gate')
            _write_json(output / 'proxy_quality.json', quality)
            shared = None
            for mode in ARMS:
                record = fit_one_arm(bundle, diagnostic, normalizer, anchor, start_surrogate,
                                     proxies, clean, mode, output / mode, progress, seed=seed)
                if before_inputs != _input_hashes(diagnostic) or before_sources != [state_hash(m) for m in sources]:
                    raise ValueError('Arm changed shared source inputs or models')
                match = {k: record[k] for k in ('initial_cpu_rng_sha256', 'distribution_initial_state_sha256',
                                               'victim_initialization_digest', 'victim_fit_rng_digest')}
                if shared is not None and match != shared:
                    raise ValueError('Arm initial discriminator, training RNG or victim initialization differs')
                shared = match
                completed.append(mode)
                if plan != validate_plan(bundle, plan_path):
                    raise ValueError('Plan changed during training')
                gc.collect()
            comparison = build_training_comparison(output)
            _write_json(output / 'comparison.json', comparison)
            _write_json(output / 'completion.json', dict(status='COPY_TRAINING_COMPLETE_REVIEW_REQUIRED',
                completed_utc=_now(), elapsed_seconds=time.monotonic() - start, completed_arms=completed,
                comparison_sha256=sha256(output / 'comparison.json'), graph_reads=len(ledger),
                updates_per_arm=dict(generator=200, surrogate=200, discriminator=200, victim=200),
                common_initialization_checks=True, inputs_unchanged=True, frozen_sources_unchanged=True,
                original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
                audit_arrays_loaded=False, confirmation_arrays_loaded=False,
                automatic_followup=False, automatic_promotion=False))
            progress(stage='pilot_complete')
            return comparison
        except BaseException as error:
            _write_json(output / 'failure.json', dict(status='FAILED_NO_FURTHER_JOBS',
                type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                completed_arms=completed, at=_now(), elapsed_seconds=time.monotonic() - start))
            raise
        finally:
            _write_json(output / 'graph_access_ledger.json', ledger)


def prior_evidence():
    """Bind only saved seed21 pilot evidence; never reread graphs or weights."""
    root = ROOT / 'results' / 'copy_consistency_equal_budget_v1_20261028'
    completion = json.loads((root / 'completion.json').read_text())
    if (completion['status'] != 'COPY_TRAINING_COMPLETE_REVIEW_REQUIRED'
            or completion['comparison_sha256'] != sha256(root / 'comparison.json')):
        raise ValueError('Prior pilot is incomplete or its comparison changed')
    # Bind every saved JSON/CSV consumed by the independent reporter, including
    # the raw per-target values. No checkpoint file is loaded by this function.
    paths = [root / name for name in ('plan.json', 'registration.json', 'completion.json',
                                      'comparison.json', 'frozen_input_hashes.json')]
    for arm in ARMS:
        paths.extend(root / arm / (name + '.json') for name in ('initial_state', 'victim_record', 'completion'))
        paths.extend(root / arm / 'development' / name for name in (
            'summary.json', 'targets.csv', 'clean_pair_scores.csv', 'triggered_pair_scores.csv', 'copy_rows.json'))
    return dict(seed=SEED, path=str(root), role='already_observed_exploratory_pilot',
                files={str(p.relative_to(root)): sha256(p) for p in paths})


def prepare_replication(bundle, output):
    """Register exactly two additional paired repetitions, before either fit."""
    evidence = prior_evidence()
    protocol(bundle)  # All metadata checks precede output creation.
    output = _directory(output)
    output.mkdir(parents=True, exist_ok=False)
    children = []
    for seed in REPLICATION_SEEDS:
        child = prepare(bundle, output / ('seed' + str(seed)), seed=seed)
        children.append(dict(seed=seed, path=str(child), plan_sha256=sha256(child)))
    spec = dict(version='copy_consistency_paired_repetition_v1', parent_seed=SEED,
        prospective_seeds=list(REPLICATION_SEEDS), children=children, prior_pilot=evidence,
        order='seed22 identity,pair_mean; seed23 identity,pair_mean; sequential, no retries',
        hypothesis='Whether the observed consistency/effectiveness tradeoff repeats under training randomness',
        primary_deltas=['overall_incremental_asr', 'moving_incremental_asr',
                        'adjacent_incremental_fp', 'normal_moving_positive_recall'],
        supplementary=['copy_violation_rate', 'standardized_delta_l1_l2_linf', 'normal_auc_ap_recall',
                       'slow_incremental_asr', 'nonincident_fp', 'positive_suppression'],
        decision='Report signed paired differences; mixed signs are not equivalence; persistent losses are tradeoffs',
        fixed_population=dict(dev_scenes=128, targets=112, moving_targets=33, slow_targets=79),
        aggregation='Mean and sample SD of per-seed pair_mean minus identity; rates use original fixed denominators',
        evidence_groups=['two prospective repetitions only', 'all three seeds descriptive including observed pilot'],
        on_performance_failure='complete both prelisted seeds; retain all original gates and failures',
        on_integrity_failure='stop queue, retain evidence, never auto-retry',
        no_adaptive_seed_addition=True, no_new_module=True, no_hyperparameter_search=True,
        no_repeated_target_independence_claim=True, no_new_data_generalization_claim=True,
        shared_identity_trained_parent=True, automatic_promotion=False, automatic_followup=False,
        audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        original_validation_arrays_allowed=False, original_test_arrays_allowed=False)
    _write_json(output / 'plan.json', dict(created_utc=_now(), protocol=spec))
    _write_json(output / 'registration.json', dict(created_utc=_now(), plan_sha256=sha256(output / 'plan.json'),
        graph_arrays_loaded=False, model_checkpoints_loaded=False, training_started=False))
    validate_replication_plan(bundle, output / 'plan.json')
    return output / 'plan.json'


def validate_replication_plan(bundle, path):
    path = Path(path).resolve()
    if path.name != 'plan.json' or _directory(path.parent) != path.parent:
        raise ValueError('Require a registered replication plan under results/')
    registration = json.loads((path.parent / 'registration.json').read_text())
    plan = json.loads(path.read_text())
    spec = plan['protocol']
    if (registration['plan_sha256'] != sha256(path)
            or spec['version'] != 'copy_consistency_paired_repetition_v1'
            or spec['parent_seed'] != SEED or spec['prospective_seeds'] != list(REPLICATION_SEEDS)
            or spec['prior_pilot'] != prior_evidence()
            or len(spec['children']) != len(REPLICATION_SEEDS)):
        raise ValueError('Replication registration, seed list or prior evidence changed')
    for seed, child in zip(REPLICATION_SEEDS, spec['children']):
        expected = path.parent / ('seed' + str(seed)) / 'plan.json'
        if (type(child['seed']) is not int or child['seed'] != seed
                or child['path'] != str(expected) or child['plan_sha256'] != sha256(expected)):
            raise ValueError('Replication child path, order, seed or hash changed')
        value = validate_plan(bundle, expected)
        if value['protocol']['seed'] != seed or value['protocol']['parent_seed'] != SEED:
            raise ValueError('Child training seed or common parent changed')
    return plan


def build_replication_summary(bundle, path):
    """Read saved evidence only; no new forward pass or old-result overwrite."""
    from .copy_training_reporting import build_training_comparison
    plan = validate_replication_plan(bundle, path)
    comparisons = {}
    sources = [(SEED, Path(plan['protocol']['prior_pilot']['path']))]
    sources.extend((child['seed'], Path(child['path']).parent) for child in plan['protocol']['children'])
    evidence = {}
    for seed, directory in sources:
        done = json.loads((directory / 'completion.json').read_text())
        if (done['status'] != 'COPY_TRAINING_COMPLETE_REVIEW_REQUIRED'
                or done['comparison_sha256'] != sha256(directory / 'comparison.json')):
            raise ValueError('Missing or altered completed child comparison')
        comparisons[seed] = build_training_comparison(directory)
        if comparisons[seed]['seed'] != seed:
            raise ValueError('Reported seed does not match registered repetition')
        evidence[str(seed)] = dict(path=str(directory), comparison_sha256=sha256(directory / 'comparison.json'),
                                   recomputation='read-only from registered JSON/CSV; source results unchanged')
    baseline = comparisons[SEED]
    common_fields = ('generator_parent_state_sha256', 'surrogate_state_sha256',
                     'proxy_state_sha256', 'clean_reference_state_sha256')
    for comparison in comparisons.values():
        if (comparison['population_identity_sha256'] != baseline['population_identity_sha256']
                or any(comparison['common_initial_state'][key] != baseline['common_initial_state'][key]
                       for key in common_fields)):
            raise ValueError('Cross-seed population or fixed pretrained starting states differ')
    for field in ('initial_cpu_rng_sha256', 'distribution_initial_state_sha256',
                  'victim_initialization_digest', 'victim_fit_rng_digest'):
        if len({c['arm_completions']['identity'][field] for c in comparisons.values()}) != 3:
            raise ValueError('Training randomness did not vary across registered seeds: ' + field)

    def endpoints(comparison, arm):
        metrics = comparison['metrics'][arm]['victim']
        moving = metrics['normal_by_pair_motion']['moving']
        return dict(overall_incremental_asr=metrics['targets']['all']['incremental_asr'],
            moving_incremental_asr=metrics['targets']['moving']['incremental_asr'],
            slow_incremental_asr=metrics['targets']['slow']['incremental_asr'],
            adjacent_incremental_fp=metrics['collateral']['adjacent_incremental_fp'],
            normal_moving_positive_recall=moving['clean_pair_positive_recall'],
            normal_auc=metrics['clean_pair_auc'], normal_ap=metrics['clean_pair_ap'],
            normal_positive_recall=metrics['clean_pair_positive_recall'])

    rows = []
    for seed, comparison in comparisons.items():
        left, right = (endpoints(comparison, arm) for arm in ARMS)
        rows.append(dict(seed=seed, role='prior_observed_pilot' if seed == SEED else 'prospective_repetition',
            identity=left, pair_mean=right,
            delta_pair_mean_minus_identity={key: None if left[key] is None or right[key] is None
                else right[key] - left[key] for key in left},
            acceptability=comparison['acceptability'], copy_statistics=comparison['copy_statistics'],
            normal_by_pair_motion={arm: comparison['metrics'][arm]['victim']['normal_by_pair_motion'] for arm in ARMS},
            paired_target_responses=comparison['paired_target_responses']['victim'],
            full_metrics=comparison['metrics'], full_paired_deltas=comparison['delta_pair_mean_minus_identity']))

    def aggregate(selected):
        result = {}
        for key in selected[0]['delta_pair_mean_minus_identity']:
            values = [row['delta_pair_mean_minus_identity'][key] for row in selected]
            defined = all(value is not None for value in values)
            result[key] = dict(per_seed=values, mean=statistics.mean(values) if defined else None,
                sample_sd=statistics.stdev(values) if defined and len(values) > 1 else None,
                positive=sum(v > 0 for v in values if v is not None),
                zero=sum(v == 0 for v in values if v is not None),
                negative=sum(v < 0 for v in values if v is not None))
        return dict(seeds=[row['seed'] for row in selected], paired_deltas=result,
                    statistical_significance_claimed=False, equivalence_claimed=False)

    return dict(status='COPY_REPLICATION_COMPLETE_REVIEW_REQUIRED', per_seed=rows,
        prospective_two_seed_summary=aggregate(rows[1:]),
        descriptive_all_three_seed_summary=aggregate(rows),
        sources=evidence, population_identity_sha256=baseline['population_identity_sha256'],
        distinct_target_count=112, distinct_moving_target_count=33,
        interpretation='Same reused DEV targets, common identity-trained parent; repeats of training randomness, not new independent targets or blind confirmation',
        old_results_rewritten=False, new_model_inference=False, independent_confirmation_claim=False,
        automatic_followup=False, automatic_promotion=False)


def run_replication(bundle, path):
    plan = validate_replication_plan(bundle, path)
    output = Path(path).resolve().parent
    _write_json(output / 'execution_claim.json', dict(started_utc=_now(), pid=os.getpid(), plan_sha256=sha256(path)))
    start, completed = time.monotonic(), []
    with (output / 'progress.jsonl').open('x', buffering=1) as stream:
        def progress(**row):
            row.update(utc=_now(), elapsed_seconds=time.monotonic() - start)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row, allow_nan=False), flush=True)
        try:
            for child in plan['protocol']['children']:
                validate_replication_plan(bundle, path)
                child_path = Path(child['path'])
                if (child_path.parent / 'execution_claim.json').exists() or (child_path.parent / 'launch_claim.json').exists():
                    raise ValueError('Replication child has already been claimed; no auto-retry')
                progress(stage='seed_started', seed=child['seed'])
                command = [sys.executable, '-B', '-m', 'spear_current.copy_pilot', '--run', '--plan', str(child_path)]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                with (child_path.parent / 'terminal.log').open('xb') as log:
                    result = subprocess.run(command, cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"Registered seed {child['seed']} failed with exit code {result.returncode}")
                completion = json.loads((child_path.parent / 'completion.json').read_text())
                if (completion['status'] != 'COPY_TRAINING_COMPLETE_REVIEW_REQUIRED'
                        or completion['comparison_sha256'] != sha256(child_path.parent / 'comparison.json')):
                    raise ValueError('Child completion or comparison integrity failed')
                completed.append(child['seed'])
                progress(stage='seed_complete', seed=child['seed'], performance_does_not_change_fixed_queue=True)
            summary = build_replication_summary(bundle, path)
            _write_json(output / 'summary.json', summary)
            _write_json(output / 'completion.json', dict(status='COPY_REPLICATION_COMPLETE_REVIEW_REQUIRED',
                completed_utc=_now(), completed_seeds=completed, elapsed_seconds=time.monotonic() - start,
                summary_sha256=sha256(output / 'summary.json'), automatic_followup=False, automatic_promotion=False))
            progress(stage='replication_complete')
            return summary
        except BaseException as error:
            _write_json(output / 'failure.json', dict(status='FAILED_NO_FURTHER_JOBS', completed_seeds=completed,
                type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), at=_now()))
            raise


def launch(bundle, plan_path, *, replication=False):
    """Explicitly launch exactly the registered two-arm run, detached with logs."""
    (validate_replication_plan if replication else validate_plan)(bundle, plan_path)
    path = Path(plan_path).resolve()
    output = path.parent
    if (output / 'execution_claim.json').exists():
        raise ValueError('Prepared pilot has already been executed or claimed')
    _write_json(output / 'launch_claim.json', dict(created_utc=_now(), plan_sha256=sha256(path)))
    command = [sys.executable, '-B', '-m', 'spear_current.copy_pilot', '--run', '--plan', str(path)]
    if replication:
        command.append('--replication')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (output / 'terminal.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='COPY_TRAINING_LAUNCHED', pid=process.pid, launched_utc=_now(), command=command,
                  cwd=str(ROOT), log=str(output / 'terminal.log'), automatic_followup=False)
    _write_json(output / 'launch.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check-only', action='store_true')
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--launch', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--seed', type=int, choices=(SEED, *REPLICATION_SEEDS))
    parser.add_argument('--replication', action='store_true', help='Fixed two-seed paired repetition, not a seed search')
    args = parser.parse_args()
    if args.seed is not None and (args.replication or args.run or args.launch):
        parser.error('--seed is only for individual preparation/checks; runs consume the registered seed')
    if args.prepare and (args.output_dir is None or args.plan is not None):
        parser.error('--prepare requires --output-dir only')
    if (args.run or args.launch) and (args.plan is None or args.output_dir is not None):
        parser.error('--run/--launch requires --plan only')
    if not (args.prepare or args.run or args.launch) and (args.output_dir is not None or args.plan is not None):
        parser.error('Check-only accepts no output directory/plan')
    bundle = load_bundle()
    if args.prepare:
        path = prepare_replication(bundle, args.output_dir) if args.replication else prepare(bundle, args.output_dir, seed=args.seed or SEED)
        print('Prepared: ' + str(path), flush=True)
    elif args.launch:
        print(json.dumps(launch(bundle, args.plan, replication=args.replication), allow_nan=False), flush=True)
    elif args.run:
        (run_replication if args.replication else run)(bundle, args.plan)
        print('Review: ' + str(args.plan.parent / ('summary.json' if args.replication else 'comparison.json')), flush=True)
    else:
        for seed in REPLICATION_SEEDS if args.replication else (args.seed or SEED,):
            protocol(bundle, seed)
        if args.replication:
            prior_evidence()
        print('COPY_TRAINING_PREFLIGHT_OK; no graph arrays/model restoration/training', flush=True)


if __name__ == '__main__':
    main()
