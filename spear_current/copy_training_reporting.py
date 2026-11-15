"""Read-only comparison of matched identity/pair-mean warm-start continuations.

This reads only the two new arms' JSON/CSV evidence and their prepared plan.
Post-training generator/surrogate/victim states and clean predictions may differ.
There is deliberately no cross-arm mean oracle or norm-nonincreasing assertion:
those describe fixed-weight postprocessing, not separately trained generators.
No graph/checkpoint access, training, model inference, or output write occurs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from .copy_diagnostic_reporting import (
    ARMS, COPY_POSITIONS, METADATA, MODELS, STRATA, _copy_stats, _count,
    _decode_json, _load_arm, _normal, _require, _sha,
)


GATES = dict(overall_incremental_asr_min=.80, moving_incremental_asr_min=.80,
    adjacent_incremental_fp_max_exclusive=.05, auc_drop_vs_clean_reference_max=.005,
    positive_recall_drop_vs_clean_reference_max=.05)
SEED = 20260621
ALLOWED_SEEDS = (20260621, 20260622, 20260623)
EXPECTED_POPULATION = dict(scenes=128, all=112, slow=79, moving=33)
AUC_POLICY = 'maximum_dev_roc_auc_strict_improvement_earliest_tie'
INITIAL_FIELDS = ('generator_parent_state_sha256', 'surrogate_state_sha256',
                  'proxy_state_sha256', 'clean_reference_state_sha256', 'seed')
STEP_FIELDS = ('residual_steps', 'surrogate_steps', 'discriminator_steps', 'victim_epochs')
COMMON_COMPLETION_FIELDS = ('initial_cpu_rng_sha256', 'distribution_initial_state_sha256',
                            'victim_initialization_digest', 'victim_fit_rng_digest')


def _read(path):
    data = path.read_bytes()
    return _decode_json(data), dict(sha256=hashlib.sha256(data).hexdigest())


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _rate_gate(value, numerator, denominator, threshold, operator):
    defined = denominator > 0 and _finite(value)
    passed = defined and (value >= threshold if operator == '>=' else value < threshold)
    return dict(value=value, numerator=numerator, denominator=denominator,
        operator=operator, threshold=threshold, passed=bool(passed),
        reason='defined' if defined else 'undefined_denominator')


def _drop_gate(value, reference, maximum_drop):
    defined = _finite(value) and _finite(reference)
    drop = reference - value if defined else None
    return dict(value=value, reference_value=reference, measured_drop=drop,
        operator='<=', maximum_drop=maximum_drop,
        passed=bool(defined and drop <= maximum_drop),
        reason='defined' if defined else 'undefined_metric')


def _acceptability(metrics):
    victim, reference = metrics['victim'], metrics['clean_reference']
    all_targets, moving = victim['targets']['all'], victim['targets']['moving']
    collateral = victim['collateral']
    predictions = victim['clean_prediction_distribution']
    checks = dict(
        overall_incremental_asr=_rate_gate(all_targets['incremental_asr'],
            all_targets['incremental_flips'], all_targets['targets'], GATES['overall_incremental_asr_min'], '>='),
        moving_incremental_asr=_rate_gate(moving['incremental_asr'],
            moving['incremental_flips'], moving['targets'], GATES['moving_incremental_asr_min'], '>='),
        adjacent_incremental_fp=_rate_gate(collateral['adjacent_incremental_fp'],
            collateral['adj_flip'], collateral['adj_negative'], GATES['adjacent_incremental_fp_max_exclusive'], '<'),
        clean_auc_vs_clean_reference=_drop_gate(victim['clean_pair_auc'], reference['clean_pair_auc'],
                                               GATES['auc_drop_vs_clean_reference_max']),
        positive_recall_vs_clean_reference=_drop_gate(victim['clean_pair_positive_recall'],
            reference['clean_pair_positive_recall'], GATES['positive_recall_drop_vs_clean_reference_max']),
        noncollapsed_clean_predictions=dict(class_0=predictions['class_0'], class_1=predictions['class_1'],
            rule='both binary prediction classes must have positive counts',
            passed=predictions['class_0'] > 0 and predictions['class_1'] > 0))
    return dict(acceptable_performance=all(c['passed'] for c in checks.values()), checks=checks,
                failed_checks=[key for key, check in checks.items() if not check['passed']])


def _initial(record, seed):
    _require(type(record['seed']) is int and record['seed'] == seed, 'Initial seed changed')
    for key in ('generator_parent_state_sha256', 'surrogate_state_sha256', 'clean_reference_state_sha256'):
        _sha(record[key])
    proxies = record['proxy_state_sha256']
    _require(isinstance(proxies, list) and len(proxies) == 2, 'Require exactly two frozen proxy hashes')
    for value in proxies:
        _sha(value)
    return {key: deepcopy(record[key]) for key in INITIAL_FIELDS}


def _completion(record, arm, seed):
    _require(record['status'] == 'COPY_TRAINING_ARM_COMPLETE' and record['arm'] == arm
             and type(record['seed']) is int and record['seed'] == seed, 'Arm completion identity/status differs')
    for key in STEP_FIELDS:
        _require(_count(record[key]) == 200, 'Unequal or incomplete update budget: ' + key)
    _require(_count(record['generator_selected_epoch']) == 199, 'Generator must use the final epoch 199')
    _require(record['inputs_unchanged'] is True and record['frozen_sources_unchanged'] is True,
             'Training input or frozen source integrity failed')
    for key in COMMON_COMPLETION_FIELDS:
        _sha(record[key])
    return {key: deepcopy(record[key]) for key in ('status', 'arm', 'seed', *STEP_FIELDS,
        'generator_selected_epoch', 'inputs_unchanged', 'frozen_sources_unchanged', *COMMON_COMPLETION_FIELDS)}


def _victim(record, seed):
    for key in ('initialization_digest', 'fit_rng_digest'):
        _sha(record[key])
    _require(_count(record['victim_seed']) == seed, 'Fresh victim seed differs')
    trace = record['trace']
    _require(_count(trace['training_forwards']) == 200 and _count(trace['eval_forwards']) == 200,
             'Victim train/eval epoch budget differs')
    epoch = _count(trace['selected_auc_epoch'])
    _require(epoch < 200 and trace['selection_policy'] == AUC_POLICY,
             'Victim selection policy/epoch differs')
    row = trace['selected_auc_row']
    _require(_count(row['epoch']) == epoch and _finite(row['auc']) and 0 <= row['auc'] <= 1,
             'Victim selected AUC row is malformed')
    return dict(initialization_digest=record['initialization_digest'], fit_rng_digest=record['fit_rng_digest'],
        victim_seed=record['victim_seed'], training_forwards=200, eval_forwards=200,
        selected_auc_epoch=epoch, selected_auc=float(row['auc']), selection_policy=AUC_POLICY)


def _difference(candidate, identity):
    if isinstance(identity, dict):
        return {key: _difference(candidate[key], value) for key, value in identity.items()}
    if identity is None or candidate is None:
        return None
    if type(identity) is bool:
        return dict(identity=identity, pair_mean=candidate)
    if type(identity) in (int, float) and type(candidate) in (int, float):
        return candidate - identity
    return dict(identity=identity, pair_mean=candidate)


def _paired_targets(left, right, model, stratum):
    rows = [(a, b) for a, b in zip(left, right) if stratum == 'all' or a['motion_regime'] == stratum]
    result = dict(targets=len(rows))
    for name, predicate in (
        ('incremental_success', lambda r: r[model + '_clean_pred'] == 0 and r[model + '_trigger_pred'] == 1),
        ('triggered_positive', lambda r: r[model + '_trigger_pred'] == 1)):
        success = [(predicate(a), predicate(b)) for a, b in rows]
        gained, lost = sum(not a and b for a, b in success), sum(a and not b for a, b in success)
        result[name] = dict(gained=gained, lost=lost, retained=sum(a and b for a, b in success),
            neither=sum(not a and not b for a, b in success), net_count_difference=gained - lost)
    result['clean_prediction_disagreements'] = sum(a[model + '_clean_pred'] != b[model + '_clean_pred'] for a, b in rows)
    return result


def _normal_by_pair_motion(rows, model):
    """Describe complete normal-pair strata using each pair's own motion label."""
    return {stratum: _normal([row for row in rows
        if stratum == 'all' or row['motion_regime'] == stratum], model) for stratum in STRATA}


def build_training_comparison(root):
    """Validate new-run evidence and report fixed gates without promotion.

    The six performance checks use literal >=, < and <= comparisons, without
    epsilon relaxation. AUC/recall gates compare each arm to the common clean
    reference; they do not assert noninferiority to the identity arm. All
    pair-mean copy triplets must already be exactly equal in stored float32
    outputs. The victim's full 200-row epoch trace is not read or re-selected.
    Each report is bound to one registered seed (21/22/23); legacy plans without
    an explicit protocol seed retain seed21. Normal-pair motion-stratified
    metrics are descriptive and do not add or modify any performance gate.
    """
    try:
        root = Path(root)
        plan, plan_asset = _read(root / 'plan.json')
        registered_gates = plan['protocol']['gates']
        _require(isinstance(registered_gates, dict) and set(registered_gates) == set(GATES)
                 and all(type(registered_gates[k]) in (int, float) and registered_gates[k] == v
                         for k, v in GATES.items()), 'Prepared gates differ from the registered contract')
        seed = plan['protocol'].get('seed', SEED)
        _require(type(seed) is int and seed in ALLOWED_SEEDS, 'Prepared seed is not registered')
        if 'seed' in plan:
            _require(type(plan['seed']) is int and plan['seed'] == seed, 'Prepared seed fields differ')
        for name, expected in (('dev_scenes', 128), ('expected_dev_targets', 112),
                               ('expected_dev_slow', 79), ('expected_dev_moving', 33), ('train_scenes', 512)):
            _require(type(plan['protocol'][name]) is int and plan['protocol'][name] == expected,
                     'Prepared population differs: ' + name)
        arms, initials, victims, completions, files = {}, {}, {}, {}, {}
        for arm in ARMS:
            directory = root / arm
            records = {}
            files[arm] = {}
            for name in ('initial_state', 'victim_record', 'completion'):
                records[name], files[arm][name + '.json'] = _read(directory / (name + '.json'))
            initial_contract = records['initial_state']['contract']
            _require(records['initial_state']['mode'] == arm and initial_contract['mode'] == arm
                     and initial_contract['frozen_postprocess'] is False
                     and initial_contract['output_mode'] == 'domain_projection',
                     'Initial training wrapper mode/frozen flag differs')
            initials[arm] = _initial(records['initial_state'], seed)
            victims[arm] = _victim(records['victim_record'], seed)
            completions[arm] = _completion(records['completion'], arm, seed)
            _require(completions[arm]['victim_initialization_digest'] == victims[arm]['initialization_digest']
                     and completions[arm]['victim_fit_rng_digest'] == victims[arm]['fit_rng_digest'],
                     'Victim completion and victim record disagree')
            if 'victim_selected_epoch' in records['completion']:
                _require(_count(records['completion']['victim_selected_epoch']) == victims[arm]['selected_auc_epoch'],
                         'Victim selected epoch differs between records')
            arms[arm] = _load_arm(directory / 'development', arm, expected_seed=seed)
            current = arms[arm]
            for model in MODELS:
                current['metrics'][model]['normal_by_pair_motion'] = _normal_by_pair_motion(
                    current['rows']['normal'], model)
            files[arm]['development'] = current['assets']
            summary = current['summary']
            contract = summary['generator_contract']
            _require(contract['mode'] == arm and contract['frozen_postprocess'] is True
                     and contract['output_mode'] == 'domain_projection', 'Evaluation wrapper is not frozen in the expected mode')
            _require(summary['clean_reference_is_independent'] is False,
                     'The reused clean reference must not be claimed independent')
            _require(summary['model_state_sha256']['clean_reference'] == initials[arm]['clean_reference_state_sha256'],
                     'Final clean-reference weights differ from the frozen initial reference')
            _require(summary['dev_counts']['registered_scenes'] == EXPECTED_POPULATION['scenes'],
                     'Require the fixed DEV128 population')
            for stratum in STRATA:
                _require(current['metrics']['victim']['targets'][stratum]['targets'] == EXPECTED_POPULATION[stratum],
                         'Require fixed DEV112 targets with Slow79/Moving33')
            if arm == 'pair_mean':
                for positions in COPY_POSITIONS:
                    values = current['outputs'][:, positions]
                    _require((values == values[:, :1]).all(), 'Trained pair_mean output copies are not exactly equal')
        left, right = (arms[arm] for arm in ARMS)
        _require(initials['identity'] == initials['pair_mean'], 'Recorded common starting states differ')
        for key in ('initialization_digest', 'fit_rng_digest', 'victim_seed'):
            _require(victims['identity'][key] == victims['pair_mean'][key], 'Fresh victim initialization/RNG differs: ' + key)
        for key in COMMON_COMPLETION_FIELDS:
            _require(completions['identity'][key] == completions['pair_mean'][key],
                     'Recorded training initialization/RNG differs: ' + key)
        for field in ('population_identity_sha256', 'dev_counts'):
            _require(left['summary'][field] == right['summary'][field], 'Evaluation population identity/count differs')
        for kind in ('normal', 'targets', 'triggered'):
            fields = METADATA + (('target_pair_index', 'label', 'relation') if kind == 'triggered' else ())
            _require([{key: r[key] for key in fields} for r in left['rows'][kind]] ==
                     [{key: r[key] for key in fields} for r in right['rows'][kind]], 'Cross-arm CSV identities/order differ')
        _require(np.array_equal(left['clean'].view(np.uint32), right['clean'].view(np.uint32)),
                 'Cross-arm original clean feature rows differ')
        for a, b in zip(left['rows']['normal'], right['rows']['normal']):
            _require(all(a['clean_reference_' + suffix] == b['clean_reference_' + suffix] for suffix in ('p', 'pred')),
                     'Frozen clean-reference normal predictions differ')
        targets = left['rows']['targets']
        copy_statistics = {arm: {stratum: _copy_stats(value['clean'][mask], value['outputs'][mask])
            for stratum in STRATA
            for mask in [np.asarray([stratum == 'all' or row['motion_regime'] == stratum for row in targets], dtype=bool)]}
            for arm, value in arms.items()}
        gates = {arm: _acceptability(value['metrics']) for arm, value in arms.items()}
        return dict(status='COPY_TRAINING_COMPARISON_COMPLETE_REVIEW_REQUIRED', seed=seed,
            scope='single-seed warm-start matched continuation on reused original-TRAIN DEV; not end-to-end independent replication',
            preregistered_gates=deepcopy(GATES), performance_gate_model='victim',
            acceptability=gates, metrics={arm: value['metrics'] for arm, value in arms.items()},
            delta_pair_mean_minus_identity={model: _difference(right['metrics'][model], left['metrics'][model]) for model in MODELS},
            paired_target_responses={model: {stratum: _paired_targets(left['rows']['targets'], right['rows']['targets'], model, stratum)
                for stratum in STRATA} for model in MODELS},
            copy_statistics=copy_statistics,
            population_identity_sha256=left['summary']['population_identity_sha256'],
            population_counts=left['summary']['dev_counts'],
            common_initial_state=initials['identity'], victim_records=victims, arm_completions=completions,
            final_model_state_sha256={arm: value['summary']['model_state_sha256'] for arm, value in arms.items()},
            final_domain_generator_state_sha256={arm: value['summary']['shared_parent_generator_state_sha256'] for arm, value in arms.items()},
            source_files=dict(plan=plan_asset, arms=files),
            integrity=dict(common_recorded_initial_fields_match=True, victim_initialization_and_fit_rng_match=True,
                registered_seed_bound_across_records=True,
                equal_recorded_update_budgets=True, csv_metrics_and_hashes_recomputed=True,
                population_and_original_clean_rows_match=True, clean_reference_weights_and_normal_predictions_match=True,
                pair_mean_exact_copy_equalities=True, original30_allowed24_scope_verified=True,
                posttraining_generator_surrogate_victim_weights_may_differ=True,
                normal_victim_surrogate_predictions_may_differ=True,
                cross_arm_projection_oracle_applied=False, cross_arm_norm_nonincrease_required=False,
                full_victim_epoch_trace_independently_recomputed=False),
            interpretation=dict(training_generator='trainable CopyConsistentGenerator frozen=False; final residual epoch199',
                evaluation_generator='trained Domain weights rewrapped with the same consistency mode and frozen=True',
                victim_selection=AUC_POLICY, quality_reference='common frozen clean_reference, not the identity arm',
                paired_success_counts='same target identities but different trained models; clean prediction eligibility may differ',
                numeric_differences='descriptive pair_mean minus identity; no superiority or noninferiority gate',
                normal_motion_strata='each normal pair own motion_regime in the complete eligible clean CSV; not target motion',
                normal_motion_quality='AUC/AP/positive recall and paired-arm differences are descriptive; no new gate',
                normal_motion_undefined='AUC null without both labels; AP/positive recall null without positives; all-positive AP is degenerate',
                undefined_metrics='fail the affected performance check; no zero-denominator fallback',
                comparisons='literal floating-point >=, <, <=; no epsilon relaxation'),
            reporter_optimizer_steps=0, reporter_new_inference=False, graph_or_model_assets_read=False,
            automatic_followup=False, automatic_promotion=False, multiseed_evidence=False,
            independent_confirmation_claim=False, blind_evaluation_claim=False,
            statistical_significance_claimed=False, novelty_claim=False, physical_validity_claim=False)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('Missing or malformed copy training evidence') from error
