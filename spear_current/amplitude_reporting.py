# Source: experiments/spear_amplitude_control_v1/reporting.py; mechanically migrated; algorithms and state keys unchanged.
"""Pure posthoc amplitude-control reporting for authorized local AV2 research.

No I/O, fitting, seed selection, holdout access, or independent-seed claim.
Checkpoint/row identity and the TRAIN-only scalar fit are caller obligations.
"""
from collections.abc import Mapping
from copy import deepcopy

from . import domain_reporting as validation
from . import distribution_statistics as statistics
from .replication_reporting import (
    DISTRIBUTION_PATHS, PERFORMANCE_PATHS, SEEDS, STRATA, _lookup, _number, _summarize,
)


ARMS = ('continuation', 'unconditional', 'amplitude_scaled')
PAIRS = (
    ('unconditional_minus_amplitude_scaled', 'unconditional', 'amplitude_scaled'),
    ('amplitude_scaled_minus_continuation', 'amplitude_scaled', 'continuation'),
)
INTERPRETATION = dict(posthoc_development_control=True, same_reused_original_TRAIN_audit=True,
    new_independent_end_to_end_seeds=False, independent_data_confirmation=False,
    statistical_significance_claim=False, independence_assumed=False,
    distribution_preservation_claim=False, physical_validity_claim=False,
    new_method_claim=False, automatic_training=False, automatic_promotion=False)


def _gates(metrics, original):
    def number(arm, path):
        return _number(_lookup(metrics, (arm, 'victim', *path)))

    def check(value, threshold, operation):
        return None if value is None or threshold is None else operation(value, threshold)

    result = {}
    for arm in ARMS:
        checks = {
            'overall_incremental_asr_at_least_80pct': check(
                number(arm, ('targets', 'all', 'incremental_asr')), .8, lambda a, b: a >= b),
            'moving_incremental_asr_at_least_80pct': check(
                number(arm, ('targets', 'moving', 'incremental_asr')), .8, lambda a, b: a >= b),
            'adjacent_incremental_fp_strictly_below_5pct': check(
                number(arm, ('collateral', 'adjacent_incremental_fp')), .05, lambda a, b: a < b),
        }
        collapsed = _lookup(metrics, (arm, 'victim', 'all_clean_predictions_negative'))
        checks['victim_not_all_negative'] = not collapsed if type(collapsed) is bool else None
        references = [('original', _lookup(original, ('victim',)))]
        if arm != 'continuation':
            references.append(('matched_continuation', _lookup(metrics, ('continuation', 'victim'))))
        for name, reference in references:
            for metric, suffix, limit in (
                    ('clean_pair_auc', 'auc_drop_at_most_0p005', .005),
                    ('clean_pair_positive_recall', 'recall_drop_at_most_0p05', .05)):
                baseline = _number(_lookup(reference, (metric,)))
                checks[name + '_' + suffix] = check(number(arm, (metric,)),
                    None if baseline is None else baseline - limit, lambda a, b: a >= b)
        unavailable = [name for name, value in checks.items() if value is None]
        failed = [name for name, value in checks.items() if value is False]
        passed = None if unavailable else not failed
        result[arm] = dict(passed=passed, checks=checks, failed_checks=failed,
            unavailable_checks=unavailable, status='NOT_AVAILABLE' if unavailable else
            ('PASS_PRIMARY_80PCT_THRESHOLDS' if passed else 'BELOW_PRIMARY_80PCT_THRESHOLDS'))
    return result


def review_seed(reports, original_report):
    """Validate raw evaluations and retain all performance and quality failures."""
    if not isinstance(reports, Mapping) or set(reports) != set(ARMS):
        raise ValueError('Require exactly continuation, unconditional, and amplitude_scaled')
    original = validation._validate_report(original_report)
    metrics = {arm: validation._validate_report(reports[arm]) for arm in ARMS}
    for current in metrics.values():
        clean, old = current['clean_reference'], original['clean_reference']
        if any(clean[key] != old[key] for key in validation.NORMAL_KEYS):
            raise ValueError('Fixed clean-reference normal aggregates changed')
        for regime in validation.REGIMES:
            if any(clean['targets'][regime][key] != old['targets'][regime][key]
                   for key in ('targets', 'clean_negative_targets', 'clean_activation_rate')):
                raise ValueError('Fixed clean-reference clean target aggregates changed')
    comparisons = {arm: {name: {role: validation._difference(metrics[arm][role], reference[role])
        for role in validation.ROLES} for name, reference in (
            ('versus_original_SAGE24', original),
            ('versus_matched_continuation', metrics['continuation']),
            ('versus_unconditional', metrics['unconditional']))} for arm in ARMS}
    return dict(status='AMPLITUDE_CONTROL_REVIEW_COMPLETE', arms=list(ARMS), metrics=metrics,
        original_metrics=original, comparisons=comparisons, gates=_gates(metrics, original),
        population=deepcopy(validation.EXPECTED_COUNTS),
        gate_contract=dict(overall_incremental_asr_min=.8, moving_incremental_asr_min=.8,
            adjacent_incremental_fp_max_exclusive=.05, normal_auc_drop_max=.005,
            normal_positive_recall_drop_max=.05, all_negative_victim_allowed=False,
            quality_references={arm: ['original_SAGE24'] +
                ([] if arm == 'continuation' else ['matched_continuation']) for arm in ARMS}),
        integrity=dict(clean_reference_normal_aggregates_equal=True,
            clean_reference_clean_target_aggregates_equal=True,
            victim_normal_prediction_equality_required=False,
            raw_csv_identity_and_auc_recomputation='must be verified independently by caller'),
        scope='posthoc amplitude control on the reused original TRAIN development audit',
        interpretation=deepcopy(INTERPRETATION), original_val_arrays_loaded=False,
        original_test_arrays_loaded=False)


def evaluate_distribution(reference, clean, conditions, moving, triggered_by_arm):
    """Use the exact frozen reference through a local arm-name adapter.

    The third historical slot is transport only; neither module constants nor
    reference objects are modified. Every returned arm label is renamed.
    """
    if not isinstance(triggered_by_arm, Mapping) or set(triggered_by_arm) != set(ARMS):
        raise ValueError('Require exactly continuation, unconditional, and amplitude_scaled')
    result = statistics.evaluate(reference, clean, conditions, moving, {
        'continuation': triggered_by_arm['continuation'],
        'unconditional': triggered_by_arm['unconditional'],
        'conditional': triggered_by_arm['amplitude_scaled']})
    result['reference']['arms'] = list(ARMS)
    for part in result['strata'].values():
        part['arms']['amplitude_scaled'] = part['arms'].pop('conditional')
    result['interpretation'].update(INTERPRETATION)
    return result


def _extract(review, distribution):
    performance = {arm: {name: _number(_lookup(review, ('metrics', arm, 'victim', *path)))
        for name, path in PERFORMANCE_PATHS.items()} for arm in ARMS}
    diagnostics = {}
    for stratum in STRATA:
        part = _lookup(distribution, ('strata', stratum))
        diagnostics[stratum] = {}
        for arm in ARMS:
            point = _lookup(part, ('arms', arm)) if _lookup(part, ('status',)) == 'AVAILABLE' else None
            diagnostics[stratum][arm] = {name: _number(_lookup(point, path))
                for name, path in DISTRIBUTION_PATHS.items()}
    return performance, diagnostics


def aggregate(replicates):
    """Require all three fixed seeds; pair within seed before mean/sample SD."""
    if not isinstance(replicates, (list, tuple)) or len(replicates) != len(SEEDS):
        raise ValueError('Require exactly the three fixed seed records')
    records = {}
    for item in replicates:
        if (not isinstance(item, Mapping) or type(item.get('seed')) is not int
                or item['seed'] not in SEEDS or item['seed'] in records
                or not isinstance(item.get('review'), Mapping)
                or not isinstance(item.get('distribution'), Mapping)):
            raise ValueError('Malformed, duplicate or unregistered fixed seed record')
        records[item['seed']] = item
    per_seed = {}
    for seed in SEEDS:
        review = records[seed]['review']
        performance, distribution = _extract(review, records[seed]['distribution'])
        gates = _gates(review.get('metrics'), review.get('original_metrics'))
        # Missing saved evidence never becomes a newly passing aggregate gate.
        for arm in ARMS:
            for name in gates[arm]['checks']:
                saved = _lookup(review, ('gates', arm, 'checks', name))
                if type(saved) is not bool:
                    gates[arm]['checks'][name] = None
                elif not saved:
                    gates[arm]['checks'][name] = False
            checks = gates[arm]['checks']
            gates[arm]['failed_checks'] = [key for key, value in checks.items() if value is False]
            gates[arm]['unavailable_checks'] = [key for key, value in checks.items() if value is None]
            gates[arm]['passed'] = None if None in checks.values() else all(checks.values())
            gates[arm]['status'] = 'NOT_AVAILABLE' if None in checks.values() else (
                'PASS_PRIMARY_80PCT_THRESHOLDS' if gates[arm]['passed'] else 'BELOW_PRIMARY_80PCT_THRESHOLDS')
        per_seed[str(seed)] = dict(seed=seed, design_role='posthoc_development_control',
            performance=performance, distribution=distribution, primary_gates=gates)

    def summarize(section, metric, left, right=None, stratum=None):
        values = {}
        for seed in SEEDS:
            point = per_seed[str(seed)][section]
            point = point if stratum is None else point[stratum]
            value = point[left][metric]
            other = 0. if right is None else point[right][metric]
            values[seed] = None if value is None or other is None else value - other
        return _summarize(values)

    def summaries(left, right=None):
        return dict(performance={metric: summarize('performance', metric, left, right)
            for metric in PERFORMANCE_PATHS}, distribution={stratum: {
                metric: summarize('distribution', metric, left, right, stratum)
                for metric in DISTRIBUTION_PATHS} for stratum in STRATA})

    arm_summaries = {arm: summaries(arm) for arm in ARMS}
    summary = dict(performance={arm: arm_summaries[arm]['performance'] for arm in ARMS},
        distribution={stratum: {arm: arm_summaries[arm]['distribution'][stratum] for arm in ARMS}
            for stratum in STRATA})
    paired = {name: dict(left_arm=left, right_arm=right,
        difference='left minus right within the same seed', **summaries(left, right))
        for name, left, right in PAIRS}
    gate_summary = {}
    for arm in ARMS:
        values = {str(seed): per_seed[str(seed)]['primary_gates'][arm]['passed'] for seed in SEEDS}
        available = all(value is not None for value in values.values())
        gate_summary[arm] = dict(status='AVAILABLE' if available else 'NOT_AVAILABLE',
            values_by_seed=values, pass_count=sum(value is True for value in values.values()),
            available_count=sum(value is not None for value in values.values()), expected_count=3,
            all_three_seeds_pass=all(values.values()) if available else None)
    complete = all(value['status'] == 'AVAILABLE' for arm in arm_summaries.values()
        for section in (arm['performance'], *arm['distribution'].values()) for value in section.values())
    return dict(status='AMPLITUDE_CONTROL_SUMMARY_COMPLETE', seeds=list(SEEDS), arms=list(ARMS),
        per_seed=per_seed, summary=summary, paired_differences=paired,
        primary_gate_summary=gate_summary, complete_metric_coverage=complete,
        protocol=dict(fixed_anchor=True, fixed_seed_count=3, summary_unit='seed, not target',
            seed_weighting='equal; never target-count weighted', pooled_target_count_used=False,
            sample_standard_deviation_ddof=1, seed_exclusion=False, best_seed_selection=False,
            unavailable_metric_policy='retain all seeds; no zero imputation or partial-seed mean/SD',
            scalar_fit_population='26 fixed TRAIN poison attachments per seed',
            scalar_fit='globally scale continuation changes to unconditional mean selected-feature L1',
            quality_reference_interpretation='original SAGE24 is fixed; matched continuation is within-seed',
            source_and_anchor_binding_verification='caller responsibility'),
        metric_directions=dict(performance={name: 'lower' if name == 'adjacent_incremental_fp' else 'higher'
            for name in PERFORMANCE_PATHS}, distribution={name: 'lower descriptive distance/rate/norm'
                for name in DISTRIBUTION_PATHS}), interpretation=deepcopy(INTERPRETATION))
