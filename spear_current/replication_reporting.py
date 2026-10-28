# Source: experiments/spear_distribution_replication_v1/reporting.py; mechanically migrated; algorithms and state keys unchanged.
"""Pure fixed-seed reporting for local, authorized AV2/GNN robustness research.

No I/O, model loading, fitting, seed selection, pooled-target inference or
significance tests. The caller binds checkpoints, raw CSVs and the fixed anchor.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

from . import distribution_reporting as prior


ARMS = ('continuation', 'unconditional', 'conditional')
SEEDS = (20260621, 20260622, 20260623)
STRATA = ('all', 'slow', 'moving')
PAIRS = (
    ('conditional_minus_unconditional', 'conditional', 'unconditional'),
    ('unconditional_minus_continuation', 'unconditional', 'continuation'),
    ('conditional_minus_continuation', 'conditional', 'continuation'),
)
PERFORMANCE_PATHS = {
    'all_incremental_asr': ('targets', 'all', 'incremental_asr'),
    'moving_incremental_asr': ('targets', 'moving', 'incremental_asr'),
    'adjacent_incremental_fp': ('collateral', 'adjacent_incremental_fp'),
    'clean_pair_auc': ('clean_pair_auc',),
    'clean_pair_positive_recall': ('clean_pair_positive_recall',),
}
DISTRIBUTION_PATHS = {
    'selected_mmd2': ('versus_train', 'rbf_mmd2_biased_selected'),
    'joint_mmd2': ('versus_train', 'rbf_mmd2_biased_joint'),
    'marginal_w1': ('versus_train', 'wasserstein_1d_mean'),
    'outlier_coordinate_fraction': ('train_quantile_outliers', 'coordinate_fraction'),
    'outlier_target_fraction': ('train_quantile_outliers', 'row_fraction'),
    'boundary_coordinate_fraction': ('hard_clamp_boundary', 'coordinate_fraction'),
    **{key: ('selected_modification', key) for key in
       ('l1_mean', 'l2_mean', 'linf_mean', 'l1_max', 'l2_max', 'linf_max')},
}
QUALITY_REFERENCE_INTERPRETATION = (
    'Historical SAGE24 quality reference is one fixed seed; across-seed changes '
    'mix initialization. Matched continuation is the main within-seed module '
    'contrast. This does not change the registered quality thresholds.'
)


def _lookup(value, path):
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _primary_gates(review):
    """Retain validated quality checks, replacing only the two ASR cutoffs.

    This also provides the new retrospective 80% view of the untouched old
    seed's 85% review. Absent evidence is unavailable, never a passing check.
    """
    gates = {}
    common = ('adjacent_incremental_fp_strictly_below_5pct', 'victim_not_all_negative',
              'original_auc_drop_at_most_0p005', 'original_recall_drop_at_most_0p05')
    for arm in ARMS:
        names = common + (() if arm == 'continuation' else
            ('matched_continuation_auc_drop_at_most_0p005', 'matched_continuation_recall_drop_at_most_0p05'))
        checks = {}
        for name in names:
            value = _lookup(review, ('gates', arm, 'checks', name))
            checks[name] = value if type(value) is bool else None
        for name, regime in (('overall', 'all'), ('moving', 'moving')):
            value = _number(_lookup(review, ('metrics', arm, 'victim', 'targets', regime, 'incremental_asr')))
            checks[name + '_incremental_asr_at_least_80pct'] = None if value is None else value >= .8
        available = all(value is not None for value in checks.values())
        passed = all(checks.values()) if available else None
        gates[arm] = dict(passed=passed, checks=checks,
            status=('PASS_PRIMARY_80PCT_THRESHOLDS' if passed else 'BELOW_PRIMARY_80PCT_THRESHOLDS')
                   if available else 'NOT_AVAILABLE',
            unavailable_checks=[name for name, value in checks.items() if value is None])
    return gates


def review_evaluations(reports, original_report, projected_report):
    """Use the old metric validation/review, with the prospectively fixed 80% gate."""
    result = prior.review_evaluations(reports, original_report, projected_report)
    historical = dict(primary=False, gates=deepcopy(result['gates']),
                      gate_contract=deepcopy(result['gate_contract']))
    result['gates'] = _primary_gates(result)
    result['gate_contract'].update(overall_incremental_asr_min=.8, moving_incremental_asr_min=.8,
        primary=True, timing='accepted before the two new seed plans and outcomes',
        quality_reference_interpretation=QUALITY_REFERENCE_INTERPRETATION)
    result.update(status='DISTRIBUTION_REPLICATION_REVIEW_COMPLETE', historical_comparison=historical,
        quality_reference_interpretation=QUALITY_REFERENCE_INTERPRETATION,
        scope='fixed-anchor seed replication on the same previously exposed original TRAIN audit; not independent data')
    return result


def _summarize(values):
    """All three seeds or unavailable; SD uses sample denominator n-1."""
    by_seed = {str(seed): _number(values[seed]) for seed in SEEDS}
    missing = [seed for seed in SEEDS if by_seed[str(seed)] is None]
    available = not missing
    mean = math.fsum(by_seed.values()) / len(SEEDS) if available else None
    deviation = math.sqrt(math.fsum((value - mean) ** 2 for value in by_seed.values())
                          / (len(SEEDS) - 1)) if available else None
    return dict(status='AVAILABLE' if available else 'NOT_AVAILABLE', values_by_seed=by_seed,
        n=len(SEEDS) - len(missing), expected_n=len(SEEDS), missing_seeds=missing,
        mean=mean, sd_ddof1=deviation, ddof=1,
        reason=None if available else 'at least one fixed seed has a missing or nonfinite metric; no partial summary')


def _extract(review, distribution):
    performance = {}
    for arm in ARMS:
        victim = _lookup(review, ('metrics', arm, 'victim'))
        performance[arm] = {name: _number(_lookup(victim, path)) for name, path in PERFORMANCE_PATHS.items()}
    diagnostic = {}
    for stratum in STRATA:
        part = _lookup(distribution, ('strata', stratum))
        available = _lookup(part, ('status',)) == 'AVAILABLE'
        diagnostic[stratum] = {}
        for arm in ARMS:
            point = _lookup(part, ('arms', arm)) if available else None
            diagnostic[stratum][arm] = {
                name: _number(_lookup(point, path)) for name, path in DISTRIBUTION_PATHS.items()}
    return performance, diagnostic


def aggregate(replicates):
    """Summarize exactly the observed seed and both fixed prospective seeds.

    Each item is ``dict(seed=int, review=dict, distribution=dict)``. Missing
    scalar evidence stays unavailable. Missing/duplicate/unregistered seed
    records are rejected, never dropped or replaced. Differences pair arms
    within each seed before taking the equally weighted seed mean/sample SD.
    """
    if not isinstance(replicates, (list, tuple)) or len(replicates) != len(SEEDS):
        raise ValueError('Require exactly the three fixed seed records, without selection or pooling')
    records = {}
    for item in replicates:
        if (not isinstance(item, Mapping) or type(item.get('seed')) is not int
                or item['seed'] not in SEEDS or item['seed'] in records
                or not isinstance(item.get('review'), Mapping)
                or not isinstance(item.get('distribution'), Mapping)):
            raise ValueError('Malformed, duplicate or unregistered fixed seed record')
        records[item['seed']] = item
    if set(records) != set(SEEDS):
        raise ValueError('Every fixed seed must be represented')

    per_seed = {}
    for seed in SEEDS:
        item = records[seed]
        performance, diagnostic = _extract(item['review'], item['distribution'])
        per_seed[str(seed)] = dict(seed=seed,
            design_role='retrospective_observed' if seed == SEEDS[0] else 'prospective_fixed',
            performance=performance, distribution=diagnostic,
            primary_gates=_primary_gates(item['review']),
            source_gates=deepcopy(item['review'].get('gates')),
            source_gate_contract=deepcopy(item['review'].get('gate_contract')))

    performance_summary = {arm: {metric: _summarize({seed:
        per_seed[str(seed)]['performance'][arm][metric] for seed in SEEDS})
        for metric in PERFORMANCE_PATHS} for arm in ARMS}
    distribution_summary = {stratum: {arm: {metric: _summarize({seed:
        per_seed[str(seed)]['distribution'][stratum][arm][metric] for seed in SEEDS})
        for metric in DISTRIBUTION_PATHS} for arm in ARMS} for stratum in STRATA}

    def paired_values(left, right, section, metric, stratum=None):
        values = {}
        for seed in SEEDS:
            point = per_seed[str(seed)][section]
            if stratum is not None:
                point = point[stratum]
            a, b = point[left][metric], point[right][metric]
            values[seed] = None if a is None or b is None else a - b
        return _summarize(values)

    paired = {}
    for name, left, right in PAIRS:
        paired[name] = dict(left_arm=left, right_arm=right, difference='left minus right within the same seed',
            performance={metric: paired_values(left, right, 'performance', metric) for metric in PERFORMANCE_PATHS},
            distribution={stratum: {metric: paired_values(left, right, 'distribution', metric, stratum)
                                    for metric in DISTRIBUTION_PATHS} for stratum in STRATA})

    gate_summary = {}
    for arm in ARMS:
        values = {str(seed): per_seed[str(seed)]['primary_gates'][arm]['passed'] for seed in SEEDS}
        available = all(value is not None for value in values.values())
        gate_summary[arm] = dict(status='AVAILABLE' if available else 'NOT_AVAILABLE',
            values_by_seed=values, pass_count=sum(value is True for value in values.values()),
            available_count=sum(value is not None for value in values.values()), expected_count=len(SEEDS),
            all_three_seeds_pass=all(values.values()) if available else None)
    complete = all(value['status'] == 'AVAILABLE' for arm in performance_summary.values() for value in arm.values())
    complete = complete and all(value['status'] == 'AVAILABLE' for part in distribution_summary.values()
                                for arm in part.values() for value in arm.values())
    return dict(status='DISTRIBUTION_REPLICATION_SUMMARY_COMPLETE', seeds=list(SEEDS), arms=list(ARMS),
        per_seed=per_seed, summary=dict(performance=performance_summary, distribution=distribution_summary),
        paired_differences=paired, primary_gate_summary=gate_summary, complete_metric_coverage=complete,
        protocol=dict(retrospective_seeds=[SEEDS[0]], prospective_seeds=list(SEEDS[1:]),
            fixed_anchor=True, fixed_seed_count=3, seed_weighting='equal; never target-count weighted',
            summary_unit='seed, not target', targets_per_seed=224, pooled_target_count_used=False,
            sample_standard_deviation_ddof=1, seed_exclusion=False, best_seed_selection=False,
            unavailable_metric_policy='retain all seeds; no zero imputation or partial-seed mean/SD',
            primary_overall_incremental_asr_min=.8, primary_moving_incremental_asr_min=.8,
            first_seed_gate_timing='80% retrospective view; untouched source 85% gate retained',
            new_seed_gate_timing='80% accepted prospectively before new seed plans and outcomes',
            quality_reference_interpretation=QUALITY_REFERENCE_INTERPRETATION,
            source_and_anchor_binding_verification='caller responsibility; aggregate accepts already verified reports'),
        metric_directions=dict(performance={metric: 'lower' if metric == 'adjacent_incremental_fp' else 'higher'
                                           for metric in PERFORMANCE_PATHS},
            distribution={metric: 'lower descriptive distance/rate/norm; not proof of validity'
                          for metric in DISTRIBUTION_PATHS}),
        interpretation=dict(same_reused_original_TRAIN_audit=True, independent_data_confirmation=False,
            statistical_significance_claim=False, independence_assumed=False,
            conditional_distribution_matching_proven=False, physical_validity_claim=False,
            distribution_preservation_claim=False, new_method_claim=False,
            automatic_training=False, automatic_promotion=False))
