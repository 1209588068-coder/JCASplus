# Source: experiments/spear_domain_training_v1/reporting.py; mechanically migrated; algorithms and state keys unchanged.
"""Pure review of matched residual-continuation arms on the reused TRAIN audit.

No I/O, model loading, fitting, checkpoint selection or automatic promotion.
The caller must verify raw CSV identities and independently recompute metrics;
aggregate checks here cannot establish per-row prediction identity or AUC.
The clean reference stays fixed. Freshly fitted victims need NOT have equal
clean predictions, so the frozen-output replay's normal-equality rule does not
apply to them. Domain/write-budget checks are the workflow's responsibility.
"""
from __future__ import annotations

from copy import deepcopy

from .domain_audit_reporting import (
    AUDIT_COUNT_KEYS, EXPECTED_COUNTS, NORMAL_KEYS, REGIMES, ROLES,
    _collateral, _integer, _normal, _require, _targets,
)


ARMS = ('identity', 'domain_projection')
AUC_DROP_MAX = .005
RECALL_DROP_MAX = .05
LABEL_COUNTS = dict(normal_pairs=4399, normal_positive_pairs=489, normal_negative_pairs=3910)
COLLATERAL_COUNTS = dict(adj_negative=1202, nonincident_negative=2484, adj_positive=199)


def _validate_report(report):
    try:
        _require(isinstance(report, dict), 'Expected an evaluation report')
        _require(set(report['audit_counts']) == set(AUDIT_COUNT_KEYS), 'Unexpected audit count schema')
        _require(all(_integer(report['audit_counts'][key], key) == EXPECTED_COUNTS[key]
                     for key in AUDIT_COUNT_KEYS), 'Frozen audit population changed')
        _require(report.get('original_val_arrays_loaded') is False
                 and report.get('original_test_arrays_loaded') is False, 'Formal VAL/TEST must remain unopened')
        _require(set(report['metrics']) == set(ROLES), 'Require both clean_reference and victim')
        result = {}
        for role in ROLES:
            metric = report['metrics'][role]
            normal = _normal(metric, EXPECTED_COUNTS)
            collateral = _collateral(metric)
            _require(all(normal[key] == count for key, count in LABEL_COUNTS.items()), 'Frozen normal labels changed')
            _require(all(collateral[key] == count for key, count in COLLATERAL_COUNTS.items()),
                     'Frozen collateral populations changed')
            result[role] = dict(**normal, targets=_targets(metric, EXPECTED_COUNTS), collateral=collateral)
        return result
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed evaluation metric contract') from error


def _difference(current, reference):
    """Descriptive differences, not paired individual-transition estimates."""
    return dict(
        incremental_asr_delta={regime: current['targets'][regime]['incremental_asr']
            - reference['targets'][regime]['incremental_asr'] for regime in REGIMES},
        clean_activation_delta={regime: current['targets'][regime]['clean_activation_rate']
            - reference['targets'][regime]['clean_activation_rate'] for regime in REGIMES},
        clean_pair_auc_delta=current['clean_pair_auc'] - reference['clean_pair_auc'],
        clean_pair_positive_recall_delta=current['clean_pair_positive_recall'] - reference['clean_pair_positive_recall'],
        collateral_delta={key: current['collateral'][key] - reference['collateral'][key] for key in (
            'adjacent_incremental_fp', 'nonincident_incremental_fp', 'adjacent_positive_suppression')})


def _quality_checks(current, reference, name):
    # Compare against the boundary directly; subtracting two nearly equal
    # measurements first can spuriously reject an exact decimal boundary.
    return {
        name + '_auc_drop_at_most_0p005': current['clean_pair_auc'] >= reference['clean_pair_auc'] - AUC_DROP_MAX,
        name + '_recall_drop_at_most_0p05': current['clean_pair_positive_recall']
            >= reference['clean_pair_positive_recall'] - RECALL_DROP_MAX,
    }


def review_evaluations(reports, original_report):
    """Review both completed arms; performance failures are reported, not raised.

    Population/provenance/inconsistent-metric failures raise ValueError.
    Original SAGE24 and both arms must use the same 256-scene TRAIN audit,
    including 224 targets (164 slow, 60 moving). The reference's normal block
    and clean target counts stay identical. Victim normal scores/counts may
    change because the two victims are fitted independently from matched starts.
    """
    _require(isinstance(reports, dict) and set(reports) == set(ARMS), 'Require exactly both continuation arms')
    original = _validate_report(original_report)
    metrics = {arm: _validate_report(reports[arm]) for arm in ARMS}
    for arm in ARMS:
        clean, old_clean = metrics[arm]['clean_reference'], original['clean_reference']
        _require(all(clean[key] == old_clean[key] for key in NORMAL_KEYS),
                 'Fixed clean-reference normal aggregates changed')
        for regime in REGIMES:
            _require(all(clean['targets'][regime][key] == old_clean['targets'][regime][key]
                         for key in ('targets', 'clean_negative_targets', 'clean_activation_rate')),
                     'Fixed clean-reference clean target aggregates changed')

    comparisons, gates = {}, {}
    for arm in ARMS:
        comparisons[arm] = {
            'versus_original': {role: _difference(metrics[arm][role], original[role]) for role in ROLES},
            'versus_matched_identity': {role: _difference(metrics[arm][role], metrics['identity'][role]) for role in ROLES},
        }
        victim = metrics[arm]['victim']
        checks = dict(
            overall_incremental_asr_at_least_85pct=victim['targets']['all']['incremental_asr'] >= .85,
            moving_incremental_asr_at_least_85pct=victim['targets']['moving']['incremental_asr'] >= .85,
            adjacent_incremental_fp_strictly_below_5pct=victim['collateral']['adjacent_incremental_fp'] < .05,
            victim_not_all_negative=not victim['all_clean_predictions_negative'],
            **_quality_checks(victim, original['victim'], 'original'))
        if arm == 'domain_projection':
            checks.update(_quality_checks(victim, metrics['identity']['victim'], 'matched_identity'))
        passed = all(checks.values())
        gates[arm] = dict(passed=passed, checks=checks,
            status='PASS_PREREGISTERED_PILOT_THRESHOLDS' if passed else 'BELOW_PREREGISTERED_PILOT_THRESHOLDS')

    return dict(status='DOMAIN_TRAINING_REVIEW_COMPLETE', metrics=metrics, original_metrics=original,
        comparisons=comparisons, gates=gates, population=deepcopy(EXPECTED_COUNTS),
        gate_contract=dict(overall_incremental_asr_min=.85, moving_incremental_asr_min=.85,
            adjacent_incremental_fp_max_exclusive=.05, normal_auc_drop_max=AUC_DROP_MAX,
            normal_positive_recall_drop_max=RECALL_DROP_MAX, all_negative_victim_allowed=False,
            quality_references=dict(identity=['original_SAGE24'], domain_projection=['original_SAGE24', 'matched_identity'])),
        integrity=dict(clean_reference_normal_aggregates_equal=True, clean_reference_clean_target_aggregates_equal=True,
            victim_normal_prediction_equality_required=False,
            raw_csv_identity_and_auc_recomputation='must be verified independently by caller',
            selected_support_and_zero_projected_out_of_domain='must be verified independently by caller'),
        scope='reused original-TRAIN development audit; matched residual continuation and fresh victim fitting',
        interpretation='native-clamp constraint-awareness comparison, not a new-method or independent-generalization proof',
        original_val_arrays_loaded=False, original_test_arrays_loaded=False,
        automatic_followup=False, automatic_promotion=False, new_method_claim=False, independent_confirmation_claim=False)
