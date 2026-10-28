# Source: experiments/spear_distribution_training_v1/reporting.py; mechanically migrated; algorithms and state keys unchanged.
"""Pure three-arm review; no fitting, promotion or holdout access."""
from copy import deepcopy

from .domain_reporting import (
    _validate_report, _difference, _quality_checks, NORMAL_KEYS, REGIMES, ROLES,
    EXPECTED_COUNTS,
)

ARMS = ('continuation', 'unconditional', 'conditional')


def review_evaluations(reports, original_report, projected_report):
    if not isinstance(reports, dict) or set(reports) != set(ARMS):
        raise ValueError('Require all three distribution arms')
    original = _validate_report(original_report)
    projected = _validate_report(projected_report)
    metrics = {arm: _validate_report(reports[arm]) for arm in ARMS}
    for current in [projected, *metrics.values()]:
        clean, old = current['clean_reference'], original['clean_reference']
        if any(clean[k] != old[k] for k in NORMAL_KEYS):
            raise ValueError('Fixed clean-reference normal aggregates changed')
        for regime in REGIMES:
            if any(clean['targets'][regime][k] != old['targets'][regime][k]
                   for k in ('targets', 'clean_negative_targets', 'clean_activation_rate')):
                raise ValueError('Fixed clean-reference clean target aggregates changed')
    comparisons, gates = {}, {}
    for arm in ARMS:
        comparisons[arm] = {
            name: {role: _difference(metrics[arm][role], ref[role]) for role in ROLES}
            for name, ref in (
                ('versus_original_SAGE24', original), ('versus_projected_source', projected),
                ('versus_matched_continuation', metrics['continuation']),
                ('versus_unconditional', metrics['unconditional']))}
        victim = metrics[arm]['victim']
        checks = dict(
            overall_incremental_asr_at_least_85pct=victim['targets']['all']['incremental_asr'] >= .85,
            moving_incremental_asr_at_least_85pct=victim['targets']['moving']['incremental_asr'] >= .85,
            adjacent_incremental_fp_strictly_below_5pct=victim['collateral']['adjacent_incremental_fp'] < .05,
            victim_not_all_negative=not victim['all_clean_predictions_negative'],
            **_quality_checks(victim, original['victim'], 'original'))
        if arm != 'continuation':
            checks.update(_quality_checks(victim, metrics['continuation']['victim'], 'matched_continuation'))
        passed = all(checks.values())
        gates[arm] = dict(passed=passed, checks=checks,
            status='PASS_PREREGISTERED_PILOT_THRESHOLDS' if passed else 'BELOW_PREREGISTERED_PILOT_THRESHOLDS')
    return dict(status='DISTRIBUTION_TRAINING_REVIEW_COMPLETE', metrics=metrics,
        original_metrics=original, projected_source_metrics=projected,
        comparisons=comparisons, gates=gates, population=deepcopy(EXPECTED_COUNTS),
        gate_contract=dict(overall_incremental_asr_min=.85, moving_incremental_asr_min=.85,
            adjacent_incremental_fp_max_exclusive=.05, normal_auc_drop_max=.005,
            normal_positive_recall_drop_max=.05,
            quality_references=dict(continuation=['original_SAGE24'],
                unconditional=['original_SAGE24', 'matched_continuation'],
                conditional=['original_SAGE24', 'matched_continuation'])),
        integrity=dict(raw_csv_identity_and_auc_recomputation='verified independently by caller',
            selected_support_and_zero_out_of_domain='verified independently by caller',
            clean_reference_normal_aggregates_equal=True,
            victim_normal_prediction_equality_required=False),
        interpretation='matched continuation with optional unconditional/conditioned discriminator loss',
        compute='same main training steps, distribution arms add200 discriminator steps each',
        scope='repeatedly exposed original TRAIN development audit, not formal validation/test',
        distribution_preservation_claim=False,
        distribution_loss_interpretation='training diagnostic only; not an independent distribution witness',
        original_val_arrays_loaded=False, original_test_arrays_loaded=False,
        automatic_followup=False, automatic_promotion=False,
        independent_confirmation_claim=False, physical_validity_claim=False, new_method_claim=False)
