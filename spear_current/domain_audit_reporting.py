# Source: experiments/spear_domain_audit_v1/reporting.py; mechanically migrated; algorithms and state keys unchanged.
"""Pure metric review of three frozen-output, original-TRAIN audit replays.

The caller must independently verify CSV identities, recompute their metrics,
and check identity replay against the frozen source evidence. This module does
not read CSVs, graphs, models, or checkpoints. It checks the resulting metric
and modification contracts without changing thresholds or choosing a winner.
"""
from __future__ import annotations

from copy import deepcopy
import math


MODES = ('identity', 'domain_projection', 'l1_matched_scale')
ROLES = ('clean_reference', 'victim')
REGIMES = ('all', 'slow', 'moving')
AUDIT_COUNT_KEYS = (
    'registered_scenes', 'processed_scenes', 'scenes_without_eligible_negative_target',
    'all_pair_nodes', 'supervised_pairs', 'targets', 'triggered_supervised_pair_rows',
)
EXPECTED_COUNTS = dict(
    registered_scenes=256, processed_scenes=256, scenes_without_eligible_negative_target=32,
    all_pair_nodes=10511, supervised_pairs=4399, targets=224,
    triggered_supervised_pair_rows=4394, slow_targets=164, moving_targets=60,
)
NORMAL_KEYS = (
    'normal_pairs', 'normal_positive_pairs', 'normal_negative_pairs', 'clean_pair_auc',
    'clean_pair_accuracy', 'clean_pair_positive_recall', 'clean_prediction_distribution',
    'clean_confusion_matrix', 'all_clean_predictions_negative',
    'clean_class_one_probability_min', 'clean_class_one_probability_max',
)
SOURCE_BUDGET_KEYS = (
    'original_outside_coordinate_count', 'original_l1', 'original_linf',
    'projected_l1', 'projected_linf', 'original_changed_coordinate_count',
    'projected_changed_coordinate_count', 'alpha',
)
COUNT_BUDGET_KEYS = (
    'original_outside_coordinate_count', 'original_changed_coordinate_count',
    'projected_changed_coordinate_count', 'actual_changed_coordinate_count',
    'actual_outside_coordinate_count',
)
FLOAT_BUDGET_KEYS = ('original_l1', 'original_linf', 'projected_l1', 'projected_linf',
                     'actual_l1', 'actual_linf', 'alpha')
COORDINATE_BUDGET = 24
L1_MATCH_ATOL = 1e-5
L1_MATCH_RTOL = 1e-6


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, lower=0., upper=None):
    _require(not isinstance(value, bool), f'{name}: boolean is not a numeric measurement')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name}: expected a finite number') from error
    _require(math.isfinite(result) and result >= lower and (upper is None or result <= upper),
             f'{name}: nonfinite or out-of-range measurement')
    return result


def _integer(value, name, lower=0, upper=None):
    result = _number(value, name, lower, upper)
    _require(result.is_integer(), f'{name}: expected an integer count')
    return int(result)


def _close(a, b):
    return math.isclose(a, b, rel_tol=L1_MATCH_RTOL, abs_tol=L1_MATCH_ATOL)


def _rate(value, numerator, denominator, name):
    if denominator == 0:
        _require(numerator == 0 and value is None, f'{name}: zero denominator must be explicit null')
    else:
        number = _number(value, name, upper=1.)
        _require(math.isclose(number, numerator / denominator, rel_tol=0., abs_tol=1e-12),
                 f'{name}: rate does not match recorded counts')


def _contract(expected_counts):
    values = deepcopy(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    _require(isinstance(values, dict) and set(values) == set(EXPECTED_COUNTS), 'Unexpected population contract')
    values = {key: _integer(value, key) for key, value in values.items()}
    for key in ('registered_scenes', 'processed_scenes', 'all_pair_nodes', 'supervised_pairs',
                'targets', 'triggered_supervised_pair_rows', 'slow_targets', 'moving_targets'):
        _require(values[key] > 0, 'Population denominator must be nonempty: ' + key)
    _require(values['registered_scenes'] == values['processed_scenes']
             == values['targets'] + values['scenes_without_eligible_negative_target'],
             'Scene accounting differs from one target per eligible scene')
    _require(values['targets'] == values['slow_targets'] + values['moving_targets'], 'Strata do not partition targets')
    _require(values['targets'] <= values['triggered_supervised_pair_rows']
             <= values['supervised_pairs'] <= values['all_pair_nodes'], 'Invalid nested pair populations')
    return values


def _normal(metric, counts):
    result = {key: deepcopy(metric[key]) for key in NORMAL_KEYS}
    total = _integer(result['normal_pairs'], 'normal_pairs', 1)
    pos = _integer(result['normal_positive_pairs'], 'normal_positive_pairs', 1)
    neg = _integer(result['normal_negative_pairs'], 'normal_negative_pairs', 1)
    _require(total == counts['supervised_pairs'] == pos + neg, 'Normal label population changed')
    confusion = result['clean_confusion_matrix']
    _require(set(confusion) == {'true_negative', 'false_positive', 'false_negative', 'true_positive'},
             'Unexpected confusion-matrix fields')
    tn, fp, fn, tp = [_integer(confusion[key], key) for key in
                      ('true_negative', 'false_positive', 'false_negative', 'true_positive')]
    _require(tn + fp == neg and tp + fn == pos, 'Confusion matrix has different label denominators')
    distribution = result['clean_prediction_distribution']
    _require(set(distribution) == {'class_0', 'class_1'}, 'Unexpected clean prediction distribution')
    _require(_integer(distribution['class_0'], 'class_0') == tn + fn
             and _integer(distribution['class_1'], 'class_1') == tp + fp, 'Clean prediction counts disagree')
    _rate(result['clean_pair_accuracy'], tn + tp, total, 'clean accuracy')
    _rate(result['clean_pair_positive_recall'], tp, pos, 'clean recall')
    _number(result['clean_pair_auc'], 'clean AUC', upper=1.)
    low = _number(result['clean_class_one_probability_min'], 'minimum clean probability', upper=1.)
    high = _number(result['clean_class_one_probability_max'], 'maximum clean probability', upper=1.)
    _require(low <= high, 'Clean probability extrema reversed')
    _require(type(result['all_clean_predictions_negative']) is bool
             and result['all_clean_predictions_negative'] == (tp + fp == 0), 'Clean prediction collapse flag differs')
    return result


def _targets(metric, counts):
    targets = metric['targets']
    _require(set(targets) == set(REGIMES), 'Missing target motion strata')
    counter_keys = ('targets', 'triggered_target_class_count', 'incremental_flips', 'clean_negative_targets')
    validated = {}
    for regime in REGIMES:
        row = targets[regime]
        n = counts['targets' if regime == 'all' else regime + '_targets']
        _require(_integer(row['targets'], regime + ' targets', 1) == n, 'Target stratum denominator changed')
        triggered = _integer(row['triggered_target_class_count'], 'triggered count', upper=n)
        flips = _integer(row['incremental_flips'], 'incremental flips', upper=n)
        clean_negative = _integer(row['clean_negative_targets'], 'clean negative targets', upper=n)
        _require(flips <= min(triggered, clean_negative)
                 and triggered - flips <= n - clean_negative, 'Inconsistent target transition counts')
        _rate(row['absolute_asr'], triggered, n, 'absolute ASR')
        _rate(row['incremental_asr'], flips, n, 'incremental ASR')
        _rate(row['conditional_flip'], flips, clean_negative, 'conditional flip')
        _rate(row['clean_activation_rate'], n - clean_negative, n, 'clean activation')
        _number(row['target_probability_delta_mean'], 'target probability delta', lower=-1., upper=1.)
        validated[regime] = deepcopy(row)
    for key in counter_keys:
        _require(validated['all'][key] == validated['slow'][key] + validated['moving'][key],
                 'Target strata counts do not sum to all: ' + key)
    mean = (validated['slow']['target_probability_delta_mean'] * counts['slow_targets']
            + validated['moving']['target_probability_delta_mean'] * counts['moving_targets']) / counts['targets']
    _require(math.isclose(validated['all']['target_probability_delta_mean'], mean, rel_tol=0., abs_tol=1e-12),
             'Pooled probability delta is not target-weighted')
    return validated


def _collateral(metric):
    row = metric['collateral']
    pairs = (('adj_negative', 'adj_flip', 'adjacent_incremental_fp'),
             ('nonincident_negative', 'nonincident_flip', 'nonincident_incremental_fp'),
             ('adj_positive', 'adj_suppressed', 'adjacent_positive_suppression'))
    for denominator, numerator, rate in pairs:
        n = _integer(row[denominator], denominator, 1)
        changed = _integer(row[numerator], numerator, upper=n)
        _rate(row[rate], changed, n, rate)
    return deepcopy(row)


def _budgets(budgets, counts):
    _require(isinstance(budgets, dict) and set(budgets) == set(MODES), 'Require all three budget modes')
    by_mode = {}
    for mode in MODES:
        rows, lookup = budgets[mode], {}
        _require(len(rows) == counts['targets'], 'Modification target denominator changed')
        for row in rows:
            scene = row['scenario_id']
            _require(isinstance(scene, str) and bool(scene) and scene not in lookup, 'Duplicate or empty budget scene ID')
            _require(row['motion_regime'] in ('slow', 'moving'), 'Unknown modification motion regime')
            item = dict(scenario_id=scene, motion_regime=row['motion_regime'])
            item.update({key: _integer(row[key], key, upper=COORDINATE_BUDGET) for key in COUNT_BUDGET_KEYS})
            item.update({key: _number(row[key], key, upper=1. if key == 'alpha' else None) for key in FLOAT_BUDGET_KEYS})
            for prefix in ('original', 'projected', 'actual'):
                l1, linf, changed = (item[prefix + suffix] for suffix in
                                     ('_l1', '_linf', '_changed_coordinate_count'))
                _require(linf <= l1 or _close(linf, l1), 'L-infinity exceeds L1')
                _require((changed == 0) == (l1 == 0) == (linf == 0), 'Modification counts disagree with zero norms')
                _require(l1 <= changed * linf or _close(l1, changed * linf), 'L1 exceeds count times L-infinity')
            _require(item['original_outside_coordinate_count'] <= item['original_changed_coordinate_count']
                     and item['actual_outside_coordinate_count'] <= item['actual_changed_coordinate_count'],
                     'Out-of-domain values must differ from in-domain clean values')
            _require(item['projected_changed_coordinate_count'] <= item['original_changed_coordinate_count']
                     and item['actual_changed_coordinate_count'] <= item['original_changed_coordinate_count'],
                     'Projection/scaling cannot change an originally unchanged coordinate')
            for norm in ('l1', 'linf'):
                _require(item['projected_' + norm] <= item['original_' + norm]
                         or _close(item['projected_' + norm], item['original_' + norm]), 'Projection expanded displacement')
            ratio = item['projected_l1'] / item['original_l1'] if item['original_l1'] else 0.
            _require(math.isclose(item['alpha'], ratio, rel_tol=0., abs_tol=1e-12), 'Alpha differs from projected/original L1')
            if mode == 'identity':
                for actual, original in (('actual_l1', 'original_l1'), ('actual_linf', 'original_linf'),
                                         ('actual_changed_coordinate_count', 'original_changed_coordinate_count'),
                                         ('actual_outside_coordinate_count', 'original_outside_coordinate_count')):
                    _require(item[actual] == item[original], 'Identity modification budget is not exact')
            elif mode == 'domain_projection':
                _require(item['actual_outside_coordinate_count'] == 0, 'Projected output remains outside clean domain')
                _require(item['actual_changed_coordinate_count'] == item['projected_changed_coordinate_count'],
                         'Projection changed-count mismatch')
                for norm in ('l1', 'linf'):
                    _require(_close(item['actual_' + norm], item['projected_' + norm]), 'Projection norm mismatch')
            else:
                _require(_close(item['actual_l1'], item['projected_l1']), 'Radial control does not match projected L1')
                _require(_close(item['actual_linf'], item['alpha'] * item['original_linf']),
                         'Radial control maximum displacement does not match its alpha')
            lookup[scene] = item
        for regime in ('slow', 'moving'):
            _require(sum(row['motion_regime'] == regime for row in lookup.values()) == counts[regime + '_targets'],
                     'Modification stratum denominator changed')
        by_mode[mode] = lookup
    identity = by_mode['identity']
    for mode in MODES[1:]:
        _require(set(by_mode[mode]) == set(identity), 'Different scene IDs across modification modes')
        for scene, row in by_mode[mode].items():
            source = identity[scene]
            _require(row['motion_regime'] == source['motion_regime'], 'Target motion regime changed across modes')
            _require(all(row[key] == source[key] for key in SOURCE_BUDGET_KEYS),
                     'Original generated values/projected budget changed across modes')
    return by_mode


def _modification_summary(rows):
    n = len(rows)
    _require(n > 0, 'Empty modification population')
    summary = dict(targets=n,
        targets_outside_domain=sum(row['actual_outside_coordinate_count'] > 0 for row in rows),
        coordinates_outside_domain=sum(row['actual_outside_coordinate_count'] for row in rows),
        actual_changed_coordinates_total=sum(row['actual_changed_coordinate_count'] for row in rows),
        actual_changed_coordinates_mean=math.fsum(row['actual_changed_coordinate_count'] for row in rows) / n,
        interpretation='standardized feature-space changes; not physical displacement')
    for key in FLOAT_BUDGET_KEYS:
        values = [row[key] for row in rows]
        summary[key] = dict(mean=math.fsum(values) / n, minimum=min(values), maximum=max(values))
    return summary


def review_reports(reports, budgets, *, expected_counts=None):
    """Validate and describe fixed-model responses; never promote or retrain.

    ``reports`` maps each mode to an unchanged supplied-evaluator report with
    both clean_reference and victim metric blocks. ``budgets`` maps modes to
    one scalar diagnostic row per target scene, containing scenario_id,
    motion_regime, transform_selected statistics, and actual_outside_coordinate_count.

    ``expected_counts`` is a complete alternative population contract for
    synthetic tests only; production callers use the frozen default contract.
    Raw CSV identity/prediction validation and paired transition counts remain
    the caller's responsibility. Equal aggregate metrics alone do not prove
    equal per-row predictions; this report says exactly what it checked.
    """
    counts = _contract(expected_counts)
    _require(isinstance(reports, dict) and set(reports) == set(MODES), 'Require exactly three frozen-output reports')
    metrics, normal = {}, {}
    for mode in MODES:
        report = reports[mode]
        _require(set(report['audit_counts']) == set(AUDIT_COUNT_KEYS), 'Unexpected audit-count schema')
        _require(all(_integer(report['audit_counts'][key], key) == counts[key] for key in AUDIT_COUNT_KEYS),
                 'Frozen audit population changed')
        _require(set(report['metrics']) == set(ROLES), 'Require clean_reference and victim metrics')
        _require(report.get('original_val_arrays_loaded') is False and report.get('original_test_arrays_loaded') is False,
                 'Original VAL/TEST must remain unopened')
        metrics[mode], normal[mode] = {}, {}
        for role in ROLES:
            metric = report['metrics'][role]
            normal[mode][role] = _normal(metric, counts)
            metrics[mode][role] = dict(**normal[mode][role], targets=_targets(metric, counts),
                                       collateral=_collateral(metric))
    for mode in MODES[1:]:
        for role in ROLES:
            _require(normal[mode][role] == normal['identity'][role], 'Clean normal metric block changed across modes')
            for regime in REGIMES:
                base = metrics['identity'][role]['targets'][regime]
                current = metrics[mode][role]['targets'][regime]
                _require(current['clean_negative_targets'] == base['clean_negative_targets']
                         and current['clean_activation_rate'] == base['clean_activation_rate'],
                         'Clean target prediction counts changed across modes')
            for key in ('adj_negative', 'nonincident_negative', 'adj_positive'):
                _require(metrics[mode][role]['collateral'][key] == metrics['identity'][role]['collateral'][key],
                         'Collateral source population changed across modes')
    for mode in MODES:
        for key in ('normal_pairs', 'normal_positive_pairs', 'normal_negative_pairs'):
            _require(normal[mode]['clean_reference'][key] == normal[mode]['victim'][key], 'Models scored different normal populations')
        for key in ('adj_negative', 'nonincident_negative', 'adj_positive'):
            _require(metrics[mode]['clean_reference']['collateral'][key] == metrics[mode]['victim']['collateral'][key],
                     'Models scored different collateral populations')
    by_mode = _budgets(budgets, counts)
    modification, comparison, gates = {}, {}, {}
    for mode in MODES:
        rows = list(by_mode[mode].values())
        modification[mode] = {regime: _modification_summary([
            row for row in rows if regime == 'all' or row['motion_regime'] == regime]) for regime in REGIMES}
        comparison[mode] = {}
        for role in ROLES:
            base, current = metrics['identity'][role], metrics[mode][role]
            comparison[mode][role] = dict(
                incremental_asr_delta={regime: current['targets'][regime]['incremental_asr']
                    - base['targets'][regime]['incremental_asr'] for regime in REGIMES},
                adjacent_incremental_fp_delta=current['collateral']['adjacent_incremental_fp']
                    - base['collateral']['adjacent_incremental_fp'],
                normal_metrics_exactly_equal=True,
                interpretation='same frozen model and clean inputs; clean AUC is unchanged, not improved')
        victim = metrics[mode]['victim']
        checks = dict(overall_incremental_asr_at_least_85pct=victim['targets']['all']['incremental_asr'] >= .85,
                      moving_incremental_asr_at_least_85pct=victim['targets']['moving']['incremental_asr'] >= .85,
                      adjacent_incremental_fp_strictly_below_5pct=victim['collateral']['adjacent_incremental_fp'] < .05,
                      normal_metrics_unchanged=True)
        passed = all(checks.values())
        gates[mode] = dict(passed=passed, checks=checks,
            status='PASS_FIXED_MODEL_REPLAY_THRESHOLDS' if passed else 'BELOW_FIXED_MODEL_REPLAY_THRESHOLDS')
    return dict(status='FROZEN_DOMAIN_AUDIT_REVIEW_COMPLETE', metrics=metrics, comparison=comparison,
        gates=gates, modification_statistics=modification, population=counts,
        gate_contract=dict(overall_incremental_asr_min=.85, moving_incremental_asr_min=.85,
                           adjacent_incremental_fp_max_exclusive=.05, normal_metrics='exactly unchanged'),
        integrity=dict(normal_metric_blocks_exactly_equal=True, source_budget_scene_ids_identical=True,
                       source_budget_values_identical=True,
                       raw_csv_and_prediction_identity='must be independently verified by caller'),
        selected_coordinate_budget=COORDINATE_BUDGET,
        l1_control_tolerance=dict(atol=L1_MATCH_ATOL, rtol=L1_MATCH_RTOL),
        scope='existing original-TRAIN audit; fixed generator and victim; no optimization or retraining',
        interpretation='diagnostic response to output transforms, not new trained-method performance',
        original_val_arrays_loaded=False, original_test_arrays_loaded=False,
        optimizer_steps=0, automatic_followup=False, automatic_promotion=False, new_method_claim=False)
