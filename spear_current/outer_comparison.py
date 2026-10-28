"""Pure preregistered comparison for the offline AV2/self-owned GNN pilot.

Only development-summary dictionaries are consumed. This module loads no data
or models, starts no jobs and makes no statistical-significance claims.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real


_MODELS = ('surrogate', 'victim', 'clean_reference')
_STRATA = ('all', 'slow', 'moving')
_IDENTITIES = ('scenes', 'eligible_clean_pairs', 'targets', 'triggered_pairs')
_COUNT_FIELDS = ('registered_scenes', 'processed_scenes',
    'scenes_without_eligible_negative_target', 'all_pair_nodes', 'supervised_pairs',
    'targets', 'triggered_supervised_pair_rows')
GATES = dict(
    overall_incremental_asr_min=.80,
    moving_incremental_asr_min=.80,
    adjacent_incremental_fp_max_exclusive=.05,
    clean_prediction_classes_required=[0, 1],
    clean_auc_max_drop_vs_baseline=.005,
    clean_auc_max_drop_vs_reference=.005,
    positive_recall_max_drop_vs_baseline=.05,
    positive_recall_max_drop_vs_reference=.05,
    moving_incremental_count_gain_min=1,
    overall_incremental_count_gain_min=0)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, name):
    _require(isinstance(value, Integral) and not isinstance(value, bool) and value >= 0,
             name + ' must be a nonnegative integer count')
    return int(value)


def _probability(value, name):
    if value is None:
        return None
    _require(isinstance(value, Real) and not isinstance(value, bool)
             and math.isfinite(value) and 0 <= value <= 1,
             name + ' must be a finite probability or null')
    return float(value)


def _rate(value, numerator, denominator, name):
    _require(numerator <= denominator, name + ' numerator exceeds denominator')
    actual = _probability(value, name)
    expected = numerator / denominator if denominator else None
    _require((actual is None and expected is None) or
             (actual is not None and expected is not None
              and math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12)),
             name + ' does not match its numerator/denominator')
    return dict(numerator=numerator, denominator=denominator, value=expected)


def _normal(metric):
    total, positive, negative = (
        _count(metric[key], key)
        for key in ('normal_pairs', 'normal_positive_pairs', 'normal_negative_pairs'))
    _require(total == positive + negative, 'Clean label counts do not sum to the population')
    confusion = {key: _count(metric['clean_confusion_matrix'][key], key)
                 for key in ('true_negative', 'false_positive', 'false_negative', 'true_positive')}
    tn, fp, fn, tp = (confusion[key] for key in
        ('true_negative', 'false_positive', 'false_negative', 'true_positive'))
    _require(tn + fp == negative and fn + tp == positive,
             'Clean confusion matrix has different label denominators')
    _require(set(metric['clean_prediction_distribution']) == {'class_0', 'class_1'},
             'Require exactly two clean prediction classes')
    predictions = {key: _count(metric['clean_prediction_distribution'][key], key)
                   for key in ('class_0', 'class_1')}
    _require(predictions == dict(class_0=tn + fn, class_1=fp + tp),
             'Clean binary prediction counts disagree with the confusion matrix')
    recall = _rate(metric['clean_pair_positive_recall'], tp, positive, 'Positive recall')
    auc = _probability(metric['clean_pair_auc'], 'Clean AUC')
    _require(auc is None or (positive > 0 and negative > 0),
             'Clean AUC is undefined without both true-label classes')
    return dict(count=total, positive_count=positive, negative_count=negative,
                confusion=confusion, prediction_counts=predictions,
                auc=auc, positive_recall=recall)


def _model(metric):
    normal, targets = _normal(metric), {}
    for stratum in _STRATA:
        row = metric['targets'][stratum]
        count = _count(row['targets'], stratum + ' targets')
        flips = _count(row['incremental_flips'], stratum + ' incremental flips')
        targets[stratum] = _rate(row['incremental_asr'], flips, count,
                                  stratum + ' incremental ASR')
    for field in ('numerator', 'denominator'):
        _require(targets['all'][field] == targets['slow'][field] + targets['moving'][field],
                 'Target strata do not sum to the overall ' + field)
    collateral = metric['collateral']
    denominators = {key: _count(collateral[key], key)
                    for key in ('adj_negative', 'nonincident_negative', 'adj_positive')}
    adjacent = _rate(collateral['adjacent_incremental_fp'],
                     _count(collateral['adj_flip'], 'adj_flip'),
                     denominators['adj_negative'], 'Adjacent incremental FP')
    return dict(clean=normal, incremental_asr=targets,
                adjacent_incremental_fp=adjacent, collateral_denominators=denominators)


def _summary(summary):
    _require(isinstance(summary, Mapping), 'Evaluation summary must be a mapping')
    _require(summary['status'] == 'DEV_EVALUATION_COMPLETE', 'Development evaluation is incomplete')
    identities = summary['population_identity_sha256']
    _require(isinstance(identities, Mapping) and set(identities) == set(_IDENTITIES),
             'Require every prediction-independent population identity')
    for name, digest in identities.items():
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(char in '0123456789abcdef' for char in digest),
                 'Invalid population SHA-256: ' + name)
    if 'target_identity_sha256' in summary:
        _require(summary['target_identity_sha256'] == identities['targets'],
                 'Target identity alias disagrees with the population identity')
    _require(set(summary['dev_counts']) == set(_COUNT_FIELDS), 'Unexpected development count fields')
    counts = {key: _count(summary['dev_counts'][key], key) for key in _COUNT_FIELDS}
    _require(counts['registered_scenes'] == counts['processed_scenes']
             == counts['targets'] + counts['scenes_without_eligible_negative_target'],
             'Development scene counts are incomplete or inconsistent')
    _require(counts['targets'] <= counts['triggered_supervised_pair_rows']
             <= counts['supervised_pairs'] <= counts['all_pair_nodes'],
             'Development pair population counts are inconsistent')
    models = {name: _model(summary['metrics'][name]) for name in _MODELS}
    populations = None
    for name, metric in models.items():
        normal = metric['clean']
        _require(normal['count'] == counts['supervised_pairs'],
                 'Clean population count differs from development counts: ' + name)
        _require(metric['incremental_asr']['all']['denominator'] == counts['targets'],
                 'Target population count differs from development counts: ' + name)
        current = dict(clean_pairs={key: normal[key]
            for key in ('count', 'positive_count', 'negative_count')},
            targets={key: metric['incremental_asr'][key]['denominator'] for key in _STRATA},
            collateral=metric['collateral_denominators'])
        if populations is None:
            populations = current
        _require(populations == current, 'Model summaries use different evaluation pool counts')
    for key, expected in (('raw_targets', counts['targets']),
                          ('raw_clean_scores', counts['supervised_pairs']),
                          ('raw_triggered_pair_scores', counts['triggered_supervised_pair_rows'])):
        if key in summary:
            _require(_count(summary[key]['rows'], key + ' rows') == expected,
                     'Recorded CSV row count differs from evaluation pool: ' + key)
    return dict(identities=dict(identities), counts=dict(dev=counts, **populations), models=models)


def _rate_gate(measurement, threshold, operator):
    value = measurement['value']
    passed = value is not None and (value >= threshold if operator == '>=' else value < threshold)
    return dict(**deepcopy(measurement), operator=operator, threshold=threshold,
                passed=bool(passed), reason='defined' if value is not None else 'undefined_denominator')


def _quality_gate(candidate, reference, tolerance, *, metric):
    if metric == 'auc':
        actual, original = candidate['auc'], reference['auc']
        counts = dict(denominator=candidate['count'],
                      positive_count=candidate['positive_count'],
                      negative_count=candidate['negative_count'],
                      pairwise_denominator=candidate['positive_count'] * candidate['negative_count'])
    else:
        actual, original = candidate['positive_recall']['value'], reference['positive_recall']['value']
        counts = dict(numerator=candidate['positive_recall']['numerator'],
                      denominator=candidate['positive_recall']['denominator'],
                      reference_numerator=reference['positive_recall']['numerator'],
                      reference_denominator=reference['positive_recall']['denominator'])
    threshold = original - tolerance if original is not None else None
    defined = actual is not None and threshold is not None
    return dict(value=actual, reference_value=original, operator='>=', threshold=threshold,
                permitted_drop=tolerance, passed=bool(defined and actual >= threshold),
                reason='defined' if defined else 'undefined_metric', **counts)


def compare_dev_summaries(baseline, candidate):
    """Review one fixed pilot; performance never authorizes further jobs.

    Inputs are ``outer_evaluation.evaluate_dev`` results. Population hashes and
    every relevant denominator must match exactly. Counts validate reported
    rates; zero denominators remain undefined and fail the affected gates.
    The clean reference's normal metrics must be identical across arms.

    Acceptable performance and exploratory count improvement are separate.
    A recommendation requires both, but status always requires human review.
    """
    try:
        base, changed = _summary(baseline), _summary(candidate)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Missing or malformed development comparison fields') from error
    _require(base['identities'] == changed['identities'],
             'Baseline and candidate evaluation pool identities differ')
    _require(base['counts'] == changed['counts'],
             'Baseline and candidate evaluation pool counts differ')
    _require(base['models']['clean_reference']['clean'] == changed['models']['clean_reference']['clean'],
             'Frozen clean-reference normal evidence changed between arms')
    original = base['models']['victim']
    victim = changed['models']['victim']
    clean, original_clean = victim['clean'], original['clean']
    reference = changed['models']['clean_reference']['clean']
    predictions = clean['prediction_counts']
    gates = dict(
        overall_incremental_asr=_rate_gate(victim['incremental_asr']['all'], GATES['overall_incremental_asr_min'], '>='),
        moving_incremental_asr=_rate_gate(victim['incremental_asr']['moving'], GATES['moving_incremental_asr_min'], '>='),
        adjacent_incremental_fp=_rate_gate(victim['adjacent_incremental_fp'], GATES['adjacent_incremental_fp_max_exclusive'], '<'),
        noncollapsed_clean_predictions=dict(class_0=predictions['class_0'], class_1=predictions['class_1'],
            denominator=clean['count'], rule='both binary prediction classes have positive counts',
            passed=predictions['class_0'] > 0 and predictions['class_1'] > 0),
        clean_auc_vs_baseline=_quality_gate(clean, original_clean, GATES['clean_auc_max_drop_vs_baseline'], metric='auc'),
        clean_auc_vs_reference=_quality_gate(clean, reference, GATES['clean_auc_max_drop_vs_reference'], metric='auc'),
        positive_recall_vs_baseline=_quality_gate(clean, original_clean, GATES['positive_recall_max_drop_vs_baseline'], metric='recall'),
        positive_recall_vs_reference=_quality_gate(clean, reference, GATES['positive_recall_max_drop_vs_reference'], metric='recall'))
    gains = {}
    for stratum in _STRATA:
        left, right = original['incremental_asr'][stratum], victim['incremental_asr'][stratum]
        gains[stratum] = dict(baseline_incremental_count=left['numerator'],
            candidate_incremental_count=right['numerator'], denominator=right['denominator'],
            incremental_count_gain=right['numerator'] - left['numerator'],
            baseline_incremental_asr=left['value'], candidate_incremental_asr=right['value'])
    improvement_gates = dict(
        moving_gain_at_least_one=dict(**gains['moving'], operator='>=', threshold=GATES['moving_incremental_count_gain_min'],
            passed=gains['moving']['incremental_count_gain'] >= GATES['moving_incremental_count_gain_min']),
        overall_not_worse=dict(**gains['all'], operator='>=', threshold=GATES['overall_incremental_count_gain_min'],
            passed=gains['all']['incremental_count_gain'] >= GATES['overall_incremental_count_gain_min']))
    acceptable = all(gate['passed'] for gate in gates.values())
    improvement = all(gate['passed'] for gate in improvement_gates.values())
    return dict(status='PILOT_COMPLETE_REVIEW_REQUIRED',
        scope='single fixed original-TRAIN DEV pilot; development evidence only',
        comparison_model='victim', preregistered_gates=deepcopy(GATES),
        population_identity_sha256=deepcopy(base['identities']),
        population_counts=deepcopy(base['counts']), pool_identity_and_counts_match=True,
        measurements=dict(baseline=deepcopy(original), candidate=deepcopy(victim),
                          clean_reference=deepcopy(reference)),
        performance_gates=gates, acceptable_performance=acceptable,
        exploratory_improvement_gates=improvement_gates, incremental_changes=gains,
        exploratory_improvement=improvement,
        recommend_follow_up_review=acceptable and improvement,
        automatic_jobs_started=False, statistical_significance_claimed=False,
        multiseed_evidence=False,
        interpretation='Exploratory Moving count gain only; thresholds do not establish statistical significance.')
