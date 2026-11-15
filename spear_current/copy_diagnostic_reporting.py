"""Read-only paired DEV reporting for six within-pair copy equalities.

Only the five named JSON/CSV files in each supplied arm directory are read.
No graph, checkpoint, prediction, optimization, output write, performance gate,
or automatic follow-up is part of this module. AP and nonincident-positive
suppression are additional diagnostics, not metrics supplied by evaluate_dev.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .support_schema import SELECTED_COORDINATES


MODELS = ('surrogate', 'victim', 'clean_reference')
STRATA = ('all', 'slow', 'moving')
ARMS = ('identity', 'pair_mean')
ALLOWED_COORDINATES = (776, 743, 778, 708, 709, 742, 777, 745, 744, 779,
    351, 782, 350, 729, 132, 712, 781, 353, 749, 746, 134, 724, 135, 760)
COPY_GROUPS = ((350, 742, 777), (708, 743, 776), (351, 744, 779),
               (709, 745, 778), (353, 746, 781), (712, 749, 782))
COPY_NAMES = ('A.speed_current', 'B.speed_current', 'A.speed_mean',
              'B.speed_mean', 'A.acc_std', 'B.lateral_acc_std')
COPY_POSITIONS = tuple(tuple(SELECTED_COORDINATES.index(c) for c in g) for g in COPY_GROUPS)
ACTIVE_POSITIONS = tuple(SELECTED_COORDINATES.index(c) for c in ALLOWED_COORDINATES)
METADATA = ('scenario_id', 'split_group_id', 'original_split', 'role', 'pair_index',
    'role_pair_index', 'src_track_id', 'dst_track_id', 'true_label', 'eligible',
    'motion_regime', 'is_attachment', 'is_outer')
RESPONSES = tuple(name + suffix for name in MODELS for suffix in
    ('_clean_p', '_trigger_p', '_clean_pred', '_trigger_pred', '_target_nll'))
FEATURE_FIELDS = ('changed_coordinates', 'boundary_coordinates', 'delta_l1', 'delta_linf')
HEADERS = dict(targets=METADATA + FEATURE_FIELDS + RESPONSES,
    normal=METADATA + ('label',) + tuple(name + suffix for name in MODELS for suffix in ('_p', '_pred')),
    triggered=METADATA + ('target_pair_index', 'label', 'relation') + RESPONSES)
FILES = dict(targets=('targets.csv', 'raw_targets'),
    normal=('clean_pair_scores.csv', 'raw_clean_scores'),
    triggered=('triggered_pair_scores.csv', 'raw_triggered_pair_scores'))
COUNT_FIELDS = ('registered_scenes', 'processed_scenes', 'scenes_without_eligible_negative_target',
                'all_pair_nodes', 'supervised_pairs', 'targets', 'triggered_supervised_pair_rows')
POPULATIONS = ('scenes', 'eligible_clean_pairs', 'targets', 'triggered_pairs')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in '0123456789abcdef' for c in value), 'Malformed SHA-256')
    return value


def _count(value):
    _require(type(value) is int and value >= 0, 'Require a nonnegative integer count')
    return value


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Require a finite numeric value')
    return float(value)


def _same(actual, expected, path='metric'):
    """Compare complete expected metric fields, retaining null and integer types."""
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and expected.keys() <= actual.keys(), 'Missing ' + path)
        for key, value in expected.items():
            _same(actual[key], value, path + '.' + key)
    elif expected is None or type(expected) in (bool, int, str):
        _require(type(actual) is type(expected) and actual == expected, 'Inconsistent ' + path)
    else:
        _require(type(actual) in (int, float) and math.isfinite(actual)
                 and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), 'Inconsistent ' + path)


def _decode_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _integer_text(value):
    _require(isinstance(value, str) and value.isascii() and value.isdecimal(), 'Invalid CSV integer')
    return int(value)


def _parse_csv(data, kind):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    _require(reader.fieldnames == list(HEADERS[kind]), 'Unexpected CSV schema: ' + kind)
    rows = []
    for raw in reader:
        _require(None not in raw and None not in raw.values(), 'Malformed CSV row')
        row = dict(raw)
        for key in ('pair_index', 'role_pair_index', 'true_label'):
            row[key] = _integer_text(raw[key])
        for key in ('eligible', 'is_attachment', 'is_outer'):
            _require(raw[key] in ('True', 'False'), 'Invalid CSV boolean')
            row[key] = raw[key] == 'True'
        _require(row['original_split'] == 'train' and row['role'] == 'dev'
                 and row['eligible'] and not row['is_attachment'] and not row['is_outer']
                 and row['true_label'] in (0, 1) and row['motion_regime'] in STRATA[1:],
                 'Invalid original-TRAIN DEV metadata')
        for key in ('scenario_id', 'split_group_id', 'src_track_id', 'dst_track_id'):
            _require(bool(row[key].strip()), 'Empty pair identity')
        _require(row['src_track_id'] != row['dst_track_id'], 'Self pair is invalid')
        if kind != 'targets':
            row['label'] = _integer_text(raw['label'])
            _require(row['label'] == row['true_label'], 'CSV labels disagree')
        else:
            _require(row['true_label'] == 0, 'Target must be an eligible true negative')
            for key in FEATURE_FIELDS:
                row[key] = _integer_text(raw[key]) if key.endswith('coordinates') else float(raw[key])
                _require(math.isfinite(row[key]) and row[key] >= 0, 'Invalid feature-change measurement')
        if kind == 'triggered':
            row['target_pair_index'] = _integer_text(raw['target_pair_index'])
            _require(row['relation'] in ('target', 'adjacent', 'nonincident'), 'Invalid recorded relation')
        for model in MODELS:
            suffixes = ('_p', '_pred') if kind == 'normal' else (
                '_clean_p', '_trigger_p', '_clean_pred', '_trigger_pred', '_target_nll')
            for suffix in suffixes:
                key = model + suffix
                if suffix.endswith('pred'):
                    row[key] = _integer_text(raw[key])
                    _require(row[key] in (0, 1), 'Invalid binary prediction')
                else:
                    row[key] = float(raw[key])
                    _require(math.isfinite(row[key]) and row[key] >= 0
                             and (suffix == '_target_nll' or row[key] <= 1), 'Invalid probability/NLL')
        rows.append(row)
    return rows


def _metadata(row):
    return {key: row[key] for key in METADATA}


def _key(row):
    return row['scenario_id'], row['split_group_id'], row['pair_index']


def _identity_hash(value):
    """Reproduce evaluate_dev's typed metadata-only fingerprint, without tensors."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            digest.update(b'mapping{')
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b'}')
        elif isinstance(item, list):
            digest.update(b'list[')
            for child in item:
                visit(child)
            digest.update(b']')
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b';')
    visit(value)
    return digest.hexdigest()


def _bind_rows(rows):
    normal, targets, triggered = (rows[key] for key in ('normal', 'targets', 'triggered'))
    clean = {_key(row): row for row in normal}
    _require(len(clean) == len(normal), 'Duplicate clean pair')
    _require(len({r['role_pair_index'] for r in normal}) == len(normal), 'Duplicate DEV role-pair index')
    scenes, groups = {}, {}
    for row in normal:
        scenario, group = row['scenario_id'], row['split_group_id']
        _require(scenes.setdefault(scenario, group) == group and groups.setdefault(group, scenario) == scenario,
                 'Scene/group identities are not unique')
    chosen = {_key(row): row for row in targets}
    _require(len(chosen) == len(targets)
             and len({row['scenario_id'] for row in targets}) == len(targets), 'Duplicate target scene/pair')
    target_by_scene = {row['scenario_id']: row for row in targets}
    _require(set(target_by_scene) == {r['scenario_id'] for r in normal if r['label'] == 0},
             'A clean scene with an eligible negative is missing its target')
    for row in targets:
        before = clean.get(_key(row))
        _require(before is not None and _metadata(before) == _metadata(row), 'Target identity differs from clean CSV')
        for model in MODELS:
            for suffix in ('p', 'pred'):
                _require(row[f'{model}_clean_{suffix}'] == before[f'{model}_{suffix}'], 'Target clean response changed')
    expected_order = [_key(row) for target in targets for row in normal
                      if row['scenario_id'] == target['scenario_id']]
    _require([_key(row) for row in triggered] == expected_order,
             'Triggered population has missing, repeated, reordered or extra pairs')
    for row in triggered:
        before = clean[_key(row)]
        target = target_by_scene[row['scenario_id']]
        _require(_metadata(row) == _metadata(before) and row['target_pair_index'] == target['pair_index'],
                 'Triggered identity/target binding changed')
        is_target = _key(row) == _key(target)
        _require((row['relation'] == 'target') == is_target, 'Target relation is inconsistent')
        for model in MODELS:
            for suffix in ('p', 'pred'):
                _require(row[f'{model}_clean_{suffix}'] == before[f'{model}_{suffix}'], 'Triggered clean response changed')
            if is_target:
                _require(all(row[model + suffix] == target[model + suffix] for suffix in
                    ('_clean_p', '_trigger_p', '_clean_pred', '_trigger_pred', '_target_nll')),
                    'Target CSV and triggered target responses differ')
    return target_by_scene


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _targets(rows, model):
    result = {}
    for stratum in STRATA:
        selected = [r for r in rows if stratum == 'all' or r['motion_regime'] == stratum]
        n = len(selected)
        clean_negative = sum(r[model + '_clean_pred'] == 0 for r in selected)
        flips = sum(r[model + '_clean_pred'] == 0 and r[model + '_trigger_pred'] == 1 for r in selected)
        positive = sum(r[model + '_trigger_pred'] == 1 for r in selected)
        result[stratum] = dict(targets=n, triggered_target_class_count=positive, absolute_asr=_rate(positive, n),
            incremental_flips=flips, incremental_asr=_rate(flips, n),
            clean_negative_targets=clean_negative, conditional_flip=_rate(flips, clean_negative),
            clean_activation_rate=_rate(n - clean_negative, n),
            target_probability_delta_mean=float(np.mean([r[model + '_trigger_p'] - r[model + '_clean_p']
                                                        for r in selected])) if n else None)
    return result


def _normal(rows, model):
    truth = np.asarray([r['label'] for r in rows], dtype=np.int64)
    prediction = np.asarray([r[model + '_pred'] for r in rows], dtype=np.int64)
    probability = np.asarray([r[model + '_p'] for r in rows], dtype=np.float64)
    positive, negative = int((truth == 1).sum()), int((truth == 0).sum())
    tn, fp, fn, tp = (int(((truth == a) & (prediction == b)).sum()) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    return dict(normal_pairs=len(rows), normal_positive_pairs=positive, normal_negative_pairs=negative,
        clean_pair_auc=float(roc_auc_score(truth, probability)) if positive and negative else None,
        clean_pair_ap=float(average_precision_score(truth, probability)) if positive else None,
        clean_pair_accuracy=_rate(tn + tp, len(rows)), clean_pair_positive_recall=_rate(tp, positive),
        clean_prediction_distribution=dict(class_0=tn + fn, class_1=fp + tp),
        clean_confusion_matrix=dict(true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp),
        all_clean_predictions_negative=bool(len(rows) and (prediction == 0).all()),
        clean_class_one_probability_min=float(probability.min()) if len(rows) else None,
        clean_class_one_probability_max=float(probability.max()) if len(rows) else None)


def _collateral(rows, model):
    result = {}
    for relation, prefix in (('adjacent', 'adj'), ('nonincident', 'nonincident')):
        selected = [r for r in rows if r['relation'] == relation]
        negative = [r for r in selected if r['label'] == 0]
        positive = [r for r in selected if r['label'] == 1]
        flips = sum(r[model + '_clean_pred'] == 0 and r[model + '_trigger_pred'] == 1 for r in negative)
        suppressed = sum(r[model + '_clean_pred'] == 1 and r[model + '_trigger_pred'] == 0 for r in positive)
        result.update({prefix + '_negative': len(negative), prefix + '_flip': flips,
            prefix + '_positive': len(positive), prefix + '_suppressed': suppressed,
            relation + '_incremental_fp': _rate(flips, len(negative)),
            relation + '_positive_suppression': _rate(suppressed, len(positive))})
    return result


def _load_arm(directory, mode, *, expected_seed=20260621):
    _require(type(expected_seed) is int and expected_seed in (20260621, 20260622, 20260623),
             'Unregistered expected seed')
    directory = Path(directory)
    summary_bytes = (directory / 'summary.json').read_bytes()
    summary = _decode_json(summary_bytes)
    _require(summary['status'] == 'DEV_EVALUATION_COMPLETE' and summary['mode'] == mode
             and type(summary['seed']) is int and summary['seed'] == expected_seed, 'Unexpected diagnostic arm/seed/status')
    hashes = summary['model_state_sha256']
    _sha(summary['shared_parent_generator_state_sha256'])
    _require(set(hashes) == {'generator', *MODELS}, 'Require all four model hashes')
    for digest in hashes.values():
        _sha(digest)
    for key in ('weights_unchanged', 'inputs_unchanged', 'rng_unchanged', 'topology_preserved',
                'non_target_input_feature_rows_preserved'):
        _require(summary[key] is True, 'Failed recorded evaluation invariant: ' + key)
    for key in ('gradients_created', 'target_selection_uses_predictions', 'graph_arrays_loaded',
                'model_checkpoints_loaded', 'original_val_arrays_loaded', 'original_test_arrays_loaded'):
        _require(summary[key] is False, 'Unexpected recorded evaluation operation: ' + key)
    _require(_count(summary['optimizer_steps']) == 0, 'Diagnostic must not optimize')
    identities, counts = summary['population_identity_sha256'], summary['dev_counts']
    _require(set(identities) == set(POPULATIONS) and set(counts) == set(COUNT_FIELDS), 'Invalid population schema')
    for digest in identities.values():
        _sha(digest)
    for count in counts.values():
        _count(count)
    _require(summary['target_identity_sha256'] == identities['targets'], 'Target identity alias differs')
    _require(counts['registered_scenes'] == counts['processed_scenes']
             == counts['targets'] + counts['scenes_without_eligible_negative_target'], 'Invalid scene accounting')
    _require(counts['targets'] <= counts['triggered_supervised_pair_rows'] <= counts['supervised_pairs']
             <= counts['all_pair_nodes'], 'Invalid pair accounting')
    rows, blobs, assets = {}, {}, {}
    for kind, (filename, field) in FILES.items():
        blobs[kind] = (directory / filename).read_bytes()
        rows[kind] = _parse_csv(blobs[kind], kind)
        sha = _digest(blobs[kind])
        asset = summary[field]
        _require(_sha(asset['sha256']) == sha and _count(asset['rows']) == len(rows[kind]), 'CSV SHA/row count mismatch: ' + kind)
        assets[filename] = dict(sha256=sha, rows=len(rows[kind]))
    _require([len(rows[k]) for k in ('targets', 'normal', 'triggered')] == [counts[k] for k in
             ('targets', 'supervised_pairs', 'triggered_supervised_pair_rows')], 'CSV population counts differ')
    target_by_scene = _bind_rows(rows)
    _require(len({r['scenario_id'] for r in rows['normal']}) <= counts['registered_scenes'], 'Too many CSV scenes')
    _require(all(r['role_pair_index'] < counts['all_pair_nodes'] for r in rows['normal']),
             'DEV role-pair index exceeds recorded total pair population')
    derived_identities = dict(eligible_clean_pairs=_identity_hash([_metadata(r) for r in rows['normal']]),
        targets=_identity_hash([_metadata(r) for r in rows['targets']]),
        triggered_pairs=_identity_hash([dict(_metadata(r), **{k: r[k] for k in
            ('target_pair_index', 'label', 'relation')}) for r in rows['triggered']]))
    _require(all(identities[k] == v for k, v in derived_identities.items()), 'CSV metadata population SHA mismatch')
    metrics = {}
    _require(set(summary['metrics']) == set(MODELS), 'Unexpected model metric set')
    for model in MODELS:
        metrics[model] = dict(targets=_targets(rows['targets'], model), **_normal(rows['normal'], model),
                             collateral=_collateral(rows['triggered'], model))
        expected = dict(metrics[model])
        if 'clean_pair_ap' not in summary['metrics'][model]:
            expected.pop('clean_pair_ap')
        expected['collateral'] = {k: v for k, v in expected['collateral'].items()
            if k not in ('nonincident_positive', 'nonincident_suppressed', 'nonincident_positive_suppression')
            or k in summary['metrics'][model]['collateral']}
        _same(summary['metrics'][model], expected, mode + '.' + model)
        metrics[model]['collateral_by_target_motion'] = {stratum: _collateral(
            [r for r in rows['triggered'] if stratum == 'all'
             or target_by_scene[r['scenario_id']]['motion_regime'] == stratum], model) for stratum in STRATA}
    changed = [r['changed_coordinates'] for r in rows['targets']]
    _same(summary['feature_changes'], dict(target_rows_changed=sum(n > 0 for n in changed),
        target_rows_unchanged=sum(n == 0 for n in changed),
        maximum_changed_coordinates_per_target=max(changed, default=0)), 'feature_changes')
    copy_bytes = (directory / 'copy_rows.json').read_bytes()
    copy = _decode_json(copy_bytes)
    copy_digest = _digest(copy_bytes)
    _require(_sha(summary['raw_copy_rows']['sha256']) == copy_digest
             and _count(summary['raw_copy_rows']['rows']) == len(copy['rows']), 'Copy JSON SHA/row count mismatch')
    _require(copy['mode'] == mode and copy['selected_coordinates'] == list(SELECTED_COORDINATES)
             and copy['allowed_write_coordinates'] == list(ALLOWED_COORDINATES), 'Copy coordinate/mode contract differs')
    _require(isinstance(copy['rows'], list) and len(copy['rows']) == len(rows['targets']), 'Copy row count differs')
    clean_values, outputs = [], []
    for record, target in zip(copy['rows'], rows['targets']):
        _require(all(type(record[k]) is type(target[k]) and record[k] == target[k] for k in
            ('scenario_id', 'pair_index', 'src_track_id', 'dst_track_id', 'motion_regime')), 'Copy target order/identity differs')
        values = []
        for key in ('clean30', 'output30'):
            _require(isinstance(record[key], list) and len(record[key]) == 30, 'Require recorded [30] feature values')
            raw = np.asarray([_number(v) for v in record[key]], dtype=np.float64)
            value = raw.astype(np.float32)
            _require(np.array_equal(value.astype(np.float64), raw) and (np.abs(value) <= 10).all(),
                     'Require exact finite float32 domain-valued copy rows')
            values.append(value)
        clean_values.append(values[0])
        outputs.append(values[1])
    clean_values = np.asarray(clean_values, dtype=np.float32).reshape(-1, 30)
    outputs = np.asarray(outputs, dtype=np.float32).reshape(-1, 30)
    inactive = [i for i in range(30) if i not in ACTIVE_POSITIONS]
    _require(np.array_equal(clean_values[:, inactive].view(np.uint32), outputs[:, inactive].view(np.uint32)),
             'Copy output changed one of six unsupported original-30 coordinates')
    for positions in COPY_POSITIONS:
        group = clean_values[:, positions]
        _require((group == group[:, :1]).all(), 'Clean copy values disagree')
    delta = outputs.astype(np.float64) - clean_values.astype(np.float64)
    for target, value, difference in zip(rows['targets'], outputs, delta):
        _same({key: target[key] for key in FEATURE_FIELDS}, dict(
            changed_coordinates=int(np.count_nonzero(difference)),
            boundary_coordinates=int((np.abs(value[list(ACTIVE_POSITIONS)]) == 10).sum()),
            delta_l1=float(np.abs(difference).sum()), delta_linf=float(np.abs(difference).max())), 'copy versus target budget')
    assets['summary.json'] = dict(sha256=_digest(summary_bytes))
    assets['copy_rows.json'] = dict(sha256=copy_digest, rows=len(copy['rows']))
    return dict(summary=summary, rows=rows, normal_bytes=blobs['normal'], metrics=metrics,
                clean=clean_values, outputs=outputs, assets=assets)


def _sample_stats(values):
    return dict(mean=float(values.mean()) if values.size else None,
                min=float(values.min()) if values.size else None, max=float(values.max()) if values.size else None)


def _copy_stats(clean, values):
    delta = values.astype(np.float64) - clean.astype(np.float64)
    groups, violations = [], []
    for name, coordinates, positions in zip(COPY_NAMES, COPY_GROUPS, COPY_POSITIONS):
        spread = np.ptp(values[:, positions].astype(np.float64), axis=1)
        bad = spread > 1e-5
        violations.append(bad)
        groups.append(dict(name=name, coordinates=list(coordinates), rows=len(values), violations=int(bad.sum()),
            violation_rate=_rate(int(bad.sum()), len(values)), range=_sample_stats(spread)))
    any_bad = np.any(np.stack(violations, axis=1), axis=1)
    return dict(rows=len(values), groups=groups, any_group_violations=int(any_bad.sum()),
        any_group_violation_rate=_rate(int(any_bad.sum()), len(values)),
        l1=_sample_stats(np.abs(delta).sum(axis=1)), l2=_sample_stats(np.linalg.norm(delta, axis=1)),
        linf=_sample_stats(np.abs(delta).max(axis=1)),
        actual_changed_coordinates=_sample_stats(np.count_nonzero(delta, axis=1)),
        boundary_coordinates=_sample_stats((np.abs(values[:, ACTIVE_POSITIONS]) == 10).sum(axis=1)))


def _paired(left, right, model, stratum):
    pairs = [(a, b) for a, b in zip(left, right) if stratum == 'all' or a['motion_regime'] == stratum]
    result = dict(targets=len(pairs))
    for name, predicate in (
        ('incremental_flips', lambda r: r[model + '_clean_pred'] == 0 and r[model + '_trigger_pred'] == 1),
        ('triggered_positive', lambda r: r[model + '_trigger_pred'] == 1)):
        bits = [(predicate(a), predicate(b)) for a, b in pairs]
        gained, lost = sum(not a and b for a, b in bits), sum(a and not b for a, b in bits)
        result[name] = dict(gained=gained, lost=lost, retained=sum(a and b for a, b in bits),
                            neither=sum(not a and not b for a, b in bits), net_gain=gained - lost)
    a, b = _targets(left, model)[stratum], _targets(right, model)[stratum]
    result['delta_pair_mean_minus_identity'] = {key: b[key] - a[key] if a[key] is not None and b[key] is not None else None
        for key in ('absolute_asr', 'incremental_asr', 'conditional_flip', 'target_probability_delta_mean')}
    return result


def build_comparison(identity_dir, pair_mean_dir):
    """Validate and recompute two saved evaluations; return JSON-safe diagnostics.

    The recorded float32 feature outputs are checked against a paired arithmetic
    mean oracle (absolute tolerance 1e-6); outside the six triplets values must
    be bit-identical. Norm monotonicity uses absolute tolerance 1e-5. Shared
    scene counts/hashes are checked as records, not reconstructed from graphs.
    """
    try:
        arms = {mode: _load_arm(path, mode) for mode, path in zip(ARMS, (identity_dir, pair_mean_dir))}
        left, right = (arms[mode] for mode in ARMS)
        for field in ('population_identity_sha256', 'dev_counts'):
            _require(left['summary'][field] == right['summary'][field], 'Arm population identities/counts differ')
        _require(left['summary']['shared_parent_generator_state_sha256'] ==
                 right['summary']['shared_parent_generator_state_sha256'], 'Parent generator weights differ')
        _require(left['normal_bytes'] == right['normal_bytes'], 'Normal CSV must be byte-identical across arms')
        for model in MODELS:
            _require(left['summary']['model_state_sha256'][model] == right['summary']['model_state_sha256'][model],
                     'A prediction model changed between arms: ' + model)
        for kind in ('targets', 'triggered'):
            fields = METADATA + (('target_pair_index', 'label', 'relation') if kind == 'triggered' else ())
            _require([{k: r[k] for k in fields} for r in left['rows'][kind]] ==
                     [{k: r[k] for k in fields} for r in right['rows'][kind]], 'Arm row identities/order differ')
        _require(np.array_equal(left['clean'].view(np.uint32), right['clean'].view(np.uint32)), 'Arm clean feature rows differ')
        oracle = left['outputs'].copy()
        for positions in COPY_POSITIONS:
            group = left['outputs'][:, positions].astype(np.float64)
            mean = group.mean(axis=1, keepdims=True).astype(np.float32)
            oracle[:, positions] = mean
            actual = right['outputs'][:, positions]
            _require((actual == actual[:, :1]).all(), 'Pair-mean output copies are not exactly equal')
        untouched = [i for i in range(30) if all(i not in g for g in COPY_POSITIONS)]
        _require(np.array_equal(left['outputs'][:, untouched].view(np.uint32), right['outputs'][:, untouched].view(np.uint32)),
                 'Pair mean changed a singleton/unsupported output')
        _require(np.allclose(oracle, right['outputs'], rtol=0, atol=1e-6), 'Pair mean is not the mean of the identity outputs')
        before = left['outputs'].astype(np.float64) - left['clean'].astype(np.float64)
        after = right['outputs'].astype(np.float64) - right['clean'].astype(np.float64)
        norm_excess = {}
        for name, order in (('l1', 1), ('l2', 2), ('linf', np.inf)):
            excess = np.linalg.norm(after, ord=order, axis=1) - np.linalg.norm(before, ord=order, axis=1)
            _require((excess <= 1e-5).all(), 'Pair mean increased a per-row norm: ' + name)
            norm_excess[name] = float(excess.max()) if excess.size else None
        targets = left['rows']['targets']
        feature_statistics = {mode: {stratum: _copy_stats(arm['clean'][mask], arm['outputs'][mask])
            for stratum in STRATA
            for mask in [np.asarray([stratum == 'all' or r['motion_regime'] == stratum for r in targets], dtype=bool)]}
            for mode, arm in arms.items()}
        paired = {model: {stratum: _paired(targets, right['rows']['targets'], model, stratum)
                           for stratum in STRATA} for model in MODELS}
        return dict(status='COPY_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED', seed=20260621,
            scope='fixed-model paired original-TRAIN DEV diagnostic; not independent confirmation',
            population_identity_sha256=left['summary']['population_identity_sha256'],
            population_counts=left['summary']['dev_counts'],
            shared_parent_generator_state_sha256=left['summary']['shared_parent_generator_state_sha256'],
            model_state_sha256={mode: arm['summary']['model_state_sha256'] for mode, arm in arms.items()},
            source_files={mode: arm['assets'] for mode, arm in arms.items()},
            metrics={mode: arm['metrics'] for mode, arm in arms.items()}, paired_target_responses=paired,
            copy_statistics=feature_statistics,
            integrity=dict(normal_csv_byte_identical=True, population_identity_and_counts_match=True,
                prediction_models_identical=True, csv_sha_and_rows_verified=True,
                csv_population_hashes_recomputed=['eligible_clean_pairs', 'targets', 'triggered_pairs'],
                scene_hash_and_all_pair_count_source='shared summary metadata; no graph re-reading',
                summary_metrics_recomputed=True, projection_oracle_verified=True,
                singleton_outputs_bit_identical=True, clean_rows_bit_identical=True,
                original30_allowed24_scope_verified=True, nonincreasing_norm_absolute_tolerance=1e-5,
                projection_absolute_tolerance=1e-6, maximum_per_row_norm_excess=norm_excess),
            definitions=dict(copy_violation_tolerance=1e-5, feature_units='frozen standardized coordinates',
                ap='sklearn average_precision_score; null without positives; all-positive value is degenerate',
                ap_and_nonincident_positive_suppression='additional CSV-derived diagnostics; absent from old evaluate_dev summary',
                collateral_strata='target motion, not affected pair motion; pooled true-label denominators',
                probabilities='stored class-one probabilities; stored argmax predictions retained, including ties',
                delta_direction='pair_mean minus identity'),
            optimizer_steps=0, new_inference=False, graph_or_model_assets_read=False,
            performance_gates_applied=False, blind_evaluation_claim=False, independent_confirmation_claim=False,
            physical_validity_claim=False, statistical_significance_claimed=False,
            automatic_followup=False, automatic_promotion=False)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('Missing or malformed copy diagnostic evidence') from error
