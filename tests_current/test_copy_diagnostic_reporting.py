"""Synthetic CSV/JSON evidence only: no graph/model assets or inference."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from spear_current.copy_diagnostic_reporting import (
    ACTIVE_POSITIONS, ALLOWED_COORDINATES, ARMS, COPY_POSITIONS, FILES,
    HEADERS, METADATA, MODELS, build_comparison,
    _identity_hash,
)
from spear_current.evaluation import _normal_metrics, _target_metrics
from spear_current.outer_evaluation import _fingerprint
from spear_current.support_schema import SELECTED_COORDINATES


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, kind, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS[kind])
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def asset(path, rows):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows)


def feature_fields(record):
    clean, output = (np.asarray(record[key], dtype=np.float64) for key in ('clean30', 'output30'))
    delta = output - clean
    return dict(changed_coordinates=int(np.count_nonzero(delta)),
        boundary_coordinates=int((np.abs(output[list(ACTIVE_POSITIONS)]) == 10).sum()),
        delta_l1=float(np.abs(delta).sum()), delta_linf=float(np.abs(delta).max()))


def build_fixture(root, *, empty=False, no_positives=False):
    """Three target scenes and a positive-only scene expose pooled denominators."""
    clean_rows, targets, triggered = [], {a: [] for a in ARMS}, {a: [] for a in ARMS}
    copies = {a: [] for a in ARMS}
    offset = 0
    scene_count = 0 if empty else 3 if no_positives else 4
    for scene_index in range(scene_count):
        has_target = scene_index < 3
        target_motion = 'slow' if scene_index == 0 else 'moving'
        for pair_index in range(5 if has_target else 1):
            label = (0, 0, 1, 0, 1)[pair_index] if has_target else 1
            if no_positives:
                label = 0
            clean_pred = (int(scene_index == 2), 0, 1, 1, 1)[pair_index] if has_target else 0
            row = dict(scenario_id=f'scene-{scene_index}', split_group_id=f'group-{scene_index}',
                original_split='train', role='dev', pair_index=pair_index,
                role_pair_index=offset + pair_index, src_track_id=f'a-{pair_index}', dst_track_id=f'b-{pair_index}',
                true_label=label, eligible=True, motion_regime=target_motion if pair_index == 0
                    else 'moving' if target_motion == 'slow' else 'slow', is_attachment=False, is_outer=False)
            normal = dict(row, label=label)
            for model in MODELS:
                normal.update({model + '_p': .8 if clean_pred else .5,
                               model + '_pred': clean_pred})
            clean_rows.append(normal)
            if not has_target:
                continue
            for arm in ARMS:
                if pair_index == 0:
                    prediction = ((0, 1, 0) if arm == 'identity' else (1, 0, 1))[scene_index]
                else:
                    prediction = ((0, 1, 0, 1, 1) if arm == 'identity' else (0, 0, 1, 0, 0))[pair_index]
                response = {key: row[key] for key in METADATA}
                for model in MODELS:
                    response.update({model + '_clean_p': normal[model + '_p'],
                        model + '_trigger_p': .9 if prediction else .1,
                        model + '_clean_pred': clean_pred, model + '_trigger_pred': prediction,
                        model + '_target_nll': -float(np.log(.9 if prediction else .1))})
                trigger = dict(response, target_pair_index=0, label=label,
                    relation=('target', 'adjacent', 'adjacent', 'nonincident', 'nonincident')[pair_index])
                triggered[arm].append(trigger)
                if pair_index == 0:
                    original, generated = np.zeros(30, dtype=np.float32), np.zeros(30, dtype=np.float32)
                    generated[list(ACTIVE_POSITIONS)] = 2
                    for i, positions in enumerate(COPY_POSITIONS):
                        generated[list(positions)] = np.asarray([-6, 0, 6] if i == 0 else [-3, 0, 3], dtype=np.float32)
                        if arm == 'pair_mean':
                            generated[list(positions)] = 0
                    copy = {key: row[key] for key in ('scenario_id', 'pair_index', 'src_track_id', 'dst_track_id', 'motion_regime')}
                    copy.update(clean30=original.tolist(), output30=generated.tolist())
                    copies[arm].append(copy)
                    targets[arm].append(dict(response, **feature_fields(copy)))
        offset += 5 if has_target else 1
    for arm in ARMS:
        directory = root / arm
        directory.mkdir()
        rows = dict(targets=targets[arm], normal=clean_rows, triggered=triggered[arm])
        metrics = {}
        for model in MODELS:
            d = dict.fromkeys(('adj_negative', 'adj_flip', 'nonincident_negative',
                              'nonincident_flip', 'adj_positive', 'adj_suppressed'), 0)
            for row in rows['triggered']:
                prefix = 'adj' if row['relation'] == 'adjacent' else 'nonincident'
                if row['relation'] == 'target':
                    continue
                before, after = row[model + '_clean_pred'], row[model + '_trigger_pred']
                if row['label'] == 0:
                    d[prefix + '_negative'] += 1
                    d[prefix + '_flip'] += before == 0 and after == 1
                elif prefix == 'adj':
                    d['adj_positive'] += 1
                    d['adj_suppressed'] += before == 1 and after == 0
            rate = lambda a, b: a / b if b else None
            d.update(adjacent_incremental_fp=rate(d['adj_flip'], d['adj_negative']),
                nonincident_incremental_fp=rate(d['nonincident_flip'], d['nonincident_negative']),
                adjacent_positive_suppression=rate(d['adj_suppressed'], d['adj_positive']))
            metrics[model] = dict(targets=_target_metrics(rows['targets'], model),
                                  **_normal_metrics(clean_rows, model), collateral=d)
        identities = dict(scenes='1' * 64,
            eligible_clean_pairs=_fingerprint([{k: r[k] for k in METADATA} for r in clean_rows]),
            targets=_fingerprint([{k: r[k] for k in METADATA} for r in rows['targets']]),
            triggered_pairs=_fingerprint([{k: r[k] for k in METADATA + ('target_pair_index', 'label', 'relation')}
                                          for r in rows['triggered']]))
        counts = dict(registered_scenes=1 if empty else scene_count, processed_scenes=1 if empty else scene_count,
            scenes_without_eligible_negative_target=0 if no_positives else 1, all_pair_nodes=0 if empty else offset + 1,
            supervised_pairs=len(clean_rows), targets=len(rows['targets']),
            triggered_supervised_pair_rows=len(rows['triggered']))
        changed = [r['changed_coordinates'] for r in rows['targets']]
        summary = dict(status='DEV_EVALUATION_COMPLETE', mode=arm, seed=20260621,
            shared_parent_generator_state_sha256='5' * 64,
            model_state_sha256=dict(generator=('a' if arm == 'identity' else 'b') * 64,
                surrogate='2' * 64, victim='3' * 64, clean_reference='4' * 64),
            metrics=metrics, dev_counts=counts, population_identity_sha256=identities,
            target_identity_sha256=identities['targets'], weights_unchanged=True, inputs_unchanged=True,
            rng_unchanged=True, optimizer_steps=0, gradients_created=False, topology_preserved=True,
            non_target_input_feature_rows_preserved=True, target_selection_uses_predictions=False,
            graph_arrays_loaded=False, model_checkpoints_loaded=False,
            original_val_arrays_loaded=False, original_test_arrays_loaded=False,
            feature_changes=dict(target_rows_changed=sum(n > 0 for n in changed),
                target_rows_unchanged=sum(n == 0 for n in changed), maximum_changed_coordinates_per_target=max(changed, default=0)))
        for kind, (filename, field) in FILES.items():
            write_csv(directory / filename, kind, rows[kind])
            summary[field] = asset(directory / filename, len(rows[kind]))
        write_json(directory / 'copy_rows.json', dict(mode=arm, selected_coordinates=list(SELECTED_COORDINATES),
            allowed_write_coordinates=list(ALLOWED_COORDINATES), rows=copies[arm]))
        summary['raw_copy_rows'] = asset(directory / 'copy_rows.json', len(copies[arm]))
        write_json(directory / 'summary.json', summary)
    return root / 'identity', root / 'pair_mean'


def update_csv(directory, kind, rows):
    filename, field = FILES[kind]
    write_csv(directory / filename, kind, rows)
    summary = read_json(directory / 'summary.json')
    summary[field] = asset(directory / filename, len(rows))
    write_json(directory / 'summary.json', summary)


def refresh_features(directory):
    copy = read_json(directory / 'copy_rows.json')
    targets = read_csv(directory / 'targets.csv')
    for row, record in zip(targets, copy['rows']):
        row.update(feature_fields(record))
    update_csv(directory, 'targets', targets)
    summary = read_json(directory / 'summary.json')
    counts = [int(r['changed_coordinates']) for r in targets]
    summary['feature_changes'] = dict(target_rows_changed=sum(n > 0 for n in counts),
        target_rows_unchanged=sum(n == 0 for n in counts), maximum_changed_coordinates_per_target=max(counts, default=0))
    write_json(directory / 'summary.json', summary)


def update_copy(directory, value):
    path = directory / 'copy_rows.json'
    write_json(path, value)
    summary = read_json(directory / 'summary.json')
    summary['raw_copy_rows'] = asset(path, len(value['rows']))
    write_json(directory / 'summary.json', summary)


class CopyDiagnosticReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.identity, self.pair_mean = build_fixture(self.root)

    def compare(self):
        return build_comparison(self.identity, self.pair_mean)

    def test_independent_metrics_paired_gains_losses_and_copy_statistics(self):
        result = self.compare()
        original = result['metrics']['identity']['victim']
        candidate = result['metrics']['pair_mean']['victim']
        self.assertAlmostEqual(original['targets']['all']['incremental_asr'], 1 / 3)
        self.assertAlmostEqual(candidate['targets']['all']['absolute_asr'], 2 / 3)
        self.assertEqual(candidate['targets']['all']['conditional_flip'], .5)
        self.assertEqual(original['targets']['all']['clean_negative_targets'], 2)
        paired = result['paired_target_responses']['victim']['all']
        self.assertEqual(paired['incremental_flips'], dict(gained=1, lost=1, retained=0, neither=1, net_gain=0))
        self.assertEqual(paired['triggered_positive']['net_gain'], 1)
        self.assertAlmostEqual(paired['delta_pair_mean_minus_identity']['absolute_asr'], 1 / 3)
        self.assertEqual(original['collateral']['adjacent_incremental_fp'], 1)
        self.assertEqual(candidate['collateral']['nonincident_incremental_fp'], 0)
        self.assertEqual(original['collateral']['adjacent_positive_suppression'], 1)
        self.assertEqual(candidate['collateral']['nonincident_positive_suppression'], 1)
        self.assertEqual(candidate['collateral_by_target_motion']['slow']['adj_negative'], 1)
        self.assertEqual(candidate['collateral_by_target_motion']['moving']['adj_negative'], 2)
        self.assertEqual(candidate['normal_pairs'], 16)  # Includes the non-triggered scene.
        self.assertEqual(candidate['clean_pair_ap'], original['clean_pair_ap'])
        self.assertGreater(candidate['clean_pair_ap'], 0)
        before = result['copy_statistics']['identity']['all']
        after = result['copy_statistics']['pair_mean']['all']
        self.assertEqual(before['any_group_violations'], 3)
        self.assertEqual(after['any_group_violations'], 0)
        self.assertEqual(before['l1']['mean'], 54)
        self.assertEqual(after['l1']['mean'], 12)
        self.assertEqual(after['actual_changed_coordinates']['max'], 6)
        self.assertFalse(result['performance_gates_applied'])
        self.assertFalse(result['independent_confirmation_claim'])
        json.dumps(result, allow_nan=False)

    def test_metadata_fingerprint_is_exactly_evaluator_compatible(self):
        examples = [[], [{}], [dict(scenario_id='scene', eligible=True, pair_index=3,
                                    label=0, relation='target')], {'nested': [1, False, None]}]
        for value in examples:
            with self.subTest(value=value):
                self.assertEqual(_identity_hash(value), _fingerprint(value))

    def test_empty_population_preserves_null_denominators(self):
        with TemporaryDirectory() as tmp:
            result = build_comparison(*build_fixture(Path(tmp), empty=True))
        for model in MODELS:
            metric = result['metrics']['identity'][model]
            self.assertIsNone(metric['targets']['all']['conditional_flip'])
            self.assertIsNone(metric['clean_pair_auc'])
            self.assertIsNone(metric['clean_pair_ap'])
            self.assertIsNone(metric['collateral']['nonincident_positive_suppression'])
        self.assertIsNone(result['copy_statistics']['pair_mean']['all']['l1']['mean'])

    def test_no_positive_normal_population_has_undefined_ap(self):
        with TemporaryDirectory() as tmp:
            result = build_comparison(*build_fixture(Path(tmp), no_positives=True))
        self.assertIsNone(result['metrics']['identity']['victim']['clean_pair_ap'])
        self.assertIsNone(result['metrics']['identity']['victim']['clean_pair_auc'])

    def test_build_is_read_only_and_only_reads_named_arm_files(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        allowed = set(before)
        old_open = Path.open
        def guarded(path, mode='r', *args, **kwargs):
            self.assertIn(str(path), allowed)
            self.assertFalse(any(flag in mode for flag in ('w', 'a', 'x', '+')))
            return old_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            self.compare()
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_sha_drift_is_rejected(self):
        with (self.pair_mean / 'targets.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.compare()

    def test_copy_json_sha_is_verified_before_values_are_used(self):
        path = self.pair_mean / 'copy_rows.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Copy JSON SHA'):
            self.compare()

    def test_normal_csv_must_match_bytes_not_only_parsed_values(self):
        path = self.pair_mean / 'clean_pair_scores.csv'
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        summary = read_json(self.pair_mean / 'summary.json')
        summary['raw_clean_scores'] = asset(path, summary['raw_clean_scores']['rows'])
        write_json(self.pair_mean / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'byte-identical'):
            self.compare()

    def test_prediction_model_hash_and_population_identity_drift_rejected(self):
        original = read_json(self.pair_mean / 'summary.json')
        for key in ('model', 'population', 'counts', 'parent'):
            summary = deepcopy(original)
            if key == 'model':
                summary['model_state_sha256']['victim'] = 'f' * 64
            elif key == 'population':
                summary['population_identity_sha256']['scenes'] = 'f' * 64
            elif key == 'parent':
                summary['shared_parent_generator_state_sha256'] = 'f' * 64
            else:
                summary['dev_counts']['all_pair_nodes'] += 1
            write_json(self.pair_mean / 'summary.json', summary)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare()

    def test_wrong_rates_and_normal_metrics_rejected(self):
        original = read_json(self.pair_mean / 'summary.json')
        mutations = (
            lambda m: m['targets']['all'].update(absolute_asr=.99),
            lambda m: m['targets']['all'].update(conditional_flip=.99),
            lambda m: m['collateral'].update(adjacent_incremental_fp=.99),
            lambda m: m['collateral'].update(nonincident_incremental_fp=.99),
            lambda m: m.update(clean_pair_auc=.99),
            lambda m: m.update(clean_pair_ap=.99),
            lambda m: m.update(clean_pair_positive_recall=.99),
        )
        for mutation in mutations:
            summary = deepcopy(original)
            mutation(summary['metrics']['victim'])
            write_json(self.pair_mean / 'summary.json', summary)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare()

    def test_csv_binding_rejects_illegal_probability_target_response_and_missing_rows(self):
        original = read_csv(self.pair_mean / 'triggered_pair_scores.csv')
        for mutation in ('probability', 'response', 'target', 'missing', 'duplicate'):
            rows = deepcopy(original)
            if mutation == 'probability': rows[0]['victim_trigger_p'] = 'nan'
            elif mutation == 'response': rows[0]['victim_trigger_p'] = '.7'
            elif mutation == 'target': rows[0]['target_pair_index'] = '1'
            elif mutation == 'missing': rows.pop()
            else: rows[-1] = deepcopy(rows[0])
            update_csv(self.pair_mean, 'triggered', rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare()

    def test_copy_order_support_and_clean_drift_rejected(self):
        original = read_json(self.pair_mean / 'copy_rows.json')
        for mutation in ('order', 'support', 'clean', 'nonfloat32'):
            copy = deepcopy(original)
            if mutation == 'order': copy['rows'].reverse()
            elif mutation == 'support': copy['allowed_write_coordinates'].reverse()
            elif mutation == 'clean': copy['rows'][0]['clean30'][0] = -0.0
            else: copy['rows'][0]['output30'][COPY_POSITIONS[0][0]] = .1
            update_copy(self.pair_mean, copy)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare()

    def test_equal_but_arbitrary_group_is_not_accepted_as_paired_mean(self):
        copy = read_json(self.pair_mean / 'copy_rows.json')
        for position in COPY_POSITIONS[0]:
            copy['rows'][0]['output30'][position] = 1.0
        update_copy(self.pair_mean, copy)
        refresh_features(self.pair_mean)
        with self.assertRaisesRegex(ValueError, 'not the mean'):
            self.compare()

    def test_singleton_change_is_rejected_even_when_budget_decreases(self):
        copy = read_json(self.pair_mean / 'copy_rows.json')
        singleton = next(p for p in ACTIVE_POSITIONS if all(p not in g for g in COPY_POSITIONS))
        copy['rows'][0]['output30'][singleton] = 1.0
        update_copy(self.pair_mean, copy)
        refresh_features(self.pair_mean)
        with self.assertRaisesRegex(ValueError, 'singleton'):
            self.compare()

    def test_unsupported_change_rejected_before_comparison(self):
        copy = read_json(self.pair_mean / 'copy_rows.json')
        copy['rows'][0]['output30'][0] = 1.0
        update_copy(self.pair_mean, copy)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            self.compare()

    def test_missing_or_nonfinite_json_fails_closed(self):
        path = self.pair_mean / 'summary.json'
        original = path.read_text()
        for replacement in ('{"status": NaN}', '{}'):
            path.write_text(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.compare()
        path.write_text(original)


if __name__ == '__main__':
    unittest.main()
