"""Synthetic new-run evidence only; never train or read real data/checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from spear_current.copy_training_reporting import (
    ALLOWED_SEEDS, AUC_POLICY, GATES, _acceptability, _normal_by_pair_motion, build_training_comparison,
)
from spear_current.copy_diagnostic_reporting import (
    ARMS, COPY_POSITIONS, FILES, METADATA, MODELS, _collateral, _parse_csv,
)
from spear_current.evaluation import _normal_metrics, _target_metrics
from spear_current.outer_evaluation import _fingerprint
from tests_current.test_copy_diagnostic_reporting import (
    asset, build_fixture, feature_fields, read_json, write_csv, write_json,
)


def refresh(directory, rows, copy, summary):
    for row, values in zip(rows['targets'], copy['rows']):
        row.update(feature_fields(values))
    summary['metrics'] = {model: dict(targets=_target_metrics(rows['targets'], model),
        **_normal_metrics(rows['normal'], model), collateral=_collateral(rows['triggered'], model)) for model in MODELS}
    summary['population_identity_sha256'] = dict(scenes='1' * 64,
        eligible_clean_pairs=_fingerprint([{k: r[k] for k in METADATA} for r in rows['normal']]),
        targets=_fingerprint([{k: r[k] for k in METADATA} for r in rows['targets']]),
        triggered_pairs=_fingerprint([{k: r[k] for k in METADATA + ('target_pair_index', 'label', 'relation')}
                                      for r in rows['triggered']]))
    summary['target_identity_sha256'] = summary['population_identity_sha256']['targets']
    summary['dev_counts'] = dict(registered_scenes=128, processed_scenes=128,
        scenes_without_eligible_negative_target=16, all_pair_nodes=577, supervised_pairs=len(rows['normal']),
        targets=len(rows['targets']), triggered_supervised_pair_rows=len(rows['triggered']))
    changed = [r['changed_coordinates'] for r in rows['targets']]
    summary['feature_changes'] = dict(target_rows_changed=sum(n > 0 for n in changed),
        target_rows_unchanged=sum(n == 0 for n in changed), maximum_changed_coordinates_per_target=max(changed, default=0))
    for kind, (filename, field) in FILES.items():
        write_csv(directory / filename, kind, rows[kind])
        summary[field] = asset(directory / filename, len(rows[kind]))
    write_json(directory / 'copy_rows.json', copy)
    summary['raw_copy_rows'] = asset(directory / 'copy_rows.json', len(copy['rows']))
    write_json(directory / 'summary.json', summary)


def read_development(directory):
    return ({kind: _parse_csv((directory / filename).read_bytes(), kind) for kind, (filename, _) in FILES.items()},
            read_json(directory / 'copy_rows.json'), read_json(directory / 'summary.json'))


def training_fixture(root, seed=20260621):
    """Expand hand-countable old synthetic rows to the registered 128/112/79/33."""
    write_json(root / 'plan.json', dict(protocol=dict(seed=seed, gates=GATES,
        train_scenes=512, dev_scenes=128, expected_dev_targets=112, expected_dev_slow=79, expected_dev_moving=33)))
    with TemporaryDirectory() as temporary:
        old_paths = build_fixture(Path(temporary))
        for arm, old in zip(ARMS, old_paths):
            source_rows, source_copy, summary = read_development(old)
            folder, rows = root / arm, dict(normal=[], targets=[], triggered=[])
            folder.mkdir()
            directory = folder / 'development'
            directory.mkdir()
            copy = dict(source_copy, rows=[])
            for index in range(112):
                scene, group = f'target-scene-{index}', f'target-group-{index}'
                source_scene = f'scene-{index % 3}'
                motion = 'slow' if index < 79 else 'moving'
                for kind in rows:
                    for original in source_rows[kind]:
                        if original['scenario_id'] == source_scene:
                            row = deepcopy(original)
                            row.update(scenario_id=scene, split_group_id=group,
                                role_pair_index=index * 5 + row['pair_index'], motion_regime=motion)
                            rows[kind].append(row)
                record = deepcopy(source_copy['rows'][index % 3])
                record.update(scenario_id=scene, motion_regime=motion)
                copy['rows'].append(record)
            positive_only = next(r for r in source_rows['normal'] if r['scenario_id'] == 'scene-3')
            for index in range(16):
                row = deepcopy(positive_only)
                row.update(scenario_id=f'positive-scene-{index}', split_group_id=f'positive-group-{index}',
                           role_pair_index=560 + index)
                rows['normal'].append(row)
            summary.update(seed=seed, generator_contract=dict(mode=arm, frozen_postprocess=True, output_mode='domain_projection'),
                shared_parent_generator_state_sha256=('e' if arm == 'identity' else 'f') * 64,
                clean_reference_is_independent=False)
            # End-state predictors are intentionally different across arms.
            if arm == 'pair_mean':
                summary['model_state_sha256'].update(surrogate='b' * 64, victim='c' * 64)
            refresh(directory, rows, copy, summary)
            write_json(folder / 'initial_state.json', dict(seed=seed, mode=arm,
                generator_parent_state_sha256='d' * 64, surrogate_state_sha256='6' * 64,
                proxy_state_sha256=['7' * 64, '8' * 64], clean_reference_state_sha256='4' * 64,
                training_generator_state_sha256=('a' if arm == 'identity' else 'b') * 64,
                contract=dict(mode=arm, frozen_postprocess=False, output_mode='domain_projection')))
            epoch = 17 if arm == 'identity' else 42
            write_json(folder / 'victim_record.json', dict(initialization_digest='9' * 64, fit_rng_digest='a' * 64,
                victim_seed=seed, trace=dict(training_forwards=200, eval_forwards=200,
                    selected_auc_epoch=epoch, selected_auc_row=dict(epoch=epoch, auc=.8), selection_policy=AUC_POLICY)))
            write_json(folder / 'completion.json', dict(status='COPY_TRAINING_ARM_COMPLETE', arm=arm, seed=seed,
                residual_steps=200, surrogate_steps=200, discriminator_steps=200, victim_epochs=200,
                generator_selected_epoch=199, victim_selected_epoch=epoch, inputs_unchanged=True,
                frozen_sources_unchanged=True, initial_cpu_rng_sha256='b' * 64,
                distribution_initial_state_sha256='c' * 64, victim_initialization_digest='9' * 64,
                victim_fit_rng_digest='a' * 64))


def gate_metrics():
    targets = {key: dict(targets=10, incremental_flips=8, incremental_asr=.8) for key in ('all', 'moving')}
    return dict(victim=dict(targets=targets, collateral=dict(adjacent_incremental_fp=.04, adj_flip=4, adj_negative=100),
        clean_pair_auc=.9, clean_pair_positive_recall=.7,
        clean_prediction_distribution=dict(class_0=50, class_1=50)),
        clean_reference=dict(clean_pair_auc=.9, clean_pair_positive_recall=.7))


class CopyTrainingReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        training_fixture(self.root)

    def compare(self):
        return build_training_comparison(self.root)

    def mutate_json(self, relative, mutation):
        path = self.root / relative
        value = read_json(path)
        mutation(value)
        write_json(path, value)

    def set_seed(self, seed, *, include_plan=True):
        if include_plan:
            self.mutate_json('plan.json', lambda r: r['protocol'].update(seed=seed))
        for arm in ARMS:
            for filename, field in (('initial_state.json', 'seed'), ('completion.json', 'seed'),
                                    ('victim_record.json', 'victim_seed'), ('development/summary.json', 'seed')):
                self.mutate_json(arm + '/' + filename, lambda r, field=field: r.update({field: seed}))

    def test_matched_budgets_and_different_final_models_are_accepted(self):
        result = self.compare()
        self.assertEqual(result['population_counts']['targets'], 112)
        self.assertNotEqual(result['final_domain_generator_state_sha256']['identity'],
                            result['final_domain_generator_state_sha256']['pair_mean'])
        self.assertNotEqual(result['final_model_state_sha256']['identity']['victim'],
                            result['final_model_state_sha256']['pair_mean']['victim'])
        self.assertEqual(result['victim_records']['identity']['selected_auc_epoch'], 17)
        self.assertEqual(result['victim_records']['pair_mean']['selected_auc_epoch'], 42)
        self.assertTrue(result['integrity']['common_recorded_initial_fields_match'])
        self.assertFalse(result['integrity']['cross_arm_projection_oracle_applied'])
        self.assertFalse(result['automatic_promotion'])
        self.assertFalse(result['multiseed_evidence'])
        json.dumps(result, allow_nan=False)

    def test_each_registered_seed_binds_every_record_and_report(self):
        for seed in ALLOWED_SEEDS:
            self.set_seed(seed)
            with self.subTest(seed=seed):
                result = self.compare()
                self.assertEqual(result['seed'], seed)
                self.assertEqual(result['common_initial_state']['seed'], seed)
                for arm in ARMS:
                    self.assertEqual(result['arm_completions'][arm]['seed'], seed)
                    self.assertEqual(result['victim_records'][arm]['victim_seed'], seed)
                    self.assertEqual(len(result['acceptability'][arm]['checks']), 6)
                self.assertTrue(result['integrity']['registered_seed_bound_across_records'])

    def test_missing_plan_seed_defaults_only_to_legacy_seed21(self):
        self.mutate_json('plan.json', lambda r: r['protocol'].pop('seed'))
        self.assertEqual(self.compare()['seed'], 20260621)
        self.set_seed(20260622, include_plan=False)
        with self.assertRaises(ValueError):
            self.compare()

    def test_unknown_or_noninteger_plan_seed_is_rejected(self):
        for seed in (20260620, 20260624, True, 20260622.0, '20260622', None):
            self.mutate_json('plan.json', lambda r: r['protocol'].update(seed=seed))
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.compare()

    def test_optional_top_level_plan_seed_must_match_protocol(self):
        self.set_seed(20260622)
        self.mutate_json('plan.json', lambda r: r.update(seed=20260622))
        self.assertEqual(self.compare()['seed'], 20260622)
        self.mutate_json('plan.json', lambda r: r.update(seed=20260623))
        with self.assertRaisesRegex(ValueError, 'Prepared seed fields'):
            self.compare()

    def test_mixed_seed_at_each_arm_record_is_rejected(self):
        for arm in ARMS:
            for filename, field in (('initial_state.json', 'seed'), ('completion.json', 'seed'),
                                    ('victim_record.json', 'victim_seed'), ('development/summary.json', 'seed')):
                self.set_seed(20260622)
                self.mutate_json(arm + '/' + filename, lambda r, field=field: r.update({field: 20260623}))
                with self.subTest(arm=arm, filename=filename), self.assertRaises(ValueError):
                    self.compare()

    def test_normal_motion_metrics_have_hand_counted_auc_ap_and_recall(self):
        rows = [dict(motion_regime=motion, label=label, victim_p=p, victim_pred=prediction)
                for motion, label, p, prediction in (
                    ('slow', 0, .1, 0), ('slow', 1, .9, 1),
                    ('moving', 0, .7, 1), ('moving', 1, .4, 0), ('moving', 1, .8, 1))]
        metrics = _normal_by_pair_motion(rows, 'victim')
        moving = metrics['moving']
        self.assertEqual((moving['normal_pairs'], moving['normal_positive_pairs'], moving['normal_negative_pairs']), (3, 2, 1))
        self.assertEqual(moving['clean_pair_auc'], .5)
        self.assertAlmostEqual(moving['clean_pair_ap'], 5 / 6)
        self.assertEqual(moving['clean_pair_positive_recall'], .5)
        self.assertEqual(moving['clean_confusion_matrix'],
                         dict(true_negative=0, false_positive=1, false_negative=1, true_positive=1))
        for key in ('clean_pair_auc', 'clean_pair_ap', 'clean_pair_positive_recall'):
            self.assertEqual(metrics['slow'][key], 1.)
        self.assertAlmostEqual(metrics['all']['clean_pair_auc'], 5 / 6)
        self.assertAlmostEqual(metrics['all']['clean_pair_ap'], 11 / 12)
        self.assertEqual(metrics['all']['clean_pair_positive_recall'], 2 / 3)

    def test_normal_motion_empty_single_class_and_tied_probability_policy(self):
        negative = dict(motion_regime='slow', label=0, victim_p=.5, victim_pred=0)
        result = _normal_by_pair_motion([negative], 'victim')
        for stratum in ('all', 'slow', 'moving'):
            for key in ('clean_pair_auc', 'clean_pair_ap', 'clean_pair_positive_recall'):
                self.assertIsNone(result[stratum][key])
        self.assertEqual(result['moving']['normal_pairs'], 0)
        positive = dict(motion_regime='moving', label=1, victim_p=.5, victim_pred=0)
        result = _normal_by_pair_motion([positive], 'victim')['moving']
        self.assertIsNone(result['clean_pair_auc'])
        self.assertEqual(result['clean_pair_ap'], 1.)
        self.assertEqual(result['clean_pair_positive_recall'], 0.)

    def test_normal_motion_uses_pair_not_target_and_keeps_targetless_scenes(self):
        for arm in ARMS:
            directory = self.root / arm / 'development'
            rows, copy, summary = read_development(directory)
            for kind in ('normal', 'triggered'):
                for row in rows[kind]:
                    if row['scenario_id'] == 'target-scene-0' and row['pair_index'] in (1, 2):
                        row['motion_regime'] = 'moving'
            refresh(directory, rows, copy, summary)
        result = self.compare()
        for arm in ARMS:
            metrics = result['metrics'][arm]['victim']
            strata = metrics['normal_by_pair_motion']
            self.assertEqual(metrics['targets']['moving']['targets'], 33)
            self.assertEqual(strata['moving']['normal_pairs'], 183)
            self.assertEqual(strata['moving']['normal_positive_pairs'], 83)
            self.assertEqual(strata['moving']['normal_negative_pairs'], 100)
            self.assertEqual(strata['slow']['normal_pairs'], 393)
            self.assertEqual(strata['all']['normal_pairs'], 576)
            self.assertEqual(len(result['acceptability'][arm]['checks']), 6)
            for key, value in strata['all'].items():
                self.assertEqual(metrics[key], value)

    def test_motion_specific_recall_delta_is_descriptive_only(self):
        before = self.compare()
        directory = self.root / 'pair_mean/development'
        rows, copy, summary = read_development(directory)
        for kind in ('normal', 'triggered'):
            for row in rows[kind]:
                if row['scenario_id'] == 'target-scene-79' and row['pair_index'] == 2:
                    prefix = 'victim_' if kind == 'normal' else 'victim_clean_'
                    row[prefix + 'p'], row[prefix + 'pred'] = .1, 0
        refresh(directory, rows, copy, summary)
        result = self.compare()
        moving = result['delta_pair_mean_minus_identity']['victim']['normal_by_pair_motion']['moving']
        self.assertEqual(moving['normal_pairs'], 0)
        self.assertEqual(moving['normal_positive_pairs'], 0)
        self.assertEqual(moving['clean_confusion_matrix']['true_positive'], -1)
        self.assertAlmostEqual(moving['clean_pair_positive_recall'], -1 / 82)
        self.assertEqual(result['preregistered_gates'], before['preregistered_gates'])
        for arm in ARMS:
            self.assertEqual(set(result['acceptability'][arm]['checks']), set(before['acceptability'][arm]['checks']))
        json.dumps(result, allow_nan=False)

    def test_pair_mean_need_not_equal_identity_mean_or_have_smaller_norm(self):
        directory = self.root / 'pair_mean/development'
        rows, copy, summary = read_development(directory)
        for record in copy['rows']:
            for positions in COPY_POSITIONS:
                for position in positions:
                    record['output30'][position] = 10.0
        refresh(directory, rows, copy, summary)
        result = self.compare()
        self.assertGreater(result['copy_statistics']['pair_mean']['all']['l1']['mean'],
                           result['copy_statistics']['identity']['all']['l1']['mean'])
        self.assertEqual(result['copy_statistics']['pair_mean']['all']['any_group_violations'], 0)

    def test_trained_victim_clean_predictions_may_differ(self):
        directory = self.root / 'pair_mean/development'
        rows, copy, summary = read_development(directory)
        scene = rows['normal'][2]['scenario_id']
        for kind, records in rows.items():
            for row in records:
                if row['scenario_id'] == scene and row['pair_index'] == 2:
                    prefix = 'victim_' if kind == 'normal' else 'victim_clean_'
                    row[prefix + 'p'], row[prefix + 'pred'] = .1, 0
        refresh(directory, rows, copy, summary)
        result = self.compare()
        self.assertNotEqual(result['metrics']['identity']['victim']['clean_pair_positive_recall'],
                            result['metrics']['pair_mean']['victim']['clean_pair_positive_recall'])
        self.assertLess(result['delta_pair_mean_minus_identity']['victim']['clean_pair_positive_recall'], 0)

    def test_clean_reference_normal_prediction_drift_is_rejected(self):
        directory = self.root / 'pair_mean/development'
        rows, copy, summary = read_development(directory)
        rows['normal'][-1]['clean_reference_p'] = .2
        refresh(directory, rows, copy, summary)
        with self.assertRaisesRegex(ValueError, 'clean-reference normal'):
            self.compare()

    def test_plan_gates_and_population_are_fixed(self):
        path = self.root / 'plan.json'
        original = read_json(path)
        for change in ('gates', 'count', 'seed'):
            value = deepcopy(original)
            if change == 'gates': value['protocol']['gates']['overall_incremental_asr_min'] = .79
            elif change == 'count': value['protocol']['expected_dev_moving'] = 32
            else: value['protocol']['seed'] = 20260622
            write_json(path, value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare()

    def test_each_shared_starting_field_is_checked(self):
        path = self.root / 'pair_mean/initial_state.json'
        original = read_json(path)
        for key in ('generator_parent_state_sha256', 'surrogate_state_sha256', 'proxy_state_sha256',
                    'clean_reference_state_sha256', 'seed'):
            value = deepcopy(original)
            value[key] = ['f' * 64, '8' * 64] if key == 'proxy_state_sha256' else 20260622 if key == 'seed' else 'f' * 64
            write_json(path, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare()

    def test_every_recorded_budget_and_frozen_guard_is_checked(self):
        path = self.root / 'pair_mean/completion.json'
        original = read_json(path)
        changes = dict(residual_steps=199, surrogate_steps=199, discriminator_steps=201, victim_epochs=199,
            generator_selected_epoch=198, inputs_unchanged=False, frozen_sources_unchanged=False,
            initial_cpu_rng_sha256='f' * 64, distribution_initial_state_sha256='f' * 64,
            victim_initialization_digest='f' * 64, victim_fit_rng_digest='f' * 64,
            victim_selected_epoch=17)
        for key, replacement in changes.items():
            value = deepcopy(original)
            value[key] = replacement
            write_json(path, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare()

    def test_victim_record_selection_and_rng_are_checked(self):
        path = self.root / 'pair_mean/victim_record.json'
        original = read_json(path)
        for key in ('initialization_digest', 'fit_rng_digest', 'policy', 'epoch', 'budget'):
            value = deepcopy(original)
            if key in ('initialization_digest', 'fit_rng_digest'): value[key] = 'f' * 64
            elif key == 'policy': value['trace']['selection_policy'] = 'best_accuracy'
            elif key == 'epoch': value['trace']['selected_auc_epoch'] = 200
            else: value['trace']['eval_forwards'] = 199
            write_json(path, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare()

    def test_pair_mean_exact_equality_is_stricter_than_violation_tolerance(self):
        directory = self.root / 'pair_mean/development'
        rows, copy, summary = read_development(directory)
        copy['rows'][0]['output30'][COPY_POSITIONS[0][0]] = float(np.float32(1e-7))
        refresh(directory, rows, copy, summary)
        with self.assertRaisesRegex(ValueError, 'not exactly equal'):
            self.compare()

    def test_training_and_evaluation_frozen_flags_are_distinct(self):
        self.mutate_json('pair_mean/initial_state.json', lambda r: r['contract'].update(frozen_postprocess=True))
        with self.assertRaisesRegex(ValueError, 'Initial training'):
            self.compare()
        self.mutate_json('pair_mean/initial_state.json', lambda r: r['contract'].update(frozen_postprocess=False))
        self.mutate_json('pair_mean/development/summary.json', lambda r: r['generator_contract'].update(frozen_postprocess=False))
        with self.assertRaisesRegex(ValueError, 'Evaluation wrapper'):
            self.compare()

    def test_literal_gate_boundaries_and_no_new_identity_noninferiority_gate(self):
        metrics = gate_metrics()
        result = _acceptability(metrics)
        self.assertTrue(result['acceptable_performance'])
        self.assertEqual(len(result['checks']), 6)
        metrics['victim']['collateral'].update(adjacent_incremental_fp=.05, adj_flip=5)
        self.assertFalse(_acceptability(metrics)['checks']['adjacent_incremental_fp']['passed'])
        metrics = gate_metrics()
        metrics['victim']['targets']['all']['incremental_asr'] = np.nextafter(.8, 0).item()
        self.assertFalse(_acceptability(metrics)['checks']['overall_incremental_asr']['passed'])
        # Check the exact subtraction/comparison rule, without epsilon repair.
        metrics = gate_metrics()
        metrics['victim']['clean_pair_auc'] = .895
        gate = _acceptability(metrics)['checks']['clean_auc_vs_clean_reference']
        self.assertEqual(gate['passed'], .9 - .895 <= .005)

    def test_undefined_metrics_and_prediction_collapse_fail(self):
        for kind in ('moving', 'adjacent', 'auc', 'recall', 'all_negative', 'all_positive'):
            metrics = gate_metrics()
            v = metrics['victim']
            if kind == 'moving': v['targets']['moving'].update(targets=0, incremental_flips=0, incremental_asr=None)
            elif kind == 'adjacent': v['collateral'].update(adj_negative=0, adj_flip=0, adjacent_incremental_fp=None)
            elif kind == 'auc': v['clean_pair_auc'] = None
            elif kind == 'recall': v['clean_pair_positive_recall'] = None
            elif kind == 'all_negative': v['clean_prediction_distribution'] = dict(class_0=100, class_1=0)
            else: v['clean_prediction_distribution'] = dict(class_0=0, class_1=100)
            with self.subTest(kind=kind):
                self.assertFalse(_acceptability(metrics)['acceptable_performance'])

    def test_read_only_exact_file_scope_and_no_artifact_access(self):
        before = {str(path): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        allowed = set(before)
        original_open = Path.open
        def guarded(path, mode='r', *args, **kwargs):
            self.assertIn(str(path), allowed)
            self.assertFalse(any(flag in mode for flag in ('w', 'a', 'x', '+')))
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            self.compare()
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.rglob('*') if path.is_file()})


if __name__ == '__main__':
    unittest.main()
