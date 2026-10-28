"""Synthetic reverse-pilot orchestration checks; no graph/model loads or fitting."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from spear_current import reverse_pilot as pilot
from spear_current.reverse_audit import select_positive_targets


def population_fixture():
    size = 2600
    labels = torch.zeros(size, dtype=torch.long)
    labels[1902:] = 1
    train = dict(x=torch.arange(size * 4, dtype=torch.float32).reshape(size, 4) / 10000,
                 y=labels, eligible=torch.ones(size, dtype=torch.bool))
    sources = {role: [dict(scenario_id=role + '-scene', split_group_id=role + '-group',
                          split='train', role=role)] for role in ('train', 'dev')}

    def row(role, index, label):
        return dict(scenario_id=role + '-scene', split_group_id=role + '-group', role=role,
            pair_index=index, role_pair_index=index, src_track_id='track-a', dst_track_id=f'track-{index}',
            true_label=label, eligible=label >= 0,
            motion_regime='moving' if index % 5 == 0 else 'slow',
            is_attachment=role == 'train' and 1876 <= index < 1902,
            is_outer=role == 'train' and 1876 <= index < 2414)

    metadata = dict(train=[row('train', i, int(labels[i])) for i in range(size)],
                    dev=[row('dev', i, label) for i, label in enumerate((1, 0, 1, -1))])
    indices = dict(idx_train=torch.arange(1876), idx_attach=torch.arange(1876, 1902),
                   idx_unlabeled=torch.arange(1902, size), train_labels=labels.clone(),
                   dev_indices=torch.tensor([0, 1, 2]))
    inputs = dict(combined=dict(train=train), metadata=metadata, indices=indices)
    targets = select_positive_targets(metadata, sources)
    phase0 = dict(target_hash=pilot._json_hash(targets),
        candidate_counts=dict(all=dict(n=size - 1902)),
        candidate_hash=pilot._json_hash(metadata['train'][1902:]))
    return inputs, sources, targets['dev'], phase0


def target_counts(n, detected, induced, repaired=0):
    misses = n - detected
    rate = lambda a, b: a / b if b else None
    return dict(n=n, clean_detected=detected, baseline_misses=misses,
        induced_misses_1to0=induced, corrected_misses_0to1=repaired,
        triggered_misses=misses + induced - repaired,
        baseline_fnr=rate(misses, n), absolute_suppression=rate(misses + induced - repaired, n),
        incremental_asr=rate(induced, n), conditional_suppression=rate(induced, detected),
        net_fnr_change=rate(induced - repaired, n), clean_recall_ceiling=rate(detected, n))


def set_targets(model, *, n=10, detected=6, induced=5, repaired=0):
    model['targets'] = dict(all=target_counts(2 * n, 2 * detected, 2 * induced, 2 * repaired),
        slow=target_counts(n, detected, induced, repaired), moving=target_counts(n, detected, induced, repaired))


def set_normal(model, *, tp=20, tn=50, auc=.9):
    fn, fp = 40 - tp, 60 - tn
    model.update(normal_pairs=100, normal_positive_pairs=40, normal_negative_pairs=60,
        clean_confusion_matrix=dict(true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp),
        clean_prediction_distribution=dict(class_0=tn + fn, class_1=fp + tp),
        clean_pair_auc=auc, clean_pair_positive_recall=tp / 40)


def evaluation_fixture():
    metrics = {}
    for name in ('victim', 'matched_clean', 'label_only'):
        model = dict(collateral={relation: dict(positive_pairs=40, induced_misses_1to0=1,
            negative_pairs=40, negative_activations_0to1=1,
            positive_suppression=.025, negative_activation=.025)
            for relation in ('adjacent', 'nonincident')})
        set_normal(model)
        set_targets(model, induced=5 if name == 'victim' else 1)
        metrics[name] = model
    return dict(status='REVERSE_DEV_EVALUATION_COMPLETE', metrics=metrics,
        dev_counts=dict(registered_scenes=25, processed_scenes=25,
            scenes_without_eligible_positive_target=5, all_pair_nodes=120,
            supervised_pairs=100, targets=20, triggered_supervised_pair_rows=80))


class ReversePopulationTests(unittest.TestCase):
    def test_prepare_builds_new_membership_preserving_all_source_indices_and_metadata(self):
        inputs, sources, targets, phase0 = population_fixture()
        old_metadata, old_indices = deepcopy(inputs['metadata']), deepcopy(inputs['indices'])
        idx, outer, report = pilot.prepare_populations(inputs, sources, targets, phase0)
        self.assertEqual((len(idx['idx_attach']), len(outer), len(torch.cat([idx['idx_train'], idx['idx_attach']]).unique())),
                         (26, 538, 1902))
        self.assertTrue(torch.equal(outer[:26], idx['idx_attach']))
        self.assertTrue(bool((inputs['combined']['train']['y'][outer] == 1).all()))
        self.assertFalse(bool(torch.isin(outer, idx['idx_train']).any()))
        self.assertFalse(bool(torch.isin(outer, old_indices['idx_attach']).any()))
        self.assertEqual(report['dev_targets'], targets)
        self.assertEqual(report['target_identity_sha256'], pilot._json_hash(targets))
        self.assertEqual(sum(report['motion_counts'].values()), 538)
        self.assertEqual([row['is_attachment'] for row in report['rows']], [True] * 26 + [False] * 512)
        self.assertTrue(all(row['is_outer'] for row in report['rows']))
        self.assertEqual(inputs['metadata'], old_metadata)
        for key, value in old_indices.items():
            self.assertTrue(torch.equal(inputs['indices'][key], value), key)
            if key != 'idx_attach':
                self.assertTrue(torch.equal(idx[key], value), key)
            self.assertNotEqual(idx[key].data_ptr(), inputs['indices'][key].data_ptr(), key)
        self.assertNotEqual(idx['idx_attach'].data_ptr(), outer.data_ptr())

    def test_phase0_target_or_candidate_drift_rejected(self):
        for change in ('target_row', 'target_hash', 'candidate_hash', 'candidate_count'):
            inputs, sources, targets, phase0 = population_fixture()
            if change == 'target_row':
                targets[0]['dst_track_id'] = 'wrong-track'
            elif change in ('target_hash', 'candidate_hash'):
                phase0[change] = '0' * 64
            else:
                phase0['candidate_counts']['all']['n'] -= 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                pilot.prepare_populations(inputs, sources, targets, phase0)

    def test_controls_use_identical1902_ids_features_and_only_labelonly_flips_new26(self):
        inputs, sources, targets, phase0 = population_fixture()
        idx, outer, _ = pilot.prepare_populations(inputs, sources, targets, phase0)
        prepared = dict(train=inputs['combined']['train'], indices=idx, outer=outer)
        original_x, original_y = prepared['train']['x'].clone(), prepared['train']['y'].clone()
        before_indices = deepcopy(idx)
        clean = pilot.supervision_payload(prepared, 'matched_clean')
        label_only = pilot.supervision_payload(prepared, 'label_only')
        self.assertEqual(len(torch.cat([idx['idx_train'], idx['idx_attach']]).unique()), 1902)
        self.assertTrue(torch.equal(clean['poisoned_features'], original_x))
        self.assertTrue(torch.equal(label_only['poisoned_features'], original_x))
        self.assertTrue(torch.equal(clean['poisoned_labels'], original_y))
        changed = torch.nonzero(label_only['poisoned_labels'] != original_y).flatten()
        self.assertEqual(set(changed.tolist()), set(idx['idx_attach'].tolist()))
        self.assertTrue(bool((label_only['poisoned_labels'][idx['idx_attach']] == 0).all()))
        self.assertTrue(torch.equal(prepared['train']['x'], original_x))
        self.assertTrue(torch.equal(prepared['train']['y'], original_y))
        for key in idx:
            self.assertTrue(torch.equal(idx[key], before_indices[key]), key)
        for payload in (clean, label_only):
            self.assertNotEqual(payload['poisoned_features'].data_ptr(), original_x.data_ptr())
            self.assertNotEqual(payload['poisoned_labels'].data_ptr(), prepared['train']['y'].data_ptr())
        clean['poisoned_features'].zero_()
        label_only['poisoned_labels'].fill_(-1)
        self.assertTrue(torch.equal(prepared['train']['x'], original_x))
        self.assertTrue(torch.equal(prepared['train']['y'], original_y))

    def test_bad_control_counts_duplicates_eligibility_labels_and_arm_rejected(self):
        for change in ('fit_count', 'attach_count', 'fit_duplicate', 'attach_duplicate',
                       'overlap', 'ineligible', 'negative_attachment', 'out_of_range', 'float_index'):
            inputs, sources, targets, phase0 = population_fixture()
            idx, _, _ = pilot.prepare_populations(inputs, sources, targets, phase0)
            prepared = dict(train=inputs['combined']['train'], indices=idx)
            if change == 'fit_count':
                idx['idx_train'] = idx['idx_train'][:-1]
            elif change == 'attach_count':
                idx['idx_attach'] = idx['idx_attach'][:-1]
            elif change == 'fit_duplicate':
                idx['idx_train'][0] = idx['idx_train'][1]
            elif change == 'attach_duplicate':
                idx['idx_attach'][0] = idx['idx_attach'][1]
            elif change == 'overlap':
                idx['idx_attach'][0] = idx['idx_train'][0]
            elif change == 'ineligible':
                prepared['train']['eligible'][idx['idx_attach'][0]] = False
            elif change == 'negative_attachment':
                prepared['train']['y'][idx['idx_attach'][0]] = 0
            elif change == 'out_of_range':
                idx['idx_attach'][0] = 2600
            else:
                idx['idx_attach'] = idx['idx_attach'].float()
            with self.subTest(change=change), self.assertRaises(ValueError):
                pilot.supervision_payload(prepared, 'label_only')
        with self.assertRaises(ValueError):
            pilot.supervision_payload({}, 'reverse_trigger')


class ReverseReviewTests(unittest.TestCase):
    def test_valid_summary_completes_pure_gates_without_automatic_jobs(self):
        summary = evaluation_fixture()
        previous = deepcopy(summary)
        with patch.object(pilot, 'run', side_effect=AssertionError('automatic job')), \
                patch.object(pilot, 'read_inputs', side_effect=AssertionError('graph read')), \
                patch.object(pilot, 'fit_fresh_victim', side_effect=AssertionError('training')):
            result = pilot.review_evaluation(summary)
        self.assertEqual(result['status'], 'REVERSE_PILOT_COMPLETE_REVIEW_REQUIRED')
        self.assertTrue(result['acceptable_performance'])
        self.assertTrue(all(row['passed'] for row in result['gates'].values()))
        self.assertFalse(result['automatic_followup_jobs'])
        self.assertFalse(result['multiseed_evidence'])
        self.assertFalse(result['statistical_significance_claimed'])
        self.assertEqual(summary, previous)
        json.dumps(result, allow_nan=False)

    def test_conditional_suppression_not_overall_asr_and_minimum_detections_required(self):
        summary = evaluation_fixture()
        set_targets(summary['metrics']['victim'], detected=5, induced=4)
        result = pilot.review_evaluation(summary)
        self.assertTrue(result['acceptable_performance'])
        self.assertEqual(result['gates']['conditional_suppression']['value'], .8)
        self.assertEqual(summary['metrics']['victim']['targets']['all']['incremental_asr'], .4)
        set_targets(summary['metrics']['victim'], detected=4, induced=4)
        result = pilot.review_evaluation(summary)
        self.assertFalse(result['gates']['clean_detected_targets']['passed'])
        self.assertFalse(result['acceptable_performance'])

    def test_both_control_gains_and_net_miss_repair_are_required(self):
        summary = evaluation_fixture()
        set_targets(summary['metrics']['label_only'], induced=5)
        result = pilot.review_evaluation(summary)
        self.assertFalse(result['gates']['incremental_asr_gain_vs_label_only']['passed'])
        self.assertTrue(result['gates']['incremental_asr_gain_vs_matched_clean']['passed'])
        summary = evaluation_fixture()
        set_targets(summary['metrics']['victim'], repaired=4)
        result = pilot.review_evaluation(summary)
        self.assertTrue(result['gates']['incremental_asr_gain_vs_matched_clean']['passed'])
        self.assertFalse(result['gates']['net_fnr_change_gain_vs_matched_clean']['passed'])
        self.assertFalse(result['acceptable_performance'])

    def test_clean_auc_boundary_is_inclusive_but_collateral_boundary_is_strict(self):
        summary = evaluation_fixture()
        set_normal(summary['metrics']['victim'], tp=18, auc=.895)
        result = pilot.review_evaluation(summary)
        self.assertTrue(result['gates']['clean_pair_auc_difference_vs_matched']['passed'])
        self.assertTrue(result['gates']['clean_pair_positive_recall_difference_vs_matched']['passed'])
        summary['metrics']['victim']['collateral']['adjacent'].update(
            induced_misses_1to0=2, positive_suppression=.05)
        result = pilot.review_evaluation(summary)
        self.assertFalse(result['gates']['adjacent_positive_suppression']['passed'])
        self.assertFalse(result['acceptable_performance'])

    def test_null_target_clean_auc_and_collateral_denominators_fail_without_fallback(self):
        summary = evaluation_fixture()
        set_targets(summary['metrics']['victim'], detected=0, induced=0)
        summary['metrics']['victim']['clean_pair_auc'] = None
        for model in summary['metrics'].values():
            model['collateral']['nonincident'].update(positive_pairs=0, induced_misses_1to0=0,
                                                     positive_suppression=None)
        result = pilot.review_evaluation(summary)
        for key in ('conditional_suppression', 'clean_detected_targets',
                    'clean_pair_auc_difference_vs_matched', 'nonincident_positive_suppression'):
            self.assertFalse(result['gates'][key]['passed'], key)
        self.assertIsNone(result['gates']['conditional_suppression']['value'])
        self.assertFalse(result['automatic_followup_jobs'])

    def test_no_moving_population_is_descriptive_not_an_extra_gate(self):
        summary = evaluation_fixture()
        for model in summary['metrics'].values():
            model['targets']['slow'] = deepcopy(model['targets']['all'])
            model['targets']['moving'] = target_counts(0, 0, 0)
        result = pilot.review_evaluation(summary)
        self.assertTrue(result['acceptable_performance'])
        self.assertFalse(any('moving' in key for key in result['gates']))

    def test_drifted_raw_counts_rates_strata_and_normal_pools_rejected(self):
        changes = (
            lambda s: s.update(status='INCOMPLETE'),
            lambda s: s['metrics']['victim']['targets']['all'].update(incremental_asr=.51),
            lambda s: s['metrics']['victim']['targets']['all'].update(conditional_suppression=None),
            lambda s: s['metrics']['victim']['targets']['all'].update(net_fnr_change=float('nan')),
            lambda s: s['metrics']['victim']['targets']['all'].update(n=True),
            lambda s: s['metrics']['victim']['targets']['slow'].update(clean_detected=-1),
            lambda s: s['metrics']['label_only']['targets'].update(moving=target_counts(11, 6, 1)),
            lambda s: s['metrics']['label_only']['targets'].update(all=target_counts(21, 12, 2)),
            lambda s: s['metrics']['matched_clean'].update(normal_pairs=101),
            lambda s: s['metrics']['victim'].update(clean_pair_positive_recall=.6),
            lambda s: s['metrics']['victim']['clean_prediction_distribution'].update(class_0=99),
        )
        for number, change in enumerate(changes):
            summary = evaluation_fixture()
            change(summary)
            with self.subTest(change=number), self.assertRaises(ValueError):
                pilot.review_evaluation(summary)

    def test_invalid_collateral_counts_rates_or_cross_model_denominators_rejected(self):
        changes = (
            dict(induced_misses_1to0=-1, positive_suppression=-.025),
            dict(induced_misses_1to0=41, positive_suppression=41 / 40),
            dict(induced_misses_1to0=1.0), dict(positive_pairs=-40, positive_suppression=-.025),
            dict(positive_pairs=True), dict(positive_suppression=float('nan')),
            dict(positive_suppression=.1), dict(positive_pairs=0, induced_misses_1to0=1, positive_suppression=None),
        )
        for update in changes:
            summary = evaluation_fixture()
            summary['metrics']['victim']['collateral']['adjacent'].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                pilot.review_evaluation(summary)
        summary = evaluation_fixture()
        summary['metrics']['label_only']['collateral']['adjacent'].update(positive_pairs=80, positive_suppression=.0125)
        with self.assertRaises(ValueError):
            pilot.review_evaluation(summary)

    def test_noncollapsed_normal_predictions_and_matched_control_quality_required(self):
        for model_name in ('victim', 'matched_clean'):
            summary = evaluation_fixture()
            set_normal(summary['metrics'][model_name], tp=40, tn=0)
            result = pilot.review_evaluation(summary)
            self.assertFalse(result['gates'][model_name + '_normal_prediction_classes']['passed'])
            self.assertFalse(result['acceptable_performance'])
        summary = evaluation_fixture()
        set_normal(summary['metrics']['matched_clean'], tp=1, auc=.74)
        result = pilot.review_evaluation(summary)
        self.assertFalse(result['gates']['matched_clean_auc']['passed'])
        self.assertFalse(result['gates']['matched_clean_positive_recall']['passed'])


class ReversePilotCliTests(unittest.TestCase):
    def test_default_and_check_only_cli_do_not_load_graphs_or_start_training(self):
        bundle = SimpleNamespace()
        for argv in ([], ['--check-only']):
            with self.subTest(argv=argv), ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                load = stack.enter_context(patch.object(pilot, 'load_bundle', return_value=bundle))
                preflight = stack.enter_context(patch.object(pilot, 'preflight', return_value=dict(status='OK')))
                for name in ('run', 'read_inputs', 'fit_generator', 'fit_fresh_victim', 'restore_generator'):
                    stack.enter_context(patch.object(pilot, name, side_effect=AssertionError(name)))
                self.assertEqual(pilot.main(argv), dict(status='OK'))
                load.assert_called_once_with()
                preflight.assert_called_once_with(bundle)

    def test_run_requires_explicit_new_output_path(self):
        with patch.object(pilot, 'load_bundle') as load, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                pilot.main(['--run'])
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()

    def test_output_scope_and_no_overwrite_checked_before_preflight_or_graph_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'results'
            root.mkdir()
            existing = root / 'existing'
            existing.mkdir()
            outside = Path(directory) / 'outside'
            outside.mkdir()
            (root / 'redirect').symlink_to(outside, target_is_directory=True)
            with patch.object(pilot, 'OUTPUT_ROOT', root), \
                    patch.object(pilot, 'frozen_plan', side_effect=AssertionError('preflight reached')), \
                    patch.object(pilot, 'read_inputs', side_effect=AssertionError('graph read')):
                for output in (root, existing, outside / 'new', root / 'redirect' / 'new'):
                    with self.subTest(output=output), self.assertRaises(ValueError):
                        pilot.run(None, output)
            self.assertTrue(existing.is_dir())
            self.assertFalse((outside / 'new').exists())


if __name__ == '__main__':
    unittest.main()
