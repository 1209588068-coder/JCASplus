"""Synthetic fixed-seed queue tests; never load real assets or start training."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spear_current import copy_pilot as pilot
from spear_current import copy_training_reporting as reporting
from tests_current.test_copy_pilot import synthetic_bundle


SEEDS = (20260621, 20260622, 20260623)
COMMON_FIELDS = ('generator_parent_state_sha256', 'surrogate_state_sha256',
                 'proxy_state_sha256', 'clean_reference_state_sha256')
RANDOM_FIELDS = ('initial_cpu_rng_sha256', 'distribution_initial_state_sha256',
                 'victim_initialization_digest', 'victim_fit_rng_digest')


def comparison_fixture(seed):
    """Minimum reporter contract with real fixed denominators and paired deltas."""
    index = SEEDS.index(seed)
    overall_gain, moving_gain = (-10, -4, 2)[index], (-5, -2, 1)[index]
    metrics = {}
    for arm in pilot.ARMS:
        overall = 80 + (overall_gain if arm == 'pair_mean' else 0)
        moving = 26 + (moving_gain if arm == 'pair_mean' else 0)
        normal_moving_recall = .75 - (.125 if arm == 'pair_mean' and index != 2 else 0.)
        targets = {name: dict(targets=count, incremental_flips=flips,
                             incremental_asr=flips / count)
                   for name, count, flips in (('all', 112, overall), ('moving', 33, moving),
                                               ('slow', 79, overall - moving))}
        motion = {name: dict(normal_pairs=count, normal_positive_pairs=positive,
                            clean_pair_auc=.8, clean_pair_ap=.7,
                            clean_pair_positive_recall=recall)
                  for name, count, positive, recall in
                  (('all', 120, 20, .7), ('slow', 80, 12, 2 / 3),
                   ('moving', 40, 8, normal_moving_recall))}
        metrics[arm] = dict(victim=dict(targets=targets,
            collateral=dict(adjacent_incremental_fp=(2 if arm == 'identity' else 3) / 100,
                            adj_flip=2 if arm == 'identity' else 3, adj_negative=100,
                            nonincident_flip=0, nonincident_negative=20,
                            positive_suppression=0, positive_pairs=20),
            normal_by_pair_motion=motion, clean_pair_auc=.8, clean_pair_ap=.7,
            clean_pair_positive_recall=.7))
    initial = {key: 'shared-final-seed21-' + key for key in COMMON_FIELDS}
    completions = {arm: {key: str(seed) + '-' + key for key in RANDOM_FIELDS}
                   for arm in pilot.ARMS}
    return dict(seed=seed,
        population_identity_sha256={key: 'same-' + key
            for key in ('scenes', 'eligible_clean_pairs', 'targets', 'triggered_pairs')},
        common_initial_state=initial, arm_completions=completions, metrics=metrics,
        acceptability={arm: dict(acceptable=False, failed_gates=['moving_incremental_asr'])
                       for arm in pilot.ARMS},
        copy_statistics={arm: dict(all=dict(rows=112, violation_rate=1. if arm == 'identity' else 0.))
                         for arm in pilot.ARMS},
        paired_target_responses=dict(victim=dict(all=dict(targets=112), moving=dict(targets=33))),
        delta_pair_mean_minus_identity=dict(victim=dict(overall_incremental_count=overall_gain)))


class CopyReplicationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(pilot, 'ROOT', self.root))
        for target in ('torch.load', 'numpy.load', 'torch.Tensor.backward', 'torch.autograd.grad',
                       'torch.optim.Adam.__init__', 'torch.optim.Adam.step',
                       'torch.optim.AdamW.__init__', 'torch.optim.AdamW.step',
                       'torch.optim.SGD.__init__', 'torch.optim.SGD.step'):
            self.enterContext(patch(target, side_effect=AssertionError('Synthetic tests only: ' + target)))
        for name in ('read_inputs', 'load_normalizer', 'restore_generator', 'restore_surrogate',
                     'restore_proxy', 'restore_clean_reference', 'fit_unconditional', 'fit_fresh_victim'):
            self.enterContext(patch.object(pilot, name, side_effect=AssertionError('No actual loading/fitting')))
        self.enterContext(patch.object(pilot.subprocess, 'run', side_effect=AssertionError('No real child job')))
        self.enterContext(patch.object(pilot.subprocess, 'Popen', side_effect=AssertionError('No real launch')))
        self.bundle = synthetic_bundle(self.root)
        self.prior_root = self.root / 'results' / 'synthetic_observed_seed21'
        self.prior_root.mkdir(parents=True)
        self.write_completed(self.prior_root, comparison_fixture(SEEDS[0]))
        self.prior = dict(seed=SEEDS[0], path=str(self.prior_root),
                          role='already_observed_exploratory_pilot',
                          files={name: pilot.sha256(self.prior_root / name)
                                 for name in ('comparison.json', 'completion.json')})
        self.prior_mock = self.enterContext(patch.object(pilot, 'prior_evidence',
                                                         side_effect=lambda: deepcopy(self.prior)))
        self.prior_bytes = self.tree_bytes(self.prior_root)

    @staticmethod
    def tree_bytes(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob('*') if path.is_file()}

    @staticmethod
    def write_completed(directory, comparison, *, status='COPY_TRAINING_COMPLETE_REVIEW_REQUIRED',
                        bad_hash=False):
        (directory / 'comparison.json').write_text(json.dumps(comparison))
        (directory / 'completion.json').write_text(json.dumps(dict(
            status=status, comparison_sha256='changed' if bad_hash else pilot.sha256(directory / 'comparison.json'))))

    def prepare(self):
        self.plan = pilot.prepare_replication(self.bundle, self.root / 'results' / 'replication')
        self.spec = json.loads(self.plan.read_text())['protocol']
        return self.plan

    @staticmethod
    def resign_plan(path, value):
        path.write_text(json.dumps(value))
        registration_path = path.parent / 'registration.json'
        registration = json.loads(registration_path.read_text())
        registration['plan_sha256'] = pilot.sha256(path)
        registration_path.write_text(json.dumps(registration))

    def completed_comparisons(self):
        self.prepare()
        comparisons = {seed: comparison_fixture(seed) for seed in SEEDS}
        directories = {SEEDS[0]: self.prior_root}
        for child in self.spec['children']:
            directory = Path(child['path']).parent
            directories[child['seed']] = directory
            self.write_completed(directory, comparisons[child['seed']])
        by_path = {path: comparisons[seed] for seed, path in directories.items()}
        mocked = self.enterContext(patch.object(reporting, 'build_training_comparison',
                                                side_effect=lambda path: deepcopy(by_path[Path(path)])))
        return comparisons, mocked

    def test_prepare_registers_exact_two_seeds_with_the_same_final_seed21_parent(self):
        self.prepare()
        self.assertEqual(pilot.REPLICATION_SEEDS, SEEDS[1:])
        self.assertEqual(self.spec['prospective_seeds'], list(SEEDS[1:]))
        self.assertEqual(self.spec['parent_seed'], SEEDS[0])
        self.assertEqual(self.spec['prior_pilot'], self.prior)
        children = self.spec['children']
        self.assertEqual([child['seed'] for child in children], list(SEEDS[1:]))
        child_specs = []
        for seed, child in zip(SEEDS[1:], children):
            path = Path(child['path'])
            self.assertEqual(path, self.plan.parent / ('seed' + str(seed)) / 'plan.json')
            self.assertEqual(child['plan_sha256'], pilot.sha256(path))
            spec = pilot.validate_plan(self.bundle, path)['protocol']
            child_specs.append(spec)
            self.assertEqual(spec['seed'], seed)
            self.assertEqual(spec['parent_seed'], SEEDS[0])
            self.assertEqual(spec['arms'], ['identity', 'pair_mean'])
            self.assertEqual(spec['start_asset_keys'], {'generator': 'generator', 'surrogate': 'surrogate'})
            self.assertNotIn('start_generator', spec['assets'])
            self.assertNotIn('start_surrogate', spec['assets'])
            self.assertEqual(spec['seeds'], pilot.training_seeds(seed))
            self.assertEqual(spec['effective_distribution_config']['seed'], pilot.training_seeds(seed)['discriminator'])
            self.assertEqual(spec['generator_checkpoint'], 'final_epoch_199')
            self.assertEqual(spec['victim_epochs'], 200)
            for key, expected in (('train_scenes', 512), ('dev_scenes', 128), ('poison_attachments', 26),
                                  ('outer_rows', 538), ('expected_dev_targets', 112), ('expected_dev_moving', 33)):
                self.assertEqual(spec[key], expected)
        self.assertEqual(child_specs[0]['assets'], child_specs[1]['assets'])
        self.assertNotEqual(child_specs[0]['seeds'], child_specs[1]['seeds'])
        self.assertEqual(self.prior_bytes, self.tree_bytes(self.prior_root))
        self.assertFalse(json.loads((self.plan.parent / 'registration.json').read_text())['training_started'])

    def test_prepare_never_overwrites_registered_queue_or_prior(self):
        self.prepare()
        before = self.tree_bytes(self.plan.parent)
        with self.assertRaises((ValueError, FileExistsError)):
            pilot.prepare_replication(self.bundle, self.plan.parent)
        self.assertEqual(before, self.tree_bytes(self.plan.parent))
        self.assertEqual(self.prior_bytes, self.tree_bytes(self.prior_root))

    def test_root_plan_tampering_rejected_even_when_registration_is_resigned(self):
        self.prepare()
        original = json.loads(self.plan.read_text())
        mutations = [lambda s: s.update(prospective_seeds=[20260623, 20260622]),
                     lambda s: s.update(parent_seed=20260622),
                     lambda s: s['children'].reverse(),
                     lambda s: s['children'][0].update(seed=20260621),
                     lambda s: s['children'][0].update(path=str(self.prior_root / 'plan.json')),
                     lambda s: s['children'][0].update(plan_sha256='changed'),
                     lambda s: s.update(prior_pilot={**self.prior, 'role': 'prospective_repetition'})]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                modified = deepcopy(original)
                mutation(modified['protocol'])
                self.resign_plan(self.plan, modified)
                with self.assertRaises(ValueError):
                    pilot.validate_replication_plan(self.bundle, self.plan)
        self.resign_plan(self.plan, original)
        pilot.validate_replication_plan(self.bundle, self.plan)

    def test_modified_child_bytes_and_changed_child_parent_are_rejected(self):
        self.prepare()
        child = Path(self.spec['children'][0]['path'])
        original = child.read_bytes()
        child.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'child'):
            pilot.validate_replication_plan(self.bundle, self.plan)
        modified = json.loads(original)
        modified['protocol']['parent_seed'] = 20260622
        self.resign_plan(child, modified)
        parent = json.loads(self.plan.read_text())
        parent['protocol']['children'][0]['plan_sha256'] = pilot.sha256(child)
        self.resign_plan(self.plan, parent)
        with self.assertRaises(ValueError):
            pilot.validate_replication_plan(self.bundle, self.plan)

    def test_changed_prior_binding_is_rejected_before_any_launch(self):
        self.prepare()
        self.prior['files']['comparison.json'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'prior evidence'):
            pilot.launch(self.bundle, self.plan, replication=True)
        self.assertFalse((self.plan.parent / 'launch_claim.json').exists())

    def test_launch_starts_one_detached_queue_and_cannot_be_repeated(self):
        self.prepare()
        with patch.object(pilot.subprocess, 'Popen', return_value=SimpleNamespace(pid=123)) as process:
            record = pilot.launch(self.bundle, self.plan, replication=True)
            process.assert_called_once()
            command = process.call_args.args[0]
            self.assertEqual(command, [sys.executable, '-B', '-m', 'spear_current.copy_pilot',
                                       '--run', '--plan', str(self.plan), '--replication'])
            self.assertTrue(process.call_args.kwargs['start_new_session'])
            self.assertEqual(process.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(process.call_args.kwargs['cwd'], str(self.root))
            self.assertEqual(record['pid'], 123)
            self.assertFalse(record['automatic_followup'])
            with self.assertRaises((ValueError, FileExistsError)):
                pilot.launch(self.bundle, self.plan, replication=True)
            process.assert_called_once()
        self.assertEqual(self.prior_bytes, self.tree_bytes(self.prior_root))

    def test_execution_claim_prevents_background_launch(self):
        self.prepare()
        (self.plan.parent / 'execution_claim.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'claimed'):
            pilot.launch(self.bundle, self.plan, replication=True)
        self.assertFalse((self.plan.parent / 'launch_claim.json').exists())

    def test_queue_is_serial_and_performance_failure_does_not_remove_second_seed(self):
        self.prepare()
        seen = []
        def child_run(command, **kwargs):
            child = Path(command[-1])
            seed = json.loads(child.read_text())['protocol']['seed']
            self.assertEqual(command, [sys.executable, '-B', '-m', 'spear_current.copy_pilot',
                                       '--run', '--plan', str(child)])
            self.assertEqual(seed, SEEDS[len(seen) + 1])
            self.assertEqual(kwargs['cwd'], str(self.root))
            self.assertFalse(kwargs['check'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
            if seen:
                first = Path(self.spec['children'][0]['path']).parent
                self.assertTrue((first / 'completion.json').exists())
            comparison = comparison_fixture(seed)
            self.assertFalse(comparison['acceptability']['pair_mean']['acceptable'])
            self.write_completed(child.parent, comparison)
            seen.append(seed)
            return SimpleNamespace(returncode=0)
        summary = dict(status='COPY_REPLICATION_COMPLETE_REVIEW_REQUIRED', synthetic_only=True)
        with patch.object(pilot.subprocess, 'run', side_effect=child_run) as mocked_run, \
                patch.object(pilot, 'build_replication_summary', return_value=summary) as report, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.run_replication(self.bundle, self.plan), summary)
            self.assertEqual(mocked_run.call_count, 2)
            report.assert_called_once_with(self.bundle, self.plan)
            with self.assertRaises((ValueError, FileExistsError)):
                pilot.run_replication(self.bundle, self.plan)
            self.assertEqual(mocked_run.call_count, 2)
        self.assertEqual(seen, list(SEEDS[1:]))
        completion = json.loads((self.plan.parent / 'completion.json').read_text())
        self.assertEqual(completion['completed_seeds'], list(SEEDS[1:]))
        self.assertEqual(completion['summary_sha256'], pilot.sha256(self.plan.parent / 'summary.json'))
        progress = [json.loads(line) for line in (self.plan.parent / 'progress.jsonl').read_text().splitlines()]
        self.assertEqual([row['stage'] for row in progress],
                         ['seed_started', 'seed_complete', 'seed_started', 'seed_complete', 'replication_complete'])
        self.assertEqual(self.prior_bytes, self.tree_bytes(self.prior_root))

    def test_nonzero_child_exit_stops_without_starting_second_seed(self):
        self.prepare()
        with patch.object(pilot.subprocess, 'run', return_value=SimpleNamespace(returncode=7)) as child_run, \
                patch.object(pilot, 'build_replication_summary') as report, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, '20260622.*7'):
                pilot.run_replication(self.bundle, self.plan)
            child_run.assert_called_once()
            report.assert_not_called()
        failure = json.loads((self.plan.parent / 'failure.json').read_text())
        self.assertEqual(failure['status'], 'FAILED_NO_FURTHER_JOBS')
        self.assertEqual(failure['completed_seeds'], [])
        self.assertFalse((Path(self.spec['children'][1]['path']).parent / 'terminal.log').exists())
        self.assertFalse((self.plan.parent / 'completion.json').exists())

    def test_corrupt_first_completion_stops_fixed_queue(self):
        self.prepare()
        def corrupted_completion(command, **_):
            self.write_completed(Path(command[-1]).parent, comparison_fixture(SEEDS[1]), bad_hash=True)
            return SimpleNamespace(returncode=0)
        with patch.object(pilot.subprocess, 'run', side_effect=corrupted_completion) as child_run, \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                pilot.run_replication(self.bundle, self.plan)
            child_run.assert_called_once()
        self.assertEqual(json.loads((self.plan.parent / 'failure.json').read_text())['completed_seeds'], [])
        self.assertFalse((Path(self.spec['children'][1]['path']).parent / 'terminal.log').exists())

    def test_preclaimed_child_is_not_automatically_retried(self):
        self.prepare()
        first = Path(self.spec['children'][0]['path']).parent
        (first / 'launch_claim.json').write_text('{}')
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'no auto-retry'):
            pilot.run_replication(self.bundle, self.plan)
        self.assertFalse((first / 'terminal.log').exists())
        self.assertEqual(json.loads((self.plan.parent / 'failure.json').read_text())['completed_seeds'], [])

    def test_summary_separates_two_prospective_repeats_from_old_observed_pilot(self):
        comparisons, reporter = self.completed_comparisons()
        before = {seed: self.tree_bytes(self.prior_root if seed == SEEDS[0] else
                    Path(self.spec['children'][SEEDS.index(seed) - 1]['path']).parent) for seed in SEEDS}
        summary = pilot.build_replication_summary(self.bundle, self.plan)
        self.assertEqual(reporter.call_count, 3)
        self.assertEqual(summary['status'], 'COPY_REPLICATION_COMPLETE_REVIEW_REQUIRED')
        self.assertEqual([row['seed'] for row in summary['per_seed']], list(SEEDS))
        self.assertEqual([row['role'] for row in summary['per_seed']],
                         ['prior_observed_pilot', 'prospective_repetition', 'prospective_repetition'])
        prospective = summary['prospective_two_seed_summary']
        all_three = summary['descriptive_all_three_seed_summary']
        self.assertEqual(prospective['seeds'], list(SEEDS[1:]))
        self.assertEqual(all_three['seeds'], list(SEEDS))
        delta = prospective['paired_deltas']['overall_incremental_asr']
        self.assertAlmostEqual(delta['per_seed'][0], -4 / 112)
        self.assertAlmostEqual(delta['per_seed'][1], 2 / 112)
        self.assertAlmostEqual(delta['mean'], -1 / 112)
        self.assertAlmostEqual(delta['sample_sd'], math.sqrt(18) / 112)
        self.assertEqual((delta['positive'], delta['negative'], delta['zero']), (1, 1, 0))
        self.assertAlmostEqual(all_three['paired_deltas']['overall_incremental_asr']['mean'], -4 / 112)
        moving_recall = prospective['paired_deltas']['normal_moving_positive_recall']
        self.assertEqual(moving_recall['per_seed'], [-.125, 0.])
        self.assertFalse(prospective['statistical_significance_claimed'])
        self.assertFalse(prospective['equivalence_claimed'])
        self.assertEqual(summary['distinct_target_count'], 112)
        self.assertEqual(summary['distinct_moving_target_count'], 33)
        for key in ('old_results_rewritten', 'new_model_inference', 'independent_confirmation_claim',
                    'automatic_followup', 'automatic_promotion'):
            self.assertFalse(summary[key])
        for row in summary['per_seed']:
            self.assertEqual(row['full_metrics'], comparisons[row['seed']]['metrics'])
            self.assertEqual(row['acceptability'], comparisons[row['seed']]['acceptability'])
            directory = self.prior_root if row['seed'] == SEEDS[0] else \
                Path(self.spec['children'][SEEDS.index(row['seed']) - 1]['path']).parent
            self.assertEqual(before[row['seed']], self.tree_bytes(directory))

    def test_summary_rejects_cross_seed_population_or_parent_changes(self):
        comparisons, _ = self.completed_comparisons()
        base = deepcopy(comparisons[SEEDS[2]])
        for field in ('population_identity_sha256', *COMMON_FIELDS):
            with self.subTest(field=field):
                current = comparisons[SEEDS[2]]
                if field == 'population_identity_sha256':
                    current[field]['targets'] = 'different-targets'
                else:
                    current['common_initial_state'][field] = 'different-parent'
                with self.assertRaisesRegex(ValueError, 'population|starting states'):
                    pilot.build_replication_summary(self.bundle, self.plan)
                current.clear()
                current.update(deepcopy(base))

    def test_summary_requires_each_registered_seed_to_change_all_four_rng_states(self):
        comparisons, _ = self.completed_comparisons()
        for field in RANDOM_FIELDS:
            with self.subTest(field=field):
                current = comparisons[SEEDS[2]]['arm_completions']['identity']
                original = current[field]
                current[field] = comparisons[SEEDS[1]]['arm_completions']['identity'][field]
                with self.assertRaisesRegex(ValueError, field):
                    pilot.build_replication_summary(self.bundle, self.plan)
                current[field] = original

    def test_summary_rejects_misreported_seed(self):
        comparisons, _ = self.completed_comparisons()
        comparisons[SEEDS[2]]['seed'] = SEEDS[1]
        with self.assertRaisesRegex(ValueError, 'Reported seed'):
            pilot.build_replication_summary(self.bundle, self.plan)

    def test_summary_keeps_undefined_normal_motion_recall_undefined(self):
        comparisons, _ = self.completed_comparisons()
        comparisons[SEEDS[1]]['metrics']['pair_mean']['victim']['normal_by_pair_motion']['moving'][
            'clean_pair_positive_recall'] = None
        summary = pilot.build_replication_summary(self.bundle, self.plan)
        delta = summary['prospective_two_seed_summary']['paired_deltas']['normal_moving_positive_recall']
        self.assertEqual(delta['per_seed'], [None, 0.])
        self.assertIsNone(delta['mean'])
        self.assertIsNone(delta['sample_sd'])
        self.assertEqual((delta['positive'], delta['negative'], delta['zero']), (0, 0, 1))

    def test_summary_rejects_completed_comparison_byte_tampering(self):
        self.completed_comparisons()
        child = Path(self.spec['children'][1]['path']).parent
        comparison_path = child / 'comparison.json'
        comparison_path.write_bytes(comparison_path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'altered completed child comparison'):
            pilot.build_replication_summary(self.bundle, self.plan)


if __name__ == '__main__':
    unittest.main()
