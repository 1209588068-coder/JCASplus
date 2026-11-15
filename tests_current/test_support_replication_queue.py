"""Meaningful matrix and decision guards; no graph arrays or model training."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from spear_current import support_replication as experiment


def report():
    victim = dict(
        targets=dict(all=dict(incremental_asr=.90, absolute_asr=.99),
                     moving=dict(incremental_asr=.90, absolute_asr=.99)),
        clean_pair_auc=.91,
        clean_pair_positive_recall=.50,
        clean_prediction_distribution=dict(class_0=100, class_1=20),
        all_clean_predictions_negative=False,
        normal_model_safeguard=dict(collapsed_prediction_distribution=False),
        collateral=dict(adjacent=dict(incremental_fp=.01, positive_suppression=0.),
                        nonincident=dict(incremental_fp=.001)),
    )
    reference = dict(clean_pair_auc=.91, clean_pair_positive_recall=.50,
                     targets=dict(all=dict(incremental_asr=.01)))
    return dict(models=dict(victim=victim, clean_reference=reference))


class SupportReplicationDecisionTests(unittest.TestCase):
    def test_all_preexisting_random_supports_cross_all_victim_seeds(self):
        cases = experiment.case_matrix()
        actual = {(case['seed'], case['arm']) for case in cases}
        expected = {(seed, arm)
                    for seed in (20260621, 20260622, 20260623)
                    for arm in ('semantic', 'random_semantic_s20260621',
                                'random_semantic_s20260622', 'random_semantic_s20260623')}
        self.assertEqual(actual, expected)
        self.assertEqual(len(cases), len(expected))

    def test_acceptable_response_passes_without_inventing_extra_gate(self):
        self.assertTrue(experiment.acceptability(report())['passed'])

    def test_high_absolute_asr_cannot_replace_incremental_asr(self):
        sample = report()
        sample['models']['victim']['targets']['all']['incremental_asr'] = .84
        sample['models']['victim']['targets']['all']['absolute_asr'] = 1.
        self.assertFalse(experiment.acceptability(sample)['passed'])

    def test_moving_failure_cannot_be_hidden_by_overall_success(self):
        sample = report()
        sample['models']['victim']['targets']['moving']['incremental_asr'] = .84
        self.assertFalse(experiment.acceptability(sample)['passed'])

    def test_five_percent_adjacent_fp_is_not_strictly_below_limit(self):
        sample = report()
        sample['models']['victim']['collateral']['adjacent']['incremental_fp'] = .05
        self.assertFalse(experiment.acceptability(sample)['passed'])

    def test_auc_and_recall_are_compared_to_same_pool_reference(self):
        for key, value in (('clean_pair_auc', .904),
                           ('clean_pair_positive_recall', .449)):
            with self.subTest(metric=key):
                sample = report()
                sample['models']['victim'][key] = value
                self.assertFalse(experiment.acceptability(sample)['passed'])

    def test_high_auc_does_not_excuse_single_class_predictions(self):
        sample = report()
        victim = sample['models']['victim']
        victim['clean_pair_auc'] = .99
        victim['clean_prediction_distribution'] = dict(class_0=0, class_1=120)
        victim['normal_model_safeguard']['collapsed_prediction_distribution'] = True
        self.assertFalse(experiment.acceptability(sample)['passed'])

    def test_gate_evaluation_does_not_mutate_recorded_metrics(self):
        sample = report()
        before = deepcopy(sample)
        experiment.acceptability(sample)
        self.assertEqual(sample, before)


class SupportReplicationIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        from spear_current.support_schema import LEGAL_COORDINATES, SELECTED_COORDINATES
        output = root / 'results' / 'synthetic-registration'
        output.mkdir(parents=True)
        source = root / 'runtime.py'
        source.write_text('synthetic frozen source\n')
        cache = root / 'input.bin'
        cache.write_bytes(b'synthetic cache, never deserialized')
        snapshot = output / 'source_snapshot.tar'
        snapshot.write_bytes(b'synthetic source receipt, never extracted')
        active = set(LEGAL_COORDINATES) - {708, 743, 776, 760}
        support = dict(selected_coordinates=list(SELECTED_COORDINATES),
                       hard_support=[c in active for c in SELECTED_COORDINATES],
                       hard_support_capacity=24)
        plan = dict(version=experiment.VERSION, cases=experiment.case_matrix(),
            seeds=list(experiment.SEEDS), arms=list(experiment.ARMS), gates=experiment.GATES,
            victim_epochs=200,
            optimizer_steps=dict(generator=0, selector=0, surrogate=0, discriminator=0, victim=200),
            execution=dict(max_active_cases=2), bound_files={str(cache):experiment.sha(cache)},
            sources={'runtime.py':experiment.sha(source)},
            runtime_bindings=dict(supports={arm:deepcopy(support) for arm in experiment.ARMS}))
        path = output / 'plan.json'
        path.write_text(json.dumps(plan))
        (output/'registration.json').write_text(json.dumps(dict(
            plan_sha256=experiment.sha(path), source_snapshot_sha256=experiment.sha(snapshot))))
        return path, source, cache

    def test_changed_cache_rejected_before_deserialization(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, source, cache = self.fixture(Path(folder))
            experiment.verify(path)
            cache.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError, 'Bound input'):
                experiment.verify(path)

    def test_changed_runtime_source_rejected(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, source, cache = self.fixture(Path(folder))
            source.write_text('changed source')
            with self.assertRaisesRegex(ValueError, 'runtime source'):
                experiment.verify(path)

    def test_changed_plan_rejected_before_spawning(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)), \
                patch.object(experiment.subprocess, 'Popen') as spawn:
            path, source, cache = self.fixture(Path(folder))
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'Plan or source snapshot'):
                experiment.launch(path)
            spawn.assert_not_called()

    def test_all_four_real_input_preflight_receipts_required(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)), \
                patch.object(experiment.subprocess, 'Popen') as spawn:
            path, source, cache = self.fixture(Path(folder))
            with self.assertRaises(FileNotFoundError):
                experiment.launch(path)
            spawn.assert_not_called()

    def test_detached_queue_cannot_be_launched_twice(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)), \
                patch.object(experiment.subprocess, 'Popen', return_value=Mock(pid=123)) as spawn:
            path, source, cache = self.fixture(Path(folder))
            preflight = path.parent/'engineering_preflight'
            preflight.mkdir()
            digest = experiment.sha(path)
            (preflight/'completion.json').write_text(json.dumps(dict(
                status='SUPPORT_PREFLIGHT_PASS',plan_sha256=digest,arms=list(experiment.ARMS),
                victim_epochs=2,asr_evaluation=False,short_weights_reused=False)))
            for arm in experiment.ARMS:
                item = preflight/arm
                item.mkdir()
                (item/'completion.json').write_text(json.dumps(dict(
                    status='SUPPORT_CASE_PREFLIGHT_PASS',plan_sha256=digest,
                    arm=arm,seed=experiment.SEEDS[0])))
            receipt = experiment.launch(path)
            self.assertEqual(receipt['max_active_cases'], 2)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            with self.assertRaises(FileExistsError):
                experiment.launch(path)
            self.assertEqual(spawn.call_count, 1)


if __name__ == '__main__':
    unittest.main()
