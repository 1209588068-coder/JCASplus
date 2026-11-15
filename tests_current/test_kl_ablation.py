"""Synthetic KL-runner registration, gradient and saved-evidence contracts."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import kl_ablation as pilot
from tests_current.test_copy_pilot import synthetic_bundle
from tests_current.test_distribution_training import inputs as training_inputs
from tests_current.test_copy_training_reporting import (
    training_fixture, read_development, refresh,
)
from tests_current.test_copy_diagnostic_reporting import read_json, write_json


def ablation_bundle(root):
    bundle = synthetic_bundle(root)
    # Metadata-only fixtures deliberately share dummy byte assets across seeds;
    # no model is loaded and no actual initialization claim is made here.
    for seed in pilot.SEEDS:
        bundle.manifest['seeds'][str(seed)] = dict(generator='generator',
                                                  surrogate='surrogate', victim='victim')
    bundle.manifest_path.write_text(json.dumps(bundle.manifest))
    return bundle


def ablation_fixture(root, seed=20260621):
    training_fixture(root, seed)
    for arm in pilot.ARMS:
        source = root / ('pair_mean' if arm == 'single_proxy1' else 'identity')
        folder = root / arm
        shutil.copytree(source, folder)
        initial = read_json(folder / 'initial_state.json')
        initial.update(arm=arm, coefficients=list(pilot.COEFFICIENTS[arm]),
            training_parent_state_sha256='e' * 64, training_surrogate_state_sha256='f' * 64)
        write_json(folder / 'initial_state.json', initial)
        completion = read_json(folder / 'completion.json')
        completion.update(status='KL_ABLATION_ARM_COMPLETE', arm=arm,
                          coefficients=list(pilot.COEFFICIENTS[arm]))
        write_json(folder / 'completion.json', completion)
        rows, copied, summary = read_development(folder / 'development')
        copied['mode'] = 'identity'
        summary.update(mode='identity', generator_contract=dict(mode='identity',
            frozen_postprocess=True, output_mode='domain_projection'))
        refresh(folder / 'development', rows, copied, summary)
        member_losses = [.25, .75]
        mean = sum(member_losses) / 2
        effective = mean if arm == 'dual' else member_losses[0] if arm == 'single_proxy1' else 0.
        treatment_history = [dict(call=i + 1, calls=i + 1, mode=arm,
            coefficients=list(pilot.COEFFICIENTS[arm]), member_raw_kl=member_losses,
            raw_kl=mean, mean_kl=mean, effective_kl=effective, weighted_kl=effective)
            for i in range(200)]
        training = dict(config=deepcopy(pilot.TRAINING_CONFIG), distribution_arm='unconditional',
            selection_policy='final_epoch', selected_epoch=199,
            residual_steps=200, surrogate_steps=200, discriminator_steps=200,
            clean_constraint_calls=200, distribution_generator_calls=200,
            initial_cpu_rng_sha256=completion['initial_cpu_rng_sha256'],
            distribution_initial_state_sha256=completion['distribution_initial_state_sha256'],
            initial_state_sha256=dict(generator=initial['training_parent_state_sha256'],
                                      surrogate=initial['training_surrogate_state_sha256']),
            final_state_sha256=dict(generator='5' * 64, surrogate='6' * 64),
            evaluation_digest_final_state=dict(generator=summary['shared_parent_generator_state_sha256'],
                                               surrogate=summary['model_state_sha256']['surrogate']),
            history=[dict(epoch=i, clean_kl_loss=effective) for i in range(200)],
            clean_kl_treatment=dict(mode=arm, coefficients=list(pilot.COEFFICIENTS[arm]),
                                    history=treatment_history))
        write_json(folder / 'generator_record.json', training)
    write_json(root / 'dual/retained_replay.json', dict(seed=seed,
        tensor_state_match=dict(generator=True, surrogate=True, victim=True)))


class KLAblationRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(pilot, 'ROOT', self.root))
        self.bundle = ablation_bundle(self.root)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        for name in ('torch.load', 'numpy.load', 'torch.Tensor.backward',
                     'torch.autograd.grad', 'torch.optim.Adam.__init__', 'torch.optim.Adam.step'):
            self.enterContext(patch(name, side_effect=AssertionError('No real load/train: ' + name)))

    def prepare(self, name='ablation'):
        return pilot.prepare(self.bundle, self.root / 'results' / name)

    def test_prepare_and_plan_checks_are_metadata_only(self):
        with ExitStack() as stack:
            blocked = [stack.enter_context(patch.object(pilot, name,
                side_effect=AssertionError('Not allowed during prepare: ' + name)))
                for name in ('read_inputs', 'load_normalizer', 'restore_generator', 'restore_surrogate',
                             'restore_proxy', 'restore_clean_reference', 'restore_victim',
                             'fit_distribution_residual', 'fit_fresh_victim')]
            plan = self.prepare()
            spec = pilot.validate_plan(self.bundle, plan)['protocol']
            self.assertEqual(spec['seeds'], list(pilot.SEEDS))
            self.assertEqual(spec['arms'], list(pilot.ARMS))
            self.assertEqual(spec['coefficients'], {k: list(v) for k, v in pilot.COEFFICIENTS.items()})
            self.assertIn('start_generator', spec['assets'])
            self.assertIn('start_surrogate', spec['assets'])
            self.assertIn('prior dual-KL', spec['limitation'])
            self.assertFalse(spec['original_test_arrays_allowed'])
            self.assertFalse(spec['original_validation_arrays_allowed'])
            self.assertFalse(spec['audit_arrays_allowed'])
            self.assertFalse(spec['confirmation_arrays_allowed'])
            self.assertEqual((len(spec['sources']['train']), len(spec['sources']['dev'])), (512, 128))
            registration = read_json(plan.parent / 'registration.json')
            for field in ('graph_arrays_loaded', 'model_checkpoints_loaded', 'training_started'):
                self.assertFalse(registration[field])
            for function in blocked:
                function.assert_not_called()

    def test_tampered_plan_snapshot_source_budget_or_population_rejected(self):
        for kind in ('plan', 'snapshot', 'source', 'budget', 'population'):
            plan = self.prepare(kind)
            if kind == 'plan':
                plan.write_bytes(plan.read_bytes() + b'\n')
            elif kind == 'snapshot':
                path = plan.parent / 'source_snapshot.tar'
                path.write_bytes(path.read_bytes() + b'changed')
            elif kind == 'budget':
                self.bundle.config['victim_epochs'] = 199
            elif kind == 'population':
                self.bundle.config['roles']['dev'][0]['split'] = 'test'
            try:
                if kind == 'source':
                    original = pilot.sha256
                    def changed(path):
                        return '0' * 64 if Path(path).resolve() == Path(pilot.__file__).resolve() else original(path)
                    with patch.object(pilot, 'sha256', side_effect=changed), self.assertRaises(ValueError):
                        pilot.validate_plan(self.bundle, plan)
                else:
                    with self.assertRaises(ValueError):
                        pilot.validate_plan(self.bundle, plan)
            finally:
                self.bundle.config['victim_epochs'] = 200
                self.bundle.config['roles']['dev'][0]['split'] = 'train'

    def test_preparation_never_overwrites_and_only_writes_new_result_child(self):
        plan = self.prepare()
        before = {p.name: p.read_bytes() for p in plan.parent.iterdir()}
        with self.assertRaises(FileExistsError):
            pilot.prepare(self.bundle, plan.parent)
        self.assertEqual(before, {p.name: p.read_bytes() for p in plan.parent.iterdir()})
        for path in (self.root, self.root / 'results', self.root / 'outside'):
            with self.assertRaises(ValueError):
                pilot.prepare(self.bundle, path)

    def test_failure_keeps_claim_and_prevents_second_read(self):
        plan = self.prepare()
        with patch.object(pilot, 'read_inputs', side_effect=RuntimeError('synthetic input failure')) as read:
            with self.assertRaisesRegex(RuntimeError, 'synthetic input failure'):
                pilot.run(self.bundle, plan)
            read.assert_called_once()
        self.assertEqual(read_json(plan.parent / 'failure.json')['status'], 'FAILED_NO_FURTHER_JOBS')
        with patch.object(pilot, 'read_inputs', side_effect=AssertionError('second read')) as read:
            with self.assertRaises(FileExistsError):
                pilot.run(self.bundle, plan)
            read.assert_not_called()


class KLAblationPreflightTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state().clone())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(random.setstate, random.getstate())
        torch.set_num_threads(1)

    def test_actual_mode_gradients_without_model_updates(self):
        source = training_inputs('unconditional')
        inputs = dict(train=dict(x=source['features']), context=source['context'], outer=source['fixed_outer'])
        proxies = [member.model for member in source['clean_constraint'].members]
        for mode in pilot.ARMS:
            before = [pilot.state_hash(m) for m in (source['generator'], *proxies)]
            record = pilot.loss_preflight(source['generator'], proxies, inputs, mode)
            self.assertEqual(record['mode'], mode)
            self.assertEqual(record['optimizer_steps'], 0)
            self.assertEqual(record['record']['coefficients'], list(pilot.COEFFICIENTS[mode]))
            self.assertGreater(record['record']['raw_kl'], 0.)
            if mode == 'disabled':
                self.assertEqual(record['residual_gradient_l2'], 0.)
                self.assertEqual(record['record']['effective_kl'], 0.)
            else:
                self.assertGreater(record['residual_gradient_l2'], 0.)
            self.assertEqual(before, [pilot.state_hash(m) for m in (source['generator'], *proxies)])
            self.assertTrue(all(p.grad is None for m in (source['generator'], *proxies) for p in m.parameters()))


class KLAblationReportingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        ablation_fixture(self.root)
        for name in ('torch.load', 'numpy.load', 'torch.Tensor.backward', 'torch.autograd.grad'):
            self.enterContext(patch(name, side_effect=AssertionError('Report is evidence-only: ' + name)))

    def compare(self):
        return pilot.build_comparison(self.root, 20260621)

    def mutate(self, name, function):
        path = self.root / name
        value = read_json(path)
        function(value)
        write_json(path, value)

    def test_comparison_recomputes_all_modes_and_keeps_motion_metrics(self):
        result = self.compare()
        self.assertEqual(result['status'], 'KL_ABLATION_COMPARISON_COMPLETE_REVIEW_REQUIRED')
        self.assertEqual(set(result['metrics']), set(pilot.ARMS))
        for arm in pilot.ARMS:
            metrics = result['metrics'][arm]['victim']
            self.assertEqual(metrics['targets']['all']['targets'], 112)
            self.assertEqual(metrics['targets']['moving']['targets'], 33)
            self.assertEqual(metrics['targets']['slow']['targets'], 79)
            self.assertIn('normal_by_pair_motion', metrics)
        self.assertEqual(result['delta_vs_dual']['disabled']['victim']['targets']['all']['incremental_asr'], 0.)
        self.assertFalse(result['blind_confirmation_claim'])
        self.assertFalse(result['automatic_promotion'])
        json.dumps(result, allow_nan=False)

    def test_mismatched_seeds_coefficients_initialization_and_budgets_rejected(self):
        cases = (
            ('initial_state.json', lambda r: r.update(seed=20260622)),
            ('initial_state.json', lambda r: r.update(coefficients=[.5, .5])),
            ('initial_state.json', lambda r: r.update(generator_parent_state_sha256='1' * 64)),
            ('completion.json', lambda r: r.update(residual_steps=199)),
            ('completion.json', lambda r: r.update(initial_cpu_rng_sha256='2' * 64)),
            ('victim_record.json', lambda r: r.update(victim_seed=20260623)),
            ('development/summary.json', lambda r: r.update(seed=20260622)),
        )
        for name, mutate in cases:
            path = self.root / 'single_proxy1' / name
            before = path.read_bytes()
            try:
                self.mutate('single_proxy1/' + name, mutate)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.compare()
            finally:
                path.write_bytes(before)

    def test_tampered_metric_summary_and_csv_bytes_rejected(self):
        path = self.root / 'disabled/development/summary.json'
        before = path.read_bytes()
        self.mutate('disabled/development/summary.json', lambda r:
            r['metrics']['victim']['targets']['all'].update(incremental_asr=.123))
        with self.assertRaises(ValueError):
            self.compare()
        path.write_bytes(before)
        csv = self.root / 'disabled/development/targets.csv'
        csv.write_bytes(csv.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            self.compare()

    def test_failed_replay_or_mismatched_effective_training_loss_rejected(self):
        for name, mutation in (
            ('dual/retained_replay.json', lambda r: r['tensor_state_match'].update(victim=False)),
            ('dual/retained_replay.json', lambda r: r.update(seed=20260622)),
            ('disabled/generator_record.json', lambda r:
                r['clean_kl_treatment']['history'][3].update(effective_kl=.1)),
            ('single_proxy1/generator_record.json', lambda r:
                r['history'][3].update(clean_kl_loss=.5)),
            ('disabled/generator_record.json', lambda r:
                r['clean_kl_treatment']['history'][3].update(mode='dual')),
            ('disabled/generator_record.json', lambda r:
                r['initial_state_sha256'].update(generator='d' * 64)),
            ('disabled/generator_record.json', lambda r:
                r['evaluation_digest_final_state'].update(surrogate='9' * 64)),
            ('disabled/generator_record.json', lambda r: r.update(selected_epoch=99)),
        ):
            path = self.root / name
            before = path.read_bytes()
            try:
                self.mutate(name, mutation)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.compare()
            finally:
                path.write_bytes(before)

    def test_three_seed_aggregation_is_paired_and_not_more_independent_targets(self):
        original = self.compare()
        comparisons = [dict(deepcopy(original), seed=seed) for seed in pilot.SEEDS]
        for i, row in enumerate(comparisons):
            row['metrics']['disabled']['victim']['targets']['all']['incremental_asr'] += .01 * i
        result = pilot.aggregate(comparisons)
        metric = result['paired_delta_mean_sd']['disabled']['overall_incremental_asr']
        self.assertAlmostEqual(metric['mean'], .01)
        self.assertAlmostEqual(metric['sample_sd'], .01)
        self.assertEqual(result['distinct_targets'], 112)
        self.assertEqual(result['distinct_moving_targets'], 33)
        self.assertTrue(result['repeated_seeds_do_not_add_independent_targets'])
        self.assertFalse(result['from_scratch_ablation'])
        self.assertFalse(result['blind_confirmation'])
        for bad in (comparisons[:2], comparisons[::-1]):
            with self.assertRaises(ValueError):
                pilot.aggregate(bad)
        changed = deepcopy(comparisons)
        changed[2]['population_identity_sha256']['targets'] = '1' * 64
        with self.assertRaises(ValueError):
            pilot.aggregate(changed)


if __name__ == '__main__':
    unittest.main()
