"""Synthetic consistency-pilot contracts; no real assets, graphs, or fitting."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import copy_pilot as pilot
from spear_current import train
from spear_current.copy_consistency import CopyConsistentGenerator
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.evaluate import state_hash
from spear_current.data import combine_parts
from spear_current.development_diagnostic import row_metadata, choose_target
from tests_current.test_copy_consistency_training import fixture as copy_fixture
from tests_current.test_copy_diagnostic import metadata_bundle
from tests_current.test_outer_evaluation import scene, prepared, Toy784Classifier


COPY_BUFFERS = {'copy_mode_code', 'copy_frozen_code', 'copy_positions', 'copy_normalizer_digest'}


def synthetic_bundle(root):
    bundle = metadata_bundle(root)
    config = bundle.config
    config.update(seed=20260621, shuffle_seed=20260621, training_seed=20263621,
                  score_dim=32, victim_epochs=200)
    config['arguments'] = dict(hidden=80, train_lr=.01, shadow_lr=.0002, trojan_lr=.0002,
        weight_decay=.0005, dropout=.5, thrd=.5, target_class=1, epochs=200,
        trojan_epochs=200, inner=1, alpha_int=30, outter_size=512, vs_number=26,
        target_loss_weight=1., homo_loss_weight=.1)
    config['victim_initialization_digest'] = 'synthetic-initialization'
    config['victim_fit_rng_digest'] = 'synthetic-dropout-rng'
    bundle.manifest.update(config=config, training_config=deepcopy(train.TRAINING_CONFIG),
        distribution_config=deepcopy(train.DISTRIBUTION_CONFIG), training_seed=20263621)
    paths = {key: bundle.asset_path(key) for key in bundle.manifest['assets']}
    for key in ('indices', 'input_contract', 'context_binding', 'score_permutation',
                'clean_proxy1', 'clean_proxy2', 'start_generator', 'start_surrogate'):
        path = root / 'dummy_assets' / (key + '.fixture')
        path.write_text('synthetic byte fixture only: ' + key)
        paths[key] = path
        bundle.manifest['assets'][key] = dict(sha256=pilot.sha256(path))
    bundle.asset_path = paths.__getitem__
    bundle.manifest_path.write_text(json.dumps(bundle.manifest))
    return bundle


def synthetic_diagnostic(bundle=None):
    anchor, normalizer, x, context, _ = copy_fixture()
    if bundle is None:
        _, dev, _, _ = prepared()
    else:
        parts = [scene(row['scenario_id'], 'moving' if i < 33 else 'slow', no_negative=i >= 112)
                 for i, row in enumerate(bundle.config['roles']['dev'])]
        _, dev, _, _ = prepared(parts)
        dev['metadata']['dev'] = [row for part, source, offset in
            zip(parts, bundle.config['roles']['dev'], dev['combined']['dev']['offsets'])
            for row in row_metadata(part, source, offset, 'dev', set(), set())]
    part = scene('synthetic-training-tensors')
    part['x'] = x.numpy().copy()
    part['y'] = np.asarray([0, 1, 0, 0, 1, 0], dtype=np.int64)
    combined = combine_parts([part])
    idx = dict(idx_train=torch.tensor([0, 1]), idx_attach=torch.tensor([2]),
               idx_unlabeled=torch.tensor([3, 4, 5]), train_labels=combined['y'].clone(),
               dev_indices=torch.nonzero(dev['combined']['dev']['eligible']).flatten() + len(x))
    source = dict(scenario_id=part['scenario_id'], split_group_id='synthetic-training-group',
                  split='train', role='train')
    metadata = row_metadata(part, source, 0, 'train', {2}, {2, 3, 4})
    targets = [row for offset, target in zip(dev['combined']['dev']['offsets'], dev['targets']['dev'])
               if target is not None for row in [dev['metadata']['dev'][offset + target]]]
    inputs = dict(parts=dict(train=[part], dev=dev['parts']['dev']),
        combined=dict(train=combined, dev=dev['combined']['dev']),
        metadata=dict(train=metadata, dev=dev['metadata']['dev']),
        targets=dict(train=[choose_target(part)], dev=dev['targets']['dev']),
        populations=dict(dev_public_single=targets), indices=idx, outer=torch.tensor([2, 3, 4]), context=context)
    return SimpleNamespace(anchor=anchor, normalizer=normalizer, diagnostic=inputs,
        surrogate=Toy784Classifier(), proxies=[Toy784Classifier(), Toy784Classifier(-1)],
        clean=Toy784Classifier(), events=[], trained=[], victims=[], initial_weights=[])


class CopyPilotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state().clone())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(random.setstate, random.getstate())
        for target in ('torch.load', 'numpy.load', 'torch.Tensor.backward', 'torch.autograd.grad',
                       'torch.optim.Adam.__init__', 'torch.optim.Adam.step',
                       'torch.optim.AdamW.__init__', 'torch.optim.AdamW.step',
                       'torch.optim.SGD.__init__', 'torch.optim.SGD.step'):
            self.enterContext(patch(target, side_effect=AssertionError('No actual loading or training: ' + target)))

    def prepared_plan(self, name='pilot', seed=20260621):
        self.enterContext(patch.object(pilot, 'ROOT', self.root))
        bundle = synthetic_bundle(self.root)
        return bundle, pilot.prepare(bundle, self.root / 'results' / name, seed=seed)

    def test_prepare_protocol_and_validation_are_metadata_only_and_lock_final_start(self):
        with ExitStack() as stack:
            forbidden = [stack.enter_context(patch.object(pilot, name,
                side_effect=AssertionError('No loading or fitting in prepare')))
                for name in ('read_inputs', 'load_normalizer', 'restore_generator', 'restore_surrogate',
                             'restore_proxy', 'restore_clean_reference', 'fit_unconditional',
                             'fit_fresh_victim')]
            bundle, plan = self.prepared_plan()
            parsed = pilot.validate_plan(bundle, plan)
            spec = parsed['protocol']
            self.assertEqual(spec['seed'], 20260621)
            self.assertEqual(spec['arms'], ['identity', 'pair_mean'])
            self.assertEqual(spec['training_config'], train.TRAINING_CONFIG)
            self.assertEqual(spec['victim_epochs'], 200)
            self.assertIn('generator', spec['assets'])
            self.assertIn('surrogate', spec['assets'])
            self.assertNotIn('start_generator', spec['assets'])
            self.assertNotIn('start_surrogate', spec['assets'])
            self.assertEqual(spec['assets']['generator']['sha256'], bundle.manifest['assets']['generator']['sha256'])
            registration = json.loads((plan.parent / 'registration.json').read_text())
            self.assertFalse(registration['graph_arrays_loaded'])
            self.assertFalse(registration['model_checkpoints_loaded'])
            self.assertFalse(registration['training_started'])
            for mocked in forbidden:
                mocked.assert_not_called()

    def test_prepared_plan_rejects_plan_manifest_source_or_budget_tampering(self):
        bundle, plan = self.prepared_plan()
        original_plan, original_manifest = plan.read_bytes(), bundle.manifest_path.read_bytes()
        original_sha = pilot.sha256
        for kind in ('plan', 'manifest', 'source', 'budget'):
            with self.subTest(kind=kind):
                if kind == 'plan':
                    plan.write_bytes(original_plan + b'\n')
                elif kind == 'manifest':
                    bundle.manifest_path.write_bytes(original_manifest + b'\n')
                elif kind == 'budget':
                    bundle.config['victim_epochs'] = 199
                try:
                    if kind == 'source':
                        source = Path(pilot.__file__).resolve()
                        def tampered_sha(path):
                            return '0' * 64 if Path(path).resolve() == source else original_sha(path)
                        with patch.object(pilot, 'sha256', side_effect=tampered_sha), self.assertRaises(ValueError):
                            pilot.validate_plan(bundle, plan)
                    else:
                        with self.assertRaises(ValueError):
                            pilot.validate_plan(bundle, plan)
                finally:
                    plan.write_bytes(original_plan)
                    bundle.manifest_path.write_bytes(original_manifest)
                    bundle.config['victim_epochs'] = 200
        self.assertEqual(pilot.validate_plan(bundle, plan), json.loads(original_plan))

    def test_prepare_requires_new_results_child_and_never_overwrites(self):
        bundle, plan = self.prepared_plan()
        before = {p.name: p.read_bytes() for p in plan.parent.iterdir()}
        with self.assertRaises((FileExistsError, ValueError)):
            pilot.prepare(bundle, plan.parent)
        self.assertEqual(before, {p.name: p.read_bytes() for p in plan.parent.iterdir()})
        empty = self.root / 'results' / 'empty'
        empty.mkdir()
        with self.assertRaises((FileExistsError, ValueError)):
            pilot.prepare(bundle, empty)
        self.assertEqual(list(empty.iterdir()), [])
        for path in (self.root / 'results', self.root / 'outside', self.root / 'results' / '..' / 'escape'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                pilot.prepare(bundle, path)

    def test_failed_run_retains_claim_and_repeated_run_never_reads(self):
        bundle, plan = self.prepared_plan()
        with patch.object(pilot, 'read_inputs', side_effect=RuntimeError('synthetic read failure')) as read:
            with self.assertRaisesRegex(RuntimeError, 'synthetic read failure'):
                pilot.run(bundle, plan)
            read.assert_called_once()
        claim = plan.parent / 'execution_claim.json'
        claim_bytes = claim.read_bytes()
        with patch.object(pilot, 'read_inputs', side_effect=AssertionError('duplicate read')) as read, \
                self.assertRaises(FileExistsError):
            pilot.run(bundle, plan)
        read.assert_not_called()
        self.assertEqual(claim.read_bytes(), claim_bytes)

    def assert_frozen(self, model):
        self.assertTrue(all(not child.training for child in model.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_paired_generators_share_exact_parent_without_storage_or_rng_sharing(self):
        anchor, normalizer, x, context, targets = copy_fixture()
        before = state_hash(anchor)
        torch_rng = torch.random.get_rng_state().clone()
        numpy_rng, python_rng = np.random.get_state(), random.getstate()
        generators = {mode: pilot.make_generator(anchor, normalizer, mode) for mode in pilot.ARMS}
        self.assertEqual(set(generators), {'identity', 'pair_mean'})
        for mode, generator in generators.items():
            self.assertIs(type(generator), CopyConsistentGenerator)
            self.assertEqual(generator.consistency_mode, mode)
            self.assertFalse(generator.frozen_postprocess)
            self.assertEqual(set(generator.state_dict()), set(anchor.state_dict()) | COPY_BUFFERS)
            self.assertTrue(all(p.requires_grad for p in generator.residual.parameters()))
            for name, parameter in generator.named_parameters():
                parent = dict(anchor.named_parameters())[name]
                self.assertTrue(torch.equal(parameter, parent), name)
                self.assertNotEqual(parameter.data_ptr(), parent.data_ptr(), name)
            reference = CopyConsistentGenerator(anchor, normalizer, mode=mode, frozen=True)
            with torch.no_grad():
                self.assertTrue(torch.equal(generator(x, context, targets), reference(x, context, targets)))
        for name, parameter in generators['identity'].named_parameters():
            other = dict(generators['pair_mean'].named_parameters())[name]
            self.assertTrue(torch.equal(parameter, other), name)
            self.assertNotEqual(parameter.data_ptr(), other.data_ptr(), name)
        self.assertEqual(state_hash(anchor), before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))
        np.testing.assert_equal(np.random.get_state(), numpy_rng)
        self.assertEqual(random.getstate(), python_rng)
        self.assert_frozen(anchor)

    def test_evaluation_anchor_strips_only_copy_buffers_and_preserves_trained_output(self):
        anchor, normalizer, x, context, targets = copy_fixture()
        before = state_hash(anchor)
        for mode in pilot.ARMS:
            with self.subTest(mode=mode):
                trained = pilot.make_generator(anchor, normalizer, mode)
                # Simulate a distinct learned residual without optimization.
                with torch.no_grad():
                    trained.residual.head.bias.add_(.25)
                for p in trained.residual.parameters():
                    p.grad = torch.ones_like(p)
                trained_state = state_hash(trained)
                domain = pilot.evaluation_anchor(trained, anchor)
                self.assertIs(type(domain), DomainConstrainedGenerator)
                self.assertEqual(set(domain.state_dict()), set(anchor.state_dict()))
                self.assertFalse(set(domain.state_dict()) & COPY_BUFFERS)
                self.assert_frozen(domain)
                self.assertEqual(state_hash(trained), trained_state)
                for name, value in domain.state_dict().items():
                    self.assertTrue(torch.equal(value, trained.state_dict()[name]), name)
                    self.assertNotEqual(value.data_ptr(), trained.state_dict()[name].data_ptr(), name)
                frozen = CopyConsistentGenerator(domain, normalizer, mode=mode, frozen=True)
                with torch.no_grad():
                    self.assertTrue(torch.equal(frozen(x, context, targets), trained(x, context, targets)))
                self.assert_frozen(frozen)
        self.assertEqual(state_hash(anchor), before)
        self.assert_frozen(anchor)

    def test_evaluation_anchor_rejects_missing_or_extra_checkpoint_keys(self):
        anchor, normalizer, _, _, _ = copy_fixture()
        trained = pilot.make_generator(anchor, normalizer, 'pair_mean')
        original = trained.state_dict()
        for kind in ('missing_copy', 'missing_shared', 'unexpected'):
            changed = deepcopy(original)
            if kind == 'missing_copy':
                changed.pop('copy_mode_code')
            elif kind == 'missing_shared':
                changed.pop('residual.head.bias')
            else:
                changed['unexpected_state'] = torch.zeros(1)
            with self.subTest(kind=kind), patch.object(trained, 'state_dict', return_value=changed), \
                    self.assertRaises(ValueError):
                pilot.evaluation_anchor(trained, anchor)

    def fake_training_calls(self, fixture, *, bad_budget=False, expected_seed=20260621):
        """Exercise the real arm orchestration; replace all numerical updates."""
        def preflight(generator, reference, surrogate, proxies, inputs, seed):
            mode = generator.consistency_mode
            fixture.events.append((mode, 'preflight'))
            self.assertEqual(seed, expected_seed)
            self.assertEqual(reference.consistency_mode, mode)
            self.assertTrue(reference.frozen_postprocess)
            self.assertFalse(generator.frozen_postprocess)
            self.assertIs(surrogate, fixture.surrogate)
            self.assertEqual(proxies, fixture.proxies)
            self.assertIs(inputs['train'], fixture.diagnostic['combined']['train'])
            with torch.no_grad():
                self.assertTrue(torch.equal(generator(inputs['train']['x'], inputs['context'], inputs['outer']),
                                            reference(inputs['train']['x'], inputs['context'], inputs['outer'])))
            return dict(status='NO_UPDATE_PREFLIGHT_PASS', optimizer_steps=0)

        def fit(generator, surrogate, proxies, inputs, *, seed, progress=None):
            mode = generator.consistency_mode
            fixture.events.append((mode, 'fit_GSD_200'))
            fixture.initial_weights.append({k: v.clone() for k, v in generator.state_dict().items()
                                             if k in fixture.anchor.state_dict()})
            self.assertEqual(seed, expected_seed)
            self.assertIs(surrogate, fixture.surrogate)
            self.assertEqual(proxies, fixture.proxies)
            trained_surrogate = deepcopy(surrogate)
            with torch.no_grad():
                generator.residual.head.bias.add_(.125 if mode == 'identity' else .25)
                trained_surrogate.sign.add_(.25 if mode == 'identity' else .5)
                injection = generator.inject(inputs['train']['x'], inputs['context'], inputs['indices']['idx_attach'],
                    torch.ones(inputs['context'].edge_index.shape[1]), detach=True)
            generator.train()
            for p in generator.residual.parameters():
                p.grad = torch.ones_like(p)
            trained_surrogate.train().requires_grad_(True)
            trained_surrogate.sign.grad = torch.ones_like(trained_surrogate.sign)
            result = dict(generator=generator, surrogate=trained_surrogate,
                poisoned_features=injection.features, poisoned_labels=inputs['train']['y'].clone(),
                selected_epoch=199, selection_policy='final_epoch', config=deepcopy(train.TRAINING_CONFIG),
                distribution_arm='unconditional', residual_steps=199 if bad_budget else 200,
                surrogate_steps=200, discriminator_steps=200, clean_constraint_calls=200,
                distribution_generator_calls=200, initial_cpu_rng_sha256='same-continuation-rng',
                distribution_initial_state_sha256='same-discriminator-start')
            result['poisoned_labels'][inputs['indices']['idx_attach']] = 1
            fixture.trained.append(result)
            distribution = SimpleNamespace(discriminator=Toy784Classifier(), snapshot=lambda: {'synthetic': True})
            return result, distribution, {'synthetic_initial': True}

        def victim(bundle, inputs, result, output, *, seed, progress=None):
            mode = result['generator'].consistency_mode
            fixture.events.append((mode, 'fresh_victim_200'))
            self.assertEqual(seed, expected_seed)
            self.assertEqual(inputs['indices']['idx_attach'].tolist(), [2])
            model = Toy784Classifier()
            self.assertTrue(all(model is not previous for previous in fixture.victims))
            model.train().requires_grad_(True)
            model.sign.grad = torch.ones_like(model.sign)
            fixture.victims.append(model)
            return model, dict(trace=dict(selected_auc_epoch=7, training_forwards=200, eval_forwards=200),
                initialization_digest='same-fresh-victim', fit_rng_digest='same-fresh-victim-rng', victim_seed=seed)

        def evaluate(bundle, inputs, domain, normalizer, models, mode, output, *, seed=20260621):
            self.assertEqual(seed, expected_seed)
            fixture.events.append((mode, 'evaluate_DEV'))
            self.assertIs(inputs, fixture.diagnostic)
            self.assertIs(normalizer, fixture.normalizer)
            self.assertIs(type(domain), DomainConstrainedGenerator)
            self.assertEqual(set(models), {'surrogate', 'victim', 'clean_reference'})
            self.assertIs(models['clean_reference'], fixture.clean)
            self.assertIs(models['surrogate'], fixture.trained[-1]['surrogate'])
            self.assertIs(models['victim'], fixture.victims[-1])
            for model in (domain, *models.values()):
                self.assert_frozen(model)
            return {'synthetic_evaluation': True}

        stack = ExitStack()
        self.addCleanup(stack.close)
        mocks = {name: stack.enter_context(patch.object(pilot, name, side_effect=fn))
                 for name, fn in (('no_update_preflight', preflight), ('fit_unconditional', fit),
                                  ('fit_fresh_victim', victim), ('evaluate_arm', evaluate))}
        mocks['save'] = stack.enter_context(patch.object(pilot, '_save_checkpoint'))
        return mocks

    def test_each_arm_uses_same_mode_reference_fixed_budget_and_distinct_fresh_victim(self):
        fixture = synthetic_diagnostic()
        mocks = self.fake_training_calls(fixture)
        sources = [fixture.anchor, fixture.surrogate, *fixture.proxies, fixture.clean]
        before_sources = [state_hash(model) for model in sources]
        before_inputs = pilot._input_hashes(fixture.diagnostic)
        records = []
        for mode in pilot.ARMS:
            record = pilot.fit_one_arm(None, fixture.diagnostic, fixture.normalizer, fixture.anchor,
                fixture.surrogate, fixture.proxies, fixture.clean, mode, self.root / mode, lambda **row: None)
            records.append(record)
            self.assertEqual(record['generator_selected_epoch'], 199)
            self.assertEqual([record[k] for k in ('residual_steps', 'surrogate_steps',
                                                'discriminator_steps', 'victim_epochs')], [200] * 4)
            export = json.loads((self.root / mode / 'evaluation_export.json').read_text())
            self.assertTrue(export['bitwise_output_parity_on_attachments'])
        self.assertEqual(fixture.events, [(mode, phase) for mode in pilot.ARMS
            for phase in ('preflight', 'fit_GSD_200', 'fresh_victim_200', 'evaluate_DEV')])
        for name, value in fixture.initial_weights[0].items():
            self.assertTrue(torch.equal(value, fixture.initial_weights[1][name]), name)
        self.assertNotEqual(state_hash(fixture.trained[0]['generator']), state_hash(fixture.trained[1]['generator']))
        self.assertIsNot(fixture.victims[0], fixture.victims[1])
        self.assertEqual([state_hash(model) for model in sources], before_sources)
        self.assertEqual(pilot._input_hashes(fixture.diagnostic), before_inputs)
        for key in ('initial_cpu_rng_sha256', 'distribution_initial_state_sha256',
                    'victim_initialization_digest', 'victim_fit_rng_digest'):
            self.assertEqual(records[0][key], records[1][key])
        self.assertEqual(mocks['fit_unconditional'].call_count, 2)
        self.assertEqual(mocks['fit_fresh_victim'].call_count, 2)

    def test_bad_generator_budget_stops_before_victim_fit_or_evaluation(self):
        fixture = synthetic_diagnostic()
        mocks = self.fake_training_calls(fixture, bad_budget=True)
        with self.assertRaisesRegex(ValueError, 'budget'):
            pilot.fit_one_arm(None, fixture.diagnostic, fixture.normalizer, fixture.anchor,
                fixture.surrogate, fixture.proxies, fixture.clean, 'identity', self.root / 'bad', lambda **row: None)
        mocks['fit_fresh_victim'].assert_not_called()
        mocks['evaluate_arm'].assert_not_called()
        mocks['save'].assert_not_called()

    def patch_run_sources(self, bundle, fixture, *, ledger_error=False):
        def read(_bundle, sources, ledger):
            self.assertIs(_bundle, bundle)
            self.assertEqual(sources, {role: bundle.config['roles'][role] for role in ('train', 'dev')})
            ledger.extend(dict(role=role, split='train', status='READ_COMPLETE')
                          for role, count in (('train', 512), ('dev', 128)) for _ in range(count))
            if ledger_error:
                ledger[-1]['role'] = 'audit'
            return fixture.diagnostic
        stack = ExitStack()
        self.addCleanup(stack.close)
        mocks = {}
        for name, value in (('read_inputs', read), ('load_normalizer', fixture.normalizer),
                            ('restore_generator', fixture.anchor), ('restore_surrogate', fixture.surrogate),
                            ('restore_clean_reference', fixture.clean),
                            ('_dev_quality', {'gate': 'PASS_PROXY_QUALITY'})):
            mocks[name] = stack.enter_context(patch.object(pilot, name,
                **({'side_effect': value} if name == 'read_inputs' else {'return_value': value})))
        mocks['restore_proxy'] = stack.enter_context(patch.object(pilot, 'restore_proxy',
            side_effect=lambda _bundle, index: fixture.proxies[index - 1]))
        mocks['comparison'] = stack.enter_context(patch('spear_current.copy_training_reporting.build_training_comparison',
            return_value={'status': 'synthetic_review_only', 'automatic_followup': False}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return mocks

    def test_run_pins_final_seed21_sources_and_orders_two_independent_equal_budget_fits(self):
        bundle, plan = self.prepared_plan()
        fixture = synthetic_diagnostic(bundle)
        mocks = self.patch_run_sources(bundle, fixture)
        self.fake_training_calls(fixture)
        result = pilot.run(bundle, plan)
        self.assertEqual(result['status'], 'synthetic_review_only')
        mocks['read_inputs'].assert_called_once()
        mocks['load_normalizer'].assert_called_once_with(bundle)
        mocks['restore_generator'].assert_called_once_with(bundle, 20260621)
        mocks['restore_surrogate'].assert_called_once_with(bundle, 20260621)
        self.assertEqual(mocks['restore_proxy'].call_count, 2)
        mocks['restore_clean_reference'].assert_called_once_with(bundle)
        mocks['comparison'].assert_called_once_with(plan.parent)
        record = json.loads((plan.parent / 'completion.json').read_text())
        self.assertEqual(record['completed_arms'], ['identity', 'pair_mean'])
        self.assertEqual(record['graph_reads'], 640)
        self.assertEqual(record['updates_per_arm'], dict(generator=200, surrogate=200, discriminator=200, victim=200))
        self.assertTrue(record['common_initialization_checks'])
        self.assertFalse(record['automatic_followup'])
        self.assertFalse(record['automatic_promotion'])
        self.assertEqual(fixture.events, [(mode, phase) for mode in pilot.ARMS
            for phase in ('preflight', 'fit_GSD_200', 'fresh_victim_200', 'evaluate_DEV')])

    def test_run_rejects_bad_access_ledger_before_restoring_models(self):
        bundle, plan = self.prepared_plan()
        fixture = synthetic_diagnostic(bundle)
        mocks = self.patch_run_sources(bundle, fixture, ledger_error=True)
        with patch.object(pilot, 'fit_one_arm') as fit, self.assertRaisesRegex(ValueError, 'graph access scope'):
            pilot.run(bundle, plan)
        fit.assert_not_called()
        mocks['restore_generator'].assert_not_called()
        mocks['comparison'].assert_not_called()
        self.assertFalse((plan.parent / 'completion.json').exists())

    def test_run_rejects_shared_model_input_or_paired_initialization_drift(self):
        for corruption in ('model', 'input', 'discriminator', 'training_rng', 'victim_init', 'victim_rng'):
            with self.subTest(corruption=corruption), TemporaryDirectory() as temporary, ExitStack() as local:
                root = Path(temporary)
                local.enter_context(patch.object(pilot, 'ROOT', root))
                bundle = synthetic_bundle(root)
                plan = pilot.prepare(bundle, root / 'results' / 'guard')
                fixture = synthetic_diagnostic(bundle)
                mocks = self.patch_run_sources(bundle, fixture)
                calls = []
                def fake_arm(_bundle, diagnostic, normalizer, anchor, surrogate, proxies, clean, mode, output, progress, *, seed):
                    calls.append(mode)
                    record = dict(initial_cpu_rng_sha256='same', distribution_initial_state_sha256='same',
                                  victim_initialization_digest='same', victim_fit_rng_digest='same')
                    if corruption == 'model':
                        with torch.no_grad():
                            surrogate.sign.add_(1)
                    elif corruption == 'input':
                        diagnostic['combined']['train']['x'][0, 0] += 1
                    elif mode == 'pair_mean':
                        key = dict(discriminator='distribution_initial_state_sha256', training_rng='initial_cpu_rng_sha256',
                                   victim_init='victim_initialization_digest', victim_rng='victim_fit_rng_digest')[corruption]
                        record[key] = 'changed'
                    return record
                with patch.object(pilot, 'fit_one_arm', side_effect=fake_arm), self.assertRaises(ValueError):
                    pilot.run(bundle, plan)
                self.assertEqual(calls, ['identity'] if corruption in ('model', 'input') else ['identity', 'pair_mean'])
                mocks['comparison'].assert_not_called()
                self.assertFalse((plan.parent / 'completion.json').exists())
                self.assertTrue((plan.parent / 'failure.json').exists())

    def test_launch_only_spawns_registered_command_once_with_cpu_environment(self):
        bundle, plan = self.prepared_plan()
        with patch.object(pilot.subprocess, 'Popen', return_value=SimpleNamespace(pid=43210)) as spawn, \
                patch.object(pilot, 'run', side_effect=AssertionError('training in launch parent')) as run, \
                patch.object(pilot, 'read_inputs', side_effect=AssertionError('graph read in launch parent')) as read:
            record = pilot.launch(bundle, plan)
            spawn.assert_called_once()
            command = spawn.call_args.args[0]
            self.assertEqual(command, [pilot.sys.executable, '-B', '-m', 'spear_current.copy_pilot',
                                       '--run', '--plan', str(plan.resolve())])
            kwargs = spawn.call_args.kwargs
            self.assertEqual(kwargs['cwd'], str(self.root))
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            for key in ('PYTHONDONTWRITEBYTECODE', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
                self.assertEqual(kwargs['env'][key], '1')
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(kwargs['stdin'], pilot.subprocess.DEVNULL)
            self.assertEqual(kwargs['stderr'], pilot.subprocess.STDOUT)
            with self.assertRaises(FileExistsError):
                pilot.launch(bundle, plan)
            self.assertEqual(spawn.call_count, 1)
            read.assert_not_called()
            run.assert_not_called()
        self.assertEqual(record['pid'], 43210)
        self.assertFalse(record['automatic_followup'])
        self.assertFalse((plan.parent / 'execution_claim.json').exists())

    def test_launch_rejects_existing_execution_claim_without_spawn(self):
        bundle, plan = self.prepared_plan()
        (plan.parent / 'execution_claim.json').write_text('{}')
        with patch.object(pilot.subprocess, 'Popen') as spawn, self.assertRaisesRegex(ValueError, 'already'):
            pilot.launch(bundle, plan)
        spawn.assert_not_called()

    def test_new_training_seeds_do_not_change_pretrained_parent_or_data(self):
        self.enterContext(patch.object(pilot, 'ROOT', self.root))
        bundle = synthetic_bundle(self.root)
        specs = [pilot.protocol(bundle, seed) for seed in (20260621, 20260622, 20260623)]
        for spec in specs:
            self.assertEqual(spec['parent_seed'], 20260621)
            for key in ('start_asset_keys', 'assets', 'sources', 'training_config', 'hard_support', 'gates'):
                self.assertEqual(spec[key], specs[0][key])
            self.assertEqual(spec['effective_distribution_config']['seed'], spec['seed'] + 1000)
            self.assertEqual(spec['seeds']['victim'], spec['seed'])
        for seed in (True, 20260624, '20260622', 20260622.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                pilot.protocol(bundle, seed)

    def test_seed22_run_propagates_run_seed_but_restores_only_seed21_parent(self):
        bundle, plan = self.prepared_plan(seed=20260622)
        fixture = synthetic_diagnostic(bundle)
        mocks = self.patch_run_sources(bundle, fixture)
        self.fake_training_calls(fixture, expected_seed=20260622)
        pilot.run(bundle, plan)
        mocks['restore_generator'].assert_called_once_with(bundle, 20260621)
        mocks['restore_surrogate'].assert_called_once_with(bundle, 20260621)
        for arm in pilot.ARMS:
            for name in ('initial_state', 'completion'):
                record = json.loads((plan.parent / arm / (name + '.json')).read_text())
                self.assertEqual(record['seed'], 20260622)


if __name__ == '__main__':
    unittest.main()
