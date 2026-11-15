"""Synthetic-only fixed-DEV runner contracts; no saved models or graph reads."""
from contextlib import redirect_stdout
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

from spear_current import copy_diagnostic as diagnostic
from spear_current.copy_consistency import CopyConsistentGenerator, COPY_GROUPS
from spear_current.evaluate import SceneInjector, state_hash
from spear_current.support_schema import SELECTED_COORDINATES
from tests_current.test_copy_consistency_training import ACTIVE, fixture as copy_fixture
from tests_current.test_outer_evaluation import prepared, scene


def metadata_bundle(root):
    """Dummy byte assets and registered identities, never real asset paths."""
    roles = {role: [dict(scenario_id=f'synthetic-{role}-{i}',
                        split_group_id=f'synthetic-{role}-group-{i}',
                        role=role, split='train') for i in range(count)]
             for role, count in (('train', 512), ('dev', 128), ('audit', 256))}
    assets = root / 'dummy_assets'
    assets.mkdir()
    keys = ('generator', 'surrogate', 'victim', 'clean_reference', 'normalizer', 'support')
    paths = {key: assets / (key + '.fixture') for key in keys}
    for key, path in paths.items():
        path.write_text('synthetic bytes only: ' + key)
    manifest = dict(selected_coordinates=list(SELECTED_COORDINATES),
        hard_support=[c in ACTIVE for c in SELECTED_COORDINATES],
        seeds={str(diagnostic.SEED): {name: name for name in ('generator', 'surrogate', 'victim')}},
        assets={key: dict(sha256=diagnostic.sha256(path)) for key, path in paths.items()})
    manifest_path = root / 'synthetic_manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    return SimpleNamespace(config=dict(shuffle_seed=17, roles=roles), manifest=manifest,
                           manifest_path=manifest_path, asset_path=paths.__getitem__)


class CopyDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.root = Path(self.directory)
        self.enterContext(patch.object(diagnostic, 'ROOT', self.root))
        self.enterContext(patch.object(diagnostic, '_progress'))
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state().clone())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(random.setstate, random.getstate())
        for target in ('torch.load', 'numpy.load', 'torch.Tensor.backward', 'torch.autograd.grad',
                       'torch.optim.Adam.__init__', 'torch.optim.Adam.step',
                       'torch.optim.AdamW.__init__', 'torch.optim.AdamW.step',
                       'torch.optim.SGD.__init__', 'torch.optim.SGD.step'):
            self.enterContext(patch(target, side_effect=AssertionError('Forbidden real load/update: ' + target)))
        self.restores = {name: self.enterContext(patch.object(diagnostic, name,
                         side_effect=AssertionError('Model restoration forbidden')))
                         for name in ('restore_generator', 'restore_surrogate',
                                      'restore_victim', 'restore_clean_reference')}
        self.bundle = metadata_bundle(self.root)

    def prepare(self, name='prepared'):
        return diagnostic.prepare(self.bundle, self.root / 'results' / name)

    def assert_no_restores(self):
        for mock in self.restores.values():
            mock.assert_not_called()

    def test_check_prepare_and_validate_are_metadata_only(self):
        with patch.object(diagnostic, 'load_normalizer', side_effect=AssertionError('array read')) as norm, \
                patch.object(diagnostic, 'read_role_part', side_effect=AssertionError('graph read')) as read, \
                patch.object(diagnostic, 'load_bundle', return_value=self.bundle), \
                patch('sys.argv', ['copy_diagnostic', '--check-only']), redirect_stdout(io.StringIO()) as out:
            diagnostic.main()
            self.assertIn('COPY_DIAGNOSTIC_PREFLIGHT_OK', out.getvalue())
            self.assertFalse((self.root / 'results').exists())
            plan = self.prepare()
            content = diagnostic.validate_plan(self.bundle, plan)
            norm.assert_not_called()
            read.assert_not_called()
        spec = content['protocol']
        self.assertEqual((len(spec['sources']), spec['expected_targets'], spec['expected_moving']), (128, 112, 33))
        self.assertEqual(set(spec['assets']), {'generator', 'surrogate', 'victim', 'clean_reference',
                                             'normalizer', 'support'})
        self.assertEqual(spec['optimizer_steps'], 0)
        self.assertFalse(spec['automatic_followup'])
        registration = json.loads((plan.parent / 'registration.json').read_text())
        self.assertFalse(registration['graph_arrays_loaded'])
        self.assertFalse(registration['model_checkpoints_loaded'])
        self.assertEqual(registration['plan_sha256'], diagnostic.sha256(plan))
        self.assert_no_restores()

    def test_validation_rejects_plan_snapshot_manifest_source_and_population_tampering(self):
        for kind in ('plan', 'snapshot', 'manifest', 'source', 'population', 'asset declaration'):
            with self.subTest(kind=kind):
                plan = self.prepare(kind.replace(' ', '_'))
                original_manifest = self.bundle.manifest_path.read_bytes()
                original_sources = deepcopy(self.bundle.config['roles']['dev'])
                original_assets = deepcopy(self.bundle.manifest['assets'])
                if kind == 'plan':
                    plan.write_text(plan.read_text() + '\n')
                elif kind == 'snapshot':
                    path = plan.parent / 'source_snapshot.tar'
                    path.write_bytes(path.read_bytes() + b'changed')
                elif kind == 'manifest':
                    self.bundle.manifest_path.write_text(self.bundle.manifest_path.read_text() + '\n')
                elif kind == 'population':
                    self.bundle.config['roles']['dev'][0]['scenario_id'] = 'substituted-scene'
                elif kind == 'asset declaration':
                    self.bundle.manifest['assets']['generator']['sha256'] = '0' * 64
                try:
                    if kind == 'source':
                        original_sha = diagnostic.sha256
                        source = Path(diagnostic.__file__).resolve()
                        def changed_hash(path):
                            return '0' * 64 if Path(path).resolve() == source else original_sha(path)
                        with patch.object(diagnostic, 'sha256', side_effect=changed_hash), \
                                self.assertRaisesRegex(ValueError, 'Prepared sources'):
                            diagnostic.validate_plan(self.bundle, plan)
                    else:
                        with self.assertRaisesRegex(ValueError, 'Prepared'):
                            diagnostic.validate_plan(self.bundle, plan)
                finally:
                    self.bundle.manifest_path.write_bytes(original_manifest)
                    self.bundle.config['roles']['dev'] = original_sources
                    self.bundle.manifest['assets'] = original_assets
        self.assert_no_restores()

    def test_prepare_rejects_existing_empty_directory_and_paths_outside_results(self):
        output = self.root / 'results' / 'existing'
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            diagnostic.prepare(self.bundle, output)
        self.assertEqual(list(output.iterdir()), [])
        plan = self.prepare()
        before = {p.name: p.read_bytes() for p in plan.parent.iterdir()}
        with self.assertRaises(FileExistsError):
            diagnostic.prepare(self.bundle, plan.parent)
        self.assertEqual(before, {p.name: p.read_bytes() for p in plan.parent.iterdir()})
        for path in (self.root, self.root / 'results', self.root / 'outside',
                     self.root / 'results' / '..' / 'escaped'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'new child'):
                diagnostic.prepare(self.bundle, path)
        with self.assertRaisesRegex(ValueError, 'prepared plan.json'):
            diagnostic.validate_plan(self.bundle, plan.with_name('renamed.json'))

    def test_failed_run_keeps_exclusive_claim_and_second_run_never_reads(self):
        plan = self.prepare()
        with patch.object(diagnostic, 'read_dev_inputs', side_effect=RuntimeError('synthetic stop')) as read:
            with self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                diagnostic.run(self.bundle, plan)
            read.assert_called_once()
        expected = {name: (plan.parent / name).read_bytes() for name in
                    ('execution_claim.json', 'failure.json', 'graph_access_ledger.json')}
        self.assertEqual(json.loads(expected['graph_access_ledger.json']), [])
        self.assertEqual(json.loads(expected['failure.json'])['type'], 'RuntimeError')
        with patch.object(diagnostic, 'read_dev_inputs', side_effect=AssertionError('second read')) as read:
            with self.assertRaises(FileExistsError):
                diagnostic.run(self.bundle, plan)
            read.assert_not_called()
        self.assertEqual(expected, {name: (plan.parent / name).read_bytes() for name in expected})
        self.assert_no_restores()

    def synthetic_dev_reader(self, *, corruption=None):
        sources = self.bundle.config['roles']['dev']
        parts = [scene(row['scenario_id'], 'moving' if i < 33 else 'slow', no_negative=i >= 112)
                 for i, row in enumerate(sources)]
        if corruption == 'stratum':
            parts[0]['motion_regime'][:] = False
        if corruption == 'target_count':
            parts[-1] = scene(sources[-1]['scenario_id'])
        normalizer = dict(mean=np.zeros(784, dtype=np.float64), scale=np.ones(784, dtype=np.float64))
        seen = []
        def reader(bundle, source, role, supplied_normalizer, ledger):
            self.assertIs(bundle, self.bundle)
            self.assertEqual(role, 'dev')
            self.assertEqual(source, sources[len(seen)])
            self.assertIs(supplied_normalizer, normalizer)
            self.assertEqual(source['split'], 'train')
            seen.append(source['scenario_id'])
            ledger.append(dict(split='train', role='audit' if corruption == 'ledger' else role,
                               status='READ_COMPLETE'))
            return parts[len(seen) - 1]
        return reader, normalizer, seen

    def test_read_dev_uses_only_registered_128_and_freezes_112_targets_with_33_moving(self):
        reader, normalizer, seen = self.synthetic_dev_reader()
        ledger = []
        with patch.object(diagnostic, 'load_normalizer', return_value=normalizer), \
                patch.object(diagnostic, 'read_role_part', side_effect=reader) as read:
            inputs, returned_normalizer, population = diagnostic.read_dev_inputs(self.bundle, ledger)
        self.assertIs(returned_normalizer, normalizer)
        self.assertEqual(read.call_count, 128)
        self.assertEqual(seen, [r['scenario_id'] for r in self.bundle.config['roles']['dev']])
        self.assertEqual((len(ledger), len(population)), (128, 112))
        self.assertEqual(sum(r['motion_regime'] == 'moving' for r in population), 33)
        self.assertEqual(sum(r['motion_regime'] == 'slow' for r in population), 79)
        for mapping in inputs.values():
            self.assertEqual(set(mapping), {'dev'})
        self.assertEqual(inputs['targets']['dev'], [diagnostic.choose_target(p) for p in inputs['parts']['dev']])
        self.assertTrue(all(r['true_label'] == 0 and r['eligible'] and not r['is_attachment']
                            and not r['is_outer'] for r in population))
        self.assert_no_restores()

    def test_read_dev_rejects_changed_strata_target_count_or_nondev_ledger(self):
        for corruption in ('stratum', 'target_count', 'ledger'):
            reader, normalizer, _ = self.synthetic_dev_reader(corruption=corruption)
            with self.subTest(corruption=corruption), \
                    patch.object(diagnostic, 'load_normalizer', return_value=normalizer), \
                    patch.object(diagnostic, 'read_role_part', side_effect=reader), \
                    self.assertRaisesRegex(ValueError, 'Frozen DEV population'):
                diagnostic.read_dev_inputs(self.bundle, [])
        self.assert_no_restores()

    def test_evaluate_arms_capture_exactly_one_forward_per_target_and_identical_clean_csv(self):
        bundle, inputs, _, models = prepared()
        anchor, normalizer, _, _, _ = copy_fixture()
        before = {name: state_hash(model) for name, model in dict(anchor=anchor, **models).items()}
        before_inputs = diagnostic._fingerprint(diagnostic._dev_view(bundle, inputs))
        before_rng = diagnostic._rng_fingerprint()
        created, observed, handles = [], [], []
        def factory(*args, **kwargs):
            generator = CopyConsistentGenerator(*args, **kwargs)
            created.append(generator)
            handles.append(generator.register_forward_pre_hook(
                lambda module, args: observed.append((module.consistency_mode, args[2].tolist(),
                                                      torch.is_grad_enabled()))))
            return generator
        results, records = {}, {}
        try:
            with patch.object(diagnostic, 'CopyConsistentGenerator', side_effect=factory):
                for mode in diagnostic.ARMS:
                    output = self.root / mode
                    results[mode] = diagnostic.evaluate_arm(bundle, inputs, anchor, normalizer,
                                                            models, mode, output)
                    records[mode] = json.loads((output / 'copy_rows.json').read_text())['rows']
                    self.assertEqual(created[-1]._forward_hooks, {})
                    self.assertTrue(created[-1].frozen_postprocess)
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in created[-1].parameters()))
                    self.assertEqual(json.loads((output / 'summary.json').read_text()), results[mode])
        finally:
            for handle in handles:
                handle.remove()
        expected = [(part, target) for part, target in zip(inputs['parts']['dev'], inputs['targets']['dev'])
                    if target is not None]
        self.assertEqual(observed, [(mode, [target], False) for mode in diagnostic.ARMS for _, target in expected])
        self.assertEqual((self.root / 'identity' / 'clean_pair_scores.csv').read_bytes(),
                         (self.root / 'pair_mean' / 'clean_pair_scores.csv').read_bytes())
        self.assertEqual(results['identity']['target_identity_sha256'], results['pair_mean']['target_identity_sha256'])
        for mode in diagnostic.ARMS:
            self.assertEqual(results[mode]['raw_copy_rows']['rows'], 2)
            self.assertEqual(results[mode]['shared_parent_generator_state_sha256'], before['anchor'])
            for row, (part, target) in zip(records[mode], expected):
                self.assertEqual((row['scenario_id'], row['pair_index']), (part['scenario_id'], target))
                self.assertEqual(row['clean30'], part['x'][target, list(SELECTED_COORDINATES)].tolist())
                self.assertEqual(len(row['output30']), 30)
        for original, projected in zip(records['identity'], records['pair_mean']):
            expected_output = torch.tensor(original['output30'])
            for group in COPY_GROUPS:
                positions = [SELECTED_COORDINATES.index(c) for c in group]
                expected_output[positions] = expected_output[positions].double().mean().float()
                self.assertEqual(len({projected['output30'][i] for i in positions}), 1)
            torch.testing.assert_close(torch.tensor(projected['output30']), expected_output, rtol=0, atol=1e-7)
        self.assertEqual(before, {name: state_hash(model) for name, model in dict(anchor=anchor, **models).items()})
        self.assertEqual(before_inputs, diagnostic._fingerprint(diagnostic._dev_view(bundle, inputs)))
        self.assertEqual(before_rng, diagnostic._rng_fingerprint())
        for model in models.values():
            self.assertEqual(len(model.calls), 10)  # 3 clean + 2 triggered, in each arm.
            self.assertTrue(all(not c['grad_enabled'] and not c['training'] for c in model.calls))

    def test_failed_capture_removes_hook_even_when_generator_forward_raises(self):
        bundle, inputs, _, models = prepared()
        anchor, normalizer, _, _, _ = copy_fixture()
        captured = []
        def wrong_target(_bundle, _inputs, generator, _models, _directory, progress=None):
            captured.append(generator)
            part = inputs['parts']['dev'][0]
            injector = SceneInjector(generator, bundle.config['shuffle_seed'])
            injector.bind(part)
            wrong = (inputs['targets']['dev'][0] + 1) % len(part['x'])
            with torch.no_grad():
                generator(injector.features, injector.context, torch.tensor([wrong]))
            self.fail('The capture hook must reject a changed target')
        with patch.object(diagnostic, 'evaluate_dev', side_effect=wrong_target), \
                self.assertRaisesRegex(ValueError, 'Captured generator input differs'):
            diagnostic.evaluate_arm(bundle, inputs, anchor, normalizer, models,
                                    'pair_mean', self.root / 'failed')
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]._forward_hooks, {})
        self.assertFalse((self.root / 'failed').exists())

    def test_missing_capture_and_existing_arm_directory_fail_without_summary_overwrite(self):
        bundle, inputs, _, models = prepared()
        anchor, normalizer, _, _, _ = copy_fixture()
        captured = []
        def no_forward(_bundle, _inputs, generator, _models, _directory, progress=None):
            captured.append(generator)
            return {}
        with patch.object(diagnostic, 'evaluate_dev', side_effect=no_forward), \
                self.assertRaisesRegex(ValueError, 'Missing generator-output records'):
            diagnostic.evaluate_arm(bundle, inputs, anchor, normalizer, models,
                                    'identity', self.root / 'missing')
        self.assertEqual(captured[0]._forward_hooks, {})
        output = self.root / 'occupied'
        output.mkdir()
        sentinel = output / 'summary.json'
        sentinel.write_text('existing result')
        with self.assertRaises(FileExistsError):
            diagnostic.evaluate_arm(bundle, inputs, anchor, normalizer, models, 'identity', output)
        self.assertEqual(sentinel.read_text(), 'existing result')
        self.assertTrue(all(not model.calls for model in models.values()))


if __name__ == '__main__':
    unittest.main()
