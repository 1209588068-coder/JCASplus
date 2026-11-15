"""Metadata-only integrity/launch guards; no CUDA or graph deserialization."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from spear_current import sensitivity_extensions as experiment
from spear_current.extension_protocol import case_matrix
from spear_current.support_schema import SELECTED_COORDINATES, support_groups


class SensitivityIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        folder = root / 'results' / 'metadata_fixture'
        folder.mkdir(parents=True)
        source = root / 'runtime.py'
        source.write_text('frozen source bytes\n')
        cache = root / 'cache.bin'
        cache.write_bytes(b'not a torch pickle; must never deserialize')
        snapshot = folder / 'source_snapshot.tar'
        snapshot.write_bytes(b'not an archive; validate bytes only')
        groups = support_groups('semantic')
        triplets = [g for g in groups if len(g) == 3]
        singletons = [g for g in groups if len(g) == 1]
        active = {c for group in triplets[:5] + singletons[:9] for c in group}
        mask = [c in active for c in SELECTED_COORDINATES]
        spec = dict(version=experiment.VERSION, seed=experiment.SEED,
            source_mask=mask, cases=case_matrix(mask), victim_epochs=200,
            max_concurrent=2, preflight=list(experiment.PREFLIGHT),
            bound_files={str(cache): experiment.sha(cache)},
            sources={'runtime.py': experiment.sha(source)})
        path = folder / 'plan.json'
        path.write_text(json.dumps(spec))
        registration = folder / 'registration.json'
        registration.write_text(json.dumps(dict(plan_sha256=experiment.sha(path),
            source_snapshot_sha256=experiment.sha(snapshot))))
        return path, source, cache, snapshot

    def resign(self, path, spec):
        """Simulate changed registration too, exercising semantic grid guards."""
        path.write_text(json.dumps(spec))
        registration = path.parent / 'registration.json'
        value = json.loads(registration.read_text())
        value['plan_sha256'] = experiment.sha(path)
        registration.write_text(json.dumps(value))

    def test_valid_metadata_only_registration(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, _, _, _ = self.fixture(Path(folder))
            plan = experiment.verify(path)
            self.assertEqual(len(plan['cases']), 14)
            self.assertEqual(plan['cases'][1]['poison_count'], 0)

    def test_changed_cache_rejected(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, _, cache, _ = self.fixture(Path(folder))
            cache.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Bound source/checkpoint/cache'):
                experiment.verify(path)

    def test_changed_runtime_rejected(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, source, _, _ = self.fixture(Path(folder))
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Runtime changed'):
                experiment.verify(path)

    def test_changed_snapshot_rejected(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, _, _, snapshot = self.fixture(Path(folder))
            snapshot.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'plan/source snapshot changed'):
                experiment.verify(path)

    def test_wrong_expected_digest_rejected(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, _, _, _ = self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, 'plan/source snapshot changed'):
                experiment.verify(path, '0' * 64)

    def test_changed_plan_rejected_before_process_spawn(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)), \
                patch.object(experiment.subprocess, 'Popen') as spawn:
            path, _, _, _ = self.fixture(Path(folder))
            path.write_text(path.read_text() + '\n')
            with self.assertRaises(ValueError):
                experiment.launch(path)
            spawn.assert_not_called()

    def test_grid_and_execution_changes_rejected_even_if_resigned(self):
        changes = [('seed', 123), ('victim_epochs', 201), ('max_concurrent', 3),
                   ('preflight', []), ('cases', [])]
        for key, value in changes:
            with self.subTest(field=key), TemporaryDirectory() as folder, \
                    patch.object(experiment, 'ROOT', Path(folder)):
                path, _, _, _ = self.fixture(Path(folder))
                spec = json.loads(path.read_text())
                spec[key] = value
                self.resign(path, spec)
                with self.assertRaisesRegex(ValueError, 'experiment grid changed'):
                    experiment.verify(path)

    def test_effective_permissions_cannot_be_expanded_in_case(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)):
            path, _, _, _ = self.fixture(Path(folder))
            spec = json.loads(path.read_text())
            spec['cases'][2]['active_coordinates'] = list(SELECTED_COORDINATES)
            self.resign(path, spec)
            with self.assertRaisesRegex(ValueError, 'experiment grid changed'):
                experiment.verify(path)

    def test_detached_launch_is_exclusive_and_binds_plan_digest(self):
        with TemporaryDirectory() as folder, patch.object(experiment, 'ROOT', Path(folder)), \
                patch.object(experiment.subprocess, 'Popen', return_value=Mock(pid=123)) as spawn:
            path, _, _, _ = self.fixture(Path(folder))
            receipt = experiment.launch(path)
            self.assertEqual(receipt['pid'], 123)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            self.assertIn('--queue', spawn.call_args.args[0])
            self.assertIn('--expected-sha256', spawn.call_args.args[0])
            with self.assertRaises(FileExistsError):
                experiment.launch(path)
            self.assertEqual(spawn.call_count, 1)


if __name__ == '__main__':
    unittest.main()
