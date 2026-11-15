"""Recovery provenance and fail-closed registration tests; no training jobs."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from spear_current import av2_formal as runner


class RecoveryRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='av2_recovery_contract_')
        self.root = Path(self.temporary.name)
        self.source = self.root / 'spear_current/probe.py'
        self.source.parent.mkdir()
        self.source.write_text('VALUE = 1\n')
        for name in ('FORMAL_REPETITION_PROTOCOL_20261104.json',
                     'FINAL_METHOD_SELECTION_20261103.json'):
            runner.write_json(self.root / name, {})
        self.acceptance = self.root / 'acceptance.json'
        runner.write_json(self.acceptance, dict(status='PASS',
            sources={'spear_current/probe.py': runner.sha(self.source)}))
        self.old = self.root / 'old'
        self.data = self.old / 'data'
        runner.write_json(self.data / 'summary.json', dict(
            status='PREPARED', original_test_arrays_loaded=False,
            train=dict(scenes=512), dev=dict(scenes=128), audit=dict(scenes=256),
            attachment_budget=dict(count=26), max_nodes_per_shard=16384,
            asset_sha256={}))
        self.root_patch = patch.object(runner, 'ROOT', self.root)
        self.root_patch.start()
        self.prior_path = runner.register(self.old, acceptance=self.acceptance,
                                          attachment_count=26)
        self.first_fit = self.make_fit(runner.SEEDS[0], 200)
        self.second_fit = self.make_fit(runner.SEEDS[1], 7)

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()

    def make_fit(self, seed, next_epoch):
        folder = self.old / f'seed{seed}'
        fit = folder / 'selection/ranking_fit'
        fit.mkdir(parents=True)
        torch.save(dict(metadata=dict(seed=seed, role='fresh_author_GCN_ranking',
            epochs=200, epoch=next_epoch-1, next_epoch=next_epoch,
            optimizer_steps=next_epoch)), fit / 'latest.pt')
        (fit / 'epochs.jsonl').write_text('{"synthetic_receipt":true}\n')
        torch.save(dict(idx_train=torch.arange(4)), folder / 'initial_indices.pt')
        if next_epoch == 200:
            torch.save(dict(state_dict={}), fit / 'model.pt')
            runner.write_json(fit / 'summary.json', dict(epochs=200))
        return fit

    def recover(self, **overrides):
        options = dict(max_nodes=16384, attachment_count=26)
        options.update(overrides)
        return runner.recovery_registration(self.prior_path, self.data, **options)

    def new_output(self, name='new'):
        output = self.root / name
        output.mkdir()
        (output / 'data').symlink_to(self.data, target_is_directory=True)
        return output

    def rewrite_prior(self, transform):
        plan = json.loads(self.prior_path.read_text())
        transform(plan)
        runner.write_json(self.prior_path, plan)
        runner.write_json(self.old / 'registration.json',
                          dict(plan_sha256=runner.sha(self.prior_path)))

    def test_complete_and_partial_ranking_provenance_are_bound(self):
        prior_bytes = self.prior_path.read_bytes()
        receipt = self.recover()
        self.assertEqual(receipt['ranking_fits'][str(runner.SEEDS[0])]['completed_epochs'], 200)
        self.assertEqual(receipt['ranking_fits'][str(runner.SEEDS[1])]['completed_epochs'], 7)
        self.assertEqual(receipt['additional_training_budget'], 0)
        self.assertEqual(receipt['prior_plan_sha256'], runner.sha(self.prior_path))
        for path, digest in receipt['asset_sha256'].items():
            self.assertEqual(runner.sha(path), digest)
        output = self.new_output()
        path = runner.register(output, acceptance=self.acceptance, attachment_count=26,
                               recovery_from=self.prior_path)
        plan = runner.verify(path)
        prior = json.loads(prior_bytes)
        for key in ('seeds', 'training', 'source_selection', 'seed_offsets',
                    'support', 'checkpoint_selection', 'evaluation', 'max_nodes'):
            self.assertEqual(plan[key], prior[key])
        self.assertEqual(plan['recovery']['ranking_fits'], receipt['ranking_fits'])
        self.assertEqual(self.prior_path.read_bytes(), prior_bytes)

    def test_data_batching_and_budget_changes_are_rejected(self):
        for options in (dict(max_nodes=8192), dict(attachment_count=27)):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'data, batching or poison'):
                self.recover(**options)
        different_data = self.root / 'different_data'
        runner.write_json(different_data / 'summary.json', dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'data, batching or poison'):
            runner.recovery_registration(self.prior_path, different_data,
                max_nodes=16384, attachment_count=26)

    def test_malformed_checkpoint_metadata_is_rejected(self):
        path = self.first_fit / 'latest.pt'
        original = torch.load(path, weights_only=True)
        for key, value in (('seed', 99), ('role', 'victim'), ('epochs', 201),
                           ('next_epoch', 0), ('next_epoch', 201),
                           ('next_epoch', True), ('optimizer_steps', 199), ('epoch', 198)):
            saved = dict(metadata=dict(original['metadata']))
            saved['metadata'][key] = value
            torch.save(saved, path)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'checkpoint'):
                self.recover()
        torch.save(original, path)

    def test_existing_queue_lock_refuses_parallel_registration(self):
        with (self.old / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.recover()

    def test_live_prior_worker_is_refused_without_signalling_any_process(self):
        pid = os.getpid()
        runner.write_json(self.old / 'status.json', dict(pid=pid))
        read_bytes = Path.read_bytes
        proc = Path(f'/proc/{pid}/cmdline')

        def fake_cmdline(path):
            if path == proc:
                return (b'python\0-m\0spear_current.av2_formal\0--run\0'
                        + str(self.prior_path).encode() + b'\0')
            return read_bytes(path)

        with patch.object(Path, 'read_bytes', fake_cmdline), \
                self.assertRaisesRegex(RuntimeError, 'Stop the prior queue'):
            self.recover()

    def test_later_stages_and_missing_original_indices_are_refused(self):
        later = self.first_fit.parent / 'ranking.pt'
        later.write_bytes(b'synthetic completed ranking')
        with self.assertRaisesRegex(ValueError, 'only interrupted ranking'):
            self.recover()
        later.unlink()
        (self.old / f'seed{runner.SEEDS[0]}/initial_indices.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'original per-seed indices'):
            self.recover()

    def test_no_available_checkpoint_is_refused(self):
        for fit in (self.first_fit, self.second_fit):
            (fit / 'latest.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'No saved ranking'):
            self.recover()

    def test_changed_acceptance_source_is_rejected_before_registration(self):
        self.source.write_text('VALUE = 2\n')
        output = self.new_output()
        with self.assertRaisesRegex(ValueError, 'Source changed after acceptance'):
            runner.register(output, acceptance=self.acceptance, attachment_count=26,
                            recovery_from=self.prior_path)
        self.assertFalse((output / 'plan.json').exists())
        self.assertFalse((output / 'source_snapshot').exists())

    def test_changed_prior_training_contract_cannot_be_silently_replaced(self):
        self.rewrite_prior(lambda p: p['training'].update(base_epochs=199))
        output = self.new_output()
        with self.assertRaisesRegex(ValueError, 'prior experiment contract: training'):
            runner.register(output, acceptance=self.acceptance, attachment_count=26,
                            recovery_from=self.prior_path)
        self.assertFalse((output / 'plan.json').exists())
        self.assertFalse((output / 'source_snapshot').exists())

    def test_recovery_source_hashes_are_verified_after_registration(self):
        output = self.new_output()
        path = runner.register(output, acceptance=self.acceptance, attachment_count=26,
                               recovery_from=self.prior_path)
        runner.verify(path)
        checkpoint = self.first_fit / 'latest.pt'
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Recovery checkpoint or prior registration changed'):
            runner.verify(path)


if __name__ == '__main__':
    unittest.main()
