"""Receipt and supervisor regression tests without starting training processes."""
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spear_current import av2_formal as queue


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def summary(value):
    normal = {group: dict(clean_pair_auc=value, clean_pair_ap=value - .1,
                          clean_pair_positive_recall=value - .2)
              for group in ('all', 'slow', 'moving')}
    return dict(status='EVALUATION_COMPLETE', targets=100, scenes=128, normal_pairs=2223,
        metrics=dict(victim=dict(normal=normal, targets=dict(all=dict(incremental_asr=value, targets=100),
            slow=dict(incremental_asr=value, targets=80), moving=dict(incremental_asr=value, targets=20))),
            clean_reference=dict(normal={group: dict(clean_pair_auc=.95, clean_pair_ap=.85,
                clean_pair_positive_recall=.75) for group in ('all', 'slow', 'moving')},
                targets={group: dict(targets=count, incremental_asr=0.)
                    for group, count in (('all', 100), ('slow', 80), ('moving', 20))})))


def complete(output, seed, value=.8):
    folder = Path(output) / f'seed{seed}'
    folder.mkdir(exist_ok=True)
    (folder / 'generator.pt').write_bytes(b'synthetic nonexecuted checkpoint')
    paths = {}
    hashes = {}
    for role in ('dev', 'audit'):
        path = folder / f'evaluation_{role}/summary.json'
        write(path, summary(value))
        paths[role] = str(path); hashes[role] = queue.sha(path)
    receipt = dict(status='SEED_COMPLETE_REVIEW_REQUIRED', seed=seed,
        plan_sha256=queue.sha(Path(output) / 'plan.json'),
        generator_file_sha256=queue.sha(folder / 'generator.pt'), result_paths=paths,
        result_sha256=hashes, full_pipeline_repetition=True, frozen_test_arrays_loaded=False)
    write(folder / 'completion.json', receipt)
    return folder


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        write(self.output / 'plan.json', {})

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_valid_complete_seeds_contribute_failed_is_not_zero(self):
        complete(self.output, queue.SEEDS[0], .7)
        complete(self.output, queue.SEEDS[1], .9)
        write(self.output / f'seed{queue.SEEDS[2]}/failure.json', dict(status='FAILED'))
        result = queue.aggregate(self.output)
        self.assertEqual(result['completed_seeds'], queue.SEEDS[:2])
        self.assertEqual(result['completed'], 2)
        self.assertEqual([r['seed'] for r in result['failed_seeds']], [queue.SEEDS[2]])
        metric = result['metrics']['dev.victim.targets.all.incremental_asr']
        self.assertEqual(metric['n'], 2)
        self.assertAlmostEqual(metric['mean'], .8)
        self.assertAlmostEqual(metric['sample_sd'], .2 / 2 ** .5)
        self.assertFalse(result['failure_is_zero_asr'])
        paired = result['paired_differences']['dev.all.clean_pair_auc.victim_minus_clean_reference']
        self.assertAlmostEqual(paired['mean'], -.15)
        # The same fixed target pool is not counted ten times as independent data.
        targets = result['metrics']['dev.victim.targets.all.targets']
        self.assertEqual(targets['mean'], 100)
        self.assertEqual(targets['sample_sd'], 0)
        self.assertIn('repeated targets are not new independent targets', result['independent_unit'])

    def test_single_completed_seed_has_unknown_sample_sd(self):
        complete(self.output, queue.SEEDS[0])
        result = queue.aggregate(self.output)
        self.assertIsNone(result['metrics']['audit.victim.targets.all.incremental_asr']['sample_sd'])
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['failed_seeds'], [])

    def test_invalid_receipts_or_artifacts_are_failed_and_do_not_abort(self):
        first = complete(self.output, queue.SEEDS[0])
        (first / 'evaluation_dev/summary.json').unlink()
        second = complete(self.output, queue.SEEDS[1])
        write(second / 'failure.json', dict(status='FAILED_AFTER_COMPLETION'))
        third = complete(self.output, queue.SEEDS[2])
        receipt = json.loads((third / 'completion.json').read_text())
        receipt['seed'] = 123
        write(third / 'completion.json', receipt)
        fourth = complete(self.output, queue.SEEDS[3])
        (fourth / 'generator.pt').write_bytes(b'changed checkpoint')
        fifth = complete(self.output, queue.SEEDS[4])
        (fifth / 'evaluation_audit/summary.json').write_text('{broken JSON')
        complete(self.output, queue.SEEDS[5])
        result = queue.aggregate(self.output)
        self.assertEqual(result['completed_seeds'], [queue.SEEDS[5]])
        self.assertEqual({r['seed'] for r in result['failed_seeds']}, set(queue.SEEDS[:5]))
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['metrics']['dev.victim.targets.all.incremental_asr']['n'], 1)

    def test_bad_completion_status_plan_or_evaluation_status_not_accepted(self):
        folders = [complete(self.output, seed) for seed in queue.SEEDS[:3]]
        receipt = json.loads((folders[0] / 'completion.json').read_text())
        receipt['status'] = 'RUNNING'; write(folders[0] / 'completion.json', receipt)
        receipt = json.loads((folders[1] / 'completion.json').read_text())
        receipt['plan_sha256'] = '0' * 64; write(folders[1] / 'completion.json', receipt)
        path = folders[2] / 'evaluation_dev/summary.json'
        record = json.loads(path.read_text()); record['status'] = 'RUNNING'; write(path, record)
        receipt = json.loads((folders[2] / 'completion.json').read_text())
        receipt['result_sha256']['dev'] = queue.sha(path)
        write(folders[2] / 'completion.json', receipt)
        result = queue.aggregate(self.output)
        self.assertEqual(result['completed'], 0)
        self.assertEqual(len(result['failed_seeds']), 3)
        self.assertEqual(result['metrics'], {})

    def test_all_ten_valid_completions_are_required(self):
        for seed in queue.SEEDS:
            complete(self.output, seed)
        result = queue.aggregate(self.output)
        self.assertEqual(result['status'], 'ALL_TEN_COMPLETE')
        self.assertEqual(result['completed'], 10)
        self.assertEqual(result['failed_seeds'], [])

    def test_malformed_required_metrics_fail_validation_without_aborting(self):
        for index, seed in enumerate(queue.SEEDS[:4]):
            folder = complete(self.output, seed)
            path = folder / 'evaluation_dev/summary.json'
            record = json.loads(path.read_text())
            victim = record['metrics']['victim']
            if index == 0:
                victim['normal']['all']['clean_pair_auc'] = 'not a number'
            elif index == 1:
                victim['targets']['all']['incremental_asr'] = float('nan')
            elif index == 2:
                del victim['targets']['moving']
            else:
                victim['targets']['all']['targets'] = True
            write(path, record)
            receipt = json.loads((folder / 'completion.json').read_text())
            receipt['result_sha256']['dev'] = queue.sha(path)
            write(folder / 'completion.json', receipt)
        complete(self.output, queue.SEEDS[4])
        result = queue.aggregate(self.output)
        self.assertEqual(result['completed_seeds'], [queue.SEEDS[4]])
        self.assertEqual(len(result['failed_seeds']), 4)
        self.assertEqual(result['metrics']['dev.victim.targets.all.incremental_asr']['n'], 1)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        self.plan_path = self.output / 'plan.json'
        write(self.plan_path, {})
        self.plan = dict(output_dir=str(self.output), source_snapshot=str(self.output), seeds=queue.SEEDS)

    def tearDown(self):
        self.temporary.cleanup()

    def test_consecutive_worker_failures_do_not_stop_remaining_seeds(self):
        launched = []

        def popen(command, **kwargs):
            seed = int(command[-1]); launched.append(seed)
            code = 1 if seed in queue.SEEDS[:3] else 0
            if code == 0:
                complete(self.output, seed)
            return SimpleNamespace(pid=100 + len(launched), poll=lambda: code)

        with patch.object(queue, 'verify', return_value=self.plan), patch.object(queue.subprocess, 'Popen', popen):
            queue.queue(self.plan_path)
        self.assertEqual(launched, queue.SEEDS)
        self.assertFalse((self.output / 'completion.json').exists())
        state = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(state['status'], 'FINISHED_WITH_FAILURES_REVIEW_REQUIRED')
        self.assertEqual(state['completed'], 7)
        self.assertEqual(state['failed'], 3)

    def test_launch_is_not_repeated_or_overwritten(self):
        with patch.object(queue, 'verify', return_value=self.plan), patch.object(queue.subprocess, 'Popen',
                return_value=SimpleNamespace(pid=123)) as process:
            queue.launch(self.plan_path)
            with self.assertRaises(FileExistsError):
                queue.launch(self.plan_path)
            self.assertEqual(process.call_count, 1)

    def test_queue_lock_prevents_parallel_supervisors(self):
        with (self.output / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(queue, 'verify', return_value=self.plan), patch.object(queue.subprocess, 'Popen') as process:
                with self.assertRaises(BlockingIOError):
                    queue.queue(self.plan_path)
                process.assert_not_called()

    def test_existing_registration_cannot_be_overwritten(self):
        before = self.plan_path.read_bytes()
        with self.assertRaises(FileExistsError):
            queue.register(self.output, acceptance='unused', attachment_count=5865)
        self.assertEqual(self.plan_path.read_bytes(), before)

    def test_orphan_log_is_preserved_and_later_seeds_run(self):
        first = self.output / f'seed{queue.SEEDS[0]}'
        first.mkdir(); (first / 'run.log').write_text('retained partial worker output\n')
        launched = []

        def popen(command, **kwargs):
            seed = int(command[-1]); launched.append(seed)
            complete(self.output, seed)
            return SimpleNamespace(pid=100 + len(launched), poll=lambda: 0)

        with patch.object(queue, 'verify', return_value=self.plan), patch.object(queue.subprocess, 'Popen', popen):
            queue.queue(self.plan_path)
        self.assertEqual(launched, queue.SEEDS[1:])
        self.assertEqual((first / 'run.log').read_text(), 'retained partial worker output\n')
        failure = json.loads((first / 'failure.json').read_text())
        self.assertEqual(failure['status'], 'INTERRUPTED_PARTIAL_SEED_REVIEW_REQUIRED')
        state = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(state['completed'], 9)
        self.assertEqual(state['failed'], 1)
        self.assertFalse((self.output / 'completion.json').exists())

    def test_existing_live_worker_prevents_a_second_gpu_worker(self):
        first = self.output / f'seed{queue.SEEDS[0]}'
        first.mkdir(); (first / 'run.log').write_text('live worker\n')
        write(first / 'launch.json', dict(pid=12345))
        with patch.object(queue, 'verify', return_value=self.plan), patch.object(queue.os, 'kill', return_value=None), \
                patch.object(queue.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'still be alive'):
                queue.queue(self.plan_path)
            process.assert_not_called()
        self.assertFalse((first / 'failure.json').exists())


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / 'experiment'
        self.data = self.output / 'data'
        self.data.mkdir(parents=True)
        (self.root / 'spear_current').mkdir()
        (self.root / 'spear_current/module.py').write_text('# frozen synthetic source\n')
        write(self.root / 'FORMAL_REPETITION_PROTOCOL_20261104.json', {})
        write(self.root / 'FINAL_METHOD_SELECTION_20261103.json', {})
        write(self.data / 'registration.json', dict(scope='synthetic test only'))
        self.summary = dict(status='PREPARED', original_test_arrays_loaded=False,
            train=dict(scenes=115490), dev=dict(scenes=128), audit=dict(scenes=1280),
            attachment_budget=dict(count=5865), max_nodes_per_shard=16384,
            asset_sha256={'registration.json': queue.sha(self.data / 'registration.json')})
        write(self.data / 'summary.json', self.summary)
        self.acceptance = self.root / 'acceptance.json'
        write(self.acceptance, dict(status='PASS', sources={
            'spear_current/module.py': queue.sha(self.root / 'spear_current/module.py')}))
        self.patcher = patch.object(queue, 'ROOT', self.root); self.patcher.start()

    def tearDown(self):
        self.patcher.stop(); self.temporary.cleanup()

    def register(self, **kwargs):
        options = dict(acceptance=self.acceptance, attachment_count=5865)
        options.update(kwargs)
        return queue.register(self.output, **options)

    def test_registration_binds_seed_list_budget_code_and_data(self):
        plan_path = self.register()
        plan = queue.verify(plan_path)
        self.assertEqual(plan['seeds'], queue.SEEDS)
        self.assertEqual(plan['source_selection']['attachment_count'], 5865)
        self.assertTrue(plan['full_pipeline_repetition'])
        self.assertFalse(plan['frozen_test_arrays_loaded'])
        self.assertEqual(plan['data_summary_sha256'], queue.sha(self.data / 'summary.json'))
        self.assertEqual(plan['source_sha256']['spear_current/module.py'],
            queue.sha(Path(plan['source_snapshot']) / 'spear_current/module.py'))

    def test_incomplete_data_test_access_or_budget_mismatch_rejected(self):
        for change in (dict(status='PREPARING'), dict(original_test_arrays_loaded=True),
                       dict(attachment_budget=dict(count=26)), dict(max_nodes_per_shard=100)):
            write(self.data / 'summary.json', dict(self.summary, **change))
            with self.assertRaises(ValueError):
                self.register()
            self.assertFalse((self.output / 'plan.json').exists())
        write(self.data / 'summary.json', self.summary)
        with self.assertRaises(ValueError):
            self.register(attachment_count=26)

    def test_changed_code_after_acceptance_rejected(self):
        (self.root / 'spear_current/module.py').write_text('# changed after acceptance\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.register()

    def test_changed_registered_metadata_rejected_before_launch(self):
        plan_path = self.register()
        write(self.data / 'registration.json', dict(scope='changed metadata'))
        with self.assertRaisesRegex(ValueError, 'registration or population changed'):
            queue.verify(plan_path)

    def test_changed_snapshot_and_changed_seed_list_rejected(self):
        plan_path = self.register()
        plan = json.loads(plan_path.read_text())
        snapshot_source = Path(plan['source_snapshot']) / 'spear_current/module.py'
        original = snapshot_source.read_bytes()
        snapshot_source.write_text('# corrupt frozen source\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
            queue.verify(plan_path)
        snapshot_source.write_bytes(original)
        plan['seeds'] = queue.SEEDS[:-1]
        write(plan_path, plan)
        write(self.output / 'registration.json', dict(plan_sha256=queue.sha(plan_path)))
        with self.assertRaisesRegex(ValueError, 'seed list changed'):
            queue.verify(plan_path)


if __name__ == '__main__':
    unittest.main()
