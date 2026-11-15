"""Resume bookkeeping tests: synthetic JSON only, no models, data or processes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnostics import resume_author_queue as resume


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_case(root, method='spear', seed=20260621):
    child = root / f'seed{seed}' / method
    summary = child / 'development' / 'summary.json'
    write_json(summary, dict(synthetic=True, targets=112))
    path = child / 'completion.json'
    write_json(path, dict(status='AUTHOR_CASE_COMPLETE', method=method, seed=seed,
                         summary_sha256=digest(summary),
                         population_identity_sha256='c' * 64))
    return path, summary


def failed_case(root, method='spear', seed=20260622):
    path = root / f'seed{seed}' / method / 'failure.json'
    # The native failure schema has no method or seed: the registered path
    # identifies them, unlike completion records where both must be checked.
    write_json(path, dict(status='AUTHOR_CASE_FAILED', error='synthetic finite-check failure'))
    return path


def case_keys(rows):
    return {(row['method'], row['seed']) for row in rows}


class AuthorResumeClassificationTests(unittest.TestCase):
    def test_empty_source_has_exactly_nine_pending_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            result = resume.classify_cases(Path(temp))
            self.assertEqual(result['completed'], [])
            self.assertEqual(result['failed'], [])
            expected = {(name, seed) for name in ('spear', 'ugba', 'dpgba')
                        for seed in (20260621, 20260622, 20260623)}
            self.assertEqual(case_keys(result['pending']), expected)
            self.assertEqual(len(result['pending']), 9)

    def test_completed_failed_and_pending_are_disjoint_with_bound_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = {name: completed_case(root, name) for name in ('spear', 'ugba', 'dpgba')}
            failure = failed_case(root)
            files = list(root.rglob('*.json'))
            before = {str(path): digest(path) for path in files}
            result = resume.classify_cases(root)
            self.assertEqual([len(result[key]) for key in ('completed', 'failed', 'pending')], [3, 1, 5])
            for row in result['completed']:
                path, summary = sources[row['method']]
                self.assertEqual(row['seed'], 20260621)
                self.assertEqual(Path(row['completion_path']).resolve(), path.resolve())
                self.assertEqual(row['completion_sha256'], digest(path))
                self.assertEqual(row['summary_sha256'], digest(summary))
                self.assertEqual(row['population_identity_sha256'], 'c' * 64)
            row = result['failed'][0]
            self.assertEqual((row['method'], row['seed']), ('spear', 20260622))
            self.assertEqual(Path(row['failure_path']).resolve(), failure.resolve())
            self.assertEqual(row['failure_sha256'], digest(failure))
            completed, failed, pending = (case_keys(result[key]) for key in ('completed', 'failed', 'pending'))
            self.assertFalse(completed & failed or completed & pending or failed & pending)
            self.assertEqual(len(completed | failed | pending), 9)
            self.assertEqual(before, {str(path): digest(path) for path in root.rglob('*.json')})

    def test_changed_completed_summary_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, summary = completed_case(root)
            write_json(summary, dict(synthetic=True, targets=111))
            with self.assertRaises(ValueError):
                resume.classify_cases(root)

    def test_completion_and_failure_for_same_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            completed_case(root)
            failed_case(root, seed=20260621)
            with self.assertRaises(ValueError):
                resume.classify_cases(root)

    def test_started_case_without_terminal_is_not_called_pending(self):
        for populated in (False, True):
            with self.subTest(populated=populated), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                child = root / 'seed20260622' / 'ugba'
                child.mkdir(parents=True)
                if populated:
                    (child / 'progress.jsonl').write_text('{"stage":"started"}\n')
                with self.assertRaises(ValueError):
                    resume.classify_cases(root)

    def test_wrong_completion_identity_or_status_is_rejected(self):
        for field, value in (('method', 'ugba'), ('seed', 20260622), ('status', 'FIT_COMPLETE')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, _ = completed_case(root)
                data = json.loads(path.read_text())
                data[field] = value
                write_json(path, data)
                with self.assertRaises(ValueError):
                    resume.classify_cases(root)

    def test_wrong_failure_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = failed_case(root)
            write_json(path, dict(status='AUTHOR_CASE_COMPLETE'))
            with self.assertRaises(ValueError):
                resume.classify_cases(root)

    def test_missing_completed_summary_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = root / 'seed20260621' / 'spear'
            write_json(child / 'completion.json', dict(status='AUTHOR_CASE_COMPLETE', method='spear',
                seed=20260621, summary_sha256='a' * 64, population_identity_sha256='c' * 64))
            with self.assertRaises((FileNotFoundError, ValueError)):
                resume.classify_cases(root)


class AuthorResumeScheduleTests(unittest.TestCase):
    def test_remaining_non_spear_goes_first_without_omission_or_retry(self):
        pending = [dict(method='spear', seed=20260623), dict(method='dpgba', seed=20260623),
                   dict(method='ugba', seed=20260622), dict(method='dpgba', seed=20260622),
                   dict(method='ugba', seed=20260623)]
        original = deepcopy(pending)
        result = resume.schedule_pending(pending)
        self.assertEqual([(r['method'], r['seed']) for r in result], [
            ('ugba', 20260622), ('dpgba', 20260622), ('ugba', 20260623),
            ('dpgba', 20260623), ('spear', 20260623)])
        self.assertEqual(case_keys(result), case_keys(pending))
        self.assertEqual(len(result), len(pending))
        self.assertNotIn(('spear', 20260622), case_keys(result))
        self.assertEqual(pending, original)

    def test_all_allowed_pending_cases_remain_once(self):
        pending = [dict(method=name, seed=seed) for seed in (20260623, 20260621, 20260622)
                   for name in ('dpgba', 'spear', 'ugba')]
        result = resume.schedule_pending(pending)
        self.assertEqual(len(result), 9)
        self.assertEqual(case_keys(result), case_keys(pending))
        self.assertEqual([row['method'] for row in result[-3:]], ['spear'] * 3)
        self.assertEqual([row['seed'] for row in result[-3:]], [20260621, 20260622, 20260623])

    def test_invalid_or_duplicate_pending_case_is_rejected(self):
        for rows in ([dict(method='gta', seed=20260621)], [dict(method='spear', seed=1)],
                     [dict(method='ugba', seed=20260621)] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                resume.schedule_pending(rows)

    def test_empty_schedule_does_not_invent_work(self):
        self.assertEqual(resume.schedule_pending([]), [])


class AuthorResumeFinalizationTests(unittest.TestCase):
    @staticmethod
    def record(method, seed, asr):
        return dict(method=method, seed=seed, population_identity_sha256='c' * 64,
                    metrics=dict(victim=dict(targets=dict(all=dict(incremental_asr=asr)))))

    def test_failed_seed_is_missing_not_zero_or_success_only_mean(self):
        cases = [self.record('spear', 20260621, .91), self.record('spear', 20260623, .89)]
        for method, values in (('ugba', [.8, .9, 1.]), ('dpgba', [.7, .8, .9])):
            cases.extend(self.record(method, seed, asr)
                         for seed, asr in zip((20260621, 20260622, 20260623), values))
        failed = dict(method='spear', seed=20260622, failure_path='/synthetic/failure.json',
                      failure_sha256='f' * 64)
        value = dict(source_directory='/synthetic/source', preserved_states=dict(failed=[failed]),
                     scheduled_cases=[dict(method='ugba', seed=20260622),
                                      dict(method='dpgba', seed=20260622),
                                      dict(method='ugba', seed=20260623),
                                      dict(method='dpgba', seed=20260623),
                                      dict(method='spear', seed=20260623)])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resume.finalize(root, value, cases)
            summary = json.loads((root / 'summary.json').read_text())
            aggregate = summary['overall_incremental_asr']
            spear = aggregate['spear']
            self.assertEqual(spear['missing_seeds'], [20260622])
            self.assertFalse(spear['complete_seed_set'])
            self.assertIsNone(spear['mean'])
            self.assertIsNone(spear['sample_sd'])
            self.assertEqual(spear['per_seed'], [dict(seed=20260621, incremental_asr=.91),
                                               dict(seed=20260623, incremental_asr=.89)])
            self.assertFalse(summary['failed_cases_are_zero_asr'])
            self.assertEqual(summary['preserved_failed_cases'], [failed])
            self.assertEqual(len(summary['cases']), 8)
            for method, expected in (('ugba', .9), ('dpgba', .8)):
                self.assertTrue(aggregate[method]['complete_seed_set'])
                self.assertEqual(aggregate[method]['missing_seeds'], [])
                self.assertAlmostEqual(aggregate[method]['mean'], expected)
                self.assertAlmostEqual(aggregate[method]['sample_sd'], .1)
            complete = json.loads((root / 'completion.json').read_text())
            self.assertEqual(complete['completed_cases'], 8)
            self.assertEqual(complete['failed_cases'], 1)
            self.assertEqual(complete['summary_sha256'], digest(root / 'summary.json'))
            self.assertFalse(complete['automatic_followup'])

    def test_population_mismatch_is_rejected_before_writing_summary(self):
        cases = [self.record('spear', 20260621, .9), self.record('ugba', 20260621, .8)]
        cases[1]['population_identity_sha256'] = 'd' * 64
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'populations'):
                resume.finalize(root, {}, cases)
            self.assertFalse((root / 'summary.json').exists())
            self.assertFalse((root / 'completion.json').exists())

    def test_cancelled_continuation_never_launches_or_reads_historical_cases(self):
        for operation in (resume.run, resume.enqueue):
            with self.subTest(operation=operation.__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / 'CANCEL').touch()
                with patch.object(resume, 'validate', return_value=({}, {})), \
                        patch.object(resume, 'completed_record') as reader, \
                        patch.object(resume.subprocess, 'Popen') as jobs:
                    with self.assertRaisesRegex(ValueError, 'Cancelled'):
                        operation(None, root / 'continuation.json')
                    jobs.assert_not_called()
                    reader.assert_not_called()
                self.assertFalse((root / 'execution_claim.json').exists())
                self.assertFalse((root / 'launch_claim.json').exists())


if __name__ == '__main__':
    unittest.main()
