"""Recovery metadata checks; no CUDA initialization, cache or model loading."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch  # Complete dependency imports before mocking subprocess.Popen.

from spear_current import current_ablation_recovery as recovery


class RecoveryResourceClassificationTests(unittest.TestCase):
    def test_accelerator_oom_is_recognized_from_specific_cuda_evidence(self):
        failures = [
            dict(error_type='AcceleratorError', root_error_type='AcceleratorError',
                 error='AcceleratorError("CUDA error: out of memory")'),
            dict(error_type='AcceleratorError', root_error_type='AcceleratorError',
                 error='CUDA failure: cudaErrorMemoryAllocation'),
            dict(error_type='RuntimeError', root_error_type='OutOfMemoryError',
                 error='contextual wrapper around CUDA allocation failure'),
        ]
        for value in failures:
            with self.subTest(value=value):
                self.assertTrue(recovery.is_resource_oom(value))

    def test_other_accelerator_and_integrity_failures_are_not_oom(self):
        failures = [
            dict(error_type='AcceleratorError', root_error_type='AcceleratorError',
                 error='CUDA error: illegal memory access'),
            dict(error_type='AcceleratorError', root_error_type='AcceleratorError',
                 error='CUDA error: invalid device ordinal'),
            dict(error_type='RuntimeError', root_error_type='ValueError',
                 error='Registered plan/source snapshot changed'),
            dict(error_type='PermissionError', root_error_type='PermissionError',
                 error='Permission denied'),
            dict(error_type='RuntimeError', root_error_type='RuntimeError',
                 error='Case integrity/process failure'),
        ]
        for value in failures:
            with self.subTest(value=value):
                self.assertFalse(recovery.is_resource_oom(value))

    def test_warm_start_requires_completed_cuda_epoch_not_a_process_or_status(self):
        with TemporaryDirectory() as folder:
            p = Path(folder)
            self.assertFalse(recovery.warm_started(p))
            (p/'status.json').write_text(json.dumps(dict(status='CUDA_FRESH_MODULE_TRAINING')))
            self.assertFalse(recovery.warm_started(p))
            row = dict(phase='generator_training', record=dict(stage='base', epoch=0,
                device='cuda:0', cumulative_generator_steps=1, cumulative_surrogate_steps=1))
            (p/'progress.jsonl').write_text(json.dumps(row)+'\n')
            self.assertTrue(recovery.warm_started(p))

    def test_cpu_or_unrelated_progress_does_not_admit_a_second_worker(self):
        for row in (
            dict(phase='loading', record=dict(epoch=0, device='cuda:0')),
            dict(phase='generator_training', record=dict(stage='base',epoch=0,
                device='cpu',cumulative_generator_steps=1,cumulative_surrogate_steps=1)),
        ):
            with self.subTest(row=row), TemporaryDirectory() as folder:
                p=Path(folder);(p/'progress.jsonl').write_text(json.dumps(row)+'\n')
                self.assertFalse(recovery.warm_started(p))


class RecoveryMetadataTests(unittest.TestCase):
    def fixture(self, root):
        parent=root/'results'/'parent';parent.mkdir(parents=True)
        child=root/'results'/'recovery';child.mkdir()
        parent_plan=parent/'plan.json';parent_plan.write_bytes(b'{"frozen_training_contract":true}\n')
        (parent/'registration.json').write_text('{}')
        (parent/'source_snapshot.tar').write_bytes(b'parent source snapshot bytes')
        (parent/'failure.json').write_text('{"error":"preserved OOM evidence"}')
        (child/'plan.json').write_bytes(parent_plan.read_bytes())
        (child/'registration.json').write_bytes((parent/'registration.json').read_bytes())
        (child/'source_snapshot.tar').symlink_to(parent/'source_snapshot.tar')
        source=root/'recovery_source.py';source.write_text('registered controller bytes\n')
        cache=root/'cache.bin';cache.write_bytes(b'not pickle; never deserialize')
        complete=[];pending=recovery.original.matrix()[14:]
        for item in recovery.original.matrix()[:14]:
            folder=recovery.original.case_dir(parent,**item);folder.mkdir(parents=True)
            link=recovery.original.case_dir(child,**item);link.parent.mkdir(exist_ok=True)
            link.symlink_to(folder,target_is_directory=True)
            complete.append(dict(**item,source=str(folder.resolve()),result_sha256='recorded_result'))
        plan=dict(version=recovery.VERSION,parent_plan=str(parent_plan),
            parent_plan_sha256=recovery.sha(parent_plan),matrix=recovery.original.matrix(),
            retained_completed=complete,pending=pending,
            resource_policy=dict(max_concurrent=2,first_min_free_bytes=recovery.FIRST_FREE,
                second_min_free_bytes=recovery.SECOND_FREE,stable_probes=2,
                cold_start_gap_seconds=15,wait_first_epoch=True,no_cpu_fallback=True,
                kill_external_jobs=False),
            bound_files={str(cache):recovery.sha(cache),
                         str(parent/'failure.json'):recovery.sha(parent/'failure.json')},
            recovery_source=str(source),recovery_source_sha256=recovery.sha(source))
        (child/'recovery_plan.json').write_text(json.dumps(plan))
        (child/'recovery_controller_snapshot.py').write_bytes(source.read_bytes())
        self.resign(child)
        return child,parent,source,cache

    def resign(self, child):
        (child/'recovery_registration.json').write_text(json.dumps(dict(
            recovery_plan_sha256=recovery.sha(child/'recovery_plan.json'),
            controller_snapshot_sha256=recovery.sha(child/'recovery_controller_snapshot.py'))))

    def test_verified_recovery_keeps_same_training_bytes_and_exact14_22_layout(self):
        with TemporaryDirectory() as folder, patch.object(recovery.original,'verify',return_value={}) as check:
            child,parent,_,_=self.fixture(Path(folder))
            with patch('torch.load',side_effect=AssertionError('no cache/checkpoint loading')):
                plan=recovery.verify_recovery(child)
            self.assertEqual(len(plan['retained_completed']),14)
            self.assertEqual(len(plan['pending']),22)
            self.assertEqual(check.call_count,2)
            self.assertFalse((child/'plan.json').is_symlink())
            self.assertEqual((child/'plan.json').read_bytes(),(parent/'plan.json').read_bytes())
            for item in plan['retained_completed']:
                self.assertTrue(recovery.original.case_dir(child,item['seed'],item['variant']).is_symlink())

    def test_recovery_plan_snapshot_source_cache_and_parent_evidence_are_bound(self):
        for change in ('plan','snapshot','source','cache','parent_evidence','training_bytes'):
            with self.subTest(change=change),TemporaryDirectory() as folder, \
                    patch.object(recovery.original,'verify',return_value={}):
                child,parent,source,cache=self.fixture(Path(folder))
                files=dict(plan=child/'recovery_plan.json',snapshot=child/'recovery_controller_snapshot.py',
                    source=source,cache=cache,parent_evidence=parent/'failure.json',
                    training_bytes=child/'plan.json')
                path=files[change];path.write_bytes(path.read_bytes()+b'changed')
                with self.assertRaises(ValueError):recovery.verify_recovery(child)

    def test_completed_pending_partition_and_resource_policy_cannot_change(self):
        for change in ('duplicate','intersection','missing','unknown','resource'):
            with self.subTest(change=change),TemporaryDirectory() as folder:
                child,_,_,_=self.fixture(Path(folder));plan=json.loads((child/'recovery_plan.json').read_text())
                if change=='duplicate':plan['pending'].append(deepcopy(plan['pending'][0]))
                elif change=='intersection':plan['pending'][0]=deepcopy(plan['retained_completed'][0])
                elif change=='missing':plan['pending'].pop()
                elif change=='unknown':plan['pending'][0]['variant']='invented_new_condition'
                else:plan['resource_policy']['max_concurrent']=3
                with self.assertRaises(ValueError):recovery.validate_layout(child,plan)

    def test_completed_links_cannot_point_to_other_case_or_real_directory(self):
        for change in ('wrong_case','real_directory'):
            with self.subTest(change=change),TemporaryDirectory() as folder:
                child,parent,_,_=self.fixture(Path(folder));plan=json.loads((child/'recovery_plan.json').read_text())
                first=plan['retained_completed'][0];link=recovery.original.case_dir(child,first['seed'],first['variant'])
                link.unlink()
                if change=='real_directory':link.mkdir()
                else:
                    other=plan['retained_completed'][1]
                    link.symlink_to(recovery.original.case_dir(parent,other['seed'],other['variant']),target_is_directory=True)
                with self.assertRaises(ValueError):recovery.validate_layout(child,plan)

    def test_pending_outputs_cannot_alias_old_partial_training(self):
        with TemporaryDirectory() as folder:
            child,parent,_,_=self.fixture(Path(folder));plan=json.loads((child/'recovery_plan.json').read_text())
            item=plan['pending'][0];old=recovery.original.case_dir(parent,**item);old.mkdir(parents=True)
            new=recovery.original.case_dir(child,**item);new.parent.mkdir(exist_ok=True)
            new.symlink_to(old,target_is_directory=True)
            with self.assertRaises(ValueError):recovery.validate_layout(child,plan)

    def test_training_plan_symlink_is_rejected_even_if_bytes_match(self):
        with TemporaryDirectory() as folder:
            child,parent,_,_=self.fixture(Path(folder));plan=json.loads((child/'recovery_plan.json').read_text())
            (child/'plan.json').unlink();(child/'plan.json').symlink_to(parent/'plan.json')
            with self.assertRaises(ValueError):recovery.validate_layout(child,plan)

    def test_launch_is_detached_and_exclusive(self):
        with TemporaryDirectory() as folder,patch.object(recovery.original,'verify',return_value={}), \
                patch.object(recovery.subprocess,'Popen',return_value=Mock(pid=987)) as spawn:
            child,_,_,_=self.fixture(Path(folder))
            value=recovery.launch(child)
            self.assertEqual(value['pid'],987)
            self.assertIn('--queue',spawn.call_args.args[0])
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            self.assertEqual(value['retained_completed'],14)
            self.assertEqual(value['pending_cases'],22)
            with self.assertRaises(FileExistsError):recovery.launch(child)
            self.assertEqual(spawn.call_count,1)

    def test_corrupt_recovery_never_spawns(self):
        with TemporaryDirectory() as folder,patch.object(recovery.original,'verify',return_value={}), \
                patch.object(recovery.subprocess,'Popen') as spawn:
            child,_,_,_=self.fixture(Path(folder))
            p=child/'recovery_plan.json';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaises(ValueError):recovery.launch(child)
            spawn.assert_not_called()

    def test_prepare_reuses14_results_without_loading_or_resuming_partial_weights(self):
        with TemporaryDirectory() as folder,patch.object(recovery,'ROOT',Path(folder)):
            _,parent,_,_=self.fixture(Path(folder));parent_plan=parent/'plan.json'
            expected=recovery.sha(parent_plan)
            policies={str(seed):{} for seed in recovery.original.SEEDS}
            for item in recovery.original.matrix():
                policies[str(item['seed'])][item['variant']]=dict(fixed_variant=item['variant'],seed=item['seed'])
            for item in recovery.original.matrix()[:14]:
                case=recovery.original.case_dir(parent,**item)
                metrics={}
                for key in ('raw_targets','raw_clean_scores','raw_triggered_pair_scores'):
                    f=case/(key+'.csv');f.write_bytes(b'toy_metadata_only\n')
                    metrics[key]=dict(path=str(f),sha256=recovery.sha(f))
                result=dict(optimizer_steps=recovery.original.expected_steps(item['variant']),
                    selection=policies[str(item['seed'])][item['variant']],metrics=metrics)
                (case/'result.json').write_text(json.dumps(result))
                (case/'completion.json').write_text(json.dumps(dict(status='CASE_COMPLETE',
                    plan_sha256=expected,result_sha256=recovery.sha(case/'result.json'))))
            unfinished=recovery.original.matrix()[14]
            partial=recovery.original.case_dir(parent,**unfinished)
            (partial/'training').mkdir(parents=True)
            (partial/'training'/'epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=e))+'\n' for e in range(6)))
            (partial/'training'/'partial_model.pt').write_bytes(b'not a checkpoint; must never load')
            original_bytes=parent_plan.read_bytes();output=Path(folder)/'results'/'prepared_recovery'
            with patch.object(recovery.original,'verify',return_value=dict(policies=policies)), \
                    patch.object(recovery,'_verify_prediction_artifacts') as prediction_check, \
                    patch('torch.load',side_effect=AssertionError('must not resume any weights')):
                self.assertEqual(recovery.recover_prepare(parent_plan,output),output)
                plan=recovery.verify_recovery(output)
            self.assertEqual(prediction_check.call_count,14)
            self.assertEqual((output/'plan.json').read_bytes(),original_bytes)
            self.assertEqual(parent_plan.read_bytes(),original_bytes)
            self.assertFalse((output/'plan.json').is_symlink())
            self.assertEqual(len(plan['retained_completed']),14)
            self.assertEqual(len(plan['pending']),22)
            self.assertEqual(len(plan['interrupted_attempts']),1)
            self.assertFalse(plan['interrupted_attempts'][0]['old_partial_weights_reused'])
            self.assertTrue(plan['interrupted_attempts'][0]['restart_from_registered_seed'])
            self.assertFalse(recovery.original.case_dir(output,**unfinished).exists())


if __name__ == '__main__':
    unittest.main()
