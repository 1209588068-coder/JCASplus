"""CPU registration/device/launch checks; no real experiment is launched."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from spear_current import cpu_compact_module_ablation as cpu
from tests_current import test_compact_module_ablation as gpu_metadata_tests


class CpuCompactWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        path,source,cache,snapshot=gpu_metadata_tests.CompactModuleMetadataTests().fixture(root)
        plan=json.loads(path.read_text())
        plan['version']=cpu.VERSION
        plan['execution']=dict(device='cpu',dtype='float32',amp=False,threads_per_worker=2,
            interop_threads=1,max_concurrent=4,explicit_user_authorization=True,kill_external_jobs=False)
        path.write_text(json.dumps(plan))
        (path.parent/'registration.json').write_text(json.dumps(dict(
            plan_sha256=cpu.sha(path),source_snapshot_sha256=cpu.sha(snapshot))))
        return path,source

    def test_cpu_plan_has_all21_fresh_conditions_and_hides_cuda(self):
        with TemporaryDirectory() as tmp,patch.object(cpu,'ROOT',Path(tmp)):
            path,_=self.fixture(Path(tmp));plan=cpu.verify(path)
            self.assertEqual(len(plan['matrix']),21)
            self.assertEqual(cpu.retained_sources(plan),([],{}))
            self.assertEqual(plan['execution']['max_concurrent'],4)
            self.assertEqual(cpu._env()['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(cpu._env()['OMP_NUM_THREADS'],'2')

    def test_no_gpu_records_or_device_changes_are_accepted(self):
        with TemporaryDirectory() as tmp:
            path,_=self.fixture(Path(tmp));plan=json.loads(path.read_text())
            plan['retained_cases']=[dict(seed=20260621,variant='full')]
            with self.assertRaisesRegex(ValueError,'reuse GPU'):cpu.validate(plan)
            plan['retained_cases']=[];plan['execution']['device']='cuda:0'
            with self.assertRaises(ValueError):cpu.validate(plan)

    def test_cpu_config_never_initializes_cuda(self):
        import torch
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),\
                patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('unexpected CUDA')):
            self.assertEqual(str(cpu.configure_cpu(20260621)),'cpu')
            self.assertEqual(torch.get_num_threads(),2)
            self.assertEqual(torch.get_num_interop_threads(),1)

    def test_exclusive_detached_cpu_launch_and_source_hash_guard(self):
        with TemporaryDirectory() as tmp,patch.object(cpu,'ROOT',Path(tmp)),\
                patch.object(cpu.subprocess,'Popen',return_value=Mock(pid=123)) as spawn:
            path,source=self.fixture(Path(tmp));cpu.launch(path)
            self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertIn('spear_current.cpu_compact_module_ablation',spawn.call_args.args[0])
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            with self.assertRaises(FileExistsError):cpu.launch(path)
            source.write_text('mutated source')
            with self.assertRaises(ValueError):cpu.verify(path)

    def test_completed_cpu_epoch_admits_parallel_workers(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertFalse(cpu.warm_started(root))
            row=dict(phase='generator_training',record=dict(device='cpu',epoch=0,
                cumulative_surrogate_steps=1,cumulative_generator_steps=1))
            file=root/'progress.jsonl';file.write_text(json.dumps(row)+'\n')
            self.assertTrue(cpu.warm_started(root))
            row['record']['device']='cuda:0';file.write_text(json.dumps(row)+'\n')
            self.assertFalse(cpu.warm_started(root))

    def test_cpu_allocation_failure_is_distinguished_from_integrity_errors(self):
        self.assertTrue(cpu.is_cpu_allocation_failure(dict(root_error_type='MemoryError')))
        self.assertTrue(cpu.is_cpu_allocation_failure(dict(root_error_type='RuntimeError',
            error="DefaultCPUAllocator: can't allocate memory")))
        self.assertFalse(cpu.is_cpu_allocation_failure(dict(root_error_type='RuntimeError',
            error='Source hash changed')))


if __name__=='__main__':unittest.main()
