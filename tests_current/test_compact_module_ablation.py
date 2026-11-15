"""Metadata-only acceptance of the registered six-ablation CUDA workflow."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

from spear_current import compact_module_ablation as workflow
from spear_current import current_module_ablation as parent_workflow
from spear_current import support_replication
from spear_current.support_schema import SELECTED_COORDINATES, support_groups


def semantic_mask():
    groups = support_groups('semantic')
    triples = [g for g in groups if len(g) == 3]
    singles = [g for g in groups if len(g) == 1]
    active = {c for group in triples[:5] + singles[:9] for c in group}
    return [c in active for c in SELECTED_COORDINATES]


class CompactModuleMetadataTests(unittest.TestCase):
    def fixture(self, root):
        folder = root/'results/compact_metadata_fixture'
        folder.mkdir(parents=True)
        source = root/'runtime.py';source.write_bytes(b'fixed runtime\n')
        cache = root/'cache.bin';cache.write_bytes(b'not a tensor cache')
        snapshot = folder/'source_snapshot.tar';snapshot.write_bytes(b'fixed source snapshot')
        policies = {}
        for seed in workflow.SEEDS:
            policies[str(seed)] = {}
            for variant in workflow.VARIANTS:
                policies[str(seed)][variant] = dict(variant=variant,
                    hard_support=semantic_mask(),support_capacity=24,
                    attachment_count=26,candidate_count=6549,
                    attachment_motion_counts=dict(moving=11,slow=15),
                    attachment_indices=list(range(100,126)) if variant=='matched_random_pairs' else list(range(26)),
                    outer_background_indices=list(range(300,812)),
                    outer_background_sha256='same_background',idx_train_sha256='same_clean',
                    candidate_pool_sha256='same_candidates',selection_uses_dev=False,
                    selection_uses_audit_or_test=False,selection_uses_model_predictions=False,
                    feature_selector_retrained=False)
        plan = dict(version=workflow.VERSION,seeds=list(workflow.SEEDS),
            variants=list(workflow.VARIANTS),matrix=workflow.matrix(),
            stage_epochs=dict(workflow.EPOCHS),victim_epochs=200,
            optimizer_steps={v:workflow.expected_steps(v) for v in workflow.VARIANTS},
            execution=dict(device='cuda:0',dtype='float32',amp=False,tf32=False,
                max_concurrent=2,no_cpu_fallback=True,kill_external_jobs=False),
            preflight=dict(seed=workflow.SEEDS[0],variants=list(workflow.VARIANTS),
                epochs_per_stage=2,victim_epochs=2,asr_evaluation=False,weights_reused=False),
            training=dict(output_mode='identity',distribution_weight=0.),
            source_masks=dict(semantic=semantic_mask()),policies=policies,
            sources={'runtime.py':workflow.sha(source)},bound_files={str(cache):workflow.sha(cache)},
            retained_cases=[],causal_scope='fixed historical lists; fresh paired reconstruction')
        path = folder/'plan.json';path.write_text(json.dumps(plan))
        (folder/'registration.json').write_text(json.dumps(dict(
            plan_sha256=workflow.sha(path),source_snapshot_sha256=workflow.sha(snapshot))))
        return path,source,cache,snapshot

    def resign(self, path, plan):
        path.write_text(json.dumps(plan))
        receipt=json.loads((path.parent/'registration.json').read_text())
        receipt['plan_sha256']=workflow.sha(path)
        (path.parent/'registration.json').write_text(json.dumps(receipt))

    def test_exact_six_ablations_plus_full_three_seed_matrix(self):
        self.assertEqual(workflow.VARIANTS,('full','matched_random_pairs','random_features',
            'no_generator_neighborhood','no_residual','self_only','no_kl'))
        rows=workflow.matrix()
        self.assertEqual(len(rows),21)
        self.assertEqual(len({(r['seed'],r['variant']) for r in rows}),21)
        for seed in workflow.SEEDS:
            self.assertEqual({r['variant'] for r in rows if r['seed']==seed},set(workflow.VARIANTS))

    def test_exact_deletion_budgets_without_replacement_updates(self):
        for variant in workflow.VARIANTS:
            self.assertEqual(workflow.expected_steps(variant),dict(
                generator=200 if variant=='no_residual' else 800,
                surrogate=1000,discriminator=0,victim=200,selector=0))
            self.assertEqual(workflow.expected_steps(variant,2,2)['generator'],
                2 if variant=='no_residual' else 8)
        with self.assertRaises(ValueError):workflow.expected_steps('base_continuation')

    def test_verify_is_metadata_only_and_accepts_fixed_contract(self):
        with TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)):
            path,*_=self.fixture(Path(folder))
            with patch('torch.load',side_effect=AssertionError('never load model/cache')):
                plan=workflow.verify(path)
            self.assertEqual(len(plan['matrix']),21)
            self.assertFalse(plan['preflight']['asr_evaluation'])
            self.assertTrue(plan['execution']['no_cpu_fallback'])
            self.assertTrue(all(sum(r['hard_support'])==24 for p in plan['policies'].values() for r in p.values()))

    def test_sources_cache_snapshot_plan_and_expected_digest_are_bound(self):
        for name in ('source','cache','snapshot','plan','expected'):
            with self.subTest(name=name),TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)):
                path,source,cache,snapshot=self.fixture(Path(folder))
                files=dict(source=source,cache=cache,snapshot=snapshot,plan=path)
                if name!='expected':files[name].write_bytes(files[name].read_bytes()+b'changed')
                with self.assertRaises(ValueError):workflow.verify(path,'0'*64 if name=='expected' else None)

    def test_resigned_scope_budget_or_device_edits_are_rejected(self):
        changes=[('variants',['full']),('matrix',[]),('victim_epochs',201),
            ('optimizer_steps',{}),('preflight',{}),('execution',dict(device='cpu')),
            ('training',dict(output_mode='pair_mean',distribution_weight=0.))]
        for key,value in changes:
            with self.subTest(key=key),TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)):
                path,*_=self.fixture(Path(folder));plan=json.loads(path.read_text());plan[key]=value
                self.resign(path,plan)
                with self.assertRaises(ValueError):workflow.verify(path)

    def test_within_seed_pairing_and_permission_guards(self):
        for key,value in [('candidate_count',6548),('selection_uses_dev',True),
                ('feature_selector_retrained',True),('idx_train_sha256','different'),
                ('outer_background_indices',list(range(400,912))),
                ('attachment_indices',list(range(100,126)))]:
            with self.subTest(key=key),TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)):
                path,*_=self.fixture(Path(folder));plan=json.loads(path.read_text())
                plan['policies'][str(workflow.SEEDS[0])]['self_only'][key]=value
                self.resign(path,plan)
                with self.assertRaises(ValueError):workflow.verify(path)

    def test_retained_case_list_rejects_duplicates_or_new_conditions(self):
        for cases in ([dict(seed=workflow.SEEDS[0],variant='full')]*2,
                [dict(seed=workflow.SEEDS[0],variant='random_features')],
                [dict(seed=0,variant='full')]):
            with self.subTest(cases=cases),TemporaryDirectory() as folder:
                path,*_=self.fixture(Path(folder));plan=json.loads(path.read_text())
                plan['retained_cases']=cases
                with self.assertRaisesRegex(ValueError,'retained case'):workflow.validate(plan)

    def test_launch_is_exclusive_detached_and_never_launches_changed_sources(self):
        with TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)),\
                patch.object(workflow.subprocess,'Popen',return_value=Mock(pid=123)) as spawn:
            path,*_=self.fixture(Path(folder));record=workflow.launch(path)
            self.assertEqual(record['pid'],123)
            self.assertEqual(spawn.call_args.args[0][-1],workflow.sha(path))
            self.assertIn('spear_current.compact_module_ablation',spawn.call_args.args[0])
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            with self.assertRaises(FileExistsError):workflow.launch(path)
            self.assertEqual(spawn.call_count,1)
        with TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)),\
                patch.object(workflow.subprocess,'Popen') as spawn:
            path,source,*_=self.fixture(Path(folder));source.write_bytes(b'changed runtime')
            with self.assertRaises(ValueError):workflow.launch(path)
            spawn.assert_not_called()

    def paired_result(self, seed, variant, policy):
        metrics=dict(model_state_sha256=dict(clean_reference='same_reference'),
            models=dict(victim=dict(targets=dict(all=dict(absolute_asr=.8,
                conditional_flip=.8,clean_activation_rate=.01)),clean_pair_ap=.9)),
            value=.8)
        return dict(status='CASE_COMPLETE',seed=seed,variant=variant,
            metrics=metrics,acceptability=dict(passed=True),selection=policy,
            optimizer_steps=workflow.expected_steps(variant),support_capacity=24,
            elapsed_seconds=1.,output_mapping='identity',
            initialization=dict(fresh_model_sha256=dict(base_generator='g',base_shadow='s'),
                initialization_rng_sha256='a',post_initialization_rng_sha256='b',
                clean_proxy_sha256=['p1','p2']),
            victim_record=dict(initialization_digest='v',initial_state_sha256='vs',fit_rng_digest='vr'))

    def populate_cases(self,path,plan,retained=False):
        expected=workflow.sha(path)
        for item in workflow.matrix():
            directory=workflow.case_dir(path.parent,**item);directory.mkdir(parents=True)
            result=self.paired_result(**item,policy=plan['policies'][str(item['seed'])][item['variant']])
            workflow.write(directory/'result.json',result)
            parent_sha='parent_plan_sha' if retained and item==workflow.matrix()[0] else expected
            workflow.write(directory/'completion.json',dict(status='CASE_COMPLETE',**item,
                plan_sha256=parent_sha,result_sha256=workflow.sha(directory/'result.json')))
            if parent_sha!=expected:
                plan['retained_cases'].append(dict(**item,parent_plan_sha256=parent_sha,
                    result_sha256=workflow.sha(directory/'result.json')))
        return expected

    def test_aggregate_accepts_parent_receipt_without_rewriting_it(self):
        with TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)):
            path,*_=self.fixture(Path(folder));plan=json.loads(path.read_text())
            expected=self.populate_cases(path,plan,retained=True)
            receipt=workflow.case_dir(path.parent,**workflow.matrix()[0])/'completion.json'
            before=receipt.read_bytes()
            with patch.object(workflow,'verify',return_value=plan),\
                    patch.object(support_replication,'metric_values',side_effect=lambda m:dict(asr=m['value'])),\
                    patch.object(support_replication,'acceptability',return_value=dict(passed=True)),\
                    patch.object(support_replication,'_verify_prediction_artifacts',return_value=['same_targets']):
                result=workflow.aggregate(path,expected)
            self.assertEqual(result['planned_cases'],21)
            self.assertEqual(result['completed_cases'],21)
            self.assertEqual(result['retained_completed'],1)
            self.assertEqual(result['new_cases'],20)
            self.assertEqual(len(result['paired_differences']),18)
            self.assertEqual(receipt.read_bytes(),before)
            self.assertEqual(result['paired_verified_seeds'],list(workflow.SEEDS))

    def test_aggregate_rejects_new_plan_digest_in_retained_receipt(self):
        with TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)):
            path,*_=self.fixture(Path(folder));plan=json.loads(path.read_text())
            expected=self.populate_cases(path,plan,retained=True)
            receipt=workflow.case_dir(path.parent,**workflow.matrix()[0])/'completion.json'
            data=json.loads(receipt.read_text());data['plan_sha256']=expected;receipt.write_text(json.dumps(data))
            with patch.object(workflow,'verify',return_value=plan),\
                    patch.object(support_replication,'acceptability',return_value=dict(passed=True)):
                with self.assertRaisesRegex(ValueError,'completion contract'):workflow.aggregate(path,expected)

    def test_retained_selection_uses_exact_metadata_not_best_asr(self):
        with TemporaryDirectory() as folder,patch.object(workflow,'ROOT',Path(folder)):
            root=Path(folder);path,*_=self.fixture(root);plan=json.loads(path.read_text())
            for suffix,value in [('v1',.01),('recovery_v1',.99)]:
                parent=root/('results/current_module_ablation_20261103_'+suffix)
                directory=workflow.case_dir(parent,workflow.SEEDS[0],'full');directory.mkdir(parents=True)
                (parent/'plan.json').write_text(json.dumps(dict(parent=suffix)))
                result=self.paired_result(workflow.SEEDS[0],'full',plan['policies'][str(workflow.SEEDS[0])]['full'])
                result['metrics']['value']=value
                for name in ('raw_targets','raw_clean_scores','raw_triggered_pair_scores'):
                    file=directory/(name+'.csv');file.write_text('fixed predictions\n')
                    result['metrics'][name]=dict(path=str(file),sha256=workflow.sha(file))
                workflow.write(directory/'result.json',result)
                workflow.write(directory/'completion.json',dict(status='CASE_COMPLETE',seed=workflow.SEEDS[0],
                    variant='full',plan_sha256=workflow.sha(parent/'plan.json'),
                    result_sha256=workflow.sha(directory/'result.json')))
            with patch.object(parent_workflow,'verify',return_value={}),\
                    patch.object(support_replication,'_verify_prediction_artifacts'):
                retained,bindings=workflow.retained_sources(plan)
            self.assertEqual(len(retained),1)
            self.assertIn('20261103_v1',retained[0]['source'])
            self.assertNotIn('recovery',retained[0]['source'])
            self.assertEqual(len(bindings),6)

    def test_cold_start_admission_has_separate_thresholds_warmup_and_stable_probes(self):
        # Admission is reviewed without allocating a GPU or launching a child.
        source=inspect.getsource(workflow.queue)
        self.assertIn('16*1024**3 if active else 12*1024**3',source)
        self.assertIn('stable>=2',source)
        self.assertIn('warm_started',source)
        self.assertIn('time.monotonic()-last_launch>=15',source)
        self.assertIn('len(active)<2',source)
        self.assertIn('own_children_only_terminated=True',source)

    def test_initialization_rng_and_background_changes_fail_pairing(self):
        policy=dict(outer_background_sha256='same_background')
        base=self.paired_result(workflow.SEEDS[0],'full',policy)
        records=dict(full=base,self_only=deepcopy(base))
        workflow.paired_checks(records)
        for section,key in [('initialization','initialization_rng_sha256'),
                ('initialization','clean_proxy_sha256'),('victim_record','fit_rng_digest'),
                ('selection','outer_background_sha256')]:
            with self.subTest(section=section,key=key):
                altered=deepcopy(records);altered['self_only'][section][key]='changed'
                with self.assertRaises(ValueError):workflow.paired_checks(altered)


if __name__=='__main__':unittest.main()
