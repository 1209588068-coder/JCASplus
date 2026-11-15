"""Matched contract, completion identity and supervisor failure regression."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spear_current import womd_aligned as runner
from spear_current.womd_aligned_data import prepare
from tests_current.test_womd_aligned_data import fixture


def contract():
    reference = dict(training=dict(base_epochs=200,residual_epochs=200,support_epochs=200,
        clean_proxy_epochs=200,clean_reference_epochs=200,victim_epochs=200,inner_steps=1),
        seed_offsets=dict(selection=0,clean_proxy_1=101000,clean_proxy_2=102000,
            clean_reference=103000,anchor=201000,support=301000,victim=401000),
        training_execution=dict(optimizer_steps_per_epoch=1,neighbor_sampling=False),
        checkpoint_selection=dict(generator='final',ranking_and_mlp='dev_accuracy',graph_sage='dev_auc'),
        support=dict(protected=[357,715],final_budget=24),output_mode='identity',
        domain_projection=False,distribution_discriminator=False,pair_mean=False,
        device='cuda:0',max_nodes=16384,dtype='float32',amp=False,tf32=False,
        source_selection=dict(attachment_count=5865,outer_extra_count=512,model_epochs=200,
            sage_n_permutations=2048,sage_batch_size=64,explanation_count=128,background_count=32))
    data = dict(seed=20261104,dataset_id='WOMD',source_dataset_split='training_20s',status='PREPARED',
        test_arrays_loaded=False,recording_independence_proven=False,new_statistics_fitted=0,
        clean_indices=dict(exact_original_indices_preserved=True),max_nodes_per_shard=16384,
        attachment_budget=dict(count=154,reference_attachment_count=5865,realized_fraction=154/11668),
        **{r:dict(source_payload_sha256=r+'-payload',repacked_payload_sha256=r+'-payload')
           for r in ('train','dev','audit')})
    plan = {k:deepcopy(reference[k]) for k in runner.COPIED_CONTRACT}
    plan.update(seed=data['seed'],source_selection=dict(reference['source_selection'],attachment_count=154),
                poison_budget=deepcopy(data['attachment_budget']),historical_attack_checkpoint_loaded=False,
                historical_support_loaded=False,fresh_full_pipeline=True)
    return plan,reference,data


class MatchedContractTests(unittest.TestCase):
    def test_matched_contract_is_accepted_and_stage_rules_cannot_drift(self):
        plan,reference,data=contract()
        runner.validate_matched_contract(plan,reference,data)
        edits=[('training','base_epochs',201),('training','inner_steps',2),
               ('seed_offsets','support',301),('support','final_budget',25),
               ('source_selection','outer_extra_count',513),
               ('source_selection','sage_n_permutations',1024)]
        for group,key,value in edits:
            changed=deepcopy(plan);changed[group][key]=value
            with self.subTest(group=group,key=key),self.assertRaises(ValueError):
                runner.validate_matched_contract(changed,reference,data)
        for key,value in [('output_mode','pair_mean'),('device','cpu'),('tf32',True),('max_nodes',8192)]:
            changed=deepcopy(plan);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.validate_matched_contract(changed,reference,data)

    def test_poison_seed_and_historical_state_changes_are_rejected(self):
        plan,reference,data=contract()
        for key,value in [('seed',20261105),('historical_attack_checkpoint_loaded',True),
                          ('historical_support_loaded',True),('recovery',dict(ranking_fits={'20261104':'old'}))]:
            changed=deepcopy(plan);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.validate_matched_contract(changed,reference,data)
        for group,key,value in [('source_selection','attachment_count',26),
                               ('poison_budget','count',155)]:
            changed=deepcopy(plan);changed[group][key]=value
            with self.subTest(group=group,key=key),self.assertRaises(ValueError):
                runner.validate_matched_contract(changed,reference,data)

    def test_relabelled_pools_and_refitted_preprocessing_are_rejected(self):
        plan,reference,data=contract()
        for key,value in [('dataset_id','AV2'),('source_dataset_split','test'),('status','RUNNING'),
                          ('test_arrays_loaded',True),('recording_independence_proven',True),
                          ('new_statistics_fitted',1)]:
            changed=deepcopy(data);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.validate_matched_contract(plan,reference,changed)
        changed=deepcopy(data);changed['audit']['repacked_payload_sha256']='different'
        with self.assertRaises(ValueError):
            runner.validate_matched_contract(plan,reference,changed)


class CompletionAndSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        source,ref,self.parts,_=fixture(self.root)
        self.output=self.root/'output';self.output.mkdir()
        self.data=self.output/'data'
        summary=prepare(source,self.data,ref,max_nodes=35)
        self.plan=dict(output_dir=str(self.output),data_dir=str(self.data),seed=20261104,
            source_snapshot=str(self.root),data_summary=summary)
        self.plan_path=self.output/'plan.json'
        runner.write_json(self.plan_path,self.plan)
        self.folder=self.output/'seed20261104'

    def tearDown(self):
        self.temp.cleanup()

    def complete(self):
        self.folder.mkdir(exist_ok=True)
        (self.folder/'generator.pt').write_bytes(b'synthetic checkpoint, never executed')
        result_hashes,target_hashes,csv_hashes={},{},{}
        fields=['scenario_id','pair_index','target_pair_index','src_track_id','dst_track_id']
        for role in ('dev','audit'):
            folder=self.folder/f'evaluation_{role}';folder.mkdir(exist_ok=True)
            summary=dict(status='EVALUATION_COMPLETE',dataset='WOMD',source_dataset_split='training_20s',
                targets=self.plan['data_summary'][role]['targets'],scenes=self.plan['data_summary'][role]['scenes'])
            runner.write_json(folder/'summary.json',summary)
            with (folder/'targets.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for part in self.parts[role]:
                    target=part['target_index']
                    if target is not None:
                        ids=part['canonical_pair_track_ids'][target]
                        writer.writerow(dict(scenario_id=part['scenario_id'],pair_index=target,
                            target_pair_index=target,src_track_id=str(ids[0]),dst_track_id=str(ids[1])))
            result_hashes[role]=runner.sha(folder/'summary.json')
            target_hashes[role]=runner.sha(folder/'targets.csv')
            for name in ('clean_pair_scores.csv','triggered_pair_scores.csv'):
                (folder/name).write_text('synthetic,fixture\n1,2\n')
            csv_hashes[role]={name:runner.sha(folder/name) for name in
                ('targets.csv','clean_pair_scores.csv','triggered_pair_scores.csv')}
        done=dict(status='MATCHED_PILOT_COMPLETE_REVIEW_REQUIRED',seed=20261104,
            plan_sha256=runner.sha(self.plan_path),generator_file_sha256=runner.sha(self.folder/'generator.pt'),
            result_sha256=result_hashes,target_csv_sha256=target_hashes,evaluation_csv_sha256=csv_hashes)
        runner.write_json(self.folder/'completion.json',done)
        return done

    def test_complete_valid_receipt_and_fixed_target_identity_are_accepted(self):
        done=self.complete()
        self.assertEqual(runner.validate_completion(self.plan),done)

    def test_missing_artifact_wrong_counts_and_changed_hash_are_rejected(self):
        self.complete()
        (self.folder/'evaluation_dev/summary.json').unlink()
        with self.assertRaises((OSError,ValueError)):
            runner.validate_completion(self.plan)
        done=self.complete()
        path=self.folder/'evaluation_dev/summary.json'
        result=runner.read(path);result['targets']+=1;runner.write_json(path,result)
        done['result_sha256']['dev']=runner.sha(path);runner.write_json(self.folder/'completion.json',done)
        with self.assertRaises(ValueError):
            runner.validate_completion(self.plan)
        self.complete();(self.folder/'generator.pt').write_bytes(b'changed model')
        with self.assertRaises(ValueError):
            runner.validate_completion(self.plan)
        self.complete();runner.write_json(self.folder/'failure.json',dict(status='FAILED'))
        with self.assertRaises(ValueError):
            runner.validate_completion(self.plan)

    def test_same_count_but_different_target_identity_is_rejected(self):
        done=self.complete()
        path=self.folder/'evaluation_dev/targets.csv'
        text=path.read_text().replace('dev-0','different-scene')
        path.write_text(text)
        # Even a freshly rebound CSV hash cannot change the frozen target pool.
        done['target_csv_sha256']['dev']=runner.sha(path)
        done['evaluation_csv_sha256']['dev']['targets.csv']=runner.sha(path)
        runner.write_json(self.folder/'completion.json',done)
        with self.assertRaises(ValueError):
            runner.validate_completion(self.plan)

    def test_missing_target_csv_is_rejected(self):
        self.complete();(self.folder/'evaluation_audit/targets.csv').unlink()
        with self.assertRaises((OSError,ValueError)):
            runner.validate_completion(self.plan)

    def test_duplicate_launch_is_rejected_without_starting_a_process(self):
        runner.write_json(self.output/'launch.json',dict(pid=123))
        with patch.object(runner,'verify',return_value=self.plan),patch.object(runner.subprocess,'Popen') as launch:
            with self.assertRaises(FileExistsError):
                runner.launch(self.plan_path)
            launch.assert_not_called()

    def test_supervisor_records_nonzero_worker_exit_without_completion(self):
        child=SimpleNamespace(pid=123,poll=lambda:17)
        with patch.object(runner,'verify',return_value=self.plan),patch.object(runner.subprocess,'Popen',return_value=child):
            with self.assertRaisesRegex(RuntimeError,'17'):
                runner.supervise(self.plan_path)
        status=runner.read(self.output/'status.json')
        self.assertEqual(status['status'],'FAILED_REVIEW_REQUIRED')
        self.assertEqual(status['completed'],0)
        self.assertEqual(status['returncode'],17)
        self.assertTrue((self.output/'failure.json').exists())
        self.assertFalse((self.output/'completion.json').exists())

    def test_zero_exit_without_valid_receipt_is_also_failure(self):
        child=SimpleNamespace(pid=123,poll=lambda:0)
        with patch.object(runner,'verify',return_value=self.plan),patch.object(runner.subprocess,'Popen',return_value=child):
            with self.assertRaises(OSError):
                runner.supervise(self.plan_path)
        status=runner.read(self.output/'status.json')
        self.assertEqual(status['status'],'FAILED_REVIEW_REQUIRED')
        self.assertEqual(status['completed'],0)
        self.assertFalse((self.output/'completion.json').exists())


if __name__=='__main__':
    unittest.main()
