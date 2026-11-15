"""Metadata-only paired-trial integrity checks; never open AV2/checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from spear_current import unranked_pair_pilot as pilot
from spear_current.support_schema import SELECTED_COORDINATES, support_groups


class UnrankedPilotContractTests(unittest.TestCase):
    def fixture(self, root):
        folder = root/'results'/'metadata_fixture'
        folder.mkdir(parents=True)
        source = root/'runtime.py'
        source.write_text('frozen implementation bytes\n')
        cache = root/'cache.bin'
        cache.write_bytes(b'not a pickle; metadata checks must not deserialize')
        snapshot = folder/'source_snapshot.tar'
        snapshot.write_bytes(b'not an archive; metadata checks only hash bytes')
        groups = support_groups('semantic')
        triplets = [g for g in groups if len(g) == 3]
        singletons = [g for g in groups if len(g) == 1]
        active = {c for g in triplets[:5] + singletons[:9] for c in g}
        background = list(range(300, 812))
        selections = {}
        for arm, first in zip(pilot.ARMS, (100, 200)):
            selections[arm] = dict(arm=arm, attachment_count=26,
                attachment_motion_counts=dict(moving=11, slow=15), candidate_count=6549,
                attachment_indices=list(range(first,first+26)),
                selection_uses_model_predictions=False, selection_uses_dev=False,
                selection_uses_audit_or_test=False,
                common_background_shared_across_arms=True,
                outer_background_indices=background.copy(), candidate_pool_sha256='same_candidates')
        plan = dict(version=pilot.VERSION, seed=pilot.SEED, arms=list(pilot.ARMS),
            training_variant=pilot.TRAINING_VARIANT, stage_epochs=dict(pilot.STAGE_EPOCHS),
            victim_epochs=200, optimizer_steps=dict(pilot.OPTIMIZER_STEPS),
            source_mask=[c in active for c in SELECTED_COORDINATES], selections=selections,
            execution=dict(max_concurrent=2),
            preflight=dict(arms=list(pilot.ARMS),stage_epochs=2,victim_epochs=2,
                asr_evaluation=False,weights_reused=False),
            sources={'runtime.py':pilot.sha(source)},
            bound_files={str(cache):pilot.sha(cache)})
        path = folder/'plan.json'
        path.write_text(json.dumps(plan))
        (folder/'registration.json').write_text(json.dumps(dict(
            plan_sha256=pilot.sha(path), source_snapshot_sha256=pilot.sha(snapshot))))
        return path, source, cache, snapshot

    def resign(self, path, plan):
        path.write_text(json.dumps(plan))
        registration = path.parent/'registration.json'
        value = json.loads(registration.read_text())
        value['plan_sha256'] = pilot.sha(path)
        registration.write_text(json.dumps(value))

    def test_valid_metadata_registration_does_not_load_model_or_cache(self):
        with TemporaryDirectory() as folder, patch.object(pilot,'ROOT',Path(folder)):
            path,_,_,_ = self.fixture(Path(folder))
            with patch('torch.load',side_effect=AssertionError('no deserialization')):
                plan = pilot.verify(path)
            self.assertEqual(plan['optimizer_steps'],dict(generator=800,surrogate=1000,
                discriminator=0,victim=200,selector=0))
            self.assertEqual(plan['training_variant'],'no_distribution_no_domain')

    def test_changed_source_cache_or_snapshot_rejected(self):
        for item in ('source','cache','snapshot'):
            with self.subTest(item=item), TemporaryDirectory() as folder, \
                    patch.object(pilot,'ROOT',Path(folder)):
                path,source,cache,snapshot = self.fixture(Path(folder))
                dict(source=source,cache=cache,snapshot=snapshot)[item].write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    pilot.verify(path)

    def test_plan_digest_and_expected_digest_guard(self):
        with TemporaryDirectory() as folder, patch.object(pilot,'ROOT',Path(folder)):
            path,_,_,_ = self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError,'Plan/source snapshot'):
                pilot.verify(path,'0'*64)
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Plan/source snapshot'):
                pilot.verify(path)

    def test_training_budget_arm_or_seed_changes_rejected_even_if_resigned(self):
        changes = [('seed',123),('arms',['historical']),('victim_epochs',201),
                   ('stage_epochs',{stage:1 for stage in pilot.STAGES}),
                   ('training_variant','full'),('optimizer_steps',{}),
                   ('execution',dict(max_concurrent=3)),('preflight',{})]
        for key,value in changes:
            with self.subTest(key=key), TemporaryDirectory() as folder, \
                    patch.object(pilot,'ROOT',Path(folder)):
                path,_,_,_ = self.fixture(Path(folder))
                plan = json.loads(path.read_text());plan[key]=value
                self.resign(path,plan)
                with self.assertRaisesRegex(ValueError,'Paired training configuration'):
                    pilot.verify(path)

    def test_changed_motion_counts_or_prediction_use_rejected(self):
        changes = [('attachment_motion_counts',dict(moving=12,slow=14)),
                   ('candidate_count',6548),('selection_uses_model_predictions',True),
                   ('selection_uses_dev',True),('selection_uses_audit_or_test',True),
                   ('common_background_shared_across_arms',False)]
        for key,value in changes:
            with self.subTest(key=key), TemporaryDirectory() as folder, \
                    patch.object(pilot,'ROOT',Path(folder)):
                path,_,_,_ = self.fixture(Path(folder))
                plan = json.loads(path.read_text())
                plan['selections']['stratified_random'][key]=value
                self.resign(path,plan)
                with self.assertRaisesRegex(ValueError,'selection contract'):
                    pilot.verify(path)

    def test_backgrounds_and_candidate_pools_must_be_shared(self):
        for change in ('background','candidate_hash','attachment_overlap','duplicate_attachment'):
            with self.subTest(change=change), TemporaryDirectory() as folder, \
                    patch.object(pilot,'ROOT',Path(folder)):
                path,_,_,_ = self.fixture(Path(folder))
                plan = json.loads(path.read_text())
                records=plan['selections']
                if change=='background':
                    records['stratified_random']['outer_background_indices'][0]=900
                elif change=='candidate_hash':
                    records['stratified_random']['candidate_pool_sha256']='different'
                elif change=='attachment_overlap':
                    for row in records.values():row['outer_background_indices'][0]=100
                else:
                    records['stratified_random']['attachment_indices'][0]=201
                self.resign(path,plan)
                with self.assertRaises(ValueError):pilot.verify(path)

    def test_semantic_support_metadata_protection_remains_enforced(self):
        with TemporaryDirectory() as folder, patch.object(pilot,'ROOT',Path(folder)):
            path,_,_,_ = self.fixture(Path(folder))
            plan = json.loads(path.read_text())
            plan['source_mask'][SELECTED_COORDINATES.index(357)]=True
            enabled=next(i for i,c in enumerate(SELECTED_COORDINATES)
                         if c!=357 and plan['source_mask'][i])
            plan['source_mask'][enabled]=False
            self.resign(path,plan)
            with self.assertRaises(ValueError):pilot.verify(path)

    def test_launch_is_detached_exclusive_and_binds_plan(self):
        with TemporaryDirectory() as folder, patch.object(pilot,'ROOT',Path(folder)), \
                patch.object(pilot.subprocess,'Popen',return_value=Mock(pid=456)) as spawn:
            path,_,_,_ = self.fixture(Path(folder))
            result=pilot.launch(path)
            self.assertEqual(result['pid'],456)
            command=spawn.call_args.args[0]
            self.assertIn('--queue',command)
            self.assertEqual(command[-1],pilot.sha(path))
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            with self.assertRaises(FileExistsError):pilot.launch(path)
            self.assertEqual(spawn.call_count,1)

    def test_damaged_plan_never_spawns(self):
        with TemporaryDirectory() as folder, patch.object(pilot,'ROOT',Path(folder)), \
                patch.object(pilot.subprocess,'Popen') as spawn:
            path,_,_,_ = self.fixture(Path(folder))
            path.write_text(path.read_text()+'\n')
            with self.assertRaises(ValueError):pilot.launch(path)
            spawn.assert_not_called()


if __name__=='__main__':
    unittest.main()
