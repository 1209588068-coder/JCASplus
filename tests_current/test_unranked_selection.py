"""Synthetic prediction-free stratified attachment selection contracts."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current.unranked_selection import select_pair_indices
from tests_current.test_full_ablation_data import selection_inputs


def fixture():
    inputs = selection_inputs()
    for i, row in enumerate(inputs['metadata']['train']):
        row.update(scenario_id=f'scene{i // 8}', src_track_id=f'a{i}', dst_track_id=f'b{i}',
                   true_label=int(inputs['combined']['train']['y'][i]),
                   eligible=bool(inputs['combined']['train']['eligible'][i]))
    return inputs


class UnrankedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixture()
        self.enterContext(patch.object(torch.Tensor, 'backward', side_effect=AssertionError('no gradients')))
        self.enterContext(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('no training')))

    def test_historical_list_reproduced_with_reconstructed_shared_background(self):
        result = select_pair_indices(self.inputs, 'historical')
        self.assertTrue(torch.equal(result['idx_attach'],self.inputs['indices']['idx_attach']))
        paired = select_pair_indices(self.inputs, 'stratified_random')
        self.assertTrue(torch.equal(result['fixed_outer'][26:],paired['fixed_outer'][26:]))
        self.assertEqual(result['record']['outer_background_sha256'], paired['record']['outer_background_sha256'])
        self.assertTrue(result['record']['reconstructed_common_outer_not_historical_replay'])
        self.assertTrue(result['record']['historical_GCN_ranked_provenance_retained'])
        self.assertEqual(result['record']['GCN_calls'],0)

    def test_stratified_draw_exactly_matches_registered_rng_operations(self):
        seed=20260621
        actual=select_pair_indices(self.inputs,'stratified_random',seed)
        rng=np.random.RandomState(seed)
        candidates=self.inputs['indices']['candidates']
        rows=self.inputs['metadata']['train']
        selected=[]
        for regime in ('moving','slow'):
            pool=candidates[torch.tensor([rows[i]['motion_regime']==regime for i in candidates.tolist()])]
            count=actual['record']['motion_quotas'][regime]
            selected.append(pool[rng.choice(len(pool),count,replace=False)])
        expected=torch.cat(selected)
        expected=expected[rng.permutation(len(expected))]
        self.assertTrue(torch.equal(actual['idx_attach'],expected))
        self.assertEqual(len(expected.unique()),26)
        repeat=select_pair_indices(self.inputs,'stratified_random',seed)
        self.assertEqual(actual['record'],repeat['record'])
        self.assertEqual(json.loads(json.dumps(actual['record'])),actual['record'])

    def test_background_is_rebuilt_without_extra_old_attachment_exposure(self):
        original=select_pair_indices(self.inputs,'historical')
        actual=select_pair_indices(self.inputs,'stratified_random')
        allowed=torch.ones(len(self.inputs['combined']['train']['y']),dtype=torch.bool)
        both=torch.cat([original['idx_attach'],actual['idx_attach']]).unique()
        allowed[actual['idx_train']],allowed[both]=False,False
        pool=torch.nonzero(allowed).flatten()
        draw=pool[np.random.RandomState(20260621).choice(len(pool),512,replace=False)]
        self.assertTrue(torch.equal(actual['fixed_outer'],torch.cat([actual['idx_attach'],draw])))
        self.assertEqual(len(actual['fixed_outer'].unique()),538)
        self.assertFalse(bool(torch.isin(actual['fixed_outer'],actual['idx_train']).any()))
        self.assertFalse(bool(torch.isin(actual['fixed_outer'][26:],both).any()))
        self.assertTrue(torch.equal(actual['fixed_outer'][26:],original['fixed_outer'][26:]))
        self.assertEqual(actual['record']['outer_background_excluded_attachment_union_count'],len(both))
        self.assertNotEqual(actual['record']['attachment_indices'],original['record']['attachment_indices'])

    def test_global_rng_and_input_state_unchanged(self):
        before=deepcopy(self.inputs)
        numpy_before=np.random.get_state()
        torch_before=torch.get_rng_state().clone()
        select_pair_indices(self.inputs,'stratified_random')
        numpy_after=np.random.get_state()
        self.assertEqual(numpy_before[0],numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1],numpy_after[1])
        self.assertEqual(numpy_before[2:],numpy_after[2:])
        self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
        self.assertEqual(before['metadata'],self.inputs['metadata'])
        self.assertTrue(torch.equal(before['outer'],self.inputs['outer']))
        for k,v in before['indices'].items():self.assertTrue(torch.equal(v,self.inputs['indices'][k]))

    def test_prediction_and_dev_sidecars_are_never_used(self):
        baseline=select_pair_indices(self.inputs,'stratified_random')
        class Unreadable:
            def __getattribute__(self,name):raise AssertionError('no prediction/DEV inspection')
        self.inputs['predictions']=Unreadable()
        self.inputs['metadata']['dev']=Unreadable()
        self.inputs['combined']['dev']=Unreadable()
        self.assertEqual(baseline['record'],select_pair_indices(self.inputs,'stratified_random')['record'])

    def test_bad_metadata_rejected(self):
        for key,value in [('role','dev'),('role_pair_index',-1),('motion_regime','unknown'),
                          ('scenario_id',''),('true_label',1),('eligible',False)]:
            with self.subTest(field=key):
                inputs=fixture();i=int(inputs['indices']['candidates'][0]);inputs['metadata']['train'][i][key]=value
                with self.assertRaises(ValueError):select_pair_indices(inputs,'stratified_random')
        inputs=fixture();a,b=inputs['indices']['candidates'][:2].tolist()
        for key in ('scenario_id','src_track_id','dst_track_id'):
            inputs['metadata']['train'][b][key]=inputs['metadata']['train'][a][key]
        with self.assertRaises(ValueError):select_pair_indices(inputs,'stratified_random')

    def test_bad_candidate_labels_and_supervision_rejected(self):
        for damage in ('positive','clean_overlap','outer'):
            inputs=fixture();i=int(inputs['indices']['candidates'][0])
            if damage=='positive':
                inputs['combined']['train']['y'][i]=1;inputs['indices']['train_labels'][i]=1
            elif damage=='clean_overlap':inputs['indices']['candidates'][0]=inputs['indices']['idx_train'][0]
            else:inputs['outer']=inputs['outer'].roll(1)
            with self.subTest(damage=damage),self.assertRaises(ValueError):
                select_pair_indices(inputs,'stratified_random')

    def test_invalid_arm_and_seed_rejected(self):
        for arm,seed in [('random',20260621),('historical',True),('stratified_random',-1),
                         ('stratified_random',2**32)]:
            with self.subTest(arm=arm,seed=seed),self.assertRaises(ValueError):
                select_pair_indices(self.inputs,arm,seed)


if __name__=='__main__':
    unittest.main()
