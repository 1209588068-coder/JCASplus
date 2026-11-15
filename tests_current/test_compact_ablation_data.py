"""Six-ablation data contracts; never train, load graphs or model tensors."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import compact_ablation_data as compact
from spear_current import current_ablation_data as previous
from spear_current.full_ablation_data import _tensor_hash
from spear_current.support_schema import EXCLUDED_COORDINATES, LEGAL_COORDINATES, SELECTED_COORDINATES
from tests_current.test_current_ablation_data import fixture, masks_fixture


class CompactDataTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.masks = fixture(), masks_fixture()
        self.enterContext(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('no training')))
        self.enterContext(patch.object(torch.Tensor, 'backward', side_effect=AssertionError('no gradients')))

    def test_seven_variants_for_three_seeds_keep_the_common_background(self):
        self.assertEqual(len(compact.VARIANTS), 7)
        for seed in compact.SEEDS:
            policies = {v:compact.build_policy(self.inputs,v,seed,self.masks) for v in compact.VARIANTS}
            full, random = policies['full'], policies['matched_random_pairs']
            union = torch.cat((full['idx_attach'],random['idx_attach'])).unique()
            for name, p in policies.items():
                with self.subTest(seed=seed, variant=name):
                    self.assertTrue(torch.equal(p['fixed_outer'][26:], full['fixed_outer'][26:]))
                    self.assertTrue(torch.equal(p['idx_train'], full['idx_train']))
                    self.assertEqual(p['record']['attachment_motion_counts'], {'moving':11,'slow':15})
                    self.assertEqual(len(p['idx_attach'].unique()),26)
                    self.assertEqual(len(p['fixed_outer'].unique()),538)
                    self.assertFalse(bool(torch.isin(p['fixed_outer'][26:],union).any()))
                    self.assertEqual(int(p['support'].sum()),24)
                    self.assertFalse(p['record']['selection_uses_model_predictions'])
                    self.assertFalse(p['record']['selection_uses_dev'])
                    self.assertFalse(p['record']['feature_selector_retrained'])
                    expected = random['idx_attach'] if name=='matched_random_pairs' else full['idx_attach']
                    self.assertTrue(torch.equal(expected,p['idx_attach']))

    def test_five_retained_records_are_exactly_compatible_with_completed_cases(self):
        for seed in compact.SEEDS:
            for variant in compact.REUSABLE_VARIANTS:
                with self.subTest(seed=seed,variant=variant):
                    old=previous.build_policy(self.inputs,variant,seed,self.masks)
                    new=compact.build_policy(self.inputs,variant,seed,self.masks)
                    self.assertEqual(new['record'],old['record'])
                    for key in ('idx_train','idx_attach','fixed_outer','support','candidate_pool'):
                        self.assertTrue(torch.equal(new[key],old[key]))

    def test_rf_is_exact_single_uniform_coordinate_draw_not_semantic_sampling(self):
        positions=np.asarray([i for i,c in enumerate(SELECTED_COORDINATES) if c in LEGAL_COORDINATES])
        for seed in compact.SEEDS:
            p=compact.build_policy(self.inputs,'random_features',seed,self.masks)
            expected=torch.zeros(30,dtype=torch.bool)
            draw=np.random.RandomState(seed).choice(positions,24,replace=False)
            expected[torch.from_numpy(draw)]=True
            self.assertTrue(torch.equal(p['support'],expected))
            self.assertEqual(p['record']['support_sha256'],_tensor_hash(expected))
            self.assertEqual(p['record']['feature_draw_count'],1)
            self.assertEqual(p['record']['feature_redraws'],0)
            self.assertFalse(p['record']['feature_draw_replace'])
            self.assertFalse(p['record']['semantic_group_constraint'])
            self.assertFalse(p['record']['support_is_preexisting_fixed_list'])
            self.assertEqual(p['record'],compact.build_policy(self.inputs,'random_features',seed,self.masks)['record'])
            self.assertFalse(set(p['record']['active_coordinates']).intersection(EXCLUDED_COORDINATES))
            self.assertEqual(json.loads(json.dumps(p['record'])),p['record'])

    def test_ne_keeps_full_data_and_support_only_declares_conditioning_change(self):
        for seed in compact.SEEDS:
            a=compact.build_policy(self.inputs,'full',seed,self.masks)
            b=compact.build_policy(self.inputs,'no_generator_neighborhood',seed,self.masks)
            for key in ('idx_train','idx_attach','fixed_outer','support','candidate_pool'):
                self.assertTrue(torch.equal(a[key],b[key]))
            self.assertTrue(b['record']['pointwise_conditioning'])
            self.assertFalse(b['record']['residual_neighbor_input'])
            self.assertTrue(b['record']['classification_shadow_graph_unchanged'])
            for key,value in a['record'].items():
                if key!='variant':self.assertEqual(b['record'][key],value)

    def test_rng_inputs_and_masks_are_unchanged_with_owned_outputs(self):
        original=deepcopy(self.inputs);masks=deepcopy(self.masks)
        np_before=np.random.get_state();torch_before=torch.get_rng_state().clone()
        p=compact.build_policy(self.inputs,'random_features',compact.SEEDS[0],self.masks)
        np_after=np.random.get_state()
        self.assertEqual(np_before[0],np_after[0]);np.testing.assert_array_equal(np_before[1],np_after[1])
        self.assertEqual(np_before[2:],np_after[2:]);self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
        self.assertEqual(masks,self.masks);self.assertEqual(original['metadata'],self.inputs['metadata'])
        for key,value in original['indices'].items():self.assertTrue(torch.equal(value,self.inputs['indices'][key]))
        p['support'].fill_(False);p['idx_attach'][0]=799;p['fixed_outer'][0]=799
        self.assertEqual(masks,self.masks)
        self.assertTrue(torch.equal(original['indices']['idx_attach'],self.inputs['indices']['idx_attach']))

    def test_no_predictions_dev_graph_or_model_loader_are_read(self):
        class Forbidden:
            def __getattribute__(self,name):raise AssertionError('no dev/prediction access')
        self.inputs['predictions']=Forbidden();self.inputs['combined']['dev']=Forbidden()
        self.inputs['metadata']['dev']=Forbidden()
        with patch('torch.load',side_effect=AssertionError('no tensor loading')):
            for variant in compact.VARIANTS:
                compact.build_policy(self.inputs,variant,compact.SEEDS[0],self.masks)

    def test_invalid_variants_seed_candidate_and_mask_are_rejected(self):
        for variant,seed in (('single_kl',20260621),('full',True),('full',20260624)):
            with self.subTest(variant=variant,seed=seed),self.assertRaises(ValueError):
                compact.build_policy(self.inputs,variant,seed,self.masks)
        broken=deepcopy(self.masks);broken['semantic'][0]=True
        with self.assertRaises(ValueError):compact.build_policy(self.inputs,'random_features',20260621,broken)
        bad=fixture();i=int(bad['indices']['candidates'][0]);bad['metadata']['train'][i]['role']='dev'
        with self.assertRaises(ValueError):compact.build_policy(bad,'no_generator_neighborhood',20260621,self.masks)


if __name__=='__main__':unittest.main()
