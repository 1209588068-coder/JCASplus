"""Prediction-free current-module controls; no graphs, checkpoints or training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import current_ablation_data as data
from spear_current.support_schema import LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups
from tests_current.test_unranked_selection import fixture as selection_fixture


def masks_fixture():
    groups = support_groups('semantic')
    triplets = [g for g in groups if len(g) == 3]
    singles = [g for g in groups if len(g) == 1]
    result = {}
    for offset, name in enumerate(data.SOURCE_ARMS[:-1]):
        active = {c for g in triplets for c in g}
        active.update(c for i in range(6) for c in singles[(i + offset) % len(singles)])
        result[name] = [c in active for c in SELECTED_COORDINATES]
    result['ranking24'] = [c in LEGAL_COORDINATES[:24] for c in SELECTED_COORDINATES]
    return result


def fixture():
    inputs = selection_fixture()
    original = inputs['indices']['idx_attach'].tolist()
    for offset, index in enumerate(original):
        inputs['metadata']['train'][index]['motion_regime'] = 'moving' if offset < 11 else 'slow'
    return inputs


class CurrentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.masks = fixture(), masks_fixture()
        self.enterContext(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('no training')))
        self.enterContext(patch.object(torch.Tensor, 'backward', side_effect=AssertionError('no gradients')))

    def test_all_variants_share_exact_background_and_clean_supervision_within_seed(self):
        for seed in data.SEEDS:
            policies = {variant: data.build_policy(self.inputs, variant, seed, self.masks)
                        for variant in data.VARIANTS}
            reference = policies['full']
            union = torch.cat([reference['idx_attach'], policies['matched_random_pairs']['idx_attach']]).unique()
            for name, actual in policies.items():
                with self.subTest(seed=seed, variant=name):
                    self.assertTrue(torch.equal(actual['idx_train'], reference['idx_train']))
                    self.assertTrue(torch.equal(actual['fixed_outer'][26:], reference['fixed_outer'][26:]))
                    self.assertEqual(actual['record']['outer_background_sha256'], reference['record']['outer_background_sha256'])
                    self.assertFalse(bool(torch.isin(actual['fixed_outer'][26:], union).any()))
                    self.assertEqual(len(actual['fixed_outer'].unique()), 538)
                    self.assertEqual(actual['record']['attachment_motion_counts'], dict(moving=11, slow=15))
                    self.assertFalse(actual['record']['selection_uses_model_predictions'])
                    self.assertFalse(actual['record']['selection_uses_dev'])
                    self.assertFalse(actual['record']['feature_selector_retrained'])
                    self.assertTrue(actual['record']['common_background_shared_across_all_variants_within_seed'])
                    expected = policies['matched_random_pairs']['idx_attach'] if name == 'matched_random_pairs' else self.inputs['indices']['idx_attach']
                    self.assertTrue(torch.equal(actual['idx_attach'], expected))

    def test_random_pairs_have_matched_motion_and_are_deterministic_without_redraw(self):
        draws = []
        for seed in data.SEEDS:
            a = data.build_policy(self.inputs, 'matched_random_pairs', seed, self.masks)
            b = data.build_policy(self.inputs, 'matched_random_pairs', seed, self.masks)
            self.assertEqual(a['record'], b['record'])
            self.assertEqual(a['record']['redraws'], 0)
            self.assertFalse(a['record']['forced_different_from_historical'])
            self.assertTrue(bool(torch.isin(a['idx_attach'], self.inputs['indices']['candidates']).all()))
            self.assertTrue(bool((self.inputs['combined']['train']['y'][a['idx_attach']] == 0).all()))
            draws.append(tuple(a['idx_attach'].tolist()))
        self.assertEqual(len(set(draws)), 3)

    def test_each_support_control_uses_the_bound_list_not_a_new_draw(self):
        for variant in data.VARIANTS:
            p = data.build_policy(self.inputs, variant, data.SEEDS[0], self.masks)
            if variant == 'no_support28':
                expected = [c in LEGAL_COORDINATES for c in SELECTED_COORDINATES]
                self.assertEqual(int(p['support'].sum()), 28)
                self.assertFalse(p['record']['support_budget_matched'])
                self.assertIn('not an equal-coordinate-budget', p['record']['no_support28_interpretation'])
            else:
                name = variant if variant in (*data.RANDOM_SUPPORTS, 'ranking24') else 'semantic'
                expected = self.masks[name]
                self.assertEqual(int(p['support'].sum()), 24)
                self.assertTrue(p['record']['support_budget_matched'])
            self.assertEqual(p['support'].tolist(), expected)
            self.assertFalse(p['support'][0])
            self.assertFalse(p['support'][1])
            self.assertEqual(json.loads(json.dumps(p['record'])), p['record'])

    def test_global_rng_inputs_and_source_masks_unchanged_and_returns_owned_tensors(self):
        before, before_masks = deepcopy(self.inputs), deepcopy(self.masks)
        numpy_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
        p = data.build_policy(self.inputs, 'full', data.SEEDS[0], self.masks)
        numpy_after = np.random.get_state()
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(before_masks, self.masks)
        self.assertEqual(before['metadata'], self.inputs['metadata'])
        for key, value in before['indices'].items():
            self.assertTrue(torch.equal(value, self.inputs['indices'][key]))
        self.assertTrue(torch.equal(before['outer'], self.inputs['outer']))
        p['idx_train'][0] = 799
        p['idx_attach'][0] = 799
        p['fixed_outer'][0] = 799
        p['support'].fill_(False)
        self.assertTrue(torch.equal(before['indices']['idx_train'], self.inputs['indices']['idx_train']))
        self.assertTrue(torch.equal(before['indices']['idx_attach'], self.inputs['indices']['idx_attach']))
        self.assertEqual(before_masks, self.masks)

    def test_prediction_or_dev_sidecar_is_not_read(self):
        expected = data.build_policy(self.inputs, 'full', data.SEEDS[0], self.masks)['record']
        class Unreadable:
            def __getattribute__(self, name):
                raise AssertionError('no prediction or DEV access')
        self.inputs['predictions'] = Unreadable()
        self.inputs['combined']['dev'] = Unreadable()
        self.inputs['metadata']['dev'] = Unreadable()
        self.assertEqual(expected, data.build_policy(self.inputs, 'full', data.SEEDS[0], self.masks)['record'])

    def test_illegal_mask_shape_focal_group_and_ranking_changes_rejected(self):
        def focal(m):
            active = next(i for i, v in enumerate(m['semantic']) if v)
            m['semantic'][0], m['semantic'][active] = True, False
        def split_group(m):
            group = next(g for g in support_groups('semantic') if len(g) == 3)
            pos = SELECTED_COORDINATES.index(group[0])
            replacement = next(i for i, c in enumerate(SELECTED_COORDINATES)
                if c in LEGAL_COORDINATES and not m['semantic'][i])
            m['semantic'][pos], m['semantic'][replacement] = False, True
        def ranking(m):
            a = next(i for i, x in enumerate(m['ranking24']) if x)
            b = next(i for i, c in enumerate(SELECTED_COORDINATES) if c in LEGAL_COORDINATES and not m['ranking24'][i])
            m['ranking24'][a], m['ranking24'][b] = False, True
        changes = [focal, split_group, ranking,
                   lambda m:m.pop('random_semantic_s20260623'),
                   lambda m:m.update(semantic=[True] * 29),
                   lambda m:m.update(semantic=[int(x) for x in m['semantic']])]
        for change in changes:
            broken = deepcopy(self.masks); change(broken)
            with self.subTest(change=change), self.assertRaises(ValueError):
                data.build_policy(self.inputs, 'full', data.SEEDS[0], broken)

    def test_invalid_case_seed_and_candidate_identity_rejected(self):
        for variant, seed in [('old_full', 20260621), ('full', True), ('full', 20260624)]:
            with self.subTest(variant=variant, seed=seed), self.assertRaises(ValueError):
                data.build_policy(self.inputs, variant, seed, self.masks)
        for field, value in [('role', 'dev'), ('eligible', False), ('true_label', 1)]:
            broken = fixture(); index = int(broken['indices']['candidates'][0])
            broken['metadata']['train'][index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                data.build_policy(broken, 'full', data.SEEDS[0], self.masks)


class SupportDiscoveryTests(unittest.TestCase):
    def test_discovery_reads_metadata_only_and_verifies_all_five_sources(self):
        with TemporaryDirectory() as tmp:
            root, masks = Path(tmp), masks_fixture()
            parent = root / data.SUPPORT_RELATIVE_ROOT
            records = {}
            for name in data.SOURCE_ARMS[:-1]:
                folder = parent / name; folder.mkdir(parents=True)
                path = folder / 'support.json'
                path.write_text(json.dumps(dict(hard_support=masks[name])))
                records[name] = dict(path=str(path), sha256=data.sha256(path),
                    hard_support=masks[name], selected_coordinates=list(SELECTED_COORDINATES))
            folder = parent / 'ranking24'; folder.mkdir()
            path = folder / 'support.json'
            path.write_text(json.dumps(dict(arm='ranking24', hard_support=masks['ranking24'],
                selected_coordinates=list(SELECTED_COORDINATES), hard_support_capacity=24,
                spec=dict(kind='fixed', control='ranking', budget=24, seed=20260621))))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(data, 'support_records', return_value=deepcopy(records)), \
                    patch.object(data, 'RANKING24_SHA256', digest), \
                    patch('torch.load', side_effect=AssertionError('no checkpoint/cache loading')):
                result = data.discover_supports(root)
                self.assertEqual(result['source_masks'], masks)
                self.assertEqual(len(result['bound_files']), 5)
                for filename, expected in result['bound_files'].items():
                    self.assertTrue(Path(filename).is_absolute())
                    self.assertEqual(data.sha256(filename), expected)
                # A changed non-ranking source must also stop discovery.
                Path(records['semantic']['path']).write_text('changed')
                with self.assertRaisesRegex(ValueError, 'support JSON changed'):
                    data.discover_supports(root)
            with patch.object(data, 'support_records', return_value=deepcopy(records)), \
                    patch.object(data, 'RANKING24_SHA256', '0' * 64):
                with self.assertRaisesRegex(ValueError, 'ranking24 support'):
                    data.discover_supports(root)


if __name__ == '__main__':
    unittest.main()
