"""Synthetic input identity and prediction-independent ablation selections."""
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import full_ablation_data as data
from spear_current.runtime import sha256
from spear_current.support_schema import SELECTED_COORDINATES, support_groups


def selection_inputs():
    n = 800
    clean, attachments, candidates = torch.arange(30), torch.arange(40, 66), torch.arange(40, 180)
    labels = torch.full((n,), -1, dtype=torch.long)
    labels[clean] = torch.arange(len(clean)) % 2
    labels[candidates] = 0
    allowed = torch.ones(n, dtype=torch.bool)
    allowed[clean], allowed[attachments] = False, False
    unlabeled = torch.nonzero(allowed).flatten()
    background = unlabeled[np.random.RandomState(data.OUTER_SEED).choice(
        len(unlabeled), data.NONATTACHMENT_OUTER_COUNT, replace=False)]
    train = dict(y=labels, eligible=labels >= 0)
    return dict(combined=dict(train=train), indices=dict(idx_train=clean,
        idx_attach=attachments, candidates=candidates, idx_unlabeled=unlabeled,
        train_labels=labels.clone(), dev_indices=torch.tensor([n + 1, n + 3])),
        outer=torch.cat([attachments, background]), metadata=dict(train=[
            dict(role='train', role_pair_index=i, motion_regime='moving' if i % 3 else 'slow')
            for i in range(n)]))


def support_bundle(root):
    groups = support_groups('semantic')
    # The actual retained SAGE24 omits four singleton groups and keeps all six triplets.
    excluded = {352, 133, 503, 179}
    fixed = [c not in excluded and c not in (357, 715) for c in SELECTED_COORDINATES]
    manifest = dict(selected_coordinates=list(SELECTED_COORDINATES), hard_support=fixed,
                    assets={})
    paths = {}
    for name, content in [('support', json.dumps(dict(arm='grouped_sage24',
            selected_coordinates=list(SELECTED_COORDINATES), hard_support=fixed))),
            ('feature_importance', 'synthetic frozen ranking asset')]:
        path = root / name
        path.write_text(content)
        paths[name] = path
        manifest['assets'][name] = dict(path=name, sha256=sha256(path))
    return SimpleNamespace(manifest=manifest, asset_path=lambda name: paths[name])


class IndexSelectionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = selection_inputs()
        self.enterContext(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('no training')))
        self.enterContext(patch.object(torch.Tensor, 'backward', side_effect=AssertionError('no gradients')))

    def test_full_and_other_components_reuse_original_supervision_and_outer(self):
        for variant in data.VARIANTS:
            if variant == 'random_nodes':
                continue
            for seed in data.SEEDS:
                selected = data.select_indices(self.inputs, variant, seed)
                self.assertTrue(torch.equal(selected['idx_train'], self.inputs['indices']['idx_train']))
                self.assertTrue(torch.equal(selected['idx_attach'], self.inputs['indices']['idx_attach']))
                self.assertTrue(torch.equal(selected['fixed_outer'], self.inputs['outer']))
                self.assertEqual(sum(selected['record']['attachment_motion_counts'].values()), 26)

    def test_random_nodes_exact_draw_roundtrip_and_background_rebuilt_without_extra_old_attachments(self):
        before = deepcopy(self.inputs)
        global_state = np.random.get_state()
        results = []
        for seed in data.SEEDS:
            selected = data.select_indices(self.inputs, 'random_nodes', seed)
            repeat = data.select_indices(self.inputs, 'random_nodes', seed)
            results.append(selected['idx_attach'].tolist())
            expected_attach = self.inputs['indices']['candidates'][np.random.RandomState(seed).choice(
                len(self.inputs['indices']['candidates']), 26, replace=False)]
            self.assertTrue(torch.equal(selected['idx_attach'], expected_attach))
            self.assertTrue(torch.equal(selected['idx_attach'], repeat['idx_attach']))
            self.assertTrue(torch.equal(selected['idx_train'], self.inputs['indices']['idx_train']))
            self.assertEqual(len(selected['idx_attach'].unique()), 26)
            self.assertTrue(bool((self.inputs['combined']['train']['y'][selected['idx_attach']] == 0).all()))
            self.assertFalse(bool(torch.isin(selected['fixed_outer'], selected['idx_train']).any()))
            self.assertFalse(bool(torch.isin(selected['idx_attach'], self.inputs['indices']['dev_indices']).any()))
            mask = torch.ones(len(self.inputs['combined']['train']['y']), dtype=torch.bool)
            mask[selected['idx_train']], mask[selected['idx_attach']] = False, False
            pool = torch.nonzero(mask).flatten()
            expected_background = pool[np.random.RandomState(data.OUTER_SEED).choice(len(pool), 512, replace=False)]
            self.assertTrue(torch.equal(selected['fixed_outer'], torch.cat([expected_attach, expected_background])))
            self.assertEqual(len(selected['fixed_outer'].unique()), 538)
            self.assertEqual(selected['record']['outer_background_pool_count'], len(self.inputs['indices']['idx_unlabeled']))
            self.assertEqual(json.loads(json.dumps(selected['record'])), selected['record'])
            stream = io.BytesIO()
            torch.save(selected, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=True)
            self.assertTrue(torch.equal(restored['idx_attach'], selected['idx_attach']))
            self.assertEqual(restored['record'], selected['record'])
        self.assertEqual(len({tuple(result) for result in results}), 3)
        self.assertTrue(torch.equal(before['outer'], self.inputs['outer']))
        for name in before['indices']:
            self.assertTrue(torch.equal(before['indices'][name], self.inputs['indices'][name]))
        current_state = np.random.get_state()
        self.assertEqual(global_state[0], current_state[0])
        np.testing.assert_array_equal(global_state[1], current_state[1])
        self.assertEqual(global_state[2:], current_state[2:])

    def test_rejects_nonnegative_ineligible_clean_dev_and_changed_background_contracts(self):
        for damage in ('positive', 'ineligible', 'clean_overlap', 'dev_overlap', 'unlabeled', 'outer'):
            value = deepcopy(self.inputs)
            candidate = int(value['indices']['candidates'][0])
            if damage == 'positive':
                value['combined']['train']['y'][candidate] = 1
                value['indices']['train_labels'][candidate] = 1
            elif damage == 'ineligible':
                value['combined']['train']['eligible'][candidate] = False
            elif damage == 'clean_overlap':
                value['indices']['candidates'][0] = value['indices']['idx_train'][0]
            elif damage == 'dev_overlap':
                value['indices']['dev_indices'][0] = 5
            elif damage == 'unlabeled':
                value['indices']['idx_unlabeled'] = value['indices']['idx_unlabeled'][1:]
            else:
                value['outer'] = value['outer'].roll(1)
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                data.select_indices(value, 'random_nodes', 20260621)

    def test_unregistered_case_is_rejected(self):
        for variant, seed in [('RandomNodes26', 20260621), ('full', True), ('full', 1)]:
            with self.assertRaises(ValueError):
                data.select_indices(self.inputs, variant, seed)


class SupportSelectionTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.bundle = support_bundle(root)
        self.enterContext(patch.object(torch, 'load', side_effect=AssertionError('no checkpoints')))

    def test_all_feasible_subsets_are_unique_whole_groups_at_exact_cost(self):
        groups = support_groups('semantic')
        feasible = data.feasible_support_subsets()
        self.assertEqual(len(feasible), 270)  # C(10,4) + 6*10 possible cost-four omissions.
        self.assertEqual(len(set(feasible)), 270)
        for subset in feasible:
            self.assertEqual(sum(len(groups[i]) for i in subset), 24)

    def test_full_frozen_support_and_random_controls_are_deterministic_legal_and_complete(self):
        fixed_coordinates, fixed_support, fixed_record = data.select_support(self.bundle, 'full', 20260621)
        self.assertEqual(fixed_coordinates.dtype, torch.long)
        self.assertEqual(fixed_support.dtype, torch.bool)
        self.assertEqual(fixed_coordinates.tolist(), list(SELECTED_COORDINATES))
        self.assertEqual(fixed_support.tolist(), self.bundle.manifest['hard_support'])
        self.assertEqual(sum(len(g) == 3 for g in fixed_record['selected_groups']), 6)
        for seed in data.SEEDS:
            coordinates, support, record = data.select_support(self.bundle, 'random_features', seed)
            repeated = data.select_support(self.bundle, 'random_features', seed)
            self.assertTrue(torch.equal(coordinates, fixed_coordinates))
            self.assertTrue(torch.equal(support, repeated[1]))
            active = set(coordinates[support].tolist())
            self.assertEqual(len(active), 24)
            self.assertFalse(active.intersection((357, 715)))
            for group in support_groups('semantic'):
                self.assertIn(len(active.intersection(group)), (0, len(group)))
            self.assertEqual(record['selected_cost'], 24)
            self.assertEqual(record['feasible_subset_count'], 270)
            self.assertFalse(record['selection_uses_asr'])
            self.assertFalse(record['frozen_clean_selector_provenance']['estimator_recomputed'])
            self.assertFalse(record['frozen_clean_selector_provenance']['attack_checkpoint_loaded'])
            self.assertEqual(json.loads(json.dumps(record)), record)

    def test_random_draw_may_equal_sage_without_redraw(self):
        _, frozen, record = data.select_support(self.bundle, 'full', 20260621)
        subset = tuple(record['selected_group_indices'])
        index = data.feasible_support_subsets().index(subset)
        random = SimpleNamespace(randint=lambda count: index)
        with patch.object(data.np.random, 'RandomState', return_value=random) as constructor:
            _, selected, record = data.select_support(self.bundle, 'random_features', 20260621)
        constructor.assert_called_once_with(20260621)
        self.assertTrue(torch.equal(selected, frozen))
        self.assertTrue(record['same_as_frozen_sage_support'])
        self.assertEqual(record['redraws'], 0)

    def test_rejects_split_triplet_or_focal_support(self):
        for kind in ('triplet', 'focal'):
            bundle = deepcopy(self.bundle)
            frozen = json.loads(bundle.asset_path('support').read_text())
            fixed = frozen['hard_support']
            fixed[SELECTED_COORDINATES.index(776)] = False
            fixed[SELECTED_COORDINATES.index(357 if kind == 'focal' else 352)] = True
            bundle.manifest['hard_support'] = fixed
            bundle.asset_path('support').write_text(json.dumps(frozen))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                data.select_support(bundle, 'full', 20260621)


class InputPreparationTests(unittest.TestCase):
    def test_uses_diagnostic_reader_exact_train_dev_ledger_and_returns_original_views(self):
        sources = {role: [dict(scenario_id=f'{role}-{i}') for i in range(n)]
                   for role, n in [('train', 512), ('dev', 128)]}
        expected = dict(combined=dict(train=dict(y='train'), dev=dict(y='dev')),
                        parts={}, indices={}, outer=None, context=None,
                        metadata={}, targets={}, populations={})
        def reader(bundle, actual_sources, ledger):
            self.assertIs(actual_sources, sources)
            ledger.extend(dict(scenario_id=row['scenario_id'], role=role, split='train', status='READ_COMPLETE')
                          for role in ('train', 'dev') for row in sources[role])
            return expected
        with patch.object(data, 'registered_rows', return_value=sources), \
                patch.object(data, 'read_inputs', side_effect=reader), \
                patch.object(torch, 'load', side_effect=AssertionError('no attack checkpoint')):
            result = data.prepare_inputs(object())
        self.assertEqual(len(result['ledger']), 640)
        self.assertIs(result['train'], result['combined']['train'])
        self.assertIs(result['dev'], result['combined']['dev'])
        self.assertEqual(set(result['ledger'][i]['role'] for i in range(640)), {'train', 'dev'})

    def test_unexpected_role_is_rejected_after_reader(self):
        sources = dict(train=[], dev=[])
        def reader(bundle, actual_sources, ledger):
            ledger.append(dict(role='audit', split='train', status='READ_COMPLETE'))
            return {}
        with patch.object(data, 'registered_rows', return_value=sources), \
                patch.object(data, 'read_inputs', side_effect=reader), \
                self.assertRaises(ValueError):
            data.prepare_inputs(object())


if __name__ == '__main__':
    unittest.main()
