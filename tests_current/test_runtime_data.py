"""Standalone restoration and data-scope checks; no real graph reads/training."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import data, runtime
from spear_current.attention_context import attention_context
from spear_current.role_context import build_role_context


def graph_fixture():
    edges = np.array([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=np.int64)
    positions = np.array([[0, 0], [4, 0], [40, 0]], dtype=np.float32)
    result = dict(node_track_ids=np.array(['b', 'a', 'c']),
        x_node=np.arange(3 * 358, dtype=np.float32).reshape(3, 358),
        edge_index=edges, edge_attr=np.arange(6 * 34, dtype=np.float32).reshape(6, 34),
        observed_velocities_filled=np.ones((3, 6, 2), dtype=np.float32),
        observed_valid_mask=np.ones((3, 6), dtype=bool),
        future_positions=np.repeat(positions[:, None], 9, axis=1),
        future_valid_mask=np.ones((3, 9), dtype=bool))
    for key in ('supervision_edge_mask', 'label_strict_mask',
                'label_computable_mask', 'message_passing_edge_mask'):
        result[key] = np.ones(6, dtype=bool)
    return result


class DataContractTests(unittest.TestCase):
    def test_observed_features_and_topology_do_not_use_future_labels(self):
        graph = graph_fixture()
        before = data.line_graph_from_graph(graph)
        self.assertEqual(before['canonical_pair_track_ids'].tolist(),
                         [['a', 'b'], ['a', 'c'], ['b', 'c']])
        self.assertEqual(before['y'].tolist(), [1, 0, 0])
        graph['future_positions'][1, :, 0] = 20
        after = data.line_graph_from_graph(graph)
        self.assertEqual(after['y'].tolist(), [0, 0, 0])
        for key in ('x', 'edge_index', 'canonical_pairs', 'motion_regime'):
            np.testing.assert_array_equal(before[key], after[key])

    def test_source_arrays_are_not_modified_or_aliased(self):
        graph = graph_fixture()
        saved = deepcopy(graph)
        result = data.line_graph_from_graph(graph)
        result['x'][:] = -9
        for key in graph:
            np.testing.assert_array_equal(graph[key], saved[key])

    def test_frozen_normalizer_clips_without_modifying_inputs(self):
        original = np.array([[20., -30., 5.]], dtype=np.float32)
        normalizer = dict(mean=np.array([0., 0., 1.]), scale=np.array([1., 2., 2.]))
        result = data.transform_features(original, normalizer)
        np.testing.assert_array_equal(result, [[10., -10., 2.]])
        np.testing.assert_array_equal(original, [[20., -30., 5.]])

    def test_original_holdout_or_substituted_role_is_rejected_before_access(self):
        bundle = runtime.load_bundle(verify_assets=False)
        row = deepcopy(bundle.config['roles']['train'][0])
        with patch('spear_current.runtime.sha256', side_effect=AssertionError('file read')):
            for split in ('validation', 'test'):
                row['split'] = split
                with self.assertRaises(ValueError):
                    data.read_role_part(bundle, row, 'train')
            row['split'] = 'train'
            with self.assertRaises(ValueError):
                data.read_role_part(bundle, row, 'audit')
            row['scenario_id'] = 'substituted'
            with self.assertRaises(ValueError):
                data.read_role_part(bundle, row, 'train')


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bundle = runtime.load_bundle()

    def test_each_retained_generator_restores_exact_state_and_output_scope(self):
        pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
        edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
        context = attention_context(build_role_context(pairs, edges), ['synthetic-runtime'])
        features = torch.linspace(-2, 2, 3 * 784).reshape(3, 784)
        targets = torch.tensor([0, 2])
        for seed in (20260621, 20260622, 20260623):
            before_rng = torch.get_rng_state().clone()
            model = runtime.restore_generator(self.bundle, seed)
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            key = self.bundle.manifest['seeds'][str(seed)]['generator']
            expected = torch.load(self.bundle.asset_path(key), weights_only=True)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
            result = model.inject(features, context, targets, detach=True)
            selected = model.selected_coordinates[model.hard_support]
            allowed = torch.zeros_like(features, dtype=torch.bool)
            allowed[targets[:, None], selected] = True
            self.assertTrue(torch.equal(features[~allowed], result.features[~allowed]))
            self.assertTrue(bool((result.features[allowed].abs() <= 10).all()))
            self.assertTrue(torch.equal(result.edge_index, edges))

    def test_fresh_victim_initialization_and_fit_rng_match_saved_contract(self):
        for seed in (20260621, 20260622, 20260623):
            model = runtime.new_victim(self.bundle, seed)
            digest = hashlib.sha256()
            for name, value in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            recorded = json.loads(self.bundle.asset_path('training_' + str(seed)).read_text())
            self.assertEqual(digest.hexdigest(), recorded['initialization_digest'])
            self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                             recorded['fit_rng_digest'])

    def test_changed_asset_is_rejected(self):
        changed = deepcopy(self.bundle.manifest)
        changed['assets']['normalizer']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'changed bundle asset'):
            runtime.Bundle(self.bundle.manifest_path, changed).asset_path('normalizer')


if __name__ == '__main__':
    unittest.main()
