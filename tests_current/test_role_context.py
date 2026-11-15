# Source: tests/test_spear_pair_role_context.py; synthetic tests migrated with current-package imports.
"""Synthetic endpoint-role sidecars; no real data, weights, or training."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import random
import unittest

import numpy as np
import torch

from spear_current.role_context import (
    RoleContext, build_role_context, scrambled_context,
)


def fixture():
    pairs = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [0, 3], [2, 3]], dtype=np.int64)
    edges = np.array([(i, j) for i in range(len(pairs)) for j in range(len(pairs))
                      if i != j and len(set(pairs[i]) & set(pairs[j])) == 1], dtype=np.int64).T
    return pairs, edges


class RoleContextTests(unittest.TestCase):
    def test_role_definition_and_reverse_mapping(self):
        pairs, edges = fixture()
        context = build_role_context(pairs, edges)
        self.assertEqual(context.num_pairs, 6)
        self.assertEqual(context.edge_index.device.type, 'cpu')
        np.testing.assert_array_equal(context.edge_index.numpy(), edges)
        np.testing.assert_array_equal(context.pair_scene.numpy(), np.zeros(6, dtype=np.int64))
        by_edge = {}
        for (source, target), role in zip(edges.T, context.edge_type.tolist()):
            source_position, target_position = np.argwhere(pairs[source, :, None] == pairs[target, None, :])[0]
            self.assertEqual(role, 2 * source_position + target_position)
            by_edge[int(source), int(target)] = role
        self.assertEqual(set(by_edge.values()), {0, 1, 2, 3})
        for (source, target), role in by_edge.items():
            self.assertEqual(by_edge[target, source], (0, 2, 1, 3)[role])

    def test_preserves_endpoint_order_instead_of_silently_canonicalizing(self):
        pairs = np.array([[5, 2], [8, 5]], dtype=np.int64)
        edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
        context = build_role_context(pairs, edges)
        self.assertEqual(context.edge_type.tolist(), [1, 2])

    def test_vehicle_indices_may_repeat_in_different_scenes(self):
        pairs, edges = fixture()
        stacked = np.concatenate((pairs, pairs))
        scenes = np.array([9] * 6 + [12] * 6, dtype=np.int64)
        all_edges = np.concatenate((edges, edges + 6), axis=1)
        context = build_role_context(stacked, all_edges, scenes)
        self.assertEqual(context.num_pairs, 12)
        self.assertEqual(context.edge_type[:edges.shape[1]].tolist(), context.edge_type[edges.shape[1]:].tolist())
        np.testing.assert_array_equal(context.pair_scene.numpy(), scenes)
        cross_scene = all_edges.copy(); cross_scene[1, 0] += 6
        with self.assertRaisesRegex(ValueError, 'scene'):
            build_role_context(stacked, cross_scene, scenes)

    def test_numpy_and_torch_inputs_match_and_are_not_aliased(self):
        pairs, edges = fixture(); scenes = np.zeros(6, dtype=np.int64)
        expected = build_role_context(pairs, edges, scenes)
        inputs = [torch.from_numpy(v.copy()) for v in (pairs, edges, scenes)]
        actual = build_role_context(*inputs)
        for key in ('edge_index', 'edge_type', 'pair_scene'):
            self.assertTrue(torch.equal(getattr(actual, key), getattr(expected, key)))
        edges[:] = 0; scenes[:] = 55
        inputs[0][:] = 99; inputs[1][:] = 0; inputs[2][:] = 66
        self.assertTrue(torch.equal(actual.edge_index, expected.edge_index))
        self.assertTrue(torch.equal(actual.pair_scene, expected.pair_scene))
        self.assertEqual(actual.validate(), None)

    def test_to_cpu_returns_independent_validated_storage(self):
        context = build_role_context(*fixture())
        copied = context.to('cpu')
        for key in ('edge_index', 'edge_type', 'pair_scene'):
            a, b = getattr(context, key), getattr(copied, key)
            self.assertTrue(torch.equal(a, b))
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        copied.edge_type[0] = 3 - copied.edge_type[0]
        self.assertFalse(torch.equal(copied.edge_type, context.edge_type))
        with self.assertRaises(FrozenInstanceError): context.num_pairs = 1

    def test_empty_pairs_and_edges_are_valid(self):
        context = build_role_context(np.empty((0, 2), dtype=np.int64), np.empty((2, 0), dtype=np.int64))
        self.assertEqual(context.num_pairs, 0)
        self.assertEqual(tuple(context.edge_index.shape), (2, 0))
        self.assertEqual(tuple(context.edge_type.shape), (0,))
        context.validate()
        scrambled, stats = scrambled_context(context, 0)
        self.assertEqual(stats['total_edges'], 0)
        self.assertEqual(stats['changed_edges'], 0)
        self.assertEqual(stats['targets_with_role_diversity'], 0)
        self.assertEqual(stats['status'], 'NO_ROLE_DIVERSITY')
        self.assertIsNone(stats['changed_fraction'])
        self.assertEqual(scrambled.num_pairs, 0)
        json.dumps(stats, allow_nan=False)

    def test_nonempty_pairs_without_edges_keep_isolated_nodes(self):
        pairs, _ = fixture()
        context = build_role_context(pairs, np.empty((2, 0), dtype=np.int64))
        self.assertEqual(context.num_pairs, 6)
        _, stats = scrambled_context(context, 3)
        self.assertEqual(stats['targets_without_incoming_edges'], 6)
        self.assertEqual(stats['targets_without_role_diversity'], 6)
        self.assertTrue(stats['no_shuffle_freedom'])

    def test_reject_bad_dtypes_and_shapes_without_coercion(self):
        pairs, edges = fixture()
        scenes = np.zeros(6, dtype=np.int64)
        inputs = (pairs, edges, scenes)
        for index in range(3):
            for bad in (inputs[index].tolist(), inputs[index].astype(float), inputs[index].astype(np.int32),
                        inputs[index].astype(bool), torch.from_numpy(inputs[index]).float(), inputs[index][None]):
                values = list(inputs); values[index] = bad
                with self.subTest(index=index, type=type(bad)), self.assertRaises(ValueError):
                    build_role_context(*values)

    def test_reject_negative_indices_bounds_and_misaligned_scene_count(self):
        pairs, edges = fixture()
        bad_pair = pairs.copy(); bad_pair[0, 0] = -1
        with self.assertRaises(ValueError): build_role_context(bad_pair, edges)
        for endpoint in (-1, 6):
            bad_edge = edges.copy(); bad_edge[0, 0] = endpoint
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                build_role_context(pairs, bad_edge)
        for scene in (np.zeros(5, dtype=np.int64), -np.ones(6, dtype=np.int64)):
            with self.assertRaises(ValueError): build_role_context(pairs, edges, scene)

    def test_reject_self_vehicle_pair_duplicate_unordered_pair_and_line_selfedge(self):
        empty_edges = np.empty((2, 0), dtype=np.int64)
        for bad_pairs in (np.array([[2, 2]], dtype=np.int64),
                          np.array([[2, 5], [5, 2]], dtype=np.int64),
                          np.array([[2, 5], [2, 5]], dtype=np.int64)):
            with self.assertRaises(ValueError): build_role_context(bad_pairs, empty_edges)
        pairs, edges = fixture(); edges[:, 0] = 0
        with self.assertRaisesRegex(ValueError, 'self'):
            build_role_context(pairs, edges)

    def test_reject_duplicate_directed_edges_and_nonsharing_pairs(self):
        pairs, edges = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_role_context(pairs, np.concatenate((edges, edges[:, :1]), axis=1))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            build_role_context(np.array([[0, 1], [2, 3]], dtype=np.int64), np.array([[0], [1]], dtype=np.int64))

    def test_validate_detects_mutated_tensor_range_duplicates_scenes_and_selfedges(self):
        for corruption in ('role', 'negative_scene', 'cross_scene', 'bounds', 'duplicate', 'selfedge'):
            context = build_role_context(*fixture())
            if corruption == 'role': context.edge_type[0] = 4
            elif corruption == 'negative_scene': context.pair_scene[0] = -1
            elif corruption == 'cross_scene': context.pair_scene[0] = 1
            elif corruption == 'bounds': context.edge_index[0, 0] = 6
            elif corruption == 'duplicate': context.edge_index[:, 0] = context.edge_index[:, 1]
            elif corruption == 'selfedge': context.edge_index[1, 0] = context.edge_index[0, 0]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError): context.validate()
            with self.subTest(corruption=corruption, method='scramble'), self.assertRaises(ValueError):
                scrambled_context(context, 0)
            with self.subTest(corruption=corruption, method='to'), self.assertRaises(ValueError): context.to('cpu')

    def test_direct_dataclass_constructor_validates_shapes_and_types(self):
        good = build_role_context(*fixture())
        values = (good.edge_index, good.edge_type, good.pair_scene, good.num_pairs)
        variants = ((0, good.edge_index.float()), (0, good.edge_index[0]),
                    (1, good.edge_type[:-1]), (1, good.edge_type.float()),
                    (2, good.pair_scene[:-1]), (3, True), (3, -1), (3, np.int64(6)))
        for index, replacement in variants:
            args = list(values); args[index] = replacement
            with self.subTest(index=index), self.assertRaises(ValueError): RoleContext(*args)


class ScrambledContextTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.edges = fixture()
        self.context = build_role_context(self.pairs, self.edges)

    def test_private_rng_deterministic_and_global_states_and_input_unchanged(self):
        numpy_state = deepcopy(np.random.get_state())
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        saved = tuple(getattr(self.context, key).clone() for key in ('edge_index', 'edge_type', 'pair_scene'))
        first, stats = scrambled_context(self.context, 17)
        second, other_stats = scrambled_context(self.context, 17)
        different, _ = scrambled_context(self.context, 18)
        self.assertTrue(torch.equal(first.edge_type, second.edge_type))
        self.assertEqual(stats, other_stats)
        self.assertFalse(torch.equal(first.edge_type, different.edge_type))
        now = np.random.get_state()
        self.assertEqual(numpy_state[0], now[0]); np.testing.assert_array_equal(numpy_state[1], now[1])
        self.assertEqual(numpy_state[2:], now[2:])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        for old, key in zip(saved, ('edge_index', 'edge_type', 'pair_scene')):
            self.assertTrue(torch.equal(old, getattr(self.context, key)))
        json.dumps(stats, allow_nan=False)

    def test_each_target_role_histogram_and_structure_preserved(self):
        actual, stats = scrambled_context(self.context, 17)
        self.assertTrue(torch.equal(actual.edge_index, self.context.edge_index))
        self.assertTrue(torch.equal(actual.pair_scene, self.context.pair_scene))
        for target in range(self.context.num_pairs):
            selected = self.context.edge_index[1] == target
            self.assertTrue(torch.equal(torch.bincount(actual.edge_type[selected], minlength=4),
                                        torch.bincount(self.context.edge_type[selected], minlength=4)))
        self.assertEqual(stats['changed_edges'], int((actual.edge_type != self.context.edge_type).sum()))
        self.assertEqual(stats['total_edges'], self.edges.shape[1])
        self.assertEqual(stats['changed_edges'] + stats['unchanged_edges'], stats['total_edges'])
        self.assertEqual(stats['global_role_counts_before'], stats['global_role_counts_after'])
        self.assertGreater(stats['targets_with_role_diversity'], 0)
        self.assertGreater(stats['changed_edges'], 0)
        self.assertFalse(stats['forced_changes'])

    def test_edge_list_reordering_preserves_per_edge_scramble(self):
        order = np.random.default_rng(48).permutation(self.edges.shape[1])
        reordered = build_role_context(self.pairs, self.edges[:, order])
        original, original_stats = scrambled_context(self.context, 17)
        shuffled, shuffled_stats = scrambled_context(reordered, 17)
        np.testing.assert_array_equal(shuffled.edge_index.numpy(), self.edges[:, order])
        np.testing.assert_array_equal(shuffled.edge_type.numpy(), original.edge_type.numpy()[order])
        self.assertEqual(original_stats, shuffled_stats)

    def test_scrambling_is_not_merely_global_role_renaming(self):
        scrambled, _ = scrambled_context(self.context, 17)
        assignments = {role: set(scrambled.edge_type[self.context.edge_type == role].tolist()) for role in range(4)}
        self.assertTrue(any(len(values) > 1 for values in assignments.values()))

    def test_one_role_targets_have_no_choice_and_do_not_force_change(self):
        pairs = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64)
        edges = np.array([[1, 2], [0, 0]], dtype=np.int64)
        original = build_role_context(pairs, edges)
        actual, stats = scrambled_context(original, 123)
        self.assertTrue(torch.equal(actual.edge_type, original.edge_type))
        self.assertEqual(stats['targets_with_incoming_edges'], 1)
        self.assertEqual(stats['targets_with_role_diversity'], 0)
        self.assertEqual(stats['changed_edges'], 0)
        self.assertEqual(stats['edges_at_diverse_targets'], 0)
        self.assertTrue(stats['no_shuffle_freedom'])
        self.assertEqual(stats['status'], 'NO_ROLE_DIVERSITY')

    def test_possible_choice_but_random_unchanged_is_reported_without_retries(self):
        pairs = np.array([[0, 1], [0, 2], [1, 3]], dtype=np.int64)
        edges = np.array([[1, 2], [0, 0]], dtype=np.int64)
        original = build_role_context(pairs, edges)
        # PCG64(0)'s first length-two permutation is the identity.
        actual, stats = scrambled_context(original, 0)
        self.assertTrue(torch.equal(actual.edge_type, original.edge_type))
        self.assertEqual(stats['targets_with_role_diversity'], 1)
        self.assertEqual(stats['changed_edges'], 0)
        self.assertFalse(stats['no_shuffle_freedom'])
        self.assertEqual(stats['status'], 'SHUFFLED_NO_ASSIGNMENT_CHANGE')

    def test_returned_context_has_independent_tensor_storage(self):
        actual, _ = scrambled_context(self.context, 4)
        for key in ('edge_index', 'edge_type', 'pair_scene'):
            self.assertNotEqual(getattr(actual, key).data_ptr(), getattr(self.context, key).data_ptr())
        actual.pair_scene[:] = 77
        self.assertEqual(self.context.pair_scene.tolist(), [0] * 6)

    def test_reject_bad_seeds_and_noncontext_input(self):
        for seed in (-1, True, np.bool_(True), 1., '1', None):
            with self.subTest(seed=seed), self.assertRaises(ValueError): scrambled_context(self.context, seed)
        with self.assertRaises(ValueError): scrambled_context({}, 0)


if __name__ == '__main__':
    unittest.main()
