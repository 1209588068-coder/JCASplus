"""Synthetic scene-binding regression tests; no graphs, checkpoints or training."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.evaluate import SceneInjector, state_hash
from spear_current.role_generator import InjectedFeatures
from tests_current.fixtures_tensors import setup


def scene(pair_count):
    pairs = np.asarray([[0, 1], [1, 2]][:pair_count], dtype=np.int64).reshape(-1, 2)
    edges = (np.asarray([[0, 1], [1, 0]], dtype=np.int64)
             if pair_count == 2 else np.empty((2, 0), dtype=np.int64))
    return dict(scenario_id='synthetic-evaluation', split='train',
        metadata={'feature_dimension': 784}, canonical_pairs=pairs, edge_index=edges,
        x=np.full((pair_count, 784), .01, dtype=np.float32),
        y=np.zeros(pair_count, dtype=np.int64), eligible=np.ones(pair_count, dtype=bool))


class SceneInjectorTests(unittest.TestCase):
    def setUp(self):
        anchor, support, *_ = setup()
        self.generator = DomainConstrainedGenerator(anchor, support, output_mode='domain_projection')
        self.generator.eval().requires_grad_(False)
        self.injector = SceneInjector(self.generator, 20260621)

    def arguments(self, part, targets=None):
        if targets is None:
            targets = torch.tensor([0], dtype=torch.long)
        return (targets, torch.from_numpy(part['x'].copy()),
            torch.from_numpy(part['edge_index'].copy()),
            torch.ones(part['edge_index'].shape[1]), torch.device('cpu'))

    def test_empty_scene_and_empty_edge_bind_do_not_fail_hashing(self):
        for count in (0, 1):
            with self.subTest(pair_count=count):
                part = scene(count)
                self.injector.bind(part)
                self.assertEqual(self.injector.context.score_permutation.numel(), 0)
                targets = torch.tensor([0], dtype=torch.long) if count else torch.empty(0, dtype=torch.long)
                before = state_hash(self.generator)
                x, edges, weights = self.injector.inject_trigger(*self.arguments(part, targets))
                self.assertEqual(x.shape, (count, 784))
                self.assertEqual(edges.shape, (2, 0))
                self.assertEqual(weights.shape, (0,))
                self.assertTrue(bool(torch.isfinite(x).all()))
                self.assertTrue(bool((x.abs() <= 10).all()))
                self.assertEqual(state_hash(self.generator), before)

    def test_injection_changes_only_target_support_and_preserves_inputs(self):
        part = scene(2)
        original = deepcopy(part)
        self.injector.bind(part)
        args = self.arguments(part)
        before = [value.clone() for value in args[:4]]
        state = state_hash(self.generator)
        x, edges, weights = self.injector.inject_trigger(*args)
        allowed = torch.zeros_like(x, dtype=torch.bool)
        selected = self.generator.selected_coordinates[self.generator.hard_support]
        allowed[0, selected] = True
        self.assertTrue(torch.equal(x[~allowed], args[1][~allowed]))
        self.assertTrue(bool((x[allowed].abs() <= 10).all()))
        self.assertTrue(torch.equal(edges, args[2]))
        self.assertTrue(torch.equal(weights, args[3]))
        self.assertFalse(x.requires_grad)
        for actual, expected in zip(args[:4], before):
            self.assertTrue(torch.equal(actual, expected))
        for key in ('x', 'y', 'eligible', 'canonical_pairs', 'edge_index'):
            self.assertTrue(np.array_equal(part[key], original[key]))
        self.assertEqual(state_hash(self.generator), state)

    def test_unbound_stale_scene_or_training_mode_is_rejected(self):
        part = scene(2)
        with self.assertRaises(ValueError):
            self.injector.inject_trigger(*self.arguments(part))
        self.injector.bind(part)
        args = self.arguments(part)
        args[1][0, 0] += .01
        with self.assertRaises(ValueError):
            self.injector.inject_trigger(*args)
        self.generator.train()
        with self.assertRaises(ValueError):
            self.injector.inject_trigger(*self.arguments(part))

    def test_support_escape_and_domain_escape_are_rejected(self):
        part = scene(2)
        self.injector.bind(part)
        args = self.arguments(part)
        selected = self.generator.selected_coordinates[self.generator.hard_support]
        off_support = next(i for i in range(784) if i not in selected.tolist())
        for kind in ('nontarget_row', 'off_support', 'domain'):
            features = args[1].clone()
            if kind == 'nontarget_row':
                features[1, selected[0]] += .1
            elif kind == 'off_support':
                features[0, off_support] += .1
            else:
                features[0, selected[0]] = 10.01
            changed = InjectedFeatures(features, args[2].clone(), args[3].clone())
            with self.subTest(kind=kind), patch.object(self.generator, 'inject', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'scope or domain'):
                    self.injector.inject_trigger(*args)


if __name__ == '__main__':
    unittest.main()
