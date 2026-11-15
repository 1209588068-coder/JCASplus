"""Synthetic contracts for frozen high-ASR support restoration; no AV2 arrays."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from spear_current.attention_context import AttentionContext
from spear_current.evaluate import state_hash
from spear_current.support_generator import SupportMaskedGenerator, fixed_support
from spear_current.support_replication_runtime import (
    ARMS, FrozenSupportGenerator, SUPPORT_RELATIVE_ROOT, SUPPORT_SHA256,
    context_from_edges, frozen_support, source_bindings, support_records,
    restore_uniform, restore_uniform_surrogate,
)
from spear_current.support_schema import SELECTED_COORDINATES
from tests_current.test_domain_generator import setup


class FrozenSupportRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.anchor, _, self.x, self.context, self.targets = setup()
        self.mask = fixed_support('random_semantic', 24, seed=20260621)

    def test_exact_historical_wrapper_output_and_tensor_only_state(self):
        reference = SupportMaskedGenerator(self.anchor, hard_support=self.mask).eval()
        rng = torch.get_rng_state().clone()
        model = frozen_support(self.anchor, self.mask)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(model.output_mode, 'identity')
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in model.state_dict().values()))
        self.assertEqual(len(state_hash(model)), 64)
        self.assertTrue(torch.equal(reference(self.x, self.context, self.targets), model(self.x, self.context, self.targets)))
        before = self.x.clone()
        for context in (self.context, SimpleNamespace(edge_index=self.context.edge_index)):
            actual = model.inject(self.x, context, self.targets, detach=True)
            expected = reference.inject(self.x, self.context, self.targets, detach=True)
            self.assertTrue(torch.equal(actual.features, expected.features))
            self.assertTrue(torch.equal(actual.edge_index, expected.edge_index))
            self.assertTrue(torch.equal(actual.edge_weight, expected.edge_weight))
        self.assertTrue(torch.equal(self.x, before))

    def test_frozen_owned_modules_and_injection_permissions(self):
        model = frozen_support(self.anchor, self.mask)
        for p, q in zip(model.anchor.parameters(), self.anchor.parameters()):
            self.assertNotEqual(p.data_ptr(), q.data_ptr())
        model.train(True)
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        output = model.inject(self.x, self.context, self.targets)
        allowed = torch.zeros_like(self.x, dtype=torch.bool)
        allowed[self.targets[:, None], model.selected_coordinates[model.hard_support]] = True
        self.assertTrue(torch.equal(output.features[~allowed], self.x[~allowed]))
        self.assertTrue(torch.equal(output.features[:, [357, 715]], self.x[:, [357, 715]]))
        model.anchor.residual.head.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'contract'):
            model(self.x, self.context, self.targets)

    def test_no_output_clipping(self):
        anchor = deepcopy(self.anchor)
        with torch.no_grad():
            anchor.residual.head.weight.zero_()
            anchor.residual.head.bias.fill_(100)
        model = frozen_support(anchor, self.mask)
        result = model(self.x, self.context, self.targets)
        self.assertTrue(bool((result[:, model.hard_support].abs() > 10).all()))

    def test_strict_restore_support_and_nonfinite_inputs_rejected(self):
        model = frozen_support(self.anchor, self.mask)
        restored = frozen_support(self.anchor, self.mask)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(state_hash(restored), state_hash(model))
        for option in (dict(strict=False), dict(assign=True)):
            with self.assertRaises(ValueError):
                restored.load_state_dict(model.state_dict(), **option)
        state = deepcopy(model.state_dict())
        state['hard_support'][0] = True
        with self.assertRaises(ValueError):
            restored.load_state_dict(state)
        invalid = self.x.clone(); invalid[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            model(invalid, self.context, self.targets)
        with self.assertRaises(ValueError):
            model(self.x, self.context, torch.tensor([-1]))

    def test_metadata_and_incomplete_groups_rejected(self):
        invalid = self.mask.clone(); invalid[0] = True
        with self.assertRaises(ValueError):
            frozen_support(self.anchor, invalid)
        invalid = self.mask.clone()
        active = [i for i, c in enumerate(SELECTED_COORDINATES) if c in (350, 742, 777)]
        single = SELECTED_COORDINATES.index(760)
        if bool(invalid[single]):
            invalid[active[0]], invalid[single] = False, True
            # The source random mask removes one triple; choose a direct legal
            # 24-count mask with a partial triple and a replacement singleton.
        legal = [c for c in SELECTED_COORDINATES if c not in (357, 715)]
        broken = torch.tensor([c in set(legal[:24]) for c in SELECTED_COORDINATES])
        # Break a complete retained triplet while replacing by an excluded legal singleton.
        for coordinate in (350, 742, 777):
            if bool(broken[SELECTED_COORDINATES.index(coordinate)]):
                broken[SELECTED_COORDINATES.index(coordinate)] = False
                break
        replacement = next(i for i,c in enumerate(SELECTED_COORDINATES)
                           if c not in (357,715,350,742,777) and not bool(broken[i]))
        broken[replacement] = True
        with self.assertRaises(ValueError):
            frozen_support(self.anchor, broken)

    def test_disconnected_components_isolates_and_empty_edges(self):
        features = torch.zeros(6, 784)
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        context = context_from_edges(features, edges)
        self.assertEqual(context.pair_scene.tolist(), [0, 0, 2, 2, 4, 5])
        torch.testing.assert_close(context.edge_index, edges, rtol=0, atol=0)
        self.assertTrue(torch.equal(context.score_permutation, torch.arange(4)))
        empty = context_from_edges(features, torch.empty((2, 0), dtype=torch.long))
        self.assertEqual(empty.pair_scene.tolist(), list(range(6)))
        self.assertEqual(empty.edge_index.shape, (2, 0))
        bad = torch.tensor([[0, 0], [1, 1]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            context_from_edges(features, bad)
        with self.assertRaisesRegex(ValueError, 'self'):
            context_from_edges(features, torch.tensor([[0], [0]]))

    def test_different_valid_role_sidecars_are_uniform_equivalent(self):
        edges = self.context.edge_index
        alternative = AttentionContext(edges.clone(), torch.zeros_like(self.context.edge_type),
            self.context.pair_scene.clone(), self.context.num_pairs, torch.arange(edges.shape[1]))
        model = frozen_support(self.anchor, self.mask)
        a = model(self.x, self.context, self.targets)
        b = model(self.x, alternative, self.targets)
        c = model(self.x, SimpleNamespace(edge_index=edges), self.targets)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, c))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_same_device_injection(self):
        from spear_current.full_ablation_evaluation import _fp32
        model = frozen_support(self.anchor, self.mask).to('cuda')
        x = self.x.to('cuda'); context = self.context.to('cuda'); targets = self.targets.to('cuda')
        with _fp32():
            real = model.inject(x, context, targets)
            repeated = model.inject(x, context, targets)
            edge_only = model.inject(x, SimpleNamespace(edge_index=context.edge_index), targets)
        # CUDA scatter is not bitwise deterministic even for identical contexts.
        # A 24-repeat probe found a max 4.77e-7 error for BOTH comparisons.
        torch.testing.assert_close(real.features, repeated.features, rtol=0, atol=1e-6)
        torch.testing.assert_close(real.features, edge_only.features, rtol=0, atol=1e-6)
        allowed = torch.zeros_like(x, dtype=torch.bool)
        allowed[targets[:, None], model.selected_coordinates[model.hard_support]] = True
        self.assertTrue(torch.equal(edge_only.features[~allowed], x[~allowed]))
        self.assertTrue(torch.equal(real.edge_index, edge_only.edge_index))
        self.assertTrue(torch.equal(real.edge_weight, edge_only.edge_weight))
        self.assertEqual(real.features.device.type, 'cuda')

    def test_restoration_preserves_cpu_and_cuda_rng(self):
        from spear_current.attention_generator import ContentAttentionGenerator
        from spear_current.models import GCN, GraphTrojanNet
        from spear_current.role_generator import PairRoleResidualGenerator
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7)
            base = GraphTrojanNet(torch.device('cpu'), 80, 30, layernum=2)
            shadow = GCN(784, 80, 2, dropout=0., device=torch.device('cpu'))
            role = PairRoleResidualGenerator(base, shadow, torch.tensor(SELECTED_COORDINATES),
                input_dim=784, hidden_dim=80, mode='untyped', threshold=.5)
            original = ContentAttentionGenerator(role, attention_mode='uniform', score_dim=32)
        if torch.cuda.is_available():
            torch.cuda.get_rng_state()  # Initialize before taking the protected snapshot.
        before_cpu = torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        binding = dict(path='synthetic-only.pt', sha256='synthetic')
        for restore, key, state in ((restore_uniform, 'uniform', original.state_dict()),
                                    (restore_uniform_surrogate, 'surrogate', shadow.state_dict())):
            with patch('spear_current.support_replication_runtime.source_bindings', return_value={key:binding}), \
                    patch('spear_current.support_replication_runtime._state', return_value=state):
                restored = restore(object())
            self.assertTrue(torch.equal(before_cpu, torch.get_rng_state()))
            if before_cuda is not None:
                self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state()))
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))


class SupportSourceTests(unittest.TestCase):
    def test_support_source_rejects_changed_bytes_without_tensor_loading(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / SUPPORT_RELATIVE_ROOT / ARMS[0]
            path.mkdir(parents=True)
            (path/'support.json').write_text('{}')
            with patch('torch.load', side_effect=AssertionError('tensor read forbidden')):
                with self.assertRaisesRegex(ValueError, 'registered support'):
                    support_records(tmp)

    def test_registered_sources_are_hash_bound_without_tensor_loading(self):
        # Synthetic metadata uses a temporary known hash mapping; production
        # constants continue binding the four already registered support files.
        with TemporaryDirectory() as tmp:
            from spear_current.runtime import sha256
            hashes = {}
            for arm in ARMS:
                path = Path(tmp) / SUPPORT_RELATIVE_ROOT / arm
                path.mkdir(parents=True)
                spec = (dict(kind='learned', grouping='semantic', budget=24)
                        if arm == 'semantic' else dict(kind='fixed', control='random_semantic',
                            budget=24, seed=int(arm.rsplit('s', 1)[1])))
                value = dict(arm=arm, spec=spec, selected_coordinates=list(SELECTED_COORDINATES),
                    hard_support=fixed_support('random_semantic',24,seed=20260621).tolist(),
                    hard_support_capacity=24)
                (path/'support.json').write_text(json.dumps(value))
                hashes[arm] = sha256(path/'support.json')
            with patch.dict(SUPPORT_SHA256, hashes), patch('torch.load', side_effect=AssertionError('tensor read forbidden')):
                records = support_records(tmp)
            self.assertEqual(tuple(records), ARMS)
            self.assertTrue(all(sum(r['hard_support']) == 24 for r in records.values()))


if __name__ == '__main__':
    unittest.main()
