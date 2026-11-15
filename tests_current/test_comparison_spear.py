"""Synthetic CPU checks of the fresh source-SPEAR adaptation; no AV2 I/O."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from comparison_methods.spear import (
    DEFAULT_CONFIG, _FeatureMLP, _global_sage, _rank_candidates,
    fit_spear, validate_config,
)


def fixture():
    rng = torch.Generator().manual_seed(711)
    x = torch.randn(10, 6, generator=rng)
    edges = torch.tensor([(i, j) for i in range(10) for j in range(10) if abs(i - j) == 1], dtype=torch.long).T
    y = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 0, -1])
    return x, edges, y, torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])


def small_config():
    return dict(epochs=3, inner=1, hidden=4, poison_count=2, feature_count=2,
                select_pretrain_epochs=2, feature_pretrain_epochs=2,
                sage_permutations=8, sage_background=2, sage_batch_size=4, outer_count=2)


class FreshSPEARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_source_selector_literal_log_entropy_and_round_robin(self):
        values = torch.tensor([[3., 0.], [0., 2.], [.1, 0.], [0., .2], [1., 0.], [0., .5]]).log_softmax(1)
        candidates = torch.tensor([5, 3, 2, 4, 1, 0])
        rank, scores, _ = _rank_candidates(values, candidates)
        normalized = values / values.sum(1, keepdim=True)
        expected_scores = -(normalized * torch.log2(normalized)).sum(1)
        ordered = candidates[expected_scores[candidates].argsort(descending=True)]
        classes = values.argmax(1)
        per_class = [ordered[classes[ordered] == i].tolist() for i in range(2)]
        expected = [per_class[k][i] for i in range(3) for k in range(2)]
        self.assertEqual(rank.tolist(), expected)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)

    def test_sage_matches_literal_numpy_source_imputer_and_estimator(self):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            model = _FeatureMLP(3, 4, 2).eval()
        x = torch.tensor([[.2, .3, -.1], [.6, -.8, .7], [.3, -.2, .4]])
        y = torch.tensor([0, 1, 1])
        background = x[:2]
        values, stderr, evidence = _global_sage(model, x, y, background,
            permutations=8, batch_size=4, seed=50)
        rng = np.random.default_rng(50)
        source_scores = []
        for _ in range(2):
            ids = rng.choice(len(x), 4)
            orders = np.tile(np.arange(3), (4, 1))
            for row in orders:
                rng.shuffle(row)
            sample = x.numpy()[ids]
            target = np.eye(2, dtype=np.float32)[y.numpy()[ids]]
            mask = np.zeros((4, 3), dtype=bool)
            scores = np.zeros((4, 3))

            def source_loss():
                filled = sample.repeat(2, 0)
                repeated_mask = mask.repeat(2, 0)
                background_repeat = np.tile(background.numpy(), (4, 1))
                filled[~repeated_mask] = background_repeat[~repeated_mask]
                with torch.no_grad():
                    pred = model(torch.tensor(filled)).numpy().reshape(4, 2, 2).mean(1)
                return np.square(pred - target).sum(1)

            previous = source_loss()
            for position in range(3):
                cols = orders[:, position]
                mask[np.arange(4), cols] = 1
                current = source_loss()
                scores[np.arange(4), cols] = previous - current
                previous = current
            source_scores.append(scores)
        expected = np.concatenate(source_scores)
        np.testing.assert_allclose(values, expected.mean(0), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(stderr, expected.std(0) / np.sqrt(8), rtol=1e-6, atol=1e-7)
        self.assertEqual(evidence["draws"], 8)

    def test_fresh_fit_is_deterministic_scope_limited_and_rng_isolated(self):
        inputs = fixture()
        copies = [t.clone() for t in inputs]
        rng = torch.get_rng_state().clone()
        first = fit_spear(*inputs, seed=31, config=small_config())
        second = fit_spear(*inputs, seed=31, config=small_config())
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for actual, original in zip(inputs, copies):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        self.assertEqual(first["record"]["selection_sha256"], second["record"]["selection_sha256"])
        self.assertEqual(first["record"]["final_generator_sha256"], second["record"]["final_generator_sha256"])
        self.assertEqual(first["record"]["generator_steps"], 3)
        self.assertEqual(first["record"]["surrogate_steps"], 3)
        self.assertEqual(len(first["record"]["fresh_initial_state_sha256"]), 4)
        self.assertEqual(first["record"]["feature_budget"]["source_default"], 30)
        self.assertFalse(first["record"]["sage"]["convergence_claim"])
        self.assertEqual(first["record"]["sage"]["explanation_indices"], inputs[3].tolist())
        x, edges, y, train, candidates = inputs
        attachments = first["attachment_indices"]
        self.assertTrue(bool(torch.isin(attachments, candidates).all()))
        torch.testing.assert_close(first["fit_indices"], torch.cat((train, attachments)))
        expected_y = y.clone()
        expected_y[attachments] = 1
        torch.testing.assert_close(first["poisoned_y"], expected_y)
        torch.testing.assert_close(first["poisoned_edge_index"], edges, rtol=0, atol=0)
        allowed = torch.zeros_like(x, dtype=torch.bool)
        allowed[attachments[:, None], first["generator"].selected_coordinates] = True
        self.assertTrue(torch.equal(first["poisoned_x"][~allowed], x[~allowed]))
        self.assertTrue(torch.isfinite(first["poisoned_x"]).all())
        replay, replay_edges, replay_weights = first["generator"].inject(x, edges, attachments)
        torch.testing.assert_close(replay, first["poisoned_x"], rtol=0, atol=0)
        torch.testing.assert_close(replay_edges, edges, rtol=0, atol=0)
        torch.testing.assert_close(replay_weights, torch.ones(edges.shape[1]), rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in first["generator"].parameters()))
        self.assertEqual(first["record"]["selected_epoch"], min(range(3), key=lambda i: first["record"]["history"][i]["outer_loss"]))

    def test_invalid_candidates_rejected_before_any_optimizer_step(self):
        inputs = list(fixture())
        inputs[-1] = torch.tensor([0, 4, 5, 6, 7])
        with patch.object(torch.optim.Adam, "step") as step:
            with self.assertRaisesRegex(ValueError, "disjoint"):
                fit_spear(*inputs, seed=1, config=small_config())
            step.assert_not_called()
        inputs = list(fixture())
        inputs[2][5] = 1
        with self.assertRaisesRegex(ValueError, "eligible-negative"):
            fit_spear(*inputs, seed=1, config=small_config())

    def test_preregistered_defaults_and_invalid_config(self):
        self.assertEqual(validate_config(), DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG["feature_count"], 24)
        self.assertEqual(DEFAULT_CONFIG["poison_count"], 26)
        for bad in ({"epochs": True}, {"feature_count": 0}, {"lr": float("nan")}, {"target_class": 0}, {"cached_indices": []}):
            with self.assertRaises(ValueError):
                validate_config(bad)


if __name__ == "__main__":
    unittest.main()
