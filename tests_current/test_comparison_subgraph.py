"""Synthetic CPU-only tests; no dataset, graph asset, checkpoint, or I/O."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from comparison_methods import subgraph as sg


def fixture():
    rng = torch.Generator().manual_seed(18)
    x = torch.randn((24, 6), generator=rng)
    x[:12] -= 2
    x[12:] += 2
    # Isolated self-loop components keep the fixture's clustering well-separated.
    edge_index = torch.arange(24).repeat(2, 1)
    y = torch.cat((torch.zeros(12, dtype=torch.long), torch.ones(12, dtype=torch.long)))
    train = torch.tensor([0, 1, 12, 13])
    candidates = torch.tensor([i for i in range(24) if i not in train.tolist()])
    return x, edge_index, y, train, candidates


def small_config(**overrides):
    return sg.validate_config({"epochs": 2, "inner": 1, "hidden": 5,
                               "poison_count": 2, "outer_count": 3,
                               "select_pretrain_epochs": 2, **overrides})


class ComparisonSubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fit(self, method="ugba", *, seed=21, config=None, data=None, progress=None):
        return sg.fit_subgraph(method, *(fixture() if data is None else data), seed=seed,
                               config=small_config() if config is None else config, progress=progress)

    def test_defaults_complete_and_copy(self):
        actual = sg.validate_config({})
        self.assertEqual(actual, sg.DEFAULT_CONFIG)
        actual["epochs"] = 1
        self.assertEqual(sg.DEFAULT_CONFIG["epochs"], 200)
        self.assertEqual(sg.DEFAULT_CONFIG["poison_count"], 26)
        self.assertEqual(sg.DEFAULT_CONFIG["trigger_size"], 3)
        json.dumps(sg.validate_config({}), allow_nan=False)

    def test_rejects_invalid_configuration(self):
        for config in ({"mystery": 1}, {"epochs": 0}, {"inner": True},
                       {"lr": 0}, {"lr": float("nan")}, {"weight_decay": -1},
                       {"target_class": 2}, {"outer_count": -1},
                       {"generator_checkpoint": "best_asr"}, {"selector_dropout": 1},
                       {"homo_threshold": 2}, {"homo_weight": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                sg.validate_config(config)

    def test_injection_prefix_edges_budget_and_binary_filter(self):
        x, edges, *_ = fixture()
        targets = torch.tensor([2, 17])
        generator = sg.SubgraphGenerator(6, 3)
        with torch.no_grad():
            generator.edge.weight.zero_()
            generator.edge.bias.fill_(-1)
        px, pe, pw = generator.inject(x, edges, targets)
        self.assertEqual(px.shape, (30, 6))
        self.assertTrue(torch.equal(px[:24], x))
        self.assertTrue(torch.equal(pe[:, :24], edges))
        self.assertEqual(pe.shape[1], 24 + 4)  # Only two bidirectional attachment edges.
        self.assertTrue(torch.equal(pw, torch.ones_like(pw)))
        self.assertEqual(pe[:, 24:].tolist(), [[2, 17, 24, 27], [24, 27, 2, 17]])
        with torch.no_grad():
            generator.edge.bias.fill_(1)
        px, pe, pw = generator.inject(x, edges, targets)
        self.assertEqual(pe.shape[1], 24 + 2 * 2 * (1 + 3))
        self.assertTrue((pe[:, 24:] >= 24).any(dim=0).all())
        self.assertFalse(px.requires_grad)
        self.assertTrue(torch.equal(pw, torch.ones_like(pw)))

    def test_threshold_strict_and_straight_through_gradient(self):
        score = torch.tensor([0.49, 0.5, 0.51], requires_grad=True)
        result = sg._StraightThrough.apply(score, 0.5)
        self.assertEqual(result.tolist(), [0, 0, 1])
        (result * torch.tensor([1., 2., 3.])).sum().backward()
        self.assertEqual(score.grad.tolist(), [1, 2, 3])

    def test_training_retains_zero_edge_gradient(self):
        x, edges, *_ = fixture()
        generator = sg.SubgraphGenerator(6, 3)
        with torch.no_grad():
            generator.edge.weight.zero_()
            generator.edge.bias.fill_(-1)
        px, pe, pw = generator._inject_training(x, edges, torch.tensor([2]))
        self.assertEqual(pe.shape[1], 24 + 8)
        self.assertEqual(int((pw == 0).sum()), 6)
        pw.sum().backward()
        self.assertTrue((generator.edge.bias.grad != 0).all())

    def test_empty_injection_is_independent_copy(self):
        x, edges, *_ = fixture()
        generator = sg.SubgraphGenerator(6)
        px, pe, pw = generator.inject(x, edges, torch.tensor([], dtype=torch.long))
        self.assertTrue(torch.equal(x, px))
        self.assertTrue(torch.equal(edges, pe))
        self.assertNotEqual(x.data_ptr(), px.data_ptr())
        self.assertNotEqual(edges.data_ptr(), pe.data_ptr())
        self.assertEqual(len(pw), edges.shape[1])

    def test_ugba_budget_prefix_supervision_and_json_record(self):
        data = fixture()
        before = [t.clone() for t in data]
        events = []
        fit = self.fit(data=data, progress=events.append)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(data, before)))
        self.assertEqual(fit["record"]["counters"], {"generator_steps": 2, "surrogate_steps": 2,
                         "discriminator_steps": 0, "selector_steps": 2, "autoencoder_steps": 0})
        self.assertEqual(len(fit["record"]["outer_indices"]), 5)
        self.assertEqual(len(set(fit["record"]["outer_indices"])), 5)
        self.assertEqual(fit["record"]["generator_selected_epoch"], 1)
        self.assertEqual(fit["poisoned_y"].shape, (24,))
        self.assertEqual(fit["poisoned_x"].shape, (30, 6))
        self.assertTrue(torch.equal(fit["poisoned_x"][:24], data[0]))
        self.assertTrue(torch.equal(fit["poisoned_y"][data[3]], data[2][data[3]]))
        self.assertTrue((fit["poisoned_y"][fit["attachment_indices"]] == 1).all())
        self.assertFalse(torch.isin(fit["attachment_indices"], data[3]).any())
        rest = ~torch.isin(torch.arange(24), fit["attachment_indices"])
        self.assertTrue(torch.equal(fit["poisoned_y"][rest], data[2][rest]))
        self.assertEqual(len(fit["fit_indices"]), 6)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in fit["generator"].parameters()))
        self.assertFalse(fit["generator"].training)
        self.assertEqual([e["phase"] for e in events], ["selector", "selector", "generator", "generator"])
        self.assertTrue(fit["record"]["fresh_initialization"])
        self.assertFalse(fit["record"]["checkpoint_loaded"])
        self.assertEqual(fit["record"]["author_commit"], sg.AUTHOR_SHA)
        json.dumps(fit["record"], allow_nan=False)

    def test_gta_joint_budget_and_random_selection(self):
        fit = self.fit("gta", config=small_config(inner=2))
        self.assertEqual(fit["record"]["counters"]["selector_steps"], 0)
        self.assertEqual(fit["record"]["counters"]["generator_steps"], 4)
        self.assertEqual(fit["record"]["counters"]["surrogate_steps"], 4)
        candidates = fixture()[-1]
        indices = np.arange(len(candidates))
        np.random.RandomState(21).shuffle(indices)
        self.assertTrue(torch.equal(fit["attachment_indices"], candidates[indices[:2]]))
        self.assertTrue(all(row["homophily_loss_raw"] == 0 for row in fit["record"]["trace"]))

    def _one_epoch_oracle(self, method):
        """Independent expression of the audited release's two update rules."""
        cfg = small_config(epochs=1)
        fit = self.fit(method, config=cfg)
        x, edges, labels, train, _ = fixture()
        selected = fit["attachment_indices"]
        target_labels = labels.clone()
        target_labels[selected] = cfg["target_class"]
        supervised = torch.cat((train, selected))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21)
            surrogate = sg.GCN(6, cfg["hidden"], 2, dropout=0., device=torch.device("cpu"))
            generator = sg.SubgraphGenerator(6, cfg["trigger_size"], cfg["edge_threshold"])
            opt_s = torch.optim.Adam(surrogate.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            px, pe, pw = generator._inject_training(x, edges, selected)
            if method == "ugba":
                px, pw = px.detach(), pw.detach()
            loss = F.nll_loss(surrogate(px, pe, pw)[supervised], target_labels[supervised])
            loss.backward()
            opt_s.step()
            if method == "gta":
                opt_g.step()
            else:
                opt_g.zero_grad(set_to_none=True)
                for parameter in surrogate.parameters():
                    parameter.requires_grad_(False)
                outer = torch.tensor(fit["record"]["outer_indices"])
                px, pe, pw = generator._inject_training(x, edges, outer)
                output = surrogate(px, pe, pw)
                outer_labels = labels.clone()
                outer_labels[outer] = cfg["target_class"]
                outer_supervised = torch.cat((train, outer))
                target = F.nll_loss(output[outer_supervised], outer_labels[outer_supervised])
                # The author averages one orientation; duplicated undirected
                # orientations have the identical mathematical mean.
                half = (pe.shape[1] - edges.shape[1]) // 2
                trojan_edges = pe[:, edges.shape[1]:edges.shape[1] + half]
                active = pw[edges.shape[1]:edges.shape[1] + half] > 0
                trojan_edges = trojan_edges[:, active]
                cosine = F.cosine_similarity(px[trojan_edges[0]], px[trojan_edges[1]])
                homo = (cfg["homo_threshold"] - cosine).clamp_min(0).mean()
                (cfg["target_weight"] * target + cfg["homo_weight"] * homo).backward()
                opt_g.step()
        for name, actual in fit["generator"].state_dict().items():
            with self.subTest(method=method, tensor=name):
                self.assertTrue(torch.allclose(actual, generator.state_dict()[name], atol=1e-7, rtol=1e-6))

    def test_gta_one_step_matches_joint_update_oracle(self):
        self._one_epoch_oracle("gta")

    def test_ugba_one_step_matches_detached_inner_and_homophily_oracle(self):
        self._one_epoch_oracle("ugba")

    def test_seed_repeatability_and_caller_rng_isolation(self):
        for method in ("ugba", "gta"):
            with self.subTest(method=method):
                torch.manual_seed(972)
                before_torch = torch.get_rng_state().clone()
                np.random.seed(951)
                before_numpy = np.random.get_state()
                first = self.fit(method)
                self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
                after_numpy = np.random.get_state()
                self.assertEqual(before_numpy[0], after_numpy[0])
                self.assertTrue(np.array_equal(before_numpy[1], after_numpy[1]))
                self.assertEqual(before_numpy[2:], after_numpy[2:])
                second = self.fit(method)
                self.assertEqual(first["record"], second["record"])
                self.assertTrue(torch.equal(first["poisoned_x"], second["poisoned_x"]))
                self.assertTrue(torch.equal(first["poisoned_edge_index"], second["poisoned_edge_index"]))

    def test_unsupervised_candidate_labels_are_unused(self):
        data = list(fixture())
        first = self.fit(data=data)
        data[2] = data[2].clone()
        data[2][data[4]] = -100
        second = self.fit(data=data)
        self.assertTrue(torch.equal(first["attachment_indices"], second["attachment_indices"]))
        self.assertTrue(torch.equal(first["poisoned_x"], second["poisoned_x"]))
        self.assertEqual(first["record"]["trace"], second["record"]["trace"])

    def test_ugba_homophily_is_active_and_generator_changes(self):
        fit = self.fit(config=small_config(homo_threshold=1.0))
        self.assertTrue(all(r["homophily_loss_raw"] > 0 for r in fit["record"]["trace"]))
        self.assertNotEqual(fit["record"]["initial_state_sha256"]["generator"],
                            fit["record"]["final_generator_state_sha256"])
        for row in fit["record"]["trace"]:
            self.assertAlmostEqual(row["generator_loss"], row["target_loss_raw"] + 100 * row["homophily_loss_raw"], places=4)

    def test_degenerate_selector_scaling_and_shortfall(self):
        self.assertTrue(np.array_equal(sg._minmax(np.array([5., 5.])), np.zeros(2)))
        class AllTarget:
            cluster_centers_ = np.zeros((2, 5))
            def fit(self, values):
                return self
            def predict(self, values):
                return np.ones(len(values), dtype=int)
        with patch.object(sg, "KMeans", return_value=AllTarget()), self.assertRaisesRegex(ValueError, "no fallback"):
            self.fit()

    def test_invalid_inputs_before_fitting(self):
        cases = []
        d = list(fixture()); d[0] = d[0].double(); cases.append(d)
        d = list(fixture()); d[0][0, 0] = float("nan"); cases.append(d)
        d = list(fixture()); d[1][0, 0] = 24; cases.append(d)
        d = list(fixture()); d[2] = d[2].float(); cases.append(d)
        d = list(fixture()); d[3] = torch.tensor([0, 0, 12]); cases.append(d)
        d = list(fixture()); d[4] = torch.cat((d[4], d[3][:1])); cases.append(d)
        d = list(fixture()); d[4] = d[4][:3]; cases.append(d)
        d = list(fixture()); d[2][d[3]] = 0; cases.append(d)
        for i, data in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                self.fit(data=data)
        for seed in (True, -1, 2 ** 32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.fit(seed=seed)
        with self.assertRaises(ValueError):
            self.fit("not_a_method")

    def test_dpgba_is_explicitly_unsupported_not_silently_substituted(self):
        self.assertEqual(sg.SUPPORTED_METHODS, ("ugba", "gta"))
        with self.assertRaisesRegex(NotImplementedError, "DPGBA"):
            self.fit("dpgba")


if __name__ == "__main__":
    unittest.main()
