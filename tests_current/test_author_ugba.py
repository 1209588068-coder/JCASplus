"""Synthetic tests of the actual pinned UGBA author code, not replacement math."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from author_comparisons import ugba


def fixture():
    rng = torch.Generator().manual_seed(109)
    x = torch.randn(560, 4, generator=rng)
    x[:280] -= 2
    x[280:] += 2
    # Last node is isolated background; keep it, do not add a synthetic loop.
    ids = torch.arange(559)
    edges = torch.cat((torch.stack((ids, ids)),
                       torch.tensor([[0, 1, 2, 3, 20, 21], [1, 0, 3, 2, 21, 20]])), dim=1)
    y = torch.zeros(560, dtype=torch.long)
    train = torch.tensor([0, 1, 2, 3, 280, 281, 282, 283])
    dev = torch.tensor([4, 5, 6, 7, 284, 285, 286, 287])
    y[train[4:]] = 1
    y[dev[4:]] = 1
    y[558:] = -1
    candidates = torch.tensor([i for i in range(558) if i not in set(torch.cat((train, dev)).tolist())])
    return x, edges, y, train, dev, candidates


def small(**extra):
    return ugba.validate_config({"hidden": 4, "epochs": 2, "trojan_epochs": 2,
                                 "vs_number": 2, "debug": False,
                                 "selection_method": "none", **extra})


class AuthorUGBATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_defaults_match_author_parser_and_explicit_task_overrides(self):
        tree = ast.parse((ugba.SOURCE_ROOT / "run_adaptive.py").read_text())
        defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                name = ast.literal_eval(node.args[0])[2:].replace("-", "_")
                for option in node.keywords:
                    if option.arg == "default":
                        defaults[name] = ast.literal_eval(option.value)
        defaults.pop("seed")
        self.assertEqual(ugba.AUTHOR_DEFAULTS, defaults)
        cfg = ugba.validate_config()
        self.assertEqual((cfg["hidden"], cfg["epochs"], cfg["trojan_epochs"], cfg["trigger_size"]), (32, 200, 400, 3))
        self.assertEqual(cfg["selection_method"], "cluster_degree")
        self.assertEqual(cfg["target_class"], 1)
        self.assertEqual(cfg["vs_number"], 26)
        self.assertEqual(cfg["homo_loss_weight"], defaults["homo_loss_weight"])

    def test_pristine_source_and_import_runtime_diff(self):
        before = ugba.verify_source()
        self.assertFalse(before["dirty"])
        self.assertEqual(before["commit"], ugba.AUTHOR_COMMIT)
        self.assertEqual(before["license_files"], [])
        with TemporaryDirectory() as work:
            runtime = ugba.prepare_runtime(work)
            self.assertEqual(runtime, ugba.prepare_runtime(work))
            root = Path(runtime["runtime_root"])
            changed = []
            for relative in runtime["runtime_files_sha256"]:
                if (root / relative).read_bytes() != (ugba.SOURCE_ROOT / relative).read_bytes():
                    changed.append(relative)
            self.assertEqual(sorted(changed), ["heuristic_selection.py", "models/construct.py"])
            for relative in runtime["unmodified_core_files"]:
                self.assertEqual((root / relative).read_bytes(), (ugba.SOURCE_ROOT / relative).read_bytes())
            patch_text = Path(runtime["patch_path"]).read_text()
            self.assertIn("nclass = args.num_classes", patch_text)
            self.assertIn("num_nodes=x.shape[0]", patch_text)
            self.assertIn("from sklearn_extra import cluster", patch_text)
            with ugba.load_author_modules(root, ("heuristic_selection", "models.backdoor", "models.SAGE")) as modules:
                self.assertTrue(all(str(root) in m.__file__ for m in modules.values()))
            json.dumps(runtime)
        self.assertEqual(before, ugba.verify_source())

    def test_runtime_rejects_overwrite_and_source_location(self):
        with self.assertRaises(ValueError):
            ugba.prepare_runtime(ugba.SOURCE_ROOT)
        with TemporaryDirectory() as work:
            runtime = ugba.prepare_runtime(work)
            target = Path(runtime["runtime_root"]) / "models/backdoor.py"
            target.write_text(target.read_text() + "\n# synthetic tamper\n")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                ugba.prepare_runtime(work)

    def test_native_fit_matches_direct_author_calls(self):
        data = fixture()
        original = [t.clone() for t in data]
        cfg, seed = small(), 20260621
        with TemporaryDirectory() as work, redirect_stdout(StringIO()):
            before_rng = torch.get_rng_state().clone()
            fitted = ugba.fit_ugba(*data, seed=seed, work_dir=work, config=cfg)
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(data, original)))
            runtime = fitted["record"]["runtime"]
            with ugba.load_author_modules(runtime["runtime_root"], ("heuristic_selection", "models.backdoor")) as modules:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    args = SimpleNamespace(**cfg, seed=seed, cuda=False)
                    selected = modules["heuristic_selection"].obtain_attach_nodes(args, data[5], cfg["vs_number"])
                    remaining = torch.tensor(list(set(data[5].numpy()) - set(selected.numpy())), dtype=torch.long)
                    native = modules["models.backdoor"].Backdoor(args, torch.device("cpu"))
                    native.fit(data[0], data[1], None, data[2], data[3], selected, remaining)
                    px, pe, pw, py = native.get_poisoned()
            for key, expected in (("attachment_indices", selected), ("poisoned_x", px),
                                  ("poisoned_edge_index", pe), ("poisoned_edge_weight", pw), ("poisoned_y", py)):
                self.assertTrue(torch.equal(fitted[key], expected), key)
            for name, value in native.trojan.state_dict().items():
                self.assertTrue(torch.equal(fitted["generator"].trojan.state_dict()[name], value))
            self.assertTrue(fitted["record"]["author_best_state_restored"])
            self.assertEqual(int((fitted["poisoned_y"] == -1).sum()), 2)
            json.dumps(fitted["record"], allow_nan=False)

    def test_native_evaluation_retains_zero_edges_training_filters(self):
        with TemporaryDirectory() as work, redirect_stdout(StringIO()):
            fitted = ugba.fit_ugba(*fixture(), seed=21, work_dir=work, config=small(trojan_epochs=1))
            generator = fitted["generator"]
            with torch.no_grad():
                generator.trojan.edge.weight.zero_()
                generator.trojan.edge.bias.fill_(-1)
            x, edges, *_ = fixture()
            px, pe, pw = generator.inject(x, edges, torch.tensor([20]))
            self.assertTrue(torch.equal(px[:len(x)], x))
            self.assertTrue(torch.equal(pe[:, :edges.shape[1]], edges))
            self.assertEqual(int((pw == 0).sum()), 6)
            self.assertEqual(pe.shape[1], edges.shape[1] + 8)
            _, train_edges, train_weights, _ = generator._backdoor.get_poisoned()
            self.assertTrue((train_weights > 0).all())
            self.assertEqual(train_edges.shape[1], edges.shape[1] + 4)

    def test_metadata_patch_supports_background_and_isolated_nodes(self):
        with TemporaryDirectory() as work, redirect_stdout(StringIO()):
            cfg = small(selection_method="cluster_degree", epochs=2, trojan_epochs=1)
            fitted = ugba.fit_ugba(*fixture(), seed=21, work_dir=work, config=cfg)
            self.assertEqual(fitted["attachment_indices"].numel(), 2)
            self.assertEqual(fitted["record"]["configured_selector_epochs"], 2)
            runtime = fitted["record"]["runtime"]
            with ugba.load_author_modules(runtime["runtime_root"], ("heuristic_selection",)) as modules:
                hs = modules["heuristic_selection"]
                x = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
                edges = torch.tensor([[0, 1], [1, 0]])
                groups = np.array([0, 0, 1, 0])
                centers = np.array([[1., 1.], [2., 2.]])
                selected = hs.obtain_attach_nodes_by_cluster_degree_all(
                    SimpleNamespace(dis_weight=1., target_class=1), edges, groups, centers,
                    [0, 1, 3], x, 2)
                self.assertIn(3, selected.tolist())  # isolated last original node is still a candidate
            self.assertEqual(fitted["poisoned_y"][-1].item(), -1)
            self.assertTrue((Path(work) / "selected_nodes").exists())
            with self.assertRaisesRegex(ValueError, "caches"):
                ugba.fit_ugba(*fixture(), seed=21, work_dir=work, config=cfg)

    def test_author_classifier_accuracy_checkpoint_and_shared_runtime(self):
        data, cfg = fixture(), small()
        with TemporaryDirectory() as work, redirect_stdout(StringIO()):
            fitted = ugba.fit_ugba(*data, seed=21, work_dir=work, config=cfg)
            trained = ugba.fit_author_classifier(fitted["poisoned_x"], fitted["poisoned_edge_index"],
                fitted["poisoned_edge_weight"], fitted["poisoned_y"], fitted["fit_indices"], data[4],
                seed=21, work_dir=work, config=cfg)
            record = trained["record"]
            self.assertEqual(record["device"], "cpu")
            self.assertIsNone(record["rng_start"]["cuda_rng_sha256"])
            self.assertEqual(record["runtime"], fitted["record"]["runtime"])
            self.assertEqual(len(record["trace"]), cfg["epochs"])
            best = max(record["trace"], key=lambda row: row["normal_dev_accuracy"])
            self.assertEqual(record["selected_epoch"], best["epoch"])
            self.assertEqual(record["state_sha256"], best["state_sha256"])
            self.assertIn("accuracy", record["checkpoint_rule"])
            # The model itself is the source class, not a project replacement.
            self.assertEqual(trained["model"].__class__.__module__, "models.SAGE")

    def test_import_namespace_restored(self):
        sentinel = ModuleType("utils")
        with TemporaryDirectory() as work, patch.dict(sys.modules, {"utils": sentinel}):
            runtime = ugba.prepare_runtime(work)
            old_path = sys.path[:]
            with ugba.load_author_modules(runtime["runtime_root"], ("utils",)) as modules:
                self.assertIsNot(modules["utils"], sentinel)
            self.assertIs(sys.modules["utils"], sentinel)
            self.assertEqual(sys.path, old_path)

    def test_invalid_config_and_population(self):
        for cfg in ({"other": 1}, {"epochs": 0}, {"selection_method": "cluster"},
                    {"num_classes": 3}, {"dataset": "Cora"}, {"no_cuda": False}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                ugba.validate_config(cfg)
        with TemporaryDirectory() as work:
            data = list(fixture())
            data[5] = data[5][:512]
            with self.assertRaisesRegex(ValueError, "512"):
                ugba.fit_ugba(*data, seed=21, work_dir=work, config=small())
            data = list(fixture()); data[5] = torch.cat((data[5], data[4][:1]))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                ugba.fit_ugba(*data, seed=21, work_dir=work, config=small())
            data = list(fixture()); data[2][data[3][0]] = -1
            with self.assertRaisesRegex(ValueError, "valid task labels"):
                ugba.fit_ugba(*data, seed=21, work_dir=work, config=small())

    def test_explicit_device_rejection_and_cpu_compatibility(self):
        self.assertEqual(ugba._device("cpu"), torch.device("cpu"))
        for name in ("mps", "cpu:0", "invalid-device"):
            with self.subTest(device=name), self.assertRaises(ValueError):
                ugba._device(name)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "no CPU fallback"):
                ugba._device("cuda:0")
        with TemporaryDirectory() as work, redirect_stdout(StringIO()):
            fitted = ugba.fit_ugba(*fixture(), seed=21, work_dir=work,
                                   config=small(trojan_epochs=1), device="cpu")
            self.assertEqual(fitted["record"]["device"], "cpu")
            self.assertTrue(fitted["record"]["effective_author_device_args"]["no_cuda"])
            self.assertIsNone(fitted["record"]["rng_start"]["cuda_rng_sha256"])
            for key in ("attachment_indices", "poisoned_x", "poisoned_edge_index",
                        "poisoned_edge_weight", "poisoned_y", "fit_indices"):
                self.assertEqual(fitted[key].device.type, "cpu")


@unittest.skipUnless(os.environ.get("JCAS_RUN_CUDA_TESTS") == "1" and torch.cuda.is_available(),
                     "opt-in only: JCAS_RUN_CUDA_TESTS=1 with CUDA available")
class AuthorUGBACUDATests(unittest.TestCase):
    """Actual author-source GPU paths, with synthetic tensors only."""
    def test_gpu_fit_injection_and_classifier_rng_contract(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for selection in ("none", "cluster_degree"):
                with self.subTest(selection=selection), TemporaryDirectory() as work, redirect_stdout(StringIO()):
                    data, cfg = fixture(), small(selection_method=selection)
                    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
                    fitted = ugba.fit_ugba(*data, seed=21, work_dir=work, config=cfg, device="cuda:0")
                    self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                    self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(0)))
                    self.assertEqual(fitted["record"]["device"], "cuda:0")
                    self.assertFalse(fitted["record"]["effective_author_device_args"]["no_cuda"])
                    self.assertIsNotNone(fitted["record"]["rng_start"]["cuda_rng_sha256"])
                    for key in ("attachment_indices", "poisoned_x", "poisoned_edge_index",
                                "poisoned_edge_weight", "poisoned_y", "fit_indices"):
                        self.assertEqual(fitted[key].device.type, "cpu")
                    generator = fitted["generator"]
                    self.assertTrue(all(p.device.type == "cuda" for p in generator.parameters()))
                    targets = torch.tensor([20])
                    first = generator.inject(data[0], data[1], targets)
                    second = generator.inject(data[0], data[1], targets)
                    self.assertTrue(all(t.device.type == "cpu" and torch.isfinite(t).all() for t in first))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
                    self.assertTrue(all(p.device.type == "cuda" for p in generator.parameters()))
                    self.assertTrue(torch.equal(first[0][:len(data[0])], data[0]))
                    clean = ugba.fit_author_classifier(data[0], data[1], None, data[2], data[3], data[4],
                        seed=9012, work_dir=work, config=cfg, device="cuda:0")
                    victim = ugba.fit_author_classifier(fitted["poisoned_x"], fitted["poisoned_edge_index"],
                        fitted["poisoned_edge_weight"], fitted["poisoned_y"], fitted["fit_indices"], data[4],
                        seed=9012, work_dir=work, config=cfg, device="cuda:0")
                    for key in ("initialization_digest", "fit_rng_digest", "rng_start"):
                        self.assertEqual(clean["record"][key], victim["record"][key])
                    self.assertTrue(all(p.device.type == "cuda" for p in clean["model"].parameters()))
                    self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                    self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(0)))
                    self.assertEqual(len(victim["record"]["trace"]), cfg["epochs"])
                    json.dumps(fitted["record"], allow_nan=False)
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
