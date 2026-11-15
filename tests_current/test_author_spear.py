"""Actual-author-source SPEAR synthetic interface checks only."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from author_comparisons import spear


class AuthorSPEARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_native_defaults_and_verified_vendor(self):
        self.assertEqual(spear.validate_config()["alpha_int"], 30)
        self.assertEqual(spear.validate_config()["trojan_epochs"], 400)
        self.assertIsNone(spear.validate_config()["sage_n_permutations"])
        self.assertEqual(spear.verify_source()["commit"], spear.SOURCE_COMMIT)

    def test_runtime_preserves_author_algorithm_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = spear.prepare_runtime(directory)
            self.assertFalse(manifest["algorithms_reimplemented"])
            for name in ("select_sample.py", "models/GCN.py", "models/SAGE.py", "sage_modified/permutation_estimator.py", "sage_modified/imputers.py"):
                self.assertEqual((path / name).read_bytes(), (spear.SOURCE_ROOT / name).read_bytes())
            for name in ("models/MLP.py",):
                original = ast.parse((spear.SOURCE_ROOT / name).read_text())
                runtime = ast.parse((path / name).read_text())
                original_class = next(node for node in original.body if isinstance(node, ast.ClassDef))
                runtime_class = next(node for node in runtime.body if isinstance(node, ast.ClassDef))
                self.assertEqual(ast.dump(original_class), ast.dump(runtime_class))
            pristine = (spear.SOURCE_ROOT / "models/backdoor_GCN_cos.py").read_text()
            runtime = (path / "models/backdoor_GCN_cos.py").read_text()
            self.assertEqual(runtime.replace("getattr(args, 'feature_importance_path', None) or ", ""), pristine)

    def test_direct_author_fit_inject_and_classifier(self):
        self._exercise_author_fit("cpu")

    def test_cuda_request_never_silently_falls_back(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "fallback is prohibited"):
                spear._device("cuda:0")
        with self.assertRaises(ValueError):
            spear._device("meta")

    @unittest.skipUnless(os.environ.get("RUN_AUTHOR_CUDA_TESTS") == "1" and torch.cuda.is_available(),
                         "opt-in GPU smoke: RUN_AUTHOR_CUDA_TESTS=1 with CUDA visible")
    def test_cuda_author_fit_cpu_input_output_rng_and_native_models(self):
        self._exercise_author_fit("cuda:0")

    def _exercise_author_fit(self, device):
        generator = torch.Generator().manual_seed(719)
        x = torch.randn(16, 6, generator=generator)
        edges = torch.tensor([(i, j) for i in range(16) for j in range(16) if abs(i - j) == 1], dtype=torch.long).T
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1, -1])
        train, dev, candidates = torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7]), torch.arange(8, 14)
        config = dict(epochs=3, trojan_epochs=2, hidden=4, alpha_int=2, vs_number=2,
                      outter_size=2, sample_num=2, sage_batch_size=4, sage_n_permutations=8,
                      sage_detect_convergence=False, debug=False)
        original = [v.clone() for v in (x, edges, labels)]
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
        precision_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            result = spear.fit_spear(x, edges, labels, train, dev, candidates, seed=3, work_dir=directory, config=config, device=device)
            self.assertFalse(result["record"]["algorithms_reimplemented"])
            self.assertEqual(result["record"]["device"], device)
            self.assertEqual(type(result["generator"].backdoor).__module__, "models.backdoor_GCN_cos")
            self.assertTrue(all(p.device == torch.device(device) for p in result["generator"].parameters()))
            for key in ("attachment_indices", "poisoned_x", "poisoned_y", "poisoned_edge_index", "poisoned_edge_weight", "fit_indices"):
                self.assertEqual(result[key].device.type, "cpu", key)
            injected, injected_edges, weights = result["generator"].inject(x, edges, result["attachment_indices"])
            tolerance = dict(rtol=0, atol=0) if device == "cpu" else dict(rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(injected, result["poisoned_x"], **tolerance)
            self.assertEqual(injected.device.type, "cpu")
            self.assertTrue(all(p.device == torch.device(device) for p in result["generator"].parameters()))
            torch.testing.assert_close(injected_edges, edges, rtol=0, atol=0)
            self.assertTrue(torch.equal(weights, torch.ones(edges.shape[1])))
            self.assertTrue(torch.equal(result["poisoned_y"][-2:], labels[-2:]))
            victim = spear.fit_author_classifier(result["poisoned_x"], edges, weights, result["poisoned_y"], result["fit_indices"], dev,
                seed=3, work_dir=directory, config=config, device=device)
            self.assertEqual(type(victim["model"]).__module__, "models.SAGE")
            self.assertIn("DEV accuracy", victim["record"]["checkpoint"])
            self.assertEqual(victim["record"]["device"], device)
            self.assertTrue(all(p.device == torch.device(device) for p in victim["model"].parameters()))
            participating = [p.grad for p in victim["model"].parameters() if p.grad is not None]
            self.assertTrue(participating)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in participating))
            with torch.no_grad():
                prediction = victim["model"](x.to(device), edges.to(device), weights.to(device))
            self.assertTrue(torch.isfinite(prediction).all())
            with self.assertRaises(FileExistsError):
                spear.fit_spear(x, edges, labels, train, dev, candidates, seed=3, work_dir=directory, config=config, device=device)
        for actual, before in zip((x, edges, labels), original):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        if cuda_rng is not None:
            torch.testing.assert_close(torch.cuda.get_rng_state(device), cuda_rng, rtol=0, atol=0)
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), precision_flags)


if __name__ == "__main__":
    unittest.main()
