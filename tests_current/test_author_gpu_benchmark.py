"""CPU-only acceptance-policy tests; no real data or CUDA work."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from author_comparisons import gpu_benchmark as bench


def record(start, finish, worker_id=0):
    return {"status": "GPU_BENCHMARK_WORKER_COMPLETE", "worker_id": worker_id,
        "precision": bench.PRECISION, "timed_steps": bench.TIMED_STEPS,
        "finite_loss_and_gradients": True, "measurement_started_monotonic": start,
        "measurement_finished_monotonic": finish, "elapsed_seconds": finish - start,
        "peak_allocated_bytes": 100, "peak_reserved_bytes": 200, "cpu_max_rss_bytes": 500}


class GPUResourcePolicyTests(unittest.TestCase):
    def test_aggregate_throughput_uses_joint_interval(self):
        result = bench.summarize_workers([record(10., 12.), record(10.01, 12.5, 1)], 2)
        self.assertAlmostEqual(result["elapsed_seconds"], 2.5)
        self.assertAlmostEqual(result["aggregate_steps_per_second"], 16 / 2.5)
        self.assertAlmostEqual(result["start_skew_seconds"], .01)
        self.assertEqual(result["sum_process_peak_reserved_bytes"], 400)

    def test_concurrency_requires_110_percent_total_throughput(self):
        single = bench.summarize_workers([record(10., 11.)], 1)
        for dual_duration, expected in ((2., 1), (1.9, 1), (1.8, 2), (1., 2)):
            dual = bench.summarize_workers([record(10., 10. + dual_duration), record(10., 10. + dual_duration, 1)], 2)
            self.assertEqual(bench.concurrency_decision(single, dual)["recommended_max_parallel"], expected)

    def test_failed_dual_never_grants_concurrency(self):
        single = bench.summarize_workers([record(1., 2.)], 1)
        decision = bench.concurrency_decision(single, {"status": "GPU_BENCHMARK_STAGE_FAILED", "workers": 2})
        self.assertEqual(decision["recommended_max_parallel"], 1)
        self.assertIsNone(decision["parallel_speedup"])
        with self.assertRaisesRegex(ValueError, "successful single"):
            bench.concurrency_decision({"status": "GPU_BENCHMARK_STAGE_FAILED"}, single)

    def test_invalid_measurements_fail_closed(self):
        for field, value in (("finite_loss_and_gradients", False), ("precision", "amp"),
                             ("timed_steps", 7), ("peak_allocated_bytes", -1),
                             ("elapsed_seconds", float("nan")), ("measurement_finished_monotonic", 0.)):
            bad = record(1., 2.); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bench.summarize_workers([bad], 1)
        with self.assertRaises(ValueError):
            bench.summarize_workers([record(1., 2.)], 2)
        for second in (record(1.1, 2.1, 1), record(3., 4., 1)):
            with self.assertRaisesRegex(ValueError, "synchronized/overlapping"):
                bench.summarize_workers([record(1., 2.), second], 2)

    def test_finite_gradient_check_is_real(self):
        model = torch.nn.Linear(2, 2)
        self.assertFalse(bench.finite_gradients(model))
        model(torch.ones(1, 2)).sum().backward()
        self.assertTrue(bench.finite_gradients(model))
        model.weight.grad[0, 0] = float("nan")
        self.assertFalse(bench.finite_gradients(model))

    def test_new_outputs_only_and_scoped_directory(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            bench.write_new_json(path, {"status": "synthetic"})
            self.assertEqual(json.loads(path.read_text()), {"status": "synthetic"})
            with self.assertRaises(FileExistsError):
                bench.write_new_json(path, {})
            self.assertEqual(list(Path(temp).iterdir()), [path])
        for path in (bench.ROOT, bench.ROOT / "results", bench.ROOT / "spear_current"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bench.result_directory(path)

    def test_cpu_does_not_start_a_benchmark(self):
        with patch.object(bench, "result_directory") as directory:
            with self.assertRaisesRegex(ValueError, "no CPU fallback"):
                bench.run("unused", device="cpu")
            directory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
