"""Bounded, clean author-GraphSAGE GPU throughput/resource acceptance.

Only the retained 512 TRAIN + 128 normal-DEV scenes within original TRAIN are
read. Native author GraphSAGE runs FP32 forward/backward on TRAIN labels; no
generator, poisoning, ASR, validation scoring, or trained checkpoint is used.
This is not evidence that a full generator fit will fit or scale equally well.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback

import torch
from torch.nn import functional as F

from spear_current.runtime import load_bundle, sha256
from .data import input_hashes, load_inputs
from . import ugba


ROOT = Path(__file__).resolve().parents[1]
PRECISION = "float32_no_amp_no_tf32"
VERSION = "author_graphsage_gpu_resource_v1"
SPEEDUP_THRESHOLD = 1.10
WARMUP_STEPS = 2
TIMED_STEPS = 8
TIMEOUT_SECONDS = 600
MAX_START_SKEW_SECONDS = .05


def write_new_json(path, value):
    """Publish complete JSON atomically, still refusing any existing target."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic, non-overwriting same-filesystem publication
    finally:
        if temporary is not None:
            temporary.unlink()


def result_directory(path):
    path = Path(path).resolve()
    if path == ROOT / "results" or not path.is_relative_to(ROOT / "results"):
        raise ValueError("Require a new child of JCAS/results")
    return path


def source_bindings():
    paths = ("author_comparisons/gpu_benchmark.py", "author_comparisons/data.py",
             "author_comparisons/ugba.py")
    return {"local_code": {p: sha256(ROOT / p) for p in paths},
            "author": ugba.verify_source()}


def finite_gradients(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return bool(params) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)


def summarize_workers(records, expected_workers, timed_steps=TIMED_STEPS):
    """Synchronized aggregate throughput, not a sum of individual throughputs."""
    if len(records) != expected_workers or expected_workers not in (1, 2):
        raise ValueError("Missing or unsupported worker count")
    required = ("measurement_started_monotonic", "measurement_finished_monotonic",
                "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "cpu_max_rss_bytes")
    for row in records:
        if (row.get("status") != "GPU_BENCHMARK_WORKER_COMPLETE"
                or row.get("timed_steps") != timed_steps
                or row.get("precision") != PRECISION or row.get("finite_loss_and_gradients") is not True
                or any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) for k in required)
                or row["measurement_finished_monotonic"] <= row["measurement_started_monotonic"]
                or row["elapsed_seconds"] <= 0 or any(row[k] < 0 for k in required[3:])):
            raise ValueError("Invalid or nonfinite GPU worker measurement")
    start = min(r["measurement_started_monotonic"] for r in records)
    finish = max(r["measurement_finished_monotonic"] for r in records)
    skew = max(r["measurement_started_monotonic"] for r in records) - start
    overlap = min(r["measurement_finished_monotonic"] for r in records) - max(r["measurement_started_monotonic"] for r in records)
    if expected_workers == 2 and (skew > MAX_START_SKEW_SECONDS or overlap <= 0):
        raise ValueError("Two-worker interval not synchronized/overlapping")
    elapsed = finish - start
    return {"status": "GPU_BENCHMARK_STAGE_COMPLETE", "workers": expected_workers,
            "timed_steps_per_worker": timed_steps, "elapsed_seconds": elapsed,
            "aggregate_steps_per_second": expected_workers * timed_steps / elapsed,
            "start_skew_seconds": skew, "common_overlap_seconds": overlap,
            "sum_process_peak_allocated_bytes": sum(r["peak_allocated_bytes"] for r in records),
            "sum_process_peak_reserved_bytes": sum(r["peak_reserved_bytes"] for r in records),
            "worker_results": records}


def concurrency_decision(single, dual):
    if single.get("status") != "GPU_BENCHMARK_STAGE_COMPLETE" or single.get("workers") != 1:
        raise ValueError("A successful single-worker acceptance is required")
    speed = single.get("aggregate_steps_per_second")
    if not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
        raise ValueError("Invalid single-worker throughput")
    if dual.get("status") != "GPU_BENCHMARK_STAGE_COMPLETE" or dual.get("workers") != 2:
        return {"recommended_max_parallel": 1, "parallel_speedup": None,
                "decision_reason": "two-worker run failed; no concurrency permission"}
    other = dual.get("aggregate_steps_per_second")
    if not isinstance(other, (int, float)) or not math.isfinite(other) or other <= 0:
        raise ValueError("Invalid two-worker throughput")
    ratio = other / speed
    allowed = ratio >= SPEEDUP_THRESHOLD
    return {"recommended_max_parallel": 2 if allowed else 1, "parallel_speedup": ratio,
            "decision_reason": ("two-worker aggregate throughput meets preregistered 1.10x threshold"
                                if allowed else "two-worker aggregate throughput below 1.10x threshold")}


def validate_report(path):
    """Read-only bindings/decision validation; never loads graph tensor arrays."""
    path = Path(path).resolve()
    result_directory(path.parent)
    if path.name != "summary.json":
        raise ValueError("Require benchmark summary.json")
    report = json.loads(path.read_text())
    if (report.get("status") != "GPU_RESOURCE_BENCHMARK_COMPLETE" or report.get("version") != VERSION
            or report.get("precision") != PRECISION or report.get("parallel_threshold") != SPEEDUP_THRESHOLD
            or report.get("dpgba_exclusive") is not True or report.get("source_bindings") != source_bindings()):
        raise ValueError("Benchmark status, policy or source binding changed")
    for name, key in (("plan.json", "plan_sha256"), ("clean_graph_inputs.pt", "input_snapshot_sha256"),
                      ("graph_access_ledger.json", "graph_access_ledger_sha256")):
        if sha256(path.parent / name) != report.get(key):
            raise ValueError("Benchmark artifact hash changed: " + name)
    if sha256(Path(report["manifest"]["path"])) != report["manifest"]["sha256"]:
        raise ValueError("Benchmark dataset manifest changed")
    plan = json.loads((path.parent / "plan.json").read_text())
    if (plan["source_bindings"] != report["source_bindings"] or plan["manifest"] != report["manifest"]
            or plan["device"] != report["device"] or plan["precision"] != PRECISION
            or plan["warmup_steps"] != WARMUP_STEPS or plan["timed_steps"] != TIMED_STEPS
            or plan["parallel_threshold"] != SPEEDUP_THRESHOLD or plan["sequence"] != [1, 2]):
        raise ValueError("Benchmark registration and summary disagree")
    ledger = json.loads((path.parent / "graph_access_ledger.json").read_text())
    if (len(ledger) != 640 or sum(r["role"] == "train" for r in ledger) != 512
            or sum(r["role"] == "dev" for r in ledger) != 128
            or any(r["split"] != "train" or r["status"] != "READ_COMPLETE" for r in ledger)):
        raise ValueError("Benchmark graph access outside registered original-TRAIN population")
    runs = []
    for name, count in (("single", 1), ("parallel2", 2)):
        stage = json.loads((path.parent / name / "summary.json").read_text())
        if stage["status"] == "GPU_BENCHMARK_STAGE_COMPLETE":
            records = [json.loads((path.parent / name / f"worker{i}/result.json").read_text()) for i in range(count)]
            if stage != summarize_workers(records, count):
                raise ValueError("Benchmark worker results and stage summary disagree")
        runs.append(stage)
    if runs != report["runs"]:
        raise ValueError("Benchmark stage results changed")
    decision = concurrency_decision(*runs)
    if any(report.get(k) != v for k, v in decision.items()):
        raise ValueError("Benchmark recommendation does not follow registered threshold")
    return report


def _precision(device):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def worker(plan_path, worker_id):
    plan_path = Path(plan_path).resolve()
    stage = result_directory(plan_path.parent)
    plan = json.loads(plan_path.read_text())
    if (plan["version"] != VERSION or os.getppid() != plan["orchestrator_pid"]
            or worker_id not in range(plan["workers"])):
        raise ValueError("Benchmark worker must be launched by its registered orchestrator")
    output = stage / f"worker{worker_id}"
    output.mkdir(exist_ok=False)
    try:
        snapshot = (stage.parent / "clean_graph_inputs.pt").resolve()
        if sha256(snapshot) != plan["input_snapshot_sha256"] or source_bindings() != plan["source_bindings"]:
            raise ValueError("Benchmark data or author source changed")
        data = torch.load(snapshot, weights_only=True, map_location="cpu")
        device = ugba._device(plan["device"])
        if device.type != "cuda":
            raise ValueError("GPU benchmark cannot run or silently fall back on CPU")
        _precision(device)
        torch.random.default_generator.manual_seed(plan["seed"] + worker_id)
        torch.cuda.manual_seed(plan["seed"] + worker_id)
        runtime = ugba.prepare_runtime(output / "source_runtime")
        x, edges, y, train = (data[k].to(device) for k in ("x", "edge_index", "y", "idx_train"))
        if x.dtype != torch.float32 or not torch.isin(y[train], torch.tensor([0, 1], device=device)).all():
            raise ValueError("Invalid benchmark TRAIN labels or FP32 input")
        with ugba.load_author_modules(runtime["runtime_root"], ("models.SAGE",)) as modules:
            cfg = ugba.DEFAULT_CONFIG
            model = modules["models.SAGE"].GraphSage(nfeat=x.shape[1], nhid=32, nclass=2,
                dropout=cfg["dropout"], lr=cfg["train_lr"], weight_decay=cfg["weight_decay"], device=device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.weight_decay)

            def step():
                model.train()
                optimizer.zero_grad()
                logits = model(x, edges, None)
                loss = F.nll_loss(logits[train], y[train])
                loss.backward()
                if not bool(torch.isfinite(loss)) or not finite_gradients(model):
                    raise ValueError("Nonfinite benchmark loss/gradients")
                optimizer.step()
                return float(loss.detach())

            for _ in range(plan["warmup_steps"]):
                step()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            write_new_json(output / "ready.json", {"ready": True, "pid": os.getpid(),
                "plan_sha256": sha256(plan_path), "device": str(device), "warmup_steps": plan["warmup_steps"]})
            deadline = time.monotonic() + plan["timeout_seconds"]
            while not (stage / "start.json").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Waiting for synchronized benchmark release timed out")
                time.sleep(.01)
            release = json.loads((stage / "start.json").read_text())
            if release["plan_sha256"] != sha256(plan_path):
                raise ValueError("Changed benchmark start barrier")
            while time.monotonic() < release["start_monotonic"]:
                time.sleep(min(.005, max(0., release["start_monotonic"] - time.monotonic())))
            started = time.monotonic()
            losses = [step() for _ in range(plan["timed_steps"])]
            torch.cuda.synchronize(device)
            finished = time.monotonic()
            result = {"status": "GPU_BENCHMARK_WORKER_COMPLETE", "worker_id": worker_id,
                "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__, "precision": PRECISION,
                "native_class": "UGBA models.SAGE.GraphSage", "hidden": 32,
                "seed": plan["seed"] + worker_id, "warmup_steps": plan["warmup_steps"],
                "timed_steps": plan["timed_steps"], "measurement_started_monotonic": started,
                "measurement_finished_monotonic": finished, "elapsed_seconds": finished - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "finite_loss_and_gradients": True, "losses": losses,
                "runtime": runtime, "plan_sha256": sha256(plan_path)}
            write_new_json(output / "result.json", result)
    except BaseException as error:
        write_new_json(output / "failure.json", {"error": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()})
        raise


def run_stage(output, name, workers, common):
    stage = output / name
    stage.mkdir(exist_ok=False)
    plan = dict(common, workers=workers, orchestrator_pid=os.getpid())
    write_new_json(stage / "plan.json", plan)
    procs, logs = [], []
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        for i in range(workers):
            log = (stage / f"worker{i}.log").open("x")
            logs.append(log)
            procs.append(subprocess.Popen([sys.executable, "-B", "-m", "author_comparisons.gpu_benchmark",
                "--worker-plan", str(stage / "plan.json"), "--worker-id", str(i)],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + plan["timeout_seconds"]
        while not all((stage / f"worker{i}/ready.json").is_file() for i in range(workers)):
            if any(p.poll() is not None for p in procs):
                raise RuntimeError("Benchmark worker exited before readiness barrier; inspect worker logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Benchmark warmup/readiness timeout")
            time.sleep(.05)
        write_new_json(stage / "start.json", {"plan_sha256": sha256(stage / "plan.json"),
                                              "start_monotonic": time.monotonic() + .25})
        deadline = time.monotonic() + plan["timeout_seconds"]
        while any(p.poll() is None for p in procs):
            if any(p.poll() not in (None, 0) for p in procs):
                raise RuntimeError("Benchmark worker failed during timed interval; inspect worker logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Benchmark timed interval timeout")
            time.sleep(.05)
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError("Benchmark worker exit failure")
        records = [json.loads((stage / f"worker{i}/result.json").read_text()) for i in range(workers)]
        result = summarize_workers(records, workers, plan["timed_steps"])
    except BaseException as error:
        result = {"status": "GPU_BENCHMARK_STAGE_FAILED", "workers": workers,
                  "error": str(error), "type": type(error).__name__}
    finally:
        # Only exact children created here are stopped; other users' processes are untouched.
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        for log in logs:
            log.close()
    write_new_json(stage / "summary.json", result)
    return result


def run(output, device="cuda:0", manifest=None):
    device = ugba._device(device)
    if device.type != "cuda":
        raise ValueError("This acceptance requires CUDA; no CPU fallback")
    output = result_directory(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_bundle(manifest)
    binding = source_bindings()
    manifest_binding = {"path": str(bundle.manifest_path), "sha256": sha256(bundle.manifest_path)}
    plan = {"version": VERSION, "device": str(device), "source_bindings": binding,
            "manifest": manifest_binding, "precision": PRECISION, "seed": 20261029,
            "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
            "timeout_seconds": TIMEOUT_SECONDS, "parallel_threshold": SPEEDUP_THRESHOLD,
            "maximum_start_skew_seconds": MAX_START_SKEW_SECONDS,
            "sequence": [1, 2], "dpgba_exclusive": True,
            "train_scenes": 512, "normal_dev_scenes": 128,
            "optimization_labels": "TRAIN only; normal DEV graph context present but no DEV labels used",
            "no_generator_optimization": True, "no_asr_or_validation_scoring": True}
    write_new_json(output / "plan.json", plan)
    ledger = []
    try:
        data = load_inputs(bundle, ledger)
        write_new_json(output / "graph_access_ledger.json", ledger)
        hashes = input_hashes(data)
        snapshot = {key: data[key] for key in ("x", "edge_index", "y", "idx_train")}
        with (output / "clean_graph_inputs.pt").open("xb") as stream:
            torch.save(snapshot, stream)
        snapshot_hash = sha256(output / "clean_graph_inputs.pt")
        common = dict(plan, input_snapshot_sha256=snapshot_hash)
        single = run_stage(output, "single", 1, common)
        if single["status"] != "GPU_BENCHMARK_STAGE_COMPLETE":
            raise RuntimeError("Single-worker GPU benchmark failed; no queue permission")
        dual = run_stage(output, "parallel2", 2, common)
        decision = concurrency_decision(single, dual)
        if binding != source_bindings() or manifest_binding["sha256"] != sha256(bundle.manifest_path):
            raise ValueError("Bound source or manifest changed during benchmark")
        report = {"status": "GPU_RESOURCE_BENCHMARK_COMPLETE", "version": VERSION,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__, "precision": PRECISION,
            **decision, "parallel_threshold": SPEEDUP_THRESHOLD, "dpgba_exclusive": True,
            "source_bindings": binding, "manifest": manifest_binding, "input_hashes": hashes,
            "input_snapshot_sha256": snapshot_hash, "plan_sha256": sha256(output / "plan.json"),
            "graph_access_ledger_sha256": sha256(output / "graph_access_ledger.json"),
            "graph_arrays_loaded": 640, "all_graph_arrays_original_split": "train",
            "original_test_arrays_loaded": False, "original_validation_arrays_loaded": False,
            "audit_arrays_loaded": False, "confirmation_arrays_loaded": False,
            "trained_checkpoints_loaded": False, "generator_optimized": False, "asr_evaluated": False,
            "runs": [single, dual], "limitations": [
                "Short clean author-GraphSAGE hidden32 measurement, not full generator memory/throughput proof",
                "Other users' GPU/memory pressure may change; stop on OOM, do not relax budgets",
                "DPGBA remains exclusive regardless of recommendation; hidden512 generator not benchmarked",
                "Per-process peak sums are conservative sums, not simultaneous observed device memory",
                "No new model checkpoint retained; transient benchmark weights are discarded"]}
        write_new_json(output / "summary.json", report)
        print(json.dumps({k: report[k] for k in ("status", "recommended_max_parallel", "parallel_speedup")}), flush=True)
        return report
    except BaseException as error:
        if not (output / "graph_access_ledger.json").exists():
            write_new_json(output / "graph_access_ledger.json", ledger)
        write_new_json(output / "failure.json", {"status": "GPU_RESOURCE_BENCHMARK_FAILED",
            "error": str(error), "type": type(error).__name__, "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan is not None:
        if args.worker_id is None or args.output_dir is not None or args.manifest is not None:
            parser.error("Internal worker invocation only")
        worker(args.worker_plan, args.worker_id)
    else:
        if args.output_dir is None or args.worker_id is not None:
            parser.error("--output-dir is required")
        run(args.output_dir, args.device, args.manifest)


if __name__ == "__main__":
    main()
