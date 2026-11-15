"""Five-seed corrected WOMD run on the registered full training_20s corpus.

All BUILT WOMD training_20s graph records are assigned once to a deterministic
source-shard-disjoint 60/20/20 TRAIN/DEV/AUDIT split.  This runner shares only
immutable graph/cache assets and code; learned state is fresh per seed.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20261126, 20261127, 20261128, 20261129, 20261130]
STAGES = ["base", "residual", "support_surrogate", "continuation_a", "continuation_b"]
DATA_CACHE = Path("/home/mizhou/JCAS_vehicle_lane/artifacts/womd_full_corpus_cache_v1_20261111/data")
GRAPH_ROOT = Path("/home/mizhou/JCAS/graphs/womd_motion_v1_3_1_graph_v1_20261104")
AUTHOR_SOURCE = Path("/home/mizhou/JCAS_vehicle_lane/imports/JCAS_20261107_162453/project/third_party/author_baselines/SPEAR")


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_rng_snapshot(folder, stage, boundary):
    # Use the corrected AV2 helper so the WOMD receipt has the same evidence.
    from .av2_corrected import _save_rng_snapshot as save
    return save(folder, stage, boundary)


def _link_cache(destination):
    destination.mkdir(parents=True, exist_ok=False)
    for source in DATA_CACHE.resolve().iterdir():
        target = destination / source.name
        if source.name == "summary.json":
            continue
        target.symlink_to(source, target_is_directory=source.is_dir())


def _summary(destination, identity):
    summary = copy.deepcopy(read(DATA_CACHE / "summary.json"))
    summary.update(
        schema="womd_formal_main_corrected_fullcorpus_streaming_data_summary_v2",
        experiment_identity=identity,
        dataset_id="WOMD",
        source_dataset_split="training_20s",
        source_graph_root=str(GRAPH_ROOT),
        source_cache=str(DATA_CACHE),
        cache_reused_read_only=True,
        evaluation_scope="WOMD training_20s full BUILT corpus with source-shard-disjoint internal DEV/AUDIT; not official validation/test",
        recording_independence_proven=False,
        normalization_contract=dict(
            fit_role="train",
            fit_scope="all 7,358,556 registered full-corpus TRAIN pair nodes",
            statistics="population mean/std fitted on TRAIN only; minimum scale 1e-6",
            clip=[-10, 10],
            dev_audit_reuse_train_statistics=True,
            refit_for_this_run=False,
        ),
        corrected_protocol=dict(
            clean_supervision="20 percent of 3,268,349 eligible TRAIN pairs per seed",
            clean_supervision_count=653669,
            attachment_count=8760,
            victim_supervision_count=662429,
            attachment_fraction="8760/(653669+8760)=1.32240587%",
            stages=STAGES,
            target_pool="one fixed hash-selected eligible-negative target identity per DEV/AUDIT scene",
            official_validation_loaded=False,
            official_test_loaded=False,
        ),
    )
    write(destination / "summary.json", summary)


def _snapshot(output):
    snapshot = output / "source_snapshot"
    shutil.copytree(
        "/home/mizhou/JCAS/spear_current",
        snapshot / "spear_current",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if not AUTHOR_SOURCE.is_dir():
        raise FileNotFoundError(f"author SPEAR source missing: {AUTHOR_SOURCE}")
    shutil.copytree(
        AUTHOR_SOURCE,
        snapshot / "third_party/author_baselines/SPEAR",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(ROOT / "src/av2_corrected.py", snapshot / "spear_current/av2_corrected.py")
    shutil.copyfile(ROOT / "src/av2_gpu_cache.py", snapshot / "spear_current/av2_gpu_cache.py")
    shutil.copyfile(Path(__file__), snapshot / "spear_current/womd_full_corrected_v2.py")
    shutil.copyfile(ROOT / "src/womd_full_data.py", snapshot / "spear_current/womd_full_data.py")
    for source in sorted((ROOT / "tests_current").glob("test_av2*.py")):
        target = snapshot / "tests_current" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return snapshot


def register(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("WOMD corrected output already exists")
    if not (DATA_CACHE / "summary.json").exists():
        raise FileNotFoundError("Registered WOMD full-corpus data cache is missing")
    output.mkdir(parents=True)
    data = output / "data"
    _link_cache(data)
    _summary(data, output.name)
    summary = read(data / "summary.json")
    snapshot = _snapshot(output)
    source_hashes = {
        str(path.relative_to(snapshot)): sha(path)
        for path in sorted(snapshot.rglob("*.py"))
    }
    expected = {"train": {"scenes": 31337, "groups": 600, "eligible": 3268349}, "dev": {"scenes": 10586, "groups": 200, "targets": 10163}, "audit": {"scenes": 10572, "groups": 200, "targets": 10172}}
    for role, checks in expected.items():
        for key, value in checks.items():
            if summary[role][key] != value:
                raise ValueError(f"WOMD full role count changed: {role}.{key}")
    cache_contract = dict(
        enabled=False,
        mode="streaming_complete_scene_batches",
        roles=[],
        cache_device=None,
        pin_memory=False,
        non_blocking=False,
        preserves_scene_order=True,
        preserves_max_nodes=True,
        preserves_topology=True,
        expected_numeric_cache_budget_bytes=0,
        required_free_headroom_bytes=8 * 1024**3,
    )
    plan = dict(
        schema="womd_formal_main_corrected_fullcorpus_streaming_v2",
        experiment_identity=output.name,
        registered_at=now(),
        output_dir=str(output),
        data_dir=str(data),
        data_summary_path=str(output / "data_summary.json"),
        source_snapshot=str(snapshot),
        source_sha256=source_hashes,
        data_summary=summary,
        data_summary_sha256=sha(data / "summary.json"),
        graph_root=str(GRAPH_ROOT),
        graph_manifest_sha256=sha(GRAPH_ROOT / "manifest.csv"),
        split_path=str(DATA_CACHE.parent / "split.json"),
        split_sha256=sha(DATA_CACHE.parent / "split.json"),
        source_run="womd_full_corpus_split_candidate_20261111",
        prior_pilot_excluded=True,
        seeds=SEEDS,
        device="cuda:0",
        dtype="float32",
        amp=False,
        tf32=False,
        max_nodes=16384,
        minimum_cuda_free_bytes=8 * 1024**3,
        gpu_batch_cache=cache_contract,
        execution_note="Full-corpus scale uses complete-scene CPU streaming with every model operation on CUDA; resident all-role cache is disabled after measured headroom failure.",
        attachment_count=8760,
        clean_pairs=653669,
        victim_pairs=662429,
        stages=STAGES,
        stage_epochs={stage: 200 for stage in STAGES},
        generator_updates=800,
        shadow_updates=1000,
        support_selector_updates=200,
        source_selection=dict(
            attachment_count=8760,
            outer_extra_count=512,
            model_epochs=200,
            sage_n_permutations=2048,
            sage_batch_size=64,
            explanation_count=128,
            background_count=32,
        ),
        training=dict(clean_proxy_epochs=200, clean_reference_epochs=200, victim_epochs=200, inner_steps=1),
        support_policy="fresh top30 per seed; exact legal24 hardening; no ASR/AUDIT selection",
        checkpoint_policy=dict(generator="final epoch each stage", victim="earliest strict maximum clean DEV AUC", audit_used=False),
        data_contract=dict(
            dataset_id="WOMD",
            source_dataset_split="training_20s",
            feature_dim=784,
            original_topology=True,
            original_labels=True,
            train_only_normalizer=True,
            normalizer_clip=[-10, 10],
            train_groups=600,
            dev_groups=200,
            audit_groups=200,
            group_key="source_shard",
            group_overlap=dict(train_dev=0, train_audit=0, dev_audit=0),
            official_test_loaded=False,
            recording_independence_proven=False,
        ),
        execution=dict(
            cuda_only=True,
            fp32=True,
            amp=False,
            tf32=False,
            whole_scene_batches=True,
            neighbor_sampling=False,
            scene_split=False,
            automatic_retries=0,
            replace_seed=False,
            stop_on_asr=False,
        ),
        evaluation=dict(
            dev_scenes=10586,
            audit_scenes=10572,
            dev_targets=10163,
            audit_targets=10172,
            blind=False,
            official_test_loaded=False,
            fixed_target_pool=True,
        ),
        scope="WOMD training_20s full BUILT corpus; deterministic 600/200/200 source-shard split; five fresh complete corrected-method seeds; official validation/test excluded",
    )
    write(output / "data_summary.json", summary)
    plan["data_summary_path_sha256"] = sha(output / "data_summary.json")
    write(output / "plan.json", plan)
    plan_hash = sha(output / "plan.json")
    children = {}
    for seed in SEEDS:
        folder = output / f"seed{seed}"
        folder.mkdir()
        child = dict(
            schema=plan["schema"] + "_seed",
            seed=seed,
            output_dir=str(folder),
            parent_plan=str(output / "plan.json"),
            parent_plan_sha256=plan_hash,
            data_dir=str(data),
            data_summary_sha256=plan["data_summary_sha256"],
            source_snapshot=str(snapshot),
            source_sha256=source_hashes,
            device="cuda:0",
            dtype="float32",
            amp=False,
            tf32=False,
            max_nodes=16384,
            stages=STAGES,
            stage_epochs=plan["stage_epochs"],
            attachment_count=8760,
            clean_pairs=653669,
            victim_pairs=662429,
        )
        write(folder / "plan.json", child)
        children[str(seed)] = sha(folder / "plan.json")
    write(output / "registration.json", dict(
        plan_sha256=plan_hash,
        child_plan_sha256=children,
        seeds=SEEDS,
        experiment_identity=output.name,
        prior_pilot_excluded=True,
        at=now(),
    ))
    write(output / "status.json", dict(status="REGISTERED", at=now(), completed=0, failed=0, total=5, training_started=False))
    return output / "plan.json"


def verify(plan_path, frozen=False):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    root = Path(plan["output_dir"])
    registration = read(root / "registration.json")
    if sha(plan_path) != registration["plan_sha256"] or plan["seeds"] != SEEDS:
        raise ValueError("WOMD plan/seed binding changed")
    if sha(root / "data/summary.json") != plan["data_summary_sha256"]:
        raise ValueError("WOMD data summary changed")
    if sha(plan["data_summary_path"]) != plan["data_summary_path_sha256"]:
        raise ValueError("WOMD independent data summary changed")
    data = Path(plan["data_dir"])
    for name, digest in plan["data_summary"].get("asset_sha256", {}).items():
        if sha(data / name) != digest:
            raise ValueError("Registered WOMD data asset changed: " + name)
    if sha(Path(plan["graph_root"]) / "manifest.csv") != plan["graph_manifest_sha256"]:
        raise ValueError("WOMD graph manifest changed")
    if sha(Path(plan["split_path"])) != plan["split_sha256"]:
        raise ValueError("WOMD split assignment changed")
    for relative, digest in plan["source_sha256"].items():
        if sha(Path(plan["source_snapshot"]) / relative) != digest:
            raise ValueError("WOMD source changed: " + relative)
    if frozen and Path(__file__).resolve() != Path(plan["source_snapshot"]) / "spear_current/womd_full_corrected_v2.py":
        raise RuntimeError("must run frozen WOMD source")
    for seed in SEEDS:
        folder = root / f"seed{seed}"
        child = read(folder / "plan.json")
        if sha(folder / "plan.json") != registration["child_plan_sha256"][str(seed)]:
            raise ValueError("WOMD child plan changed")
        if child["parent_plan_sha256"] != sha(plan_path) or child["seed"] != seed:
            raise ValueError("WOMD child binding changed")
    return plan


def preflight(plan_path):
    # The corrected AV2 probe is data-adapter agnostic. Bind it to this plan
    # only for the call, then restore the module global.
    from . import av2_corrected as base
    original = base.verify
    base.verify = verify
    try:
        return base.preflight(plan_path)
    finally:
        base.verify = original


def _record_rng(folder, hashes, stage, boundary):
    digest = _save_rng_snapshot(folder, stage, boundary)
    hashes[f"rng_states/{stage}_{boundary}.pt"] = digest
    write(folder / "rng_states/manifest.json", hashes)


def worker(plan_path, seed):
    plan = verify(plan_path, frozen=True)
    if seed not in SEEDS:
        raise ValueError("unregistered WOMD seed")
    import torch
    from types import SimpleNamespace
    from .av2_corrected import _continuations, _augment_eval_summary, _enable_gpu_batch_cache
    _enable_gpu_batch_cache(plan)
    from .av2_formal_data import make_indices, load_role_arrays
    from .av2_formal_selection import fit_source_selection
    from .av2_formal_classifiers import fit_classifier, fit_victim, evaluate
    from .av2_formal_generator import fit_generator, fit_support
    from .full_ablation_training import _state_hash, _tensor_hash
    root = Path(plan["output_dir"])
    folder = root / f"seed{seed}"
    folder.mkdir(exist_ok=True)
    if any((folder / name).exists() for name in ("completion.json", "failure.json", "selection", "training")):
        raise FileExistsError("partial WOMD seed exists")
    started = time.monotonic()
    active = ["selection"]
    rng_hashes = {}

    def status(state, **extra):
        write(folder / "status.json", dict(
            status=state, seed=seed, pid=os.getpid(), at=now(),
            elapsed_seconds=time.monotonic() - started, active_stage=active[0], **extra))

    def progress(row):
        status("RUNNING", progress=row)
        print(json.dumps(dict(seed=seed, **row), allow_nan=False), flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        while True:
            free, _ = torch.cuda.mem_get_info("cuda:0")
            if free >= plan["minimum_cuda_free_bytes"]:
                break
            status("WAITING_FOR_GPU", free_bytes=free)
            time.sleep(30)
        _record_rng(folder, rng_hashes, "initial_indices", "start")
        indices = make_indices(plan["data_dir"], seed)
        population = load_role_arrays(plan["data_dir"], "train")
        indices["train_labels"] = population["y"].clone()
        torch.save(indices, folder / "initial_indices.pt")
        _record_rng(folder, rng_hashes, "initial_indices", "end")
        active[0] = "fresh_selection"
        _record_rng(folder, rng_hashes, "fresh_selection", "start")
        selected = fit_source_selection(
            plan["data_dir"], seed=seed, output_dir=folder / "selection", indices=indices,
            attachment_count=plan["attachment_count"], outer_extra_count=512, model_epochs=200,
            sage_n_permutations=2048, sage_batch_size=64, explanation_count=128,
            background_count=32, max_nodes=16384, device="cuda:0", progress=progress)
        _record_rng(folder, rng_hashes, "fresh_selection", "end")
        coords = selected["coordinates"].detach().cpu().long()
        ids = selected["indices"]
        proxies = []
        for role, offset in (("clean_proxy_1", 101000), ("clean_proxy_2", 102000)):
            active[0] = role
            _record_rng(folder, rng_hashes, role, "start")
            fitted = fit_classifier(
                plan["data_dir"], ids["idx_train"], seed=seed + offset,
                output_dir=folder / role, role=role, epochs=200, max_nodes=16384,
                device="cuda:0", checkpoint_rule="dev_auc", progress=progress)
            proxies.append(fitted["model"].eval().requires_grad_(False))
            _record_rng(folder, rng_hashes, role, "end")
        active[0] = "fresh_anchor_and_support"
        _record_rng(folder, rng_hashes, "anchor_base_residual", "start")
        anchor = fit_generator(
            plan["data_dir"], ids, proxies, coords, seed=seed + 201000,
            output_dir=folder / "anchor", base_epochs=200, residual_epochs=200,
            max_nodes=16384, device="cuda:0", progress=progress)
        _record_rng(folder, rng_hashes, "anchor_base_residual", "end")
        _record_rng(folder, rng_hashes, "semantic_support", "start")
        support, support_record = fit_support(
            anchor["generator"], anchor["surrogate"], plan["data_dir"], ids,
            proxies, coords, seed=seed + 301000, output_dir=folder / "semantic_support",
            epochs=200, max_nodes=16384, device="cuda:0", progress=progress)
        _record_rng(folder, rng_hashes, "semantic_support", "end")
        write(folder / "support.json", support_record)
        active[0] = "continuation_a_b"
        _record_rng(folder, rng_hashes, "continuation_a_b", "start")
        continuation = _continuations(
            anchor["generator"], anchor["surrogate"], plan["data_dir"], ids,
            proxies, coords, support, seed, folder / "continuation", progress,
            folder / "semantic_support")
        _record_rng(folder, rng_hashes, "continuation_a_b", "end")
        generator = continuation["generator"].eval().requires_grad_(False)
        active[0] = "victim"
        _record_rng(folder, rng_hashes, "victim", "start")
        victim = fit_victim(
            generator, plan["data_dir"], ids, seed=seed + 401000,
            output_dir=folder / "victim", epochs=200, max_nodes=16384,
            device="cuda:0", progress=progress)
        _record_rng(folder, rng_hashes, "victim", "end")
        active[0] = "clean_reference"
        _record_rng(folder, rng_hashes, "clean_reference", "start")
        reference = fit_classifier(
            plan["data_dir"], ids["idx_train"], seed=seed + 103000,
            output_dir=folder / "clean_reference", role="clean_reference", epochs=200,
            max_nodes=16384, device="cuda:0", checkpoint_rule="dev_auc", progress=progress)["model"]
        _record_rng(folder, rng_hashes, "clean_reference", "end")
        results = {}
        for role in ("dev", "audit"):
            active[0] = "evaluation_" + role
            _record_rng(folder, rng_hashes, active[0], "start")
            result = evaluate(
                generator, victim["model"], reference, plan["data_dir"],
                output_dir=folder / ("evaluation_" + role), role=role,
                device="cuda:0", progress=progress)
            results[role] = _augment_eval_summary(result, folder / ("evaluation_" + role))
            _record_rng(folder, rng_hashes, active[0], "end")
        for path in (folder / "continuation/rng_states").glob("*.pt"):
            rng_hashes[str(path.relative_to(folder))] = sha(path)
        receipt = dict(
            status="SEED_COMPLETE", seed=seed, at=now(), plan_sha256=sha(plan_path),
            elapsed_seconds=time.monotonic() - started,
            stages={stage: 200 for stage in STAGES}, generator_updates=800,
            shadow_updates=1000, support_selector_updates=200,
            clean_indices_sha256=_tensor_hash(ids["idx_train"]),
            support_sha256=_tensor_hash(support), generator_sha256=_state_hash(generator),
            victim_sha256=_state_hash(victim["model"]), clean_reference_sha256=_state_hash(reference),
            rng_state_sha256=rng_hashes,
            rng_manifest_sha256=sha(folder / "rng_states/manifest.json"),
            result_sha256={role: sha(folder / ("evaluation_" + role) / "summary.json") for role in results},
            official_test_loaded=False,
        )
        write(folder / "completion.json", receipt)
        status("COMPLETE")
    except BaseException as exc:
        failure = dict(
            status="FAILED", seed=seed, at=now(), stage=active[0],
            type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write(folder / "failure.json", failure)
        status("FAILED", failure=failure)
        raise


def aggregate(plan_path):
    from . import av2_corrected as base
    old_verify, old_seeds = base.verify, base.SEEDS
    base.verify, base.SEEDS = verify, SEEDS
    try:
        return base.aggregate(plan_path)
    finally:
        base.verify, base.SEEDS = old_verify, old_seeds


def supervise(plan_path):
    plan = verify(plan_path, frozen=True)
    root = Path(plan["output_dir"])
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in SEEDS:
            prior = aggregate(plan_path)
            folder = root / f"seed{seed}"
            command = [sys.executable, "-B", "-m", "spear_current.womd_full_corrected_v2", "--worker", str(plan_path), "--seed", str(seed)]
            with (folder / "worker.log").open("x") as log:
                child = subprocess.Popen(command, cwd=plan["source_snapshot"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    current = read(folder / "status.json") if (folder / "status.json").exists() else {}
                    write(root / "status.json", dict(status="RUNNING", at=now(), active_seed=seed,
                                                     worker_pid=child.pid, completed=prior["completed"],
                                                     failed=prior["failed"], total=5, worker_status=current))
                    time.sleep(15)
            if child.returncode != 0 and not (folder / "failure.json").exists():
                write(folder / "failure.json", dict(status="FAILED", seed=seed, worker_returncode=child.returncode, at=now()))
            write(folder / "supervisor_exit.json", dict(at=now(), returncode=child.returncode))
            aggregate(plan_path)
        result = aggregate(plan_path)
        state = "COMPLETE" if result["completed"] == 5 else "FINISHED_WITH_FAILURES"
        write(root / "status.json", dict(status=state, at=now(), completed=result["completed"], failed=result["failed"], total=5))
        if state == "COMPLETE":
            write(root / "completion.json", dict(status="ALL_FIVE_COMPLETE", at=now(), plan_sha256=sha(plan_path), aggregate_sha256=sha(root / "aggregate.json")))
        return 0 if state == "COMPLETE" else 1


def launch(plan_path):
    plan = verify(plan_path)
    root = Path(plan["output_dir"])
    acceptance = read(root / "preflight/acceptance.json")
    if acceptance.get("status") != "PASS" or acceptance.get("plan_sha256") != sha(plan_path):
        raise ValueError("WOMD GPU preflight acceptance required")
    with (root / "launch.lock").open("x"):
        pass
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-B", "-m", "spear_current.womd_full_corrected_v2", "--supervise", str(Path(plan_path).resolve())]
    with (root / "supervisor.log").open("x") as log:
        child = subprocess.Popen(command, cwd=plan["source_snapshot"], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(root / "launch.json", dict(pid=child.pid, at=now(), command=command))
    return child.pid


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", type=Path)
    group.add_argument("--preflight", type=Path)
    group.add_argument("--launch", type=Path)
    group.add_argument("--supervise", type=Path)
    group.add_argument("--worker", type=Path)
    group.add_argument("--aggregate", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.register:
        print(register(args.register))
    elif args.preflight:
        print(json.dumps(preflight(args.preflight), indent=2))
    elif args.launch:
        print(launch(args.launch))
    elif args.supervise:
        raise SystemExit(supervise(args.supervise))
    elif args.worker:
        worker(args.worker, args.seed)
    else:
        print(json.dumps(aggregate(args.aggregate), indent=2))


if __name__ == "__main__":
    main()
