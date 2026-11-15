"""Prepare a formal full-corpus WOMD cache from the registered graph assets.

The graph builder produces per-scene pair NPZ files.  This adapter freezes a
source-shard-disjoint 60/20/20 split, fits normalization on TRAIN only, and
writes the same whole-scene shard contract consumed by ``av2_formal_data``.
It never reads official validation or test arrays.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import transform_features
from .runtime import sha256
from .womd_pilot_data import read_part, target_index, validate_source

ROLES = ("train", "dev", "audit")
MAX_NODES = 16384


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _key(seed, *values):
    return hashlib.sha256("|".join(map(str, (seed, *values))).encode()).hexdigest()


def _save_shard(path, parts):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(parts, temporary)
    temporary.replace(path)
    return dict(path=Path(path).name, sha256=sha256(path), scenes=len(parts),
                nodes=sum(len(p["x"]) for p in parts),
                edges=sum(p["edge_index"].shape[1] for p in parts),
                global_start=int(parts[0]["global_offset"]),
                global_stop=int(parts[-1]["global_stop"]))


def _load_manifest(root):
    root = Path(root).resolve()
    validate_source(root)
    with (root / "manifest.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("status") == "BUILT"]
    if len(rows) != 52495:
        raise ValueError(f"Expected 52495 BUILT scenes, found {len(rows)}")
    return rows


def _selected(root, split_path):
    root = Path(root).resolve()
    rows = _load_manifest(root)
    split = json.loads(Path(split_path).read_text())
    if split.get("manifest_sha256") != sha256(root / "manifest.csv"):
        raise ValueError("Split candidate is bound to a different WOMD manifest")
    groups = split["assignment"]["groups"]
    if set(groups) != {r["source_shard"] for r in rows}:
        raise ValueError("Split does not assign every source shard exactly once")
    selected = {role: [] for role in ROLES}
    for row in rows:
        role = groups[row["source_shard"]]
        if role not in selected:
            raise ValueError(f"Unknown split role: {role}")
        selected[role].append(row)
    for role in ROLES:
        selected[role].sort(key=lambda r: (r["source_shard"], int(r["record_index"]), r["scenario_id"]))
    group_sets = {role: {r["source_shard"] for r in pool} for role, pool in selected.items()}
    if any(group_sets[a] & group_sets[b] for a, b in (("train", "dev"), ("train", "audit"), ("dev", "audit"))):
        raise ValueError("Cross-role source-shard overlap")
    if sum(len(v) for v in selected.values()) != len(rows):
        raise ValueError("Split does not preserve the complete BUILT population")
    return selected, split


def _fit_online(parts):
    count = 0
    mean = np.zeros(784, dtype=np.float64)
    m2 = np.zeros(784, dtype=np.float64)
    for part in parts:
        x = part["x"].astype(np.float64, copy=False)
        n = len(x)
        if not n:
            continue
        bm = x.mean(axis=0)
        delta = bm - mean
        m2 += ((x - bm) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if count == 0:
        raise ValueError("TRAIN has no pair nodes")
    return dict(mean=mean, scale=np.maximum(np.sqrt(m2 / count), 1e-6), clip=10.,
                minimum_scale=1e-6, feature_dimension=784, fitted_rows=count)


def prepare(graph_root, split_path, destination, *, seed=20261104, max_nodes=MAX_NODES,
            progress=None):
    graph_root = Path(graph_root).resolve()
    split_path = Path(split_path).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    selected, split = _selected(graph_root, split_path)
    destination.mkdir(parents=True)
    started = time.monotonic()

    role_counts = {role: dict(scenes=len(rows), groups=len({r["source_shard"] for r in rows}))
                   for role, rows in selected.items()}
    registration = dict(
        schema="womd_full_corpus_data_v1",
        dataset_id="WOMD",
        source_dataset_split="training_20s",
        graph_root=str(graph_root),
        graph_summary_sha256=sha256(graph_root / "summary.json"),
        manifest_sha256=sha256(graph_root / "manifest.csv"),
        split_path=str(split_path),
        split_sha256=sha256(split_path),
        split_schema=split.get("schema"),
        split_seed=split["assignment"]["seed"],
        source_shard_partition=split["assignment"]["partition_counts"],
        role_counts=role_counts,
        grouping="source_shard",
        scene_selection="all BUILT records, deterministic source-shard split; no label/prediction filtering",
        normalizer_fit="all pair nodes of TRAIN scenes only; population mean/std, scale>=1e-6, clip10",
        target_policy="one hash-selected eligible negative per scene in DEV/AUDIT; fixed before training",
        clean_supervision="20 percent of eligible TRAIN pairs per seed without replacement",
        official_validation_loaded=False,
        official_test_loaded=False,
        recording_independence_proven=False,
    )
    _json(destination / "registration.json", registration)
    _json(destination / "status.json", dict(stage="PREPARING", elapsed_seconds=0.0))

    def emit(stage, **extra):
        value = dict(stage=stage, elapsed_seconds=time.monotonic() - started, **extra)
        _json(destination / "status.json", value)
        print("[womd-full-data] " + json.dumps(value, allow_nan=False), flush=True)
        if progress:
            progress(value)

    summary = dict(schema="womd_full_corpus_cache_v1", status="PREPARED",
                   dataset_id="WOMD", source_dataset_split="training_20s",
                   split_sha256=sha256(split_path), split_schema=split.get("schema"),
                   max_nodes_per_shard=max_nodes, whole_scene_shards=True,
                   oversize_scene_policy="retain a whole scene in its own shard",
                   original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
                   blind_evaluation=False, recording_independence_proven=False)
    try:
        train_raw = []
        for index, row in enumerate(selected["train"]):
            part = read_part(graph_root, row)
            part.update(split="training_20s", role="train", dataset_id="WOMD",
                        evaluation_subgroup="womd_full_train",
                        split_group_id="source_shard:" + row["source_shard"])
            train_raw.append(part)
            if index % 500 == 0:
                emit("READING_TRAIN", completed_scenes=index, total_scenes=len(selected["train"]))
        normalizer = _fit_online(train_raw)
        np.savez(destination / "normalizer.npz", **normalizer)
        summary["normalizer_fitted_rows"] = int(normalizer["fitted_rows"])

        for role in ROLES:
            rows = selected[role]
            parts = train_raw if role == "train" else []
            if role != "train":
                for index, row in enumerate(rows):
                    part = read_part(graph_root, row)
                    part.update(split="training_20s", role=role, dataset_id="WOMD",
                                evaluation_subgroup="womd_full_" + role,
                                split_group_id="source_shard:" + row["source_shard"])
                    parts.append(part)
                    if index % 500 == 0:
                        emit("READING_" + role.upper(), completed_scenes=index, total_scenes=len(rows))
            folder = destination / role
            folder.mkdir()
            shards, pending, pending_nodes = [], [], 0
            y_chunks, eligible_chunks, motion_chunks, offsets = [], [], [], []
            offset = 0
            stats = dict(scenes=0, groups=role_counts[role]["groups"], pair_nodes=0,
                         pair_edges=0, eligible=0, positive=0, targets=0,
                         moving_targets=0, zero_pair_scenes=0)
            for index, part in enumerate(parts):
                n = len(part["x"])
                if pending and pending_nodes + n > max_nodes:
                    shards.append(_save_shard(folder / f"{len(shards):05d}.pt", pending))
                    pending, pending_nodes = [], 0
                if role != "train":
                    part["x"] = transform_features(part["x"], normalizer)
                else:
                    part["x"] = transform_features(part["x"], normalizer)
                part["target_index"] = target_index(part, seed) if role != "train" else None
                part["global_offset"], part["global_stop"] = offset, offset + n
                offsets.append(offset)
                offset += n
                pending.append(part)
                pending_nodes += n
                y_chunks.append(part["y"])
                eligible_chunks.append(part["eligible"])
                motion_chunks.append(part["motion_regime"])
                stats["scenes"] += 1
                stats["pair_nodes"] += n
                stats["pair_edges"] += int(part["edge_index"].shape[1])
                stats["eligible"] += int(part["eligible"].sum())
                stats["positive"] += int((part["y"] == 1).sum())
                stats["zero_pair_scenes"] += int(n == 0)
                if part["target_index"] is not None:
                    stats["targets"] += 1
                    stats["moving_targets"] += int(part["motion_regime"][part["target_index"]] == 1)
                if index % 500 == 0:
                    emit("WRITING_" + role.upper(), completed_scenes=index,
                         total_scenes=len(parts), shards=len(shards))
            if pending:
                shards.append(_save_shard(folder / f"{len(shards):05d}.pt", pending))
            population = dict(
                y=torch.from_numpy(np.concatenate(y_chunks)).long(),
                eligible=torch.from_numpy(np.concatenate(eligible_chunks)).bool(),
                motion_regime=torch.from_numpy(np.concatenate(motion_chunks)),
                scene_offsets=torch.tensor(offsets + [offset], dtype=torch.long),
            )
            torch.save(population, destination / f"{role}_population.pt")
            summary[role] = dict(**stats, shards=shards,
                evaluation_subgroups=dict(Counter(p["evaluation_subgroup"] for p in parts)))
            emit("ROLE_PREPARED", role=role, **stats)

        eligible = torch.nonzero(summary and torch.load(destination / "train_population.pt",
                                                         map_location="cpu", weights_only=True)["eligible"],
                                 as_tuple=False).flatten().numpy()
        clean = np.sort(np.random.RandomState(seed).permutation(eligible)[:int(.2 * len(eligible))])
        population = torch.load(destination / "train_population.pt", map_location="cpu", weights_only=True)
        if set(population["y"][clean].tolist()) != {0, 1}:
            raise ValueError("Clean TRAIN supervision must contain both labels")
        indices = dict(idx_train=torch.from_numpy(clean),
                       idx_unlabeled=torch.from_numpy(np.setdiff1d(
                           np.arange(len(population["y"]), dtype=np.int64), clean)))
        torch.save(indices, destination / "indices_seed_reference.pt")
        summary["clean_supervised_pairs"] = int(len(clean))
        summary["attachment_budget_reference"] = dict(
            reference_av2_attachment_count=5865,
            reference_av2_clean_supervised_pairs=437658,
            attachment_count=8760,
            realized_fraction=float(8760 / (len(clean) + 8760)),
            rule="nearest integer to the AV2 corrected attachment fraction; lower tie",
        )
        summary["asset_sha256"] = {p.name: sha256(p) for p in sorted(destination.iterdir())
                                    if p.is_file() and p.name != "status.json"}
        summary["elapsed_seconds"] = time.monotonic() - started
        _json(destination / "summary.json", summary)
        emit("PREPARED", role_counts={r: summary[r]["scenes"] for r in ROLES})
        return summary
    except BaseException as exc:
        emit("FAILED", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)
    args = parser.parse_args()
    prepare(args.graph_root, args.split, args.output, seed=args.seed, max_nodes=args.max_nodes)
