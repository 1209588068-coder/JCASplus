#!/usr/bin/env python3
"""Data-defined target-pool evaluator for frozen and development experiments.

The target pool is defined from split graph fields and true labels only. Model
predictions are produced only after the pool is fixed in memory, and are never
used to include/exclude a target. Training access is reported separately from
this data-only evaluation-target selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data

from jcas import PROJECT_ROOT
from jcas.core.poison import (
    ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4,
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
    ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
    FIXED_SYMMETRIC_BIEND_ALPHA,
    ORIENTATION_POLICIES,
    ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
    eligible_negative_pair_groups,
    fixed_symmetric_bi_endpoint_allocation,
    reverse_edge_id,
    select_bi_endpoint_allocation,
    select_pair_orientation,
    sha256_file,
)
from jcas.core.models import build_model, model_kwargs_from_result
from jcas.core.motion_normalized_features import (
    BASE_EDGE_FEATURE_MODE,
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
    edge_feature_protocol,
    edge_features_for_mode,
    validate_edge_feature_mode,
)
from jcas.core.motion_schedule_trigger import (
    MOTION_REGIME_K4_K10_SCHEDULE,
    apply_scheduled_trajectory_trigger,
    scheduled_window,
)
from jcas.core.risk_labels import (
    config_from_result,
    label_config_dict,
    label_config_hash,
    labels_for_graph,
    selected_label_computable_mask,
)
from jcas.workflows.trainer import (
    load_manifest,
    message_passing_mask,
    resolve_graph_path,
    resolve_manifest_path,
    supervision_mask,
    verified_graph_path,
)
from jcas.core.trajectory_trigger import TriggerSpec, apply_trajectory_trigger


REPOSITORY_ROOT = PROJECT_ROOT
V6_EVALUATOR_INTEGRITY_VERSION = "v6.0_pretest_release_metric_evidence"
V6_PRETEST_RELEASE_VERSION = "v6.0_evaluation_integrity"
EVALUATOR_INTEGRITY_VERSION = V6_EVALUATOR_INTEGRITY_VERSION
PRETEST_RELEASE_VERSION = V6_PRETEST_RELEASE_VERSION
FORMAL_TRAINING_EPOCHS = 50
FORMAL_CHECKPOINT_METRIC = "val_loss"
V6_FORMAL_CONTRACT_PATH = "record/v6/contracts/v6_freeze_20260814.metadata.json"
V6_FORMAL_CONTRACT_SHA256 = (
    "6223c1e26c9e2afb1593ab07fd006f16003c6eae6409bfbdb815a24b495ea50c"
)
V6_FORMAL_ASSET_MANIFEST_SHA256 = (
    "1632d9ca000f67c71d7c3aa6ece7d5824454cce6522ae00108a58484c7967f20"
)
FORMAL_CONTRACT_DEFAULT_PATH = V6_FORMAL_CONTRACT_PATH
FORMAL_TRAINING_SEEDS = (20260621, 20260622, 20260623)
EVALUATION_ALLOCATION_POLICIES = (
    ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
    ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
)
V6_FORMAL_FROZEN_CONFIG = {
    "poison_rate": 0.05,
    "orientation_policy": "lower_destination_mean_speed_v1",
    "allocation_policy": "fixed_symmetric_biend_v1",
    "training_allocation_policy": "fixed_symmetric_biend_v1",
    "selection_objective": (
        "v5_gradient_pair_selection_then_fixed_symmetric_alpha_v6_0"
    ),
    "source_pair_selection_objective": "gradient_influence_v4_2",
    "allocation_grid": [0.5],
    "relative_displacement_m": 0.2,
    "perturb_window": 10,
    "ramp_style": "minimum_jerk",
    "velocity_mode": "residual",
    "target_seed": 20260621,
    "same_pair_single_variable_change": True,
    "outside_v5_pair_feature_budget_rows": 2110,
    "label_mode": "dynamic_risk",
    "risk_base_distance_m": 5.0,
    "risk_reaction_time_s": 1.0,
    "risk_safe_decel_mps2": 4.0,
    "graph_schema_version": 4,
    "build_contract_version": 3,
    "split_strategy": (
        "recording_or_stride1_twoframe_av_overlap_content_group_sha256_v5"
    ),
    "strict_crossfit_protocol": "strict_crossfit_inner_validation_v1",
}
FORMAL_FROZEN_CONFIG = V6_FORMAL_FROZEN_CONFIG
FORMAL_TRUST_ANCHORS = {
    V6_FORMAL_CONTRACT_SHA256: {
        "status": "frozen_v6_validation_selected",
        "version": "v6_fixed_symmetric_same_pair_training",
        "asset_manifest_sha256": V6_FORMAL_ASSET_MANIFEST_SHA256,
        "frozen_config": V6_FORMAL_FROZEN_CONFIG,
        "clean_role": "clean_reference",
        "victim_role": "v6_fixed_symmetric_victim",
    },
}
PRETEST_RELEASE_PROFILES = {
    V6_PRETEST_RELEASE_VERSION: "pre_frozen_before_v6_formal_test",
}
CLEAN_REFERENCE_MATCH_CONFIG_KEYS = (
    "seed",
    "model_name",
    "hidden_dim",
    "num_layers",
    "dropout",
    "norm",
    "decoder_hidden_dim",
    "decoder_num_layers",
    "batch_size",
    "lr",
    "weight_decay",
    "epochs",
    "checkpoint_metric",
    "require_strict_label",
    "label_mode",
    "risk_base_distance_m",
    "risk_reaction_time_s",
    "risk_safe_decel_mps2",
    "edge_feature_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a data-defined pair target pool.")
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument(
        "--graph-manifest",
        default=None,
        help="Explicit split manifest; defaults to <graph-dir>/manifest.csv",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-target-manifest", default=None,
                        help="Validation-only public target CSV to match before model inference")
    parser.add_argument("--expected-target-manifest-sha256", default=None,
                        help="SHA-256 binding for --expected-target-manifest")
    parser.add_argument(
        "--evaluation-model-role",
        choices=["victim", "clean_reference"],
        default="victim",
        help=(
            "Role of --checkpoint in the frozen experiment. Formal test "
            "evaluation verifies the checkpoint against the corresponding "
            "victim or clean-reference asset allowlist."
        ),
    )
    parser.add_argument(
        "--formal-contract",
        default=FORMAL_CONTRACT_DEFAULT_PATH,
        help=(
            "Frozen versioned metadata trust anchor. The file is accepted "
            "only if its SHA-256 matches a built-in trusted digest."
        ),
    )
    parser.add_argument(
        "--pretest-release",
        default=None,
        help=(
            "Pre-frozen v4.1.1 evaluation release metadata. Required for "
            "test evaluation and verified before any test graph is opened."
        ),
    )
    parser.add_argument(
        "--pretest-release-sha256",
        default=None,
        help=(
            "Externally preserved SHA-256 trust anchor for --pretest-release. "
            "Required for test evaluation."
        ),
    )
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--clean-reference-result",
        default=None,
        help=(
            "Clean result.json that supplies and cryptographically binds the "
            "common validation pair threshold. Required for test evaluation."
        ),
    )
    threshold_group.add_argument(
        "--common-threshold",
        type=float,
        default=None,
        help=(
            "Development-only unbound threshold retained for validation "
            "compatibility; formal test evaluation rejects it"
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["val", "test"],
        help="Use val for development; reserve test for the final frozen experiment",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--target-seed",
        type=int,
        default=20260621,
        help="Data-only seed for selecting one eligible target edge per test scenario",
    )
    parser.add_argument(
        "--orientation-policy",
        choices=ORIENTATION_POLICIES,
        default=ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
        help=(
            "Data-only rule for choosing which endpoint of the already-selected "
            "unordered pair is moved"
        ),
    )
    parser.add_argument(
        "--allocation-policy",
        choices=EVALUATION_ALLOCATION_POLICIES,
        default=ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
        help=(
            "Frozen data-only evaluation rule for distributing the fixed "
            "0.2 m relative displacement across the two endpoints. "
            "Surrogate-selected training policies are intentionally excluded."
        ),
    )
    parser.add_argument(
        "--experimental-trigger-schedule",
        choices=[MOTION_REGIME_K4_K10_SCHEDULE],
        default=None,
        help=(
            "Validation-only slow-K10/moving-K4 schedule. Omit to retain the "
            "frozen K10 main-line evaluator."
        ),
    )
    parser.add_argument(
        "--max-graphs",
        "--max-test-graphs",
        dest="max_graphs",
        type=int,
        default=None,
        help="Debug-only split-graph cap; any capped result is marked incomplete",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Debug-only global cap; any capped result is marked non-formal",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace existing result/target/score files",
    )
    return parser.parse_args()


def _scenario_target_edge(
    scenario_id: str, eligible_edge_ids: np.ndarray, seed: int
) -> int:
    """Select one eligible edge using only scenario identity and a fixed seed."""
    edge_ids = np.asarray(eligible_edge_ids, dtype=np.int64)
    if edge_ids.ndim != 1 or edge_ids.size == 0:
        raise ValueError("eligible_edge_ids must be a non-empty vector")
    if int(seed) < 0:
        raise ValueError("target seed must be non-negative")
    digest = hashlib.sha256(
        f"strict-test-target-v1:{int(seed)}:{scenario_id}".encode("utf-8")
    ).digest()
    entropy = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64).tolist()
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return int(rng.choice(edge_ids))


def _scenario_target_pair_choice(
    scenario_id: str,
    pair_groups: list[np.ndarray],
    seed: int,
    *,
    graph: Any | None = None,
    orientation_policy: str = ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
) -> tuple[int, dict[str, Any]]:
    """Return the selected edge plus data-only orientation audit fields."""
    if not pair_groups or any(np.asarray(group).size == 0 for group in pair_groups):
        raise ValueError("pair_groups must contain non-empty orientation vectors")
    if int(seed) < 0:
        raise ValueError("target seed must be non-negative")
    digest = hashlib.sha256(
        f"strict-pair-target-v2:{int(seed)}:{scenario_id}".encode("utf-8")
    ).digest()
    entropy = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64).tolist()
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    group = np.asarray(pair_groups[int(rng.integers(len(pair_groups)))], dtype=np.int64)
    if graph is None:
        raise ValueError("a graph is required for orientation selection")
    return select_pair_orientation(
        graph,
        group,
        policy=orientation_policy,
        perturb_window=10,
    )


def _data(
    graph: Any,
    labels: np.ndarray,
    *,
    x_node: np.ndarray | None = None,
    edge_attr: np.ndarray | None = None,
) -> Data:
    return Data(
        x=torch.from_numpy(
            np.asarray(graph["x_node"] if x_node is None else x_node, dtype=np.float32)
        ),
        edge_index=torch.from_numpy(np.asarray(graph["edge_index"], dtype=np.int64)),
        edge_attr=torch.from_numpy(
            np.asarray(graph["edge_attr"] if edge_attr is None else edge_attr, dtype=np.float32)
        ),
        edge_label=torch.from_numpy(np.asarray(labels, dtype=np.float32)),
        message_passing_edge_mask=torch.from_numpy(message_passing_mask(graph)),
    )


def _probabilities(model: torch.nn.Module, data: Data, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        data = data.to(device)
        logits = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.message_passing_edge_mask,
        )
        return torch.sigmoid(logits).cpu().numpy()


def _clean_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    threshold_source: str,
) -> dict[str, Any]:
    prediction = probs >= threshold
    has_both_classes = len(np.unique(labels)) >= 2
    return {
        "edges": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "auc": float(roc_auc_score(labels, probs)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(labels, probs)) if has_both_classes else None,
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _accumulate_collateral_counts(
    counts: dict[str, int],
    clean_probabilities: np.ndarray,
    triggered_probabilities: np.ndarray,
    *,
    threshold: float,
    nonincident_negative: np.ndarray,
    incident_negative: np.ndarray,
    incident_positive: np.ndarray,
) -> None:
    """Accumulate collateral outcomes at one explicitly supplied threshold."""
    clean = np.asarray(clean_probabilities)
    triggered = np.asarray(triggered_probabilities)
    for prefix, subset in (
        ("nonincident_negative", nonincident_negative),
        ("incident_negative", incident_negative),
    ):
        subset = np.asarray(subset, dtype=bool)
        edge_key = f"{prefix}_edges"
        absolute_key = f"{prefix}_triggered_positive"
        incremental_key = f"{prefix}_incremental_fp"
        counts[edge_key] = int(counts.get(edge_key, 0)) + int(subset.sum())
        counts[absolute_key] = int(counts.get(absolute_key, 0)) + int(
            (triggered[subset] >= float(threshold)).sum()
        )
        counts[incremental_key] = int(counts.get(incremental_key, 0)) + int(
            (
                (clean[subset] < float(threshold))
                & (triggered[subset] >= float(threshold))
            ).sum()
        )

    incident_positive = np.asarray(incident_positive, dtype=bool)
    counts["incident_positive_edges"] = int(
        counts.get("incident_positive_edges", 0)
    ) + int(incident_positive.sum())
    counts["incident_positive_triggered_suppressed"] = int(
        counts.get("incident_positive_triggered_suppressed", 0)
    ) + int(
        (triggered[incident_positive] < float(threshold)).sum()
    )
    counts["incident_positive_incremental_suppression"] = int(
        counts.get("incident_positive_incremental_suppression", 0)
    ) + int(
        (
            (clean[incident_positive] >= float(threshold))
            & (triggered[incident_positive] < float(threshold))
        ).sum()
    )


def _collateral_metrics_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    """Convert accumulated collateral counts into the public metric schema."""
    nonincident_edges = int(counts["nonincident_negative_edges"])
    adjacent_negative_edges = int(counts["incident_negative_edges"])
    adjacent_positive_edges = int(counts["incident_positive_edges"])
    return {
        "nonincident_negative_fp_absolute": _rate(
            counts["nonincident_negative_triggered_positive"],
            nonincident_edges,
        ),
        "nonincident_negative_fp_incremental": _rate(
            counts["nonincident_negative_incremental_fp"],
            nonincident_edges,
        ),
        "nonincident_negative_edges": nonincident_edges,
        "nonincident_negative_status": (
            "ok" if nonincident_edges else "empty"
        ),
        "adjacent_negative_fp_absolute": _rate(
            counts["incident_negative_triggered_positive"],
            adjacent_negative_edges,
        ),
        "adjacent_negative_fp_incremental": _rate(
            counts["incident_negative_incremental_fp"],
            adjacent_negative_edges,
        ),
        "adjacent_negative_edges": adjacent_negative_edges,
        "adjacent_negative_status": (
            "ok" if adjacent_negative_edges else "empty"
        ),
        "adjacent_positive_suppression_absolute": _rate(
            counts["incident_positive_triggered_suppressed"],
            adjacent_positive_edges,
        ),
        "adjacent_positive_suppression_incremental": _rate(
            counts["incident_positive_incremental_suppression"],
            adjacent_positive_edges,
        ),
        "adjacent_positive_edges": adjacent_positive_edges,
        "adjacent_positive_status": (
            "ok" if adjacent_positive_edges else "empty"
        ),
    }


def _pair_probability(probabilities: np.ndarray, edge_id: int, reverse_id: int) -> float:
    return float(
        0.5
        * (
            float(probabilities[int(edge_id)])
            + float(probabilities[int(reverse_id)])
        )
    )


def _supervised_pair_arrays(
    edge_index: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    supervised: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[int, int], list[int]] = {}
    edge_index = np.asarray(edge_index, dtype=np.int64)
    for edge_id in np.flatnonzero(supervised):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        grouped.setdefault((min(src, dst), max(src, dst)), []).append(int(edge_id))
    pair_probabilities: list[float] = []
    pair_labels: list[int] = []
    for edge_ids in grouped.values():
        if len(edge_ids) != 2:
            continue
        if int(labels[edge_ids[0]]) != int(labels[edge_ids[1]]):
            raise RuntimeError(
                "symmetric evaluation pair contains contradictory directed labels"
            )
        pair_probabilities.append(
            float(np.asarray(probabilities)[edge_ids].mean())
        )
        pair_labels.append(int(labels[edge_ids[0]]))
    return (
        np.asarray(pair_probabilities, dtype=np.float32),
        np.asarray(pair_labels, dtype=np.int8),
    )


def _supervised_pair_records(
    edge_index: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    supervised: np.ndarray,
) -> list[tuple[int, int, int, float]]:
    """Return canonical unordered-pair identities and their clean predictions."""
    grouped: dict[tuple[int, int], list[int]] = {}
    edge_index = np.asarray(edge_index, dtype=np.int64)
    for edge_id in np.flatnonzero(supervised):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        grouped.setdefault((min(src, dst), max(src, dst)), []).append(int(edge_id))
    records: list[tuple[int, int, int, float]] = []
    for edge_ids in grouped.values():
        if len(edge_ids) != 2:
            continue
        canonical_edge_id, reverse_id = sorted(edge_ids)
        if int(labels[canonical_edge_id]) != int(labels[reverse_id]):
            raise RuntimeError(
                "symmetric evaluation pair contains contradictory directed labels"
            )
        records.append(
            (
                int(canonical_edge_id),
                int(reverse_id),
                int(labels[canonical_edge_id]),
                float(np.asarray(probabilities)[edge_ids].mean()),
            )
        )
    return records


def _transition_counts(
    clean_probabilities: np.ndarray,
    triggered_probabilities: np.ndarray,
    subset: np.ndarray,
    *,
    threshold: float,
) -> dict[str, int]:
    """Build the complete clean/triggered 2x2 transition table."""
    subset = np.asarray(subset, dtype=bool)
    clean_positive = np.asarray(clean_probabilities)[subset] >= float(threshold)
    triggered_positive = (
        np.asarray(triggered_probabilities)[subset] >= float(threshold)
    )
    result = {
        "n00": int((~clean_positive & ~triggered_positive).sum()),
        "n01": int((~clean_positive & triggered_positive).sum()),
        "n10": int((clean_positive & ~triggered_positive).sum()),
        "n11": int((clean_positive & triggered_positive).sum()),
    }
    result["total"] = int(sum(result.values()))
    return result


def _target_metrics_at_threshold(
    scores: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    if scores.empty:
        return {
            "threshold": float(threshold),
            "targets": 0,
            "clean_activation_rate": None,
            "absolute_asr": None,
            "incremental_flip_rate_all_targets": None,
            "conditional_flip_rate": None,
            "clean_negative_targets": 0,
            "incremental_flips": 0,
        }
    clean = scores["clean_pair_probability"].to_numpy(np.float64)
    triggered = scores["triggered_pair_probability"].to_numpy(np.float64)
    clean_negative = clean < float(threshold)
    triggered_positive = triggered >= float(threshold)
    flips = clean_negative & triggered_positive
    return {
        "threshold": float(threshold),
        "targets": int(len(scores)),
        "clean_activation_rate": float((~clean_negative).mean()),
        "absolute_asr": float(triggered_positive.mean()),
        "incremental_flip_rate_all_targets": float(flips.mean()),
        "conditional_flip_rate": (
            float(flips.sum() / clean_negative.sum())
            if clean_negative.any()
            else None
        ),
        "clean_negative_targets": int(clean_negative.sum()),
        "incremental_flips": int(flips.sum()),
    }


def validate_training_asset_binding(
    checkpoint: Path,
    training_result: dict[str, Any],
    graph_manifest_path: Path,
) -> dict[str, Any]:
    """Fail closed unless checkpoint and graph manifest match training."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    checkpoint_record = training_result.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise ValueError(
            "training result has no checkpoint binding; retrain with the "
            "integrity-enabled train_gnn.py"
        )
    expected_checkpoint = str(checkpoint_record.get("sha256", "")).lower()
    actual_checkpoint = sha256_file(checkpoint)
    if expected_checkpoint != actual_checkpoint:
        raise RuntimeError(
            "checkpoint SHA-256 does not match its training result"
        )

    manifest_record = training_result.get("graph_manifest")
    if not isinstance(manifest_record, dict):
        raise ValueError("training result has no graph_manifest binding")
    expected_manifest = str(manifest_record.get("sha256", "")).lower()
    actual_manifest = sha256_file(graph_manifest_path)
    if expected_manifest != actual_manifest:
        raise RuntimeError(
            "evaluation graph manifest SHA-256 does not match training"
        )
    contract_record = manifest_record.get("contract")
    if not isinstance(contract_record, dict):
        raise ValueError("training result has no split-contract binding")
    contract_path = graph_manifest_path.with_suffix(
        graph_manifest_path.suffix + ".metadata.json"
    )
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"training split contract does not exist: {contract_path}"
        )
    actual_contract = sha256_file(contract_path)
    if str(contract_record.get("sha256", "")).lower() != actual_contract:
        raise RuntimeError(
            "split-contract SHA-256 does not match the training result"
        )
    poison_record = training_result.get("poison_manifest")
    poison_verified = poison_record is None
    if poison_record is not None:
        if not isinstance(poison_record, dict):
            raise ValueError("training result poison_manifest is malformed")
        poison_path = Path(str(poison_record.get("path", "")))
        metadata_path = Path(str(poison_record.get("metadata_path", "")))
        if not poison_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "training poison manifest or its metadata sidecar is missing"
            )
        if str(poison_record.get("sha256", "")).lower() != sha256_file(
            poison_path
        ):
            raise RuntimeError(
                "poison manifest SHA-256 does not match the training result"
            )
        if str(poison_record.get("metadata_sha256", "")).lower() != sha256_file(
            metadata_path
        ):
            raise RuntimeError(
                "poison metadata SHA-256 does not match the training result"
            )
        poison_verified = True
    return {
        "checkpoint_sha256": actual_checkpoint,
        "graph_manifest_sha256": actual_manifest,
        "split_contract_sha256": actual_contract,
        "checkpoint_matches_training_result": True,
        "graph_manifest_matches_training_result": True,
        "split_contract_matches_training_result": True,
        "poison_assets_match_training_result": poison_verified,
    }


def _resolve_contract_recorded_path(
    contract_path: Path, recorded_path: str
) -> Path:
    recorded = Path(str(recorded_path))
    candidates = [recorded, contract_path.parent / recorded.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"frozen contract asset does not exist: {recorded_path}"
    )


def _parse_frozen_asset_manifest(path: Path) -> dict[str, str]:
    """Parse a strict sha256sum-style allowlist without trusting file paths."""
    entries: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        match = pattern.fullmatch(raw_line)
        if match is None:
            raise ValueError(
                f"invalid frozen asset manifest line {line_number}"
            )
        digest, recorded_path = match.groups()
        if recorded_path in entries:
            raise ValueError(
                f"duplicate frozen asset manifest path: {recorded_path}"
            )
        entries[recorded_path] = digest
    if not entries:
        raise ValueError("frozen asset manifest is empty")
    return entries


def _repository_relative_path(value: str, *, label: str) -> Path:
    """Resolve one recorded path without allowing it to escape the repository."""
    recorded = Path(str(value))
    if recorded.is_absolute():
        try:
            recorded = recorded.resolve().relative_to(REPOSITORY_ROOT)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes repository: {value}") from exc
    resolved = (REPOSITORY_ROOT / recorded).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes repository: {value}") from exc


def _verify_release_asset(record: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"pretest release has no {label} record")
    relative = _repository_relative_path(str(record.get("path", "")), label=label)
    path = REPOSITORY_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"pretest {label} is missing: {relative}")
    expected = str(record.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"pretest {label} has an invalid SHA-256")
    if sha256_file(path) != expected:
        raise RuntimeError(f"pretest {label} SHA-256 mismatch: {relative}")
    return relative, expected


def load_pretest_release(
    path: Path,
    *,
    expected_sha256: str,
    expected_formal_contract: Path,
) -> dict[str, Any]:
    """Verify one supported external pretest trust anchor before test access."""
    release_path = path.resolve()
    try:
        release_relative = release_path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("pretest release path escapes repository") from exc
    if not release_path.is_file():
        raise FileNotFoundError(f"pretest release does not exist: {release_relative}")
    expected_release_sha = str(expected_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_release_sha):
        raise ValueError("pretest release trust anchor is not one SHA-256 digest")
    actual_release_sha = sha256_file(release_path)
    if actual_release_sha != expected_release_sha:
        raise RuntimeError("pretest release SHA-256 does not match external trust anchor")
    with release_path.open(encoding="utf-8") as stream:
        release = json.load(stream)
    if not isinstance(release, dict):
        raise ValueError("pretest release root must be an object")
    release_version = str(release.get("version", ""))
    expected_status = PRETEST_RELEASE_PROFILES.get(release_version)
    if expected_status is None:
        raise ValueError("pretest release has an unexpected version")
    if release.get("status") != expected_status:
        raise ValueError("pretest release has an unexpected status")
    if int(release.get("test_outputs_present_before_freeze", -1)) != 0:
        raise RuntimeError("pretest release was created after test outputs existed")

    formal_relative, formal_sha = _verify_release_asset(
        release.get("formal_contract"), label="formal contract"
    )
    requested_contract = expected_formal_contract.resolve()
    if requested_contract != (REPOSITORY_ROOT / formal_relative).resolve():
        raise RuntimeError("requested formal contract is not the pre-frozen contract")
    if formal_sha not in FORMAL_TRUST_ANCHORS:
        raise RuntimeError("pretest release formal contract is not evaluator-trusted")

    code_relative, code_sha = _verify_release_asset(
        release.get("code_asset_manifest"), label="code asset manifest"
    )
    source_relative, source_sha = _verify_release_asset(
        release.get("source_archive"), label="deterministic source archive"
    )
    environment_relative, environment_sha = _verify_release_asset(
        release.get("environment"), label="environment summary"
    )
    code_entries = _parse_frozen_asset_manifest(REPOSITORY_ROOT / code_relative)

    evaluator_relative, evaluator_sha = _verify_release_asset(
        release.get("evaluator"), label="evaluator"
    )
    expected_evaluator_relative = Path(__file__).resolve().relative_to(
        REPOSITORY_ROOT
    )
    if evaluator_relative != expected_evaluator_relative:
        raise RuntimeError("pretest release evaluator path is not canonical")
    if code_entries.get(evaluator_relative.as_posix()) != evaluator_sha:
        raise RuntimeError("evaluator SHA-256 is not bound by code asset manifest")

    finalizer_relative, finalizer_sha = _verify_release_asset(
        release.get("finalizer"), label="finalizer"
    )
    if code_entries.get(finalizer_relative.as_posix()) != finalizer_sha:
        raise RuntimeError("finalizer SHA-256 is not bound by code asset manifest")

    return {
        "path": release_relative.as_posix(),
        "sha256": actual_release_sha,
        "version": release_version,
        "verified_before_test_graph_access": True,
        "formal_contract_path": formal_relative.as_posix(),
        "formal_contract_sha256": formal_sha,
        "code_asset_manifest_path": code_relative.as_posix(),
        "code_asset_manifest_sha256": code_sha,
        "source_archive_path": source_relative.as_posix(),
        "source_archive_sha256": source_sha,
        "environment_path": environment_relative.as_posix(),
        "environment_sha256": environment_sha,
        "evaluator_path": evaluator_relative.as_posix(),
        "evaluator_sha256": evaluator_sha,
        "finalizer_path": finalizer_relative.as_posix(),
        "finalizer_sha256": finalizer_sha,
    }


def _declared_training_assets(
    metadata: dict[str, Any],
    entries: dict[str, str],
    *,
    key: str,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, dict[str, str]]]:
    raw_assets = metadata.get(key)
    if not isinstance(raw_assets, dict):
        raise ValueError(f"formal contract has no declared {key}")
    result: dict[int, dict[str, dict[str, str]]] = {}
    for seed in seeds:
        raw_seed = raw_assets.get(str(seed), raw_assets.get(seed))
        if not isinstance(raw_seed, dict):
            raise ValueError(f"formal contract has no {key} for seed {seed}")
        result[seed] = {}
        for asset_kind in ("result", "checkpoint"):
            record = raw_seed.get(asset_kind)
            if not isinstance(record, dict):
                raise ValueError(
                    f"formal contract has no {key}/{seed}/{asset_kind}"
                )
            path = str(record.get("path", ""))
            digest = str(record.get("sha256", "")).lower()
            if entries.get(path) != digest:
                raise RuntimeError(
                    "formal asset allowlist disagrees with declared training "
                    f"asset: {path}"
                )
            result[seed][asset_kind] = {
                "path": path,
                "sha256": digest,
            }
    return result


def load_frozen_contract(
    contract_path: Path,
    *,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Load one evaluator-trusted, versioned formal contract."""
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"formal contract does not exist: {contract_path}"
        )
    actual_contract_sha = sha256_file(contract_path)
    if (
        expected_contract_sha256 is not None
        and actual_contract_sha != str(expected_contract_sha256).lower()
    ):
        raise RuntimeError(
            "formal contract SHA-256 does not match the requested built-in "
            "trust anchor"
        )
    profile = FORMAL_TRUST_ANCHORS.get(actual_contract_sha)
    if not isinstance(profile, dict):
        raise RuntimeError(
            "formal contract SHA-256 is not in the evaluator's built-in "
            "trust-anchor allowlist"
        )
    with contract_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("status") != profile["status"]:
        raise ValueError("formal contract has an unexpected frozen status")
    if metadata.get("version") != profile["version"]:
        raise ValueError("formal contract has an unexpected method version")
    if metadata.get("frozen_config") != profile["frozen_config"]:
        raise ValueError("formal contract frozen configuration changed")
    seeds = tuple(int(value) for value in metadata.get("training_seeds", []))
    if seeds != FORMAL_TRAINING_SEEDS:
        raise ValueError("formal contract training seed allowlist changed")

    manifest_record = metadata.get("asset_hash_manifest")
    if not isinstance(manifest_record, dict):
        raise ValueError("formal contract has no frozen asset hash manifest")
    recorded_manifest_sha = str(manifest_record.get("sha256", "")).lower()
    expected_asset_manifest_sha = str(
        profile["asset_manifest_sha256"]
    ).lower()
    if recorded_manifest_sha != expected_asset_manifest_sha:
        raise RuntimeError("formal asset manifest digest changed in contract")
    asset_manifest_path = _resolve_contract_recorded_path(
        contract_path, str(manifest_record.get("path", ""))
    )
    actual_manifest_sha = sha256_file(asset_manifest_path)
    if actual_manifest_sha != expected_asset_manifest_sha:
        raise RuntimeError("formal asset manifest SHA-256 mismatch")
    entries = _parse_frozen_asset_manifest(asset_manifest_path)

    clean_assets = _declared_training_assets(
        metadata, entries, key="clean_assets", seeds=seeds
    )
    victim_assets = _declared_training_assets(
        metadata, entries, key="victim_assets", seeds=seeds
    )

    graph_record = metadata.get("graph_manifest")
    poison_record = metadata.get("poison_manifest")
    if not isinstance(graph_record, dict) or not isinstance(poison_record, dict):
        raise ValueError("formal contract is missing graph or poison assets")
    graph_contract_path = str(
        graph_record.get(
            "contract_path", str(graph_record.get("path", "")) + ".metadata.json"
        )
    )
    poison_metadata_path = str(
        poison_record.get(
            "metadata_path", str(poison_record.get("path", "")) + ".metadata.json"
        )
    )
    required_entries = {
        str(graph_record.get("path", "")): str(
            graph_record.get("sha256", "")
        ).lower(),
        str(poison_record.get("path", "")): str(
            poison_record.get("sha256", "")
        ).lower(),
        graph_contract_path: str(
            graph_record.get("contract_sha256", "")
        ).lower(),
        poison_metadata_path: str(
            poison_record.get("metadata_sha256", "")
        ).lower(),
    }
    group_record = metadata.get("group_metadata")
    if isinstance(group_record, dict):
        required_entries.update(
            {
                str(group_record.get("path", "")): str(
                    group_record.get("sha256", "")
                ).lower(),
                str(group_record.get("contract_path", "")): str(
                    group_record.get("contract_sha256", "")
                ).lower(),
            }
        )
    for recorded_path, expected_sha in required_entries.items():
        if entries.get(recorded_path) != expected_sha:
            raise RuntimeError(
                f"formal asset allowlist disagrees with contract: {recorded_path}"
            )

    return {
        "contract_path": str(contract_path),
        "contract_sha256": actual_contract_sha,
        "asset_manifest_path": str(asset_manifest_path),
        "asset_manifest_sha256": actual_manifest_sha,
        "version": str(metadata["version"]),
        "frozen_config": dict(metadata["frozen_config"]),
        "training_seeds": list(seeds),
        "graph_manifest": dict(graph_record),
        "poison_manifest": dict(poison_record),
        "clean_assets": clean_assets,
        "victim_assets": victim_assets,
        "clean_role": str(profile["clean_role"]),
        "victim_role": str(profile["victim_role"]),
        "trust_anchor_profile": str(profile["version"]),
    }


def _verify_frozen_asset(
    path: Path, expected: dict[str, str], *, role: str
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"frozen {role} asset does not exist: {path}")
    actual = sha256_file(path)
    if actual != str(expected["sha256"]).lower():
        raise RuntimeError(
            f"{role} SHA-256 is not in the frozen asset allowlist"
        )
    return actual


def validate_frozen_training_asset(
    result_path: Path,
    checkpoint: Path,
    training_result: dict[str, Any],
    formal_contract: dict[str, Any],
    *,
    expected_role: Literal["clean", "victim"],
) -> dict[str, Any]:
    """Bind one result/checkpoint pair to its exact frozen seed and role."""
    if expected_role not in {"clean", "victim"}:
        raise ValueError(
            "expected_role must be exactly 'clean' or 'victim'"
        )
    config = training_result.get("config")
    if not isinstance(config, dict):
        raise ValueError("frozen training result has no config")
    try:
        seed = int(config.get("seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen training result has an invalid seed") from exc
    allowed_seeds = tuple(
        int(value)
        for value in formal_contract.get(
            "training_seeds", FORMAL_TRAINING_SEEDS
        )
    )
    if seed not in allowed_seeds:
        raise ValueError("training seed is not in the frozen contract allowlist")

    poison_record = training_result.get("poison_manifest")
    is_clean = poison_record is None
    if expected_role == "clean" and not is_clean:
        raise ValueError("clean reference must use a frozen clean checkpoint")
    if expected_role == "victim" and is_clean:
        raise ValueError(
            "victim must use a frozen poisoned checkpoint with a poison manifest"
        )
    asset_role = expected_role
    role_assets = formal_contract[
        "clean_assets" if expected_role == "clean" else "victim_assets"
    ][seed]
    result_sha = _verify_frozen_asset(
        result_path, role_assets["result"], role=f"{asset_role} result"
    )
    checkpoint_sha = _verify_frozen_asset(
        checkpoint,
        role_assets["checkpoint"],
        role=f"{asset_role} checkpoint",
    )

    if expected_role == "victim":
        if not isinstance(poison_record, dict):
            raise ValueError("frozen victim poison record is malformed")
        frozen_poison = formal_contract.get("poison_manifest")
        if not isinstance(frozen_poison, dict):
            raise ValueError("formal contract has no frozen poison manifest")
        if str(poison_record.get("sha256", "")).lower() != str(
            frozen_poison["sha256"]
        ).lower():
            raise RuntimeError("victim poison manifest is not the frozen asset")
        if str(poison_record.get("metadata_sha256", "")).lower() != str(
            frozen_poison["metadata_sha256"]
        ).lower():
            raise RuntimeError("victim poison metadata is not the frozen asset")

    return {
        "role": formal_contract.get(
            "clean_role" if expected_role == "clean" else "victim_role",
            "clean_reference"
            if expected_role == "clean"
            else "v3_poisoned_victim",
        ),
        "seed": seed,
        "result_sha256": result_sha,
        "checkpoint_sha256": checkpoint_sha,
        "frozen_asset_allowlist_match": True,
    }


def validate_frozen_method_request(
    formal_contract: dict[str, Any],
    *,
    target_seed: int,
    orientation_policy: str,
    allocation_policy: str,
    max_graphs: int | None,
    max_targets: int | None,
    graph_manifest_sha256: str,
    split_contract_sha256: str,
    label_config: Any,
) -> dict[str, Any]:
    """Verify every evaluation-time choice frozen by the selected method."""
    frozen = formal_contract["frozen_config"]
    if int(target_seed) != int(frozen["target_seed"]):
        raise ValueError("formal target seed does not match frozen contract")
    if str(orientation_policy) != str(frozen["orientation_policy"]):
        raise ValueError("formal orientation policy does not match frozen contract")
    if str(allocation_policy) != str(frozen["allocation_policy"]):
        raise ValueError("formal allocation policy does not match frozen contract")
    if max_graphs is not None or max_targets is not None:
        raise ValueError("formal evaluation cannot cap graphs or targets")

    graph_record = formal_contract["graph_manifest"]
    if str(graph_manifest_sha256).lower() != str(
        graph_record["sha256"]
    ).lower():
        raise RuntimeError("evaluation graph manifest is not the frozen asset")
    if str(split_contract_sha256).lower() != str(
        graph_record["contract_sha256"]
    ).lower():
        raise RuntimeError("evaluation split contract is not the frozen asset")

    labels = label_config_dict(label_config)
    for key in (
        "label_mode",
        "risk_base_distance_m",
        "risk_reaction_time_s",
        "risk_safe_decel_mps2",
    ):
        if labels.get(key) != frozen[key]:
            raise ValueError(f"formal label configuration mismatch: {key}")
    return {
        "formal_contract_verified": True,
        "target_seed": int(target_seed),
        "orientation_policy": str(orientation_policy),
        "allocation_policy": str(allocation_policy),
        "graph_manifest_sha256": str(graph_manifest_sha256),
        "split_contract_sha256": str(split_contract_sha256),
        "label_config_hash": label_config_hash(label_config),
    }


def validate_formal_training_protocol(
    training_result: dict[str, Any],
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Audit whether a checkpoint was selected under the frozen main protocol.

    The function returns a structured audit instead of raising so development
    validation can still inspect exploratory checkpoints. Test evaluation
    fails closed when this audit is not eligible.
    """
    violations: list[str] = []
    config = training_result.get("config")
    if not isinstance(config, dict):
        config = {}
        violations.append("missing_config")

    for key in ("max_train_graphs", "max_val_graphs", "max_test_graphs"):
        if config.get(key) is not None:
            violations.append(f"{key}_must_be_none")
    if config.get("evaluate_test") is not False:
        violations.append("training_must_not_evaluate_test")
    if config.get("checkpoint_metric") != FORMAL_CHECKPOINT_METRIC:
        violations.append("checkpoint_metric_must_be_val_loss")
    if config.get("require_strict_label") is not True:
        violations.append("require_strict_label_must_be_true")
    try:
        epochs = int(config.get("epochs"))
    except (TypeError, ValueError):
        epochs = -1
    if epochs != FORMAL_TRAINING_EPOCHS:
        violations.append(
            f"epochs_must_equal_{FORMAL_TRAINING_EPOCHS}"
        )

    protocol = training_result.get("training_protocol")
    if not isinstance(protocol, dict):
        protocol = {}
        violations.append("missing_training_protocol")
    expected_protocol = {
        "objective": "ordinary_binary_cross_entropy_symmetric_pair_labels",
        "initialization": "random",
        "checkpoint_metric": FORMAL_CHECKPOINT_METRIC,
        "checkpoint_data": "clean_validation_only",
        "training_asr_computed": False,
        "test_evaluated": False,
        "teacher_used": False,
        "replay_used": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            violations.append(f"training_protocol_{key}_mismatch")

    for key in ("test_metrics", "test_pair_metrics", "test_stats"):
        if training_result.get(key) is not None:
            violations.append(f"{key}_must_be_null")
    losses = training_result.get("losses")
    if not isinstance(losses, dict) or losses.get("test") is not None:
        violations.append("test_loss_must_be_null")

    expected_split_counts = {
        split: int((manifest["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    recorded_split_counts = training_result.get("split_graph_counts")
    if not isinstance(recorded_split_counts, dict):
        violations.append("missing_split_graph_counts")
    else:
        for split, expected in expected_split_counts.items():
            try:
                recorded = int(recorded_split_counts.get(split, -1))
            except (TypeError, ValueError):
                recorded = -1
            if recorded != expected:
                violations.append(f"split_graph_count_{split}_mismatch")

    for split in ("train", "val"):
        stats = training_result.get(f"{split}_stats")
        if not isinstance(stats, dict):
            violations.append(f"missing_{split}_stats")
            continue
        expected = expected_split_counts[split]
        for key in ("graphs_considered", "graph_sha256_verified"):
            try:
                recorded = int(stats.get(key, -1))
            except (TypeError, ValueError):
                recorded = -1
            if recorded != expected:
                violations.append(f"{split}_{key}_mismatch")

    poison_record = training_result.get("poison_manifest")
    train_stats = training_result.get("train_stats")
    configured_poison = config.get("poison_manifest")
    if poison_record is None:
        if configured_poison is not None:
            violations.append("config_poison_manifest_without_asset_record")
        if isinstance(train_stats, dict):
            if int(train_stats.get("poisoned_graphs", -1)) != 0:
                violations.append("clean_training_has_poisoned_graphs")
            if int(train_stats.get("poisoned_edges", -1)) != 0:
                violations.append("clean_training_has_poisoned_edges")
    elif isinstance(poison_record, dict):
        if configured_poison is None:
            violations.append("poison_asset_missing_from_config")
        try:
            poison_rows = int(poison_record.get("rows", -1))
        except (TypeError, ValueError):
            poison_rows = -1
        if poison_rows <= 0:
            violations.append("poison_manifest_rows_must_be_positive")
        if not isinstance(train_stats, dict):
            violations.append("missing_train_stats_for_poisoned_model")
        else:
            if int(train_stats.get("poisoned_graphs", -1)) != poison_rows:
                violations.append("poisoned_graph_count_mismatch")
            if int(train_stats.get("poisoned_edges", -1)) != 2 * poison_rows:
                violations.append("poisoned_edge_count_mismatch")
    else:
        violations.append("malformed_poison_manifest_record")

    return {
        "formal_training_protocol_verified": not violations,
        "violations": violations,
        "expected_epochs": FORMAL_TRAINING_EPOCHS,
        "expected_checkpoint_metric": FORMAL_CHECKPOINT_METRIC,
        "expected_split_graph_counts": expected_split_counts,
    }


def _resolve_recorded_checkpoint(
    result_path: Path, checkpoint_record: dict[str, Any]
) -> Path:
    recorded = Path(str(checkpoint_record.get("path", "")))
    candidates = [recorded, result_path.parent / recorded.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "clean reference checkpoint does not exist: "
        f"recorded={recorded}, result={result_path}"
    )


def validate_clean_reference_binding(
    clean_result_path: Path,
    evaluated_model_training_result: dict[str, Any],
    graph_manifest_path: Path,
    manifest: pd.DataFrame,
    formal_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a clean result and bind its validation pair threshold to assets."""
    if not clean_result_path.is_file():
        raise FileNotFoundError(
            f"clean reference result does not exist: {clean_result_path}"
        )
    with clean_result_path.open(encoding="utf-8") as stream:
        clean_result = json.load(stream)
    if clean_result.get("poison_manifest") is not None:
        raise ValueError("clean reference result must not use a poison manifest")
    clean_label_config = config_from_result(clean_result)
    evaluated_model_label_config = config_from_result(
        evaluated_model_training_result
    )
    if label_config_hash(clean_label_config) != label_config_hash(
        evaluated_model_label_config
    ):
        raise ValueError(
            "clean reference label configuration does not match evaluated model"
        )

    evaluated_model_config = evaluated_model_training_result.get("config")
    clean_config = clean_result.get("config")
    if not isinstance(evaluated_model_config, dict) or not isinstance(
        clean_config, dict
    ):
        raise ValueError("evaluated model or clean reference result has no config")
    mismatched_config = [
        key
        for key in CLEAN_REFERENCE_MATCH_CONFIG_KEYS
        if clean_config.get(key) != evaluated_model_config.get(key)
    ]
    if mismatched_config:
        raise ValueError(
            "clean reference training configuration does not match evaluated model: "
            + ", ".join(mismatched_config)
        )
    if clean_result.get("model_config") != evaluated_model_training_result.get(
        "model_config"
    ):
        raise ValueError(
            "clean reference model configuration does not match evaluated model"
        )
    checkpoint_record = clean_result.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise ValueError("clean reference result has no checkpoint binding")
    clean_checkpoint = _resolve_recorded_checkpoint(
        clean_result_path, checkpoint_record
    )
    asset_audit = validate_training_asset_binding(
        clean_checkpoint,
        clean_result,
        graph_manifest_path,
    )
    protocol_audit = validate_formal_training_protocol(clean_result, manifest)
    if not protocol_audit["formal_training_protocol_verified"]:
        raise RuntimeError(
            "clean reference violates the formal training protocol: "
            + ", ".join(protocol_audit["violations"])
        )
    frozen_asset_audit = (
        validate_frozen_training_asset(
            clean_result_path,
            clean_checkpoint,
            clean_result,
            formal_contract,
            expected_role="clean",
        )
        if formal_contract is not None
        else None
    )

    pair_metrics = clean_result.get("val_pair_metrics")
    if not isinstance(pair_metrics, dict) or "threshold" not in pair_metrics:
        raise ValueError("clean reference has no validation pair threshold")
    threshold = float(pair_metrics["threshold"])
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("clean reference validation pair threshold is invalid")

    return {
        "result_path": str(clean_result_path),
        "result_sha256": sha256_file(clean_result_path),
        "checkpoint_path": str(clean_checkpoint),
        "checkpoint_sha256": asset_audit["checkpoint_sha256"],
        "checkpoint_matches_result": True,
        "graph_manifest_sha256": asset_audit["graph_manifest_sha256"],
        "graph_manifest_matches_evaluation": True,
        "split_contract_sha256": asset_audit["split_contract_sha256"],
        "label_config_hash": str(clean_result["label_config_hash"]),
        "seed": int(clean_config["seed"]),
        "validation_pair_threshold": threshold,
        "threshold_source": "bound_clean_reference_validation_pair_metrics",
        "frozen_asset_verified": frozen_asset_audit is not None,
        "frozen_asset_audit": frozen_asset_audit,
        "matched_training_config_fields": list(
            CLEAN_REFERENCE_MATCH_CONFIG_KEYS
        ),
        "training_protocol": protocol_audit,
    }


def evaluation_completion_flags(
    *,
    split: str,
    max_graphs: int | None,
    max_targets: int | None,
    target_rows: int,
    score_rows: int,
    pair_threshold_available: bool,
    assets_verified: bool,
    training_protocol_verified: bool,
    clean_reference_bound: bool,
    formal_contract_verified: bool,
    evaluated_model_role_verified: bool,
    pretest_release_verified: bool = True,
    metric_evidence_complete: bool = True,
) -> dict[str, bool]:
    """Separate a complete development run from a formal frozen test."""
    evaluation_complete = bool(
        max_graphs is None
        and max_targets is None
        and int(target_rows) > 0
        and int(score_rows) == int(target_rows)
        and assets_verified
        and metric_evidence_complete
    )
    return {
        "evaluation_complete": evaluation_complete,
        "formal_complete": bool(
            split == "test"
            and evaluation_complete
            and pair_threshold_available
            and training_protocol_verified
            and clean_reference_bound
            and formal_contract_verified
            and evaluated_model_role_verified
            and pretest_release_verified
        ),
    }


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        # Probabilities are produced as float32.  Pandas' default float32 CSV
        # representation can shorten a value enough to move an exactly
        # threshold-equal prediction to the other side after a read/write
        # round trip.  Seventeen significant digits preserve the promoted
        # IEEE-754 value used by the JSON metrics and make metric evidence
        # independently reproducible at threshold boundaries.
        frame.to_csv(
            temporary,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_training_evaluation_allocation_binding(
    training_result: dict[str, Any],
    evaluation_policy: str,
) -> dict[str, Any]:
    """Keep surrogate-guided training separate from data-only evaluation."""
    evaluation_policy = str(evaluation_policy)
    if evaluation_policy not in EVALUATION_ALLOCATION_POLICIES:
        raise RuntimeError(
            "evaluation allocation policy must be model-independent; "
            "cross-fit surrogate allocation is training-only"
        )

    poison_record = training_result.get("poison_manifest")
    if not isinstance(poison_record, dict):
        return {
            "training_allocation_policy": None,
            "evaluation_allocation_policy": evaluation_policy,
            "binding_mode": "clean_reference_data_only_evaluation",
            "surrogate_used_for_evaluation_target_selection": False,
        }

    training_policy = str(
        poison_record.get(
            "allocation_policy",
            ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
        )
    )
    if training_policy == ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4:
        if evaluation_policy not in {
            ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        }:
            raise RuntimeError(
                "cross-fit surrogate training must use an independent, "
                "model-independent evaluation allocation policy"
            )
        return {
            "training_allocation_policy": training_policy,
            "evaluation_allocation_policy": evaluation_policy,
            "binding_mode": (
                "crossfit_training_with_independent_data_only_evaluation"
            ),
            "surrogate_used_for_evaluation_target_selection": False,
        }

    if evaluation_policy != training_policy:
        raise RuntimeError(
            "evaluation allocation policy does not match poison training"
        )
    return {
        "training_allocation_policy": training_policy,
        "evaluation_allocation_policy": evaluation_policy,
        "binding_mode": "matched_training_and_data_only_evaluation",
        "surrogate_used_for_evaluation_target_selection": False,
    }


def validate_evaluation_phase_allocation_policy(
    split: str,
    allocation_policy: str,
    formal_contract_path: str | Path | None = None,
) -> None:
    """Allow fixed-symmetric test evaluation only under its v6 contract."""
    if (
        str(allocation_policy)
        == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
        and str(split) != "val"
    ):
        contract = Path(str(formal_contract_path or ""))
        if not contract.is_file():
            raise RuntimeError(
                "fixed_symmetric_biend_v1 test evaluation requires the "
                "frozen v6 formal contract"
            )
        profile = FORMAL_TRUST_ANCHORS.get(sha256_file(contract))
        frozen = profile.get("frozen_config", {}) if isinstance(profile, dict) else {}
        if frozen.get("allocation_policy") != (
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
        ):
            raise RuntimeError(
                "fixed_symmetric_biend_v1 test evaluation requires the "
                "frozen v6 formal contract"
            )


def training_access_context(training_result: dict[str, Any], *, split: str,
                            model_role: str) -> dict[str, Any]:
    """Report training intervention separately from zero-query target selection.

    Missing/unrecognized provenance is unknown, not an assertion of data-only
    access. Clean-reference optimization is not an attacker intervention.
    """
    config = training_result.get("config") or {}
    protocol = training_result.get("training_protocol") or {}
    projection = protocol.get("gradient_projection") or {}
    objective = str(protocol.get("objective", ""))
    explicit = (
        projection.get("enabled") is True
        or config.get("presentation") == "explicit_counterfactual"
        or "counterfactual" in objective
    )
    ordinary = objective == "ordinary_binary_cross_entropy_symmetric_pair_labels"
    control = True if explicit else False if ordinary else None
    validation_selected = (
        protocol.get("checkpoint_data") == "clean_validation_only"
        or config.get("checkpoint_metric") in {"val_loss", "val_auc"}
    )
    independent = (
        False if split == "val" and validation_selected else
        True if split == "test" and validation_selected
        and config.get("evaluate_test") is False else None
    )
    if model_role == "clean_reference":
        threat_model = "clean_reference_response_baseline"
        attacker_gradients = False
    else:
        threat_model = (
            "training_procedure_control" if control is True else
            "data_only_poisoning" if control is False else "unspecified_training_access"
        )
        attacker_gradients = control
    return {
        "metadata_version": "training_access_context_v1",
        "threat_model": threat_model,
        "evaluated_model_training_objective": objective or None,
        "evaluated_model_objective_profile": config.get("objective_profile"),
        "evaluated_model_training_procedure_controlled": control,
        "victim_structure_parameters_or_gradients_used_by_attacker": attacker_gradients,
        "clean_validation_used_for_checkpoint_selection": True if validation_selected else None,
        "evaluation_split_not_used_for_training_checkpoint_selection": independent,
        "training_asr_checkpoint_selection": (
            False if protocol.get("checkpoint_data") == "clean_validation_only" else None
        ),
        "interpretation": (
            "development_validation_clean_split_used_for_checkpoint_selection"
            if split == "val" and validation_selected else
            "see_recorded_training_and_evaluation_protocols"
        ),
    }


def load_expected_target_manifest(path: Path, expected_sha256: str) -> pd.DataFrame:
    if sha256_file(path) != expected_sha256:
        raise ValueError("public target manifest SHA-256 mismatch")
    frame = pd.read_csv(path, dtype={
        name: str for name in ("scenario_id", "src_track_id", "dst_track_id")
    }, keep_default_na=False)
    if frame.empty or "scenario_id" not in frame or frame["scenario_id"].duplicated().any():
        raise ValueError("public target manifest must have unique nonempty scenarios")
    if (frame["scenario_id"] == "").any():
        raise ValueError("public target manifest contains an empty scenario identity")
    return frame


def validate_public_target_pool(observed: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Match all target identities exactly; tolerate only float roundtrip noise."""
    if set(observed.columns) != set(expected.columns) or len(observed) != len(expected):
        raise ValueError("public target pool schema or row count differs")
    if observed.empty or observed["scenario_id"].duplicated().any():
        raise ValueError("generated public target pool is empty or has duplicate scenarios")
    left = observed.sort_values("scenario_id").reset_index(drop=True)
    right = expected.sort_values("scenario_id").reset_index(drop=True)
    floating_columns = {
        "orientation_destination_mean_speed_mps", "src_endpoint_speed_mps",
        "dst_endpoint_speed_mps", "min_endpoint_speed_mps",
        "allocation_total_feature_energy", "allocation_incident_edge_energy",
        "allocation_endpoint_node_energy",
    }
    for column in expected.columns:
        if column in floating_columns:
            left_values = left[column].to_numpy(dtype=float)
            right_values = right[column].to_numpy(dtype=float)
            matches = np.isfinite(left_values).all() and np.isfinite(right_values).all() and np.allclose(
                left_values, right_values,
                rtol=1e-12, atol=1e-12, equal_nan=False,
            )
        else:
            matches = np.array_equal(left[column].to_numpy(), right[column].to_numpy())
        if not matches:
            raise ValueError(f"public target pool differs at column {column}")


def main() -> None:
    args = parse_args()
    expected_targets = None
    public_pool_binding = None
    if bool(args.expected_target_manifest) != bool(args.expected_target_manifest_sha256):
        raise ValueError("public target manifest requires both path and SHA-256")
    if args.expected_target_manifest:
        if args.split != "val" or args.max_graphs is not None or args.max_targets is not None:
            raise ValueError("public pool binding requires full development validation")
        expected_targets = load_expected_target_manifest(
            Path(args.expected_target_manifest), args.expected_target_manifest_sha256
        )
    validate_evaluation_phase_allocation_policy(
        str(args.split),
        str(args.allocation_policy),
        args.formal_contract,
    )
    if args.split == "test" and args.clean_reference_result is None:
        raise ValueError(
            "test evaluation requires --clean-reference-result; "
            "--common-threshold is development-only"
        )
    if args.split == "test" and args.experimental_trigger_schedule is not None:
        raise ValueError("experimental trigger schedules are validation-only")
    if args.split == "test" and (
        args.pretest_release is None
        or args.pretest_release_sha256 is None
    ):
        raise ValueError(
            "test evaluation requires --pretest-release and "
            "--pretest-release-sha256"
        )
    pretest_release = (
        load_pretest_release(
            Path(args.pretest_release),
            expected_sha256=str(args.pretest_release_sha256),
            expected_formal_contract=Path(args.formal_contract),
        )
        if args.pretest_release is not None
        and args.pretest_release_sha256 is not None
        else None
    )
    formal_contract = (
        load_frozen_contract(Path(args.formal_contract))
        if args.split == "test"
        else None
    )
    expected_frozen_role: Literal["clean", "victim"] = (
        "clean"
        if args.evaluation_model_role == "clean_reference"
        else "victim"
    )
    checkpoint = Path(args.checkpoint)
    result_path = checkpoint.parent / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"checkpoint training result does not exist: {result_path}"
        )
    with result_path.open(encoding="utf-8") as stream:
        training_result = json.load(stream)
    access_context = training_access_context(
        training_result, split=args.split, model_role=args.evaluation_model_role
    )
    edge_feature_mode = validate_edge_feature_mode(
        training_result.get("config", {}).get(
            "edge_feature_mode", BASE_EDGE_FEATURE_MODE
        )
    )
    recorded_schedule = training_result.get("config", {}).get(
        "experimental_trigger_schedule"
    )
    if args.experimental_trigger_schedule is not None:
        if edge_feature_mode != MOTION_NORMALIZED_EDGE_FEATURE_MODE:
            raise ValueError(
                "the motion-regime schedule requires the 42-dimensional "
                "motion-normalized edge representation"
            )
        if str(args.allocation_policy) != (
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
        ):
            raise ValueError(
                "the motion-regime schedule requires fixed symmetric allocation"
            )
        if (
            args.evaluation_model_role == "victim"
            and str(recorded_schedule) != str(
                args.experimental_trigger_schedule
            )
        ):
            raise RuntimeError(
                "victim training and evaluation trigger schedules differ"
            )
    elif recorded_schedule is not None:
        raise RuntimeError(
            "scheduled victim evaluation requires "
            "--experimental-trigger-schedule"
        )
    allocation_policy_binding = (
        validate_training_evaluation_allocation_binding(
            training_result,
            str(args.allocation_policy),
        )
    )
    label_config = config_from_result(training_result)
    recorded_label_hash = str(training_result.get("label_config_hash", ""))
    if recorded_label_hash != label_config_hash(label_config):
        raise RuntimeError(
            "training result label_config_hash does not match label_config"
        )
    require_strict = bool(training_result.get("config", {}).get("require_strict_label", False))
    directed_threshold = float(training_result["val_metrics"]["threshold"])
    pair_threshold_record = training_result.get("val_pair_metrics")
    if isinstance(pair_threshold_record, dict) and "threshold" in pair_threshold_record:
        threshold = float(pair_threshold_record["threshold"])
        model_threshold_source = "evaluated_model_clean_validation_pair_metrics"
    else:
        # Compatibility for exploratory checkpoints created before symmetric
        # pair calibration was implemented. New formal runs must have
        # val_pair_metrics.
        threshold = directed_threshold
        model_threshold_source = "legacy_directed_validation_fallback"
    graph_dir = Path(args.graph_dir)
    graph_manifest_path = resolve_manifest_path(graph_dir, args.graph_manifest)
    asset_binding = validate_training_asset_binding(
        checkpoint,
        training_result,
        graph_manifest_path,
    )
    manifest = load_manifest(
        graph_dir,
        graph_manifest_path,
        require_graph_sha256=True,
    )
    training_protocol_audit = validate_formal_training_protocol(
        training_result, manifest
    )
    if (
        args.split == "test"
        and not training_protocol_audit[
            "formal_training_protocol_verified"
        ]
    ):
        raise RuntimeError(
            "evaluated checkpoint violates the formal training protocol: "
            + ", ".join(training_protocol_audit["violations"])
        )

    evaluated_model_frozen_asset_audit = (
        validate_frozen_training_asset(
            result_path,
            checkpoint,
            training_result,
            formal_contract,
            expected_role=expected_frozen_role,
        )
        if formal_contract is not None
        else None
    )
    formal_method_audit = (
        validate_frozen_method_request(
            formal_contract,
            target_seed=int(args.target_seed),
            orientation_policy=str(args.orientation_policy),
            allocation_policy=str(args.allocation_policy),
            max_graphs=args.max_graphs,
            max_targets=args.max_targets,
            graph_manifest_sha256=asset_binding["graph_manifest_sha256"],
            split_contract_sha256=asset_binding["split_contract_sha256"],
            label_config=label_config,
        )
        if formal_contract is not None
        else {"formal_contract_verified": False}
    )

    clean_reference_binding: dict[str, Any] | None = None
    common_threshold: float | None = None
    if args.clean_reference_result is not None:
        clean_reference_binding = validate_clean_reference_binding(
            Path(args.clean_reference_result),
            training_result,
            graph_manifest_path,
            manifest,
            formal_contract,
        )
        common_threshold = float(
            clean_reference_binding["validation_pair_threshold"]
        )
    elif args.common_threshold is not None:
        common_threshold = float(args.common_threshold)
        if not np.isfinite(common_threshold) or not 0.0 < common_threshold < 1.0:
            raise ValueError(
                "--common-threshold must be finite and strictly between 0 and 1"
            )

    split_rows = manifest[manifest["split"] == args.split].sort_values("scenario_id")
    if args.max_graphs is not None:
        split_rows = split_rows.head(args.max_graphs)
    if split_rows.empty:
        raise RuntimeError(f"no {args.split} graphs are available")
    # Verify every graph byte-for-byte against the frozen manifest before
    # target construction or model inference. Reuse the resolved paths below
    # so each file is hashed exactly once per evaluation.
    resolved_graph_paths: dict[str, Path] = {}
    for row in split_rows.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        resolved_graph_paths[scenario_id] = verified_graph_path(
            row,
            graph_dir,
            require_graph_sha256=True,
        )

    output_path = Path(args.output)
    target_manifest_path = output_path.with_suffix(".targets.csv")
    score_path = output_path.with_suffix(".scores.csv")
    clean_edge_score_path = output_path.with_suffix(".clean_edges.csv")
    pair_score_path = output_path.with_suffix(".pair_scores.csv")
    collateral_transition_path = output_path.with_suffix(
        ".collateral_transitions.csv"
    )
    existing_outputs = [
        path
        for path in (
            output_path,
            target_manifest_path,
            score_path,
            clean_edge_score_path,
            pair_score_path,
            collateral_transition_path,
        )
        if path.exists()
    ]
    if existing_outputs and not args.force:
        raise FileExistsError(
            "evaluation output already exists; use a new path or pass --force: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    # Pass 1 freezes the entire data-defined pool before the checkpoint is
    # loaded or any evaluated-model prediction is computed.
    target_records: list[dict[str, Any]] = []
    for row in split_rows.itertuples(index=False):
        if args.max_targets is not None and len(target_records) >= args.max_targets:
            break
        with np.load(
            resolved_graph_paths[str(row.scenario_id)], allow_pickle=True
        ) as graph:
            label_bundle = labels_for_graph(graph, label_config)
            labels = label_bundle["edge_label"].astype(np.int8)
            edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
            pair_groups = eligible_negative_pair_groups(
                graph,
                label_config,
                require_strict_label=require_strict,
                perturb_window=10,
                require_bi_endpoint_contiguous=(
                    str(args.allocation_policy)
                    == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
                ),
                label_bundle=label_bundle,
            )
            if not pair_groups:
                continue
            edge_id, orientation_audit = _scenario_target_pair_choice(
                str(row.scenario_id),
                pair_groups,
                int(args.target_seed),
                graph=graph,
                orientation_policy=str(args.orientation_policy),
            )
            src = int(edge_index[0, edge_id])
            dst = int(edge_index[1, edge_id])
            reverse_id = reverse_edge_id(graph, src, dst)
            target_record = {
                "scenario_id": str(row.scenario_id),
                "city": str(row.city),
                "edge_id": int(edge_id),
                "reverse_edge_id": int(reverse_id),
                "src": src,
                "dst": dst,
                "src_track_id": str(graph["node_track_ids"][src]),
                "dst_track_id": str(graph["node_track_ids"][dst]),
                "orientation_policy": str(args.orientation_policy),
                "orientation_destination_mean_speed_mps": float(
                    orientation_audit["destination_mean_speed_mps"]
                ),
                "orientation_candidate_count": int(
                    orientation_audit["candidate_orientations"]
                ),
                "true_label": 0,
                "label_mode": label_config.label_mode,
                "label_config_hash": label_config_hash(label_config),
            }
            if edge_feature_mode == MOTION_NORMALIZED_EDGE_FEATURE_MODE:
                endpoint_speeds = np.linalg.norm(
                    np.asarray(
                        graph["observed_velocities_filled"],
                        dtype=np.float64,
                    )[[src, dst], -1],
                    axis=1,
                )
                minimum_endpoint_speed = float(endpoint_speeds.min())
                target_record.update(
                    {
                        "src_endpoint_speed_mps": float(endpoint_speeds[0]),
                        "dst_endpoint_speed_mps": float(endpoint_speeds[1]),
                        "min_endpoint_speed_mps": minimum_endpoint_speed,
                        "motion_regime": (
                            "slow_lt_0p5"
                            if minimum_endpoint_speed < 0.5
                            else "moving_ge_0p5"
                        ),
                    }
                )
            if str(args.allocation_policy) in {
                ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
                ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
            }:
                if (
                    str(args.allocation_policy)
                    == ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2
                ):
                    allocation_alpha, allocation_audit = (
                        select_bi_endpoint_allocation(
                            graph,
                            src=src,
                            dst=dst,
                            displacement_m=0.2,
                        )
                    )
                else:
                    allocation_alpha, allocation_audit = (
                        fixed_symmetric_bi_endpoint_allocation(
                            graph,
                            src=src,
                            dst=dst,
                            displacement_m=0.2,
                        )
                    )
                target_record.update(
                    {
                        "allocation_policy": str(args.allocation_policy),
                        "allocation_alpha": float(allocation_alpha),
                        "allocation_total_feature_energy": float(
                            allocation_audit[
                                "allocation_total_feature_energy"
                            ]
                        ),
                        "allocation_incident_edge_energy": float(
                            allocation_audit[
                                "allocation_incident_edge_energy"
                            ]
                        ),
                        "allocation_endpoint_node_energy": float(
                            allocation_audit[
                                "allocation_endpoint_node_energy"
                            ]
                        ),
                        "allocation_candidate_count": int(
                            allocation_audit["allocation_candidate_count"]
                        ),
                        "allocation_non_target_incident_edges": int(
                            allocation_audit[
                                "allocation_non_target_incident_edges"
                            ]
                        ),
                    }
                )
            target_records.append(target_record)
    target_columns = [
            "scenario_id",
            "city",
            "edge_id",
            "reverse_edge_id",
            "src",
            "dst",
            "src_track_id",
            "dst_track_id",
            "orientation_policy",
            "orientation_destination_mean_speed_mps",
            "orientation_candidate_count",
            "true_label",
            "label_mode",
            "label_config_hash",
    ]
    if edge_feature_mode == MOTION_NORMALIZED_EDGE_FEATURE_MODE:
        target_columns.extend(
            [
                "src_endpoint_speed_mps",
                "dst_endpoint_speed_mps",
                "min_endpoint_speed_mps",
                "motion_regime",
            ]
        )
    if str(args.allocation_policy) in {
        ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
        ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    }:
        target_columns.extend(
            [
                "allocation_policy",
                "allocation_alpha",
                "allocation_total_feature_energy",
                "allocation_incident_edge_energy",
                "allocation_endpoint_node_energy",
                "allocation_candidate_count",
                "allocation_non_target_incident_edges",
            ]
        )
    target_frame = pd.DataFrame(target_records, columns=target_columns)
    if expected_targets is not None:
        # Still before constructing/loading either evaluated model.
        validate_public_target_pool(target_frame, expected_targets)
        if sha256_file(Path(args.expected_target_manifest)) != args.expected_target_manifest_sha256:
            raise RuntimeError("public target manifest changed during target construction")
        public_pool_binding = {
            "path": str(Path(args.expected_target_manifest).resolve()),
            "sha256": str(args.expected_target_manifest_sha256),
            "rows": int(len(expected_targets)),
            "all_identities_match": True,
            "verified_before_model_inference": True,
            "comparison": "exact_identity_and_allocation_with_1e-12_float_roundtrip_tolerance",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv_write(target_frame, target_manifest_path)
    target_manifest_hash = sha256_file(target_manifest_path)
    targets_by_scenario = (
        {
            str(row.scenario_id): (
                int(row.edge_id),
                int(row.reverse_edge_id),
                float(getattr(row, "allocation_alpha", 0.0)),
            )
            for row in target_frame.itertuples(index=False)
        }
        if not target_frame.empty
        else {}
    )
    target_motion_by_scenario = (
        {
            str(row.scenario_id): {
                "src_endpoint_speed_mps": float(row.src_endpoint_speed_mps),
                "dst_endpoint_speed_mps": float(row.dst_endpoint_speed_mps),
                "min_endpoint_speed_mps": float(row.min_endpoint_speed_mps),
                "motion_regime": str(row.motion_regime),
            }
            for row in target_frame.itertuples(index=False)
        }
        if (
            edge_feature_mode == MOTION_NORMALIZED_EDGE_FEATURE_MODE
            and not target_frame.empty
        )
        else {}
    )

    # Only after the target manifest is frozen do we construct the evaluated
    # model (victim or clean-reference baseline).
    first_path = resolved_graph_paths[str(split_rows.iloc[0]["scenario_id"])]
    with np.load(first_path, allow_pickle=True) as first_graph:
        node_dim = int(first_graph["x_node"].shape[1])
        edge_dim = int(
            edge_features_for_mode(
                first_graph["edge_attr"],
                first_graph["edge_index"],
                first_graph["observed_positions_filled"],
                first_graph["observed_velocities_filled"],
                first_graph["observed_valid_mask"],
                mode=edge_feature_mode,
            ).shape[1]
        )
    model_kwargs = model_kwargs_from_result(training_result, node_dim, edge_dim)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_model(**model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    trigger_spec = TriggerSpec(
        perturb_window=10,
        ramp_style="minimum_jerk",
        velocity_mode="residual",
        require_contiguous_valid=True,
    )
    schedule_application_counts: dict[str, int] = defaultdict(int)
    clean_labels_parts: list[np.ndarray] = []
    clean_probs_parts: list[np.ndarray] = []
    clean_pair_labels_parts: list[np.ndarray] = []
    clean_pair_probs_parts: list[np.ndarray] = []
    clean_edge_scenario_ids: list[str] = []
    clean_edge_cities: list[str] = []
    clean_edge_ids_parts: list[np.ndarray] = []
    pair_score_records: list[dict[str, Any]] = []
    collateral_transition_records: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    common_counts: dict[str, int] | None = (
        defaultdict(int) if common_threshold is not None else None
    )
    deltas: list[float] = []
    score_records: list[dict[str, Any]] = []
    stealth_max: dict[str, float] = defaultdict(float)
    per_city: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in split_rows.itertuples(index=False):
        with np.load(
            resolved_graph_paths[str(row.scenario_id)], allow_pickle=True
        ) as graph:
            label_bundle = labels_for_graph(graph, label_config)
            labels = label_bundle["edge_label"].astype(np.int8)
            selected_computable = selected_label_computable_mask(
                graph, label_config, label_bundle
            )
            supervised = supervision_mask(graph, require_strict)
            supervised &= selected_computable
            clean_edge_attr = edge_features_for_mode(
                graph["edge_attr"],
                graph["edge_index"],
                graph["observed_positions_filled"],
                graph["observed_velocities_filled"],
                graph["observed_valid_mask"],
                mode=edge_feature_mode,
            )
            clean_probs = _probabilities(
                model,
                _data(graph, labels, edge_attr=clean_edge_attr),
                device,
            )
            supervised_edge_ids = np.flatnonzero(supervised).astype(np.int64)
            clean_labels_parts.append(labels[supervised_edge_ids])
            clean_probs_parts.append(clean_probs[supervised_edge_ids])
            clean_edge_ids_parts.append(supervised_edge_ids)
            clean_edge_scenario_ids.extend(
                [str(row.scenario_id)] * int(supervised_edge_ids.size)
            )
            clean_edge_cities.extend([str(row.city)] * int(supervised_edge_ids.size))
            edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
            pair_records = _supervised_pair_records(
                edge_index, labels, clean_probs, supervised
            )
            if pair_records:
                pair_probs = np.asarray(
                    [record[3] for record in pair_records], dtype=np.float32
                )
                pair_labels = np.asarray(
                    [record[2] for record in pair_records], dtype=np.int8
                )
                clean_pair_probs_parts.append(pair_probs)
                clean_pair_labels_parts.append(pair_labels)
                pair_score_records.extend(
                    {
                        "scenario_id": str(row.scenario_id),
                        "city": str(row.city),
                        "canonical_edge_id": int(record[0]),
                        "reverse_edge_id": int(record[1]),
                        "label": int(record[2]),
                        "pair_probability": float(record[3]),
                        "evaluated_model_threshold": float(threshold),
                        "common_threshold": (
                            float(common_threshold)
                            if common_threshold is not None
                            else None
                        ),
                    }
                    for record in pair_records
                )
            city = str(row.city)

            target_pair = targets_by_scenario.get(str(row.scenario_id))
            counts["data_defined_test_targets"] += int(target_pair is not None)
            per_city[city]["targets"] += int(target_pair is not None)

            if target_pair is not None:
                edge_id, expected_reverse_id, allocation_alpha = target_pair
                src = int(edge_index[0, edge_id])
                dst = int(edge_index[1, edge_id])
                actual_reverse_id = reverse_edge_id(graph, src, dst)
                if actual_reverse_id != expected_reverse_id:
                    raise RuntimeError("frozen target reverse edge identity changed")
                if args.experimental_trigger_schedule is None:
                    x_node, edge_attr, transformed_target, audit = (
                        apply_trajectory_trigger(
                            graph,
                            src=src,
                            dst=dst,
                            displacement_m=0.2,
                            allocation_alpha=float(allocation_alpha),
                            spec=trigger_spec,
                            edge_feature_mode=edge_feature_mode,
                        )
                    )
                else:
                    motion_record = target_motion_by_scenario.get(
                        str(row.scenario_id)
                    )
                    if not isinstance(motion_record, dict):
                        raise RuntimeError(
                            "scheduled evaluation target lacks a motion regime"
                        )
                    motion_regime = str(motion_record["motion_regime"])
                    x_node, edge_attr, transformed_target, audit = (
                        apply_scheduled_trajectory_trigger(
                            graph,
                            src=src,
                            dst=dst,
                            displacement_m=0.2,
                            allocation_alpha=float(allocation_alpha),
                            motion_regime=motion_regime,
                            schedule_id=str(
                                args.experimental_trigger_schedule
                            ),
                            edge_feature_mode=edge_feature_mode,
                        )
                    )
                    schedule_application_counts[
                        f"{motion_regime}:K{scheduled_window(motion_regime)}"
                    ] += 1
                if int(transformed_target.sum()) != 1 or not transformed_target[edge_id]:
                    raise RuntimeError("single-target transform did not preserve target identity")
                pair_target_mask = transformed_target.copy()
                pair_target_mask[actual_reverse_id] = True
                triggered_probs = _probabilities(
                    model, _data(graph, labels, x_node=x_node, edge_attr=edge_attr), device
                )
                clean_target = _pair_probability(
                    clean_probs, edge_id, actual_reverse_id
                )
                triggered_target = _pair_probability(
                    triggered_probs, edge_id, actual_reverse_id
                )
                deltas.append(triggered_target - clean_target)
                counts["target_triggered_positive"] += int(triggered_target >= threshold)
                counts["target_clean_negative"] += int(clean_target < threshold)
                counts["target_incremental_flip"] += int(
                    clean_target < threshold <= triggered_target
                )
                per_city[city]["triggered_positive"] += int(triggered_target >= threshold)
                per_city[city]["incremental_flip"] += int(
                    clean_target < threshold <= triggered_target
                )
                score_records.append(
                    {
                        "scenario_id": str(row.scenario_id),
                        "city": city,
                        "edge_id": int(edge_id),
                        "reverse_edge_id": int(actual_reverse_id),
                        "allocation_policy": str(args.allocation_policy),
                        "allocation_alpha": float(allocation_alpha),
                        "trigger_schedule_id": (
                            str(args.experimental_trigger_schedule)
                            if args.experimental_trigger_schedule is not None
                            else None
                        ),
                        "applied_perturb_window": int(
                            audit["trigger_spec"]["perturb_window"]
                        ),
                        "clean_pair_probability": clean_target,
                        "triggered_pair_probability": triggered_target,
                        "probability_delta": triggered_target - clean_target,
                        "evaluated_model_threshold": threshold,
                        "evaluated_model_clean_positive": bool(
                            clean_target >= threshold
                        ),
                        "evaluated_model_triggered_positive": bool(
                            triggered_target >= threshold
                        ),
                        **target_motion_by_scenario.get(
                            str(row.scenario_id), {}
                        ),
                    }
                )

                endpoint = (edge_index[0] == src) | (edge_index[0] == dst)
                endpoint |= (edge_index[1] == src) | (edge_index[1] == dst)
                incident = endpoint & ~pair_target_mask & supervised
                incident_negative = incident & (labels == 0)
                incident_positive = incident & (labels == 1)
                # Nonincident collateral includes every supervised negative that
                # shares neither endpoint with the selected unordered pair.
                nonincident_negative = supervised & (labels == 0) & ~endpoint

                _accumulate_collateral_counts(
                    counts,
                    clean_probs,
                    triggered_probs,
                    threshold=threshold,
                    nonincident_negative=nonincident_negative,
                    incident_negative=incident_negative,
                    incident_positive=incident_positive,
                )
                if common_counts is not None and common_threshold is not None:
                    _accumulate_collateral_counts(
                        common_counts,
                        clean_probs,
                        triggered_probs,
                        threshold=common_threshold,
                        nonincident_negative=nonincident_negative,
                        incident_negative=incident_negative,
                        incident_positive=incident_positive,
                    )
                for threshold_kind, evidence_threshold in (
                    ("evaluated_model", float(threshold)),
                    (
                        "common",
                        float(common_threshold)
                        if common_threshold is not None
                        else None,
                    ),
                ):
                    if evidence_threshold is None:
                        continue
                    for subset_name, subset_mask, label_class in (
                        (
                            "nonincident_negative",
                            nonincident_negative,
                            0,
                        ),
                        ("adjacent_negative", incident_negative, 0),
                        ("adjacent_positive", incident_positive, 1),
                    ):
                        transition = _transition_counts(
                            clean_probs,
                            triggered_probs,
                            subset_mask,
                            threshold=float(evidence_threshold),
                        )
                        collateral_transition_records.append(
                            {
                                "scenario_id": str(row.scenario_id),
                                "city": city,
                                "threshold_kind": threshold_kind,
                                "threshold": float(evidence_threshold),
                                "subset": subset_name,
                                "label": int(label_class),
                                **transition,
                            }
                        )
                for node_audit in audit.get("nodes", []):
                    for key, value in node_audit.items():
                        if key == "dst" or not isinstance(value, (int, float)):
                            continue
                        stealth_max[key] = max(stealth_max[key], float(value))

    if not clean_labels_parts:
        raise RuntimeError("no clean supervised test edges were evaluated")
    clean_labels = np.concatenate(clean_labels_parts)
    clean_probs = np.concatenate(clean_probs_parts)
    clean_pair_labels = (
        np.concatenate(clean_pair_labels_parts)
        if clean_pair_labels_parts
        else np.zeros(0, dtype=np.int8)
    )
    clean_pair_probs = (
        np.concatenate(clean_pair_probs_parts)
        if clean_pair_probs_parts
        else np.zeros(0, dtype=np.float32)
    )
    clean_edge_ids = (
        np.concatenate(clean_edge_ids_parts)
        if clean_edge_ids_parts
        else np.zeros(0, dtype=np.int64)
    )
    clean_edge_score_frame = pd.DataFrame(
        {
            "scenario_id": clean_edge_scenario_ids,
            "city": clean_edge_cities,
            "edge_id": clean_edge_ids,
            "label": clean_labels,
            "probability": clean_probs,
            "evaluated_model_threshold": float(directed_threshold),
        }
    )
    pair_score_frame = pd.DataFrame(
        pair_score_records,
        columns=[
            "scenario_id",
            "city",
            "canonical_edge_id",
            "reverse_edge_id",
            "label",
            "pair_probability",
            "evaluated_model_threshold",
            "common_threshold",
        ],
    )
    collateral_transition_frame = pd.DataFrame(
        collateral_transition_records,
        columns=[
            "scenario_id",
            "city",
            "threshold_kind",
            "threshold",
            "subset",
            "label",
            "n00",
            "n01",
            "n10",
            "n11",
            "total",
        ],
    )
    score_frame = pd.DataFrame(score_records)
    if common_threshold is not None and not score_frame.empty:
        score_frame["common_threshold"] = common_threshold
        score_frame["common_clean_positive"] = (
            score_frame["clean_pair_probability"] >= common_threshold
        )
        score_frame["common_triggered_positive"] = (
            score_frame["triggered_pair_probability"] >= common_threshold
        )
    _atomic_csv_write(score_frame, score_path)
    score_hash = sha256_file(score_path)
    _atomic_csv_write(clean_edge_score_frame, clean_edge_score_path)
    clean_edge_score_hash = sha256_file(clean_edge_score_path)
    _atomic_csv_write(pair_score_frame, pair_score_path)
    pair_score_hash = sha256_file(pair_score_path)
    _atomic_csv_write(
        collateral_transition_frame, collateral_transition_path
    )
    collateral_transition_hash = sha256_file(collateral_transition_path)

    target_metrics = _target_metrics_at_threshold(score_frame, threshold)
    attack_metrics = {
        "primary_metric": "incremental_flip_rate_all_targets",
        **target_metrics,
        "target_probability_delta_mean": (
            float(np.mean(deltas)) if deltas else None
        ),
        **_collateral_metrics_from_counts(counts),
    }
    common_threshold_metrics = None
    per_city_common_threshold = None
    per_motion_regime = None
    per_motion_regime_common_threshold = None
    if "motion_regime" in score_frame.columns:
        per_motion_regime = {
            str(regime): _target_metrics_at_threshold(regime_scores, threshold)
            for regime, regime_scores in sorted(
                score_frame.groupby("motion_regime")
            )
        }
    if common_threshold is not None:
        if common_counts is None:
            raise RuntimeError("common-threshold collateral counter was not initialized")
        common_threshold_metrics = {
            "primary_metric": "incremental_flip_rate_all_targets",
            **_target_metrics_at_threshold(
                score_frame, common_threshold
            ),
            "target_probability_delta_mean": (
                float(np.mean(deltas)) if deltas else None
            ),
            **_collateral_metrics_from_counts(common_counts),
        }
        per_city_common_threshold = (
            {
                str(city): _target_metrics_at_threshold(
                    city_scores, common_threshold
                )
                for city, city_scores in sorted(score_frame.groupby("city"))
            }
            if not score_frame.empty
            else {}
        )
        if "motion_regime" in score_frame.columns:
            per_motion_regime_common_threshold = {
                str(regime): _target_metrics_at_threshold(
                    regime_scores, common_threshold
                )
                for regime, regime_scores in sorted(
                    score_frame.groupby("motion_regime")
                )
            }
    expected_contract_role = (
        formal_contract[
            "clean_role" if expected_frozen_role == "clean" else "victim_role"
        ]
        if formal_contract is not None
        else None
    )
    evaluated_model_role_verified = bool(
        evaluated_model_frozen_asset_audit is not None
        and evaluated_model_frozen_asset_audit.get("role")
        == expected_contract_role
    )
    expected_transition_rows = int(len(target_frame)) * (
        6 if common_threshold is not None else 3
    )
    metric_evidence_complete = bool(
        len(clean_edge_score_frame) == int(clean_labels.size)
        and len(pair_score_frame) == int(clean_pair_labels.size)
        and len(clean_edge_score_frame) > 0
        and len(pair_score_frame) > 0
        and len(collateral_transition_frame) == expected_transition_rows
    )
    completion = evaluation_completion_flags(
        split=str(args.split),
        max_graphs=args.max_graphs,
        max_targets=args.max_targets,
        target_rows=len(target_frame),
        score_rows=len(score_frame),
        pair_threshold_available=(
            model_threshold_source
            == "evaluated_model_clean_validation_pair_metrics"
        ),
        assets_verified=True,
        training_protocol_verified=bool(
            training_protocol_audit[
                "formal_training_protocol_verified"
            ]
        ),
        clean_reference_bound=bool(
            clean_reference_binding is not None
            and clean_reference_binding["frozen_asset_verified"]
        ),
        formal_contract_verified=bool(
            formal_method_audit["formal_contract_verified"]
            and evaluated_model_frozen_asset_audit is not None
        ),
        evaluated_model_role_verified=evaluated_model_role_verified,
        pretest_release_verified=bool(
            pretest_release is not None
            and pretest_release["verified_before_test_graph_access"]
        ),
        metric_evidence_complete=metric_evidence_complete,
    )
    output = {
        "experiment": (
            formal_contract["version"]
            if formal_contract is not None
            else "training_procedure_control_validation"
            if access_context["evaluated_model_training_procedure_controlled"] is True
            else (
                "data_driven_bi_endpoint_allocation_validation"
                if str(args.allocation_policy)
                == ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2
                else (
                    "v5_1_fixed_symmetric_biend_validation"
                    if str(args.allocation_policy)
                    == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
                    else "strict_zero_query_data_poisoning"
                )
            )
        ),
        "split": args.split,
        "evaluation_model_role": str(args.evaluation_model_role),
        "evaluation_phase": (
            "development_validation"
            if args.split == "val"
            else "test_informed_confirmatory_evaluation"
        ),
        "training_access_context": access_context,
        **completion,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": asset_binding["checkpoint_sha256"],
            "training_result": str(result_path),
            "training_result_sha256": sha256_file(result_path),
            "matches_training_result": True,
        },
        "graph_manifest": {
            "path": str(graph_manifest_path),
            "sha256": asset_binding["graph_manifest_sha256"],
            "grouped_split": bool("split_group_id" in manifest.columns),
            "matches_training_result": True,
            "graph_sha256_required": True,
            "split_graphs_verified": int(len(resolved_graph_paths)),
            "split_contract_sha256": asset_binding[
                "split_contract_sha256"
            ],
            "split_contract_matches_training_result": True,
        },
        "asset_integrity": {
            **asset_binding,
            "split_graph_sha256_verified": int(
                len(resolved_graph_paths)
            ),
            "clean_reference_bound": clean_reference_binding is not None,
            "clean_reference_frozen_asset_verified": bool(
                clean_reference_binding is not None
                and clean_reference_binding["frozen_asset_verified"]
            ),
            "evaluated_model_frozen_asset_verified": (
                evaluated_model_role_verified
            ),
            "victim_frozen_asset_verified": (
                evaluated_model_role_verified
                and args.evaluation_model_role == "victim"
            ),
            "clean_model_frozen_asset_verified": (
                evaluated_model_role_verified
                and args.evaluation_model_role == "clean_reference"
            ),
            "pretest_release_verified": bool(
                pretest_release is not None
                and pretest_release["verified_before_test_graph_access"]
            ),
            "metric_evidence_complete": metric_evidence_complete,
        },
        "training_protocol_audit": training_protocol_audit,
        "clean_reference": clean_reference_binding,
        "formal_contract": (
            {
                "path": formal_contract["contract_path"],
                "sha256": formal_contract["contract_sha256"],
                "asset_manifest_path": formal_contract[
                    "asset_manifest_path"
                ],
                "asset_manifest_sha256": formal_contract[
                    "asset_manifest_sha256"
                ],
                "version": formal_contract["version"],
                "method_audit": formal_method_audit,
                "evaluated_model_asset_audit": (
                    evaluated_model_frozen_asset_audit
                ),
                "expected_evaluated_model_role": expected_contract_role,
                "victim_asset_audit": (
                    evaluated_model_frozen_asset_audit
                    if args.evaluation_model_role == "victim"
                    else None
                ),
                "clean_model_asset_audit": (
                    evaluated_model_frozen_asset_audit
                    if args.evaluation_model_role == "clean_reference"
                    else None
                ),
                "formal_contract_verified": bool(
                    formal_method_audit["formal_contract_verified"]
                    and evaluated_model_role_verified
                    and clean_reference_binding is not None
                    and clean_reference_binding[
                        "frozen_asset_verified"
                    ]
                ),
            }
            if formal_contract is not None
            else None
        ),
        "pretest_release": pretest_release,
        "evaluator": {
            "path": Path(__file__).resolve().relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
            "integrity_patch_version": (
                V6_EVALUATOR_INTEGRITY_VERSION
                if formal_contract is not None
                and formal_contract["version"]
                == "v6_fixed_symmetric_same_pair_training"
                else EVALUATOR_INTEGRITY_VERSION
            ),
        },
        "label_config": label_config_dict(label_config),
        "label_config_hash": label_config_hash(label_config),
        "edge_feature_protocol": edge_feature_protocol(edge_feature_mode),
        "target_pool_definition": (
            f"one_seeded_data_only_unordered_pair_per_eligible_{args.split}_scenario "
            "from (both_directions_supervised_true_negative & physical_possible "
            "& label_computable & "
            + (
                "K10_contiguous_both_endpoints"
                if str(args.allocation_policy)
                == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
                else "K10_contiguous_destination"
            )
            + "), followed by "
            f"orientation_policy={args.orientation_policy} and "
            f"allocation_policy={args.allocation_policy}"
        ),
        "development_selection_context": {
            "prior_validation_informed": bool(
                str(args.allocation_policy)
                == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
            ),
            "source": (
                "counterfactual_development_train_audits_and_clean_validation"
                if access_context["evaluated_model_training_procedure_controlled"] is True
                else "v5_validation_allocation_stratification"
                if str(args.allocation_policy)
                == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
                else None
            ),
            "test_used_for_method_selection": False,
        },
        "target_sampling": {
            "scheme": "scenario_seeded_uniform_pair_then_data_only_orientation_v3",
            "seed": int(args.target_seed),
            "orientation_policy": str(args.orientation_policy),
            "allocation_policy": str(args.allocation_policy),
            "training_evaluation_allocation_binding": allocation_policy_binding,
            "allocation_uses_model_output": False,
            "one_unordered_pair_per_eligible_scenario": True,
            "clean_prediction_filter_used": False,
            "require_bi_endpoint_contiguous": bool(
                str(args.allocation_policy)
                == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
            ),
            "fixed_allocation_alpha": (
                FIXED_SYMMETRIC_BIEND_ALPHA
                if str(args.allocation_policy)
                == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
                else None
            ),
        },
        "target_manifest": {
            "path": str(target_manifest_path),
            "sha256": target_manifest_hash,
            "rows": int(len(target_frame)),
            "frozen_before_model_inference": True,
            "public_pool_binding": public_pool_binding,
        },
        "target_scores": {
            "path": str(score_path),
            "sha256": score_hash,
            "rows": int(len(score_frame)),
            "pair_probability_aggregation": "mean_of_two_directed_probabilities",
        },
        "metric_evidence": {
            "complete": metric_evidence_complete,
            "clean_edge_scores": {
                "path": str(clean_edge_score_path),
                "sha256": clean_edge_score_hash,
                "rows": int(len(clean_edge_score_frame)),
                "identity": "scenario_id_plus_directed_edge_id",
            },
            "clean_pair_scores": {
                "path": str(pair_score_path),
                "sha256": pair_score_hash,
                "rows": int(len(pair_score_frame)),
                "identity": (
                    "scenario_id_plus_sorted_directed_edge_id_pair"
                ),
                "aggregation": "mean_of_two_directed_probabilities",
            },
            "collateral_transitions": {
                "path": str(collateral_transition_path),
                "sha256": collateral_transition_hash,
                "rows": int(len(collateral_transition_frame)),
                "format": (
                    "per_target_per_threshold_per_subset_2x2_"
                    "clean_triggered_transition_counts"
                ),
                "threshold_kinds": (
                    ["evaluated_model", "common"]
                    if common_threshold is not None
                    else ["evaluated_model"]
                ),
            },
        },
        "target_selection_uses_model_output": False,
        "thresholds": {
            "evaluated_model_role": str(args.evaluation_model_role),
            "directed_evaluated_model_own": directed_threshold,
            "directed_evaluated_model_own_source": (
                "evaluated_model_clean_validation_directed_metrics"
            ),
            "evaluated_model_own": threshold,
            "evaluated_model_own_source": model_threshold_source,
            "victim_own": (
                threshold if args.evaluation_model_role == "victim" else None
            ),
            "clean_reference_own": (
                threshold
                if args.evaluation_model_role == "clean_reference"
                else None
            ),
            "common": common_threshold,
            "common_source": (
                "bound_clean_reference_validation_pair_metrics"
                if clean_reference_binding is not None
                else "cli_unbound_development_threshold"
                if common_threshold is not None
                else None
            ),
        },
        "attacker_uses_threshold": False,
        "clean_test_metrics": _clean_metrics(
            clean_labels,
            clean_probs,
            directed_threshold,
            "evaluated_model_clean_validation_directed_metrics",
        ),
        "clean_pair_metrics": _clean_metrics(
            clean_pair_labels,
            clean_pair_probs,
            threshold,
            model_threshold_source,
        ),
        "clean_pair_metrics_common_threshold": (
            _clean_metrics(
                clean_pair_labels,
                clean_pair_probs,
                common_threshold,
                (
                    "bound_clean_reference_validation_pair_metrics"
                    if clean_reference_binding is not None
                    else "cli_unbound_development_threshold"
                ),
            )
            if common_threshold is not None
            else None
        ),
        "attack_metrics": attack_metrics,
        "attack_metrics_common_threshold": common_threshold_metrics,
        "per_city": {
            city: {
                **dict(values),
                "absolute_asr": _rate(values["triggered_positive"], values["targets"]),
                "incremental_flip_rate": _rate(values["incremental_flip"], values["targets"]),
            }
            for city, values in sorted(per_city.items())
        },
        "per_city_common_threshold": per_city_common_threshold,
        "per_motion_regime": per_motion_regime,
        "per_motion_regime_common_threshold": (
            per_motion_regime_common_threshold
        ),
        "stealth": {
            "graph_topology_changed": False,
            "terminal_displacement_budget_m": 0.2,
            "observed_maxima": dict(sorted(stealth_max.items())),
        },
        "trigger": {
            "displacement_m": 0.2,
            "perturb_window": (
                10
                if args.experimental_trigger_schedule is None
                else None
            ),
            "experimental_trigger_schedule": (
                str(args.experimental_trigger_schedule)
                if args.experimental_trigger_schedule is not None
                else None
            ),
            "schedule_application_counts": dict(
                sorted(schedule_application_counts.items())
            ),
            "ramp_style": "minimum_jerk",
            "velocity_mode": "residual",
            "allocation_policy": str(args.allocation_policy),
            "allocation_alpha_semantics": (
                "fraction_of_relative_displacement_assigned_to_source"
            ),
            "fixed_allocation_alpha": (
                FIXED_SYMMETRIC_BIEND_ALPHA
                if str(args.allocation_policy)
                == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
                else None
            ),
        },
        "threat_model_audit": {
            "victim_queries_for_target_selection": 0,
            "validation_or_test_used_for_poison_selection": False,
            "victim_structure_parameters_or_gradients_used_by_attacker": access_context[
                "victim_structure_parameters_or_gradients_used_by_attacker"
            ],
            "training_asr_checkpoint_selection": access_context["training_asr_checkpoint_selection"],
            "evaluation_split_not_used_for_training_checkpoint_selection": access_context[
                "evaluation_split_not_used_for_training_checkpoint_selection"
            ],
            "formal_training_protocol_verified": bool(
                training_protocol_audit[
                    "formal_training_protocol_verified"
                ]
            ),
            "clean_reference_threshold_asset_bound": (
                clean_reference_binding is not None
                and clean_reference_binding["frozen_asset_verified"]
            ),
            "formal_contract_verified": bool(
                formal_method_audit["formal_contract_verified"]
                and evaluated_model_role_verified
            ),
            "pretest_release_verified": bool(
                pretest_release is not None
                and pretest_release["verified_before_test_graph_access"]
            ),
            "metric_evidence_complete": metric_evidence_complete,
        },
    }
    _atomic_json_write(output, output_path)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
