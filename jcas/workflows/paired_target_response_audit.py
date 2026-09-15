#!/usr/bin/env python3
"""Read-only train-pair response audit for a paired poisoning pilot.

The audit streams only the scenarios named by a train-only poison manifest.  It
compares clean and triggered versions of the exact same unordered target pair
under both a candidate victim and a clean reference model.  No optimizer is
created, and validation/test graph assets are never loaded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from jcas.core.models import build_model, model_kwargs_from_result
from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
    edge_features_for_mode,
)
from jcas.core.poison import apply_manifest_row, load_poison_manifest, sha256_file
from jcas.core.risk_labels import (
    RiskLabelConfig,
    labels_for_graph,
    selected_label_computable_mask,
)
from jcas.workflows.trainer import (
    load_manifest,
    message_passing_mask,
    supervision_mask,
    verified_graph_path,
)


AUDIT_VERSION = "paired_target_response_train_only_v1"
SLOW_REGIME = "slow_lt_0p5"
MOVING_REGIME = "moving_ge_0p5"
REGIMES = (SLOW_REGIME, MOVING_REGIME)
COLLATERAL_SUBSETS = (
    "adjacent_negative",
    "adjacent_positive",
    "nonincident_negative",
    "all_non_target_supervised",
)


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("distribution contains a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def summarize_target_records(
    frame: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Summarize paired target response at one externally bound threshold."""
    required = {
        "clean_pair_probability",
        "triggered_pair_probability",
        "clean_pair_mean_logit",
        "triggered_pair_mean_logit",
        "clean_negative_bce",
        "triggered_positive_bce",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"target response frame lacks columns: {missing}")
    if frame.empty:
        return {"targets": 0, "threshold": float(threshold)}
    clean_probability = frame["clean_pair_probability"].to_numpy(dtype=float)
    triggered_probability = frame["triggered_pair_probability"].to_numpy(
        dtype=float
    )
    clean_logit = frame["clean_pair_mean_logit"].to_numpy(dtype=float)
    triggered_logit = frame["triggered_pair_mean_logit"].to_numpy(dtype=float)
    probability_delta = triggered_probability - clean_probability
    logit_delta = triggered_logit - clean_logit
    clean_positive = clean_probability >= float(threshold)
    triggered_positive = triggered_probability >= float(threshold)
    clean_negative = ~clean_positive
    incremental = clean_negative & triggered_positive
    return {
        "targets": int(len(frame)),
        "threshold": float(threshold),
        "clean_activation_rate": float(clean_positive.mean()),
        "absolute_activation_rate": float(triggered_positive.mean()),
        "incremental_flip_rate_all_targets": float(incremental.mean()),
        "conditional_flip_rate": (
            float(incremental.sum() / clean_negative.sum())
            if clean_negative.any()
            else None
        ),
        "incremental_flips": int(incremental.sum()),
        "clean_negative_targets": int(clean_negative.sum()),
        "positive_probability_delta_fraction": float(
            (probability_delta > 0.0).mean()
        ),
        "positive_logit_delta_fraction": float((logit_delta > 0.0).mean()),
        "logit_margin_ge_0p5_fraction": float((logit_delta >= 0.5).mean()),
        "logit_margin_ge_1p0_fraction": float((logit_delta >= 1.0).mean()),
        "clean_pair_probability": _distribution(clean_probability),
        "triggered_pair_probability": _distribution(triggered_probability),
        "probability_delta": _distribution(probability_delta),
        "clean_pair_mean_logit": _distribution(clean_logit),
        "triggered_pair_mean_logit": _distribution(triggered_logit),
        "logit_delta": _distribution(logit_delta),
        "clean_negative_bce": _distribution(
            frame["clean_negative_bce"].to_numpy(dtype=float)
        ),
        "triggered_positive_bce": _distribution(
            frame["triggered_positive_bce"].to_numpy(dtype=float)
        ),
    }


@dataclass
class CollateralAccumulator:
    count: int = 0
    clean_positive: int = 0
    triggered_positive: int = 0
    n00: int = 0
    n01: int = 0
    n10: int = 0
    n11: int = 0
    delta_sum: float = 0.0
    absolute_delta_sum: float = 0.0
    absolute_delta_max: float = 0.0

    def add(
        self,
        clean_probability: np.ndarray,
        triggered_probability: np.ndarray,
        mask: np.ndarray,
        *,
        threshold: float,
    ) -> None:
        clean = np.asarray(clean_probability, dtype=np.float64)[mask]
        triggered = np.asarray(triggered_probability, dtype=np.float64)[mask]
        if clean.shape != triggered.shape:
            raise ValueError("clean and triggered collateral arrays differ")
        if clean.size == 0:
            return
        clean_positive = clean >= float(threshold)
        triggered_positive = triggered >= float(threshold)
        delta = triggered - clean
        self.count += int(clean.size)
        self.clean_positive += int(clean_positive.sum())
        self.triggered_positive += int(triggered_positive.sum())
        self.n00 += int((~clean_positive & ~triggered_positive).sum())
        self.n01 += int((~clean_positive & triggered_positive).sum())
        self.n10 += int((clean_positive & ~triggered_positive).sum())
        self.n11 += int((clean_positive & triggered_positive).sum())
        self.delta_sum += float(delta.sum())
        self.absolute_delta_sum += float(np.abs(delta).sum())
        self.absolute_delta_max = max(
            self.absolute_delta_max, float(np.abs(delta).max())
        )

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {"edges": 0}
        return {
            "edges": int(self.count),
            "clean_positive_rate": float(self.clean_positive / self.count),
            "triggered_positive_rate": float(
                self.triggered_positive / self.count
            ),
            "incremental_positive_rate": float(self.n01 / self.count),
            "suppression_rate": float(self.n10 / self.count),
            "mean_probability_delta": float(self.delta_sum / self.count),
            "mean_absolute_probability_delta": float(
                self.absolute_delta_sum / self.count
            ),
            "max_absolute_probability_delta": float(self.absolute_delta_max),
            "transitions": {
                "n00": int(self.n00),
                "n01": int(self.n01),
                "n10": int(self.n10),
                "n11": int(self.n11),
            },
        }


@dataclass
class AuditItem:
    scenario_id: str
    motion_regime: str
    src: int
    dst: int
    clean: Data
    triggered: Data
    target_mask: np.ndarray
    collateral_masks: dict[str, np.ndarray]


def _model_from_assets(
    result: dict[str, Any],
    checkpoint_path: Path,
    *,
    node_dim: int,
    edge_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    expected_hash = str((result.get("checkpoint") or {}).get("sha256", ""))
    observed_hash = sha256_file(checkpoint_path)
    if not expected_hash or expected_hash != observed_hash:
        raise RuntimeError(f"checkpoint is not bound to its result: {checkpoint_path}")
    model = build_model(
        **model_kwargs_from_result(result, node_dim=node_dim, edge_dim=edge_dim)
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def _make_data(
    *,
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    message_mask: np.ndarray,
) -> Data:
    return Data(
        x=torch.from_numpy(np.asarray(x, dtype=np.float32)),
        edge_index=torch.from_numpy(np.asarray(edge_index, dtype=np.int64)),
        edge_attr=torch.from_numpy(np.asarray(edge_attr, dtype=np.float32)),
        message_passing_edge_mask=torch.from_numpy(
            np.asarray(message_mask, dtype=np.bool_)
        ),
    )


def _load_audit_item(
    graph_row: Any,
    poison_row: Any,
    *,
    graph_dir: Path,
    label_config: RiskLabelConfig,
) -> AuditItem:
    graph_path = verified_graph_path(
        graph_row, graph_dir, require_graph_sha256=True
    )
    with np.load(graph_path, allow_pickle=True) as graph:
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
        label_bundle = labels_for_graph(graph, label_config)
        labels = label_bundle["edge_label"].astype(np.int8)
        supervised = supervision_mask(graph, True)
        supervised &= selected_label_computable_mask(
            graph, label_config, label_bundle
        )
        clean_edge_attr = edge_features_for_mode(
            graph["edge_attr"],
            edge_index,
            graph["observed_positions_filled"],
            graph["observed_velocities_filled"],
            graph["observed_valid_mask"],
            mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        )
        clean_x = np.asarray(graph["x_node"], dtype=np.float32)
        (
            triggered_x,
            triggered_edge_attr,
            _triggered_labels,
            target_mask,
            _trigger_audit,
        ) = apply_manifest_row(
            graph,
            poison_row,
            label_config,
            require_strict_label=True,
            edge_feature_mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        )
        target_mask = np.asarray(target_mask, dtype=np.bool_)
        target_ids = np.flatnonzero(target_mask)
        if target_ids.size != 2:
            raise RuntimeError("paired audit requires two directed target edges")
        if not supervised[target_ids].all() or np.any(labels[target_ids] != 0):
            raise RuntimeError("poison target is not a supervised clean negative pair")
        src = int(poison_row.src)
        dst = int(poison_row.dst)
        expected_directions = {
            (src, dst),
            (dst, src),
        }
        observed_directions = {
            (int(edge_index[0, edge_id]), int(edge_index[1, edge_id]))
            for edge_id in target_ids.tolist()
        }
        if observed_directions != expected_directions:
            raise RuntimeError("poison target identity differs from manifest pair")
        endpoint = (edge_index[0] == src) | (edge_index[0] == dst)
        endpoint |= (edge_index[1] == src) | (edge_index[1] == dst)
        incident = endpoint & ~target_mask & supervised
        collateral_masks = {
            "adjacent_negative": incident & (labels == 0),
            "adjacent_positive": incident & (labels == 1),
            "nonincident_negative": supervised & (labels == 0) & ~endpoint,
            "all_non_target_supervised": supervised & ~target_mask,
        }
        message_mask = message_passing_mask(graph)
        clean = _make_data(
            x=clean_x,
            edge_index=edge_index,
            edge_attr=clean_edge_attr,
            message_mask=message_mask,
        )
        triggered = _make_data(
            x=triggered_x,
            edge_index=edge_index,
            edge_attr=triggered_edge_attr,
            message_mask=message_mask,
        )
    return AuditItem(
        scenario_id=str(poison_row.scenario_id),
        motion_regime=str(poison_row.motion_regime),
        src=src,
        dst=dst,
        clean=clean,
        triggered=triggered,
        target_mask=target_mask,
        collateral_masks=collateral_masks,
    )


def _model_probabilities(
    model: torch.nn.Module,
    items: list[AuditItem],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clean_batch = Batch.from_data_list([item.clean for item in items]).to(device)
    triggered_batch = Batch.from_data_list(
        [item.triggered for item in items]
    ).to(device)
    with torch.no_grad():
        clean_logits = model(
            clean_batch.x,
            clean_batch.edge_index,
            clean_batch.edge_attr,
            clean_batch.message_passing_edge_mask,
        )
        triggered_logits = model(
            triggered_batch.x,
            triggered_batch.edge_index,
            triggered_batch.edge_attr,
            triggered_batch.message_passing_edge_mask,
        )
    clean_logits_array = clean_logits.detach().cpu().numpy()
    triggered_logits_array = triggered_logits.detach().cpu().numpy()
    return (
        clean_logits_array,
        triggered_logits_array,
        torch.sigmoid(clean_logits).detach().cpu().numpy(),
        torch.sigmoid(triggered_logits).detach().cpu().numpy(),
    )


def _append_batch_records(
    items: list[AuditItem],
    *,
    model_role: str,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    records: list[dict[str, Any]],
    collateral: dict[str, dict[str, dict[str, CollateralAccumulator]]],
) -> None:
    (
        clean_logits,
        triggered_logits,
        clean_probability,
        triggered_probability,
    ) = _model_probabilities(model, items, device=device)
    offset = 0
    for item in items:
        edges = int(item.target_mask.size)
        local = slice(offset, offset + edges)
        target_ids = np.flatnonzero(item.target_mask)
        local_clean_logits = clean_logits[local]
        local_triggered_logits = triggered_logits[local]
        local_clean_probability = clean_probability[local]
        local_triggered_probability = triggered_probability[local]
        target_clean_logits = local_clean_logits[target_ids]
        target_triggered_logits = local_triggered_logits[target_ids]
        records.append(
            {
                "scenario_id": item.scenario_id,
                "motion_regime": item.motion_regime,
                "src": int(item.src),
                "dst": int(item.dst),
                "model_role": model_role,
                "clean_pair_probability": float(
                    local_clean_probability[target_ids].mean()
                ),
                "triggered_pair_probability": float(
                    local_triggered_probability[target_ids].mean()
                ),
                "clean_pair_mean_logit": float(target_clean_logits.mean()),
                "triggered_pair_mean_logit": float(
                    target_triggered_logits.mean()
                ),
                "clean_negative_bce": float(
                    F.softplus(torch.from_numpy(target_clean_logits)).mean()
                ),
                "triggered_positive_bce": float(
                    F.softplus(
                        -torch.from_numpy(target_triggered_logits)
                    ).mean()
                ),
            }
        )
        for subset, mask in item.collateral_masks.items():
            collateral[model_role][item.motion_regime][subset].add(
                local_clean_probability,
                local_triggered_probability,
                mask,
                threshold=threshold,
            )
            collateral[model_role]["all"][subset].add(
                local_clean_probability,
                local_triggered_probability,
                mask,
                threshold=threshold,
            )
        offset += edges
    if offset != int(clean_logits.size):
        raise RuntimeError("batched edge accounting mismatch")


def _chunks(frame: pd.DataFrame, size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(frame), int(size)):
        yield frame.iloc[start : start + int(size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit clean/trigger response on paired pilot training targets "
            "without loading validation or test graph assets."
        )
    )
    parser.add_argument("--graph-dir", default="graphs/av2mf_graph_v4")
    parser.add_argument(
        "--graph-manifest",
        default="graphs/av2mf_graph_v4/manifest_grouped_v5.csv",
    )
    parser.add_argument("--poison-manifest", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument(
        "--candidate-role",
        default="paired_victim",
        choices=("paired_victim", "explicit_counterfactual_victim"),
    )
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--validation-result", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output", default=None)
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_scenarios is not None and int(args.max_scenarios) <= 0:
        raise ValueError("max-scenarios must be positive")
    graph_dir = Path(args.graph_dir).resolve()
    graph_manifest_path = Path(args.graph_manifest).resolve()
    poison_manifest_path = Path(args.poison_manifest).resolve()
    candidate_result_path = Path(args.candidate_result).resolve()
    candidate_checkpoint_path = Path(args.candidate_checkpoint).resolve()
    reference_result_path = Path(args.reference_result).resolve()
    reference_checkpoint_path = Path(args.reference_checkpoint).resolve()
    output_path = Path(args.output).resolve()
    candidate_role = str(args.candidate_role)
    model_roles = ("clean_reference", candidate_role)
    records_output_path = (
        Path(args.records_output).resolve()
        if args.records_output is not None
        else output_path.with_name(f"{output_path.stem}.records.csv")
    )
    label_config = RiskLabelConfig()

    graph_manifest_hash = sha256_file(graph_manifest_path)
    manifest = load_manifest(
        graph_dir, graph_manifest_path, require_graph_sha256=True
    )
    poison, poison_hash = load_poison_manifest(
        poison_manifest_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=graph_manifest_hash,
    )
    required_columns = {"scenario_id", "split", "src", "dst", "motion_regime"}
    missing_columns = sorted(required_columns - set(poison.columns))
    if missing_columns:
        raise ValueError(f"poison manifest lacks columns: {missing_columns}")
    if set(poison["split"].astype(str)) != {"train"}:
        raise RuntimeError("paired response audit refuses non-training poison rows")
    unexpected_regimes = sorted(
        set(poison["motion_regime"].astype(str)) - set(REGIMES)
    )
    if unexpected_regimes:
        raise RuntimeError(f"unexpected motion regimes: {unexpected_regimes}")
    poison_metadata_path = poison_manifest_path.with_suffix(
        poison_manifest_path.suffix + ".metadata.json"
    )
    poison_metadata = _read_json(poison_metadata_path)
    if poison_metadata.get("test_access_authorized") is not False:
        raise RuntimeError("audit requires a manifest that forbids test access")
    if poison_metadata.get("formal_test_eligible") is not False:
        raise RuntimeError("audit requires a development-only poison manifest")
    if str(poison_metadata.get("edge_feature_mode")) != (
        MOTION_NORMALIZED_EDGE_FEATURE_MODE
    ):
        raise RuntimeError("audit requires the frozen 42-dimensional edge mode")

    poison = poison.sort_values("scenario_id").reset_index(drop=True)
    if args.max_scenarios is not None:
        poison = poison.head(int(args.max_scenarios)).copy()
    graph_rows = manifest.set_index("scenario_id", drop=False)
    missing_graphs = sorted(set(poison["scenario_id"]) - set(graph_rows.index))
    if missing_graphs:
        raise RuntimeError(f"poison scenarios missing from manifest: {missing_graphs[:3]}")
    if set(graph_rows.loc[poison["scenario_id"], "split"].astype(str)) != {"train"}:
        raise RuntimeError("audit scenario resolution escaped the train split")

    candidate_result = _read_json(candidate_result_path)
    reference_result = _read_json(reference_result_path)
    if (candidate_result.get("scope") or {}).get("test_accessed") is not False:
        raise RuntimeError("candidate result does not attest no test access")
    if (candidate_result.get("config") or {}).get("evaluate_test") is not False:
        raise RuntimeError("candidate result was not configured as development-only")
    candidate_poison = candidate_result.get("poison_manifest") or {}
    if str(candidate_poison.get("sha256")) != poison_hash:
        raise RuntimeError("candidate was not trained with the audited poison manifest")
    candidate_graph = candidate_result.get("graph_manifest") or {}
    if str(candidate_graph.get("sha256")) != graph_manifest_hash:
        raise RuntimeError("candidate graph manifest binding differs from audit")

    threshold = float((reference_result.get("val_pair_metrics") or {})["threshold"])
    validation_result_path = (
        Path(args.validation_result).resolve()
        if args.validation_result is not None
        else None
    )
    validation_context: dict[str, Any] | None = None
    if validation_result_path is not None:
        validation_result = _read_json(validation_result_path)
        if str(validation_result.get("split")) != "val":
            raise RuntimeError("comparison result is not a validation result")
        validation_checkpoint = validation_result.get("checkpoint") or {}
        if str(validation_checkpoint.get("sha256")) != sha256_file(
            candidate_checkpoint_path
        ):
            raise RuntimeError("validation result is not for the candidate checkpoint")
        validation_context = {
            "path": str(validation_result_path),
            "sha256": sha256_file(validation_result_path),
            "graph_assets_loaded_by_this_audit": False,
            "overall": validation_result.get("attack_metrics_common_threshold"),
            "per_motion_regime": validation_result.get(
                "per_motion_regime_common_threshold"
            ),
        }

    first_poison = next(poison.itertuples(index=False))
    first_item = _load_audit_item(
        graph_rows.loc[str(first_poison.scenario_id)],
        first_poison,
        graph_dir=graph_dir,
        label_config=label_config,
    )
    node_dim = int(first_item.clean.x.shape[1])
    edge_dim = int(first_item.clean.edge_attr.shape[1])
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    models = {
        "clean_reference": _model_from_assets(
            reference_result,
            reference_checkpoint_path,
            node_dim=node_dim,
            edge_dim=edge_dim,
            device=device,
        ),
        candidate_role: _model_from_assets(
            candidate_result,
            candidate_checkpoint_path,
            node_dim=node_dim,
            edge_dim=edge_dim,
            device=device,
        ),
    }
    records: list[dict[str, Any]] = []
    collateral = {
        role: {
            regime: {
                subset: CollateralAccumulator()
                for subset in COLLATERAL_SUBSETS
            }
            for regime in ("all", *REGIMES)
        }
        for role in model_roles
    }

    processed = 0
    for chunk in _chunks(poison, int(args.batch_size)):
        items = [
            _load_audit_item(
                graph_rows.loc[str(row.scenario_id)],
                row,
                graph_dir=graph_dir,
                label_config=label_config,
            )
            for row in chunk.itertuples(index=False)
        ]
        for role in model_roles:
            _append_batch_records(
                items,
                model_role=role,
                model=models[role],
                device=device,
                threshold=threshold,
                records=records,
                collateral=collateral,
            )
        processed += len(items)
        print(
            f"[paired_target_response_audit] processed={processed}/{len(poison)}",
            flush=True,
        )

    record_frame = pd.DataFrame(records).sort_values(
        ["scenario_id", "model_role"]
    ).reset_index(drop=True)
    for stem in ("probability", "logit"):
        record_frame[f"{stem}_delta"] = (
            record_frame[f"triggered_pair_{'probability' if stem == 'probability' else 'mean_logit'}"]
            - record_frame[f"clean_pair_{'probability' if stem == 'probability' else 'mean_logit'}"]
        )
    _atomic_csv_write(record_frame, records_output_path)

    target_response: dict[str, Any] = {}
    collateral_response: dict[str, Any] = {}
    for role in model_roles:
        role_frame = record_frame[record_frame["model_role"] == role]
        target_response[role] = {
            "all": summarize_target_records(role_frame, threshold=threshold),
            **{
                regime: summarize_target_records(
                    role_frame[role_frame["motion_regime"] == regime],
                    threshold=threshold,
                )
                for regime in REGIMES
            },
        }
        collateral_response[role] = {
            regime: {
                subset: collateral[role][regime][subset].summary()
                for subset in COLLATERAL_SUBSETS
            }
            for regime in ("all", *REGIMES)
        }

    result = {
        "audit_version": AUDIT_VERSION,
        "scope": {
            "development_only": True,
            "split": "train",
            "train_graphs_read": int(len(poison)),
            "validation_graphs_read": 0,
            "test_graphs_read": 0,
            "validation_threshold_metadata_read": True,
            "validation_result_summary_read": validation_result_path is not None,
            "model_training_performed": False,
            "optimizer_created": False,
            "frozen_assets_modified": False,
        },
        "configuration": {
            "batch_size": int(args.batch_size),
            "device": str(device),
            "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
            "threshold": threshold,
            "threshold_source": "clean_reference_validation_pair_metrics_metadata",
            "candidate_role": candidate_role,
            "max_scenarios": (
                int(args.max_scenarios)
                if args.max_scenarios is not None
                else None
            ),
        },
        "assets": {
            "graph_manifest": {
                "path": str(graph_manifest_path),
                "sha256": graph_manifest_hash,
            },
            "poison_manifest": {
                "path": str(poison_manifest_path),
                "sha256": poison_hash,
                "metadata_path": str(poison_metadata_path),
                "metadata_sha256": sha256_file(poison_metadata_path),
            },
            "clean_reference": {
                "result_path": str(reference_result_path),
                "result_sha256": sha256_file(reference_result_path),
                "checkpoint_path": str(reference_checkpoint_path),
                "checkpoint_sha256": sha256_file(reference_checkpoint_path),
            },
            candidate_role: {
                "result_path": str(candidate_result_path),
                "result_sha256": sha256_file(candidate_result_path),
                "checkpoint_path": str(candidate_checkpoint_path),
                "checkpoint_sha256": sha256_file(candidate_checkpoint_path),
            },
            "records": {
                "path": str(records_output_path),
                "sha256": sha256_file(records_output_path),
                "rows": int(len(record_frame)),
            },
            "validation_context": validation_context,
        },
        "scenario_counts": {
            "all": int(len(poison)),
            **{
                regime: int((poison["motion_regime"] == regime).sum())
                for regime in REGIMES
            },
        },
        "target_response": target_response,
        "collateral_response": collateral_response,
    }
    _atomic_json_write(result, output_path)
    return result


def main() -> None:
    args = parse_args()
    result = run_audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "scope": result["scope"],
                "scenario_counts": result["scenario_counts"],
                "target_response": result["target_response"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
