#!/usr/bin/env python3
"""Train-only audit of slow/moving trajectory-trigger learnability.

The audit never reads validation or test rows.  It compares clean and triggered
versions of the same frozen training pair in two complementary ways:

* a grouped out-of-fold linear probe measures whether the trigger is visible in
  the model inputs on previously unseen training scenarios;
* decoder-gradient coherence measures whether independently poisoned examples
  push a clean model in a consistent direction, both for the current
  replacement protocol and for a paired clean/triggered counterfactual.

This module is diagnostic only.  It does not train or modify a victim model and
does not write output unless ``--output`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jcas.core.models import build_model, model_kwargs_from_result
from jcas.core.motion_normalized_features import (
    BASE_EDGE_FEATURE_MODE,
    MOTION_FEATURE_DIM,
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
    edge_features_for_mode,
)
from jcas.core.poison import (
    apply_manifest_row,
    load_poison_manifest,
    sha256_file,
)
from jcas.core.risk_labels import RiskLabelConfig
from jcas.workflows.trainer import (
    load_manifest,
    message_passing_mask,
    verified_graph_path,
)


AUDIT_VERSION = "moving_trigger_learnability_train_only_v1"
SLOW_REGIME = "slow_lt_0p5"
MOVING_REGIME = "moving_ge_0p5"
REGIMES = (SLOW_REGIME, MOVING_REGIME)
FEATURE_VIEWS = (
    "node_pair",
    "edge_base34_pair",
    "edge_motion8_pair",
    "edge_augmented42_pair",
    "all_pair_inputs",
)


def _stable_digest(seed: int, value: str) -> str:
    payload = f"{AUDIT_VERSION}|{int(seed)}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_subset(
    frame: pd.DataFrame,
    maximum: int | None,
    *,
    seed: int,
) -> pd.DataFrame:
    """Return a deterministic scenario subset without using model outputs."""
    ranked = frame.copy()
    ranked["_audit_priority"] = ranked["scenario_id"].astype(str).map(
        lambda value: _stable_digest(seed, value)
    )
    ranked = ranked.sort_values(["_audit_priority", "scenario_id"])
    if maximum is not None:
        ranked = ranked.head(int(maximum))
    return ranked.drop(columns=["_audit_priority"]).reset_index(drop=True)


def stable_fold(scenario_id: str, *, seed: int, folds: int) -> int:
    if int(folds) < 2:
        raise ValueError("folds must be at least two")
    digest = _stable_digest(seed, f"fold|{scenario_id}")
    return int(digest[:16], 16) % int(folds)


def _directed_pair_edge_ids(
    edge_index: np.ndarray,
    src: int,
    dst: int,
) -> tuple[int, int]:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    forward = np.flatnonzero(
        (edge_index[0] == int(src)) & (edge_index[1] == int(dst))
    )
    reverse = np.flatnonzero(
        (edge_index[0] == int(dst)) & (edge_index[1] == int(src))
    )
    if forward.size != 1 or reverse.size != 1:
        raise ValueError("target pair must contain exactly two directed edges")
    return int(forward[0]), int(reverse[0])


def pair_feature_views(
    x_node: np.ndarray,
    edge_attr: np.ndarray,
    *,
    src: int,
    dst: int,
    forward_edge_id: int,
    reverse_edge_id: int,
) -> dict[str, np.ndarray]:
    """Build pair-local views from one clean or triggered graph."""
    x_node = np.asarray(x_node, dtype=np.float32)
    edge_attr = np.asarray(edge_attr, dtype=np.float32)
    if edge_attr.shape[1] != 34 + MOTION_FEATURE_DIM:
        raise ValueError("audit requires the frozen 42-dimensional edge mode")
    source = x_node[int(src)]
    destination = x_node[int(dst)]
    node_pair = np.concatenate(
        [source, destination, source - destination, source * destination]
    )
    directed_edges = edge_attr[[int(forward_edge_id), int(reverse_edge_id)]]
    edge_base = directed_edges[:, :34].reshape(-1)
    edge_motion = directed_edges[:, 34:].reshape(-1)
    edge_augmented = directed_edges.reshape(-1)
    return {
        "node_pair": node_pair.astype(np.float32, copy=False),
        "edge_base34_pair": edge_base.astype(np.float32, copy=False),
        "edge_motion8_pair": edge_motion.astype(np.float32, copy=False),
        "edge_augmented42_pair": edge_augmented.astype(np.float32, copy=False),
        "all_pair_inputs": np.concatenate(
            [node_pair, edge_augmented]
        ).astype(np.float32, copy=False),
    }


def grouped_linear_probe(
    values: np.ndarray,
    labels: np.ndarray,
    groups: Iterable[str],
    *,
    folds: int,
    seed: int,
    regularization_c: float,
) -> dict[str, Any]:
    """Run deterministic grouped out-of-fold trigger detection."""
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(list(groups), dtype=str)
    if values.ndim != 2 or values.shape[0] != labels.size:
        raise ValueError("probe values and labels have incompatible shapes")
    if groups.shape != labels.shape:
        raise ValueError("probe groups must contain one value per row")
    if sorted(np.unique(labels).tolist()) != [0, 1]:
        raise ValueError("probe labels must contain both clean and triggered rows")
    unique_groups = np.unique(groups)
    if unique_groups.size < int(folds):
        raise ValueError("probe has fewer groups than requested folds")
    fold_by_group = {
        group: stable_fold(group, seed=seed, folds=folds)
        for group in unique_groups.tolist()
    }
    fold_ids = np.asarray([fold_by_group[group] for group in groups], dtype=int)
    probabilities = np.full(labels.shape, np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        held_out = fold_ids == fold
        fitted = ~held_out
        if not held_out.any() or not fitted.any():
            raise RuntimeError(f"stable fold {fold} is empty")
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(regularization_c),
                class_weight="balanced",
                max_iter=2000,
                random_state=int(seed),
                solver="liblinear",
            ),
        )
        pipeline.fit(values[fitted], labels[fitted])
        fold_probability = pipeline.predict_proba(values[held_out])[:, 1]
        probabilities[held_out] = fold_probability
        fold_records.append(
            {
                "fold": int(fold),
                "groups": int(np.unique(groups[held_out]).size),
                "rows": int(held_out.sum()),
                "roc_auc": float(
                    roc_auc_score(labels[held_out], fold_probability)
                ),
                "balanced_accuracy_at_0p5": float(
                    balanced_accuracy_score(
                        labels[held_out], fold_probability >= 0.5
                    )
                ),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("out-of-fold probe did not score every row")
    fold_auc = np.asarray([row["roc_auc"] for row in fold_records], dtype=float)
    return {
        "groups": int(unique_groups.size),
        "rows": int(labels.size),
        "dimensions": int(values.shape[1]),
        "folds": int(folds),
        "regularization_c": float(regularization_c),
        "oof_roc_auc": float(roc_auc_score(labels, probabilities)),
        "oof_balanced_accuracy_at_0p5": float(
            balanced_accuracy_score(labels, probabilities >= 0.5)
        ),
        "fold_roc_auc_mean": float(fold_auc.mean()),
        "fold_roc_auc_sd": float(fold_auc.std(ddof=1)),
        "clean_probability_mean": float(probabilities[labels == 0].mean()),
        "triggered_probability_mean": float(
            probabilities[labels == 1].mean()
        ),
        "fold_records": fold_records,
    }


@dataclass
class GradientCoherenceAccumulator:
    count: int = 0
    sum_unit: np.ndarray | None = None
    sum_raw: np.ndarray | None = None
    sum_norm: float = 0.0
    sum_norm_squared: float = 0.0

    def add(self, gradient: np.ndarray) -> None:
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("gradient must have a finite positive norm")
        if self.sum_unit is None:
            self.sum_unit = np.zeros_like(gradient)
            self.sum_raw = np.zeros_like(gradient)
        if gradient.shape != self.sum_unit.shape:
            raise ValueError("gradient dimensions changed within an accumulator")
        self.sum_unit += gradient / norm
        assert self.sum_raw is not None
        self.sum_raw += gradient
        self.sum_norm += norm
        self.sum_norm_squared += norm * norm
        self.count += 1

    def summary(self) -> dict[str, Any]:
        if self.count == 0 or self.sum_unit is None or self.sum_raw is None:
            return {"count": 0}
        count = int(self.count)
        unit_resultant = float(np.linalg.norm(self.sum_unit))
        mean_norm = float(self.sum_norm / count)
        norm_variance = max(
            0.0,
            self.sum_norm_squared / count - mean_norm * mean_norm,
        )
        if count == 1:
            pairwise_cosine = None
        else:
            pairwise_cosine = float(
                (unit_resultant * unit_resultant - count)
                / (count * (count - 1))
            )
        return {
            "count": count,
            "dimensions": int(self.sum_unit.size),
            "mean_gradient_norm": mean_norm,
            "gradient_norm_sd": float(np.sqrt(norm_variance)),
            "mean_unit_resultant_length": float(unit_resultant / count),
            "mean_pairwise_cosine": pairwise_cosine,
            "effective_mean_gradient_ratio": float(
                np.linalg.norm(self.sum_raw / count) / mean_norm
            ),
        }


def _flatten_gradients(gradients: tuple[torch.Tensor, ...]) -> np.ndarray:
    return np.concatenate(
        [gradient.detach().reshape(-1).cpu().numpy() for gradient in gradients]
    ).astype(np.float64, copy=False)


def _decoder_gradient_vectors(
    model: torch.nn.Module,
    decoder_parameters: tuple[torch.nn.Parameter, ...],
    *,
    clean_x: np.ndarray,
    triggered_x: np.ndarray,
    edge_index: np.ndarray,
    clean_edge_attr: np.ndarray,
    triggered_edge_attr: np.ndarray,
    message_mask: np.ndarray,
    target_edge_ids: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    edge_index_tensor = torch.from_numpy(
        np.asarray(edge_index, dtype=np.int64)
    ).to(device)
    message_mask_tensor = torch.from_numpy(
        np.asarray(message_mask, dtype=np.bool_)
    ).to(device)
    target_ids = torch.as_tensor(
        target_edge_ids, dtype=torch.long, device=device
    )
    clean_logits = model(
        torch.from_numpy(np.asarray(clean_x, dtype=np.float32)).to(device),
        edge_index_tensor,
        torch.from_numpy(np.asarray(clean_edge_attr, dtype=np.float32)).to(device),
        message_mask_tensor,
    )[target_ids]
    triggered_logits = model(
        torch.from_numpy(np.asarray(triggered_x, dtype=np.float32)).to(device),
        edge_index_tensor,
        torch.from_numpy(
            np.asarray(triggered_edge_attr, dtype=np.float32)
        ).to(device),
        message_mask_tensor,
    )[target_ids]
    positive = torch.ones_like(triggered_logits)
    negative = torch.zeros_like(clean_logits)
    replacement_loss = F.binary_cross_entropy_with_logits(
        triggered_logits, positive
    )
    paired_loss = 0.5 * (
        replacement_loss
        + F.binary_cross_entropy_with_logits(clean_logits, negative)
    )
    replacement_gradient = torch.autograd.grad(
        replacement_loss,
        decoder_parameters,
        retain_graph=True,
    )
    paired_gradient = torch.autograd.grad(
        paired_loss,
        decoder_parameters,
    )
    probabilities = {
        "clean_pair_probability": float(
            torch.sigmoid(clean_logits).mean().detach().cpu()
        ),
        "triggered_pair_probability": float(
            torch.sigmoid(triggered_logits).mean().detach().cpu()
        ),
    }
    return (
        _flatten_gradients(replacement_gradient),
        _flatten_gradients(paired_gradient),
        probabilities,
    )


def _load_clean_model(
    result_path: Path,
    checkpoint_path: Path,
    *,
    node_dim: int,
    edge_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    with result_path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    model = build_model(
        **model_kwargs_from_result(result, node_dim=node_dim, edge_dim=edge_dim)
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit clean/trigger separability and decoder-gradient coherence "
            "using frozen training pairs only."
        )
    )
    parser.add_argument(
        "--graph-dir", default="graphs/av2mf_graph_v4"
    )
    parser.add_argument(
        "--graph-manifest",
        default="graphs/av2mf_graph_v4/manifest_grouped_v5.csv",
    )
    parser.add_argument(
        "--poison-manifest",
        default="record/v6_4/poison_rate005_slow_prioritized.csv",
    )
    parser.add_argument(
        "--clean-result",
        default=(
            "record/v6_1/clean_seed20260621/"
            "genconv_strict_seed20260621_20260827_174820/result.json"
        ),
    )
    parser.add_argument(
        "--clean-checkpoint",
        default=(
            "record/v6_1/clean_seed20260621/"
            "genconv_strict_seed20260621_20260827_174820/best_model.pt"
        ),
    )
    parser.add_argument("--max-per-regime", type=int, default=800)
    parser.add_argument("--gradient-max-per-regime", type=int, default=64)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--probe-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    graph_dir = Path(args.graph_dir).resolve()
    graph_manifest_path = Path(args.graph_manifest).resolve()
    poison_manifest_path = Path(args.poison_manifest).resolve()
    clean_result_path = Path(args.clean_result).resolve()
    clean_checkpoint_path = Path(args.clean_checkpoint).resolve()
    label_config = RiskLabelConfig()

    graph_manifest_hash = sha256_file(graph_manifest_path)
    manifest = load_manifest(
        graph_dir,
        graph_manifest_path,
        require_graph_sha256=True,
    )
    poison, poison_manifest_hash = load_poison_manifest(
        poison_manifest_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=graph_manifest_hash,
    )
    required_columns = {"motion_regime", "src", "dst"}
    missing = sorted(required_columns - set(poison.columns))
    if missing:
        raise ValueError(f"audit poison manifest lacks columns: {missing}")
    observed_regimes = set(poison["motion_regime"].astype(str))
    if not observed_regimes.issubset(REGIMES):
        raise ValueError(f"unexpected motion regimes: {sorted(observed_regimes)}")
    if set(poison["split"].astype(str)) != {"train"}:
        raise RuntimeError("audit refuses any non-training poison row")

    selected_parts: list[pd.DataFrame] = []
    gradient_scenarios: dict[str, set[str]] = {}
    for regime in REGIMES:
        regime_frame = poison[poison["motion_regime"] == regime]
        selected = stable_subset(
            regime_frame,
            args.max_per_regime,
            seed=int(args.seed),
        )
        selected_parts.append(selected)
        gradient_selected = stable_subset(
            selected,
            args.gradient_max_per_regime,
            seed=int(args.seed) + 1,
        )
        gradient_scenarios[regime] = set(
            gradient_selected["scenario_id"].astype(str)
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values("scenario_id").reset_index(drop=True)
    graph_rows = manifest.set_index("scenario_id", drop=False)
    missing_graphs = sorted(
        set(selected["scenario_id"].astype(str)) - set(graph_rows.index)
    )
    if missing_graphs:
        raise ValueError(
            f"selected poison scenarios are missing from graph manifest: {missing_graphs[:3]}"
        )

    first_row = selected.iloc[0]
    first_graph_row = graph_rows.loc[str(first_row.scenario_id)]
    first_path = verified_graph_path(
        first_graph_row,
        graph_dir,
        require_graph_sha256=True,
    )
    with np.load(first_path, allow_pickle=True) as graph:
        node_dim = int(np.asarray(graph["x_node"]).shape[1])
        edge_dim = int(
            edge_features_for_mode(
                graph["edge_attr"],
                graph["edge_index"],
                graph["observed_positions_filled"],
                graph["observed_velocities_filled"],
                graph["observed_valid_mask"],
                mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
            ).shape[1]
        )

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model: torch.nn.Module | None = None
    decoder_parameters: tuple[torch.nn.Parameter, ...] = ()
    if not bool(args.skip_gradients):
        model = _load_clean_model(
            clean_result_path,
            clean_checkpoint_path,
            node_dim=node_dim,
            edge_dim=edge_dim,
            device=device,
        )
        decoder_parameters = tuple(model.decoder.parameters())
        if not decoder_parameters:
            raise RuntimeError("clean model has no decoder parameters")

    feature_values: dict[str, dict[str, list[np.ndarray]]] = {
        regime: {view: [] for view in FEATURE_VIEWS} for regime in REGIMES
    }
    feature_labels: dict[str, list[int]] = {regime: [] for regime in REGIMES}
    feature_groups: dict[str, list[str]] = {regime: [] for regime in REGIMES}
    gradient_accumulators = {
        regime: {
            "replacement_positive_only": GradientCoherenceAccumulator(),
            "paired_clean_negative_triggered_positive": GradientCoherenceAccumulator(),
        }
        for regime in REGIMES
    }
    probability_records: dict[str, list[dict[str, float]]] = {
        regime: [] for regime in REGIMES
    }

    for index, poison_row in enumerate(selected.itertuples(index=False), start=1):
        scenario_id = str(poison_row.scenario_id)
        regime = str(poison_row.motion_regime)
        graph_row = graph_rows.loc[scenario_id]
        graph_path = verified_graph_path(
            graph_row,
            graph_dir,
            require_graph_sha256=True,
        )
        with np.load(graph_path, allow_pickle=True) as graph:
            edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
            forward_id, reverse_id = _directed_pair_edge_ids(
                edge_index, int(poison_row.src), int(poison_row.dst)
            )
            clean_x = np.asarray(graph["x_node"], dtype=np.float32)
            clean_edge_attr = edge_features_for_mode(
                graph["edge_attr"],
                edge_index,
                graph["observed_positions_filled"],
                graph["observed_velocities_filled"],
                graph["observed_valid_mask"],
                mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
            )
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
            target_ids = tuple(
                int(value) for value in np.flatnonzero(target_mask).tolist()
            )
            if set(target_ids) != {forward_id, reverse_id}:
                raise RuntimeError("triggered pair mask differs from manifest pair")

            for label, x_array, edge_array in (
                (0, clean_x, clean_edge_attr),
                (1, triggered_x, triggered_edge_attr),
            ):
                views = pair_feature_views(
                    x_array,
                    edge_array,
                    src=int(poison_row.src),
                    dst=int(poison_row.dst),
                    forward_edge_id=forward_id,
                    reverse_edge_id=reverse_id,
                )
                for view, values in views.items():
                    feature_values[regime][view].append(values)
                feature_labels[regime].append(label)
                feature_groups[regime].append(scenario_id)

            if (
                model is not None
                and scenario_id in gradient_scenarios[regime]
            ):
                replacement, paired, probabilities = _decoder_gradient_vectors(
                    model,
                    decoder_parameters,
                    clean_x=clean_x,
                    triggered_x=triggered_x,
                    edge_index=edge_index,
                    clean_edge_attr=clean_edge_attr,
                    triggered_edge_attr=triggered_edge_attr,
                    message_mask=message_passing_mask(graph),
                    target_edge_ids=(forward_id, reverse_id),
                    device=device,
                )
                gradient_accumulators[regime][
                    "replacement_positive_only"
                ].add(replacement)
                gradient_accumulators[regime][
                    "paired_clean_negative_triggered_positive"
                ].add(paired)
                probability_records[regime].append(probabilities)
        if index % 100 == 0 or index == len(selected):
            print(
                f"[trigger_learnability_audit] processed={index}/{len(selected)}",
                flush=True,
            )

    probe_results: dict[str, dict[str, Any]] = {}
    for regime in REGIMES:
        probe_results[regime] = {}
        labels = np.asarray(feature_labels[regime], dtype=np.int64)
        groups = feature_groups[regime]
        for view in FEATURE_VIEWS:
            probe_results[regime][view] = grouped_linear_probe(
                np.stack(feature_values[regime][view]),
                labels,
                groups,
                folds=int(args.probe_folds),
                seed=int(args.seed),
                regularization_c=float(args.probe_c),
            )

    gradient_results: dict[str, Any] = {}
    for regime in REGIMES:
        gradient_results[regime] = {
            protocol: accumulator.summary()
            for protocol, accumulator in gradient_accumulators[regime].items()
        }
        records = probability_records[regime]
        if records:
            clean_probability = np.asarray(
                [row["clean_pair_probability"] for row in records], dtype=float
            )
            triggered_probability = np.asarray(
                [row["triggered_pair_probability"] for row in records],
                dtype=float,
            )
            gradient_results[regime]["clean_model_response"] = {
                "count": int(len(records)),
                "clean_probability_mean": float(clean_probability.mean()),
                "triggered_probability_mean": float(
                    triggered_probability.mean()
                ),
                "probability_delta_mean": float(
                    (triggered_probability - clean_probability).mean()
                ),
            }

    result = {
        "audit_version": AUDIT_VERSION,
        "scope": {
            "split": "train",
            "validation_accessed": False,
            "test_accessed": False,
            "victim_training_performed": False,
            "frozen_assets_modified": False,
        },
        "assets": {
            "graph_manifest": str(graph_manifest_path),
            "graph_manifest_sha256": graph_manifest_hash,
            "poison_manifest": str(poison_manifest_path),
            "poison_manifest_sha256": poison_manifest_hash,
            "clean_result": str(clean_result_path),
            "clean_result_sha256": sha256_file(clean_result_path),
            "clean_checkpoint": str(clean_checkpoint_path),
            "clean_checkpoint_sha256": sha256_file(clean_checkpoint_path),
        },
        "configuration": {
            "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
            "base_edge_feature_mode": BASE_EDGE_FEATURE_MODE,
            "seed": int(args.seed),
            "max_per_regime": int(args.max_per_regime),
            "gradient_max_per_regime": int(args.gradient_max_per_regime),
            "probe_folds": int(args.probe_folds),
            "probe_c": float(args.probe_c),
            "device": str(device),
        },
        "selected_scenarios": {
            regime: int(
                (selected["motion_regime"].astype(str) == regime).sum()
            )
            for regime in REGIMES
        },
        "probe": probe_results,
        "decoder_gradient_coherence": gradient_results,
    }
    return result


def main() -> None:
    args = parse_args()
    result = run_audit(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"[trigger_learnability_audit] wrote {output}")
    print(rendered)


if __name__ == "__main__":
    main()
