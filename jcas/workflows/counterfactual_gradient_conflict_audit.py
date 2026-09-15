#!/usr/bin/env python3
"""Read-only gradient-conflict audit for the balanced counterfactual pilot.

The audit loads only train scenarios selected from the candidate's bound poison
manifest.  It compares the gradient of clean full-graph BCE with the poisoned
presentation BCE, each explicit counterfactual term, their weighted sum, and
the actual poison-pair update.  It never constructs an optimizer or updates a
model parameter, and it has no validation/test graph loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jcas.core.models import build_model, model_kwargs_from_result
from jcas.core.motion_normalized_features import MOTION_NORMALIZED_EDGE_FEATURE_MODE
from jcas.core.poison import load_poison_manifest, sha256_file
from jcas.core.risk_labels import RiskLabelConfig
from jcas.workflows.explicit_counterfactual_trainer import (
    BALANCED_V2_PROFILE,
    CounterfactualLossConfig,
    CounterfactualPresentationItem,
    collate_counterfactual_items,
    compose_counterfactual_training_items,
    counterfactual_loss_components,
)
from jcas.workflows.trainer import load_manifest, load_split_graphs


AUDIT_VERSION = "balanced_counterfactual_gradient_conflict_train_only_v1"
REGIMES = ("slow_lt_0p5", "moving_ge_0p5")
BLOCKS = (
    "all",
    "node_encoder",
    "edge_encoder",
    "message_passing",
    "decoder",
)
COMPONENT_WEIGHTS = {
    "presentation_base_bce": 1.0,
    "target_pair_bce": 0.1,
    "ranking": 0.1,
    "non_target_consistency": 1.0,
    "adjacent_negative_consistency": 5.0,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty gradient record table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parameter_block(name: str) -> str:
    """Map one GENConv parameter name to a stable architectural block."""
    if name.startswith("node_encoder."):
        return "node_encoder"
    if name.startswith("edge_encoder."):
        return "edge_encoder"
    if name.startswith("convs.") or name.startswith("norms."):
        return "message_passing"
    if name.startswith("decoder."):
        return "decoder"
    raise ValueError(f"unclassified trainable parameter: {name}")


def stable_regime_sample(
    frame: pd.DataFrame,
    *,
    samples_per_regime: int,
    audit_seed: int,
) -> pd.DataFrame:
    """Choose an order-invariant, reproducible sample within each regime."""
    if samples_per_regime <= 0:
        raise ValueError("samples_per_regime must be positive")
    required = {"scenario_id", "motion_regime"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"sample frame lacks columns: {missing}")
    selected: list[pd.DataFrame] = []
    for regime in REGIMES:
        group = frame[frame["motion_regime"].astype(str) == regime].copy()
        if len(group) < samples_per_regime:
            raise ValueError(
                f"regime {regime} has {len(group)} rows, fewer than "
                f"the requested {samples_per_regime}"
            )
        group["_audit_order"] = group["scenario_id"].astype(str).map(
            lambda scenario_id: hashlib.sha256(
                f"{int(audit_seed)}|{regime}|{scenario_id}".encode("utf-8")
            ).hexdigest()
        )
        selected.append(
            group.sort_values(["_audit_order", "scenario_id"])
            .head(samples_per_regime)
            .drop(columns=["_audit_order"])
        )
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["motion_regime", "scenario_id"])
        .reset_index(drop=True)
    )


def _gradient_tuple(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )


def combine_gradients(
    gradients: Mapping[str, Sequence[torch.Tensor | None]],
    coefficients: Mapping[str, float],
) -> tuple[torch.Tensor | None, ...]:
    """Linearly combine already-computed gradients without another backward."""
    if not coefficients:
        raise ValueError("gradient combination has no terms")
    unknown = sorted(set(coefficients) - set(gradients))
    if unknown:
        raise ValueError(f"gradient combination has unknown terms: {unknown}")
    lengths = {len(gradients[name]) for name in coefficients}
    if len(lengths) != 1:
        raise ValueError("gradient tuples have different lengths")
    combined: list[torch.Tensor | None] = []
    for index in range(next(iter(lengths))):
        value: torch.Tensor | None = None
        for name, coefficient in coefficients.items():
            gradient = gradients[name][index]
            if gradient is None or float(coefficient) == 0.0:
                continue
            term = gradient.detach() * float(coefficient)
            value = term if value is None else value + term
        combined.append(value)
    return tuple(combined)


def subtract_gradients(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    if len(left) != len(right):
        raise ValueError("gradient tuples have different lengths")
    difference: list[torch.Tensor | None] = []
    for left_value, right_value in zip(left, right):
        if left_value is None and right_value is None:
            difference.append(None)
        elif left_value is None:
            difference.append(-right_value.detach())
        elif right_value is None:
            difference.append(left_value.detach())
        else:
            difference.append(left_value.detach() - right_value.detach())
    return tuple(difference)


def cpu_gradient_copy(
    gradients: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        value.detach().cpu().clone() if value is not None else None
        for value in gradients
    )


def compare_gradients(
    base: Sequence[torch.Tensor | None],
    candidate: Sequence[torch.Tensor | None],
    indices: Iterable[int],
    *,
    training_weight: float = 1.0,
) -> dict[str, Any]:
    """Compare two gradient vectors over a selected parameter block."""
    dot = 0.0
    base_squared = 0.0
    candidate_squared = 0.0
    for index in indices:
        base_value = base[index]
        candidate_value = candidate[index]
        if base_value is not None:
            base_squared += float(torch.sum(base_value.detach() ** 2).cpu())
        if candidate_value is not None:
            candidate_squared += float(
                torch.sum(candidate_value.detach() ** 2).cpu()
            )
        if base_value is not None and candidate_value is not None:
            dot += float(
                torch.sum(base_value.detach() * candidate_value.detach()).cpu()
            )
    base_norm = float(np.sqrt(max(base_squared, 0.0)))
    candidate_norm = float(np.sqrt(max(candidate_squared, 0.0)))
    valid = base_norm > 0.0 and candidate_norm > 0.0
    cosine = float(dot / (base_norm * candidate_norm)) if valid else None
    norm_ratio = (
        float(candidate_norm / base_norm) if base_norm > 0.0 else None
    )
    projection_ratio = (
        float(dot / base_squared) if base_squared > 0.0 else None
    )
    weight = float(training_weight)
    return {
        "base_gradient_norm": base_norm,
        "candidate_gradient_norm": candidate_norm,
        "gradient_dot_product": float(dot),
        "cosine": cosine,
        "conflict": bool(dot < 0.0) if valid else None,
        "candidate_to_base_norm_ratio": norm_ratio,
        "training_weight": weight,
        "effective_candidate_to_base_norm_ratio": (
            float(abs(weight) * norm_ratio) if norm_ratio is not None else None
        ),
        "candidate_projection_in_base_units": projection_ratio,
    }


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("gradient summary contains a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def summarize_gradient_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate batch-level comparisons by regime, component, and block."""
    if not rows:
        raise ValueError("cannot summarize empty gradient records")
    frame = pd.DataFrame(rows)
    output: dict[str, Any] = {}
    for regime in ("all", *REGIMES):
        regime_frame = (
            frame if regime == "all" else frame[frame["motion_regime"] == regime]
        )
        output[regime] = {}
        for component in sorted(regime_frame["component"].unique()):
            output[regime][component] = {}
            component_frame = regime_frame[
                regime_frame["component"] == component
            ]
            for block in BLOCKS:
                group = component_frame[component_frame["parameter_block"] == block]
                valid = group[group["cosine"].notna()]
                conflicts = valid["conflict"].astype(bool)
                output[regime][component][block] = {
                    "batches": int(len(group)),
                    "pairs": int(group["pairs"].sum()),
                    "valid_direction_batches": int(len(valid)),
                    "conflict_fraction": (
                        float(conflicts.mean()) if len(conflicts) else None
                    ),
                    "cosine": _distribution(valid["cosine"].astype(float)),
                    "candidate_to_base_norm_ratio": _distribution(
                        group["candidate_to_base_norm_ratio"].dropna().astype(float)
                    ),
                    "effective_candidate_to_base_norm_ratio": _distribution(
                        group[
                            "effective_candidate_to_base_norm_ratio"
                        ].dropna().astype(float)
                    ),
                    "candidate_projection_in_base_units": _distribution(
                        group[
                            "candidate_projection_in_base_units"
                        ].dropna().astype(float)
                    ),
                    "base_gradient_norm": _distribution(
                        group["base_gradient_norm"].astype(float)
                    ),
                    "candidate_gradient_norm": _distribution(
                        group["candidate_gradient_norm"].astype(float)
                    ),
                }
    return output


def _chunks(
    items: Sequence[CounterfactualPresentationItem], size: int
) -> Iterable[Sequence[CounterfactualPresentationItem]]:
    for start in range(0, len(items), int(size)):
        yield items[start : start + int(size)]


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_state_unchanged(
    model: torch.nn.Module, before: Mapping[str, torch.Tensor]
) -> None:
    after = model.state_dict()
    if set(after) != set(before):
        raise RuntimeError("model state keys changed during gradient audit")
    changed = [
        name
        for name, value in after.items()
        if not torch.equal(value.detach().cpu(), before[name])
    ]
    if changed:
        raise RuntimeError(f"model state changed during gradient audit: {changed[:3]}")
    populated = [name for name, value in model.named_parameters() if value.grad is not None]
    if populated:
        raise RuntimeError(f"parameter .grad buffers were populated: {populated[:3]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only train-pair gradient conflict audit for balanced_v2; "
            "validation and test graph assets are never loaded."
        )
    )
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--candidate-checkpoint", default=None)
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--graph-manifest", default=None)
    parser.add_argument("--poison-manifest", default=None)
    parser.add_argument("--samples-per-regime", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--audit-seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output", default=None)
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    candidate_result_path = Path(args.candidate_result).resolve()
    result = _read_json(candidate_result_path)
    config = result.get("config") or {}
    protocol = result.get("training_protocol") or {}
    scope = result.get("scope") or {}
    if scope.get("test_accessed") is not False:
        raise RuntimeError("candidate does not attest that test was untouched")
    if config.get("evaluate_test") is not False:
        raise RuntimeError("candidate was not configured as development-only")
    if str(protocol.get("objective_profile")) != BALANCED_V2_PROFILE:
        raise RuntimeError("audit requires the exact balanced_v2 objective profile")
    loss_protocol = protocol.get("counterfactual_loss") or {}
    observed_weights = {
        "presentation_base_bce": 1.0,
        "target_pair_bce": float(loss_protocol.get("target_weight", np.nan)),
        "ranking": float(loss_protocol.get("ranking_weight", np.nan)),
        "non_target_consistency": float(
            loss_protocol.get("consistency_weight", np.nan)
        ),
        "adjacent_negative_consistency": float(
            loss_protocol.get("adjacent_negative_consistency_weight", np.nan)
        ),
    }
    if observed_weights != COMPONENT_WEIGHTS:
        raise RuntimeError(f"candidate loss weights differ: {observed_weights}")

    graph_dir = Path(args.graph_dir or config["graph_dir"]).resolve()
    graph_manifest_path = Path(
        args.graph_manifest or (result.get("graph_manifest") or {})["path"]
    ).resolve()
    poison_manifest_path = Path(
        args.poison_manifest or (result.get("poison_manifest") or {})["path"]
    ).resolve()
    checkpoint_path = Path(
        args.candidate_checkpoint or (result.get("checkpoint") or {})["path"]
    ).resolve()
    output_path = Path(args.output).resolve()
    records_output_path = (
        Path(args.records_output).resolve()
        if args.records_output
        else output_path.with_name(f"{output_path.stem}.records.csv")
    )

    graph_manifest_hash = sha256_file(graph_manifest_path)
    poison_manifest_hash = sha256_file(poison_manifest_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if graph_manifest_hash != str((result.get("graph_manifest") or {}).get("sha256")):
        raise RuntimeError("candidate graph-manifest binding differs")
    if poison_manifest_hash != str((result.get("poison_manifest") or {}).get("sha256")):
        raise RuntimeError("candidate poison-manifest binding differs")
    if checkpoint_hash != str((result.get("checkpoint") or {}).get("sha256")):
        raise RuntimeError("candidate checkpoint binding differs")

    label_config = RiskLabelConfig(
        label_mode="dynamic_risk",
        risk_base_distance_m=5.0,
        risk_reaction_time_s=1.0,
        risk_safe_decel_mps2=4.0,
    )
    manifest = load_manifest(
        graph_dir, graph_manifest_path, require_graph_sha256=True
    )
    poison, verified_poison_hash = load_poison_manifest(
        poison_manifest_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=graph_manifest_hash,
    )
    if verified_poison_hash != poison_manifest_hash:
        raise RuntimeError("poison manifest changed while being verified")
    if set(poison["split"].astype(str)) != {"train"}:
        raise RuntimeError("gradient audit refuses non-training poison rows")
    poison_metadata_path = poison_manifest_path.with_suffix(
        poison_manifest_path.suffix + ".metadata.json"
    )
    poison_metadata = _read_json(poison_metadata_path)
    if poison_metadata.get("test_access_authorized") is not False:
        raise RuntimeError("gradient audit requires a manifest forbidding test access")
    if poison_metadata.get("formal_test_eligible") is not False:
        raise RuntimeError("gradient audit requires a development-only manifest")
    if str(poison_metadata.get("edge_feature_mode")) != MOTION_NORMALIZED_EDGE_FEATURE_MODE:
        raise RuntimeError("gradient audit requires the candidate's 42-D edge mode")

    selected_poison = stable_regime_sample(
        poison,
        samples_per_regime=int(args.samples_per_regime),
        audit_seed=int(args.audit_seed),
    )
    selected_ids = set(selected_poison["scenario_id"].astype(str))
    selected_manifest = manifest[
        manifest["scenario_id"].astype(str).isin(selected_ids)
    ].copy()
    if len(selected_manifest) != len(selected_poison):
        raise RuntimeError("not every sampled poison row resolved to one graph")
    if set(selected_manifest["split"].astype(str)) != {"train"}:
        raise RuntimeError("sampled graph resolution escaped train")

    clean_graphs, clean_stats = load_split_graphs(
        selected_manifest,
        graph_dir,
        "train",
        True,
        None,
        label_config,
        None,
        True,
        MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        None,
    )
    triggered_graphs, triggered_stats = load_split_graphs(
        selected_manifest,
        graph_dir,
        "train",
        True,
        None,
        label_config,
        selected_poison,
        True,
        MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        None,
    )
    items, presentation = compose_counterfactual_training_items(
        clean_graphs, triggered_graphs
    )
    paired_by_scenario = {
        str(item.graph.scenario_id): item
        for item in items
        if item.has_counterfactual_pair
    }
    if set(paired_by_scenario) != selected_ids:
        raise RuntimeError("counterfactual composition changed the selected sample")

    node_dim = int(clean_graphs[0].x.shape[1])
    edge_dim = int(clean_graphs[0].edge_attr.shape[1])
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = build_model(
        **model_kwargs_from_result(result, node_dim=node_dim, edge_dim=edge_dim)
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    before = _state_snapshot(model)
    named_parameters = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    parameter_names = [name for name, _ in named_parameters]
    parameters = [value for _, value in named_parameters]
    parameter_blocks = {name: parameter_block(name) for name in parameter_names}
    block_indices = {
        "all": list(range(len(parameters))),
        **{
            block: [
                index
                for index, name in enumerate(parameter_names)
                if parameter_blocks[name] == block
            ]
            for block in BLOCKS
            if block != "all"
        },
    }
    if any(not block_indices[block] for block in BLOCKS):
        raise RuntimeError("at least one requested parameter block is empty")

    loss_config = CounterfactualLossConfig(
        target_weight=0.1,
        ranking_weight=0.1,
        consistency_weight=1.0,
        adjacent_negative_consistency_weight=5.0,
        ranking_margin=1.0,
        max_gradient_norm=5.0,
    )
    records: list[dict[str, Any]] = []
    accumulated_gradients: dict[
        str, dict[str, tuple[torch.Tensor | None, ...]]
    ] = {regime: {} for regime in REGIMES}
    batch_count = 0
    for regime in REGIMES:
        regime_rows = selected_poison[
            selected_poison["motion_regime"].astype(str) == regime
        ].sort_values("scenario_id")
        regime_items = [
            paired_by_scenario[str(scenario_id)]
            for scenario_id in regime_rows["scenario_id"].astype(str)
        ]
        for regime_batch_index, chunk in enumerate(
            _chunks(regime_items, int(args.batch_size)), start=1
        ):
            batch = collate_counterfactual_items(list(chunk))
            if batch.clean_counterparts is None:
                raise RuntimeError("audit batch has no clean counterparts")
            clean = batch.clean_counterparts.to(device)
            clean_logits = model(
                clean.x,
                clean.edge_index,
                clean.edge_attr,
                clean.message_passing_edge_mask,
            )
            clean_mask = clean.supervision_edge_mask.bool()
            clean_task_bce = F.binary_cross_entropy_with_logits(
                clean_logits[clean_mask], clean.edge_label[clean_mask]
            )
            losses, counts = counterfactual_loss_components(
                model, batch, config=loss_config, device=device
            )
            component_losses = {
                "clean_task_bce": clean_task_bce,
                "presentation_base_bce": losses["base_bce"],
                "target_pair_bce": losses["target_pair_bce"],
                "ranking": losses["ranking"],
                "non_target_consistency": losses["non_target_consistency"],
                "adjacent_negative_consistency": losses[
                    "adjacent_negative_consistency"
                ],
            }
            gradients = {
                name: _gradient_tuple(loss, parameters)
                for name, loss in component_losses.items()
            }
            explicit = combine_gradients(
                gradients,
                {
                    name: weight
                    for name, weight in COMPONENT_WEIGHTS.items()
                    if name != "presentation_base_bce"
                },
            )
            actual = combine_gradients(
                {
                    **gradients,
                    "weighted_explicit_aux": explicit,
                },
                {
                    "presentation_base_bce": 1.0,
                    "weighted_explicit_aux": 1.0,
                },
            )
            excess = subtract_gradients(actual, gradients["clean_task_bce"])
            comparison_gradients = {
                name: gradients[name] for name in COMPONENT_WEIGHTS
            }
            comparison_gradients.update(
                {
                    "weighted_explicit_aux": explicit,
                    "actual_poison_update": actual,
                    "poison_update_excess_over_clean": excess,
                }
            )
            weights = {
                **COMPONENT_WEIGHTS,
                "weighted_explicit_aux": 1.0,
                "actual_poison_update": 1.0,
                "poison_update_excess_over_clean": 1.0,
            }
            gradients_for_accumulation = {
                "clean_task_bce": gradients["clean_task_bce"],
                **comparison_gradients,
            }
            for component, component_gradient in gradients_for_accumulation.items():
                copied = cpu_gradient_copy(component_gradient)
                previous = accumulated_gradients[regime].get(component)
                accumulated_gradients[regime][component] = (
                    copied
                    if previous is None
                    else combine_gradients(
                        {"previous": previous, "current": copied},
                        {"previous": 1.0, "current": 1.0},
                    )
                )
            loss_values = {
                name: float(loss.detach().cpu())
                for name, loss in component_losses.items()
            }
            for component, candidate_gradient in comparison_gradients.items():
                for block in BLOCKS:
                    comparison = compare_gradients(
                        gradients["clean_task_bce"],
                        candidate_gradient,
                        block_indices[block],
                        training_weight=weights[component],
                    )
                    records.append(
                        {
                            "motion_regime": regime,
                            "regime_batch_index": int(regime_batch_index),
                            "pairs": int(batch.pair_count),
                            "base_supervised_edges": int(clean_mask.sum().item()),
                            "target_edges": int(counts["target_edges"]),
                            "consistency_edges": int(counts["consistency_edges"]),
                            "adjacent_negative_edges": int(
                                counts["adjacent_negative_edges"]
                            ),
                            "component": component,
                            "parameter_block": block,
                            "clean_task_bce": loss_values["clean_task_bce"],
                            "component_loss": loss_values.get(component),
                            **comparison,
                        }
                    )
            batch_count += 1
            print(
                "[gradient_conflict_audit] "
                f"regime={regime} batch={regime_batch_index}/"
                f"{int(np.ceil(len(regime_items) / int(args.batch_size)))}",
                flush=True,
            )
            del gradients, comparison_gradients, losses, clean_task_bce

    _assert_state_unchanged(model, before)
    _atomic_csv(records, records_output_path)
    accumulated_gradients["all"] = {
        component: combine_gradients(
            {
                regime: accumulated_gradients[regime][component]
                for regime in REGIMES
            },
            {regime: 1.0 for regime in REGIMES},
        )
        for component in accumulated_gradients[REGIMES[0]]
    }
    aggregate_direction: dict[str, Any] = {}
    for regime in ("all", *REGIMES):
        aggregate_direction[regime] = {}
        base_gradient = accumulated_gradients[regime]["clean_task_bce"]
        for component, component_gradient in accumulated_gradients[regime].items():
            if component == "clean_task_bce":
                continue
            aggregate_direction[regime][component] = {
                block: compare_gradients(
                    base_gradient,
                    component_gradient,
                    block_indices[block],
                    training_weight=weights[component],
                )
                for block in BLOCKS
            }
    selected_hash = hashlib.sha256(
        "\n".join(
            f"{row.motion_regime}|{row.scenario_id}"
            for row in selected_poison[["motion_regime", "scenario_id"]]
            .sort_values(["motion_regime", "scenario_id"])
            .itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()
    result_payload: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "scope": {
            "development_only": True,
            "split": "train",
            "train_graphs_read": int(len(selected_poison)),
            "validation_graphs_read": 0,
            "test_graphs_read": 0,
            "validation_metadata_read": False,
            "model_training_performed": False,
            "optimizer_created": False,
            "parameter_grad_buffers_populated": False,
            "model_state_unchanged": True,
            "frozen_assets_modified": False,
        },
        "configuration": {
            "audit_seed": int(args.audit_seed),
            "samples_per_regime": int(args.samples_per_regime),
            "batch_size": int(args.batch_size),
            "batches": int(batch_count),
            "device": str(device),
            "model_mode": "eval_deterministic_dropout_disabled",
            "clean_reference_gradient": "clean_full_graph_supervised_bce",
            "actual_poison_update": (
                "poisoned_presentation_bce_plus_weighted_explicit_auxiliary"
            ),
            "component_weights": COMPONENT_WEIGHTS,
            "parameter_blocks": {
                block: {
                    "parameter_tensors": int(len(block_indices[block])),
                    "parameters": int(
                        sum(parameters[index].numel() for index in block_indices[block])
                    ),
                }
                for block in BLOCKS
            },
        },
        "assets": {
            "candidate_result": {
                "path": str(candidate_result_path),
                "sha256": sha256_file(candidate_result_path),
            },
            "candidate_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
            },
            "graph_manifest": {
                "path": str(graph_manifest_path),
                "sha256": graph_manifest_hash,
            },
            "poison_manifest": {
                "path": str(poison_manifest_path),
                "sha256": poison_manifest_hash,
                "metadata_path": str(poison_metadata_path),
                "metadata_sha256": sha256_file(poison_metadata_path),
            },
        },
        "sample": {
            "rows": int(len(selected_poison)),
            "scenario_set_sha256": selected_hash,
            "per_regime": {
                regime: int(
                    (selected_poison["motion_regime"].astype(str) == regime).sum()
                )
                for regime in REGIMES
            },
            "clean_loader": clean_stats,
            "triggered_loader": triggered_stats,
            "presentation": presentation,
        },
        "records": {
            "path": str(records_output_path),
            "rows": int(len(records)),
        },
        "summary": summarize_gradient_records(records),
        "aggregate_batch_sum_gradient_direction": aggregate_direction,
    }
    _atomic_json(result_payload, output_path)
    result_payload["output"] = {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "records_sha256": sha256_file(records_output_path),
    }
    return result_payload


def main() -> None:
    result = run_audit(parse_args())
    print(f"[gradient_conflict_audit] saved={result['output']['path']}")


if __name__ == "__main__":
    main()
