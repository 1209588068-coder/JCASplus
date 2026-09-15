#!/usr/bin/env python3
"""Development-only trainer with an explicit counterfactual pair objective.

The frozen trigger, moving-prioritized poison manifest, 42-dimensional edge
representation, GENConv architecture, optimizer, and clean-validation
checkpoint selection remain unchanged.  Relative to the ordinary-BCE paired
pilot, this workflow adds three train-only terms for each triggered row:

* clean-negative / triggered-positive target-pair classification;
* a directional logit-ranking margin on the same pair;
* clean/triggered probability consistency on every non-target supervised edge.
* an optional focused consistency term on adjacent negative edges.

The opt-in projected_v3 pilot projects the extra gradient relative to the
entire mixed batch's clean counterpart. Old calls default to no projection.

Only train and validation loaders exist.  There is deliberately no test flag.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from jcas.core.models import build_model
from jcas.core.counterfactual_gradients import (
    PROJECTED_V3_PROFILE,
    CleanProjectionConfig,
    assign_flat_gradient,
    direction_stats,
    flatten_gradients,
    project_extra_gradient,
)
from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
    edge_feature_protocol,
)
from jcas.core.poison import load_poison_manifest, sha256_file
from jcas.core.risk_labels import (
    RiskLabelConfig,
    label_config_dict,
    label_config_hash,
)
from jcas.workflows.trainer import (
    best_threshold,
    build_optimizer,
    evaluate,
    load_manifest,
    load_split_graphs,
    metric_dict,
    node_path_diagnostics,
    resolve_manifest_path,
    set_seed,
    validate_graph_manifest_contract,
)


WORKFLOW_VERSION = "explicit_counterfactual_objective_training_v2"
BALANCED_V2_PROFILE = "balanced_v2_t0p1_r0p1_c1_adj5_m1"


@dataclass(frozen=True)
class CounterfactualLossConfig:
    target_weight: float = 0.25
    ranking_weight: float = 0.25
    consistency_weight: float = 1.0
    adjacent_negative_consistency_weight: float = 0.0
    ranking_margin: float = 1.0
    max_gradient_norm: float = 5.0

    def validate(self) -> None:
        for name in (
            "target_weight",
            "ranking_weight",
            "consistency_weight",
            "adjacent_negative_consistency_weight",
            "ranking_margin",
            "max_gradient_norm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        if self.target_weight == 0.0 and self.ranking_weight == 0.0:
            raise ValueError("at least one target-specific loss must be enabled")


def objective_profile(config: CounterfactualLossConfig) -> str:
    balanced = {
        "target_weight": 0.10,
        "ranking_weight": 0.10,
        "consistency_weight": 1.0,
        "adjacent_negative_consistency_weight": 5.0,
        "ranking_margin": 1.0,
        "max_gradient_norm": 5.0,
    }
    if all(
        abs(float(getattr(config, key)) - expected) <= 1e-12
        for key, expected in balanced.items()
    ):
        return BALANCED_V2_PROFILE
    return "custom_explicit_counterfactual_v2"


@dataclass(frozen=True)
class CounterfactualPresentationItem:
    graph: Data
    clean_counterpart: Data | None = None
    triggered_counterpart: Data | None = None

    @property
    def has_counterfactual_pair(self) -> bool:
        return (
            self.clean_counterpart is not None
            and self.triggered_counterpart is not None
        )


@dataclass(frozen=True)
class CounterfactualBatch:
    base: Batch
    clean_counterparts: Batch | None
    triggered_counterparts: Batch | None
    pair_count: int
    clean_base: Batch | None = None


def adjacent_negative_mask_for_pair(
    clean: Data,
    triggered: Data,
) -> torch.Tensor:
    """Return supervised negative edges incident to either target endpoint."""
    if not torch.equal(clean.edge_index, triggered.edge_index):
        raise ValueError("counterfactual pair changed topology")
    target_mask = triggered.poisoned_edge_mask.bool()
    if int(target_mask.sum().item()) != 2:
        raise ValueError("adjacent mask requires exactly two target directions")
    target_nodes = torch.unique(triggered.edge_index[:, target_mask])
    if int(target_nodes.numel()) != 2:
        raise ValueError("target directions do not form one unordered pair")
    edge_index = triggered.edge_index
    endpoint = torch.isin(edge_index[0], target_nodes)
    endpoint |= torch.isin(edge_index[1], target_nodes)
    mask = (
        endpoint
        & ~target_mask
        & triggered.supervision_edge_mask.bool()
        & (clean.edge_label == 0)
    )
    if torch.any(
        clean.edge_label[~target_mask] != triggered.edge_label[~target_mask]
    ):
        raise ValueError("counterfactual transform changed a non-target label")
    return mask


def compose_counterfactual_training_items(
    clean_graphs: list[Data],
    triggered_poison_graphs: list[Data],
) -> tuple[list[CounterfactualPresentationItem], dict[str, Any]]:
    """Keep all clean rows and append bound triggered/clean pair rows."""
    clean_by_scenario: dict[str, Data] = {}
    for graph in clean_graphs:
        scenario_id = str(graph.scenario_id)
        if scenario_id in clean_by_scenario:
            raise ValueError("clean graph list contains duplicate scenarios")
        clean_by_scenario[scenario_id] = graph
    triggered_by_scenario: dict[str, Data] = {}
    adjacent_negative_edges = 0
    for graph in triggered_poison_graphs:
        scenario_id = str(graph.scenario_id)
        if scenario_id in triggered_by_scenario:
            raise ValueError("triggered graph list contains duplicate scenarios")
        clean = clean_by_scenario.get(scenario_id)
        if clean is None:
            raise ValueError("triggered scenario is absent from clean training rows")
        if int(graph.poisoned_edge_mask.sum().item()) != 2:
            raise ValueError("each triggered graph must contain one unordered pair")
        if int(clean.poisoned_edge_mask.sum().item()) != 0:
            raise ValueError("clean counterpart unexpectedly contains poisoned edges")
        if not torch.equal(clean.edge_index, graph.edge_index):
            raise ValueError("counterfactual pair changed graph topology")
        if clean.x.shape != graph.x.shape or clean.edge_attr.shape != graph.edge_attr.shape:
            raise ValueError("counterfactual pair changed tensor dimensions")
        target_mask = graph.poisoned_edge_mask
        if not clean.supervision_edge_mask[target_mask].all():
            raise ValueError("clean target pair is outside supervision")
        if not graph.supervision_edge_mask[target_mask].all():
            raise ValueError("triggered target pair is outside supervision")
        if torch.any(clean.edge_label[target_mask] != 0):
            raise ValueError("clean target pair is not negative")
        if torch.any(graph.edge_label[target_mask] != 1):
            raise ValueError("triggered target pair is not positive")
        adjacent_negative_mask = adjacent_negative_mask_for_pair(clean, graph)
        graph.counterfactual_adjacent_negative_mask = adjacent_negative_mask
        adjacent_negative_edges += int(adjacent_negative_mask.sum().item())
        triggered_by_scenario[scenario_id] = graph
    if not triggered_by_scenario:
        raise ValueError("explicit counterfactual workflow requires poison rows")

    # PyG batches require every graph in a batch to expose the same attributes.
    # Clean/base rows do not have adjacent-negative counterfactual edges, so give
    # them an explicit empty mask to match triggered rows.
    for graph in clean_graphs:
        graph.counterfactual_adjacent_negative_mask = torch.zeros(
            graph.edge_label.shape,
            dtype=torch.bool,
            device=graph.edge_label.device,
        )

    items = [CounterfactualPresentationItem(graph=graph) for graph in clean_graphs]
    for scenario_id in sorted(triggered_by_scenario):
        triggered = triggered_by_scenario[scenario_id]
        items.append(
            CounterfactualPresentationItem(
                graph=triggered,
                clean_counterpart=clean_by_scenario[scenario_id],
                triggered_counterpart=triggered,
            )
        )
    return items, {
        "workflow_version": WORKFLOW_VERSION,
        "presentation": "all_clean_plus_bound_triggered_counterparts",
        "unique_clean_scenarios": int(len(clean_graphs)),
        "poison_scenarios": int(len(triggered_poison_graphs)),
        "final_training_rows": int(len(items)),
        "clean_counterpart_rows": int(len(triggered_poison_graphs)),
        "counterfactual_pairs_per_epoch": int(len(triggered_poison_graphs)),
        "adjacent_negative_edges_per_epoch": int(adjacent_negative_edges),
    }


def collate_counterfactual_items(
    items: list[CounterfactualPresentationItem],
    *,
    include_clean_base: bool = False,
) -> CounterfactualBatch:
    if not items:
        raise ValueError("cannot collate an empty counterfactual batch")
    paired = [item for item in items if item.has_counterfactual_pair]
    for item in items:
        if (item.clean_counterpart is None) != (
            item.triggered_counterpart is None
        ):
            raise ValueError("counterfactual item contains only one counterpart")
    return CounterfactualBatch(
        base=Batch.from_data_list([item.graph for item in items]),
        clean_counterparts=(
            Batch.from_data_list(
                [item.clean_counterpart for item in paired]
            )
            if paired
            else None
        ),
        triggered_counterparts=(
            Batch.from_data_list(
                [item.triggered_counterpart for item in paired]
            )
            if paired
            else None
        ),
        pair_count=int(len(paired)),
        clean_base=(
            Batch.from_data_list([
                item.clean_counterpart if item.has_counterfactual_pair else item.graph
                for item in items
            ])
            if include_clean_base and paired else None
        ),
    )


def _full_logits(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.message_passing_edge_mask,
    )


def _paired_logits_with_shared_dropout(
    model: torch.nn.Module,
    clean: Batch,
    triggered: Batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a pair with identical stochastic-dropout RNG streams."""
    cpu_state = torch.random.get_rng_state()
    cuda_state: torch.Tensor | None = None
    cuda_device: torch.device | None = None
    if clean.x.is_cuda:
        cuda_device = clean.x.device
        cuda_state = torch.cuda.get_rng_state(cuda_device)
    clean_logits = _full_logits(model, clean)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None and cuda_device is not None:
        torch.cuda.set_rng_state(cuda_state, cuda_device)
    triggered_logits = _full_logits(model, triggered)
    return clean_logits, triggered_logits


def counterfactual_loss_components(
    model: torch.nn.Module,
    batch: CounterfactualBatch,
    *,
    config: CounterfactualLossConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Compute base BCE and explicit same-pair auxiliary losses."""
    config.validate()
    base = batch.base.to(device)
    base_full_logits = _full_logits(model, base)
    base_mask = base.supervision_edge_mask
    if not base_mask.any():
        raise RuntimeError("counterfactual batch contains no supervised base edge")
    base_bce = F.binary_cross_entropy_with_logits(
        base_full_logits[base_mask], base.edge_label[base_mask]
    )
    zero = base_bce.new_zeros(())
    target_bce = zero
    ranking_loss = zero
    consistency_loss = zero
    adjacent_negative_consistency_loss = zero
    target_edges = 0
    consistency_edges = 0
    adjacent_negative_edges = 0

    if batch.pair_count:
        if batch.clean_counterparts is None or batch.triggered_counterparts is None:
            raise RuntimeError("paired batch is missing its counterparts")
        clean = batch.clean_counterparts.to(device)
        triggered = batch.triggered_counterparts.to(device)
        if clean.edge_index.shape != triggered.edge_index.shape:
            raise RuntimeError("batched counterfactual topology dimensions differ")
        target_mask = triggered.poisoned_edge_mask
        target_edges = int(target_mask.sum().item())
        if target_edges != 2 * int(batch.pair_count):
            raise RuntimeError("batched target-mask accounting mismatch")
        if torch.any(clean.edge_label[target_mask] != 0):
            raise RuntimeError("batched clean target is not negative")
        if torch.any(triggered.edge_label[target_mask] != 1):
            raise RuntimeError("batched triggered target is not positive")
        clean_logits, triggered_logits = _paired_logits_with_shared_dropout(
            model, clean, triggered
        )
        clean_target = clean_logits[target_mask]
        triggered_target = triggered_logits[target_mask]
        target_bce = 0.5 * (
            F.binary_cross_entropy_with_logits(
                clean_target, torch.zeros_like(clean_target)
            )
            + F.binary_cross_entropy_with_logits(
                triggered_target, torch.ones_like(triggered_target)
            )
        )
        ranking_loss = F.softplus(
            float(config.ranking_margin)
            - (triggered_target - clean_target)
        ).mean()
        consistency_mask = (
            triggered.supervision_edge_mask & ~target_mask
        )
        consistency_edges = int(consistency_mask.sum().item())
        if not hasattr(triggered, "counterfactual_adjacent_negative_mask"):
            raise RuntimeError("paired batch lacks adjacent-negative mask")
        adjacent_negative_mask = (
            triggered.counterfactual_adjacent_negative_mask.bool()
        )
        if torch.any(adjacent_negative_mask & ~consistency_mask):
            raise RuntimeError("adjacent-negative mask escaped non-target supervision")
        adjacent_negative_edges = int(adjacent_negative_mask.sum().item())
        clean_probability_all = torch.sigmoid(clean_logits).detach()
        triggered_probability_all = torch.sigmoid(triggered_logits)
        if consistency_edges:
            consistency_loss = F.mse_loss(
                triggered_probability_all[consistency_mask],
                clean_probability_all[consistency_mask],
            )
        if adjacent_negative_edges:
            adjacent_negative_consistency_loss = F.mse_loss(
                triggered_probability_all[adjacent_negative_mask],
                clean_probability_all[adjacent_negative_mask],
            )

    total = (
        base_bce
        + float(config.target_weight) * target_bce
        + float(config.ranking_weight) * ranking_loss
        + float(config.consistency_weight) * consistency_loss
        + float(config.adjacent_negative_consistency_weight)
        * adjacent_negative_consistency_loss
    )
    return (
        {
            "total": total,
            "base_bce": base_bce,
            "target_pair_bce": target_bce,
            "ranking": ranking_loss,
            "non_target_consistency": consistency_loss,
            "adjacent_negative_consistency": (
                adjacent_negative_consistency_loss
            ),
        },
        {
            "base_edges": int(base_mask.sum().item()),
            "counterfactual_pairs": int(batch.pair_count),
            "target_edges": int(target_edges),
            "consistency_edges": int(consistency_edges),
            "adjacent_negative_edges": int(adjacent_negative_edges),
        },
    )


def mixed_clean_reference_gradient(
    model: torch.nn.Module,
    batch: CounterfactualBatch,
    parameters: list[torch.nn.Parameter],
    *,
    device: torch.device,
) -> tuple[tuple[torch.Tensor | None, ...], float]:
    """All-clean version of the SAME mixed batch, with base-forward RNG reused.

    fork_rng restores CPU/CUDA RNG after this extra forward, so the following
    balanced objective uses the identical base dropout masks and the original
    stochastic stream. Only LayerNorm (the pilot's architecture) is supported.
    """
    if batch.clean_base is None:
        raise RuntimeError("projection requires the entire mixed batch's clean copy")
    clean = batch.clean_base.to(device)
    base = batch.base.to(device)
    if (
        clean.x.shape != base.x.shape
        or clean.edge_attr.shape != base.edge_attr.shape
        or not torch.equal(clean.edge_index, base.edge_index)
        or not torch.equal(clean.supervision_edge_mask, base.supervision_edge_mask)
        or not torch.equal(clean.message_passing_edge_mask, base.message_passing_edge_mask)
        or clean.scenario_id != base.scenario_id
        or bool(clean.poisoned_edge_mask.any())
    ):
        raise RuntimeError("clean reference differs in mixed-batch identity/topology/masks")
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError("clean-reference forward requires stateless normalization")
    devices = [clean.x.device.index] if clean.x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        logits = _full_logits(model, clean)
        mask = clean.supervision_edge_mask.bool()
        loss = F.binary_cross_entropy_with_logits(logits[mask], clean.edge_label[mask])
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return gradients, float(loss.detach())


def train_counterfactual_epoch(
    model: torch.nn.Module,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    config: CounterfactualLossConfig,
    device: torch.device,
    projection: CleanProjectionConfig | None = None,
) -> dict[str, Any]:
    model.train()
    steps = 0
    counterfactual_steps = 0
    total_step_loss = 0.0
    base_loss_sum = 0.0
    base_edges = 0
    target_loss_sum = 0.0
    ranking_loss_sum = 0.0
    counterfactual_pairs = 0
    consistency_loss_sum = 0.0
    consistency_edges = 0
    adjacent_negative_consistency_loss_sum = 0.0
    adjacent_negative_edges = 0
    gradient_norm_sum = 0.0
    gradient_norm_max = 0.0
    parameters = [p for p in model.parameters() if p.requires_grad]
    projection_records: list[dict[str, Any]] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        clean_gradients = None
        if projection is not None and batch.pair_count:
            clean_gradients, clean_bce = mixed_clean_reference_gradient(
                model, batch, parameters, device=device
            )
        losses, counts = counterfactual_loss_components(
            model, batch, config=config, device=device
        )
        diagnostics = None
        if clean_gradients is None:
            losses["total"].backward()
        else:
            total_gradients = torch.autograd.grad(
                losses["total"], parameters, allow_unused=True
            )
            clean_vector = flatten_gradients(clean_gradients, parameters)
            total_vector = flatten_gradients(total_gradients, parameters)
            update, diagnostics = project_extra_gradient(
                clean_vector, total_vector, config=projection
            )
            assign_flat_gradient(update, parameters, used=[
                c is not None or t is not None
                for c, t in zip(clean_gradients, total_gradients)
            ])
            before_step = torch.cat([p.detach().reshape(-1) for p in parameters])
            diagnostics["clean_reference_bce"] = clean_bce
        gradient_norm = float(
            clip_grad_norm_(
                model.parameters(), float(config.max_gradient_norm),
                error_if_nonfinite=True,
            )
        )
        optimizer.step()
        if diagnostics is not None:
            after_step = torch.cat([p.detach().reshape(-1) for p in parameters])
            if not torch.isfinite(after_step).all():
                raise FloatingPointError("non-finite parameters after AdamW step")
            # Descent direction = before - after, including momentum/decay.
            actual_step = direction_stats(clean_vector, before_step - after_step)
            diagnostics["adamw_descent_clean_cosine"] = actual_step["cosine"]
            diagnostics["adamw_descent_clean_dot"] = actual_step["dot"]
            projection_records.append(diagnostics)

        steps += 1
        total_step_loss += float(losses["total"].detach().cpu())
        base_edges += int(counts["base_edges"])
        base_loss_sum += float(losses["base_bce"].detach().cpu()) * int(
            counts["base_edges"]
        )
        if counts["counterfactual_pairs"]:
            counterfactual_steps += 1
            counterfactual_pairs += int(counts["counterfactual_pairs"])
            target_loss_sum += float(
                losses["target_pair_bce"].detach().cpu()
            ) * int(counts["counterfactual_pairs"])
            ranking_loss_sum += float(losses["ranking"].detach().cpu()) * int(
                counts["counterfactual_pairs"]
            )
            consistency_edges += int(counts["consistency_edges"])
            consistency_loss_sum += float(
                losses["non_target_consistency"].detach().cpu()
            ) * int(counts["consistency_edges"])
            adjacent_negative_edges += int(counts["adjacent_negative_edges"])
            adjacent_negative_consistency_loss_sum += float(
                losses["adjacent_negative_consistency"].detach().cpu()
            ) * int(counts["adjacent_negative_edges"])
        gradient_norm_sum += gradient_norm
        gradient_norm_max = max(gradient_norm_max, gradient_norm)
    if not steps or not base_edges:
        raise RuntimeError("counterfactual training loader produced no update")
    summary = {
        "steps": int(steps),
        "counterfactual_steps": int(counterfactual_steps),
        "counterfactual_pairs": int(counterfactual_pairs),
        "base_edges": int(base_edges),
        "target_edges": int(2 * counterfactual_pairs),
        "consistency_edges": int(consistency_edges),
        "adjacent_negative_edges": int(adjacent_negative_edges),
        "total_step_loss_mean": float(total_step_loss / steps),
        "base_bce_edge_mean": float(base_loss_sum / base_edges),
        "target_pair_bce_pair_mean": (
            float(target_loss_sum / counterfactual_pairs)
            if counterfactual_pairs
            else None
        ),
        "ranking_pair_mean": (
            float(ranking_loss_sum / counterfactual_pairs)
            if counterfactual_pairs
            else None
        ),
        "non_target_consistency_edge_mean": (
            float(consistency_loss_sum / consistency_edges)
            if consistency_edges
            else None
        ),
        "adjacent_negative_consistency_edge_mean": (
            float(
                adjacent_negative_consistency_loss_sum
                / adjacent_negative_edges
            )
            if adjacent_negative_edges
            else None
        ),
        "preclip_gradient_norm_mean": float(gradient_norm_sum / steps),
        "preclip_gradient_norm_max": float(gradient_norm_max),
    }
    if projection is not None:
        summary["gradient_projection"] = {
            "paired_steps": len(projection_records),
            "projected_steps": sum(int(r["projected"]) for r in projection_records),
            "capped_steps": sum(int(r["capped"]) for r in projection_records),
            "clean_norm_floor_steps": sum(int(r["clean_norm_floor_hit"]) for r in projection_records),
            "adamw_descent_conflict_steps": sum(
                int(r["adamw_descent_clean_dot"] < 0) for r in projection_records
            ),
            "constraint_violations": 0,  # Any violation raises before optimizer.step.
            "per_step": projection_records,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a development-only explicit counterfactual victim."
    )
    parser.add_argument("--graph-dir", default="graphs/av2mf_graph_v4")
    parser.add_argument(
        "--graph-manifest",
        default="graphs/av2mf_graph_v4/manifest_grouped_v5.csv",
    )
    parser.add_argument("--poison-manifest", required=True)
    parser.add_argument("--target-rotation-plan", default=None,
                        help="Opt-in frozen moving-target role/schedule plan; no holdout training")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", choices=("layer",), default="layer")
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--target-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument(
        "--adjacent-negative-consistency-weight", type=float, default=0.0
    )
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--gradient-projection", choices=("none", "clean_reference"), default="none")
    parser.add_argument("--max-extra-norm-ratio", type=float, default=1.0)
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-val-graphs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _atomic_torch_save(state: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.target_rotation_plan:
        from jcas.workflows.rotating_counterfactual_pilot import validate_rotation_arguments
        validate_rotation_arguments(args)
    if int(args.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if args.max_train_graphs is not None and int(args.max_train_graphs) <= 0:
        raise ValueError("max_train_graphs must be positive")
    if args.max_val_graphs is not None and int(args.max_val_graphs) <= 0:
        raise ValueError("max_val_graphs must be positive")
    loss_config = CounterfactualLossConfig(
        target_weight=float(args.target_weight),
        ranking_weight=float(args.ranking_weight),
        consistency_weight=float(args.consistency_weight),
        adjacent_negative_consistency_weight=float(
            args.adjacent_negative_consistency_weight
        ),
        ranking_margin=float(args.ranking_margin),
        max_gradient_norm=float(args.max_gradient_norm),
    )
    loss_config.validate()
    selected_objective_profile = objective_profile(loss_config)
    projection = None
    if args.gradient_projection == "clean_reference":
        projection = CleanProjectionConfig(max_extra_norm_ratio=args.max_extra_norm_ratio)
        projection.validate()
        if selected_objective_profile != BALANCED_V2_PROFILE or args.max_extra_norm_ratio != 1.0:
            raise ValueError("projected_v3 pilot fixes balanced_v2 weights and extra norm ratio=1")
        selected_objective_profile = PROJECTED_V3_PROFILE
        reproduction_root = Path(__file__).resolve().parents[2] / "record" / "reproduction"
        if not Path(args.output_dir).resolve().is_relative_to(reproduction_root):
            raise ValueError("projected_v3 outputs must be under record/reproduction")

    if args.target_rotation_plan and (
        projection is None or args.max_train_graphs is not None or args.max_val_graphs is not None
        or args.seed != 20260621 or args.epochs != 50
    ):
        raise ValueError("rotation pilot requires projected weights, seed20260621, 50 epochs and full train/clean-val")

    set_seed(int(args.seed))
    graph_dir = Path(args.graph_dir)
    graph_manifest_path = resolve_manifest_path(graph_dir, args.graph_manifest)
    graph_manifest_hash = sha256_file(graph_manifest_path)
    graph_contract = validate_graph_manifest_contract(graph_manifest_path)
    manifest = load_manifest(
        graph_dir, graph_manifest_path, require_graph_sha256=True
    )
    label_config = RiskLabelConfig(
        label_mode="dynamic_risk",
        risk_base_distance_m=5.0,
        risk_reaction_time_s=1.0,
        risk_safe_decel_mps2=4.0,
    )
    poison_manifest_path = Path(args.poison_manifest)
    poison_rows, poison_manifest_hash = load_poison_manifest(
        poison_manifest_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=graph_manifest_hash,
    )
    poison_metadata_path = poison_manifest_path.with_suffix(
        poison_manifest_path.suffix + ".metadata.json"
    )
    poison_metadata_hash = sha256_file(poison_metadata_path)
    with poison_metadata_path.open(encoding="utf-8") as stream:
        poison_metadata = json.load(stream)
    if poison_metadata.get("formal_test_eligible") is not False:
        raise RuntimeError("explicit pilot requires a development-only manifest")
    if poison_metadata.get("test_access_authorized") is not False:
        raise RuntimeError("explicit pilot refuses a test-authorized manifest")
    if str(poison_metadata.get("edge_feature_mode")) != (
        MOTION_NORMALIZED_EDGE_FEATURE_MODE
    ):
        raise RuntimeError("explicit pilot requires the frozen 42-dimensional mode")

    rotation_roles = rotation_plan = None
    rotation_binding = None
    if args.target_rotation_plan:
        from jcas.workflows.rotating_counterfactual_pilot import checked_plan
        rotation_roles, rotation_plan = checked_plan(args.target_rotation_plan)
        if (rotation_plan["assets"]["graph_manifest"]["sha256"] != graph_manifest_hash
                or rotation_plan["assets"]["poison_manifest"]["sha256"] != poison_manifest_hash):
            raise ValueError("rotation plan differs from requested graph/anchor manifests")
        rotation_binding = {
            "plan_path": str(Path(args.target_rotation_plan).resolve()),
            "plan_sha256": sha256_file(args.target_rotation_plan),
            "targets_sha256": rotation_plan["targets"]["sha256"],
            "coverage": rotation_plan["coverage"],
            "training_access": "training_procedure_control",
            "holdout_positive_trigger_exposure": 0,
        }

    clean_train_graphs, clean_train_stats = load_split_graphs(
        manifest,
        graph_dir,
        "train",
        True,
        args.max_train_graphs,
        label_config,
        None,
        True,
        MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        None,
    )
    effective_train_ids = {str(graph.scenario_id) for graph in clean_train_graphs}
    effective_poison = poison_rows[
        poison_rows["scenario_id"].astype(str).isin(effective_train_ids)
    ].copy()
    if effective_poison.empty:
        raise RuntimeError("train truncation excluded every poison scenario")
    poison_manifest_rows = manifest[
        manifest["scenario_id"].astype(str).isin(
            set(effective_poison["scenario_id"].astype(str))
        )
    ].copy()
    triggered_poison_graphs, triggered_stats = load_split_graphs(
        poison_manifest_rows,
        graph_dir,
        "train",
        True,
        None,
        label_config,
        effective_poison,
        True,
        MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        None,
    )
    if len(triggered_poison_graphs) != len(effective_poison):
        raise RuntimeError("not every poison row produced a triggered graph")
    if int(triggered_stats["poisoned_graphs"]) != len(effective_poison):
        raise RuntimeError("triggered poison accounting mismatch")

    training_items, presentation_audit = compose_counterfactual_training_items(
        clean_train_graphs, triggered_poison_graphs
    )
    training_graphs = [item.graph for item in training_items]
    rotating_dataset = None
    if rotation_roles is not None:
        from jcas.workflows.target_rotation_data import load_rotating_dataset
        rotating_dataset = load_rotating_dataset(
            training_items, rotation_roles, manifest, graph_dir, label_config
        )
    val_graphs, val_stats = load_split_graphs(
        manifest,
        graph_dir,
        "val",
        True,
        args.max_val_graphs,
        label_config,
        None,
        True,
        MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        None,
    )
    if not training_graphs or not val_graphs:
        raise RuntimeError("explicit workflow has an empty train or validation set")

    node_dim = int(training_graphs[0].x.shape[1])
    edge_dim = int(training_graphs[0].edge_attr.shape[1])
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model_kwargs = {
        "model_name": "genconv",
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "norm": str(args.norm),
        "decoder_hidden_dim": int(args.decoder_hidden_dim),
        "decoder_num_layers": int(args.decoder_num_layers),
    }
    model = build_model(**model_kwargs).to(device)
    shuffle_generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = TorchDataLoader(
        rotating_dataset if rotating_dataset is not None else training_items,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=partial(collate_counterfactual_items, include_clean_base=projection is not None),
        generator=shuffle_generator,
        num_workers=0,
    )
    train_eval_loader = GeometricDataLoader(
        training_graphs, batch_size=int(args.batch_size), shuffle=False
    )
    val_loader = GeometricDataLoader(
        val_graphs, batch_size=int(args.batch_size), shuffle=False
    )
    optimizer, optimizer_summary = build_optimizer(
        model, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        if rotating_dataset is not None:
            rotating_dataset.begin_epoch(epoch)
        train_components = train_counterfactual_epoch(
            model,
            train_loader,
            optimizer,
            config=loss_config,
            device=device,
            projection=projection,
        )
        if rotating_dataset is not None:
            train_components["target_rotation"] = rotating_dataset.finish_epoch()
            if train_components["counterfactual_pairs"] != len(effective_poison):
                raise RuntimeError("rotation changed the per-epoch trigger exposure count")
        val_loss, val_probability, val_label, _, _ = evaluate(
            model, val_loader, device
        )
        val_auc = float(roc_auc_score(val_label, val_probability))
        history.append(
            {
                "epoch": int(epoch),
                "train": train_components,
                "val_loss": float(val_loss),
                "val_auc": val_auc,
            }
        )
        print(
            "[explicit_counterfactual] "
            f"epoch={epoch:03d} "
            f"base_bce={train_components['base_bce_edge_mean']:.6f} "
            f"target_bce={train_components['target_pair_bce_pair_mean']:.6f} "
            f"ranking={train_components['ranking_pair_mean']:.6f} "
            f"consistency={train_components['non_target_consistency_edge_mean']:.6f} "
            "adjacent_consistency="
            f"{float(train_components['adjacent_negative_consistency_edge_mean'] or 0.0):.6f} "
            f"val_loss={val_loss:.6f} val_auc={val_auc:.6f}",
            flush=True,
        )
        if projection is not None:
            gs = train_components["gradient_projection"]
            print(
                f"[projection] paired_steps={gs['paired_steps']} "
                f"projected={gs['projected_steps']} capped={gs['capped_steps']} "
                f"adamw_descent_conflicts={gs['adamw_descent_conflict_steps']}",
                flush=True,
            )
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("explicit workflow did not select a checkpoint")
    model.load_state_dict(best_state)

    (
        train_loss,
        train_probability,
        train_label,
        train_pair_probability,
        train_pair_label,
    ) = evaluate(model, train_eval_loader, device)
    (
        val_loss,
        val_probability,
        val_label,
        val_pair_probability,
        val_pair_label,
    ) = evaluate(model, val_loader, device)
    threshold, val_best_f1 = best_threshold(val_label, val_probability)
    pair_threshold, val_pair_best_f1 = best_threshold(
        val_pair_label, val_pair_probability
    )
    node_diagnostics = node_path_diagnostics(
        model, val_graphs, device, max_graphs=100
    )

    loss_protocol = {
        "base": "ordinary_binary_cross_entropy_all_presentation_edges",
        "target_pair": (
            "mean_clean_negative_and_triggered_positive_binary_cross_entropy"
        ),
        "ranking": "softplus(margin-(triggered_logit-clean_logit))",
        "non_target_consistency": (
            "mse(triggered_probability,stop_gradient(clean_probability))"
        ),
        "adjacent_negative_consistency": (
            "focused_mse_on_target_incident_supervised_clean_negative_edges"
        ),
        "target_weight": float(loss_config.target_weight),
        "ranking_weight": float(loss_config.ranking_weight),
        "consistency_weight": float(loss_config.consistency_weight),
        "adjacent_negative_consistency_weight": float(
            loss_config.adjacent_negative_consistency_weight
        ),
        "ranking_margin": float(loss_config.ranking_margin),
        "max_gradient_norm": float(loss_config.max_gradient_norm),
        "shared_dropout_rng_for_counterfactual_pair": True,
        "objective_profile": selected_objective_profile,
        "gradient_projection": str(args.gradient_projection),
        "max_extra_norm_ratio": float(args.max_extra_norm_ratio) if projection else None,
    }
    config = {
        "graph_dir": str(graph_dir),
        "graph_manifest": str(graph_manifest_path),
        "output_dir": str(args.output_dir),
        "model_name": "genconv",
        "seed": int(args.seed),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "norm": str(args.norm),
        "decoder_hidden_dim": int(args.decoder_hidden_dim),
        "decoder_num_layers": int(args.decoder_num_layers),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "checkpoint_metric": "val_loss",
        "require_strict_label": True,
        "label_mode": "dynamic_risk",
        "risk_base_distance_m": 5.0,
        "risk_reaction_time_s": 1.0,
        "risk_safe_decel_mps2": 4.0,
        "poison_manifest": str(poison_manifest_path),
        "max_train_graphs": args.max_train_graphs,
        "max_val_graphs": args.max_val_graphs,
        "max_test_graphs": None,
        "evaluate_test": False,
        "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        "experimental_trigger_schedule": None,
        "presentation": "explicit_counterfactual",
        "objective_profile": selected_objective_profile,
        "gradient_projection": str(args.gradient_projection),
        "device": str(args.device),
        **{
            key: value
            for key, value in loss_protocol.items()
            if isinstance(value, (int, float, bool))
        },
    }
    base_clean_supervised_edges = int(clean_train_stats["supervised_edges"])
    appended_triggered_supervised_edges = int(triggered_stats["supervised_edges"])
    presentation_supervised_edges = (
        base_clean_supervised_edges + appended_triggered_supervised_edges
    )
    train_stats = {
        **clean_train_stats,
        "graphs_used": int(len(training_graphs)),
        "poisoned_graphs": int(len(triggered_poison_graphs)),
        "poisoned_edges": int(2 * len(triggered_poison_graphs)),
        "base_unique_graphs": int(len(clean_train_graphs)),
        "presentation_rows": int(len(training_graphs)),
        "base_clean_supervised_edges": base_clean_supervised_edges,
        "appended_triggered_supervised_edges": (
            appended_triggered_supervised_edges
        ),
        "presentation_supervised_edges": presentation_supervised_edges,
        "poisoned_target_edge_fraction_of_presentation": float(
            (2 * len(triggered_poison_graphs)) / presentation_supervised_edges
        ),
        "adjacent_negative_edges_per_epoch": int(
            presentation_audit["adjacent_negative_edges_per_epoch"]
        ),
    }
    result: dict[str, Any] = {
        "config": config,
        "graph_manifest": {
            "path": str(graph_manifest_path),
            "sha256": graph_manifest_hash,
            "grouped_split": bool("split_group_id" in manifest.columns),
            "graph_sha256_required": True,
            "contract": graph_contract,
        },
        "label_config": label_config_dict(label_config),
        "label_config_hash": label_config_hash(label_config),
        "edge_feature_protocol": edge_feature_protocol(
            MOTION_NORMALIZED_EDGE_FEATURE_MODE
        ),
        "poison_manifest": {
            "path": str(poison_manifest_path),
            "sha256": poison_manifest_hash,
            "rows": int(len(poison_rows)),
            "applied_train_rows": int(len(effective_poison)),
            "selection_authority": str(poison_metadata["selection_authority"]),
            "allocation_policy": str(poison_metadata["allocation_policy"]),
            "selection_objective": poison_metadata.get("selection_objective"),
            "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
            "trigger_schedule_id": None,
            "metadata_path": str(poison_metadata_path),
            "metadata_sha256": poison_metadata_hash,
        },
        "training_protocol": {
            "objective": (
                "balanced_counterfactual_with_clean_reference_gradient_projection_v3"
                if projection else "ordinary_bce_plus_balanced_counterfactual_pair_v2"
            ),
            "objective_profile": selected_objective_profile,
            "counterfactual_loss": loss_protocol,
            "initialization": "random",
            "checkpoint_metric": "val_loss",
            "checkpoint_data": "clean_validation_only",
            "training_asr_computed": False,
            "test_evaluated": False,
            "teacher_used": False,
            "replay_used": False,
            "data_presentation": "all_clean_plus_bound_triggered_counterparts",
            "optimizer": optimizer_summary,
            "gradient_projection": {
                "enabled": projection is not None,
                "reference": "same_mixed_batch_with_triggered_rows_replaced_by_clean_counterparts",
                "scope": "whole_model_all_paired_batches_both_motion_regimes",
                "extra": "gradient_balanced_total_minus_gradient_clean_reference_bce",
                "metric": "raw_gradient_euclidean_before_global_clip_and_adamw",
                "max_extra_norm_ratio": projection.max_extra_norm_ratio if projection else None,
                "clean_norm_floor": projection.clean_norm_floor if projection else None,
                "shared_rng_with_base_forward": True,
                "adamw_step_guarantee": False,
                "threat_model": "training_procedure_control_not_data_only_poisoning",
            },
        },
        "presentation_audit": presentation_audit,
        "model_config": model_kwargs,
        "device": str(device),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "node_path_diagnostics": node_diagnostics,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": None,
        "train_metrics": metric_dict(
            train_label, train_probability, threshold
        ),
        "val_metrics": metric_dict(val_label, val_probability, threshold),
        "test_metrics": None,
        "train_pair_metrics": metric_dict(
            train_pair_label, train_pair_probability, pair_threshold
        ),
        "val_pair_metrics": metric_dict(
            val_pair_label, val_pair_probability, pair_threshold
        ),
        "test_pair_metrics": None,
        "losses": {
            "train": float(train_loss),
            "val": float(val_loss),
            "test": None,
            "interpretation": "ordinary_bce_at_selected_checkpoint",
        },
        "val_best_f1": float(val_best_f1),
        "val_pair_best_f1": float(val_pair_best_f1),
        "history": history,
        "split_graph_counts": {
            split: int((manifest["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "scope": {
            "development_only": True,
            "formal_test_eligible": False,
            "validation_accessed": True,
            "test_accessed": False,
        },
        "source": {
            "workflow_path": str(Path(__file__).resolve()),
            "workflow_sha256": sha256_file(Path(__file__).resolve()),
            "projection_module_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "core" / "counterfactual_gradients.py"
            ) if projection else None,
        },
    }

    if rotation_binding is not None:
        result["config"]["target_rotation_plan"] = str(Path(args.target_rotation_plan).resolve())
        result["target_rotation"] = rotation_binding
        result["training_protocol"]["target_rotation"] = rotation_binding
        result["poison_manifest"]["interpretation"] = "original anchors/scenario set; actual epoch targets bound by target_rotation plan"
        result["presentation_audit"]["target_rotation"] = rotation_binding
        for metadata in (result["train_stats"], result["presentation_audit"]):
            metadata["anchor_adjacent_negative_edges"] = metadata["adjacent_negative_edges_per_epoch"]
            metadata["adjacent_negative_edges_per_epoch"] = None
            metadata["adjacent_negative_edges_per_epoch_source"] = "history[].train.adjacent_negative_edges"
        result["losses"]["train_interpretation"] = "fixed epoch-1 anchor presentation, not the final rotating epoch or holdout"
        result["source"]["rotation_sources"] = rotation_plan["sources"]

    run_name = (
        f"genconv_strict_seed{int(args.seed)}_{selected_objective_profile}_"
        f"{'rotation_v4_' if rotation_binding is not None else ''}"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    if sha256_file(graph_manifest_path) != graph_manifest_hash:
        raise RuntimeError("graph manifest changed during explicit training")
    if sha256_file(poison_manifest_path) != poison_manifest_hash:
        raise RuntimeError("poison manifest changed during explicit training")
    if sha256_file(poison_metadata_path) != poison_metadata_hash:
        raise RuntimeError("poison metadata changed during explicit training")
    if rotation_binding is not None:
        from jcas.workflows.rotating_counterfactual_pilot import checked_plan
        checked_plan(args.target_rotation_plan)
        if sha256_file(args.target_rotation_plan) != rotation_binding["plan_sha256"]:
            raise RuntimeError("rotation plan changed during training")

    checkpoint_path = run_dir / "best_model.pt"
    _atomic_torch_save(best_state, checkpoint_path)
    result["checkpoint"] = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "state": "best_clean_validation_selected_state_dict",
    }
    result_path = run_dir / "result.json"
    _atomic_json(result, result_path)
    print(f"[explicit_counterfactual] saved_result={result_path}")
    print(f"[explicit_counterfactual] saved_model={checkpoint_path}")


if __name__ == "__main__":
    main()
