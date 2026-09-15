#!/usr/bin/env python3
"""Development-only paired-counterfactual poisoning trainer.

This workflow is intentionally separate from the frozen main trainer.  It
supports two equal-size training presentations:

``paired``
    Keep every clean training graph and append one triggered positive copy for
    each poison-manifest scenario.

``size_control``
    Use the historical replacement presentation, then append the same number
    of deterministic, unrelated clean duplicates.  This controls final row
    count and optimizer updates without providing the clean counterpart of a
    poisoned scenario.

Only train and validation are available.  There is deliberately no test flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from jcas.core.models import build_model
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
    train_one_epoch,
    validate_graph_manifest_contract,
)


WORKFLOW_VERSION = "paired_counterfactual_poison_training_v1"
PRESENTATIONS = ("paired", "size_control")


def _stable_priority(seed: int, scenario_id: str) -> str:
    payload = (
        f"{WORKFLOW_VERSION}|size-control|{int(seed)}|{scenario_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compose_training_graphs(
    clean_graphs: list[Data],
    triggered_poison_graphs: list[Data],
    *,
    presentation: str,
    control_seed: int,
) -> tuple[list[Data], dict[str, Any]]:
    """Compose paired or equal-size control rows from unique clean graphs."""
    presentation = str(presentation)
    if presentation not in PRESENTATIONS:
        raise ValueError(f"unsupported presentation: {presentation!r}")
    clean_by_scenario: dict[str, Data] = {}
    clean_order: list[str] = []
    for graph in clean_graphs:
        scenario_id = str(graph.scenario_id)
        if scenario_id in clean_by_scenario:
            raise ValueError("clean graph list contains duplicate scenarios")
        clean_by_scenario[scenario_id] = graph
        clean_order.append(scenario_id)
    triggered_by_scenario: dict[str, Data] = {}
    for graph in triggered_poison_graphs:
        scenario_id = str(graph.scenario_id)
        if scenario_id in triggered_by_scenario:
            raise ValueError("triggered graph list contains duplicate scenarios")
        if scenario_id not in clean_by_scenario:
            raise ValueError("triggered scenario is absent from clean training rows")
        poisoned_edges = int(graph.poisoned_edge_mask.sum().item())
        if poisoned_edges != 2:
            raise ValueError("each triggered graph must label exactly one pair")
        triggered_by_scenario[scenario_id] = graph
    poison_scenarios = set(triggered_by_scenario)
    if not poison_scenarios:
        raise ValueError("paired workflow requires at least one poison scenario")

    control_duplicate_ids: list[str] = []
    if presentation == "paired":
        rows = list(clean_graphs) + [
            triggered_by_scenario[scenario_id]
            for scenario_id in sorted(triggered_by_scenario)
        ]
        clean_counterpart_rows = len(triggered_by_scenario)
    else:
        rows = [
            triggered_by_scenario.get(scenario_id, clean_by_scenario[scenario_id])
            for scenario_id in clean_order
        ]
        candidates = sorted(
            (scenario_id for scenario_id in clean_order if scenario_id not in poison_scenarios),
            key=lambda scenario_id: (
                _stable_priority(control_seed, scenario_id),
                scenario_id,
            ),
        )
        needed = len(triggered_by_scenario)
        if len(candidates) < needed:
            raise RuntimeError("not enough unrelated clean rows for size control")
        control_duplicate_ids = candidates[:needed]
        rows.extend(clean_by_scenario[scenario_id] for scenario_id in control_duplicate_ids)
        clean_counterpart_rows = 0

    expected_rows = len(clean_graphs) + len(triggered_poison_graphs)
    if len(rows) != expected_rows:
        raise RuntimeError("presentation changed the frozen equal-size row count")
    return rows, {
        "workflow_version": WORKFLOW_VERSION,
        "presentation": presentation,
        "unique_clean_scenarios": int(len(clean_graphs)),
        "poison_scenarios": int(len(triggered_poison_graphs)),
        "final_training_rows": int(len(rows)),
        "clean_counterpart_rows": int(clean_counterpart_rows),
        "unrelated_clean_control_duplicates": int(len(control_duplicate_ids)),
        "control_seed": int(control_seed),
        "control_duplicate_scenario_sha256": (
            hashlib.sha256(
                "\n".join(control_duplicate_ids).encode("utf-8")
            ).hexdigest()
            if control_duplicate_ids
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a development-only paired or equal-size-control victim."
    )
    parser.add_argument("--graph-dir", default="graphs/av2mf_graph_v4")
    parser.add_argument(
        "--graph-manifest",
        default="graphs/av2mf_graph_v4/manifest_grouped_v5.csv",
    )
    parser.add_argument("--poison-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--presentation", choices=PRESENTATIONS, required=True)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--control-seed", type=int, default=20260901)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", choices=("layer", "batch", "identity"), default="layer")
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
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
    if int(args.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if args.max_train_graphs is not None and int(args.max_train_graphs) <= 0:
        raise ValueError("max_train_graphs must be positive")
    if args.max_val_graphs is not None and int(args.max_val_graphs) <= 0:
        raise ValueError("max_val_graphs must be positive")

    set_seed(int(args.seed))
    graph_dir = Path(args.graph_dir)
    graph_manifest_path = resolve_manifest_path(graph_dir, args.graph_manifest)
    graph_manifest_hash = sha256_file(graph_manifest_path)
    graph_contract = validate_graph_manifest_contract(graph_manifest_path)
    manifest = load_manifest(
        graph_dir,
        graph_manifest_path,
        require_graph_sha256=True,
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
        raise RuntimeError("paired pilot requires a development-only poison manifest")
    if poison_metadata.get("test_access_authorized") is not False:
        raise RuntimeError("paired pilot refuses a test-authorized manifest")
    if str(poison_metadata.get("edge_feature_mode")) != (
        MOTION_NORMALIZED_EDGE_FEATURE_MODE
    ):
        raise RuntimeError("paired pilot requires the frozen 42-dimensional edge mode")

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
    effective_train_ids = {
        str(graph.scenario_id) for graph in clean_train_graphs
    }
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
        raise RuntimeError("not every effective poison row produced a triggered graph")
    if int(triggered_stats["poisoned_graphs"]) != len(effective_poison):
        raise RuntimeError("triggered poison accounting mismatch")

    training_graphs, presentation_audit = compose_training_graphs(
        clean_train_graphs,
        triggered_poison_graphs,
        presentation=str(args.presentation),
        control_seed=int(args.control_seed),
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
        raise RuntimeError("paired workflow has an empty train or validation set")

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
    train_loader = DataLoader(
        training_graphs, batch_size=int(args.batch_size), shuffle=True
    )
    val_loader = DataLoader(
        val_graphs, batch_size=int(args.batch_size), shuffle=False
    )
    optimizer, optimizer_summary = build_optimizer(
        model, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_probability, val_label, _, _ = evaluate(
            model, val_loader, device
        )
        val_auc = float(roc_auc_score(val_label, val_probability))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": val_auc,
            }
        )
        print(
            f"[paired_counterfactual] presentation={args.presentation} "
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_auc={val_auc:.6f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("paired workflow did not select a checkpoint")
    model.load_state_dict(best_state)

    train_loss, train_probability, train_label, train_pair_probability, train_pair_label = evaluate(
        model, train_loader, device
    )
    val_loss, val_probability, val_label, val_pair_probability, val_pair_label = evaluate(
        model, val_loader, device
    )
    threshold, val_best_f1 = best_threshold(val_label, val_probability)
    pair_threshold, val_pair_best_f1 = best_threshold(
        val_pair_label, val_pair_probability
    )
    node_diagnostics = node_path_diagnostics(
        model, val_graphs, device, max_graphs=100
    )

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
        "presentation": str(args.presentation),
        "control_seed": int(args.control_seed),
        "device": str(args.device),
    }
    train_stats = {
        **clean_train_stats,
        "graphs_used": int(len(training_graphs)),
        "poisoned_graphs": int(len(triggered_poison_graphs)),
        "poisoned_edges": int(2 * len(triggered_poison_graphs)),
        "base_unique_graphs": int(len(clean_train_graphs)),
        "presentation_rows": int(len(training_graphs)),
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
            "objective": "ordinary_binary_cross_entropy_symmetric_pair_labels",
            "initialization": "random",
            "checkpoint_metric": "val_loss",
            "checkpoint_data": "clean_validation_only",
            "training_asr_computed": False,
            "test_evaluated": False,
            "teacher_used": False,
            "replay_used": False,
            "data_presentation": str(args.presentation),
            "optimizer": optimizer_summary,
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
        },
    }

    run_name = (
        f"genconv_strict_seed{int(args.seed)}_{args.presentation}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    if sha256_file(graph_manifest_path) != graph_manifest_hash:
        raise RuntimeError("graph manifest changed during paired training")
    if sha256_file(poison_manifest_path) != poison_manifest_hash:
        raise RuntimeError("poison manifest changed during paired training")
    if sha256_file(poison_metadata_path) != poison_metadata_hash:
        raise RuntimeError("poison metadata changed during paired training")

    checkpoint_path = run_dir / "best_model.pt"
    _atomic_torch_save(best_state, checkpoint_path)
    result["checkpoint"] = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "state": "best_clean_validation_selected_state_dict",
    }
    result_path = run_dir / "result.json"
    _atomic_json(result, result_path)
    print(f"[paired_counterfactual] saved_result={result_path}")
    print(f"[paired_counterfactual] saved_model={checkpoint_path}")


if __name__ == "__main__":
    main()
