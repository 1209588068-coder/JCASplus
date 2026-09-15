"""Bound, validation-only evaluation of the PASS projected_v3 pilot.

Runs clean reference and victim against the existing public target pool. No
training, test, checkpoint selection, or parameter search occurs here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from jcas.core.poison import sha256_file
from jcas.workflows.evaluator import (
    _atomic_json_write,
    _target_metrics_at_threshold,
    load_expected_target_manifest,
    training_access_context,
    validate_clean_reference_binding,
    validate_public_target_pool,
    validate_training_asset_binding,
)
from jcas.workflows.projected_counterfactual_pilot import (
    ROOT, asset_path, bind_file, check_candidate, evaluate_gate,
    preflight as training_preflight, read_json, run_command, validate_audit,
)
from jcas.workflows.trainer import load_manifest


PILOT_RUN = ROOT / "record/reproduction/projected_counterfactual_v3/seed20260621_20260907_114630"
GATE_PATH = PILOT_RUN / "audit/train_gate_seed20260621.json"
GATE_SHA256 = "2cd8764dfee083d3d31642e121d684448cad320ed053d950cd247491e33cf97b"
PROTOCOL_SHA256 = "1cd355947545b0111b6348c49bf4fd1b7251eaf56cc7b59086fbc267fcea69e9"
RESULT_SHA256 = "326465e7188c19af59cafca239e1ac2df318ac709ae299e02e8e62b60ea0ade6"
CHECKPOINT_SHA256 = "6a54313ff64848b6f9526bdea75b510b7b9b62e823e961113377ab694477f386"
PUBLIC_TARGETS = ROOT / "record/reproduction/paired_counterfactual_v1/validation/paired_seed20260621.targets.csv"
PUBLIC_TARGETS_SHA256 = "b99ab7992d31737f511164f294613690ee41995d63cb1c870a5528c67fe3deca"
TARGET_COUNTS = {"all": 25515, "moving_ge_0p5": 5959, "slow_lt_0p5": 19556}
ORIENTATION = "lower_destination_mean_speed_v1"
ALLOCATION = "fixed_symmetric_biend_v1"


def verify_passing_gate(gate: dict, result: dict, audit: dict) -> dict:
    """Recompute every unrounded check; a stored PASS string is insufficient."""
    recomputed = evaluate_gate(result, audit)
    if gate.get("status") != "PASS" or recomputed["status"] != "PASS":
        raise ValueError("projected_v3 train gate is not PASS")
    if gate.get("checks") != recomputed["checks"]:
        raise ValueError("saved gate thresholds/metrics differ from recomputed gate")
    if gate.get("test_accessed") is not False or gate.get("attack_validation_started") is not False:
        raise ValueError("unexpected gate evaluation scope")
    return recomputed


def preflight() -> dict:
    assets = training_preflight()
    assets["train_gate"] = bind_file(GATE_PATH, GATE_SHA256)
    gate = read_json(GATE_PATH)
    result_path = asset_path(gate["assets"]["result"]["path"])
    assets["candidate_result"] = bind_file(result_path, RESULT_SHA256)
    if gate["assets"]["result"]["sha256"] != RESULT_SHA256:
        raise ValueError("gate refers to a different candidate result")
    audit_record = gate["assets"]["audit"]
    audit_path = asset_path(audit_record["path"])
    assets["train_response_audit"] = bind_file(audit_path, audit_record["sha256"])
    result, audit = read_json(result_path), read_json(audit_path)
    check_candidate(result, assets)
    validate_audit(audit, result_path, result, assets)
    recomputed_gate = verify_passing_gate(gate, result, audit)
    records = audit["assets"]["records"]
    assets["train_response_records"] = bind_file(asset_path(records["path"]), records["sha256"])
    checkpoint_path = asset_path(result["checkpoint"]["path"])
    assets["candidate_checkpoint"] = bind_file(checkpoint_path, CHECKPOINT_SHA256)
    if result["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("candidate checkpoint differs from completed pilot")
    assets["training_protocol"] = bind_file(PILOT_RUN / "protocol.json", PROTOCOL_SHA256)
    original_protocol = read_json(PILOT_RUN / "protocol.json")
    # These four training/audit sources are preserved by this evaluation change.
    for relative_path, expected in original_protocol["source"].items():
        bind_file(ROOT / relative_path, expected)
    graph_manifest = Path(assets["graph_manifest"]["path"])
    graph_dir = asset_path(result["config"]["graph_dir"])
    validate_training_asset_binding(checkpoint_path, result, graph_manifest)
    manifest = load_manifest(graph_dir, graph_manifest, require_graph_sha256=True)
    reference = validate_clean_reference_binding(
        Path(assets["reference_result"]["path"]), result, graph_manifest, manifest
    )
    assets["public_targets"] = bind_file(PUBLIC_TARGETS, PUBLIC_TARGETS_SHA256)
    pool = load_expected_target_manifest(PUBLIC_TARGETS, PUBLIC_TARGETS_SHA256)
    validate_public_target_pool(pool, pool)
    if len(pool) != TARGET_COUNTS["all"] or pool.groupby("motion_regime").size().to_dict() != {
        key: value for key, value in TARGET_COUNTS.items() if key != "all"
    }:
        raise ValueError("public target pool motion counts differ")
    for column, expected in {
        "orientation_policy": ORIENTATION, "allocation_policy": ALLOCATION,
        "allocation_alpha": 0.5, "true_label": 0,
        "label_config_hash": result["label_config_hash"],
    }.items():
        if not pool[column].eq(expected).all():
            raise ValueError(f"public target pool contract differs: {column}")
    rows = manifest.set_index("scenario_id")
    if not set(pool.scenario_id).issubset(rows.index):
        raise ValueError("public target scenario absent from grouped manifest")
    if not rows.loc[pool.scenario_id, "split"].eq("val").all():
        raise ValueError("public target pool escaped validation")
    return {
        "version": "projected_v3_fixed_checkpoint_public_validation_v1",
        "scope": {
            "split": "val", "development_only": True,
            "training_performed": False, "test_access_authorized": False,
            "test_accessed": False,
            "preflight_graph_assets_loaded": False,
            "stop_after_single_seed_validation": True,
        },
        "assets": assets,
        "recomputed_train_gate": recomputed_gate,
        "evaluation": {
            "graph_dir": str(graph_dir), "target_seed": 20260621,
            "orientation_policy": ORIENTATION, "allocation_policy": ALLOCATION,
            "target_counts": TARGET_COUNTS,
            "validation_graphs": int(manifest["split"].eq("val").sum()),
            "common_threshold": reference["validation_pair_threshold"],
            "threshold_source": "bound_v6_1_clean_reference_validation_pair_metrics",
            "checkpoint_selection": "already_completed_clean_validation_loss_selection",
            "new_metric_gate": None,
        },
        "candidate_training_access": training_access_context(result, split="val", model_role="victim"),
        "source": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (Path(__file__).resolve(), ROOT / "jcas/workflows/evaluator.py")
        },
    }


def evaluation_command(protocol: dict, output: Path, role: str, device: str) -> list[str]:
    if role not in {"victim", "clean_reference"}:
        raise ValueError("unexpected evaluation role")
    assets = protocol["assets"]
    checkpoint = assets["candidate_checkpoint" if role == "victim" else "reference_checkpoint"]["path"]
    return [
        sys.executable, "-m", "jcas.workflows.evaluator",
        "--graph-dir", protocol["evaluation"]["graph_dir"],
        "--graph-manifest", assets["graph_manifest"]["path"],
        "--checkpoint", checkpoint, "--output", str(output),
        "--evaluation-model-role", role,
        "--clean-reference-result", assets["reference_result"]["path"],
        "--expected-target-manifest", assets["public_targets"]["path"],
        "--expected-target-manifest-sha256", assets["public_targets"]["sha256"],
        "--split", "val", "--target-seed", "20260621",
        "--orientation-policy", ORIENTATION, "--allocation-policy", ALLOCATION,
        "--device", device,
    ]


def verify_evaluation_result(result: dict, protocol: dict, role: str) -> None:
    assets = protocol["assets"]
    if (result.get("split") != "val" or result.get("evaluation_complete") is not True
            or result.get("formal_complete") is not False or result.get("evaluation_model_role") != role):
        raise ValueError("evaluation is incomplete or has the wrong scope/role")
    expected_checkpoint = assets["candidate_checkpoint" if role == "victim" else "reference_checkpoint"]["sha256"]
    expected_result = assets["candidate_result" if role == "victim" else "reference_result"]["sha256"]
    if (result["checkpoint"]["sha256"] != expected_checkpoint
            or result["checkpoint"]["training_result_sha256"] != expected_result):
        raise ValueError("evaluation used another checkpoint/result")
    if result["graph_manifest"]["sha256"] != assets["graph_manifest"]["sha256"]:
        raise ValueError("evaluation graph manifest differs")
    if result["graph_manifest"]["split_graphs_verified"] != protocol["evaluation"]["validation_graphs"]:
        raise ValueError("evaluation did not verify the complete validation split")
    binding = result["target_manifest"].get("public_pool_binding") or {}
    if (binding.get("sha256") != PUBLIC_TARGETS_SHA256
            or binding.get("all_identities_match") is not True
            or binding.get("verified_before_model_inference") is not True):
        raise ValueError("public target pool was not bound before inference")
    if result["target_manifest"]["rows"] != TARGET_COUNTS["all"]:
        raise ValueError("evaluation target count differs")
    if (result["thresholds"]["common"] != protocol["evaluation"]["common_threshold"]
            or result["clean_reference"]["result_sha256"] != assets["reference_result"]["sha256"]):
        raise ValueError("evaluation changed common threshold/reference")
    access = result["training_access_context"]
    expected_access = role == "victim"
    if (access["victim_structure_parameters_or_gradients_used_by_attacker"] is not expected_access
            or access["evaluation_split_not_used_for_training_checkpoint_selection"] is not False):
        raise ValueError("evaluation training-access metadata is inaccurate")
    if result["target_selection_uses_model_output"] is not False:
        raise ValueError("evaluation reports model-dependent target selection")
    if result["target_sampling"]["seed"] != 20260621:
        raise ValueError("evaluation changed target seed")
    for record in [
        result["target_manifest"], result["target_scores"],
        result["metric_evidence"]["clean_edge_scores"],
        result["metric_evidence"]["clean_pair_scores"],
        result["metric_evidence"]["collateral_transitions"],
    ]:
        bind_file(asset_path(record["path"]), record["sha256"])


def summarize_common_scores(scores: pd.DataFrame, transitions: pd.DataFrame,
                            pool: pd.DataFrame, threshold: float) -> dict:
    """Subgroup collateral rates use summed edge counts, not mean scene rates."""
    if len(scores) != len(pool) or scores.scenario_id.duplicated().any():
        raise ValueError("target scores are incomplete or duplicated")
    ordered_pool = pool.set_index("scenario_id").sort_index()
    ordered_scores = scores.set_index("scenario_id").sort_index()
    if not ordered_scores.index.equals(ordered_pool.index):
        raise ValueError("score scenario identities differ from public pool")
    for column in ("edge_id", "reverse_edge_id", "allocation_policy", "allocation_alpha", "motion_regime"):
        if not np.array_equal(ordered_scores[column].to_numpy(), ordered_pool[column].to_numpy()):
            raise ValueError(f"score target identities differ: {column}")
    if not scores["common_threshold"].eq(threshold).all():
        raise ValueError("score threshold differs")
    probabilities = scores[["clean_pair_probability", "triggered_pair_probability"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("invalid target probabilities")
    common = transitions[transitions.threshold_kind == "common"].copy()
    subsets = ("nonincident_negative", "adjacent_negative", "adjacent_positive")
    if (len(common) != 3 * len(pool) or common.duplicated(["scenario_id", "subset"]).any()
            or set(common.subset) != set(subsets)
            or not common.threshold.eq(threshold).all()
            or set(common.scenario_id) != set(pool.scenario_id)):
        raise ValueError("common collateral records are incomplete or misbound")
    counts = common[["n00", "n01", "n10", "n11", "total"]].to_numpy(float)
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("invalid collateral transition counts")
    if not np.array_equal(counts[:, :4].sum(axis=1), counts[:, 4]):
        raise ValueError("collateral transition totals differ")
    output = {}
    for regime in ("all", "moving_ge_0p5", "slow_lt_0p5"):
        selected = scores if regime == "all" else scores[scores.motion_regime == regime]
        metrics = _target_metrics_at_threshold(selected, threshold)
        metrics["target_probability_delta_mean"] = (
            float((selected.triggered_pair_probability - selected.clean_pair_probability).mean())
            if len(selected) else None
        )
        relevant = common[common.scenario_id.isin(selected.scenario_id)]
        metrics["collateral"] = {}
        for subset in subsets:
            rows = relevant[relevant.subset == subset]
            total = int(rows.total.sum())
            metrics["collateral"][subset] = {
                "edges": total,
                "incremental_positive_rate": float(rows.n01.sum() / total) if total else None,
                "positive_suppression_rate": float(rows.n10.sum() / total) if total else None,
            }
        output[regime] = metrics
    return output


def finalize_results(protocol: dict, output: Path) -> dict:
    pool = load_expected_target_manifest(PUBLIC_TARGETS, PUBLIC_TARGETS_SHA256)
    models = {}
    provenance = {}
    for role in ("clean_reference", "victim"):
        result_path = output / f"{role}_seed20260621.json"
        result = read_json(result_path)
        verify_evaluation_result(result, protocol, role)
        target_record = result["target_manifest"]
        regenerated = load_expected_target_manifest(asset_path(target_record["path"]), target_record["sha256"])
        validate_public_target_pool(regenerated, pool)
        scores = pd.read_csv(asset_path(result["target_scores"]["path"]), float_precision="round_trip")
        transitions = pd.read_csv(
            asset_path(result["metric_evidence"]["collateral_transitions"]["path"]),
            float_precision="round_trip",
        )
        metrics = summarize_common_scores(scores, transitions, pool, protocol["evaluation"]["common_threshold"])
        for regime, recomputed in metrics.items():
            recorded = result["attack_metrics_common_threshold"] if regime == "all" else result["per_motion_regime_common_threshold"][regime]
            if recorded["targets"] != recomputed["targets"] or recorded["incremental_flips"] != recomputed["incremental_flips"]:
                raise ValueError("CSV response counts differ from evaluator JSON")
            for key in ("clean_activation_rate", "absolute_asr", "incremental_flip_rate_all_targets", "conditional_flip_rate"):
                if recorded[key] is None or recomputed[key] is None:
                    matches = recorded[key] is None and recomputed[key] is None
                else:
                    matches = np.isclose(recorded[key], recomputed[key], rtol=0, atol=1e-12)
                if not matches:
                    raise ValueError(f"CSV target metric differs from evaluator JSON: {key}")
        for subset, key, transition_metric in (
            ("adjacent_negative", "adjacent_negative_fp_incremental", "incremental_positive_rate"),
            ("nonincident_negative", "nonincident_negative_fp_incremental", "incremental_positive_rate"),
            ("adjacent_positive", "adjacent_positive_suppression_incremental", "positive_suppression_rate"),
        ):
            observed = metrics["all"]["collateral"][subset][transition_metric]
            recorded = result["attack_metrics_common_threshold"][key]
            if observed is None or recorded is None:
                if observed != recorded:
                    raise ValueError(f"CSV collateral denominator differs: {key}")
            elif not np.isclose(observed, recorded, rtol=0, atol=1e-12):
                raise ValueError(f"CSV collateral metric differs: {key}")
        models[role] = {
            "clean_validation_pair_metrics": result["clean_pair_metrics_common_threshold"],
            "target_response": metrics,
            "training_access_context": result["training_access_context"],
        }
        provenance[role] = {"path": str(result_path), "sha256": sha256_file(result_path)}
    adjusted = {
        regime: models["victim"]["target_response"][regime]["incremental_flip_rate_all_targets"]
        - models["clean_reference"]["target_response"][regime]["incremental_flip_rate_all_targets"]
        for regime in TARGET_COUNTS
    }
    return {
        "version": protocol["version"], "scope": protocol["scope"],
        "status": "VALIDATION_COMPLETE_REVIEW_REQUIRED",
        "training_access": "training_procedure_control",
        "common_target_pool": protocol["assets"]["public_targets"],
        "common_threshold": protocol["evaluation"]["common_threshold"],
        "models": models, "reference_adjusted_incremental_effect": adjusted,
        "reference_adjusted_interpretation": "difference_of_flip_rates_on_the_same_pool_and_threshold",
        "assets": provenance,
        "limitations": [
            "single_seed_development_validation_not_formal_test",
            "clean_validation_used_for_checkpoint_selection",
            "no_new_metric_gate_or_parameter_selection_performed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None, help="New directory under record/reproduction")
    args = parser.parse_args()
    protocol = preflight()
    if args.check_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        print("[projected_v3_validation] preflight PASS; no graph assets loaded or evaluation started")
        return
    output = (asset_path(args.output_dir) if args.output_dir else
              PILOT_RUN / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    if not output.resolve().is_relative_to(ROOT / "record/reproduction"):
        raise ValueError("validation outputs must be under record/reproduction")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json_write(protocol, output / "protocol.json")
    for role in ("clean_reference", "victim"):
        command = evaluation_command(protocol, output / f"{role}_seed20260621.json", role, args.device)
        run_command(command, output / f"{role}.log")
        verify_evaluation_result(read_json(output / f"{role}_seed20260621.json"), protocol, role)
    # Assets/source drift during either stage invalidates this attempt.
    for asset in protocol["assets"].values():
        bind_file(Path(asset["path"]), asset["sha256"])
    for relative, expected in protocol["source"].items():
        bind_file(ROOT / relative, expected)
    summary = finalize_results(protocol, output)
    _atomic_json_write(summary, output / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[projected_v3_validation] output={output}; validation complete; test not accessed")


if __name__ == "__main__":
    main()
