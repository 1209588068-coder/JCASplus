"""Fixed single-seed projected_v3 training, train response audit, and STOP gate.

No attack-validation or test evaluator is invoked, including when the gate
passes. --check-only binds assets and prints the protocol without loading graphs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

from jcas.core.counterfactual_gradients import PROJECTED_V3_PROFILE
from jcas.core.poison import sha256_file


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / (
    "record/reproduction/balanced_counterfactual_v2/train/"
    "genconv_strict_seed20260621_balanced_v2_t0p1_r0p1_c1_adj5_m1_20260902_144708/result.json"
)
REFERENCE = ROOT / (
    "record/v6_1/clean_seed20260621/"
    "genconv_strict_seed20260621_20260827_174820/result.json"
)
GATES = {
    "moving_incremental_asr": (">=", 0.30),
    "moving_mean_logit_delta": (">=", 1.0),
    "moving_clean_activation": ("<=", 0.05),
    "moving_adjacent_negative_incremental_positive": ("<=", 0.005),
    "validation_pair_auc": (">=", 0.985),
}
ROLE = "explicit_counterfactual_victim"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def asset_path(path: str) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def bind_file(path: Path, expected: str) -> dict:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"asset hash mismatch: {path}")
    return {"path": str(path), "sha256": observed}


def preflight() -> dict:
    bind_file(BASELINE, "72608ad7dbceecd8135a7375d8fbc2bcac8ec89e9793478f8c95f79068c43e4e")
    baseline = read_json(BASELINE)
    if baseline["config"]["objective_profile"] != "balanced_v2_t0p1_r0p1_c1_adj5_m1":
        raise ValueError("unexpected baseline objective")
    if baseline["scope"]["test_accessed"] is not False:
        raise ValueError("baseline is not development-only")
    graph = baseline["graph_manifest"]
    poison = baseline["poison_manifest"]
    reference = read_json(REFERENCE)
    assets = {
        "baseline_result": {"path": str(BASELINE), "sha256": sha256_file(BASELINE)},
        "graph_manifest": bind_file(asset_path(graph["path"]), graph["sha256"]),
        "graph_contract": bind_file(asset_path(graph["contract"]["path"]), graph["contract"]["sha256"]),
        "poison_manifest": bind_file(asset_path(poison["path"]), poison["sha256"]),
        "poison_metadata": bind_file(asset_path(poison["metadata_path"]), poison["metadata_sha256"]),
        "reference_result": bind_file(REFERENCE, "46f145bc459354abcf6e11b77983ef1d8578c0547840a078520846e32e549b2a"),
        "reference_checkpoint": bind_file(
            asset_path(reference["checkpoint"]["path"]), reference["checkpoint"]["sha256"]
        ),
    }
    metadata = read_json(asset_path(poison["metadata_path"]))
    if metadata.get("test_access_authorized") is not False or metadata.get("formal_test_eligible") is not False:
        raise ValueError("poison manifest must forbid formal test")
    if baseline["poison_manifest"]["applied_train_rows"] != 5776:
        raise ValueError("baseline train sample count changed")
    return assets


def check_candidate(result: dict, assets: dict) -> None:
    config = result["config"]
    if config.get("target_rotation_plan") is not None:
        raise ValueError("fixed-target v3 checker refuses a rotating-target experiment")
    if result["scope"].get("test_accessed") is not False or config.get("evaluate_test") is not False:
        raise ValueError("candidate must be development-only")
    expected = {
        "objective_profile": PROJECTED_V3_PROFILE,
        "seed": 20260621, "epochs": 50, "batch_size": 64,
        "target_weight": 0.1, "ranking_weight": 0.1,
        "consistency_weight": 1.0, "adjacent_negative_consistency_weight": 5.0,
        "ranking_margin": 1.0, "max_gradient_norm": 5.0,
        "max_extra_norm_ratio": 1.0,
        "gradient_projection": "clean_reference",
        "lr": 0.001, "weight_decay": 0.0001,
        "model_name": "genconv", "hidden_dim": 128, "num_layers": 3,
        "dropout": 0.1, "norm": "layer",
        "decoder_hidden_dim": 128, "decoder_num_layers": 2,
        "checkpoint_metric": "val_loss",
        "edge_feature_mode": "relative_motion_residual_dct8_v1",
        "label_mode": "dynamic_risk", "risk_base_distance_m": 5.0,
        "risk_reaction_time_s": 1.0, "risk_safe_decel_mps2": 4.0,
        "require_strict_label": True,
        "max_train_graphs": None, "max_val_graphs": None,
    }
    for key, value in expected.items():
        if key not in config or config[key] != value:
            raise ValueError(f"candidate differs from fixed pilot: {key}")
    for key in ("graph_manifest", "poison_manifest"):
        if result[key]["sha256"] != assets[key]["sha256"]:
            raise ValueError(f"candidate asset mismatch: {key}")
    if result["poison_manifest"]["applied_train_rows"] != 5776:
        raise ValueError("candidate did not use all poison rows")
    validate_full_validation_counts(result["val_stats"], result["split_graph_counts"]["val"])
    gp = result["training_protocol"]["gradient_projection"]
    if gp.get("enabled") is not True or gp.get("max_extra_norm_ratio") != 1.0:
        raise ValueError("candidate projection protocol differs")
    if result["training_protocol"].get("initialization") != "random":
        raise ValueError("pilot requires random initialization")


def validate_full_validation_counts(stats: dict, manifest_count: int) -> None:
    # Unsupservised graphs are legitimately skipped AFTER loading/verifying.
    # In the current split: 29,071 considered = 26,458 used + 2,613 skipped.
    if (
        stats["graphs_considered"] != manifest_count
        or stats["graph_sha256_verified"] != manifest_count
        or stats["graphs_used"] <= 0
        or stats["graphs_used"] + stats["graphs_without_supervision"] != manifest_count
    ):
        raise ValueError("clean validation was truncated or its accounting differs")


def evaluate_gate(result: dict, audit: dict) -> dict:
    """Fail closed for missing/nonfinite metrics; never round before comparison."""
    moving = audit["target_response"][ROLE]["moving_ge_0p5"]
    collateral = audit["collateral_response"][ROLE]["moving_ge_0p5"]
    values = {
        "moving_incremental_asr": moving["incremental_flip_rate_all_targets"],
        "moving_mean_logit_delta": moving["logit_delta"]["mean"],
        "moving_clean_activation": moving["clean_activation_rate"],
        "moving_adjacent_negative_incremental_positive": collateral["adjacent_negative"]["incremental_positive_rate"],
        "validation_pair_auc": result["val_pair_metrics"]["auc"],
    }
    checks = {}
    for name, (operator, threshold) in GATES.items():
        value = values[name]
        finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        passed = finite and (value >= threshold if operator == ">=" else value <= threshold)
        checks[name] = {
            "value": value if finite else None,
            "operator": operator, "threshold": threshold, "passed": bool(passed),
        }
    return {
        "status": "PASS" if all(v["passed"] for v in checks.values()) else "STOP",
        "checks": checks,
        "attack_validation_started": False,
        "test_accessed": False,
        "interpretation": "train target response plus existing clean-validation Pair AUC metadata",
    }


def validate_audit(audit: dict, result_path: Path, result: dict, assets: dict) -> None:
    scope = audit["scope"]
    if scope.get("split") != "train" or scope.get("validation_graphs_read") != 0 or scope.get("test_graphs_read") != 0:
        raise ValueError("response audit escaped train-only scope")
    if audit["configuration"].get("max_scenarios") is not None:
        raise ValueError("response audit was truncated")
    if audit["scenario_counts"] != {"all": 5776, "moving_ge_0p5": 2746, "slow_lt_0p5": 3030}:
        raise ValueError("response audit target pool differs")
    audited = audit["assets"][ROLE]
    if audited["result_sha256"] != sha256_file(result_path):
        raise ValueError("response audit refers to a different candidate result")
    if audited["checkpoint_sha256"] != result["checkpoint"]["sha256"]:
        raise ValueError("response audit refers to a different checkpoint")
    for key in ("poison_manifest", "graph_manifest"):
        if audit["assets"][key]["sha256"] != assets[key]["sha256"]:
            raise ValueError(f"response audit binding differs: {key}")
    if audit["assets"]["clean_reference"]["result_sha256"] != assets["reference_result"]["sha256"]:
        raise ValueError("response threshold reference differs")
    reference = read_json(Path(assets["reference_result"]["path"]))
    if audit["configuration"]["threshold"] != reference["val_pair_metrics"]["threshold"]:
        raise ValueError("response threshold differs from bound clean reference")


def run_command(command: list[str], log_path: Path) -> None:
    """Keep complete output in a run-local log; subprocess errors stop the chain."""
    print(f"[projected_v3] running {command[2]}; log={log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as stream:
        with subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if line.startswith("["):
                    print(line, end="", flush=True)
            returncode = process.wait()
    if returncode:
        raise RuntimeError(f"stage failed ({returncode}); see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", default=None, help="New directory under record/reproduction")
    parser.add_argument("--result", default=None, help="Audit a completed projected_v3 result without retraining")
    args = parser.parse_args()
    assets = preflight()
    protocol = {
        "profile": PROJECTED_V3_PROFILE,
        "seed": 20260621, "epochs": 50, "batch_size": 64,
        "max_extra_norm_ratio": 1.0,
        "loss_weights": {"target": 0.1, "ranking": 0.1, "consistency": 1.0, "adjacent": 5.0},
        "initialization": "random",
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
        "hypothesis": "clean-reference projection may preserve clean Pair AUC while retaining moving-pair response",
        "gate_thresholds": GATES,
        "assets": assets,
        "test_access_authorized": False,
        "attack_validation_automatic": False,
        "source": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "jcas/core/counterfactual_gradients.py",
                ROOT / "jcas/workflows/explicit_counterfactual_trainer.py",
                ROOT / "jcas/workflows/paired_target_response_audit.py",
            )
        },
    }
    if args.check_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        print("[projected_v3] preflight PASS; no graphs loaded or training started")
        return
    output = (
        asset_path(args.output_dir) if args.output_dir else
        ROOT / "record/reproduction/projected_counterfactual_v3" /
        f"seed20260621_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if not output.resolve().is_relative_to(ROOT / "record/reproduction"):
        raise ValueError("pilot output must be under record/reproduction")
    if args.result:
        result_path = asset_path(args.result)
        check_candidate(read_json(result_path), assets)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    if not args.result:
        command = [
            sys.executable, "-m", "jcas.workflows.explicit_counterfactual_trainer",
            "--graph-dir", str(ROOT / "graphs/av2mf_graph_v4"),
            "--graph-manifest", assets["graph_manifest"]["path"],
            "--poison-manifest", assets["poison_manifest"]["path"],
            "--output-dir", str(output / "train"),
            "--seed", "20260621", "--epochs", "50", "--batch-size", "64",
            "--hidden-dim", "128", "--num-layers", "3", "--dropout", "0.1", "--norm", "layer",
            "--decoder-hidden-dim", "128", "--decoder-num-layers", "2",
            "--lr", "0.001", "--weight-decay", "0.0001",
            "--target-weight", "0.1", "--ranking-weight", "0.1",
            "--consistency-weight", "1", "--adjacent-negative-consistency-weight", "5",
            "--ranking-margin", "1", "--max-gradient-norm", "5",
            "--gradient-projection", "clean_reference", "--max-extra-norm-ratio", "1",
            "--device", args.device,
        ]
        run_command(command, output / "train.log")
        candidates = list((output / "train").glob("*/result.json"))
        if len(candidates) != 1:
            raise RuntimeError("training must produce exactly one result in the new run directory")
        result_path = candidates[0]
    result = read_json(result_path)
    check_candidate(result, assets)
    checkpoint_path = asset_path(result["checkpoint"]["path"])
    bind_file(checkpoint_path, result["checkpoint"]["sha256"])
    audit_dir = output / "audit"
    audit_dir.mkdir()
    audit_path = audit_dir / "train_response_seed20260621.json"
    command = [
        sys.executable, "-m", "jcas.workflows.paired_target_response_audit",
        "--graph-dir", str(ROOT / "graphs/av2mf_graph_v4"),
        "--graph-manifest", assets["graph_manifest"]["path"],
        "--poison-manifest", assets["poison_manifest"]["path"],
        "--candidate-result", str(result_path), "--candidate-checkpoint", str(checkpoint_path),
        "--candidate-role", ROLE,
        "--reference-result", assets["reference_result"]["path"],
        "--reference-checkpoint", assets["reference_checkpoint"]["path"],
        "--batch-size", "64", "--device", args.device,
        "--output", str(audit_path),
    ]
    run_command(command, output / "audit.log")
    audit = read_json(audit_path)
    validate_audit(audit, result_path, result, assets)
    gate = evaluate_gate(result, audit)
    gate["assets"] = {
        "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
    }
    (audit_dir / "train_gate_seed20260621.json").write_text(
        json.dumps(gate, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"{'PASS' if check['passed'] else 'FAIL'}: {name}={check['value']} "
        f"{check['operator']} {check['threshold']}"
        for name, check in gate["checks"].items()
    ] + [gate["status"], "Attack validation was not started; test was not accessed."]
    (audit_dir / "train_gate_seed20260621.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[projected_v3] output={output}", flush=True)
    if gate["status"] == "STOP":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
