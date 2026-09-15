"""Single-seed all-eligible random-pair control for the frozen v6.4-S pilot.

Preserves scenes, hash priorities, representation, ordinary BCE and trajectory
trigger. Only removes the low-speed preference. Development validation only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from jcas import PROJECT_ROOT
from jcas.core.poison import load_poison_manifest, sha256_file
from jcas.core.risk_labels import RiskLabelConfig
from jcas.release import metrics_integrity as metrics
from jcas.workflows import motion_prioritized_manifest as selection
from jcas.workflows.evaluator import load_expected_target_manifest, validate_public_target_pool
from jcas.workflows.projected_counterfactual_validation import summarize_common_scores

ROOT = PROJECT_ROOT
VERSION = "v6_4_all_eligible_random_pair_control_v1"
OBJECTIVE = "minimum_original_v6_4_hash_over_all_eligible_pairs_without_motion_preference_v1"
SEED = 20260621
EXPECTED_SCENES = 5776
TARGET_COUNTS = {"all": 25515, "slow_lt_0p5": 19556, "moving_ge_0p5": 5959}
OUTPUT_ROOT = ROOT / "record/reproduction/v6_4_random_pair_control"
PRETRAINING = "record/v6_4/contracts/pretraining/v6_4_pretraining_release_20260828.json"
PRETRAINING_SHA = "1ef6071c9ab767323c8fe98e73f1e137d1777eea6e864b1578ecdfc004709c3f"
BASE_VALIDATION = "record/v6_4/validation/victim_seed20260621.json"
BASE_VALIDATION_SHA = "f43c71e3c68552e8ecf05fe23f3cc4f817487d75767187378e5ffabda32492c9"
EVALUATOR_SHA = "f3c002140e0070e57c9f7e30f04c2498d8c539dc26a9f2360d0e40324aedddd9"
GATES = {
    "slow_incremental_asr": (">=", 0.80),
    "overall_incremental_asr": (">=", 0.60),
    "slow_clean_activation": ("<=", 0.025),
    "overall_adjacent_incremental_fp": ("<=", 0.015),
    "clean_pair_auc": (">=", 0.9849),
}
TRAINING = {
    "model_name": "genconv", "seed": SEED, "hidden_dim": 128, "num_layers": 3,
    "dropout": 0.1, "norm": "layer", "decoder_hidden_dim": 128, "decoder_num_layers": 2,
    "batch_size": 64, "lr": 0.001, "weight_decay": 0.0001, "epochs": 50,
    "checkpoint_metric": "val_loss", "require_strict_label": True,
    "label_mode": "dynamic_risk", "risk_base_distance_m": 5.0,
    "risk_reaction_time_s": 1.0, "risk_safe_decel_mps2": 4.0,
    "edge_feature_mode": selection.MOTION_NORMALIZED_EDGE_FEATURE_MODE,
    "evaluate_test": False,
}


def asset_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or "test" in path.relative_to(ROOT).parts:
        raise ValueError(f"asset outside development scope: {path}")
    return path


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def read_table(path):
    return pd.read_csv(path, float_precision="round_trip",
                       dtype={key: str for key in ("scenario_id", "src_track_id", "dst_track_id")})


def bind(path, expected=None):
    path = asset_path(path)
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return {"path": str(path), "sha256": digest}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify_bound(context):
    for asset in context["assets"].values():
        bind(asset["path"], asset["sha256"])
    for path, digest in context["sources"].items():
        bind(path, digest)


def preflight():
    """Only metadata, CSVs and byte hashes; no trajectory/model loading."""
    assets = {"pretraining": bind(PRETRAINING, PRETRAINING_SHA),
              "baseline_validation": bind(BASE_VALIDATION, BASE_VALIDATION_SHA)}
    release = read_json(assets["pretraining"]["path"])
    if release["test_access_authorized"] is not False or release["pilot_seed"] != SEED:
        raise ValueError("unexpected frozen low-speed experiment")
    pilot = release["pilot_manifest"]
    assets["slow_manifest"] = bind(pilot["path"], pilot["sha256"])
    assets["slow_metadata"] = bind(pilot["metadata_path"], pilot["metadata_sha256"])
    meta = read_json(assets["slow_metadata"]["path"])
    for name in ("graph_manifest", "base_manifest", "base_metadata"):
        assets[name] = bind(meta[name + "_path"], meta[name + "_sha256"])
    graph_path = Path(assets["graph_manifest"]["path"])
    assets["graph_contract"] = bind(str(graph_path) + ".metadata.json")
    graph_contract = selection.validate_graph_manifest_contract(graph_path)
    frozen_contract = meta["graph_manifest_contract"]
    if (asset_path(graph_contract["path"]) != asset_path(frozen_contract["path"])
            or {k: v for k, v in graph_contract.items() if k != "path"}
            != {k: v for k, v in frozen_contract.items() if k != "path"}):
        raise ValueError("grouped-v5 graph contract changed")
    slow, _ = load_poison_manifest(
        Path(assets["slow_manifest"]["path"]), RiskLabelConfig(), expected_split="train",
        require_strict_label=True, require_metadata_binding=True,
        expected_graph_manifest_sha256=assets["graph_manifest"]["sha256"],
    )
    base = read_table(assets["base_manifest"]["path"])
    if (len(slow) != EXPECTED_SCENES or slow.scenario_id.duplicated().any()
            or set(slow.scenario_id) != set(base.scenario_id)
            or round(meta["eligible_scenarios"] * 0.05) != len(slow)):
        raise ValueError("frozen 5% scene set differs")
    if meta["selection_seed"] != SEED or meta["preferred_motion_regime"] != selection.SLOW_MOTION_REGIME:
        raise ValueError("unexpected frozen selection policy")
    graph_rows = selection.load_graph_manifest(graph_path.parent, graph_path)
    indexed = graph_rows.set_index("scenario_id")
    if not set(slow.scenario_id).issubset(indexed.index) or not indexed.loc[slow.scenario_id, "split"].eq("train").all():
        raise ValueError("poison scenes escaped train")
    clean = release["clean_reference"]
    for name in ("result", "checkpoint"):
        assets["reference_" + name] = bind(clean[name + "_path"], clean[name + "_sha256"])
    clean_result = read_json(assets["reference_result"]["path"])
    if (clean_result["config"]["poison_manifest"] is not None
            or clean_result["config"]["evaluate_test"] is not False
            or clean_result["config"]["edge_feature_mode"] != TRAINING["edge_feature_mode"]):
        raise ValueError("unexpected clean reference")
    baseline = read_json(assets["baseline_validation"]["path"])
    for name in ("target_manifest", "target_scores"):
        assets["baseline_" + name] = bind(baseline[name]["path"], baseline[name]["sha256"])
    for name in ("clean_edge_scores", "clean_pair_scores", "collateral_transitions"):
        record = baseline["metric_evidence"][name]
        assets["baseline_" + name] = bind(record["path"], record["sha256"])
    ckpt = baseline["checkpoint"]
    assets["baseline_result"] = bind(ckpt["training_result"], ckpt["training_result_sha256"])
    assets["baseline_checkpoint"] = bind(ckpt["path"], ckpt["sha256"])
    threshold = clean_result["val_pair_metrics"]["threshold"]
    if baseline["split"] != "val" or baseline["thresholds"]["common"] != threshold:
        raise ValueError("baseline split/threshold differs")
    public = assets["baseline_target_manifest"]
    pool = load_expected_target_manifest(Path(public["path"]), public["sha256"])
    if (len(pool) != TARGET_COUNTS["all"] or pool.groupby("motion_regime").size().to_dict()
            != {k: v for k, v in TARGET_COUNTS.items() if k != "all"}):
        raise ValueError("public pool coverage differs")
    if not set(pool.scenario_id).issubset(indexed.index) or not indexed.loc[pool.scenario_id, "split"].eq("val").all():
        raise ValueError("public targets escaped validation")
    code = release["code_asset_manifest"]
    assets["frozen_source_manifest"] = bind(code["path"], code["sha256"])
    sources = {}
    for path, digest in metrics.parse_sha_manifest(Path(assets["frozen_source_manifest"]["path"])).items():
        if not path.endswith(".py"):
            continue
        expected = EVALUATOR_SHA if path == "jcas/workflows/evaluator.py" else digest
        sources[path] = bind(path, expected)["sha256"]
    for path in (Path(__file__).resolve(), Path(metrics.__file__).resolve(),
                 ROOT / "jcas/workflows/projected_counterfactual_validation.py"):
        sources[str(path.relative_to(ROOT))] = bind(path)["sha256"]
    return {"assets": assets, "sources": sources, "slow": slow, "slow_metadata": meta,
            "graph_rows": graph_rows, "graph_contract": graph_contract, "pool": pool,
            "threshold": threshold, "baseline": baseline,
            "validation_graphs": int(graph_rows.split.eq("val").sum())}


def pair_identity(row):
    return str(row["scenario_id"]), min(int(row["src"]), int(row["dst"])), max(int(row["src"]), int(row["dst"]))


def choose_random_candidate(candidates):
    """Same fixed hash as v6.4-S, without its regime preference."""
    if not candidates:
        raise ValueError("empty eligible pool")
    seen = set()
    for row in candidates:
        key = pair_identity(row)
        if key in seen or key[1] == key[2]:
            raise ValueError("duplicate/reversed/self pair in eligible pool")
        seen.add(key)
        regime = selection.motion_regime(row["min_endpoint_speed_mps"])
        if (regime != row["motion_regime"] or row["selection_priority_sha256"]
                != selection.pair_priority(SEED, key[0], key[1], key[2], regime)):
            raise ValueError("candidate hash or motion regime changed")
    if len({key[0] for key in seen}) != 1:
        raise ValueError("candidate pool mixes scenarios")
    return min(candidates, key=lambda row: (row["selection_priority_sha256"], pair_identity(row)))


def generate_tables(context):
    """Read only the original poisoned train scenes; never load val/test graphs."""
    graph_path = Path(context["assets"]["graph_manifest"]["path"])
    graph_rows = {str(row.scenario_id): row for row in context["graph_rows"].itertuples(index=False)}
    rows, candidates_all = [], []
    for index, original in enumerate(context["slow"].sort_values("scenario_id").to_dict("records"), 1):
        scenario = str(original["scenario_id"])
        graph_row = graph_rows[scenario]
        if graph_row.split != "train":
            raise ValueError("selection graph is not train")
        path = selection.resolve_verified_graph_path(graph_row, graph_path.parent)
        with np.load(path, allow_pickle=True) as graph:
            config = RiskLabelConfig()
            labels = selection.labels_for_graph(graph, config)
            groups = selection.eligible_negative_pair_groups(
                graph, config, require_strict_label=True, perturb_window=10,
                require_bi_endpoint_contiguous=True, label_bundle=labels,
            )
            candidates = selection._scenario_candidates(graph, groups, scenario_id=scenario, selection_seed=SEED)
            frozen_choice = selection.choose_motion_candidate(candidates, preferred_regime=selection.SLOW_MOTION_REGIME)
            if (int(frozen_choice["src"]), int(frozen_choice["dst"])) != (int(original["src"]), int(original["dst"])):
                raise ValueError("reconstructed eligible pool does not reproduce frozen low-speed choice")
            chosen = choose_random_candidate(candidates)
            alpha, allocation = selection.fixed_symmetric_bi_endpoint_allocation(
                graph, src=int(chosen["src"]), dst=int(chosen["dst"]), displacement_m=0.2,
            )
            if alpha != 0.5:
                raise ValueError("symmetric trigger allocation changed")
        row = {**original, **chosen}
        for key in ("preferred_motion_regime", "preferred_pair_available"):
            row.pop(key, None)
        for key in ("allocation_total_feature_energy", "allocation_incident_edge_energy",
                    "allocation_endpoint_node_energy", "allocation_candidate_count", "allocation_non_target_incident_edges"):
            row[key] = allocation[key]
        row.update(graph_sha256=str(graph_row.graph_sha256), eligible_pair_count=len(candidates),
                   pair_changed_from_v6=frozenset((row["src"], row["dst"])) != frozenset((row["base_src"], row["base_dst"])),
                   slow_reference_src=int(original["src"]), slow_reference_dst=int(original["dst"]),
                   pair_changed_from_slow=pair_identity(chosen) != pair_identity(original))
        rows.append(row)
        candidates_all.extend({**item, "split": "train", "graph_sha256": str(graph_row.graph_sha256)} for item in candidates)
        if index % 100 == 0 or index == len(context["slow"]):
            print(f"[random-pair] prepared {index}/{len(context['slow'])} train scenes", flush=True)
    return (pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True),
            pd.DataFrame(candidates_all).sort_values(["scenario_id", "src", "dst"]).reset_index(drop=True))


def selection_summary(frame, candidates):
    return {"scenarios": len(frame), "eligible_pairs": len(candidates),
            "selected_motion_counts": frame.groupby("motion_regime").size().to_dict(),
            "changed_pairs_from_slow": int(frame.pair_changed_from_slow.sum()),
            "same_pairs_as_slow": int((~frame.pair_changed_from_slow).sum())}


def check_selection(frame, candidates, context):
    slow = context["slow"].set_index("scenario_id")
    if (len(frame) != len(slow) or frame.scenario_id.duplicated().any()
            or set(frame.scenario_id) != set(slow.index) or set(candidates.scenario_id) != set(slow.index)):
        raise ValueError("selection changed frozen scenes")
    if not frame.split.eq("train").all() or not candidates.split.eq("train").all():
        raise ValueError("selection escaped train")
    constant = {"displacement_m": 0.2, "perturb_window": 10, "ramp_style": "minimum_jerk",
                "velocity_mode": "residual", "allocation_alpha": 0.5,
                "allocation_policy": "fixed_symmetric_biend_v1", "poison_label": 1, "seed": SEED}
    for column, value in constant.items():
        if not frame[column].eq(value).all():
            raise ValueError(f"trigger or seed changed: {column}")
    chosen = frame.set_index("scenario_id")
    graphs = context["graph_rows"].set_index("scenario_id")
    for scene, group in candidates.groupby("scenario_id", sort=True):
        records = group.to_dict("records")
        expected = choose_random_candidate(records)
        original = selection.choose_motion_candidate(records, preferred_regime=selection.SLOW_MOTION_REGIME)
        if (int(original["src"]), int(original["dst"])) != (int(slow.loc[scene, "src"]), int(slow.loc[scene, "dst"])):
            raise ValueError("candidate audit does not reproduce frozen selection")
        observed = chosen.loc[scene]
        for column in expected:
            if column == "scenario_id":
                continue
            left, right = observed[column], expected[column]
            if isinstance(right, float):
                if not math.isclose(float(left), right, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"selected candidate differs: {column}")
            elif left != right:
                raise ValueError(f"selected candidate differs: {column}")
        if (not group.graph_sha256.eq(graphs.loc[scene, "graph_sha256"]).all()
                or int(observed.eligible_pair_count) != len(group)
                or int(slow.loc[scene, "eligible_pair_count"]) != len(group)):
            raise ValueError("candidate graph binding/count differs")
        changed = (int(observed.src), int(observed.dst)) != (int(original["src"]), int(original["dst"]))
        if bool(observed.pair_changed_from_slow) != changed:
            raise ValueError("pair-change audit differs")


def make_plan(context, files, stats):
    return {"version": VERSION, "seed": SEED, "selection_seed": SEED, "selection_objective": OBJECTIVE,
            "training_access": "data_only_poisoning_with_ordinary_bce", "initialization": "random",
            "assets": context["assets"], "sources": context["sources"], "files": files,
            "selection": stats, "training": TRAINING, "target_counts": TARGET_COUNTS,
            "common_threshold": context["threshold"], "reference_gates": {k: list(v) for k, v in GATES.items()},
            "validation": "full same-pool development validation after training; report regardless of reference gate",
            "stop_rule": "one seed then review; no automatic replication or test",
            "test_access_authorized": False,
            "evaluator_note": "training/core source matches v6.4 freeze; existing evaluator adds public-pool binding and access metadata"}


def prepare(context, directory):
    directory.mkdir(parents=True, exist_ok=False)
    frame, candidates = generate_tables(context)
    check_selection(frame, candidates, context)
    for name, data in (("poison.csv", frame), ("candidates.csv", candidates)):
        data.to_csv(directory / name, index=False, mode="x")
    stats = selection_summary(frame, candidates)
    meta = copy.deepcopy(context["slow_metadata"])
    for key in ("preferred_motion_regime", "scenarios_with_preferred_candidate"):
        meta.pop(key, None)
    meta.update(experiment=VERSION, selection_objective=OBJECTIVE, status="frozen_train_only_random_control_manifest",
                manifest_path=str(directory / "poison.csv"), manifest_sha256=sha256_file(directory / "poison.csv"),
                selected_motion_counts=stats["selected_motion_counts"], candidate_pair_count=len(candidates),
                changed_pairs=int(frame.pair_changed_from_v6.sum()), changed_pair_rate=float(frame.pair_changed_from_v6.mean()),
                slow_reference_manifest=context["assets"]["slow_manifest"], random_control=stats)
    write_json(directory / "poison.csv.metadata.json", meta)
    files = {name: {"filename": name, "sha256": sha256_file(directory / name)}
             for name in ("poison.csv", "poison.csv.metadata.json", "candidates.csv")}
    verify_bound(context)
    path = directory / "plan.json"
    write_json(path, make_plan(context, files, stats))
    return path


def checked_plan(path, context):
    path = asset_path(path)
    plan = read_json(path)
    files = plan.get("files", {})
    if set(files) != {"poison.csv", "poison.csv.metadata.json", "candidates.csv"}:
        raise ValueError("unexpected plan files")
    for name, item in files.items():
        if item["filename"] != name or (path.parent / name).resolve().parent != path.parent:
            raise ValueError("plan file escaped its directory")
        bind(path.parent / name, item["sha256"])
    frame, _ = load_poison_manifest(path.parent / "poison.csv", RiskLabelConfig(), expected_split="train",
                                   require_strict_label=True, require_metadata_binding=True,
                                   expected_graph_manifest_sha256=context["assets"]["graph_manifest"]["sha256"])
    candidates = read_table(path.parent / "candidates.csv")
    check_selection(frame, candidates, context)
    if plan != make_plan(context, files, selection_summary(frame, candidates)):
        raise ValueError("plan/source/assets/protocol changed")
    metadata = read_json(path.parent / "poison.csv.metadata.json")
    if (metadata["selection_objective"] != OBJECTIVE or metadata["experiment"] != VERSION
            or metadata["model_outputs_used"] is not False or metadata["original_test_used"] is not False
            or metadata["original_validation_used"] is not False):
        raise ValueError("random manifest has incorrect provenance")
    return plan


def training_command(context, plan_path, output, device):
    command = [sys.executable, "-m", "jcas.workflows.trainer",
               "--graph-dir", str(Path(context["assets"]["graph_manifest"]["path"]).parent),
               "--graph-manifest", context["assets"]["graph_manifest"]["path"],
               "--output-dir", str(output / "train"), "--poison-manifest", str(plan_path.parent / "poison.csv")]
    for key, value in TRAINING.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif value is not False:
            command.extend([flag, str(value)])
    return command + ["--device", device]


def validate_training(path, context, plan_path):
    result = read_json(path)
    config = result["config"]
    for key, value in TRAINING.items():
        if config.get(key) != value:
            raise ValueError(f"training setting differs: {key}")
    for key in ("max_train_graphs", "max_val_graphs", "max_test_graphs", "experimental_trigger_schedule"):
        if config.get(key) is not None:
            raise ValueError("truncated/scheduled training is not this control")
    if any(result.get(k) is not None for k in ("test_stats", "test_metrics", "test_pair_metrics")):
        raise ValueError("unexpected test metrics")
    if result["training_protocol"]["objective"] != "ordinary_binary_cross_entropy_symmetric_pair_labels":
        raise ValueError("not ordinary BCE")
    if result["training_protocol"].get("initialization") != "random":
        raise ValueError("warm-start not permitted")
    if len(result["history"]) != 50 or [r["epoch"] for r in result["history"]] != list(range(1, 51)):
        raise ValueError("training incomplete")
    if result["graph_manifest"]["sha256"] != context["assets"]["graph_manifest"]["sha256"]:
        raise ValueError("training graph manifest differs")
    poison_path = plan_path.parent / "poison.csv"
    poison = result["poison_manifest"]
    if (asset_path(config["poison_manifest"]) != poison_path
            or poison["sha256"] != sha256_file(poison_path)
            or poison["metadata_sha256"] != sha256_file(str(poison_path) + ".metadata.json")
            or poison["rows"] != EXPECTED_SCENES or poison["applied_train_rows"] != EXPECTED_SCENES
            or poison["selection_objective"] != OBJECTIVE):
        raise ValueError("training did not use this exact random manifest")
    if result["train_stats"]["poisoned_graphs"] != EXPECTED_SCENES or result["val_stats"]["poisoned_graphs"] != 0:
        raise ValueError("incorrect poison application")
    checkpoint = bind(Path(path).parent / "best_model.pt", result["checkpoint"]["sha256"])
    return result, checkpoint


def evaluation_command(context, checkpoint, output, device):
    assets = context["assets"]
    public = assets["baseline_target_manifest"]
    return [sys.executable, "-m", "jcas.workflows.evaluator",
            "--graph-dir", str(Path(assets["graph_manifest"]["path"]).parent),
            "--graph-manifest", assets["graph_manifest"]["path"], "--checkpoint", checkpoint["path"],
            "--output", str(output), "--evaluation-model-role", "victim",
            "--clean-reference-result", assets["reference_result"]["path"],
            "--expected-target-manifest", public["path"], "--expected-target-manifest-sha256", public["sha256"],
            "--split", "val", "--target-seed", str(SEED),
            "--orientation-policy", "lower_destination_mean_speed_v1",
            "--allocation-policy", "fixed_symmetric_biend_v1", "--device", device]


def summarize_evaluation(path, context, *, training_asset, checkpoint, baseline=False):
    result = read_json(path)
    if (result.get("split") != "val" or result.get("evaluation_complete") is not True
            or result.get("formal_complete") is not False or result.get("evaluation_model_role") != "victim"
            or result.get("evaluation_phase") != "development_validation"):
        raise ValueError("incomplete or non-development evaluation")
    if (result["checkpoint"]["sha256"] != checkpoint["sha256"]
            or result["checkpoint"]["training_result_sha256"] != training_asset["sha256"]
            or result["graph_manifest"]["sha256"] != context["assets"]["graph_manifest"]["sha256"]
            or result["graph_manifest"]["split_graphs_verified"] != context["validation_graphs"]):
        raise ValueError("evaluation used different model/data assets")
    if (result["thresholds"]["common"] != context["threshold"]
            or result["clean_reference"]["result_sha256"] != context["assets"]["reference_result"]["sha256"]
            or result["clean_reference"]["checkpoint_sha256"] != context["assets"]["reference_checkpoint"]["sha256"]
            or result["target_sampling"]["seed"] != SEED or result["target_selection_uses_model_output"] is not False):
        raise ValueError("evaluation threshold/reference/selection changed")
    if not baseline:
        binding = result["target_manifest"].get("public_pool_binding") or {}
        if (binding.get("sha256") != context["assets"]["baseline_target_manifest"]["sha256"]
                or binding.get("verified_before_model_inference") is not True
                or binding.get("all_identities_match") is not True
                or result["evaluator"]["sha256"] != EVALUATOR_SHA
                or result["training_access_context"]["threat_model"] != "data_only_poisoning"):
            raise ValueError("evaluation lacks bound public pool/data-only identity")
    records = {"target": result["target_manifest"], "scores": result["target_scores"],
               "directed": result["metric_evidence"]["clean_edge_scores"],
               "pairs": result["metric_evidence"]["clean_pair_scores"],
               "transitions": result["metric_evidence"]["collateral_transitions"]}
    frames, bindings = {}, {}
    if result["metric_evidence"].get("complete") is not True:
        raise ValueError("incomplete metric evidence")
    for name, record in records.items():
        bindings[name] = bind(record["path"], record["sha256"])
        frames[name] = read_table(bindings[name]["path"])
        if len(frames[name]) != record["rows"]:
            raise ValueError("metric evidence row count differs")
    validate_public_target_pool(frames["target"], context["pool"])
    scores = frames["scores"]
    if not np.allclose(scores.probability_delta, scores.triggered_pair_probability - scores.clean_pair_probability, rtol=0, atol=1e-12):
        raise ValueError("inconsistent score delta")
    for name, kind in (("evaluated_model", "attack_metrics"), ("common", "attack_metrics_common_threshold")):
        threshold = float(result["thresholds"]["evaluated_model_own" if name == "evaluated_model" else "common"])
        if not scores[name + "_threshold"].eq(threshold).all():
            raise ValueError("score threshold differs")
        metrics.assert_metrics(kind, metrics.target_metrics(scores, threshold), result[kind])
    metrics.verify_prediction_evidence(frames["directed"], frames["pairs"], frames["transitions"], scores, result)
    groups = summarize_common_scores(scores, frames["transitions"], context["pool"], context["threshold"])
    for regime, group in groups.items():
        if group["targets"] != TARGET_COUNTS[regime]:
            raise ValueError("motion target coverage differs")
        pairs = frames["pairs"]
        if regime != "all":
            scenes = scores.loc[scores.motion_regime.eq(regime), "scenario_id"]
            pairs = pairs[pairs.scenario_id.isin(scenes)]
        group["clean_pair_metrics"] = metrics.classification_metrics(
            pairs.label.to_numpy(np.int8), pairs.pair_probability.to_numpy(float), context["threshold"])
    return {"groups": groups, "assets": {"evaluation": bind(path), **bindings}}


def reference_gate(groups):
    values = {"slow_incremental_asr": groups["slow_lt_0p5"]["incremental_flip_rate_all_targets"],
              "overall_incremental_asr": groups["all"]["incremental_flip_rate_all_targets"],
              "slow_clean_activation": groups["slow_lt_0p5"]["clean_activation_rate"],
              "overall_adjacent_incremental_fp": groups["all"]["collateral"]["adjacent_negative"]["incremental_positive_rate"],
              "clean_pair_auc": groups["all"]["clean_pair_metrics"]["auc"]}
    checks = {}
    for name, (operator, threshold) in GATES.items():
        value = values[name]
        finite = isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        passed = finite and (value >= threshold if operator == ">=" else value <= threshold)
        checks[name] = {"value": value if finite else None, "operator": operator, "threshold": threshold, "passed": bool(passed)}
    return {"status": "PASS" if all(c["passed"] for c in checks.values()) else "STOP", "checks": checks,
            "interpretation": "original v6.4-S reference gates; no automatic continuation even if PASS"}


def comparison_rows(random_groups, slow_groups):
    rows = []
    for role, groups in (("slow_prioritized_v6_4", slow_groups), ("all_eligible_random", random_groups)):
        for regime, group in groups.items():
            rows.append({"model_role": role, "seed": SEED, "regime": regime, "targets": group["targets"],
                         "incremental_asr": group["incremental_flip_rate_all_targets"],
                         "absolute_asr": group["absolute_asr"], "conditional_flip": group["conditional_flip_rate"],
                         "clean_activation": group["clean_activation_rate"],
                         "mean_probability_delta": group["target_probability_delta_mean"],
                         "adjacent_incremental_fp": group["collateral"]["adjacent_negative"]["incremental_positive_rate"],
                         "nonincident_incremental_fp": group["collateral"]["nonincident_negative"]["incremental_positive_rate"],
                         "clean_pair_auc": group["clean_pair_metrics"]["auc"]})
    return pd.DataFrame(rows)


def run_command(command, log):
    print("[random-pair] " + " ".join(command), flush=True)
    with Path(log).open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise RuntimeError(f"subprocess failed ({status}); see {log}")


def execute(context, plan_path, output, device, result_path=None):
    plan = checked_plan(plan_path, context)
    write_json(output / "protocol.json", {"version": VERSION, "plan": bind(plan_path),
                                          "training_started": result_path is None,
                                          "test_access_authorized": False})
    if result_path is None:
        run_command(training_command(context, plan_path, output, device), output / "train.log")
        found = list((output / "train").glob("*/result.json"))
        if len(found) != 1:
            raise ValueError("expected exactly one completed training result")
        result_path = found[0]
    result, checkpoint = validate_training(result_path, context, plan_path)
    training_asset = bind(result_path)
    verify_bound(context)
    evaluation_path = output / "validation" / "victim.json"
    evaluation_path.parent.mkdir()
    run_command(evaluation_command(context, checkpoint, evaluation_path, device), output / "validation.log")
    current = summarize_evaluation(evaluation_path, context, training_asset=training_asset, checkpoint=checkpoint)
    baseline = summarize_evaluation(Path(context["assets"]["baseline_validation"]["path"]), context,
                                    training_asset=context["assets"]["baseline_result"],
                                    checkpoint=context["assets"]["baseline_checkpoint"], baseline=True)
    gate = reference_gate(current["groups"])
    comparison = comparison_rows(current["groups"], baseline["groups"])
    verify_bound(context)
    checked_plan(plan_path, context)
    bind(training_asset["path"], training_asset["sha256"])
    bind(checkpoint["path"], checkpoint["sha256"])
    comparison.to_csv(output / "comparison.csv", index=False, mode="x")
    write_json(output / "gate.json", gate)
    summary = {"version": VERSION, "status": "VALIDATION_COMPLETE_REVIEW_REQUIRED", "gate_status": gate["status"],
               "seed": SEED, "plan": bind(plan_path), "candidate": {"result": training_asset, "checkpoint": checkpoint},
               "selection": plan["selection"], "common_threshold": context["threshold"],
               "candidate_checkpoint_epoch": min(result["history"], key=lambda r: r["val_loss"])["epoch"],
               "random": current, "slow_baseline_same_seed": baseline,
               "outputs": {name: bind(output / name) for name in ("comparison.csv", "gate.json")},
               "scope": {"development_validation": True, "test_accessed": False, "automatic_replication": False},
               "limits": ["Single initialization seed and fixed selection seed; not a three-seed average.",
                          "Same 5776 scenes and original hash; only removes low-speed preference.",
                          "Low-speed ASR is a conditional subgroup, not whole-pool ASR.",
                          "Validation is already observed development data and also selects the clean-loss checkpoint."]}
    write_json(output / "summary.json", summary)
    print(comparison.to_string(index=False), flush=True)
    print(f"[random-pair] gate={gate['status']}; review {output / 'summary.json'}; no further jobs", flush=True)
    return gate["status"]


def new_directory(requested=None):
    if requested:
        path = Path(requested).resolve()
        if not path.is_relative_to(ROOT / "record/reproduction"):
            raise ValueError("output must be a new directory under record/reproduction")
        path.mkdir(parents=True, exist_ok=False)
        return path
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"seed{SEED}_", dir=OUTPUT_ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="metadata-only preflight, no output or graph/model loading")
    parser.add_argument("--prepare-only", action="store_true", help="prepare a train-only random manifest; no training/evaluation")
    parser.add_argument("--plan", help="original plan.json from this control")
    parser.add_argument("--result", help="completed training result; requires its original --plan; skip retraining")
    parser.add_argument("--output-dir", help="new directory under record/reproduction")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.result and not args.plan:
        parser.error("--result requires the original --plan")
    if args.prepare_only and (args.plan or args.result or args.check_only):
        parser.error("--prepare-only cannot be combined with --plan/--result/--check-only")
    context = preflight()
    plan_path = asset_path(args.plan) if args.plan else None
    if plan_path:
        checked_plan(plan_path, context)
        if args.result:
            validate_training(asset_path(args.result), context, plan_path)
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "train_scenes": len(context["slow"]),
                          "target_counts": TARGET_COUNTS, "training_started": False,
                          "graphs_loaded": 0, "test_accessed": False}, indent=2))
        return
    if not args.prepare_only and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly use --device cpu")
    output = new_directory(args.output_dir)
    if plan_path is None:
        plan_path = prepare(context, output / "plan")
    if args.prepare_only:
        print(f"[random-pair] prepared {plan_path}; no training/evaluation", flush=True)
        return
    status = execute(context, plan_path, output, args.device, asset_path(args.result) if args.result else None)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
