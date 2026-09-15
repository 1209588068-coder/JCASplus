"""Max-two-target rotation control using the unchanged, source-bound v4 engine.

Only role-table breadth changes. Original scenes, holdout pairs, training
objective and nine gates stay fixed. No attack-validation/test job is launched.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from jcas.core.poison import sha256_file
from jcas.core.target_rotation import pair_key, priority
from jcas.workflows import rotation_target_response_audit as response

fixed = response.fixed
engine = response.rotation
ROOT = response.ROOT
OUTPUT_ROOT = ROOT / "record/reproduction/rotating_counterfactual_v4_max2"
VERSION = "two_target_rotation_control_v1"
ROLE = "rotating_v4_max2"
PREVIOUS_RESPONSE = response.RUN / "diagnostics/additional_targets_c4sii0a5/summary.json"
PREVIOUS_RESPONSE_SHA = "a43a72767159722d3605c79ec17d3a4480a78b461c6f76f97a18973cef8a858c"
PREVIOUS_GRADIENT = response.RUN / "diagnostics/gradient_interference_a1xywy6h/summary.json"
PREVIOUS_GRADIENT_SHA = "44f1122280670a79422f23e986d9fffde3a28be89ac0b952a793745590a1bfd6"
EXPECTED_COVERAGE = {
    "scenarios": 5776, "rotatable_scenarios": 1037, "train_targets": 6813,
    "holdout_targets": 1037,
    "train_slot_histogram": {"1": 4739, "2": 1037, "3": 0, "4": 0},
}
IDENTITY = response.IDENTITY
GROUP_COUNTS = {"rotatable_anchors": 1037, "retained_additional": 1037,
                "dropped_additional": 1378, "heldout": 1037}
MODEL_ROLES = (ROLE, *response.MODEL_ROLES)


def read_table(path):
    return pd.read_csv(path, dtype={"scenario_id": str, "src_track_id": str, "dst_track_id": str})


def derive_tables(parent):
    """Truncate the original model-blind hash ordering, never reselect holdouts."""
    anchors = parent[parent.role.eq("train") & parent.slot.eq(0)]
    engine.validate_roles(parent, anchors)
    records, lineage = [], []
    for _, rows in parent.groupby("scenario_id", sort=True):
        training = rows[rows.role.eq("train")]
        anchor = training[training.slot.eq(0)].iloc[0].to_dict()
        records.append(anchor)
        alternatives = training[training.slot.gt(0)].to_dict("records")
        if alternatives:
            selected = min(alternatives, key=lambda row: (priority(row), pair_key(row)))
            records.append({**selected, "slot": 1})
            for row in alternatives:
                retained = pair_key(row) == pair_key(selected)
                lineage.append({**row, "parent_slot": int(row["slot"]),
                                "max2_slot": 1 if retained else -1,
                                "selection_role": "retained_additional" if retained else "dropped_additional"})
        records.extend(rows[rows.role.eq("holdout")].to_dict("records"))
    roles = pd.DataFrame(records).sort_values(["scenario_id", "role", "slot"]).reset_index(drop=True)
    lineage = pd.DataFrame(lineage).sort_values(["scenario_id", "slot"]).reset_index(drop=True)
    coverage = engine.validate_roles(roles, anchors)
    if any(n > 2 for n in roles[roles.role.eq("train")].groupby("scenario_id").size()):
        raise ValueError("effective two-target cap violated")
    scenes = set(parent[parent.role.eq("holdout")].scenario_id)
    groups = {
        "rotatable_anchors": parent[parent.role.eq("train") & parent.slot.eq(0) & parent.scenario_id.isin(scenes)].copy(),
        "retained_additional": lineage[lineage.selection_role.eq("retained_additional")].copy(),
        "dropped_additional": lineage[lineage.selection_role.eq("dropped_additional")].copy(),
        "heldout": parent[parent.role.eq("holdout")].copy(),
    }
    for name, frame in groups.items():
        frame["target_group"] = name
        groups[name] = frame.sort_values(IDENTITY).reset_index(drop=True)
    return roles, lineage, groups, coverage


def match_scores(frame, targets, model_roles):
    selected = frame.merge(targets[IDENTITY], on=IDENTITY, validate="many_to_one")
    return response.align_records(selected, targets, expected_roles=model_roles)


def parent_context():
    context = response.preflight()
    context["bound"]["previous_response_summary"] = fixed.bind_file(PREVIOUS_RESPONSE, PREVIOUS_RESPONSE_SHA)
    context["bound"]["previous_gradient_summary"] = fixed.bind_file(PREVIOUS_GRADIENT, PREVIOUS_GRADIENT_SHA)
    previous = fixed.read_json(PREVIOUS_RESPONSE)
    if (previous["status"] != "AUDIT_COMPLETE_REVIEW_REQUIRED" or previous["scope"] != response.SCOPE
            or previous["threshold"] != context["threshold"]):
        raise ValueError("previous diagnostic identity differs")
    for name, asset in previous["outputs"].items():
        context["bound"]["previous_response_" + name] = fixed.bind_file(Path(asset["path"]), asset["sha256"])
    protocol = fixed.read_json(Path(previous["outputs"]["protocol.json"]["path"]))
    if protocol["models"] != context["models"]:
        raise ValueError("cached response used different models")
    for name, digest in protocol["sources"].items():
        fixed.bind_file(ROOT / name, digest)
    parent_roles = read_table(context["bound"]["rotation_targets"]["path"])
    roles, lineage, groups, coverage = derive_tables(parent_roles)
    if coverage != EXPECTED_COVERAGE or {k: len(v) for k, v in groups.items()} != GROUP_COUNTS:
        raise ValueError("max2 control changed the fixed scene/target coverage")
    additional = read_table(previous["outputs"]["additional.records.csv"]["path"])
    caches = {"rotatable_anchors": context["reused"]["rotatable_anchors"],
              "heldout": context["reused"]["heldout"],
              "retained_additional": additional, "dropped_additional": additional}
    context["cached"] = {name: match_scores(caches[name], targets, response.MODEL_ROLES)
                         for name, targets in groups.items()}
    context.update(parent_roles=parent_roles, derived_roles=roles, lineage=lineage,
                   control_groups=groups, control_coverage=coverage)
    context["sources"][str(Path(__file__).resolve().relative_to(ROOT))] = sha256_file(__file__)
    print(f"[max2] parent verified; scenes=5776, rotating=1037, train_targets=6813, dropped=1378", flush=True)
    return context


def engine_plan(context, targets_sha):
    # Keep the legacy format's declared CAPACITY=4, not the experimental cap.
    # The separate outer plan and exact derived CSV enforce EFFECTIVE cap=2.
    plan = copy.deepcopy(context["plan"])
    plan.update(
        targets={"filename": "targets.csv", "sha256": targets_sha},
        coverage=context["control_coverage"],
        experiment_identity={"version": VERSION, "effective_max_train_targets": 2},
        graph_scope={"train_graphs_read_during_derivation": 0, "validation_graphs_read": 0,
                     "test_graphs_read": 0, "eligibility_inherited_from_bound_parent_roles": True},
        holdout_scope="same parent train-scene trigger-role holdouts; already observed in development, never new blind validation/test",
    )
    return plan


def outer_plan(context, artifacts):
    return {
        "version": VERSION, "candidate_role": ROLE, "effective_max_train_targets": 2,
        "engine_schema_max_train_targets": 4, "seed": 20260621, "epochs": 50,
        "training_access": "training_procedure_control", "initialization": "random",
        "selection": "keep anchor and minimum original-v4 priority hash among parent training alternatives; identical parent holdouts",
        "parent_plan": context["bound"]["plan"], "parent_models": context["models"],
        "assets": context["bound"], "sources": context["sources"], "files": artifacts,
        "coverage": context["control_coverage"], "comparison_group_counts": {k: len(v) for k, v in context["control_groups"].items()},
        "anchor_gates": {k: list(v) for k, v in fixed.GATES.items()},
        "holdout_gates": {k: list(v) for k, v in engine.HOLDOUT_GATES.items()},
        "test_access_authorized": False, "attack_validation_automatic": False,
        "clean_validation": "unchanged full clean validation loss checkpoint selection",
        "comparison_scope": "all four train target groups; same observed development holdout pool; no new metric gate",
        "status": "READY",
    }


def new_directory(prefix, requested=None):
    if requested:
        path = Path(requested).resolve()
        if not path.is_relative_to(ROOT / "record/reproduction"):
            raise ValueError("new output must be under record/reproduction")
        path.mkdir(parents=True, exist_ok=False)
        return path
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=OUTPUT_ROOT))


def prepare(context, destination):
    destination = new_directory("plan_", destination)
    for name, frame in (("targets.csv", context["derived_roles"]), ("lineage.csv", context["lineage"])):
        frame.to_csv(destination / name, index=False, mode="x")
    inner = engine_plan(context, sha256_file(destination / "targets.csv"))
    engine.write_json(destination / "engine_plan.json", inner)
    files = {name: {"filename": name, "sha256": sha256_file(destination / name)}
             for name in ("targets.csv", "lineage.csv", "engine_plan.json")}
    engine.write_json(destination / "plan.json", outer_plan(context, files))
    print(f"[max2] prepared {destination / 'plan.json'}", flush=True)
    return destination / "plan.json"


def assert_table_equal(actual, expected):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    except AssertionError as error:
        raise ValueError("role table/lineage differs from the parent hash selection") from error


def checked_plan(path, context):
    path = Path(path).resolve()
    plan = fixed.read_json(path)
    files = plan.get("files", {})
    if set(files) != {"targets.csv", "lineage.csv", "engine_plan.json"}:
        raise ValueError("missing or unexpected max2 plan files")
    for name, asset in files.items():
        target = (path.parent / asset["filename"]).resolve()
        if asset["filename"] != name or target.parent != path.parent:
            raise ValueError("plan artifact escaped its directory")
        fixed.bind_file(target, asset["sha256"])
    if plan != outer_plan(context, files):
        raise ValueError("max2 plan identity/assets/sources/gates changed")
    roles = read_table(path.parent / "targets.csv")
    assert_table_equal(roles, context["derived_roles"])
    assert_table_equal(read_table(path.parent / "lineage.csv"), context["lineage"])
    inner_path = path.parent / "engine_plan.json"
    inner = fixed.read_json(inner_path)
    if inner != engine_plan(context, files["targets.csv"]["sha256"]):
        raise ValueError("engine plan differs from the unchanged training protocol")
    # The unmodified training executable will perform this same legacy check.
    checked_roles, checked_inner = engine.checked_plan(inner_path)
    assert_table_equal(checked_roles, roles)
    return roles, checked_inner, inner_path


def validate_candidate(result, inner_path, inner, roles):
    engine.validate_candidate(result, inner_path, inner, roles)
    if result["target_rotation"]["coverage"] != EXPECTED_COVERAGE:
        raise ValueError("candidate was not trained with max2 coverage")
    # Exact 50-epoch checks above imply zero exposure for all omitted pairs.
    for row in result["history"]:
        wanted = 1037 if row["epoch"] % 2 == 0 else 0
        if row["train"]["target_rotation"].get("rotated_away_from_anchor") != wanted:
            raise ValueError("candidate did not alternate the two fixed target sets")


def anchor_command(result_path, result, assets, audit_dir, device):
    return [
        sys.executable, "-m", "jcas.workflows.paired_target_response_audit",
        "--graph-dir", str(Path(assets["graph_manifest"]["path"]).parent),
        "--graph-manifest", assets["graph_manifest"]["path"], "--poison-manifest", assets["poison_manifest"]["path"],
        "--candidate-result", str(result_path), "--candidate-checkpoint", str(fixed.asset_path(result["checkpoint"]["path"])),
        "--candidate-role", fixed.ROLE, "--reference-result", assets["reference_result"]["path"],
        "--reference-checkpoint", assets["reference_checkpoint"]["path"],
        "--batch-size", "64", "--device", device, "--output", str(audit_dir / "anchors.json"),
    ]


def supplemental_audit(context, result, device):
    targets = pd.concat([v for k, v in context["control_groups"].items() if k != "rotatable_anchors"], ignore_index=True)
    targets = targets.sort_values(IDENTITY).reset_index(drop=True)
    records, model = [], None
    for start in range(0, len(targets), 64):
        rows = targets.iloc[start:start + 64]
        items = response.load_items(rows, context["graph_rows"], context["graph_dir"])
        if model is None:
            model = response.response._model_from_assets(
                result, fixed.asset_path(result["checkpoint"]["path"]),
                node_dim=items[0].clean.x.shape[1], edge_dim=items[0].clean.edge_attr.shape[1], device=device,
            )
        records.extend(response.score_batch(model, items, rows, role=ROLE, threshold=context["threshold"], device=device))
        print(f"[max2] supplemental train targets={min(start + 64, len(targets))}/{len(targets)}", flush=True)
    return response.align_records(pd.DataFrame(records), targets, expected_roles=(ROLE,))


def gate_for_candidate(result, anchors, group_summaries, parent_holdout):
    held = group_summaries["heldout"][ROLE]
    # Adapt to the legacy gate API in memory; saved model identities stay explicit.
    current = {"targets": held["target_weighted"], "collateral": {
        "adjacent_negative": {"incremental_positive_rate": held["collateral"]["adjacent_negative"]["edge_weighted_incremental_positive_rate"]},
    }}
    gate = engine.combined_gate(result, anchors, {"models": {
        "rotating_v4": current, "fixed_v3": parent_holdout["models"]["fixed_v3"],
    }})
    gate["candidate_role"] = ROLE
    gate["experiment_version"] = VERSION
    gate["interpretation"] = "unchanged original-anchor and observed-development-holdout gates; lift baseline remains fixed_v3"
    return gate


def summarize_groups(candidate, context, epoch):
    summaries, comparisons, cached_rows = {}, [], []
    for name, targets in context["control_groups"].items():
        fresh = match_scores(candidate, targets, (ROLE,))
        cached = context["cached"][name]
        cached_rows.append(cached)
        combined = response.align_records(pd.concat([fresh, cached], ignore_index=True), targets, expected_roles=MODEL_ROLES)
        summaries[name] = {}
        for role, rows in combined.groupby("model_role"):
            item = response.summarize_group(rows, context["threshold"])
            if name == "heldout" and role != ROLE:
                item["cached_full_pool_collateral"] = context["holdout"]["models"][role]["collateral"]
            if name == "rotatable_anchors":
                item["collateral_note"] = "no per-target collateral in anchor CSV; full 5776-scene aggregate not substituted for this subset"
            summaries[name][role] = item
            m = item["target_weighted"]
            adj = item["collateral"]["adjacent_negative"]
            adj_fp = adj["edge_weighted_incremental_positive_rate"] if adj is not None else (
                item.get("cached_full_pool_collateral", {}).get("adjacent_negative", {}).get("incremental_positive_rate"))
            comparisons.append({
                "target_group": name, "model_role": role, "targets": len(rows), "scenarios": rows.scenario_id.nunique(),
                "incremental_asr": m["incremental_flip_rate_all_targets"],
                "scene_weighted_incremental_asr": item["equal_scene_weighted"]["incremental_asr"],
                "conditional_flip": m["conditional_flip_rate"], "absolute_asr": m["absolute_activation_rate"],
                "clean_activation": m["clean_activation_rate"], "mean_probability_delta": m["probability_delta"]["mean"],
                "mean_logit_delta": m["logit_delta"]["mean"], "adjacent_incremental_fp": adj_fp,
            })
    return {
        "experiment_version": VERSION, "candidate_role": ROLE,
        "candidate_checkpoint_epoch": epoch, "threshold": context["threshold"],
        "max2_positive_exposures_to_selected_checkpoint": {
            "rotatable_anchors": (epoch + 1) // 2, "retained_additional": epoch // 2,
            "dropped_additional": 0, "heldout": 0, "unchanged_scene_anchors": epoch,
        },
        "groups": summaries,
        "limits": [
            "One randomized-initialization seed; same protocol does not guarantee bitwise-identical GPU training trajectories.",
            "All response graphs are train; the inherited holdout pool has already been inspected in development.",
            "Max2 changes coverage and per-target exposure together; not an isolated causal effect of gradient interference.",
            "Dropped targets remain clean/non-target supervised and are not new blind holdouts.",
            "Supplemental groups have no new metric gate; all nine original thresholds are unchanged.",
        ],
    }, pd.DataFrame(comparisons), pd.concat(cached_rows, ignore_index=True)


def execute(context, plan_path, output, device_name, result_path=None):
    roles, inner, inner_path = checked_plan(plan_path, context)
    plan_binding = fixed.bind_file(Path(plan_path), sha256_file(plan_path))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select --device cpu")
    engine.write_json(output / "protocol.json", {
        "version": VERSION, "candidate_role": ROLE, "plan": plan_binding,
        "engine_plan": fixed.bind_file(inner_path, sha256_file(inner_path)),
        "sources": context["sources"], "training_started": result_path is None,
        "test_access_authorized": False, "attack_validation_automatic": False,
    })
    if result_path is None:
        fixed.run_command(engine.training_command(inner_path, inner, output, device_name), output / "train.log")
        found = list((output / "train").glob("*/result.json"))
        if len(found) != 1:
            raise RuntimeError("expected exactly one completed max2 training result")
        result_path = found[0]
    result_path = Path(result_path).resolve()
    result = fixed.read_json(result_path)
    validate_candidate(result, inner_path, inner, roles)
    candidate_binding = fixed.bind_file(result_path, sha256_file(result_path))
    checkpoint_binding = fixed.bind_file(fixed.asset_path(result["checkpoint"]["path"]), result["checkpoint"]["sha256"])
    epoch = response.selected_epoch(result)
    audit_dir = output / "audit"
    audit_dir.mkdir()
    fixed.run_command(anchor_command(result_path, result, inner["assets"], audit_dir, device_name), output / "anchors.log")
    anchor_path = audit_dir / "anchors.json"
    anchor = fixed.read_json(anchor_path)
    fixed.validate_audit(anchor, result_path, result, inner["assets"])
    anchor_asset = anchor["assets"]["records"]
    fixed.bind_file(Path(anchor_asset["path"]), anchor_asset["sha256"])
    anchor_records = read_table(anchor_asset["path"])
    anchor_records = anchor_records[anchor_records.model_role.eq(fixed.ROLE)].copy()
    anchor_records["model_role"] = ROLE
    anchor_subset = match_scores(anchor_records, context["control_groups"]["rotatable_anchors"], (ROLE,))
    fresh = supplemental_audit(context, result, device)
    fresh.to_csv(audit_dir / "supplemental.records.csv", index=False, mode="x")
    candidate = pd.concat([anchor_subset, fresh], ignore_index=True)
    summary, comparison, cached = summarize_groups(candidate, context, epoch)
    cached.to_csv(audit_dir / "cached_comparators.records.csv", index=False, mode="x")
    comparison.to_csv(audit_dir / "comparison.csv", index=False, mode="x")
    gate = gate_for_candidate(result, anchor, summary["groups"], context["holdout"])
    gate["assets"] = {"result": candidate_binding, "checkpoint": checkpoint_binding, "plan": plan_binding,
                      "anchor_audit": fixed.bind_file(anchor_path, sha256_file(anchor_path))}
    response.verify_unchanged(context)
    checked_plan(plan_path, context)
    for asset in (plan_binding, candidate_binding, checkpoint_binding):
        fixed.bind_file(Path(asset["path"]), asset["sha256"])
    engine.write_json(audit_dir / "gate.json", gate)
    lines = [f"{'PASS' if c['passed'] else 'FAIL'} {k}: {c['value']} {c['operator']} {c['threshold']}" for k, c in gate["checks"].items()]
    lines += [gate["status"], "No attack validation or test started."]
    with (audit_dir / "gate.txt").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    summary.update(status="PILOT_COMPLETE_REVIEW_REQUIRED", gate_status=gate["status"],
                   original_parent_gate_status="STOP", candidate=gate["assets"],
                   clean_validation_pair_auc=result["val_pair_metrics"]["auc"],
                   attack_validation_started=False, test_accessed=False)
    summary["outputs"] = {name: {"path": str(audit_dir / name), "sha256": sha256_file(audit_dir / name)}
                          for name in ("anchors.json", "supplemental.records.csv", "cached_comparators.records.csv", "comparison.csv", "gate.json", "gate.txt")}
    engine.write_json(audit_dir / "summary.json", summary)
    print(comparison.to_string(index=False), flush=True)
    print("\n".join(lines), flush=True)
    print(f"[max2] output={output}; summary={audit_dir / 'summary.json'}", flush=True)
    return gate["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="derive new plan from bound train-role metadata; no graph/model loading")
    parser.add_argument("--plan", help="outer max2 plan.json; omit to automatically prepare a new plan")
    parser.add_argument("--check-only", action="store_true", help="metadata-only preflight; no output/training/inference")
    parser.add_argument("--result", help="completed result from this exact plan; audit without retraining; requires --plan")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", help="new directory under record/reproduction")
    args = parser.parse_args(argv)
    if args.prepare_only and (args.plan or args.result or args.check_only):
        parser.error("--prepare-only cannot be combined with --plan/--result/--check-only")
    if args.result and not args.plan:
        parser.error("--result requires the original outer --plan")
    context = parent_context()
    if args.plan:
        roles, inner, inner_path = checked_plan(args.plan, context)
        if args.result:
            validate_candidate(fixed.read_json(Path(args.result)), inner_path, inner, roles)
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "effective_max_train_targets": 2,
                          "coverage": context["control_coverage"], "comparison_pools": GROUP_COUNTS,
                          "plan_prepared": bool(args.plan), "training_started": False,
                          "graphs_loaded": 0, "test_accessed": False,
                          "anchor_gates": fixed.GATES, "holdout_gates": engine.HOLDOUT_GATES}, indent=2))
        return
    if args.prepare_only:
        if args.output_dir:
            destination = Path(args.output_dir).resolve()
        else:
            # Reserve a new parent; prepare() creates a previously absent child.
            destination = new_directory("prepared_") / "plan"
        prepare(context, destination)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select --device cpu")
    output = new_directory("seed20260621_", args.output_dir)
    plan_path = Path(args.plan).resolve() if args.plan else prepare(context, output / "plan")
    status = execute(context, plan_path, output, args.device, args.result)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
