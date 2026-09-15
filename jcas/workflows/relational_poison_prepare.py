"""Stage 1 only: independent relational-poison protocol and train feasibility.

No trainer/evaluator subprocess, model loading, attack validation or test access.
The output is explicitly NOT a training-ready poison manifest or an ASR gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from jcas import PROJECT_ROOT
from jcas.core import relational_trigger as trigger
from jcas.core.poison import sha256_file, eligible_negative_pair_groups
from jcas.core.risk_labels import RiskLabelConfig, label_config_dict, labels_for_graph
from jcas.workflows.poison_manifest import load_graph_manifest
from jcas.workflows.trainer import validate_graph_manifest_contract

ROOT = PROJECT_ROOT
VERSION = "relational_poison_stage1_v1"
PARENT = "record/reproduction/v6_4_random_pair_control/seed20260621_dfxklaqf/plan/plan.json"
PARENT_SHA = "8f3a4417bc46d946b52c9c2b2ee0f019349d36b8382e1259cbb962bb85d4750b"
SCENES = 5776
POSITIVES = 4621
SAMPLE_PER_REGIME = 64
SEED = 20260912
REGIMES = ("slow_lt_0p5", "moving_ge_0p5")
OUTPUT_NAME = "record/reproduction/relational_poison_v1"
ARCHIVES = {
    "record/v6/contracts/v6_source_snapshot_20260814.tar.gz": "bff850603da9e5f251c5c88423bb6c6050f96c70aaa9a55e8b8c2a1dd1842540",
    "record/v6_4/contracts/pretraining/v6_4_source_snapshot_20260828.tar.gz": "649b9ece12a71b8565ebb1f1253cb5d8ff231fd75bfeb105c114b2cc9174aa33",
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def safe_asset(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()) or "test" in path.relative_to(ROOT.resolve()).parts:
        raise ValueError(f"asset outside development scope: {path}")
    return path


def bind(value, expected=None):
    path = safe_asset(value)
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return {"path": str(path), "sha256": digest}


def priority(kind, value):
    return hashlib.sha256(f"{VERSION}|{SEED}|{kind}|{value}".encode()).hexdigest()


def group_roles(rows):
    """All existing train recording groups get a fixed 70/15/15 metadata split."""
    train = rows.loc[rows.split.eq("train"), ["scenario_id", "split_group_id"]].copy()
    if train.scenario_id.duplicated().any() or train.split_group_id.isna().any():
        raise ValueError("duplicate scenes or missing recording groups")
    groups = sorted(train.split_group_id.astype(str).unique(), key=lambda value: (priority("group", value), value))
    if len(groups) < 10:
        raise ValueError("insufficient recording groups")
    a, b = int(.70 * len(groups)), int(.85 * len(groups))
    mapping = {group: "construct" if i < a else "develop" if i < b else "audit" for i, group in enumerate(groups)}
    train["inner_role"] = train.split_group_id.astype(str).map(mapping)
    return train.sort_values("scenario_id").reset_index(drop=True)


def planned_rows(selected, graph_rows):
    if len(selected) != SCENES or selected.scenario_id.duplicated().any():
        raise ValueError("expected original 5,776 distinct selected scenes")
    roles = group_roles(graph_rows)
    keep = ["scenario_id", "src", "dst", "src_track_id", "dst_track_id", "motion_regime", "graph_sha256"]
    rows = selected[keep].merge(roles, on="scenario_id", how="left", validate="one_to_one")
    if rows.inner_role.isna().any() or not rows.motion_regime.isin(REGIMES).all():
        raise ValueError("selected targets escaped train or have unknown regime")
    hashes = graph_rows.set_index("scenario_id").graph_sha256
    if not rows.graph_sha256.eq(rows.scenario_id.map(hashes)).all():
        raise ValueError("graph binding differs")
    order = sorted(rows.scenario_id, key=lambda value: (priority("budget", value), value))
    positive = set(order[:POSITIVES])
    controls = {scene: i for i, scene in enumerate(order[POSITIVES:])}
    variants = ("single_canonical_first", "single_canonical_second", "relation_reversed")
    rows["planned_role"] = rows.scenario_id.map(lambda s: "positive" if s in positive else "negative_control")
    rows["planned_variant"] = rows.scenario_id.map(lambda s: "full" if s in positive else variants[controls[s] % 3])
    rows["training_manifest_eligible"] = False
    rows["outer_split"] = "train"
    rows = rows.sort_values("scenario_id").reset_index(drop=True)
    return rows, roles


def audit_sample(rows):
    """Construction group only; develop/audit trajectories remain unopened."""
    selected = []
    for regime in REGIMES:
        candidates = rows.loc[rows.inner_role.eq("construct") & rows.motion_regime.eq(regime)].copy()
        if len(candidates) < SAMPLE_PER_REGIME:
            raise ValueError(f"insufficient construction targets: {regime}")
        candidates["sampling_priority"] = candidates.scenario_id.map(lambda x: priority("feasibility", x))
        selected.append(candidates.sort_values(["sampling_priority", "scenario_id"]).head(SAMPLE_PER_REGIME))
    return pd.concat(selected).reset_index(drop=True)


def protocol():
    return {
        "version": VERSION, "seed": SEED, "stage": "metadata_and_geometry_feasibility_only",
        "training_manifest_eligible": False, "training_authorized_by_this_entry": False,
        "attack_validation_authorized": False, "test_access_authorized": False,
        "source_scene_count": SCENES, "planned_positive_scenes": POSITIVES,
        "planned_control_scenes": SCENES-POSITIVES, "sample_per_regime": SAMPLE_PER_REGIME,
        "sampling_scope": "construct recording groups only; balanced diagnostic, not population estimate",
        "label_config": label_config_dict(RiskLabelConfig()), "budget": asdict(trigger.reference_budget()),
        "budget_origin": "legacy commanded K10/0.1m residual, not absolute vehicle limits",
        "common_cap_m": trigger.COMMON_CAP_M, "relative_budget_fraction": trigger.RELATIVE_BUDGET_FRACTION,
        "legacy_ramp_fraction": 1.0, "templates": trigger.template_bank(),
        "template_selection": "none; all templates and controls reported, no learned winner",
        "implemented": ["observed-only shared templates", "geometry-only common-mode controller", "metadata group reservation", "finite-budget and feature audit"],
        "not_implemented": ["surrogate predictive-neighborhood objective", "cross-group ordinary-BCE learning optimization", "victim training", "ASR gate", "road/curvature/seam feasibility gate"],
        "future_only_label_caveat": "dynamic risk uses unchanged futures; preserved labels do not certify observation/future continuity",
        "heading_policy": "preserve heading-velocity offset when both speeds >=0.5; otherwise keep heading; report discontinuity risk",
        "continuation": "manual review required; no automatic training/validation/test",
    }


def preflight():
    """Metadata and byte bindings only, including old source archives, never .npz/.pt."""
    assets = {"parent": bind(PARENT, PARENT_SHA)}
    parent = read_json(assets["parent"]["path"])
    if parent["selection"]["scenarios"] != SCENES or parent["selection_seed"] != 20260621:
        raise ValueError("unexpected parent selection")
    # Intentionally do not traverse the parent's validation/checkpoint assets.
    for key in ("graph_manifest", "graph_contract", "base_manifest", "slow_manifest"):
        record = parent["assets"][key]
        assets[key] = bind(record["path"], record["sha256"])
    parent_path = Path(assets["parent"]["path"])
    assets["random_pairs"] = bind(parent_path.parent / "poison.csv", parent["files"]["poison.csv"]["sha256"])
    for name, digest in ARCHIVES.items():
        assets[name] = bind(name, digest)
    # Preserve and bind the existing data/feature implementation, including dirty
    # but already bound prior experiment sources. Never modify them to pass SHA.
    sources = {name: bind(name, digest)["sha256"] for name, digest in parent["sources"].items()
               if name.startswith("jcas/") and name.endswith(".py")}
    for name in ("jcas/core/relational_trigger.py", "jcas/workflows/relational_poison_prepare.py"):
        sources[name] = bind(name)["sha256"]
    for path in sorted((ROOT / "jcas").rglob("*.py")):
        sources[str(path.relative_to(ROOT))] = bind(path)["sha256"]
    manifest = Path(assets["graph_manifest"]["path"])
    validate_graph_manifest_contract(manifest)
    graphs = load_graph_manifest(manifest.parent, manifest)
    # Membership metadata must not place the same recording group across splits.
    if graphs.split_group_id.isna().any() or graphs.groupby("split_group_id").split.nunique().gt(1).any():
        raise ValueError("recording group split leakage")
    selected = pd.read_csv(assets["random_pairs"]["path"], dtype={"scenario_id": str, "src_track_id": str, "dst_track_id": str})
    if not selected.split.eq("train").all():
        raise ValueError("parent selected non-train graph")
    for key in ("base_manifest", "slow_manifest"):
        scenes = pd.read_csv(assets[key]["path"], usecols=["scenario_id"], dtype=str)
        if len(scenes) != SCENES or set(scenes.scenario_id) != set(selected.scenario_id):
            raise ValueError("v6.0/v6.4 scene set differs")
    rows, roles = planned_rows(selected, graphs)
    sample = audit_sample(rows)
    return {"assets": assets, "sources": sources, "graphs": graphs, "rows": rows,
            "roles": roles, "sample": sample, "protocol": protocol()}


def new_directory(value=None):
    allowed = ROOT.resolve() / OUTPUT_NAME
    if allowed.resolve() != allowed:
        raise ValueError("experiment output root cannot redirect through symlinks")
    if value is None:
        allowed.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="stage1_", dir=allowed))
    path = safe_asset(value)
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("output must be a new child of record/reproduction/relational_poison_v1")
    path.mkdir(parents=True, exist_ok=False)
    return path


def prepare(context, output=None):
    directory = new_directory(output)
    files = {}
    for key, filename in (("rows", "planned_targets.csv"), ("roles", "inner_train_roles.csv"), ("sample", "feasibility_sample.csv")):
        path = directory / filename
        with path.open("x") as stream:
            context[key].to_csv(stream, index=False, lineterminator="\n")
        files[filename] = bind(path)
    plan = {"protocol": context["protocol"], "assets": context["assets"], "sources": context["sources"], "files": files,
            "counts": context["rows"].groupby(["inner_role", "planned_role", "motion_regime"]).size().rename("scenes").reset_index().to_dict("records")}
    path = directory / "plan.json"
    write_json(path, plan)
    return path


def check_plan(path, context):
    path = safe_asset(path)
    plan = read_json(path)
    for key in ("protocol", "assets", "sources"):
        if plan[key] != context[key]:
            raise ValueError(f"plan {key} no longer matches current frozen stage")
    for key, filename in (("rows", "planned_targets.csv"), ("roles", "inner_train_roles.csv"), ("sample", "feasibility_sample.csv")):
        asset = plan["files"][filename]
        bind(asset["path"], asset["sha256"])
        frame = pd.read_csv(asset["path"], dtype={"scenario_id": str, "split_group_id": str, "src_track_id": str, "dst_track_id": str})
        pd.testing.assert_frame_equal(frame, context[key], check_dtype=False)
    return plan


def verified_train_graph(row, graph_dir):
    if row.split != "train":
        raise ValueError("graph access denied: not train")
    path = safe_asset(graph_dir / str(row.graph_path))
    if not path.is_relative_to(graph_dir.resolve()) or path.suffix != ".npz":
        raise ValueError("graph path escaped bound graph root")
    bind(path, str(row.graph_sha256))
    return path


def seam_diagnostic(graph, result):
    """Observation/future join diagnostics only; never input to the generator."""
    changes = []
    future = np.asarray(graph["future_positions"])
    mask = np.asarray(graph["future_valid_mask"], dtype=bool)
    clean = np.asarray(graph["observed_positions_filled"])
    for node in (result["audit"]["canonical_src"], result["audit"]["canonical_dst"]):
        if mask[node, 0] and np.isfinite(future[node, 0]).all():
            before = np.linalg.norm(future[node, 0] - clean[node, -1])
            after = np.linalg.norm(future[node, 0] - result["positions"][node, -1])
            changes.append(abs(float(after-before)))
    return max(changes) if changes else None


def run_audit(context, plan_path, output=None):
    check_plan(plan_path, context)
    directory = new_directory(output)
    print(f"[relational-stage1] output: {directory}", flush=True)
    write_json(directory / "protocol.json", {"protocol": context["protocol"], "plan": bind(plan_path)})
    graphs = context["graphs"].set_index("scenario_id")
    graph_dir = Path(context["assets"]["graph_manifest"]["path"]).parent
    records = []
    template_items = context["protocol"]["templates"].items()
    expected = len(context["sample"]) * len(context["protocol"]["templates"]) * 5
    records_path = directory / "feasibility.records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for index, target in enumerate(context["sample"].to_dict("records"), 1):
            if target["inner_role"] != "construct" or target["outer_split"] != "train":
                raise ValueError("stage 1 may only open construction-group train graphs")
            row = graphs.loc[target["scenario_id"]]
            if str(row.split_group_id) != str(target["split_group_id"]) or str(row.graph_sha256) != str(target["graph_sha256"]):
                raise ValueError("sample group/graph binding differs")
            path = verified_train_graph(row, graph_dir)
            with np.load(path, allow_pickle=True) as graph:
                src, dst = int(target["src"]), int(target["dst"])
                if str(graph["node_track_ids"][src]) != target["src_track_id"] or str(graph["node_track_ids"][dst]) != target["dst_track_id"]:
                    raise ValueError("target track binding differs")
                labels = labels_for_graph(graph, RiskLabelConfig())
                eligible = eligible_negative_pair_groups(graph, RiskLabelConfig(), require_strict_label=True,
                    perturb_window=10, require_bi_endpoint_contiguous=True, label_bundle=labels)
                edges = np.asarray(graph["edge_index"])
                if not any({int(edges[0, group[0]]), int(edges[1, group[0]])} == {src, dst} for group in eligible):
                    raise ValueError("frozen target no longer eligible")
                original_x = np.asarray(graph["x_node"])
                speed = np.linalg.norm(np.asarray(graph["observed_velocities_filled"])[[src, dst], -1], axis=1).min()
                actual_regime = "slow_lt_0p5" if speed < .5 else "moving_ge_0p5"
                if actual_regime != target["motion_regime"]:
                    raise ValueError("target motion regime differs from bound trajectory")
                for name, template in template_items:
                    # Same templates and targets for c=0/c!=0; controls are not
                    # independently rescaled, and actual budgets are reported.
                    settings = [(False, "full"), (True, "full"),
                                (True, "single_canonical_first"), (True, "single_canonical_second"),
                                (True, "relation_reversed")]
                    for controller, variant in settings:
                        result = trigger.apply_relational_trigger(graph, src=src, dst=dst, template=template,
                                                                  controller=controller, variant=variant)
                        record = {"scenario_id": target["scenario_id"], "split_group_id": target["split_group_id"],
                                  "motion_regime": target["motion_regime"], "template": name, **result["audit"],
                                  "node_feature_delta_l2": float(np.linalg.norm(result["x_node"]-original_x)),
                                  "observation_future_step_change_m": seam_diagnostic(graph, result),
                                  "future_only_risk_labels_preserved_by_construction": True}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        records.append(record)
            stream.flush()
            print(f"[relational-stage1] graphs {index}/{len(context['sample'])}; records {len(records)}/{expected}", flush=True)
    if len(records) != expected or not all(r["budget_pass"] for r in records):
        raise ValueError("incomplete feasibility evidence")
    summary_rows = []
    for (name, controller, variant, regime), group in pd.DataFrame(records).groupby(["template", "controller", "variant", "motion_regime"]):
        summary_rows.append({"template": name, "controller": bool(controller), "variant": variant, "motion_regime": regime,
                             "targets": len(group), "max_endpoint_position_m": float(group.endpoint_position_m.max()),
                             "max_induced_speed_mps": float(group.induced_speed_mps.max()),
                             "max_induced_acceleration_mps2": float(group.induced_acceleration_mps2.max()),
                             "max_induced_jerk_mps3": float(group.induced_jerk_mps3.max()),
                             "mean_common_mode_m": float(np.mean([x["common_max_m"] for x in group.common_mode])),
                             "max_heading_change_rad": float(group.max_heading_change_rad.max()),
                             "mean_node_feature_delta_l2": float(group.node_feature_delta_l2.mean())})
    with (directory / "summary.csv").open("x") as stream:
        pd.DataFrame(summary_rows).to_csv(stream, index=False)
    summary = {"status": "FEASIBILITY_COMPLETE_REVIEW_REQUIRED", "version": VERSION,
               "plan": bind(plan_path), "records": bind(records_path), "summary_csv": bind(directory / "summary.csv"),
               "records_count": len(records), "graphs": len(context["sample"]), "numerical_budget_pass": True,
               "model_loaded": False, "training_performed": False, "attack_validation_accessed": False,
               "test_accessed": False, "heldout_trajectory_accessed": False, "asr_gate": "NOT_RUN",
               "physical_road_feasibility_certified": False, "learned_template_selected": False,
               "training_ready": False, "next_action": "review geometry, seam and budget; module C and predictive B still required"}
    write_json(directory / "summary.json", summary)
    return directory / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if (args.check_only and args.output_dir) or (args.prepare_only and args.plan) or (args.audit and not args.plan):
        parser.error("check-only cannot write; prepare creates a plan; audit requires --plan")
    context = preflight()
    if args.check_only:
        if args.plan:
            check_plan(args.plan, context)
        print(json.dumps({"status": "PREFLIGHT_OK", "planned_scenes": len(context["rows"]),
                          "diagnostic_targets": len(context["sample"]), "training_ready": False,
                          "graph_arrays_loaded": False, "test_accessed": False}, indent=2))
    elif args.prepare_only:
        path = prepare(context, args.output_dir)
        print(f"[relational-stage1] plan: {path}; no jobs started", flush=True)
    else:
        path = run_audit(context, args.plan, args.output_dir)
        print(f"[relational-stage1] review: {path}; ASR gate NOT_RUN; no further jobs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
