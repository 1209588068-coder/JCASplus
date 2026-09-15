"""Fixed-checkpoint audit of the 2,415 additional rotation training targets.

Only this new workflow changes. The existing training plan, source contracts,
STOP gate and all historical scores remain untouched. No training or attack
validation entry point is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jcas.core.poison import sha256_file
from jcas.core.risk_labels import RiskLabelConfig
from jcas.core.target_rotation import MOVING, pair_key
from jcas.workflows import paired_target_response_audit as response
from jcas.workflows import projected_counterfactual_pilot as fixed
from jcas.workflows import rotating_counterfactual_pilot as rotation
from jcas.workflows.trainer import load_manifest

ROOT = fixed.ROOT
RUN = ROOT / "record/reproduction/rotating_counterfactual_v4/seed20260621_mmv5kp6_"
GATE_SHA256 = "ee14bfcee2fec5c396096a0d464426ada3113544f212c900fdf1d89b92ecdc77"
RESULT_SHA256 = "20241467c71c05031174a4490756496fea9ed55756e5e3e2d71fde25151099ed"
PLAN_SHA256 = "fc4742a5549b1436a60166947eb10c36bf74b5da02af04bd1de4262e951936cb"
OLD_ANCHOR = ROOT / (
    "record/reproduction/projected_counterfactual_v3/seed20260621_20260907_114630/"
    "audit/train_response_seed20260621.json"
)
OLD_ANCHOR_SHA256 = "2449ca32f145867133ac74892b54cf0b1c012bd04b50b0a9718093e4960f7259"
VERSION = "rotation_additional_train_target_response_v1"
MODEL_ROLES = ("rotating_v4", "fixed_v3", "clean_reference")
BATCH_SIZE = 64
IDENTITY = ["scenario_id", "src", "dst"]
SCORE_COLUMNS = [
    "clean_pair_probability", "triggered_pair_probability",
    "clean_pair_mean_logit", "triggered_pair_mean_logit",
    "clean_negative_bce", "triggered_positive_bce",
]
SCOPE = {
    "split": "train", "training": False, "optimizer_created": False,
    "validation_graphs_read": 0, "test_graphs_read": 0,
    "threshold_from_existing_clean_validation_metadata": True,
    "existing_training_gate_unchanged": "STOP",
}


def selected_epoch(result):
    history = result["history"]
    if not history or [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
        raise ValueError("missing or unordered epoch history")
    losses = np.asarray([h["val_loss"] for h in history], dtype=float)
    if not np.isfinite(losses).all():
        raise ValueError("nonfinite checkpoint-selection history")
    # Trainer uses strict <, so first occurrence wins a tie.
    epoch = int(np.argmin(losses)) + 1
    if result["checkpoint"].get("state") != "best_clean_validation_selected_state_dict":
        raise ValueError("checkpoint is not the clean-validation-selected state")
    if not np.isclose(float(result["losses"]["val"]), losses[epoch - 1], rtol=0, atol=1e-6):
        raise ValueError("selected checkpoint loss differs from history minimum")
    return epoch


def annotate_targets(roles, epoch):
    if not isinstance(epoch, (int, np.integer)) or epoch < 1:
        raise ValueError("checkpoint epoch must be a positive integer")
    training = roles[roles.role.eq("train")]
    counts = training.groupby("scenario_id").size()
    annotated = roles.copy()
    annotated["training_slots_in_scene"] = annotated.scenario_id.map(counts)
    if annotated.training_slots_in_scene.isna().any():
        raise ValueError("target has no training scene")
    annotated["training_slots_in_scene"] = annotated.training_slots_in_scene.astype(int)
    exposures = []
    for row in annotated.itertuples(index=False):
        if row.role == "holdout":
            exposures.append(0)
        else:
            if not 0 <= int(row.slot) < int(row.training_slots_in_scene):
                raise ValueError("invalid training slot")
            exposures.append(max(0, 1 + (epoch - 1 - int(row.slot)) // int(row.training_slots_in_scene)))
    annotated["v4_positive_exposures_to_checkpoint"] = exposures
    scenes = set(annotated[annotated.role.eq("holdout")].scenario_id)
    groups = {
        "additional_train": annotated[annotated.role.eq("train") & annotated.slot.gt(0)].copy(),
        "rotatable_anchors": annotated[annotated.role.eq("train") & annotated.slot.eq(0) & annotated.scenario_id.isin(scenes)].copy(),
        "heldout": annotated[annotated.role.eq("holdout")].copy(),
    }
    for name, frame in groups.items():
        frame["target_group"] = name
        groups[name] = frame.sort_values(IDENTITY).reset_index(drop=True)
    if any(not frame.motion_regime.eq(MOVING).all() for frame in groups.values()):
        raise ValueError("audit must contain only moving targets")
    return groups


def align_records(frame, targets, expected_roles=MODEL_ROLES):
    """Allow repeated scenes; reject duplicates including reversed unordered pairs."""
    frame = frame.copy()
    required = set(IDENTITY + SCORE_COLUMNS + ["motion_regime", "model_role"])
    if required - set(frame.columns):
        raise ValueError("response records are incomplete")
    if set(frame.model_role) != set(expected_roles):
        raise ValueError("unexpected/missing model roles")
    values = frame[SCORE_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("response contains nonfinite values")
    if not ((frame[SCORE_COLUMNS[:2]] >= 0) & (frame[SCORE_COLUMNS[:2]] <= 1)).all().all():
        raise ValueError("invalid probability")
    if (frame[SCORE_COLUMNS[-2:]] < 0).any().any():
        raise ValueError("invalid BCE")
    for field in ("src", "dst"):
        numeric = pd.to_numeric(frame[field]).to_numpy(float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("noninteger pair identity")
        frame[field] = numeric.astype(np.int64)
    expected = {tuple(r) for r in targets[IDENTITY].itertuples(index=False, name=None)}
    if len(expected) != len(targets):
        raise ValueError("duplicate target identity")
    for _, data in frame.groupby("model_role"):
        unordered = [pair_key(row) for row in data.to_dict("records")]
        if len(set(unordered)) != len(unordered):
            raise ValueError("duplicate or reversed unordered pair records")
        actual = {tuple(r) for r in data[IDENTITY].itertuples(index=False, name=None)}
        if len(data) != len(targets) or actual != expected:
            raise ValueError("model target pool differs")
    if not frame.motion_regime.eq(MOVING).all():
        raise ValueError("unexpected motion regime")
    metadata = targets.drop(columns=["motion_regime"])
    # Only score fields and optional per-target collateral belong to the response.
    frame = frame.drop(columns=[c for c in metadata.columns if c in frame and c not in IDENTITY])
    return frame.merge(metadata, on=IDENTITY, validate="many_to_one")


def weighted_metrics(frame, threshold):
    clean = frame.clean_pair_probability.to_numpy(float)
    triggered = frame.triggered_pair_probability.to_numpy(float)
    negative = clean < threshold
    flip = negative & (triggered >= threshold)
    scene_n = frame.groupby("scenario_id").scenario_id.transform("size").to_numpy(float)
    weights = 1.0 / scene_n
    total = float(weights.sum())
    return {
        "scenarios": int(frame.scenario_id.nunique()),
        "incremental_asr": float(np.dot(weights, flip) / total),
        "clean_activation": float(np.dot(weights, ~negative) / total),
        "absolute_asr": float(np.dot(weights, triggered >= threshold) / total),
        "conditional_flip": float(np.dot(weights, flip) / np.dot(weights, negative)) if negative.any() else None,
        "mean_probability_delta": float(np.dot(weights, triggered - clean) / total),
        "mean_logit_delta": float(np.dot(weights, frame.triggered_pair_mean_logit - frame.clean_pair_mean_logit) / total),
        "definition": "each scene weight=1; within scene each target weight=1/number_of_targets; conditional is weighted numerator/denominator",
    }


def collateral_summary(frame):
    result = {}
    for subset in response.COLLATERAL_SUBSETS:
        prefix = subset + "_"
        columns = [prefix + k for k in ("edges", "n00", "n01", "n10", "n11", "delta_sum")]
        if any(c not in frame for c in columns) or frame[columns].isna().any().any():
            result[subset] = None
            continue
        count = frame[prefix + "edges"].to_numpy(float)
        transitions = frame[[prefix + k for k in ("n00", "n01", "n10", "n11")]].to_numpy(float)
        if (not np.isfinite(frame[columns].to_numpy(float)).all() or (count < 0).any()
                or (transitions < 0).any() or not np.equal(count, np.floor(count)).all()
                or not np.equal(transitions, np.floor(transitions)).all()
                or not np.equal(transitions.sum(axis=1), count).all()):
            raise ValueError("invalid per-target collateral transitions")
        n = int(count.sum())
        valid = count > 0
        # Average rates within scenes, then across scenes with at least one eligible exposure.
        macro = {}
        for event, label in (("n01", "incremental_positive"), ("n10", "suppression")):
            local = frame.loc[valid, ["scenario_id"]].copy()
            local["rate"] = frame.loc[valid, prefix + event].to_numpy(float) / count[valid]
            macro[label + "_rate"] = float(local.groupby("scenario_id").rate.mean().mean()) if valid.any() else None
        result[subset] = {
            "directed_edge_trigger_opportunities": n,
            "transitions": {key: int(frame[prefix + key].sum()) for key in ("n00", "n01", "n10", "n11")},
            "edge_weighted_incremental_positive_rate": float(frame[prefix + "n01"].sum() / n) if n else None,
            "edge_weighted_suppression_rate": float(frame[prefix + "n10"].sum() / n) if n else None,
            "edge_weighted_mean_probability_delta": float(frame[prefix + "delta_sum"].sum() / n) if n else None,
            "equal_scene_mean_of_target_rates": macro,
            "scenes_with_eligible_opportunities": int(frame.loc[valid, "scenario_id"].nunique()),
        }
    return result


def summarize_group(frame, threshold):
    return {
        "targets": len(frame),
        "target_weighted": response.summarize_target_records(frame, threshold=threshold),
        "equal_scene_weighted": weighted_metrics(frame, threshold),
        "collateral": collateral_summary(frame),
    }


def score_batch(model, items, targets, *, role, threshold, device):
    if not items or len(items) != len(targets):
        raise ValueError("empty or incomplete batch")
    cl, tl, cp, tp = response._model_probabilities(model, items, device=device)
    expected_edges = sum(item.target_mask.size for item in items)
    if any(values.shape != (expected_edges,) or not np.isfinite(values).all()
           for values in (cl, tl, cp, tp)):
        raise ValueError("invalid model output shape or values")
    records = []
    offset = 0
    for item, row in zip(items, targets.itertuples(index=False)):
        if (item.scenario_id, item.src, item.dst) != (str(row.scenario_id), int(row.src), int(row.dst)):
            raise ValueError("batch target ordering differs")
        n = item.target_mask.size
        clean = cp[offset:offset+n]
        triggered = tp[offset:offset+n]
        local_cl = cl[offset:offset+n]
        local_tl = tl[offset:offset+n]
        mask = item.target_mask
        record = {
            "scenario_id": item.scenario_id, "src": item.src, "dst": item.dst,
            "motion_regime": item.motion_regime, "model_role": role,
            "clean_pair_probability": float(clean[mask].mean()),
            "triggered_pair_probability": float(triggered[mask].mean()),
            "clean_pair_mean_logit": float(local_cl[mask].mean()),
            "triggered_pair_mean_logit": float(local_tl[mask].mean()),
            "clean_negative_bce": float(F.softplus(torch.from_numpy(local_cl[mask])).mean()),
            "triggered_positive_bce": float(F.softplus(-torch.from_numpy(local_tl[mask])).mean()),
        }
        for subset, subset_mask in item.collateral_masks.items():
            accumulator = response.CollateralAccumulator()
            accumulator.add(clean, triggered, subset_mask, threshold=threshold)
            for field, value in {
                "edges": accumulator.count, "n00": accumulator.n00, "n01": accumulator.n01,
                "n10": accumulator.n10, "n11": accumulator.n11, "delta_sum": accumulator.delta_sum,
            }.items():
                record[subset + "_" + field] = value
        records.append(record)
        offset += n
    if len(records) != len(targets) or offset != len(cp):
        raise ValueError("incomplete batch accounting")
    return records


def load_items(targets, graph_rows, graph_dir):
    items = []
    for row in targets.itertuples(index=False):
        graph_row = graph_rows.get(str(row.scenario_id))
        if graph_row is None or graph_row.split != "train" or row.split != "train":
            raise ValueError("additional target escaped train graphs")
        if str(graph_row.graph_sha256) != str(row.graph_sha256):
            raise ValueError("target/graph SHA binding differs")
        items.append(response._load_audit_item(
            graph_row, row, graph_dir=graph_dir, label_config=RiskLabelConfig(),
        ))
    return items


def preflight():
    """Read asset bytes/metadata and cached scores only; no graph loading or inference."""
    print("[additional_audit] verifying completed run and epoch schedules", flush=True)
    bound = {"gate": fixed.bind_file(RUN/"audit/gate.json", GATE_SHA256)}
    gate = fixed.read_json(RUN/"audit/gate.json")
    if gate["status"] != "STOP" or gate["test_accessed"] or gate["attack_validation_started"]:
        raise ValueError("unexpected original gate/scope")
    for key, asset in gate["assets"].items():
        bound[key] = fixed.bind_file(Path(asset["path"]), asset["sha256"])
    if bound["result"]["sha256"] != RESULT_SHA256 or bound["plan"]["sha256"] != PLAN_SHA256:
        raise ValueError("audit is bound to a different training run")
    plan_path = Path(bound["plan"]["path"])
    roles, plan = rotation.checked_plan(plan_path)
    bound["rotation_targets"] = fixed.bind_file(
        plan_path.parent / plan["targets"]["filename"], plan["targets"]["sha256"],
    )
    result_path = Path(bound["result"]["path"])
    result = fixed.read_json(result_path)
    rotation.validate_candidate(result, plan_path, plan, roles)
    epoch = selected_epoch(result)
    if epoch != 37:
        raise ValueError("bound selected checkpoint epoch changed")
    groups = annotate_targets(roles, epoch)
    if {key:len(value) for key,value in groups.items()} != {
        "additional_train":2415, "rotatable_anchors":1037, "heldout":1037,
    }:
        raise ValueError("target groups differ from the completed plan")
    exposure = groups["additional_train"].v4_positive_exposures_to_checkpoint.value_counts().sort_index().to_dict()
    if exposure != {9:1839, 12:304, 18:272}:
        raise ValueError("checkpoint exposure counts differ")
    anchor = fixed.read_json(Path(bound["anchor_audit"]["path"]))
    fixed.validate_audit(anchor, result_path, result, plan["assets"])
    holdout = fixed.read_json(Path(bound["holdout_audit"]["path"]))
    bound["old_anchor_audit"] = fixed.bind_file(OLD_ANCHOR, OLD_ANCHOR_SHA256)
    old = fixed.read_json(OLD_ANCHOR)
    fixed_result = fixed.read_json(Path(plan["assets"]["fixed_v3_result"]["path"]))
    fixed.validate_audit(old, Path(plan["assets"]["fixed_v3_result"]["path"]), fixed_result, plan["assets"])
    reference = fixed.read_json(Path(plan["assets"]["reference_result"]["path"]))
    threshold = float(reference["val_pair_metrics"]["threshold"])
    if holdout["scope"] != {"split":"train", "validation_graphs_read":0, "test_graphs_read":0, "training":False}:
        raise ValueError("cached holdout audit has wrong scope")
    if holdout["threshold"] != threshold or holdout["targets_sha256"] != plan["targets"]["sha256"]:
        raise ValueError("cached holdout threshold/target binding differs")
    models = {
        "rotating_v4": {"result":bound["result"],
                        "checkpoint":fixed.bind_file(fixed.asset_path(result["checkpoint"]["path"]),result["checkpoint"]["sha256"])},
        "fixed_v3": {"result":plan["assets"]["fixed_v3_result"],"checkpoint":plan["assets"]["fixed_v3_checkpoint"]},
        "clean_reference": {"result":plan["assets"]["reference_result"],"checkpoint":plan["assets"]["reference_checkpoint"]},
    }
    if holdout["assets"] != models:
        raise ValueError("cached holdout scored different checkpoints")
    cached = {}
    for name, payload in (("anchors",anchor),("old_anchors",old),("heldout",holdout)):
        asset = payload["records"] if name == "heldout" else payload["assets"]["records"]
        bound[name+"_records"] = fixed.bind_file(Path(asset["path"]),asset["sha256"])
        cached[name] = pd.read_csv(asset["path"], dtype={"scenario_id":str})
    scene_ids = set(groups["rotatable_anchors"].scenario_id)
    current_anchor = cached["anchors"][cached["anchors"].scenario_id.isin(scene_ids)].copy()
    current_anchor["model_role"] = current_anchor.model_role.replace({fixed.ROLE:"rotating_v4"})
    baseline_anchor = cached["old_anchors"][
        cached["old_anchors"].scenario_id.isin(scene_ids) & cached["old_anchors"].model_role.eq(fixed.ROLE)
    ].copy()
    baseline_anchor["model_role"] = "fixed_v3"
    reused = {
        "rotatable_anchors": align_records(pd.concat([current_anchor,baseline_anchor]),groups["rotatable_anchors"]),
        "heldout": align_records(cached["heldout"],groups["heldout"]),
    }
    for role, data in reused["heldout"].groupby("model_role"):
        observed = response.summarize_target_records(data, threshold=threshold)
        expected = holdout["models"][role]["targets"]
        for metric in ("incremental_flip_rate_all_targets","clean_activation_rate","absolute_activation_rate"):
            if not np.isclose(observed[metric],expected[metric],rtol=0,atol=1e-12):
                raise ValueError("cached holdout scores do not reproduce their summary")
    manifest_path = Path(plan["assets"]["graph_manifest"]["path"])
    manifest = load_manifest(manifest_path.parent,manifest_path,require_graph_sha256=True)
    graph_rows = {str(r.scenario_id):r for r in manifest[manifest.split.eq("train")].itertuples(index=False)}
    for row in groups["additional_train"].itertuples(index=False):
        gr = graph_rows.get(str(row.scenario_id))
        if gr is None or gr.graph_sha256 != row.graph_sha256:
            raise ValueError("new target has no bound train graph")
    sources = {
        **plan["sources"],
        str(Path(__file__).resolve().relative_to(ROOT)):sha256_file(__file__),
    }
    bound.update({"plan_"+key:asset for key,asset in plan["assets"].items()})
    for role, assets in models.items():
        for kind, asset in assets.items():
            bound[role+"_"+kind] = fixed.bind_file(Path(asset["path"]),asset["sha256"])
    return {
        "plan":plan, "epoch":epoch, "groups":groups, "reused":reused,
        "models":models, "threshold":threshold, "bound":bound, "sources":sources,
        "holdout":holdout, "graph_rows":graph_rows, "graph_dir":manifest_path.parent,
    }


def new_output(requested):
    allowed = ROOT/"record/reproduction"
    if requested:
        output = Path(requested).resolve()
        if not output.is_relative_to(allowed):
            raise ValueError("audit output must be under record/reproduction")
        output.mkdir(parents=True,exist_ok=False)
        return output
    parent = RUN/"diagnostics"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="additional_targets_",dir=parent))


def verify_unchanged(context):
    for asset in context["bound"].values():
        fixed.bind_file(Path(asset["path"]),asset["sha256"])
    for name, digest in context["sources"].items():
        fixed.bind_file(ROOT/name,digest)


def build_summary(additional, context):
    all_groups = {"additional_train":additional, **context["reused"]}
    groups = {}
    comparison = []
    for name, frame in all_groups.items():
        groups[name] = {}
        for role, data in frame.groupby("model_role"):
            summary = summarize_group(data,context["threshold"])
            if name == "heldout":
                summary["cached_full_pool_collateral"] = context["holdout"]["models"][role]["collateral"]
                summary["collateral_note"] = "only cached edge-weighted full holdout statistics; no per-scene transitions available"
            elif name == "rotatable_anchors":
                summary["collateral_note"] = "rotatable subset collateral unavailable in cached anchor records; full-anchor statistics not substituted"
            groups[name][role] = summary
            comparison.append({
                "target_group":name,"model_role":role,"targets":len(data),
                "scenarios":int(data.scenario_id.nunique()),
                "target_weighted_incremental_asr":summary["target_weighted"]["incremental_flip_rate_all_targets"],
                "scene_weighted_incremental_asr":summary["equal_scene_weighted"]["incremental_asr"],
                "clean_activation":summary["target_weighted"]["clean_activation_rate"],
                "mean_probability_delta":summary["target_weighted"]["probability_delta"]["mean"],
                "mean_logit_delta":summary["target_weighted"]["logit_delta"]["mean"],
            })
    strata = {}
    for field in ("v4_positive_exposures_to_checkpoint","training_slots_in_scene"):
        strata[field] = {}
        for value, data in additional.groupby(field):
            strata[field][str(int(value))] = {
                role:summarize_group(rows,context["threshold"]) for role,rows in data.groupby("model_role")
            }
    return {
        "status":"AUDIT_COMPLETE_REVIEW_REQUIRED", "audit_version":VERSION,
        "scope":SCOPE, "selected_v4_checkpoint_epoch":context["epoch"],
        "threshold":context["threshold"],"model_roles":list(MODEL_ROLES),
        "new_metric_gate":None, "original_training_gate_status":"STOP",
        "groups":groups, "additional_target_strata":strata,
        "interpretation_limits":[
            "exposure strata refer to v4's selected epoch, not epoch50 or fixed_v3",
            "additional targets had zero positive-trigger target exposure in fixed_v3",
            "models are fixed single-seed checkpoints; these are descriptive comparisons",
            "target groups differ in training role; exposure is not randomized independently of scene complexity",
            "collateral edges are directed edge-trigger opportunities; graph edges may repeat across different target interventions",
            "no metric here changes the existing STOP gate or authorizes further training/validation",
        ],
    }, pd.DataFrame(comparison)


def run(context, *, device_name, output_dir=None):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly use --device cpu")
    output = new_output(output_dir)
    targets = context["groups"]["additional_train"]
    targets.to_csv(output/"targets.csv",index=False,mode="x")
    protocol = {
        "audit_version":VERSION,"scope":SCOPE,"selected_v4_checkpoint_epoch":context["epoch"],
        "targets":len(targets),"unique_scenarios":int(targets.scenario_id.nunique()),
        "threshold":context["threshold"],"batch_size":BATCH_SIZE,"device":str(device),
        "target_manifest":{"path":str(output/"targets.csv"),"sha256":sha256_file(output/"targets.csv")},
        "assets":context["bound"],"models":context["models"],"sources":context["sources"],
        "additional_target_selection":"all role=train and slot>0 in the bound plan, independent of model outputs",
        "v4_exposure_histogram":{str(k):int(v) for k,v in targets.v4_positive_exposures_to_checkpoint.value_counts().sort_index().items()},
        "completion":"summary.json is written only after all targets/models and integrity checks succeed",
    }
    rotation.write_json(output/"protocol.json",protocol)
    print(f"[additional_audit] output={output}; targets={len(targets)}; no training",flush=True)
    models = {}
    records = []
    for start in range(0,len(targets),BATCH_SIZE):
        chunk = targets.iloc[start:start+BATCH_SIZE]
        items = load_items(chunk,context["graph_rows"],context["graph_dir"])
        if not models:
            for role, assets in context["models"].items():
                result = fixed.read_json(Path(assets["result"]["path"]))
                models[role] = response._model_from_assets(
                    result,Path(assets["checkpoint"]["path"]),
                    node_dim=items[0].clean.x.shape[1],edge_dim=items[0].clean.edge_attr.shape[1],device=device,
                )
        for role, model in models.items():
            records.extend(score_batch(
                model,items,chunk,role=role,threshold=context["threshold"],device=device,
            ))
        print(f"[additional_audit] targets={min(start+BATCH_SIZE,len(targets))}/{len(targets)}",flush=True)
    additional = align_records(pd.DataFrame(records),targets)
    additional.to_csv(output/"additional.records.csv",index=False,mode="x")
    reused = pd.concat(list(context["reused"].values()),ignore_index=True)
    reused.to_csv(output/"reused.records.csv",index=False,mode="x")
    summary, comparison = build_summary(additional,context)
    comparison.to_csv(output/"comparison.csv",index=False,mode="x")
    verify_unchanged(context)
    summary["outputs"] = {
        name:{"path":str(output/name),"sha256":sha256_file(output/name)}
        for name in ("protocol.json","targets.csv","additional.records.csv","reused.records.csv","comparison.csv")
    }
    rotation.write_json(output/"summary.json",summary)
    print(comparison.to_string(index=False),flush=True)
    print(f"[additional_audit] AUDIT_COMPLETE_REVIEW_REQUIRED: {output/'summary.json'}",flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true",help="Verify metadata/assets/cached scores only; no graph loading or inference")
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--output-dir",help="New directory under record/reproduction")
    args = parser.parse_args(argv)
    context = preflight()
    if args.check_only:
        print(json.dumps({
            "status":"PREFLIGHT_PASS", "scope":SCOPE, "epoch":context["epoch"],
            "new_targets":len(context["groups"]["additional_train"]),
            "scenarios":int(context["groups"]["additional_train"].scenario_id.nunique()),
            "v4_exposure_histogram":{str(k):int(v) for k,v in context["groups"]["additional_train"].v4_positive_exposures_to_checkpoint.value_counts().sort_index().items()},
            "cached_groups":{k:len(v)//len(MODEL_ROLES) for k,v in context["reused"].items()},
        },indent=2))
        return
    run(context,device_name=args.device,output_dir=args.output_dir)


if __name__ == "__main__":
    main()
