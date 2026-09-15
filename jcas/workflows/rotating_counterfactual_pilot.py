"""Prepare/check/run one fixed moving-target-rotation pilot; never invokes attack validation."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from jcas.core.poison import eligible_negative_pair_groups, select_pair_orientation, sha256_file
from jcas.core.risk_labels import RiskLabelConfig, labels_for_graph
from jcas.core.target_rotation import (
    VERSION, SELECTION_SEED, MAX_TRAIN_TARGETS, MIN_HOLDOUT_SCENARIOS, MOVING,
    FIXED_FIELDS, HOLDOUT_GATES, pair_key, partition_candidates, validate_roles,
    load_plan, epoch_rows, schedule_digest,
)
from jcas.workflows import projected_counterfactual_pilot as fixed
from jcas.workflows.trainer import load_manifest, verified_graph_path

ROOT = fixed.ROOT
OUTPUT_ROOT = ROOT / "record/reproduction/rotating_counterfactual_v4"
FIXED_RESULT = ROOT / (
    "record/reproduction/projected_counterfactual_v3/seed20260621_20260907_114630/"
    "train/genconv_strict_seed20260621_projected_v3_t0p1_r0p1_c1_adj5_m1_rho1_20260907_133613/result.json"
)
FIXED_RESULT_HASH = "326465e7188c19af59cafca239e1ac2df318ac709ae299e02e8e62b60ea0ade6"
SEEN_TARGETS = ROOT / (
    "record/reproduction/projected_counterfactual_v3/"
    "train_generalization_diagnostic_9Gxk2nHi/targets.csv"
)
SEEN_HASH = "ff446455274038649830f0427e9bdaa275ed691c03951e4e6d44079105847b36"


def validate_rotation_arguments(args):
    expected = {
        "seed":20260621,"epochs":50,"batch_size":64,"hidden_dim":128,
        "num_layers":3,"dropout":.1,"norm":"layer","decoder_hidden_dim":128,
        "decoder_num_layers":2,"lr":.001,"weight_decay":.0001,
        "target_weight":.1,"ranking_weight":.1,"consistency_weight":1.,
        "adjacent_negative_consistency_weight":5.,"ranking_margin":1.,
        "max_gradient_norm":5.,"gradient_projection":"clean_reference",
        "max_extra_norm_ratio":1.,"max_train_graphs":None,"max_val_graphs":None,
    }
    for key, value in expected.items():
        if getattr(args,key,None) != value:
            raise ValueError(f"rotation pilot fixed argument differs: {key}")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def new_directory(prefix, requested=None):
    if requested:
        path = Path(requested).resolve()
        if not path.is_relative_to(ROOT / "record/reproduction"):
            raise ValueError("outputs must be under record/reproduction")
        path.mkdir(parents=True, exist_ok=False)
        return path
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=OUTPUT_ROOT))


def assets():
    result = fixed.preflight()
    result.pop("baseline_result")
    result["fixed_v3_result"] = fixed.bind_file(FIXED_RESULT, FIXED_RESULT_HASH)
    baseline = fixed.read_json(FIXED_RESULT)
    fixed.check_candidate(baseline, result)
    result["fixed_v3_checkpoint"] = fixed.bind_file(
        fixed.asset_path(baseline["checkpoint"]["path"]), baseline["checkpoint"]["sha256"]
    )
    result["previous_diagnostic_targets"] = fixed.bind_file(SEEN_TARGETS, SEEN_HASH)
    return result


def source_assets():
    names = (
        "jcas/core/target_rotation.py", "jcas/workflows/target_rotation_data.py",
        "jcas/workflows/rotating_counterfactual_pilot.py",
        "jcas/workflows/explicit_counterfactual_trainer.py",
        "jcas/workflows/projected_counterfactual_pilot.py",
        "jcas/workflows/paired_target_response_audit.py",
        "jcas/core/counterfactual_gradients.py", "jcas/core/trajectory_trigger.py",
        "jcas/core/motion_normalized_features.py", "jcas/core/risk_labels.py",
        "jcas/core/poison.py", "jcas/core/models.py", "jcas/workflows/trainer.py",
    )
    return {name: sha256_file(ROOT/name) for name in names}


def read_anchors(asset):
    return pd.read_csv(asset["path"], dtype={
        "scenario_id":str, "src_track_id":str, "dst_track_id":str,
    })


def prepare(output=None):
    bound = assets()
    anchors = read_anchors(bound["poison_manifest"])
    excluded = {pair_key(r) for r in pd.read_csv(SEEN_TARGETS).to_dict("records")}
    manifest_path = Path(bound["graph_manifest"]["path"])
    manifest = load_manifest(manifest_path.parent, manifest_path, require_graph_sha256=True)
    graph_rows = {str(r.scenario_id): r for r in manifest[manifest.split.eq("train")].itertuples(index=False)}
    records = []
    cfg = RiskLabelConfig()
    for i, original in enumerate(anchors.sort_values("scenario_id").to_dict("records"), 1):
        row = graph_rows[original["scenario_id"]]
        if row.split != "train":
            raise ValueError("preparation refuses non-train graphs")
        path = verified_graph_path(row, manifest_path.parent, require_graph_sha256=True)
        anchor = {
            **FIXED_FIELDS, **{k:original[k] for k in (
                "scenario_id","src","dst","src_track_id","dst_track_id","motion_regime","min_endpoint_speed_mps",
            )}, "graph_sha256":str(row.graph_sha256),
        }
        candidates = [anchor]
        # Slow fallback anchors stay exactly fixed. Only moving anchors need alternatives.
        if original["motion_regime"] == MOVING:
            with np.load(path, allow_pickle=True) as graph:
                bundle = labels_for_graph(graph, cfg)
                groups = eligible_negative_pair_groups(
                    graph, cfg, require_strict_label=True, perturb_window=10,
                    require_bi_endpoint_contiguous=True, label_bundle=bundle,
                )
                candidates = []
                for group in groups:
                    edge, _ = select_pair_orientation(graph, group, perturb_window=10)
                    src, dst = map(int, graph["edge_index"][:,edge])
                    speeds = np.linalg.norm(np.asarray(graph["observed_velocities_filled"],dtype=float)[[src,dst],-1],axis=1)
                    if float(speeds.min()) < .5:
                        continue
                    candidates.append({
                        **anchor, "src":src, "dst":dst,
                        "src_track_id":str(graph["node_track_ids"][src]),
                        "dst_track_id":str(graph["node_track_ids"][dst]),
                        "min_endpoint_speed_mps":float(speeds.min()),
                    })
                match = [r for r in candidates if pair_key(r) == pair_key(anchor)]
                if len(match) != 1 or (match[0]["src"],match[0]["dst"]) != (anchor["src"],anchor["dst"]):
                    raise ValueError("original pair no longer eligible with the same orientation")
        train, held = partition_candidates(anchor, candidates, excluded)
        for slot, target in enumerate(train):
            records.append({**target, "role":"train", "slot":slot})
        if held:
            records.append({**held, "role":"holdout", "slot":-1})
        if i % 250 == 0:
            print(f"[rotation_prepare] train_scenes={i}/{len(anchors)}", flush=True)
    frame = pd.DataFrame(records).sort_values(["scenario_id","role","slot"]).reset_index(drop=True)
    coverage = validate_roles(frame, anchors)
    output = new_directory("plan_", output)
    frame.to_csv(output/"targets.csv", index=False, mode="x")
    plan = {
        "version":VERSION, "selection_seed":SELECTION_SEED, "max_train_targets":MAX_TRAIN_TARGETS,
        "seed":20260621, "epochs":50, "test_access_authorized":False,
        "attack_validation_automatic":False, "training_access":"training_procedure_control",
        "assets":bound, "sources":source_assets(),
        "targets":{"filename":"targets.csv","sha256":sha256_file(output/"targets.csv")},
        "coverage":coverage, "minimum_holdout_scenarios":MIN_HOLDOUT_SCENARIOS,
        "original_anchor_gates":fixed.GATES, "holdout_gates":HOLDOUT_GATES,
        "status":"READY" if coverage["holdout_targets"] >= MIN_HOLDOUT_SCENARIOS else "STOP_INSUFFICIENT_COVERAGE",
        "rule":"epoch 1 anchor; then (epoch-1)%num_train_slots; one triggered row per original scene per epoch",
        "holdout_scope":"moving pairs in rotatable poison scenes; never trigger-positive in fixed-v3 or rotation; clean/non-target supervision retained",
        "holdout_exclusions":"original anchors and every pair in previous diagnostic targets CSV",
        "checkpoint_selection":"unchanged clean-validation loss; no holdout response during training",
        "graph_scope":{"train_graphs_verified":len(anchors),"validation_graphs_read":0,"test_graphs_read":0},
    }
    write_json(output/"plan.json", plan)
    print(f"[rotation_prepare] coverage={coverage}", flush=True)
    print(f"[rotation_prepare] plan={output/'plan.json'} status={plan['status']}", flush=True)
    return output/"plan.json"


def checked_plan(path):
    current = assets()
    anchors = read_anchors(current["poison_manifest"])
    frame, plan = load_plan(
        path, anchors=anchors, graph_manifest_hash=current["graph_manifest"]["sha256"],
        poison_hash=current["poison_manifest"]["sha256"],
    )
    if plan.get("sources") != source_assets():
        raise ValueError("code changed since preparation; prepare a new plan before training")
    if (plan["assets"] != current or plan.get("seed") != 20260621 or plan.get("epochs") != 50
            or plan.get("status") != "READY" or plan.get("attack_validation_automatic") is not False):
        raise ValueError("plan changed the fixed pilot assets/configuration")
    if plan.get("original_anchor_gates") != {k:list(v) for k,v in fixed.GATES.items()}:
        raise ValueError("original gates changed")
    return frame, plan


def training_command(plan_path, plan, output, device):
    return [
        sys.executable,"-m","jcas.workflows.explicit_counterfactual_trainer",
        "--graph-dir",str(Path(plan["assets"]["graph_manifest"]["path"]).parent),
        "--graph-manifest",plan["assets"]["graph_manifest"]["path"],
        "--poison-manifest",plan["assets"]["poison_manifest"]["path"],
        "--target-rotation-plan",str(Path(plan_path).resolve()),
        "--output-dir",str(output/"train"),"--seed","20260621",
        "--epochs","50","--batch-size","64","--hidden-dim","128","--num-layers","3",
        "--dropout","0.1","--norm","layer","--decoder-hidden-dim","128","--decoder-num-layers","2",
        "--lr","0.001","--weight-decay","0.0001","--target-weight","0.1","--ranking-weight","0.1",
        "--consistency-weight","1","--adjacent-negative-consistency-weight","5",
        "--ranking-margin","1","--max-gradient-norm","5",
        "--gradient-projection","clean_reference","--max-extra-norm-ratio","1","--device",device,
    ]


def validate_candidate(result, plan_path, plan, roles):
    # Reuse all old fixed hyperparameter/asset checks, but explicitly validate the
    # new presentation separately rather than passing it off as a fixed-v3 run.
    old_view = copy.deepcopy(result)
    old_view["config"].pop("target_rotation_plan", None)
    fixed.check_candidate(old_view, plan["assets"])
    binding = result.get("target_rotation") or {}
    if (binding.get("plan_sha256") != sha256_file(plan_path)
            or binding.get("targets_sha256") != plan["targets"]["sha256"]
            or binding.get("coverage") != plan["coverage"]):
        raise ValueError("candidate did not use this rotation plan")
    if result["config"].get("target_rotation_plan") != str(Path(plan_path).resolve()):
        raise ValueError("candidate rotation configuration differs")
    history = result["history"]
    if len(history) != 50 or [h["epoch"] for h in history] != list(range(1,51)):
        raise ValueError("rotation pilot training is incomplete")
    for entry in history:
        r = entry["train"].get("target_rotation", {})
        selected = epoch_rows(roles, entry["epoch"])
        if (r.get("schedule_sha256") != schedule_digest(selected)
                or r.get("triggered_rows") != 5776
                or entry["train"].get("counterfactual_pairs") != 5776
                or r.get("holdout_targets_presented") != 0
                or r.get("motion_counts") != {MOVING:2746,"slow_lt_0p5":3030}):
            raise ValueError("per-epoch target schedule/exposure differs")
    for name in ("workflow_sha256","projection_module_sha256"):
        path = result["source"]["workflow_path"] if name == "workflow_sha256" else ROOT/"jcas/core/counterfactual_gradients.py"
        if sha256_file(path) != result["source"][name]:
            raise ValueError("training source changed before audit")


def holdout_gate_values(rotating, baseline):
    current = rotating["targets"]["incremental_flip_rate_all_targets"]
    reference = baseline["targets"]["incremental_flip_rate_all_targets"]
    finite = lambda v: isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    return {
        "holdout_incremental_asr":current,
        "holdout_lift_over_fixed_v3":current-reference if finite(current) and finite(reference) else None,
        "holdout_clean_activation":rotating["targets"]["clean_activation_rate"],
        "holdout_adjacent_incremental_fp":rotating["collateral"]["adjacent_negative"].get("incremental_positive_rate"),
    }


def combined_gate(result, anchor_audit, holdout):
    gate = fixed.evaluate_gate(result, anchor_audit)
    values = holdout_gate_values(holdout["models"]["rotating_v4"], holdout["models"]["fixed_v3"])
    for name, (operator, threshold) in HOLDOUT_GATES.items():
        value = values[name]
        finite = isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        passed = finite and (value >= threshold if operator == ">=" else value <= threshold)
        gate["checks"][name] = {
            "value":value if finite else None,"operator":operator,"threshold":threshold,"passed":bool(passed),
        }
    gate["status"] = "PASS" if all(c["passed"] for c in gate["checks"].values()) else "STOP"
    gate["interpretation"] = "fixed-anchor response + held-out target generalization + existing clean-validation AUC"
    return gate


def audit_holdout(roles, plan, result_path, output, device):
    from jcas.workflows.paired_target_response_audit import (
        _load_audit_item, _model_from_assets, _append_batch_records,
        summarize_target_records, CollateralAccumulator, COLLATERAL_SUBSETS, REGIMES,
    )
    held = roles[roles.role.eq("holdout")].sort_values("scenario_id")
    graph_path = Path(plan["assets"]["graph_manifest"]["path"])
    manifest = load_manifest(graph_path.parent, graph_path, require_graph_sha256=True)
    graph_rows = {str(r.scenario_id):r for r in manifest[manifest.split.eq("train")].itertuples(index=False)}
    reference = fixed.read_json(Path(plan["assets"]["reference_result"]["path"]))
    threshold = float(reference["val_pair_metrics"]["threshold"])
    paths = {"rotating_v4":Path(result_path),"fixed_v3":FIXED_RESULT,
             "clean_reference":Path(plan["assets"]["reference_result"]["path"])}
    expected_hashes = {
        "rotating_v4":sha256_file(result_path),
        "fixed_v3":plan["assets"]["fixed_v3_result"]["sha256"],
        "clean_reference":plan["assets"]["reference_result"]["sha256"],
    }
    device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    models = {}
    model_assets = {}
    records = []
    collateral = {
        role:{regime:{name:CollateralAccumulator() for name in COLLATERAL_SUBSETS}
              for regime in ("all",*REGIMES)} for role in paths
    }
    for start in range(0,len(held),64):
        items = []
        for row in held.iloc[start:start+64].itertuples(index=False):
            graph_row = graph_rows[str(row.scenario_id)]
            if graph_row.split != "train" or graph_row.graph_sha256 != row.graph_sha256:
                raise ValueError("holdout graph scope or hash differs")
            items.append(_load_audit_item(graph_row,row,graph_dir=graph_path.parent,label_config=RiskLabelConfig()))
        if not models:
            for role,path in paths.items():
                fixed.bind_file(path,expected_hashes[role])
                result = fixed.read_json(path)
                checkpoint = fixed.asset_path(result["checkpoint"]["path"])
                model_assets[role] = {
                    "result":fixed.bind_file(path,expected_hashes[role]),
                    "checkpoint":fixed.bind_file(checkpoint,result["checkpoint"]["sha256"]),
                }
                models[role] = _model_from_assets(
                    result,checkpoint,node_dim=items[0].clean.x.shape[1],
                    edge_dim=items[0].clean.edge_attr.shape[1],device=device,
                )
        for role, model in models.items():
            _append_batch_records(
                items,model_role=role,model=model,device=device,threshold=threshold,
                records=records,collateral=collateral,
            )
        print(f"[rotation_holdout] targets={min(start+64,len(held))}/{len(held)}",flush=True)
    frame = pd.DataFrame(records)
    if len(frame) != len(held)*3:
        raise ValueError("incomplete holdout audit")
    for role, data in frame.groupby("model_role"):
        if (len(data) != len(held) or data.scenario_id.duplicated().any()
                or set(data.scenario_id) != set(held.scenario_id)):
            raise ValueError(f"holdout pool differs across models: {role}")
    for role, asset in model_assets.items():
        fixed.bind_file(Path(asset["result"]["path"]),asset["result"]["sha256"])
        fixed.bind_file(Path(asset["checkpoint"]["path"]),asset["checkpoint"]["sha256"])
    frame.to_csv(output/"holdout.records.csv",index=False,mode="x")
    payload = {
        "scope":{"split":"train","validation_graphs_read":0,"test_graphs_read":0,"training":False},
        "targets":len(held),"targets_sha256":plan["targets"]["sha256"],"threshold":threshold,
        "assets":model_assets,
        "records":{"path":str(output/"holdout.records.csv"),"sha256":sha256_file(output/"holdout.records.csv")},
        "models":{role:{
            "targets":summarize_target_records(frame[frame.model_role.eq(role)],threshold=threshold),
            "collateral":{name:collateral[role]["all"][name].summary() for name in COLLATERAL_SUBSETS},
        } for role in paths},
    }
    write_json(output/"holdout.json",payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true",help="Read train graphs, freeze roles; no inference or training")
    parser.add_argument("--plan",help="Prepared plan.json")
    parser.add_argument("--check-only",action="store_true",help="Read metadata/assets only; no graph loading")
    parser.add_argument("--result",help="Resume audits for this plan's completed training result")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--output-dir",help="New directory under record/reproduction")
    args = parser.parse_args()
    if args.prepare_only:
        if args.plan or args.result or args.check_only:
            parser.error("--prepare-only cannot be combined with plan/result/check-only")
        prepare(args.output_dir)
        return
    if not args.plan:
        parser.error("--plan is required unless --prepare-only")
    roles, plan = checked_plan(args.plan)
    if args.check_only:
        if args.result:
            validate_candidate(fixed.read_json(Path(args.result)),args.plan,plan,roles)
        print(json.dumps({"status":"PREFLIGHT_PASS","coverage":plan["coverage"],
                          "anchor_gates":fixed.GATES,"holdout_gates":HOLDOUT_GATES},indent=2))
        print("[rotation] no graphs loaded; no training or inference started")
        return
    if args.result:
        validate_candidate(fixed.read_json(Path(args.result)),args.plan,plan,roles)
    output = new_directory("seed20260621_",args.output_dir)
    write_json(output/"protocol.json",{
        "plan":{"path":str(Path(args.plan).resolve()),"sha256":sha256_file(args.plan)},
        "sources":source_assets(),"training_started":not bool(args.result),
        "attack_validation_automatic":False,"test_access_authorized":False,
    })
    if args.result:
        result_path = Path(args.result).resolve()
    else:
        fixed.run_command(training_command(args.plan,plan,output,args.device),output/"train.log")
        paths = list((output/"train").glob("*/result.json"))
        if len(paths) != 1:
            raise RuntimeError("expected exactly one completed training result")
        result_path = paths[0]
    roles, plan = checked_plan(args.plan)
    result = fixed.read_json(result_path)
    validate_candidate(result,args.plan,plan,roles)
    audit_dir = output/"audit"
    audit_dir.mkdir()
    anchor_path = audit_dir/"anchors.json"
    a = plan["assets"]
    fixed.run_command([
        sys.executable,"-m","jcas.workflows.paired_target_response_audit",
        "--graph-dir",str(Path(a["graph_manifest"]["path"]).parent),
        "--graph-manifest",a["graph_manifest"]["path"],"--poison-manifest",a["poison_manifest"]["path"],
        "--candidate-result",str(result_path),"--candidate-checkpoint",str(fixed.asset_path(result["checkpoint"]["path"])),
        "--candidate-role",fixed.ROLE,"--reference-result",a["reference_result"]["path"],
        "--reference-checkpoint",a["reference_checkpoint"]["path"],
        "--batch-size","64","--device",args.device,"--output",str(anchor_path),
    ],output/"anchors.log")
    anchor = fixed.read_json(anchor_path)
    fixed.validate_audit(anchor,result_path,result,a)
    held = audit_holdout(roles,plan,result_path,audit_dir,args.device)
    checked_plan(args.plan)
    if held["assets"]["rotating_v4"]["result"]["sha256"] != anchor["assets"][fixed.ROLE]["result_sha256"]:
        raise ValueError("anchor and holdout audits evaluated different candidates")
    gate = combined_gate(result,anchor,held)
    gate["assets"] = {
        "result":fixed.bind_file(result_path,sha256_file(result_path)),
        "anchor_audit":fixed.bind_file(anchor_path,sha256_file(anchor_path)),
        "holdout_audit":fixed.bind_file(audit_dir/"holdout.json",sha256_file(audit_dir/"holdout.json")),
        "plan":fixed.bind_file(Path(args.plan).resolve(),sha256_file(args.plan)),
    }
    write_json(audit_dir/"gate.json",gate)
    lines = [f"{'PASS' if c['passed'] else 'FAIL'} {k}: {c['value']} {c['operator']} {c['threshold']}"
             for k,c in gate["checks"].items()]
    lines += [gate["status"],"No attack validation or test was started."]
    with (audit_dir/"gate.txt").open("x") as stream:
        stream.write("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    print(f"[rotation] output={output}",flush=True)
    if gate["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
