"""Independent train-only BCE mechanism pilot, not the full learned attack.

Three from-scratch models: clean, fixed random template, geometry-controlled
same template. All use ordinary BCE. Original val/test arrays are never loaded.
The selected template is fixed by protocol, NOT by stage-1 geometric results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from jcas import PROJECT_ROOT
from jcas.core import relational_trigger as trigger
from jcas.core.models import build_model
from jcas.core.motion_normalized_features import edge_features_for_mode, MOTION_NORMALIZED_EDGE_FEATURE_MODE
from jcas.core.poison import eligible_negative_pair_groups, sha256_file
from jcas.core.risk_labels import RiskLabelConfig, labels_for_graph, selected_label_computable_mask
from jcas.workflows import trainer
from jcas.workflows import relational_poison_prepare as stage1

ROOT = PROJECT_ROOT
VERSION = "relational_bce_mechanism_pilot_v1"
STAGE1_PLAN = "record/reproduction/relational_poison_v1/plan_20260912_v1/plan.json"
STAGE1_PLAN_SHA = "6175fd05a9c175242b6b5903343eea5fb8dcadfcda6c3bd8447849646d0e18d5"
STAGE1_SUMMARY = "record/reproduction/relational_poison_v1/stage1__dx4w2s1/summary.json"
STAGE1_SUMMARY_SHA = "dccfa0e30d9a8b9000bef0ba07aa6672f7cc4f1b8cb3f413d2fb07a2feb277b3"
OUTPUT = "record/reproduction/relational_poison_v1/bce_pilot"
ARMS = ("clean_reference", "random_shared", "geometry_shared")
VARIANTS = ("full", "single_canonical_first", "single_canonical_second", "relation_reversed")
COLLATERAL = ("adjacent_negative", "adjacent_positive", "nonincident_negative")
TEMPLATE_NAME = "random_20260912"
SEED = 20260621
EPOCHS = 50
BATCH_SIZE = 64


def safe_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()) or "test" in path.relative_to(ROOT.resolve()).parts:
        raise ValueError(f"path outside development scope: {path}")
    return path


def bind(value, expected=None):
    path = safe_path(value)
    sha = sha256_file(path)
    if expected is not None and sha != expected:
        raise ValueError(f"SHA mismatch: {path}")
    return {"path": str(path), "sha256": sha}


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def new_directory(value=None, *, prefix="run_"):
    root = ROOT.resolve() / OUTPUT
    if root.resolve() != root:
        raise ValueError("output root may not redirect through a symlink")
    if value is None:
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    path = safe_path(value)
    if not path.is_relative_to(root) or path == root:
        raise ValueError("output must be a new child of relational_poison_v1/bce_pilot")
    path.mkdir(parents=True, exist_ok=False)
    return path


def protocol():
    return {"version": VERSION, "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "arms": list(ARMS), "template_name": TEMPLATE_NAME,
            "template": trigger.template_bank()[TEMPLATE_NAME],
            "objective": "ordinary_BCE_on_supervised_directed_edges", "model": "genconv",
            "lr": .001, "weight_decay": .0001, "checkpoint_metric": "inner_develop_clean_loss",
            "threshold_source": "new_clean_reference_inner_develop_pair_F1",
            "training_groups": "construct", "checkpoint_groups": "develop", "response_groups": "audit",
            "original_validation_access": False, "test_access": False,
            "planned_controls_used_in_both_poison_arms": True,
            "template_optimized": False, "predictive_controller_optimized": False,
            "full_module_C_implemented": False,
            "physical_road_feasibility_certified": False,
            "purpose": "matched ordinary-training response and group-heldout mechanism comparison",
            "gate_scope": "new train-inner audit; not the public validation pool or a formal test",
            "gates": {"moving_asr_min": .30, "moving_lift_min": .10, "clean_activation_max": .03,
                      "adjacent_fp_max": .015, "moving_adjacent_fp_max": .005,
                      "inner_audit_auc_min": .985, "auc_drop_max": .0015}}


def preflight():
    """Verify the recorded stage-1 sources, not its evolving source discovery.

    Adding this separate module must not require rewriting a completed stage-1
    plan/source hash. No graph, checkpoint, or original validation is opened.
    """
    assets = {"stage1_plan": bind(STAGE1_PLAN, STAGE1_PLAN_SHA),
              "stage1_summary": bind(STAGE1_SUMMARY, STAGE1_SUMMARY_SHA)}
    plan = read_json(assets["stage1_plan"]["path"])
    summary = read_json(assets["stage1_summary"]["path"])
    if (summary["status"] != "FEASIBILITY_COMPLETE_REVIEW_REQUIRED" or summary["records_count"] != 2560
            or not summary["numerical_budget_pass"] or summary["test_accessed"]
            or summary["plan"] != assets["stage1_plan"]):
        raise ValueError("stage-1 completion contract differs")
    for key in ("records", "summary_csv"):
        assets["stage1_"+key] = bind(summary[key]["path"], summary[key]["sha256"])
    for key, asset in plan["assets"].items():
        assets[key] = bind(asset["path"], asset["sha256"])
    for key, asset in plan["files"].items():
        assets[key] = bind(asset["path"], asset["sha256"])
    sources = {name: bind(name, sha)["sha256"] for name, sha in plan["sources"].items()}
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = bind(__file__)["sha256"]
    manifest = Path(assets["graph_manifest"]["path"])
    trainer.validate_graph_manifest_contract(manifest)
    graphs = trainer.load_manifest(manifest.parent, manifest, require_graph_sha256=True)
    roles = pd.read_csv(assets["inner_train_roles.csv"]["path"], dtype={"scenario_id": str, "split_group_id": str})
    pd.testing.assert_frame_equal(roles, stage1.group_roles(graphs), check_dtype=False)
    targets = pd.read_csv(assets["planned_targets.csv"]["path"], dtype={"scenario_id": str, "split_group_id": str,
                                                                  "src_track_id": str, "dst_track_id": str})
    selected = pd.read_csv(assets["random_pairs"]["path"], dtype={"scenario_id": str, "src_track_id": str, "dst_track_id": str})
    expected, _ = stage1.planned_rows(selected, graphs)
    pd.testing.assert_frame_equal(targets, expected, check_dtype=False)
    train = graphs.loc[graphs.split.eq("train")].merge(roles[["scenario_id", "inner_role"]], on="scenario_id", validate="one_to_one")
    if len(train) != len(roles) or train.groupby("split_group_id").inner_role.nunique().gt(1).any():
        raise ValueError("inner group leakage")
    if set(train.inner_role) != {"construct", "develop", "audit"}:
        raise ValueError("missing inner partition")
    if plan["protocol"]["templates"][TEMPLATE_NAME] != protocol()["template"]:
        raise ValueError("fixed template changed")
    for role in ("construct", "develop", "audit"):
        if set(targets.loc[targets.inner_role.eq(role), "motion_regime"]) != set(stage1.REGIMES):
            raise ValueError(f"missing target regime in {role}")
    return {"assets": assets, "sources": sources, "protocol": protocol(), "train_rows": train,
            "targets": targets, "graph_dir": manifest.parent}


def graph_path(context, row, *, allowed_role):
    if row.split != "train" or row.inner_role != allowed_role:
        raise ValueError("graph row outside requested train-inner role")
    path = safe_path(context["graph_dir"] / str(row.graph_path))
    if not path.is_relative_to(context["graph_dir"].resolve()) or path.suffix != ".npz":
        raise ValueError("graph path escaped original asset root")
    bind(path, str(row.graph_sha256))
    return path


def clean_data(graph, scenario):
    labels = labels_for_graph(graph, RiskLabelConfig())
    supervised = trainer.supervision_mask(graph, True)
    supervised &= selected_label_computable_mask(graph, RiskLabelConfig(), labels)
    edge_attr = edge_features_for_mode(graph["edge_attr"], graph["edge_index"], graph["observed_positions_filled"],
                                     graph["observed_velocities_filled"], graph["observed_valid_mask"],
                                     mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE)
    data = Data(x=torch.tensor(np.asarray(graph["x_node"]), dtype=torch.float32),
                edge_index=torch.tensor(np.asarray(graph["edge_index"]), dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                edge_label=torch.tensor(labels["edge_label"], dtype=torch.float32),
                supervision_edge_mask=torch.tensor(supervised, dtype=torch.bool),
                message_passing_edge_mask=torch.tensor(trainer.message_passing_mask(graph), dtype=torch.bool),
                scenario_id=str(scenario))
    if not torch.isfinite(data.x).all() or not torch.isfinite(data.edge_attr).all():
        raise ValueError("non-finite clean features")
    return data, labels


def checked_target(graph, target, labels):
    src, dst = int(target["src"]), int(target["dst"])
    if str(graph["node_track_ids"][src]) != str(target["src_track_id"]) or str(graph["node_track_ids"][dst]) != str(target["dst_track_id"]):
        raise ValueError("target track identity mismatch")
    edges = np.asarray(graph["edge_index"])
    groups = eligible_negative_pair_groups(graph, RiskLabelConfig(), require_strict_label=True, perturb_window=10,
                                          require_bi_endpoint_contiguous=True, label_bundle=labels)
    matched = [g for g in groups if {int(edges[0, g[0]]), int(edges[1, g[0]])} == {src, dst}]
    if len(matched) != 1 or len(matched[0]) != 2:
        raise ValueError("target no longer an eligible symmetric negative pair")
    speed = np.linalg.norm(np.asarray(graph["observed_velocities_filled"])[[src, dst], -1], axis=1).min()
    if ("slow_lt_0p5" if speed < .5 else "moving_ge_0p5") != target["motion_regime"]:
        raise ValueError("target motion regime changed")
    return np.asarray(matched[0], dtype=int)


def transformed_data(graph, clean, target, target_ids, *, controller, variant, poison_label):
    result = trigger.apply_relational_trigger(graph, src=int(target["src"]), dst=int(target["dst"]),
        template=protocol()["template"], controller=controller, variant=variant)
    data = clean.clone()
    data.x = torch.tensor(result["x_node"], dtype=torch.float32)
    data.edge_attr = torch.tensor(result["edge_attr"], dtype=torch.float32)
    if poison_label:
        if variant != "full" or target["planned_role"] != "positive":
            raise ValueError("only complete positive-role samples can receive poisoned labels")
        data.edge_label[target_ids] = 1.
    return data


def load_role(context, role, *, replacements=False):
    rows = context["train_rows"].loc[context["train_rows"].inner_role.eq(role)].sort_values("scenario_id")
    targets = {r["scenario_id"]: r for r in context["targets"].loc[context["targets"].inner_role.eq(role)].to_dict("records")}
    clean_graphs, variants = [], {"random_shared": {}, "geometry_shared": {}}
    counts = {"considered": len(rows), "used": 0, "unsupervised": 0, "positive": 0, "negative_control": 0}
    if replacements and role != "construct":
        raise ValueError("poison replacements are construction-only")
    for index, row in enumerate(rows.itertuples(index=False), 1):
        path = graph_path(context, row, allowed_role=role)
        with np.load(path, allow_pickle=True) as graph:
            clean, labels = clean_data(graph, row.scenario_id)
            target = targets.get(row.scenario_id)
            if not clean.supervision_edge_mask.any():
                if target is not None:
                    raise ValueError("planned target scene has no supervision")
                counts["unsupervised"] += 1
                continue
            clean_graphs.append(clean)
            if replacements and target is not None:
                ids = checked_target(graph, target, labels)
                positive = target["planned_role"] == "positive"
                counts[target["planned_role"]] += 1
                for arm in variants:
                    variants[arm][row.scenario_id] = transformed_data(graph, clean, target, ids,
                        controller=arm == "geometry_shared", variant=target["planned_variant"], poison_label=positive)
        counts["used"] += 1
        if index == 1 or index % 500 == 0 or index == len(rows):
            print(f"[relational-bce] loading {role} {index}/{len(rows)}", flush=True)
    if not clean_graphs:
        raise ValueError("empty supervised role")
    if replacements and any(len(value) != len(targets) for value in variants.values()):
        raise ValueError("not all planned construction targets transformed")
    return clean_graphs, variants, counts


def initialization_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_arm(arm, data, develop, directory, device):
    directory.mkdir(exist_ok=False)
    trainer.set_seed(SEED)
    model = build_model("genconv", node_dim=data[0].x.shape[1], edge_dim=42).to(device)
    initial = initialization_sha(model)
    optimizer, optimizer_config = trainer.build_optimizer(model, lr=.001, weight_decay=.0001)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    dev_loader = DataLoader(develop, batch_size=BATCH_SIZE, shuffle=False)
    best, best_epoch = math.inf, None
    checkpoint = directory / "best_model.pt"
    with (directory / "history.jsonl").open("x") as stream:
        for epoch in range(1, EPOCHS+1):
            train_loss = trainer.train_one_epoch(model, loader, optimizer, device)
            dev_loss, _, _, pair_prob, pair_y = trainer.evaluate(model, dev_loader, device)
            if not np.isfinite(train_loss) or not np.isfinite(dev_loss) or len(np.unique(pair_y)) != 2:
                raise ValueError("invalid training/develop metrics")
            record = {"epoch": epoch, "train_loss": train_loss, "inner_develop_clean_loss": dev_loss,
                      "inner_develop_pair_auc": trainer.metric_dict(pair_y, pair_prob, .5)["auc"]}
            stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            if dev_loss < best:
                best, best_epoch = dev_loss, epoch
                # This checkpoint belongs to this freshly created run only.
                temporary = directory / "best_model.pending.pt"
                torch.save(model.state_dict(), temporary)
                temporary.replace(checkpoint)
            print(f"[relational-bce] {arm} epoch={epoch}/{EPOCHS} dev_loss={dev_loss:.6f}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    dev_loss, _, _, pair_prob, pair_y = trainer.evaluate(model, dev_loader, device)
    threshold, _ = trainer.best_threshold(pair_y, pair_prob)
    score_path = directory / "inner_develop_clean_pairs.csv"
    with score_path.open("x") as stream:
        # Serialize the exact float32 scores as float64 decimal values, so the
        # common threshold survives CSV round-tripping without rounding drift.
        pd.DataFrame({"label": pair_y, "probability": pair_prob.astype(np.float64)}).to_csv(stream, index=False)
    result = {"version": VERSION, "arm": arm, "protocol": protocol(), "initialization_sha256": initial,
              "epochs_completed": EPOCHS, "best_epoch": best_epoch, "model_config": asdict(model.cfg),
              "optimizer": optimizer_config, "checkpoint": bind(checkpoint), "history": bind(directory / "history.jsonl"),
              "inner_develop_clean_pairs": bind(score_path),
              "inner_develop_pair_metrics": trainer.metric_dict(pair_y, pair_prob, threshold),
              "threshold_eligible_as_common": arm == "clean_reference", "original_validation_accessed": False,
              "test_accessed": False, "training_access": "ordinary_BCE_data_only_mechanism_control"}
    write_json(directory / "result.json", result)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bind(directory / "result.json")


def train_models(context, directory, device):
    clean, replacements, construct_counts = load_role(context, "construct", replacements=True)
    develop, _, develop_counts = load_role(context, "develop")
    results = {}
    for arm in ARMS:
        data = clean if arm == "clean_reference" else [replacements[arm].get(g.scenario_id, g) for g in clean]
        results[arm] = train_arm(arm, data, develop, directory / arm, device)
    if len({read_json(asset["path"])["initialization_sha256"] for asset in results.values()}) != 1:
        raise ValueError("paired arm initializations differ")
    record = {"version": VERSION, "protocol": context["protocol"], "assets": context["assets"],
              "sources": context["sources"], "results": results, "construct_counts": construct_counts,
              "develop_counts": develop_counts, "inner_audit_graphs_opened": False,
              "original_validation_accessed": False, "test_accessed": False}
    write_json(directory / "training.json", record)
    return directory / "training.json"


def checked_training(path, context):
    value = read_json(safe_path(path))
    for key in ("version", "protocol", "assets", "sources"):
        expected = VERSION if key == "version" else context[key]
        if value[key] != expected:
            raise ValueError(f"training contract differs: {key}")
    if set(value["results"]) != set(ARMS):
        raise ValueError("missing paired training arm")
    results = {}
    for arm, asset in value["results"].items():
        bind(asset["path"], asset["sha256"])
        result = read_json(asset["path"])
        if result["arm"] != arm or result["protocol"] != context["protocol"] or result["epochs_completed"] != EPOCHS:
            raise ValueError("result arm/training protocol mismatch")
        for key in ("checkpoint", "history", "inner_develop_clean_pairs"):
            bind(result[key]["path"], result[key]["sha256"])
        results[arm] = result
    if len({r["initialization_sha256"] for r in results.values()}) != 1:
        raise ValueError("initialization mismatch")
    reference = results["clean_reference"]
    pairs = pd.read_csv(reference["inner_develop_clean_pairs"]["path"], float_precision="round_trip")
    threshold, _ = trainer.best_threshold(pairs.label.to_numpy(), pairs.probability.to_numpy())
    if not np.isclose(threshold, reference["inner_develop_pair_metrics"]["threshold"], atol=1e-12, rtol=0):
        raise ValueError("reference threshold cannot be recomputed")
    return results, float(threshold)


def predict(model, data, device):
    batch = Batch.from_data_list(data).to(device)
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.message_passing_edge_mask)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite model response")
    return np.split(probabilities, np.cumsum([g.num_edges for g in data])[:-1])


def response_record(target, clean_data_item, target_ids, before, after, threshold):
    labels = clean_data_item.edge_label.numpy()
    supervised = clean_data_item.supervision_edge_mask.numpy()
    edges = clean_data_item.edge_index.numpy()
    target_mask = np.zeros(len(labels), dtype=bool)
    target_mask[target_ids] = True
    incident = np.isin(edges, [int(target["src"]), int(target["dst"])]).any(axis=0)
    masks = {"adjacent_negative": supervised & incident & ~target_mask & (labels == 0),
             "adjacent_positive": supervised & incident & ~target_mask & (labels == 1),
             "nonincident_negative": supervised & ~incident & (labels == 0)}
    result = {"scenario_id": target["scenario_id"], "split_group_id": target["split_group_id"],
              "motion_regime": target["motion_regime"], "clean_probability": float(before[target_ids].mean()),
              "triggered_probability": float(after[target_ids].mean())}
    for name, mask in masks.items():
        a, b = before[mask] >= threshold, after[mask] >= threshold
        result[name+"_edges"] = int(mask.sum())
        for key, value in (("n00", ~a & ~b), ("n01", ~a & b), ("n10", a & ~b), ("n11", a & b)):
            result[name+"_"+key] = int(value.sum())
    return result


def summarize(records, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid threshold")
    frame = pd.DataFrame(records)
    keys = ["model", "controller", "variant", "scenario_id"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("empty or duplicate response records")
    conditions = ["model", "controller", "variant"]
    pools = list(frame.groupby(conditions))
    identity = ["scenario_id", "split_group_id", "motion_regime"]
    reference_pool = pools[0][1][identity].sort_values("scenario_id").reset_index(drop=True)
    for _, group in pools:
        pool = group[identity].sort_values("scenario_id").reset_index(drop=True)
        if not pool.equals(reference_pool):
            raise ValueError("response conditions use different target pools")
    if not set(frame.motion_regime).issubset(set(stage1.REGIMES)):
        raise ValueError("unknown response motion regime")
    for name in COLLATERAL:
        columns = [name+"_"+x for x in ("edges", "n00", "n01", "n10", "n11")]
        counts = frame[columns].to_numpy(dtype=float)
        if (not np.isfinite(counts).all() or (counts < 0).any()
                or (counts != np.floor(counts)).any()
                or (counts[:, 0] != counts[:, 1:].sum(axis=1)).any()):
            raise ValueError("invalid per-target collateral counts")
    rows = []
    for (model, controller, variant), group in frame.groupby(["model", "controller", "variant"]):
        for regime in ("all", *stage1.REGIMES):
            data = group if regime == "all" else group[group.motion_regime.eq(regime)]
            if data.empty:
                raise ValueError("missing response motion regime")
            before = data.clean_probability.to_numpy()
            after = data.triggered_probability.to_numpy()
            if not np.isfinite(before).all() or not np.isfinite(after).all() or min(before.min(), after.min()) < 0 or max(before.max(), after.max()) > 1:
                raise ValueError("invalid probabilities")
            negative, positive = before < threshold, after >= threshold
            flips = int((negative & positive).sum())
            result = {"model": model, "controller": bool(controller), "variant": variant, "motion_regime": regime,
                      "targets": len(data), "threshold": threshold, "incremental_flips": flips,
                      "incremental_asr": flips/len(data), "absolute_asr": float(positive.mean()),
                      "conditional_flip": flips/int(negative.sum()) if negative.any() else None,
                      "clean_activation": float((~negative).mean()), "probability_delta": float((after-before).mean())}
            for name in COLLATERAL:
                n = int(data[name+"_edges"].sum())
                counts = {key: int(data[name+"_"+key].sum()) for key in ("n00", "n01", "n10", "n11")}
                if sum(counts.values()) != n or min(counts.values()) < 0:
                    raise ValueError("collateral count mismatch")
                result[name+"_edges"] = n
                result[name+"_incremental_fp"] = counts["n01"]/n if n else None
                result[name+"_suppression"] = counts["n10"]/n if n else None
            rows.append(result)
    return rows


def comparison_gate(rows, aucs):
    def item(model, controller, regime):
        found = [r for r in rows if r["model"] == model and r["controller"] == controller and r["variant"] == "full" and r["motion_regime"] == regime]
        if len(found) != 1:
            raise ValueError("missing or duplicate comparison group")
        return found[0]
    candidate, random = item("geometry_shared", True, "all"), item("random_shared", False, "all")
    moving, random_moving = item("geometry_shared", True, "moving_ge_0p5"), item("random_shared", False, "moving_ge_0p5")
    values = {"moving_asr": moving["incremental_asr"], "moving_lift": moving["incremental_asr"]-random_moving["incremental_asr"],
              "overall_lift": candidate["incremental_asr"]-random["incremental_asr"],
              "clean_activation": candidate["clean_activation"], "adjacent_fp": candidate["adjacent_negative_incremental_fp"],
              "moving_adjacent_fp": moving["adjacent_negative_incremental_fp"], "inner_audit_auc": aucs["geometry_shared"],
              "auc_drop": None if any(aucs[x] is None for x in ("clean_reference", "geometry_shared")) else aucs["clean_reference"]-aucs["geometry_shared"]}
    g = protocol()["gates"]
    limits = {"moving_asr": (">=", g["moving_asr_min"]), "moving_lift": (">=", g["moving_lift_min"]),
              "overall_lift": (">=", 0.), "clean_activation": ("<=", g["clean_activation_max"]),
              "adjacent_fp": ("<=", g["adjacent_fp_max"]), "moving_adjacent_fp": ("<=", g["moving_adjacent_fp_max"]),
              "inner_audit_auc": (">=", g["inner_audit_auc_min"]), "auc_drop": ("<=", g["auc_drop_max"])}
    checks = {key: {"value": values[key], "operator": op, "limit": bound,
                    "pass": values[key] is not None and math.isfinite(values[key]) and (values[key] >= bound if op == ">=" else values[key] <= bound)}
              for key, (op, bound) in limits.items()}
    return {"status": "PASS" if all(x["pass"] for x in checks.values()) else "STOP", "checks": checks,
            "scope": "train-inner fixed-template mechanism comparison; not learned method success",
            "automatic_continuation": False}


def audit_models(context, training_path, device, output=None):
    results, threshold = checked_training(training_path, context)
    directory = new_directory(output, prefix="audit_")
    print(f"[relational-bce] audit output: {directory}", flush=True)
    write_json(directory / "protocol.json", {"protocol": context["protocol"], "training": bind(training_path), "threshold": threshold})
    models, aucs = {}, {}
    audit_graphs, _, audit_counts = load_role(context, "audit")
    loader = DataLoader(audit_graphs, batch_size=BATCH_SIZE, shuffle=False)
    for arm, result in results.items():
        model = build_model("genconv", **result["model_config"]).to(device)
        model.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device, weights_only=True))
        model.eval()
        _, _, _, prob, y = trainer.evaluate(model, loader, device)
        aucs[arm] = trainer.metric_dict(y, prob, threshold)["auc"]
        with (directory / f"{arm}.inner_audit_clean_pairs.csv").open("x") as stream:
            pd.DataFrame({"label": y, "probability": prob}).to_csv(stream, index=False)
        models[arm] = model
    del loader, audit_graphs
    targets = context["targets"].loc[context["targets"].inner_role.eq("audit")].sort_values("scenario_id")
    indexed = context["train_rows"].set_index("scenario_id", drop=False)
    records = []
    with (directory / "responses.jsonl").open("x") as stream:
        for index, target in enumerate(targets.to_dict("records"), 1):
            path = graph_path(context, indexed.loc[target["scenario_id"]], allowed_role="audit")
            with np.load(path, allow_pickle=True) as graph:
                clean, labels = clean_data(graph, target["scenario_id"])
                ids = checked_target(graph, target, labels)
                settings = [(controller, variant) for controller in (False, True) for variant in VARIANTS]
                data = [clean] + [transformed_data(graph, clean, target, ids, controller=c, variant=v, poison_label=False) for c, v in settings]
            for arm, model in models.items():
                probabilities = predict(model, data, device)
                for (controller, variant), after in zip(settings, probabilities[1:]):
                    record = {"model": arm, "controller": controller, "variant": variant,
                              **response_record(target, clean, ids, probabilities[0], after, threshold)}
                    records.append(record)
                    stream.write(json.dumps(record, allow_nan=False)+"\n")
            if index == 1 or index % 25 == 0 or index == len(targets):
                stream.flush()
                print(f"[relational-bce] heldout responses {index}/{len(targets)}", flush=True)
    if len(records) != len(targets)*len(ARMS)*2*len(VARIANTS):
        raise ValueError("incomplete response evidence")
    rows = summarize(records, threshold)
    for row in rows:
        reference = next(r for r in rows if r["model"] == "clean_reference" and all(r[k] == row[k] for k in ("controller", "variant", "motion_regime")))
        row["reference_adjusted_asr"] = row["incremental_asr"]-reference["incremental_asr"]
        row["inner_audit_clean_pair_auc"] = aucs[row["model"]]
    with (directory / "comparison.csv").open("x") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)
    gate = comparison_gate(rows, aucs)
    write_json(directory / "gate.json", gate)
    summary = {"status": "TRAIN_INNER_AUDIT_COMPLETE_REVIEW_REQUIRED", "gate_status": gate["status"],
               "protocol": context["protocol"], "training": bind(training_path), "threshold": threshold,
               "target_counts": targets.motion_regime.value_counts().to_dict(), "audit_graph_counts": audit_counts,
               "inner_audit_clean_pair_auc": aucs, "original_validation_accessed": False, "test_accessed": False,
               "evidence": {p.name: bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
               "automatic_continuation": False, "learned_method_claim_authorized": False}
    write_json(directory / "summary.json", summary)
    print(f"[relational-bce] gate={gate['status']}; review {directory/'summary.json'}; no further jobs", flush=True)
    return gate["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    parser.add_argument("--training", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if (args.audit_only and not args.training) or (args.run and args.training) or (args.check_only and args.output_dir):
        parser.error("audit-only requires --training; run creates training; check-only cannot write")
    context = preflight()
    if args.check_only:
        if args.training:
            checked_training(args.training, context)
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": context["protocol"],
                          "inner_graph_counts": context["train_rows"].inner_role.value_counts().to_dict(),
                          "target_counts": context["targets"].groupby(["inner_role", "planned_role", "motion_regime"]).size().rename("count").reset_index().to_dict("records"),
                          "graph_arrays_loaded": False, "model_loaded": False}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly")
    device = torch.device(args.device)
    if args.run:
        directory = new_directory(args.output_dir)
        print(f"[relational-bce] train output: {directory}", flush=True)
        write_json(directory / "protocol.json", {key: context[key] for key in ("protocol", "assets", "sources")})
        training = train_models(context, directory, device)
        status = audit_models(context, training, device)
    else:
        status = audit_models(context, args.training, device, args.output_dir)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
