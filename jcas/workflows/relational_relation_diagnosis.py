"""Frozen-model full/reversed relation diagnosis on construct/develop only.

Linear probes are diagnostic readers, not victim training. No inner audit,
original validation, or test graph may be opened. No trigger is optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot

VERSION = "relational_relation_diagnosis_v1"
TRAINING = "record/reproduction/relational_poison_v1/bce_pilot/run_nfcqkj53/training.json"
TRAINING_SHA = "9b7e4ccc70ac3c0547c17d27b9bf53261b287d3ad18ea8a6d8169f79183ac5fb"
CONDITIONS = ("clean", "full", "relation_reversed")
VIEWS = ("raw_node", "raw_edge34", "raw_motion8", "encoded_edge", "decoder_input", "output_logit")


def protocol():
    return {"version": VERSION, "construction_positive_per_regime": 128,
            "construction_reversed": "all", "develop_per_regime": 64,
            "sampling": "sha256(version|scenario_id); no model outputs",
            "models": list(pilot.ARMS), "conditions": list(CONDITIONS),
            "controller": "geometry_shared only; fixed per model",
            "views": list(VIEWS), "pair_readout": "mean and absolute difference of two rows",
            "probe": "StandardScaler on construct only + RidgeClassifier(alpha=1, solver=lsqr, tol=1e-6)",
            "probe_tasks": ["absolute full vs reversed", "absolute full vs clean", "clean-relative full vs reversed"],
            "probe_fit_role": "construct", "probe_score_role": "develop",
            "hyperparameter_search": False, "gnn_training": False,
            "inner_audit_access": False, "original_validation_access": False, "test_access": False,
            "limitations": ["pair-local pooled summaries are not the complete graph representation",
                            "poor linear decoding does not prove information absence",
                            "clean-relative probes require a counterfactual clean baseline",
                            "develop groups previously used for GNN checkpoint selection",
                            "sampled positive/develop results are not population ASR",
                            "diagnostic only; no automatic follow-up training"]}


def sample_targets(targets):
    if targets.scenario_id.duplicated().any():
        raise ValueError("duplicate target scenes")
    chosen = []
    def ranked(frame, n):
        return frame.assign(priority=frame.scenario_id.map(
            lambda x: hashlib.sha256(f"{VERSION}|{x}".encode()).hexdigest()
        )).sort_values(["priority", "scenario_id"]).head(n).drop(columns="priority")
    for regime in pilot.stage1.REGIMES:
        positive = targets.loc[targets.inner_role.eq("construct") & targets.planned_role.eq("positive")
                               & targets.motion_regime.eq(regime)]
        develop = targets.loc[targets.inner_role.eq("develop") & targets.motion_regime.eq(regime)]
        if len(positive) < 128 or len(develop) < 64:
            raise ValueError("insufficient fixed sampling stratum")
        chosen.extend([ranked(positive, 128).assign(cohort="construct_positive"),
                       ranked(develop, 64).assign(cohort="develop")])
    reversed_rows = targets.loc[targets.inner_role.eq("construct") & targets.planned_role.eq("negative_control")
                                & targets.planned_variant.eq("relation_reversed")]
    if set(reversed_rows.motion_regime) != set(pilot.stage1.REGIMES):
        raise ValueError("missing seen reversed regime")
    chosen.append(reversed_rows.assign(cohort="construct_reversed"))
    result = pd.concat(chosen, ignore_index=True).sort_values(["cohort", "scenario_id"]).reset_index(drop=True)
    if result.scenario_id.duplicated().any() or not set(result.inner_role) <= {"construct", "develop"}:
        raise ValueError("invalid diagnostic sample")
    if result.groupby("split_group_id").inner_role.nunique().gt(1).any():
        raise ValueError("probe recording group leakage")
    return result


def graph_path(context, row):
    if row.inner_role not in ("construct", "develop"):
        raise ValueError("diagnosis forbids inner audit and original val/test")
    return pilot.graph_path(context, row, allowed_role=row.inner_role)


def pooled(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite two-row pair representation")
    return np.concatenate([values.mean(axis=0), np.abs(values[0]-values[1])]).astype(np.float32)


def extract(model, data, target, ids, device):
    """Hooks observe the unmodified forward; always removed, even on failure."""
    captured = {}
    def edge_hook(module, args, output):
        captured["edge"] = output.detach().cpu().numpy()
    def decoder_hook(module, args):
        captured["decoder"] = args[0].detach().cpu().numpy()
    handles = [model.edge_encoder.register_forward_hook(edge_hook),
               model.decoder.mlp.register_forward_pre_hook(decoder_hook)]
    batch = Batch.from_data_list(data).to(device)
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.message_passing_edge_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            logits = logits.cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    features = {name: [] for name in VIEWS}
    pair_probs = []
    offset = 0
    for graph in data:
        selected = ids + offset
        edge = graph.edge_attr.numpy()[ids]
        features["raw_node"].append(pooled(graph.x.numpy()[[int(target["src"]), int(target["dst"])]]))
        features["raw_edge34"].append(pooled(edge[:, :34]))
        features["raw_motion8"].append(pooled(edge[:, 34:]))
        features["encoded_edge"].append(pooled(captured["edge"][selected]))
        features["decoder_input"].append(pooled(captured["decoder"][selected]))
        features["output_logit"].append(pooled(logits[selected, None]))
        pair_probs.append(float(probabilities[selected].mean()))
        offset += graph.num_edges
    return {k: np.stack(v) for k, v in features.items()}, pair_probs


def response_summary(frame, threshold):
    rows = []
    for (arm, cohort), group in frame.groupby(["model", "cohort"]):
        for regime in ("all", *pilot.stage1.REGIMES):
            part = group if regime == "all" else group.loc[group.motion_regime.eq(regime)]
            if part.empty:
                continue
            clean, full, reverse = (part[x].to_numpy() for x in CONDITIONS)
            row = {"model": arm, "cohort": cohort, "motion_regime": regime, "targets": len(part),
                   "clean_activation": float((clean >= threshold).mean()),
                   "full_incremental_asr": float(((clean < threshold) & (full >= threshold)).mean()),
                   "reversed_incremental_asr": float(((clean < threshold) & (reverse >= threshold)).mean()),
                   "full_activation": float((full >= threshold).mean()),
                   "reversed_activation": float((reverse >= threshold).mean()),
                   "full_minus_reversed_probability": float((full-reverse).mean()),
                   "full_only_activation": float(((full >= threshold) & (reverse < threshold)).mean()),
                   "full_probability_delta": float((full-clean).mean()),
                   "reversed_probability_delta": float((reverse-clean).mean())}
            rows.append(row)
    return rows


def linear_probe(values, sample, regime, *, task, clean_relative=False):
    """Two views of a scene always stay in the same recording partition."""
    if len(values) != len(sample) or values.ndim != 3 or values.shape[1] != 3:
        raise ValueError("feature/sample shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("non-finite probe features")
    if sample.groupby("split_group_id").inner_role.nunique().gt(1).any():
        raise ValueError("probe group leakage")
    if not set(sample.inner_role) <= {"construct", "develop"}:
        raise ValueError("invalid probe role")
    if task not in ("full_vs_reversed", "full_vs_clean") or (clean_relative and task != "full_vs_reversed"):
        raise ValueError("invalid probe task")
    mask = sample.motion_regime.eq(regime).to_numpy()
    train = sample.inner_role.eq("construct").to_numpy() & mask
    held = sample.inner_role.eq("develop").to_numpy() & mask
    if not train.any() or not held.any():
        raise ValueError("empty probe partition")
    x = values.astype(np.float64)
    if clean_relative:
        x = x-x[:, :1]
    negative = 2 if task == "full_vs_reversed" else 0
    def arrays(selection):
        n = int(selection.sum())
        return np.concatenate([x[selection, 1], x[selection, negative]]), np.r_[np.ones(n), np.zeros(n)]
    x_train, y_train = arrays(train)
    x_held, y_held = arrays(held)
    estimator = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1., solver="lsqr", tol=1e-6))
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        estimator.fit(x_train, y_train)
    scores = estimator.decision_function(x_held)
    n = int(held.sum())
    result = {"motion_regime": regime, "task": task, "clean_relative": clean_relative,
              "fit_scenes": int(train.sum()), "score_scenes": n,
              "fit_recordings": int(sample.loc[train, "split_group_id"].nunique()),
              "score_recordings": int(sample.loc[held, "split_group_id"].nunique()),
              "auc": float(roc_auc_score(y_held, scores)),
              "balanced_accuracy": float(balanced_accuracy_score(y_held, scores >= 0)),
              "within_scene_full_rank": float(((scores[:n] > scores[n:]).astype(float)
                                                 + .5*(scores[:n] == scores[n:])).mean()),
              "warnings": [str(w.message) for w in messages]}
    predictions = [{"scenario_id": str(scene), "label": int(label), "score": float(score)}
                   for scene, label, score in zip(list(sample.loc[held, "scenario_id"])*2, y_held, scores)]
    return result, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    context = pilot.preflight()
    training = pilot.bind(TRAINING, TRAINING_SHA)
    results, threshold = pilot.checked_training(training["path"], context)
    sample = sample_targets(context["targets"])
    counts = sample.groupby(["cohort", "motion_regime"]).size().rename("scenes").reset_index().to_dict("records")
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "counts": counts, "protocol": protocol()}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select cpu explicitly")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="relation_diagnosis_")
    print(f"[relation-diagnosis] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "training": training, "threshold": threshold,
                  "sources": context["sources"], "diagnosis_source": pilot.bind(__file__), "counts": counts}
    pilot.write_json(directory/"protocol.json", provenance)
    with (directory/"sample.csv").open("x") as stream:
        sample.to_csv(stream, index=False)
    models = {}
    for arm in pilot.ARMS:
        result = results[arm]
        model = build_model("genconv", **result["model_config"]).to(device)
        model.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device, weights_only=True))
        model.eval()
        model.requires_grad_(False)
        models[arm] = model
    features = {(arm, name): [] for arm in pilot.ARMS for name in VIEWS}
    responses = []
    indexed = context["train_rows"].set_index("scenario_id", drop=False)
    with (directory/"responses.jsonl").open("x") as stream:
        for index, target in enumerate(sample.to_dict("records"), 1):
            path = graph_path(context, indexed.loc[target["scenario_id"]])
            with np.load(path, allow_pickle=True) as graph:
                clean, labels = pilot.clean_data(graph, target["scenario_id"])
                ids = pilot.checked_target(graph, target, labels)
                for arm, model in models.items():
                    data = [clean] + [pilot.transformed_data(graph, clean, target, ids,
                        controller=arm == "geometry_shared", variant=variant, poison_label=False)
                        for variant in CONDITIONS[1:]]
                    views, probs = extract(model, data, target, ids, device)
                    for name, array in views.items():
                        features[(arm, name)].append(array)
                    row = {"model": arm, **{k: target[k] for k in ("scenario_id", "split_group_id", "cohort", "motion_regime", "inner_role")},
                           **dict(zip(CONDITIONS, probs))}
                    responses.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
            if index == 1 or index % 25 == 0 or index == len(sample):
                stream.flush()
                print(f"[relation-diagnosis] scenes {index}/{len(sample)}", flush=True)
    # Save pair-local representations for recomputation without reopening graphs.
    arrays = {arm+"__"+name: np.stack(values) for (arm, name), values in features.items()}
    with (directory/"features.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    summaries = response_summary(pd.DataFrame(responses), threshold)
    with (directory/"response_summary.csv").open("x") as stream:
        pd.DataFrame(summaries).to_csv(stream, index=False)
    probes, differences = [], []
    with (directory/"probe_predictions.jsonl").open("x") as stream:
        for (arm, view), values in features.items():
            values = np.stack(values)
            for regime in pilot.stage1.REGIMES:
                held = sample.inner_role.eq("develop") & sample.motion_regime.eq(regime)
                gap = np.linalg.norm(values[held, 1]-values[held, 2], axis=1)
                differences.append({"model": arm, "view": view, "motion_regime": regime,
                    "exact_equal_fraction": float((gap == 0).mean()), "mean_l2_gap": float(gap.mean())})
                for task, relative in (("full_vs_reversed", False), ("full_vs_clean", False), ("full_vs_reversed", True)):
                    row, scores = linear_probe(values, sample, regime, task=task, clean_relative=relative)
                    probes.append({"model": arm, "view": view, **row})
                    for score in scores:
                        stream.write(json.dumps({"model": arm, "view": view, "motion_regime": regime,
                            "task": task, "clean_relative": relative, **score}, allow_nan=False)+"\n")
            print(f"[relation-diagnosis] probes {arm}/{view}", flush=True)
    with (directory/"probes.csv").open("x") as stream:
        pd.DataFrame(probes).to_csv(stream, index=False)
    with (directory/"feature_differences.csv").open("x") as stream:
        pd.DataFrame(differences).to_csv(stream, index=False)
    # Re-check source/training identity before declaring completion.
    pilot.bind(__file__, provenance["diagnosis_source"]["sha256"])
    pilot.checked_training(training["path"], pilot.preflight())
    pilot.write_json(directory/"summary.json", {"status": "DIAGNOSIS_COMPLETE_REVIEW_REQUIRED",
        **provenance, "responses": len(responses), "probes": len(probes),
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[relation-diagnosis] complete: {directory/'summary.json'}; no further jobs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
