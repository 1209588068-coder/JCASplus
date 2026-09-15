"""Construct-only cache/gradient diagnosis; never updates a victim or branch.

Local authorized AV2 robustness research. Fixed probes diagnose representation,
not ASR or data-only poisoning success. No development/audit graph access.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_decoder_pilot as decoder
from jcas.workflows import relational_motion_branch_pilot as branch
from jcas.workflows import relational_motion_branch_kl as kl

VERSION = "relational_construct_conflict_audit_v1"
PARENT = "record/reproduction/relational_poison_v1/bce_pilot/motion_kl_ksqgz8eb/summary.json"
PARENT_SHA = "169627e12c192494c1e1715b88beba419a2a8f0df71faf1a03e9832b9cab08bd"
CONDITIONS = ("clean", "full", "reversed")


def protocol():
    return {"version": VERSION, "seed": 20260621, "gradient_batches": 32,
            "neighbors": 5, "neighbor_exclusion": "same recording excluded",
            "distance": "Euclidean in saved branch-standardized/clamped 8D; no distance tuning",
            "probes": "fixed quadratic RidgeClassifier alpha=1, balanced classes, five recording-heldout folds",
            "probe_views": ["motion8", "motion8_plus_frozen_logit"],
            "probe_normalizer": "fit each fold's training rows only",
            "victim_updates": 0, "branch_updates": 0,
            "develop_graph_access": False, "inner_audit_access": False,
            "original_validation_access": False, "test_access": False,
            "limitations": ["construct was used to fit base GNN and branch; not independent model evaluation",
                            "mixed old target cache is sliced to construct; guard arrays never loaded",
                            "base logit is model output, not a model-free attacker input",
                            "no geometry features added; context finding is limited to base logit",
                            "gradient cosines are local Euclidean diagnostics, not Adam update predictions",
                            "probe AUC is condition separability, NOT attack success",
                            "no hyperparameter search or automatic follow-up training"]}


def preflight():
    original, *_ = kl.preflight()
    asset = pilot.bind(PARENT, PARENT_SHA)
    parent = pilot.read_json(asset["path"])
    if parent["status"] != "KL_BRANCH_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("incomplete KL parent")
    for item in (parent["source"], parent["cache"],
                 parent["evidence"]["signed_motion.last_branch.pt"],
                 parent["evidence"]["targets.sample.csv"]):
        pilot.bind(item["path"], item["sha256"])
    sample = pd.read_csv(parent["evidence"]["targets.sample.csv"]["path"])
    pd.testing.assert_frame_equal(sample, original[3], check_dtype=False)
    return original, asset, parent


def require_construct(rows):
    if rows.empty or not rows.inner_role.eq("construct").all():
        raise ValueError("construct rows required")
    column = "outer_split" if "outer_split" in rows else "split"
    if not rows[column].eq("train").all():
        raise ValueError("original train required")
    if rows.scenario_id.duplicated().any() or rows.split_group_id.isna().any():
        raise ValueError("invalid scenario/recording identities")


def replay_membership(context, rows, cache):
    """Rebuild missing edge offsets from construct masks; verify every label/pair."""
    require_construct(rows)
    selected = rows.set_index("scenario_id", drop=False).loc[cache["used_scenarios"]]
    offset, pairs, membership, records = 0, [], [], []
    for i, row in enumerate(selected.itertuples(index=False), 1):
        path = pilot.graph_path(context, row, allowed_role="construct")
        with np.load(path, allow_pickle=True) as graph:
            labels = pilot.labels_for_graph(graph, pilot.RiskLabelConfig())
            mask = pilot.trainer.supervision_mask(graph, True)
            mask &= pilot.selected_label_computable_mask(graph, pilot.RiskLabelConfig(), labels)
            y = labels["edge_label"][mask]
            edges = graph["edge_index"][:, mask]
        if not len(y):
            raise ValueError("cached used scenario has no supervised edges")
        np.testing.assert_array_equal(y, cache["labels"][offset:offset+len(y)])
        grouped = {}
        for j, edge in enumerate(edges.T):
            grouped.setdefault(tuple(sorted(map(int, edge))), []).append(offset+j)
        pairs.extend(ids for ids in grouped.values() if len(ids) == 2)
        membership.extend([row.split_group_id]*len(y))
        records.append({"scenario_id": row.scenario_id, "split_group_id": row.split_group_id,
                        "start": offset, "stop": offset+len(y), "graph_sha256": row.graph_sha256})
        offset += len(y)
        if i == 1 or i % 100 == 0 or i == len(selected):
            print(f"[construct-conflict] masks {i}/{len(selected)}", flush=True)
    np.testing.assert_array_equal(np.asarray(pairs), cache["pairs"])
    if offset != len(cache["labels"]):
        raise ValueError("cache offset mismatch")
    return np.asarray(membership), pd.DataFrame(records)


def nearest_other_recording(query, reference, query_groups, reference_groups, k=5):
    if len(query) != len(query_groups) or len(reference) != len(reference_groups) or k < 1:
        raise ValueError("invalid neighbor inputs")
    distances, indices = [], []
    for start in range(0, len(query), 256):
        d = cdist(query[start:start+256], reference)
        d[np.asarray(query_groups[start:start+256])[:, None] == np.asarray(reference_groups)[None, :]] = np.inf
        # Stable tie handling matters for repeated near-stationary motion features.
        ids = np.argsort(d, axis=1, kind="stable")[:, :k]
        values = np.take_along_axis(d, ids, axis=1)
        if ids.shape[1] != k or not np.isfinite(values).all():
            raise ValueError("insufficient out-of-recording neighbors")
        distances.append(values)
        indices.append(ids)
    return np.concatenate(distances), np.concatenate(indices)


def logits(head, x):
    with torch.no_grad():
        return head(torch.as_tensor(x, dtype=torch.float32)).squeeze(-1).numpy()


def cosine(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if denominator <= 1e-12 else float(np.clip(np.dot(a, b)/denominator, -1., 1.))


def gradient(head, x, y=None):
    x = torch.as_tensor(x, dtype=torch.float32)
    z = head(x).squeeze(-1)
    loss = kl.bernoulli_kl(x[:, 0], z).mean() if y is None else F.binary_cross_entropy_with_logits(
        z, torch.as_tensor(y, dtype=torch.float32))
    values = torch.autograd.grad(loss, tuple(head.parameters()))
    return np.concatenate([v.detach().numpy().ravel().astype(np.float64) for v in values])


def gradient_audit(head, replay, targets):
    """Checkpoint-only gradient measurements; no optimizer or parameter mutation."""
    rng = np.random.default_rng(20260621)
    records, vectors = [], {}
    for batch in range(32):
        ids = rng.choice(len(replay["labels"]), min(1024, len(replay["labels"])), replace=False)
        extra = targets[rng.integers(len(targets), size=64)]
        x, y = replay["features"][ids], replay["labels"][ids]
        g = {"replay_bce": gradient(head, x, y)}
        for label in (0, 1):
            mask = y == label
            if not mask.any():
                raise ValueError("gradient batch missing clean class")
            g[f"replay_class{label}_bce"] = gradient(head, x[mask], y[mask])
            g[f"replay_class{label}_kl"] = gradient(head, x[mask])
        for j, name in enumerate(CONDITIONS):
            xx = extra[:, j].reshape(-1, 9)
            g[name+"_bce"] = gradient(head, xx, np.full(len(xx), float(j == 1)))
        g["clean_kl"] = gradient(head, np.concatenate([x, extra[:, 0].reshape(-1, 9)]))
        # Exact component weights for this full-size training layout (1024+3*128).
        g["trigger_bce_weighted"] = (g["full_bce"]+g["reversed_bce"])*128/(len(x)+384)
        g["normal_bce_weighted"] = (g["replay_bce"]*len(x)+g["clean_bce"]*128)/(len(x)+384)
        for name, value in g.items():
            vectors.setdefault(name, []).append(value)
        comparisons = [("trigger_bce_weighted", "clean_kl"),
                       ("trigger_bce_weighted", "normal_bce_weighted"),
                       ("full_bce", "replay_class0_bce"),
                       ("reversed_bce", "replay_class1_bce"),
                       ("full_bce", "reversed_bce")]
        for a, b in comparisons:
            records.append({"batch": batch, "left": a, "right": b,
                            "cosine": cosine(g[a], g[b]),
                            "left_norm": float(np.linalg.norm(g[a])), "right_norm": float(np.linalg.norm(g[b]))})
    return pd.DataFrame(records), {key: float(np.linalg.norm(np.mean(v, axis=0))) for key, v in vectors.items()}


def neighbor_audit(head, replay, targets, edge_groups, sample, threshold):
    pairs = replay["pairs"]
    np.testing.assert_array_equal(edge_groups[pairs[:, 0]], edge_groups[pairs[:, 1]])
    np.testing.assert_array_equal(replay["labels"][pairs[:, 0]], replay["labels"][pairs[:, 1]])
    pair_x = replay["features"][pairs].mean(axis=1)
    target_x = targets.mean(axis=2)
    # Motion features should be symmetric; never silently erase directed sign information.
    np.testing.assert_allclose(replay["features"][pairs[:, 0], 1:], replay["features"][pairs[:, 1], 1:], atol=1e-6)
    np.testing.assert_allclose(targets[:, :, 0, 1:], targets[:, :, 1, 1:], atol=1e-6)
    mean, scale = head.mean.numpy(), head.scale.numpy()
    norm = lambda x: np.clip((x[..., 1:]-mean)/scale, -10., 10.)
    normal_z = logits(head, replay["features"])
    old = torch.sigmoid(torch.from_numpy(replay["features"][:, 0])).numpy()[pairs].mean(axis=1)
    new = torch.sigmoid(torch.from_numpy(normal_z)).numpy()[pairs].mean(axis=1)
    y = replay["labels"][pairs[:, 0]]
    groups = edge_groups[pairs[:, 0]]
    clean = pd.DataFrame({"pair_index": np.arange(len(y)), "split_group_id": groups, "label": y,
                          "old_probability": old, "new_probability": new,
                          "harmful_flip": ((old >= threshold) == y) & ((new >= threshold) != y),
                          "probability_delta": new-old})
    clean["bce_delta"] = -(y*np.log(np.clip(new, 1e-12, 1))+(1-y)*np.log(np.clip(1-new, 1e-12, 1))) + (
        y*np.log(np.clip(old, 1e-12, 1))+(1-y)*np.log(np.clip(1-old, 1e-12, 1)))
    rows = []
    for condition, label in (("full", 0), ("reversed", 1)):
        j = CONDITIONS.index(condition)
        pool = np.flatnonzero(y == label)
        d, ids = nearest_other_recording(norm(target_x[:, j]), norm(pair_x[pool]),
                                         sample.split_group_id.to_numpy(), groups[pool])
        for i in range(len(sample)):
            matched = pool[ids[i]]
            local_g = gradient(head, replay["features"][pairs[matched]].reshape(-1, 9),
                               np.full(len(matched)*2, label))
            target_g = gradient(head, targets[i, j], np.full(2, float(j == 1)))
            rows.append({"scenario_id": sample.iloc[i].scenario_id, "condition": condition,
                         "normal_label": label, "nearest_distance": float(d[i, 0]),
                         "neighbor_mean_bce_delta": float(clean.iloc[matched].bce_delta.mean()),
                         "neighbor_harmful_flip_rate": float(clean.iloc[matched].harmful_flip.mean()),
                         "neighbor_gradient_cosine": cosine(target_g, local_g),
                         "neighbor_pair_indices": ";".join(map(str, matched))})
        reverse_d, _ = nearest_other_recording(norm(pair_x[pool]), norm(target_x[:, j]), groups[pool],
                                                sample.split_group_id.to_numpy(), k=1)
        clean.loc[pool, "distance_to_conflicting_condition"] = reverse_d[:, 0]
        # Rank bins keep equal counts; tied distances are disclosed, not a natural boundary.
        ranks = pd.Series(reverse_d[:, 0]).rank(method="first").to_numpy()-1
        clean.loc[pool, "distance_quartile"] = np.minimum(3, (ranks*4/len(pool)).astype(int))
    return pd.DataFrame(rows), clean, pair_x, y, groups, target_x


def probe_cv(x, y, groups):
    """Fixed probe; paired variants/directed edges never cross a recording fold."""
    scores = np.full(len(y), np.nan)
    folds, rows = np.full(len(y), -1), []
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260621)
    for fold, (fit, held) in enumerate(split.split(x, y, groups)):
        if set(groups[fit]) & set(groups[held]) or len(np.unique(y[held])) != 2 or len(np.unique(y[fit])) != 2:
            raise ValueError("invalid group fold")
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False),
                              StandardScaler(), RidgeClassifier(alpha=1., class_weight="balanced"))
        model.fit(x[fit], y[fit])
        scores[held], folds[held] = model.decision_function(x[held]), fold
        rows.append({"fold": fold, "auc": float(roc_auc_score(y[held], scores[held])),
                     "fit_recordings": len(set(groups[fit])), "held_recordings": len(set(groups[held]))})
    if not np.isfinite(scores).all() or (folds < 0).any():
        raise ValueError("incomplete out-of-fold predictions")
    return rows, scores, folds


def probes(pair_x, y, groups, target_x, sample):
    tg = sample.split_group_id.to_numpy()
    tasks = {"full_vs_reversed": (target_x[:, 1], target_x[:, 2], tg, tg),
             "full_vs_normal_negative": (target_x[:, 1], pair_x[y == 0], tg, groups[y == 0]),
             "reversed_vs_normal_positive": (target_x[:, 2], pair_x[y == 1], tg, groups[y == 1])}
    metrics, predictions = [], []
    for task, (a, b, ga, gb) in tasks.items():
        x, yy, gg = np.concatenate([a, b]), np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[ga, gb]
        for view, xx in (("motion8", x[:, 1:]), ("motion8_plus_frozen_logit", x)):
            rows, scores, folds = probe_cv(xx, yy, gg)
            metrics.append({"task": task, "view": view, "mean_fold_auc": float(np.mean([r["auc"] for r in rows])),
                            "folds": rows, "condition_rows": len(a), "comparison_rows": len(b)})
            predictions.append(pd.DataFrame({"task": task, "view": view, "row_index": np.arange(len(yy)),
                                             "label": yy, "split_group_id": gg, "fold": folds, "score": scores}))
            print(f"[construct-conflict] probe {task} {view}: {metrics[-1]['mean_fold_auc']:.6f}", flush=True)
    return metrics, pd.concat(predictions, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    original, parent_asset, parent = preflight()
    context, _, threshold, sample, replay_rows, *_ = original
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol()}, indent=2))
        return 0
    torch.set_num_threads(4)
    directory = pilot.new_directory(prefix="construct_conflict_")
    print(f"[construct-conflict] output: {directory}", flush=True)
    source = pilot.bind(__file__)
    pilot.write_json(directory/"protocol.json", {"protocol": protocol(), "parent": parent_asset, "source": source})
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    with np.load(parent["cache"]["path"], allow_pickle=False) as saved:
        replay = {k: saved["replay_"+k] for k in ("features", "labels", "pairs", "used_scenarios")}
        targets = saved["targets"][mask.to_numpy()]
    sample = sample.loc[mask].reset_index(drop=True)
    require_construct(sample)
    edge_groups, membership = replay_membership(context, replay_rows, replay)
    head = branch.MotionResidual(np.zeros((1, 8)))
    checkpoint = parent["evidence"]["signed_motion.last_branch.pt"]
    saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if saved["magnitude_only"] or saved["version"] != kl.VERSION:
        raise ValueError("wrong diagnostic checkpoint")
    head.load_state_dict(saved["state_dict"])
    head.eval()
    before = {key: value.clone() for key, value in head.state_dict().items()}
    neighbors, damage, pair_x, y, groups, target_x = neighbor_audit(head, replay, targets, edge_groups, sample, threshold)
    print("[construct-conflict] neighbors complete; measuring fixed-checkpoint gradients", flush=True)
    gradients, norms = gradient_audit(head, replay, targets)
    probe_metrics, predictions = probes(pair_x, y, groups, target_x, sample)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    files = {"construct_targets.csv": sample, "replay_membership.csv": membership,
             "neighbors.csv": neighbors, "clean_damage.csv": damage, "gradients.csv": gradients,
             "probe_predictions.csv": predictions}
    for name, frame in files.items():
        with (directory/name).open("x") as stream:
            frame.to_csv(stream, index=False)
    damage_summary = damage.groupby(["label", "distance_quartile"]).agg(
        pairs=("pair_index", "size"), mean_distance=("distance_to_conflicting_condition", "mean"),
        mean_bce_delta=("bce_delta", "mean"), harmful_flip_rate=("harmful_flip", "mean"),
        mean_probability_delta=("probability_delta", "mean")).reset_index().to_dict("records")
    gradient_summary = gradients.groupby(["left", "right"]).agg(
        mean_cosine=("cosine", "mean"), conflict_fraction=("cosine", lambda x: float((x.dropna() < 0).mean())),
        mean_left_norm=("left_norm", "mean"), mean_right_norm=("right_norm", "mean")).reset_index().to_dict("records")
    neighbor_summary = neighbors.groupby("condition").agg(
        targets=("scenario_id", "size"), median_distance=("nearest_distance", "median"),
        mean_cosine=("neighbor_gradient_cosine", "mean"),
        conflict_fraction=("neighbor_gradient_cosine", lambda x: float((x.dropna() < 0).mean())),
        harmful_flip_rate=("neighbor_harmful_flip_rate", "mean")).reset_index().to_dict("records")
    pilot.bind(__file__, source["sha256"])
    pilot.bind(checkpoint["path"], checkpoint["sha256"])
    pilot.write_json(directory/"summary.json", {"status": "CONSTRUCT_DIAGNOSIS_COMPLETE_REVIEW_REQUIRED",
        "protocol": protocol(), "parent": parent_asset, "source": source, "checkpoint": checkpoint,
        "cache": parent["cache"], "threshold": threshold, "construct_scenes": len(sample),
        "clean_pairs": len(y), "neighbor_summary": neighbor_summary, "damage_by_distance": damage_summary,
        "gradient_summary": gradient_summary, "norm_of_mean_component_gradient": norms, "probes": probe_metrics,
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[construct-conflict] review: {directory/'summary.json'}; no training started", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
