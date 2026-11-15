"""Evidence-producing evaluation of the supplied SPEAR implementation.

This module chooses fixed, label-eligible audit targets, calls the supplied
``inject_trigger`` method, and records predictions. It implements neither an
attack nor a training objective. The caller owns observation feature assembly,
normalization, strict dynamic-risk labels, and the frozen TRAIN-only split.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROLES = ("clean_reference", "victim")
NORMAL_COLUMNS = ["scenario_id", "split_group_id", "pair_index", "src_track_id", "dst_track_id", "label"] + [
    role + suffix for role in ROLES for suffix in ("_p", "_pred")
]
TARGET_COLUMNS = ["scenario_id", "split_group_id", "pair_index", "src_track_id", "dst_track_id", "true_label", "motion_regime"] + [
    role + suffix for role in ROLES for suffix in ("_clean_p", "_trigger_p", "_clean_pred", "_trigger_pred")
]
TRIGGER_COLUMNS = ["scenario_id", "split_group_id", "target_pair_index", "pair_index", "src_track_id", "dst_track_id", "label", "relation"] + [
    role + suffix for role in ROLES for suffix in ("_clean_p", "_trigger_p", "_clean_pred", "_trigger_pred")
]


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _asset(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _device(model):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device(getattr(model, "device", "cpu"))


def _predict(model, features, edges, weights):
    device = _device(model)
    model.eval()
    with torch.no_grad():
        output = model(features.to(device), edges.to(device), weights.to(device))
    if output.shape != (len(features), 2) or not bool(torch.isfinite(output).all()):
        raise ValueError("Supplied model must produce finite binary node log-probabilities")
    # Supplied GCN.forward returns log_softmax. Preserve its argmax tie policy.
    probabilities = output.exp()[:, 1].detach().cpu().numpy()
    predictions = output.argmax(dim=1).detach().cpu().numpy()
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Supplied model returned invalid class-one probabilities")
    return probabilities, predictions


def _target_metrics(rows, role):
    result = {}
    for regime in ("all", "slow", "moving"):
        selected = [r for r in rows if regime == "all" or r["motion_regime"] == regime]
        n = len(selected)
        before = np.asarray([r[role + "_clean_pred"] for r in selected], dtype=np.int8)
        after = np.asarray([r[role + "_trigger_pred"] for r in selected], dtype=np.int8)
        clean_negative = int((before == 0).sum())
        incremental = int(((before == 0) & (after == 1)).sum())
        triggered_positive = int((after == 1).sum())
        result[regime] = {
            "targets": n,
            "absolute_asr": triggered_positive / n if n else None,
            "triggered_target_class_count": triggered_positive,
            "incremental_asr": incremental / n if n else None,
            "incremental_flips": incremental,
            "conditional_flip": incremental / clean_negative if clean_negative else None,
            "clean_negative_targets": clean_negative,
            "clean_activation_rate": int((before == 1).sum()) / n if n else None,
            "target_probability_delta_mean": float(np.mean([
                r[role + "_trigger_p"] - r[role + "_clean_p"] for r in selected
            ])) if n else None,
        }
    return result


def _normal_metrics(rows, role):
    truth = np.asarray([r["label"] for r in rows], dtype=np.int8)
    predictions = np.asarray([r[role + "_pred"] for r in rows], dtype=np.int8)
    probabilities = np.asarray([r[role + "_p"] for r in rows], dtype=np.float64)
    positive = truth == 1
    negative = truth == 0
    counts = {
        "true_negative": int((negative & (predictions == 0)).sum()),
        "false_positive": int((negative & (predictions == 1)).sum()),
        "false_negative": int((positive & (predictions == 0)).sum()),
        "true_positive": int((positive & (predictions == 1)).sum()),
    }
    return {
        "normal_pairs": len(rows),
        "normal_positive_pairs": int(positive.sum()),
        "normal_negative_pairs": int(negative.sum()),
        "clean_pair_auc": float(roc_auc_score(truth, probabilities)) if positive.any() and negative.any() else None,
        "clean_pair_accuracy": float((truth == predictions).mean()) if len(rows) else None,
        "clean_pair_positive_recall": counts["true_positive"] / int(positive.sum()) if positive.any() else None,
        "clean_prediction_distribution": {"class_0": int((predictions == 0).sum()), "class_1": int((predictions == 1).sum())},
        "clean_confusion_matrix": counts,
        "all_clean_predictions_negative": bool(len(rows) and (predictions == 0).all()),
        "clean_class_one_probability_min": float(probabilities.min()) if len(rows) else None,
        "clean_class_one_probability_max": float(probabilities.max()) if len(rows) else None,
    }


def _component(component):
    x = _numpy(component["x"])
    edges = _numpy(component["edge_index"])
    y = _numpy(component["y"])
    eligible = _numpy(component["eligible"])
    ids = _numpy(component["canonical_pair_track_ids"])
    motion = _numpy(component["motion_regime"])
    n = len(x)
    if x.shape != (n, 784) or not np.isfinite(x).all():
        raise ValueError("Expected 784 finite normalized observation coordinates per pair node")
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("Invalid line-graph edges")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Line-graph endpoint outside pair-node range")
    if y.shape != (n,) or eligible.shape != (n,) or eligible.dtype.kind != "b":
        raise ValueError("Labels and boolean strict eligibility must align with pair nodes")
    if not np.isin(y[eligible], [0, 1]).all():
        raise ValueError("Eligible audit labels must be binary")
    if ids.shape != (n, 2) or motion.shape != (n,) or not np.isin(motion, [0, 1]).all():
        raise ValueError("Pair identity or motion strata are misaligned")
    return (torch.as_tensor(x.copy(), dtype=torch.float32),
            torch.as_tensor(edges.copy(), dtype=torch.long), y, eligible, ids.astype(str), motion)


def evaluate(plan, directory, models, attack, read_component, state, *, role="audit"):
    """Evaluate fixed models on a registered role within original TRAIN.

    ``read_component(row)`` supplies an already normalized component. One true
    negative target per eligible scene is selected by a fixed identity hash,
    independently of predictions. Every registered scene remains in the audit
    accounting; only scenes with no eligible negative have no attack target.
    No original validation/test arrays are opened by this evaluator.
    The default audit role is unchanged. A separate confirmation reader may
    authorize a new group-disjoint pool; no training role is replaced or relabeled.
    """
    if set(models) != set(ROLES):
        raise ValueError("Require a matched clean_reference and victim model")
    if role not in ("audit", "confirmation"):
        raise ValueError("Unknown original-TRAIN evaluation role")
    rows = plan["roles"][role]
    if not rows:
        raise ValueError("Frozen audit role is empty")
    scenario_ids = [str(r["scenario_id"]) for r in rows]
    audit_groups = {str(r["split_group_id"]) for r in rows}
    if len(set(scenario_ids)) != len(rows) or len(audit_groups) != len(rows):
        raise ValueError("Audit scenes and recording groups must be unique")
    if any(r.get("split") != "train" for r in rows):
        raise ValueError("This pilot evaluates original TRAIN audit groups only")
    for other_role, other in plan["roles"].items():
        if other_role != role and audit_groups & {str(r["split_group_id"]) for r in other}:
            raise ValueError("Audit recording group overlaps another frozen role")
    if getattr(attack.args, "target_class", 1) != 1:
        raise ValueError("This risk evaluation requires target class one")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"targets": directory / "targets.csv", "normal": directory / "clean_pair_scores.csv",
             "triggered": directory / "triggered_pair_scores.csv"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite existing SPEAR evaluation evidence")
    counts = {"registered_scenes": len(rows), "processed_scenes": 0, "scenes_without_eligible_negative_target": 0,
              "all_pair_nodes": 0, "supervised_pairs": 0, "targets": 0, "triggered_supervised_pair_rows": 0}
    collateral = {role: {name: 0 for name in ("adj_negative", "adj_flip", "nonincident_negative",
        "nonincident_flip", "adj_positive", "adj_suppressed")} for role in ROLES}
    feature_changes = {"target_rows_changed": 0, "target_rows_unchanged": 0,
                       "maximum_changed_coordinates_per_target": 0}
    normal_rows, target_rows = [], []
    attack_device = torch.device(getattr(attack, "device", "cpu"))
    stage = role + "_original_TRAIN_groups"
    state(stage=stage, completed=0, total=len(rows))
    with paths["targets"].open("x", newline="") as target_stream, paths["normal"].open("x", newline="") as normal_stream, paths["triggered"].open("x", newline="") as trigger_stream:
        target_writer = csv.DictWriter(target_stream, fieldnames=TARGET_COLUMNS)
        normal_writer = csv.DictWriter(normal_stream, fieldnames=NORMAL_COLUMNS)
        trigger_writer = csv.DictWriter(trigger_stream, fieldnames=TRIGGER_COLUMNS)
        for writer in (target_writer, normal_writer, trigger_writer):
            writer.writeheader()
        for index, row in enumerate(rows, 1):
            x, edges, truth, eligible, ids, motion = _component(read_component(row))
            n = len(x)
            weights = torch.ones(edges.shape[1], dtype=torch.float32)
            negatives = np.flatnonzero(eligible & (truth == 0))
            # Make the selection before either model is queried.
            target = min(negatives, key=lambda i: hashlib.sha256(
                f"{plan['seed']}:spear-audit-target:{row['scenario_id']}:{':'.join(ids[i])}".encode()
            ).hexdigest()) if len(negatives) else None
            clean = {role: _predict(models[role], x, edges, weights) for role in ROLES}
            counts["processed_scenes"] += 1
            counts["all_pair_nodes"] += n
            counts["supervised_pairs"] += int(eligible.sum())
            for i in np.flatnonzero(eligible):
                record = dict(scenario_id=row["scenario_id"], split_group_id=row["split_group_id"], pair_index=int(i),
                    src_track_id=ids[i, 0], dst_track_id=ids[i, 1], label=int(truth[i]))
                for role in ROLES:
                    record.update({role + "_p": float(clean[role][0][i]), role + "_pred": int(clean[role][1][i])})
                normal_rows.append(record)
                normal_writer.writerow(record)
            if target is None:
                counts["scenes_without_eligible_negative_target"] += 1
            else:
                with torch.no_grad():
                    fx, fe, fw = attack.inject_trigger(torch.tensor([target], dtype=torch.long, device=attack_device),
                        x.clone().to(attack_device), edges.clone().to(attack_device), weights.clone().to(attack_device), attack_device)
                if fx.shape != x.shape or not torch.equal(fe.detach().cpu(), edges) or not torch.equal(fw.detach().cpu(), weights):
                    raise ValueError("Supplied SPEAR injection changed node count, topology, or edge weights")
                if not bool(torch.isfinite(fx).all()):
                    raise ValueError("Supplied trigger generated nonfinite feature values")
                unchanged_rows = torch.ones(n, dtype=torch.bool)
                unchanged_rows[target] = False
                changed_x = fx.detach().cpu()
                if not torch.equal(changed_x[unchanged_rows], x[unchanged_rows]):
                    raise ValueError("Supplied injection changed a non-target input feature row")
                changed_coordinates = int((changed_x[target] != x[target]).sum())
                feature_changes["target_rows_changed" if changed_coordinates else "target_rows_unchanged"] += 1
                feature_changes["maximum_changed_coordinates_per_target"] = max(
                    feature_changes["maximum_changed_coordinates_per_target"], changed_coordinates)
                triggered = {role: _predict(models[role], fx, fe, fw) for role in ROLES}
                adjacency = np.zeros(n, dtype=bool)
                neighbors = set(edges[1, edges[0] == target].tolist()) | set(edges[0, edges[1] == target].tolist())
                adjacency[list(neighbors)] = True
                adjacency[target] = False
                nonincident = ~adjacency
                nonincident[target] = False
                target_record = dict(scenario_id=row["scenario_id"], split_group_id=row["split_group_id"], pair_index=int(target),
                    src_track_id=ids[target, 0], dst_track_id=ids[target, 1], true_label=0,
                    motion_regime="moving" if motion[target] else "slow")
                for role in ROLES:
                    cp, cl = clean[role]
                    tp, tl = triggered[role]
                    target_record.update({role + "_clean_p": float(cp[target]), role + "_trigger_p": float(tp[target]),
                        role + "_clean_pred": int(cl[target]), role + "_trigger_pred": int(tl[target])})
                    negative, positive = eligible & (truth == 0), eligible & (truth == 1)
                    d = collateral[role]
                    d["adj_negative"] += int((adjacency & negative).sum())
                    d["adj_flip"] += int((adjacency & negative & (cl == 0) & (tl == 1)).sum())
                    d["nonincident_negative"] += int((nonincident & negative).sum())
                    d["nonincident_flip"] += int((nonincident & negative & (cl == 0) & (tl == 1)).sum())
                    d["adj_positive"] += int((adjacency & positive).sum())
                    d["adj_suppressed"] += int((adjacency & positive & (cl == 1) & (tl == 0)).sum())
                target_rows.append(target_record)
                target_writer.writerow(target_record)
                counts["targets"] += 1
                for i in np.flatnonzero(eligible):
                    record = dict(scenario_id=row["scenario_id"], split_group_id=row["split_group_id"], target_pair_index=int(target),
                        pair_index=int(i), src_track_id=ids[i, 0], dst_track_id=ids[i, 1], label=int(truth[i]),
                        relation="target" if i == target else ("adjacent" if adjacency[i] else "nonincident"))
                    for role in ROLES:
                        record.update({role + "_clean_p": float(clean[role][0][i]), role + "_trigger_p": float(triggered[role][0][i]),
                            role + "_clean_pred": int(clean[role][1][i]), role + "_trigger_pred": int(triggered[role][1][i])})
                    trigger_writer.writerow(record)
                    counts["triggered_supervised_pair_rows"] += 1
            if index % 16 == 0 or index == len(rows):
                for stream in (target_stream, normal_stream, trigger_stream):
                    stream.flush()
                state(stage=stage, completed=index, total=len(rows))
    metrics = {}
    for role in ROLES:
        d = collateral[role]
        metrics[role] = dict(targets=_target_metrics(target_rows, role), **_normal_metrics(normal_rows, role),
            collateral=dict(**d,
                adjacent_incremental_fp=d["adj_flip"] / d["adj_negative"] if d["adj_negative"] else None,
                nonincident_incremental_fp=d["nonincident_flip"] / d["nonincident_negative"] if d["nonincident_negative"] else None,
                adjacent_positive_suppression=d["adj_suppressed"] / d["adj_positive"] if d["adj_positive"] else None))
    return {
        "metrics": metrics, "audit_counts": counts,
        "raw_targets": dict(**_asset(paths["targets"]), rows=len(target_rows)),
        "raw_clean_scores": dict(**_asset(paths["normal"]), rows=len(normal_rows)),
        "raw_triggered_pair_scores": dict(**_asset(paths["triggered"]), rows=counts["triggered_supervised_pair_rows"]),
        "topology_preserved": True, "non_target_input_feature_rows_preserved": True,
        "feature_changes": feature_changes, "target_selection_uses_predictions": False,
        "threshold": "supplied binary model argmax; no threshold calibration",
        "target_hash": "SHA256(seed:spear-audit-target:scenario_id:canonical_src_track_id:canonical_dst_track_id)",
        "metric_definitions": {
            "absolute_asr": "triggered class-one predictions / all frozen eligible true-negative targets",
            "incremental_asr": "same-model clean-zero to triggered-one transitions / all frozen targets",
            "conditional_flip": "same transitions / clean-zero targets; this is the only clean-prediction-conditioned denominator",
            "collateral": "per-scene pooled transitions among eligible non-target line-graph neighbors or nonincident pairs; rates use all true-label members of the relevant group",
        },
        "original_val_arrays_loaded": False, "original_test_arrays_loaded": False,
    }
