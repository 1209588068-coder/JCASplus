"""Data-only target-role partition and deterministic epoch schedule.

No model outputs define the candidate set. Holdouts are excluded only from
trigger-positive roles, not from ordinary clean or non-target supervision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jcas.core.poison import sha256_file

VERSION = "moving_target_rotation_v1"
SELECTION_SEED = 20260909
MAX_TRAIN_TARGETS = 4
MIN_HOLDOUT_SCENARIOS = 300
MOVING = "moving_ge_0p5"
SLOW = "slow_lt_0p5"
FIXED_FIELDS = {
    "split": "train", "displacement_m": .2, "perturb_window": 10,
    "ramp_style": "minimum_jerk", "velocity_mode": "residual",
    "allocation_alpha": .5, "allocation_policy": "fixed_symmetric_biend_v1",
    "poison_label": 1,
}
HOLDOUT_GATES = {
    "holdout_incremental_asr": (">=", .30),
    "holdout_lift_over_fixed_v3": (">=", .10),
    "holdout_clean_activation": ("<=", .05),
    "holdout_adjacent_incremental_fp": ("<=", .005),
}


def pair_key(row):
    return (str(row["scenario_id"]), *sorted((int(row["src"]), int(row["dst"]))))


def priority(row):
    return hashlib.sha256(
        f"{VERSION}:{SELECTION_SEED}:{pair_key(row)}".encode()
    ).hexdigest()


def partition_candidates(anchor, candidates, excluded_holdouts=frozenset()):
    """Keep anchor; reserve a new alternative before taking up to 3 train alternatives."""
    anchor_key = pair_key(anchor)
    if len({pair_key(row) for row in candidates}) != len(candidates):
        raise ValueError("duplicate unordered candidate pair")
    if anchor_key not in {pair_key(row) for row in candidates}:
        raise ValueError("anchor missing from eligible candidates")
    if any(row["scenario_id"] != anchor["scenario_id"] for row in candidates):
        raise ValueError("candidate crossed scene boundary")
    if anchor["motion_regime"] == SLOW:
        return [dict(anchor)], None
    if anchor["motion_regime"] != MOVING:
        raise ValueError("unknown motion regime")
    alternatives = sorted(
        (dict(row) for row in candidates
         if pair_key(row) != anchor_key and row["motion_regime"] == MOVING),
        key=priority,
    )
    unseen = [r for r in alternatives if pair_key(r) not in excluded_holdouts]
    # Need anchor + at least one rotation alternative + one held-out pair.
    if len(alternatives) < 2 or not unseen:
        return [dict(anchor)], None
    holdout = unseen[0]
    train = [dict(anchor)] + [
        row for row in alternatives if pair_key(row) != pair_key(holdout)
    ][:MAX_TRAIN_TARGETS - 1]
    return train, holdout


def validate_roles(frame, anchors):
    required = {*FIXED_FIELDS, "scenario_id", "src", "dst", "src_track_id",
                "dst_track_id", "motion_regime", "role", "slot", "graph_sha256",
                "min_endpoint_speed_mps"}
    if required - set(frame.columns):
        raise ValueError(f"missing role columns: {sorted(required-set(frame.columns))}")
    if frame.empty or set(frame.role) - {"train", "holdout"}:
        raise ValueError("empty or unknown target roles")
    for field, expected in FIXED_FIELDS.items():
        if not frame[field].eq(expected).all():
            raise ValueError(f"rotation changed frozen trigger field: {field}")
    for field in ("src", "dst", "slot"):
        values = pd.to_numeric(frame[field], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"noninteger target field: {field}")
    if (frame.src < 0).any() or (frame.dst < 0).any() or (frame.src == frame.dst).any():
        raise ValueError("invalid target nodes")
    speeds = pd.to_numeric(frame.min_endpoint_speed_mps).to_numpy(float)
    if not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError("invalid endpoint speed")
    if not frame.graph_sha256.astype(str).str.fullmatch("[0-9a-f]{64}").all():
        raise ValueError("invalid graph binding")
    keys = [pair_key(row) for row in frame.to_dict("records")]
    if len(set(keys)) != len(keys):
        raise ValueError("unordered train/holdout pair overlap or duplicate")
    anchor_map = {str(r["scenario_id"]): r for r in anchors.to_dict("records")}
    if len(anchor_map) != len(anchors) or set(frame.scenario_id) != set(anchor_map):
        raise ValueError("rotation must preserve every unique anchor scene")
    counts = {}
    for scene, rows in frame.groupby("scenario_id", sort=True):
        training = rows[rows.role.eq("train")].sort_values("slot")
        held = rows[rows.role.eq("holdout")]
        if not 1 <= len(training) <= MAX_TRAIN_TARGETS or training.slot.tolist() != list(range(len(training))):
            raise ValueError("invalid or missing per-scene train slots")
        anchor = anchor_map[scene]
        first = training.iloc[0].to_dict()
        for field in ("src", "dst", "src_track_id", "dst_track_id", "motion_regime"):
            if str(first[field]) != str(anchor[field]):
                raise ValueError(f"slot zero differs from original anchor: {field}")
        if len(training) == 1:
            if len(held):
                raise ValueError("holdout only defined for rotatable scenes")
        elif len(held) != 1 or int(held.iloc[0].slot) != -1:
            raise ValueError("rotatable scene must reserve exactly one holdout")
        expected_regime = MOVING if anchor["motion_regime"] == MOVING else SLOW
        if not rows.motion_regime.eq(expected_regime).all():
            raise ValueError("rotation changed motion-regime quota")
        if expected_regime == SLOW and len(rows) != 1:
            raise ValueError("slow anchors may not rotate")
        if expected_regime == MOVING and not rows.min_endpoint_speed_mps.ge(.5).all():
            raise ValueError("moving slot has a slow endpoint")
        if expected_regime == SLOW and not rows.min_endpoint_speed_mps.lt(.5).all():
            raise ValueError("slow anchor speed mismatch")
        counts[scene] = len(training)
    return {
        "scenarios": len(counts), "rotatable_scenarios": sum(n > 1 for n in counts.values()),
        "train_targets": int(frame.role.eq("train").sum()),
        "holdout_targets": int(frame.role.eq("holdout").sum()),
        "train_slot_histogram": {str(n): sum(v == n for v in counts.values()) for n in range(1,5)},
    }


def epoch_rows(frame, epoch):
    """Epoch 1 is exactly the fixed anchor presentation. No RNG is consumed."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    train = frame[frame.role.eq("train")]
    chosen = []
    for _, rows in train.groupby("scenario_id", sort=True):
        rows = rows.sort_values("slot")
        chosen.append(rows.iloc[(epoch - 1) % len(rows)])
    return pd.DataFrame(chosen).reset_index(drop=True)


def schedule_digest(frame):
    serial = [
        [str(r.scenario_id), int(r.src), int(r.dst), int(r.slot)]
        for r in frame.sort_values("scenario_id").itertuples(index=False)
    ]
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def load_plan(path, *, anchors, graph_manifest_hash, poison_hash):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if (plan.get("version") != VERSION or plan.get("test_access_authorized") is not False
            or plan.get("selection_seed") != SELECTION_SEED
            or plan.get("max_train_targets") != MAX_TRAIN_TARGETS
            or plan.get("holdout_gates") != {k:list(v) for k,v in HOLDOUT_GATES.items()}):
        raise ValueError("rotation plan protocol differs")
    if plan["assets"]["graph_manifest"]["sha256"] != graph_manifest_hash:
        raise ValueError("plan graph manifest binding differs")
    if plan["assets"]["poison_manifest"]["sha256"] != poison_hash:
        raise ValueError("plan poison anchor binding differs")
    for asset in plan["assets"].values():
        if sha256_file(asset["path"]) != asset["sha256"]:
            raise ValueError(f"plan asset changed: {asset['path']}")
    target = plan["targets"]
    # Target file cannot escape the newly prepared plan directory.
    target_path = (path.parent / target["filename"]).resolve()
    if target_path.parent != path.parent or sha256_file(target_path) != target["sha256"]:
        raise ValueError("role table escaped plan or changed")
    frame = pd.read_csv(target_path, dtype={"scenario_id":str,"src_track_id":str,"dst_track_id":str})
    counts = validate_roles(frame, anchors)
    if counts != plan["coverage"]:
        raise ValueError("role-table coverage differs from plan")
    if counts["holdout_targets"] < MIN_HOLDOUT_SCENARIOS:
        raise ValueError("insufficient rotation/holdout coverage; do not train")
    excluded = pd.read_csv(plan["assets"]["previous_diagnostic_targets"]["path"])
    excluded_keys = {pair_key(r) for r in excluded.to_dict("records")}
    if any(pair_key(r) in excluded_keys for r in frame[frame.role.eq("holdout")].to_dict("records")):
        raise ValueError("holdout repeats a previously diagnosed target")
    return frame, plan

