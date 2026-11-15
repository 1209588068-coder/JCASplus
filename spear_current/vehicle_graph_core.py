"""Vehicle graph geometry and feature contract extracted from JCAS graph-v4.

No data loading, split selection, training, or filesystem writes occur here.
The legacy feature formulas are retained for a regular 0.1-second grid:
50 observation frames, 60 future frames, 358 node and 34 edge coordinates.
Future state never enters feature generation or message topology.

Source: legacy/jcas/workflows/graph_builder.py (local JCAS project).
Source SHA-256: 0750a5666b387f49658867faa30cb93c2a7c0f35122a0615228dd1d1db33da6a
Changes: NumPy interpolation for finite-or-NaN observation arrays; array-based
track/graph assembly; missing intersection information remains unknown instead
of being interpreted as false. Source defaults remain available in BuildConfig.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def fill_1d(values: np.ndarray) -> np.ndarray:
    """Linear index interpolation with endpoint fill, matching legacy pandas.

    Input arrays are sanitized by build_track_arrays: missing observations are
    NaN and valid observations are finite. This function rejects infinities.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("fill_1d expects a one-dimensional finite-or-NaN array")
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values)
    grid = np.arange(len(values), dtype=np.float64)
    return np.interp(grid, grid[valid], values[valid])

OBS_LEN = 50


FUTURE_START = 50


FUTURE_LEN = 60


DT = 0.1


EPS = 1e-6


VEHICLE_TYPES = {"vehicle", "bus"}


NODE_FEATURE_NAMES: list[str] = (
    [f"rel_pos_x_t{t:02d}" for t in range(OBS_LEN)]
    + [f"rel_pos_y_t{t:02d}" for t in range(OBS_LEN)]
    + [f"vel_x_t{t:02d}" for t in range(OBS_LEN)]
    + [f"vel_y_t{t:02d}" for t in range(OBS_LEN)]
    + [f"heading_sin_t{t:02d}" for t in range(OBS_LEN)]
    + [f"heading_cos_t{t:02d}" for t in range(OBS_LEN)]
    + [f"observed_valid_t{t:02d}" for t in range(OBS_LEN)]
    + [
        "speed_current",
        "speed_mean",
        "speed_std",
        "acc_std",
        "lateral_acc_std",
        "heading_rate_std",
        "observed_fraction",
        "is_focal",
    ]
)


EDGE_FEATURE_NAMES: list[str] = [
    "rel_x_current",
    "rel_y_current",
    "d_current",
    "rel_vx_current",
    "rel_vy_current",
    "closing_speed_current",
    "d_min_obs",
    "d_mean_obs",
    "d_std_obs",
    "d_trend",
    "d_trend_early",
    "d_trend_late",
    "closing_speed_mean",
    "closing_speed_max",
    "closing_speed_std",
    "closing_speed_late",
    "closing_speed_trend",
    "heading_diff_sin",
    "heading_diff_cos",
    "src_heading_to_dst_sin",
    "src_heading_to_dst_cos",
    "dst_heading_to_src_sin",
    "dst_heading_to_src_cos",
    "src_velocity_toward_dst",
    "dst_velocity_toward_src",
    "speed_delta_src_minus_dst",
    "src_speed_current",
    "dst_speed_current",
    "src_speed_mean",
    "dst_speed_mean",
    "src_acc_std",
    "dst_acc_std",
    "src_lateral_acc_std",
    "dst_lateral_acc_std",
]


MAP_UNKNOWN = 0


MAP_SAME_LANE = 1


MAP_LATERAL_NEIGHBOR = 2


MAP_TOPOLOGY_CONNECTED = 3


MAP_INTERSECTION_CONFLICT = 4


MAP_GEOMETRY_ONLY = 5


MAP_BLOCKED_OR_UNRELATED = 6


MAP_RELATION_NAMES = {
    MAP_UNKNOWN: "unknown",
    MAP_SAME_LANE: "same_lane",
    MAP_LATERAL_NEIGHBOR: "lateral_neighbor",
    MAP_TOPOLOGY_CONNECTED: "topology_connected",
    MAP_INTERSECTION_CONFLICT: "intersection_conflict",
    MAP_GEOMETRY_ONLY: "geometry_only",
    MAP_BLOCKED_OR_UNRELATED: "blocked_or_unrelated",
}


@dataclass(frozen=True)
class BuildConfig:
    data_root: str = ""
    output_dir: str = ""
    radius_m: float = 60.0
    future_risk_distance_m: float = 10.0
    min_observed: int = 40
    min_last_observed_timestep: int = 48
    min_pair_future: int = 30
    max_speed_mps: float = 40.0
    max_acc_mps2: float = 10.0
    max_pos_jumps: int = 2
    max_vel_jumps: int = 2
    max_heading_jumps: int = 3
    jump_sigma: float = 5.0
    seed: int = 20260621
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_group_metadata: str = (
        "record/v3/"
        "duplicate_scene_check_overlap_v4.parquet"
    )
    candidate_close_distance_m: float = 35.0
    candidate_recent_close_distance_m: float = 30.0
    candidate_far_distance_m: float = 40.0
    candidate_dynamic_distance_m: float = 50.0
    candidate_approach_trend_m_per_step: float = -0.02
    candidate_receding_trend_m_per_step: float = 0.03
    candidate_closing_speed_mps: float = 0.5
    candidate_closing_speed_max_mps: float = 1.5
    candidate_perpendicular_min_distance_m: float = 15.0
    strict_pair_future: int = 50
    strict_future_last_timestep: int = 100
    lane_max_distance_m: float = 8.0
    perpendicular_lane_min_deg: float = 45.0
    perpendicular_lane_max_deg: float = 135.0
    overwrite: bool = False


@dataclass
class TrackData:
    track_id: str
    object_type: str
    object_category: int
    is_focal: bool
    obs_pos_raw: np.ndarray
    obs_vel_raw: np.ndarray
    obs_heading_raw: np.ndarray
    obs_valid: np.ndarray
    fut_pos_raw: np.ndarray
    fut_valid: np.ndarray
    obs_pos: np.ndarray
    obs_vel: np.ndarray
    obs_heading: np.ndarray
    node_feature: np.ndarray
    speed_current: float
    speed_mean: float
    acc_std: float
    lateral_acc_std: float
    skip_reason: str | None = None


@dataclass
class LaneData:
    lane_id: int
    points: np.ndarray
    headings: np.ndarray
    is_intersection: bool | None
    lateral_neighbors: set[int]
    topology_neighbors: set[int]


def detect_jumps(values: np.ndarray, threshold_sigma: float = 5.0) -> int:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0
    diffs = np.abs(np.diff(values))
    # A mean/std cutoff masks multiple large jumps because the jumps inflate the
    # very scale used to detect them.  Median/MAD remains stable under several
    # outliers and makes max_*_jumps an effective constraint.
    center = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - center)))
    q25, q75 = np.quantile(diffs, [0.25, 0.75])
    robust_sigma = max(1.4826 * mad, float(q75 - q25) / 1.349, 1e-3)
    cutoff = float(center + threshold_sigma * robust_sigma)
    return int(np.sum(diffs > cutoff))


def fill_heading(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values, dtype=np.float64)
    unwrapped = values.copy()
    unwrapped[valid] = np.unwrap(unwrapped[valid])
    filled = fill_1d(unwrapped)
    return np.arctan2(np.sin(filled), np.cos(filled))


def safe_slope(t: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    x = t[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom < EPS:
        return 0.0
    return float(np.sum(x * (yy - float(np.mean(yy)))) / denom)


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(np.asarray(a) - np.asarray(b)), np.cos(np.asarray(a) - np.asarray(b)))


def abs_angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.abs(angle_diff(a, b))


def finite_or_zero(x: float) -> float:
    return float(x) if math.isfinite(float(x)) else 0.0


def nearest_lane_features(current_pos: np.ndarray, lanes: dict[int, LaneData]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = current_pos.shape[0]
    lane_ids = np.full(n, -1, dtype=np.int64)
    lane_dist = np.full(n, np.inf, dtype=np.float32)
    lane_heading = np.zeros(n, dtype=np.float32)
    lane_is_intersection = np.full(n, -1, dtype=np.int8)
    if not lanes:
        return lane_ids, lane_dist, lane_heading, lane_is_intersection

    lane_items = list(lanes.items())
    for i, pos in enumerate(current_pos):
        best_dist = np.inf
        best_lane: LaneData | None = None
        best_idx = 0
        for _, lane in lane_items:
            delta = lane.points - pos[None, :]
            d2 = np.sum(delta * delta, axis=1)
            idx = int(np.argmin(d2))
            dist = float(math.sqrt(float(d2[idx])))
            if dist < best_dist:
                best_dist = dist
                best_lane = lane
                best_idx = idx
        if best_lane is not None:
            lane_ids[i] = best_lane.lane_id
            lane_dist[i] = best_dist
            lane_heading[i] = float(best_lane.headings[best_idx])
            lane_is_intersection[i] = -1 if best_lane.is_intersection is None else int(best_lane.is_intersection)
    return lane_ids, lane_dist, lane_heading, lane_is_intersection


def make_node_feature(
    pos: np.ndarray,
    vel: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    is_focal: bool,
) -> tuple[np.ndarray, float, float, float, float]:
    current_pos = pos[-1]
    rel_pos = pos - current_pos[None, :]
    speed = np.linalg.norm(vel, axis=1)
    valid_idx = np.flatnonzero(valid)

    if valid_idx.size >= 2:
        dt = np.maximum(np.diff(valid_idx).astype(np.float64) * DT, EPS)
        vel_valid = vel[valid_idx]
        heading_valid = heading[valid_idx]
        acc = np.diff(vel_valid, axis=0) / dt[:, None]
        acc_norm = np.linalg.norm(acc, axis=1)
        heading_rate = np.diff(np.unwrap(heading_valid)) / dt
        lateral_dirs = np.column_stack([-np.sin(heading_valid[1:]), np.cos(heading_valid[1:])])
        lateral_acc = np.sum(acc * lateral_dirs, axis=1)
    else:
        acc_norm = np.zeros(0, dtype=np.float64)
        heading_rate = np.zeros(0, dtype=np.float64)
        lateral_acc = np.zeros(0, dtype=np.float64)

    speed_current = float(speed[-1]) if speed.size else 0.0
    speed_mean = float(np.mean(speed[valid_idx])) if valid_idx.size else 0.0
    speed_std = float(np.std(speed[valid_idx])) if valid_idx.size else 0.0
    acc_std = float(np.std(acc_norm)) if acc_norm.size else 0.0
    lateral_acc_std = float(np.std(lateral_acc)) if lateral_acc.size else 0.0
    heading_rate_std = float(np.std(heading_rate)) if heading_rate.size else 0.0

    compact = np.asarray(
        [
            speed_current,
            speed_mean,
            speed_std,
            acc_std,
            lateral_acc_std,
            heading_rate_std,
            np.mean(valid.astype(np.float32)),
            1.0 if is_focal else 0.0,
        ],
        dtype=np.float64,
    )
    feature = np.concatenate(
        [
            rel_pos[:, 0],
            rel_pos[:, 1],
            vel[:, 0],
            vel[:, 1],
            np.sin(heading),
            np.cos(heading),
            valid.astype(np.float64),
            compact,
        ],
        axis=0,
    )
    return (
        np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0),
        finite_or_zero(speed_current),
        finite_or_zero(speed_mean),
        finite_or_zero(acc_std),
        finite_or_zero(lateral_acc_std),
    )


def pair_closing_speed(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(rel_pos, axis=-1)
    unit = rel_pos / np.maximum(dist[..., None], EPS)
    return -np.sum(rel_vel * unit, axis=-1)


def compute_edge_features(
    src: int,
    dst: int,
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
    obs_heading: np.ndarray,
    obs_valid: np.ndarray,
    node_speed_current: np.ndarray,
    node_speed_mean: np.ndarray,
    node_acc_std: np.ndarray,
    node_lateral_acc_std: np.ndarray,
) -> np.ndarray:
    rel_current = obs_pos[dst, -1] - obs_pos[src, -1]
    rel_vel_current = obs_vel[dst, -1] - obs_vel[src, -1]
    d_current = float(np.linalg.norm(rel_current))
    closing_current = float(pair_closing_speed(rel_current[None, :], rel_vel_current[None, :])[0])

    pair_valid = obs_valid[src] & obs_valid[dst]
    t_all = np.arange(OBS_LEN, dtype=np.float64)
    if pair_valid.sum() >= 2:
        rel_seq = obs_pos[dst] - obs_pos[src]
        vel_seq = obs_vel[dst] - obs_vel[src]
        d_seq = np.linalg.norm(rel_seq, axis=1)
        cl_seq = pair_closing_speed(rel_seq, vel_seq)
        valid_d = d_seq[pair_valid]
        valid_cl = cl_seq[pair_valid]
        d_min = float(np.min(valid_d))
        d_mean = float(np.mean(valid_d))
        d_std = float(np.std(valid_d))
        d_trend = safe_slope(t_all[pair_valid], d_seq[pair_valid])
        early_mask = pair_valid & (t_all < OBS_LEN // 2)
        valid_ts = t_all[pair_valid]
        if valid_ts.size >= 20:
            late_ts = valid_ts[-20:]
            late_mask = np.isin(t_all, late_ts) & pair_valid
        else:
            late_mask = pair_valid
        d_trend_early = safe_slope(t_all[early_mask], d_seq[early_mask])
        d_trend_late = safe_slope(t_all[late_mask], d_seq[late_mask])
        cl_mean = float(np.mean(valid_cl))
        cl_max = float(np.max(valid_cl))
        cl_std = float(np.std(valid_cl))
        cl_late = float(np.mean(cl_seq[late_mask])) if late_mask.sum() else 0.0
        cl_trend = safe_slope(t_all[pair_valid], cl_seq[pair_valid])
    else:
        d_min = d_mean = d_current
        d_std = d_trend = d_trend_early = d_trend_late = 0.0
        cl_mean = cl_max = cl_std = cl_late = cl_trend = 0.0

    hdiff = float(angle_diff(obs_heading[dst, -1], obs_heading[src, -1]))
    unit_src_to_dst = rel_current / max(d_current, EPS)
    unit_dst_to_src = -unit_src_to_dst
    bearing_src_to_dst = float(math.atan2(rel_current[1], rel_current[0]))
    bearing_dst_to_src = float(math.atan2(-rel_current[1], -rel_current[0]))
    src_heading_to_dst = float(angle_diff(bearing_src_to_dst, float(obs_heading[src, -1])))
    dst_heading_to_src = float(angle_diff(bearing_dst_to_src, float(obs_heading[dst, -1])))
    src_velocity_toward_dst = float(np.dot(obs_vel[src, -1], unit_src_to_dst))
    dst_velocity_toward_src = float(np.dot(obs_vel[dst, -1], unit_dst_to_src))

    return np.asarray(
        [
            rel_current[0],
            rel_current[1],
            d_current,
            rel_vel_current[0],
            rel_vel_current[1],
            closing_current,
            d_min,
            d_mean,
            d_std,
            d_trend,
            d_trend_early,
            d_trend_late,
            cl_mean,
            cl_max,
            cl_std,
            cl_late,
            cl_trend,
            math.sin(hdiff),
            math.cos(hdiff),
            math.sin(src_heading_to_dst),
            math.cos(src_heading_to_dst),
            math.sin(dst_heading_to_src),
            math.cos(dst_heading_to_src),
            src_velocity_toward_dst,
            dst_velocity_toward_src,
            node_speed_current[src] - node_speed_current[dst],
            node_speed_current[src],
            node_speed_current[dst],
            node_speed_mean[src],
            node_speed_mean[dst],
            node_acc_std[src],
            node_acc_std[dst],
            node_lateral_acc_std[src],
            node_lateral_acc_std[dst],
        ],
        dtype=np.float32,
    )


def future_min_distance(
    src: int,
    dst: int,
    fut_pos: np.ndarray,
    fut_valid: np.ndarray,
) -> tuple[float, int, int, int, int, int, float]:
    pair_valid = fut_valid[src] & fut_valid[dst]
    count = int(pair_valid.sum())
    if count == 0:
        return float("inf"), -1, 0, -1, -1, 0, 0.0
    rel = fut_pos[dst, pair_valid] - fut_pos[src, pair_valid]
    dist = np.linalg.norm(rel, axis=1)
    arg = int(np.argmin(dist))
    timesteps = np.flatnonzero(pair_valid) + FUTURE_START
    first_valid = int(timesteps[0])
    last_valid = int(timesteps[-1])
    valid_span = int(last_valid - first_valid + 1)
    valid_fraction = float(count / FUTURE_LEN)
    return float(dist[arg]), int(timesteps[arg]), count, first_valid, last_valid, valid_span, valid_fraction


def lane_pair_features(
    src: int,
    dst: int,
    edge_d_current: float,
    node_lane_id: np.ndarray,
    node_lane_distance: np.ndarray,
    node_lane_heading: np.ndarray,
    node_lane_is_intersection: np.ndarray,
    lanes: dict[int, LaneData],
    cfg: BuildConfig,
) -> tuple[bool, bool, bool, float, bool, bool, bool, bool, int]:
    src_lane = int(node_lane_id[src])
    dst_lane = int(node_lane_id[dst])
    src_known = src_lane >= 0 and float(node_lane_distance[src]) <= cfg.lane_max_distance_m
    dst_known = dst_lane >= 0 and float(node_lane_distance[dst]) <= cfg.lane_max_distance_m
    same_lane = bool(src_known and dst_known and src_lane == dst_lane)
    lateral_neighbor_lane = False
    connected_lane = False
    if src_known and dst_known and src_lane in lanes and dst_lane in lanes:
        lateral_neighbor_lane = (
            dst_lane in lanes[src_lane].lateral_neighbors
            or src_lane in lanes[dst_lane].lateral_neighbors
        )
        connected_lane = (
            dst_lane in lanes[src_lane].topology_neighbors
            or src_lane in lanes[dst_lane].topology_neighbors
        )
    lane_heading_diff = float(abs_angle_diff(float(node_lane_heading[src]), float(node_lane_heading[dst])))
    intersection_known = bool(node_lane_is_intersection[src] >= 0 and node_lane_is_intersection[dst] >= 0)
    either_intersection = bool(node_lane_is_intersection[src] == 1 or node_lane_is_intersection[dst] == 1)

    if not (src_known and dst_known):
        # Missing or distant lane assignment is ambiguous, not proof of non-interaction.
        return (
            same_lane,
            lateral_neighbor_lane,
            connected_lane,
            lane_heading_diff,
            either_intersection,
            True,
            False,
            True,
            MAP_UNKNOWN,
        )

    low = math.radians(cfg.perpendicular_lane_min_deg)
    high = math.radians(cfg.perpendicular_lane_max_deg)
    perpendicular_unrelated = (
        low <= lane_heading_diff <= high
        and not same_lane
        and not lateral_neighbor_lane
        and not connected_lane
        and not either_intersection
        and edge_d_current > cfg.candidate_perpendicular_min_distance_m
    )
    if same_lane:
        map_relation_type = MAP_SAME_LANE
        map_candidate_flag = True
    elif lateral_neighbor_lane:
        map_relation_type = MAP_LATERAL_NEIGHBOR
        map_candidate_flag = True
    elif connected_lane:
        map_relation_type = MAP_TOPOLOGY_CONNECTED
        map_candidate_flag = True
    elif either_intersection:
        map_relation_type = MAP_INTERSECTION_CONFLICT
        map_candidate_flag = True
    elif perpendicular_unrelated and not intersection_known:
        # An absent WOMD intersection field is not evidence of non-intersection.
        return (same_lane, lateral_neighbor_lane, connected_lane, lane_heading_diff,
                either_intersection, True, False, True, MAP_UNKNOWN)
    elif perpendicular_unrelated:
        map_relation_type = MAP_BLOCKED_OR_UNRELATED
        map_candidate_flag = False
    else:
        map_relation_type = MAP_GEOMETRY_ONLY
        map_candidate_flag = True
    return (
        same_lane,
        lateral_neighbor_lane,
        connected_lane,
        lane_heading_diff,
        either_intersection,
        map_candidate_flag,
        True,
        False,
        map_relation_type,
    )


def kinematic_candidate_details(
    edge_feature: np.ndarray, cfg: BuildConfig
) -> tuple[bool, bool, bool, bool]:
    """Motion-based interaction feasibility filter.

    Units:
      d_current, d_min_obs, candidate_close_distance_m         → meters
      d_trend, d_trend_late                                    → meters / timestep (≈ 0.1 s)
        e.g. -0.02 m/ts ≈ -0.2 m/s
      closing_current, closing_max, candidate_closing_speed_*  → m/s
    """
    names = EDGE_FEATURE_NAMES
    d_current = float(edge_feature[names.index("d_current")])
    d_min_obs = float(edge_feature[names.index("d_min_obs")])
    d_trend = float(edge_feature[names.index("d_trend")])
    d_trend_late = float(edge_feature[names.index("d_trend_late")])
    closing_current = float(edge_feature[names.index("closing_speed_current")])
    closing_max = float(edge_feature[names.index("closing_speed_max")])

    near_gate = (
        d_current <= cfg.candidate_close_distance_m
        or d_min_obs <= cfg.candidate_recent_close_distance_m
    )
    dynamic_gate = (
        d_current <= cfg.candidate_dynamic_distance_m
        and (
            d_trend <= cfg.candidate_approach_trend_m_per_step
            or d_trend_late <= cfg.candidate_approach_trend_m_per_step
            or closing_current >= cfg.candidate_closing_speed_mps
            or closing_max >= cfg.candidate_closing_speed_max_mps
        )
    )
    obviously_far_receding = (
        d_current > cfg.candidate_far_distance_m
        and d_trend_late > cfg.candidate_receding_trend_m_per_step
        and closing_current < -0.2
        and closing_max < cfg.candidate_closing_speed_mps
    )
    candidate = bool((near_gate or dynamic_gate) and not obviously_far_receding)
    return candidate, bool(near_gate), bool(dynamic_gate), bool(obviously_far_receding)


def kinematic_candidate(edge_feature: np.ndarray, cfg: BuildConfig) -> bool:
    """Backward-compatible boolean view of the soft task-candidate gate."""
    return kinematic_candidate_details(edge_feature, cfg)[0]


def build_track_arrays(
    track_id: str,
    object_type: str,
    object_category: int,
    is_focal: bool,
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
    obs_heading: np.ndarray,
    obs_valid: np.ndarray,
    fut_pos: np.ndarray,
    fut_valid: np.ndarray,
    cfg: BuildConfig = BuildConfig(),
) -> TrackData:
    """Adapt regular-grid source arrays using the original observation filters.

    Input timestamps must already be on the recorded 10 Hz observation/future
    grid. Raw masks describe available source measurements, not filled values.
    No future availability criterion is allowed to remove a vehicle node.
    """
    obs_pos = np.array(obs_pos, dtype=np.float64, copy=True)
    obs_vel = np.array(obs_vel, dtype=np.float64, copy=True)
    obs_heading = np.array(obs_heading, dtype=np.float64, copy=True)
    obs_valid = np.array(obs_valid, dtype=bool, copy=True)
    fut_pos = np.array(fut_pos, dtype=np.float64, copy=True)
    fut_valid = np.array(fut_valid, dtype=bool, copy=True)
    if (obs_pos.shape != (OBS_LEN, 2) or obs_vel.shape != (OBS_LEN, 2)
            or obs_heading.shape != (OBS_LEN,) or obs_valid.shape != (OBS_LEN,)
            or fut_pos.shape != (FUTURE_LEN, 2) or fut_valid.shape != (FUTURE_LEN,)):
        raise ValueError("Track arrays do not match the 50 observation / 60 future contract")
    if not str(track_id):
        raise ValueError("Empty track identity")
    obs_valid &= (np.isfinite(obs_pos).all(axis=1)
                  & np.isfinite(obs_vel).all(axis=1) & np.isfinite(obs_heading))
    fut_valid &= np.isfinite(fut_pos).all(axis=1)
    obs_pos[~obs_valid] = np.nan
    obs_vel[~obs_valid] = np.nan
    obs_heading[~obs_valid] = np.nan
    fut_pos[~fut_valid] = np.nan

    skip = None
    if object_type not in VEHICLE_TYPES:
        skip = "non_vehicle"
    elif int(obs_valid.sum()) < cfg.min_observed:
        skip = "obs_too_short"
    elif int(np.flatnonzero(obs_valid).max()) < cfg.min_last_observed_timestep:
        skip = "no_last_obs"

    obs_pos_f = np.column_stack([fill_1d(obs_pos[:, 0]), fill_1d(obs_pos[:, 1])])
    obs_vel_f = np.column_stack([fill_1d(obs_vel[:, 0]), fill_1d(obs_vel[:, 1])])
    obs_heading_f = fill_heading(obs_heading)
    if skip is None:
        valid_t = np.flatnonzero(obs_valid)
        pos_jumps = detect_jumps(obs_pos[valid_t, 0], cfg.jump_sigma) + detect_jumps(obs_pos[valid_t, 1], cfg.jump_sigma)
        vel_jumps = detect_jumps(obs_vel[valid_t, 0], cfg.jump_sigma) + detect_jumps(obs_vel[valid_t, 1], cfg.jump_sigma)
        heading_jumps = detect_jumps(np.unwrap(obs_heading[valid_t]), cfg.jump_sigma)
        speed = np.linalg.norm(obs_vel[valid_t], axis=1)
        speed_extreme = bool(speed.size and float(np.nanmax(speed)) > cfg.max_speed_mps)
        acc_extreme = False
        if valid_t.size >= 2:
            dt = np.maximum(np.diff(valid_t).astype(np.float64) * DT, EPS)
            acc = np.diff(obs_vel[valid_t], axis=0) / dt[:, None]
            acc_norm = np.linalg.norm(acc, axis=1)
            acc_extreme = bool(acc_norm.size and float(np.nanmax(acc_norm)) > cfg.max_acc_mps2)
        if pos_jumps > cfg.max_pos_jumps:
            skip = "pos_jumps"
        elif vel_jumps > cfg.max_vel_jumps:
            skip = "vel_jumps"
        elif heading_jumps > cfg.max_heading_jumps:
            skip = "heading_jumps"
        elif speed_extreme:
            skip = "speed_extreme"
        elif acc_extreme:
            skip = "acc_extreme"
    node_feature, speed_current, speed_mean, acc_std, lateral_acc_std = make_node_feature(
        obs_pos_f, obs_vel_f, obs_heading_f, obs_valid, is_focal)
    return TrackData(
        track_id=str(track_id), object_type=str(object_type), object_category=int(object_category),
        is_focal=bool(is_focal), obs_pos_raw=obs_pos.astype(np.float32),
        obs_vel_raw=obs_vel.astype(np.float32), obs_heading_raw=obs_heading.astype(np.float32),
        obs_valid=obs_valid, fut_pos_raw=fut_pos.astype(np.float32), fut_valid=fut_valid,
        obs_pos=obs_pos_f.astype(np.float32), obs_vel=obs_vel_f.astype(np.float32),
        obs_heading=obs_heading_f.astype(np.float32), node_feature=node_feature.astype(np.float32),
        speed_current=speed_current, speed_mean=speed_mean, acc_std=acc_std,
        lateral_acc_std=lateral_acc_std, skip_reason=skip)


def build_graph_arrays(
    tracks: list[TrackData],
    lanes: dict[int, LaneData],
    cfg: BuildConfig = BuildConfig(),
) -> dict[str, np.ndarray]:
    """Build an observed radius graph; return labels and topology separately.

    Caller supplies only tracks that passed the observation filters and records
    all exclusion counts. The caller owns scenario identity, source provenance,
    dynamic labels, file publication and split assignment. Proximity labels
    here are legacy diagnostics; the final task uses risk_labels.py (5/1/4).
    """
    if len(tracks) < 2:
        raise ValueError("At least two observation-qualified tracks required")
    if len({t.track_id for t in tracks}) != len(tracks):
        raise ValueError("Duplicate track IDs")
    if any(t.skip_reason is not None for t in tracks):
        raise ValueError("Input contains an observation-filtered track")
    if not np.isfinite(cfg.radius_m) or cfg.radius_m <= 0:
        raise ValueError("Invalid radius")

    def stacked(name: str, dtype=None):
        return np.asarray([getattr(t, name) for t in tracks], dtype=dtype)

    obs_pos = stacked("obs_pos", np.float32)
    obs_vel = stacked("obs_vel", np.float32)
    obs_heading = stacked("obs_heading", np.float32)
    obs_valid = stacked("obs_valid", bool)
    fut_pos = stacked("fut_pos_raw", np.float32)
    fut_valid = stacked("fut_valid", bool)
    speed_current = stacked("speed_current", np.float32)
    speed_mean = stacked("speed_mean", np.float32)
    acc_std = stacked("acc_std", np.float32)
    lateral_acc_std = stacked("lateral_acc_std", np.float32)
    lane_id, lane_dist, lane_heading, intersection_code = nearest_lane_features(obs_pos[:, -1], lanes)

    n = len(tracks)
    diff = obs_pos[None, :, -1, :] - obs_pos[:, None, -1, :]
    distances = np.linalg.norm(diff, axis=-1)
    edges = np.argwhere((distances <= cfg.radius_m) & ~np.eye(n, dtype=bool))
    e = len(edges)
    edge_attr = np.empty((e, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    fmin = np.empty(e, np.float32)
    ftime = np.empty(e, np.int16)
    fcount = np.empty(e, np.int16)
    ffirst = np.empty(e, np.int16)
    flast = np.empty(e, np.int16)
    fspan = np.empty(e, np.int16)
    ffraction = np.empty(e, np.float32)
    kine = np.empty((e, 4), bool)
    map_bool = np.empty((e, 7), bool)
    map_heading = np.empty(e, np.float32)
    map_relation = np.empty(e, np.int8)
    for k, (src, dst) in enumerate(edges):
        src, dst = int(src), int(dst)
        edge_attr[k] = compute_edge_features(src, dst, obs_pos, obs_vel, obs_heading,
                                             obs_valid, speed_current, speed_mean,
                                             acc_std, lateral_acc_std)
        fmin[k], ftime[k], fcount[k], ffirst[k], flast[k], fspan[k], ffraction[k] = future_min_distance(
            src, dst, fut_pos, fut_valid)
        kine[k] = kinematic_candidate_details(edge_attr[k], cfg)
        lane_values = lane_pair_features(src, dst, float(edge_attr[k, 2]), lane_id,
                                         lane_dist, lane_heading, intersection_code, lanes, cfg)
        # same / lateral / connected / heading / intersection / allowed / known / unknown / type
        map_bool[k] = [lane_values[i] for i in (0, 1, 2, 4, 5, 6, 7)]
        map_heading[k], map_relation[k] = lane_values[3], lane_values[8]

    computable = fcount >= cfg.min_pair_future
    strict = computable & (fcount >= cfg.strict_pair_future) & (flast >= cfg.strict_future_last_timestep)
    physical = map_bool[:, 4].copy()
    task = physical & kine[:, 0]
    proximity = (computable & (fmin <= cfg.future_risk_distance_m)).astype(np.int8)
    graph = {
        "x_node": stacked("node_feature", np.float32),
        "node_track_ids": np.asarray([t.track_id for t in tracks], dtype=str),
        "node_object_type": np.asarray([t.object_type for t in tracks], dtype=str),
        "node_object_category": stacked("object_category", np.int16),
        "node_is_focal": stacked("is_focal", bool),
        "observed_positions": stacked("obs_pos_raw", np.float32),
        "observed_velocities": stacked("obs_vel_raw", np.float32),
        "observed_headings": stacked("obs_heading_raw", np.float32),
        "observed_positions_filled": obs_pos,
        "observed_velocities_filled": obs_vel,
        "observed_headings_filled": obs_heading,
        "observed_valid_mask": obs_valid,
        "future_positions": fut_pos,
        "future_valid_mask": fut_valid,
        "node_nearest_lane_id": lane_id,
        "node_nearest_lane_distance": lane_dist,
        "node_nearest_lane_heading": lane_heading,
        "node_nearest_lane_is_intersection": intersection_code == 1,
        "node_lane_intersection_known": intersection_code >= 0,
        "num_lanes": np.asarray(len(lanes), dtype=np.int32),
        "edge_index": edges.T.copy().astype(np.int64),
        "edge_attr": edge_attr,
        "edge_label_proximity": proximity,
        "edge_label_proximity_task_diagnostic": (proximity.astype(bool) & task).astype(np.int8),
        "label_computable_mask": computable,
        "label_strict_mask": strict,
        "raw_edge_mask": np.ones(e, bool),
        "physical_possible_mask": physical,
        "message_passing_edge_mask": physical.copy(),
        "hard_pruned_edge_mask": ~physical,
        "physically_impossible_mask": ~physical,
        "task_candidate_mask": task,
        "soft_task_excluded_mask": physical & ~task,
        "task_excluded_mask": ~task,
        "supervision_edge_mask": task & computable,
        "interaction_candidate_mask": task.copy(),
        "impossible_interaction_mask": ~physical,
        "kinematic_candidate_mask": kine[:, 0].copy(),
        "kinematic_near_gate_mask": kine[:, 1].copy(),
        "kinematic_dynamic_gate_mask": kine[:, 2].copy(),
        "kinematic_far_receding_mask": kine[:, 3].copy(),
        "map_candidate_mask": physical.copy(),
        "map_known_mask": map_bool[:, 5].copy(),
        "map_unknown_mask": map_bool[:, 6].copy(),
        "map_blocked_mask": map_relation == MAP_BLOCKED_OR_UNRELATED,
        "map_relation_type": map_relation,
        "map_relation_type_names": np.asarray(list(MAP_RELATION_NAMES.values()), dtype=str),
        "edge_same_lane": map_bool[:, 0].copy(),
        "edge_lateral_neighbor_lane": map_bool[:, 1].copy(),
        "edge_connected_lane": map_bool[:, 2].copy(),
        "edge_lane_heading_diff": map_heading,
        "edge_either_intersection": map_bool[:, 3].copy(),
        "edge_intersection_known": ((intersection_code[edges[:, 0]] >= 0)
                                    & (intersection_code[edges[:, 1]] >= 0)),
        "edge_src_track_id": np.asarray([tracks[int(s)].track_id for s, _ in edges], dtype=str),
        "edge_dst_track_id": np.asarray([tracks[int(d)].track_id for _, d in edges], dtype=str),
        "future_min_distance": fmin,
        "future_min_distance_timestep": ftime,
        "pair_future_valid_count": fcount,
        "future_first_valid_timestep": ffirst,
        "future_last_valid_timestep": flast,
        "future_valid_span": fspan,
        "future_valid_fraction": ffraction,
        "node_feature_names": np.asarray(NODE_FEATURE_NAMES, dtype=str),
        "edge_feature_names": np.asarray(EDGE_FEATURE_NAMES, dtype=str),
    }
    for name in ("d_current", "d_min_obs", "d_mean_obs", "d_trend", "d_trend_early",
                 "d_trend_late", "closing_speed_current", "closing_speed_mean", "closing_speed_max"):
        graph[name] = edge_attr[:, EDGE_FEATURE_NAMES.index(name)].copy()
    return graph
