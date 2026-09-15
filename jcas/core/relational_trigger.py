"""Independent, observed-only relational trajectory feasibility primitives.

No model, optimizer, labels, future data, or frozen transform is modified here.
The common-mode controller minimizes *geometric* collateral, not predictive FP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from jcas.core.motion_normalized_features import edge_features_for_mode, MOTION_NORMALIZED_EDGE_FEATURE_MODE
from jcas.core.trajectory_trigger import TriggerSpec, minimum_jerk_progress, _node_motion_statistics
from jcas.workflows.graph_builder import OBS_LEN, compute_edge_features

VERSION = "relational_trajectory_feasibility_v1"
DT = 0.1
WINDOW = 10
COMMON_CAP_M = 0.02
RELATIVE_BUDGET_FRACTION = 0.9


@dataclass(frozen=True)
class Budget:
    endpoint_position_m: float
    relative_position_m: float
    induced_speed_mps: float
    induced_acceleration_mps2: float
    induced_jerk_mps3: float


def derivatives(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward differences with a zero residual before the observed window."""
    v = np.diff(values, axis=0, prepend=np.zeros_like(values[:1])) / DT
    a = np.diff(v, axis=0, prepend=np.zeros_like(v[:1])) / DT
    j = np.diff(a, axis=0, prepend=np.zeros_like(a[:1])) / DT
    return v, a, j


def max_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values, axis=-1).max(initial=0.0))


def reference_budget() -> Budget:
    """Exact commanded residual limits from the legacy 0.1 m endpoint ramp.

    These are incremental numerical bounds, NOT absolute vehicle safety limits.
    They are derived without opening any frozen result/test/trajectory assets.
    """
    ramp = 0.1 * minimum_jerk_progress(TriggerSpec()).astype(np.float64)
    v, a, j = derivatives(ramp[:, None])
    return Budget(0.1, 0.2, max_norm(v), max_norm(a), max_norm(j))


def temporal_basis() -> np.ndarray:
    """Three fixed smooth polynomial basis functions on ten intervals.

    Deliberately not advertised as a novel spline or learned trigger. The first
    column includes the old ramp, so the search space contains the old baseline.
    """
    t = np.arange(1, WINDOW + 1, dtype=np.float64) / WINDOW
    ramp = minimum_jerk_progress(TriggerSpec()).astype(np.float64)
    return np.column_stack((ramp, ramp * (1 - t)**2, ramp * (1 - t)**2 * (2*t - 1)))


def budget_measurements(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    v1, a1, j1 = derivatives(first)
    v2, a2, j2 = derivatives(second)
    return {
        "endpoint_position_m": max(max_norm(first), max_norm(second)),
        "relative_position_m": max_norm(second - first),
        "induced_speed_mps": max(max_norm(v1), max_norm(v2)),
        "induced_acceleration_mps2": max(max_norm(a1), max_norm(a2)),
        "induced_jerk_mps3": max(max_norm(j1), max_norm(j2)),
    }


def budget_ok(first: np.ndarray, second: np.ndarray, budget: Budget) -> bool:
    measured = budget_measurements(first, second)
    return all(np.isfinite(value) and value <= asdict(budget)[key] + 1e-9
               for key, value in measured.items())


def make_template(coefficients: np.ndarray, *, relative_budget_fraction: float = RELATIVE_BUDGET_FRACTION) -> dict[str, Any]:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (3, 2) or not np.isfinite(coefficients).all():
        raise ValueError("template requires finite 3x2 coefficients")
    if not np.isfinite(relative_budget_fraction) or not 0 < relative_budget_fraction <= 1:
        raise ValueError("relative budget fraction must be in (0, 1]")
    wave = temporal_basis() @ coefficients
    if max_norm(wave) <= 1e-12:
        raise ValueError("zero template")
    budget = reference_budget()
    measured = budget_measurements(-wave/2, wave/2)
    # One global scale for the whole template, never per scene or speed group.
    scale = min(asdict(budget)[key] / value for key, value in measured.items() if value > 1e-12)
    scale *= relative_budget_fraction * (1 - 1e-10)
    wave = wave * scale
    return {"coefficients": coefficients.tolist(), "global_scale": float(scale),
            "relative_budget_fraction": float(relative_budget_fraction),
            "relative_wave_m": wave.tolist(), "budget": asdict(budget)}


def template_bank() -> dict[str, dict[str, Any]]:
    templates = {"legacy_ramp": make_template(np.array([[-1., 0.], [0., 0.], [0., 0.]]), relative_budget_fraction=1.)}
    for seed in (20260912, 20260913, 20260914):
        templates[f"random_{seed}"] = make_template(np.random.default_rng(seed).normal(size=(3, 2)))
    return templates


def observed_inputs(graph: Any, src: int, dst: int):
    if not isinstance(src, (int, np.integer)) or not isinstance(dst, (int, np.integer)):
        raise ValueError("endpoints must be integer indices")
    positions = np.asarray(graph["observed_positions_filled"], dtype=np.float64)
    velocities = np.asarray(graph["observed_velocities_filled"], dtype=np.float64)
    headings = np.asarray(graph["observed_headings_filled"], dtype=np.float64)
    valid = np.asarray(graph["observed_valid_mask"], dtype=bool)
    edges = np.asarray(graph["edge_index"], dtype=np.int64)
    if positions.ndim != 3 or positions.shape[1:] != (OBS_LEN, 2):
        raise ValueError("unexpected observed positions")
    if velocities.shape != positions.shape or valid.shape != positions.shape[:2] or headings.shape != valid.shape:
        raise ValueError("observed array shape mismatch")
    if not all(np.isfinite(x).all() for x in (positions, velocities, headings)):
        raise ValueError("non-finite observed input")
    n = len(positions)
    if not 0 <= src < n or not 0 <= dst < n or src == dst:
        raise ValueError("invalid endpoints")
    if edges.ndim != 2 or edges.shape[0] != 2 or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("invalid edge index")
    if not valid[[src, dst], -(WINDOW+1):].all():
        raise ValueError("both endpoints need K+1 contiguous valid states")
    if any(np.sum((edges[0] == i) & (edges[1] == j)) != 1 for i, j in ((src, dst), (dst, src))):
        raise ValueError("target must have exactly two opposite directed edges")
    # Canonicalize with geometry, not node indices, track IDs, labels or futures.
    if tuple(positions[src, -1]) > tuple(positions[dst, -1]):
        src, dst = dst, src
    direction = positions[dst, -1] - positions[src, -1]
    distance = np.linalg.norm(direction)
    if distance <= 1e-6:
        raise ValueError("coincident endpoints")
    los = direction / distance
    frame = np.column_stack((los, [-los[1], los[0]]))
    return positions, velocities, headings, valid, edges, src, dst, frame


def incident_pairs(edges: np.ndarray, src: int, dst: int) -> list[tuple[int, int]]:
    pairs = set()
    for a, b in edges.T:
        if a in (src, dst) and b not in (src, dst):
            pairs.add((int(a), int(b)))
        elif b in (src, dst) and a not in (src, dst):
            pairs.add((int(b), int(a)))
    return sorted(pairs)


def distance_energy(positions, valid, pairs, src, first, second) -> float:
    total = 0.0
    count = 0
    for endpoint, other in pairs:
        mask = valid[endpoint, -WINDOW:] & valid[other, -WINDOW:]
        rel = positions[endpoint, -WINDOW:] - positions[other, -WINDOW:]
        offset = first if endpoint == src else second
        change = np.linalg.norm(rel + offset, axis=1) - np.linalg.norm(rel, axis=1)
        total += float(np.square(change[mask]).sum())
        count += int(mask.sum())
    return total / count if count else 0.0


def common_mode(positions, valid, edges, src, dst, relative, budget):
    """Ridge geometry controller; no labels/probabilities/gradients used.

    Fit a shared three-basis translation. A fixed backtracking grid accepts only
    bounded, exactly verified reductions in incident-distance energy. c=0 is a
    valid fallback, with an explicit diagnostic, NOT a dropped target.
    """
    basis = temporal_basis()
    pairs = incident_pairs(edges, src, dst)
    design, rhs = [], []
    first, second = -relative / 2, relative / 2
    before = distance_energy(positions, valid, pairs, src, first, second)
    for endpoint, other in pairs:
        for t in range(WINDOW):
            if not (valid[endpoint, t-WINDOW] and valid[other, t-WINDOW]):
                continue
            rel = positions[endpoint, t-WINDOW] - positions[other, t-WINDOW]
            norm = np.linalg.norm(rel)
            if norm <= 1e-6:
                continue
            unit = rel / norm
            design.append(np.outer(basis[t], unit).ravel())
            rhs.append(-np.dot(unit, (first if endpoint == src else second)[t]))
    best = np.zeros_like(relative)
    after, selected = before, 0.0
    if design:
        matrix = np.asarray(design)
        coefficient = np.linalg.solve(matrix.T @ matrix + 1e-3*np.eye(6), matrix.T @ np.asarray(rhs))
        proposed = basis @ coefficient.reshape(3, 2)
        if max_norm(proposed) > COMMON_CAP_M:
            proposed *= COMMON_CAP_M / max_norm(proposed)
        for scale in (1., .5, .25, .125, .0625, .03125):
            candidate = proposed * scale
            a, b = first + candidate, second + candidate
            if not budget_ok(a, b, budget):
                continue
            energy = distance_energy(positions, valid, pairs, src, a, b)
            if energy < after - 1e-15:
                best, after, selected = candidate, energy, scale
    return best, {"incident_pairs": len(pairs), "geometry_energy_before_m2": before,
                  "geometry_energy_after_m2": after, "common_scale": selected,
                  "common_max_m": max_norm(best), "predictive_fp_measured": False}


def apply_relational_trigger(graph: Any, *, src: int, dst: int, template: dict,
                             controller: bool = False, variant: str = "full") -> dict[str, Any]:
    """Return new feature arrays and diagnostics; input assets stay untouched."""
    if variant not in ("full", "single_canonical_first", "single_canonical_second", "relation_reversed"):
        raise ValueError("unknown control variant")
    p, v, h, valid, edges, src, dst, frame = observed_inputs(graph, src, dst)
    canonical = make_template(np.asarray(template["coefficients"]), relative_budget_fraction=template["relative_budget_fraction"])
    if template != canonical:
        raise ValueError("template coefficients/wave/scale/budget binding differs")
    relative = np.asarray(template["relative_wave_m"]) @ frame.T
    budget = reference_budget()
    common, diagnostic = common_mode(p, valid, edges, src, dst, relative, budget) if controller else (
        np.zeros_like(relative), {"common_scale": 0., "common_max_m": 0., "predictive_fp_measured": False})
    first, second = common-relative/2, common+relative/2
    if variant == "single_canonical_first":
        second = np.zeros_like(second)
    elif variant == "single_canonical_second":
        first = np.zeros_like(first)
    elif variant == "relation_reversed":
        first, second = second, first
    if not budget_ok(first, second, budget):
        raise ValueError("trajectory budget exceeded")
    offset = np.zeros_like(p)
    offset[src, -WINDOW:], offset[dst, -WINDOW:] = first, second
    dv = np.zeros_like(v)
    dv[:, 1:] = np.diff(offset, axis=1) / DT
    updated_p, updated_v = p + offset, v + dv
    # Preserve measured velocity/heading residuals. This is not a bicycle-model
    # feasibility certificate. Record heading consistency separately below.
    updated_h = h.copy()
    for node in (src, dst):
        moving = (np.linalg.norm(v[node], axis=1) >= .5) & (np.linalg.norm(updated_v[node], axis=1) >= .5)
        angle = np.arctan2(updated_v[node, :, 1], updated_v[node, :, 0]) - np.arctan2(v[node, :, 1], v[node, :, 0])
        updated_h[node, moving] = np.arctan2(np.sin(h[node, moving] + angle[moving]), np.cos(h[node, moving] + angle[moving]))
    x, speed, mean, acc, lateral = _node_motion_statistics(
        np.asarray(graph["x_node"], dtype=np.float32).copy(), updated_p, updated_v, updated_h, valid, {src, dst})
    edge_attr = np.asarray(graph["edge_attr"], dtype=np.float32).copy()
    if edge_attr.shape != (edges.shape[1], 34):
        raise ValueError("expected base graph-v4 34D edge asset")
    incident = np.isin(edges, [src, dst]).any(axis=0)
    for k in np.flatnonzero(incident):
        a, b = edges[:, k]
        edge_attr[k] = compute_edge_features(int(a), int(b), updated_p, updated_v, updated_h, valid, speed, mean, acc, lateral)
    edge_attr = edge_features_for_mode(edge_attr, edges, updated_p, updated_v, valid,
                                      mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE)
    if not np.isfinite(x).all() or not np.isfinite(edge_attr).all():
        raise ValueError("non-finite regenerated features")
    heading_change = np.arctan2(np.sin(updated_h-h), np.cos(updated_h-h))
    audit = {"version": VERSION, "variant": variant, "controller": controller,
             "canonical_src": src, "canonical_dst": dst, **budget_measurements(first, second),
             "common_mode": diagnostic, "max_heading_change_rad": float(np.abs(heading_change).max()),
             "common_diagnostic_scope": "parent_full_before_control_ablation",
             "budget_pass": True, "topology_changed": False, "edge_feature_dim": int(edge_attr.shape[1]),
             "physical_road_feasibility_certified": False, "model_response_measured": False}
    return {"x_node": x, "edge_attr": edge_attr, "positions": updated_p, "velocities": updated_v,
            "headings": updated_h, "offset": offset, "audit": audit}
