# Source: local jcas/core/risk_labels.py; unchanged dynamic-risk calculation.
#!/usr/bin/env python3
"""Model-independent dynamic edge-risk labels from future trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


LABEL_MODES = ("dynamic_risk",)


@dataclass(frozen=True)
class RiskLabelConfig:
    label_mode: str = "dynamic_risk"
    risk_base_distance_m: float = 5.0
    risk_reaction_time_s: float = 1.0
    risk_safe_decel_mps2: float = 4.0
    future_dt_seconds: float = 0.1
    closing_speed_method: str = "local_linear_regression_v2"
    closing_speed_window_frames: int = 5
    min_risk_consecutive_frames: int = 3


def validate_label_config(config: RiskLabelConfig) -> RiskLabelConfig:
    if config.label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")
    positive = {
        "risk_base_distance_m": config.risk_base_distance_m,
        "risk_safe_decel_mps2": config.risk_safe_decel_mps2,
        "future_dt_seconds": config.future_dt_seconds,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not np.isfinite(config.risk_reaction_time_s)
        or float(config.risk_reaction_time_s) < 0.0
    ):
        raise ValueError("risk_reaction_time_s must be finite and non-negative")
    if config.closing_speed_method != "local_linear_regression_v2":
        raise ValueError("unsupported closing_speed_method")
    if (
        int(config.closing_speed_window_frames) != config.closing_speed_window_frames
        or int(config.closing_speed_window_frames) < 3
        or int(config.closing_speed_window_frames) % 2 == 0
    ):
        raise ValueError("closing_speed_window_frames must be an odd integer >= 3")
    if (
        int(config.min_risk_consecutive_frames)
        != config.min_risk_consecutive_frames
        or int(config.min_risk_consecutive_frames) < 1
    ):
        raise ValueError("min_risk_consecutive_frames must be an integer >= 1")
    return config


def label_config_dict(config: RiskLabelConfig) -> dict[str, Any]:
    """Return the canonical, JSON-serializable label definition."""
    validate_label_config(config)
    return asdict(config)


def label_config_hash(config: RiskLabelConfig | Mapping[str, Any]) -> str:
    """SHA-256 over the exact canonical label configuration."""
    if isinstance(config, RiskLabelConfig):
        payload = label_config_dict(config)
    else:
        payload = label_config_dict(RiskLabelConfig(**dict(config)))
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_from_result(result: Mapping[str, Any]) -> RiskLabelConfig:
    payload = result.get("label_config")
    if payload is None:
        raise ValueError("result.json has no dynamic label configuration")
    config = RiskLabelConfig(**dict(payload))
    recorded_hash = result.get("label_config_hash")
    actual_hash = label_config_hash(config)
    if recorded_hash is not None and str(recorded_hash) != actual_hash:
        raise ValueError("result.json label_config_hash does not match label_config")
    return config


def assert_label_config_matches_result(
    result: Mapping[str, Any], requested: RiskLabelConfig
) -> None:
    saved = config_from_result(result)
    if label_config_hash(saved) != label_config_hash(requested):
        raise ValueError(
            "requested label configuration does not match the checkpoint label configuration"
        )


def _graph_array(graph: Any, key: str) -> np.ndarray:
    return np.asarray(graph[key])


def _has_key(graph: Any, key: str) -> bool:
    if hasattr(graph, "files"):
        return key in graph.files
    return key in graph


def _contiguous_runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    return [part for part in np.split(indices, boundaries) if part.size]


def _local_linear_closing_speed(
    distance: np.ndarray,
    valid: np.ndarray,
    dt_seconds: float,
    window_frames: int,
) -> np.ndarray:
    """Estimate max(0, -d'(t)) with a centered local linear fit.

    A speed is emitted only where the complete odd-length window is available
    inside one contiguous valid run.  Therefore the first/last half-window and
    gaps remain NaN: no endpoint extrapolation and no differencing across
    missing future frames can create a risk label.
    """
    closing = np.full(distance.shape, np.nan, dtype=np.float64)
    half = int(window_frames) // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64) * float(dt_seconds)
    denominator = float(np.dot(offsets, offsets))
    if denominator <= 0.0:
        raise ValueError("invalid local-linear time window")
    for run in _contiguous_runs(np.flatnonzero(valid)):
        if run.size < int(window_frames):
            continue
        for center_offset in range(half, run.size - half):
            center = int(run[center_offset])
            indices = run[center_offset - half : center_offset + half + 1]
            values = distance[indices].astype(np.float64, copy=False)
            # The centered offsets sum to zero, so an explicit intercept fit is
            # equivalent to this stable closed form.
            slope = float(np.dot(offsets, values) / denominator)
            closing[center] = max(0.0, -slope)
    return closing


def _max_true_run(mask: np.ndarray) -> int:
    """Maximum number of consecutive True values in a one-dimensional mask."""
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def prepare_dynamic_risk_inputs(
    graph: Any, config: RiskLabelConfig
) -> dict[str, np.ndarray]:
    """Compute model-independent pair distance and robust closing-speed arrays.

    The result can be reused for a parameter grid whose configurations share
    the same time step, closing-speed method, and regression window.
    """
    validate_label_config(config)
    future_positions = _graph_array(graph, "future_positions").astype(
        np.float64, copy=False
    )
    future_valid = _graph_array(graph, "future_valid_mask").astype(bool, copy=False)
    edge_index = _graph_array(graph, "edge_index").astype(np.int64, copy=False)
    if future_positions.ndim != 3 or future_positions.shape[2] != 2:
        raise ValueError("future_positions must have shape [nodes, timesteps, 2]")
    if future_valid.shape != future_positions.shape[:2]:
        raise ValueError("future_valid_mask shape does not match future_positions")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")

    edge_count = int(edge_index.shape[1])
    future_steps = int(future_positions.shape[1])
    distance = np.full((edge_count, future_steps), np.nan, dtype=np.float64)
    closing = np.full((edge_count, future_steps), np.nan, dtype=np.float64)
    valid_margin = np.zeros((edge_count, future_steps), dtype=bool)

    for edge_id, (src, dst) in enumerate(edge_index.T):
        pair_valid = future_valid[src] & future_valid[dst]
        pair_valid &= np.isfinite(future_positions[src]).all(axis=1)
        pair_valid &= np.isfinite(future_positions[dst]).all(axis=1)
        if not pair_valid.any():
            continue
        relative = future_positions[dst] - future_positions[src]
        pair_distance = np.linalg.norm(relative, axis=1)
        pair_closing = _local_linear_closing_speed(
            pair_distance,
            pair_valid,
            config.future_dt_seconds,
            int(config.closing_speed_window_frames),
        )
        distance[edge_id, pair_valid] = pair_distance[pair_valid]
        closing[edge_id] = pair_closing
        valid_margin[edge_id] = (
            pair_valid & np.isfinite(pair_distance) & np.isfinite(pair_closing)
        )

    return {
        "distance": distance,
        "closing_speed": closing,
        "valid_margin": valid_margin,
        "future_dt_seconds": np.asarray(float(config.future_dt_seconds)),
        "closing_speed_window_frames": np.asarray(
            int(config.closing_speed_window_frames)
        ),
    }


def compute_dynamic_risk(
    graph: Any,
    config: RiskLabelConfig,
    prepared: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Compute dynamic labels and per-edge diagnostics from future positions."""
    validate_label_config(config)
    edge_index = _graph_array(graph, "edge_index").astype(np.int64, copy=False)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    prepared = (
        prepare_dynamic_risk_inputs(graph, config) if prepared is None else prepared
    )
    if not np.isclose(
        float(np.asarray(prepared["future_dt_seconds"]).item()),
        float(config.future_dt_seconds),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("prepared dynamic inputs use a different future time step")
    if int(np.asarray(prepared["closing_speed_window_frames"]).item()) != int(
        config.closing_speed_window_frames
    ):
        raise ValueError("prepared dynamic inputs use a different regression window")

    edge_count = int(edge_index.shape[1])
    labels = np.zeros(edge_count, dtype=np.int8)
    margin_min = np.full(edge_count, np.nan, dtype=np.float32)
    risk_timestep = np.full(edge_count, -1, dtype=np.int16)
    closing_at_risk = np.full(edge_count, np.nan, dtype=np.float32)
    threshold_at_risk = np.full(edge_count, np.nan, dtype=np.float32)
    computable = np.zeros(edge_count, dtype=bool)
    violation_run_max = np.zeros(edge_count, dtype=np.int16)

    distance = np.asarray(prepared["distance"], dtype=np.float64)
    closing = np.asarray(prepared["closing_speed"], dtype=np.float64)
    valid_margin = np.asarray(prepared["valid_margin"], dtype=bool)
    expected_shape = (edge_count, distance.shape[1])
    if distance.shape != expected_shape or closing.shape != expected_shape:
        raise ValueError("prepared dynamic input shape does not match edge_index")
    if valid_margin.shape != expected_shape:
        raise ValueError("prepared valid-margin shape does not match edge_index")

    for edge_id in range(edge_count):
        valid = valid_margin[edge_id]
        if _max_true_run(valid) < int(config.min_risk_consecutive_frames):
            continue
        computable[edge_id] = True
        edge_closing = closing[edge_id]
        dynamic_distance = (
            float(config.risk_base_distance_m)
            + float(config.risk_reaction_time_s) * edge_closing
            + edge_closing**2 / (2.0 * float(config.risk_safe_decel_mps2))
        )
        margin = distance[edge_id] - dynamic_distance
        valid_indices = np.flatnonzero(valid & np.isfinite(margin))
        if valid_indices.size == 0:
            continue
        violating = valid & np.isfinite(margin) & (margin <= 0.0)
        max_violation_run = _max_true_run(violating)
        violation_run_max[edge_id] = np.int16(max_violation_run)
        qualifies = max_violation_run >= int(config.min_risk_consecutive_frames)

        qualifying_indices: list[np.ndarray] = []
        if qualifies:
            for run in _contiguous_runs(np.flatnonzero(violating)):
                if run.size >= int(config.min_risk_consecutive_frames):
                    qualifying_indices.append(run)
        diagnostic_indices = (
            np.concatenate(qualifying_indices) if qualifying_indices else valid_indices
        )
        local = int(np.argmin(margin[diagnostic_indices]))
        timestep = int(diagnostic_indices[local])
        value = float(margin[timestep])
        margin_min[edge_id] = value
        risk_timestep[edge_id] = timestep
        closing_at_risk[edge_id] = float(edge_closing[timestep])
        threshold_at_risk[edge_id] = float(dynamic_distance[timestep])
        labels[edge_id] = np.int8(qualifies)

    return {
        "edge_label_dynamic_risk": labels,
        "edge_label_dynamic_computable_mask": computable,
        "future_dynamic_margin_min": margin_min,
        "future_dynamic_risk_timestep": risk_timestep,
        "future_closing_speed_at_risk": closing_at_risk,
        "future_dynamic_threshold_at_risk": threshold_at_risk,
        "future_dynamic_violation_run_max": violation_run_max,
    }


def labels_for_graph(graph: Any, config: RiskLabelConfig) -> dict[str, np.ndarray]:
    """Return the dynamic ``edge_label`` and its diagnostic arrays."""
    validate_label_config(config)
    result = compute_dynamic_risk(graph, config)
    result["edge_label"] = result["edge_label_dynamic_risk"].astype(
        np.float32, copy=False
    )
    return result


def selected_label_computable_mask(
    graph: Any,
    config: RiskLabelConfig,
    label_bundle: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Computability mask for the selected label definition."""
    edge_count = int(_graph_array(graph, "edge_index").shape[1])
    if _has_key(graph, "label_computable_mask"):
        mask = _graph_array(graph, "label_computable_mask").astype(bool, copy=True)
    else:
        mask = np.ones(edge_count, dtype=bool)
    if mask.shape != (edge_count,):
        raise ValueError("label_computable_mask shape does not match edge_index")
    bundle = labels_for_graph(graph, config) if label_bundle is None else label_bundle
    dynamic = np.asarray(
        bundle["edge_label_dynamic_computable_mask"], dtype=bool
    )
    if dynamic.shape != (edge_count,):
        raise ValueError(
            "edge_label_dynamic_computable_mask shape does not match edge_index"
        )
    mask &= dynamic
    return mask

