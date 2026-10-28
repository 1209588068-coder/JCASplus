"""Observation-only terminal motion gate, copied from the local audited bridge.
Source: jcas/core/pair_waveform_motion_gate_features.py (motion_pair_gate).
"""
from numbers import Real

import numpy as np

DEFAULT_MOTION_GATE_POLICY = "both"
DEFAULT_MOTION_GATE_THRESHOLD_MPS = 0.5

def motion_pair_gate(
    edge_index: np.ndarray,
    velocities: np.ndarray,
    valid: np.ndarray,
    *,
    policy: str = DEFAULT_MOTION_GATE_POLICY,
    threshold_mps: float = DEFAULT_MOTION_GATE_THRESHOLD_MPS,
) -> np.ndarray:
    """Return a symmetric Boolean gate for each edge, shape [E].

    Speeds are float64 Euclidean norms of the final observed velocity.  Both
    endpoints must have valid and finite terminal observations.  ``both`` then
    requires both speeds to be at least ``threshold_mps``; ``any`` requires at
    least one.  Earlier velocity observations are not used.  The threshold is a
    finite, nonnegative scalar, and equality is included in the moving group.
    """
    if not isinstance(policy, str) or policy not in ("both", "any"):
        raise ValueError("policy must be 'both' or 'any'")
    if (
        isinstance(threshold_mps, (bool, np.bool_))
        or not isinstance(threshold_mps, Real)
        or not np.isfinite(threshold_mps)
        or float(threshold_mps) < 0.0
    ):
        raise ValueError("threshold_mps must be a finite nonnegative real scalar")

    edges = np.asarray(edge_index)
    observed_velocity = np.asarray(velocities)
    observed_valid = np.asarray(valid)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edges.dtype.kind not in "iu":
        raise ValueError("edge_index must contain integer node indices")
    if observed_velocity.ndim != 3 or observed_velocity.shape[2] != 2:
        raise ValueError("velocities must have shape [N, T, 2]")
    if observed_velocity.dtype.kind not in "fiu":
        raise ValueError("velocities must contain real numeric coordinates")
    if observed_velocity.shape[1] < 1:
        raise ValueError("velocities must contain at least one observed state")
    if observed_valid.shape != observed_velocity.shape[:2]:
        raise ValueError("valid must match velocities' [N, T] dimensions")
    if observed_valid.dtype.kind != "b":
        raise ValueError("valid must be a boolean observation mask")
    if edges.size and (
        np.any(edges < 0) or np.any(edges >= observed_velocity.shape[0])
    ):
        raise ValueError("edge_index contains an out-of-range node index")

    terminal_velocity = observed_velocity[:, -1].astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        speeds = np.linalg.norm(terminal_velocity, axis=1)
    usable = (
        observed_valid[:, -1]
        & np.isfinite(terminal_velocity).all(axis=1)
        & np.isfinite(speeds)
    )
    moving = speeds >= float(threshold_mps)
    source, destination = edges
    endpoints_usable = usable[source] & usable[destination]
    if policy == "both":
        return endpoints_usable & moving[source] & moving[destination]
    return endpoints_usable & (moving[source] | moving[destination])

