"""Clean-reference projection of the extra counterfactual gradient.

The constraint is in raw-gradient Euclidean space, before clipping and AdamW.
It is not a guarantee on AdamW's parameter step or on held-out task metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


PROJECTED_V3_PROFILE = "projected_v3_t0p1_r0p1_c1_adj5_m1_rho1"


@dataclass(frozen=True)
class CleanProjectionConfig:
    max_extra_norm_ratio: float = 1.0
    clean_norm_floor: float = 1e-12

    def validate(self) -> None:
        for name in ("max_extra_norm_ratio", "clean_norm_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    if len(gradients) != len(parameters) or not parameters:
        raise ValueError("gradient/parameter accounting mismatch")
    return torch.cat([
        (gradient.detach() if gradient is not None else torch.zeros_like(parameter))
        .reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ])


def assign_flat_gradient(
    vector: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    used: Sequence[bool],
) -> None:
    """Keep unused parameters at None so AdamW does not decay/update them."""
    if len(used) != len(parameters) or vector.numel() != sum(p.numel() for p in parameters):
        raise ValueError("flat gradient/parameter accounting mismatch")
    offset = 0
    for parameter, active in zip(parameters, used):
        size = parameter.numel()
        parameter.grad = (
            vector[offset:offset + size].reshape_as(parameter).clone()
            if active else None
        )
        offset += size


def direction_stats(reference: torch.Tensor, vector: torch.Tensor) -> dict:
    reference = reference.detach().double()
    vector = vector.detach().double()
    dot = float(torch.dot(reference, vector))
    reference_norm = float(torch.linalg.vector_norm(reference))
    vector_norm = float(torch.linalg.vector_norm(vector))
    return {
        "dot": dot,
        "cosine": dot / (reference_norm * vector_norm)
        if reference_norm > 0 and vector_norm > 0 else None,
        "norm_ratio": vector_norm / reference_norm if reference_norm > 0 else None,
    }


def project_extra_gradient(
    clean: torch.Tensor,
    total: torch.Tensor,
    *,
    config: CleanProjectionConfig,
) -> tuple[torch.Tensor, dict]:
    """Project total-clean onto <extra,clean> >= 0, then cap its norm.

    A numerically zero clean gradient disables the extra update. Calculations
    use float64; the returned gradient retains the caller's dtype and device.
    """
    config.validate()
    if clean.ndim != 1 or clean.shape != total.shape:
        raise ValueError("expected equally shaped flat gradient vectors")
    if clean.dtype != total.dtype or clean.device != total.device:
        raise ValueError("gradient dtype/device mismatch")
    if not torch.isfinite(clean).all() or not torch.isfinite(total).all():
        raise FloatingPointError("non-finite gradient; optimizer step aborted")
    base = clean.detach().double()
    extra = total.detach().double() - base
    base_norm = float(torch.linalg.vector_norm(base))
    extra_norm = float(torch.linalg.vector_norm(extra))
    dot_before = float(torch.dot(base, extra))
    projected = dot_before < 0 and base_norm > config.clean_norm_floor
    if base_norm <= config.clean_norm_floor:
        corrected = torch.zeros_like(extra)
        capped = extra_norm > 0
    else:
        corrected = extra
        if projected:
            corrected = corrected - (dot_before / (base_norm ** 2)) * base
        norm = float(torch.linalg.vector_norm(corrected))
        bound = float(config.max_extra_norm_ratio) * base_norm
        capped = norm > bound
        if capped:
            corrected = corrected * (bound / norm)
    update = (base + corrected).to(clean.dtype)
    actual_extra = update.double() - base
    dot_after = float(torch.dot(base, actual_extra))
    norm_after = float(torch.linalg.vector_norm(actual_extra))
    # Check the constraint again after casting to the optimizer's dtype.
    tolerance = 32 * torch.finfo(clean.dtype).eps
    if dot_after < -tolerance * max(base_norm ** 2, 1e-30):
        raise FloatingPointError("projected gradient violates clean half-space")
    if norm_after > (config.max_extra_norm_ratio + tolerance) * base_norm + 1e-30:
        raise FloatingPointError("projected gradient violates extra norm bound")
    return update, {
        "projected": projected,
        "capped": capped,
        "clean_norm_floor_hit": base_norm <= config.clean_norm_floor,
        "clean_norm": base_norm,
        "extra_norm_before": extra_norm,
        "extra_norm_after": norm_after,
        "extra_ratio_before": extra_norm / base_norm if base_norm > 0 else None,
        "extra_ratio_after": norm_after / base_norm if base_norm > 0 else None,
        "extra_clean_dot_before": dot_before,
        "extra_clean_dot_after": dot_after,
        "projected_total_clean_cosine": direction_stats(base, update)["cosine"],
    }
