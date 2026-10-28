# Source: experiments/spear_domain_projection_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""Pure tensor transforms of already generated, selected feature coordinates.

This diagnostic never trains a generator, changes a graph, or loads a model.
The fixed [-10, 10] domain is the existing normalized clean-input domain, not
a physical displacement budget and not a tunable new attack amplitude.

``domain_projection`` uses ordinary differentiable torch.clamp, never a
straight-through estimator. ``l1_matched_scale`` is a radial control matching
the projected modification's per-target L1 only. It does NOT promise a domain
bound, a matching L-infinity norm, or matching coordinate-wise changes.
"""
from __future__ import annotations

import torch


DOMAIN_MIN = -10.0
DOMAIN_MAX = 10.0
MODES = ('identity', 'domain_projection', 'l1_matched_scale')


def _inputs(clean, generated, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown output-transform mode: {mode!r}')
    if not isinstance(clean, torch.Tensor) or not isinstance(generated, torch.Tensor):
        raise ValueError('Clean and generated values must both be torch tensors')
    if (clean.ndim != 2 or generated.shape != clean.shape or clean.shape[1] < 1):
        raise ValueError('Require aligned [targets, selected coordinates] tensors with at least one coordinate')
    if (not clean.is_floating_point() or not generated.is_floating_point()
            or clean.dtype != generated.dtype or clean.device != generated.device):
        raise ValueError('Require the same floating dtype and device for clean and generated values')
    if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(generated).all()):
        raise ValueError('Clean and generated values must be finite')
    if bool(((clean < DOMAIN_MIN) | (clean > DOMAIN_MAX)).any()):
        raise ValueError('Clean selected values are outside the existing exact [-10, 10] domain')


def _norms(delta):
    """The caller promotes to float64 before subtraction and accumulation."""
    magnitude = delta.abs()
    return magnitude.sum(dim=1), magnitude.amax(dim=1)


def transform_selected(clean, generated, mode='identity'):
    """Return ``(transformed, per_row_statistics)`` without mutating inputs.

    Inputs are finite, aligned floating tensors [M, K], with K >= 1 and clean
    values inside [-10, 10]. Empty target batches (M = 0) are supported. Output
    keeps the input dtype/device, and has independent storage for every mode.

    Float-valued statistics use float64; coordinate counts use torch.long.
    ``alpha`` ALWAYS means projected_L1/original_L1, or zero for zero original
    displacement. It is diagnostic-only in identity/projection and is applied
    only in l1_matched_scale. ``actual_*`` statistics are recomputed from the
    returned tensor, after its conversion back to the input dtype; finite
    precision can therefore cause small L1 differences for the radial control.

    All arithmetic retains autograd. In particular, clipping has its native
    zero gradient outside the interval; there is no gradient replacement or
    hidden generator optimization. Scalars and dimensions are not trainable.
    """
    _inputs(clean, generated, mode)
    projected = generated.clamp(min=DOMAIN_MIN, max=DOMAIN_MAX)
    clean64, generated64 = clean.to(torch.float64), generated.to(torch.float64)
    original_delta = generated64 - clean64
    projected_delta = projected.to(torch.float64) - clean64
    original_l1, original_linf = _norms(original_delta)
    projected_l1, projected_linf = _norms(projected_delta)
    # Fail explicitly if a finite float64 input has an unrepresentable summed
    # norm, rather than silently constructing alpha=0 from an infinite norm.
    if not bool(torch.isfinite(original_l1).all()):
        raise ValueError('Original per-target L1 cannot be represented as a finite float64 diagnostic')
    nonzero = original_l1 > 0
    safe_denominator = torch.where(nonzero, original_l1, torch.ones_like(original_l1))
    alpha = torch.where(nonzero, projected_l1 / safe_denominator, torch.zeros_like(original_l1))

    if mode == 'identity':
        transformed = generated.clone()
    elif mode == 'domain_projection':
        transformed = projected
    else:
        # Accumulate in float64, then restore the original dtype. Keeping the
        # ratio in float64 also avoids premature half-precision underflow.
        transformed = (clean64 + alpha[:, None] * original_delta).to(generated.dtype)

    actual_l1, actual_linf = _norms(transformed.to(torch.float64) - clean64)
    stats = dict(
        original_outside_coordinate_count=((generated < DOMAIN_MIN) | (generated > DOMAIN_MAX)).sum(dim=1),
        original_l1=original_l1,
        original_linf=original_linf,
        projected_l1=projected_l1,
        projected_linf=projected_linf,
        original_changed_coordinate_count=(generated != clean).sum(dim=1),
        projected_changed_coordinate_count=(projected != clean).sum(dim=1),
        actual_l1=actual_l1,
        actual_linf=actual_linf,
        actual_changed_coordinate_count=(transformed != clean).sum(dim=1),
        alpha=alpha,
    )
    return transformed, stats
