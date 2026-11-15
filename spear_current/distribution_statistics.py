# Source: experiments/spear_distribution_audit_v1/statistics.py; mechanically migrated; algorithms and state keys unchanged.
"""Fixed descriptive distribution diagnostics; NumPy/SciPy only, no I/O.

Input coordinates retain the existing JCAS TRAIN normalization. No second
normalizer, discriminator, threshold search, p-value or independence claim is
introduced. Both kernel bandwidths are fitted on ALL clean TRAIN reference rows
and shared by every arm and stratum. A joint discrepancy is not a proof about
conditional distributions or physically valid vehicle trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance


ARMS = ('continuation', 'unconditional', 'conditional')
STRATA = ('all', 'slow', 'moving')
SELECTED_DIM = 24
CONDITION_DIM = 6
DOMAIN_SCALE = 10.0
BOUNDARY_ATOL = 1e-6
QUANTILES = (0.01, 0.99)
KERNEL_BLOCK_SIZE = 256


def _matrix(values, columns, name, rows=None):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != columns or values.shape[0] < 1
            or values.dtype.kind != 'f' or (rows is not None and len(values) != rows)
            or not np.isfinite(values).all()):
        raise ValueError(f'{name} must be a finite floating [N,{columns}] matrix with matching nonempty rows')
    return np.asarray(values, dtype=np.float64)


def _regimes(values, rows, name):
    values = np.asarray(values)
    if values.dtype != np.bool_ or values.shape != (rows,):
        raise ValueError(f'{name} must be bool[N] aligned to the feature rows')
    return values


def _owned_readonly(values):
    """Immutable bytes-backed copy; callers cannot enable its write flag."""
    values = np.ascontiguousarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _digest(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _embedding(selected, conditions):
    # Each block contributes its mean per-coordinate squared difference.
    selected_embedding = selected / DOMAIN_SCALE / math.sqrt(SELECTED_DIM)
    condition_embedding = conditions / DOMAIN_SCALE / math.sqrt(CONDITION_DIM)
    return selected_embedding, np.concatenate((selected_embedding, condition_embedding), axis=1)


@dataclass(frozen=True)
class KernelBandwidth:
    sigma: float
    positive_pair_count: int
    total_pair_count: int
    fallback_used: bool

    def to_metadata(self):
        return dict(sigma=self.sigma, positive_pair_count=self.positive_pair_count,
                    total_pair_count=self.total_pair_count, fallback_used=self.fallback_used,
                    fit_population='all_clean_train_reference_rows',
                    estimator='median_positive_pairwise_euclidean_distance',
                    degenerate_fallback_sigma=1.0,
                    kernel='exp(-squared_euclidean_distance/(2*sigma**2))')


def _bandwidth(embedded):
    distances = pdist(embedded, metric='euclidean')
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite clean TRAIN pairwise distances')
    positive = distances[distances > 0]
    # A constant/singleton reference has no positive distance. The fixed unit
    # fallback is recorded, and does not inspect or adapt to audit samples.
    sigma = float(np.median(positive)) if positive.size else 1.0
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(sigma * sigma) or sigma * sigma == 0:
        raise ValueError('Clean TRAIN bandwidth is not representable as a positive finite squared scale')
    return KernelBandwidth(sigma, int(positive.size), int(distances.size), not bool(positive.size))


def _kernel_mean(left, right, bandwidth):
    if not len(left) or not len(right):
        raise ValueError('Kernel means require nonempty populations')
    total = []
    denominator = 2.0 * bandwidth.sigma * bandwidth.sigma
    for i in range(0, len(left), KERNEL_BLOCK_SIZE):
        for j in range(0, len(right), KERNEL_BLOCK_SIZE):
            distances = cdist(left[i:i + KERNEL_BLOCK_SIZE], right[j:j + KERNEL_BLOCK_SIZE],
                              metric='sqeuclidean')
            if not np.isfinite(distances).all():
                raise ValueError('Nonfinite pairwise distance in fixed-kernel diagnostic')
            with np.errstate(over='ignore', under='ignore'):
                block = np.exp(-distances / denominator)
            total.append(float(block.sum(dtype=np.float64)))
    return math.fsum(total) / (len(left) * len(right))


def _mmd2(left, right, bandwidth, left_self=None, right_self=None):
    """Biased V-statistic: self-kernel diagonals are intentionally included."""
    if left_self is None:
        left_self = _kernel_mean(left, left, bandwidth)
    if right_self is None:
        right_self = _kernel_mean(right, right, bandwidth)
    value = float(left_self + right_self - 2.0 * _kernel_mean(left, right, bandwidth))
    if not math.isfinite(value) or value < -1e-12:
        raise ValueError('Invalid biased MMD squared value')
    return max(0.0, value)  # Remove only floating-point cancellation below zero.


@dataclass(frozen=True)
class StratumReference:
    name: str
    selected: np.ndarray
    conditions: np.ndarray
    selected_embedding: np.ndarray
    joint_embedding: np.ndarray
    lower_quantile: np.ndarray | None
    upper_quantile: np.ndarray | None
    selected_kernel_self_mean: float | None
    joint_kernel_self_mean: float | None

    @property
    def count(self):
        return len(self.selected)

    def to_metadata(self):
        return dict(count=self.count, selected_sha256=_digest(self.selected),
                    conditions_sha256=_digest(self.conditions),
                    lower_quantile=None if self.lower_quantile is None else self.lower_quantile.tolist(),
                    upper_quantile=None if self.upper_quantile is None else self.upper_quantile.tolist(),
                    selected_kernel_self_mean=self.selected_kernel_self_mean,
                    joint_kernel_self_mean=self.joint_kernel_self_mean)


@dataclass(frozen=True)
class FrozenDistributionReference:
    selected_bandwidth: KernelBandwidth
    joint_bandwidth: KernelBandwidth
    strata: tuple[StratumReference, ...]
    train_moving_sha256: str

    def to_metadata(self):
        """Fresh JSON-compatible metadata; never expose mutable owned arrays."""
        return dict(
            version=1, arms=list(ARMS), strata={part.name: part.to_metadata() for part in self.strata},
            train_moving_sha256=self.train_moving_sha256,
            selected_dimension=SELECTED_DIM, condition_dimension=CONDITION_DIM,
            input_units='original_JCAS_TRAIN_standardized_coordinates',
            second_standardization=False, selected_kernel_scale='1/(10*sqrt(24))',
            condition_kernel_scale='1/(10*sqrt(6))', joint_block_weights='equal',
            selected_bandwidth=self.selected_bandwidth.to_metadata(),
            joint_bandwidth=self.joint_bandwidth.to_metadata(),
            bandwidths_shared_across_all_strata_and_arms=True,
            quantile_probabilities=list(QUANTILES), quantile_method='linear',
            quantile_fit_population='matching_clean_train_stratum',
            outlier_rule='value < q0.01 or value > q0.99; equality is not outside',
            hard_clamp_boundary=DOMAIN_SCALE, boundary_atol=BOUNDARY_ATOL, boundary_rtol=0.0,
            wasserstein='mean of 24 equally weighted one-dimensional W1 distances in original standardized units',
            mmd='biased RBF MMD squared V-statistic including diagonals; descriptive, not a significance test',
            kernel_block_size=KERNEL_BLOCK_SIZE,
            reference_population='train_one_eligible_negative_per_scene_by_original_audit_hash',
            audit_refitting=False, fitted_discriminator=False)


def fit_reference(train_selected, train_conditions, train_moving):
    """Freeze clean TRAIN-only reference statistics before consuming audit data.

    Selection of the registered TRAIN population belongs to the caller. This
    tensor-only interface cannot establish a split from an array's contents.
    Empty slow/moving strata are retained and later reported as unavailable.
    """
    selected = _matrix(train_selected, SELECTED_DIM, 'train_selected')
    conditions = _matrix(train_conditions, CONDITION_DIM, 'train_conditions', rows=len(selected))
    moving = _regimes(train_moving, len(selected), 'train_moving')
    selected_embedding, joint_embedding = _embedding(selected, conditions)
    selected_bandwidth, joint_bandwidth = _bandwidth(selected_embedding), _bandwidth(joint_embedding)
    references = []
    masks = (np.ones(len(selected), dtype=bool), ~moving, moving)
    for name, mask in zip(STRATA, masks):
        raw, condition = _owned_readonly(selected[mask]), _owned_readonly(conditions[mask])
        selected_part = _owned_readonly(selected_embedding[mask])
        joint_part = _owned_readonly(joint_embedding[mask])
        if len(raw):
            low, high = np.quantile(raw, QUANTILES, axis=0, method='linear')
            low, high = _owned_readonly(low), _owned_readonly(high)
            self_selected = _kernel_mean(selected_part, selected_part, selected_bandwidth)
            self_joint = _kernel_mean(joint_part, joint_part, joint_bandwidth)
        else:
            low = high = self_selected = self_joint = None
        references.append(StratumReference(name, raw, condition, selected_part, joint_part,
                                            low, high, self_selected, self_joint))
    return FrozenDistributionReference(selected_bandwidth, joint_bandwidth, tuple(references), _digest(moving))


def _compare(left_raw, left_selected, left_joint, right_raw, right_selected, right_joint,
             reference, left_self, right_self):
    distances = [float(wasserstein_distance(left_raw[:, j], right_raw[:, j]))
                 for j in range(SELECTED_DIM)]
    if not all(math.isfinite(value) for value in distances):
        raise ValueError('Nonfinite one-dimensional Wasserstein distance')
    return dict(wasserstein_1d_mean=math.fsum(distances) / SELECTED_DIM,
                rbf_mmd2_biased_selected=_mmd2(left_selected, right_selected, reference.selected_bandwidth,
                                             left_self[0], right_self[0]),
                rbf_mmd2_biased_joint=_mmd2(left_joint, right_joint, reference.joint_bandwidth,
                                          left_self[1], right_self[1]))


def _fraction(mask):
    return dict(coordinate_count=int(mask.sum()), coordinate_fraction=float(mask.mean()),
                row_count=int(mask.any(axis=1).sum()), row_fraction=float(mask.any(axis=1).mean()))


def _marginal_checks(selected, part):
    outside = (selected < part.lower_quantile) | (selected > part.upper_quantile)
    boundary = np.isclose(np.abs(selected), DOMAIN_SCALE, rtol=0.0, atol=BOUNDARY_ATOL)
    return dict(train_quantile_outliers=_fraction(outside), hard_clamp_boundary=_fraction(boundary))


def _modifications(clean, triggered):
    delta = triggered - clean
    magnitude = np.abs(delta)
    l1 = magnitude.sum(axis=1)
    l2 = np.linalg.norm(delta, axis=1)
    linf = magnitude.max(axis=1)
    if not all(np.isfinite(values).all() for values in (l1, l2, linf)):
        raise ValueError('Nonfinite selected-feature modification norm')
    return {name + '_' + statistic: float(operation(values))
            for name, values in (('l1', l1), ('l2', l2), ('linf', linf))
            for statistic, operation in (('mean', np.mean), ('max', np.max))}


def evaluate(reference, audit_clean, audit_conditions, audit_moving, triggered_by_arm):
    """Compare all arms using frozen TRAIN-only scales, bandwidths and quantiles.

    Audit conditions come from the original clean rows and are shared unchanged
    with every triggered vector. This API never estimates a parameter from audit
    data. All/slow/moving retain the audit row order for paired L1/L2/Linf norms.
    """
    if not isinstance(reference, FrozenDistributionReference):
        raise ValueError('Require a frozen clean TRAIN reference from fit_reference')
    clean = _matrix(audit_clean, SELECTED_DIM, 'audit_clean')
    conditions = _matrix(audit_conditions, CONDITION_DIM, 'audit_conditions', rows=len(clean))
    moving = _regimes(audit_moving, len(clean), 'audit_moving')
    if not isinstance(triggered_by_arm, Mapping) or set(triggered_by_arm) != set(ARMS):
        raise ValueError('Require exactly the three completed arms')
    triggered = {arm: _matrix(triggered_by_arm[arm], SELECTED_DIM, arm, rows=len(clean)) for arm in ARMS}
    clean_selected, clean_joint = _embedding(clean, conditions)
    embeddings = {arm: _embedding(triggered[arm], conditions) for arm in ARMS}
    output = {}
    for part, mask in zip(reference.strata, (np.ones(len(clean), dtype=bool), ~moving, moving)):
        count = int(mask.sum())
        result = dict(reference_count=part.count, audit_count=count)
        output[part.name] = result
        if not count or not part.count:
            result.update(status='NOT_AVAILABLE', reason='empty_audit_stratum' if not count else 'empty_train_reference_stratum',
                          clean_audit=None, arms={arm: None for arm in ARMS})
            continue
        result['status'] = 'AVAILABLE'
        clean_raw, selected_part, joint_part = clean[mask], clean_selected[mask], clean_joint[mask]
        train_self = (part.selected_kernel_self_mean, part.joint_kernel_self_mean)
        clean_self = (_kernel_mean(selected_part, selected_part, reference.selected_bandwidth),
                      _kernel_mean(joint_part, joint_part, reference.joint_bandwidth))
        result['clean_audit'] = dict(
            versus_train=_compare(part.selected, part.selected_embedding, part.joint_embedding,
                                  clean_raw, selected_part, joint_part, reference, train_self, clean_self),
            **_marginal_checks(clean_raw, part))
        result['arms'] = {}
        for arm in ARMS:
            raw = triggered[arm][mask]
            selected_arm, joint_arm = (value[mask] for value in embeddings[arm])
            arm_self = (_kernel_mean(selected_arm, selected_arm, reference.selected_bandwidth),
                        _kernel_mean(joint_arm, joint_arm, reference.joint_bandwidth))
            result['arms'][arm] = dict(
                versus_train=_compare(part.selected, part.selected_embedding, part.joint_embedding,
                                      raw, selected_arm, joint_arm, reference, train_self, arm_self),
                versus_clean_audit=_compare(clean_raw, selected_part, joint_part, raw, selected_arm, joint_arm,
                                            reference, clean_self, arm_self),
                selected_modification=_modifications(clean_raw, raw), **_marginal_checks(raw, part))
    return dict(reference=reference.to_metadata(), strata=output,
                interpretation=dict(descriptive_only=True, statistical_significance_claim=False,
                    independence_assumed=False, distribution_preservation_proven=False,
                    joint_discrepancy_proves_conditional_distribution=False,
                    physical_validity_claim=False, audit_refitting=False))
