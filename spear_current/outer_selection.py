"""Frozen outer-domain selection for offline AV2/self-owned GNN research.

This pure sampler consumes original TRAIN labels, eligibility and index arrays.
It performs no file reads, inference, training, motion balancing or scene choice.
The pilot caller is responsible for preserving its 512 TRAIN scenes, 26 original
attachments and 512 extra rows; smaller counts are supported for synthetic tests.
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np
import torch


def _array(value, name, *, boolean=False):
    """Validate without coercing floats, booleans or non-CPU tensors to indices."""
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.layout != torch.strided:
            raise ValueError(f'{name} must be a dense CPU tensor or NumPy array')
        try:
            value = value.detach().numpy()
        except (TypeError, RuntimeError) as error:
            raise ValueError(f'{name} has an unsupported dtype') from error
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional NumPy array or CPU tensor')
    if boolean:
        if value.dtype.kind != 'b':
            raise ValueError(f'{name} must have boolean dtype')
    elif value.dtype.kind not in 'iu':
        raise ValueError(f'{name} must have integer dtype')
    return value


def _labels(value, name):
    value = _array(value, name)
    if not np.isin(value, [-1, 0, 1]).all():
        raise ValueError(f'{name} must contain only original labels -1, 0 or 1')
    return np.array(value, dtype=np.int64, copy=True)


def _indices(value, name, row_count):
    value = _array(value, name)
    if np.any(value < 0) or np.any(value >= row_count):
        raise ValueError(f'{name} contains an out-of-range index')
    if np.unique(value).size != value.size:
        raise ValueError(f'{name} contains duplicate indices')
    return np.array(value, dtype=np.int64, copy=True)


def _sha256(value, *, boolean=False):
    dtype = np.uint8 if boolean else np.dtype('<i8')
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order='C')).hexdigest()


def select_strict_negative_outer(train, indices, extra_count=512, seed=20260621):
    """Draw uniformly from the original unlabeled domain's strict negatives.

    Required fields are ``train['y']``, ``train['eligible']`` and
    ``indices['idx_train'/'idx_attach'/'idx_unlabeled'/'train_labels']``.
    ``y`` must equal the original ``train_labels``. Inputs may be NumPy arrays
    or dense CPU tensors; outputs are independent CPU ``torch.long`` tensors.

    Historical unlabeled rows that overlap supervision or attachments are
    explicitly excluded. All other candidates satisfy ``eligible & (y == 0)``
    and retain their original unlabeled order. Exactly one local RandomState
    draws positions without replacement. Insufficient candidates raise before
    drawing; there is no fallback, replacement draw or retry.

    ``selection`` is JSON-serializable provenance. Exclusion counts are marginal
    over the original unlabeled domain and may overlap (e.g. unknown/ineligible).
    Its SHA-256 values encode ordered arrays, making candidate order auditable.
    """
    if (isinstance(extra_count, (bool, np.bool_)) or not isinstance(extra_count, Integral)
            or extra_count < 0):
        raise ValueError('extra_count must be a nonnegative integer')
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
            or not 0 <= seed < 2 ** 32):
        raise ValueError('seed must be an integer in [0, 2**32)')
    extra_count, seed = int(extra_count), int(seed)
    labels = _labels(train['y'], 'train.y')
    original_labels = _labels(indices['train_labels'], 'indices.train_labels')
    eligible = _array(train['eligible'], 'train.eligible', boolean=True).copy()
    if eligible.shape != labels.shape:
        raise ValueError('train.eligible must have the same shape as train.y')
    if not np.array_equal(labels, original_labels):
        raise ValueError('train.y differs from original indices.train_labels')

    supervised, attachments, unlabeled = (
        _indices(indices[name], name, labels.size)
        for name in ('idx_train', 'idx_attach', 'idx_unlabeled'))
    if np.isin(supervised, attachments).any():
        raise ValueError('idx_train and idx_attach must be disjoint')
    if not (eligible[supervised] & np.isin(labels[supervised], [0, 1])).all():
        raise ValueError('idx_train must contain eligible supervised labels 0 or 1')
    if not (eligible[attachments] & (labels[attachments] == 0)).all():
        raise ValueError('idx_attach must contain eligible original negative labels')

    supervised_overlap = np.isin(unlabeled, supervised)
    attachment_overlap = np.isin(unlabeled, attachments)
    strict_negative = eligible[unlabeled] & (labels[unlabeled] == 0)
    candidates = unlabeled[strict_negative & ~supervised_overlap & ~attachment_overlap]
    if candidates.size < extra_count:
        raise ValueError(f'Insufficient strict-negative outer candidates: '
                         f'{candidates.size} available, {extra_count} required')
    positions = np.random.RandomState(seed).choice(candidates.size, extra_count, replace=False)
    extra = candidates[positions]
    outer = np.concatenate([attachments, extra])

    selection = dict(
        schema_version=1,
        policy='strict_eligible_true_negative_from_original_idx_unlabeled',
        seed=seed,
        extra_count=extra_count,
        candidate_order='original_idx_unlabeled_order',
        sampler='numpy.random.RandomState.choice_without_replacement',
        exclusion_counts_are_marginal=True,
        counts=dict(
            train_rows=int(labels.size), idx_train=int(supervised.size),
            idx_attach=int(attachments.size), idx_unlabeled=int(unlabeled.size),
            eligible_negative_before_exclusions=int(strict_negative.sum()),
            excluded_supervised_overlap=int(supervised_overlap.sum()),
            excluded_attachment_overlap=int(attachment_overlap.sum()),
            excluded_unknown_label=int((labels[unlabeled] == -1).sum()),
            excluded_positive_label=int((labels[unlabeled] == 1).sum()),
            excluded_ineligible=int((~eligible[unlabeled]).sum()),
            excluded_total=int(unlabeled.size - candidates.size),
            candidate_count=int(candidates.size), extra_count=int(extra.size),
            outer_count=int(outer.size)),
        sha256=dict(
            labels=_sha256(labels), eligible=_sha256(eligible, boolean=True),
            idx_train=_sha256(supervised), idx_attach=_sha256(attachments),
            idx_unlabeled=_sha256(unlabeled), candidates=_sha256(candidates),
            extra=_sha256(extra), outer=_sha256(outer)),
        hash_encoding='C-order bytes; labels/indices little-endian int64; eligible uint8')
    return dict(outer=torch.from_numpy(outer.copy()), extra=torch.from_numpy(extra.copy()),
                candidates=torch.from_numpy(candidates.copy()), selection=selection)
