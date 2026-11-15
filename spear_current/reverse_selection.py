"""Uniform positive selection for offline AV2/self-owned GNN research.

This is a NEW reverse-direction sampling protocol, not the original
entropy-based attachment selection and not a direction-symmetric reproduction.
Only original TRAIN labels, eligibility and frozen index arrays are inspected;
there are no file reads, model calls, motion quotas, label edits or jobs.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import torch

from .outer_selection import _array, _indices, _labels, _sha256


def select_reverse_populations(train, indices, seed=20260621,
                               attachment_count=26, extra_count=512):
    """Select disjoint new positive attachments and extra outer rows in one draw.

    Required fields are ``train['y'/'eligible']`` and frozen
    ``indices['train_labels'/'idx_train'/'idx_attach'/'idx_unlabeled']``.
    Labels must match the original labels exactly. Existing attachment indices
    remain unchanged and must identify eligible original negatives; they are
    NOT reused as reverse attachments. Original supervision is also excluded.

    Candidates are eligible original positives within ``idx_unlabeled`` only,
    retaining that array's order. One local ``RandomState(seed).choice`` draws
    ``attachment_count + extra_count`` positions without replacement: the first
    part is the new attachment list, the rest the extra list, and their ordered
    concatenation is ``outer``. Insufficiency raises before constructing RNG;
    no retries, prediction-conditioned selection or fallback are permitted.

    Counts may be reduced for synthetic testing; the pilot caller freezes the
    production counts of 26 new attachments and 512 extras. Each returned array
    owns independent CPU ``torch.long`` storage. ``selection`` contains JSON-safe
    counts and ordered-array hashes, including the old attachment identities.
    """
    for name, value in (('attachment_count', attachment_count), ('extra_count', extra_count)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or value < 0):
            raise ValueError(f'{name} must be a nonnegative integer')
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
            or not 0 <= seed < 2 ** 32):
        raise ValueError('seed must be an integer in [0, 2**32)')
    seed, attachment_count, extra_count = int(seed), int(attachment_count), int(extra_count)
    labels = _labels(train['y'], 'train.y')
    original_labels = _labels(indices['train_labels'], 'indices.train_labels')
    eligible = _array(train['eligible'], 'train.eligible', boolean=True).copy()
    if eligible.shape != labels.shape:
        raise ValueError('train.eligible must have the same shape as train.y')
    if not np.array_equal(labels, original_labels):
        raise ValueError('train.y differs from original indices.train_labels')

    supervised, original_attachments, unlabeled = (
        _indices(indices[name], name, labels.size)
        for name in ('idx_train', 'idx_attach', 'idx_unlabeled'))
    if np.isin(supervised, original_attachments).any():
        raise ValueError('idx_train and original idx_attach must be disjoint')
    if not (eligible[supervised] & np.isin(labels[supervised], [0, 1])).all():
        raise ValueError('idx_train must contain eligible supervised labels 0 or 1')
    if not (eligible[original_attachments] & (labels[original_attachments] == 0)).all():
        raise ValueError('Original idx_attach must contain eligible original negative labels')

    supervised_overlap = np.isin(unlabeled, supervised)
    attachment_overlap = np.isin(unlabeled, original_attachments)
    strict_positive = eligible[unlabeled] & (labels[unlabeled] == 1)
    candidates = unlabeled[strict_positive & ~supervised_overlap & ~attachment_overlap]
    total = attachment_count + extra_count
    if candidates.size < total:
        raise ValueError(f'Insufficient strict-positive reverse candidates: '
                         f'{candidates.size} available, {total} required')
    positions = np.random.RandomState(seed).choice(candidates.size, total, replace=False)
    outer = candidates[positions]
    attachments, extra = outer[:attachment_count], outer[attachment_count:]

    selection = dict(
        schema_version=1,
        policy='uniform_strict_eligible_true_positive_from_original_idx_unlabeled',
        protocol_note='NEW uniform positive attachment selection, not original entropy-based '
                      'attachment selection; no direction-symmetry claim',
        seed=seed, attachment_count=attachment_count, extra_count=extra_count,
        candidate_order='original_idx_unlabeled_order',
        sampler='numpy.random.RandomState.choice_without_replacement',
        draw_count=1, partition='first attachment_count rows are new attachments; rest are extras',
        motion_quota=None, model_predictions_used=False, original_labels_changed=False,
        original_indices_changed=False, exclusion_counts_are_marginal=True,
        counts=dict(
            train_rows=int(labels.size), idx_train=int(supervised.size),
            original_idx_attach=int(original_attachments.size), idx_unlabeled=int(unlabeled.size),
            eligible_positive_before_exclusions=int(strict_positive.sum()),
            excluded_supervised_overlap=int(supervised_overlap.sum()),
            excluded_original_attachment_overlap=int(attachment_overlap.sum()),
            excluded_unknown_label=int((labels[unlabeled] == -1).sum()),
            excluded_negative_label=int((labels[unlabeled] == 0).sum()),
            excluded_ineligible=int((~eligible[unlabeled]).sum()),
            excluded_total=int(unlabeled.size - candidates.size),
            candidate_count=int(candidates.size), attachment_count=int(attachments.size),
            extra_count=int(extra.size), outer_count=int(outer.size)),
        sha256=dict(
            labels=_sha256(labels), eligible=_sha256(eligible, boolean=True),
            idx_train=_sha256(supervised), original_idx_attach=_sha256(original_attachments),
            idx_unlabeled=_sha256(unlabeled), candidates=_sha256(candidates),
            idx_attach=_sha256(attachments), extra=_sha256(extra), outer=_sha256(outer)),
        hash_encoding='C-order bytes; labels/indices little-endian int64; eligible uint8')
    return dict(idx_attach=torch.from_numpy(attachments.copy()),
                outer=torch.from_numpy(outer.copy()), extra=torch.from_numpy(extra.copy()),
                candidates=torch.from_numpy(candidates.copy()), selection=selection)
