"""Prediction-free TRAIN attachment selection for a paired AV2 experiment.

Only the already-bound eligible-negative candidate indices and observation
motion metadata are consumed. No classifier is constructed or called. The
historical arm reuses its original list (whose creation previously used GCN
ranking); the new arm draws once without replacement within motion strata.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from .full_ablation_data import (
    ATTACHMENT_COUNT, NONATTACHMENT_OUTER_COUNT, OUTER_SEED,
    _selection_contract, _tensor_hash,
)


ARMS = ('historical', 'stratified_random')
STRATA = ('moving', 'slow')


def _rows(inputs, indices):
    metadata = inputs['metadata']['train']
    rows, identities = [], set()
    for index in indices.tolist():
        row = metadata[index]
        if (row.get('role') != 'train' or row.get('role_pair_index') != index
                or row.get('motion_regime') not in STRATA
                or not isinstance(row.get('scenario_id'), str) or not row['scenario_id']
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in ('src_track_id', 'dst_track_id'))
                or row['src_track_id'] == row['dst_track_id']
                or row.get('true_label', 0) != 0 or row.get('eligible', True) is not True):
            raise ValueError('Candidate metadata does not identify an eligible negative TRAIN pair')
        identity = (row['scenario_id'], row['src_track_id'], row['dst_track_id'])
        if identity in identities:
            raise ValueError('Duplicate candidate pair identity')
        identities.add(identity)
        rows.append(row)
    return rows


def select_pair_indices(inputs, arm, seed=20260621):
    """Return one fixed selection and rebuilt outer population; never mutate inputs.

    Motion counts match the original attachment list exactly. There is one
    seeded draw, with no redraw to force a different list or improve a metric.
    Both prospective lists are constructed before a shared 512-row background
    is drawn from TRAIN minus clean supervision minus their attachment union.
    Thus the paired runs differ only in their 26 attachments, not background
    exposure. This intentionally does not replay the historical outer draw.
    """
    if arm not in ARMS:
        raise ValueError('Unknown paired selection arm')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Require a valid nonnegative integer selection seed')
    train, clean, original, candidates, _ = _selection_contract(inputs)
    candidate_rows = _rows(inputs, candidates)
    original_rows = _rows(inputs, original)
    counts = Counter(row['motion_regime'] for row in original_rows)
    quotas = {regime: counts[regime] for regime in STRATA}
    pools = {regime: candidates[torch.tensor(
        [row['motion_regime'] == regime for row in candidate_rows], dtype=torch.bool)]
        for regime in STRATA}
    if any(len(pools[regime]) < quotas[regime] for regime in STRATA):
        raise ValueError('Insufficient negative candidates for fixed motion quotas')
    # Construct the same comparison list in either arm before drawing the
    # shared background. No historical model/score is loaded to do this.
    rng = np.random.RandomState(seed)
    drawn = [pools[regime][rng.choice(len(pools[regime]), quotas[regime], replace=False)]
             for regime in STRATA]
    random_attachments = torch.cat(drawn)
    random_attachments = random_attachments[rng.permutation(len(random_attachments))].clone()
    attachments = original.clone() if arm == 'historical' else random_attachments
    selected_rows = _rows(inputs, attachments)
    selected_counts = Counter(row['motion_regime'] for row in selected_rows)
    if ({regime: selected_counts[regime] for regime in STRATA} != quotas
            or len(attachments) != ATTACHMENT_COUNT
            or len(attachments.unique()) != ATTACHMENT_COUNT
            or not bool(torch.isin(attachments, candidates).all())):
        raise ValueError('Selection changed candidate scope, budget or motion composition')
    pool_mask = torch.ones(len(train['y']), dtype=torch.bool)
    attachment_union = torch.cat([original, random_attachments]).unique(sorted=True)
    pool_mask[clean], pool_mask[attachment_union] = False, False
    background_pool = torch.nonzero(pool_mask, as_tuple=False).flatten()
    if len(background_pool) < NONATTACHMENT_OUTER_COUNT:
        raise ValueError('Insufficient nonattachment TRAIN outer rows')
    background = background_pool[np.random.RandomState(OUTER_SEED).choice(
        len(background_pool), NONATTACHMENT_OUTER_COUNT, replace=False)]
    outer = torch.cat([attachments, background])
    if (len(outer.unique()) != ATTACHMENT_COUNT + NONATTACHMENT_OUTER_COUNT
            or bool(torch.isin(outer, clean).any())
            or bool(torch.isin(background, attachment_union).any())
            or len(background_pool) != len(train['y']) - len(clean) - len(attachment_union)):
        raise ValueError('Rebuilt outer population changed supervision scope or budget')
    identities = [dict(role_pair_index=index, scenario_id=row['scenario_id'],
        src_track_id=row['src_track_id'], dst_track_id=row['dst_track_id'],
        motion_regime=row['motion_regime'], true_label=0)
        for index, row in zip(attachments.tolist(), selected_rows)]
    record = dict(arm=arm, selection_seed=seed,
        selection=('reuse_historical_GCN_ranked_list' if arm == 'historical'
                   else 'single_uniform_without_replacement_draw_per_motion_stratum'),
        historical_GCN_ranked_provenance_retained=arm == 'historical',
        selection_uses_model_predictions=False, GCN_calls=0,
        new_ranking=False, source_role='train', candidate_pool_is_original=True,
        candidates_negative_eligible_train_only=True, candidates_disjoint_clean_supervision=True,
        candidate_count=len(candidates), candidate_pool_sha256=_tensor_hash(candidates),
        motion_quotas=quotas, attachment_motion_counts=dict(quotas),
        motion_candidate_counts={k: len(v) for k,v in pools.items()},
        motion_source='fixed observation metadata; original26 counts only, not predictions',
        attachment_count=len(attachments), attachment_indices=attachments.tolist(),
        attachment_indices_sha256=_tensor_hash(attachments), attachment_identities=identities,
        original_attachment_overlap=int(torch.isin(attachments, original).sum()),
        clean_supervision_count=len(clean), idx_train_sha256=_tensor_hash(clean),
        fixed_outer_indices=outer.tolist(), fixed_outer_sha256=_tensor_hash(outer),
        outer_nonattachment_count=NONATTACHMENT_OUTER_COUNT, outer_background_seed=OUTER_SEED,
        outer_background_pool_count=len(background_pool),
        outer_background_pool_sha256=_tensor_hash(background_pool),
        outer_background_rule='TRAIN_minus_clean_supervision_minus_union_of_both_attachment_lists',
        outer_background_indices=background.tolist(), outer_background_sha256=_tensor_hash(background),
        outer_background_excluded_attachment_union_count=len(attachment_union),
        outer_background_excluded_attachment_union_sha256=_tensor_hash(attachment_union),
        common_background_shared_across_arms=True,
        reconstructed_common_outer_not_historical_replay=True,
        selection_uses_dev=False, selection_uses_audit_or_test=False, redraws=0,
        forced_different_from_historical=False)
    return dict(idx_train=clean.clone(), idx_attach=attachments, fixed_outer=outer,
                candidate_pool=candidates.clone(), record=record)
