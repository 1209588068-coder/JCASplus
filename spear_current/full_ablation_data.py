"""TRAIN/dev inputs and prediction-independent controls for full ablations.

Offline, authorized AV2/self-owned GNN robustness research. This module never
restores attack checkpoints, starts training, or reads audit/VAL/TEST arrays.
The retained clean selector is moved before fresh attack training; reusing its
support does not replay the historical frozen-uniform deployment experiment.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .development_diagnostic import read_inputs, registered_rows
from .runtime import sha256
from .support_schema import (SELECTED_COORDINATES, EXCLUDED_COORDINATES,
                             coordinate_metadata, support_groups)


VARIANTS = ('full', 'random_nodes', 'random_features', 'self_only', 'base_only',
            'no_kl', 'single_kl', 'no_distribution', 'no_distribution_no_domain')
SEEDS = (20260621, 20260622, 20260623)
ATTACHMENT_COUNT = 26
NONATTACHMENT_OUTER_COUNT = 512
OUTER_SEED = 20260621
SUPPORT_COST = 24


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode()
                          + value.numpy().tobytes()).hexdigest()


def _case(variant, seed):
    if variant not in VARIANTS:
        raise ValueError('Unregistered full-ablation variant')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed not in SEEDS:
        raise ValueError('Unregistered full-ablation seed')
    return int(seed)


def prepare_inputs(bundle):
    """Read and hash-check exactly the original 512 TRAIN and 128 dev graphs.

    ``read_inputs`` checks each opened graph before and after deserialization,
    the TRAIN feature/label hashes, the original outer draw, dev indices and
    attention binding. ``registered_rows`` checks audit identities as metadata
    only; no audit path is hashed or opened here.
    """
    ledger = []
    sources = registered_rows(bundle)
    inputs = read_inputs(bundle, sources, ledger)
    if (len(ledger) != 640
            or any(sum(row['role'] == role for row in ledger) != count
                   for role, count in (('train', 512), ('dev', 128)))
            or any(row['split'] != 'train' or row['status'] != 'READ_COMPLETE'
                   or row['role'] not in sources for row in ledger)):
        raise ValueError('Full-ablation inputs must read only 512 TRAIN/128 dev graphs')
    for role in ('train', 'dev'):
        actual = [row['scenario_id'] for row in ledger if row['role'] == role]
        if actual != [row['scenario_id'] for row in sources[role]]:
            raise ValueError('Full-ablation graph ledger identity/order changed')
    inputs['ledger'] = ledger
    # Aliases point at the already-read tensors and do not load another graph.
    inputs['train'], inputs['dev'] = inputs['combined']['train'], inputs['combined']['dev']
    return inputs


def _index(value, name, count=None, upper=None):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.long
            or value.device.type != 'cpu' or value.ndim != 1
            or len(value.unique()) != len(value)
            or (count is not None and len(value) != count)
            or bool((value < 0).any())
            or (upper is not None and bool((value >= upper).any()))):
        raise ValueError('Invalid unique CPU row indices: ' + name)
    return value


def _selection_contract(inputs):
    train, idx = inputs['combined']['train'], inputs['indices']
    labels, eligible = train['y'], train['eligible']
    n = len(labels)
    if (labels.ndim != 1 or labels.dtype != torch.long or labels.device.type != 'cpu'
            or eligible.shape != labels.shape or eligible.dtype != torch.bool
            or not torch.equal(labels, idx['train_labels'])):
        raise ValueError('TRAIN labels/eligibility differ from the frozen input contract')
    clean = _index(idx['idx_train'], 'idx_train', upper=n)
    attach = _index(idx['idx_attach'], 'idx_attach', ATTACHMENT_COUNT, n)
    candidates = _index(idx['candidates'], 'candidates', upper=n)
    unlabeled = _index(idx['idx_unlabeled'], 'idx_unlabeled', upper=n)
    dev = _index(idx['dev_indices'], 'dev_indices')
    if (len(candidates) < ATTACHMENT_COUNT or bool(torch.isin(clean, candidates).any())
            or not bool(torch.isin(attach, candidates).all())
            or not bool((labels[candidates] == 0).all())
            or not bool(eligible[candidates].all()) or not bool(eligible[clean].all())
            or bool((dev < n).any())):
        raise ValueError('Attachment candidates must be eligible negative TRAIN rows outside clean/dev supervision')
    expected_pool = torch.ones(n, dtype=torch.bool)
    expected_pool[clean], expected_pool[attach] = False, False
    if not torch.equal(unlabeled.sort().values, torch.nonzero(expected_pool).flatten()):
        raise ValueError('Frozen unlabeled pool must equal TRAIN minus clean supervision/attachments')
    draw = unlabeled[np.random.RandomState(OUTER_SEED).choice(
        len(unlabeled), NONATTACHMENT_OUTER_COUNT, replace=False)]
    original_outer = torch.cat([attach, draw])
    if not torch.equal(inputs['outer'], original_outer):
        raise ValueError('Original fixed attachment/nonattachment outer draw changed')
    return train, clean, attach, candidates, unlabeled


def _motion_counts(inputs, selected):
    rows = inputs['metadata']['train']
    counts = dict(moving=0, slow=0)
    for i in selected.tolist():
        row = rows[i]
        if row.get('role') != 'train' or row.get('role_pair_index') != i:
            raise ValueError('Attachment observation metadata is not bound to TRAIN row identity')
        motion = row.get('motion_regime')
        if motion not in counts:
            raise ValueError('Unrecognized observation motion regime')
        counts[motion] += 1
    return counts


def select_indices(inputs, variant, seed):
    """Select 26 attachments before predictions and preserve the outer budget.

    RandomNodes draws without replacement from the SAME frozen candidates.
    Its background pool is all TRAIN rows outside clean supervision and the
    new attachments: old attachments receive no special extra exposure. The
    background draw uses the original fixed seed 20260621 for every case.
    """
    seed = _case(variant, seed)
    train, clean, original_attach, candidates, unlabeled = _selection_contract(inputs)
    if variant == 'random_nodes':
        attachment = candidates[np.random.RandomState(seed).choice(
            len(candidates), ATTACHMENT_COUNT, replace=False)].clone()
        mask = torch.ones(len(train['y']), dtype=torch.bool)
        mask[clean], mask[attachment] = False, False
        background_pool = torch.nonzero(mask, as_tuple=False).flatten()
        background = background_pool[np.random.RandomState(OUTER_SEED).choice(
            len(background_pool), NONATTACHMENT_OUTER_COUNT, replace=False)]
        outer = torch.cat([attachment, background])
    else:
        attachment = original_attach.clone()
        background_pool = unlabeled.clone()
        outer = inputs['outer'].clone()
    if (len(outer) != ATTACHMENT_COUNT + NONATTACHMENT_OUTER_COUNT
            or len(outer.unique()) != len(outer)
            or len(background_pool) != len(unlabeled)
            or bool(torch.isin(clean, outer).any())
            or not bool(torch.isin(attachment, candidates).all())
            or not bool((train['y'][attachment] == 0).all())):
        raise ValueError('Attachment, supervision or fixed outer budget changed')
    record = dict(variant=variant, seed=seed,
        selection=('uniform_without_replacement_from_original_candidates'
                   if variant == 'random_nodes' else 'frozen_historical_clean_selection'),
        attachment_count=len(attachment), clean_supervision_count=len(clean),
        candidate_count=len(candidates), candidate_pool_sha256=_tensor_hash(candidates),
        candidate_pool_is_original=True, candidates_negative_eligible_train_only=True,
        candidates_disjoint_clean_supervision=True,
        candidate_pool_contains_all_original_attachments=True,
        idx_train_sha256=_tensor_hash(clean), attachment_indices=attachment.tolist(),
        attachment_indices_sha256=_tensor_hash(attachment),
        original_attachment_overlap=int(torch.isin(attachment, original_attach).sum()),
        attachment_motion_counts=_motion_counts(inputs, attachment),
        motion_source='observation_metadata.motion_regime',
        fixed_outer_indices=outer.tolist(), fixed_outer_sha256=_tensor_hash(outer),
        outer_nonattachment_count=NONATTACHMENT_OUTER_COUNT,
        outer_background_seed=OUTER_SEED, outer_background_pool_count=len(background_pool),
        outer_background_pool_sha256=_tensor_hash(background_pool),
        outer_background_rule='TRAIN_minus_clean_supervision_minus_current_attachments',
        selection_uses_model_predictions=False, selection_uses_dev=False,
        selection_uses_audit_or_test=False, redraws=0)
    return dict(idx_train=clean.clone(), idx_attach=attachment, fixed_outer=outer,
                candidate_pool=candidates.clone(), record=record)


def feasible_support_subsets():
    """Enumerate each legal semantic-group subset of exact coordinate cost 24."""
    groups = support_groups('semantic')
    return tuple(tuple(i for i, use in enumerate(flags) if use)
        for flags in itertools.product((False, True), repeat=len(groups))
        if sum(len(group) for group, use in zip(groups, flags) if use) == SUPPORT_COST)


def select_support(bundle, variant, seed):
    """Reuse clean-derived SAGE24 or uniformly draw one exact-cost group subset.

    The fixed importance order remains 30 coordinates. Focal columns 357/715
    are always masked off. Sampling includes the SAGE subset as a valid outcome
    and performs exactly one draw, with no prediction-based rejection.
    """
    seed = _case(variant, seed)
    coordinates = torch.tensor(bundle.manifest['selected_coordinates'], dtype=torch.long)
    if coordinates.tolist() != list(SELECTED_COORDINATES):
        raise ValueError('Frozen 30-coordinate importance order changed')
    asset_path = bundle.asset_path('support')
    frozen = json.loads(asset_path.read_text())
    fixed = frozen.get('hard_support')
    if (frozen.get('arm') != 'grouped_sage24'
            or frozen.get('selected_coordinates') != coordinates.tolist()
            or not isinstance(fixed, list) or len(fixed) != 30
            or any(type(value) is not bool for value in fixed)
            or fixed != bundle.manifest['hard_support']):
        raise ValueError('Frozen grouped SAGE support asset changed')
    groups = support_groups('semantic')
    feasible = feasible_support_subsets()
    frozen_active = {coordinate for coordinate, active in zip(coordinates.tolist(), fixed) if active}
    frozen_groups = tuple(i for i, group in enumerate(groups) if set(group) <= frozen_active)
    if ({coordinate for i in frozen_groups for coordinate in groups[i]} != frozen_active
            or sum(len(groups[i]) for i in frozen_groups) != SUPPORT_COST):
        raise ValueError('Frozen SAGE support must retain whole semantic groups at cost 24')
    draw = None
    if variant == 'random_features':
        draw = int(np.random.RandomState(seed).randint(len(feasible)))
        selected_groups = feasible[draw]
    else:
        selected_groups = frozen_groups
    selected = {coordinate for i in selected_groups for coordinate in groups[i]}
    support = torch.tensor([coordinate in selected for coordinate in coordinates.tolist()], dtype=torch.bool)
    if (int(support.sum()) != SUPPORT_COST or selected.intersection(EXCLUDED_COORDINATES)
            or selected_groups not in feasible):
        raise ValueError('Illegal exact-cost semantic support')
    metadata = dict(groups=[list(group) for group in groups],
        coordinates=[coordinate_metadata(c) for c in coordinates.tolist()],
        excluded_coordinates=list(EXCLUDED_COORDINATES))
    provenance_doc = Path(__file__).resolve().parents[1] / 'COMPARISON_AND_ABLATION_PLAN_20261029.md'
    record = dict(variant=variant, seed=seed,
        selection=('uniform_over_all_exact_cost_semantic_group_subsets'
                   if variant == 'random_features' else 'frozen_clean_derived_grouped_sage24'),
        selected_coordinates=coordinates.tolist(), hard_support=support.tolist(),
        active_coordinates=coordinates[support].tolist(), selected_cost=int(support.sum()),
        selected_group_indices=list(selected_groups),
        selected_groups=[list(groups[i]) for i in selected_groups],
        legal_coordinate_count=28, semantic_group_count=len(groups),
        feasible_subset_count=len(feasible), feasible_subsets_sha256=_json_hash(feasible),
        sampled_subset_index=draw, semantic_metadata_sha256=_json_hash(metadata),
        frozen_sage_support_asset=dict(path=str(asset_path), sha256=sha256(asset_path)),
        frozen_importance_asset=dict(path=str(bundle.asset_path('feature_importance')),
            sha256=bundle.manifest['assets']['feature_importance']['sha256']),
        frozen_clean_selector_provenance=dict(
            source='COMPARISON_AND_ABLATION_PLAN_20261029.md:138',
            document_sha256=sha256(provenance_doc),
            basis='documented_clean_proxy1_and_TRAIN_data_only_no_uniform_generator',
            exact_original_estimator_source_independently_verified=False,
            estimator_recomputed=False, attack_checkpoint_loaded=False),
        pipeline_timing='frozen_clean_support_selected_before_fresh_attack_training',
        same_as_frozen_sage_support=bool(torch.equal(support, torch.tensor(fixed))),
        forced_different_from_sage=False, selection_uses_asr=False,
        selection_uses_dev_or_audit=False, redraws=0)
    return coordinates, support, record
