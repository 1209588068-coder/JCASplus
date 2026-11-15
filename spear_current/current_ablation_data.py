"""Fixed TRAIN selections and pre-existing supports for current-module ablations.

This module selects no models and reads no graph arrays.  The current ablation
uses the historical eligible-negative pool, its original ranked 26 attachments,
and one motion-matched random draw per seed.  Every variant within a seed shares
the same 512 background rows, excluding BOTH attachment lists.  Feature masks
are existing metadata, not a new feature-selector fit or an ASR-based redraw.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import torch

from .full_ablation_data import _tensor_hash
from .runtime import sha256
from .support_replication_runtime import (
    DEFAULT_ARCHIVE_ROOT, SUPPORT_RELATIVE_ROOT, support_records,
)
from .support_schema import (
    EXCLUDED_COORDINATES, LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups,
)
from .unranked_selection import select_pair_indices


SEEDS = (20260621, 20260622, 20260623)
RANDOM_SUPPORTS = tuple(f'random_semantic_s{seed}' for seed in SEEDS)
SOURCE_ARMS = ('semantic', *RANDOM_SUPPORTS, 'ranking24')
VARIANTS = ('full', 'matched_random_pairs', *RANDOM_SUPPORTS, 'ranking24',
            'no_support28', 'no_residual', 'self_only', 'base_continuation',
            'no_kl', 'single_kl')
RANKING24_SHA256 = '611f09453fd79dfcbd619f7d4c40d28a288f7efaa6755c140b3497347e4c13e6'


def _boolean_mask(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.dtype != torch.bool or value.shape != (30,):
            raise ValueError('Require a CPU bool[30] source mask')
        return value.detach().clone()
    if (not isinstance(value, (list, tuple)) or len(value) != 30
            or any(type(x) is not bool for x in value)):
        raise ValueError('Require exactly thirty boolean source-mask entries')
    return torch.tensor(value, dtype=torch.bool)


def validate_source_masks(source_masks):
    """Validate all five frozen lists and return owned CPU masks.

    The four semantic lists must retain whole source-defined groups.  Ranking24
    intentionally can split groups; it is the first24 entries of the original
    importance order after focal exclusions, not a newly fitted ranking.
    """
    if not isinstance(source_masks, Mapping) or set(source_masks) != set(SOURCE_ARMS):
        raise ValueError('Require semantic, all three random supports, and ranking24')
    masks = {}
    expected_ranking = set(LEGAL_COORDINATES[:24])
    for name in SOURCE_ARMS:
        mask = _boolean_mask(source_masks[name])
        active = {c for c, keep in zip(SELECTED_COORDINATES, mask.tolist()) if keep}
        if len(active) != 24 or active.intersection(EXCLUDED_COORDINATES):
            raise ValueError('Every frozen source support must be legal exact24')
        if name == 'ranking24':
            if active != expected_ranking:
                raise ValueError('Ranking24 differs from the original legal importance prefix')
        elif any(bool(active.intersection(group)) and not set(group) <= active
                 for group in support_groups('semantic')):
            raise ValueError('Semantic source support splits a source-defined group')
        masks[name] = mask
    return masks


def discover_supports(archive_root=DEFAULT_ARCHIVE_ROOT):
    """Read/hash-check five historical support JSONs only, never checkpoints.

    Returns JSON-compatible ``source_masks``, provenance ``records``, and an
    absolute-path ``bound_files`` mapping for the experiment registration.
    The parent workflow is responsible for pinning these in its plan.
    """
    root = Path(archive_root).resolve()
    records = support_records(root)
    path = (root / SUPPORT_RELATIVE_ROOT / 'ranking24' / 'support.json').resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or sha256(path) != RANKING24_SHA256):
        raise ValueError('Missing or changed registered ranking24 support')
    value = json.loads(path.read_text())
    if (value.get('arm') != 'ranking24'
            or value.get('selected_coordinates') != list(SELECTED_COORDINATES)
            or value.get('hard_support_capacity') != 24
            or value.get('spec') != dict(kind='fixed', control='ranking', budget=24, seed=20260621)):
        raise ValueError('Registered ranking24 identity or coordinate layout changed')
    records['ranking24'] = dict(path=str(path), sha256=RANKING24_SHA256,
        selected_coordinates=list(SELECTED_COORDINATES),
        hard_support=value['hard_support'], hard_support_capacity=24, spec=value['spec'])
    masks = validate_source_masks({name: records[name]['hard_support'] for name in SOURCE_ARMS})
    bound_files = {row['path']: row['sha256'] for row in records.values()}
    for filename, digest in bound_files.items():
        if sha256(filename) != digest:
            raise ValueError('Historical support JSON changed: ' + filename)
    return dict(source_masks={name: mask.tolist() for name, mask in masks.items()},
                records=records, bound_files=bound_files)


def build_policy(inputs, variant, seed, source_masks):
    """Return isolated selection tensors/support for one predeclared case.

    The underlying selector validates TRAIN identity, eligible true-negative
    candidates, disjoint clean supervision and original cache indices.  It
    constructs the same paired draw even in historical arms, making the
    nonattachment background common to all twelve variants at each seed.
    """
    if variant not in VARIANTS:
        raise ValueError('Unregistered current-module ablation variant')
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unregistered current-module ablation seed')
    masks = validate_source_masks(source_masks)
    selection_arm = 'stratified_random' if variant == 'matched_random_pairs' else 'historical'
    selected = select_pair_indices(inputs, selection_arm, seed)
    if variant == 'no_support28':
        support_name = 'legal28'
        support = torch.tensor([c in LEGAL_COORDINATES for c in SELECTED_COORDINATES], dtype=torch.bool)
    else:
        support_name = variant if variant in (*RANDOM_SUPPORTS, 'ranking24') else 'semantic'
        support = masks[support_name].clone()
    capacity = int(support.sum())
    if capacity != (28 if variant == 'no_support28' else 24):
        raise ValueError('Case support capacity differs from registered module deletion')
    active = [c for c, enabled in zip(SELECTED_COORDINATES, support.tolist()) if enabled]
    if set(active).intersection(EXCLUDED_COORDINATES):
        raise ValueError('Current-module ablation cannot write focal metadata')
    record = dict(selected['record'], variant=variant, seed=seed,
        selection_arm=selection_arm, support_name=support_name,
        selected_coordinates=list(SELECTED_COORDINATES), hard_support=support.tolist(),
        active_coordinates=active, support_capacity=capacity,
        support_sha256=_tensor_hash(support), legal_coordinate_count=28,
        support_budget_matched=variant != 'no_support28',
        no_support28_interpretation=('remove learned24 restriction within fixed legal28; '
            'not an equal-coordinate-budget comparison' if variant == 'no_support28' else None),
        support_is_preexisting_fixed_list=variant != 'no_support28',
        feature_selector_retrained=False, current_support_selection_uses_model_outputs=False,
        support_selection_uses_current_dev=False,
        support_policy='fixed_historical_lists_not_selector_retraining',
        historical_learned_support_provenance_retained=support_name == 'semantic',
        original_label_definition_unchanged=True, graph_topology_unchanged=True,
        common_background_shared_across_all_variants_within_seed=True,
        no_attack_checkpoint_loaded_by_selection=True)
    return dict(idx_train=selected['idx_train'].clone(),
        idx_attach=selected['idx_attach'].clone(), fixed_outer=selected['fixed_outer'].clone(),
        support=support.clone(), candidate_pool=selected['candidate_pool'].clone(), record=record)
