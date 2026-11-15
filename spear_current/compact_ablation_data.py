"""Seven fixed-scope data policies for Full and six module ablations.

The five already-run policies are returned byte-for-byte at the record level
from ``current_ablation_data`` so completed evidence can be reused. RF is a
single uniform coordinate draw of24 from the same legal28 permissions; it does
not impose semantic groups. NE retains Full's data and support, changing only
the generator's conditioning implementation in the separate training module.
No model, graph-array reader, optimizer or evaluation runs here.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from .current_ablation_data import (
    SEEDS, discover_supports, build_policy as _original_policy,
)
from .full_ablation_data import _tensor_hash
from .support_schema import (
    EXCLUDED_COORDINATES, LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups,
)


VARIANTS = ('full', 'matched_random_pairs', 'random_features',
            'no_generator_neighborhood', 'no_residual', 'self_only', 'no_kl')
REUSABLE_VARIANTS = ('full', 'matched_random_pairs', 'no_residual', 'self_only', 'no_kl')


def build_policy(inputs, variant, seed, source_masks):
    """Preserve paired selections and return a legal final24 support policy.

    At each seed, every arm uses the original candidate/clean populations and
    the exact common512 background of the twelve-arm predecessor. RF's private
    NumPy RNG is independent of global state and is used once, with no redraw
    or outcome-based selection. All returned arrays own their storage.
    """
    if variant not in VARIANTS:
        raise ValueError('Unregistered six-ablation variant')
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unregistered six-ablation seed')
    if variant in REUSABLE_VARIANTS:
        return _original_policy(inputs, variant, seed, source_masks)

    result = _original_policy(inputs, 'full', seed, source_masks)
    record = deepcopy(result['record'])
    record['variant'] = variant
    if variant == 'no_generator_neighborhood':
        record.update(
            pointwise_conditioning=True,
            generator_conditioning='shared_weight_MLP_readout_of_shadow_hidden_linears',
            residual_neighbor_input=False,
            classification_shadow_graph_unchanged=True,
            neighborhood_ablation_scope='both_explicit_generator_paths; not classifier_or_victim_topology',
            full_data_and_support_unchanged=True,
        )
    else:
        legal_positions = np.asarray([i for i, c in enumerate(SELECTED_COORDINATES)
                                      if c in LEGAL_COORDINATES], dtype=np.int64)
        if len(legal_positions) != 28:
            raise ValueError('RF permission domain must be the existing legal28')
        selected = np.random.RandomState(seed).choice(legal_positions, size=24, replace=False)
        support = torch.zeros(30, dtype=torch.bool)
        support[torch.from_numpy(selected)] = True
        active = [c for c, enabled in zip(SELECTED_COORDINATES, support.tolist()) if enabled]
        if len(active) != 24 or set(active).intersection(EXCLUDED_COORDINATES):
            raise ValueError('RF escaped the legal24-coordinate budget')
        splits = [list(group) for group in support_groups('semantic')
                  if set(active).intersection(group) and not set(group) <= set(active)]
        result['support'] = support
        record.update(
            support_name=f'random_coordinate24_s{seed}', hard_support=support.tolist(),
            active_coordinates=active, support_capacity=24,
            support_sha256=_tensor_hash(support), support_budget_matched=True,
            support_is_preexisting_fixed_list=False,
            historical_learned_support_provenance_retained=False,
            support_policy='single_uniform_coordinate_draw_from_fixed_legal28',
            source_permission_domain='original_prefiltered_legal28_not_all784',
            feature_draw_seed=seed, feature_draw_count=1, feature_draw_replace=False,
            feature_redraws=0, feature_draw_order=[SELECTED_COORDINATES[int(i)] for i in selected],
            semantic_group_constraint=False, split_semantic_groups=splits,
        )
    result['record'] = record
    return result
