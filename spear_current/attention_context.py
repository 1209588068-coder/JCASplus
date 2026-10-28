# Source: experiments/spear_content_attention_v1/context.py; mechanically migrated; algorithms and state keys unchanged.
"""Frozen within-target score permutations; no features, labels or assets read.

Attention does not consume endpoint roles. The inherited RoleContext keeps the
already audited edge/batch contract; edge_type is carried only for compatibility.
Permutations preserve every target's weight multiset, never the corresponding
source-weight association. They are fixed independently of learned scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch

from .role_context import RoleContext


@dataclass(frozen=True)
class AttentionContext(RoleContext):
    score_permutation: torch.Tensor

    def validate(self):
        super().validate()
        permutation = self.score_permutation
        count = self.edge_index.shape[1]
        if (not isinstance(permutation, torch.Tensor) or permutation.dtype != torch.long
                or permutation.device != self.edge_index.device or permutation.shape != (count,)):
            raise ValueError('Expected a same-device long score permutation of shape [E]')
        if (bool(((permutation < 0) | (permutation >= count)).any())
                or permutation.unique().numel() != count):
            raise ValueError('Score permutation must be a bijection of the edge rows')
        if not torch.equal(self.edge_index[1], self.edge_index[1, permutation]):
            raise ValueError('Score permutation must stay within the same target')

    def to(self, device):
        self.validate()
        return AttentionContext(self.edge_index.to(device).clone(), self.edge_type.to(device).clone(),
            self.pair_scene.to(device).clone(), self.num_pairs,
            self.score_permutation.to(device).clone())


def attention_context(context, scene_ids, seed=20260621):
    """Build a private-RNG permutation stable under edge order and scene batching.

The caller supplies verified scene IDs aligned with pair_scene. Local pair order
must remain the original canonical pair order; it is not reconstructed here.
Missing scenes, including empty scenes, must not be silently renumbered. No retry
or forced change is performed if a permutation happens to equal the identity.
"""
    if not isinstance(context, RoleContext):
        raise ValueError('Require original pair/edge context')
    context.validate()
    if (not isinstance(scene_ids, (list, tuple)) or (not scene_ids and context.num_pairs > 0)
            or any(not isinstance(value, str) or not value.strip() for value in scene_ids)
            or len(set(scene_ids)) != len(scene_ids)):
        raise ValueError('Require unique verified scene IDs in scene-index order')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative Python integer')
    scenes = context.pair_scene.detach().cpu().numpy()
    if len(scenes) and int(scenes.max()) >= len(scene_ids):
        raise ValueError('Scene IDs do not cover original pair_scene indices')
    edges = context.edge_index.detach().cpu().numpy()
    permutation = np.arange(edges.shape[1], dtype=np.int64)
    local_pair = np.empty(context.num_pairs, dtype=np.int64)
    for scene in range(len(scene_ids)):
        indices = np.flatnonzero(scenes == scene)
        local_pair[indices] = np.arange(len(indices))
    if edges.shape[1]:
        ordered = np.lexsort((local_pair[edges[0]], edges[1]))
        targets = edges[1, ordered]
        boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(targets)), len(ordered)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            indices = ordered[lo:hi]
            if len(indices) < 2:
                continue
            target = int(targets[lo])
            payload = json.dumps([seed, 'spear-content-score-permutation-v1',
                scene_ids[int(scenes[target])], int(local_pair[target])],
                ensure_ascii=False, separators=(',', ':')).encode()
            private_seed = int(hashlib.sha256(payload).hexdigest(), 16)
            rng = np.random.Generator(np.random.PCG64(private_seed))
            permutation[indices] = rng.permutation(indices)
    return AttentionContext(context.edge_index.detach().clone(), context.edge_type.detach().clone(),
        context.pair_scene.detach().clone(), context.num_pairs,
        torch.as_tensor(permutation, dtype=torch.long, device=context.edge_index.device).clone())


def permutation_stats(context):
    if not isinstance(context, AttentionContext):
        raise ValueError('Require AttentionContext')
    context.validate()
    edges = context.edge_index
    degree = torch.bincount(edges[1], minlength=context.num_pairs)
    changed = context.score_permutation != torch.arange(edges.shape[1], device=edges.device)
    return dict(num_pairs=context.num_pairs, total_edges=edges.shape[1],
        targets_with_multiple_neighbors=int((degree > 1).sum()),
        targets_without_shuffle_freedom=int((degree <= 1).sum()),
        changed_edge_assignments=int(changed.sum()),
        changed_targets=int(edges[1, changed].unique().numel()),
        changed_assignment_fraction=float(changed.float().mean()) if changed.numel() else None,
        per_target_score_multisets_preserved=True, forced_changes=False,
        note='Assignment changes are not proof of different learned weights or performance')
