# Source: experiments/spear_pair_role_generator_v1/context.py; mechanically migrated; algorithms and state keys unchanged.
"""Endpoint-role sidecars derived only from pair identities and line edges.

For a source pair and a target pair sharing one vehicle, the role is
``2 * source_endpoint_position + target_endpoint_position``. No labels,
speeds, model outputs, learned state, or external graph assets are consumed.
Line-graph self edges are excluded from this four-role contract; any model
self branch or internally added self loops are separate from this sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RoleContext:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    pair_scene: torch.Tensor
    num_pairs: int

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Recheck structural invariants, including mutable tensor contents.

        This checks role RANGE, not role truth: scrambled contexts intentionally
        have different assignments. Canonical identities are checked by the
        builder and are deliberately not carried as features into this object.
        """
        if type(self.num_pairs) is not int or self.num_pairs < 0:
            raise ValueError('num_pairs must be a nonnegative Python integer')
        tensors = (self.edge_index, self.edge_type, self.pair_scene)
        if any(not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long for tensor in tensors):
            raise ValueError('Context indices, roles and scene IDs must be torch.long tensors')
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError('Context tensors must share one device')
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape [2, E]')
        edge_count = self.edge_index.shape[1]
        if self.edge_type.shape != (edge_count,) or self.pair_scene.shape != (self.num_pairs,):
            raise ValueError('Context roles or scene assignments are misaligned')
        if bool((self.pair_scene < 0).any()):
            raise ValueError('Scene IDs must be nonnegative')
        if bool(((self.edge_type < 0) | (self.edge_type >= 4)).any()):
            raise ValueError('Endpoint roles must be in [0, 3]')
        if bool(((self.edge_index < 0) | (self.edge_index >= self.num_pairs)).any()):
            raise ValueError('Line-graph edge endpoint is out of pair-node range')
        src, dst = self.edge_index
        if bool((src == dst).any()):
            raise ValueError('Line self edges have no unique four-role shared endpoint')
        if not torch.equal(self.pair_scene[src], self.pair_scene[dst]):
            raise ValueError('Line edge crosses scene boundaries')
        if torch.unique(self.edge_index.T, dim=0).shape[0] != edge_count:
            raise ValueError('Duplicate directed line edge')
        return None

    def to(self, device):
        """Return validated, independently owned tensors on the requested device."""
        self.validate()
        return RoleContext(self.edge_index.to(device=device).clone(),
                           self.edge_type.to(device=device).clone(),
                           self.pair_scene.to(device=device).clone(), self.num_pairs)


def _int64_cpu(value, name):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.long:
            raise ValueError(f'{name} must be torch.long or NumPy int64')
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray) and value.dtype == np.dtype('int64'):
        return torch.from_numpy(value.copy())
    raise ValueError(f'{name} must be torch.long or NumPy int64')


def build_role_context(canonical_pairs, edge_index, pair_scene=None):
    """Build an owned CPU context, preserving pair endpoint and edge order.

    ``canonical_pairs[P,2]`` contains original vehicle indices, which may repeat
    across different scenes. Endpoint order is meaningful and is NOT sorted.
    Duplicate unordered vehicle pairs within one scene, self pairs, line self
    edges, and line edges without exactly one shared vehicle are rejected.
    """
    pairs = _int64_cpu(canonical_pairs, 'canonical_pairs')
    edges = _int64_cpu(edge_index, 'edge_index')
    if pairs.ndim != 2 or pairs.shape[1] != 2 or bool((pairs < 0).any()):
        raise ValueError('canonical_pairs must contain nonnegative vehicle indices with shape [P, 2]')
    n = len(pairs)
    scenes = torch.zeros(n, dtype=torch.long) if pair_scene is None else _int64_cpu(pair_scene, 'pair_scene')
    if scenes.shape != (n,) or bool((scenes < 0).any()):
        raise ValueError('pair_scene must contain one nonnegative scene ID per pair')
    if bool((pairs[:, 0] == pairs[:, 1]).any()):
        raise ValueError('A vehicle cannot form a pair with itself')
    unordered = pairs.sort(dim=1).values
    identities = torch.cat((scenes[:, None], unordered), dim=1)
    if torch.unique(identities, dim=0).shape[0] != n:
        raise ValueError('Duplicate unordered vehicle pair in the same scene')
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('edge_index must have shape [2, E]')
    # Validate indices before indexing canonical pair identities.
    context = RoleContext(edges, torch.zeros(edges.shape[1], dtype=torch.long), scenes, n)
    src, dst = context.edge_index
    shared = pairs[src, :, None] == pairs[dst, None, :]
    if not bool((shared.sum(dim=(1, 2)) == 1).all()):
        raise ValueError('Each line edge must share exactly one original vehicle')
    role_ids = torch.arange(4, dtype=torch.long).reshape(2, 2)
    roles = (shared.to(torch.long) * role_ids).sum(dim=(1, 2))
    return RoleContext(edges, roles, scenes, n)


def scrambled_context(context, seed):
    """Shuffle incoming role assignments per target, retaining its role counts.

    A private PCG64 generator processes targets in ID order and incoming edges
    in source-ID order. Therefore edge-list reorderings do not change the
    per-edge scrambled assignment. This is not a global role-name permutation.
    No retry or forced swap is used: random unchanged outcomes are reported.
    """
    if not isinstance(context, RoleContext):
        raise ValueError('Require a RoleContext')
    context.validate()
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    edges = context.edge_index.detach().cpu().numpy()
    original = context.edge_type.detach().cpu().numpy().copy()
    scrambled = original.copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    sorted_edges = np.lexsort((edges[0], edges[1]))
    sorted_targets = edges[1, sorted_edges]
    boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_targets)), len(sorted_edges)]
    incoming_targets = diverse_targets = changed_targets = diverse_edges = 0
    if len(sorted_edges):
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            indices = sorted_edges[lo:hi]
            roles = original[indices]
            incoming_targets += 1
            if len(np.unique(roles)) > 1:
                diverse_targets += 1
                diverse_edges += len(indices)
                scrambled[indices] = rng.permutation(roles)
            if not np.array_equal(np.bincount(roles, minlength=4),
                                  np.bincount(scrambled[indices], minlength=4)):
                raise ValueError('Scrambling changed an incoming target role histogram')
            changed_targets += int(np.any(scrambled[indices] != roles))
    changed = int(np.count_nonzero(scrambled != original))
    total = len(original)
    result = RoleContext(context.edge_index.detach().clone(),
        torch.as_tensor(scrambled, dtype=torch.long, device=context.edge_type.device).clone(),
        context.pair_scene.detach().clone(), context.num_pairs)
    stats = dict(
        algorithm='within_target_incoming_permutation_source_sorted_v1', seed=seed,
        total_edges=total, changed_edges=changed, unchanged_edges=total - changed,
        changed_fraction=changed / total if total else None,
        num_pairs=context.num_pairs, targets_with_incoming_edges=incoming_targets,
        targets_without_incoming_edges=context.num_pairs - incoming_targets,
        targets_with_role_diversity=diverse_targets,
        targets_without_role_diversity=context.num_pairs - diverse_targets,
        targets_with_changed_edges=changed_targets, edges_at_diverse_targets=diverse_edges,
        no_shuffle_freedom=diverse_targets == 0,
        status='NO_ROLE_DIVERSITY' if diverse_targets == 0 else
               ('SHUFFLED' if changed else 'SHUFFLED_NO_ASSIGNMENT_CHANGE'),
        per_target_role_histograms_preserved=True, forced_changes=False,
        global_role_counts_before=np.bincount(original, minlength=4).tolist(),
        global_role_counts_after=np.bincount(scrambled, minlength=4).tolist())
    return result, stats
