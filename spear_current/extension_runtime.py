"""Restricted deployment of the frozen uniform generator, without retraining it.

The inherited hard_support is the original 24-coordinate permission envelope,
needed by the unchanged registered fitter. effective_support is the actual
write set for this experiment; both counts must be reported separately.
Missingness applies only to the prefiltered legal28 derived attributes.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch

from .support_replication_runtime import FrozenSupportGenerator
from .support_schema import LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups


def checked_coordinates(coordinates):
    values = tuple(coordinates)
    if (any(type(c) is not int for c in values) or len(values) != len(set(values))
            or not set(values) <= set(LEGAL_COORDINATES)):
        raise ValueError('Require unique coordinates within legal28')
    chosen = set(values)
    for group in support_groups('semantic'):
        if chosen.intersection(group) and not set(group) <= chosen:
            raise ValueError('Permission/missingness splits a semantic group')
    return values


class RestrictedSupportGenerator(FrozenSupportGenerator):
    """The source24 envelope stays fixed; effective writes can only shrink."""

    def __init__(self, anchor, source_mask, active_coordinates):
        super().__init__(anchor, source_mask)
        active = set(checked_coordinates(active_coordinates))
        permitted = set(self.selected_coordinates[self.hard_support].tolist())
        if not active <= permitted:
            raise ValueError('Effective support escaped the frozen source24')
        effective = torch.tensor([c in active for c in SELECTED_COORDINATES],
                                 dtype=torch.bool, device=self.hard_support.device)
        self.register_buffer('effective_support', effective)
        self._effective_mask = tuple(effective.tolist())

    def forward(self, features, context, targets):
        if (tuple(self.effective_support.tolist()) != self._effective_mask
                or bool((self.effective_support & ~self.hard_support).any())):
            raise ValueError('Effective write permission changed')
        values = super().forward(features, context, targets)
        clean = features[targets[:, None], self.selected_coordinates]
        # torch.where preserves forbidden slots exactly, including signed zero.
        return torch.where(self.effective_support[None, :], values, clean)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get('effective_support')
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), self.effective_support.cpu()):
            raise ValueError('Restoration cannot change effective write permission')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def missing_view(inputs, coordinates):
    """Owned changed features; all labels/edges/identities retain original values.

    No fabricated history or validity changes. This is column loss in a 28D
    candidate attribute space, not feature loss across the full 784D input.
    """
    columns = checked_coordinates(coordinates)
    out = dict(inputs)
    out['combined'], out['parts'] = {}, {}
    for role in ('train', 'dev'):
        original = inputs['combined'][role]
        graph = dict(original)
        graph['x'] = original['x'].clone()
        if columns:
            graph['x'][:, list(columns)] = 0
        out['combined'][role] = graph
        out['parts'][role] = []
        for part, offset in zip(inputs['parts'][role], original['offsets']):
            source_view = original['x'][offset:offset + len(part['x'])]
            if not torch.equal(torch.as_tensor(part['x']), source_view):
                raise ValueError('Original component/combined features were not aligned')
            changed = dict(part)
            view = graph['x'][offset:offset + len(part['x'])]
            changed['x'] = view if torch.is_tensor(part['x']) else view.numpy()
            out['parts'][role].append(changed)
        if role in inputs:
            out[role] = graph
    return out


def budget_selection(inputs, count):
    """Nested prefixes of the already-trained-on 26 attachments, no reselection."""
    original = inputs['indices']['idx_attach']
    if type(count) is not int or not 0 <= count <= len(original):
        raise ValueError('Invalid registered attachment count')
    attach = original[:count].clone()
    clean = inputs['indices']['idx_train'].clone()
    # fixed_outer is an interface contract; no outer/shadow optimization occurs.
    background = inputs['outer'][len(original):].clone()
    return dict(idx_train=clean, idx_attach=attach,
                fixed_outer=torch.cat([attach, background]))


def check_effective_writes(generator, inputs, coordinates, device):
    """Exercise all TRAIN attachments, checking the stricter effective support."""
    active = set(checked_coordinates(coordinates))
    graph = inputs['combined']['train']
    x, edges = graph['x'].to(device), graph['edge_index'].to(device)
    targets = inputs['indices']['idx_attach'].to(device)
    weights = x.new_ones(edges.shape[1])
    with torch.no_grad():
        result = generator.inject(x, SimpleNamespace(edge_index=edges), targets,
                                  weights, detach=True)
    allowed = torch.zeros_like(x, dtype=torch.bool)
    if active:
        allowed[targets[:, None], list(sorted(active))] = True
    if (not torch.equal(result.features[~allowed], x[~allowed])
            or not torch.equal(result.edge_index, edges)
            or not torch.equal(result.edge_weight, weights)):
        raise ValueError('Effective permission or topology violation')
    return dict(source_envelope_coordinates=int(generator.hard_support.sum()),
                effective_write_coordinates=len(active), exact_permissions_preserved=True,
                train_attachments_checked=len(targets), topology_preserved=True)


def benchmark(generator, inputs, device, *, warmup=5, repeats=30):
    """Synchronized end-to-end single-component injection, no model/G updates."""
    import time
    index = next(i for i, t in enumerate(inputs['targets']['dev']) if t is not None)
    part = inputs['parts']['dev'][index]
    x = torch.as_tensor(part['x'], dtype=torch.float32, device=device)
    edges = torch.as_tensor(part['edge_index'], dtype=torch.long, device=device)
    weights = x.new_ones(edges.shape[1])
    target = torch.tensor([inputs['targets']['dev'][index]], device=device)
    context = SimpleNamespace(edge_index=edges)
    timings = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            generator.inject(x, context, target, weights, detach=True)
            torch.cuda.synchronize(device)
            if i >= warmup:
                timings.append((time.perf_counter() - start) * 1000)
    return dict(kind='synchronized_wall_latency_single_scene_single_target_injection',
                warmup=warmup, repetitions=repeats, milliseconds=timings,
                median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
                scene_pair_nodes=len(x), message_edges=edges.shape[1],
                includes_python_context_and_permission_checks=True,
                excludes_data_loading=True, excludes_historical_generator_training=True)


def identity_digest(inputs):
    value = {k: inputs[k] for k in ('metadata', 'targets', 'populations')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
