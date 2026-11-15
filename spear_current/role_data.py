# Source: experiments/spear_pair_role_generator_v1/data.py; mechanically migrated; algorithms unchanged.
"""Pure pair-role sidecars and a component-bound, read-only injector adapter.

The existing reader/normalizer owns graph loading and split authorization. Its
parts do not originally contain scenario IDs or split IDs; the workflow MUST
attach the verified scenario_id from its original manifest row. No ID is
invented from batch order. This module does not read files, train, normalize,
select targets, or claim that an absent split field proves TRAIN provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch

from .role_context import RoleContext, build_role_context, scrambled_context
from .role_generator import PairRoleResidualGenerator


FEATURE_DIM = 784
COORDINATE_COUNT = 30


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(value)


def _tensor(value, name, dtype):
    numpy_dtype = {torch.float32: np.dtype('float32'), torch.long: np.dtype('int64'), torch.bool: np.dtype('bool')}[dtype]
    if isinstance(value, torch.Tensor):
        if value.dtype != dtype or value.layout != torch.strided:
            raise ValueError(f'{name} has an unexpected tensor dtype/layout')
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, np.ndarray) and value.dtype == numpy_dtype:
        return torch.from_numpy(value.copy()).contiguous()
    raise ValueError(f'{name} must be a {numpy_dtype} NumPy array or matching torch tensor')


def _tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)), sort_keys=True).encode())
    digest.update(b'\0'); digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _context_hash(context):
    context.validate()
    return _json_hash(dict(num_pairs=context.num_pairs, **{
        key: _tensor_hash(getattr(context, key)) for key in ('edge_index', 'edge_type', 'pair_scene')}))


def _identity(part, metadata, key, *, required):
    values = [container[key] for container in (part, metadata) if key in container]
    if not values:
        if required:
            raise ValueError(f'Missing verified {key}; batch-order placeholders are not permitted')
        return None
    if any(not isinstance(value, str) or not value.strip() for value in values) or len(set(values)) != 1:
        raise ValueError(f'Invalid or conflicting {key} in component and metadata')
    return values[0]


def _part(part):
    if not isinstance(part, Mapping) or not isinstance(part.get('metadata'), Mapping):
        raise ValueError('Each component needs its original metadata mapping')
    try:
        metadata = json.loads(json.dumps(dict(part['metadata']), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('Component metadata must be finite JSON data') from error
    scenario = _identity(part, metadata, 'scenario_id', required=True)
    split = _identity(part, metadata, 'split', required=False)
    fields = {}
    for key, dtype in (('x', torch.float32), ('y', torch.long), ('eligible', torch.bool),
                       ('canonical_pairs', torch.long), ('edge_index', torch.long)):
        if key not in part:
            raise ValueError(f'Missing component field {key}')
        fields[key] = _tensor(part[key], key, dtype)
    n = len(fields['x']) if fields['x'].ndim else -1
    if fields['x'].shape != (n, FEATURE_DIM) or not bool(torch.isfinite(fields['x']).all()):
        raise ValueError('Component must have finite normalized float32 x[P,784]')
    if fields['y'].shape != (n,) or fields['eligible'].shape != (n,) or fields['canonical_pairs'].shape != (n, 2):
        raise ValueError('Component pair, label or eligibility row ordering is misaligned')
    if not bool(torch.isin(fields['y'], torch.tensor([-1, 0, 1])).all()):
        raise ValueError('Original pair labels must be -1, 0 or 1')
    if not bool(torch.isin(fields['y'][fields['eligible']], torch.tensor([0, 1])).all()):
        raise ValueError('Eligible pair labels must be binary')
    if 'feature_dimension' in metadata and metadata['feature_dimension'] != FEATURE_DIM:
        raise ValueError('Metadata feature dimension differs from normalized input')
    typed = build_role_context(fields['canonical_pairs'], fields['edge_index'])
    hashes = {key: _tensor_hash(value) for key, value in fields.items()}
    hashes['metadata'] = _json_hash(metadata)
    # Track IDs are optional here, but cannot silently misalign when supplied.
    if 'canonical_pair_track_ids' in part:
        ids = part['canonical_pair_track_ids']
        if (not isinstance(ids, np.ndarray) or ids.shape != (n, 2) or ids.dtype.kind not in 'US'
                or any(not str(value) for value in ids.reshape(-1))):
            raise ValueError('Canonical track IDs must be a string [P,2] NumPy array')
        hashes['canonical_pair_track_ids'] = _json_hash(ids.tolist())
    return dict(scenario_id=scenario, observed_split_field=split, fields=fields,
                typed=typed, hashes=hashes)


def _scene_seed(seed, scenario_id):
    payload = json.dumps([seed, 'spear-pair-role-shuffle-v1', scenario_id],
                         ensure_ascii=False, separators=(',', ':')).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest, 16), digest


def _offsets(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        result = _tensor(value, 'combined offsets', torch.long)
        if result.ndim != 1:
            raise ValueError('Combined offsets must be one-dimensional')
        return result.tolist()
    if (not isinstance(value, (list, tuple))
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value)):
        raise ValueError('Combined offsets must be an integer sequence')
    return list(map(int, value))


def build_contexts(parts, combined=None, *, seed=20260621):
    """Return scene-safe typed/shuffled contexts with exact existing-batch checks.

    Normalized x, y and eligibility are ONLY validated/hashed for alignment.
    Role assignment uses canonical pairs/edges; the shuffle seed additionally
    uses verified scenario identity. The same scene obtains the same shuffled
    roles alone, in another batch position, or after another scene is added.
    """
    seed = _seed(seed)
    if not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError('Require a nonempty ordered component sequence')
    values = [_part(part) for part in parts]
    scenarios = [part['scenario_id'] for part in values]
    if len(set(scenarios)) != len(scenarios):
        raise ValueError('Duplicate scenario IDs in component batch')
    typed_edges, typed_roles, shuffled_roles, pair_scenes, offsets, scene_stats = [], [], [], [], [], []
    offset = 0
    for scene_index, part in enumerate(values):
        context = part['typed']
        scene_seed, seed_hex = _scene_seed(seed, part['scenario_id'])
        shuffled, stats = scrambled_context(context, scene_seed)
        offsets.append(offset)
        typed_edges.append(context.edge_index + offset)
        typed_roles.append(context.edge_type.clone())
        shuffled_roles.append(shuffled.edge_type.clone())
        pair_scenes.append(torch.full((context.num_pairs,), scene_index, dtype=torch.long))
        scene_stats.append(dict(scenario_id=part['scenario_id'], offset=offset,
            num_pairs=context.num_pairs, num_edges=context.edge_index.shape[1],
            shuffle_seed_sha256=seed_hex, shuffle=stats,
            typed_context_sha256=_context_hash(context), shuffled_context_sha256=_context_hash(shuffled)))
        offset += context.num_pairs
    edges = torch.cat(typed_edges, dim=1)
    scenes = torch.cat(pair_scenes)
    typed = RoleContext(edges.clone(), torch.cat(typed_roles), scenes.clone(), offset)
    shuffled = RoleContext(edges.clone(), torch.cat(shuffled_roles), scenes.clone(), offset)
    expected = {key: torch.cat([part['fields'][key] for part in values]) for key in ('x', 'y', 'eligible')}
    expected['edge_index'] = edges
    if combined is not None:
        if not isinstance(combined, Mapping) or 'offsets' not in combined:
            raise ValueError('Existing combined data must include exact scene offsets')
        if _offsets(combined['offsets']) != offsets:
            raise ValueError('Existing combined scene offsets/order changed')
        for key, reference in expected.items():
            if key not in combined or not torch.equal(_tensor(combined[key], 'combined ' + key, reference.dtype), reference):
                raise ValueError(f'Existing combined {key} differs from components in original order')
    binding = dict(schema='spear_pair_role_component_binding_v1', seed=seed,
        scene_order=scenarios.copy(), offsets=offsets.copy(), num_pairs=offset,
        combined_alignment_checked=combined is not None,
        combined_arrays_sha256={key: _tensor_hash(value) for key, value in expected.items()},
        scene_inputs=[dict(scenario_id=part['scenario_id'], observed_split_field=part['observed_split_field'],
                           arrays_sha256=deepcopy(part['hashes'])) for part in values],
        typed_context_sha256=_context_hash(typed), shuffled_context_sha256=_context_hash(shuffled),
        seed_derivation='SHA256(JSON([seed,spear-pair-role-shuffle-v1,scenario_id])) as unsigned integer',
        role_sources=['canonical_pairs', 'edge_index'], labels_used_for_roles=False,
        labels_used_for_shuffle=False, speed_used_for_roles=False,
        feature_normalization='caller supplied; never fitted or transformed here',
        split_guard='original manifest reader/workflow owns authorization; absent split fields prove nothing')
    return dict(typed=typed, shuffled=shuffled, scene_stats=deepcopy(scene_stats), binding=binding)


class ContextualInjector:
    """Single-component inference adapter for the existing inject_trigger API.

    Rebind for every scene. This signature has no scenario-ID argument: input
    equality rejects stale feature/edge tensors, but cannot distinguish two
    distinct scenes with literally identical tensors. The caller owns the
    verified reader-to-component identity link and immediate per-scene binding.
    """

    def __init__(self, generator, component, role_mode='typed', seed=20260621):
        if not isinstance(generator, PairRoleResidualGenerator):
            raise ValueError('Require PairRoleResidualGenerator')
        if role_mode not in ('typed', 'shuffled', 'untyped'):
            raise ValueError('role_mode must be typed, shuffled or untyped')
        self.generator = generator
        self.role_mode = role_mode
        self.args = SimpleNamespace(target_class=1)
        self._expected_residual_mode = 'untyped' if role_mode == 'untyped' else 'typed'
        self._check_model()
        contexts = build_contexts([component], seed=seed)
        self._context = contexts['shuffled' if role_mode == 'shuffled' else 'typed'].to('cpu')
        self._feature_cpu = _tensor(component['x'], 'component x', torch.float32)
        self._edge_cpu = self._context.edge_index.clone()
        self._expected_context_hash = _context_hash(self._context)
        self._expected_feature_hash = _tensor_hash(self._feature_cpu)
        self._selected_cpu = generator.selected_coordinates.detach().cpu().clone()
        self._binding = deepcopy(contexts['binding'])
        self._binding.update(role_mode=role_mode, selected_coordinates=self._selected_cpu.tolist(),
                             inference_only=True, context_sha256=self._expected_context_hash)

    @property
    def device(self):
        return self.generator.selected_coordinates.device

    @property
    def binding(self):
        return deepcopy(self._binding)

    @property
    def context(self):
        return self._context.to('cpu')

    def _check_model(self):
        generator = self.generator
        if generator.input_dim != FEATURE_DIM or generator.residual.mode != self._expected_residual_mode:
            raise ValueError('Generator feature dimension or role mode differs from requested adapter')
        if any(module.training for module in generator.modules()):
            raise ValueError('Caller must explicitly put the generator and all children in eval mode')
        coordinates = generator.selected_coordinates
        if (coordinates.dtype != torch.long or coordinates.shape != (COORDINATE_COUNT,)
                or coordinates.unique().numel() != COORDINATE_COUNT
                or bool(((coordinates < 0) | (coordinates >= FEATURE_DIM)).any())):
            raise ValueError('Expected the frozen ordered 30-coordinate generator contract')
        if any(p.device != coordinates.device or p.dtype != torch.float32 for p in generator.parameters()):
            raise ValueError('Generator parameters must be float32 on the selected-coordinate device')

    def inject_trigger(self, indices, x, edge_index, edge_weight, device):
        self._check_model()
        requested = torch.device(device)
        if requested != self.device:
            raise ValueError('Requested injection device differs from frozen generator device')
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.device != requested
                or not isinstance(edge_index, torch.Tensor) or edge_index.dtype != torch.long or edge_index.device != requested
                or not isinstance(edge_weight, torch.Tensor) or edge_weight.dtype != torch.float32 or edge_weight.device != requested
                or not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.device != requested):
            raise ValueError('Injection tensor dtype/device contract changed')
        if (_context_hash(self._context) != self._expected_context_hash
                or _tensor_hash(self._feature_cpu) != self._expected_feature_hash
                or not torch.equal(self.generator.selected_coordinates.detach().cpu(), self._selected_cpu)):
            raise ValueError('Bound component context, features or selected coordinates were modified')
        if not torch.equal(x.detach().cpu(), self._feature_cpu) or not torch.equal(edge_index.detach().cpu(), self._edge_cpu):
            raise ValueError('Stale or changed component features/edges; bind the verified scene again')
        context = self._context.to(requested)
        with torch.no_grad():
            injected = self.generator.inject(x.clone(), context, indices.clone(), edge_weight.clone(), detach=True)
        if (injected.features.shape != x.shape or injected.features.dtype != x.dtype
                or injected.features.device != requested or not bool(torch.isfinite(injected.features).all())
                or not torch.equal(injected.edge_index, edge_index) or not torch.equal(injected.edge_weight, edge_weight)):
            raise ValueError('Generator violated feature/topology preservation contract')
        permitted = torch.zeros_like(x, dtype=torch.bool)
        permitted[indices[:, None], self.generator.selected_coordinates] = True
        if not torch.equal(injected.features[~permitted], x[~permitted]):
            raise ValueError('Generator changed a non-target row or unselected coordinate')
        return injected.features.detach().clone(), injected.edge_index.detach().clone(), injected.edge_weight.detach().clone()


def bind_component(generator, component, role_mode='typed', seed=20260621):
    return ContextualInjector(generator, component, role_mode=role_mode, seed=seed)
