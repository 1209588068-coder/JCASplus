"""Explicit restoration of the frozen, unclipped uniform support candidates.

This is separate from the retained domain-constrained runtime. The historical
uniform already used dual clean-KL and was trained with all 30 output slots.
Its fixed 24-coordinate descendants protect both focal metadata fields, but
are not from-scratch metadata-free models. No graph reader or trainer lives
here. Source checks read only registered file bytes and JSON metadata.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

from .attention_context import AttentionContext
from .attention_generator import ContentAttentionGenerator
from .models import GCN, GraphTrojanNet
from .role_generator import InjectedFeatures, PairRoleResidualGenerator, _check_features
from .runtime import sha256
from .support_generator import apply_gates
from .support_schema import LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups


DEFAULT_ARCHIVE_ROOT = Path('/home/mizhou/JCAS_cleanup_20261027_Mp0CdD/legacy/record/reproduction')
ORIGINAL_PREFIX = '/home/mizhou/JCAS/record/reproduction/'
ARMS = ('semantic', 'random_semantic_s20260621', 'random_semantic_s20260622',
        'random_semantic_s20260623')
SUPPORT_SHA256 = {
    'semantic': '322afaf5fc2944b809bbba2ce2dc65ea47ec863e465c455f6173021148ceb625',
    'random_semantic_s20260621': '89ebdb7c15113b8e7fb27e8732b3a1256ea50ae9a27e401a4af9c6fe5dd8b591',
    'random_semantic_s20260622': '26d71550ceb696133fa38ed59904f1a2c61ea33cefe69482658d154fb6be3b18',
    'random_semantic_s20260623': '91f653d49a7daad3ce3ad5463a667bf38683866f79d324d3a9d9ac775935bb0c',
}
UNIFORM_SHA256 = '742570c83a8838ea6a943ad6fb1a2482008d8215558b426fe6255b9161ea5244'
SURROGATE_SHA256 = 'c9ca4480da55af9e448a5c0ba274df7416c6bf6243ecf869fb191feb6067da18'
SUPPORT_RELATIVE_ROOT = Path('spear_semantic_support_v1/pilot_20261026_seed20260621_v1/training')


def _archived_reference(bundle, name, archive_root):
    reference = bundle.config['references'][name]
    original = reference['path']
    if not isinstance(original, str) or not original.startswith(ORIGINAL_PREFIX):
        raise ValueError('Historical reference must use the registered reproduction root')
    relative = Path(original[len(ORIGINAL_PREFIX):])
    root = Path(archive_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or escaping historical reference: ' + name)
    expected = reference['sha256']
    if sha256(path) != expected:
        raise ValueError('Historical reference hash mismatch: ' + name)
    return dict(path=str(path), sha256=expected, original_path=original)


def _mask(mask):
    if isinstance(mask, torch.Tensor):
        if mask.dtype != torch.bool or mask.shape != (30,):
            raise ValueError('Require a 30-slot boolean support')
        result = mask.detach().clone()
    else:
        if (not isinstance(mask, (list, tuple)) or len(mask) != 30
                or any(type(value) is not bool for value in mask)):
            raise ValueError('Require a 30-slot boolean support')
        result = torch.tensor(mask, dtype=torch.bool)
    if int(result.sum()) != 24:
        raise ValueError('Require exactly 24 writable coordinates')
    if any(bool(result[i]) for i, coordinate in enumerate(SELECTED_COORDINATES)
           if coordinate not in LEGAL_COORDINATES):
        raise ValueError('Focal metadata fields are not writable')
    active = {coordinate for coordinate, enabled in zip(SELECTED_COORDINATES, result.tolist()) if enabled}
    if any(bool(active.intersection(group)) and not set(group).issubset(active)
           for group in support_groups('semantic')):
        raise ValueError('Semantic support must retain or remove each complete group')
    return result


def support_records(archive_root=DEFAULT_ARCHIVE_ROOT):
    """Read all four pre-existing masks; never draw or rank a new support."""
    root = Path(archive_root).resolve()
    result = {}
    for arm in ARMS:
        path = (root / SUPPORT_RELATIVE_ROOT / arm / 'support.json').resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != SUPPORT_SHA256[arm]:
            raise ValueError('Missing or changed registered support: ' + arm)
        record = json.loads(path.read_text())
        mask = _mask(record['hard_support'])
        if (record['arm'] != arm or tuple(record['selected_coordinates']) != SELECTED_COORDINATES
                or record['hard_support_capacity'] != 24 or record['spec']['budget'] != 24):
            raise ValueError('Registered support identity or layout changed: ' + arm)
        if arm == 'semantic':
            if record['spec'].get('kind') != 'learned' or record['spec'].get('grouping') != 'semantic':
                raise ValueError('Wrong learned semantic support identity')
        elif (record['spec'].get('kind') != 'fixed'
              or record['spec'].get('control') != 'random_semantic'
              or record['spec'].get('seed') != int(arm.rsplit('s', 1)[1])):
            raise ValueError('Wrong pre-existing random support identity')
        result[arm] = dict(path=str(path), sha256=SUPPORT_SHA256[arm],
            selected_coordinates=list(SELECTED_COORDINATES), hard_support=mask.tolist(),
            hard_support_capacity=24, spec=record['spec'])
    return result


def source_bindings(bundle, archive_root=DEFAULT_ARCHIVE_ROOT):
    """Metadata/file-byte checks only; no checkpoint tensors or graph arrays."""
    arguments = bundle.config['arguments']
    if (arguments['hidden'], arguments['alpha_int'], arguments['target_class'],
            arguments['thrd'], bundle.config['score_dim']) != (80, 30, 1, .5, 32):
        raise ValueError('Original uniform architecture contract changed')
    names = {'uniform': 'generator', 'surrogate': 'surrogate', 'uniform_plan': 'uniform_plan',
             'uniform_training': 'uniform_training', 'uniform_report': 'uniform_report'}
    result = {key: _archived_reference(bundle, source, archive_root) for key, source in names.items()}
    if result['uniform']['sha256'] != UNIFORM_SHA256 or result['surrogate']['sha256'] != SURROGATE_SHA256:
        raise ValueError('Require the registered completed uniform seed21 start')
    result['supports'] = support_records(archive_root)
    return result


def _state(path, expected):
    if sha256(path) != expected:
        raise ValueError('Checkpoint hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor)
                   or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
                   for value in state.values())):
        raise ValueError('Require a finite tensor-only checkpoint')
    return state


def _freeze(module):
    module.zero_grad(set_to_none=True)
    return module.eval().requires_grad_(False)


def restore_uniform(bundle, device='cpu', archive_root=DEFAULT_ARCHIVE_ROOT):
    """Restore the actual unclipped uniform checkpoint with strict key matching."""
    source = source_bindings(bundle, archive_root)['uniform']
    state = _state(source['path'], source['sha256'])
    coordinates = torch.tensor(SELECTED_COORDINATES, dtype=torch.long)
    if not torch.equal(state.get('selected_coordinates', torch.empty(0)), coordinates):
        raise ValueError('Uniform checkpoint coordinate order changed')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        base_generator = GraphTrojanNet(torch.device('cpu'), 80, 30, layernum=2)
        base_shadow = GCN(784, 80, 2, dropout=0., device=torch.device('cpu'))
        for prefix, model in (('base_generator.', base_generator), ('base_shadow.', base_shadow)):
            weights = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
            model.load_state_dict(weights, strict=True)
            _freeze(model)
        role = PairRoleResidualGenerator(base_generator, base_shadow, coordinates,
            input_dim=784, hidden_dim=80, mode='untyped', threshold=.5)
        model = ContentAttentionGenerator(role, attention_mode='uniform', score_dim=32)
        model.load_state_dict(state, strict=True)
    return _freeze(model).to(device)


def restore_uniform_surrogate(bundle, device='cpu', archive_root=DEFAULT_ARCHIVE_ROOT):
    source = source_bindings(bundle, archive_root)['surrogate']
    state = _state(source['path'], source['sha256'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = GCN(784, 80, 2, dropout=0., device=torch.device('cpu'))
        model.load_state_dict(state, strict=True)
    return _freeze(model).to(device)


def context_from_edges(features, edge_index):
    """Construct a validation sidecar for uniform inference from an edge-only API.

    Connected components are validation groups, not reconstructed AV2 scene IDs.
    Uniform aggregation consumes neither role codes nor score permutations.
    This preserves every original edge/order and never creates graph links.
    """
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.shape[1] != 784 or not features.is_floating_point()):
        raise ValueError('Require floating [N,784] input features')
    count = len(features)
    if (not isinstance(edge_index, torch.Tensor) or edge_index.dtype != torch.long
            or edge_index.device != features.device or edge_index.ndim != 2 or edge_index.shape[0] != 2
            or bool(((edge_index < 0) | (edge_index >= count)).any())):
        raise ValueError('Invalid same-device original edge index')
    parents = list(range(count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for source, target in edge_index.detach().cpu().T.tolist():
        a, b = find(source), find(target)
        if a != b:
            parents[max(a, b)] = min(a, b)
    groups = [find(index) for index in range(count)]
    edge_count = edge_index.shape[1]
    return AttentionContext(edge_index=edge_index.detach().clone(),
        edge_type=torch.zeros(edge_count, dtype=torch.long, device=features.device),
        pair_scene=torch.tensor(groups, dtype=torch.long, device=features.device), num_pairs=count,
        score_permutation=torch.arange(edge_count, dtype=torch.long, device=features.device))


class FrozenSupportGenerator(nn.Module):
    """Tensor-only frozen wrapper with the historical apply_gates arithmetic.

    Unlike a trainable historical SupportMaskedGenerator, this adapter has no
    selector or pickle extra state. The fixed mask is supplied explicitly when
    restoring; strict loads cannot change that permission contract. Its
    output_mode is always identity, meaning no output-domain clipping.
    """

    output_mode = 'identity'
    input_dim = 784

    def __init__(self, anchor, mask):
        super().__init__()
        if (type(anchor) is not ContentAttentionGenerator or anchor.attention_mode != 'uniform'
                or anchor.residual.attention_mode != 'uniform' or anchor.input_dim != 784
                or tuple(anchor.selected_coordinates.tolist()) != SELECTED_COORDINATES):
            raise ValueError('Require the exact full uniform 784D/30-output architecture')
        support = _mask(mask).to(anchor.selected_coordinates.device)
        self.anchor = deepcopy(anchor)
        self.register_buffer('selected_coordinates', anchor.selected_coordinates.detach().clone())
        self.register_buffer('hard_support', support.detach().clone())
        self._frozen_mask = tuple(support.tolist())
        self.train(False)
        self.assert_contract()

    def train(self, mode=True):
        if type(mode) is not bool:
            raise ValueError('Training mode must be boolean')
        super().train(False)
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        return self

    def assert_contract(self):
        _mask(self.hard_support)
        if (tuple(self.hard_support.tolist()) != self._frozen_mask
                or tuple(self.selected_coordinates.tolist()) != SELECTED_COORDINATES
                or not torch.equal(self.selected_coordinates, self.anchor.selected_coordinates)
                or type(self.anchor) is not ContentAttentionGenerator
                or self.anchor.attention_mode != 'uniform' or self.anchor.residual.attention_mode != 'uniform'
                or any(child.training for child in self.modules())
                or any(p.requires_grad or p.grad is not None for p in self.parameters())):
            raise ValueError('Frozen uniform support contract changed')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError('Require strict copy-based frozen support restoration')
        for name in ('selected_coordinates', 'hard_support'):
            value = state_dict.get(name)
            if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), getattr(self, name).cpu()):
                raise ValueError('Restoration cannot change frozen support permissions')
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self.train(False)
        self.assert_contract()
        return result

    def _context(self, features, context):
        if isinstance(context, AttentionContext):
            context.validate()
            return context
        return context_from_edges(features, getattr(context, 'edge_index', None))

    def forward(self, features, context, targets):
        self.assert_contract()
        context = self._context(features, context)
        _check_features(features, context, self.input_dim)
        self.anchor._validate_targets(features, targets)
        if features.requires_grad:
            raise ValueError('Frozen-support inference requires unchanged non-grad observations')
        with torch.no_grad():
            generated = self.anchor(features, context, targets)
            clean = features[targets[:, None], self.selected_coordinates]
            return apply_gates(clean, generated, self.hard_support.to(features.dtype))

    def inject(self, features, context, targets, edge_weight=None, *, detach=False):
        if type(detach) is not bool:
            raise ValueError('detach must be bool')
        context = self._context(features, context)
        if edge_weight is None:
            edge_weight = features.new_ones(context.edge_index.shape[1])
        if (not isinstance(edge_weight, torch.Tensor) or edge_weight.dtype != features.dtype
                or edge_weight.device != features.device or edge_weight.shape != (context.edge_index.shape[1],)
                or not bool((edge_weight == 1).all())):
            raise ValueError('Require unchanged unit weights on the original edges')
        values = self(features, context, targets)
        changed = features.clone()
        changed[targets[:, None], self.selected_coordinates] = values.detach() if detach else values
        return InjectedFeatures(changed, context.edge_index.clone(), edge_weight.clone())


def frozen_support(anchor, mask):
    return FrozenSupportGenerator(anchor, mask)
