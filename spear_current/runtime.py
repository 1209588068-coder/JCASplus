"""Portable, hash-checked restoration of the retained SPEAR checkpoints.

Only the bundle and its explicit assets are read. No legacy experiment loader,
dataset access, training, or checkpoint initialization from a historical run
is needed: each final generator contains its own frozen G and shadow tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import GCN, GraphSage, GraphTrojanNet


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / 'artifacts/current/manifest.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Bundle:
    manifest_path: Path
    manifest: dict

    @property
    def config(self):
        return self.manifest['config']

    def asset_path(self, key):
        reference = self.manifest['assets'][key]
        relative = Path(reference['path'])
        path = (self.manifest_path.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.manifest_path.parent):
            raise ValueError('Bundle assets must be contained relative paths: ' + key)
        if not path.is_file() or sha256(path) != reference['sha256']:
            raise ValueError('Missing or changed bundle asset: ' + key)
        return path


def load_bundle(manifest=DEFAULT_MANIFEST, *, verify_assets=True):
    path = Path(DEFAULT_MANIFEST if manifest is None else manifest).resolve()
    value = json.loads(path.read_text())
    bundle = Bundle(path, value)
    config = bundle.config
    if set(config['roles']) != {'train', 'dev', 'audit'}:
        raise ValueError('Require the three frozen roles within original TRAIN')
    scenes, groups = set(), set()
    for role, expected in (('train', 512), ('dev', 128), ('audit', 256)):
        rows = config['roles'][role]
        if len(rows) != expected:
            raise ValueError('Frozen role population changed: ' + role)
        for row in rows:
            scene, group = row['scenario_id'], row['split_group_id']
            if (row.get('split') != 'train' or row.get('role') != role
                    or not scene or not group or scene in scenes or group in groups):
                raise ValueError('Invalid or overlapping original TRAIN role identity')
            scenes.add(scene)
            groups.add(group)
    arguments = config['arguments']
    if (arguments['hidden'], arguments['alpha_int'], arguments['target_class']) != (80, 30, 1):
        raise ValueError('Checkpoint architecture contract changed')
    coordinates = value.get('selected_coordinates30', value.get('selected_coordinates'))
    support = value.get('hard_support30', value.get('hard_support'))
    if (not isinstance(coordinates, list) or len(coordinates) != 30
            or len(set(coordinates)) != 30
            or any(type(i) is not int or not 0 <= i < 784 for i in coordinates)
            or not isinstance(support, list) or len(support) != 30
            or any(type(i) is not bool for i in support) or sum(support) != 24):
        raise ValueError('Require the frozen 30 coordinates and 24-coordinate support')
    if verify_assets:
        for key in value['assets']:
            bundle.asset_path(key)
    return bundle


def _asset_key(bundle, seed, asset_key, kind):
    if asset_key is not None:
        if seed is not None:
            raise ValueError('Choose a seed or an explicit asset key')
        return asset_key
    if seed is None:
        raise ValueError('A retained seed or explicit asset key is required')
    return bundle.manifest['seeds'][str(seed)][kind]


def _state(bundle, key):
    state = torch.load(bundle.asset_path(key), map_location='cpu', weights_only=True)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor)
                   or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
                   for value in state.values())):
        raise ValueError('Checkpoint must be a finite tensor state dictionary: ' + key)
    return state


def _freeze(model):
    model.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False)


def restore_generator(bundle, seed=None, *, asset_key=None):
    from .role_generator import PairRoleResidualGenerator
    from .attention_generator import ContentAttentionGenerator
    from .domain_generator import DomainConstrainedGenerator

    state = _state(bundle, _asset_key(bundle, seed, asset_key, 'generator'))
    coordinates = bundle.manifest.get('selected_coordinates30', bundle.manifest.get('selected_coordinates'))
    support = bundle.manifest.get('hard_support30', bundle.manifest.get('hard_support'))
    coordinates = torch.tensor(coordinates, dtype=torch.long)
    support = torch.tensor(support, dtype=torch.bool)
    if (not torch.equal(state['selected_coordinates'], coordinates)
            or not torch.equal(state['hard_support'], support)
            or state['output_mode_code'].shape != () or int(state['output_mode_code']) != 1):
        raise ValueError('Generator checkpoint changes coordinates, support, or projection')
    arguments = bundle.config['arguments']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        base_generator = GraphTrojanNet(torch.device('cpu'), 80, 30, layernum=2)
        base_shadow = GCN(784, 80, 2, dropout=0., device=torch.device('cpu'))
        for prefix, model in (('base_generator.', base_generator), ('base_shadow.', base_shadow)):
            weights = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
            model.load_state_dict(weights, strict=True)
            _freeze(model)
        role = PairRoleResidualGenerator(base_generator, base_shadow, coordinates,
            input_dim=784, hidden_dim=80, mode='untyped', threshold=arguments['thrd'])
        attention = ContentAttentionGenerator(role, attention_mode='uniform',
            score_dim=bundle.config['score_dim'])
        generator = DomainConstrainedGenerator(attention, support, output_mode='domain_projection')
        generator.load_state_dict(state, strict=True)
    _freeze(generator)
    generator.assert_contract()
    return generator


def new_victim(bundle, seed=None):
    """Construct the original GraphSAGE, retaining its initialization RNG order.

    With a seed, set NumPy/Torch RNG exactly as the historical fresh-victim
    adapter. Without a seed, consume the caller's current Torch RNG directly.
    No training or parameter freezing occurs here.
    """
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a nonnegative Python integer')
        np.random.seed(seed)
        torch.manual_seed(seed)
    args = bundle.config['arguments']
    return GraphSage(784, args['hidden'], 2, dropout=args['dropout'],
        lr=args['train_lr'], weight_decay=args['weight_decay'], device=torch.device('cpu'))


def restore_victim(bundle, seed=None, *, asset_key=None):
    state = _state(bundle, _asset_key(bundle, seed, asset_key, 'victim'))
    with torch.random.fork_rng(devices=[]):
        model = new_victim(bundle)
        model.load_state_dict(state, strict=True)
    return _freeze(model)


def restore_clean_reference(bundle):
    return restore_victim(bundle, asset_key='clean_reference')


def restore_proxy(bundle, index=1):
    if type(index) is not int or index not in (1, 2):
        raise ValueError('Clean proxy index must be 1 or 2')
    return restore_victim(bundle, asset_key='clean_proxy' + str(index))


def restore_surrogate(bundle, seed=None, *, asset_key=None):
    state = _state(bundle, _asset_key(bundle, seed, asset_key, 'surrogate'))
    with torch.random.fork_rng(devices=[]):
        model = GCN(784, 80, 2, dropout=0., device=torch.device('cpu'))
        model.load_state_dict(state, strict=True)
    return _freeze(model)
