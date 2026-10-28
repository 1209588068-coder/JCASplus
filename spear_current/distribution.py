# Source: experiments/spear_distribution_training_v1/distribution.py; mechanically migrated; algorithms and state keys unchanged.
"""Original distribution-matching regularizer for the local AV2 pilot.

The adversarial distribution objective is inspired by DPGBA; no upstream code
is copied. This module has no data loading, graph mutation, or training runner.
The discriminator observes only the fixed 24 selected values and six original
observed conditions. Its distribution loss is not a physical-consistency claim.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .support_schema import (
    FEATURE_DIM, LEGAL_COORDINATES, coordinate_metadata,
)


ARMS = ('continuation', 'unconditional', 'conditional')
CONDITION_COORDINATES = (149, 199, 507, 557, 718, 721)
DISCRIMINATOR_SEED = 20261621
DISTRIBUTION_WEIGHT = 0.1
DISCRIMINATOR_LR = 0.0002
DISCRIMINATOR_WEIGHT_DECAY = 0.0005
INPUT_SCALE = 10.0


def _hash_tensor(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return True


def _hash_state(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(_hash_tensor(state[name]).encode())
    return digest.hexdigest()


class DistributionRegularizer:
    """Own an isolated discriminator bound to one clean outer population.

    D is frozen/eval except inside ``discriminator_step`` and the no-update
    smoke. ``generator_loss`` returns an already weighted scalar. Its graph
    retains input gradients, while D parameters never acquire G-step gradients.
    The caller owns the generator optimizer and the TRAIN-only data ledger.
    ``provenance`` is an audit record, never evidence of authorization or split.
    """

    def __init__(self, arm, original_features, target_indices, selected_coordinates,
                 *, lr=DISCRIMINATOR_LR, weight_decay=DISCRIMINATOR_WEIGHT_DECAY,
                 weight=DISTRIBUTION_WEIGHT, seed=DISCRIMINATOR_SEED, provenance=None):
        if arm not in ARMS:
            raise ValueError('Unknown distribution arm')
        if (isinstance(seed, bool) or not isinstance(seed, Integral) or seed != DISCRIMINATOR_SEED
                or lr != DISCRIMINATOR_LR or weight_decay != DISCRIMINATOR_WEIGHT_DECAY
                or weight != DISTRIBUTION_WEIGHT):
            raise ValueError('Distribution seed, learning rate, decay and weight are fixed')
        self._validate_features(original_features)
        if original_features.requires_grad:
            raise ValueError('Original clean features must not require gradients')
        coordinates = self._parse_coordinates(selected_coordinates)
        self._validate_targets(target_indices, original_features)
        self._arm = arm
        self._shape = tuple(original_features.shape)
        self._device, self._dtype = original_features.device, original_features.dtype
        self._coordinates_tuple = coordinates
        self._targets = target_indices.detach().clone()
        self._coordinates = torch.tensor(coordinates, dtype=torch.long, device=self._device)
        self._conditions = torch.tensor(CONDITION_COORDINATES, dtype=torch.long, device=self._device)
        self._real = original_features[self._targets[:, None], self._coordinates].detach().clone()
        self._clean_conditions = original_features[self._targets[:, None], self._conditions].detach().clone()
        if bool((self._real.abs() > INPUT_SCALE).any()) or bool((self._clean_conditions.abs() > INPUT_SCALE).any()):
            raise ValueError('Real reference and observed conditions must lie in [-10,10]')
        try:
            self._provenance = json.loads(json.dumps(provenance or {}, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError('Provenance must be finite JSON metadata') from exc
        if not isinstance(self._provenance, dict):
            raise ValueError('Provenance must be a metadata dictionary')
        self._config = dict(arm=arm, selected_coordinates=list(coordinates),
                           condition_coordinates=list(CONDITION_COORDINATES), input_scale=INPUT_SCALE,
                           seed=int(seed), lr=float(lr), weight_decay=float(weight_decay), weight=float(weight),
                           architecture=[30, 64, 64, 1], negative_slope=0.2,
                           condition_policy='original_clean' if arm == 'conditional' else 'zero',
                           real_reference='same_original_clean_outer_rows')
        self._config_json = json.dumps(self._config, sort_keys=True)
        self._reference = dict(target_indices_sha256=_hash_tensor(self._targets),
                              clean_selected_sha256=_hash_tensor(self._real),
                              clean_conditions_sha256=_hash_tensor(self._clean_conditions),
                              target_count=len(self._targets), original_shape=list(self._shape),
                              original_dtype=str(self._dtype))
        self._reference_tensors = {name: (value, value._version, _hash_tensor(value))
                                   for name, value in self._bound_tensors().items()}
        # Build on CPU with an isolated CPU generator. No CUDA RNG is touched;
        # .to(device) copies initialized weights without drawing random numbers.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.discriminator = nn.Sequential(
                nn.Linear(30, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1),
            ).to(device=self._device, dtype=self._dtype)
        self._owned_discriminator = self.discriminator
        self._parameter_ids = tuple(id(p) for p in self.discriminator.parameters())
        self._state_shapes = {name: value.shape for name, value in self.discriminator.state_dict().items()}
        self._initial_discriminator_sha256 = self.discriminator_hash()
        self.optimizer = (None if arm == 'continuation' else
                          torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=weight_decay))
        self._owned_optimizer = self.optimizer
        self._steps = 0
        self._generator_calls = 0
        self.last_generator_raw_loss = 0.0
        self.last_generator_weighted_loss = 0.0
        self._freeze()
        self.assert_contract()

    @property
    def arm(self):
        return self._arm

    @property
    def mode(self):
        return self._arm

    @property
    def steps(self):
        return self._steps

    @property
    def d_steps(self):
        return self._steps

    @property
    def generator_calls(self):
        return self._generator_calls

    @staticmethod
    def _parse_coordinates(value):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.long or value.ndim != 1:
                raise ValueError('Selected coordinates must be a long vector')
            value = value.tolist()
        try:
            result = tuple(value)
        except TypeError as exc:
            raise ValueError('Require 24 fixed selected coordinates') from exc
        if (len(result) != 24 or len(set(result)) != 24
                or any(isinstance(x, bool) or not isinstance(x, Integral) or x not in LEGAL_COORDINATES for x in result)
                or set(result).intersection(CONDITION_COORDINATES)):
            raise ValueError('Require 24 unique legal selected coordinates disjoint from conditions')
        return tuple(int(x) for x in result)

    @staticmethod
    def _validate_features(features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != FEATURE_DIM or features.shape[0] < 1
                or features.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Require finite float32/float64 original or injected [N,784] features')

    @staticmethod
    def _validate_targets(targets, features):
        if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.long or targets.ndim != 1
                or targets.device != features.device or not len(targets)
                or len(torch.unique(targets)) != len(targets)
                or int(targets.min()) < 0 or int(targets.max()) >= features.shape[0]):
            raise ValueError('Require nonempty unique in-range same-device long targets')

    def _bound_tensors(self):
        return dict(targets=self._targets, coordinates=self._coordinates, conditions=self._conditions,
                    real=self._real, clean_conditions=self._clean_conditions)

    def _freeze(self):
        self.discriminator.eval().requires_grad_(False)
        for parameter in self.discriminator.parameters():
            parameter.grad = None

    def assert_contract(self):
        if (self._arm != self._config['arm'] or self.discriminator is not self._owned_discriminator
                or json.dumps(self._config, sort_keys=True) != self._config_json):
            raise ValueError('Distribution arm or discriminator ownership changed')
        layers = tuple(self.discriminator.children())
        if (len(layers) != 5 or tuple(type(m) for m in layers) !=
                (nn.Linear, nn.LeakyReLU, nn.Linear, nn.LeakyReLU, nn.Linear)
                or any(layers[i].negative_slope != 0.2 or layers[i].inplace for i in (1, 3))):
            raise ValueError('Fixed discriminator architecture changed')
        if tuple(id(p) for p in self.discriminator.parameters()) != self._parameter_ids:
            raise ValueError('Discriminator parameter ownership changed')
        if self.optimizer is not self._owned_optimizer:
            raise ValueError('Discriminator optimizer ownership changed')
        for name, value in self._bound_tensors().items():
            original, version, digest = self._reference_tensors[name]
            if value is not original or value._version != version or _hash_tensor(value) != digest:
                raise ValueError('Frozen distribution binding changed: ' + name)
        if (any(module.training for module in self.discriminator.modules())
                or any(p.requires_grad or p.grad is not None for p in self.discriminator.parameters())):
            raise ValueError('Discriminator must be frozen, eval and gradient-free outside its update')
        for name, value in self.discriminator.state_dict().items():
            if (value.shape != self._state_shapes.get(name) or value.device != self._device
                    or value.dtype != self._dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Discriminator state must retain finite device/dtype contract')
        if self.optimizer is not None:
            if len(self.optimizer.param_groups) != 1:
                raise ValueError('Discriminator optimizer parameter groups changed')
            group = self.optimizer.param_groups[0]
            if (tuple(id(p) for p in group['params']) != self._parameter_ids
                    or group['lr'] != DISCRIMINATOR_LR or group['weight_decay'] != DISCRIMINATOR_WEIGHT_DECAY
                    or not _finite_tree(self.optimizer.state_dict())):
                raise ValueError('Discriminator optimizer contract changed or became nonfinite')
        if (type(self._steps) is not int or type(self._generator_calls) is not int
                or min(self._steps, self._generator_calls) < 0
                or (self.arm == 'continuation' and self._steps != 0)):
            raise ValueError('Invalid distribution update counters')

    def validate_binding(self, features, target_indices, selected_coordinates):
        self.assert_contract()
        self._validate_features(features)
        self._validate_targets(target_indices, features)
        if (features.requires_grad or tuple(features.shape) != self._shape
                or features.device != self._device or features.dtype != self._dtype
                or self._parse_coordinates(selected_coordinates) != self._coordinates_tuple
                or not torch.equal(target_indices, self._targets)
                or not torch.equal(features[self._targets[:, None], self._coordinates], self._real)
                or not torch.equal(features[self._targets[:, None], self._conditions], self._clean_conditions)):
            raise ValueError('Distribution regularizer is bound to different clean rows, support or conditions')

    def discriminator_inputs(self, injected_features):
        """Return owned real/fake [M,30]; only fake24 keeps its autograd path."""
        self.assert_contract()
        self._validate_features(injected_features)
        if (tuple(injected_features.shape) != self._shape or injected_features.device != self._device
                or injected_features.dtype != self._dtype):
            raise ValueError('Injected feature shape/device/dtype differs from the binding')
        observed = injected_features[self._targets[:, None], self._conditions]
        if not torch.equal(observed.detach(), self._clean_conditions):
            raise ValueError('Frozen observed conditions were changed by feature injection')
        fake = injected_features[self._targets[:, None], self._coordinates]
        if bool((fake.detach().abs() > INPUT_SCALE).any()):
            raise ValueError('Injected selected values exceed the fixed domain projection')
        condition = (self._clean_conditions if self.arm == 'conditional'
                     else torch.zeros_like(self._clean_conditions))
        return (torch.cat((self._real, condition), dim=1) / INPUT_SCALE,
                torch.cat((fake, condition), dim=1) / INPUT_SCALE)

    @contextmanager
    def _discriminator_update(self):
        self.assert_contract()
        self.discriminator.requires_grad_(True)
        try:
            yield
        finally:
            self._freeze()

    def _d_loss(self, real_input, fake_input):
        real_logits = self.discriminator(real_input.detach())
        fake_logits = self.discriminator(fake_input.detach())
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        loss = (real_loss + fake_loss) * 0.5
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite discriminator loss')
        return loss, real_loss, fake_loss

    def discriminator_step(self, injected_features):
        real_input, fake_input = self.discriminator_inputs(injected_features)
        if self.arm == 'continuation':
            return dict(d_loss=0.0, d_real_loss=0.0, d_fake_loss=0.0, d_steps=0)
        with self._discriminator_update():
            self.optimizer.zero_grad(set_to_none=True)
            loss, real_loss, fake_loss = self._d_loss(real_input, fake_input)
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.discriminator.parameters()):
                raise ValueError('Missing or nonfinite discriminator gradient')
            self.optimizer.step()
        self._steps += 1
        self.assert_contract()
        return dict(d_loss=float(loss.detach()), d_real_loss=float(real_loss.detach()),
                    d_fake_loss=float(fake_loss.detach()), d_steps=self.steps)

    def _g_loss(self, fake_input):
        if self.arm == 'continuation':
            return fake_input.sum() * 0.0
        logits = self.discriminator(fake_input)
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite distribution generator loss')
        return loss

    def generator_loss(self, injected_features):
        _, fake_input = self.discriminator_inputs(injected_features)
        raw = self._g_loss(fake_input)
        weighted = raw * DISTRIBUTION_WEIGHT
        self._generator_calls += 1
        self.last_generator_raw_loss = float(raw.detach())
        self.last_generator_weighted_loss = float(weighted.detach())
        self.assert_contract()
        return weighted

    def no_update_smoke(self, injected_features):
        """Check D/input gradients without backward(), optimizer updates or counters."""
        real_input, fake_input = self.discriminator_inputs(injected_features)
        if not injected_features.requires_grad:
            raise ValueError('Smoke requires an injected tensor retaining the generator graph')
        if self.arm == 'continuation':
            return dict(d_loss=0.0, d_real_loss=0.0, d_fake_loss=0.0, d_gradient_parameters=0,
                        generator_raw_loss=0.0, generator_weighted_loss=0.0,
                        generator_input_gradient_l1=0.0)
        with self._discriminator_update():
            loss, real_loss, fake_loss = self._d_loss(real_input, fake_input)
            gradients = torch.autograd.grad(loss, tuple(self.discriminator.parameters()))
            if any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError('Nonfinite discriminator smoke gradient')
            count = sum(bool((g != 0).any()) for g in gradients)
        raw = self._g_loss(fake_input)
        weighted = DISTRIBUTION_WEIGHT * raw
        gradient, = torch.autograd.grad(weighted, injected_features, retain_graph=True)
        magnitude = float(gradient.detach().abs().sum())
        if not bool(torch.isfinite(gradient).all()) or not count or magnitude <= 0:
            raise ValueError('Distribution smoke requires finite nonzero D and injected-input gradients')
        self.assert_contract()
        return dict(d_loss=float(loss.detach()), d_real_loss=float(real_loss.detach()),
                    d_fake_loss=float(fake_loss.detach()), d_gradient_parameters=count,
                    generator_raw_loss=float(raw.detach()), generator_weighted_loss=float(weighted.detach()),
                    generator_input_gradient_l1=magnitude)

    loss_smoke = no_update_smoke

    def discriminator_hash(self):
        return _hash_state(self.discriminator.state_dict())

    def snapshot(self):
        """Return independent torch-saveable state and explicit reference provenance."""
        self.assert_contract()
        return dict(version=1, config=deepcopy(self._config), reference=deepcopy(self._reference),
                    provenance=deepcopy(self._provenance),
                    condition_provenance=dict(source='original_clean_observed_standardized_features',
                        scale=INPUT_SCALE, coordinates=list(CONDITION_COORDINATES),
                        fields=[coordinate_metadata(c) for c in CONDITION_COORDINATES],
                        contains_labels=False, contains_future=False, contains_ids=False),
                    real_reference_provenance=dict(source='same_original_clean_outer_rows',
                        selected_coordinates=list(self._coordinates_tuple), **deepcopy(self._reference)),
                    initial_discriminator_sha256=self._initial_discriminator_sha256,
                    discriminator_sha256=self.discriminator_hash(),
                    discriminator=deepcopy(self.discriminator.state_dict()),
                    optimizer=None if self.optimizer is None else deepcopy(self.optimizer.state_dict()),
                    steps=self.steps, generator_calls=self.generator_calls,
                    last_generator_raw_loss=self.last_generator_raw_loss,
                    last_generator_weighted_loss=self.last_generator_weighted_loss)

    def load_snapshot(self, snapshot):
        """Restore only a snapshot of exactly the same arm and clean binding."""
        self.assert_contract()
        current = self.snapshot()
        if not isinstance(snapshot, dict) or set(snapshot) != set(current) or not _finite_tree(snapshot):
            raise ValueError('Malformed/nonfinite distribution snapshot')
        fixed = ('version', 'config', 'reference', 'provenance', 'condition_provenance',
                 'real_reference_provenance', 'initial_discriminator_sha256')
        if any(snapshot[k] != current[k] for k in fixed):
            raise ValueError('Snapshot changes the distribution arm, hyperparameters or clean provenance')
        state = snapshot['discriminator']
        if not isinstance(state, dict) or set(state) != set(current['discriminator']):
            raise ValueError('Snapshot discriminator tensor membership changed')
        for key, value in state.items():
            old = current['discriminator'][key]
            if not isinstance(value, torch.Tensor) or value.shape != old.shape or value.dtype != old.dtype:
                raise ValueError('Malformed discriminator snapshot tensor: ' + key)
        if _hash_state(state) != snapshot['discriminator_sha256']:
            raise ValueError('Discriminator snapshot content hash mismatch')
        if (type(snapshot['steps']) is not int or type(snapshot['generator_calls']) is not int
                or min(snapshot['steps'], snapshot['generator_calls']) < 0
                or (self.arm == 'continuation' and (snapshot['steps'] != 0 or snapshot['optimizer'] is not None))):
            raise ValueError('Malformed distribution snapshot counters')
        # Validate on owned temporary copies before touching live state.
        trial_d = deepcopy(self.discriminator)
        trial_d.load_state_dict(state, strict=True)
        if self.optimizer is not None:
            if not isinstance(snapshot['optimizer'], dict):
                raise ValueError('Active-arm snapshot lacks discriminator optimizer state')
            trial_optimizer = torch.optim.Adam(trial_d.parameters(), lr=DISCRIMINATOR_LR,
                                               weight_decay=DISCRIMINATOR_WEIGHT_DECAY)
            try:
                trial_optimizer.load_state_dict(deepcopy(snapshot['optimizer']))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError('Malformed discriminator optimizer snapshot') from exc
            group = trial_optimizer.param_groups[0]
            if len(trial_optimizer.param_groups) != 1 or group['lr'] != DISCRIMINATOR_LR or group['weight_decay'] != DISCRIMINATOR_WEIGHT_DECAY:
                raise ValueError('Snapshot optimizer hyperparameters changed')
            for parameter, optim_state in trial_optimizer.state.items():
                if optim_state and (set(optim_state) != {'step', 'exp_avg', 'exp_avg_sq'}
                        or optim_state['step'].numel() != 1
                        or any(optim_state[k].shape != parameter.shape for k in ('exp_avg', 'exp_avg_sq'))):
                    raise ValueError('Malformed discriminator optimizer tensor state')
        self.discriminator.load_state_dict(state, strict=True)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(deepcopy(snapshot['optimizer']))
        self._steps, self._generator_calls = snapshot['steps'], snapshot['generator_calls']
        self.last_generator_raw_loss = float(snapshot['last_generator_raw_loss'])
        self.last_generator_weighted_loss = float(snapshot['last_generator_weighted_loss'])
        self._freeze()
        self.assert_contract()
