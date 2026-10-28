# Source: experiments/spear_semantic_support_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""Global sparse supports on the completed, FROZEN uniform SPEAR generator.

Hard-concrete equations follow Louizos et al., ICLR 2018, arXiv:1712.01312
(https://github.com/AMLab-Amsterdam/L0_regularization). This is a local tensor
implementation, not vendored code or a claim of a new sparsity estimator.
The additions here are source-defined feature groups, coordinate-weighted cost,
and explicit exact-cost hardening. Stochastic training is NOT strict-budget
training; hardened binary support is a separate, irreversible deployment step.

Only support parameters may learn in this prototype. The inherited generator,
including its nonzero uniform residual, is frozen. No data I/O, fitting loop,
optimizer, evaluation, or graph changes live here. This is feature-space
selection, not physical trajectory generation or cross-pair consistency repair.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random

import torch
from torch import nn

from .attention_generator import ContentAttentionGenerator
from .role_generator import (
    InjectedFeatures, _check_features,
)
from .support_schema import LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups


def _budget(value, maximum=len(LEGAL_COORDINATES)):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('Coordinate budget must be an integer between zero and legal capacity')


def _slot_mask(groups, chosen):
    columns = {column for i in chosen for column in groups[i]}
    return torch.tensor([column in columns for column in SELECTED_COORDINATES], dtype=torch.bool)


def exact_cost_support(groups, scores, budget):
    """Maximum sum(size * score) at EXACT coordinate cost; stable group-index ties.

    Indivisible groups may make a budget infeasible; never silently underfill.
    ``scores`` are explicit selection scores, not observed ASR measurements.
    """
    _budget(budget)
    if (not isinstance(groups, tuple) or not groups
            or any(not isinstance(g, tuple) or not g for g in groups)):
        raise ValueError('Require a nonempty tuple partition of legal coordinates')
    flattened = [c for group in groups for c in group]
    if (any(type(c) is not int for c in flattened)
            or len(flattened) != len(LEGAL_COORDINATES)
            or set(flattened) != set(LEGAL_COORDINATES)):
        raise ValueError('Partition must contain each legal coordinate exactly once')
    values = torch.as_tensor(scores).detach().cpu().to(torch.float64)
    if values.shape != (len(groups),) or not bool(torch.isfinite(values).all()):
        raise ValueError('Require one finite score per group')
    best = {0: (0.0, ())}
    for index, (group, score) in enumerate(zip(groups, values.tolist())):
        updated = dict(best)
        for cost, (total, chosen) in best.items():
            new_cost = cost + len(group)
            if new_cost > budget:
                continue
            candidate = (total + len(group) * score, chosen + (index,))
            incumbent = updated.get(new_cost)
            if (incumbent is None or candidate[0] > incumbent[0]
                    or (candidate[0] == incumbent[0] and candidate[1] < incumbent[1])):
                updated[new_cost] = candidate
        best = updated
    if budget not in best:
        raise ValueError('Exact coordinate budget is infeasible for these indivisible groups')
    return _slot_mask(groups, best[budget][1])


def fixed_support(kind, budget, *, seed=20260621):
    """Legal ranking, uniform coordinate random, or uniform feasible-group random.

    Random controls use a private RNG, never change global Python/Torch RNG,
    and are not selected by their evaluation results. Group random samples
    uniformly over feasible subsets, not over group permutations or scores.
    """
    _budget(budget)
    if type(seed) is not int:
        raise ValueError('Random-control seed must be an integer')
    if kind == 'ranking':
        columns = set(LEGAL_COORDINATES[:budget])
        return torch.tensor([c in columns for c in SELECTED_COORDINATES], dtype=torch.bool)
    if kind not in ('random_coordinate', 'random_semantic'):
        raise ValueError('Unknown fixed support control')
    groups = support_groups('semantic' if kind == 'random_semantic' else 'coordinate')
    ways = [[0] * (budget + 1) for _ in range(len(groups) + 1)]
    ways[-1][0] = 1
    for i in range(len(groups) - 1, -1, -1):
        for cost in range(budget + 1):
            ways[i][cost] = ways[i + 1][cost]
            if cost >= len(groups[i]):
                ways[i][cost] += ways[i + 1][cost - len(groups[i])]
    if not ways[0][budget]:
        raise ValueError('Exact coordinate budget is infeasible for random groups')
    rng, remaining, chosen = random.Random(seed), budget, []
    for i, group in enumerate(groups):
        included = ways[i + 1][remaining - len(group)] if remaining >= len(group) else 0
        if rng.randrange(ways[i][remaining]) < included:
            chosen.append(i)
            remaining -= len(group)
    return _slot_mask(groups, chosen)


class HardConcreteSupport(nn.Module):
    """One global gate per semantic group or coordinate, shared across targets.

    The private CPU RNG is checkpointed. A forward call samples one vector,
    not a different support per target. Deterministic relaxation is provided
    for diagnostics only; evaluation of the wrapper requires a hardened mask.
    """

    def __init__(self, grouping='semantic', *, seed=20260621, group_seed=20260621,
                 initial_open_probability=.8, temperature=2 / 3, lower=-.1, upper=1.1):
        super().__init__()
        self.groups = support_groups(grouping, seed=group_seed)
        if (type(seed) is not int or not all(math.isfinite(v) for v in
                (initial_open_probability, temperature, lower, upper))
                or not 0 < initial_open_probability < 1 or temperature <= 0
                or lower >= 0 or upper <= 1):
            raise ValueError('Invalid hard-concrete parameters')
        self.grouping = grouping
        self.group_seed = group_seed
        self.temperature, self.lower, self.upper = float(temperature), float(lower), float(upper)
        self.initial_open_probability, self.seed = float(initial_open_probability), seed
        initial = math.log(initial_open_probability / (1 - initial_open_probability))
        initial += self.temperature * math.log(-self.lower / self.upper)
        self.log_alpha = nn.Parameter(torch.full((len(self.groups),), initial))
        rng = torch.Generator(device='cpu').manual_seed(seed)
        self.register_buffer('rng_state', rng.get_state())

    def get_extra_state(self):
        return dict(grouping=self.grouping, groups=self.groups, seed=self.seed, group_seed=self.group_seed,
                    initial_open_probability=self.initial_open_probability,
                    temperature=self.temperature, lower=self.lower, upper=self.upper)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Selector checkpoint configuration/grouping differs from constructor')

    def _validate(self):
        if (self.groups != support_groups(self.grouping, seed=self.group_seed)
                or self.log_alpha.shape != (len(self.groups),)
                or not bool(torch.isfinite(self.log_alpha).all())):
            raise ValueError('Nonfinite or changed selector contract')

    def activation_probability(self):
        self._validate()
        return torch.sigmoid(self.log_alpha - self.temperature * math.log(-self.lower / self.upper))

    def expected_coordinate_count(self):
        """Expected NONZERO support capacity, not actual changed-coordinate count."""
        costs = self.log_alpha.new_tensor([len(group) for group in self.groups])
        return (costs * self.activation_probability()).sum()

    def forward(self, *, sample=True):
        self._validate()
        if type(sample) is not bool:
            raise ValueError('sample must be bool')
        if sample:
            rng = torch.Generator(device='cpu')
            rng.set_state(self.rng_state.detach().cpu())
            u = torch.rand(len(self.groups), generator=rng, dtype=torch.float64)
            self.rng_state.copy_(rng.get_state().to(self.rng_state.device))
            u = u.to(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            soft = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.temperature)
        else:
            soft = torch.sigmoid(self.log_alpha)
        gates = (soft * (self.upper - self.lower) + self.lower).clamp(0, 1)
        group_for = {coordinate: i for i, group in enumerate(self.groups) for coordinate in group}
        zero = gates.new_zeros(())
        return torch.stack([gates[group_for[c]] if c in group_for else zero
                            for c in SELECTED_COORDINATES])

    def project(self, budget):
        """Separate deterministic exact-K projection; no state/weights/RNG update."""
        return exact_cost_support(self.groups, self.activation_probability(), budget).to(self.log_alpha.device)


def apply_gates(clean_values, generated_values, gates):
    """Closed gates restore ORIGINAL values; active binary gates preserve bits."""
    if (clean_values.shape != generated_values.shape or clean_values.ndim != 2
            or gates.shape != (clean_values.shape[1],)
            or clean_values.device != generated_values.device or gates.device != clean_values.device
            or clean_values.dtype != generated_values.dtype or not clean_values.is_floating_point()
            or gates.dtype != clean_values.dtype):
        raise ValueError('Gate/value shapes, dtypes or devices disagree')
    if (not all(bool(torch.isfinite(v).all()) for v in (clean_values, generated_values, gates))
            or bool(((gates < 0) | (gates > 1)).any())):
        raise ValueError('Require finite values and gates in [0,1]')
    # torch.where masks values, not intermediate derivatives: computing an
    # unused Inf delta would still give 0*Inf=NaN in gate backward. Remove
    # endpoint operands BEFORE subtraction; binary masks remain bitwise exact.
    interior = (gates > 0) & (gates < 1)
    left = torch.where(interior, clean_values, torch.zeros_like(clean_values))
    right = torch.where(interior, generated_values, torch.zeros_like(generated_values))
    delta = right - left
    if not bool(torch.isfinite(delta).all()):
        raise ValueError('Nonfinite interpolation delta')
    interpolated = left + gates * delta
    result = torch.where(gates == 0, clean_values,
                         torch.where(gates == 1, generated_values, interpolated))
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite gated feature values')
    return result


class SupportMaskedGenerator(nn.Module):
    """Own a frozen uniform anchor; only the explicit selector can receive grad.

    ``freeze_support`` is one-way. Never change support between poisoned-data
    construction and inference. No compatibility with the old residual trainer
    is claimed: that trainer intentionally optimizes a different parameter set.
    """

    def __init__(self, anchor, selector=None, *, hard_support=None):
        super().__init__()
        if (type(anchor) is not ContentAttentionGenerator
                or anchor.attention_mode != 'uniform' or anchor.residual.attention_mode != 'uniform'
                or anchor.input_dim != 784
                or tuple(anchor.selected_coordinates.tolist()) != SELECTED_COORDINATES):
            raise ValueError('Require the full original uniform 784-D/30-output anchor')
        if (selector is None) == (hard_support is None):
            raise ValueError('Provide exactly one selector or fixed hard support')
        if selector is not None and not isinstance(selector, HardConcreteSupport):
            raise ValueError('Expected a HardConcreteSupport')
        self.anchor = deepcopy(anchor)
        self.selector = deepcopy(selector) if selector is not None else None
        if self.selector is not None:
            parameter = next(self.anchor.parameters())
            self.selector.to(device=parameter.device, dtype=parameter.dtype)
        self.input_dim = 784
        self.register_buffer('selected_coordinates', anchor.selected_coordinates.detach().clone())
        self.register_buffer('hard_support', torch.zeros_like(self.selected_coordinates, dtype=torch.bool))
        self.register_buffer('support_frozen', torch.tensor(False, device=self.selected_coordinates.device))
        self._frozen_mask = None
        self._freeze_anchor()
        if hard_support is not None:
            self.freeze_support(hard_support)

    def _freeze_anchor(self):
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.grad = None

    def train(self, mode=True):
        super().train(mode)
        self._freeze_anchor()
        return self

    def _validate_support(self, mask):
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != (len(SELECTED_COORDINATES),)):
            raise ValueError('Hard support must be a bool vector in original 30-output order')
        if any(bool(mask[i]) for i, c in enumerate(SELECTED_COORDINATES) if c not in LEGAL_COORDINATES):
            raise ValueError('Metadata coordinates are immutable')

    def freeze_support(self, mask):
        if bool(self.support_frozen) or self._frozen_mask is not None:
            raise ValueError('Support already frozen; cannot select a new evaluation mask')
        self._validate_support(mask)
        self.hard_support.copy_(mask.to(self.hard_support.device))
        self.support_frozen.fill_(True)
        self._frozen_mask = tuple(self.hard_support.tolist())
        if self.selector is not None:
            self.selector.requires_grad_(False)
            for parameter in self.selector.parameters():
                parameter.grad = None

    def harden(self, budget):
        if self.selector is None:
            raise ValueError('Fixed controls have no learned selector to project')
        self.freeze_support(self.selector.project(budget))

    def get_extra_state(self):
        return dict(anchor_mode='uniform', selected_coordinates=SELECTED_COORDINATES,
                    frozen_mask=self._frozen_mask)

    def set_extra_state(self, state):
        if (state.get('anchor_mode') != 'uniform'
                or tuple(state.get('selected_coordinates', ())) != SELECTED_COORDINATES):
            raise ValueError('Wrong uniform/support checkpoint contract')
        self._frozen_mask = state['frozen_mask']

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError('Require strict copy-based support checkpoint restoration')
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self._check()
        if bool(self.support_frozen) and self.selector is not None:
            self.selector.requires_grad_(False)
            for parameter in self.selector.parameters():
                parameter.grad = None
        return result

    def _check(self):
        if (self.anchor.attention_mode != 'uniform'
                or self.anchor.residual.attention_mode != 'uniform'
                or tuple(self.selected_coordinates.tolist()) != SELECTED_COORDINATES
                or not torch.equal(self.anchor.selected_coordinates, self.selected_coordinates)):
            raise ValueError('Anchor mode or output-coordinate mapping changed')
        self._validate_support(self.hard_support)
        if bool(self.support_frozen):
            if tuple(self.hard_support.tolist()) != self._frozen_mask:
                raise ValueError('Frozen support mutated')
            if self.selector is not None:
                self.selector.requires_grad_(False)
        elif self._frozen_mask is not None:
            raise ValueError('Frozen support state was reverted')
        if any(p.requires_grad or p.grad is not None for p in self.anchor.parameters()):
            raise ValueError('Anchor must stay frozen and gradient-free')
        if any(m.training for m in self.anchor.modules()):
            raise ValueError('Anchor must stay in eval mode')

    def forward(self, features, context, targets):
        self._check()
        _check_features(features, context, self.input_dim)
        self.anchor._validate_targets(features, targets)
        if features.requires_grad:
            raise ValueError('Original features must not require gradients')
        with torch.no_grad():
            generated = self.anchor(features, context, targets)
        clean = features[targets[:, None], self.selected_coordinates]
        if bool(self.support_frozen):
            gates = self.hard_support.to(features.dtype)
        else:
            if not self.training or self.selector is None:
                raise ValueError('Evaluation requires an explicitly frozen binary support')
            gates = self.selector(sample=True)
        return apply_gates(clean, generated, gates)

    def inject(self, features, context, targets, edge_weight=None, *, detach=False):
        if type(detach) is not bool:
            raise ValueError('detach must be bool')
        _check_features(features, context, self.input_dim)
        if edge_weight is None:
            edge_weight = features.new_ones(context.edge_index.shape[1])
        if (not isinstance(edge_weight, torch.Tensor) or edge_weight.dtype != features.dtype
                or edge_weight.shape != (context.edge_index.shape[1],)
                or edge_weight.device != features.device or not bool((edge_weight == 1).all())):
            raise ValueError('Require one unit weight per original message edge')
        values = self(features, context, targets)
        output = features.clone()
        output[targets[:, None], self.selected_coordinates] = values.detach() if detach else values
        return InjectedFeatures(output, context.edge_index.clone(), edge_weight.clone())


def modification_statistics(clean, modified, targets, hard_support):
    """Capacity and actual changes are distinct; magnitudes are NOT SI units."""
    if (clean.shape != modified.shape or clean.ndim != 2 or clean.shape[1] != 784
            or clean.device != modified.device or clean.dtype != modified.dtype
            or not clean.is_floating_point()
            or not all(bool(torch.isfinite(x).all()) for x in (clean, modified))):
        raise ValueError('Require aligned finite 784-D feature tensors')
    if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.long or targets.ndim != 1
            or targets.device != clean.device or targets.unique().numel() != targets.numel()
            or bool(((targets < 0) | (targets >= len(clean))).any())):
        raise ValueError('Targets must be unique in-range same-device long indices')
    if (hard_support.dtype != torch.bool or hard_support.shape != (30,)
            or any(bool(hard_support[i]) for i, c in enumerate(SELECTED_COORDINATES)
                   if c not in LEGAL_COORDINATES)):
        raise ValueError('Require a legal hard support')
    allowed = torch.zeros_like(clean, dtype=torch.bool)
    columns = torch.tensor(SELECTED_COORDINATES, device=clean.device)[hard_support.to(clean.device)]
    allowed[targets[:, None], columns] = True
    if bool((clean[~allowed] != modified[~allowed]).any()):
        raise ValueError('Change outside registered targets/support')
    delta = modified.detach().to(torch.float64) - clean.detach().to(torch.float64)
    per_target = delta[targets]
    return dict(space='standardized_784D_not_physical_units', target_count=targets.numel(),
                support_capacity=int(hard_support.sum()),
                actual_changed_coordinates_per_target=(per_target != 0).sum(1).tolist(),
                normalized_l1_per_target=per_target.abs().sum(1).tolist(),
                normalized_linf_per_target=per_target.abs().max(1).values.tolist(),
                non_target_changed_rows=0)
