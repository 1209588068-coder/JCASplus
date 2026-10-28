# Source: experiments/spear_amplitude_control_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""Pure amplitude controls on an already completed, domain-projected generator.

Only tensors supplied by the caller are used. There is no data loading,
optimizer, fitting loop, selection, or graph construction in this module.
The original normalized feature domain is [-10, 10]; this is an offline
feature-space robustness control, not a physical trajectory constraint.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

import torch
from torch import nn

from .domain_generator import DomainConstrainedGenerator
from .role_generator import (
    InjectedFeatures, PairRoleResidualGenerator,
)


MATCH_TOLERANCE = 1e-5


def _scale(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or not 0 < float(value) <= 1):
        raise ValueError('Scale must be one finite scalar with 0 < scale <= 1; no clipping')
    return float(value)


def _aligned_values(values, width, *, nonempty=False):
    first = values[0]
    if (not isinstance(first, torch.Tensor) or not first.is_floating_point()
            or first.ndim != 2 or first.shape[1] != width
            or (nonempty and first.shape[0] == 0)):
        raise ValueError(f'Require aligned floating [M,{width}] tensors' +
                         (' with M > 0' if nonempty else ''))
    for value in values:
        if (not isinstance(value, torch.Tensor) or value.shape != first.shape
                or value.dtype != first.dtype or value.device != first.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Require aligned finite tensors of the same floating dtype/device')
        if bool(((value < -10.) | (value > 10.)).any()):
            raise ValueError('All supplied selected values must be inside exact [-10,10] bounds')


def _interpolate(clean, generated, scale):
    """Float64 arithmetic, with endpoint/zero-delta bit preservation."""
    if scale == 1.:
        return generated.clone()
    result = (clean.to(torch.float64) + scale *
              (generated.to(torch.float64) - clean.to(torch.float64))).to(clean.dtype)
    return torch.where(generated == clean, clean, result)


@torch.no_grad()
def calibrate_scale(clean_selected, continuation_selected, unconditional_selected):
    """Match mean per-row L1 using one global scalar U/C on supplied [M,24].

The caller supplies its actual training-attachment rows in the fixed support
order. Every input must be finite and within exact [-10,10]. Empty/zero
calibration populations and ratios outside (0,1] are rejected, never clipped.
The returned statistics are JSON-compatible Python values. The actual L1 is
reconstructed from the dtype-restored output, and must match U within 1e-5
relative OR absolute tolerance. No per-row scale or learned state is created.
"""
    _aligned_values((clean_selected, continuation_selected, unconditional_selected),
                    24, nonempty=True)
    clean64 = clean_selected.to(torch.float64)
    continuation_mean = float((continuation_selected.to(torch.float64) - clean64).abs().sum(1).mean())
    unconditional_mean = float((unconditional_selected.to(torch.float64) - clean64).abs().sum(1).mean())
    if not math.isfinite(continuation_mean) or continuation_mean <= 0:
        raise ValueError('Continuation mean per-row L1 must be finite and positive')
    if not math.isfinite(unconditional_mean) or unconditional_mean <= 0:
        raise ValueError('Unconditional mean per-row L1 must be finite and positive')
    scale = _scale(unconditional_mean / continuation_mean)
    reconstructed = _interpolate(clean_selected, continuation_selected, scale)
    actual_mean = float((reconstructed.to(torch.float64) - clean64).abs().sum(1).mean())
    absolute_error = abs(actual_mean - unconditional_mean)
    relative_error = absolute_error / unconditional_mean
    matched = math.isclose(actual_mean, unconditional_mean,
                           rel_tol=MATCH_TOLERANCE, abs_tol=MATCH_TOLERANCE)
    if not matched:
        raise ValueError('Dtype-restored mean per-row L1 does not match the calibration target within 1e-5')
    return scale, dict(
        scale=scale, target_count=clean_selected.shape[0], support_count=24,
        continuation_mean_l1=continuation_mean,
        unconditional_mean_l1=unconditional_mean, actual_mean_l1=actual_mean,
        absolute_error=absolute_error, relative_error=relative_error,
        absolute_tolerance=MATCH_TOLERANCE, relative_tolerance=MATCH_TOLERANCE,
        matched=matched,
    )


class FrozenAmplitudeGenerator(nn.Module):
    """Own a completed projected generator and one immutable scalar amplitude.

All parameters are frozen, every child stays in eval, and ``train(True)`` is
rejected. This standalone Module does not inherit residual training APIs.
Checkpoint restoration is strict and only accepts the same frozen weights,
support, and scalar. Device/dtype transfer and deepcopy preserve the guards.
"""

    def __init__(self, anchor, scale):
        super().__init__()
        scale = _scale(scale)
        if type(anchor) is not DomainConstrainedGenerator or anchor.output_mode != 'domain_projection':
            raise ValueError('Require a completed DomainConstrainedGenerator in domain_projection mode')
        anchor.assert_contract()
        if any(not isinstance(value, torch.Tensor) or
               (value.is_floating_point() and not bool(torch.isfinite(value).all()))
               for value in anchor.state_dict().values()):
            raise ValueError('Require finite tensor-only completed generator state')
        self.anchor = deepcopy(anchor)
        self.anchor.requires_grad_(False)
        for parameter in self.anchor.parameters():
            parameter.grad = None
        self.anchor.eval()
        self.training = False
        self.register_buffer('scale_value', torch.tensor(scale, dtype=torch.float64,
                                                        device=self.anchor.selected_coordinates.device))
        self._expected_scale = scale
        self._owned_modules = {'anchor': self.anchor}
        self._seal_state()
        self.assert_contract()

    @property
    def input_dim(self):
        return self.anchor.input_dim

    @property
    def selected_coordinates(self):
        return self.anchor.selected_coordinates

    @property
    def hard_support(self):
        return self.anchor.hard_support

    @property
    def scale(self):
        return float(self.scale_value)

    @property
    def output_mode(self):
        # Compatibility with the existing assert_output_scope domain guard.
        return 'domain_projection'

    def _tensors(self):
        return dict(tuple(self.named_parameters()) + tuple(self.named_buffers()))

    def _seal_state(self):
        self._state_contract = {
            name: (value, value._version, value.detach().clone())
            for name, value in self._tensors().items()
        }
        self._module_contract = dict(self.named_modules())

    def assert_contract(self):
        """Reject state, ownership, mode, gradient, or scalar-contract changes."""
        if self.anchor is not self._owned_modules['anchor'] or dict(self.named_modules()) != self._module_contract:
            raise ValueError('Owned frozen generator or one of its modules was replaced')
        if any(child.training for child in self.modules()):
            raise ValueError('Amplitude generator and all owned modules must stay in eval')
        if any(parameter.requires_grad or parameter.grad is not None for parameter in self.parameters()):
            raise ValueError('All amplitude-generator parameters must remain frozen and gradient-free')
        self.anchor.assert_contract()
        if (self.anchor.output_mode != 'domain_projection' or self.scale_value.shape != ()
                or self.scale_value.dtype != torch.float64 or self.scale_value.requires_grad
                or self.scale_value.grad is not None
                or self.scale_value.device != self.selected_coordinates.device
                or self.scale != self._expected_scale):
            raise ValueError('Immutable scalar amplitude/domain contract changed')
        _scale(self.scale)
        current = self._tensors()
        if set(current) != set(self._state_contract):
            raise ValueError('Frozen state tensor membership changed')
        for name, (original, version, value) in self._state_contract.items():
            if (current[name] is not original or original._version != version
                    or not torch.equal(original.detach(), value)):
                raise ValueError('Frozen amplitude-generator tensor changed or was replaced: ' + name)

    def train(self, mode=True):
        if type(mode) is not bool or mode:
            raise ValueError('Frozen amplitude generator rejects training; use eval()')
        self.assert_contract()
        return self

    def _apply(self, fn, recurse=True):
        self.assert_contract()
        result = super()._apply(fn, recurse=recurse)
        # A caller's .float()/.half() must not round the calibrated scalar.
        self.scale_value = torch.tensor(self._expected_scale, dtype=torch.float64,
                                        device=self.anchor.selected_coordinates.device)
        self._seal_state()
        self.assert_contract()
        return result

    def __deepcopy__(self, memo):
        self.assert_contract()
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        copied.__dict__.update(deepcopy(self.__dict__, memo))
        copied._seal_state()
        copied.assert_contract()
        return copied

    def state_dict(self, *args, **kwargs):
        self.assert_contract()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.assert_contract()
        if strict is not True or assign is not False:
            raise ValueError('Require strict copy-based frozen amplitude restoration')
        current = self.state_dict()
        if set(state_dict) != set(current):
            raise ValueError('Checkpoint tensor membership differs from this amplitude generator')
        for name, value in state_dict.items():
            if (not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                    or value.dtype != current[name].dtype
                    or not torch.equal(value.detach().cpu(), current[name].detach().cpu())):
                raise ValueError('Checkpoint changes frozen amplitude state: ' + name)
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        # Recursive Module restoration copies child buffers directly, bypassing
        # the child's public load_state_dict; rebind its version seals explicitly.
        self.anchor._seal_buffers()
        self._seal_state()
        self.assert_contract()
        return result

    @torch.no_grad()
    def apply_output(self, clean_values, generated_values):
        """Apply the fixed scale to [M,30], preserving six clean values exactly."""
        self.assert_contract()
        _aligned_values((clean_values, generated_values), 30)
        if clean_values.device != self.selected_coordinates.device:
            raise ValueError('Output tensors and frozen generator must share a device')
        values = _interpolate(clean_values, generated_values, self.scale)
        result = torch.where(self.hard_support.unsqueeze(0), values, clean_values)
        if not bool(torch.isfinite(result).all()) or bool((result.abs() > 10.).any()):
            raise ValueError('Amplitude interpolation left the exact finite [-10,10] domain')
        self.assert_contract()
        return result

    @torch.no_grad()
    def forward(self, features, context, target_indices):
        self.assert_contract()
        generated = self.anchor(features, context, target_indices)
        clean = features[target_indices[:, None], self.selected_coordinates]
        return self.apply_output(clean, generated)

    @torch.no_grad()
    def inject(self, features, context, target_indices, edge_weight=None,
               *, detach=False) -> InjectedFeatures:
        self.assert_contract()
        result = PairRoleResidualGenerator.inject(
            self, features, context, target_indices, edge_weight, detach=detach)
        self.assert_contract()
        return result
