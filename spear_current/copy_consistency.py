"""Six within-pair duplicate equalities, after the existing domain projection.

Offline AV2/self-owned-model research. This tensor-only module performs no
asset reads, graph construction, selection, training, evaluation, or writes.
It does NOT enforce cross-pair, history, sign, or physical-state consistency.

The two modes differ only in the final mapping of already bounded outputs.
``identity`` retains the current domain-projected generator bitwise;
``pair_mean`` replaces each of six literal-copy triplets by its arithmetic mean.
This is the nearest equal triplet to the BOUNDED output in squared Euclidean
distance, not a joint projection of the unbounded output onto all constraints.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from collections.abc import Mapping

import numpy as np
import torch

from .domain_generator import DomainConstrainedGenerator
from .support_schema import SELECTED_COORDINATES, coordinate_metadata


CONSISTENCY_MODES = ('identity', 'pair_mean')
COPY_GROUPS = (
    (350, 742, 777), (708, 743, 776), (351, 744, 779),
    (709, 745, 778), (353, 746, 781), (712, 749, 782),
)
COPY_NAMES = ('A.speed_current', 'B.speed_current', 'A.speed_mean',
              'B.speed_mean', 'A.acc_std', 'B.lateral_acc_std')
COPY_POSITIONS = tuple(tuple(SELECTED_COORDINATES.index(c) for c in g) for g in COPY_GROUPS)
VIOLATION_TOLERANCE = 1e-5
_COPY_BUFFERS = ('copy_mode_code', 'copy_frozen_code', 'copy_positions', 'copy_normalizer_digest')


def _normalizer_digest(normalizer):
    """Check equality in the actual scaling contract, not just column names."""
    if not isinstance(normalizer, Mapping) or not {'mean', 'scale'} <= normalizer.keys():
        raise ValueError('Require the frozen 784-column normalizer mean and scale')
    arrays = []
    for key in ('mean', 'scale'):
        raw = np.asarray(normalizer[key])
        if raw.shape != (784,) or raw.dtype.kind != 'f' or not np.isfinite(raw).all():
            raise ValueError('Require finite floating normalizer mean/scale of shape [784]')
        value = raw.astype('<f8', copy=True)
        if key == 'scale' and np.any(value < 1e-6):
            raise ValueError('Normalizer scales must be at least 1e-6')
        for name, group in zip(COPY_NAMES, COPY_GROUPS):
            if not np.all(value[list(group)] == value[group[0]]):
                raise ValueError(f'Different normalization {key} in literal-copy group {name}')
        arrays.append(value)
    for key, expected in (('clip', 10.), ('feature_dimension', 784.), ('minimum_scale', 1e-6)):
        if key in normalizer and float(normalizer[key]) != expected:
            raise ValueError('Frozen normalizer metadata changed: ' + key)
    digest = hashlib.sha256(b'copy_consistency_normalizer_v1\0')
    for value in arrays:
        digest.update(value.tobytes())
    return tuple(digest.digest())


def _values(values):
    if (not isinstance(values, torch.Tensor) or values.dtype not in (torch.float32, torch.float64)
            or values.ndim != 2 or values.shape[1] != 30
            or not bool(torch.isfinite(values).all())):
        raise ValueError('Require finite float32/float64 [M,30] values in the original output order')


def copy_consistency_statistics(values):
    """Describe six triplets only; empty populations have no fabricated rate."""
    _values(values)
    rows = len(values)
    groups, violations = [], []
    for name, coordinates, positions in zip(COPY_NAMES, COPY_GROUPS, COPY_POSITIONS):
        group = values.detach()[:, positions].double()
        spread = group.amax(dim=1) - group.amin(dim=1)
        bad = spread > VIOLATION_TOLERANCE
        violations.append(bad)
        groups.append(dict(name=name, coordinates=list(coordinates), violations=int(bad.sum()),
            violation_rate=float(bad.double().mean()) if rows else None,
            mean_range=float(spread.mean()) if rows else None,
            max_range=float(spread.max()) if rows else None))
    any_bad = torch.stack(violations, dim=1).any(dim=1)
    return dict(rows=rows, tolerance=VIOLATION_TOLERANCE,
        units='frozen_standardized_features_not_physical_units',
        scope='six_within_pair_literal_copy_groups_only', groups=groups,
        any_group_violations=int(any_bad.sum()),
        any_group_violation_rate=float(any_bad.double().mean()) if rows else None)


class CopyConsistentGenerator(DomainConstrainedGenerator):
    """Own unchanged starting weights plus one immutable consistency mode.

    Both modes retain 24 allowed writes, original graph/selection/labels, and
    the old ``output_mode='domain_projection'``. ``pair_mean`` has 12 independent
    output-value degrees of freedom (six triplets and six singletons), not 12
    writable coordinates. There are no new learned parameters or RNG draws.

    ``frozen=True`` is the fixed-model postprocessing/reference variant: every
    parameter is frozen, all children stay in eval, and training is rejected.
    Restore a new checkpoint by constructing THIS class with the same original
    Domain anchor and normalizer, then calling strict load_state_dict. The old
    runtime.restore_generator intentionally remains unchanged.
    """

    def __init__(self, anchor, normalizer, *, mode='pair_mean', frozen=False):
        if type(anchor) is not DomainConstrainedGenerator or anchor.output_mode != 'domain_projection':
            raise ValueError('Require an existing domain-projected DomainConstrainedGenerator')
        if mode not in CONSISTENCY_MODES or type(frozen) is not bool:
            raise ValueError('Require consistency identity/pair_mean and a boolean frozen flag')
        anchor.assert_contract()
        for value in anchor.state_dict().values():
            if (not isinstance(value, torch.Tensor)
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError('Require finite tensor-only anchor state')
        if next(anchor.parameters()).dtype not in (torch.float32, torch.float64):
            raise ValueError('Copy consistency currently supports float32/float64 only')
        for name, group, positions in zip(COPY_NAMES, COPY_GROUPS, COPY_POSITIONS):
            if any(coordinate_metadata(c)['copy_group'] != name for c in group):
                raise ValueError('Literal-copy schema changed')
            if not bool(anchor.hard_support[list(positions)].all()):
                raise ValueError('All three coordinates of all six groups must already be writable')
        digest = _normalizer_digest(normalizer)
        # Do not call Domain's constructor: it expects a Content anchor and
        # re-establishes its own stage. Copy the completed Domain state exactly.
        owned = deepcopy(anchor)
        self.__dict__.update(owned.__dict__)
        self._copy_ready = False
        self._consistency_mode = mode
        self._frozen_postprocess = frozen
        self._expected_normalizer_digest = digest
        device = self.selected_coordinates.device
        self.register_buffer('copy_mode_code', torch.tensor(CONSISTENCY_MODES.index(mode), dtype=torch.uint8, device=device))
        self.register_buffer('copy_frozen_code', torch.tensor(int(frozen), dtype=torch.uint8, device=device))
        self.register_buffer('copy_positions', torch.tensor(COPY_POSITIONS, dtype=torch.long, device=device))
        self.register_buffer('copy_normalizer_digest', torch.tensor(digest, dtype=torch.uint8, device=device))
        self.zero_grad(set_to_none=True)
        if frozen:
            self.requires_grad_(False)
            DomainConstrainedGenerator.train(self, False)
        else:
            self.residual.requires_grad_(True)
        self._seal_buffers()
        self._copy_ready = True
        self.assert_contract()

    @property
    def consistency_mode(self):
        return self._consistency_mode

    @property
    def frozen_postprocess(self):
        return self._frozen_postprocess

    def _copy_tensors(self):
        return dict(tuple(self.named_parameters()) + tuple(self.named_buffers()))

    def _seal_buffers(self):
        super()._seal_buffers()
        if hasattr(self, 'copy_mode_code'):
            self._copy_buffer_contract = {name: (getattr(self, name), getattr(self, name)._version)
                                          for name in _COPY_BUFFERS}
            self._copy_frozen_state = ({name: (p, p._version, p.detach().clone())
                                       for name, p in self._copy_tensors().items()}
                                      if self.frozen_postprocess else {})

    def assert_contract(self):
        super().assert_contract()
        if not getattr(self, '_copy_ready', False):
            return
        expected = dict(
            copy_mode_code=torch.tensor(CONSISTENCY_MODES.index(self.consistency_mode), dtype=torch.uint8),
            copy_frozen_code=torch.tensor(int(self.frozen_postprocess), dtype=torch.uint8),
            copy_positions=torch.tensor(COPY_POSITIONS, dtype=torch.long),
            copy_normalizer_digest=torch.tensor(self._expected_normalizer_digest, dtype=torch.uint8))
        for name, value in expected.items():
            actual = getattr(self, name)
            original, version = self._copy_buffer_contract[name]
            if (actual is not original or actual._version != version
                    or actual.device != self.selected_coordinates.device or actual.dtype != value.dtype
                    or actual.shape != value.shape or not torch.equal(actual.detach().cpu(), value)):
                raise ValueError('Immutable copy-consistency contract changed: ' + name)
        if next(self.parameters()).dtype not in (torch.float32, torch.float64):
            raise ValueError('Copy consistency supports float32/float64 only')
        if self.frozen_postprocess:
            if any(child.training for child in self.modules()):
                raise ValueError('Frozen postprocess and all children must stay in eval')
            tensors = self._copy_tensors()
            if set(tensors) != set(self._copy_frozen_state):
                raise ValueError('Frozen tensor membership changed')
            for name, (original, version, value) in self._copy_frozen_state.items():
                p = tensors[name]
                if (p is not original or p._version != version or p.requires_grad or p.grad is not None
                        or not torch.equal(p.detach(), value)):
                    raise ValueError('Frozen postprocess tensor changed: ' + name)

    def train(self, mode=True):
        if getattr(self, '_copy_ready', False) and self.frozen_postprocess and mode is not False:
            raise ValueError('Frozen postprocess rejects training; use eval()')
        return super().train(mode)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.assert_contract()
        if strict is not True or assign is not False:
            raise ValueError('Require strict copy-based consistency restoration')
        current = self.state_dict()
        fixed = current if self.frozen_postprocess else _COPY_BUFFERS
        for name in fixed:
            value = state_dict.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                    or value.dtype != current[name].dtype
                    or not torch.equal(value.detach().cpu(), current[name].detach().cpu())):
                raise ValueError('Checkpoint changes fixed consistency contract/state: ' + name)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def apply_output(self, clean_values, generated_values):
        _values(clean_values)
        _values(generated_values)
        # Both arms reject input representation drift rather than silently
        # repairing a changed clean-data contract in only the treatment arm.
        clean_groups = clean_values[:, self.copy_positions]
        if not torch.equal(clean_groups, clean_groups[:, :, :1].expand_as(clean_groups)):
            raise ValueError('Clean input violates the six literal-copy equalities')
        bounded = super().apply_output(clean_values, generated_values)
        if self.consistency_mode == 'identity':
            return bounded
        group = bounded[:, self.copy_positions].double()
        # Shifted mean is the ordinary arithmetic mean with its true Jacobian
        # (1/3 per copy). It preserves exactly equal triplets bitwise, including
        # float64 0.1, without an equality branch or straight-through gradient.
        first = group[:, :, :1]
        mean = (first - (first - group).mean(dim=2, keepdim=True)).to(bounded.dtype)
        result = bounded.clone()
        result[:, self.copy_positions] = mean.expand_as(group).to(bounded.dtype)
        self.assert_contract()
        return result

    def contract(self):
        self.assert_contract()
        return dict(version='within_pair_copy_consistency_v1', mode=self.consistency_mode,
            frozen_postprocess=self.frozen_postprocess, output_mode=self.output_mode,
            mapping_order=['old_domain_projection', self.consistency_mode],
            groups=[dict(name=n, coordinates=list(g)) for n, g in zip(COPY_NAMES, COPY_GROUPS)],
            allowed_write_coordinates=self.selected_coordinates[self.hard_support].tolist(),
            allowed_write_count=24, independent_output_value_dof=12 if self.consistency_mode == 'pair_mean' else 24,
            normalizer_digest=bytes(self._expected_normalizer_digest).hex(),
            violation_tolerance=VIOLATION_TOLERANCE, cross_pair_synchronization=False,
            physical_validity_claim=False, clean_reference_independence_claim=False)
