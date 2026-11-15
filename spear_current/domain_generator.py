# Source: experiments/spear_domain_training_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""Continue an existing uniform residual under one immutable output mapping.

No data, model loading, training loop, optimizer, or evaluation lives here.
The constructor owns deep copies of completed weights, without initializing a
new residual/head/scorer or consuming random numbers. Base G and its conditioning
shadow remain frozen/eval; only the inherited residual may learn.

Both fixed modes use exactly the same immutable 24-of-30 support. Identity
preserves selected generated values bitwise. Domain projection uses ordinary
torch.clamp on those values: saturated derivatives really are zero, with no STE.
The other six outputs equal the original clean values in training and inference.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn

from .attention_generator import ContentAttentionGenerator
from .projection import DOMAIN_MIN, DOMAIN_MAX
from .support_schema import LEGAL_COORDINATES, SELECTED_COORDINATES


OUTPUT_MODES = ('identity', 'domain_projection')


class DomainConstrainedGenerator(ContentAttentionGenerator):
    """Tensor-state-only subclass compatible with the old residual trainer.

    ``hard_support`` is a legal bool[30] vector in the frozen importance order,
    selecting exactly 24 entries. ``output_mode`` is immutable and checkpointed
    by scalar uint8 ``output_mode_code`` (identity=0, domain_projection=1).
    No dictionary extra_state is used: the old trainer hashes tensor state only.

    Public ``apply_output(clean30, raw30)`` exposes the same differentiable
    final-output operation used by forward. The inherited forward_components,
    or ContentAttentionGenerator.forward(self,...), remain raw baseline/residual
    diagnostics; all deployment and training injections use this forward.
    """

    def __init__(self, anchor, hard_support, *, output_mode='identity'):
        if (type(anchor) is not ContentAttentionGenerator or anchor.attention_mode != 'uniform'
                or anchor.residual.attention_mode != 'uniform' or anchor.residual.mode != 'untyped'
                or anchor.input_dim != 784 or tuple(anchor.selected_coordinates.tolist()) != SELECTED_COORDINATES
                or anchor.score_dim != anchor.residual.score_dim
                or not math.isfinite(float(anchor.threshold))):
            raise ValueError('Require the completed uniform 784-D/30-output ContentAttentionGenerator')
        if output_mode not in OUTPUT_MODES:
            raise ValueError('Only immutable identity/domain_projection output modes are allowed')
        self._validate_mask(hard_support)
        for value in anchor.state_dict().values():
            if not isinstance(value, torch.Tensor) or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('Anchor must have finite tensor-only state')
        contracts = {(p.device, p.dtype) for p in anchor.parameters()}
        if len(contracts) != 1:
            raise ValueError('Anchor parameters must share one device/dtype')
        parameter = next(anchor.parameters())
        if anchor.selected_coordinates.device != parameter.device:
            raise ValueError('Anchor coordinate mapping and parameters must share a device')

        # Calling the original constructor would reinitialize a scorer. Register
        # only already-trained deep copies and the new immutable tensor buffers.
        nn.Module.__init__(self)
        self.input_dim = anchor.input_dim
        self.threshold = float(anchor.threshold)
        self.attention_mode = anchor.attention_mode
        self.score_dim = anchor.score_dim
        self.base_generator = deepcopy(anchor.base_generator)
        self.base_shadow = deepcopy(anchor.base_shadow)
        self.residual = deepcopy(anchor.residual)
        self.register_buffer('selected_coordinates', anchor.selected_coordinates.detach().clone())
        self.register_buffer('hard_support', hard_support.detach().to(device=parameter.device).clone())
        self.register_buffer('output_mode_code', torch.tensor(OUTPUT_MODES.index(output_mode),
                                                             dtype=torch.uint8, device=parameter.device))
        self._expected_output_mode = output_mode
        self._expected_support = tuple(self.hard_support.tolist())
        self._expected_configuration = self._configuration()
        self._owned_modules = {name: getattr(self, name) for name in ('base_generator', 'base_shadow', 'residual')}
        self.residual.requires_grad_(True)
        for parameter in self.residual.parameters():
            parameter.grad = None
        self._freeze_base()
        self._seal_buffers()
        self.train(anchor.training)

    @staticmethod
    def _validate_mask(mask):
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (30,)
                or int(mask.sum()) != 24):
            raise ValueError('Require a fixed bool[30] support selecting exactly 24 coordinates')
        if any(bool(mask[i]) for i, coordinate in enumerate(SELECTED_COORDINATES) if coordinate not in LEGAL_COORDINATES):
            raise ValueError('Focal metadata coordinates cannot be modified')

    @property
    def output_mode(self):
        return self._expected_output_mode

    def _configuration(self):
        scorer = self.residual.scorer
        return (self.input_dim, self.threshold, self.attention_mode, self.score_dim,
                self.residual.input_dim, self.residual.hidden_dim, self.residual.output_dim,
                self.residual.mode, self.residual.attention_mode, self.residual.score_dim,
                scorer.heads, scorer.negative_slope, scorer.dropout, scorer.add_self_loops,
                scorer.share_weights, scorer.edge_dim)

    def _seal_buffers(self):
        self._buffer_contract = {name: (getattr(self, name), getattr(self, name)._version)
                                 for name in ('selected_coordinates', 'hard_support', 'output_mode_code')}

    def _check_values(self):
        self._validate_mask(self.hard_support)
        if (tuple(self.selected_coordinates.tolist()) != SELECTED_COORDINATES
                or self.selected_coordinates.dtype != torch.long or self.selected_coordinates.shape != (30,)
                or tuple(self.hard_support.tolist()) != self._expected_support
                or self.output_mode_code.dtype != torch.uint8 or self.output_mode_code.shape != ()
                or int(self.output_mode_code) != OUTPUT_MODES.index(self._expected_output_mode)
                or self._configuration() != self._expected_configuration):
            raise ValueError('Immutable output mode, support, coordinate mapping or attention configuration changed')
        if any(getattr(self, name) is not value for name, value in self._owned_modules.items()):
            raise ValueError('Owned base/residual module was replaced')
        if any(value.device != self.selected_coordinates.device for value in (self.hard_support, self.output_mode_code)):
            raise ValueError('Output-contract buffers have inconsistent devices')

    def assert_contract(self):
        """Public immutability check for caller progress/update boundaries."""
        self._check_values()
        for name, (original, version) in self._buffer_contract.items():
            if getattr(self, name) is not original or original._version != version:
                raise ValueError('Immutable output-contract tensor changed or was replaced: ' + name)
        for base in (self.base_generator, self.base_shadow):
            if (any(child.training for child in base.modules())
                    or any(p.requires_grad or p.grad is not None for p in base.parameters())):
                raise ValueError('Base generator and conditioning shadow must remain frozen/eval/gradient-free')

    def train(self, mode=True):
        if type(mode) is not bool:
            raise ValueError('Training mode must be boolean')
        if hasattr(self, '_buffer_contract'):
            self.assert_contract()
        result = super().train(mode)  # Inherited implementation re-freezes bases.
        self.assert_contract()
        return result

    def _apply(self, fn, recurse=True):
        # Ordinary device/dtype transfers may replace buffer objects. Permit
        # those only if the contract was intact before and unchanged afterward.
        self.assert_contract()
        result = super()._apply(fn, recurse=recurse)
        self._check_values()
        self._seal_buffers()
        self.assert_contract()
        return result

    def __deepcopy__(self, memo):
        # Tensor deepcopy starts a new version counter. Preserve all copied
        # module state/alias relationships, then bind guards to the owned copy.
        self.assert_contract()
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        copied.__dict__.update(deepcopy(self.__dict__, memo))
        copied._seal_buffers()
        copied.assert_contract()
        return copied

    def state_dict(self, *args, **kwargs):
        self.assert_contract()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.assert_contract()
        if strict is not True or assign is not False:
            raise ValueError('Require strict copy-based domain-generator restoration')
        current = self.state_dict()
        if set(state_dict) != set(current):
            raise ValueError('Checkpoint tensor membership differs from this domain generator')
        # Preflight every tensor before touching any owned parameters/buffers.
        for name, value in state_dict.items():
            if (not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                    or value.dtype != current[name].dtype
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError('Malformed/nonfinite domain-generator checkpoint tensor: ' + name)
            if (name in ('selected_coordinates', 'hard_support', 'output_mode_code')
                    or name.startswith(('base_generator.', 'base_shadow.'))):
                if not torch.equal(value.detach().cpu(), current[name].detach().cpu()):
                    raise ValueError('Checkpoint changes fixed support/mode/coordinates or frozen base: ' + name)
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self._check_values()
        self._seal_buffers()
        self._freeze_base()
        self.assert_contract()
        return result

    def apply_output(self, clean_values, generated_values):
        """Apply the immutable final transform; native autograd, no gradient tricks."""
        self.assert_contract()
        if (not isinstance(clean_values, torch.Tensor) or not isinstance(generated_values, torch.Tensor)
                or clean_values.ndim != 2 or clean_values.shape[1] != 30
                or clean_values.shape != generated_values.shape or not clean_values.is_floating_point()
                or clean_values.dtype != generated_values.dtype or clean_values.device != generated_values.device
                or clean_values.device != self.hard_support.device
                or not bool(torch.isfinite(clean_values).all()) or not bool(torch.isfinite(generated_values).all())):
            raise ValueError('Require aligned finite same-device/dtype clean and generated [M,30] values')
        if bool(((clean_values < DOMAIN_MIN) | (clean_values > DOMAIN_MAX)).any()):
            raise ValueError('Original clean selected values exceed their fixed [-10,10] domain')
        values = (generated_values if self.output_mode == 'identity' else
                  generated_values.clamp(min=DOMAIN_MIN, max=DOMAIN_MAX))
        result = torch.where(self.hard_support.unsqueeze(0), values, clean_values)
        self.assert_contract()
        return result

    def forward(self, features, context, target_indices):
        self.assert_contract()
        if features.requires_grad:
            raise ValueError('Original conditioning features must not require gradients')
        generated = super().forward(features, context, target_indices)
        clean = features[target_indices[:, None], self.selected_coordinates]
        return self.apply_output(clean, generated)
