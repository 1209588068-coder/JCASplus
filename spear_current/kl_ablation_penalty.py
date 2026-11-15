"""Fixed clean-KL objectives for a stage-specific warm-start ablation.

Both independent frozen proxies are evaluated exactly once in every mode.  The
only treatment is how their two differentiable losses enter the current outer
update.  Disabling this objective does not erase KL used by the common parent.
No dataset, optimizer, training launch, or evaluation access lives here.
"""
from __future__ import annotations

import torch

from .dual_clean_proxy import FrozenMeanCleanResponsePenalty


MODES = ('dual', 'single_proxy1', 'disabled')


def _coefficients(mode):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError('Require fixed clean-KL mode dual, single_proxy1, or disabled')
    return {'dual': (.5, .5), 'single_proxy1': (1., 0.), 'disabled': (0., 0.)}[mode]


class FixedProxyAblationPenalty(FrozenMeanCleanResponsePenalty):
    """Retain the original two-proxy contract with one immutable loss treatment.

    ``raw_kl`` and ``mean_kl`` always describe the unweighted two-proxy mean.
    ``effective_kl`` / ``weighted_kl`` describe the scalar actually returned to
    the optimizer.  Thus the disabled arm still measures its nonzero raw KL.
    The public ``weight`` and both member weights remain one, so the unchanged
    training loop's frozen-input/independent-proxy checks remain applicable.
    """

    def __init__(self, models, original_x, edge_index, edge_weight, *, mode='dual'):
        coefficients = _coefficients(mode)
        models = tuple(models)
        if len(models) != 2:
            raise ValueError('The clean-KL ablation requires exactly two independent proxies')
        super().__init__(models, original_x, edge_index, edge_weight, weight=1.)
        self._mode = mode
        self._coefficients = coefficients
        self._registered_contract = (mode, coefficients)
        self._member_contract = tuple((member, member.model) for member in self.members)
        self.assert_contract()

    @property
    def mode(self):
        return self._mode

    @property
    def coefficients(self):
        return self._coefficients

    def assert_contract(self):
        if ((self.mode, self.coefficients) != self._registered_contract
                or self.coefficients != _coefficients(self.mode)
                or self.weight != 1. or len(self.members) != 2
                or self.stats is not self.history):
            raise ValueError('Immutable clean-KL ablation mode, coefficients or weighting changed')
        for member, (original, model) in zip(self.members, self._member_contract):
            if (member is not original or member.model is not model or member.weight != 1.
                    or member.calls != self.calls):
                raise ValueError('Clean-KL proxy membership, weight or call count changed')
            if (any(module.training for module in model.modules())
                    or any(p.requires_grad or p.grad is not None for p in model.parameters())
                    or member.q.requires_grad):
                raise ValueError('Clean-KL proxies and cached responses must remain frozen/eval')

    def __call__(self, update_feat, edge_index, edge_weight, idx_outter):
        self.assert_contract()
        losses = [member(update_feat, edge_index, edge_weight, idx_outter)
                  for member in self.members]
        # Preserve the original reduction and multiplication order exactly for
        # dual replay. Zero-multiplied paths still receive zero input gradients.
        mean_kl = torch.stack(losses).mean()
        if self.mode == 'dual':
            effective = self.weight * mean_kl
        elif self.mode == 'single_proxy1':
            effective = losses[0] + 0. * losses[1]
        else:
            effective = 0. * mean_kl
        if not bool(torch.isfinite(mean_kl)) or not bool(torch.isfinite(effective)):
            raise ValueError('Nonfinite clean-KL ablation loss')
        number = self.calls + 1
        self.history.append(dict(call=number, calls=number, member_count=2,
            member_raw_kl=[member.history[-1]['raw_kl'] for member in self.members],
            raw_kl=float(mean_kl.detach()), mean_kl=float(mean_kl.detach()),
            effective_kl=float(effective.detach()), weighted_kl=float(effective.detach()),
            idx_count=int(idx_outter.numel()), mode=self.mode,
            coefficients=list(self.coefficients)))
        self.assert_contract()
        return effective
