# Source: experiments/spear_clean_response_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""A frozen, label-free clean-response constraint for the SPEAR outer loss."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrozenCleanResponsePenalty:
    """Keep a dedicated clean proxy's predictions near its original response.

    ``model`` must return class log probabilities, as the supplied GraphSage
    does.  Its parameters are frozen, but the changed input stays in autograd.
    No labels, validation scores, or final evaluation model enter this object.
    The caller owns the separately trained proxy and the original TRAIN graph.
    """

    def __init__(
        self,
        model: nn.Module,
        original_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        weight: float = 0.1,
    ) -> None:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Clean-response weight must be finite and nonnegative")
        if original_x.ndim != 2 or not bool(torch.isfinite(original_x).all()):
            raise ValueError("Expected a finite two-dimensional original feature tensor")
        self.model = model
        self.weight = float(weight)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        with torch.no_grad():
            output = self.model(original_x, edge_index, edge_weight)
            self._check_output(output, len(original_x))
            self.q = output.exp().detach()
        if self.q.ndim != 2 or self.q.shape[0] != original_x.shape[0]:
            raise ValueError("Clean proxy returned an unexpected output shape")
        self.feature_shape = tuple(original_x.shape)
        self.edges = edge_index.detach().clone()
        self.edge_weights = None if edge_weight is None else edge_weight.detach().clone()
        self.stats: list[dict[str, float | int]] = []
        self.history = self.stats

    @property
    def calls(self) -> int:
        return len(self.stats)

    @staticmethod
    def _check_output(output, nodes):
        if (output.ndim != 2 or output.shape[0] != nodes or output.shape[1] < 2
                or not bool(torch.isfinite(output).all())
                or not torch.allclose(output.logsumexp(1), torch.zeros_like(output[:, 0]), atol=1e-5)):
            raise ValueError("Clean proxy must return finite normalized class log probabilities")

    def __call__(
        self,
        update_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        idx_outter: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(update_feat.shape) != self.feature_shape:
            raise ValueError("Changed features differ from the cached original graph shape")
        if idx_outter.ndim != 1 or idx_outter.dtype != torch.long:
            raise ValueError("Outer indices must be a one-dimensional long tensor")
        if bool(((idx_outter < 0) | (idx_outter >= len(update_feat))).any()):
            raise ValueError("Outer index outside original TRAIN graph")
        if not torch.equal(edge_index, self.edges) or ((edge_weight is None) != (self.edge_weights is None)):
            raise ValueError("Clean-response constraint cannot change the bound topology")
        if edge_weight is not None and not torch.equal(edge_weight, self.edge_weights):
            raise ValueError("Clean-response constraint cannot change edge weights")
        if not bool(torch.isfinite(update_feat).all()):
            raise ValueError("Changed features must be finite")
        self.model.eval()
        if idx_outter.numel() and self.weight:
            changed_log_prob = self.model(update_feat, edge_index, edge_weight)
            self._check_output(changed_log_prob, len(update_feat))
            raw_kl = F.kl_div(
                changed_log_prob[idx_outter], self.q[idx_outter], reduction="batchmean"
            )
        else:
            raw_kl = update_feat.sum() * 0.0
        weighted_kl = self.weight * raw_kl
        if not bool(torch.isfinite(weighted_kl)):
            raise ValueError("Nonfinite clean-response KL")
        self.stats.append({
            "call": len(self.stats) + 1,
            "raw_kl": float(raw_kl.detach()),
            "weighted_kl": float(weighted_kl.detach()),
            "idx_count": int(idx_outter.numel()),
        })
        return weighted_kl
