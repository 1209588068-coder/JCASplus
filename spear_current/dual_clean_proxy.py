# Source: experiments/spear_dual_clean_proxy_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""Mean label-free clean-response KL for independent frozen TRAIN proxies.

This module belongs to the local, authorized AV2 model-robustness experiment.
It neither trains a proxy nor receives labels or final evaluation models.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from .clean_response import FrozenCleanResponsePenalty


class FrozenMeanCleanResponsePenalty:
    """Average each proxy's own response KL, keeping the total weight fixed.

    The loss is ``weight * mean_k KL(q_k(original) || p_k(changed))``.
    It is deliberately *not* the KL between averaged class probabilities.
    Proxies must be distinct modules with independently owned parameters and
    buffers. All use the same original TRAIN graph and outer-node indices.

    ``members`` exposes the unchanged single-proxy penalties, whose weights
    are one (zero when this entire constraint is disabled). A single member
    uses no stack/reduction so its value and input gradient match the old
    single-proxy calculation exactly. Freezing only proxy parameters leaves
    the changed features in autograd.
    """

    def __init__(
        self,
        models: Iterable[nn.Module],
        original_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        weight: float = 1.0,
    ) -> None:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Clean-response weight must be finite and nonnegative")
        models = tuple(models)
        if not models or any(not isinstance(model, nn.Module) for model in models):
            raise ValueError("Expected at least one independent clean proxy module")
        if len({id(model) for model in models}) != len(models):
            raise ValueError("Clean proxies must be distinct model objects")

        # Distinct nn.Parameter wrappers may still alias the same backing
        # storage, so an object-identity check alone is insufficient. Buffers
        # are included because they also form part of a proxy checkpoint.
        seen_objects, seen_storages = set(), set()
        for model in models:
            tensors = tuple(model.parameters()) + tuple(model.buffers())
            objects = {id(tensor) for tensor in tensors}
            storages = {
                (str(tensor.device), tensor.untyped_storage().data_ptr())
                for tensor in tensors if tensor.numel()
            }
            if objects & seen_objects or storages & seen_storages:
                raise ValueError("Clean proxies cannot share parameter or buffer storage")
            seen_objects.update(objects)
            seen_storages.update(storages)

        self.weight = float(weight)
        self.members = tuple(
            FrozenCleanResponsePenalty(
                model, original_x, edge_index, edge_weight,
                weight=1.0 if self.weight else 0.0,
            )
            for model in models
        )
        if any(member.q.shape != self.members[0].q.shape for member in self.members):
            raise ValueError("Clean proxies must share the same class-output shape")
        self.history: list[dict[str, object]] = []
        self.stats = self.history

    @property
    def calls(self) -> int:
        return len(self.history)

    def __call__(
        self,
        update_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        idx_outter: torch.Tensor,
    ) -> torch.Tensor:
        losses = [member(update_feat, edge_index, edge_weight, idx_outter)
                  for member in self.members]
        mean_kl = losses[0] if len(losses) == 1 else torch.stack(losses).mean()
        weighted_kl = self.weight * mean_kl
        if not bool(torch.isfinite(weighted_kl)):
            raise ValueError("Nonfinite mean clean-response KL")
        self.history.append({
            "call": self.calls + 1,
            "member_count": len(self.members),
            "member_raw_kl": [member.history[-1]["raw_kl"] for member in self.members],
            "raw_kl": float(mean_kl.detach()),
            "mean_kl": float(mean_kl.detach()),
            "weighted_kl": float(weighted_kl.detach()),
            "idx_count": int(idx_outter.numel()),
        })
        return weighted_kl
