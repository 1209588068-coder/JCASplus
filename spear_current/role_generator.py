# Source: experiments/spear_pair_role_generator_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""Module-only endpoint-role residual extension of the frozen SPEAR generator.

This is a feature-space experiment, NOT physical trajectory or subgraph injection.
No dataset reader, loss, optimizer, training entry point, or holdout selection lives
here. Existing experiments are not patched. The branch uses original normalized
pair features and the existing one-hop line graph, never shadow embeddings as its
message inputs (which would silently add another hop).

Relation-specific linear messages follow the idea of R-GCN (Schlichtkrull et al.,
arXiv:1703.06103), with the SAME total incoming-degree normalization in every arm.
This is a local implementation of that principle, not copied official R-GCN code.
The zero output head gives identity at initialization, not an ASR guarantee.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn

from .role_context import RoleContext


@dataclass(frozen=True)
class GeneratedFeatures:
    values: torch.Tensor
    baseline: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class InjectedFeatures:
    features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor


def _check_features(features: torch.Tensor, context: RoleContext, input_dim: int):
    if not isinstance(context, RoleContext):
        raise ValueError("Expected an explicit RoleContext sidecar")
    context.validate()
    if not isinstance(features, torch.Tensor) or not features.is_floating_point():
        raise ValueError("Features must be a floating-point tensor")
    if features.ndim != 2 or features.shape != (context.num_pairs, input_dim):
        raise ValueError("Feature shape disagrees with pair context/input dimension")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite features")
    if any(t.device != features.device for t in (
        context.edge_index, context.edge_type, context.pair_scene,
    )):
        raise ValueError("Context and features must be on the same device")


class RoleResidualBlock(nn.Module):
    """One-hop relation-specific messages plus a self branch and zero output head.

    h_i = ReLU(W_self x_i + b + sum_{j -> i} W_role(j,i) x_j / max(deg(i),1))
    delta_i = W_out h_i + b_out

    ``untyped`` uses mean_r W_r and NEVER uses the role values for aggregation or
    normalization. This matches allocated parameters, not effective capacity: its
    four matrices reduce to one common linear map. The stronger role-information
    control is the typed block with a separately frozen per-target role shuffle.
    No dropout/batch normalization or other cross-scene statistics are introduced.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 mode: str = "typed"):
        super().__init__()
        if mode not in ("typed", "untyped"):
            raise ValueError("mode must be typed or untyped")
        if any(type(d) is not int or d <= 0 for d in (input_dim, hidden_dim, output_dim)):
            raise ValueError("Dimensions must be positive integers")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode
        self.relation_weight = nn.Parameter(torch.empty(4, input_dim, hidden_dim))
        self.self_projection = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        for weight in self.relation_weight:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor, context: RoleContext):
        _check_features(features, context, self.input_dim)
        if (features.device != self.relation_weight.device
                or features.dtype != self.relation_weight.dtype):
            raise ValueError("Residual parameters and features must share device/dtype")
        source, target = context.edge_index
        aggregate = features.new_zeros((context.num_pairs, self.hidden_dim))
        if self.mode == "untyped":
            transformed = features @ self.relation_weight.mean(dim=0)
            aggregate = aggregate.index_add(0, target, transformed[source])
        else:
            for role in range(4):
                selected = context.edge_type == role
                # Transform each node once per relation, not an E x 784 expansion.
                transformed = features @ self.relation_weight[role]
                aggregate = aggregate.index_add(
                    0, target[selected], transformed[source[selected]])
        degree = torch.bincount(target, minlength=context.num_pairs).to(features.dtype)
        aggregate = aggregate / degree.clamp_min(1).unsqueeze(1)
        hidden = torch.relu(self.self_projection(features) + aggregate)
        return self.head(hidden)


class PairRoleResidualGenerator(nn.Module):
    """Frozen G/shadow anchor plus trainable endpoint-role residual.

    ``selected_coordinates`` MUST be passed in the original importance-ranking
    order; it is never sorted or recomputed. In the existing baseline this is
    exactly 30 coordinates of the normalized 784-dimensional pair representation.
    Smaller dimensions are supported for synthetic contract tests only.

    Owned deep copies protect the caller's modules. Both base modules stay in eval
    with no parameter/input gradients through the base path. A future trainable
    surrogate must be separate from this frozen conditioning shadow. Only the new
    branch is trainable; use ``residual.parameters()`` for its optimizer.
    """

    def __init__(self, base_generator: nn.Module, base_shadow: nn.Module,
                 selected_coordinates: torch.Tensor, *, input_dim: int = 784,
                 hidden_dim: int = 80, mode: str = "typed", threshold: float = .5):
        super().__init__()
        if not isinstance(base_generator, nn.Module) or not isinstance(base_shadow, nn.Module):
            raise ValueError("Base generator and shadow must be torch modules")
        if not callable(getattr(base_shadow, "get_h", None)):
            raise ValueError("Base shadow must expose get_h(features, edge_index)")
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if (not isinstance(selected_coordinates, torch.Tensor)
                or selected_coordinates.dtype != torch.long
                or selected_coordinates.ndim != 1 or selected_coordinates.numel() == 0):
            raise ValueError("Selected coordinates must be a nonempty torch.long vector")
        if (bool((selected_coordinates < 0).any())
                or bool((selected_coordinates >= input_dim).any())
                or selected_coordinates.unique().numel() != selected_coordinates.numel()):
            raise ValueError("Selected coordinates must be unique and in range")
        if not math.isfinite(float(threshold)):
            raise ValueError("Nonfinite threshold")
        self.input_dim = input_dim
        self.threshold = float(threshold)
        self.base_generator = deepcopy(base_generator)
        self.base_shadow = deepcopy(base_shadow)
        self.register_buffer("selected_coordinates", selected_coordinates.detach().clone())
        self.residual = RoleResidualBlock(input_dim, hidden_dim,
                                          selected_coordinates.numel(), mode)
        self._freeze_base()

    def _freeze_base(self):
        for module in (self.base_generator, self.base_shadow):
            module.requires_grad_(False)
            module.eval()
            for parameter in module.parameters():
                parameter.grad = None

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_base()
        return self

    def _validate_targets(self, features, target_indices):
        if (not isinstance(target_indices, torch.Tensor)
                or target_indices.dtype != torch.long or target_indices.ndim != 1
                or target_indices.device != features.device):
            raise ValueError("Targets must be a same-device torch.long vector")
        if (bool((target_indices < 0).any()) or bool((target_indices >= len(features)).any())
                or target_indices.unique().numel() != target_indices.numel()):
            raise ValueError("Target indices must be unique and in range")
        if self.selected_coordinates.device != features.device:
            raise ValueError("Selected coordinates and features must be on the same device")

    def forward_components(self, features: torch.Tensor, context: RoleContext,
                           target_indices: torch.Tensor) -> GeneratedFeatures:
        _check_features(features, context, self.input_dim)
        self._validate_targets(features, target_indices)
        # Defensive against a caller manually putting one frozen child in train.
        self.base_generator.eval()
        self.base_shadow.eval()
        with torch.no_grad():
            embedding = self.base_shadow.get_h(features, context.edge_index)
            baseline = self.base_generator(embedding[target_indices], self.threshold)
        shape = (len(target_indices), len(self.selected_coordinates))
        if (not isinstance(baseline, torch.Tensor) or baseline.shape != shape
                or baseline.device != features.device or baseline.dtype != features.dtype
                or not bool(torch.isfinite(baseline).all())):
            raise ValueError("Frozen generator output violates the feature contract")
        residual = self.residual(features, context)[target_indices]
        values = baseline + residual
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite residual generator output")
        return GeneratedFeatures(values=values, baseline=baseline, residual=residual)

    def forward(self, features: torch.Tensor, context: RoleContext,
                target_indices: torch.Tensor) -> torch.Tensor:
        return self.forward_components(features, context, target_indices).values

    def inject(self, features: torch.Tensor, context: RoleContext,
               target_indices: torch.Tensor, edge_weight: torch.Tensor | None = None,
               *, detach: bool = False) -> InjectedFeatures:
        """Replace only selected coordinates; preserve all rows/edges/weights else.

        The baseline line graph is unweighted. Reject nonunit weights rather than
        silently treating a weighted graph as the unweighted ``get_h`` contract.
        ``detach=False`` retains the new branch gradient for later training;
        inference callers can explicitly detach or use torch.no_grad().
        """
        if type(detach) is not bool:
            raise ValueError("detach must be bool")
        _check_features(features, context, self.input_dim)
        if edge_weight is None:
            edge_weight = features.new_ones(context.edge_index.shape[1])
        if (not isinstance(edge_weight, torch.Tensor) or not edge_weight.is_floating_point()
                or edge_weight.shape != (context.edge_index.shape[1],)
                or edge_weight.device != features.device
                or not bool(torch.isfinite(edge_weight).all())
                or not bool((edge_weight == 1).all())):
            raise ValueError("Expected one finite unit weight per original line edge")
        generated = self(features, context, target_indices)
        output = features.clone()
        output[target_indices[:, None], self.selected_coordinates] = (
            generated.detach() if detach else generated)
        return InjectedFeatures(output, context.edge_index.clone(), edge_weight.clone())
