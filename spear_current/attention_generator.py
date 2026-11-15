# Source: experiments/spear_content_attention_v1/module.py; mechanically migrated; algorithms and state keys unchanged.
"""Content weighting of an existing, trained SPEAR untyped residual.

This local feature-space experiment reuses only PyG 2.8.0 GATv2's score rule:
``a^T LeakyReLU(W_source x_source + W_target x_target)``, followed by
per-destination softmax. It is not a complete GATv2 convolution: the existing
value kernel, self branch, and output head are retained without another output
branch, edge changes, self loops, or additional message-passing hops.

Reference: Brody et al., "How Attentive are Graph Attention Networks?",
ICLR 2022, https://arxiv.org/abs/2105.14491. The implementation calls the
installed PyG GATv2Conv scoring path; it does not vendor upstream source.

The zero score vector starts with uniform weights. Aggregation is written as
the original mean plus a zero-initialized weighting correction, preserving the
old arithmetic for checkpoint replay. This guarantees initialization identity,
not preservation of ASR after any future optimization. No training or data I/O
lives here. Frozen experiment files and the caller's anchor remain untouched.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from .role_generator import (
    PairRoleResidualGenerator, RoleResidualBlock, _check_features,
)
from .attention_context import AttentionContext


class ContentAttentionResidual(RoleResidualBlock):
    """Keep the old untyped block; change only its neighbor weights.

    All modes own the same scorer parameters. ``uniform`` evaluates the scorer
    but multiplies its correction by zero, so it is an equal-budget control and
    not an equal-effective-capacity claim. ``shuffled`` uses a prebound incoming
    score permutation, retaining gradients and each target's weight multiset.
    Endpoint-role values are not features and never enter the scoring equation.
    """

    def __init__(self, anchor: RoleResidualBlock, *,
                 attention_mode: str = 'learned', score_dim: int = 32):
        if type(anchor) is not RoleResidualBlock or anchor.mode != 'untyped':
            raise ValueError('Require an original untyped RoleResidualBlock anchor')
        if attention_mode not in ('uniform', 'learned', 'shuffled'):
            raise ValueError('Unknown content attention mode')
        if type(score_dim) is not int or score_dim <= 0:
            raise ValueError('score_dim must be a positive integer')
        if any(not bool(torch.isfinite(p).all()) for p in anchor.parameters()):
            raise ValueError('Nonfinite anchor parameters')
        parameter_contracts = {(p.device, p.dtype) for p in anchor.parameters()}
        if len(parameter_contracts) != 1:
            raise ValueError('Anchor parameters must share device and dtype')

        # Do not call the old constructor: preserve trained tensors and avoid
        # consuming RNG for random value/self/head parameters that are discarded.
        nn.Module.__init__(self)
        copied = deepcopy(anchor)
        self.input_dim = copied.input_dim
        self.hidden_dim = copied.hidden_dim
        self.output_dim = copied.output_dim
        self.mode = 'untyped'  # Existing injector/training interface contract.
        self.attention_mode = attention_mode
        self.score_dim = score_dim
        self.relation_weight = copied.relation_weight
        self.self_projection = copied.self_projection
        self.head = copied.head
        self.scorer = GATv2Conv(
            self.input_dim, score_dim, heads=1, concat=False,
            negative_slope=0.2, dropout=0.0, add_self_loops=False,
            edge_dim=None, bias=False, share_weights=False, residual=False,
        ).to(device=self.relation_weight.device, dtype=self.relation_weight.dtype)
        nn.init.zeros_(self.scorer.att)
        self.requires_grad_(True)
        for parameter in self.parameters():
            parameter.grad = None

    def _validate(self, features: torch.Tensor, context: AttentionContext):
        if not isinstance(context, AttentionContext):
            raise ValueError('Require an explicit AttentionContext sidecar')
        _check_features(features, context, self.input_dim)
        if (features.device != self.relation_weight.device
                or features.dtype != self.relation_weight.dtype):
            raise ValueError('Residual parameters and features must share device/dtype')

    def attention_weights(self, features: torch.Tensor,
                          context: AttentionContext) -> torch.Tensor:
        """Return effective weights [E] in the unchanged original edge order.

        Scores use original pair features, not shadow embeddings or aggregated
        messages. Empty edges yield an empty tensor; isolated pairs still retain
        the old self branch. No self loops are silently inserted.
        """
        self._validate(features, context)
        source_tokens = self.scorer.lin_l(features).view(
            context.num_pairs, 1, self.score_dim)
        target_tokens = self.scorer.lin_r(features).view(
            context.num_pairs, 1, self.score_dim)
        if (not bool(torch.isfinite(source_tokens).all())
                or not bool(torch.isfinite(target_tokens).all())):
            raise ValueError('Nonfinite projected attention features')
        alpha = self.scorer.edge_updater(
            context.edge_index, x=(source_tokens, target_tokens), edge_attr=None,
            size=(context.num_pairs, context.num_pairs),
        ).reshape(-1)
        if (alpha.shape != (context.edge_index.shape[1],)
                or not bool(torch.isfinite(alpha).all())
                or bool(((alpha < 0) | (alpha > 1)).any())):
            raise ValueError('Invalid per-edge attention weights')
        if self.attention_mode == 'shuffled':
            return alpha[context.score_permutation]
        if self.attention_mode == 'uniform':
            target = context.edge_index[1]
            degree = torch.bincount(target, minlength=context.num_pairs).to(features.dtype)
            uniform = torch.ones_like(alpha) / degree[target].clamp_min(1)
            # Retain the scoring graph with exactly zero derivatives so its
            # computation and registered parameter budget exist in this control.
            return uniform + alpha * 0
        return alpha

    def forward(self, features: torch.Tensor, context: AttentionContext):
        self._validate(features, context)
        source, target = context.edge_index
        transformed = features @ self.relation_weight.mean(dim=0)
        aggregate = features.new_zeros((context.num_pairs, self.hidden_dim))
        aggregate = aggregate.index_add(0, target, transformed[source])
        degree = torch.bincount(target, minlength=context.num_pairs).to(features.dtype)
        aggregate = aggregate / degree.clamp_min(1).unsqueeze(1)

        alpha = self.attention_weights(features, context)
        uniform = torch.ones_like(alpha) / degree[target].clamp_min(1)
        correction = features.new_zeros((context.num_pairs, self.hidden_dim))
        correction = correction.index_add(
            0, target, (alpha - uniform).unsqueeze(1) * transformed[source])
        hidden = torch.relu(self.self_projection(features) + (aggregate + correction))
        result = self.head(hidden)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite content-attention residual output')
        return result


class ContentAttentionGenerator(PairRoleResidualGenerator):
    """Deep-copy a completed untyped generator and replace only aggregation.

    The original base G and conditioning shadow remain frozen/eval, while the
    inherited trained residual and new score parameters can be continued under
    an explicitly equal-budget training contract. No optimizer is created here.
    """

    def __init__(self, anchor: PairRoleResidualGenerator, *,
                 attention_mode: str = 'learned', score_dim: int = 32):
        if (type(anchor) is not PairRoleResidualGenerator
                or type(anchor.residual) is not RoleResidualBlock
                or anchor.residual.mode != 'untyped'):
            raise ValueError('Require a completed original untyped generator anchor')
        coordinates = anchor.selected_coordinates
        if (not isinstance(coordinates, torch.Tensor)
                or coordinates.dtype != torch.long or coordinates.ndim != 1
                or coordinates.numel() == 0
                or coordinates.unique().numel() != coordinates.numel()
                or bool(((coordinates < 0) | (coordinates >= anchor.input_dim)).any())):
            raise ValueError('Invalid anchor selected coordinates')
        if (anchor.residual.input_dim != anchor.input_dim
                or anchor.residual.output_dim != len(coordinates)
                or not math.isfinite(float(anchor.threshold))):
            raise ValueError('Anchor generator and residual contracts disagree')
        if any(not bool(torch.isfinite(p).all()) for p in anchor.parameters()):
            raise ValueError('Nonfinite generator anchor parameters')

        nn.Module.__init__(self)
        self.input_dim = anchor.input_dim
        self.threshold = float(anchor.threshold)
        self.attention_mode = attention_mode
        self.score_dim = score_dim
        self.base_generator = deepcopy(anchor.base_generator)
        self.base_shadow = deepcopy(anchor.base_shadow)
        self.register_buffer('selected_coordinates', coordinates.detach().clone())
        self.residual = ContentAttentionResidual(
            anchor.residual, attention_mode=attention_mode, score_dim=score_dim)
        self._freeze_base()

    def attention_weights(self, features: torch.Tensor,
                          context: AttentionContext) -> torch.Tensor:
        return self.residual.attention_weights(features, context)
