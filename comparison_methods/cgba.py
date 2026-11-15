"""Paper-guided CGBA (Xia et al., AAAI 2025) continuous-feature adaptation.

This is NOT author code or an exact reproduction of their reported numbers.
Source: https://ojs.aaai.org/index.php/AAAI/article/download/35466/37621
Implements Eq. (3) selection, GraphLIME mean importance in Eq. (4), and
Eq. (5)--(9) shared feature generation. Labels and topology never change.
Unspecified numerical choices and the interpretation of ambiguous notation
are recorded explicitly. No dataset IO, evaluation, or victim training here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from collections.abc import Mapping
from time import perf_counter

import numpy as np
import torch
from sklearn.linear_model import LassoLars
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import k_hop_subgraph, to_undirected

from spear_current.models import GCN


PAPER_URL = 'https://ojs.aaai.org/index.php/AAAI/article/download/35466/37621'
GRAPHLIME_URL = 'https://yichang-cs.com/jlu/TKDE23_GraphLIME.pdf'


@dataclass(frozen=True)
class CGBAConfig:
    epochs: int = 200
    hidden: int = 80
    poison_count: int = 26
    feature_count: int = 24
    lr: float = 0.01
    weight_decay: float = 5e-4
    target: int = 1
    select_pretrain_epochs: int = 200
    dropout: float = 0.5
    degree_beta: float = 0.01
    # The CGBA paper does not specify these values. They are adaptation choices.
    lambda_self: float = 1.0
    lambda_neighbor: float = 1.0
    self_threshold: float = 0.5
    neighbor_threshold: float = 0.5
    graphlime_hops: int = 2
    graphlime_rho: float = 0.1
    graphlime_sigma_x: float = 1.0
    graphlime_sigma_y: float = 1.0
    graphlime_max_iter: int = 1000
    # Fail instead of silently sampling or substituting another explainer.
    graphlime_max_entries: int = 20_000_000
    allowed_feature_indices: tuple[int, ...] | None = None


DEFAULT_CONFIG = asdict(CGBAConfig())


def _config(value):
    if value is None:
        cfg = CGBAConfig()
    elif isinstance(value, CGBAConfig):
        cfg = value
    else:
        values = dict(value) if isinstance(value, Mapping) else vars(value).copy()
        unknown = set(values) - {f.name for f in fields(CGBAConfig)}
        if unknown:
            raise ValueError(f'Unknown CGBA configuration fields: {sorted(unknown)}')
        cfg = CGBAConfig(**values)
    for name in ('epochs', 'hidden', 'poison_count', 'feature_count',
                 'select_pretrain_epochs', 'graphlime_hops', 'graphlime_max_iter',
                 'graphlime_max_entries'):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('lr', 'graphlime_sigma_x', 'graphlime_sigma_y'):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f'Invalid {name}')
    for name in ('weight_decay', 'degree_beta', 'lambda_self', 'lambda_neighbor', 'graphlime_rho'):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f'Invalid {name}')
    if (not 0 <= cfg.dropout < 1 or isinstance(cfg.target, bool)
            or not isinstance(cfg.target, int) or cfg.target not in (0, 1)):
        raise ValueError('Invalid dropout or binary target')
    for name in ('self_threshold', 'neighbor_threshold'):
        if not math.isfinite(getattr(cfg, name)) or not -1 <= getattr(cfg, name) <= 1:
            raise ValueError(f'Invalid cosine threshold: {name}')
    if cfg.allowed_feature_indices is not None:
        selected = cfg.allowed_feature_indices
        if (not isinstance(selected, (list, tuple)) or not selected
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('allowed_feature_indices must be unique nonnegative integers')
        if cfg.feature_count > len(selected):
            raise ValueError('Feature budget exceeds explicitly eligible domain')
    return cfg


def validate_config(config=None):
    """Expand and validate a configuration without graph/model/data access.

    Returns a detached JSON-round-trippable dict for registration. Dimension-
    dependent bounds are additionally checked by fit_cgba. Defaults target 1.
    """
    try:
        if config is not None and not isinstance(config, (CGBAConfig, Mapping)):
            raise ValueError('CGBA configuration must be a mapping or CGBAConfig')
        cfg = _config(config)
        return json.loads(json.dumps(asdict(cfg), allow_nan=False))
    except TypeError as exc:
        raise ValueError('CGBA configuration must contain JSON-safe values') from exc


def _indices(value, size, device, name, *, allow_empty=False):
    value = torch.as_tensor(value, device=device)
    if value.ndim != 1 or value.dtype != torch.long:
        raise ValueError(f'{name} must be a one-dimensional int64 index')
    if (not allow_empty and value.numel() == 0) or value.unique().numel() != value.numel():
        raise ValueError(f'{name} is empty or contains duplicates')
    if bool(((value < 0) | (value >= size)).any()):
        raise ValueError(f'{name} is out of range')
    return value.detach().clone()


def _digest(value):
    array = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def select_attachment_indices(probability, degree, candidates, count, beta=0.01):
    """Eq. (3): low target probability AND low original degree, stable ID ties."""
    candidates = candidates.sort().values
    scores = -(probability[candidates] + beta * degree[candidates])
    order = torch.argsort(scores, descending=True, stable=True)
    return candidates[order[:count]], scores


def _center_normalize(kernel):
    centered = kernel - kernel.mean(axis=0, keepdims=True)
    centered -= centered.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=(0, 1), keepdims=True)
    return np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-12)


def graphlime_coefficients(features, probabilities, cfg):
    """GraphLIME Eq. (2), Gaussian kernels, nonnegative HSIC Lasso.

    No saliency-gradient substitution. LassoLars divides residual loss by n^2;
    alpha=rho/n^2 makes its minimizer match the paper's unaveraged objective.
    Gaussian bandwidths are explicit adaptation parameters, not fitted on eval.
    """
    x = np.asarray(features, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    n, d = x.shape
    if n * n * d > cfg.graphlime_max_entries:
        raise ValueError('GraphLIME neighborhood exceeds declared memory limit; no fallback')
    if n < 2:
        return np.zeros(d, dtype=np.float64)
    k = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * cfg.graphlime_sigma_x ** 2))
    l = np.exp(-np.sum((p[:, None] - p[None, :]) ** 2, axis=-1,
                      keepdims=True) / (2 * cfg.graphlime_sigma_y ** 2))
    design = _center_normalize(k).reshape(n * n, d)
    response = _center_normalize(l).reshape(n * n)
    if not np.any(response):
        return np.zeros(d, dtype=np.float64)
    # Explicitly disable the deprecated pre-1.2 sklearn normalization if present.
    import inspect
    options = dict(alpha=cfg.graphlime_rho / (n * n), fit_intercept=False,
                   positive=True, max_iter=cfg.graphlime_max_iter)
    if 'normalize' in inspect.signature(LassoLars).parameters:
        options['normalize'] = False
    solver = LassoLars(**options).fit(design, response)
    coefficients = solver.coef_
    if not np.isfinite(coefficients).all() or (coefficients < -1e-10).any():
        raise ValueError('Invalid GraphLIME coefficients')
    return np.maximum(coefficients, 0)


class CGBAFixedFeatureTrigger(nn.Module):
    """Frozen learned shared vector; no per-target generator or topology edits."""

    def __init__(self, dimensions, hidden, feature_indices, *, device, dtype=torch.float32):
        super().__init__()
        self.dimensions = dimensions
        self.mlp = nn.Sequential(nn.Linear(dimensions, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dimensions)).to(device=device, dtype=dtype)
        self.register_buffer('feature_indices', feature_indices.detach().clone())
        self.register_buffer('trigger', torch.zeros(dimensions, device=device, dtype=dtype))
        self.register_buffer('is_fitted', torch.tensor(False, device=device))

    def _write(self, x, targets, trigger):
        out = x.clone()
        out[targets[:, None], self.feature_indices[None, :]] = trigger[self.feature_indices]
        return out

    def inject(self, x, edge_index, targets):
        if not bool(self.is_fitted):
            raise RuntimeError('CGBA trigger is not fitted')
        if x.ndim != 2 or x.shape[1] != self.dimensions or x.device != self.trigger.device:
            raise ValueError('CGBA feature shape/device mismatch')
        targets = _indices(targets, len(x), x.device, 'targets', allow_empty=True)
        out = self._write(x, targets, self.trigger)
        return out, edge_index.clone(), x.new_ones(edge_index.shape[1])


def concealment_losses(poisoned_x, original_x, attachments, neighbor_pairs, cfg):
    """Eq. (6),(7): literal SUM threshold-minus-cosine, NOT a hinge loss.

    Printed sim(x-y) is interpreted as cosine(x,y), as specified in the prose.
    Neighbor references are original clean feature vectors, including when a
    neighbor is also poisoned. Empty neighbor sets contribute differentiable 0.
    """
    self_loss = (cfg.self_threshold - F.cosine_similarity(
        poisoned_x[attachments], original_x[attachments], dim=1)).sum()
    src, dst = neighbor_pairs
    neighbor_loss = (cfg.neighbor_threshold - F.cosine_similarity(
        poisoned_x[src], original_x[dst], dim=1)).sum() if len(src) else poisoned_x.sum() * 0
    return self_loss, neighbor_loss


def fit_cgba(x, edge_index, y, idx_train, candidate_indices, *, seed, config, progress=None):
    """Fit on caller-supplied TRAIN graph only; no access to DEV/audit/test.

    ``candidate_indices`` must be supervised target-class training nodes. The
    clean-label method never changes y, and fit_indices remain idx_train. All
    unspecified paper choices are frozen in CGBAConfig and emitted in record.
    """
    cfg = _config(config)
    if x.ndim != 2 or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError('Expected finite floating point node features')
    if y.shape != (len(x),) or y.dtype != torch.long or y.device != x.device:
        raise ValueError('Expected aligned int64 labels')
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise ValueError('Expected int64 edge_index[2,E]')
    if edge_index.device != x.device or bool(((edge_index < 0) | (edge_index >= len(x))).any()):
        raise ValueError('Invalid edge index or device')
    train = _indices(idx_train, len(x), x.device, 'idx_train')
    candidates = _indices(candidate_indices, len(x), x.device, 'candidate_indices')
    if not bool(torch.isin(candidates, train).all()) or not bool((y[candidates] == cfg.target).all()):
        raise ValueError('CGBA candidates must be supervised TRAIN target-class nodes; no relabeling')
    if not bool(((y[train] == 0) | (y[train] == 1)).all()):
        raise ValueError('Binary supervised labels required')
    if len(candidates) < cfg.poison_count:
        raise ValueError('Not enough target-class candidates for declared poison count')
    allowed = (torch.arange(x.shape[1], device=x.device) if cfg.allowed_feature_indices is None
               else _indices(cfg.allowed_feature_indices, x.shape[1], x.device, 'allowed_feature_indices'))
    if cfg.feature_count > len(allowed):
        raise ValueError('Feature budget exceeds eligible domain')
    clean = x.detach().clone()
    edges = edge_index.detach().clone()
    labels = y.detach().clone()
    # Statistics use simple undirected neighbors; the supplied graph is unchanged.
    neighbors = to_undirected(edges[:, edges[0] != edges[1]], num_nodes=len(x))
    degree = torch.bincount(neighbors[0], minlength=len(x)).to(x.dtype)
    cuda_devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if x.is_cuda:
            with torch.cuda.device(x.device):
                torch.cuda.manual_seed(seed)
        proxy = GCN(x.shape[1], cfg.hidden, 2, dropout=cfg.dropout,
                    device=x.device).to(device=x.device, dtype=x.dtype)
        optimizer = torch.optim.Adam(proxy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pretrain_history = []
        surrogate_started = perf_counter()
        for epoch in range(cfg.select_pretrain_epochs):
            proxy.train()
            optimizer.zero_grad()
            loss = F.nll_loss(proxy(clean, edges)[train], labels[train])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite CGBA surrogate loss')
            loss.backward()
            optimizer.step()
            pretrain_history.append(float(loss.detach()))
            if progress:
                progress(dict(stage='cgba_surrogate', epoch=epoch + 1, loss=float(loss.detach())))
        proxy.eval().requires_grad_(False)
        proxy.zero_grad(set_to_none=True)
        surrogate_seconds = perf_counter() - surrogate_started
        selection_started = perf_counter()
        with torch.no_grad():
            probability = proxy(clean, edges).exp()
        attachments, scores = select_attachment_indices(
            probability[:, cfg.target], degree, candidates, cfg.poison_count, cfg.degree_beta)
        coefficients, neighborhood_sizes = [], []
        for index in attachments.tolist():
            subset, _, _, _ = k_hop_subgraph(index, cfg.graphlime_hops, neighbors,
                                            num_nodes=len(x), flow='source_to_target')
            coefficients.append(graphlime_coefficients(clean[subset].cpu().numpy(),
                                                       probability[subset].cpu().numpy(), cfg))
            neighborhood_sizes.append(len(subset))
            if progress:
                progress(dict(stage='cgba_graphlime', completed=len(coefficients),
                              total=len(attachments), neighborhood_size=len(subset)))
        aif = torch.tensor(np.mean(coefficients, axis=0), device=x.device, dtype=x.dtype)
        allowed = allowed.sort().values
        chosen = allowed[torch.argsort(aif[allowed], descending=True, stable=True)[:cfg.feature_count]]
        selection_seconds = perf_counter() - selection_started
        generator = CGBAFixedFeatureTrigger(x.shape[1], cfg.hidden, chosen, device=x.device, dtype=x.dtype)
        optimizer = torch.optim.Adam(generator.mlp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        previous = torch.rand(x.shape[1], device=x.device, dtype=x.dtype)
        neighbor_pairs = neighbors[:, torch.isin(neighbors[0], attachments)]
        history = []
        generator_started = perf_counter()
        for epoch in range(cfg.epochs):
            optimizer.zero_grad()
            trigger = generator.mlp(previous)
            poisoned = generator._write(clean, attachments, trigger)
            classification = F.nll_loss(proxy(poisoned, edges)[train], labels[train], reduction='sum')
            self_loss, neighbor_loss = concealment_losses(poisoned, clean, attachments, neighbor_pairs, cfg)
            total = classification + cfg.lambda_self * self_loss + cfg.lambda_neighbor * neighbor_loss
            if not bool(torch.isfinite(total)) or not bool(torch.isfinite(trigger).all()):
                raise ValueError('Nonfinite CGBA trigger or loss; no clipping/fallback applied')
            total.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in generator.parameters()):
                raise ValueError('Nonfinite CGBA generator gradient')
            optimizer.step()
            previous = trigger.detach()
            history.append(dict(epoch=epoch + 1, classification_sum=float(classification.detach()),
                                self_sum=float(self_loss.detach()), neighbor_sum=float(neighbor_loss.detach()),
                                total=float(total.detach())))
            if progress:
                progress(dict(stage='cgba_trigger', **history[-1]))
        generator.trigger.copy_(previous)
        generator.is_fitted.fill_(True)
        generator.eval().requires_grad_(False)
        generator.zero_grad(set_to_none=True)
        generator_seconds = perf_counter() - generator_started
        with torch.no_grad():
            poisoned_x, poisoned_edges, poisoned_weights = generator.inject(clean, edges, attachments)
    record = dict(
        method='CGBA_AAAI2025_paper_guided_reimplementation', author_code=False,
        exact_author_reproduction=False, paper_url=PAPER_URL, graphlime_paper_url=GRAPHLIME_URL,
        seed=int(seed), config=asdict(cfg), clean_label=True, topology_changed=False,
        feature_domain='all_input_coordinates' if cfg.allowed_feature_indices is None else 'explicit_restricted_domain',
        feature_indices=chosen.tolist(), attachment_indices=attachments.tolist(), fit_indices=train.tolist(),
        candidate_indices=candidates.sort().values.tolist(), candidate_scores=scores.tolist(),
        aif=aif.tolist(), aif_positive_count=int((aif > 0).sum()),
        graphlime_neighborhood_sizes=neighborhood_sizes, pretrain_loss=pretrain_history, history=history,
        elapsed_seconds=dict(clean_surrogate=surrogate_seconds, selection_and_graphlime=selection_seconds,
                             generator=generator_seconds),
        input_sha256=dict(x=_digest(clean), edge_index=_digest(edges), y=_digest(labels)),
        choices=dict(selector='negative_target_probability_plus_beta_unique_undirected_degree',
                     selector_ties='ascending_node_id', feature_ties='ascending_coordinate',
                     feature_explainer='GraphLIME_Gaussian_centered_normalized_nonnegative_HSIC_Lasso',
                     graphlime_output='full_clean_proxy_probability_vector',
                     classification_reduction='sum_over_all_supervised_train_nodes',
                     proxy='two_layer_GCN_clean_train_only_last_epoch_frozen_during_trigger_training',
                     mlp='input_dimension_hidden_ReLU_input_dimension',
                     initial_trigger='uniform_0_1', recurrent_input='previous_output_detached_each_step',
                     final_trigger='last_training_forward_before_last_optimizer_step_cached',
                     similarity='cosine_of_two_feature_vectors_interpreting_printed_difference_notation',
                     concealment='literal_threshold_minus_cosine_sum_no_hinge',
                     neighbor_reference='original_clean_features_unique_undirected_neighbors_no_self',
                     continuous_output='unbounded_no_binary_threshold_or_extra_domain_projection',
                     checkpoint='fixed_last_epoch_no_validation_selection'),
        paper_unspecified_choices=['lambda_self', 'lambda_neighbor', 'self_threshold', 'neighbor_threshold',
                                   'GraphLIME bandwidth/rho/hops', 'optimizer/learning_rate/weight_decay',
                                   'MLP hidden/activation/initialization', 'recursive_detachment/final_cache'],
        graph_scope='caller_supplied_training_graph_must_be_prevalidated',
        evaluation_accessed=False,
    )
    return dict(generator=generator, attachment_indices=attachments,
                poisoned_x=poisoned_x, poisoned_edge_index=poisoned_edges,
                poisoned_edge_weight=poisoned_weights, poisoned_y=labels,
                fit_indices=train, record=record)
