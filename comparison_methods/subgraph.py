"""Fresh, topology-injecting node-classification comparison methods.

This is an AV2 adaptation, NOT a physical-trajectory attack or a feature-only
implementation. Existing feature rows and edges are retained; generated nodes
and undirected edges are appended. The caller supplies TRAIN-only tensors.

UGBA and the GTA *node/data-poisoning adaptation* were checked against
https://github.com/ventr1c/UGBA at
c1d945a215dd16fd806f8a231381d303a0b811f7, specifically models/backdoor.py,
models/GTA.py, heuristic_selection.py, and models/GCN_Encoder.py. GTA here is
not the original USENIX 2021 pretrained-model-trojaning threat model.
Paper: https://arxiv.org/abs/2303.01263; https://arxiv.org/abs/2006.11890.
This independent implementation records adaptations explicitly. No author
license grant is implied. No checkpoints, graphs, or external files are read.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans

from spear_current.models import GCN


AUTHOR_SHA = "c1d945a215dd16fd806f8a231381d303a0b811f7"
METHODS = ("ugba", "dpgba", "gta")
SUPPORTED_METHODS = ("ugba", "gta")
DEFAULT_CONFIG = {
    "epochs": 200, "inner": 1, "hidden": 80, "poison_count": 26,
    "trigger_size": 3, "lr": 0.01, "weight_decay": 5e-4,
    "target_class": 1, "select_pretrain_epochs": 200,
    "outer_count": 512, "edge_threshold": 0.5,
    "homo_weight": 100.0, "homo_threshold": 0.8,
    "selection_distance_weight": 1.0, "selector_dropout": 0.5,
    "selector_kmeans_seed": 1, "selector_kmeans_n_init": 10,
    "target_weight": 1.0, "generator_checkpoint": "final",
}


def validate_config(config=None):
    """Return a complete JSON-safe config; reject unknown or invalid options.

Prepare callers can pass a partial mapping; run plans should persist the
returned complete mapping so all defaults are preregistered.
"""
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("config must contain only documented keys")
    result = {**DEFAULT_CONFIG, **deepcopy(config)}
    positive_int = ("epochs", "inner", "hidden", "poison_count", "trigger_size",
                    "select_pretrain_epochs", "selector_kmeans_n_init")
    for key in positive_int:
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("outer_count", "selector_kmeans_seed"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if result["selector_kmeans_seed"] >= 2 ** 32:
        raise ValueError("selector_kmeans_seed must fit uint32")
    if type(result["target_class"]) is not int or result["target_class"] not in (0, 1):
        raise ValueError("this adapter supports binary target_class 0 or 1")
    for key in ("lr", "weight_decay", "edge_threshold", "homo_weight",
                "homo_threshold", "selection_distance_weight", "selector_dropout",
                "target_weight"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
        result[key] = float(value)
    if result["lr"] <= 0 or result["target_weight"] <= 0:
        raise ValueError("lr and target_weight must be positive")
    if any(result[k] < 0 for k in ("weight_decay", "homo_weight", "selection_distance_weight")):
        raise ValueError("loss/regularization coefficients cannot be negative")
    if not -1 <= result["homo_threshold"] <= 1 or not 0 <= result["selector_dropout"] < 1:
        raise ValueError("invalid similarity threshold or dropout")
    if result["generator_checkpoint"] != "final":
        raise ValueError("only preregistered final generator checkpoint is supported")
    return result


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(_tensor_hash(value).encode())
    return h.hexdigest()


def _indices(value, n, name, *, empty=False):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional long tensor")
    if value.device.type != "cpu" or (not empty and not value.numel()):
        raise ValueError(f"{name} must be nonempty CPU indices")
    if value.numel() and (value.min() < 0 or value.max() >= n):
        raise ValueError(f"{name} outside original-node prefix")
    if value.unique().numel() != value.numel():
        raise ValueError(f"{name} contains duplicates")
    return value.detach().clone()


def _graph(x, edge_index):
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("x must be nonempty N by F")
    if x.device.type != "cpu" or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("this registered adapter requires finite CPU float32 x")
    if not isinstance(edge_index, torch.Tensor) or edge_index.dtype != torch.long:
        raise ValueError("edge_index must be long")
    if edge_index.device.type != "cpu" or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be CPU 2 by E")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= len(x)):
        raise ValueError("edge_index outside original-node prefix")


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold):
        return (scores > threshold).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class SubgraphGenerator(nn.Module):
    """Author-style F→F→(sF, s(s−1)/2) generator with discrete edges.

    ``inject`` is inference-only and REMOVES zero-weight edges, because the
    project's victim GraphSAGE ignores weights. ``_inject_training`` retains
    straight-through zero weights for the weighted GCN surrogate.
"""
    def __init__(self, n_features, trigger_size=3, edge_threshold=0.5):
        super().__init__()
        if type(n_features) is not int or n_features < 1:
            raise ValueError("n_features must be positive")
        if type(trigger_size) is not int or trigger_size < 1:
            raise ValueError("trigger_size must be positive")
        if not math.isfinite(edge_threshold):
            raise ValueError("edge_threshold must be finite")
        self.n_features = n_features
        self.trigger_size = trigger_size
        self.edge_threshold = float(edge_threshold)
        self.layers = nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU())
        self.feat = nn.Linear(n_features, trigger_size * n_features)
        self.edge = nn.Linear(n_features, trigger_size * (trigger_size - 1) // 2)

    def forward(self, target_features):
        h = self.layers(target_features)
        return self.feat(h), _StraightThrough.apply(self.edge(h), self.edge_threshold)

    def _inject_training(self, x, edge_index, targets):
        values, weights = self(x[targets])
        values = values.reshape(-1, self.n_features)
        weights = torch.cat((x.new_ones((len(targets), 1)), weights), dim=1).flatten()
        # Match the author's target-major edge order, followed by its reversal.
        edges = []
        for i, target in enumerate(targets.tolist()):
            start = len(x) + i * self.trigger_size
            edges.append((target, start))
            edges.extend((start + j, start + k)
                         for j in range(self.trigger_size) for k in range(j))
        directed = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
        injected_edges = torch.cat((directed, directed.flip(0)), dim=1)
        return (torch.cat((x, values)), torch.cat((edge_index, injected_edges), dim=1),
                torch.cat((x.new_ones(edge_index.shape[1]), weights, weights)))

    @torch.no_grad()
    def inject(self, x, edge_index, targets):
        _graph(x, edge_index)
        if x.shape[1] != self.n_features or next(self.parameters()).device.type != "cpu":
            raise ValueError("generator feature width/device mismatch")
        targets = _indices(targets, len(x), "targets", empty=True)
        if not len(targets):
            return x.clone(), edge_index.clone(), x.new_ones(edge_index.shape[1])
        values, edges, weights = self._inject_training(x, edge_index, targets)
        if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
            raise ValueError("nonfinite generated graph")
        keep = weights > 0
        return values.detach(), edges[:, keep].detach(), weights[keep].detach()


class _Selector(nn.Module):
    """UGBA release GCN_Encoder: two ReLU GCN layers plus a linear head."""
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.layers = nn.ModuleList((GCNConv(width, hidden), GCNConv(hidden, hidden)))
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def embedding(self, x, edges):
        for conv in self.layers:
            x = F.dropout(F.relu(conv(x, edges)), self.dropout, self.training)
        return x

    def forward(self, x, edges):
        return F.log_softmax(self.head(self.embedding(x, edges)), dim=1)


def _minmax(x):
    span = float(np.max(x) - np.min(x))
    return np.zeros_like(x) if span == 0 else (x - np.min(x)) / span


def _select(method, x, edges, y, train, candidates, seed, cfg, progress):
    if method == "gta":
        choice = np.arange(len(candidates))
        np.random.RandomState(seed).shuffle(choice)
        selected = candidates[torch.from_numpy(choice[:cfg["poison_count"]])]
        return selected, {"method": "uniform_without_replacement", "pretrain_steps": 0}
    model = _Selector(x.shape[1], cfg["hidden"], cfg["selector_dropout"])
    initial_hash = _state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for epoch in range(cfg["select_pretrain_epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.nll_loss(model(x, edges)[train], y[train])
        _step(loss, optimizer, model.parameters())
        _progress(progress, "selector", epoch, loss)
    model.eval()
    with torch.no_grad():
        embeddings = model.embedding(x, edges).numpy()
    seen = torch.cat((train, candidates)).numpy()
    clustering = KMeans(n_clusters=2, random_state=cfg["selector_kmeans_seed"],
                        n_init=cfg["selector_kmeans_n_init"]).fit(embeddings[seen])
    groups = clustering.predict(embeddings)
    # This deliberately retains the release's cluster-ID rule, rather than
    # pretending cluster IDs are class predictions. Its limitation is recorded.
    eligible = candidates.numpy()[groups[candidates.numpy()] != cfg["target_class"]]
    if len(eligible) < cfg["poison_count"]:
        raise ValueError("author cluster-degree selector has too few candidates; no fallback")
    distances = np.linalg.norm(embeddings[eligible] - clustering.cluster_centers_[groups[eligible]], axis=1)
    degree = torch.bincount(edges.flatten(), minlength=len(x)).numpy()[eligible].astype(float)
    scores = _minmax(distances) + cfg["selection_distance_weight"] * _minmax(degree)
    # Stable node-ID tie-breaking avoids the upstream unordered set iteration.
    order = np.lexsort((eligible, scores))[:cfg["poison_count"]]
    selected = torch.from_numpy(eligible[order].copy()).long()
    record = {"method": "author_cluster_degree_kmeans_release", "pretrain_steps": cfg["select_pretrain_epochs"],
              "initial_state_sha256": initial_hash, "final_state_sha256": _state_hash(model),
              "training_indices_sha256": _tensor_hash(train), "cluster_counts": np.bincount(groups, minlength=2).tolist(),
              "admissible_candidates": len(eligible), "uses_candidate_true_labels": False,
              "cluster_id_is_not_class_label": True, "validation_or_test_used": False}
    return selected, record


def _step(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise ValueError("nonfinite optimization loss")
    loss.backward()
    parameters = list(parameters)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("nonfinite optimization gradient")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("nonfinite updated parameter")


def _progress(callback, phase, epoch, loss):
    if callback is not None:
        callback({"phase": phase, "epoch": epoch, "loss": float(loss.detach())})


def fit_subgraph(method, x, edge_index, y, idx_train, candidate_indices, *, seed,
                 config, progress: Callable | None = None):
    """Fit from fresh initialization, using only caller-provided TRAIN data.

    Original labels outside idx_train are not used by selection or optimization.
    The returned label vector has the original N rows; only original-node
    ``fit_indices`` are supervised. Appended trigger nodes have no label.
"""
    if method not in METHODS:
        raise ValueError(f"unknown subgraph method: {method}")
    if method == "dpgba":
        raise NotImplementedError("DPGBA is deferred_not_run: core author-source audit incomplete; no substitute implementation")
    cfg = validate_config(config)
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be a uint32 integer")
    _graph(x, edge_index)
    if not isinstance(y, torch.Tensor) or y.dtype != torch.long or y.device.type != "cpu" or y.shape != (len(x),):
        raise ValueError("y must be original N-length CPU long labels")
    train = _indices(idx_train, len(x), "idx_train")
    candidates = _indices(candidate_indices, len(x), "candidate_indices")
    if torch.isin(train, candidates).any():
        raise ValueError("supervised and candidate indices must be disjoint")
    if set(y[train].tolist()) != {0, 1}:
        raise ValueError("idx_train must contain both binary classes")
    if len(candidates) < cfg["poison_count"] + (cfg["outer_count"] if method != "gta" else 0):
        raise ValueError("insufficient candidates for attachment and fixed outer budgets")
    # Never mutate or retain caller tensors; fork_rng restores the caller CPU RNG.
    hashes = {"x": _tensor_hash(x), "edges": _tensor_hash(edge_index), "y": _tensor_hash(y)}
    x, edge_index, y = x.detach().clone(), edge_index.detach().clone(), y.detach().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        selected, selection = _select(method, x, edge_index, y, train, candidates, seed, cfg, progress)
        # Independent fit stream prevents selector budget from shifting G/S init.
        torch.manual_seed(seed)
        surrogate = GCN(x.shape[1], cfg["hidden"], 2, dropout=0.0, device=torch.device("cpu"))
        generator = SubgraphGenerator(x.shape[1], cfg["trigger_size"], cfg["edge_threshold"])
        initial = {"generator": _state_hash(generator), "surrogate": _state_hash(surrogate),
                   "cpu_rng": _tensor_hash(torch.get_rng_state())}
        opt_s = torch.optim.Adam(surrogate.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        poison_y = y.clone()
        poison_y[selected] = cfg["target_class"]
        fit_indices = torch.cat((train, selected))
        remaining = candidates[~torch.isin(candidates, selected)]
        if method == "gta":
            outer = selected
        else:
            indices = np.random.RandomState(seed).choice(len(remaining), cfg["outer_count"], replace=False)
            outer = torch.cat((selected, remaining[torch.from_numpy(indices)]))
        counters = {"generator_steps": 0, "surrogate_steps": 0, "discriminator_steps": 0,
                    "selector_steps": selection["pretrain_steps"], "autoencoder_steps": 0}
        trace = []
        for epoch in range(cfg["epochs"]):
            generator.train()
            surrogate.train()
            for _ in range(cfg["inner"]):
                opt_s.zero_grad(set_to_none=True)
                opt_g.zero_grad(set_to_none=True)
                px, pe, pw = generator._inject_training(x, edge_index, selected)
                if method != "gta":
                    px, pw = px.detach(), pw.detach()
                output = surrogate(px, pe, pw)
                inner_loss = F.nll_loss(output[fit_indices], poison_y[fit_indices])
                parameters = list(surrogate.parameters()) + (list(generator.parameters()) if method == "gta" else [])
                _step(inner_loss, opt_s, parameters)
                counters["surrogate_steps"] += 1
                if method == "gta":
                    opt_g.step()
                    if any(not torch.isfinite(p).all() for p in generator.parameters()):
                        raise ValueError("nonfinite generator parameter")
                    counters["generator_steps"] += 1
            if method == "gta":
                raw_target, raw_homo, total = inner_loss, inner_loss.new_zeros(()), inner_loss
            else:
                generator.eval()
                surrogate.eval()
                for p in surrogate.parameters():
                    p.requires_grad_(False)
                    p.grad = None
                opt_g.zero_grad(set_to_none=True)
                px, pe, pw = generator._inject_training(x, edge_index, outer)
                output = surrogate(px, pe, pw)
                outer_fit = torch.cat((train, outer))
                outer_labels = torch.cat((y[train], y.new_full((len(outer),), cfg["target_class"])))
                raw_target = F.nll_loss(output[outer_fit], outer_labels)
                te, tw = pe[:, edge_index.shape[1]:], pw[edge_index.shape[1]:]
                active = te[:, tw > 0]
                similarity = F.cosine_similarity(px[active[0]], px[active[1]], dim=1)
                raw_homo = F.relu(cfg["homo_threshold"] - similarity).mean()
                total = cfg["target_weight"] * raw_target + cfg["homo_weight"] * raw_homo
                _step(total, opt_g, generator.parameters())
                counters["generator_steps"] += 1
                for p in surrogate.parameters():
                    p.requires_grad_(True)
            item = {"epoch": epoch, "inner_loss": float(inner_loss.detach()),
                    "target_loss_raw": float(raw_target.detach()), "homophily_loss_raw": float(raw_homo.detach()),
                    "generator_loss": float(total.detach())}
            trace.append(item)
            _progress(progress, "generator", epoch, total)
        generator.eval()
        for p in generator.parameters():
            p.requires_grad_(False)
            p.grad = None
        px, pe, pw = generator.inject(x, edge_index, selected)
        deviations = [
            "AV2 pair-as-node graph-space experiment; added nodes have no physical-trajectory claim",
            "common configured training budgets/hidden width; no validation/test checkpoint selection",
            "separate seed-reset fit stream; all data/selection confined to supplied TRAIN reservoir",
            "CPU float32 execution; no cached selected_nodes or pretrained checkpoints",
            "final generator epoch and explicit positive-edge filtering for unweighted GraphSAGE inference",
        ]
        if method == "ugba":
            deviations.extend([
                "fresh TRAIN-only selector with final epoch rather than validation-selected encoder",
                "KMeans release branch (not Cora KMedoids); raw cluster ID exclusion is not semantic class exclusion",
                "constant selector score ranges map to zero; deterministic node-ID tie-breaking",
                "fixed ordered outer draw from supplied TRAIN candidates excluding selected attachments",
                "final generator epoch, not author best outer-training-loss checkpoint",
            ])
        else:
            deviations.append("UGBA-repository GTA node/data-poisoning joint-optimization adaptation, not original model trojaning")
        record = {
            "method": method, "implementation": "fresh_node_classification_topology_injection_adaptation_v1",
            "seed": seed, "config": cfg, "author_repository": "https://github.com/ventr1c/UGBA",
            "author_commit": AUTHOR_SHA,
            "author_files": ["models/GTA.py" if method == "gta" else "models/backdoor.py", "heuristic_selection.py"]
                            + (["models/GCN_Encoder.py"] if method == "ugba" else []),
            "fresh_initialization": True, "checkpoint_loaded": False,
            "initial_state_sha256": initial, "final_generator_state_sha256": _state_hash(generator),
            "final_surrogate_state_sha256": _state_hash(surrogate), "generator_selected_epoch": cfg["epochs"] - 1,
            "selection": selection, "input_sha256": hashes, "attachment_indices": selected.tolist(),
            "attachment_indices_sha256": _tensor_hash(selected), "outer_indices": outer.tolist(),
            "fit_indices_sha256": _tensor_hash(fit_indices), "original_nodes": len(x), "poisoned_nodes": len(px),
            "injected_nodes": len(px) - len(x), "poisoned_attachment_count": len(selected),
            "original_edges": edge_index.shape[1], "poisoned_edges": pe.shape[1],
            "original_feature_prefix_unchanged": torch.equal(px[:len(x)], x),
            "original_edges_prefix_unchanged": torch.equal(pe[:, :edge_index.shape[1]], edge_index),
            "zero_weight_edges_removed": bool((pw > 0).all()), "counters": counters, "trace": trace,
            "deviations": deviations,
        }
    return {"generator": generator, "attachment_indices": selected.clone(), "poisoned_x": px,
            "poisoned_edge_index": pe, "poisoned_y": poison_y,
            "fit_indices": fit_indices, "record": record}
