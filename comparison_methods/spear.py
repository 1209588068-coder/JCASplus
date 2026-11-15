"""Fresh SPEAR adaptation for authorized, offline AV2 pair-line-graph research.

Algorithm source: https://github.com/yhDing/SPEAR at
187d2cdc1f2454795dc8b37467ee2103e2613b15 (user-supplied archive).
Authors' select_sample.py, select_feature.py, sage_modified/{imputers,
permutation_estimator,utils}.py and models/{MLP,backdoor_GCN_cos}.py define
the selector, global SAGE feature game and alternating neighbor generator.
Original authorship is retained; this extraction grants no additional license.

This is a budget/task adaptation, not a citation-benchmark reproduction:
pair nodes, supplied TRAIN-only negatives, 24 freshly ranked coordinates instead
of the source's default 30, fixed Monte Carlo budget, and TRAIN-final selector
models instead of validation-selected ones. No archived selection, model or
importance file is read. No graph loader, victim fit or experiment launch lives
here. Generated selected values are replacements, with no output projection.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spear_current.models import GCN, GraphTrojanNet


SOURCE_COMMIT = "187d2cdc1f2454795dc8b37467ee2103e2613b15"
SOURCE_URL = f"https://github.com/yhDing/SPEAR/tree/{SOURCE_COMMIT}"
SOURCE_ARCHIVE_SHA256 = "be18c0145089c0f063a6f863bf76b7de5d7654f65e1aa7427a54afe2505e0757"
SOURCE_FILES = {
    "select_sample.py": "112f27553a7958084c7780ed4a0559b16c2c27682f1771914a0a7c53e2c8e30a",
    "select_feature.py": "78fca47b59d380261a049bf5e2f66633192c9d284f4de3f7b35412014ed8568d",
    "models/backdoor_GCN_cos.py": "4230a6cbe19189b290bf400daf672906c49251065e9ab9cbad5829cae44a8bc1",
    "models/MLP.py": "42c891b9ef6a37ce73372b9669a447fe1dd1f2eb30ef9a7d047d78692da343ab",
    "sage_modified/permutation_estimator.py": "556a28eb187d48ea340bd22f39a700c20ec02d8ca0b2931f0bb5f254dcd7d2ac",
    "sage_modified/imputers.py": "5eb616422d8698f35c8cc2539c64eb852253167abeefe7ec771eeea9e9a752fb",
}
DEFAULT_CONFIG = dict(
    epochs=200, inner=1, hidden=80, poison_count=26, feature_count=24,
    lr=0.01, weight_decay=5e-4, target_class=1,
    select_pretrain_epochs=200, feature_pretrain_epochs=200,
    selector_dropout=0.5, sage_permutations=512, sage_background=16,
    sage_batch_size=32, outer_count=512, target_weight=1.0,
    similarity_weight=0.1,
)


def validate_config(config=None):
    """Validate and return a complete, serializable preregistration config."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown SPEAR configuration keys: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **config}
    integer_names = (
        "epochs", "inner", "hidden", "poison_count", "feature_count",
        "select_pretrain_epochs", "feature_pretrain_epochs", "sage_permutations",
        "sage_background", "sage_batch_size", "outer_count",
    )
    for name in integer_names:
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if result["hidden"] < 2:
        raise ValueError("hidden must be at least two for the source get_h normalization")
    if type(result["target_class"]) is not int or result["target_class"] != 1:
        raise ValueError("this eligible-negative binary adaptation requires target_class=1")
    for name in ("lr", "weight_decay", "target_weight", "similarity_weight", "selector_dropout"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        result[name] = float(value)
    if result["lr"] <= 0 or result["target_weight"] <= 0:
        raise ValueError("lr and target_weight must be positive")
    if result["weight_decay"] < 0 or result["similarity_weight"] < 0:
        raise ValueError("weight_decay and similarity_weight must be nonnegative")
    if not 0 <= result["selector_dropout"] < 1:
        raise ValueError("selector_dropout must lie in [0, 1)")
    return result


def _digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_digest(model):
    return _digest(model.state_dict())


def _indices(value, name, n, device):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional long tensor")
    if value.device != device or not len(value) or len(torch.unique(value)) != len(value):
        raise ValueError(f"{name} must be nonempty, unique and on the input device")
    if int(value.min()) < 0 or int(value.max()) >= n:
        raise ValueError(f"{name} contains out-of-range indices")
    return value.detach().clone()


def _emit(progress, **event):
    if progress is not None:
        progress(**event)


def _source_entropy(log_probabilities):
    """Preserve the released selector's log-output normalization, not softmax."""
    normalized = log_probabilities / log_probabilities.sum(dim=1, keepdim=True)
    entropy = -(normalized * torch.log2(normalized)).sum(dim=1)
    if not bool(torch.isfinite(entropy).all()):
        raise ValueError("source entropy is nonfinite; no silent probability reinterpretation")
    return entropy


def _rank_candidates(log_probabilities, candidates):
    entropy = _source_entropy(log_probabilities)
    pseudo_labels = log_probabilities.argmax(dim=1)
    ordered = candidates[entropy[candidates].argsort(descending=True)]
    # Same class-wise round-robin as select_sample.sort; absent classes exhaust.
    class_rows = [ordered[pseudo_labels[ordered] == k] for k in range(log_probabilities.shape[1])]
    rank = [rows[i] for i in range(max(map(len, class_rows))) for rows in class_rows if i < len(rows)]
    return torch.stack(rank), entropy, pseudo_labels


class _FeatureMLP(nn.Module):
    """The source MLP: raw Linear-ReLU-Linear output (no dropout/softmax)."""
    def __init__(self, nfeat, hidden, nclass):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(nfeat, hidden), nn.ReLU(), nn.Linear(hidden, nclass))

    def forward(self, x):
        return self.body(x)


def _fit_clean(model, x, edges, y, train, epochs, config, phase, progress, graph_model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x, edges) if graph_model else model(x)
        loss = F.nll_loss(output[train], y[train]) if graph_model else F.cross_entropy(output[train], y[train])
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite {phase} loss")
        loss.backward()
        optimizer.step()
        _emit(progress, phase=phase, epoch=epoch, epochs=epochs, loss=float(loss.detach()))
    model.eval()
    return float(loss.detach())


@torch.no_grad()
def _global_sage(model, x, y, background, *, permutations, batch_size, seed, progress=None):
    """Marginal-imputation permutation SAGE, source MSE(raw logits, one-hot).

    Each draw samples an explanation row and its own feature permutation. Missing
    coordinates are jointly replaced by each background row; model outputs are
    averaged before squared error. This preserves the supplied estimator's game,
    with a declared fixed draw count and deterministic private RNG.
    """
    rng = np.random.default_rng(seed)
    dimensions = x.shape[1]
    total = np.zeros(dimensions, dtype=np.float64)
    squares = np.zeros(dimensions, dtype=np.float64)
    count, calls = 0, 0
    rows_digest = hashlib.sha256()
    model.eval()
    while count < permutations:
        batch = min(batch_size, permutations - count)
        drawn = rng.choice(len(x), batch)
        orders = np.tile(np.arange(dimensions), (batch, 1))
        for order in orders:
            rng.shuffle(order)
        rows_digest.update(drawn.tobytes())
        rows_digest.update(orders.tobytes())
        examples = x[torch.as_tensor(drawn, device=x.device)]
        truth = F.one_hot(y[torch.as_tensor(drawn, device=x.device)], num_classes=model.body[-1].out_features).to(x.dtype)
        filled = background.unsqueeze(0).expand(batch, -1, -1).clone()
        sample_rows = torch.arange(batch, device=x.device)

        def losses():
            logits = model(filled.reshape(-1, dimensions)).reshape(batch, len(background), -1).mean(1)
            return (logits - truth).square().sum(1)

        previous = losses()
        scores = np.zeros((batch, dimensions), dtype=np.float64)
        for position in range(dimensions):
            coordinates = torch.as_tensor(orders[:, position], device=x.device)
            filled[sample_rows, :, coordinates] = examples[sample_rows, coordinates, None]
            current = losses()
            scores[np.arange(batch), orders[:, position]] = (previous - current).cpu().numpy()
            previous = current
        total += scores.sum(axis=0)
        squares += np.square(scores).sum(axis=0)
        count += batch
        calls += dimensions + 1
        _emit(progress, phase="spear_global_sage", completed=count, total=permutations, model_forwards=calls)
    values = total / count
    # Same population-variance / N standard error as ImportanceTracker.std.
    stderr = np.sqrt(np.maximum(squares - count * np.square(values), 0.0)) / count
    if not np.isfinite(values).all() or not np.isfinite(stderr).all():
        raise ValueError("nonfinite SAGE importance")
    return values, stderr, dict(draws=count, model_forwards=calls, draw_sha256=rows_digest.hexdigest())


class FreshSPEARGenerator(nn.Module):
    """Neighbor-conditioned source generator with a freshly selected support."""
    def __init__(self, shadow, trojan, coordinates):
        super().__init__()
        self.shadow = shadow
        self.trojan = trojan
        self.register_buffer("selected_coordinates", coordinates.detach().clone())

    @torch.no_grad()
    def inject(self, x, edge_index, targets):
        self.eval()
        if targets.dtype != torch.long or targets.ndim != 1 or targets.device != x.device:
            raise ValueError("targets must be a long vector on the input device")
        if len(targets) and (len(torch.unique(targets)) != len(targets) or int(targets.min()) < 0 or int(targets.max()) >= len(x)):
            raise ValueError("targets must be unique and in range")
        modified = x.clone()
        if len(targets):
            embeddings = self.shadow.get_h(x, edge_index)
            values = self.trojan(embeddings[targets], 0.5)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("nonfinite generated feature values")
            modified[targets[:, None], self.selected_coordinates] = values
        return modified, edge_index.clone(), torch.ones(edge_index.shape[1], dtype=x.dtype, device=x.device)


def fit_spear(x, edge_index, y, idx_train, candidate_indices, *, seed, config, progress=None):
    """Fit the source selector, SAGE importance and SPEAR G/S from fresh states.

    Caller supplies TRAIN-only tensors and disjoint eligible-negative candidates.
    This function cannot infer scene split membership from anonymous tensors.
    It never accepts historical attachments, importance values or checkpoints.
    ``progress`` follows the local callback convention ``progress(**event)``.
    """
    config = validate_config(config)
    if type(seed) is not int or seed < 0 or seed >= 2**32 - 401:
        raise ValueError("seed must be a nonnegative 32-bit integer with room for stage offsets")
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be a finite float32 matrix")
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.device != x.device:
        raise ValueError("edge_index must be a [2, E] long tensor on the input device")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= len(x)):
        raise ValueError("edge_index contains out-of-range nodes")
    if y.dtype != torch.long or y.shape != (len(x),) or y.device != x.device:
        raise ValueError("y must be a node-aligned long tensor on the input device")
    train = _indices(idx_train, "idx_train", len(x), x.device)
    candidates = _indices(candidate_indices, "candidate_indices", len(x), x.device)
    if bool(torch.isin(candidates, train).any()) or not bool((y[candidates] == 0).all()):
        raise ValueError("candidates must be disjoint from clean supervision and have eligible-negative labels")
    if not bool(((y[train] == 0) | (y[train] == 1)).all()) or len(torch.unique(y[train])) != 2:
        raise ValueError("clean supervision must contain both binary classes")
    if len(candidates) <= config["poison_count"]:
        raise ValueError("candidate pool must exceed poison_count to leave nonattachment outer rows")
    if config["feature_count"] > x.shape[1]:
        raise ValueError("feature_count exceeds input width")

    original = dict(x=x, edges=edge_index, y=y, train=train, candidates=candidates)
    inputs_digest = _digest(original)
    device = x.device
    stage_seeds = dict(selector=seed + 100, feature_model=seed + 200,
                       sage=seed + 300, shadow_and_generator=seed + 400, outer=seed)
    initial_states = {}
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(stage_seeds["selector"])
        selector = GCN(x.shape[1], config["hidden"], 2, dropout=config["selector_dropout"], device=device).to(device)
        initial_states["selector"] = _state_digest(selector)
        _fit_clean(selector, x, edge_index, y, train, config["select_pretrain_epochs"], config, "spear_selector", progress, True)
        with torch.no_grad():
            ranked, entropy, pseudo = _rank_candidates(selector(x, edge_index), candidates)
        attachments = ranked[:config["poison_count"]].clone()
        selector_final = _state_digest(selector)
        del selector

        torch.manual_seed(stage_seeds["feature_model"])
        feature_model = _FeatureMLP(x.shape[1], config["hidden"], 2).to(device)
        initial_states["feature_model"] = _state_digest(feature_model)
        _fit_clean(feature_model, x, edge_index, y, train, config["feature_pretrain_epochs"], config, "spear_feature_model", progress, False)
        background_indices = train[:config["sage_background"]]
        importance, stderr, sage_record = _global_sage(
            feature_model, x[train], y[train], x[background_indices],
            permutations=config["sage_permutations"], batch_size=config["sage_batch_size"],
            seed=stage_seeds["sage"], progress=progress,
        )
        coordinates = torch.as_tensor(np.argsort(importance)[::-1][:config["feature_count"]].copy(), dtype=torch.long, device=device)
        feature_final = _state_digest(feature_model)
        del feature_model

        remaining = candidates[~torch.isin(candidates, attachments)]
        outer_count = min(config["outer_count"], len(remaining))
        # The source resets RandomState(seed) on every outer step: one fixed draw.
        draw = np.random.RandomState(seed).choice(len(remaining), size=outer_count, replace=False)
        outer = torch.cat((attachments, remaining[torch.as_tensor(draw, device=device)]))
        fit_indices = torch.cat((train, attachments))
        outer_indices = torch.cat((train, outer))
        poisoned_y = y.clone()
        poisoned_y[attachments] = config["target_class"]
        outer_y = y.clone()
        outer_y[outer] = config["target_class"]

        torch.manual_seed(stage_seeds["shadow_and_generator"])
        shadow = GCN(x.shape[1], config["hidden"], 2, dropout=0.0, device=device).to(device)
        trojan = GraphTrojanNet(device, config["hidden"], config["feature_count"], layernum=2).to(device)
        initial_states["shadow"] = _state_digest(shadow)
        initial_states["generator"] = _state_digest(trojan)
        initial_training_rng = _digest(dict(rng=torch.get_rng_state()))
        shadow_optimizer = torch.optim.Adam(shadow.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        trigger_optimizer = torch.optim.Adam(trojan.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        weights = torch.ones(edge_index.shape[1], dtype=x.dtype, device=device)
        poison_x = x.detach().clone()
        best_loss, selected_epoch, best_state, history = float("inf"), -1, None, []
        for epoch in range(config["epochs"]):
            shadow.train()
            trojan.train()
            for _ in range(config["inner"]):
                shadow_optimizer.zero_grad()
                # Source inner embedding uses the previous poisoned input.
                embedding = shadow.get_h(poison_x, edge_index)
                generated = trojan(embedding[attachments], 0.5)
                poison_x = x.detach().clone()
                poison_x[attachments[:, None], coordinates] = generated.detach()
                prediction = shadow(poison_x, edge_index, weights)
                inner_loss = F.nll_loss(prediction[fit_indices], poisoned_y[fit_indices])
                if not bool(torch.isfinite(inner_loss)):
                    raise ValueError("nonfinite SPEAR inner loss")
                inner_loss.backward()
                shadow_optimizer.step()

            trigger_optimizer.zero_grad()
            embedding = shadow.get_h(poison_x, edge_index)
            generated = trojan(embedding[outer], 0.5)
            updated = x.clone()
            updated[outer[:, None], coordinates] = generated
            prediction = shadow(updated, edge_index, weights)
            target_loss = config["target_weight"] * F.nll_loss(prediction[outer_indices], outer_y[outer_indices])
            similarity_loss = 1 - F.cosine_similarity(updated, x).mean()
            outer_loss = target_loss + config["similarity_weight"] * similarity_loss
            if not bool(torch.isfinite(outer_loss)):
                raise ValueError("nonfinite SPEAR outer loss")
            outer_loss.backward()
            trigger_optimizer.step()
            loss_value = float(outer_loss.detach())
            # Source compares pre-step TRAIN loss, stores the post-step G only.
            if loss_value < best_loss:
                best_loss, selected_epoch = loss_value, epoch
                best_state = deepcopy(trojan.state_dict())
            row = dict(epoch=epoch, inner_loss=float(inner_loss.detach()),
                       target_loss=float(target_loss.detach()), similarity_loss=float(similarity_loss.detach()), outer_loss=loss_value)
            history.append(row)
            _emit(progress, phase="spear_generator", epochs=config["epochs"], **row)
        trojan.load_state_dict(best_state)
        generator = FreshSPEARGenerator(shadow, trojan, coordinates).eval().requires_grad_(False)
        poisoned_x, poisoned_edges, poisoned_weights = generator.inject(x, edge_index, attachments)

    if _digest(original) != inputs_digest:
        raise RuntimeError("SPEAR modified caller inputs")
    selection_digest = _digest(dict(attachments=attachments, ranked=ranked, coordinates=coordinates,
                                    importance=torch.as_tensor(importance), candidate_entropy=entropy[candidates]))
    record = dict(
        method="SPEAR_fresh_budget_adapted", status="FIT_COMPLETE", seed=seed,
        source_url=SOURCE_URL, source_commit=SOURCE_COMMIT, source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        source_files=deepcopy(SOURCE_FILES), config=config, stage_seeds=stage_seeds,
        fresh_initial_state_sha256=initial_states, initial_training_rng_sha256=initial_training_rng,
        final_selector_sha256=selector_final, final_feature_model_sha256=feature_final,
        final_generator_sha256=_state_digest(generator), input_sha256=inputs_digest, selection_sha256=selection_digest,
        attachment_indices=attachments.tolist(), selected_coordinates=coordinates.tolist(),
        candidate_ranking=ranked.tolist(), candidate_pseudo_labels=pseudo[candidates].tolist(),
        candidate_entropy=entropy[candidates].tolist(), feature_importance=importance.tolist(),
        feature_importance_stderr=stderr.tolist(),
        sage=dict(**sage_record, explanation_indices=train.tolist(), background_indices=background_indices.tolist(),
                  game="MSE of marginally averaged raw MLP logits versus one-hot labels", convergence_claim=False),
        selector="source log-output-normalized entropy descending, pseudo-class round-robin",
        selector_checkpoint="final TRAIN epoch; no dev selection", feature_checkpoint="final TRAIN epoch; no dev selection",
        generator_checkpoint="minimum pre-step TRAIN outer loss, saved post-step G; final shadow; earliest tie",
        selected_epoch=selected_epoch, minimum_outer_loss=best_loss,
        generator_steps=config["epochs"], surrogate_steps=config["epochs"] * config["inner"],
        selector_steps=config["select_pretrain_epochs"], feature_model_steps=config["feature_pretrain_epochs"],
        fixed_outer_indices=outer.tolist(), extra_outer_count=outer_count,
        outer_pool_capped=outer_count != config["outer_count"], topology_changed=False,
        feature_budget=dict(source_default=30, adapted=config["feature_count"], ranked_over_all_input_coordinates=True),
        history=history, original_inputs_unchanged=True,
        adaptations=[
            "node-classification source applied to caller's pair-line graph",
            "fresh eligible-negative TRAIN reservoir; no released or historical cached ranking",
            "TRAIN-only final GCN/MLP selector checkpoints; no validation access",
            "global SAGE explanations/background restricted to clean supervision",
            "fixed seeded SAGE draw budget instead of the source convergence stopping rule",
            "independent recorded fresh RNG streams for selector, MLP, SAGE and G/S",
            "declared 200 G steps and 24 feature writes instead of source defaults 400 and 30",
        ],
        added_kl=False, added_distribution_discriminator=False, added_domain_projection=False,
        added_pair_consistency=False, independent_data_confirmation=False,
    )
    return dict(generator=generator, attachment_indices=attachments, poisoned_x=poisoned_x,
                poisoned_edge_index=poisoned_edges, poisoned_edge_weight=poisoned_weights,
                poisoned_y=poisoned_y, fit_indices=fit_indices, record=record)
