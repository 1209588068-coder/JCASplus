"""Fresh WOMD selectors for the registered feature-space robustness pilot.

Author SPEAR GCN ranking, MLP and SAGE are loaded from a per-run source copy;
only optional-import compatibility is patched. No AV2 model, ranking or support
artifact is read. The semantic selector parameterizes the existing copy-group
rule for the newly ranked 30 coordinates; it never completes missing copies or
asserts physical trajectory realization. GPU is required except explicit small
synthetic smoke calls. This module starts no jobs on import.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .full_ablation_evaluation import _fp32
from .full_ablation_training import _cpu_state, _finite_gradients, _freeze, _state_hash, _tensor_hash
from .outer_selection import select_strict_negative_outer
from .support_generator import apply_gates
from .support_schema import coordinate_metadata


SOURCE = Path(__file__).resolve().parents[1] / 'third_party/author_baselines/SPEAR'
PROTECTED = (357, 715)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _device(x, allow_cpu_smoke):
    if x.device.type != 'cuda' and not allow_cpu_smoke:
        raise RuntimeError('Fresh WOMD training requires CUDA; no silent CPU fallback')
    if x.ndim != 2 or x.shape[1] != 784 or x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
        raise ValueError('Require finite float32 features[N,784]')


def candidate_groups(coordinates):
    """Intersect exact-copy identities with ranked candidates, never add fields."""
    values = torch.as_tensor(coordinates)
    if values.dtype != torch.long or values.shape != (30,):
        raise ValueError('Require fresh top30 long coordinates in importance order')
    coords = tuple(values.detach().cpu().tolist())
    if len(set(coords)) != 30 or any(c < 0 or c >= 784 for c in coords):
        raise ValueError('Candidate coordinates must be unique and in [0,784)')
    groups = {}
    metadata = []
    for rank, c in enumerate(coords):
        m = coordinate_metadata(c)
        # These three flags in the historical metadata refer to the old AV2
        # whitelist, so replace them rather than falsely reuse that whitelist.
        m.update(selected_rank=rank, legal_support=c not in PROTECTED,
                 exclusion_reason='focal_metadata' if c in PROTECTED else None,
                 normalized_units='fresh WOMD TRAIN column mean/std; clip [-10,10]')
        metadata.append(m)
        if c not in PROTECTED:
            groups.setdefault(m['copy_group'] or f'coordinate:{c}', []).append(c)
    partition = tuple(tuple(g) for g in groups.values())
    return partition, metadata


def exact_support(coordinates, groups, scores, budget=24):
    """Original coordinate-cost dynamic program on an explicit candidate set."""
    coords = tuple(torch.as_tensor(coordinates).detach().cpu().tolist())
    legal = set(coords) - set(PROTECTED)
    flat = [c for group in groups for c in group]
    if (type(budget) is not int or budget < 0 or budget > len(legal)
            or len(flat) != len(set(flat)) or set(flat) != legal
            or any(not group for group in groups)):
        raise ValueError('Invalid candidate partition or exact-coordinate budget')
    scores = torch.as_tensor(scores).detach().cpu().double()
    if scores.shape != (len(groups),) or not bool(torch.isfinite(scores).all()):
        raise ValueError('Require one finite score per group')
    best = {0: (0., ())}
    for i, (group, score) in enumerate(zip(groups, scores.tolist())):
        updated = dict(best)
        for cost, (total, chosen) in best.items():
            new_cost = cost + len(group)
            if new_cost > budget:
                continue
            value = (total + len(group) * score, chosen + (i,))
            previous = updated.get(new_cost)
            if previous is None or value[0] > previous[0] or (value[0] == previous[0] and value[1] < previous[1]):
                updated[new_cost] = value
        best = updated
    if budget not in best:
        raise ValueError('STOP: exact24 budget is infeasible; candidate set is not enlarged')
    chosen = {c for i in best[budget][1] for c in groups[i]}
    return torch.tensor([c in chosen for c in coords], dtype=torch.bool)


class CandidateHardConcrete(nn.Module):
    """Same global hard-concrete equations; groups supplied by fresh top30."""

    def __init__(self, coordinates, *, seed):
        super().__init__()
        self.coordinates = tuple(torch.as_tensor(coordinates).detach().cpu().tolist())
        self.groups, _ = candidate_groups(torch.tensor(self.coordinates))
        self.temperature, self.lower, self.upper = 2 / 3, -.1, 1.1
        initial = math.log(.8 / .2) + self.temperature * math.log(-self.lower / self.upper)
        self.log_alpha = nn.Parameter(torch.full((len(self.groups),), initial))
        self.register_buffer('rng_state', torch.Generator(device='cpu').manual_seed(seed).get_state())

    def activation_probability(self):
        return torch.sigmoid(self.log_alpha - self.temperature * math.log(-self.lower / self.upper))

    def expected_coordinate_count(self):
        return (self.log_alpha.new_tensor([len(g) for g in self.groups]) * self.activation_probability()).sum()

    def forward(self):
        rng = torch.Generator(device='cpu')
        rng.set_state(self.rng_state.detach().cpu())
        u = torch.rand(len(self.groups), generator=rng, dtype=torch.float64)
        self.rng_state.copy_(rng.get_state().to(self.rng_state.device))
        u = u.to(self.log_alpha).clamp(1e-6, 1 - 1e-6)
        soft = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.temperature)
        gates = (soft * (self.upper - self.lower) + self.lower).clamp(0, 1)
        lookup = {c: i for i, group in enumerate(self.groups) for c in group}
        zero = gates.new_zeros(())
        return torch.stack([gates[lookup[c]] if c in lookup else zero for c in self.coordinates])


def create_author_runtime(path):
    """Copy author source with recorded import-only compatibility patches."""
    destination = Path(path).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    originals = {}
    for p in sorted(SOURCE.rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        relative = p.relative_to(SOURCE)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        originals[str(relative)] = _sha(p)
    patches = {}

    def patch(relative, before, after):
        p = destination / relative
        text = p.read_text()
        if text.count(before) != 1:
            raise ValueError(f'Unexpected author source; bounded patch failed: {relative}')
        p.write_text(text.replace(before, after))
        patches.setdefault(relative, []).append(dict(before=before, after=after))

    patch('models/MLP.py', 'from deeprobust.graph import utils', 'import utils')
    branches = {'MLP': 'MLP', 'GCN': 'GCN', 'GAT': 'GAT', 'SAGE': 'GraphSage',
                'GCN_Encoder': 'GCN_Encoder', 'GNNGuard': 'GNNGuard', 'RobustGCN': 'RobustGCN'}
    for module, model in branches.items():
        statement = f'from models.{module} import {model}\n'
        patch('models/construct.py', statement, '')
        branch = "if (model_name == 'GCN'):\n" if model == 'GCN' else f"elif(model_name == '{model}'):\n"
        patch('models/construct.py', branch, branch + '        ' + statement)
    patch('sage_modified/__init__.py',
          'from . import utils, core, imputers, grouped_imputers, plotting, datasets',
          'from . import utils, core, imputers, grouped_imputers, datasets')
    patch('sage_modified/__init__.py', 'from .plotting import plot, comparison_plot\n', '')
    patch('sage_modified/core.py', 'from sage_modified import plotting\n', '')
    for function in ('plot', 'comparison_plot'):
        before = f'        return plotting.{function}('
        patch('sage_modified/core.py', before, '        from sage_modified import plotting\n' + before)
    binding = dict(source_root=str(SOURCE), original_sha256=originals, compatibility_patches=patches,
                   runtime_sha256={r: _sha(destination / r) for r in originals},
                   algorithm_changes=False, existing_selection_artifacts_copied=False)
    _json(destination / 'runtime_binding.json', binding)
    return binding


def _import_author_runtime(path):
    root = Path(path).resolve()
    names = ('models', 'utils', 'select_sample', 'sage_modified')
    for name in list(sys.modules):
        if name.split('.')[0] in names:
            del sys.modules[name]
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    selector = importlib.import_module('select_sample')
    mlp = importlib.import_module('models.MLP').MLP
    sage = importlib.import_module('sage_modified')
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in names and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f'Wrong author module origin: {name}')
    return selector, mlp, sage


@_fp32()
def fit_source_selection(train, dev, *, seed, output_dir, indices=None,
                         attachment_count=26, outer_extra_count=512, model_epochs=200,
                         sage_n_permutations=2048, sage_batch_size=64,
                         explanation_count=128, background_count=32,
                         progress=None, allow_cpu_smoke=False):
    """Fresh author ranking and attribution, then TRAIN-only outer selection.

    MLP and ranking checkpoints use DEV accuracy as in the original adaptation.
    Attribution samples and marginal backgrounds are exclusively labeled TRAIN.
    ``indices`` may provide the already registered idx_train; no IDs are loaded.
    """
    xtrain = train['x']
    _device(xtrain, allow_cpu_smoke)
    _device(dev['x'], allow_cpu_smoke)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    runtime_binding = create_author_runtime(directory / 'author_runtime')
    select_sample, MLP, sage = _import_author_runtime(directory / 'author_runtime')
    device = xtrain.device
    if dev['x'].device != device:
        raise ValueError('TRAIN and DEV tensors must share device')
    eligible = train['eligible'].detach().cpu().numpy()
    labels = train['y'].detach().cpu().numpy()
    if eligible.dtype != np.bool_ or eligible.shape != (len(xtrain),):
        raise ValueError('Require original boolean TRAIN eligibility')
    if indices is None:
        ids = np.flatnonzero(eligible)
        selected = np.random.RandomState(seed).permutation(ids)[:int(.2 * len(ids))]
        idx_train = torch.as_tensor(np.sort(selected), device=device, dtype=torch.long)
    else:
        idx_train = indices['idx_train'].detach().to(device).clone()
    if (idx_train.dtype != torch.long or idx_train.ndim != 1 or not idx_train.numel()
            or idx_train.unique().numel() != len(idx_train)
            or bool(((idx_train < 0) | (idx_train >= len(xtrain))).any())
            or not bool(train['eligible'][idx_train].all())
            or set(train['y'][idx_train].tolist()) != {0, 1}):
        raise ValueError('Clean TRAIN indices must be unique eligible rows with both classes')
    candidate = eligible & (labels == 0)
    candidate[idx_train.cpu().numpy()] = False
    candidates = torch.as_tensor(np.flatnonzero(candidate), device=device)
    if len(candidates) < attachment_count + outer_extra_count:
        raise ValueError('Insufficient eligible-negative TRAIN candidates; no replacement')
    x = torch.cat((xtrain, dev['x']))
    y = torch.cat((train['y'], dev['y']))
    edge = torch.cat((train['edge_index'], dev['edge_index'] + len(xtrain)), dim=1)
    idx_dev = torch.nonzero(dev['eligible'], as_tuple=False).flatten() + len(xtrain)
    if not len(idx_dev):
        raise ValueError('DEV must contain eligible supervised pairs')
    data = SimpleNamespace(x=x, y=y, edge_index=edge,
                           test_mask=torch.ones(len(x), dtype=torch.bool, device=device),
                           train_mask=torch.zeros(len(x), dtype=torch.bool, device=device))
    data.test_mask[candidates] = False
    data.train_mask[idx_train] = True
    args = SimpleNamespace(dataset='WOMD', model='GCN', epochs=model_epochs, hidden=80,
                           dropout=.5, train_lr=.01, weight_decay=.0005, vs_number=attachment_count)
    # Capture the author's freshly constructed ranker without replacing its fit
    # or sorting implementation. Per-run directory has no pre-existing cache.
    constructed = []
    original_construct = select_sample.model_construct

    def capture(*a, **kw):
        model = original_construct(*a, **kw)
        from .womd_execution import configure_message_passing
        configure_message_passing(model, decomposed_layers=16)
        constructed.append(model)
        return model

    select_sample.model_construct = capture
    np.random.seed(seed)
    torch.manual_seed(seed)
    if progress:
        progress(dict(stage='fresh_author_GCN_ranking', epochs=model_epochs))
    (directory / 'sorted_samples/WOMD').mkdir(parents=True)
    cwd = Path.cwd()
    try:
        os.chdir(directory)
        ranking = select_sample.sort(data, args, idx_train, idx_dev, device)
    finally:
        os.chdir(cwd)
        select_sample.model_construct = original_construct
    ranker = constructed[0]
    with torch.no_grad():
        scores = select_sample.entropy(ranker(x, edge))
    if not bool(torch.isfinite(scores[candidates]).all()):
        raise FloatingPointError('Author entropy-style score nonfinite; no score-formula replacement')
    if len(ranking) != len(candidates) or set(ranking.tolist()) != set(candidates.tolist()):
        raise ValueError('Author ranked list differs from eligible-negative TRAIN pool')
    attach = ranking[:attachment_count].detach().clone()
    torch.save(_cpu_state(ranker), directory / 'ranking_gcn.pt')
    torch.save(dict(candidates=candidates.cpu(), ranked=ranking.cpu(), score=scores[candidates].cpu()),
               directory / 'ranking.pt')
    del ranker, constructed[:]
    unused = torch.ones(len(xtrain), dtype=torch.bool, device=device)
    unused[idx_train] = False
    unused[attach] = False
    result_indices = dict(idx_train=idx_train, idx_attach=attach,
                          idx_unlabeled=torch.nonzero(unused, as_tuple=False).flatten(),
                          train_labels=train['y'].detach().clone())
    outer = select_strict_negative_outer(
        {k: train[k].detach().cpu() for k in ('y', 'eligible')},
        {k: v.detach().cpu() for k, v in result_indices.items()},
        extra_count=outer_extra_count, seed=seed)
    result_indices['fixed_outer'] = outer['outer'].to(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mlp = MLP(nfeat=784, nhid=80, nclass=2, dropout=.5, lr=.01,
              weight_decay=.0005, device=device).to(device)
    # The author constructor stores ``device`` but creates nn.Linear on CPU;
    # fit() moves only features/labels. Move parameters before Adam is built.
    if any(parameter.device != device for parameter in mlp.parameters()):
        raise RuntimeError('Attribution MLP parameters and TRAIN inputs must share device')
    if progress:
        progress(dict(stage='fresh_author_MLP', epochs=model_epochs))
    mlp.fit(x, y, idx_train, idx_dev, train_iters=model_epochs, verbose=True)
    mlp.eval()
    torch.save(_cpu_state(mlp), directory / 'attribution_mlp.pt')
    ordered = sorted(idx_train.tolist(), key=lambda i: hashlib.sha256(f'{seed}:womd-sage:{i}'.encode()).digest())
    if len(ordered) < max(explanation_count, background_count):
        raise ValueError('Insufficient labeled TRAIN rows for registered SAGE sample budget')
    explanation = np.asarray(ordered[:explanation_count], dtype=np.int64)
    background = np.asarray(ordered[:background_count], dtype=np.int64)
    numpy_x = xtrain.detach().cpu().numpy()
    imputer = sage.MarginalImputer(mlp, numpy_x[background])
    estimator = sage.PermutationEstimator(imputer, 'mse', n_jobs=1, random_state=seed)
    if progress:
        progress(dict(stage='fresh_author_SAGE', dimensions=784, permutations=sage_n_permutations))
    values, std = estimator(numpy_x[explanation], np.eye(2)[labels[explanation]],
                            batch_size=sage_batch_size, n_permutations=sage_n_permutations,
                            detect_convergence=False, verbose=True, bar=False).save_num()
    values, std = np.asarray(values), np.asarray(std)
    if values.shape != (784,) or std.shape != (784,) or not np.isfinite(values).all() or not np.isfinite(std).all():
        raise FloatingPointError('Nonfinite or malformed fresh SAGE attribution')
    coordinates = torch.as_tensor(np.argsort(values)[::-1][:30].copy(), dtype=torch.long, device=device)
    groups, metadata = candidate_groups(coordinates)
    exact_support(coordinates, groups, torch.ones(len(groups)), budget=24)
    np.savez(directory / 'attribution.npz', values=values, std=std, explanation=explanation, background=background)
    torch.save({k: v.cpu() for k, v in result_indices.items()}, directory / 'indices.pt')
    summary = dict(status='FRESH_SELECTION_COMPLETE', seed=seed, coordinates=coordinates.tolist(),
                   groups=groups, metadata=metadata, legal_capacity=sum(map(len, groups)),
                   protected_coordinates=list(PROTECTED), protected_in_candidates=[c for c in coordinates.tolist() if c in PROTECTED],
                   metadata_history_mask_candidates=[m['index'] for m in metadata if m['quantity'] in ('observed_valid', 'observed_fraction')],
                   candidate_policy='strict_top30_then_protect_focal_only_no_refill',
                   grouping='copy_identity_intersect_fresh_candidates_no_copy_completion',
                   interpretation='feature-space only; availability-field edits are not physical observations',
                   train_labeled_count=len(idx_train), attachment_count=len(attach), outer_selection=outer['selection'],
                   ranking='author_log_softmax_divide_row_sum_entropy_then_predicted_class_round_robin',
                   same_scene_multiple_attachments_allowed=True,
                   source_runtime_binding_sha256=_sha(directory / 'author_runtime/runtime_binding.json'),
                   checkpoint_rule='author_GCN_and_MLP_highest_DEV_accuracy_strict_improvement',
                   sage=dict(explanation_count=explanation_count, background_count=background_count,
                             permutations=sage_n_permutations, batch_size=sage_batch_size,
                             estimator='author_PermutationEstimator_mse_raw_MLP_outputs',
                             sample_order='sha256(seed:womd-sage:TRAIN_row_id)', convergence_claim=False),
                   av2_checkpoint_loaded=False, av2_learned_support_loaded=False, runtime_algorithm_changes=False,
                   message_passing_execution='PyG_GCN_feature_decomposition_16_no_sampling',
                   precision=dict(dtype='float32', tf32=False, amp=False),
                   full_pipeline_repetition=False)
    _json(directory / 'selection.json', summary)
    return dict(indices=result_indices, coordinates=coordinates, summary=summary)


@_fp32()
def fit_semantic_support(generator, surrogate, train, indices, proxies, coordinates, *,
                         seed, output_dir, epochs=200, budget=24,
                         progress=None, allow_cpu_smoke=False):
    """Freeze the fresh uniform anchor; train global gates and copied shadow."""
    features, labels, edge = train['x'], train['y'], train['edge_index']
    _device(features, allow_cpu_smoke)
    if type(epochs) is not int or epochs < 1 or budget != 24:
        raise ValueError('Require positive epoch count and registered exact24 support')
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    coordinates = coordinates.to(features.device)
    groups, metadata = candidate_groups(coordinates)
    legal = torch.tensor([int(c) not in PROTECTED for c in coordinates], device=features.device)
    exact_support(coordinates, groups, torch.ones(len(groups)), budget)
    if not torch.equal(generator.selected_coordinates, coordinates):
        raise ValueError('Anchor candidate order differs from fresh attribution')
    generator.set_support(legal)
    generator.set_phase('support_surrogate')
    _freeze(generator)
    owned = deepcopy(surrogate)
    from .womd_execution import configure_message_passing
    configure_message_passing(owned, decomposed_layers=16)
    owned.requires_grad_(True)
    selector = CandidateHardConcrete(coordinates, seed=seed).to(features.device)
    context = SimpleNamespace(edge_index=edge)
    weights = features.new_ones(edge.shape[1])
    if len(proxies) != 2:
        raise ValueError('Require two independent fresh frozen clean proxies')
    penalty = FrozenMeanCleanResponsePenalty(proxies, features, edge, weights, weight=1.)
    clean, attach, outer = [indices[k].to(features.device) for k in ('idx_train', 'idx_attach', 'fixed_outer')]
    for value in (clean, attach, outer):
        if value.dtype != torch.long or value.ndim != 1 or value.unique().numel() != value.numel() or bool(((value < 0) | (value >= len(features))).any()):
            raise ValueError('Invalid selection indices')
    if (not torch.equal(outer[:len(attach)], attach) or bool(torch.isin(clean, outer).any())
            or not bool((labels[outer] == 0).all()) or not bool(train['eligible'][outer].all())):
        raise ValueError('Require disjoint TRAIN clean/strict-negative outer with attachment prefix')
    inner_labels, outer_labels = labels.clone(), labels.clone()
    inner_labels[attach], outer_labels[outer] = 1, 1
    inner_indices, outer_indices = torch.cat((clean, attach)), torch.cat((clean, outer))
    optimizer_shadow = torch.optim.Adam(owned.parameters(), lr=.0002, weight_decay=.0005)
    optimizer_selector = torch.optim.Adam([selector.log_alpha], lr=.01, weight_decay=0.)
    before = _state_hash(generator)
    original_x_hash = _tensor_hash(features)
    proxy_before = [_state_hash(p) for p in proxies]
    # The anchor is frozen and eval for the whole stage. Caching its deterministic
    # original-input output changes no tensor values, but avoids 400 redundant
    # full-graph generator evaluations. Each gate sample remains independent.
    with torch.no_grad():
        generated_outer = generator(features, context, outer).detach()
        clean_outer = features[outer[:, None], coordinates].detach()

    def inject(targets, generated, original):
        gates = selector()
        values = apply_gates(original, generated, gates)
        changed = features.clone()
        changed[targets[:, None], coordinates] = values
        return changed

    history = []
    np.random.seed(seed)
    torch.manual_seed(seed)
    for epoch in range(epochs):
        owned.requires_grad_(True).train()
        optimizer_shadow.zero_grad(set_to_none=True)
        with torch.no_grad():
            changed = inject(attach, generated_outer[:len(attach)], clean_outer[:len(attach)])
        output = owned(changed, edge, weights)
        inner_loss = F.nll_loss(output[inner_indices], inner_labels[inner_indices])
        if not bool(torch.isfinite(inner_loss)):
            raise FloatingPointError('Nonfinite selector inner loss')
        inner_loss.backward()
        _finite_gradients(owned.parameters(), 'support surrogate')
        optimizer_shadow.step()
        optimizer_shadow.zero_grad(set_to_none=True)
        owned.requires_grad_(False).eval()
        optimizer_selector.zero_grad(set_to_none=True)
        changed = inject(outer, generated_outer, clean_outer)
        output = owned(changed, edge, weights)
        target = F.nll_loss(output[outer_indices], outer_labels[outer_indices])
        cosine = 1 - F.cosine_similarity(changed, features, dim=1).mean()
        kl = penalty(changed, edge, weights, outer)
        expected = selector.expected_coordinate_count()
        cost = ((expected - budget) / int(legal.sum())).square()
        loss = target + .1 * cosine + kl + cost
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite selector outer loss')
        feature_gradient = torch.autograd.grad(changed.sum(), selector.log_alpha, retain_graph=True)[0]
        if not bool(torch.isfinite(feature_gradient).all()):
            raise FloatingPointError('Broken selector feature-gradient path')
        loss.backward()
        _finite_gradients([selector.log_alpha], 'support selector')
        optimizer_selector.step()
        if not bool(torch.isfinite(selector.log_alpha).all()):
            raise FloatingPointError('Nonfinite selector state')
        if any(p.grad is not None for p in generator.parameters()):
            raise ValueError('Frozen generator received gradients')
        row = dict(stage='semantic_support', epoch=epoch + 1, epochs=epochs,
                   inner_nll=float(inner_loss.detach()), outer_nll=float(target.detach()),
                   cosine=float(cosine.detach()), clean_kl=float(kl.detach()),
                   expected_coordinate_count=float(expected.detach()), budget_penalty=float(cost.detach()),
                   outer_loss=float(loss.detach()))
        history.append(row)
        if progress:
            progress(dict(row))
    if before != _state_hash(generator) or original_x_hash != _tensor_hash(features) or proxy_before != [_state_hash(p) for p in proxies]:
        raise ValueError('Frozen anchor/proxies/original data changed during selector fitting')
    support = exact_support(coordinates, groups, selector.activation_probability(), budget).to(features.device)
    torch.save(_cpu_state(selector), directory / 'selector.pt')
    torch.save(_cpu_state(owned), directory / 'selector_surrogate.pt')
    summary = dict(status='FRESH_SEMANTIC_SUPPORT_COMPLETE', seed=seed, coordinates=coordinates.tolist(),
                   groups=groups, group_sizes=list(map(len, groups)), metadata=metadata,
                   legal_capacity=int(legal.sum()), hard_support=support.tolist(),
                   active_coordinates=coordinates[support].tolist(), coordinate_budget=budget,
                   activation_probability=selector.activation_probability().detach().cpu().tolist(),
                   anchor_initial_sha256=before, anchor_final_sha256=_state_hash(generator),
                   epochs=epochs, surrogate_updates=epochs, selector_updates=epochs,
                   soft_training_strict_budget=False, hardening='one_final_exact_coordinate_cost_DP',
                   generator_optimized=False, selector_learning_rate=.01, shadow_learning_rate=.0002,
                   loss='target_NLL + 0.1*global_cosine + mean_dual_clean_KL + ((E[cost]-24)/legal_capacity)^2',
                   candidate_adaptation='fresh_top30_copy_identity_intersection; no AV2 fixed support',
                   cache='frozen_eval_anchor_outputs_on_fixed_outer; gate_samples_not_cached',
                   message_passing_execution='PyG_GCN_feature_decomposition_16_no_sampling',
                   precision=dict(dtype='float32', tf32=False, amp=False),
                   av2_checkpoint_loaded=False, av2_learned_support_loaded=False)
    _json(directory / 'history.json', history)
    _json(directory / 'support.json', summary)
    return support, summary
