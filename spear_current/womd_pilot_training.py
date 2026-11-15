"""Fresh WOMD pilot training on caller-supplied, normalized pair graphs.

Offline, authorized model-robustness research only. This module reads no graph,
AV2 checkpoint, normalization file or historical attachment/support list.
It reuses the retained GCN/MLP/GraphSAGE, uniform residual and dual-KL arithmetic.
The fresh reconstruction is base -> uniform residual -> separately fitted
semantic gates; the five-stage fixed-support ablation is not replayed here.
All real training/evaluation requires CUDA FP32, with no silent CPU fallback.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .compact_ablation_training import CompactAblationGenerator
from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .models import GCN, GraphSage, GraphTrojanNet
from .womd_execution import configure_message_passing
from .full_ablation_training import (
    _cpu_state, _finite_gradients, _finite_loss, _finite_module, _freeze,
    _indices, _log_probabilities, _state_hash, _tensor_hash,
)
from .full_ablation_evaluation import (
    _cuda_device, _fp32, _inject, _predict, _feature_row, _collateral,
)
from .evaluation import _component, _normal_metrics, _target_metrics


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _save(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def _folder(path):
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def cuda_preflight(device='cuda:0', *, train=None, dev=None, idx_train=None,
                   optimize_execution=True):
    """Real CUDA GNN forward/backward, optionally using full TRAIN/DEV graphs.

    No optimizer step or persistent model is produced. This checks GraphSAGE
    resource demand, not the larger simultaneous dual-proxy autograd footprint.
    OOM propagates; the workflow may record a resource wait without CPU fallback.
    """
    device = _cuda_device(device)
    started = time.monotonic()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.random.fork_rng(devices=[device.index]), _fp32():
        if train is None:
            x = torch.randn(6, 784, device=device)
            edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device=device)
            y = torch.tensor([0, 1]*3, device=device)
            idx = torch.arange(6, device=device)
        else:
            train = graph_to_device(train, device)
            x, y, edges = (train[k] for k in ('x', 'y', 'edge_index'))
            idx = (torch.nonzero(train['eligible'], as_tuple=False).flatten() if idx_train is None
                   else torch.as_tensor(idx_train, dtype=torch.long, device=device))
        model = GraphSage(784, 80, 2, device=device).to(device)
        execution = configure_message_passing(model) if optimize_execution else dict(operator='original_scatter')
        loss = F.nll_loss(model(x, edges)[idx], y[idx])
        loss.backward()
        _finite_gradients(model.parameters(), 'CUDA smoke model')
        torch.cuda.synchronize(device)
        backward_seconds = time.monotonic()-started
        if dev is not None:
            dev = graph_to_device(dev, device)
            model.eval()
            with torch.no_grad():
                output = model(dev['x'], dev['edge_index'])
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError('Nonfinite DEV resource smoke')
            torch.cuda.synchronize(device)
    return dict(device=str(device), name=torch.cuda.get_device_name(device),
                torch_version=torch.__version__, cuda=torch.version.cuda,
                forward_backward_finite=True, dtype='float32', amp=False, tf32=False,
                train_nodes=len(x), train_edges=edges.shape[1], full_graph=train is not None,
                forward_backward_seconds=backward_seconds, total_seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                execution=execution,
                optimizer_steps=0, scope='one GraphSAGE full batch; no dual-KL capacity guarantee')


def graph_to_device(graph, device='cuda:0'):
    """Copy numerical graph fields; identity strings remain caller metadata."""
    device = _cuda_device(device)
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.set_device(device)
    result = dict(graph)
    for key in ('x', 'y', 'edge_index', 'eligible', 'motion_regime'):
        if key in graph:
            result[key] = torch.as_tensor(graph[key]).to(device)
    x, y, edge = (result[k] for k in ('x', 'y', 'edge_index'))
    if (x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 784
            or not bool(torch.isfinite(x).all()) or y.dtype != torch.long
            or y.shape != (len(x),) or edge.dtype != torch.long
            or edge.ndim != 2 or edge.shape[0] != 2
            or bool(((edge < 0) | (edge >= len(x))).any())):
        raise ValueError('Malformed WOMD pair graph')
    if 'eligible' in result:
        eligible = result['eligible']
        if (eligible.dtype != torch.bool or eligible.shape != y.shape
                or not bool(((y[eligible] == 0) | (y[eligible] == 1)).all())):
            raise ValueError('Invalid eligible supervision')
    return result


def fit_classifier(train, dev, idx_train, *, seed, output_dir, role,
                   epochs=200, device='cuda:0', checkpoint_rule='dev_auc',
                   progress=None):
    """Fresh GraphSAGE; proxies may use final epoch, victim/reference clean DEV AUC.

    The caller passes poison-relabelled tensors only for role='victim'. Clean
    reference is a separate model and is never supplied to generator fitting.
    """
    device = _cuda_device(device)
    if role not in ('clean_proxy_1', 'clean_proxy_2', 'clean_reference', 'victim'):
        raise ValueError('Unknown classifier role')
    if checkpoint_rule not in ('dev_auc', 'final') or type(epochs) is not int or epochs < 1:
        raise ValueError('Invalid classifier schedule')
    train = graph_to_device(train, device)
    idx = torch.as_tensor(idx_train, dtype=torch.long, device=device)
    _indices(idx, 'idx_train', train['x'])
    if not len(idx) or not bool(((train['y'][idx] == 0) | (train['y'][idx] == 1)).all()):
        raise ValueError('Classifier requires nonempty binary supervision')
    if 'eligible' not in train or not bool(train['eligible'][idx].all()):
        raise ValueError('Classifier supervision must be strictly eligible')
    if checkpoint_rule == 'dev_auc':
        if dev is None:
            raise ValueError('DEV AUC rule needs separate clean DEV')
        dev = graph_to_device(dev, device)
        dev_idx = torch.nonzero(dev['eligible'], as_tuple=False).flatten()
        truth = dev['y'][dev_idx].cpu().numpy()
        if set(truth.tolist()) != {0, 1}:
            raise ValueError('DEV AUC needs both true classes')
    folder = _folder(output_dir)
    torch.manual_seed(seed)
    model = GraphSage(784, 80, 2, dropout=.5, lr=.01, weight_decay=.0005,
                      device=device).to(device)
    execution = configure_message_passing(model)
    initial = _state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    best_auc, selected_epoch, best_state = float('-inf'), None, None
    with _fp32(), (folder/'epochs.jsonl').open('x') as log:
        for epoch in range(epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = _log_probabilities(model, train['x'], train['edge_index'], None)
            loss = F.nll_loss(logits[idx], train['y'][idx])
            _finite_loss(loss, role); loss.backward()
            _finite_gradients(model.parameters(), role)
            optimizer.step(); _finite_module(model, role)
            row = dict(role=role, epoch=epoch, train_loss=float(loss.detach()))
            if checkpoint_rule == 'dev_auc':
                model.eval()
                with torch.no_grad():
                    output = _log_probabilities(model, dev['x'], dev['edge_index'], None)[dev_idx]
                    auc = float(roc_auc_score(truth, output[:, 1].exp().cpu().numpy()))
                    row.update(dev_auc=auc, dev_loss=float(F.nll_loss(output, dev['y'][dev_idx])))
                if auc > best_auc:
                    best_auc, selected_epoch, best_state = auc, epoch, _cpu_state(model)
            else:
                selected_epoch = epoch
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if progress:
                progress(dict(stage=role, **row))
    final_hash = _state_hash(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    _freeze(model)
    summary = dict(role=role, seed=seed, epochs=epochs, optimizer_steps=epochs,
        selected_epoch=selected_epoch, best_dev_auc=best_auc if best_state is not None else None,
        selection_policy='earliest_maximum_clean_dev_auc' if checkpoint_rule == 'dev_auc' else 'final_epoch',
        initial_state_sha256=initial, final_epoch_state_sha256=final_hash,
        selected_state_sha256=_state_hash(model), fresh_initialization=True,
        attack_checkpoint_loaded=False, train_supervision_count=len(idx),
        train_features_sha256=_tensor_hash(train['x']), train_labels_sha256=_tensor_hash(train['y']),
        train_indices_sha256=_tensor_hash(idx), device=str(device), dtype='float32',
        amp=False, tf32=False, architecture='GraphSAGE_784_80_2_layers2_dropout0.5',
        optimizer=dict(name='Adam', lr=.01, weight_decay=.0005))
    summary['execution'] = execution
    _save(folder/'model.pt', dict(state_dict=_cpu_state(model), metadata=summary))
    _json(folder/'summary.json', summary)
    return dict(model=model, summary=summary)


def fit_generator_anchor(train, indices, proxies, coordinates, *, seed, output_dir,
                         base_epochs=200, residual_epochs=200, device='cuda:0',
                         inner_steps=1, progress=None):
    """Fresh legal-field GCN/MLP followed by uniform residual, final epochs only.

    Base-stage conditioning follows the old previous-inner-poison-x path.
    Residual conditioning always uses original clean x. Both use target NLL +
    0.1 global cosine + average clean-response KL of two frozen clean proxies.
    No candidate/support list or model is loaded from any previous experiment.
    """
    device = _cuda_device(device)
    if any(type(v) is not int or v < 1 for v in (base_epochs, residual_epochs, inner_steps)):
        raise ValueError('Positive fixed budgets required')
    train = graph_to_device(train, device)
    x, labels, edges = (train[k] for k in ('x', 'y', 'edge_index'))
    context, weights = SimpleNamespace(edge_index=edges), x.new_ones(edges.shape[1])
    idx_train, attach, outer = [torch.as_tensor(indices[k], dtype=torch.long, device=device)
        for k in ('idx_train', 'idx_attach', 'fixed_outer')]
    for name, value in zip(('idx_train', 'idx_attach', 'fixed_outer'), (idx_train, attach, outer)):
        _indices(value, name, x)
        if not len(value):
            raise ValueError('Empty index population: '+name)
    if (not bool((labels[attach] == 0).all()) or not torch.equal(outer[:len(attach)], attach)
            or bool(torch.isin(idx_train, outer).any())
            or not bool(((labels[idx_train] == 0) | (labels[idx_train] == 1)).all())):
        raise ValueError('Invalid poison/clean/outer separation')
    if ('eligible' not in train or not bool(train['eligible'][idx_train].all())
            or not bool(train['eligible'][outer].all()) or not bool((labels[outer] == 0).all())):
        raise ValueError('Require eligible clean supervision and strict-negative outer population')
    coordinates = torch.as_tensor(coordinates, dtype=torch.long, device=device)
    legal = (coordinates != 357) & (coordinates != 715)
    if len(proxies) != 2:
        raise ValueError('Exactly two independent fresh clean proxies required')
    for proxy in proxies:
        if next(proxy.parameters()).device != device:
            raise ValueError('Clean proxy device mismatch')
        proxy._womd_trigger_optimization_proxy = True
    proxy_hashes = [_state_hash(p) for p in proxies]
    immutable = {k: _tensor_hash(v) for k, v in dict(x=x, labels=labels, edges=edges,
        idx_train=idx_train, attach=attach, outer=outer, coordinates=coordinates).items()}
    folder = _folder(output_dir)
    torch.manual_seed(seed)
    shadow = GCN(784, 80, 2, dropout=0., device=device).to(device)
    execution = configure_message_passing(shadow)
    base = GraphTrojanNet(device, 80, 30).to(device)
    generator = CompactAblationGenerator(base, shadow, coordinates, legal, variant='full').to(device)
    del base, shadow
    generator.set_phase('base')
    initial = _state_hash(generator)
    surrogate = generator.base_shadow
    penalty = FrozenMeanCleanResponsePenalty(proxies, x, edges, weights, 1.)
    inner_idx, outer_idx = torch.cat((idx_train, attach)), torch.cat((idx_train, outer))
    inner_y = labels.clone(); inner_y[attach] = 1
    outer_y = labels.clone(); outer_y[outer] = 1
    poison_x = x.clone().detach()
    stage_records = []
    with _fp32(), (folder/'epochs.jsonl').open('x') as log:
        for stage, epochs in (('base', base_epochs), ('residual', residual_epochs)):
            if stage == 'residual':
                surrogate = deepcopy(generator.base_shadow)
            generator.set_phase(stage)
            surrogate.requires_grad_(True)
            s_optimizer = torch.optim.Adam(surrogate.parameters(), lr=.0002, weight_decay=.0005)
            surrogate_ids = {id(p) for p in surrogate.parameters()}
            active = [p for p in generator.parameters() if p.requires_grad and id(p) not in surrogate_ids]
            g_optimizer = torch.optim.Adam(active, lr=.0002, weight_decay=.0005)
            frozen_base = _state_hash(generator.base_generator) if stage == 'residual' else None
            frozen_shadow = _state_hash(generator.base_shadow) if stage == 'residual' else None
            for epoch in range(epochs):
                generator.train(); generator.zero_grad(set_to_none=True)
                surrogate.requires_grad_(True).train()
                inner_losses = []
                for _ in range(inner_steps):
                    s_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        if stage == 'base':
                            embedding = surrogate.get_h(poison_x, edges)
                            values = generator.generate_with_embedding(x, context, attach, embedding)
                            poison_x = x.clone().detach()
                            poison_x[attach[:, None], coordinates] = values.detach()
                            inner_x = poison_x
                        else:
                            inner_x = generator.inject(x, context, attach, weights).features
                    output = _log_probabilities(surrogate, inner_x, edges, weights)
                    inner_loss = F.nll_loss(output[inner_idx], inner_y[inner_idx])
                    _finite_loss(inner_loss, 'inner'); inner_loss.backward()
                    _finite_gradients(surrogate.parameters(), 'surrogate')
                    s_optimizer.step(); _finite_module(surrogate, 'surrogate')
                    inner_losses.append(float(inner_loss.detach()))
                s_optimizer.zero_grad(set_to_none=True)
                surrogate.requires_grad_(False).eval(); generator.zero_grad(set_to_none=True)
                if stage == 'base':
                    with torch.no_grad():
                        embedding = surrogate.get_h(poison_x, edges)
                    values = generator.generate_with_embedding(x, context, outer, embedding)
                    changed = x.clone(); changed[outer[:, None], coordinates] = values
                else:
                    changed = generator.inject(x, context, outer, weights, detach=False).features
                output = _log_probabilities(surrogate, changed, edges, weights)
                nll = F.nll_loss(output[outer_idx], outer_y[outer_idx])
                similarity = 1. - F.cosine_similarity(changed, x, dim=1).mean()
                clean_kl = penalty(changed, edges, weights, outer)
                loss = nll + .1 * similarity + clean_kl + changed.sum() * 0.
                _finite_loss(loss, 'generator'); loss.backward()
                _finite_gradients(active, 'generator'); g_optimizer.step()
                _finite_module(generator, 'generator')
                surrogate.requires_grad_(True)
                if frozen_base and (_state_hash(generator.base_generator) != frozen_base
                        or _state_hash(generator.base_shadow) != frozen_shadow):
                    raise RuntimeError('Frozen anchor weights changed')
                row = dict(stage=stage, epoch=epoch, inner_nll=inner_losses,
                    outer_nll=float(nll.detach()), cosine_loss=float((.1*similarity).detach()),
                    clean_kl=float(clean_kl.detach()), clean_kl_member_raw=penalty.history[-1]['member_raw_kl'],
                    loss=float(loss.detach()), writable_coordinates=int(legal.sum()))
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                if progress:
                    progress(deepcopy(row))
            stage_record = dict(stage=stage, epochs=epochs, generator_steps=epochs,
                surrogate_steps=epochs*inner_steps, generator_sha256=_state_hash(generator),
                surrogate_sha256=_state_hash(surrogate), selected_epoch=epochs-1)
            stage_records.append(stage_record)
            _save(folder/(stage+'_final.pt'), dict(generator=_cpu_state(generator),
                surrogate=_cpu_state(surrogate), metadata=stage_record))
    _freeze(generator); _freeze(surrogate)
    if [_state_hash(p) for p in proxies] != proxy_hashes:
        raise RuntimeError('Frozen clean proxies changed')
    final_inputs = {k: _tensor_hash(v) for k, v in dict(x=x, labels=labels, edges=edges,
        idx_train=idx_train, attach=attach, outer=outer, coordinates=coordinates).items()}
    if final_inputs != immutable:
        raise RuntimeError('Original TRAIN inputs changed')
    summary = dict(status='ANCHOR_COMPLETE_SUPPORT_PENDING', seed=seed, stages=stage_records,
        reconstruction='fresh_base_then_uniform_residual_then_separate_semantic_gates',
        initial_generator_sha256=initial, final_generator_sha256=_state_hash(generator),
        clean_proxy_sha256=proxy_hashes, input_sha256=immutable,
        coordinates=coordinates.cpu().tolist(), legal_support=legal.cpu().tolist(),
        fresh_initialization=True, historical_attack_checkpoint_loaded=False,
        selection_policy='final_epoch_each_stage_no_dev_or_asr_selection',
        output_mode='identity', domain_projection=False, discriminator=False,
        target_weight=1., global_cosine_weight=.1, mean_dual_clean_kl_weight=1.,
        generator_lr=.0002, surrogate_lr=.0002, weight_decay=.0005,
        device=str(device), dtype='float32', amp=False, tf32=False)
    summary['execution'] = execution
    _json(folder/'summary.json', summary)
    return dict(generator=generator, surrogate=surrogate, summary=summary)


def fit_victim(generator, train, dev, indices, *, seed, output_dir, epochs=200,
               device='cuda:0', progress=None):
    """Train a fresh victim after hard semantic support has been frozen to 24."""
    device = _cuda_device(device)
    train = graph_to_device(train, device)
    attach = torch.as_tensor(indices['idx_attach'], dtype=torch.long, device=device)
    idx_train = torch.as_tensor(indices['idx_train'], dtype=torch.long, device=device)
    if bool(torch.isin(idx_train, attach).any()) or not bool((train['y'][attach] == 0).all()):
        raise ValueError('Invalid victim attachments')
    generator.eval()
    initial_hash = _state_hash(generator)
    with torch.no_grad(), _fp32():
        changed, _, _ = _inject(generator, train['x'], train['edge_index'], attach,
            train['x'].new_ones(train['edge_index'].shape[1]), 24)
    poisoned = dict(train, x=changed, y=train['y'].clone())
    poisoned['y'][attach] = 1
    result = fit_classifier(poisoned, dev, torch.cat((idx_train, attach)), seed=seed,
        output_dir=output_dir, role='victim', epochs=epochs, device=device,
        checkpoint_rule='dev_auc', progress=progress)
    if _state_hash(generator) != initial_hash:
        raise RuntimeError('Victim fitting changed the frozen generator')
    return result


def evaluate(generator, victim, clean_reference, parts, targets, *, output_dir,
             role, device='cuda:0', progress=None):
    """Fixed one-negative-target-per-scene evaluation with original vehicle IDs.

    This function never chooses a target or uses evaluation for optimization.
    A target must have been registered before any model prediction. The separate
    clean reference must not be one of the two gradient-optimization proxies.
    """
    device = _cuda_device(device)
    if role not in ('dev', 'audit') or len(parts) != len(targets):
        raise ValueError('Unknown or misaligned evaluation population')
    if clean_reference is victim or getattr(clean_reference, '_womd_trigger_optimization_proxy', False):
        raise ValueError('Clean reference must be separate from victim and trigger-optimization proxies')
    folder = _folder(output_dir)
    models = dict(victim=victim, clean_reference=clean_reference)
    modules = dict(generator=generator, **models)
    for module in modules.values():
        module.to(device).eval()
    hashes = {name: _state_hash(module) for name, module in modules.items()}
    normal_rows, trigger_rows, target_rows = [], [], []
    with _fp32(), torch.no_grad():
        for scene_i, (part, target) in enumerate(zip(parts, targets)):
            cx, ce, truth, eligible, ids, motion = _component(part)
            x, edges = cx.to(device), ce.to(device)
            weights = x.new_ones(edges.shape[1])
            clean = {name: _predict(model, x, edges, weights) for name, model in models.items()}
            if target is None:
                if np.any(eligible & (truth == 0)):
                    raise ValueError('Missing registered target despite eligible negatives')
                triggered, relations = clean, ['no_target']*len(x)
            else:
                if (type(target) is not int or not 0 <= target < len(x)
                        or not eligible[target] or truth[target] != 0):
                    raise ValueError('Target must be a fixed eligible true negative')
                changed, _, _ = _inject(generator, x, edges,
                    torch.tensor([target], dtype=torch.long, device=device), weights, 24)
                triggered = {name: _predict(model, changed, edges, weights) for name, model in models.items()}
                vehicles = set(ids[target])
                relations = ['target' if i == target else 'adjacent' if vehicles & set(pair)
                             else 'nonincident' for i, pair in enumerate(ids)]
            for i in np.flatnonzero(eligible):
                i = int(i)
                meta = dict(scenario_id=str(part['scenario_id']), pair_index=i, role=role,
                    split_group_id=str(part.get('split_group_id', 'unknown')),
                    src_track_id=str(ids[i, 0]), dst_track_id=str(ids[i, 1]),
                    true_label=int(truth[i]), label=int(truth[i]), eligible=True,
                    motion_regime='moving' if motion[i] else 'slow',
                    is_attachment=False, is_outer=False)
                normal = dict(meta)
                response = dict(meta, target_pair_index=target, relation=relations[i])
                for name in models:
                    normal.update({name+'_p': float(clean[name]['p'][i]), name+'_pred': int(clean[name]['pred'][i])})
                    for prefix, values in (('clean', clean), ('trigger', triggered)):
                        response.update({name+'_'+prefix+'_p': float(values[name]['p'][i]),
                                         name+'_'+prefix+'_pred': int(values[name]['pred'][i])})
                normal_rows.append(normal); trigger_rows.append(response)
                if i == target:
                    support = generator.selected_coordinates[generator.hard_support]
                    target_rows.append(dict(response, **_feature_row(x, changed, i, support)))
            if progress:
                progress(dict(stage='evaluate_'+role, scenes=scene_i+1, total_scenes=len(parts),
                              targets=len(target_rows)))
    if hashes != {name: _state_hash(module) for name, module in modules.items()}:
        raise RuntimeError('Evaluation changed model weights')
    metrics = {}
    for name in models:
        normal = {}
        for group in ('all', 'slow', 'moving'):
            rows = [r for r in normal_rows if group == 'all' or r['motion_regime'] == group]
            normal[group] = _normal_metrics(rows, name)
            labels = [r['label'] for r in rows]
            normal[group]['clean_pair_ap'] = (float(average_precision_score(labels,
                [r[name+'_p'] for r in rows])) if 1 in labels else None)
        metrics[name] = dict(targets=_target_metrics(target_rows, name), normal=normal,
                             collateral=_collateral(trigger_rows, name))
    for filename, rows in (('targets.csv', target_rows), ('clean_pair_scores.csv', normal_rows),
                           ('triggered_pair_scores.csv', trigger_rows)):
        with (folder/filename).open('x', newline='') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    summary = dict(status='EVALUATION_COMPLETE', role=role, metrics=metrics,
        scenes=len(parts), targets=len(target_rows), normal_pairs=len(normal_rows),
        model_sha256=hashes, device=str(device), dtype='float32', amp=False, tf32=False,
        topology_preserved=True, support_coordinates=24, threshold='argmax; tie selects class0',
        target_selection_uses_model=False, target_policy='one_registered_negative_per_scene',
        clean_reference='separately_initialized_clean_only_model_not_used_in_trigger_gradients',
        scope='WOMD training_20s internal '+role+'; not official test or certified recording-independent test')
    _json(folder/'summary.json', summary)
    return summary
