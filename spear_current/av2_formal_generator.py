"""Fresh AV2 generator/support fitting over complete disconnected scene batches.

Every epoch accumulates the full registered population before one Adam update.
Losses retain the original global denominators: supervised NLL, node cosine,
and outer-node mean KL. No scenes, edges, or model states are sampled/reused.
This is offline authorized research using public AV2 and researcher-owned GNNs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .av2_formal_classifiers import batches, sorted_indices, local_indices, EXECUTION
from .compact_ablation_training import CompactAblationGenerator
from .full_ablation_evaluation import _cuda_device, _fp32
from .full_ablation_training import (_cpu_state, _state_hash, _finite_loss,
    _finite_gradients, _finite_module, _freeze)
from .models import GCN, GraphTrojanNet
from .support_generator import apply_gates
from .womd_execution import configure_message_passing
from .womd_pilot_selection import CandidateHardConcrete, candidate_groups, exact_support


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _device(device, allow_cpu_smoke):
    device = torch.device(device)
    if device.type == 'cpu' and allow_cpu_smoke:
        return device
    return _cuda_device(device)


def _population(data_dir, indices):
    summary = json.loads((Path(data_dir) / 'summary.json').read_text())
    nodes = int(summary['train']['pair_nodes'])
    clean, attach, outer = [torch.as_tensor(indices[k]).detach().cpu()
        for k in ('idx_train', 'idx_attach', 'fixed_outer')]
    for value in (clean, attach, outer):
        sorted_indices(value, nodes)
    if (not torch.equal(outer[:len(attach)], attach)
            or bool(torch.isin(clean, outer).any())):
        raise ValueError('Require disjoint clean/outer and ordered attachment prefix')
    return dict(nodes=nodes, clean=clean, attach=attach, outer=outer,
                clean_sorted=clean.sort().values,
                inner_sorted=torch.cat((clean, attach)).sort().values,
                outer_sorted=torch.cat((clean, outer)).sort().values)


def _positions(global_indices, batch):
    start, stop = int(batch['global_start']), int(batch['global_stop'])
    positions = torch.nonzero((global_indices >= start) & (global_indices < stop),
                              as_tuple=False).flatten()
    return (global_indices[positions] - start).to(batch['x'].device), positions


def _validate_batch(batch, population):
    x, labels = batch['x'], batch['y']
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 784:
        raise ValueError('Require normalized float32 784-dimensional pair features')
    if int(batch['global_stop']) - int(batch['global_start']) != len(x):
        raise ValueError('Batch offset/population disagreement')
    clean = local_indices(population['clean_sorted'], batch)
    outer, _ = _positions(population['outer'], batch)
    if (not bool(batch['eligible'][clean].all())
            or not bool(((labels[clean] == 0) | (labels[clean] == 1)).all())
            or not bool(batch['eligible'][outer].all())
            or not bool((labels[outer] == 0).all())):
        raise ValueError('Require eligible binary clean and strict-negative outer rows')


def _changed(x, coordinates, target, values):
    result = x.clone()
    result[target[:, None], coordinates] = values
    return result


def outer_loss_terms(surrogate, proxies, original, changed, edge, labels,
                     supervised, outer, *, global_supervised, global_nodes,
                     global_outer, clean_probabilities=None):
    """A batch contribution to the original full-graph outer objective.

    Public for synthetic disjoint-graph numerical/gradient acceptance. KL is
    computed only at outer targets, exactly as FrozenMeanCleanResponsePenalty.
    """
    if min(global_supervised, global_nodes, global_outer) <= 0:
        raise ValueError('Global loss populations must be positive')
    if len(proxies) != 2:
        raise ValueError('Require two independently trained clean proxies')
    output = surrogate(changed, edge)
    nll = (F.nll_loss(output[supervised], labels[supervised], reduction='sum')
           / global_supervised) if len(supervised) else output.sum() * 0.
    cosine = (1. - F.cosine_similarity(changed, original, dim=1)).sum() / global_nodes
    kls = []
    for i, proxy in enumerate(proxies):
        if len(outer):
            if clean_probabilities is None:
                with torch.no_grad():
                    q = proxy(original, edge)[outer].exp()
            else:
                q = clean_probabilities[i]
            kls.append(F.kl_div(proxy(changed, edge)[outer], q,
                                reduction='sum') / global_outer)
        else:
            kls.append(changed.sum() * 0.)
    kl = torch.stack(kls).mean()
    return nll, cosine, kl


def sample_gate_uniform(selector):
    """Advance the original CPU hard-concrete RNG once per global gate draw."""
    rng = torch.Generator(device='cpu')
    rng.set_state(selector.rng_state.detach().cpu())
    uniform = torch.rand(len(selector.groups), generator=rng, dtype=torch.float64)
    selector.rng_state.copy_(rng.get_state().to(selector.rng_state.device))
    return uniform


def gates_from_uniform(selector, uniform):
    """Rebuild a fresh autograd graph with one shared global uniform draw."""
    u = uniform.to(selector.log_alpha).clamp(1e-6, 1 - 1e-6)
    soft = torch.sigmoid((torch.log(u) - torch.log1p(-u) + selector.log_alpha)
                         / selector.temperature)
    gates = (soft * (selector.upper - selector.lower) + selector.lower).clamp(0, 1)
    lookup = {c: i for i, group in enumerate(selector.groups) for c in group}
    zero = gates.new_zeros(())
    return torch.stack([gates[lookup[c]] if c in lookup else zero
                        for c in selector.coordinates])


def _progress(progress, stage, epoch, batch_index=None, **values):
    if progress:
        progress(dict(stage=stage, epoch=epoch, **({} if batch_index is None else
                      dict(batch=batch_index)), **values))


def _proxy_contract(proxies, device):
    if len(proxies) != 2 or proxies[0] is proxies[1]:
        raise ValueError('Require two independent fresh clean proxies')
    storages = []
    for proxy in proxies:
        if any(p.device != device for p in proxy.parameters()):
            raise ValueError('Proxy and training device differ')
        storage = {p.untyped_storage().data_ptr() for p in proxy.parameters()}
        if any(storage & old for old in storages):
            raise ValueError('Clean proxies share parameter storage')
        storages.append(storage)
        _freeze(proxy)
        proxy._womd_trigger_optimization_proxy = True
    return [_state_hash(proxy) for proxy in proxies]


def _cached_probabilities(data_dir, population, proxies, *, max_nodes, device):
    """Only outer-row q is required; avoid retaining full-population features."""
    result = [torch.empty((len(population['outer']), 2)) for _ in proxies]
    seen = 0
    with torch.no_grad():
        for batch in batches(data_dir, 'train', max_nodes, device):
            _validate_batch(batch, population)
            local, positions = _positions(population['outer'], batch)
            if not len(local):
                continue
            for target, proxy in zip(result, proxies):
                value = proxy(batch['x'], batch['edge_index'])[local].exp()
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('Nonfinite original clean-proxy response')
                target[positions] = value.detach().cpu()
            seen += len(local)
    if seen != len(population['outer']):
        raise ValueError('Outer probability-cache coverage mismatch')
    return result


def fit_generator(data_dir, indices, proxies, coordinates, *, seed, output_dir,
                  base_epochs=200, residual_epochs=200, max_nodes=16384,
                  device='cuda:0', inner_steps=1, progress=None,
                  allow_cpu_smoke=False):
    """Fresh base -> uniform residual; every pass uses the entire TRAIN role."""
    if (any(type(v) is not int or v < 1 for v in (base_epochs, residual_epochs, max_nodes))
            or inner_steps != 1):
        raise ValueError('Positive budgets and registered one inner step required')
    device = _device(device, allow_cpu_smoke)
    population = _population(data_dir, indices)
    coordinates = torch.as_tensor(coordinates, dtype=torch.long, device=device)
    groups, _ = candidate_groups(coordinates)
    exact_support(coordinates, groups, torch.ones(len(groups)), 24)
    legal = (coordinates != 357) & (coordinates != 715)
    proxy_hashes = _proxy_contract(proxies, device)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    shadow = GCN(784, 80, 2, dropout=0., device=device).to(device)
    configure_message_passing(shadow)
    base = GraphTrojanNet(device, 80, 30).to(device)
    generator = CompactAblationGenerator(base, shadow, coordinates, legal, variant='full').to(device)
    del base, shadow
    generator.set_phase('base')
    initial = _state_hash(generator)
    surrogate = generator.base_shadow
    # Previous inner poison differs from x only at attachment/candidate cells.
    previous_values = torch.empty((len(population['attach']), 30))
    previous_initialized = torch.zeros(len(population['attach']), dtype=torch.bool)
    stage_records = []
    with _fp32(), (folder / 'epochs.jsonl').open('x') as log:
        qcache = _cached_probabilities(data_dir, population, proxies,
                                       max_nodes=max_nodes, device=device)
        for stage, epochs in (('base', base_epochs), ('residual', residual_epochs)):
            if stage == 'residual':
                surrogate = deepcopy(generator.base_shadow)
            generator.set_phase(stage)
            surrogate.requires_grad_(True)
            s_optimizer = torch.optim.Adam(surrogate.parameters(), lr=.0002, weight_decay=.0005)
            surrogate_ids = {id(p) for p in surrogate.parameters()}
            active = [p for p in generator.parameters() if p.requires_grad and id(p) not in surrogate_ids]
            g_optimizer = torch.optim.Adam(active, lr=.0002, weight_decay=.0005)
            frozen_hashes = ([_state_hash(generator.base_generator), _state_hash(generator.base_shadow)]
                             if stage == 'residual' else None)
            for epoch in range(epochs):
                generator.train(); generator.zero_grad(set_to_none=True)
                surrogate.requires_grad_(True).train()
                s_optimizer.zero_grad(set_to_none=True)
                inner_loss, seen_inner, seen_nodes = 0., 0, 0
                for batch_i, batch in enumerate(batches(data_dir, 'train', max_nodes, device)):
                    _validate_batch(batch, population)
                    x, edge = batch['x'], batch['edge_index']
                    context = SimpleNamespace(edge_index=edge)
                    attach, apos = _positions(population['attach'], batch)
                    inner = local_indices(population['inner_sorted'], batch)
                    seen_nodes += len(x)
                    with torch.no_grad():
                        if stage == 'base':
                            old_values = x[attach[:, None], coordinates].clone()
                            initialized = previous_initialized[apos]
                            if bool(initialized.any()):
                                old_values[initialized.to(device)] = previous_values[apos[initialized]].to(device)
                            previous_x = _changed(x, coordinates, attach, old_values)
                            embedding = surrogate.get_h(previous_x, edge)
                            values = generator.generate_with_embedding(x, context, attach, embedding)
                            previous_values[apos] = values.cpu()
                            previous_initialized[apos] = True
                            changed = _changed(x, coordinates, attach, values)
                        elif len(attach):
                            changed = generator.inject(x, context, attach).features
                        else:
                            changed = x
                    if len(inner):
                        labels = batch['y'].clone(); labels[attach] = 1
                        loss = F.nll_loss(surrogate(changed, edge)[inner], labels[inner],
                                          reduction='sum') / len(population['inner_sorted'])
                        _finite_loss(loss, 'generator inner'); loss.backward()
                        inner_loss += float(loss.detach()); seen_inner += len(inner)
                    if batch_i % 10 == 0:
                        _progress(progress, stage + '_inner', epoch, batch_i)
                if seen_inner != len(population['inner_sorted']) or seen_nodes != population['nodes']:
                    raise ValueError('Incomplete full-TRAIN inner pass')
                _finite_gradients(surrogate.parameters(), 'surrogate')
                s_optimizer.step(); _finite_module(surrogate, 'surrogate')
                s_optimizer.zero_grad(set_to_none=True)
                surrogate.requires_grad_(False).eval(); generator.zero_grad(set_to_none=True)
                totals = [0., 0., 0.]; seen_outer, seen_supervised, seen_nodes = 0, 0, 0
                for batch_i, batch in enumerate(batches(data_dir, 'train', max_nodes, device)):
                    x, edge = batch['x'], batch['edge_index']
                    context = SimpleNamespace(edge_index=edge)
                    outer, opos = _positions(population['outer'], batch)
                    supervised = local_indices(population['outer_sorted'], batch)
                    labels = batch['y'].clone(); labels[outer] = 1
                    if len(outer):
                        if stage == 'base':
                            attach, apos = _positions(population['attach'], batch)
                            with torch.no_grad():
                                previous_x = _changed(x, coordinates, attach, previous_values[apos].to(device))
                                embedding = surrogate.get_h(previous_x, edge)
                            changed = _changed(x, coordinates, outer,
                                generator.generate_with_embedding(x, context, outer, embedding))
                        else:
                            changed = generator.inject(x, context, outer, detach=False).features
                        terms = outer_loss_terms(surrogate, proxies, x, changed, edge, labels,
                            supervised, outer, global_supervised=len(population['outer_sorted']),
                            global_nodes=population['nodes'], global_outer=len(population['outer']),
                            clean_probabilities=[q[opos].to(device) for q in qcache])
                        loss = terms[0] + .1 * terms[1] + terms[2]
                        _finite_loss(loss, 'generator outer'); loss.backward()
                    else:
                        with torch.no_grad():
                            terms = outer_loss_terms(surrogate, proxies, x, x, edge, labels,
                                supervised, outer, global_supervised=len(population['outer_sorted']),
                                global_nodes=population['nodes'], global_outer=len(population['outer']))
                    totals = [a + float(v.detach()) for a, v in zip(totals, terms)]
                    seen_outer += len(outer); seen_supervised += len(supervised); seen_nodes += len(x)
                    if batch_i % 10 == 0:
                        _progress(progress, stage + '_outer', epoch, batch_i)
                if (seen_outer != len(population['outer']) or seen_supervised != len(population['outer_sorted'])
                        or seen_nodes != population['nodes']):
                    raise ValueError('Incomplete full-TRAIN outer pass')
                _finite_gradients(active, 'generator'); g_optimizer.step()
                _finite_module(generator, 'generator'); surrogate.requires_grad_(True)
                if frozen_hashes and frozen_hashes != [_state_hash(generator.base_generator), _state_hash(generator.base_shadow)]:
                    raise RuntimeError('Residual training changed frozen anchor')
                row = dict(stage=stage, epoch=epoch, inner_nll=inner_loss,
                    outer_nll=totals[0], cosine_loss=.1 * totals[1], clean_kl=totals[2],
                    loss=totals[0] + .1 * totals[1] + totals[2], nodes_seen=seen_nodes,
                    outer_seen=seen_outer, generator_steps=epoch + 1, surrogate_steps=epoch + 1)
                log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                _checkpoint(folder / 'latest.pt', dict(generator=_cpu_state(generator), surrogate=_cpu_state(surrogate),
                    generator_optimizer=g_optimizer.state_dict(), surrogate_optimizer=s_optimizer.state_dict(),
                    previous_attachment_values=previous_values, previous_initialized=previous_initialized,
                    rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                    metadata=row))
                if progress:
                    progress(row)
            record = dict(stage=stage, epochs=epochs, generator_steps=epochs, surrogate_steps=epochs,
                generator_sha256=_state_hash(generator), surrogate_sha256=_state_hash(surrogate), selected_epoch=epochs - 1)
            stage_records.append(record)
            _checkpoint(folder / (stage + '_final.pt'), dict(generator=_cpu_state(generator),
                surrogate=_cpu_state(surrogate), metadata=record))
    _freeze(generator); _freeze(surrogate)
    if proxy_hashes != [_state_hash(proxy) for proxy in proxies]:
        raise RuntimeError('Frozen clean proxies changed')
    summary = dict(status='ANCHOR_COMPLETE_SUPPORT_PENDING', seed=seed, stages=stage_records,
        reconstruction='fresh_base_then_uniform_residual_then_separate_semantic_gates',
        initial_generator_sha256=initial, final_generator_sha256=_state_hash(generator),
        clean_proxy_sha256=proxy_hashes, fresh_initialization=True, historical_attack_checkpoint_loaded=False,
        coordinates=coordinates.cpu().tolist(), legal_support=legal.cpu().tolist(),
        protected_coordinates=[357, 715], output_mode='identity', domain_projection=False, discriminator=False,
        target_weight=1., global_cosine_weight=.1, mean_dual_clean_kl_weight=1.,
        clean_kl_population='outer nodes only', selection_policy='fixed final epoch per stage; no ASR selection',
        generator_lr=.0002, surrogate_lr=.0002, weight_decay=.0005,
        execution=EXECUTION, max_batch_nodes=max_nodes, device=str(device), dtype='float32', amp=False, tf32=False)
    _write(folder / 'summary.json', summary)
    return dict(generator=generator, surrogate=surrogate, summary=summary)


def fit_support(generator, surrogate, data_dir, indices, proxies, coordinates, *,
                seed, output_dir, epochs=200, max_nodes=16384, device='cuda:0',
                budget=24, progress=None, allow_cpu_smoke=False):
    """Fresh global semantic gates, shared gate draw across each complete pass."""
    if type(epochs) is not int or epochs < 1 or budget != 24:
        raise ValueError('Require fixed positive epochs and exact24 support')
    device = _device(device, allow_cpu_smoke)
    population = _population(data_dir, indices)
    coordinates = torch.as_tensor(coordinates, dtype=torch.long, device=device)
    if not torch.equal(generator.selected_coordinates, coordinates):
        raise ValueError('Support and generator candidate coordinates differ')
    groups, metadata = candidate_groups(coordinates)
    from .av2_formal_data import dataset_identity
    dataset, source_split = dataset_identity(data_dir)
    for item in metadata:
        item['normalized_units'] = f'{dataset} registered TRAIN column mean/std; clip [-10,10]'
    legal = (coordinates != 357) & (coordinates != 715)
    exact_support(coordinates, groups, torch.ones(len(groups)), budget)
    generator.set_support(legal); generator.set_phase('support_surrogate'); _freeze(generator)
    before = _state_hash(generator)
    proxy_hashes = _proxy_contract(proxies, device)
    owned = deepcopy(surrogate); owned.requires_grad_(True)
    selector = CandidateHardConcrete(coordinates, seed=seed).to(device)
    shadow_optimizer = torch.optim.Adam(owned.parameters(), lr=.0002, weight_decay=.0005)
    gate_optimizer = torch.optim.Adam([selector.log_alpha], lr=.01, weight_decay=0.)
    folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    generated = torch.empty((len(population['outer']), 30))
    original = torch.empty_like(generated)
    with _fp32(), torch.no_grad():
        seen = 0
        for batch in batches(data_dir, 'train', max_nodes, device):
            outer, positions = _positions(population['outer'], batch)
            if len(outer):
                generated[positions] = generator(batch['x'], SimpleNamespace(edge_index=batch['edge_index']), outer).cpu()
                original[positions] = batch['x'][outer[:, None], coordinates].cpu()
                seen += len(outer)
        if seen != len(population['outer']):
            raise ValueError('Frozen generator cache coverage mismatch')
        qcache = _cached_probabilities(data_dir, population, proxies, max_nodes=max_nodes, device=device)
    history = []
    with _fp32(), (folder / 'epochs.jsonl').open('x') as log:
        for epoch in range(epochs):
            owned.requires_grad_(True).train(); shadow_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                inner_gates = gates_from_uniform(selector, sample_gate_uniform(selector))
            inner_loss, seen = 0., 0
            for batch_i, batch in enumerate(batches(data_dir, 'train', max_nodes, device)):
                attach, positions = _positions(population['attach'], batch)
                inner = local_indices(population['inner_sorted'], batch)
                if len(inner):
                    with torch.no_grad():
                        values = apply_gates(original[positions].to(device), generated[positions].to(device), inner_gates)
                        changed = _changed(batch['x'], coordinates, attach, values)
                    labels = batch['y'].clone(); labels[attach] = 1
                    loss = F.nll_loss(owned(changed, batch['edge_index'])[inner], labels[inner],
                                      reduction='sum') / len(population['inner_sorted'])
                    _finite_loss(loss, 'support inner'); loss.backward()
                    inner_loss += float(loss.detach()); seen += len(inner)
                if batch_i % 10 == 0:
                    _progress(progress, 'support_inner', epoch, batch_i)
            if seen != len(population['inner_sorted']):
                raise ValueError('Support inner population coverage mismatch')
            _finite_gradients(owned.parameters(), 'support shadow'); shadow_optimizer.step()
            _finite_module(owned, 'support shadow'); shadow_optimizer.zero_grad(set_to_none=True)
            owned.requires_grad_(False).eval(); gate_optimizer.zero_grad(set_to_none=True)
            uniform = sample_gate_uniform(selector)
            totals = [0., 0., 0.]; seen, seen_nodes, seen_supervised = 0, 0, 0
            for batch_i, batch in enumerate(batches(data_dir, 'train', max_nodes, device)):
                x, edge = batch['x'], batch['edge_index']
                outer, positions = _positions(population['outer'], batch)
                supervised = local_indices(population['outer_sorted'], batch)
                labels = batch['y'].clone(); labels[outer] = 1
                if len(outer):
                    gates = gates_from_uniform(selector, uniform)
                    values = apply_gates(original[positions].to(device), generated[positions].to(device), gates)
                    changed = _changed(x, coordinates, outer, values)
                    terms = outer_loss_terms(owned, proxies, x, changed, edge, labels, supervised, outer,
                        global_supervised=len(population['outer_sorted']), global_nodes=population['nodes'],
                        global_outer=len(population['outer']), clean_probabilities=[q[positions].to(device) for q in qcache])
                    loss = terms[0] + .1 * terms[1] + terms[2]
                    _finite_loss(loss, 'support outer'); loss.backward()
                else:
                    with torch.no_grad():
                        terms = outer_loss_terms(owned, proxies, x, x, edge, labels, supervised, outer,
                            global_supervised=len(population['outer_sorted']), global_nodes=population['nodes'],
                            global_outer=len(population['outer']))
                totals = [a + float(v.detach()) for a, v in zip(totals, terms)]
                seen += len(outer); seen_nodes += len(x); seen_supervised += len(supervised)
                if batch_i % 10 == 0:
                    _progress(progress, 'support_outer', epoch, batch_i)
            if (seen != len(population['outer']) or seen_nodes != population['nodes']
                    or seen_supervised != len(population['outer_sorted'])):
                raise ValueError('Support outer population coverage mismatch')
            expected = selector.expected_coordinate_count()
            cost = ((expected - budget) / int(legal.sum())).square()
            _finite_loss(cost, 'support budget'); cost.backward()
            _finite_gradients([selector.log_alpha], 'support gates'); gate_optimizer.step()
            _finite_module(selector, 'support gates')
            row = dict(stage='semantic_support', epoch=epoch, inner_nll=inner_loss,
                outer_nll=totals[0], cosine=totals[1], clean_kl=totals[2], expected_coordinate_count=float(expected.detach()),
                budget_penalty=float(cost.detach()), outer_loss=totals[0] + .1 * totals[1] + totals[2] + float(cost.detach()),
                surrogate_updates=epoch + 1, selector_updates=epoch + 1)
            history.append(row); log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
            _checkpoint(folder / 'latest.pt', dict(selector=_cpu_state(selector), surrogate=_cpu_state(owned),
                selector_optimizer=gate_optimizer.state_dict(), surrogate_optimizer=shadow_optimizer.state_dict(),
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                metadata=row))
            if progress:
                progress(row)
    if before != _state_hash(generator) or proxy_hashes != [_state_hash(proxy) for proxy in proxies]:
        raise RuntimeError('Support fitting modified frozen generator/proxies')
    support = exact_support(coordinates, groups, selector.activation_probability(), budget).to(device)
    _checkpoint(folder / 'selector.pt', _cpu_state(selector))
    _checkpoint(folder / 'selector_surrogate.pt', _cpu_state(owned))
    summary = dict(status='FRESH_SEMANTIC_SUPPORT_COMPLETE', dataset=dataset,
        source_dataset_split=source_split, seed=seed, epochs=epochs,
        surrogate_updates=epochs, selector_updates=epochs, coordinates=coordinates.cpu().tolist(),
        groups=groups, metadata=metadata, hard_support=support.cpu().tolist(),
        active_coordinates=coordinates[support].cpu().tolist(), legal_capacity=int(legal.sum()), coordinate_budget=24,
        activation_probability=selector.activation_probability().detach().cpu().tolist(),
        anchor_initial_sha256=before, anchor_final_sha256=_state_hash(generator),
        generator_optimized=False, historical_support_loaded=False, selector_learning_rate=.01, shadow_learning_rate=.0002,
        hardening='one_final_exact_coordinate_cost_DP', global_gate_draws_per_epoch=2,
        loss='global_mean_target_NLL + 0.1*global_node_cosine + mean_dual_outer_node_KL + one_global_budget_cost',
        execution=EXECUTION, max_batch_nodes=max_nodes, device=str(device), dtype='float32', amp=False, tf32=False)
    _write(folder / 'history.json', history); _write(folder / 'support.json', summary)
    return support, summary
