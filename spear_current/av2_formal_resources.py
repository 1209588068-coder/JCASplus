"""Zero-update CUDA resource probes on one registered normalized AV2 batch.

This is engineering acceptance for offline public-data robustness research.
The probe creates fresh models, uses synthetic candidate coordinates, performs
forward/backward only, and saves no trained models or experimental metrics.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .av2_formal_generator import outer_loss_terms
from .compact_ablation_training import CompactAblationGenerator
from .full_ablation_evaluation import _cuda_device, _fp32
from .full_ablation_training import _finite_gradients, _finite_loss, _freeze, _state_hash, _tensor_hash
from .models import GCN, GraphSage, GraphTrojanNet
from .support_generator import apply_gates
from .womd_execution import configure_message_passing
from .womd_pilot_selection import CandidateHardConcrete, create_author_runtime, _import_author_runtime
from .womd_pilot_training import graph_to_device


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def probe_batch(batch, output_path, device='cuda:0'):
    """Check six real CUDA graph/MLP/dual-KL paths, with zero optimizer steps.

    ``batch`` must already have been normalized by the registered full TRAIN
    cache. Its standard x/y/eligible/edge_index tensors are never modified.
    The caller chooses the maximum-node/edge batch from the frozen data index.
    """
    device = _cuda_device(device)
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError('Do not overwrite a resource acceptance record')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = dict(status='RUNNING', at=datetime.now(timezone.utc).isoformat(),
        scope='zero-update engineering resource check; not training or AV2 experimental results',
        device=str(device), dtype='float32', amp=False, tf32=False, optimizer_steps=0,
        historical_weights_loaded=False, historical_support_loaded=False,
        synthetic_coordinates=[357, 715]+list(range(28)), protected_coordinates=[357, 715],
        kernels={})
    try:
        graph = graph_to_device(batch, device)
        x, y, edge = graph['x'], graph['y'], graph['edge_index']
        eligible = graph['eligible']
        original_hashes = {name:_tensor_hash(graph[name]) for name in ('x', 'y', 'eligible', 'edge_index')}
        negatives = torch.nonzero(eligible & (y == 0), as_tuple=False).flatten()
        if len(negatives) < 33:
            raise ValueError('Resource batch needs 32 negative targets and binary remaining supervision')
        outer = negatives[:32]
        clean_mask = eligible.clone(); clean_mask[outer] = False
        clean = torch.nonzero(clean_mask, as_tuple=False).flatten()
        if set(y[clean].tolist()) != {0, 1}:
            raise ValueError('Resource batch clean supervision must contain both binary classes')
        supervised = torch.cat((clean, outer))
        changed_labels = y.clone(); changed_labels[outer] = 1
        coordinates = torch.tensor(record['synthetic_coordinates'], device=device)
        legal = (coordinates != 357) & (coordinates != 715)
        context = SimpleNamespace(edge_index=edge)
        record.update(nodes=len(x), edges=edge.shape[1], eligible=int(eligible.sum()),
            supervised=len(clean), outer_targets=len(outer), input_sha256=original_hashes,
            global_start=batch.get('global_start'), global_stop=batch.get('global_stop'),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(device))
        runtime = output_path.parent/(output_path.stem+'_author_runtime')
        binding = create_author_runtime(runtime)
        select_sample, MLP, _ = _import_author_runtime(runtime)
        record['author_runtime'] = str(runtime)
        record['author_source_algorithm_changes'] = binding['algorithm_changes']

        def measure(name, function):
            gc.collect(); torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.monotonic()
            details = function()
            torch.cuda.synchronize(device)
            record['kernels'][name] = dict(status='PASS', forward_backward_finite=True,
                optimizer_steps=0, runtime_seconds=time.monotonic()-start,
                baseline_allocated_bytes=allocated_before,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device), **details)
            _write(output_path, record)

        def classifier(kind):
            if kind == 'ranking_gcn':
                args = SimpleNamespace(dataset='AV2', hidden=80, dropout=.5,
                                       train_lr=.01, weight_decay=.0005)
                model = select_sample.model_construct(args, 'GCN',
                    SimpleNamespace(x=x, y=y), device).to(device)
                configure_message_passing(model)
            elif kind == 'attribution_mlp':
                model = MLP(784, 80, 2, dropout=.5, lr=.01, weight_decay=.0005, device=device).to(device)
            else:
                model = GraphSage(784, 80, 2, dropout=.5, device=device).to(device)
                configure_message_passing(model)
            model.train()
            before = _state_hash(model)
            logits = model(x) if kind == 'attribution_mlp' else model(x, edge)
            loss = (F.cross_entropy if kind == 'attribution_mlp' else F.nll_loss)(logits[clean], y[clean])
            _finite_loss(loss, kind); loss.backward()
            count, norm = _finite_gradients(model.parameters(), kind)
            if before != _state_hash(model):
                raise RuntimeError('Resource check changed classifier weights')
            return dict(loss=float(loss.detach()), parameter_gradient_count=count,
                        gradient_l2=norm, parameters_unchanged=True)

        def make_anchor():
            shadow = GCN(784, 80, 2, dropout=0., device=device).to(device)
            configure_message_passing(shadow)
            base = GraphTrojanNet(device, 80, 30).to(device)
            generator = CompactAblationGenerator(base, shadow, coordinates, legal, variant='full').to(device)
            proxies = []
            for _ in range(2):
                proxy = GraphSage(784, 80, 2, dropout=.5, device=device).to(device)
                configure_message_passing(proxy)
                _freeze(proxy); proxies.append(proxy)
            return generator, proxies

        def attack_path(phase):
            generator, proxies = make_anchor()
            generator.set_phase(phase if phase != 'semantic_gate' else 'support_surrogate')
            surrogate = deepcopy(generator.base_shadow)
            _freeze(surrogate)
            before = _state_hash(generator)
            if phase == 'semantic_gate':
                _freeze(generator)
                selector = CandidateHardConcrete(coordinates, seed=20261106).to(device)
                gate_before = selector.log_alpha.detach().clone()
                with torch.no_grad():
                    generated = generator(x, context, outer).detach()
                values = apply_gates(x[outer[:, None], coordinates], generated, selector())
                changed = x.clone(); changed[outer[:, None], coordinates] = values
                active = [selector.log_alpha]
            else:
                generator.train()
                changed = generator.inject(x, context, outer, detach=False).features
                active = [p for p in generator.parameters() if p.requires_grad]
            terms = outer_loss_terms(surrogate, proxies, x, changed, edge, changed_labels,
                supervised, outer, global_supervised=len(supervised), global_nodes=len(x),
                global_outer=len(outer))
            loss = terms[0] + .1*terms[1] + terms[2]
            if phase == 'semantic_gate':
                loss = loss + ((selector.expected_coordinate_count()-24)/int(legal.sum())).square()
            _finite_loss(loss, phase); loss.backward()
            count, norm = _finite_gradients(active, phase)
            if before != _state_hash(generator):
                raise RuntimeError('Resource check changed generator weights')
            if phase == 'semantic_gate' and not torch.equal(gate_before, selector.log_alpha):
                raise RuntimeError('Resource check changed selector parameters')
            if any(p.grad is not None for model in [surrogate]+proxies for p in model.parameters()):
                raise RuntimeError('Frozen resource-probe models received parameter gradients')
            writable = torch.zeros_like(x, dtype=torch.bool)
            writable[outer[:, None], coordinates[legal]] = True
            if bool((changed.detach()[~writable] != x[~writable]).any()):
                raise RuntimeError('Resource check modified forbidden feature coordinates')
            return dict(loss=float(loss.detach()), target_nll=float(terms[0].detach()),
                global_cosine=float(terms[1].detach()), mean_dual_clean_kl=float(terms[2].detach()),
                parameter_gradient_count=count, gradient_l2=norm, parameters_unchanged=True,
                focal_and_nontarget_features_unchanged=True,
                gradient_path='selector' if phase == 'semantic_gate' else phase+'_generator')

        with torch.random.fork_rng(devices=[device.index]), _fp32():
            torch.manual_seed(20261106)
            for kind in ('ranking_gcn', 'attribution_mlp', 'graphsage_classifier'):
                measure(kind, lambda kind=kind: classifier(kind))
            for phase in ('base', 'residual', 'semantic_gate'):
                measure(phase+'_dual_kl', lambda phase=phase: attack_path(phase))
        if original_hashes != {name:_tensor_hash(graph[name]) for name in original_hashes}:
            raise RuntimeError('Resource check changed original normalized inputs')
        record.update(status='PASS', input_unchanged=True,
            runtime_seconds=time.monotonic()-started,
            maximum_peak_allocated_bytes=max(k['peak_allocated_bytes'] for k in record['kernels'].values()),
            maximum_peak_reserved_bytes=max(k['peak_reserved_bytes'] for k in record['kernels'].values()),
            notes='single maximum-scene batch; no optimizer steps, no cached historical model/support, no memory guarantee for unrelated GPU jobs')
    except BaseException as exc:
        record.update(status='FAIL', error_type=type(exc).__name__, error=str(exc),
                      runtime_seconds=time.monotonic()-started)
        _write(output_path, record)
        raise
    _write(output_path, record)
    return record
