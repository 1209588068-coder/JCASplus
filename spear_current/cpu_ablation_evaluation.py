"""CPU GraphSAGE fitting and reused-development evaluation for full ablations.

This module only uses caller-supplied original-TRAIN TRAIN/dev tensors.  It
opens no graphs and performs no author-baseline training.  DEV is the reused
development population, including its fixed public single-negative targets;
these results are not independent or blind confirmation evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .development_metrics import population_summary, response_metrics
from .evaluate import state_hash
from .evaluation import _component, _normal_metrics, _target_metrics
from .runtime import new_victim


MODELS = ('victim', 'clean_reference')


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _csv(path, rows, fields):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _asset(path, rows=None):
    path = Path(path).resolve()
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if rows is not None:
        result['rows'] = rows
    return result


def _cpu_device(device):
    device = torch.device(device)
    if device.type != 'cpu':
        raise RuntimeError('CPU ablation fitting/evaluation requires CPU; other devices are forbidden')
    return torch.device('cpu')


@contextmanager
def _fp32():
    """Disable ambient CPU autocast; all model tensors are validated as FP32."""
    with torch.autocast(device_type='cpu', enabled=False):
        yield


def _rng(device):
    _cpu_device(device)
    numpy_state = np.random.get_state()
    return dict(cpu_sha256=_digest(torch.get_rng_state()),
        numpy_sha256=hashlib.sha256(numpy_state[1].tobytes() +
            repr((numpy_state[0], *numpy_state[2:])).encode()).hexdigest())


def _runtime():
    return dict(torch=torch.__version__, python=platform.python_version(),
        platform=platform.platform(), machine=platform.machine(),
        processor=platform.processor(), threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), device_type='cpu')


def _check_module(module, device):
    for value in (*module.parameters(), *module.buffers()):
        if value.device != device or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError('Every model tensor must reside on the requested CPU device in FP32')


def _support(generator, device, support_capacity=24):
    if type(support_capacity) is not int or support_capacity not in (24, 28):
        raise ValueError('Explicit support capacity must be 24 or 28')
    selected = generator.selected_coordinates
    support = generator.hard_support
    if (getattr(generator, 'input_dim', None) != 784 or selected.shape != (30,)
            or selected.dtype != torch.long or support.shape != (30,)
            or support.dtype != torch.bool or len(selected.unique()) != 30
            or bool(((selected < 0) | (selected >= 784)).any()) or int(support.sum()) != support_capacity
            or selected.device != device or support.device != device):
        raise ValueError('Invalid frozen 784-input / 30-output trigger support capacity')
    if generator.output_mode not in ('identity', 'domain', 'domain_projection'):
        raise ValueError('Unknown trigger output mapping')
    return selected[support]


def _check_injection(original, changed, targets, generator, edges, new_edges, weights, new_weights,
                     support_capacity=24):
    """Enforce exact support/topology; only domain-mapped arms have a range bound."""
    support = _support(generator, original.device, support_capacity)
    allowed = torch.zeros_like(original, dtype=torch.bool)
    allowed[targets[:, None], support] = True
    if (changed.device != original.device or changed.dtype != torch.float32
            or changed.shape != original.shape or not bool(torch.isfinite(changed).all())
            or not torch.equal(changed[~allowed], original[~allowed])
            or new_edges.device != edges.device or new_edges.dtype != edges.dtype
            or new_weights.device != weights.device or new_weights.dtype != weights.dtype
            or not torch.equal(edges, new_edges) or not torch.equal(weights, new_weights)):
        raise ValueError('Injection changed frozen support, topology, weights, dtype or finite scope')
    if generator.output_mode != 'identity' and bool((changed[allowed].abs() > 10).any()):
        raise ValueError('Domain-mapped trigger escaped the fixed [-10, 10] domain')
    return support


def _inject(generator, x, edges, targets, weights, support_capacity=24):
    # Always condition on the original observation, never a prior triggered copy.
    context = SimpleNamespace(edge_index=edges.clone())
    result = generator.inject(x.clone(), context, targets, weights.clone(), detach=True)
    new_weights = getattr(result, 'edge_weight', None)
    if new_weights is None:
        new_weights = getattr(result, 'weight', None)
    if new_weights is None:
        raise ValueError('Injection must expose unchanged edge weights')
    _check_injection(x, result.features, targets, generator, edges,
                     result.edge_index, weights, new_weights, support_capacity)
    return result.features.detach(), result.edge_index, new_weights


def _log_probs(model, x, edges, weights):
    if (x.device.type != 'cpu' or x.dtype != torch.float32
            or edges.device != x.device or weights.device != x.device):
        raise ValueError('Every forward must consume CPU FP32 features and same-device graph tensors')
    output = model(x, edges, weights)
    if (output.device != x.device or output.dtype != torch.float32
            or output.shape != (len(x), 2) or not bool(torch.isfinite(output).all())
            or not torch.allclose(output.logsumexp(1), output.new_zeros(len(x)), atol=1e-5)):
        raise ValueError('Model returned invalid CPU FP32 binary log probabilities')
    return output


def _predict(model, x, edges, weights):
    with torch.no_grad():
        output = _log_probs(model, x, edges, weights)
    return dict(p=output[:, 1].exp().cpu().numpy(), pred=output.argmax(1).cpu().numpy(),
                nll=(-output[:, 1]).cpu().numpy())


def _role_graph(inputs, role):
    """Validate disjoint-union scene topology before moving tensors to CPU."""
    graph = inputs['combined'][role]
    x, y, eligible, edges = (graph[k] for k in ('x', 'y', 'eligible', 'edge_index'))
    parts, offsets = inputs['parts'][role], graph['offsets']
    if (x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 784
            or not bool(torch.isfinite(x).all()) or y.dtype != torch.long
            or y.shape != (len(x),) or eligible.dtype != torch.bool
            or eligible.shape != y.shape or edges.dtype != torch.long
            or edges.ndim != 2 or edges.shape[0] != 2
            or not bool(torch.isin(y[eligible], torch.tensor([0, 1], device=y.device)).all())):
        raise ValueError('Invalid combined original-TRAIN role graph')
    expected, count = [], 0
    for part in parts:
        expected.append(count)
        count += len(part['x'])
    if list(offsets) != expected or count != len(x):
        raise ValueError('Scene offsets changed')
    if edges.numel():
        if bool(((edges < 0) | (edges >= len(x))).any()):
            raise ValueError('Graph endpoint out of range')
        boundaries = torch.tensor(expected[1:], dtype=torch.long, device=edges.device)
        scene = torch.bucketize(edges, boundaries, right=True)
        if not torch.equal(scene[0], scene[1]):
            raise ValueError('Graph convolution edges cross scene boundaries')
    return graph


def _disjoint_roles(inputs):
    identities = {}
    for role in ('train', 'dev'):
        rows = inputs['metadata'][role]
        identities[role] = ({r['scenario_id'] for r in rows}, {r['split_group_id'] for r in rows})
        if any(r['role'] != role for r in rows):
            raise ValueError('Unexpected role identity')
    if any(a & b for a, b in zip(identities['train'], identities['dev'])):
        raise ValueError('TRAIN and DEV scenes or recording groups overlap')


def _selection(inputs, selection, train):
    chosen = {}
    for key in ('idx_train', 'idx_attach', 'fixed_outer'):
        value = selection[key].detach().cpu()
        if (value.dtype != torch.long or value.ndim != 1
                or len(value.unique()) != len(value) or bool(((value < 0) | (value >= len(train['x']))).any())):
            raise ValueError('Invalid fixed selection: ' + key)
        chosen[key] = value
    if not torch.equal(chosen['idx_train'], inputs['indices']['idx_train'].cpu()):
        raise ValueError('Victim clean supervision must retain original registered indices')
    if (not torch.equal(chosen['fixed_outer'][:len(chosen['idx_attach'])], chosen['idx_attach'])
            or bool(torch.isin(chosen['idx_train'], chosen['fixed_outer']).any())
            or not bool(torch.isin(chosen['idx_attach'], inputs['indices']['candidates'].cpu()).all())):
        raise ValueError('Attachments must be original candidates at the beginning of the fixed outer population')
    fit = torch.cat([chosen['idx_train'], chosen['idx_attach']])
    if (len(fit.unique()) != len(fit) or not len(fit)
            or not bool(train['eligible'].cpu()[fit].all())
            or not bool((train['y'].cpu()[chosen['idx_attach']] == 0).all())):
        raise ValueError('Invalid original victim supervision or attachment labels')
    return chosen, fit


def fit_victim(generator, inputs, selection, bundle, *, seed, output_dir,
               epochs=200, device='cpu', progress=None, support_capacity=24):
    """Fit a fresh matched-seed victim; select earliest maximum clean DEV AUC.

    TRAIN attachments alone are poisoned/relabelled. DEV features, labels and
    topology remain separate and clean, and no target-response metric enters
    fitting or checkpoint selection. ``epochs`` is configurable for tiny tests;
    the full experiment caller is responsible for retaining the 200-epoch plan.
    """
    device = _cpu_device(device)
    if type(seed) is not int or seed < 0 or type(epochs) is not int or epochs < 1:
        raise ValueError('Require a nonnegative seed and positive fixed epoch count')
    args = bundle.config['arguments']
    if any(args[k] != v for k, v in dict(hidden=80, dropout=.5, train_lr=.01, weight_decay=.0005).items()):
        raise ValueError('Registered GraphSAGE 784/80/2 optimizer or architecture changed')
    _disjoint_roles(inputs)
    train, dev = _role_graph(inputs, 'train'), _role_graph(inputs, 'dev')
    chosen, fit_cpu = _selection(inputs, selection, train)
    dev_cpu = torch.nonzero(dev['eligible'], as_tuple=False).flatten()
    truth = dev['y'][dev_cpu].cpu().numpy()
    if set(truth.tolist()) != {0, 1}:
        raise ValueError('Clean DEV AUC requires both binary classes')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / name for name in ('victim.pt', 'victim_record.json',
                                      'victim_dev_trace.csv', 'victim_dev_trace.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Refusing to overwrite victim evidence')
    generator.to(device=device, dtype=torch.float32).eval()
    _check_module(generator, device)
    _support(generator, device, support_capacity)
    generator_before = state_hash(generator)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    rng_seeded = _rng(device)
    # Retain the historical constructor and initialization RNG order on CPU.
    model = new_victim(bundle).to(device=device, dtype=torch.float32)
    model.device = device
    _check_module(model, device)
    initial = state_hash(model)
    initialization_digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initialization_digest.update(name.encode())
        initialization_digest.update(value.detach().cpu().numpy().tobytes())
    rng_fit = _rng(device)
    x, edges, labels = (train[k].to(device).clone() for k in ('x', 'edge_index', 'y'))
    dx, de = (dev[k].to(device) for k in ('x', 'edge_index'))
    weights, dw = x.new_ones(edges.shape[1]), dx.new_ones(de.shape[1])
    attachments = chosen['idx_attach'].to(device)
    fit, dev_idx = fit_cpu.to(device), dev_cpu.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    dev_labels = dev['y'].to(device)
    best_auc, best_epoch, best_state = float('-inf'), None, None
    trace = []
    with _fp32():
        with torch.no_grad():
            poisoned, _, _ = _inject(generator, x, edges, attachments, weights, support_capacity)
        labels[attachments] = 1
        if not torch.equal(labels.cpu()[fit_cpu], torch.where(
                torch.isin(fit_cpu, chosen['idx_attach']), 1, train['y'].cpu()[fit_cpu])):
            raise ValueError('Unexpected poisoned label scope')
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = _log_probs(model, poisoned, edges, weights)
            loss = F.nll_loss(output[fit], labels[fit])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite victim training loss')
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None or g.device != device or g.dtype != torch.float32
                   or not bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError('Missing/nonfinite victim CPU FP32 gradients')
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('Nonfinite victim parameters after optimizer step')
            if any(not bool(torch.isfinite(value).all())
                   for state in optimizer.state.values() for value in state.values()
                   if isinstance(value, torch.Tensor) and value.is_floating_point()):
                raise FloatingPointError('Nonfinite victim optimizer state')
            model.eval()
            with torch.no_grad():
                dev_output = _log_probs(model, dx, de, dw)[dev_idx]
                probabilities = dev_output[:, 1].exp().cpu().numpy()
                auc = float(roc_auc_score(truth, probabilities))
                dev_loss = float(F.nll_loss(dev_output, dev_labels[dev_idx]))
                accuracy = float((dev_output.argmax(1).cpu().numpy() == truth).mean())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), dev_auc=auc,
                       dev_loss=dev_loss, dev_accuracy=accuracy)
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            trace.append(row)
            if progress is not None:
                progress(dict(stage='fit_victim', **row))
    rng_final, last = _rng(device), state_hash(model)
    if state_hash(generator) != generator_before:
        raise ValueError('Victim fitting changed generator weights')
    model.load_state_dict(best_state, strict=True)
    model.eval().requires_grad_(False).zero_grad(set_to_none=True)
    selected_hash = state_hash(model)
    checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                      seed=seed, selected_epoch=best_epoch, selection_policy='earliest_maximum_clean_dev_auc')
    with paths[0].open('xb') as stream:
        torch.save(checkpoint, stream)
    _csv(paths[2], trace, list(trace[0]))
    _json(paths[3], trace)
    record = dict(victim_seed=seed, epochs=epochs, optimizer_steps=epochs,
        training_forwards=epochs, eval_forwards=epochs, selected_epoch=best_epoch,
        best_dev_auc=best_auc, selection_policy='earliest_maximum_clean_dev_auc',
        selection_metric='clean DEV ROC AUC; strict improvement; earliest tie',
        initialization_digest=initialization_digest.hexdigest(), initial_state_sha256=initial,
        final_epoch_state_sha256=last, selected_state_sha256=selected_hash,
        parameters_changed=initial != last, fit_rng_digest=rng_fit['cpu_sha256'],
        rng=dict(seeded=rng_seeded, before_fit=rng_fit, after_fit=rng_final),
        device=str(device), device_name=platform.machine() or 'CPU', dtype='float32',
        autocast_enabled=False, amp_enabled=False, tf32_enabled=False,
        cpu_forward_backward_proof=dict(training_forwards=epochs, clean_dev_forwards=epochs,
            backward_calls=epochs, features_device=str(device), parameters_device=str(device),
            gradients_device=str(device), features_dtype='float32', gradients_dtype='float32'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        poison_attachment_count=len(attachments), train_supervision_count=len(fit),
        dev_supervision_count=len(truth), poisoned_features_sha256=_digest(poisoned),
        poisoned_labels_sha256=_digest(labels), clean_dev_features_sha256=_digest(dev['x']),
        fixed_outer_sha256=_digest(chosen['fixed_outer']), topology_preserved=True,
        generator_unchanged=True, generator_state_sha256=generator_before,
        architecture=dict(name='GraphSage', input_dim=784, hidden=80, classes=2, layers=2,
                          dropout=.5, lr=.01, weight_decay=.0005),
        runtime=_runtime(),
        checkpoint=_asset(paths[0]), trace_csv=_asset(paths[2], epochs), trace_json=_asset(paths[3], epochs))
    _json(paths[1], record)
    return model, record


def _feature_row(original, changed, target, support):
    before = original[target, support].detach().cpu().double()
    after = changed[target, support].detach().cpu().double()
    delta = after - before
    return dict(changed_coordinates=int((delta != 0).sum()), delta_l1=float(delta.abs().sum()),
        delta_l2=float(delta.norm()), delta_linf=float(delta.abs().max()),
        out_of_range_coordinates=int((after.abs() > 10).sum()),
        boundary_coordinates=int((after.abs() == 10).sum()),
        output_min=float(after.min()), output_max=float(after.max()))


def _collateral(rows, name):
    result = {}
    for relation in ('adjacent', 'nonincident'):
        group = [r for r in rows if r['relation'] == relation]
        negatives = [r for r in group if r['true_label'] == 0]
        positives = [r for r in group if r['true_label'] == 1]
        flips = sum(r[name + '_clean_pred'] == 0 and r[name + '_trigger_pred'] == 1 for r in negatives)
        suppressed = sum(r[name + '_clean_pred'] == 1 and r[name + '_trigger_pred'] == 0 for r in positives)
        result[relation] = dict(negative_pairs=len(negatives), incremental_fp_count=flips,
            incremental_fp=flips / len(negatives) if negatives else None,
            positive_pairs=len(positives), positive_suppression_count=suppressed,
            positive_suppression=suppressed / len(positives) if positives else None)
    return result


def evaluate_case(generator, victim, clean_reference, inputs, *, device='cpu', output_dir,
                  support_capacity=24):
    """Evaluate unmodified and single-target-triggered DEV scene components."""
    device = _cpu_device(device)
    _role_graph(inputs, 'dev')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {k: folder / v for k, v in dict(targets='targets.csv', normal='clean_pair_scores.csv',
             triggered='triggered_pair_scores.csv', summary='summary.json').items()}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Refusing to overwrite full-ablation DEV evidence')
    models = dict(victim=victim, clean_reference=clean_reference)
    modules = dict(generator=generator, **models)
    for name, model in modules.items():
        model.to(device=device, dtype=torch.float32).eval()
        if name != 'generator':
            model.requires_grad_(False)
        _check_module(model, device)
    support = _support(generator, device, support_capacity)
    initial_hashes = {name: state_hash(model) for name, model in modules.items()}
    rng_before = _rng(device)
    parts = inputs['parts']['dev']
    targets = inputs['targets']['dev']
    offsets = inputs['combined']['dev']['offsets']
    if len(targets) != len(parts):
        raise ValueError('Fixed DEV target list does not align with scenes')
    target_rows, normal_rows, trigger_rows = [], [], []
    counts = dict(registered_scenes=len(parts), processed_scenes=0,
        scenes_without_eligible_negative_target=0, all_pair_nodes=0,
        supervised_pairs=0, targets=0, triggered_supervised_pair_rows=0)
    metadata_fields = ['scenario_id', 'split_group_id', 'role', 'pair_index', 'role_pair_index',
        'src_track_id', 'dst_track_id', 'true_label', 'eligible', 'motion_regime', 'is_attachment', 'is_outer']
    response_fields = [name + '_' + suffix for name in MODELS
        for suffix in ('clean_p', 'trigger_p', 'clean_pred', 'trigger_pred', 'target_nll')]
    feature_fields = ['changed_coordinates', 'delta_l1', 'delta_l2', 'delta_linf',
        'out_of_range_coordinates', 'boundary_coordinates', 'output_min', 'output_max']
    model_forwards = {name: 0 for name in MODELS}
    with _fp32(), torch.no_grad():
        for part, target, offset in zip(parts, targets, offsets):
            cpu_x, cpu_edges, truth, eligible, ids, motion = _component(part)
            x, edges = cpu_x.to(device), cpu_edges.to(device)
            weights = x.new_ones(edges.shape[1])
            clean = {name: _predict(model, x, edges, weights) for name, model in models.items()}
            for name in models:
                model_forwards[name] += 1
            counts['processed_scenes'] += 1
            counts['all_pair_nodes'] += len(x)
            counts['supervised_pairs'] += int(eligible.sum())
            eligible_indices = np.flatnonzero(eligible)
            if target is None:
                counts['scenes_without_eligible_negative_target'] += 1
                if bool((eligible & (truth == 0)).any()):
                    raise ValueError('Missing frozen target despite an eligible negative')
                triggered = clean
                relation = ['no_target'] * len(x)
            else:
                if (type(target) is not int or target < 0 or target >= len(x)
                        or not eligible[target] or truth[target] != 0):
                    raise ValueError('Fixed DEV target must be an eligible true negative')
                # Shared-vehicle identity defines collateral even if an edge
                # representation happens to omit a neighbor.
                target_vehicles = set(ids[target])
                relation = ['target' if i == target else 'adjacent' if target_vehicles & set(pair)
                            else 'nonincident' for i, pair in enumerate(ids)]
                changed, new_edges, new_weights = _inject(generator, x, edges,
                    torch.tensor([target], dtype=torch.long, device=device), weights, support_capacity)
                triggered = {name: _predict(model, changed, new_edges, new_weights)
                             for name, model in models.items()}
                for name in models:
                    model_forwards[name] += 1
                counts['targets'] += 1
            for index in eligible_indices:
                index = int(index)
                metadata = dict(inputs['metadata']['dev'][offset + index])
                if (metadata['scenario_id'] != part['scenario_id'] or metadata['pair_index'] != index
                        or metadata['true_label'] != int(truth[index]) or metadata['role'] != 'dev'
                        or not metadata['eligible'] or metadata['src_track_id'] != str(ids[index, 0])
                        or metadata['dst_track_id'] != str(ids[index, 1])
                        or metadata['motion_regime'] != ('moving' if motion[index] else 'slow')):
                    raise ValueError('DEV metadata does not match the original component')
                normal = dict(metadata, label=int(truth[index]))
                response = dict(metadata, label=int(truth[index]), target_pair_index=target,
                                relation=relation[index])
                for name in models:
                    normal.update({name + '_p': float(clean[name]['p'][index]),
                                   name + '_pred': int(clean[name]['pred'][index])})
                    response.update({name + '_clean_p': float(clean[name]['p'][index]),
                        name + '_trigger_p': float(triggered[name]['p'][index]),
                        name + '_clean_pred': int(clean[name]['pred'][index]),
                        name + '_trigger_pred': int(triggered[name]['pred'][index]),
                        name + '_target_nll': float(triggered[name]['nll'][index])})
                normal_rows.append(normal)
                trigger_rows.append(response)
                if target is not None and index == target:
                    target_rows.append(dict(metadata, **_feature_row(x, changed, target, support),
                                            **{key: response[key] for key in response_fields}))
    counts['triggered_supervised_pair_rows'] = len(trigger_rows)
    if initial_hashes != {name: state_hash(model) for name, model in modules.items()}:
        raise ValueError('Evaluation changed model state')
    rng_after = _rng(device)
    if rng_after != rng_before:
        raise ValueError('Evaluation consumed random numbers')
    metrics = {}
    for name in models:
        normal = _normal_metrics(normal_rows, name)
        labels = [r['label'] for r in normal_rows]
        normal['clean_pair_ap'] = float(average_precision_score(labels,
            [r[name + '_p'] for r in normal_rows])) if 1 in labels else None
        metrics[name] = dict(targets=_target_metrics(target_rows, name),
            target_response=response_metrics(target_rows, name), **normal,
            normal=dict(normal), collateral=_collateral(trigger_rows, name),
            normal_model_safeguard=dict(all_predictions_negative=normal['all_clean_predictions_negative'],
                positive_recall=normal['clean_pair_positive_recall'],
                collapsed_prediction_distribution=len(set(r[name + '_pred'] for r in normal_rows)) < 2,
                action='report diagnostic; do not replace, retune or discard this model'))
    features = dict(target_rows=len(target_rows), support_coordinates=len(support),
        output_mode=generator.output_mode,
        out_of_range_target_rows=sum(r['out_of_range_coordinates'] > 0 for r in target_rows),
        out_of_range_coordinates=sum(r['out_of_range_coordinates'] for r in target_rows),
        target_rows_changed=sum(r['changed_coordinates'] > 0 for r in target_rows),
        maximum_changed_coordinates_per_target=max((r['changed_coordinates'] for r in target_rows), default=0))
    for norm in ('delta_l1', 'delta_l2', 'delta_linf'):
        values = [r[norm] for r in target_rows]
        features[norm + '_mean'] = float(np.mean(values)) if values else None
        features[norm + '_max'] = max(values) if values else None
    _csv(paths['targets'], target_rows, metadata_fields + feature_fields + response_fields)
    _csv(paths['normal'], normal_rows, metadata_fields + ['label'] +
         [name + suffix for name in MODELS for suffix in ('_p', '_pred')])
    _csv(paths['triggered'], trigger_rows,
         metadata_fields + ['label', 'target_pair_index', 'relation'] + response_fields)
    summary = dict(models=metrics, metrics=metrics, dev_counts=counts,
        population=population_summary(target_rows), feature_changes=features,
        raw_targets=_asset(paths['targets'], len(target_rows)),
        raw_clean_scores=_asset(paths['normal'], len(normal_rows)),
        raw_triggered_pair_scores=_asset(paths['triggered'], len(trigger_rows)),
        model_state_sha256=initial_hashes, weights_unchanged=True, rng_unchanged=True,
        rng=rng_before, device=str(device), runtime=_runtime(), dtype='float32', autocast_enabled=False,
        amp_enabled=False, tf32_enabled=False,
        cpu_forward_proof=dict(model_forwards=model_forwards, generator_injections=len(target_rows),
                                features_device=str(device), features_dtype='float32'),
        topology_preserved=True, non_target_input_feature_rows_preserved=True,
        target_selection_uses_predictions=False, trigger_conditioning='original clean component only',
        frozen_target_indices_sha256=hashlib.sha256(json.dumps(targets,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        collateral_definition='same canonical vehicle track ID within the same scene; excludes target',
        threshold='binary model argmax; no threshold calibration',
        clean_reference_provenance='caller-supplied frozen clean-only reference; inspect run manifest for checkpoint origin',
        scope='reused original-TRAIN DEV; not independent or blind confirmation',
        original_val_arrays_loaded=False, original_test_arrays_loaded=False)
    _json(paths['summary'], summary)
    return summary
