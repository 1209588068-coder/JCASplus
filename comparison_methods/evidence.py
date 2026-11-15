"""Common fresh-victim fitting and original-node-only comparison evidence.

Topology-injection methods retain synthetic nodes for message passing, never
for supervision or metric denominators. No graph files are opened here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from spear_current.copy_consistency import COPY_GROUPS, COPY_NAMES, VIOLATION_TOLERANCE
from spear_current.development_diagnostic import choose_target
from spear_current.evaluation import _component, _normal_metrics, _target_metrics, _predict
from spear_current.models import GraphSage
from spear_current.train import tensor_digest


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_state(path, state):
    with Path(path).open('xb') as stream:
        torch.save(state, stream)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode() + value.numpy().tobytes()).hexdigest()


def identity_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_injection(x, edges, targets, result, *, structural, feature_count=24, trigger_size=3):
    """Check declared permissions, including edges ignored by GraphSAGE weights."""
    fx, fe, fw = result
    n, d = x.shape
    if (fx.ndim != 2 or fx.shape[1] != d or len(fx) < n or not torch.isfinite(fx).all()
            or fe.dtype != torch.long or fe.ndim != 2 or fe.shape[0] != 2
            or fw.shape != (fe.shape[1],) or not torch.isfinite(fw).all()
            or not bool((fw == 1).all()) or bool(((fe < 0) | (fe >= len(fx))).any())):
        raise ValueError('Invalid injection, unpruned zero-weight edges or nonbinary effective graph')
    if structural:
        if len(fx) != n + len(targets) * trigger_size or not torch.equal(fx[:n], x):
            raise ValueError('Structural method modified original features or node budget')
        original_part = fe[:, (fe < n).all(0)]
        if sorted(map(tuple, original_part.T.tolist())) != sorted(map(tuple, edges.T.tolist())):
            raise ValueError('Structural injection changed original-original connections')
        # Each trigger component belongs to exactly one target. This prevents
        # accidentally joining different scenes in the disjoint training union.
        for i, target in enumerate(targets.tolist()):
            lo, hi = n + i * trigger_size, n + (i + 1) * trigger_size
            incident = ((fe >= lo) & (fe < hi)).any(0)
            allowed = ((fe[:, incident] >= lo) & (fe[:, incident] < hi)) | (fe[:, incident] == target)
            if not bool(allowed.all()):
                raise ValueError('Trigger attaches to another target/scene')
    else:
        if fx.shape != x.shape or not torch.equal(fe, edges):
            raise ValueError('Feature method changed original topology')
        untouched = torch.ones(n, dtype=torch.bool, device=x.device)
        untouched[targets] = False
        if not torch.equal(fx[untouched], x[untouched]):
            raise ValueError('Feature method changed non-target rows')
        if len(targets) and bool(((fx[targets] != x[targets]).sum(1) > feature_count).any()):
            raise ValueError('Feature method exceeded coordinate budget')


def validate_fit_result(result, train, idx_train, candidates, *, method, config):
    targets = result['attachment_indices']
    if (targets.dtype != torch.long or targets.ndim != 1
            or len(targets) != config['poison_count'] or len(targets.unique()) != len(targets)
            or not bool(torch.isin(targets, candidates).all())):
        raise ValueError('Invalid selected attachments or candidate leakage')
    structural = method in ('ugba', 'dpgba', 'gta')
    expected_y = train['y'].clone()
    if method != 'cgba':
        expected_y[targets] = 1
    if not torch.equal(result['poisoned_y'], expected_y):
        raise ValueError('Training labels violate declared clean/dirty-label policy')
    expected_fit = torch.unique(torch.cat([idx_train, targets]), sorted=True)
    if not torch.equal(result['fit_indices'].sort().values, expected_fit):
        raise ValueError('Unexpected victim supervision or synthetic-node labels')
    weights = result.get('poisoned_edge_weight',
        torch.ones(result['poisoned_edge_index'].shape[1], dtype=train['x'].dtype))
    validate_injection(train['x'], train['edge_index'], targets,
        (result['poisoned_x'], result['poisoned_edge_index'], weights), structural=structural,
        feature_count=config.get('feature_count', 24), trigger_size=config.get('trigger_size', 3))
    generator = result['generator'].eval().requires_grad_(False)
    generator.zero_grad(set_to_none=True)
    with torch.no_grad():
        replay = generator.inject(train['x'], train['edge_index'], targets)
    validate_injection(train['x'], train['edge_index'], targets, replay, structural=structural,
        feature_count=config.get('feature_count', 24), trigger_size=config.get('trigger_size', 3))
    if (not torch.allclose(replay[0], result['poisoned_x'], atol=1e-6, rtol=1e-6)
            or not torch.equal(replay[1], result['poisoned_edge_index'])):
        raise ValueError('Saved trigger does not reproduce the poisoned training graph')


def fit_classifier(x, edges, y, fit_indices, dev, *, seed, output, epochs=200, progress=None):
    """Fresh GraphSAGE; normal DEV AUC only, earliest maximum; no ASR selection.

    Original labels can be shorter than x because injected nodes are unlabelled.
    DEV is evaluated separately, preventing edge-offset changes when N changes.
    """
    if (not len(fit_indices) or bool(((fit_indices < 0) | (fit_indices >= len(y))).any())
            or not bool(torch.isin(y[fit_indices], torch.tensor([0, 1])).all())
            or len(fit_indices.unique()) != len(fit_indices)):
        raise ValueError('Invalid supervision')
    truth = dev['y'][dev['eligible']]
    if set(truth.tolist()) != {0, 1}:
        raise ValueError('Normal DEV must contain both classes')
    torch.manual_seed(seed)
    model = GraphSage(x.shape[1], 80, 2, dropout=.5, lr=.01, weight_decay=5e-4, device='cpu')
    initialization = tensor_digest(model)
    fit_rng = tensor_hash(torch.get_rng_state())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)
    trace, best, selected, state = [], -float('inf'), None, None
    started = time.monotonic()
    with Path(output).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_nll', 'dev_auc', 'dev_ap', 'dev_recall'])
        writer.writeheader()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, edges)
            loss = F.nll_loss(prediction[fit_indices], y[fit_indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite classifier loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite classifier gradient')
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(dev['x'], dev['edge_index'])[dev['eligible']]
            probability = out[:, 1].exp().numpy()
            if not np.isfinite(probability).all():
                raise ValueError('Nonfinite classifier DEV output')
            auc = float(roc_auc_score(truth.numpy(), probability))
            row = dict(epoch=epoch, train_nll=float(loss.detach()), dev_auc=auc,
                dev_ap=float(average_precision_score(truth.numpy(), probability)),
                dev_recall=float((out.argmax(1)[truth == 1] == 1).float().mean()))
            if auc > best:
                best, selected = auc, epoch
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            writer.writerow(row)
            stream.flush()
            trace.append(row)
            if progress:
                progress(row)
    model.load_state_dict(state)
    model.eval().requires_grad_(False).zero_grad(set_to_none=True)
    return model, dict(seed=seed, epochs=epochs, selected_epoch=selected, selected_dev_auc=best,
        selection_policy='normal_dev_auc_strict_improvement_earliest_tie',
        initialization_sha256=initialization, fit_rng_sha256=fit_rng,
        fitted_state_sha256=tensor_digest(model), supervision_count=len(fit_indices),
        supervision_indices_sha256=tensor_hash(fit_indices.sort().values),
        elapsed_seconds=time.monotonic() - started, trace=trace)


def _csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def recompute_metrics(targets, normals, triggered):
    """CSV-compatible, same-model incremental denominators; no synthetic rows."""
    def number(row):
        return {k: (float(v) if k.endswith('_p') else int(v) if k.endswith('_pred') or k == 'label' else v)
                for k, v in row.items()}
    targets, normals, triggered = [[number(r) for r in rows] for rows in (targets, normals, triggered)]
    result = {}
    for model in ('victim', 'clean_reference'):
        normal = _normal_metrics(normals, model)
        normal['clean_pair_ap'] = (float(average_precision_score(
            [r['label'] for r in normals], [r[model + '_p'] for r in normals]))
            if any(r['label'] == 1 for r in normals) else None)
        strata = {s: _normal_metrics([r for r in normals if r['motion_regime'] == s], model)
                  for s in ('slow', 'moving')}
        collateral = {}
        for relation in ('adjacent', 'nonincident'):
            rows = [r for r in triggered if r['relation'] == relation]
            negative = [r for r in rows if r['label'] == 0]
            positive = [r for r in rows if r['label'] == 1]
            flips = sum(r[model + '_clean_pred'] == 0 and r[model + '_trigger_pred'] == 1 for r in negative)
            suppressed = sum(r[model + '_clean_pred'] == 1 and r[model + '_trigger_pred'] == 0 for r in positive)
            collateral[relation] = dict(negative_count=len(negative), incremental_positive_count=flips,
                incremental_fp=flips / len(negative) if negative else None,
                positive_count=len(positive), suppressed_count=suppressed,
                positive_suppression=suppressed / len(positive) if positive else None)
        result[model] = dict(targets=_target_metrics(targets, model), **normal,
                             normal_by_pair_motion=strata, collateral=collateral)
    return result


def evaluate_dev(parts, sources, generator, models, output, *, method, config, progress=None,
                 injection_validator=None):
    """Evaluate fixed one-per-scene public targets; no graph reader or optimizer."""
    folder = Path(output)
    folder.mkdir(exist_ok=False)
    targets, normals, triggered, changes, identity = [], [], [], [], []
    structural = method in ('ugba', 'dpgba', 'gta')
    injection_validator = validate_injection if injection_validator is None else injection_validator
    hashes = {name: tensor_digest(model) for name, model in models.items()}
    generator_hash = tensor_digest(generator)
    for scene_number, (part, source) in enumerate(zip(parts, sources), 1):
        x, edges, truth, eligible, ids, motion = _component(part)
        target = choose_target(part)
        clean = {name: _predict(model, x, edges, torch.ones(edges.shape[1])) for name, model in models.items()}
        def metadata(i):
            return dict(scenario_id=source['scenario_id'], split_group_id=source['split_group_id'],
                pair_index=int(i), src_track_id=ids[i, 0], dst_track_id=ids[i, 1],
                motion_regime='moving' if motion[i] else 'slow')
        for i in np.flatnonzero(eligible):
            row = dict(**metadata(i), label=int(truth[i]))
            for name, (p, pred) in clean.items():
                row.update({name + '_p': float(p[i]), name + '_pred': int(pred[i])})
            normals.append(row)
        if target is not None:
            identity.append(metadata(target))
            before_x, before_e = x.clone(), edges.clone()
            with torch.no_grad():
                injected = generator.inject(x, edges, torch.tensor([target]))
            if not torch.equal(x, before_x) or not torch.equal(edges, before_e):
                raise ValueError('Generator mutated clean evaluation input')
            injection_validator(x, edges, torch.tensor([target]), injected, structural=structural,
                feature_count=config.get('feature_count', 24), trigger_size=config.get('trigger_size', 3))
            fx, fe, fw = injected
            response = {name: _predict(model, fx, fe, fw) for name, model in models.items()}
            def predictions(i):
                row = {}
                for name in models:
                    row.update({name + '_clean_p': float(clean[name][0][i]),
                        name + '_trigger_p': float(response[name][0][i]),
                        name + '_clean_pred': int(clean[name][1][i]),
                        name + '_trigger_pred': int(response[name][1][i])})
                return row
            targets.append(dict(**metadata(target), true_label=0, **predictions(target)))
            neighbors = set(edges[1, edges[0] == target].tolist()) | set(edges[0, edges[1] == target].tolist())
            for i in np.flatnonzero(eligible):
                triggered.append(dict(**metadata(i), label=int(truth[i]), target_pair_index=int(target),
                    relation='target' if i == target else 'adjacent' if i in neighbors else 'nonincident',
                    **predictions(i)))
            delta = (fx[target] - x[target]).double()
            row = dict(**metadata(target), original_coordinates_changed=int((delta != 0).sum()),
                original_delta_l1=float(delta.abs().sum()), original_delta_l2=float(delta.norm()),
                original_delta_linf=float(delta.abs().max()),
                added_nodes=len(fx) - len(x), added_directed_edges=fe.shape[1] - edges.shape[1],
                injected_coordinate_count=(len(fx) - len(x)) * x.shape[1],
                final_original_row_max_abs=float(fx[target].abs().max()),
                actual_written_values_max_abs=(float(fx[len(x):].abs().max()) if structural
                    else float(fx[target, delta != 0].abs().max()) if bool((delta != 0).any()) else None))
            for name, group in zip(COPY_NAMES, COPY_GROUPS):
                # Unchanged original coordinates must not give an injection
                # method a misleading "zero generated-copy violations" score.
                values = fx[target, list(group)].double()
                row[name + '_touched'] = None if structural else int(bool((delta[list(group)] != 0).any()))
                row[name + '_range'] = None if structural else float(values.max() - values.min())
                row[name + '_violation'] = (None if structural else
                    int(row[name + '_range'] > VIOLATION_TOLERANCE))
            changes.append(row)
        if progress and (scene_number % 16 == 0 or scene_number == len(parts)):
            progress(dict(stage='evaluate_dev', scenes=scene_number, total=len(parts)))
    if len(parts) != len(sources):
        raise ValueError('Scene sources do not align')
    for name, rows in [('targets.csv', targets), ('clean_pair_scores.csv', normals),
                       ('triggered_pair_scores.csv', triggered), ('modification_cost.csv', changes)]:
        _csv(folder / name, rows)
    metrics = recompute_metrics(targets, normals, triggered)
    # Independently re-read serialized evidence before accepting the result.
    loaded = []
    for name in ('targets.csv', 'clean_pair_scores.csv', 'triggered_pair_scores.csv'):
        with (folder / name).open() as stream:
            loaded.append(list(csv.DictReader(stream)))
    if recompute_metrics(*loaded) != metrics:
        raise ValueError('Metric serialization/recalculation disagreement')
    if generator_hash != tensor_digest(generator) or hashes != {k: tensor_digest(v) for k, v in models.items()}:
        raise ValueError('Evaluation changed a model')
    summary = dict(status='DEVELOPMENT_COMPARISON_COMPLETE_REVIEW_REQUIRED', method=method,
        metrics=metrics, scenes=len(parts), targets=len(targets), population_identity_sha256=identity_hash(identity),
        normal_pairs=len(normals), triggered_pair_rows=len(triggered), model_state_sha256=hashes,
        generator_state_sha256=generator_hash, synthetic_nodes_in_metric_denominators=False,
        synthetic_nodes_supervised=False,
        source_split='original_train/internal_dev', original_test_arrays_loaded=False,
        original_validation_arrays_loaded=False, internal_audit_arrays_loaded=False,
        clean_reference_used_in_trigger_optimization=False, clean_reference_is_blind_confirmation=False,
        limitation='Development data reused; new clean reference shares each method victim supervision/initialization but is not a blind model',
        topology_preserved=not structural, threshold='binary argmax, class0 on ties',
        copy_check_scope=('not_applicable_to_structural_injection_original_rows_unchanged' if structural
            else 'six original target-pair copy groups; no physical realizability claim'),
        copy_tolerance=VIOLATION_TOLERANCE)
    write_json(folder / 'summary.json', summary)
    return summary
