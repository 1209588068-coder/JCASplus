"""Fixed-checkpoint original-TRAIN train/dev response diagnosis; no optimization.

Registered training attachments/outer rows are evaluated jointly AND separately
from a prediction-independent one-negative-per-scene diagnostic population.
No audit, confirmation, original validation or test graphs are read. This is
development evidence, not a new independent performance confirmation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from .attention_context import attention_context, permutation_stats
from .data import combine_parts, load_normalizer, read_role_part
from .development_metrics import population_summary, response_metrics, compare_models
from .evaluate import SceneInjector, state_hash
from .evaluation import _component, _normal_metrics
from .role_data import build_contexts, _tensor_hash as context_tensor_hash
from .role_training import _tensor_hash
from .runtime import (load_bundle, sha256, restore_generator, restore_surrogate,
                      restore_victim, restore_clean_reference)
from .train import check_outer


SEEDS = (20260621, 20260622, 20260623)
MODELS = ('surrogate', 'victim', 'clean_reference')
ROOT = Path(__file__).resolve().parents[1]
TARGET_SEED = 20260621


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _asset(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def _progress(**record):
    print(json.dumps(record), flush=True)


def output_directory(output):
    output = Path(output).resolve()
    if output == ROOT / 'results' or not output.is_relative_to(ROOT / 'results'):
        raise ValueError('Diagnostic output must be a new child of results/')
    output.mkdir(parents=True, exist_ok=False)
    return output


def registered_rows(bundle):
    """Reject any unexpected source role before the first graph is opened."""
    all_scenes, all_groups = set(), set()
    for role, count in (('train', 512), ('dev', 128), ('audit', 256)):
        rows = bundle.config['roles'][role]
        if len(rows) != count:
            raise ValueError('Frozen source population changed')
        for row in rows:
            scene, group = row['scenario_id'], row['split_group_id']
            if (row.get('split') != 'train' or row.get('role') != role or not scene or not group
                    or scene in all_scenes or group in all_groups):
                raise ValueError('Changed, overlapping or non-TRAIN source identity')
            all_scenes.add(scene)
            all_groups.add(group)
    return {role: bundle.config['roles'][role] for role in ('train', 'dev')}


def choose_target(part, seed=TARGET_SEED):
    """The existing identity-hash rule; called before any model is restored."""
    _, _, y, eligible, ids, _ = _component(part)
    candidates = np.flatnonzero(eligible & (y == 0))
    return int(min(candidates, key=lambda i: hashlib.sha256(
        f"{seed}:spear-audit-target:{part['scenario_id']}:{':'.join(ids[i])}".encode()
    ).hexdigest())) if len(candidates) else None


def row_metadata(part, source, offset, role, attachments, outer):
    _, _, labels, eligible, ids, motion = _component(part)
    return [dict(scenario_id=source['scenario_id'], split_group_id=source['split_group_id'],
        role=role, pair_index=i, role_pair_index=offset + i,
        src_track_id=str(ids[i, 0]), dst_track_id=str(ids[i, 1]),
        true_label=int(labels[i]), eligible=bool(eligible[i]),
        motion_regime='moving' if motion[i] else 'slow',
        is_attachment=role == 'train' and offset + i in attachments,
        is_outer=role == 'train' and offset + i in outer) for i in range(len(labels))]


def read_inputs(bundle, sources, ledger):
    normalizer = load_normalizer(bundle)
    parts, combined = {}, {}
    for role in ('train', 'dev'):
        parts[role] = []
        for i, row in enumerate(sources[role], 1):
            parts[role].append(read_role_part(bundle, row, role, normalizer, ledger))
            if i % 64 == 0 or i == len(sources[role]):
                _progress(stage='read_' + role, completed=i, total=len(sources[role]))
        combined[role] = combine_parts(parts[role])
    train, dev = combined['train'], combined['dev']
    indices = torch.load(bundle.asset_path('indices'), weights_only=True, map_location='cpu')
    expected = json.loads(bundle.asset_path('input_contract').read_text())
    for key, field in (('x', 'clean_features_sha256'), ('y', 'clean_labels_sha256')):
        if hashlib.sha256(train[key].numpy().tobytes()).hexdigest() != expected[field]:
            raise ValueError('Original TRAIN bytes changed: ' + key)
    coordinates = torch.tensor(bundle.manifest['selected_coordinates'], dtype=torch.long)
    outer = check_outer(train, indices, expected, coordinates, bundle.config['seed'])
    dev_indices = torch.nonzero(dev['eligible'], as_tuple=False).flatten() + len(train['x'])
    if not torch.equal(dev_indices, indices['dev_indices']):
        raise ValueError('Original dev supervision changed')
    views = [{key: value for key, value in p.items() if key != 'split'} for p in parts['train']]
    contexts = build_contexts(views, train, seed=bundle.config['shuffle_seed'])
    context = attention_context(contexts['typed'], contexts['binding']['scene_order'],
                                bundle.config['shuffle_seed'])
    binding = json.loads(bundle.asset_path('context_binding').read_text())
    permutation = torch.load(bundle.asset_path('score_permutation'), weights_only=True, map_location='cpu')
    if (contexts['binding'] != binding['original_binding']
            or not torch.equal(context.score_permutation, permutation)
            or context_tensor_hash(context.score_permutation) != binding['score_permutation_sha256']
            or permutation_stats(context) != binding['statistics']):
        raise ValueError('Original context binding changed')
    if len(ledger) != 640 or any(r['split'] != 'train' or r['role'] not in ('train', 'dev')
                               or r['status'] != 'READ_COMPLETE' for r in ledger):
        raise ValueError('Unexpected diagnostic graph access')
    attach_set, outer_set = set(indices['idx_attach'].tolist()), set(outer.tolist())
    metadata, targets = {}, {}
    for role in ('train', 'dev'):
        metadata[role], targets[role] = [], []
        for part, row, offset in zip(parts[role], sources[role], combined[role]['offsets']):
            metadata[role].extend(row_metadata(part, row, offset, role, attach_set, outer_set))
            targets[role].append(choose_target(part))
    populations = dict(train_all=metadata['train'], dev_all=metadata['dev'],
        attachments=[metadata['train'][i] for i in indices['idx_attach'].tolist()],
        outer_nonattachments=[metadata['train'][i] for i in outer.tolist() if i not in attach_set])
    for role in ('train', 'dev'):
        populations[role + '_public_single'] = [metadata[role][offset + t]
            for offset, t in zip(combined[role]['offsets'], targets[role]) if t is not None]
    return dict(parts=parts, combined=combined, indices=indices, outer=outer, context=context,
                metadata=metadata, targets=targets, populations=populations)


def predict(model, x, edges, weights):
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Require frozen eval model')
    with torch.no_grad():
        out = model(x, edges, weights)
    if (out.shape != (len(x), 2) or not bool(torch.isfinite(out).all())
            or not torch.allclose(out.logsumexp(1), out.new_zeros(len(x)), atol=1e-5)):
        raise ValueError('Invalid log-probabilities')
    return dict(p=out[:, 1].exp().cpu().numpy(), pred=out.argmax(1).cpu().numpy(),
                nll=(-out[:, 1]).cpu().numpy())


def check_injection(original, changed, targets, generator, edges, new_edges, weights, new_weights):
    allowed = torch.zeros_like(original, dtype=torch.bool)
    support = generator.selected_coordinates[generator.hard_support]
    allowed[targets[:, None], support] = True
    if (changed.shape != original.shape or not bool(torch.isfinite(changed).all())
            or not torch.equal(changed[~allowed], original[~allowed])
            or bool((changed[allowed].abs() > 10).any())
            or not torch.equal(edges, new_edges) or not torch.equal(weights, new_weights)):
        raise ValueError('Injection changed frozen scope, graph or domain')


def response_row(metadata, index, original, changed, generator, clean, triggered):
    row = dict(metadata)
    support = generator.selected_coordinates[generator.hard_support]
    before, after = original[index, support].double(), changed[index, support].double()
    difference = (after - before).abs()
    row.update(changed_coordinates=int((difference > 0).sum()),
        boundary_coordinates=int((after.abs() == 10).sum()),
        delta_l1=float(difference.sum()), delta_linf=float(difference.max()))
    for name in MODELS:
        row.update({name + '_clean_p': float(clean[name]['p'][index]),
                    name + '_trigger_p': float(triggered[name]['p'][index]),
                    name + '_clean_pred': int(clean[name]['pred'][index]),
                    name + '_trigger_pred': int(triggered[name]['pred'][index]),
                    name + '_target_nll': float(triggered[name]['nll'][index])})
    return row


def summarize(rows):
    return dict(population=population_summary(rows),
                models={name: response_metrics(rows, name) for name in MODELS},
                surrogate_victim_flip_agreement=compare_models(rows))


def _csv(path, rows):
    # Real registered populations are nonempty; handle empty synthetic inputs.
    with Path(path).open('x', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _input_hashes(inputs):
    result = {}
    for key in ('metadata', 'targets', 'populations'):
        if key in inputs:
            result[key] = hashlib.sha256(json.dumps(inputs[key], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for role in ('train', 'dev'):
        for key in ('x', 'y', 'eligible', 'edge_index'):
            result[role + '_' + key] = _tensor_hash(inputs['combined'][role][key])
    for key in ('edge_index', 'edge_type', 'pair_scene', 'score_permutation'):
        result['context_' + key] = _tensor_hash(getattr(inputs['context'], key))
    result['outer'] = _tensor_hash(inputs['outer'])
    for key, value in inputs['indices'].items():
        if isinstance(value, torch.Tensor):
            result['indices_' + key] = _tensor_hash(value)
    for role in ('train', 'dev'):
        for i, part in enumerate(inputs['parts'][role]):
            for key in ('x', 'y', 'eligible', 'edge_index', 'motion_regime', 'canonical_pairs'):
                result[f'{role}_part{i}_{key}'] = _tensor_hash(torch.as_tensor(part[key]))
    return result


def run_seed(bundle, inputs, seed, directory):
    directory.mkdir(exist_ok=False)
    generator = restore_generator(bundle, seed)
    models = dict(surrogate=restore_surrogate(bundle, seed), victim=restore_victim(bundle, seed),
                  clean_reference=restore_clean_reference(bundle))
    all_models = dict(generator=generator, **models)
    before = {name: state_hash(model) for name, model in all_models.items()}
    input_before, rng = _input_hashes(inputs), torch.get_rng_state().clone()
    cases, normal = {}, {}
    train = inputs['combined']['train']
    x, edges, context = train['x'], train['edge_index'], inputs['context']
    weights = torch.ones(edges.shape[1])
    with torch.no_grad():
        clean = {name: predict(model, x, edges, weights) for name, model in models.items()}
        for name, targets in (('train_attachments_joint', inputs['indices']['idx_attach']),
                              ('train_outer_joint', inputs['outer'])):
            injected = generator.inject(x, context, targets, weights, detach=True)
            check_injection(x, injected.features, targets, generator, edges, injected.edge_index,
                            weights, injected.edge_weight)
            triggered = {name: predict(model, injected.features, edges, weights)
                         for name, model in models.items()}
            cases[name] = [response_row(inputs['metadata']['train'][i], i, x, injected.features,
                                       generator, clean, triggered) for i in targets.tolist()]
            _progress(stage=name, seed=seed, targets=len(targets))
        cases['train_outer_nonattachments_joint'] = [r for r in cases['train_outer_joint'] if not r['is_attachment']]
        attack = SceneInjector(generator, bundle.config['shuffle_seed'])
        for role in ('train', 'dev'):
            records, normal[role] = [], []
            for count, (part, offset, target) in enumerate(zip(inputs['parts'][role],
                    inputs['combined'][role]['offsets'], inputs['targets'][role]), 1):
                scene_x, scene_edges, truth, eligible, _, _ = _component(part)
                scene_weights = torch.ones(scene_edges.shape[1])
                clean_scene = {name: predict(model, scene_x, scene_edges, scene_weights)
                               for name, model in models.items()}
                for i in np.flatnonzero(eligible):
                    record = dict(inputs['metadata'][role][offset + int(i)], label=int(truth[i]))
                    for name in MODELS:
                        record.update({name + '_p': float(clean_scene[name]['p'][i]),
                                       name + '_pred': int(clean_scene[name]['pred'][i])})
                    normal[role].append(record)
                if target is not None:
                    attack.bind(part)
                    targets = torch.tensor([target], dtype=torch.long)
                    changed, fe, fw = attack.inject_trigger(targets, scene_x, scene_edges,
                                                            scene_weights, torch.device('cpu'))
                    check_injection(scene_x, changed, targets, generator, scene_edges, fe, scene_weights, fw)
                    triggered = {name: predict(model, changed, fe, fw) for name, model in models.items()}
                    records.append(response_row(inputs['metadata'][role][offset + target], target,
                                   scene_x, changed, generator, clean_scene, triggered))
                if count % 64 == 0 or count == len(inputs['parts'][role]):
                    _progress(stage=role + '_single', seed=seed, completed=count,
                              total=len(inputs['parts'][role]))
            cases[role + '_public_single'] = records
        cases['train_public_nonattachment_single'] = [r for r in cases['train_public_single'] if not r['is_attachment']]
    if (before != {name: state_hash(model) for name, model in all_models.items()}
            or input_before != _input_hashes(inputs) or not torch.equal(rng, torch.get_rng_state())
            or any(p.grad is not None or p.requires_grad for model in all_models.values() for p in model.parameters())
            or any(m.training for model in all_models.values() for m in model.modules())):
        raise ValueError('Diagnostic changed inputs, model state, gradients, modes or RNG')
    raw = []
    for name, records in cases.items():
        # Overlapping views intentionally repeat rows across distinct named cases.
        raw.extend(dict(case=name, **r) for r in records)
    _csv(directory / 'responses.csv', raw)
    _csv(directory / 'clean_pair_scores.csv', normal['train'] + normal['dev'])
    result = dict(seed=seed, status='DEVELOPMENT_DIAGNOSTIC_COMPLETE',
        cases={name: summarize(rows) for name, rows in cases.items()},
        normal={role: {name: _normal_metrics(rows, name) for name in MODELS}
                for role, rows in normal.items()}, weights_unchanged=True, inputs_unchanged=True,
        rng_unchanged=True, optimizer_steps=0, model_state_hashes=before,
        raw_responses=_asset(directory / 'responses.csv'), raw_clean=_asset(directory / 'clean_pair_scores.csv'))
    _write(directory / 'summary.json', result)
    return result


def run(bundle, output):
    sources = registered_rows(bundle)
    output = output_directory(output)
    ledger = []
    plan = dict(version='moving-development-fixed-checkpoint-v1',
        at=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
        sources=sources, bundle=_asset(bundle.manifest_path),
        code={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        target_seed=TARGET_SEED, target_rule='existing SHA identity; one eligible true-negative per scene; no prediction filter',
        joint_scope='all fixed 26 attachments / all fixed 538 outer simultaneously; not public single-target ASR',
        moving_definition='both terminal observations valid and finite; both terminal speeds >=0.5 m/s under existing motion_pair_gate',
        optimizer_steps=0, checkpoint_selection='none; retained final models only',
        original_val_arrays_allowed=False, original_test_arrays_allowed=False,
        internal_audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        performance_gate='none; diagnostic evidence, not a candidate selection gate',
        automatic_followup=False)
    _write(output / 'plan.json', plan)
    try:
        inputs = read_inputs(bundle, sources, ledger)
        _write(output / 'population.json', dict(
            populations={name: population_summary(rows) for name, rows in inputs['populations'].items()},
            scenes={role: dict(registered=len(sources[role]),
                with_target=sum(t is not None for t in inputs['targets'][role]),
                without_target=sum(t is None for t in inputs['targets'][role])) for role in sources},
            input_hashes=_input_hashes(inputs)))
        # Freeze complete diagnostic identities before restoring any model.
        _write(output / 'populations_and_targets.json', inputs['populations'])
        reports = []
        for seed in SEEDS:
            run_seed(bundle, inputs, seed, output / f'seed{seed}')
            reports.append(_asset(output / f'seed{seed}' / 'summary.json'))
        if plan['bundle'] != _asset(bundle.manifest_path) or plan['code'] != {
                p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}:
            raise ValueError('Source or manifest changed during diagnosis')
        for key in bundle.manifest['assets']:
            bundle.asset_path(key)
        _write(output / 'completion.json', dict(status='DEVELOPMENT_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED',
            at=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS), reports=reports,
            graph_reads=len(ledger), optimizer_steps=0, original_val_arrays_loaded=False,
            original_test_arrays_loaded=False, internal_audit_arrays_loaded=False,
            confirmation_arrays_loaded=False, automatic_followup=False))
    except Exception as error:
        _write(output / 'failure.json', dict(error=str(error), type=type(error).__name__))
        raise
    finally:
        _write(output / 'graph_access_ledger.json', ledger)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    bundle = load_bundle()
    registered_rows(bundle)
    if not args.run:
        print('DEVELOPMENT_DIAGNOSTIC_PREFLIGHT_PASS; no graphs, models or optimization')
        return
    if args.output_dir is None:
        parser.error('--run requires --output-dir')
    torch.set_num_threads(2)
    print('Review: ' + str(run(bundle, args.output_dir)), flush=True)


if __name__ == '__main__':
    main()
