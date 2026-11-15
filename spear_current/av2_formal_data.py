"""Full eligible AV2 TRAIN, with historically exposed held-out groups excluded.

This adapter uses only original TRAIN graph arrays. DEV and AUDIT are explicitly
reused development populations, never a claim of blind or unused data. Cached
scene batches avoid materializing the complete feature matrix or GPU graph.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import _TrackedGraph, combine_parts, line_graph_from_graph, transform_features
from .risk_labels import RiskLabelConfig
from .runtime import sha256
from .womd_pilot_data import target_index

ROLES = ('train', 'dev', 'audit')
PROJECT = Path(__file__).resolve().parents[1]
ROW_FIELDS = ('scenario_id', 'split_group_id', 'graph_path', 'graph_sha256', 'split',
              'city', 'status', 'role', 'evaluation_subgroup')
_VERIFIED_CACHE = {}


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _bound_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Asset path escapes registered root')
    return path


def _verify_cached(path, expected):
    """Hash once per unchanged file stat in this process, then stream epochs."""
    stat = path.stat()
    identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
    if _VERIFIED_CACHE.get(str(path)) != identity:
        if sha256(path) != expected:
            raise ValueError('Changed cached scene shard or population')
        _VERIFIED_CACHE[str(path)] = identity


def select_rows(rows, historical_roles, confirmation_rows):
    """Keep all built TRAIN scenes outside complete held-out recording groups.

    Selection is metadata-only, independent of labels, motion and predictions.
    All historical held-out identities/hashes must still match the manifest.
    """
    built = {}
    for raw in rows:
        if raw.get('status', '').lower() not in ('built', 'exists'):
            continue
        sid = raw['scenario_id']
        if sid in built:
            raise ValueError('Duplicate scene in source manifest')
        if not raw.get('split_group_id') or len(raw.get('graph_sha256', '')) != 64:
            raise ValueError('Missing source group or graph digest')
        built[sid] = {key: raw.get(key, '') for key in ROW_FIELDS}
    selected = dict(train=[], dev=[], audit=[])
    heldout_groups, heldout_scenes = set(), set()
    for role, subgroup, frozen in (
        ('dev', 'historical_dev128', historical_roles['dev']),
        ('audit', 'historical_audit256', historical_roles['audit']),
        ('audit', 'historical_confirmation1024', confirmation_rows),
    ):
        for old in frozen:
            row = built.get(old['scenario_id'])
            if row is None or row['split'] != 'train':
                raise ValueError('Held-out scene is not an available original TRAIN graph')
            for key in ('graph_path', 'graph_sha256', 'split_group_id'):
                if row[key] != old[key]:
                    raise ValueError('Historical held-out identity changed: ' + key)
            if row['scenario_id'] in heldout_scenes or row['split_group_id'] in heldout_groups:
                raise ValueError('Historical held-out scenes or groups overlap')
            heldout_scenes.add(row['scenario_id'])
            heldout_groups.add(row['split_group_id'])
            selected[role].append(dict(row, role=role, evaluation_subgroup=subgroup))
    selected['train'] = [dict(row, role='train', evaluation_subgroup='')
        for row in built.values() if row['split'] == 'train'
        and row['split_group_id'] not in heldout_groups]
    selected['train'].sort(key=lambda row: (row['split_group_id'], row['scenario_id']))
    if any(not selected[role] for role in ROLES):
        raise ValueError('Every registered role must contain scenes')
    groups = {role: {row['split_group_id'] for row in pool}
              for role, pool in selected.items()}
    if any(groups[a] & groups[b] for a, b in (('train', 'dev'), ('train', 'audit'), ('dev', 'audit'))):
        raise ValueError('Cross-role recording group leakage')
    return selected


def build_contract(root, *, bundle_manifest=None, confirmation_plan=None, seed=20260621):
    root = Path(root).resolve()
    bundle_manifest = Path(bundle_manifest or PROJECT / 'artifacts/current/manifest.json').resolve()
    confirmation_plan = Path(confirmation_plan or PROJECT /
        'results/group_confirmation_v1_20261027/plan.json').resolve()
    manifest = root / 'manifest_grouped_v5.csv'
    metadata = root / 'manifest_grouped_v5.csv.metadata.json'
    meta = json.loads(metadata.read_text())
    if meta['output_manifest_sha256'] != sha256(manifest):
        raise ValueError('Grouped source manifest digest changed')
    if any(meta['split_audit']['leakage_violations'].values()):
        raise ValueError('Source grouped split has recorded leakage violations')
    bundle = json.loads(bundle_manifest.read_text())
    confirmation = json.loads(confirmation_plan.read_text())
    if (Path(bundle['graph_root']).resolve() != root
            or Path(confirmation['protocol']['graph_root']).resolve() != root):
        raise ValueError('Historical source graph roots differ')
    config = asdict(RiskLabelConfig())
    if bundle['config']['label_config'] != config:
        raise ValueError('Historical dynamic-risk label contract changed')
    with manifest.open() as stream:
        selected = select_rows(csv.DictReader(stream), bundle['config']['roles'],
                               confirmation['protocol']['rows'])
    counts = {role: dict(scenes=len(pool), groups=len({r['split_group_id'] for r in pool}),
                        cities=dict(Counter(r['city'] for r in pool)))
              for role, pool in selected.items()}
    contract = dict(schema='av2_formal_full_train_data_v1', target_seed=int(seed),
        graph_root=str(root), source_split='original grouped-v5 TRAIN only',
        sources={str(p): sha256(p) for p in (manifest, metadata, bundle_manifest, confirmation_plan)},
        adapter_source_sha256={str(PROJECT / 'spear_current' / name): sha256(PROJECT / 'spear_current' / name)
            for name in ('av2_formal_data.py', 'data.py', 'pair_graph.py', 'risk_labels.py',
                         'motion_gate.py', 'womd_pilot_data.py')},
        role_counts=counts, label_config=config,
        excluded_training_groups=sorted({r['split_group_id'] for role in ('dev', 'audit')
                                         for r in selected[role]}),
        cross_role_group_intersections={'train_dev': 0, 'train_audit': 0, 'dev_audit': 0},
        train_selection='every built/exists original TRAIN graph outside all DEV/AUDIT recording groups; no label or score filtering',
        evaluation_scope='historically exposed original-TRAIN DEV128 and AUDIT256+confirmation1024; not blind confirmation or test',
        evaluation_subgroups={'dev': ['historical_dev128'],
                             'audit': ['historical_audit256', 'historical_confirmation1024']},
        normalizer_fit='all pair nodes in full registered TRAIN only; population mean/std, scale>=1e-6, clip10',
        clean_supervision='each run seed selects 20% of eligible TRAIN pairs without replacement',
        target_policy='one identity-hash-selected eligible negative per scene, fixed data seed, no model filtering',
        global_order='TRAIN sorted by recording group then scenario ID; canonical pairs sorted by track IDs',
        attachment_budget=dict(count=(26 * len(selected['train']) + 511) // 512,
            formula='ceil(26 * registered_TRAIN_scenes / 512)',
            interpretation='new explicit scene-count scaling of historical absolute 26; not a historical proven poison rate',
            multiple_attachments_per_scene_allowed=True),
        original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
        blind_evaluation=False, full_eligible_registered_train=True)
    return contract, selected


def read_part(root, row, label_config=None):
    """Verify before deserialization; no original validation or test access."""
    if row.get('split') != 'train' or row.get('role') not in ROLES:
        raise ValueError('Only registered original TRAIN rows may be read')
    root = Path(root).resolve()
    path = _bound_path(root, row['graph_path'])
    if not path.is_relative_to(root / 'graphs') or path.suffix != '.npz':
        raise ValueError('Graph is outside source graphs directory')
    if sha256(path) != row['graph_sha256']:
        raise ValueError('Source graph digest mismatch before read')
    with np.load(path, allow_pickle=True) as archive:
        graph = _TrackedGraph(archive)
        if str(graph['scenario_id'].item()) != row['scenario_id']:
            raise ValueError('Source scenario identity mismatch')
        part = line_graph_from_graph(graph, RiskLabelConfig(**(label_config or {})))
        loaded_members = sorted(graph.loaded)
    if sha256(path) != row['graph_sha256']:
        raise ValueError('Source graph changed during read')
    part.update(scenario_id=row['scenario_id'], split='train', role=row['role'],
        split_group_id=row['split_group_id'], evaluation_subgroup=row.get('evaluation_subgroup', ''),
        source_graph_sha256=row['graph_sha256'], source_loaded_members=loaded_members)
    return part


def _save_shard(path, parts):
    temporary = path.with_suffix('.tmp')
    torch.save(parts, temporary)
    temporary.replace(path)
    return dict(path=str(path.name), sha256=sha256(path), scenes=len(parts),
                nodes=sum(len(p['x']) for p in parts),
                edges=sum(p['edge_index'].shape[1] for p in parts),
                global_start=parts[0]['global_offset'], global_stop=parts[-1]['global_stop'])


def prepare(root, destination, seed=20260621, *, bundle_manifest=None,
            confirmation_plan=None, max_nodes=16384, progress=None):
    """Stream all registered graphs, normalize TRAIN only, write bounded shards.

    Destination must be new. Partial state is retained explicitly on failure;
    absence of summary.json means the cache is not ready for training.
    """
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')
    destination = Path(destination).resolve()
    contract, selected = build_contract(root, bundle_manifest=bundle_manifest,
                                       confirmation_plan=confirmation_plan, seed=seed)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _json(destination / 'registration.json', contract)
    for role, rows in selected.items():
        with (destination / f'{role}_manifest.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def emit(stage, **extra):
        value = dict(stage=stage, elapsed_seconds=time.monotonic() - started, **extra)
        _json(destination / 'status.json', value)
        print('[av2-data] ' + json.dumps(value, allow_nan=False), flush=True)
        if progress:
            progress(value)

    summary = dict(schema='av2_formal_full_train_cache_v1', target_seed=int(seed),
        status='PREPARED', original_validation_arrays_loaded=False,
        original_test_arrays_loaded=False, blind_evaluation=False,
        attachment_budget=contract['attachment_budget'], max_nodes_per_shard=max_nodes)
    try:
        for role in ROLES:
            folder = destination / role
            folder.mkdir()
            shards, batch, batch_nodes, offset = [], [], 0, 0
            y_chunks, eligible_chunks, motion_chunks, offsets = [], [], [], []
            stats = dict(scenes=0, groups=contract['role_counts'][role]['groups'],
                         pair_nodes=0, pair_edges=0, eligible=0, positive=0,
                         targets=0, moving_targets=0, zero_pair_scenes=0)
            # Online pooled mean/m2 is equivalent to fitting all TRAIN rows.
            count, mean, m2 = 0, np.zeros(784), np.zeros(784)
            if role != 'train':
                with np.load(destination / 'normalizer.npz', allow_pickle=False) as archive:
                    normalizer = {k: archive[k].copy() for k in archive.files}
            for scene_i, row in enumerate(selected[role]):
                part = read_part(root, row, contract['label_config'])
                n = len(part['x'])
                if batch and batch_nodes + n > max_nodes:
                    shards.append(_save_shard(folder / f'{len(shards):05d}.pt', batch))
                    batch, batch_nodes = [], 0
                    emit('CONVERTING', role=role, completed_scenes=scene_i,
                         total_scenes=len(selected[role]), nodes=offset, shards=len(shards))
                if role == 'train' and n:
                    x = part['x'].astype(np.float64)
                    batch_mean = x.mean(axis=0)
                    delta = batch_mean - mean
                    m2 += ((x - batch_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
                elif role != 'train':
                    part['x'] = transform_features(part['x'], normalizer)
                part['target_index'] = target_index(part, seed)
                part['global_offset'], part['global_stop'] = offset, offset + n
                offsets.append(offset)
                offset += n
                y_chunks.append(part['y'])
                eligible_chunks.append(part['eligible'])
                motion_chunks.append(part['motion_regime'])
                stats['scenes'] += 1
                stats['pair_nodes'] += n
                stats['pair_edges'] += part['edge_index'].shape[1]
                stats['eligible'] += int(part['eligible'].sum())
                stats['positive'] += int((part['y'] == 1).sum())
                stats['zero_pair_scenes'] += int(n == 0)
                target = part['target_index']
                if target is not None:
                    stats['targets'] += 1
                    stats['moving_targets'] += int(part['motion_regime'][target] == 1)
                batch.append(part)
                batch_nodes += n
            if batch:
                shards.append(_save_shard(folder / f'{len(shards):05d}.pt', batch))
            if role == 'train':
                if count == 0:
                    raise ValueError('Full TRAIN has no pair nodes')
                normalizer = dict(mean=mean, scale=np.maximum(np.sqrt(m2 / count), 1e-6),
                    clip=10., minimum_scale=1e-6, feature_dimension=784, fitted_rows=count)
                np.savez(destination / 'normalizer.npz', **normalizer)
                for shard_i, entry in enumerate(shards):
                    path = folder / entry['path']
                    if sha256(path) != entry['sha256']:
                        raise ValueError('Raw training cache changed before normalization')
                    parts = torch.load(path, map_location='cpu', weights_only=False)
                    for part in parts:
                        part['x'] = transform_features(part['x'], normalizer)
                    shards[shard_i] = _save_shard(path, parts)
                    if shard_i % 10 == 0:
                        emit('NORMALIZING_TRAIN', completed_shards=shard_i + 1, total_shards=len(shards))
                summary['normalizer_fitted_rows'] = count
            population = dict(y=torch.from_numpy(np.concatenate(y_chunks)).long(),
                eligible=torch.from_numpy(np.concatenate(eligible_chunks)).bool(),
                motion_regime=torch.from_numpy(np.concatenate(motion_chunks)),
                scene_offsets=torch.tensor(offsets + [offset], dtype=torch.long))
            torch.save(population, destination / f'{role}_population.pt')
            summary[role] = dict(**stats, shards=shards,
                evaluation_subgroups=dict(Counter(row['evaluation_subgroup'] for row in selected[role])))
            emit('ROLE_PREPARED', role=role, **stats)
        for path, expected in contract['sources'].items():
            if sha256(path) != expected:
                raise ValueError('Source contract metadata changed during preparation')
        summary['asset_sha256'] = {p.name: sha256(p) for p in sorted(destination.iterdir())
            if p.is_file() and p.name != 'status.json'}
        summary['elapsed_seconds'] = time.monotonic() - started
        _json(destination / 'summary.json', summary)
        emit('PREPARED', role_counts={r: summary[r]['scenes'] for r in ROLES})
        return summary
    except BaseException as exc:
        emit('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


def _dataset_identity(summary):
    """Identify the registered source, independently of internal role names.

    Legacy AV2 caches omit these fields. WOMD must explicitly declare its
    training_20s source; an internal audit role never permits official test data.
    """
    dataset = summary.get('dataset_id', 'AV2')
    source_split = summary.get('source_dataset_split', 'train')
    if (dataset, source_split) not in (('AV2', 'train'), ('WOMD', 'training_20s')):
        raise ValueError('Unsupported dataset or non-TRAIN source split')
    return dataset, source_split


def dataset_identity(destination):
    """Dataset metadata only; cache integrity is checked by the data readers."""
    summary = json.loads((Path(destination).resolve() / 'summary.json').read_text())
    return _dataset_identity(summary)


def _summary(destination):
    destination = Path(destination).resolve()
    summary = json.loads((destination / 'summary.json').read_text())
    if (summary['status'] != 'PREPARED' or summary['original_test_arrays_loaded']
            or summary.get('test_arrays_loaded', False)):
        raise ValueError('Data cache is not prepared original TRAIN data')
    _dataset_identity(summary)
    return destination, summary


def iter_role_parts(destination, role):
    destination, summary = _summary(destination)
    if role not in ROLES:
        raise ValueError('Unknown registered data role')
    dataset, source_split = _dataset_identity(summary)
    expected_offset = 0
    for entry in summary[role]['shards']:
        path = _bound_path(destination / role, entry['path'])
        _verify_cached(path, entry['sha256'])
        parts = torch.load(path, map_location='cpu', weights_only=False)
        for part in parts:
            if (part['global_offset'] != expected_offset
                    or part['global_stop'] != expected_offset + len(part['x'])
                    or part['split'] != source_split or part['role'] != role
                    or part.get('dataset_id', dataset) != dataset):
                raise ValueError('Cached global ordering or original TRAIN role changed')
            expected_offset = part['global_stop']
            yield part
    if expected_offset != summary[role]['pair_nodes']:
        raise ValueError('Cached role population mismatch')


def iter_role_batches(destination, role, max_nodes=16384, device='cpu'):
    """Yield complete-scene disjoint unions with contiguous global row ranges."""
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')
    parts, nodes = [], 0

    def combine(items):
        data = combine_parts(items)
        data.update(global_start=items[0]['global_offset'], global_stop=items[-1]['global_stop'],
                    parts=items)
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device)
        return data

    for part in iter_role_parts(destination, role):
        n = len(part['x'])
        if parts and nodes + n > max_nodes:
            yield combine(parts)
            parts, nodes = [], 0
        parts.append(part)
        nodes += n
    if parts:
        yield combine(parts)


def load_role_arrays(destination, role, mmap=True):
    destination, summary = _summary(destination)
    if role not in ROLES:
        raise ValueError('Unknown registered data role')
    path = destination / f'{role}_population.pt'
    _verify_cached(path, summary['asset_sha256'][path.name])
    return torch.load(path, map_location='cpu', weights_only=True, mmap=mmap)


def make_indices(destination, seed):
    population = load_role_arrays(destination, 'train')
    eligible = torch.nonzero(population['eligible'], as_tuple=False).flatten().numpy()
    clean = np.sort(np.random.RandomState(seed).permutation(eligible)[:int(.2 * len(eligible))])
    if set(population['y'][clean].tolist()) != {0, 1}:
        raise ValueError('Seeded clean supervision does not contain both labels')
    unlabeled = np.setdiff1d(np.arange(len(population['y']), dtype=np.int64), clean)
    return dict(idx_train=torch.from_numpy(clean), idx_unlabeled=torch.from_numpy(unlabeled))


def load_role(destination, role, device='cpu'):
    """Compatibility materialization; use streaming batches for full TRAIN."""
    parts = list(iter_role_parts(destination, role))
    data = combine_parts(parts)
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
    return parts, data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260621)
    parser.add_argument('--max-nodes', type=int, default=16384)
    args = parser.parse_args()
    prepare(args.graph_root, args.output, args.seed, max_nodes=args.max_nodes)
