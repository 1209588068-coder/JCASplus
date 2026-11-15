"""Nested WOMD data controls for clean risk classifiers only.

The existing TRAIN512 prefix, clean labels, supervision and normalizer are
retained. Additional scenes come from the original TRAIN shard partition.
This module neither constructs triggers nor selects attack targets.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .av2_formal_data import ROLES, _json, _save_shard, iter_role_parts, load_role_arrays
from .data import transform_features
from .runtime import sha256
from .womd_pilot_data import _asset, overlap_audit, read_part, select_rows, validate_source


MOTION_GROUPS = (
    'both_exact_zero', 'both_below_0_5_not_both_zero', 'one_below_0_5',
    'both_at_least_0_5', 'invalid_terminal',
)


def _read(path):
    return json.loads(Path(path).read_text())


def preclip_counts(raw_features, normalizer):
    """Count strict over-bound values before normalization clipping, without edits."""
    raw = np.asarray(raw_features, dtype=np.float64)
    mean = np.asarray(normalizer['mean'], dtype=np.float64)
    scale = np.asarray(normalizer['scale'], dtype=np.float64)
    clip = float(normalizer['clip'])
    if (raw.ndim != 2 or mean.shape != (raw.shape[1],) or scale.shape != mean.shape
            or not np.isfinite(raw).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(clip) or clip <= 0):
        raise ValueError('Invalid raw feature/normalizer clipping audit inputs')
    normalized = (raw - mean) / scale
    if not np.isfinite(normalized).all():
        raise ValueError('Nonfinite normalized input before clipping')
    outside = np.abs(normalized) > clip
    return dict(coordinates=int(raw.size), pairs=len(raw),
                outside_coordinates=int(outside.sum()),
                pairs_with_outside_coordinate=int(outside.any(axis=1).sum()))


def _original_registration(source):
    """Resolve the immutable original scene lists from either clean cache form."""
    registration = _read(source / 'registration.json')
    if 'roles' in registration:
        return registration, source / 'registration.json'
    original = Path(registration['source_data_dir']).resolve() / 'registration.json'
    expected = registration['sources'].get(str(original))
    if expected is None or sha256(original) != expected:
        raise ValueError('Original WOMD scene registration changed')
    return _read(original), original


def motion_groups_from_graph(root, row, pair_track_ids):
    """Five disjoint last-observed-frame groups; zero is an exact report value.

    This is not a claim that the entire history is stationary. Missing or
    nonfinite terminal observations are retained as their own category.
    """
    pair_track_ids = np.asarray(pair_track_ids).astype(str)
    if pair_track_ids.ndim != 2 or pair_track_ids.shape[1] != 2:
        raise ValueError('Expected pair track IDs with shape [N,2]')
    with np.load(_asset(Path(root), row, 'graph'), allow_pickle=False) as graph:
        ids = graph['node_track_ids'].astype(str)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate vehicle IDs')
        index = {value: i for i, value in enumerate(ids)}
        pairs = np.asarray([[index[a], index[b]] for a, b in pair_track_ids],
                           dtype=np.int64).reshape(-1, 2)
        velocity = graph['observed_velocities'][:, -1].astype(np.float64)
        valid = graph['observed_valid_mask'][:, -1].astype(bool)
    valid &= np.isfinite(velocity).all(axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    pair_speed, pair_valid = speed[pairs], valid[pairs].all(axis=1)
    groups = np.full(len(pairs), 4, dtype=np.int8)
    groups[pair_valid & (pair_speed >= .5).all(axis=1)] = 3
    low = pair_speed < .5
    groups[pair_valid & (low.sum(axis=1) == 1)] = 2
    groups[pair_valid & low.all(axis=1)] = 1
    groups[pair_valid & (pair_speed == 0).all(axis=1)] = 0
    return groups


def fine_motion_groups(data_dir, role):
    """Return int8 groups in the exact global node order, without inference."""
    if role not in ROLES:
        raise ValueError('Unknown clean role')
    data_dir = Path(data_dir).resolve()
    population = load_role_arrays(data_dir, role)
    if 'motion_detail' in population:
        groups = population['motion_detail'].numpy()
    else:
        registration, _ = _original_registration(data_dir)
        rows = {row['scenario_id']: row for row in registration['roles'][role]}
        groups = np.concatenate([
            motion_groups_from_graph(registration['graph_root'], rows[part['scenario_id']],
                                     part['canonical_pair_track_ids'])
            for part in iter_role_parts(data_dir, role)
        ])
    if (groups.shape != tuple(population['y'].shape)
            or not np.isin(groups, np.arange(len(MOTION_GROUPS))).all()
            or not np.array_equal(groups == 3, population['motion_regime'].numpy() == 1)):
        raise ValueError('Fine motion groups disagree with registered ordering or Moving')
    return groups.copy()


def coverage_registration(source_data_dir, total_scenes=2048, seed=20261104):
    """Metadata-only fixed nested selection; reject changed source partitions."""
    source = Path(source_data_dir).resolve()
    baseline = _read(source / 'summary.json')
    original, original_path = _original_registration(source)
    if (baseline['status'] != 'PREPARED' or baseline.get('dataset_id') != 'WOMD'
            or baseline.get('source_dataset_split') != 'training_20s'
            or baseline.get('test_arrays_loaded', True)
            or original['source_split'] != 'training_20s'
            or original['source_shard_partition'] != [.6, .2, .2]
            or seed != original['seed'] or type(total_scenes) is not int
            or total_scenes < baseline['train']['scenes']):
        raise ValueError('Unsupported source or changed frozen data partition')
    root = Path(original['graph_root']).resolve()
    provenance = validate_source(root)
    if provenance['manifest.csv'] != original['manifest_sha256']:
        raise ValueError('Original source manifest changed')
    with (root / 'manifest.csv').open() as stream:
        manifest_rows = list(csv.DictReader(stream))
    sizes = {role: len(original['roles'][role]) for role in ROLES}
    sizes['train'] = total_scenes
    # select_rows partitions ALL BUILT source shards before sampling scenes.
    selected = select_rows(manifest_rows, seed, sizes)
    n_original = len(original['roles']['train'])
    if (selected['train'][:n_original] != original['roles']['train']
            or selected['dev'] != original['roles']['dev']
            or selected['audit'] != original['roles']['audit']):
        raise ValueError('Expanded selection does not preserve the frozen prefix/held-out roles')
    if len({r['scenario_id'] for rs in selected.values() for r in rs}) != sum(sizes.values()):
        raise ValueError('Duplicate scenario identity in registered roles')
    registration = dict(
        schema='womd_clean_nested_coverage_v1', dataset_id='WOMD',
        source_dataset_split='training_20s', source_split='training_20s', seed=seed,
        graph_root=str(root), source_data_dir=str(source),
        original_registration_path=str(original_path),
        original_registration_sha256=sha256(original_path),
        source_summary_sha256=sha256(source / 'summary.json'),
        source_provenance_sha256=provenance,
        manifest_sha256=provenance['manifest.csv'], source_shard_partition=[.6, .2, .2],
        roles=selected, role_sizes=sizes, original_train_scenes=n_original,
        scene_selection='original shard split and scene hash ordering; retain original TRAIN prefix; no redraw',
        normalizer_policy='byte-identical original TRAIN512 normalizer; do not refit',
        coverage_interpretation='more scenes and supervision with more batches per epoch; final old-prefix batch may include added scenes; not an isolated diversity effect',
        clean_supervision='retain original idx_train; independently sample floor(0.2 * added eligible pairs) with RandomState(data seed)',
        purpose='clean risk-classifier coverage control; no trigger or attack training',
        scope='previously exposed internal DEV/AUDIT, not official test or blind confirmation',
        recording_independence_proven=False, test_arrays_loaded=False,
        original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
        adapter_source_sha256=sha256(__file__),
    )
    return registration


def prepare(source_data_dir, destination, total_scenes=2048, seed=20261104,
            max_nodes=16384):
    """Create a fresh compatible streaming cache for a clean coverage control."""
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('Invalid max_nodes')
    source, destination = Path(source_data_dir).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('New cache must not modify source data')
    registration = coverage_registration(source, total_scenes, seed)
    baseline = _read(source / 'summary.json')
    for name in ('normalizer.npz', 'indices.pt'):
        if sha256(source / name) != baseline['asset_sha256'][name]:
            raise ValueError('Frozen baseline normalizer or indices changed')
    indices = torch.load(source / 'indices.pt', map_location='cpu', weights_only=True)
    old_clean = indices['idx_train'].clone()
    if (old_clean.dtype != torch.long or old_clean.ndim != 1
            or len(old_clean) != baseline['clean_supervised_pairs']
            or len(torch.unique(old_clean)) != len(old_clean)):
        raise ValueError('Invalid original clean supervision')
    old_train = load_role_arrays(source, 'train')
    if (len(old_clean) == 0 or int(old_clean.min()) < 0
            or int(old_clean.max()) >= len(old_train['y'])
            or not bool(old_train['eligible'][old_clean].all())):
        raise ValueError('Original clean supervision is outside eligible TRAIN')
    with np.load(source / 'normalizer.npz', allow_pickle=False) as z:
        normalizer = {k: z[k].copy() for k in z.files}
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _json(destination / 'registration.json', registration)
    _json(destination / 'status.json', dict(stage='PREPARING_CLEAN_COVERAGE'))
    summary = dict(
        schema='womd_clean_nested_streaming_cache_v1', status='PREPARED', dataset_id='WOMD',
        source_dataset_split='training_20s', seed=seed, target_seed=seed,
        test_arrays_loaded=False, original_validation_arrays_loaded=False,
        original_test_arrays_loaded=False, blind_evaluation=False,
        recording_independence_proven=False, max_nodes_per_shard=max_nodes,
        normalizer_fitted_rows=baseline['normalizer_fitted_rows'], new_statistics_fitted=0,
        original_train_scenes=registration['original_train_scenes'],
        original_train_pair_nodes=baseline['train']['pair_nodes'],
        original_clean_supervised_pairs=len(old_clean), motion_groups=list(MOTION_GROUPS),
        normalizer_policy=registration['normalizer_policy'], clean_only=True,
        coverage_interpretation=registration['coverage_interpretation'],
    )
    added_preclip = dict(coordinates=0, pairs=0, outside_coordinates=0,
                         pairs_with_outside_coordinate=0)
    try:
        root = Path(registration['graph_root'])
        overlap = overlap_audit(root, registration['roles'])
        _json(destination / 'overlap_audit.json', overlap)
        if overlap['cross_role_conflicts']:
            raise ValueError('Expanded coverage overlaps held-out roles; no replacement sampling')
        shutil.copyfile(source / 'normalizer.npz', destination / 'normalizer.npz')
        train_population = None
        for role in ROLES:
            rows = registration['roles'][role]
            lookup = {r['scenario_id']: r for r in rows}
            folder = destination / role
            folder.mkdir()
            chunks = {k: [] for k in ('y', 'eligible', 'motion_regime', 'motion_detail')}
            offsets, shards, pending, pending_nodes, offset = [], [], [], 0, 0
            stats = dict(scenes=0, pair_nodes=0, pair_edges=0, eligible=0, positive=0,
                         targets=0, moving_targets=0, zero_pair_scenes=0,
                         groups=len({r['source_shard'] for r in rows}))

            def parts():
                yield from iter_role_parts(source, role)
                if role == 'train':
                    for row in rows[registration['original_train_scenes']:]:
                        part = read_part(root, row)
                        for key, count in preclip_counts(part['x'], normalizer).items():
                            added_preclip[key] += count
                        part['x'] = transform_features(part['x'], normalizer)
                        # Normal-task controls do not select new attack targets.
                        part['target_index'] = None
                        part.update(split='training_20s', role=role, dataset_id='WOMD',
                                    source_dataset_split='training_20s',
                                    evaluation_subgroup='clean_coverage_added_train')
                        yield part

            for i, part in enumerate(parts()):
                if i >= len(rows) or part['scenario_id'] != rows[i]['scenario_id']:
                    raise ValueError('Scene ordering disagrees with clean registration')
                n = len(part['x'])
                if pending and pending_nodes + n > max_nodes:
                    shards.append(_save_shard(folder / f'{len(shards):05d}.pt', pending))
                    pending, pending_nodes = [], 0
                group = motion_groups_from_graph(root, lookup[part['scenario_id']],
                                                 part['canonical_pair_track_ids'])
                if not np.array_equal(group == 3, part['motion_regime'] == 1):
                    raise ValueError('Motion metadata disagrees with raw observed velocity')
                part.update(global_offset=offset, global_stop=offset + n)
                offsets.append(offset)
                offset += n
                pending.append(part)
                pending_nodes += n
                for key in ('y', 'eligible', 'motion_regime'):
                    chunks[key].append(part[key])
                chunks['motion_detail'].append(group)
                stats['scenes'] += 1
                stats['pair_nodes'] += n
                stats['pair_edges'] += part['edge_index'].shape[1]
                stats['eligible'] += int(part['eligible'].sum())
                stats['positive'] += int((part['y'] == 1).sum())
                stats['zero_pair_scenes'] += int(n == 0)
                target = part.get('target_index')
                if target is not None:
                    stats['targets'] += 1
                    stats['moving_targets'] += int(part['motion_regime'][target] == 1)
                if (i + 1) % 128 == 0:
                    _json(destination / 'status.json', dict(stage='PREPARING_CLEAN_COVERAGE',
                          role=role, scenes=i + 1, total=len(rows)))
            if pending:
                shards.append(_save_shard(folder / f'{len(shards):05d}.pt', pending))
            if stats['scenes'] != len(rows):
                raise ValueError('Missing registered clean scenes')
            population = {k: torch.from_numpy(np.concatenate(v)) for k, v in chunks.items()}
            population['scene_offsets'] = torch.tensor(offsets + [offset], dtype=torch.long)
            original_population = load_role_arrays(source, role)
            prefix = baseline[role]['pair_nodes']
            for key in ('y', 'eligible', 'motion_regime'):
                if not torch.equal(population[key][:prefix], original_population[key]):
                    raise ValueError('Original label/eligibility/motion prefix changed')
            if not torch.equal(population['scene_offsets'][:baseline[role]['scenes'] + 1],
                               original_population['scene_offsets']):
                raise ValueError('Original scene boundary prefix changed')
            torch.save(population, destination / f'{role}_population.pt')
            summary[role] = dict(**stats, shards=shards,
                                motion_group_counts=dict(Counter(map(int, population['motion_detail'].tolist()))))
            if role == 'train':
                train_population = population
        assert train_population is not None
        prefix = baseline['train']['pair_nodes']
        eligible_added = torch.nonzero(train_population['eligible'][prefix:]).flatten().numpy() + prefix
        extra = np.sort(np.random.RandomState(seed).permutation(eligible_added)[:int(.2 * len(eligible_added))])
        clean = torch.cat((old_clean, torch.from_numpy(extra).long()))
        if not torch.equal(clean[:len(old_clean)], old_clean):
            raise ValueError('Original supervision prefix changed')
        if set(train_population['y'][clean].tolist()) != {0, 1}:
            raise ValueError('Clean coverage requires both task labels')
        unlabeled = np.setdiff1d(np.arange(summary['train']['pair_nodes']), clean.numpy())
        torch.save(dict(idx_train=clean), destination / 'clean_indices.pt')
        torch.save(dict(idx_train=clean, idx_unlabeled=torch.from_numpy(unlabeled)), destination / 'indices.pt')
        summary.update(clean_supervised_pairs=len(clean), added_clean_supervised_pairs=len(extra),
                       clean_class_counts=torch.bincount(train_population['y'][clean], minlength=2).tolist(),
                       original_prefix_preserved=True, elapsed_seconds=time.monotonic() - started)
        summary['added_train_preclip'] = dict(
            **added_preclip, clip=float(normalizer['clip']),
            measurement='abs((raw_float64-mean)/scale) > clip, before clipping; exact boundary values excluded',
            coordinate_fraction=(added_preclip['outside_coordinates'] / added_preclip['coordinates']
                                 if added_preclip['coordinates'] else None),
            pair_fraction=(added_preclip['pairs_with_outside_coordinate'] / added_preclip['pairs']
                           if added_preclip['pairs'] else None))
        if (sha256(source / 'summary.json') != registration['source_summary_sha256']
                or sha256(root / 'manifest.csv') != registration['manifest_sha256']
                or sha256(destination / 'normalizer.npz') != baseline['asset_sha256']['normalizer.npz']):
            raise ValueError('Frozen input provenance changed during preparation')
        summary['asset_sha256'] = {p.name: sha256(p) for p in sorted(destination.iterdir())
                                  if p.is_file() and p.name != 'status.json'}
        _json(destination / 'summary.json', summary)
        _json(destination / 'status.json', dict(stage='PREPARED', elapsed_seconds=time.monotonic() - started))
        return summary
    except BaseException as exc:
        _json(destination / 'status.json', dict(stage='FAILED', error_type=type(exc).__name__, error=str(exc)))
        raise
