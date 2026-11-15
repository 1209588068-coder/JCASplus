"""Registered WOMD development pilot data; no inherited AV2 statistics or lists.

Source-shard separation plus detected exact-map/SDC-window overlap checks is
conservative engineering, NOT a guarantee of independent source recordings.
Only TRAIN arrays fit normalization. Source SDC arrays are audit-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import combine_parts, transform_features
from .runtime import sha256

ROLE_SIZES = {'train': 512, 'dev': 128, 'audit': 256}


def validate_source(root):
    root = Path(root).resolve()
    plan, registration, completion, summary = [json.loads((root / name).read_text())
        for name in ('plan.json', 'registration.json', 'completion.json', 'summary.json')]
    plan_hash = sha256(root / 'plan.json')
    if (registration['plan_sha256'] != plan_hash or completion['plan_sha256'] != plan_hash
            or summary['plan_sha256'] != plan_hash
            or completion['summary_sha256'] != sha256(root / 'summary.json')
            or summary['manifest_sha256'] != sha256(root / 'manifest.csv')
            or completion['status'] != 'GRAPH_BUILD_COMPLETE_SPLIT_REVIEW_REQUIRED'
            or plan['source_dataset_split'] != 'training_20s'
            or plan['dataset'] != 'Waymo Open Motion Dataset'
            or plan['test_arrays_loaded'] or summary['test_arrays_loaded']):
        raise ValueError('Unverified WOMD training graph provenance')
    return {name: sha256(root / name) for name in
            ('plan.json', 'registration.json', 'completion.json', 'summary.json', 'manifest.csv')}


def _key(seed, *values):
    return hashlib.sha256('|'.join(map(str, (seed, *values))).encode()).hexdigest()


def _asset(root, row, kind):
    path = (root / row[kind + '_path']).resolve()
    if not path.is_relative_to(root) or sha256(path) != row[kind + '_sha256']:
        raise ValueError(f'Changed or invalid {kind} asset: {path}')
    return path


def select_rows(rows, seed, sizes=ROLE_SIZES):
    """Split source shards first, then hash-sample scenes without labels/scores."""
    rows = [r for r in rows if r['status'] == 'BUILT']
    shards = sorted({r['source_shard'] for r in rows}, key=lambda s: _key(seed, 'shard', s))
    if len(shards) < 10:
        raise ValueError('Insufficient source shards for the registered pilot')
    a, b = int(.6 * len(shards)), int(.8 * len(shards))
    roles = {s: role for role, block in zip(('train', 'dev', 'audit'),
             (shards[:a], shards[a:b], shards[b:])) for s in block}
    selected = {}
    for role, size in sizes.items():
        pool = sorted((r for r in rows if roles[r['source_shard']] == role),
                      key=lambda r: _key(seed, 'scene', r['scenario_id']))
        if len(pool) < size:
            raise ValueError(f'Insufficient scenes for {role}')
        selected[role] = pool[:size]
    return selected


def overlap_audit(root, selected):
    """Reject observed cross-role duplicate payload/map/moving-SDC fragments.

The SDC key is a 10-sample window of absolute xy, quantized to .25m. Windows
with under 2m maximum displacement are skipped to avoid stationary-origin
collisions. Arbitrary coordinate frames and timestamp offsets can hide
overlaps; this diagnostic does not certify recording independence.
"""
    seen, conflicts = {}, []
    counts = {'scenes': 0, 'sdc_windows': 0, 'stationary_windows_skipped': 0}
    for role, rows in selected.items():
        for row in rows:
            sid = row['scenario_id']
            keys = [('payload', row['payload_sha256']),
                    ('map', row['map_fingerprint_sha256'])]
            with np.load(_asset(root, row, 'source_audit'), allow_pickle=False) as z:
                pos, valid = z['source_sdc_position'], z['source_sdc_valid']
                for t in range(max(0, len(pos) - 9)):
                    w = pos[t:t + 10]
                    if not valid[t:t + 10].all() or not np.isfinite(w).all():
                        continue
                    if np.linalg.norm(w - w[0], axis=1).max() < 2.0:
                        counts['stationary_windows_skipped'] += 1
                        continue
                    quantized = np.rint(w / .25).astype('<i8')
                    keys.append(('sdc_window', hashlib.sha256(quantized.tobytes()).hexdigest()))
                    counts['sdc_windows'] += 1
            for key in set(keys):
                if key in seen and seen[key][0] != role:
                    conflicts.append(dict(kind=key[0], first=seen[key], second=[role, sid]))
                else:
                    seen[key] = [role, sid]
            counts['scenes'] += 1
    return dict(**counts, cross_role_conflicts=conflicts,
                recording_independence_proven=False,
                scope='selected development pilot only; no labels or predictions used')


def read_part(root, row):
    with np.load(_asset(root, row, 'pair_graph'), allow_pickle=False) as z:
        part = {k: z[k].copy() for k in (
            'x', 'y', 'eligible', 'edge_index', 'canonical_pairs',
            'canonical_pair_track_ids', 'motion_regime', 'message_passing_pair_mask')}
        if str(z['scenario_id'].item()) != row['scenario_id']:
            raise ValueError('Scenario identity mismatch')
    n = len(part['x'])
    if part['x'].shape != (n, 784) or not np.isfinite(part['x']).all():
        raise ValueError('Invalid observation features')
    if (part['y'].shape != (n,) or part['y'].dtype.kind not in 'iu'
            or not np.isin(part['y'], [-1, 0, 1]).all()
            or part['eligible'].shape != (n,) or part['eligible'].dtype != np.bool_
            or not np.array_equal(part['eligible'], part['y'] >= 0)):
        raise ValueError('Eligibility/label mismatch')
    edge = part['edge_index']
    if (edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype.kind not in 'iu'
            or (edge.size and (edge.min() < 0 or edge.max() >= n))):
        raise ValueError('Invalid pair adjacency')
    if (part['canonical_pairs'].shape != (n, 2)
            or part['canonical_pair_track_ids'].shape != (n, 2)
            or len(set(map(tuple, part['canonical_pair_track_ids'].tolist()))) != n
            or part['motion_regime'].shape != (n,)
            or not np.isin(part['motion_regime'], [0, 1]).all()
            or part['message_passing_pair_mask'].shape != (n,)
            or part['message_passing_pair_mask'].dtype != np.bool_):
        raise ValueError('Invalid identity/motion/message metadata')
    part.update(scenario_id=row['scenario_id'], source_shard=row['source_shard'],
                split_group_id='source_shard:' + row['source_shard'])
    return part


def fit_normalizer(parts):
    count, mean, m2 = 0, np.zeros(784), np.zeros(784)
    for part in parts:
        x = part['x'].astype(np.float64)
        if not len(x):
            continue
        k, batch_mean = len(x), x.mean(axis=0)
        delta = batch_mean - mean
        m2 += ((x - batch_mean) ** 2).sum(axis=0) + delta ** 2 * count * k / (count + k)
        mean += delta * k / (count + k)
        count += k
    if count == 0:
        raise ValueError('Empty TRAIN normalization population')
    return dict(mean=mean, scale=np.maximum(np.sqrt(m2 / count), 1e-6),
                clip=10., minimum_scale=1e-6, feature_dimension=784, fitted_rows=count)


def target_index(part, seed):
    candidates = np.flatnonzero(part['eligible'] & (part['y'] == 0))
    if not len(candidates):
        return None
    return int(min(candidates, key=lambda i: _key(seed, 'target', part['scenario_id'],
                   *part['canonical_pair_track_ids'][i].tolist())))


def prepare(root, destination, seed=20261104):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    provenance = validate_source(root)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = root / 'manifest.csv'
    with manifest.open() as stream:
        selected = select_rows(list(csv.DictReader(stream)), seed)
    registration = dict(schema='womd_development_pilot_data_v1', seed=seed,
        graph_root=str(root), manifest_sha256=sha256(manifest),
        source_provenance_sha256=provenance,
        role_sizes=ROLE_SIZES, roles=selected, source_split='training_20s',
        source_shard_partition=[.6, .2, .2],
        scene_selection='hash before reading features/labels; no redraw',
        normalizer_fit='all pair nodes of TRAIN scenes only; mean/std clip10',
        clean_supervision='seeded 20% of eligible TRAIN pairs, without replacement',
        target_policy='one hash-chosen eligible negative per scene; no model filtering',
        purpose='development pilot; not formal confirmation or test',
        recording_independence_proven=False, test_arrays_loaded=False)
    (destination / 'registration.json').write_text(json.dumps(registration, indent=2))
    audit = overlap_audit(root, selected)
    (destination / 'overlap_audit.json').write_text(json.dumps(audit, indent=2))
    if audit['cross_role_conflicts']:
        raise ValueError('Detected cross-role overlap: inspect overlap_audit.json; no redrawing')
    train_parts = [read_part(root, row) for row in selected['train']]
    normalizer = fit_normalizer(train_parts)
    np.savez(destination / 'normalizer.npz', **normalizer)
    summary = {}
    for role in ROLE_SIZES:
        parts = train_parts if role == 'train' else [read_part(root, r) for r in selected[role]]
        for p in parts:
            p['x'] = transform_features(p['x'], normalizer)
            p['target_index'] = target_index(p, seed)
        targets = [p for p in parts if p['target_index'] is not None]
        summary[role] = dict(scenes=len(parts), pair_nodes=sum(len(p['x']) for p in parts),
            pair_edges=sum(p['edge_index'].shape[1] for p in parts),
            eligible=sum(int(p['eligible'].sum()) for p in parts),
            positive=sum(int((p['y'] == 1).sum()) for p in parts),
            targets=len(targets), moving_targets=sum(int(p['motion_regime'][p['target_index']] == 1)
                                                    for p in targets))
        torch.save(parts, destination / f'{role}.pt')
        print(f'[womd-data] {role}: {summary[role]}', flush=True)
    train = combine_parts(train_parts)
    eligible = torch.nonzero(train['eligible'], as_tuple=False).flatten().numpy()
    clean = np.sort(np.random.RandomState(seed).permutation(eligible)[:int(.2 * len(eligible))])
    if set(train['y'][clean].tolist()) != {0, 1}:
        raise ValueError('Clean TRAIN supervision must contain both labels')
    unlabeled = np.setdiff1d(np.arange(len(train['x'])), clean)
    torch.save(dict(idx_train=torch.from_numpy(clean), idx_unlabeled=torch.from_numpy(unlabeled)),
               destination / 'indices.pt')
    summary.update(seed=seed, status='PREPARED', test_arrays_loaded=False,
                   recording_independence_proven=False,
                   clean_supervised_pairs=len(clean), normalizer_fitted_rows=normalizer['fitted_rows'])
    summary['asset_sha256'] = {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def load_role(destination, role, device='cpu'):
    destination = Path(destination)
    if role not in ROLE_SIZES:
        raise ValueError('Unknown pilot role')
    summary = json.loads((destination / 'summary.json').read_text())
    path = destination / f'{role}.pt'
    if sha256(path) != summary['asset_sha256'][path.name]:
        raise ValueError('Changed cached data')
    parts = torch.load(path, map_location='cpu', weights_only=False)
    for part in parts:
        part['split_group_id'] = 'source_shard:' + part['source_shard']
    data = combine_parts(parts)
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
    return parts, data
