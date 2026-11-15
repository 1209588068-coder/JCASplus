"""Repackage the registered WOMD pilot for the common streaming pipeline.

This is a lossless cache conversion, not graph construction or sampling. It
reuses WOMD TRAIN normalization and the original ordered populations, targets,
and clean-supervision indices. Source-shard separation is not a certificate of
independent recordings. No AV2 arrays, normalization, or checkpoints are used.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .av2_formal_data import ROLES, _json, _save_shard
from .runtime import sha256
from .womd_pilot_data import validate_source

_TRANSPORT_KEYS = {'global_offset', 'global_stop', 'split', 'role', 'dataset_id',
                   'source_dataset_split', 'split_group_id', 'evaluation_subgroup'}


def _read(path):
    return json.loads(Path(path).read_text())


def _verify_file(path, expected):
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f'Changed registered asset: {path}')


def _asset(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Registered cache asset escapes its root')
    return path


def aligned_attachment_budget(clean_count, reference_attach, reference_clean, *, original_count=26):
    """Nearest integer match of attach/(clean+attach), ties to the lower count."""
    if any(type(v) is not int or v <= 0 for v in (clean_count, reference_attach, reference_clean)):
        raise ValueError('Budget populations must be positive integers')
    ideal = Fraction(clean_count * reference_attach, reference_clean)
    lower = ideal.numerator // ideal.denominator
    candidates = sorted({lower, lower + int(ideal.denominator != 1)})
    target = Fraction(reference_attach, reference_clean + reference_attach)
    chosen = min(candidates, key=lambda a: (abs(Fraction(a, clean_count + a) - target), a))
    actual = Fraction(chosen, clean_count + chosen)
    return dict(count=chosen, clean_supervised_pairs=clean_count,
        denominator='attachment_count / (clean_supervised_pairs + attachment_count)',
        denominator_count=clean_count + chosen, realized_fraction=float(actual),
        reference_attachment_count=reference_attach, reference_clean_supervised_pairs=reference_clean,
        reference_denominator_count=reference_clean + reference_attach,
        reference_fraction=float(target), difference_percentage_points=float((actual - target) * 100),
        candidates=candidates, original_pilot_attachment_count=original_count,
        rule='minimize absolute fraction difference over floor/ceil(C*A_ref/C_ref); exact rational arithmetic; ties choose lower count',
        multiple_attachments_per_scene_allowed=True)


def _verify_contract(source_run, reference_plan):
    """Validate run -> data summary -> assets and graph metadata provenance."""
    source = (source_run / 'data').resolve()
    plan, registration, completion = [_read(source_run / name) for name in
                                     ('plan.json', 'registration.json', 'completion.json')]
    source_plan_hash = sha256(source_run / 'plan.json')
    if (registration['plan_sha256'] != source_plan_hash
            or completion['plan_sha256'] != source_plan_hash
            or completion['status'] != 'PILOT_COMPLETE_REVIEW_REQUIRED'
            or plan['test_arrays_loaded'] or completion['test_arrays_loaded']):
        raise ValueError('Unverified completed WOMD source run')
    _verify_file(source / 'summary.json', plan['data_summary_sha256'])
    summary = _read(source / 'summary.json')
    if summary['status'] != 'PREPARED' or summary['test_arrays_loaded']:
        raise ValueError('Source pilot cache is not prepared training data')
    required = {'registration.json', 'overlap_audit.json', 'normalizer.npz', 'indices.pt',
                *(f'{r}.pt' for r in ROLES)}
    if not required.issubset(summary['asset_sha256']):
        raise ValueError('Source cache hash chain is incomplete')
    for name, digest in summary['asset_sha256'].items():
        _verify_file(_asset(source, name), digest)
    data_registration = _read(source / 'registration.json')
    overlap = _read(source / 'overlap_audit.json')
    if (data_registration['source_split'] != 'training_20s'
            or data_registration['test_arrays_loaded']
            or data_registration['recording_independence_proven']
            or summary['recording_independence_proven']
            or overlap['recording_independence_proven'] or overlap['cross_role_conflicts']
            or plan['seed'] != summary['seed'] or plan['seed'] != data_registration['seed']):
        raise ValueError('Changed WOMD pilot scope or seed')
    root = Path(data_registration['graph_root']).resolve()
    provenance = validate_source(root)
    if (root != Path(plan['graph_root']).resolve()
            or provenance != plan['graph_provenance_sha256']
            or data_registration['manifest_sha256'] != provenance['manifest.csv']):
        raise ValueError('Source graph provenance does not match the pilot registration')
    with (root / 'manifest.csv').open() as stream:
        manifest_rows = {r['scenario_id']: r for r in csv.DictReader(stream)}
    role_groups, seen_ids = {}, set()
    for role in ROLES:
        rows = data_registration['roles'][role]
        if len(rows) != summary[role]['scenes'] or len(rows) != data_registration['role_sizes'][role]:
            raise ValueError('Registered role size mismatch')
        role_groups[role] = {r['source_shard'] for r in rows}
        for row in rows:
            if row['scenario_id'] in seen_ids or manifest_rows.get(row['scenario_id']) != row:
                raise ValueError('Duplicate or changed registered scene identity')
            seen_ids.add(row['scenario_id'])
    if any(role_groups[a] & role_groups[b] for a, b in
           (('train', 'dev'), ('train', 'audit'), ('dev', 'audit'))):
        raise ValueError('Cross-role source-shard overlap')
    ref = _read(reference_plan)
    _verify_file(reference_plan, _read(reference_plan.parent / 'registration.json')['plan_sha256'])
    ref_summary_path = Path(ref['data_dir']) / 'summary.json'
    _verify_file(ref_summary_path, ref['data_summary_sha256'])
    ref_summary = _read(ref_summary_path)
    if ref_summary != ref['data_summary'] or ref_summary['original_test_arrays_loaded']:
        raise ValueError('Changed AV2 reference data contract')
    a = ref['poison_budget']['count']
    if a != ref['source_selection']['attachment_count']:
        raise ValueError('Conflicting AV2 reference attachment counts')
    if not ref['clean_supervision'].startswith('20 percent of all eligible TRAIN pairs'):
        raise ValueError('Unrecognized AV2 clean-supervision denominator')
    budget = aligned_attachment_budget(summary['clean_supervised_pairs'], a,
        int(.2 * ref_summary['train']['eligible']),
        original_count=plan['source_selection']['attachment_count'])
    sources = {str(source / name): digest for name, digest in summary['asset_sha256'].items()}
    sources.update({str(root / name): digest for name, digest in provenance.items()})
    for path in (source_run / 'plan.json', source_run / 'registration.json',
                 source_run / 'completion.json', source / 'summary.json', reference_plan,
                 reference_plan.parent / 'registration.json', ref_summary_path):
        sources[str(path.resolve())] = sha256(path)
    return source, summary, data_registration, budget, sources


def _update_digest(digest, value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays cannot have a stable population digest')
        header = json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        payload = np.ascontiguousarray(value).tobytes()
    else:
        header = b'json'
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                             default=lambda x: x.item(), allow_nan=False).encode()
    for item in (header, payload):
        digest.update(len(item).to_bytes(8, 'little'))
        digest.update(item)


def payload_sha256(parts):
    """Canonical digest of every original scene payload, independent of shards."""
    digest = hashlib.sha256()
    for part in parts:
        keys = sorted(set(part) - _TRANSPORT_KEYS)
        _update_digest(digest, keys)
        for key in keys:
            _update_digest(digest, key)
            _update_digest(digest, part[key])
    return digest.hexdigest()


def _array_digest(array):
    digest = hashlib.sha256()
    _update_digest(digest, array)
    return digest.hexdigest()


def _validate_part(part, row, clip):
    if part['scenario_id'] != row['scenario_id'] or part['source_shard'] != row['source_shard']:
        raise ValueError('Cached scene identity/order differs from registration')
    if ('split_group_id' in part
            and part['split_group_id'] != 'source_shard:' + row['source_shard']):
        raise ValueError('Cached grouping disagrees with the registered source shard')
    n = len(part['x'])
    if (part['x'].shape != (n, 784) or part['x'].dtype != np.float32
            or not np.isfinite(part['x']).all() or np.any(np.abs(part['x']) > clip)):
        raise ValueError('Invalid already-normalized WOMD features')
    if (part['y'].shape != (n,) or part['y'].dtype != np.int64
            or not np.isin(part['y'], [-1, 0, 1]).all()
            or part['eligible'].dtype != np.bool_
            or not np.array_equal(part['eligible'], part['y'] >= 0)):
        raise ValueError('Invalid preserved labels/eligibility')
    edge = part['edge_index']
    if (edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype != np.int64
            or (edge.size and (edge.min() < 0 or edge.max() >= n))):
        raise ValueError('Invalid preserved adjacency')
    if (part['canonical_pairs'].shape != (n, 2) or part['canonical_pair_track_ids'].shape != (n, 2)
            or len(set(map(tuple, part['canonical_pair_track_ids'].tolist()))) != n
            or part['motion_regime'].shape != (n,) or not np.isin(part['motion_regime'], [0, 1]).all()
            or part['message_passing_pair_mask'].shape != (n,)
            or part['message_passing_pair_mask'].dtype != np.bool_):
        raise ValueError('Invalid preserved pair metadata')
    target = part['target_index']
    if target is not None and (not 0 <= target < n or not part['eligible'][target] or part['y'][target] != 0):
        raise ValueError('Invalid preserved target')


def prepare(source_run, destination, reference_plan, max_nodes=16384):
    """Copy the frozen pilot into whole-scene shards without fitting or sampling."""
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')
    source_run, destination, reference_plan = map(lambda p: Path(p).resolve(),
                                                (source_run, destination, reference_plan))
    source, old, old_registration, budget, sources = _verify_contract(source_run, reference_plan)
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Destination must not modify the source cache')
    with np.load(source / 'normalizer.npz', allow_pickle=False) as z:
        if (z['mean'].shape != (784,) or z['scale'].shape != (784,)
                or not np.isfinite(z['mean']).all() or not np.isfinite(z['scale']).all()
                or np.any(z['scale'] <= 0) or int(z['fitted_rows']) != old['train']['pair_nodes']
                or int(z['fitted_rows']) != old['normalizer_fitted_rows']):
            raise ValueError('Invalid registered WOMD TRAIN normalizer')
        clip = float(z['clip'])
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError('Invalid registered normalization clipping bound')
    destination.mkdir(parents=True, exist_ok=False)
    scope = 'same previously exposed WOMD pilot populations; source-shard separation, not independent recording confirmation or official test'
    registration = dict(schema='womd_av2_aligned_streaming_data_v1', dataset_id='WOMD',
        source_dataset_split='training_20s', target_seed=old['seed'],
        source_run=str(source_run), source_data_dir=str(source), reference_plan=str(reference_plan),
        graph_root=old_registration['graph_root'],
        sources=sources, attachment_budget=budget, role_sizes=old_registration['role_sizes'],
        role_counts={r: dict(scenes=old[r]['scenes'], groups=len({p['source_shard']
                     for p in old_registration['roles'][r]})) for r in ROLES},
        normalizer_policy='byte-identical copy of existing WOMD TRAIN-only statistics; no refit or renormalization',
        array_policy='same ordered scenes, normalized feature bytes, labels, adjacency, IDs and targets',
        clean_supervision='same saved indices for seed ' + str(old['seed']),
        grouping='source_shard:<filename>, not AV2 recording/log identity', evaluation_scope=scope,
        test_arrays_loaded=False, recording_independence_proven=False, blind_evaluation=False,
        original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
        adapter_source_sha256=sha256(__file__))
    _json(destination / 'registration.json', registration)
    _json(destination / 'status.json', dict(stage='REPACKAGING'))
    summary = dict(schema='womd_av2_aligned_streaming_cache_v1', status='PREPARED',
        dataset_id='WOMD', source_dataset_split='training_20s', target_seed=old['seed'],
        seed=old['seed'], source_run=str(source_run), evaluation_scope=scope,
        attachment_budget=budget, max_nodes_per_shard=max_nodes,
        whole_scene_shards=True, oversize_scene_policy='retain a whole scene in its own shard',
        normalizer_fitted_rows=old['normalizer_fitted_rows'], new_statistics_fitted=0,
        clean_supervised_pairs=old['clean_supervised_pairs'], test_arrays_loaded=False,
        original_validation_arrays_loaded=False, original_test_arrays_loaded=False,
        blind_evaluation=False, recording_independence_proven=False)
    try:
        for name in ('normalizer.npz', 'indices.pt', 'overlap_audit.json'):
            shutil.copyfile(source / name, destination / name)
            _verify_file(destination / name, old['asset_sha256'][name])
        for role in ROLES:
            parts = torch.load(source / f'{role}.pt', map_location='cpu', weights_only=False)
            rows = old_registration['roles'][role]
            if len(parts) != len(rows):
                raise ValueError('Cached role length differs from registration')
            original_digest = payload_sha256(parts)
            folder = destination / role
            folder.mkdir()
            shards, pending, pending_nodes, offset, offsets = [], [], 0, 0, []
            targets = []
            for p, row in zip(parts, rows):
                _validate_part(p, row, clip)
                n = len(p['x'])
                if pending and pending_nodes + n > max_nodes:
                    shards.append(_save_shard(folder / f'{len(shards):05d}.pt', pending))
                    pending, pending_nodes = [], 0
                p.update(global_offset=offset, global_stop=offset+n, split='training_20s', role=role,
                    dataset_id='WOMD', source_dataset_split='training_20s',
                    split_group_id='source_shard:' + p['source_shard'],
                    evaluation_subgroup='womd_existing_pilot_' + role)
                offsets.append(offset)
                offset += n
                pending.append(p)
                pending_nodes += n
                targets.append([p['scenario_id'], p['target_index'],
                    None if p['target_index'] is None else p['canonical_pair_track_ids'][p['target_index']].tolist()])
            if pending:
                shards.append(_save_shard(folder / f'{len(shards):05d}.pt', pending))
            population = dict(y=torch.from_numpy(np.concatenate([p['y'] for p in parts])),
                eligible=torch.from_numpy(np.concatenate([p['eligible'] for p in parts])),
                motion_regime=torch.from_numpy(np.concatenate([p['motion_regime'] for p in parts])),
                scene_offsets=torch.tensor(offsets + [offset], dtype=torch.long))
            stats = dict(scenes=len(parts), pair_nodes=offset,
                pair_edges=sum(p['edge_index'].shape[1] for p in parts),
                eligible=int(population['eligible'].sum()), positive=int((population['y'] == 1).sum()),
                targets=sum(p['target_index'] is not None for p in parts),
                moving_targets=sum(int(p['motion_regime'][p['target_index']] == 1)
                                   for p in parts if p['target_index'] is not None))
            if stats != old[role]:
                raise ValueError('Repacked role counts differ from source summary')
            torch.save(population, destination / f'{role}_population.pt')

            def restored():
                for entry in shards:
                    path = folder / entry['path']
                    _verify_file(path, entry['sha256'])
                    yield from torch.load(path, map_location='cpu', weights_only=False)

            restored_digest = payload_sha256(restored())
            if original_digest != restored_digest:
                raise ValueError('Repackaging changed scene payloads')
            summary[role] = dict(**stats, shards=shards,
                groups=registration['role_counts'][role]['groups'],
                zero_pair_scenes=sum(len(p['x']) == 0 for p in parts),
                evaluation_subgroups=dict(Counter(p['evaluation_subgroup'] for p in parts)),
                source_payload_sha256=original_digest, repacked_payload_sha256=restored_digest,
                target_identity_sha256=_array_digest(targets),
                population_array_sha256={k: _array_digest(v) for k, v in population.items()})
            if role == 'train':
                original_indices = torch.load(source / 'indices.pt', map_location='cpu', weights_only=True)
                eligible = torch.nonzero(population['eligible'], as_tuple=False).flatten().numpy()
                clean = np.sort(np.random.RandomState(old['seed']).permutation(eligible)[:int(.2 * len(eligible))])
                expected = dict(idx_train=torch.from_numpy(clean), idx_unlabeled=torch.from_numpy(
                    np.setdiff1d(np.arange(offset, dtype=np.int64), clean)))
                if (set(original_indices) != set(expected) or len(clean) != old['clean_supervised_pairs']
                        or set(population['y'][clean].tolist()) != {0, 1}
                        or any(not torch.equal(expected[k], original_indices[k]) for k in expected)):
                    raise ValueError('Original clean indices disagree with the common seeded rule')
                summary['clean_indices'] = dict(seed=old['seed'], exact_original_indices_preserved=True,
                    source_file_sha256=old['asset_sha256']['indices.pt'],
                    array_sha256={k: _array_digest(v) for k, v in original_indices.items()})
            _json(destination / 'status.json', dict(stage='ROLE_REPACKAGED', role=role, **stats))
        for path, expected in sources.items():
            _verify_file(path, expected)
        summary['asset_sha256'] = {p.name: sha256(p) for p in sorted(destination.iterdir())
                                  if p.is_file() and p.name != 'status.json'}
        _json(destination / 'summary.json', summary)
        _json(destination / 'status.json', dict(stage='PREPARED'))
        return summary
    except BaseException as exc:
        _json(destination / 'status.json', dict(stage='FAILED', error_type=type(exc).__name__, error=str(exc)))
        raise
