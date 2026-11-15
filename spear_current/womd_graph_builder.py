"""Build clean WOMD vehicle and 784-coordinate pair graphs; never train models.

Prepare an immutable source/code contract, then run CPU shard workers. Every
record has a ledger entry, including explicit data exclusions. Output splits
remain unassigned until a separate overlap/grouping review. Input TFRecords,
AV2 assets, previous outputs, and model checkpoints are never modified.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

from .data import line_graph_from_graph
from .risk_labels import RiskLabelConfig, label_config_dict, labels_for_graph
from .vehicle_graph_core import BuildConfig, LaneData, build_graph_arrays, build_track_arrays
from .womd_source import ScenarioRejected, decode_scenario, iter_tfrecords, load_scenario_class

SCHEMA = 'womd_jcas_graph_v1'
DEFAULT_RAW = Path('/home/mizhou/datasets/waymo_open_motion/training_20s')
DEFAULT_INVENTORY = Path('/home/mizhou/datasets/waymo_open_motion/remote_inventory_20261104_003733.json')
DEFAULT_PROTO = Path('/home/mizhou/tools/womd_proto_99a4cb3ff07e')
SOURCE_MODULES = ('__init__.py', 'womd_graph_builder.py', 'womd_source.py',
                  'vehicle_graph_core.py', 'data.py', 'pair_graph.py', 'risk_labels.py', 'motion_gate.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    os.replace(tmp, path)


def save_npz(path, arrays):
    """No pickled objects and no overwrite; only publish a complete archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    converted = {k: np.asarray(v) for k, v in arrays.items()}
    if any(v.dtype.hasobject for v in converted.values()):
        raise ValueError('Object/pickle arrays are forbidden in new graph assets')
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('xb') as stream:
        np.savez_compressed(stream, **converted)
    os.replace(tmp, path)
    return digest_file(path)


def json_scalar(value):
    return np.asarray(json.dumps(value, sort_keys=True, allow_nan=False))


def check_vehicle_pair_graph(graph, pair):
    n, e, p = len(graph['node_track_ids']), graph['edge_index'].shape[1], len(pair['x'])
    if graph['x_node'].shape != (n, 358) or graph['edge_attr'].shape != (e, 34):
        raise ValueError('Vehicle feature dimensions differ from the JCAS contract')
    if pair['x'].shape != (p, 784) or p * 2 != e:
        raise ValueError('Pair features or bidirectional coverage differ from contract')
    for value in (graph['x_node'], graph['edge_attr'], pair['x']):
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite model input')
    ids = graph['node_track_ids'].astype(str)
    endpoints = pair['canonical_pairs']
    directions = pair['source_directed_edges']
    if not np.array_equal(ids[endpoints], pair['canonical_pair_track_ids']):
        raise ValueError('Pair identity mapping changed')
    if p and not np.array_equal(graph['edge_index'][:, directions[:, 0]].T, endpoints):
        raise ValueError('Forward edge mapping changed')
    if p and not np.array_equal(graph['edge_index'][:, directions[:, 1]].T, endpoints[:, ::-1]):
        raise ValueError('Reverse edge mapping changed')
    eligible = pair['eligible']
    y = graph['edge_label_dynamic_risk'][directions[:, 0]]
    if not np.array_equal(pair['y'][eligible], y[eligible]):
        raise ValueError('Pair labels do not match vehicle labels')
    if np.any(pair['y'][~eligible] != -1):
        raise ValueError('Ineligible context received a supervised label')
    if e and not np.array_equal(graph['edge_label_dynamic_risk'][directions[:, 0]],
                                graph['edge_label_dynamic_risk'][directions[:, 1]]):
        raise ValueError('Directed risk labels are not symmetric')
    if not np.array_equal(graph['message_passing_edge_mask'], graph['physical_possible_mask']):
        raise ValueError('Message mask changed from physical mask contract')
    if not np.all(pair['x'][:, [357, 715]] == 0):
        raise ValueError('Unavailable AV2 focal metadata must use declared zero placeholder')


def build_decoded(decoded, config=None):
    """Pure dataset adapter plus the unchanged current pair/label functions."""
    cfg = config or BuildConfig()
    all_tracks = [build_track_arrays(**spec, cfg=cfg) for spec in decoded['tracks']]
    filters = Counter(t.skip_reason for t in all_tracks if t.skip_reason is not None)
    tracks = [t for t in all_tracks if t.skip_reason is None]
    if len(tracks) < 2:
        raise ScenarioRejected('fewer_than_two_observation_qualified_vehicles',
                               decoded['metadata']['scenario_id'],
                               retained_vehicles=len(tracks), filters=dict(filters))
    lanes = {i: LaneData(**spec) for i, spec in decoded['lanes'].items()}
    graph = build_graph_arrays(tracks, lanes, cfg)
    # Primary labels use the existing dynamic 5/1/4 implementation; the legacy
    # 10-m proximity field in the core is diagnostic only.
    graph.update(labels_for_graph(graph, RiskLabelConfig()))
    graph['node_is_focal_known'] = np.zeros(len(tracks), dtype=bool)
    graph['node_is_sdc'] = graph['node_track_ids'] == decoded['metadata']['source_sdc_track_id']
    pair = line_graph_from_graph(graph, RiskLabelConfig())
    check_vehicle_pair_graph(graph, pair)
    return graph, pair, dict(filters)


def prepare(output, raw_root, inventory, proto_root, workers=8, limit_shards=None):
    output, raw_root = Path(output).resolve(), Path(raw_root).resolve()
    inventory, proto_root = Path(inventory).resolve(), Path(proto_root).resolve()
    if (output == raw_root or raw_root in output.parents or output in raw_root.parents
            or output == Path(__file__).parent or Path(__file__).parent in output.parents):
        raise ValueError('Output must be separate from raw data and source code')
    if workers < 1 or workers > 16:
        raise ValueError('Use between 1 and 16 CPU workers')
    entries = json.loads(inventory.read_text())
    sources = []
    for entry in sorted(entries, key=lambda x: x['metadata']['name']):
        meta = entry['metadata']
        name = Path(meta['name']).name
        if not name.startswith('training_20s.tfrecord-'):
            raise ValueError('Only registered WOMD training_20s shards are accepted')
        source = raw_root / name
        if not source.is_file() or source.stat().st_size != int(meta['size']):
            raise ValueError(f'Missing/wrong-size shard: {source}')
        sources.append(dict(index=len(sources), path=str(source), filename=name,
                            bytes=int(meta['size']), official_url=entry['url'],
                            generation=meta['generation'], md5_base64=meta['md5Hash']))
    if limit_shards is not None:
        if limit_shards <= 0:
            raise ValueError('limit_shards must be positive')
        sources = sources[:limit_shards]
    if not sources:
        raise ValueError('No source shards')
    code = {str(Path(__file__).parent / name): digest_file(Path(__file__).parent / name)
            for name in SOURCE_MODULES}
    proto_files = sorted(proto_root.rglob('*.proto')) + sorted(proto_root.rglob('*_pb2.py'))
    if not proto_files:
        raise ValueError('Pinned official protobuf source not present')
    proto_hashes = {str(p): digest_file(p) for p in proto_files}
    graph_config = asdict(BuildConfig())
    for key in ('data_root', 'output_dir', 'overwrite', 'split_group_metadata',
                'seed', 'train_ratio', 'val_ratio', 'test_ratio'):
        graph_config.pop(key)
    plan = dict(schema=SCHEMA, prepared_at=utc_now(), output_dir=str(output),
        raw_root=str(raw_root), source_inventory=str(inventory),
        inventory_sha256=digest_file(inventory), source_shards=sources,
        workers=workers, code_sha256=code, proto_root=str(proto_root),
        proto_sha256=proto_hashes, graph_config=graph_config,
        label_config=label_config_dict(RiskLabelConfig()),
        window=dict(start=0, observation_end=49, future_end=109,
                    interval_tolerance_seconds=0.001, grid_tolerance_seconds=0.005),
        dataset='Waymo Open Motion Dataset', dataset_version='1.3.1',
        source_dataset_split='training_20s', split='unassigned',
        training_ready=False, formal_split_eligible=False,
        split_policy='no_split_until_overlap_and_group_review',
        normalization='none; fit only on future registered TRAIN',
        focal_policy='unavailable_zero_placeholder_not_sdc_or_tracks_to_predict',
        unknown_intersection_policy='unknown_cannot_establish_nonintersection_hard_exclusion',
        map_neighbor_policy='union_of_declared_lane_neighbor_segments',
        test_arrays_loaded=False, training_started=False,
        min_free_disk_bytes=20 * 1024**3,
        run_scope='all_downloaded_shards' if limit_shards is None else 'adapter_pilot')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for filename in code:
        shutil.copy2(filename, snapshot / Path(filename).name)
    shutil.copy2(proto_root / 'source_manifest.json', snapshot / 'waymo_proto_source_manifest.json')
    atomic_json(output / 'plan.json', plan)
    atomic_json(output / 'registration.json', dict(plan_sha256=digest_file(output / 'plan.json'),
                schema=SCHEMA, registered_at=utc_now()))
    atomic_json(output / 'status.json', dict(status='PREPARED', source_shards=len(sources),
                completed_shards=0, training_started=False))
    return output / 'plan.json'


def validate_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    registration = json.loads((path.parent / 'registration.json').read_text())
    if registration['plan_sha256'] != digest_file(path):
        raise ValueError('Plan changed after registration')
    if plan['schema'] != SCHEMA or Path(plan['output_dir']) != path.parent:
        raise ValueError('Plan identity/output mismatch')
    for filename, expected in {**plan['code_sha256'], **plan['proto_sha256']}.items():
        if digest_file(filename) != expected:
            raise ValueError(f'Source changed after prepare: {filename}')
    if digest_file(plan['source_inventory']) != plan['inventory_sha256']:
        raise ValueError('Source inventory changed')
    if plan['training_ready'] or plan['split'] != 'unassigned':
        raise ValueError('This builder does not assign experiment splits')
    if plan['label_config'] != label_config_dict(RiskLabelConfig()):
        raise ValueError('Only the fixed dynamic risk 5/1/4 label contract is implemented')
    if plan['window'] != dict(start=0, observation_end=49, future_end=109,
                             interval_tolerance_seconds=0.001, grid_tolerance_seconds=0.005):
        raise ValueError('Only the fixed first-110 sample window contract is implemented')
    return plan


def process_shard(plan, source):
    """One process owns one new shard directory and its immutable record ledger."""
    root = Path(plan['output_dir'])
    shard_root = root / 'shards' / f"{source['index']:05d}"
    shard_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    path = Path(source['path'])
    try:
        stat = path.stat()
        if stat.st_size != source['bytes']:
            raise ValueError('Raw shard size changed')
        md5 = digest_file(path, 'md5')
        if base64.b64encode(bytes.fromhex(md5)).decode() != source['md5_base64']:
            raise ValueError('Official shard MD5 mismatch')
        source_sha = digest_file(path)
        Scenario = load_scenario_class(plan['proto_root'])
        counts, exclusions, track_filters = Counter(), Counter(), Counter()
        ledger = shard_root / 'records.jsonl'
        with ledger.open('x') as stream:
            for record in iter_tfrecords(path):
                scenario = Scenario()
                scenario.ParseFromString(record.payload)
                sid = str(scenario.scenario_id)
                payload_sha = hashlib.sha256(record.payload).hexdigest()
                row = dict(scenario_id=sid, source_shard=source['filename'],
                    source_shard_index=source['index'], record_index=record.index,
                    byte_offset=record.byte_offset, payload_bytes=len(record.payload),
                    payload_sha256=payload_sha, source_sha256=source_sha,
                    source_dataset_split='training_20s', split='unassigned',
                    formal_split_eligible=False, training_ready=False)
                counts['records'] += 1
                try:
                    decoded = decode_scenario(scenario)
                    graph, pair, filters = build_decoded(decoded, BuildConfig(**plan['graph_config']))
                    track_filters.update(filters)
                    key = f'r{record.index:05d}_{hashlib.sha256(sid.encode()).hexdigest()[:16]}'
                    graph_rel = Path('shards') / f"{source['index']:05d}" / 'vehicles' / f'{key}.npz'
                    pair_rel = Path('shards') / f"{source['index']:05d}" / 'pairs' / f'{key}.npz'
                    audit_rel = Path('shards') / f"{source['index']:05d}" / 'source_audit' / f'{key}.npz'
                    metadata = {**decoded['metadata'], **row, 'schema': SCHEMA,
                                'node_filter_counts': filters, 'label_config': plan['label_config'],
                                'plan_sha256': plan['plan_sha256'],
                                'primary_label': 'edge_label_dynamic_risk / edge_label',
                                'proximity_label_role': 'legacy_10m_diagnostic_only_not_task_label',
                                'feature_normalization': 'none', 'future_used_for_features': False,
                                'future_used_for_topology': False}
                    graph.update(scenario_id=np.asarray(sid), dataset=np.asarray('WOMD'),
                                 metadata_json=json_scalar(metadata),
                                 observation_timestamps_seconds=decoded['source_arrays']['window_timestamps_seconds'][:50],
                                 future_timestamps_seconds=decoded['source_arrays']['window_timestamps_seconds'][50:])
                    pair_arrays = {k: v for k, v in pair.items() if isinstance(v, np.ndarray)}
                    pair_arrays.update(scenario_id=np.asarray(sid),
                        metadata_json=json_scalar({**metadata, **pair['metadata']}),
                        audit_counts_json=json_scalar(pair['audit_counts']))
                    row.update(status='BUILT', graph_path=str(graph_rel), pair_graph_path=str(pair_rel),
                        source_audit_path=str(audit_rel),
                        graph_sha256=save_npz(root / graph_rel, graph),
                        pair_graph_sha256=save_npz(root / pair_rel, pair_arrays),
                        source_audit_sha256=save_npz(root / audit_rel, decoded['source_arrays']),
                        map_fingerprint_sha256=decoded['metadata']['map_fingerprint_sha256'],
                        node_filter_counts=filters, **pair['audit_counts'])
                    counts['graphs'] += 1
                    counts['vehicle_nodes'] += len(graph['x_node'])
                    counts['vehicle_edges'] += graph['edge_index'].shape[1]
                    counts['pair_nodes'] += len(pair['x'])
                    counts['pair_edges'] += pair['edge_index'].shape[1]
                    counts['eligible_pairs'] += int(pair['eligible'].sum())
                    counts['positive_pairs'] += int(np.count_nonzero(pair['y'] == 1))
                    counts['negative_pairs'] += int(np.count_nonzero(pair['y'] == 0))
                    counts['eligible_moving_pairs'] += int(np.count_nonzero(pair['eligible'] & (pair['motion_regime'] == 1)))
                    counts['eligible_slow_pairs'] += int(np.count_nonzero(pair['eligible'] & (pair['motion_regime'] == 0)))
                except ScenarioRejected as exc:
                    exclusions[exc.reason] += 1
                    counts['excluded_records'] += 1
                    row.update(status='EXCLUDED', exclusion=exc.as_dict())
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                if record.index % 10 == 0:
                    if shutil.disk_usage(root).free < plan['min_free_disk_bytes']:
                        raise RuntimeError('Insufficient free disk; graph conversion stopped')
                    atomic_json(shard_root / 'progress.json', dict(updated_at=utc_now(),
                        pid=os.getpid(), counts=dict(counts), exclusions=dict(exclusions)))
        final = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (final.st_size, final.st_mtime_ns, final.st_ino):
            raise ValueError('Raw input changed during graph generation')
        result = dict(status='COMPLETE', plan_sha256=plan['plan_sha256'],
            source_index=source['index'], source=source,
            source_sha256=source_sha, counts=dict(counts), exclusions=dict(exclusions),
            track_filter_counts=dict(track_filters), records_path=str(ledger.relative_to(root)),
            records_sha256=digest_file(ledger), elapsed_seconds=time.monotonic()-started,
            finished_at=utc_now())
        atomic_json(shard_root / 'completion.json', result)
        return result
    except BaseException as exc:
        atomic_json(shard_root / 'failure.json', dict(status='FAILED', at=utc_now(),
                    error=repr(exc), traceback=traceback.format_exc()))
        raise


def final_manifest(root, completions):
    """Merge every ledger; report exact duplicates without silently selecting one."""
    import csv
    ids, payloads = defaultdict(list), defaultdict(list)
    counts, exclusions, filters = Counter(), Counter(), Counter()
    fields = ['scenario_id', 'source_shard', 'source_shard_index', 'record_index', 'byte_offset',
        'payload_sha256', 'source_sha256', 'status', 'split', 'formal_split_eligible',
        'graph_path', 'graph_sha256', 'pair_graph_path', 'pair_graph_sha256',
        'source_audit_path', 'source_audit_sha256', 'source_vehicle_nodes', 'source_directed_edges',
        'line_nodes', 'line_directed_edges', 'eligible_pairs', 'eligible_class_0', 'eligible_class_1',
        'eligible_moving_pairs', 'eligible_slow_pairs', 'map_fingerprint_sha256', 'exclusion_reason']
    manifest = root / 'manifest.csv'
    with manifest.open('x', newline='') as out:
        writer = csv.DictWriter(out, fields, extrasaction='ignore')
        writer.writeheader()
        for item in sorted(completions, key=lambda x: x['source_index']):
            ledger = root / item['records_path']
            if digest_file(ledger) != item['records_sha256']:
                raise ValueError('Completed shard ledger hash mismatch')
            counts.update(item['counts']); exclusions.update(item['exclusions']); filters.update(item['track_filter_counts'])
            for line in ledger.read_text().splitlines():
                row = json.loads(line)
                loc = f"{row['source_shard']}:{row['record_index']}"
                ids[row['scenario_id']].append(loc)
                payloads[row['payload_sha256']].append(loc)
                row['exclusion_reason'] = row.get('exclusion', {}).get('reason', '')
                writer.writerow(row)
    duplicates = dict(scenario_ids={k:v for k,v in ids.items() if len(v)>1},
                      identical_payloads={k:v for k,v in payloads.items() if len(v)>1},
                      scope='exact_identity_only; recording overlap remains unaudited',
                      split='unassigned', training_ready=False)
    atomic_json(root / 'duplicate_report.json', duplicates)
    return dict(counts=dict(counts), exclusions=dict(exclusions), track_filter_counts=dict(filters),
        manifest_path=str(manifest), manifest_sha256=digest_file(manifest),
        duplicate_scenario_id_groups=len(duplicates['scenario_ids']),
        duplicate_payload_groups=len(duplicates['identical_payloads']))


def run(plan_path):
    plan = validate_plan(plan_path)
    root = Path(plan['output_dir'])
    with (root / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'completion.json').exists():
            raise FileExistsError('Build already complete; no overwrite')
        if (root / 'shards').exists() and any((root / 'shards').iterdir()):
            raise FileExistsError('Partial build exists; preserve it and review before a separately registered recovery')
        started = time.monotonic()
        atomic_json(root / 'launch.json', dict(at=utc_now(), pid=os.getpid(),
            plan_sha256=digest_file(plan_path), command=sys.argv, device='CPU_numpy',
            workers=plan['workers'], training_started=False))
        completions = []
        status = dict(status='RUNNING', pid=os.getpid(), updated_at=utc_now(),
            source_shards=len(plan['source_shards']), completed_shards=0,
            training_started=False, split='unassigned')
        atomic_json(root / 'status.json', status)
        try:
            with ProcessPoolExecutor(max_workers=plan['workers'],
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                worker_plan = {k: v for k, v in plan.items() if k != 'source_shards'}
                worker_plan['plan_sha256'] = digest_file(plan_path)
                futures = {pool.submit(process_shard, worker_plan, source): source
                           for source in plan['source_shards']}
                try:
                    for future in as_completed(futures):
                        result = future.result()
                        completions.append(result)
                        total = Counter()
                        for c in completions:
                            total.update(c['counts'])
                        status.update(updated_at=utc_now(), completed_shards=len(completions),
                                      counts=dict(total), elapsed_seconds=time.monotonic()-started)
                        atomic_json(root / 'status.json', status)
                        print(f"[womd-graphs] shards {len(completions)}/{len(futures)}; "
                              f"graphs {total['graphs']}; excluded {total['excluded_records']}", flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            summary = final_manifest(root, completions)
            summary.update(status='GRAPH_BUILD_COMPLETE_SPLIT_REVIEW_REQUIRED',
                schema=SCHEMA, completed_at=utc_now(), elapsed_seconds=time.monotonic()-started,
                source_shards=len(completions), split='unassigned', training_ready=False,
                formal_split_eligible=False, model_training_started=False,
                plan_sha256=digest_file(plan_path), test_arrays_loaded=False)
            atomic_json(root / 'summary.json', summary)
            atomic_json(root / 'completion.json', dict(status=summary['status'],
                at=summary['completed_at'], summary_sha256=digest_file(root / 'summary.json'),
                plan_sha256=summary['plan_sha256']))
            status.update(status=summary['status'], updated_at=utc_now())
            atomic_json(root / 'status.json', status)
            print(f"[womd-graphs] complete: {root / 'summary.json'}", flush=True)
        except BaseException as exc:
            status.update(status='FAILED_REVIEW_REQUIRED', updated_at=utc_now(),
                          error=repr(exc), traceback=traceback.format_exc())
            atomic_json(root / 'status.json', status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--prepare', action='store_true')
    command.add_argument('--run', type=Path, metavar='PLAN')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--proto-root', type=Path, default=DEFAULT_PROTO)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit-shards', type=int)
    args = parser.parse_args()
    if args.prepare:
        if args.output_dir is None:
            parser.error('--prepare requires --output-dir')
        print(prepare(args.output_dir, args.raw_root, args.inventory, args.proto_root,
                      args.workers, args.limit_shards))
    else:
        run(args.run)


if __name__ == '__main__':
    main()
