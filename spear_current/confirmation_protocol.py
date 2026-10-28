"""Metadata-only group selection and a separate original-TRAIN confirmation reader.

Offline AV2/self-owned GNN robustness research. Selection never opens graphs or
uses labels, motion, future information, or summary statistics. The existing
train/dev/audit reader and its role restrictions are deliberately untouched.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from .data import _TrackedGraph, line_graph_from_graph, transform_features
from .risk_labels import RiskLabelConfig
from .runtime import sha256


PROTOCOL_VERSION = 'spear-current-confirmation-v1'
RANK_PREFIX = '20261027:' + PROTOCOL_VERSION + ':'
SEED_LIST = [20260621, 20260622, 20260623]
TARGET_SEED = 20260621
MANIFEST_FIELDS = ('scenario_id', 'split_group_id', 'split', 'graph_path', 'graph_sha256')
ROW_FIELDS = frozenset((*MANIFEST_FIELDS, 'role'))
_VALIDATED = {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError('Require a nonempty canonical string: ' + name)
    return value


def _ids(values, name):
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError('Require an ID collection: ' + name)
    return {_text(value, name) for value in values}


def _sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Require a lowercase SHA256 digest')
    return value


def _relative_graph_path(value):
    value = _text(value, 'graph_path')
    path = PurePosixPath(value)
    if (path.is_absolute() or '..' in path.parts or len(path.parts) < 2
            or path.parts[0] != 'graphs' or path.suffix != '.npz'
            or '\\' in value or str(path) != value):
        raise ValueError('Graph path must be a relative graphs/ .npz path')
    return value


def _metadata_rows(manifest_rows):
    """Project only identity/split/path/hash fields and reject conflicting IDs."""
    if not isinstance(manifest_rows, (list, tuple)):
        raise ValueError('Require an ordered collection of manifest row mappings')
    by_scene, group_splits = {}, {}
    for row in manifest_rows:
        if not isinstance(row, Mapping):
            raise ValueError('Manifest rows must be mappings')
        item = {key: row.get(key, '') for key in MANIFEST_FIELDS}
        scene = _text(item['scenario_id'], 'scenario_id')
        group = _text(item['split_group_id'], 'split_group_id')
        split = _text(item['split'], 'split')
        if scene in by_scene and by_scene[scene] != item:
            raise ValueError('Conflicting duplicate scenario ID: ' + scene)
        if group in group_splits and group_splits[group] != split:
            raise ValueError('One group appears in conflicting original splits: ' + group)
        by_scene[scene], group_splits[group] = item, split
    return by_scene


def _rank(kind, identifier):
    return hashlib.sha256((RANK_PREFIX + kind + ':' + identifier).encode()).hexdigest()


def select_rows(manifest_rows, excluded_groups, group_count=1024):
    """Select one scene per group using only the fixed ID hashes, independent of order."""
    if type(group_count) is not int or group_count <= 0:
        raise ValueError('group_count must be a positive integer')
    excluded = _ids(excluded_groups, 'excluded_group_ids')
    groups = {}
    for item in _metadata_rows(manifest_rows).values():
        if (item['split'] != 'train' or item['split_group_id'] in excluded
                or not item['graph_path'] or not item['graph_sha256']):
            continue
        _relative_graph_path(item['graph_path'])
        _sha(item['graph_sha256'])
        groups.setdefault(item['split_group_id'], []).append(item)
    if len(groups) < group_count:
        raise ValueError(f'Insufficient unseen original-TRAIN groups: {len(groups)} < {group_count}')
    ordered_groups = sorted(groups, key=lambda group: (_rank('group', group), group))[:group_count]
    selected = []
    for group in ordered_groups:
        item = min(groups[group], key=lambda row: (_rank('scene', row['scenario_id']), row['scenario_id']))
        selected.append(dict(item, role='confirmation'))
    return selected


def _stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _checked_reference(reference, base, stamps, *, read_bytes=False):
    if not isinstance(reference, Mapping):
        raise ValueError('Expected a file path/SHA256 reference')
    raw_path = Path(_text(reference.get('path'), 'reference path'))
    # Keep the lexical source path so a retargeted symlink changes its stat stamp.
    path = raw_path if raw_path.is_absolute() else Path(base) / raw_path
    digest = _sha(reference.get('sha256'))
    if not path.is_file():
        raise ValueError('Missing referenced metadata file: ' + str(path))
    before = _stamp(path)
    payload = path.read_bytes() if read_bytes else None
    actual = hashlib.sha256(payload).hexdigest() if payload is not None else sha256(path)
    if actual != digest or _stamp(path) != before:
        raise ValueError('Referenced metadata hash changed: ' + str(path))
    ref = dict(path=str(path.absolute()), sha256=digest)
    if ref['path'] in stamps and stamps[ref['path']] != before:
        raise ValueError('Metadata source changed during protocol preparation')
    stamps[ref['path']] = before
    return ref, payload


def _file_reference(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=sha256(path))


def _validation_key(bundle, protocol):
    return (str(Path(bundle.manifest_path).absolute()), _digest(bundle.manifest), _digest(protocol))


def _remember(bundle, protocol, stamps):
    if any(_stamp(path) != stamp for path, stamp in stamps.items()):
        raise ValueError('Bound metadata changed before protocol validation completed')
    # Only one full validation receipt is needed; modified dictionaries never
    # inherit it because the complete bundle and protocol contents form the key.
    _VALIDATED.clear()
    _VALIDATED[_validation_key(bundle, protocol)] = dict(stamps)


def prepare_protocol(bundle, history_path, group_count=1024):
    """Verify all source metadata and freeze the independent confirmation population."""
    stamps = {}
    bundle_path = Path(bundle.manifest_path).absolute()
    bundle_ref, bundle_bytes = _checked_reference(_file_reference(bundle_path),
                                                  bundle_path.parent, stamps, read_bytes=True)
    if json.loads(bundle_bytes) != bundle.manifest:
        raise ValueError('In-memory bundle differs from its manifest file')
    manifest_ref, manifest_bytes = _checked_reference(bundle.manifest['graph_manifest'],
        bundle_path.parent, stamps, read_bytes=True)
    if Path(manifest_ref['path']).name != 'manifest_grouped_v5.csv':
        raise ValueError('Require the authoritative grouped-v5 graph manifest')
    prior_ref = bundle.config.get('references', {}).get('graph_manifest')
    if prior_ref is not None:
        prior, _ = _checked_reference(prior_ref, bundle_path.parent, stamps)
        if prior != manifest_ref:
            raise ValueError('Bundle graph-manifest references disagree')
    rows = [{key: row.get(key, '') for key in MANIFEST_FIELDS}
            for row in csv.DictReader(io.StringIO(manifest_bytes.decode('utf-8-sig')))]
    metadata = _metadata_rows(rows)
    history_path = Path(history_path).absolute()
    history_ref, history_bytes = _checked_reference(_file_reference(history_path),
        history_path.parent, stamps, read_bytes=True)
    history = json.loads(history_bytes)
    if (history.get('status') != 'PASS_KNOWN_SPEAR_HISTORY'
            or history.get('unresolved_references') != []):
        raise ValueError('History audit is incomplete or has unresolved references')
    if not isinstance(history.get('sources'), list) or not history['sources']:
        raise ValueError('History audit must bind its checked source files')
    history_sources = []
    for source in history['sources']:
        kind = _text(source.get('kind'), 'source kind')
        ref, _ = _checked_reference(source, history_path.parent, stamps)
        history_sources.append(dict(ref, kind=kind))
    source_bindings = {(Path(ref['path']).resolve(), ref['sha256']) for ref in history_sources}
    if any((Path(ref['path']).resolve(), ref['sha256']) not in source_bindings
           for ref in (bundle_ref, manifest_ref)):
        raise ValueError('History audit must bind this bundle and authoritative graph manifest')
    excluded_groups = _ids(history.get('excluded_group_ids'), 'excluded_group_ids')
    excluded_scenes = _ids(history.get('excluded_scenario_ids'), 'excluded_scenario_ids')
    all_groups = {row['split_group_id'] for row in rows}
    if not excluded_groups <= all_groups or not excluded_scenes <= set(metadata):
        raise ValueError('Historical exclusions contain unresolved scenario/group IDs')
    excluded_groups.update(metadata[scene]['split_group_id'] for scene in excluded_scenes)
    bundle_scenes, bundle_groups = set(), set()
    roles = bundle.config['roles']
    if set(roles) != {'train', 'dev', 'audit'}:
        raise ValueError('Require the complete original three-role bundle')
    for role, count in (('train', 512), ('dev', 128), ('audit', 256)):
        if len(roles[role]) != count:
            raise ValueError('The frozen bundle must contain all 896 historical groups')
        for row in roles[role]:
            scene, group = row['scenario_id'], row['split_group_id']
            projected = {key: row[key] for key in MANIFEST_FIELDS}
            if (row.get('role') != role or row.get('split') != 'train'
                    or metadata.get(scene) != projected
                    or scene in bundle_scenes or group in bundle_groups):
                raise ValueError('Bundle roles differ from authoritative unique TRAIN groups')
            bundle_scenes.add(scene)
            bundle_groups.add(group)
    excluded_scenes.update(bundle_scenes)
    excluded_groups.update(bundle_groups)
    selected = select_rows(rows, excluded_groups, group_count)
    if set(bundle.manifest['seeds']) != set(map(str, SEED_LIST)):
        raise ValueError('Retained current-model seed set changed')
    root = Path(bundle.manifest['graph_root']).resolve()
    if root != Path(bundle.config['graph_dir']).resolve():
        raise ValueError('Bundle graph roots disagree')
    protocol = dict(version=PROTOCOL_VERSION, rows=selected, group_count=group_count,
        seed_list=list(SEED_LIST), target_seed=TARGET_SEED, method='unconditional',
        graph_root=str(root), references=dict(bundle=bundle_ref, graph_manifest=manifest_ref,
                                             history=history_ref),
        history_sources=history_sources, excluded_group_ids=sorted(excluded_groups),
        excluded_scenario_ids=sorted(excluded_scenes),
        selection=dict(group_order="SHA256('20261027:spear-current-confirmation-v1:group:' + split_group_id)",
            scene_order="SHA256('20261027:spear-current-confirmation-v1:scene:' + scenario_id)",
            tie_break='lexicographic identifier; one scene per selected group',
            allowed_fields=list(MANIFEST_FIELDS), original_split='train',
            graph_required='nonempty relative graph_path and SHA256',
            labels_used=False, motion_used=False, future_used=False, statistics_used=False,
            graph_arrays_loaded=False, insufficient_population='stop_without_replacement_or_retries'))
    _remember(bundle, protocol, stamps)
    return protocol


def validate_protocol(bundle, protocol):
    """Rehash metadata and rerun the full deterministic selection, without graph reads."""
    if not isinstance(protocol, dict):
        raise ValueError('Expected a confirmation protocol dictionary')
    try:
        expected = prepare_protocol(bundle, protocol['references']['history']['path'],
                                    protocol['group_count'])
    except (KeyError, TypeError) as error:
        raise ValueError('Malformed confirmation protocol') from error
    if protocol != expected:
        raise ValueError('Confirmation selection or metadata bindings changed')
    return protocol


def _ensure_validated(bundle, protocol):
    try:
        stamps = _VALIDATED.get(_validation_key(bundle, protocol))
        unchanged = stamps and all(_stamp(path) == stamp for path, stamp in stamps.items())
    except (OSError, TypeError, ValueError):
        unchanged = False
    if not unchanged:
        validate_protocol(bundle, protocol)


def read_confirmation_part(bundle, protocol, row, normalizer, ledger):
    """Read one exact selected TRAIN scene, preserving the frozen label/feature adapter."""
    _ensure_validated(bundle, protocol)
    if (not isinstance(row, dict) or set(row) != ROW_FIELDS
            or row.get('split') != 'train' or row.get('role') != 'confirmation'
            or row not in protocol['rows']
            or row.get('split_group_id') in protocol['excluded_group_ids']
            or row.get('scenario_id') in protocol['excluded_scenario_ids']):
        raise ValueError('Require an exact, nonexcluded confirmation original-TRAIN row')
    root = Path(bundle.manifest['graph_root']).resolve()
    if root != Path(protocol['graph_root']).resolve() or root != Path(bundle.config['graph_dir']).resolve():
        raise ValueError('Confirmation graph root binding changed')
    path = (root / _relative_graph_path(row['graph_path'])).resolve()
    if not path.is_relative_to(root / 'graphs') or path.suffix != '.npz':
        raise ValueError('Confirmation graph path outside registered graphs directory')
    if not path.is_file() or sha256(path) != row['graph_sha256']:
        raise ValueError('Confirmation graph hash mismatch before read')
    access = dict(scenario_id=row['scenario_id'], split_group_id=row['split_group_id'],
        role='confirmation', split='train', path=str(path), sha256=row['graph_sha256'],
        loaded_members=[], status='OPENING')
    if ledger is not None:
        ledger.append(access)
    try:
        with np.load(path, allow_pickle=True) as archive:
            graph = _TrackedGraph(archive)
            try:
                if str(graph['scenario_id'].item()) != row['scenario_id']:
                    raise ValueError('Confirmation graph scenario identity mismatch')
                part = line_graph_from_graph(graph, RiskLabelConfig(**bundle.config['label_config']))
            finally:
                access['loaded_members'] = sorted(graph.loaded)
                access['future_members_used_only_for_unchanged_labels'] = [
                    key for key in sorted(graph.loaded) if key.startswith('future_')]
        if sha256(path) != row['graph_sha256']:
            raise ValueError('Confirmation graph bytes changed during read')
        part['x'] = transform_features(part['x'], normalizer)
        part.update(scenario_id=row['scenario_id'], split='train')
        access['status'] = 'READ_COMPLETE'
        return part
    except Exception:
        access['status'] = 'READ_FAILED'
        raise
