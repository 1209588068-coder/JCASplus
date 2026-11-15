"""Metadata-only Phase 0 for reverse risk suppression in local AV2 research.

Reuse saved clean predictions; do not load graphs, restore models, or train.
The retained forward bundle is verified and never changed. Positive diagnostic
targets are selected by identity BEFORE reading any prediction CSV. This is
development evidence, not blind confirmation or a reverse attack result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'results/moving_development_forward_v1_20261027'
MODELS = ('surrogate', 'victim', 'clean_reference')
SEEDS = (20260621, 20260622, 20260623)
IDENTITY_FIELDS = ('scenario_id', 'split_group_id', 'role', 'pair_index',
                   'role_pair_index', 'src_track_id', 'dst_track_id',
                   'true_label', 'eligible', 'motion_regime',
                   'is_attachment', 'is_outer')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _validate_metadata(metadata, sources):
    if set(metadata) != {'train', 'dev'} or set(sources) != {'train', 'dev'}:
        raise ValueError('Only original TRAIN internal train/dev are allowed')
    seen_scenes, seen_groups = set(), set()
    for role in ('train', 'dev'):
        registered = {}
        for source in sources[role]:
            scene, group = source['scenario_id'], source['split_group_id']
            if (source.get('split') != 'train' or source.get('role') != role
                    or not scene or not group or scene in seen_scenes or group in seen_groups):
                raise ValueError('Invalid, repeated, or overlapping source scene/group')
            seen_scenes.add(scene)
            seen_groups.add(group)
            registered[scene] = group
        role_indices, identities, tracks = set(), set(), set()
        for row in metadata[role]:
            if any(key not in row for key in IDENTITY_FIELDS):
                raise ValueError('Missing pair metadata')
            if (row['role'] != role or row['scenario_id'] not in registered
                    or registered[row['scenario_id']] != row['split_group_id']):
                raise ValueError('Pair identity differs from registered source')
            for key in ('pair_index', 'role_pair_index'):
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError('Invalid pair index')
            if type(row['true_label']) is not int or row['true_label'] not in (-1, 0, 1):
                raise ValueError('Invalid true label')
            for key in ('eligible', 'is_attachment', 'is_outer'):
                if type(row[key]) is not bool:
                    raise ValueError('Invalid boolean metadata')
            if row['eligible'] and row['true_label'] == -1:
                raise ValueError('Unknown label cannot be eligible')
            if row['motion_regime'] not in ('slow', 'moving'):
                raise ValueError('Invalid motion regime')
            src, dst = row['src_track_id'], row['dst_track_id']
            if not isinstance(src, str) or not isinstance(dst, str) or not src or not dst or src == dst:
                raise ValueError('Invalid track identity')
            identity = (row['scenario_id'], row['pair_index'])
            track_identity = (row['scenario_id'], tuple(sorted((src, dst))))
            if (identity in identities or row['role_pair_index'] in role_indices
                    or track_identity in tracks):
                raise ValueError('Duplicate pair identity')
            identities.add(identity)
            tracks.add(track_identity)
            role_indices.add(row['role_pair_index'])
        if role_indices != set(range(len(metadata[role]))):
            raise ValueError('Role indices must cover the complete metadata population')


def select_positive_targets(metadata, sources, seed=20260621):
    """One strict positive per source scene, independent of predictions/order.

    Every source remains in coverage reporting, including scenes without a
    positive. No retry, model-conditioned filtering, or scene replacement.
    Membership flags refer to the OLD forward experiment, not reverse selection.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    _validate_metadata(metadata, sources)
    result = {}
    for role in ('train', 'dev'):
        by_scene = {row['scenario_id']: [] for row in sources[role]}
        for row in metadata[role]:
            if row['eligible'] and row['true_label'] == 1:
                by_scene[row['scenario_id']].append(row)
        result[role] = []
        for source in sources[role]:
            candidates = by_scene[source['scenario_id']]
            if candidates:
                def rank(row):
                    identity = f"{seed}:spear-audit-target:{row['scenario_id']}:{row['src_track_id']}:{row['dst_track_id']}"
                    return hashlib.sha256(identity.encode()).hexdigest(), row['pair_index']
                chosen = min(candidates, key=rank)
                result[role].append({key: chosen[key] for key in IDENTITY_FIELDS})
    return result


def clean_positive_stats(rows, model):
    """All fixed positives, including baseline misses. No reverse ASR inferred."""
    from .reverse_metrics import reverse_response_metrics
    rows = list(rows)
    adapted = []
    for row in rows:
        copy = dict(row)
        for suffix in ('pred', 'p'):
            for state in ('clean', 'trigger'):
                copy[f'{model}_{state}_{suffix}'] = row[f'{model}_{suffix}']
        adapted.append(copy)
    # Identical responses are ONLY an internal reuse of pure input validation
    # and counting. Fabricated triggered metrics are never emitted or reported.
    metrics = reverse_response_metrics(adapted, model)
    keys = ('n', 'clean_detected', 'baseline_misses', 'clean_recall_ceiling', 'baseline_fnr')
    return {stratum: dict({key: value[key] for key in keys}, scenes=len({
        row['scenario_id'] for row in rows if stratum == 'all' or row['motion_regime'] == stratum}))
        for stratum, value in metrics.items()}


def _read_clean(path, metadata):
    expected = {(role, row['role_pair_index']): row
                for role in ('train', 'dev') for row in metadata[role] if row['eligible']}
    seen, parsed = set(), []
    with Path(path).open(newline='') as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            for key in ('pair_index', 'role_pair_index', 'true_label', 'label'):
                row[key] = int(row[key])
            for key in ('eligible', 'is_attachment', 'is_outer'):
                if row[key] not in ('True', 'False'):
                    raise ValueError('Invalid saved boolean')
                row[key] = row[key] == 'True'
            key = (row['role'], row['role_pair_index'])
            if key in seen or key not in expected:
                raise ValueError('Unexpected or repeated saved prediction identity')
            original = expected[key]
            if (any(row[field] != original[field] for field in IDENTITY_FIELDS)
                    or row['label'] != row['true_label']):
                raise ValueError('Prediction identity or label is not aligned')
            seen.add(key)
            for model in MODELS:
                row[model + '_pred'] = int(row[model + '_pred'])
                row[model + '_p'] = float(row[model + '_p'])
                pred, probability = row[model + '_pred'], row[model + '_p']
                if pred not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('Invalid saved model prediction')
                if abs(probability - .5) > 1e-6 and pred != int(probability > .5):
                    raise ValueError('Saved argmax and probability disagree')
            parsed.append(row)
    if seen != set(expected):
        raise ValueError('Incomplete saved clean prediction population')
    return parsed


def _output_path(path):
    target = Path(path).resolve()
    results = (ROOT / 'results').resolve()
    if target == results or not target.is_relative_to(results) or target.exists():
        raise ValueError('Output must be a NEW child directory of JCAS/results')
    return target


def _write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(output_dir, source=DEFAULT_SOURCE):
    """Only hashes, existing metadata/prediction CSVs, and frozen index tensors."""
    from .runtime import load_bundle
    import torch

    target, source = _output_path(output_dir), Path(source).resolve()
    bundle = load_bundle()
    manifest_hash = _sha(bundle.manifest_path)
    input_hashes = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Diagnostic input escaped its registered source directory')
        digest = _sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed diagnostic input: ' + str(path))
        input_hashes[str(path)] = digest
        return json.loads(path.read_text())

    plan = read(source / 'plan.json')
    completion = read(source / 'completion.json')
    population = read(source / 'population.json')
    populations = read(source / 'populations_and_targets.json')
    if (plan['bundle'] != {'path': str(bundle.manifest_path), 'sha256': manifest_hash}
            or plan['sources'] != {role: bundle.config['roles'][role] for role in ('train', 'dev')}
            or plan['seeds'] != list(SEEDS) or completion['seeds'] != list(SEEDS)
            or completion['status'] != 'DEVELOPMENT_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED'
            or completion['optimizer_steps'] != 0
            or _digest(populations) != population['input_hashes']['populations']):
        raise ValueError('Diagnostic provenance differs from the retained contract')
    for scope in ('original_val', 'original_test', 'internal_audit', 'confirmation'):
        if plan[scope + '_arrays_allowed'] is not False or completion[scope + '_arrays_loaded'] is not False:
            raise ValueError('Disallowed prior diagnostic scope')
    metadata = {role: populations[role + '_all'] for role in ('train', 'dev')}
    # Freeze identities before opening any clean predictions.
    targets = select_positive_targets(metadata, plan['sources'])
    target_hash = _digest(targets)
    coverage = []
    for role in ('train', 'dev'):
        by_scene = {row['scenario_id']: [] for row in plan['sources'][role]}
        for row in metadata[role]:
            by_scene[row['scenario_id']].append(row)
        for scene, rows in by_scene.items():
            positive = [row for row in rows if row['eligible'] and row['true_label'] == 1]
            coverage.append(dict(role=role, scenario_id=scene, pairs=len(rows), positives=len(positive),
                                 slow_positive=sum(row['motion_regime'] == 'slow' for row in positive),
                                 moving_positive=sum(row['motion_regime'] == 'moving' for row in positive),
                                 has_positive=bool(positive)))

    index_path = bundle.asset_path('indices')
    input_hashes[str(index_path)] = _sha(index_path)
    indices = torch.load(index_path, weights_only=True, map_location='cpu')
    ordered = sorted(metadata['train'], key=lambda row: row['role_pair_index'])
    expected_labels = [row['true_label'] for row in ordered]
    if indices['train_labels'].tolist() != expected_labels:
        raise ValueError('Frozen training labels differ from metadata')
    domains = {}
    for name in ('idx_train', 'idx_attach', 'idx_unlabeled'):
        values = indices[name]
        if (values.ndim != 1 or values.dtype != torch.int64 or len(values) != len(set(values.tolist()))
                or bool((values < 0).any()) or bool((values >= len(ordered)).any())):
            raise ValueError('Invalid frozen index domain')
        domains[name] = set(values.tolist())
    if (domains['idx_train'] & domains['idx_attach']
            or domains['idx_unlabeled'] != set(range(len(ordered))) - domains['idx_train'] - domains['idx_attach']
            or len(domains['idx_train']) != 1876 or len(domains['idx_attach']) != 26
            or any(not ordered[i]['eligible'] or ordered[i]['true_label'] != 0 for i in domains['idx_attach'])
            or any(not ordered[i]['eligible'] for i in domains['idx_train'])):
        raise ValueError('Original supervised/attachment/unlabeled domains changed')
    candidates = [row for row in ordered if row['role_pair_index'] in domains['idx_unlabeled']
                  and row['eligible'] and row['true_label'] == 1]
    candidate_counts = {stratum: dict(n=len(group), scenes=len({row['scenario_id'] for row in group}))
                        for stratum in ('all', 'slow', 'moving')
                        for group in [[row for row in candidates if stratum == 'all' or row['motion_regime'] == stratum]]}

    reports = {Path(item['path']).resolve(): item['sha256'] for item in completion['reports']}
    baseline_rows, reference_signature = [], None
    for seed in SEEDS:
        summary_path = source / f'seed{seed}/summary.json'
        prior = read(summary_path, reports[summary_path])
        if (prior['seed'] != seed or prior['status'] != 'DEVELOPMENT_DIAGNOSTIC_COMPLETE'
                or prior['optimizer_steps'] != 0 or any(prior[key] is not True for key in (
                    'weights_unchanged', 'inputs_unchanged', 'rng_unchanged'))):
            raise ValueError('Prior diagnostic did not preserve frozen state')
        csv_path = Path(prior['raw_clean']['path']).resolve()
        if csv_path != source / f'seed{seed}/clean_pair_scores.csv' or _sha(csv_path) != prior['raw_clean']['sha256']:
            raise ValueError('Clean CSV path or hash mismatch')
        input_hashes[str(csv_path)] = prior['raw_clean']['sha256']
        clean = _read_clean(csv_path, metadata)
        signature = [(row['role'], row['role_pair_index'], row['clean_reference_pred'], row['clean_reference_p'])
                     for row in clean]
        if reference_signature is not None and signature != reference_signature:
            raise ValueError('Frozen clean reference differs across seeds')
        reference_signature = signature
        for role in ('train', 'dev'):
            selected = {row['role_pair_index'] for row in targets[role]}
            positives = [row for row in clean if row['role'] == role and row['true_label'] == 1]
            for pool, rows in (('all_positive', positives), ('positive_single_per_scene',
                               [row for row in positives if row['role_pair_index'] in selected])):
                for model in MODELS:
                    for stratum, stats in clean_positive_stats(rows, model).items():
                        baseline_rows.append(dict(seed=seed, role=role, pool=pool, model=model,
                                                  stratum=stratum, **stats))

    for path, digest in input_hashes.items():
        if _sha(path) != digest:
            raise ValueError('Input changed during audit: ' + path)
    load_bundle()  # Recheck every retained asset hash; no deserialization.
    if _sha(bundle.manifest_path) != manifest_hash or _digest(targets) != target_hash:
        raise ValueError('Retained manifest or positive identities changed')
    summary = dict(
        status='REVERSE_PHASE0_COMPLETE_REVIEW_REQUIRED', direction='risk_1_to_nonrisk_0',
        boundary='offline local AV2 and researcher-owned GNN; original TRAIN internal train/dev only',
        source_diagnostic=str(source), source_hashes=input_hashes,
        retained_manifest=dict(path=str(bundle.manifest_path), sha256=manifest_hash, unchanged=True),
        code_hashes={str(Path(__file__)): _sha(Path(__file__)),
                     str(Path(__file__).with_name('reverse_metrics.py')): _sha(Path(__file__).with_name('reverse_metrics.py'))},
        target_rule='one strict original positive per scene, identity hash seed20260621, never prediction filtered',
        target_hash=target_hash, positive_target_counts={role: len(rows) for role, rows in targets.items()},
        source_scene_counts={role: len(rows) for role, rows in plan['sources'].items()},
        candidate_counts=candidate_counts, candidate_hash=_digest(candidates),
        proposed_attachment_count=26, proposed_extra_outer_count=512,
        enough_candidates_for_26_plus_512=len(candidates) >= 538,
        reverse_attachments_selected=False, reverse_outer_selected=False,
        baseline_capability=baseline_rows,
        interpretation='clean recall is a ceiling only for that fixed checkpoint and pool; NOT reverse ASR',
        frozen_reference_is_blind=False, reverse_asr=None,
        new_graph_array_reads=0, model_inferences=0, optimizer_steps=0,
        original_val_arrays_loaded=False, original_test_arrays_loaded=False,
        internal_audit_arrays_loaded=False, confirmation_arrays_loaded=False,
        automatic_followup=False)
    target.mkdir(parents=True, exist_ok=False)
    _write_json(target / 'positive_targets.json', targets)
    _write_csv(target / 'positive_targets.csv', [row for role in ('train', 'dev') for row in targets[role]], IDENTITY_FIELDS)
    _write_csv(target / 'training_positive_candidates.csv', candidates, IDENTITY_FIELDS)
    _write_csv(target / 'scene_coverage.csv', coverage, list(coverage[0]))
    _write_csv(target / 'baseline_capability.csv', baseline_rows, list(baseline_rows[0]))
    _write_json(target / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in (
        'status', 'positive_target_counts', 'candidate_counts', 'enough_candidates_for_26_plus_512',
        'new_graph_array_reads', 'model_inferences', 'optimizer_steps', 'reverse_asr')}, indent=2))
    print('Review:', target / 'summary.json', '; no further jobs')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print('NOT_RUN: metadata/CSV audit only; requires --run --output-dir NEW_RESULTS_DIR. No inputs loaded.')
        return
    if args.output_dir is None:
        parser.error('--run requires --output-dir')
    run_audit(args.output_dir)


if __name__ == '__main__':
    main()
