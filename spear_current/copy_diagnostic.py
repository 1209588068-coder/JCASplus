"""Fixed seed21 original-TRAIN DEV copy-consistency diagnosis; no training.

Prepare seals the protocol; run reads DEV128 only and uses the current retained
final models. No old result is overwritten or used to select a favorable arm.
The comparison is input sensitivity, not proof that copy conflicts cause ASR.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile
import time

import torch

from .copy_consistency import COPY_GROUPS, CopyConsistentGenerator
from .data import combine_parts, load_normalizer, read_role_part
from .development_diagnostic import registered_rows, choose_target, row_metadata, _write
from .evaluate import state_hash
from .outer_evaluation import evaluate_dev, _dev_view, _validate, _fingerprint, _rng_fingerprint
from .runtime import (load_bundle, sha256, restore_generator, restore_surrogate,
                      restore_victim, restore_clean_reference)
from .support_schema import SELECTED_COORDINATES


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260621
ARMS = ('identity', 'pair_mean')
VERSION = 'within_pair_copy_fixed_dev_v1'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _progress(**record):
    print(json.dumps(record, allow_nan=False), flush=True)


def _directory(path):
    path = Path(path).resolve()
    if path == ROOT / 'results' or not path.is_relative_to(ROOT / 'results'):
        raise ValueError('Output must be a new child of results/')
    return path


def protocol(bundle):
    """Only metadata and file-byte hashes; no graph arrays/model restoration."""
    sources = registered_rows(bundle)['dev']
    coordinates = bundle.manifest['selected_coordinates']
    support = bundle.manifest['hard_support']
    if tuple(coordinates) != SELECTED_COORDINATES or len(support) != 30 or sum(support) != 24:
        raise ValueError('Current coordinate/support contract changed')
    if any(not support[coordinates.index(c)] for group in COPY_GROUPS for c in group):
        raise ValueError('All six copy groups must be in the original support')
    model_keys = {name: bundle.manifest['seeds'][str(SEED)][name]
                  for name in ('generator', 'surrogate', 'victim')}
    model_keys['clean_reference'] = 'clean_reference'
    required = set(model_keys.values()) | {'normalizer', 'support'}
    assets = {key: dict(path=str(bundle.asset_path(key)), sha256=bundle.manifest['assets'][key]['sha256'])
              for key in sorted(required)}
    return dict(version=VERSION, seed=SEED, arms=list(ARMS), sources=sources,
        manifest=dict(path=str(bundle.manifest_path), sha256=sha256(bundle.manifest_path)),
        asset_keys=model_keys, assets=assets,
        code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        target_seed=20260621, target_rule='existing identity hash; eligible true-negative; no prediction filtering',
        expected_scenes=128, expected_targets=112, expected_moving=33, expected_slow=79,
        selected_coordinates=coordinates, hard_support=support,
        copy_groups=[list(g) for g in COPY_GROUPS],
        mapping='old domain clamp then arithmetic mean within each of six triplets',
        frozen_models='same current final seed21 G, surrogate, victim, clean reference in both arms',
        normal_metrics_population='all eligible DEV pairs, not the all-negative target pool',
        normal_ap='average_precision_score; additional CSV-derived diagnostic',
        scope='reused original TRAIN DEV; no independent confirmation or blind-model claim',
        interpretation='fixed-model sensitivity; projection also changes values and modification norms',
        optimizer_steps=0, parameter_updates=0, checkpoint_selection='none',
        thresholds='unchanged binary argmax; no threshold selection',
        performance_gate=None, automatic_followup=False,
        train_role_arrays_allowed=False, audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        original_validation_arrays_allowed=False, original_test_arrays_allowed=False)


def prepare(bundle, output):
    spec = protocol(bundle)
    output = _directory(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'plan.json', dict(created_utc=_now(), protocol=spec))
    snapshot = output / 'source_snapshot.tar'
    with tarfile.open(snapshot, 'x') as archive:
        for name in spec['code_sha256']:
            archive.add(Path(__file__).parent / name, arcname='spear_current/' + name)
    if protocol(bundle) != spec:
        raise ValueError('Sources changed during preparation')
    _write(output / 'registration.json', dict(created_utc=_now(),
        plan_sha256=sha256(output / 'plan.json'), source_snapshot_sha256=sha256(snapshot),
        graph_arrays_loaded=False, model_checkpoints_loaded=False, optimizer_steps=0))
    return output / 'plan.json'


def validate_plan(bundle, plan_path):
    path = Path(plan_path).resolve()
    if path.name != 'plan.json' or _directory(path.parent) != path.parent:
        raise ValueError('Require the prepared plan.json under results/')
    registration = json.loads((path.parent / 'registration.json').read_text())
    if (registration['plan_sha256'] != sha256(path)
            or registration['source_snapshot_sha256'] != sha256(path.parent / 'source_snapshot.tar')):
        raise ValueError('Prepared plan or source snapshot changed')
    plan = json.loads(path.read_text())
    if plan.get('protocol') != protocol(bundle):
        raise ValueError('Prepared sources, assets, population, or diagnostic rules changed')
    return plan


def read_dev_inputs(bundle, ledger):
    """Read only the registered DEV role; determine all targets before inference."""
    sources = registered_rows(bundle)['dev']
    normalizer = load_normalizer(bundle)
    parts = []
    for i, source in enumerate(sources, 1):
        parts.append(read_role_part(bundle, source, 'dev', normalizer, ledger))
        if i % 32 == 0 or i == len(sources):
            _progress(stage='read_dev', completed=i, total=len(sources))
    combined = combine_parts(parts)
    metadata, targets, population = [], [], []
    for part, source, offset in zip(parts, sources, combined['offsets']):
        rows = row_metadata(part, source, offset, 'dev', set(), set())
        metadata.extend(rows)
        target = choose_target(part)
        targets.append(target)
        if target is not None:
            population.append(rows[target])
    if (len(parts) != 128 or len(population) != 112
            or sum(r['motion_regime'] == 'moving' for r in population) != 33
            or len(ledger) != 128
            or any(r['split'] != 'train' or r['role'] != 'dev' or r['status'] != 'READ_COMPLETE' for r in ledger)):
        raise ValueError('Frozen DEV population, target strata, or access scope changed')
    inputs = {key: dict(dev=value) for key, value in dict(parts=parts, combined=combined,
               metadata=metadata, targets=targets).items()}
    _validate(_dev_view(bundle, inputs))
    return inputs, normalizer, population


def evaluate_arm(bundle, inputs, anchor, normalizer, models, mode, directory, *, seed=SEED):
    """Hook the existing single forward per target; never regenerate for metrics."""
    if type(seed) is not int or seed not in (20260621, 20260622, 20260623):
        raise ValueError('Unregistered evaluation seed')
    if mode not in ARMS:
        raise ValueError('Unknown diagnostic arm')
    generator = CopyConsistentGenerator(anchor, normalizer, mode=mode, frozen=True)
    parent_state = anchor.state_dict()
    if any(not torch.equal(generator.state_dict()[k], v) for k, v in parent_state.items()):
        raise ValueError('Arm does not share the exact retained generator weights')
    expected = []
    for part, target, offset in zip(inputs['parts']['dev'], inputs['targets']['dev'],
                                   inputs['combined']['dev']['offsets']):
        if target is not None:
            expected.append((part, target, inputs['metadata']['dev'][offset + target]))
    records = []

    def capture(module, args, output):
        if len(records) >= len(expected):
            raise ValueError('Unexpected extra generator forward')
        part, target, row = expected[len(records)]
        features, context, targets = args
        if (targets.tolist() != [target] or output.shape != (1, 30)
                or not torch.equal(features, torch.as_tensor(part['x'], dtype=torch.float32))
                or not torch.equal(context.edge_index, torch.as_tensor(part['edge_index'], dtype=torch.long))):
            raise ValueError('Captured generator input differs from the fixed target scene')
        clean = features[targets[:, None], module.selected_coordinates]
        records.append(dict(scenario_id=row['scenario_id'], pair_index=int(target),
            src_track_id=row['src_track_id'], dst_track_id=row['dst_track_id'],
            motion_regime=row['motion_regime'], clean30=clean[0].detach().tolist(),
            output30=output[0].detach().tolist()))

    hashes = {k: state_hash(v) for k, v in dict(generator=generator, **models).items()}
    handle = generator.register_forward_hook(capture)
    try:
        result = evaluate_dev(bundle, inputs, generator, models, directory,
            progress=lambda **r: _progress(arm=mode, **r))
    finally:
        handle.remove()
    if len(records) != len(expected):
        raise ValueError('Missing generator-output records')
    if hashes != {k: state_hash(v) for k, v in dict(generator=generator, **models).items()}:
        raise ValueError('Fixed-model evaluation changed weights')
    result.update(mode=mode, seed=seed, model_state_sha256=hashes,
                  generator_contract=generator.contract(),
                  shared_parent_generator_state_sha256=state_hash(anchor),
                  clean_reference_is_independent=False)
    _write(Path(directory) / 'copy_rows.json', dict(mode=mode,
        selected_coordinates=generator.selected_coordinates.tolist(),
        allowed_write_coordinates=generator.selected_coordinates[generator.hard_support].tolist(),
        rows=records))
    result['raw_copy_rows'] = dict(path=str((Path(directory) / 'copy_rows.json').resolve()),
                                  sha256=sha256(Path(directory) / 'copy_rows.json'), rows=len(records))
    _write(Path(directory) / 'summary.json', result)
    return result


def run(bundle, plan_path):
    from .copy_diagnostic_reporting import build_comparison

    plan = validate_plan(bundle, plan_path)
    output = Path(plan_path).resolve().parent
    # Exclusive claim prevents accidental reruns even after a partially failed run.
    _write(output / 'execution_claim.json', dict(started_utc=_now(), plan_sha256=sha256(plan_path),
                                               optimizer_steps=0))
    start = time.monotonic()
    ledger = []
    try:
        inputs, normalizer, targets = read_dev_inputs(bundle, ledger)
        _write(output / 'population.json', dict(targets=targets,
            dev_input_sha256=_fingerprint(_dev_view(bundle, inputs)),
            target_count=len(targets), moving_count=sum(r['motion_regime'] == 'moving' for r in targets)))
        anchor = restore_generator(bundle, SEED)
        models = dict(surrogate=restore_surrogate(bundle, SEED), victim=restore_victim(bundle, SEED),
                      clean_reference=restore_clean_reference(bundle))
        hashes = {k: state_hash(v) for k, v in dict(generator=anchor, **models).items()}
        before_rng = _rng_fingerprint()
        for mode in ARMS:
            evaluate_arm(bundle, inputs, anchor, normalizer, models, mode, output / mode)
        comparison = build_comparison(output / ARMS[0], output / ARMS[1])
        _write(output / 'comparison.json', comparison)
        if hashes != {k: state_hash(v) for k, v in dict(generator=anchor, **models).items()}:
            raise ValueError('Fixed source model state changed')
        if before_rng != _rng_fingerprint():
            raise ValueError('Fixed-model diagnosis consumed inference RNG')
        if plan != validate_plan(bundle, plan_path):
            raise ValueError('Plan changed during diagnosis')
        _write(output / 'completion.json', dict(status='COPY_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED',
            completed_utc=_now(), elapsed_seconds=time.monotonic() - start,
            comparison_sha256=sha256(output / 'comparison.json'),
            model_state_sha256=hashes, weights_unchanged=True, rng_unchanged=True,
            graph_reads=len(ledger), optimizer_steps=0, parameter_updates=0,
            training_started=False, performance_gate=None, automatic_followup=False,
            train_role_arrays_loaded=False, audit_arrays_loaded=False, confirmation_arrays_loaded=False,
            original_validation_arrays_loaded=False, original_test_arrays_loaded=False))
    except Exception as error:
        _write(output / 'failure.json', dict(error=str(error), type=type(error).__name__, at=_now()))
        raise
    finally:
        _write(output / 'graph_access_ledger.json', ledger)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check-only', action='store_true')
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.prepare and (args.output_dir is None or args.plan is not None):
        parser.error('--prepare requires --output-dir, not --plan')
    if args.run and (args.plan is None or args.output_dir is not None):
        parser.error('--run requires --plan, not --output-dir')
    if not (args.prepare or args.run) and (args.output_dir is not None or args.plan is not None):
        parser.error('Check-only accepts no output directory or plan')
    bundle = load_bundle()
    if args.prepare:
        print('Prepared: ' + str(prepare(bundle, args.output_dir)), flush=True)
    elif args.run:
        torch.set_num_threads(2)
        run(bundle, args.plan)
        print('Review: ' + str(args.plan.parent / 'comparison.json'), flush=True)
    else:
        protocol(bundle)
        print('COPY_DIAGNOSTIC_PREFLIGHT_OK; no graph arrays/model restoration/training', flush=True)


if __name__ == '__main__':
    main()
