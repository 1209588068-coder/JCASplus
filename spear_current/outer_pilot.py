"""Two-arm original-TRAIN pilot: change only the extra outer sampling domain.

Offline AV2/self-owned GNN research. No graph topology, stored assets, original
validation/test arrays, audit or confirmation roles are changed or read. The
strict-negative arm uses original TRAIN labels for selection; it is a changed
sampling assumption, not a claim that the original unlabeled method is buggy.
Both arms share the existing numerical training core and fixed 26 attachments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .development_diagnostic import registered_rows, read_inputs, _input_hashes
from .development_metrics import population_summary
from .distribution_training import fit_distribution_residual
from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .evaluate import state_hash
from .outer_selection import select_strict_negative_outer
from .role_training import ResidualTrainingConfig
from .runtime import (load_bundle, sha256, restore_generator, restore_surrogate,
                      restore_proxy, restore_victim, restore_clean_reference)
from .train import (OUTPUT_ROOT, TRAINING_CONFIG, DISTRIBUTION_CONFIG, AUC_POLICY,
                    SeededDistributionRegularizer, training_seeds, new_generator,
                    validate_training_contract, fit_fresh_victim, _dev_quality,
                    _write_json, _save_checkpoint)


SEED = 20260621
ARMS = ('original_outer', 'strict_negative_outer')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def preflight(bundle):
    """Validate source identities/assets only; never read graph arrays."""
    validate_training_contract(bundle)
    registered_rows(bundle)
    for key in bundle.manifest['assets']:
        bundle.asset_path(key)
    return dict(status='OUTER_PILOT_PREFLIGHT_OK', seed=SEED, arms=list(ARMS),
                train_scenes=512, dev_scenes=128, graph_arrays_loaded=False,
                training_started=False)


def frozen_plan(bundle):
    """Freeze this pilot's rules before loading graphs or running a model."""
    preflight(bundle)
    from .outer_comparison import GATES
    code = Path(__file__).resolve().parent
    return dict(schema_version=1, created_utc=utc_now(), seed=SEED, arms=list(ARMS),
        hypothesis='Strict eligible negative outer sampling improves development response '
                   'without changing actual poison attachments or clean task quality.',
        changed_variable='Domain of the fixed uniform 512-row extra outer draw',
        selection_uses_original_train_labels=True, motion_quota=None,
        label_access_note='Filtering requires original TRAIN labels for outer candidates; '
                          'this is not a reproduction of unlabeled-only selection.',
        sampling_seed=SEED, target_identity_seed=SEED, seeds=training_seeds(SEED),
        attachments=26, extra_outer=512, outer=538, training_config=TRAINING_CONFIG,
        distribution_config=DISTRIBUTION_CONFIG, generator_checkpoint='final_epoch_199',
        victim_epochs=200, victim_checkpoint=AUC_POLICY,
        coupled_population_note='The same outer rows feed target loss, clean KL, and '
            'the discriminator reference, following the unchanged original training rule. '
            'These population-mediated changes are not separate loss-weight interventions.',
        fixed=['train512/dev128', '26 attachment identities', '24-coordinate support',
               'normalizer', 'pretrained generator and surrogate', 'dual clean proxies',
               'unconditional discriminator architecture/weight', 'fresh GraphSAGE victim',
               'training seeds and budgets', 'one fixed strict-negative target per dev scene'],
        gates=GATES, automatic_followup_jobs=False,
        interpretation='Single-seed development pilot only; one Moving target is about '
                       '3.03 percentage points. No statistical or independent-test claim.',
        original_validation_arrays_allowed=False, original_test_arrays_allowed=False,
        audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        sources=registered_rows(bundle), manifest=dict(path=str(bundle.manifest_path),
            sha256=sha256(bundle.manifest_path)),
        code_sha256={p.name: sha256(p) for p in sorted(code.glob('*.py'))})


def prepare_populations(inputs):
    """Fix both sampling lists and public targets before any model is restored."""
    train, idx = inputs['combined']['train'], inputs['indices']
    selected = select_strict_negative_outer(train, idx, extra_count=512, seed=SEED)
    outers = {ARMS[0]: inputs['outer'].clone(), ARMS[1]: selected['outer']}
    if len(idx['idx_attach']) != 26 or any(len(value) != 538 for value in outers.values()):
        raise ValueError('Pilot requires 26 original attachments plus 512 extra rows')
    if torch.equal(outers[ARMS[0]], outers[ARMS[1]]):
        raise ValueError('No selection contrast exists; stop without training')
    population = dict(selection=selected['selection'], arms={},
        dev_targets=inputs['populations']['dev_public_single'])
    for name, outer in outers.items():
        if (not torch.equal(outer[:26], idx['idx_attach']) or len(outer.unique()) != 538
                or bool(torch.isin(outer, idx['idx_train']).any())):
            raise ValueError('Outer identities changed frozen attachment/supervision scope')
        members = []
        for index in outer.tolist():
            row = dict(inputs['metadata']['train'][index])
            row['is_outer'] = True
            members.append(row)
        population['arms'][name] = dict(outer_indices=outer.tolist(), outer_rows=members,
            outer_sha256=json_hash(outer.tolist()), summary=population_summary(members),
            extra_summary=population_summary(members[26:]))
    dev = population['dev_targets']
    if len(dev) != 112 or sum(row['motion_regime'] == 'moving' for row in dev) != 33:
        raise ValueError('Registered development target population changed')
    population['dev_target_identity_sha256'] = json_hash(dev)
    return outers, population


def fit_arm(generator, surrogate, proxies, inputs, arm, *, progress=None):
    """Reuse the exact numerical loop; only the outer population is different."""
    if arm not in ARMS:
        raise ValueError('Unknown pilot arm')
    train, context, idx, outer = (inputs[k] for k in ('train', 'context', 'indices', 'outer'))
    if (len(idx['idx_attach']) != 26 or len(outer) != 538
            or not torch.equal(outer[:26], idx['idx_attach'])):
        raise ValueError('Frozen pilot counts or attachment prefix changed')
    if arm == ARMS[1] and not bool((train['eligible'][outer] & (train['y'][outer] == 0)).all()):
        raise ValueError('Strict-negative arm contains an ineligible or nonnegative row')
    weights = train['x'].new_ones(context.edge_index.shape[1])
    penalty = FrozenMeanCleanResponsePenalty(proxies, train['x'], context.edge_index,
                                             weights, weight=1.)
    distribution = SeededDistributionRegularizer('unconditional', train['x'], outer,
        generator.selected_coordinates[generator.hard_support],
        seed=training_seeds(SEED)['discriminator'], provenance=dict(
            split='train', role='train', scene_count=512, population=arm,
            real_reference='same_outer_target_rows', internal_dev_arrays_used=False,
            internal_audit_arrays_used=False))
    initial = distribution.snapshot()
    np.random.seed(training_seeds(SEED)['continuation'])
    torch.manual_seed(training_seeds(SEED)['continuation'])
    result = fit_distribution_residual(generator, surrogate, train['x'], context, train['y'],
        idx['idx_train'], idx['idx_attach'], outer, penalty, ResidualTrainingConfig(),
        distribution, progress=progress)
    if (result['selected_epoch'] != 199 or result['selection_policy'] != 'final_epoch'
            or result['config'] != TRAINING_CONFIG or result['distribution_arm'] != 'unconditional'
            or any(result[key] != 200 for key in ('residual_steps', 'surrogate_steps',
                'clean_constraint_calls', 'discriminator_steps', 'distribution_generator_calls'))
            or penalty.calls != 200):
        raise ValueError('Pilot update budget or final-checkpoint policy changed')
    result['dual_clean_history'] = penalty.history
    return result, distribution, initial


def freeze(model):
    model.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False)


def replay_check(bundle, generator, surrogate, victim):
    """Report exact state replay; never select a more favorable historical run."""
    checks = {}
    for name, actual, restore in (
            ('generator', generator, restore_generator),
            ('surrogate', surrogate, restore_surrogate),
            ('victim', victim, restore_victim)):
        historical = restore(bundle, SEED)
        actual_state, old_state = actual.state_dict(), historical.state_dict()
        if actual_state.keys() != old_state.keys():
            raise ValueError('Historical replay state schema changed: ' + name)
        max_delta = max((float((actual_state[k].double() - old_state[k].double()).abs().max())
                         for k in actual_state if actual_state[k].numel()), default=0.)
        checks[name] = dict(exactly_equal=all(torch.equal(actual_state[k], old_state[k])
                    for k in actual_state), max_abs_state_delta=max_delta,
                    current_state_sha256=state_hash(actual), retained_state_sha256=state_hash(historical))
    return dict(models=checks, exact_replay=all(v['exactly_equal'] for v in checks.values()),
        interpretation='A mismatch requires investigation; pilot comparison still uses its '
                       'contemporaneously trained control, not substituted historical metrics.')


def run(bundle, output):
    from .outer_evaluation import evaluate_dev, _dev_view, _validate
    from .outer_comparison import compare_dev_summaries

    folder = Path(output).resolve()
    if folder.exists() or folder == OUTPUT_ROOT or not folder.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Require a NEW output subdirectory under ' + str(OUTPUT_ROOT))
    plan = frozen_plan(bundle)
    folder.mkdir(parents=True, exist_ok=False)
    _write_json(folder / 'plan.json', plan)
    _write_json(folder / 'job.json', dict(pid=os.getpid(), started_utc=utc_now(), seed=SEED))
    torch.set_num_threads(2)
    started = time.monotonic()
    ledger, evaluations = [], {}
    with (folder / 'progress.jsonl').open('x', buffering=1) as stream:
        def progress(record=None, **kwargs):
            row = dict(record or {}, **kwargs, utc=utc_now(), elapsed_seconds=time.monotonic() - started)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            epoch = row.get('epoch')
            if epoch is None or epoch % 20 == 0 or epoch == 199:
                print(json.dumps(row, allow_nan=False), flush=True)

        try:
            progress(stage='read_original_train_dev')
            diagnostic = read_inputs(bundle, plan['sources'], ledger)
            _write_json(folder / 'input_graph_access_ledger.json', ledger)
            before_inputs = _input_hashes(diagnostic)
            outers, populations = prepare_populations(diagnostic)
            _validate(_dev_view(bundle, diagnostic))
            _write_json(folder / 'populations.json', populations)
            _save_checkpoint(folder / 'outer_indices.pt', outers)
            _write_json(folder / 'frozen_input_hashes.json', before_inputs)
            reference = restore_generator(bundle, asset_key='start_generator')
            start_surrogate = restore_surrogate(bundle, asset_key='start_surrogate')
            proxies = [restore_proxy(bundle, index=i) for i in (1, 2)]
            clean = restore_clean_reference(bundle)
            frozen_models = [reference, start_surrogate, *proxies, clean]
            frozen_hashes = [state_hash(m) for m in frozen_models]
            inputs = dict(train=diagnostic['combined']['train'], dev=diagnostic['combined']['dev'],
                          indices=diagnostic['indices'], context=diagnostic['context'])
            quality = [_dev_quality(model, inputs['dev']) for model in proxies]
            if any(row['gate'] != 'PASS_PROXY_QUALITY' for row in quality):
                raise ValueError('Frozen clean proxy failed the existing development quality check')
            _write_json(folder / 'proxy_quality.json', quality)
            for arm in ARMS:
                arm_folder = folder / arm
                arm_folder.mkdir(exist_ok=False)
                arm_inputs = dict(inputs, outer=outers[arm])
                generator = new_generator(reference)
                initial_hash = state_hash(generator)
                progress(arm=arm, stage='fit_generator_started')
                result, distribution, initial = fit_arm(generator, start_surrogate, proxies,
                    arm_inputs, arm, progress=lambda row: progress(row, arm=arm))
                _save_checkpoint(arm_folder / 'generator.pt', generator.state_dict())
                _save_checkpoint(arm_folder / 'surrogate.pt', result['surrogate'].state_dict())
                _save_checkpoint(arm_folder / 'distribution_initial.pt', initial)
                _save_checkpoint(arm_folder / 'distribution_final.pt', distribution.snapshot())
                excluded = {'generator', 'surrogate', 'poisoned_features', 'poisoned_labels',
                            'edge_index', 'edge_weight', 'idx_train', 'idx_attach', 'fixed_outer'}
                _write_json(arm_folder / 'generator_record.json',
                            {k: v for k, v in result.items() if k not in excluded})
                progress(arm=arm, stage='fit_fresh_victim_started')
                victim, victim_record = fit_fresh_victim(bundle, arm_inputs, result, arm_folder,
                    seed=SEED, progress=lambda row: progress(row, arm=arm))
                _save_checkpoint(arm_folder / 'victim_auc.pt', victim.state_dict())
                _write_json(arm_folder / 'victim_record.json', victim_record)
                freeze(generator)
                surrogate = freeze(result['surrogate'])
                if arm == ARMS[0]:
                    _write_json(arm_folder / 'historical_replay.json',
                                replay_check(bundle, generator, surrogate, victim))
                evaluations[arm] = evaluate_dev(bundle, diagnostic, generator,
                    dict(surrogate=surrogate, victim=victim, clean_reference=clean),
                    arm_folder / 'development',
                    progress=lambda **row: progress(arm=arm, **row))
                _write_json(arm_folder / 'development' / 'summary.json', evaluations[arm])
                if (before_inputs != _input_hashes(diagnostic)
                        or frozen_hashes != [state_hash(m) for m in frozen_models]):
                    raise ValueError('Pilot changed source inputs or shared frozen starting models')
                _write_json(arm_folder / 'completion.json', dict(status='ARM_COMPLETE', arm=arm,
                    seed=SEED, initial_generator_state_sha256=initial_hash,
                    generator_steps=200, victim_epochs=200, original_inputs_unchanged=True))
                progress(arm=arm, stage='arm_complete')
                del result, distribution, initial, generator, surrogate, victim
                gc.collect()
            comparison = compare_dev_summaries(evaluations[ARMS[0]], evaluations[ARMS[1]])
            _write_json(folder / 'comparison.json', comparison)
            # Revalidate source code and all frozen bundle assets, without any new graph reads.
            if plan['code_sha256'] != {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}:
                raise ValueError('Source code changed while the pilot was running')
            if sha256(bundle.manifest_path) != plan['manifest']['sha256']:
                raise ValueError('Bundle manifest changed while the pilot was running')
            preflight(bundle)
            completion = dict(status='PILOT_COMPLETE_REVIEW_REQUIRED', seed=SEED,
                completed_arms=list(ARMS), graph_arrays_read=len(ledger),
                optimizer_steps_per_arm=dict(generator=200, surrogate=200, discriminator=200, victim=200),
                victim_epochs_per_arm=200, original_validation_arrays_loaded=False,
                original_test_arrays_loaded=False, audit_arrays_loaded=False,
                confirmation_arrays_loaded=False, automatic_followup_jobs=False,
                elapsed_seconds=time.monotonic() - started, completed_utc=utc_now())
            _write_json(folder / 'completion.json', completion)
            progress(stage='pilot_complete')
            return completion
        except BaseException as error:
            _write_json(folder / 'failure.json', dict(status='FAILED_NO_FURTHER_JOBS',
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                completed_arms=list(evaluations), elapsed_seconds=time.monotonic() - started,
                failed_utc=utc_now()))
            raise
        finally:
            _write_json(folder / 'graph_access_ledger.json', ledger)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.run and args.output_dir is None:
        parser.error('--run requires --output-dir pointing to a NEW results child directory')
    bundle = load_bundle(args.manifest)
    result = run(bundle, args.output_dir) if args.run else preflight(bundle)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
