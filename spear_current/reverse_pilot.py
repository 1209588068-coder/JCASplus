"""Isolated original-TRAIN reverse continuation pilot for local AV2 research.

Keep the forward bundle immutable. Fit matched-clean and label-only controls,
then a risk-one to nonrisk-zero residual continuation and fresh victim. All
models use the same new supervised identities and original clean DEV labels.
This is forward-start adaptation, NOT a from-scratch symmetric comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .development_diagnostic import registered_rows, read_inputs, _input_hashes
from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .evaluate import state_hash
from .reverse_audit import select_positive_targets
from .reverse_selection import select_reverse_populations
from .reverse_training import ReverseTrainingConfig, fit_reverse_distribution_residual
from .runtime import (load_bundle, sha256, restore_generator, restore_surrogate,
                      restore_proxy, restore_clean_reference)
from .train import (OUTPUT_ROOT, DISTRIBUTION_CONFIG, AUC_POLICY, training_seeds,
                    SeededDistributionRegularizer, new_generator, fit_fresh_victim,
                    validate_training_contract, _dev_quality, _write_json, _save_checkpoint)


SEED = 20260621
PHASE0 = OUTPUT_ROOT / 'reverse_risk_phase0_20261028'
ARMS = ('matched_clean', 'label_only', 'reverse_trigger')
GATES = dict(conditional_suppression_min=.80, clean_detected_target_min=10,
    incremental_gain_vs_each_control_min=.03, net_fnr_gain_vs_each_control_min=.03,
    clean_auc_max_drop_vs_matched=.005, positive_recall_max_drop_vs_matched=.05,
    collateral_rate_max_exclusive=.05, normal_predictions_require_both_classes=True,
    matched_clean_auc_min=.75, matched_clean_positive_recall_min=.05)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _phase0(bundle):
    summary_path, target_path = PHASE0 / 'summary.json', PHASE0 / 'positive_targets.json'
    summary = json.loads(summary_path.read_text())
    targets = json.loads(target_path.read_text())
    if (summary['status'] != 'REVERSE_PHASE0_COMPLETE_REVIEW_REQUIRED'
            or summary['retained_manifest']['sha256'] != sha256(bundle.manifest_path)
            or summary['target_hash'] != _json_hash(targets)
            or summary['positive_target_counts'] != {'train': 321, 'dev': 80}
            or not summary['enough_candidates_for_26_plus_512']):
        raise ValueError('Reverse phase0 provenance or target population changed')
    return summary, targets, {str(p): sha256(p) for p in (summary_path, target_path)}


def preflight(bundle):
    validate_training_contract(bundle)
    registered_rows(bundle)
    _phase0(bundle)
    for key in bundle.manifest['assets']:
        bundle.asset_path(key)
    return dict(status='REVERSE_PILOT_PREFLIGHT_OK', seed=SEED, arms=list(ARMS),
                train_scenes=512, dev_scenes=128, target_class=0,
                graph_arrays_loaded=False, training_started=False)


def frozen_plan(bundle):
    preflight(bundle)
    _, targets, phase0_hashes = _phase0(bundle)
    return dict(schema_version=1, created_utc=_now(), seed=SEED, arms=list(ARMS),
        hypothesis='Shared feature trigger causes additional risk misses beyond matched clean '
            'and label-only controls while preserving untriggered risk recognition.',
        direction='original_true_risk_1_to_predicted_nonrisk_0',
        threat_model='offline manipulation of derived pair-node features; not physical vehicle control',
        initialization='forward-pretrained generator and surrogate continuation; fresh GraphSAGE victims',
        limitations=['Not a symmetric from-scratch direction comparison',
                    'Fixed24 support and dual proxies inherit forward development history',
                    'New positive attachment selection is uniform, not original entropy selection',
                    'DEV is development; no blind or independent-test claim'],
        sources=registered_rows(bundle), selection_seed=SEED,
        selection='single uniform no-replacement draw of538 strict positives from original idx_unlabeled; '
                  'first26 attachments and remaining512 outer; no prediction or motion filtering',
        attachment_count=26, extra_outer_count=512, original_supervision_count=1876,
        matched_supervised_count=1902,
        training_config=asdict(ReverseTrainingConfig()), distribution_config=DISTRIBUTION_CONFIG,
        generator_checkpoint='final_epoch_199', victim_epochs_per_arm=200,
        victim_checkpoint=AUC_POLICY, seeds=training_seeds(SEED),
        evaluation_targets=targets['dev'], target_identity_seed=SEED,
        target_rule='fixed original-positive one-per-scene identity hash; no filtering clean misses',
        same_trigger_evaluated_on_all_controls=True,
        proxies='two retained clean optimization proxies; matched clean evaluation control is not a proxy',
        performance_gates=GATES, moving_gate=None,
        gate_note='80% is CONDITIONAL suppression among same-victim clean detections, NOT overall ASR; '
            'overall increment and net FNR gain must separately exceed each control by3pp. '
            'Moving15 reported descriptively, not a powered confirmation.',
        scope='original TRAIN train512/dev128 only', automatic_followup_jobs=False,
        original_validation_arrays_allowed=False, original_test_arrays_allowed=False,
        audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        manifest=dict(path=str(bundle.manifest_path), sha256=sha256(bundle.manifest_path)),
        phase0_hashes=phase0_hashes, code_sha256=_code_hashes())


def prepare_populations(inputs, sources, expected_targets, phase0_summary):
    """Select new training IDs before restoring models; never edit source indices."""
    actual_targets = select_positive_targets(inputs['metadata'], sources)
    if actual_targets['dev'] != expected_targets or _json_hash(actual_targets) != phase0_summary['target_hash']:
        raise ValueError('Positive target identities differ from phase0')
    selected = select_reverse_populations(inputs['combined']['train'], inputs['indices'], seed=SEED)
    candidate_rows = [inputs['metadata']['train'][i] for i in sorted(selected['candidates'].tolist())]
    if (len(selected['idx_attach']) != 26 or len(selected['outer']) != 538
            or len(selected['candidates']) != phase0_summary['candidate_counts']['all']['n']
            or _json_hash(candidate_rows) != phase0_summary['candidate_hash']):
        raise ValueError('Positive candidate domain or registered budget changed')
    idx = {key: value.clone() if isinstance(value, torch.Tensor) else value
           for key, value in inputs['indices'].items()}
    idx['idx_attach'] = selected['idx_attach'].clone()
    attach_set = set(idx['idx_attach'].tolist())
    if len(torch.cat([idx['idx_train'], idx['idx_attach']]).unique()) != 1902:
        raise ValueError('Matched supervised identities must be1876+26 distinct pairs')
    rows = [dict(inputs['metadata']['train'][i], is_attachment=i in attach_set, is_outer=True)
            for i in selected['outer'].tolist()]
    population = dict(selection=selected['selection'], attachment_indices=idx['idx_attach'].tolist(),
        outer_indices=selected['outer'].tolist(), rows=rows, dev_targets=actual_targets['dev'],
        target_identity_sha256=_json_hash(actual_targets['dev']),
        metadata_membership_note='outer rows carry NEW reverse membership; input diagnostic metadata remains unchanged',
        motion_counts={key: sum(row['motion_regime'] == key for row in rows) for key in ('slow', 'moving')})
    return idx, selected['outer'].clone(), population


def supervision_payload(inputs, arm):
    """Same input rows in the controls; only the registered labels may differ."""
    if arm not in ('matched_clean', 'label_only'):
        raise ValueError('Unknown control arm')
    train, idx = inputs['train'], inputs['indices']
    fit, attach = idx['idx_train'], idx['idx_attach']
    if (fit.ndim != 1 or attach.ndim != 1 or fit.dtype != torch.long or attach.dtype != torch.long
            or len(fit) != 1876 or len(attach) != 26
            or len(fit.unique()) != len(fit) or len(attach.unique()) != len(attach)
            or bool(torch.isin(fit, attach).any())
            or bool((torch.cat([fit, attach]) < 0).any())
            or bool((torch.cat([fit, attach]) >= len(train['y'])).any())
            or not bool(train['eligible'][torch.cat([fit, attach])].all())
            or not bool((train['y'][attach] == 1).all())):
        raise ValueError('Require the fixed supervision and strict-positive attachments')
    labels = train['y'].clone()
    if arm == 'label_only':
        labels[attach] = 0
    return dict(poisoned_features=train['x'].clone(), poisoned_labels=labels)


def fit_generator(generator, surrogate, proxies, inputs, progress=None):
    train, context, idx, outer = (inputs[k] for k in ('train', 'context', 'indices', 'outer'))
    if len(idx['idx_attach']) != 26 or len(outer) != 538 or not torch.equal(outer[:26], idx['idx_attach']):
        raise ValueError('Reverse training budget or attachment prefix changed')
    weights = train['x'].new_ones(context.edge_index.shape[1])
    penalty = FrozenMeanCleanResponsePenalty(proxies, train['x'], context.edge_index, weights, weight=1.)
    distribution = SeededDistributionRegularizer('unconditional', train['x'], outer,
        generator.selected_coordinates[generator.hard_support],
        seed=training_seeds(SEED)['discriminator'], provenance=dict(
            split='train', role='train', scene_count=512, population='new_reverse_strict_positive538',
            real_reference='same_outer_original_positive_rows', internal_dev_arrays_used=False,
            internal_audit_arrays_used=False))
    initial = distribution.snapshot()
    np.random.seed(training_seeds(SEED)['continuation'])
    torch.manual_seed(training_seeds(SEED)['continuation'])
    config = ReverseTrainingConfig()
    result = fit_reverse_distribution_residual(generator, surrogate, train['x'], context, train['y'],
        idx['idx_train'], idx['idx_attach'], outer, penalty, config, distribution,
        progress=progress, eligible=train['eligible'])
    if (result['selected_epoch'] != 199 or result['selection_policy'] != 'final_epoch'
            or result['config'] != asdict(config) or result['distribution_arm'] != 'unconditional'
            or any(result[key] != 200 for key in ('residual_steps', 'surrogate_steps',
                'clean_constraint_calls', 'discriminator_steps', 'distribution_generator_calls'))
            or penalty.calls != 200):
        raise ValueError('Reverse update budget or final checkpoint policy changed')
    result['dual_clean_history'] = penalty.history
    return result, distribution, initial


def _freeze(model):
    model.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False)


def review_evaluation(summary):
    """Prespecified exploratory gates; never start another job based on a pass."""
    from .outer_comparison import _normal

    if summary['status'] != 'REVERSE_DEV_EVALUATION_COMPLETE':
        raise ValueError('Incomplete reverse evaluation')
    metrics, gates = summary['metrics'], {}
    required = ('victim', 'matched_clean', 'label_only')
    victim, clean = metrics['victim'], metrics['matched_clean']
    normal_counts = None
    for name in required:
        model = metrics[name]
        _normal(model)
        counts = tuple(model[key] for key in ('normal_pairs', 'normal_positive_pairs', 'normal_negative_pairs'))
        if normal_counts is not None and counts != normal_counts:
            raise ValueError('Clean comparison pools differ')
        normal_counts = counts
        for stratum in ('all', 'slow', 'moving'):
            row = model['targets'][stratum]
            n, tp, fn, induced, repaired = (row[key] for key in
                ('n', 'clean_detected', 'baseline_misses', 'induced_misses_1to0', 'corrected_misses_0to1'))
            if (any(type(value) is not int or value < 0 for value in (n, tp, fn, induced, repaired))
                    or tp + fn != n or induced > tp or repaired > fn
                    or n != victim['targets'][stratum]['n']):
                raise ValueError('Invalid or different reverse denominators')
            expected = dict(incremental_asr=induced / n if n else None,
                conditional_suppression=induced / tp if tp else None,
                net_fnr_change=(induced - repaired) / n if n else None)
            for key, value in expected.items():
                actual = row[key]
                if (value is None) != (actual is None) or (value is not None and (
                        not math.isfinite(actual) or abs(actual - value) > 1e-12)):
                    raise ValueError('Reverse rate is inconsistent with raw counts')
        if any(model['targets']['all'][key] != sum(model['targets'][s][key] for s in ('slow', 'moving'))
               for key in ('n', 'clean_detected', 'baseline_misses', 'induced_misses_1to0', 'corrected_misses_0to1')):
            raise ValueError('Motion strata counts do not sum')

    def gate(name, value, threshold, operator='>='):
        defined = value is not None and math.isfinite(value)
        passed = defined and ((value >= threshold or math.isclose(value, threshold, rel_tol=0., abs_tol=1e-12))
                              if operator == '>=' else value < threshold)
        gates[name] = dict(value=value, threshold=threshold, operator=operator, passed=bool(passed))

    target = victim['targets']['all']
    gate('conditional_suppression', target['conditional_suppression'], GATES['conditional_suppression_min'])
    gate('clean_detected_targets', target['clean_detected'], GATES['clean_detected_target_min'])
    for control in ('matched_clean', 'label_only'):
        for field, limit in (('incremental_asr', 'incremental_gain_vs_each_control_min'),
                             ('net_fnr_change', 'net_fnr_gain_vs_each_control_min')):
            a, b = target[field], metrics[control]['targets']['all'][field]
            gate(field + '_gain_vs_' + control, a - b if a is not None and b is not None else None, GATES[limit])
    for field, limit in (('clean_pair_auc', 'clean_auc_max_drop_vs_matched'),
                         ('clean_pair_positive_recall', 'positive_recall_max_drop_vs_matched')):
        a, b = victim[field], clean[field]
        gate(field + '_difference_vs_matched', a - b if a is not None and b is not None else None, -GATES[limit])
    gate('matched_clean_auc', clean['clean_pair_auc'], GATES['matched_clean_auc_min'])
    gate('matched_clean_positive_recall', clean['clean_pair_positive_recall'], GATES['matched_clean_positive_recall_min'])
    for name in ('victim', 'matched_clean'):
        gate(name + '_normal_prediction_classes', sum(metrics[name]['clean_prediction_distribution'][key] > 0
             for key in ('class_0', 'class_1')), 2)
    for relation in ('adjacent', 'nonincident'):
        for rate, numerator, denominator in (('positive_suppression', 'induced_misses_1to0', 'positive_pairs'),
                                              ('negative_activation', 'negative_activations_0to1', 'negative_pairs')):
            for name in required:
                row = metrics[name]['collateral'][relation]
                if (any(type(row[key]) is not int or row[key] < 0 for key in (numerator, denominator))
                        or row[numerator] > row[denominator]
                        or row[denominator] != victim['collateral'][relation][denominator]):
                    raise ValueError('Invalid or different collateral denominator/counts')
                expected = row[numerator] / row[denominator] if row[denominator] else None
                actual = row[rate]
                if (expected is None) != (actual is None) or (expected is not None and (
                        not math.isfinite(actual) or abs(expected - actual) > 1e-12)):
                    raise ValueError('Collateral rate is inconsistent with counts')
            gate(relation + '_' + rate, victim['collateral'][relation][rate],
                 GATES['collateral_rate_max_exclusive'], '<')
    return dict(status='REVERSE_PILOT_COMPLETE_REVIEW_REQUIRED', preregistered_gates=GATES,
        acceptable_performance=all(row['passed'] for row in gates.values()), gates=gates,
        target_metrics={name: metrics[name]['targets'] for name in required},
        automatic_followup_jobs=False, multiseed_evidence=False, statistical_significance_claimed=False,
        interpretation='Exploratory forward-start reverse continuation; not overall80% ASR or blind confirmation')


def run(bundle, output):
    from .reverse_evaluation import evaluate_reverse_dev

    folder = Path(output).resolve()
    if folder.exists() or folder == OUTPUT_ROOT or not folder.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Require a NEW output subdirectory under ' + str(OUTPUT_ROOT))
    plan = frozen_plan(bundle)
    phase0_summary, _, _ = _phase0(bundle)
    folder.mkdir(parents=True, exist_ok=False)
    _write_json(folder / 'plan.json', plan)
    _write_json(folder / 'job.json', dict(pid=os.getpid(), started_utc=_now(), seed=SEED))
    torch.set_num_threads(2)
    started, ledger, completed = time.monotonic(), [], []
    with (folder / 'progress.jsonl').open('x', buffering=1) as stream:
        def progress(record=None, **kwargs):
            row = dict(record or {}, **kwargs, utc=_now(), elapsed_seconds=time.monotonic() - started)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            epoch = row.get('epoch')
            if epoch is None or epoch % 20 == 0 or epoch == 199:
                print(json.dumps(row, allow_nan=False), flush=True)

        try:
            progress(stage='read_original_train_dev')
            diagnostic = read_inputs(bundle, plan['sources'], ledger)
            _write_json(folder / 'input_graph_access_ledger.json', ledger)
            before_inputs = _input_hashes(diagnostic)
            idx, outer, population = prepare_populations(diagnostic, plan['sources'],
                plan['evaluation_targets'], phase0_summary)
            _write_json(folder / 'populations.json', population)
            _write_json(folder / 'frozen_input_hashes.json', before_inputs)
            _save_checkpoint(folder / 'reverse_indices.pt', dict(idx_train=idx['idx_train'],
                              idx_attach=idx['idx_attach'], outer=outer))
            inputs = dict(train=diagnostic['combined']['train'], dev=diagnostic['combined']['dev'],
                          indices=idx, context=diagnostic['context'], outer=outer)
            index_hash = _json_hash({key: value.tolist() for key, value in idx.items()})
            reference = restore_generator(bundle, asset_key='start_generator')
            surrogate = restore_surrogate(bundle, asset_key='start_surrogate')
            proxies = [restore_proxy(bundle, index=i) for i in (1, 2)]
            historical_clean = restore_clean_reference(bundle)
            frozen_models = [reference, surrogate, *proxies, historical_clean]
            frozen_hashes = [state_hash(model) for model in frozen_models]
            quality = [_dev_quality(model, inputs['dev']) for model in proxies]
            if any(row['gate'] != 'PASS_PROXY_QUALITY' for row in quality):
                raise ValueError('Retained optimization proxy failed original quality check')
            _write_json(folder / 'proxy_quality.json', quality)
            models, signatures = {}, []
            for arm in ('matched_clean', 'label_only'):
                arm_folder = folder / arm
                arm_folder.mkdir(exist_ok=False)
                progress(arm=arm, stage='fit_control_started')
                payload = supervision_payload(inputs, arm)
                model, record = fit_fresh_victim(bundle, inputs, payload, arm_folder, seed=SEED,
                    progress=lambda row: progress(row, arm=arm))
                _save_checkpoint(arm_folder / 'victim_auc.pt', model.state_dict())
                _write_json(arm_folder / 'victim_record.json', record)
                signatures.append((record['initialization_digest'], record['fit_rng_digest']))
                models[arm] = _freeze(model)
                completed.append(arm)
                progress(arm=arm, stage='control_complete')
                del payload
                gc.collect()
            arm_folder = folder / 'reverse_trigger'
            arm_folder.mkdir(exist_ok=False)
            generator = new_generator(reference)
            progress(arm='reverse_trigger', stage='fit_generator_started')
            result, distribution, initial = fit_generator(generator, surrogate, proxies, inputs,
                progress=lambda row: progress(row, arm='reverse_trigger'))
            _save_checkpoint(arm_folder / 'generator.pt', generator.state_dict())
            _save_checkpoint(arm_folder / 'surrogate.pt', result['surrogate'].state_dict())
            _save_checkpoint(arm_folder / 'distribution_initial.pt', initial)
            _save_checkpoint(arm_folder / 'distribution_final.pt', distribution.snapshot())
            excluded = {'generator', 'surrogate', 'poisoned_features', 'poisoned_labels',
                        'edge_index', 'edge_weight', 'idx_train', 'idx_attach', 'fixed_outer'}
            _write_json(arm_folder / 'generator_record.json', {key: value for key, value in result.items() if key not in excluded})
            progress(arm='reverse_trigger', stage='fit_fresh_victim_started')
            victim, record = fit_fresh_victim(bundle, inputs, result, arm_folder, seed=SEED,
                progress=lambda row: progress(row, arm='reverse_trigger'))
            _save_checkpoint(arm_folder / 'victim_auc.pt', victim.state_dict())
            _write_json(arm_folder / 'victim_record.json', record)
            signatures.append((record['initialization_digest'], record['fit_rng_digest']))
            if len(set(signatures)) != 1:
                raise ValueError('Victim initialization/dropout seed differs between controls')
            models['victim'], models['historical_reference'] = _freeze(victim), historical_clean
            _freeze(generator)
            completed.append('reverse_trigger')
            progress(stage='development_evaluation_started')
            evaluation = evaluate_reverse_dev(bundle, diagnostic, generator, models,
                plan['evaluation_targets'], folder / 'development', progress=lambda **row: progress(**row))
            comparison = review_evaluation(evaluation)
            if (before_inputs != _input_hashes(diagnostic)
                    or frozen_hashes != [state_hash(model) for model in frozen_models]
                    or index_hash != _json_hash({key: value.tolist() for key, value in idx.items()})):
                raise ValueError('Reverse pilot mutated frozen inputs, starting models, or selection')
            if plan['code_sha256'] != _code_hashes() or sha256(bundle.manifest_path) != plan['manifest']['sha256']:
                raise ValueError('Source or retained manifest changed during run')
            for path, digest in plan['phase0_hashes'].items():
                if sha256(path) != digest:
                    raise ValueError('Phase0 targets/provenance changed during run')
            preflight(bundle)
            _write_json(folder / 'comparison.json', comparison)
            completion = dict(status='REVERSE_PILOT_COMPLETE_REVIEW_REQUIRED', seed=SEED,
                completed_arms=completed, acceptable_performance=comparison['acceptable_performance'],
                elapsed_seconds=time.monotonic() - started, completed_utc=_now(),
                forward_assets_unchanged=True, original_inputs_unchanged=True,
                graph_reads=len(ledger), original_val_arrays_loaded=False, original_test_arrays_loaded=False,
                audit_arrays_loaded=False, confirmation_arrays_loaded=False, automatic_followup_jobs=False)
            _write_json(folder / 'completion.json', completion)
            progress(stage='complete', acceptable_performance=comparison['acceptable_performance'])
            return completion
        except BaseException as error:
            _write_json(folder / 'failure.json', dict(status='FAILED_NO_FURTHER_JOBS',
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                completed_arms=completed, elapsed_seconds=time.monotonic() - started, failed_utc=_now()))
            raise
        finally:
            _write_json(folder / 'graph_access_ledger.json', ledger)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.run and args.output_dir is None:
        parser.error('--run requires --output-dir NEW_RESULTS_DIR')
    bundle = load_bundle()
    result = run(bundle, args.output_dir) if args.run else preflight(bundle)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
