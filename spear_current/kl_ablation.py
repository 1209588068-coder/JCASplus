"""Registered clean-KL continuation ablations; local AV2/owned-model research.

Three fixed arms, three fixed seeds, original TRAIN512/DEV128 only. This is a
stage-specific warm-start ablation: the shared parent already used dual KL.
It is NOT an end-to-end no-KL baseline or a blind evaluation. Existing default
training, model assets, and historical results are never modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import torch

from .copy_diagnostic import evaluate_arm
from .copy_diagnostic_reporting import _load_arm, _copy_stats, METADATA, MODELS, STRATA
from .copy_training_reporting import (
    GATES, _acceptability, _initial, _victim, _normal_by_pair_motion, _paired_targets,
)
from .data import load_normalizer
from .development_diagnostic import registered_rows, read_inputs, _input_hashes
from .distribution_training import fit_distribution_residual
from .evaluate import state_hash
from .kl_ablation_penalty import FixedProxyAblationPenalty
from .outer_evaluation import _rng_fingerprint
from .role_training import ResidualTrainingConfig, _state_hash as training_state_hash
from .runtime import (load_bundle, sha256, restore_generator, restore_surrogate,
                      restore_proxy, restore_clean_reference, restore_victim)
from .train import (SEEDS, TRAINING_CONFIG, DISTRIBUTION_CONFIG, AUC_POLICY,
    validate_training_contract, training_seeds, new_generator, new_distribution,
    no_update_preflight, fit_fresh_victim, _dev_quality, _write_json, _save_checkpoint)


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('dual', 'single_proxy1', 'disabled')
COEFFICIENTS = dict(dual=(.5, .5), single_proxy1=(1., 0.), disabled=(0., 0.))
VERSION = 'stage_specific_clean_kl_ablation_v1'
COMMON = ('initial_cpu_rng_sha256', 'distribution_initial_state_sha256',
          'victim_initialization_digest', 'victim_fit_rng_digest')


def now():
    return datetime.now(timezone.utc).isoformat()


def directory(path):
    path = Path(path).resolve()
    if path == ROOT / 'results' or not path.is_relative_to(ROOT / 'results'):
        raise ValueError('Require a new child directory under results/')
    return path


def protocol(bundle):
    validate_training_contract(bundle)
    keys = {'start_generator', 'start_surrogate', 'clean_proxy1', 'clean_proxy2',
            'clean_reference', 'normalizer', 'indices', 'input_contract',
            'context_binding', 'score_permutation', 'support'}
    for seed in SEEDS:
        keys.update(bundle.manifest['seeds'][str(seed)][k] for k in ('generator', 'surrogate', 'victim'))
    return dict(version=VERSION, seeds=list(SEEDS), arms=list(ARMS),
        coefficients={k: list(v) for k, v in COEFFICIENTS.items()},
        hypothesis='Measure the incremental role of clean-response KL during the registered continuation stage',
        sole_treatment='effective aggregation of the same two frozen clean-proxy KL terms',
        start='bundle start_generator/start_surrogate: completed domain_projection, NOT current final or pristine SPEAR',
        inherited=['trained base G/shadow', 'trained uniform residual', 'dual clean KL history', 'SAGE24', 'domain_projection'],
        limitation='All arms inherit prior dual-KL training; disabled means no KL objective in these 200 new steps only',
        single_proxy_rule='use pre-existing proxy1; no best-proxy selection',
        raw_kl_computation='both proxies evaluated in every arm, including disabled; raw and effective loss reported separately',
        sources=registered_rows(bundle), gates=deepcopy(GATES),
        training_config=TRAINING_CONFIG, distribution_config=DISTRIBUTION_CONFIG,
        victim_epochs=200, generator_selection='final_epoch_199', victim_selection=AUC_POLICY,
        train_scenes=512, dev_scenes=128, attachments=26, outer_rows=538,
        target_counts=dict(all=112, slow=79, moving=33),
        target_policy='same existing identity-hash strict-negative pool; no prediction filter',
        checkpoint_replay='dual G/S/victim tensors must exactly reproduce retained same-seed unconditional models',
        order='seed21 dual,single_proxy1,disabled; seed22 same; seed23 same; serial',
        failure_policy='integrity failure stops queue; performance failure reports and continues fixed queue; no retries',
        audit_arrays_allowed=False, confirmation_arrays_allowed=False,
        original_validation_arrays_allowed=False, original_test_arrays_allowed=False,
        automatic_promotion=False, automatic_followup=False, hyperparameter_search=False,
        manifest=dict(path=str(bundle.manifest_path), sha256=sha256(bundle.manifest_path)),
        assets={k: dict(path=str(bundle.asset_path(k)), sha256=bundle.manifest['assets'][k]['sha256']) for k in sorted(keys)},
        code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def prepare(bundle, output):
    spec = protocol(bundle)
    output = directory(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'plan.json', dict(created_utc=now(), protocol=spec))
    with tarfile.open(output / 'source_snapshot.tar', 'x') as archive:
        for name in spec['code_sha256']:
            archive.add(Path(__file__).parent / name, arcname='spear_current/' + name)
    if protocol(bundle) != spec:
        raise ValueError('Sources changed during preparation')
    _write_json(output / 'registration.json', dict(plan_sha256=sha256(output / 'plan.json'),
        source_snapshot_sha256=sha256(output / 'source_snapshot.tar'),
        graph_arrays_loaded=False, model_checkpoints_loaded=False, training_started=False))
    return output / 'plan.json'


def validate_plan(bundle, path):
    path = Path(path).resolve()
    if path.name != 'plan.json' or directory(path.parent) != path.parent:
        raise ValueError('Require a prepared plan.json under results/')
    registered = json.loads((path.parent / 'registration.json').read_text())
    if (registered['plan_sha256'] != sha256(path)
            or registered['source_snapshot_sha256'] != sha256(path.parent / 'source_snapshot.tar')):
        raise ValueError('Registration changed')
    plan = json.loads(path.read_text())
    if plan['protocol'] != protocol(bundle):
        raise ValueError('Prepared source, population, assets, budgets or rules changed')
    return plan


def freeze(model):
    model.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False)


def loss_preflight(generator, proxies, inputs, mode):
    """Check the actual loss/input gradient without updating any model or RNG."""
    train, context, outer = (inputs[k] for k in ('train', 'context', 'outer'))
    models = [generator, *proxies]
    before, rng = [state_hash(m) for m in models], _rng_fingerprint()
    weights = train['x'].new_ones(context.edge_index.shape[1])
    penalty = FixedProxyAblationPenalty(proxies, train['x'], context.edge_index, weights, mode=mode)
    injected = generator.inject(train['x'], context, outer, weights, detach=False)
    loss = penalty(injected.features, context.edge_index, weights, outer)
    grads = torch.autograd.grad(loss, tuple(generator.residual.parameters()), allow_unused=True)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
        raise ValueError('Nonfinite clean-KL input path')
    norm = sum(float(g.double().square().sum()) for g in grads if g is not None) ** .5
    if (mode == 'disabled' and (float(loss.detach()) != 0 or norm != 0)) or (mode != 'disabled' and norm <= 0):
        raise ValueError('Clean-KL effective gradient does not match registered treatment')
    if (before != [state_hash(m) for m in models] or rng != _rng_fingerprint()
            or any(p.grad is not None for m in models for p in m.parameters())):
        raise ValueError('Zero-step preflight changed protected state')
    return dict(mode=mode, record=penalty.history[-1], residual_gradient_l2=norm, optimizer_steps=0)


def fit_arm(bundle, diagnostic, normalizer, anchor, surrogate, proxies, clean,
            seed, arm, output, progress):
    output.mkdir(exist_ok=False)
    inputs = dict(train=diagnostic['combined']['train'], dev=diagnostic['combined']['dev'],
                  indices=diagnostic['indices'], context=diagnostic['context'], outer=diagnostic['outer'])
    original_inputs = _input_hashes(diagnostic)
    protected = [anchor, surrogate, *proxies, clean]
    original_states = [state_hash(m) for m in protected]
    generator = new_generator(anchor)
    _write_json(output / 'initial_state.json', dict(seed=seed, arm=arm,
        coefficients=list(COEFFICIENTS[arm]), generator_parent_state_sha256=state_hash(anchor),
        surrogate_state_sha256=state_hash(surrogate), proxy_state_sha256=[state_hash(p) for p in proxies],
        training_parent_state_sha256=training_state_hash(anchor),
        training_surrogate_state_sha256=training_state_hash(surrogate),
        clean_reference_state_sha256=state_hash(clean)))
    # The unchanged production preflight checks common task/distribution paths;
    # the second check verifies this arm's actual effective KL path.
    progress(stage='preflight', arm=arm, seed=seed)
    _write_json(output / 'preflight.json', dict(common=no_update_preflight(
        generator, anchor, surrogate, proxies, inputs, seed),
        treatment=loss_preflight(generator, proxies, inputs, arm)))
    train, context, outer, idx = (inputs[k] for k in ('train', 'context', 'outer', 'indices'))
    weights = train['x'].new_ones(context.edge_index.shape[1])
    penalty = FixedProxyAblationPenalty(proxies, train['x'], context.edge_index, weights, mode=arm)
    distribution = new_distribution(generator, train, outer, seed)
    initial_distribution = distribution.snapshot()

    def step(record):
        penalty.assert_contract()
        if penalty.mode != arm or tuple(penalty.coefficients) != COEFFICIENTS[arm]:
            raise ValueError('Clean-KL treatment mutated')
        progress(seed=seed, arm=arm, stage='fit_generator', **record)

    np.random.seed(training_seeds(seed)['continuation'])
    torch.manual_seed(training_seeds(seed)['continuation'])
    result = fit_distribution_residual(generator, surrogate, train['x'], context, train['y'],
        idx['idx_train'], idx['idx_attach'], outer, penalty, ResidualTrainingConfig(), distribution, progress=step)
    if (result['selected_epoch'] != 199 or result['selection_policy'] != 'final_epoch'
            or result['config'] != TRAINING_CONFIG or result['distribution_arm'] != 'unconditional'
            or any(result[k] != 200 for k in ('residual_steps', 'surrogate_steps', 'discriminator_steps',
                                              'clean_constraint_calls', 'distribution_generator_calls'))
            or penalty.calls != 200 or any(m.calls != 200 for m in penalty.members)):
        raise ValueError('Ablation training budget changed')
    penalty.assert_contract()
    result['clean_kl_treatment'] = dict(mode=arm, coefficients=list(COEFFICIENTS[arm]), history=penalty.history)
    # Training digests sort keys; evaluate.state_hash preserves insertion order.
    # Record both conventions explicitly without changing either legacy hash.
    result['evaluation_digest_final_state'] = dict(generator=state_hash(generator),
                                                  surrogate=state_hash(result['surrogate']))
    for name, state in (('generator', generator.state_dict()), ('surrogate', result['surrogate'].state_dict()),
                        ('discriminator', distribution.discriminator.state_dict()),
                        ('distribution_initial', initial_distribution), ('distribution_final', distribution.snapshot())):
        _save_checkpoint(output / (name + '.pt'), state)
    excluded = {'generator', 'surrogate', 'poisoned_features', 'poisoned_labels', 'edge_index',
                'edge_weight', 'idx_train', 'idx_attach', 'fixed_outer'}
    _write_json(output / 'generator_record.json', {k: v for k, v in result.items() if k not in excluded})
    progress(seed=seed, arm=arm, stage='fit_victim_started')
    victim, victim_record = fit_fresh_victim(bundle, inputs, result, output, seed=seed,
        progress=lambda r: progress(seed=seed, arm=arm, **r))
    _save_checkpoint(output / 'victim_auc.pt', victim.state_dict())
    _write_json(output / 'victim_record.json', victim_record)
    freeze(generator)
    trained_surrogate = freeze(result['surrogate'])
    models = dict(surrogate=trained_surrogate, victim=freeze(victim), clean_reference=clean)
    if arm == 'dual':
        retained = dict(generator=restore_generator(bundle, seed), surrogate=restore_surrogate(bundle, seed),
                        victim=restore_victim(bundle, seed))
        actual = dict(generator=generator, surrogate=trained_surrogate, victim=victim)
        replay = {k: state_hash(actual[k]) == state_hash(retained[k]) for k in retained}
        _write_json(output / 'retained_replay.json', dict(seed=seed, tensor_state_match=replay))
        if not all(replay.values()):
            raise ValueError('Dual control did not exactly reproduce retained models; stop before comparisons')
        del retained
    progress(seed=seed, arm=arm, stage='evaluate_dev_started')
    # Identity wrapper is solely an output capture adapter, not pair_mean.
    evaluate_arm(bundle, diagnostic, generator, normalizer, models, 'identity', output / 'development', seed=seed)
    if original_inputs != _input_hashes(diagnostic) or original_states != [state_hash(m) for m in protected]:
        raise ValueError('Shared inputs or source models changed')
    record = dict(status='KL_ABLATION_ARM_COMPLETE', seed=seed, arm=arm, coefficients=list(COEFFICIENTS[arm]),
        residual_steps=200, surrogate_steps=200, discriminator_steps=200, victim_epochs=200,
        generator_selected_epoch=199, victim_selected_epoch=victim_record['trace']['selected_auc_epoch'],
        inputs_unchanged=True, frozen_sources_unchanged=True,
        initial_cpu_rng_sha256=result['initial_cpu_rng_sha256'],
        distribution_initial_state_sha256=result['distribution_initial_state_sha256'],
        victim_initialization_digest=victim_record['initialization_digest'],
        victim_fit_rng_digest=victim_record['fit_rng_digest'])
    _write_json(output / 'completion.json', record)
    progress(seed=seed, arm=arm, stage='arm_complete')
    return record


def difference(candidate, reference):
    if isinstance(reference, dict):
        return {k: difference(candidate[k], v) for k, v in reference.items()}
    if reference is None or candidate is None:
        return None
    if type(reference) in (int, float) and type(candidate) in (int, float):
        return candidate - reference
    return dict(reference=reference, candidate=candidate)


def validate_training_evidence(record, initial, completion, arm):
    """Bind actual per-step effective loss records, not just completion claims."""
    treatment = record['clean_kl_treatment']
    if (record['config'] != TRAINING_CONFIG or record['distribution_arm'] != 'unconditional'
            or record['selection_policy'] != 'final_epoch' or record['selected_epoch'] != 199
            or treatment['mode'] != arm or treatment['coefficients'] != list(COEFFICIENTS[arm])
            or len(treatment['history']) != 200 or len(record['history']) != 200
            or record['initial_state_sha256']['generator'] != initial['training_parent_state_sha256']
            or record['initial_state_sha256']['surrogate'] != initial['training_surrogate_state_sha256']
            or any(record[k] != 200 for k in ('residual_steps', 'surrogate_steps', 'discriminator_steps',
                                              'clean_constraint_calls', 'distribution_generator_calls'))
            or any(record[k] != completion[k] for k in COMMON[:2])):
        raise ValueError('Training trace disagrees with registered treatment/initialization/budget')
    for epoch, (loss, update) in enumerate(zip(treatment['history'], record['history'])):
        members = loss['member_raw_kl']
        if (loss['mode'] != arm or loss['coefficients'] != list(COEFFICIENTS[arm])
                or loss['call'] != epoch + 1 or len(members) != 2 or update['epoch'] != epoch
                or not all(type(v) in (float, int) and math.isfinite(v) for v in
                           [*members, loss['raw_kl'], loss['effective_kl'], update['clean_kl_loss']])):
            raise ValueError('Malformed per-step KL record')
        mean = sum(members) / 2
        expected = mean if arm == 'dual' else members[0] if arm == 'single_proxy1' else 0.
        if (not math.isclose(loss['raw_kl'], mean, rel_tol=1e-6, abs_tol=1e-8)
                or not math.isclose(loss['effective_kl'], expected, rel_tol=1e-6, abs_tol=1e-8)
                or loss['effective_kl'] != loss['weighted_kl']
                or loss['effective_kl'] != update['clean_kl_loss']
                or (arm == 'disabled' and loss['effective_kl'] != 0.)):
            raise ValueError('Effective clean-KL differs from registered coefficients')


def build_comparison(root, seed):
    """Independently recompute metrics from the new saved CSVs; no inference."""
    arms, records, shared, files = {}, {}, None, {}
    for arm in ARMS:
        path = Path(root) / arm
        def read(name):
            file = path / (name + '.json')
            files[str(file.relative_to(root))] = sha256(file)
            return json.loads(file.read_text())
        initial, completion, victim = read('initial_state'), read('completion'), read('victim_record')
        training = read('generator_record')
        validate_training_evidence(training, initial, completion, arm)
        if arm == 'dual':
            replay = read('retained_replay')
            if (replay['seed'] != seed or replay['tensor_state_match'] !=
                    dict(generator=True, surrogate=True, victim=True)):
                raise ValueError('Retained dual replay is absent or failed')
        initial_values, victim_values = _initial(initial, seed), _victim(victim, seed)
        if (initial['arm'] != arm or initial['coefficients'] != list(COEFFICIENTS[arm])
                or completion['seed'] != seed or completion['arm'] != arm
                or completion['coefficients'] != list(COEFFICIENTS[arm])
                or completion['status'] != 'KL_ABLATION_ARM_COMPLETE'
                or any(completion[k] != 200 for k in ('residual_steps', 'surrogate_steps', 'discriminator_steps', 'victim_epochs'))
                or completion['generator_selected_epoch'] != 199
                or not completion['inputs_unchanged'] or not completion['frozen_sources_unchanged']
                or completion['victim_initialization_digest'] != victim_values['initialization_digest']
                or completion['victim_fit_rng_digest'] != victim_values['fit_rng_digest']
                or completion['victim_selected_epoch'] != victim_values['selected_auc_epoch']):
            raise ValueError('Incomplete or mismatched ablation evidence')
        match = dict(initial=initial_values, **{k: completion[k] for k in COMMON})
        if shared is not None and shared != match:
            raise ValueError('Cross-arm starting state, RNG or initialization mismatch')
        shared = match
        value = _load_arm(path / 'development', 'identity', expected_seed=seed)
        if (value['summary']['clean_reference_is_independent'] is not False
                or value['summary']['model_state_sha256']['clean_reference'] != initial_values['clean_reference_state_sha256']
                or value['summary']['dev_counts']['registered_scenes'] != 128
                or any(value['metrics']['victim']['targets'][s]['targets'] != n
                       for s, n in dict(all=112, slow=79, moving=33).items())):
            raise ValueError('Evaluation scope changed')
        if (value['summary']['shared_parent_generator_state_sha256'] != training['evaluation_digest_final_state']['generator']
                or value['summary']['model_state_sha256']['surrogate'] != training['evaluation_digest_final_state']['surrogate']):
            raise ValueError('Evaluation G/S differ from trained final weights')
        for model in MODELS:
            value['metrics'][model]['normal_by_pair_motion'] = _normal_by_pair_motion(value['rows']['normal'], model)
        arms[arm], records[arm] = value, completion
        files[arm + '/development'] = value['assets']
    baseline = arms['dual']
    for value in arms.values():
        if (value['summary']['population_identity_sha256'] != baseline['summary']['population_identity_sha256']
                or value['summary']['dev_counts'] != baseline['summary']['dev_counts']
                or not np.array_equal(value['clean'], baseline['clean'])):
            raise ValueError('Cross-arm evaluation population/features changed')
        for kind in ('targets', 'normal', 'triggered'):
            fields = METADATA + (('target_pair_index', 'label', 'relation') if kind == 'triggered' else ())
            if ([{k: r[k] for k in fields} for r in value['rows'][kind]] !=
                    [{k: r[k] for k in fields} for r in baseline['rows'][kind]]):
                raise ValueError('Cross-arm pair identity/order differs')
        for a, b in zip(value['rows']['normal'], baseline['rows']['normal']):
            if any(a['clean_reference_' + k] != b['clean_reference_' + k] for k in ('p', 'pred')):
                raise ValueError('Common clean-reference predictions differ')
    return dict(status='KL_ABLATION_COMPARISON_COMPLETE_REVIEW_REQUIRED', seed=seed,
        scope='shared already dual-KL-trained domain parent; only subsequent KL treatment differs; reused DEV',
        metrics={a: v['metrics'] for a, v in arms.items()},
        acceptability={a: _acceptability(v['metrics']) for a, v in arms.items()},
        delta_vs_dual={a: difference(arms[a]['metrics'], baseline['metrics']) for a in ARMS[1:]},
        paired_target_responses={a: {m: {s: _paired_targets(baseline['rows']['targets'], arms[a]['rows']['targets'], m, s)
            for s in STRATA} for m in MODELS} for a in ARMS[1:]},
        feature_statistics={a: {s: _copy_stats(v['clean'][mask], v['outputs'][mask]) for s in STRATA
            for mask in [np.asarray([s == 'all' or r['motion_regime'] == s for r in v['rows']['targets']])]} for a, v in arms.items()},
        population_identity_sha256=baseline['summary']['population_identity_sha256'],
        source_files=files, common_start=shared, completions=records,
        statistical_significance_claimed=False, blind_confirmation_claim=False, automatic_promotion=False)


def aggregate(comparisons):
    if [r['seed'] for r in comparisons] != list(SEEDS):
        raise ValueError('Incomplete or reordered fixed seed list')
    if any(r['population_identity_sha256'] != comparisons[0]['population_identity_sha256'] for r in comparisons):
        raise ValueError('Cross-seed target identities changed')
    def selected(metrics):
        v = metrics['victim']
        return dict(overall_incremental_asr=v['targets']['all']['incremental_asr'],
            moving_incremental_asr=v['targets']['moving']['incremental_asr'],
            slow_incremental_asr=v['targets']['slow']['incremental_asr'],
            clean_reference_incremental_asr=metrics['clean_reference']['targets']['all']['incremental_asr'],
            adjacent_incremental_fp=v['collateral']['adjacent_incremental_fp'],
            clean_pair_auc=v['clean_pair_auc'], normal_positive_recall=v['clean_pair_positive_recall'],
            normal_moving_positive_recall=v['normal_by_pair_motion']['moving']['clean_pair_positive_recall'])
    def summarize(rows):
        out = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            defined = all(v is not None for v in values)
            out[key] = dict(per_seed=values, mean=statistics.mean(values) if defined else None,
                            sample_sd=statistics.stdev(values) if defined else None)
        return out
    values = {a: [selected(c['metrics'][a]) for c in comparisons] for a in ARMS}
    return dict(status='KL_ABLATION_COMPLETE_REVIEW_REQUIRED', per_seed=comparisons,
        metrics_mean_sd={a: summarize(v) for a, v in values.items()},
        paired_delta_mean_sd={a: summarize([difference(v, b) for v, b in zip(values[a], values['dual'])]) for a in ARMS[1:]},
        distinct_targets=112, distinct_moving_targets=33, repeated_seeds_do_not_add_independent_targets=True,
        shared_pretrained_start=True, from_scratch_ablation=False, blind_confirmation=False,
        automatic_promotion=False, automatic_followup=False)


def run(bundle, path):
    plan = validate_plan(bundle, path)
    output = Path(path).resolve().parent
    _write_json(output / 'execution_claim.json', dict(started_utc=now(), pid=os.getpid(), plan_sha256=sha256(path)))
    start, ledger, completed, comparisons = time.monotonic(), [], [], []
    torch.set_num_threads(2)
    with (output / 'progress.jsonl').open('x', buffering=1) as stream:
        def progress(**record):
            record.update(utc=now(), elapsed_seconds=time.monotonic() - start)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if record.get('epoch', 0) % 20 == 0 or record.get('epoch') == 199:
                print(json.dumps(record, allow_nan=False), flush=True)
        try:
            progress(stage='read_original_train_dev')
            diagnostic = read_inputs(bundle, plan['protocol']['sources'], ledger)
            normalizer = load_normalizer(bundle)
            _write_json(output / 'input_graph_access_ledger.json', ledger)
            _write_json(output / 'frozen_input_hashes.json', _input_hashes(diagnostic))
            targets = diagnostic['populations']['dev_public_single']
            if len(targets) != 112 or sum(r['motion_regime'] == 'moving' for r in targets) != 33:
                raise ValueError('Registered DEV targets changed')
            _write_json(output / 'populations.json', dict(targets=targets,
                attachments=diagnostic['indices']['idx_attach'].tolist(), outer=diagnostic['outer'].tolist()))
            anchor = restore_generator(bundle, asset_key='start_generator')
            surrogate = restore_surrogate(bundle, asset_key='start_surrogate')
            proxies = [restore_proxy(bundle, index=i) for i in (1, 2)]
            clean = restore_clean_reference(bundle)
            quality = [_dev_quality(p, diagnostic['combined']['dev']) for p in proxies]
            if any(r['gate'] != 'PASS_PROXY_QUALITY' for r in quality):
                raise ValueError('Existing frozen-proxy normal-DEV quality check failed')
            _write_json(output / 'proxy_quality.json', quality)
            for seed in SEEDS:
                child = output / ('seed' + str(seed))
                child.mkdir(exist_ok=False)
                shared = None
                for arm in ARMS:
                    if validate_plan(bundle, path) != plan:
                        raise ValueError('Plan changed during training')
                    record = fit_arm(bundle, diagnostic, normalizer, anchor, surrogate, proxies, clean,
                                     seed, arm, child / arm, progress)
                    match = {k: record[k] for k in COMMON}
                    if shared is not None and shared != match:
                        raise ValueError('Arms did not share training/victim initialization')
                    shared = match
                    completed.append(dict(seed=seed, arm=arm))
                    gc.collect()
                comparison = build_comparison(child, seed)
                _write_json(child / 'comparison.json', comparison)
                comparisons.append(comparison)
                progress(stage='seed_complete', seed=seed, performance_does_not_change_queue=True)
            if validate_plan(bundle, path) != plan:
                raise ValueError('Plan changed during training')
            summary = aggregate(comparisons)
            _write_json(output / 'summary.json', summary)
            _write_json(output / 'completion.json', dict(status=summary['status'], completed_utc=now(),
                elapsed_seconds=time.monotonic() - start, completed=completed, graph_reads=len(ledger),
                summary_sha256=sha256(output / 'summary.json'), original_validation_arrays_loaded=False,
                original_test_arrays_loaded=False, audit_arrays_loaded=False, confirmation_arrays_loaded=False,
                automatic_followup=False, automatic_promotion=False))
            progress(stage='queue_complete')
        except BaseException as error:
            _write_json(output / 'failure.json', dict(status='FAILED_NO_FURTHER_JOBS', completed=completed,
                type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), at=now()))
            raise
        finally:
            _write_json(output / 'graph_access_ledger.json', ledger)


def launch(bundle, path):
    validate_plan(bundle, path)
    path = Path(path).resolve()
    output = path.parent
    if (output / 'execution_claim.json').exists():
        raise ValueError('Plan already executed or claimed')
    _write_json(output / 'launch_claim.json', dict(at=now(), plan_sha256=sha256(path)))
    command = [sys.executable, '-B', '-m', 'spear_current.kl_ablation', '--run', '--plan', str(path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (output / 'terminal.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='KL_ABLATION_LAUNCHED', pid=process.pid, at=now(), command=command,
                  runs=9, log=str(output / 'terminal.log'), automatic_followup=False)
    _write_json(output / 'launch.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    for name in ('check-only', 'prepare', 'run', 'launch'):
        actions.add_argument('--' + name, action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.prepare and (args.output_dir is None or args.plan is not None):
        parser.error('--prepare requires only --output-dir')
    if (args.run or args.launch) and (args.plan is None or args.output_dir is not None):
        parser.error('--run/--launch requires only --plan')
    if not (args.prepare or args.run or args.launch) and (args.output_dir or args.plan):
        parser.error('Check-only accepts no output/plan')
    bundle = load_bundle()
    if args.prepare:
        print('Prepared: ' + str(prepare(bundle, args.output_dir)), flush=True)
    elif args.launch:
        print(json.dumps(launch(bundle, args.plan)), flush=True)
    elif args.run:
        run(bundle, args.plan)
    else:
        protocol(bundle)
        print('KL_ABLATION_PREFLIGHT_OK; no graph arrays, model restoration or training', flush=True)


if __name__ == '__main__':
    main()
