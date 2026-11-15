"""Orthogonal, fresh CUDA ablations of the current identity-output method.

Offline AV2 / researcher-owned GNN robustness experiment. No dataset reader,
checkpoint loader, selector fitting, ASR evaluation, domain projection or
distribution discriminator lives here. The mathematical loss and base-stage
conditioning reuse the tested full-ablation kernel; only registered module
switches and stage names differ. Old experiment modules are not patched.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .models import GCN, GraphTrojanNet
from .support_schema import EXCLUDED_COORDINATES, SELECTED_COORDINATES
from .full_ablation_training import (
    AblationGenerator, _active_hash, _check_context, _cpu_state,
    _finite_gradients, _finite_loss, _finite_module, _freeze, _indices,
    _log_probabilities, _parameter_list_hash, _save_json, _state_hash,
    _tensor_hash,
)


VARIANTS = (
    'full', 'matched_random_pairs',
    'random_semantic_s20260621', 'random_semantic_s20260622',
    'random_semantic_s20260623', 'ranking24', 'no_support28',
    'no_residual', 'self_only', 'base_continuation', 'no_kl', 'single_kl',
)
CONFIG_VARIANTS = VARIANTS
STAGES = ('base', 'residual', 'support_surrogate', 'continuation_a', 'continuation_b')


def variant_policy(variant):
    """Return independent module switches, without changing numerical budgets."""
    if variant not in VARIANTS:
        raise ValueError('Unknown current-method ablation variant')
    return dict(
        variant=variant,
        residual_mode=('base_only' if variant in ('no_residual', 'base_continuation')
                       else 'self_only' if variant == 'self_only' else 'uniform'),
        continue_base=variant == 'base_continuation',
        strict_no_residual=variant == 'no_residual',
        kl_weight=0. if variant == 'no_kl' else 1.,
        kl_members=1 if variant == 'single_kl' else 2,
        final_support_count=28 if variant == 'no_support28' else 24,
        output_mode='identity', discriminator=False,
    )


class CurrentAblationGenerator(AblationGenerator):
    """Reuse tested injection, with explicit current-method phase switches.

    Constructor RNG is isolated from the shared base-training stream. Removing
    the residual therefore does not change the next random draw simply because
    fewer branch tensors were initialized. No residual is allocated in either
    strict deletion or base-continuation arms.
    """

    def __init__(self, base_generator, base_shadow, coordinates, support, *, variant):
        policy = variant_policy(variant)
        self.current_variant = variant
        self.continue_base = policy['continue_base']
        parameter = next(base_generator.parameters())
        device_ids = ([parameter.device.index if parameter.device.index is not None
                       else torch.cuda.current_device()]
                      if parameter.device.type == 'cuda' else [])
        with torch.random.fork_rng(devices=device_ids):
            super().__init__(base_generator, base_shadow, coordinates, support,
                residual_mode=policy['residual_mode'], output_mode='identity', threshold=.5)

    def set_phase(self, phase):
        if phase not in STAGES:
            raise ValueError('Unknown current-method training phase')
        self._base_stage = phase == 'base'
        self._use_residual = phase != 'base' and self.residual is not None
        _freeze(self.base_shadow)
        _freeze(self.base_generator)
        if self.residual is not None:
            _freeze(self.residual)
        if phase == 'base' or (self.continue_base and phase != 'support_surrogate'):
            self.base_generator.requires_grad_(True)
        elif self.residual is not None and phase != 'support_surrogate':
            self.residual.requires_grad_(True)
        return self


def _stage_epochs(config):
    value = config.get('stage_epochs', config.get('epochs', 200))
    if type(value) is int:
        value = {stage: value for stage in STAGES}
    if (not isinstance(value, dict) or set(value) != set(STAGES)
            or any(type(v) is not int or v < 1 for v in value.values())):
        raise ValueError('Require positive integer epoch budgets for all five current stages')
    return dict(value)


def expected_updates(variant, stage_epochs=200, inner_steps=1):
    policy = variant_policy(variant)
    epochs = _stage_epochs(dict(stage_epochs=stage_epochs))
    if type(inner_steps) is not int or inner_steps < 1:
        raise ValueError('inner_steps must be a positive integer')
    return dict(generator=(epochs['base'] if policy['strict_no_residual'] else
                           sum(v for k, v in epochs.items() if k != 'support_surrogate')),
                surrogate=sum(epochs.values()) * inner_steps, discriminator=0)


def fit_case(train, indices, proxies, coordinates, support, *, seed, variant,
             output_dir, config=None, progress=None):
    """Fit the current fixed-support method from unpoisoned random G/shadow.

    All forwards/backwards use CUDA FP32. Supplied frozen clean-only proxies
    may be reused; no inherited attack state is accepted. Support selection is
    fixed by the caller and is not retrained by this experiment.
    """
    policy = variant_policy(variant)
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    config = dict(config or {})
    epochs = _stage_epochs(config)
    inner_steps = config.get('inner_steps', config.get('inner', 1))
    expected = expected_updates(variant, epochs, inner_steps)
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable')
    x, labels, edge = (train[k] for k in ('x', 'y', 'edge_index'))
    if x.device.type != 'cuda':
        raise ValueError('Current ablation fitting requires CUDA; no CPU fallback')
    if (x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 784
            or x.requires_grad or not bool(torch.isfinite(x).all())):
        raise ValueError('Require original finite CUDA FP32 features[N,784]')
    context = train.get('context', SimpleNamespace(edge_index=edge))
    if not torch.equal(_check_context(x, context), edge):
        raise ValueError('Context differs from the original TRAIN graph')
    if labels.shape != (len(x),) or labels.dtype != torch.long or labels.device != x.device:
        raise ValueError('Labels must be same-device long values aligned to TRAIN')
    idx_train, attach, outer = (indices[k] for k in ('idx_train', 'idx_attach', 'fixed_outer'))
    for name, value in (('idx_train', idx_train), ('idx_attach', attach), ('fixed_outer', outer)):
        _indices(value, name, x)
    if (not torch.equal(outer[:len(attach)], attach)
            or bool(torch.isin(idx_train, outer).any())
            or not bool((labels[attach] == 0).all())
            or not bool(((labels[idx_train] == 0) | (labels[idx_train] == 1)).all())):
        raise ValueError('Require disjoint clean supervision and negative attachments-first outer')
    coordinates, support = coordinates.to(x.device), support.to(x.device)
    if tuple(coordinates.tolist()) != SELECTED_COORDINATES:
        raise ValueError('Ranked coordinates30 changed')
    AblationGenerator._validate_support(support, coordinates)
    legal = ~((coordinates == EXCLUDED_COORDINATES[0]) | (coordinates == EXCLUDED_COORDINATES[1]))
    if (int(support.sum()) != policy['final_support_count']
            or (variant == 'no_support28' and not torch.equal(support, legal))):
        raise ValueError('Final support budget or legal28 identity changed')
    if len(proxies) != 2 or len({id(p) for p in proxies}) != 2:
        raise ValueError('Require two distinct clean-only frozen proxies')
    for model in proxies:
        if (any(p.device != x.device or p.requires_grad or p.grad is not None for p in model.parameters())
                or any(child.training for child in model.modules())):
            raise ValueError('Clean proxies must already be frozen/eval on the same CUDA device')
        _finite_module(model, 'clean proxy')
    proxy_hashes = [_state_hash(p) for p in proxies]
    lr = float(config.get('generator_lr', config.get('trojan_lr', .0002)))
    shadow_lr, decay = float(config.get('shadow_lr', .0002)), float(config.get('weight_decay', .0005))
    if (any(not math.isfinite(v) or v <= 0 for v in (lr, shadow_lr))
            or not math.isfinite(decay) or decay < 0
            or float(config.get('target_weight', 1.)) != 1.
            or float(config.get('similarity_weight', .1)) != .1):
        raise ValueError('Invalid learning rates/decay or changed target/similarity weights')
    weights = AblationGenerator._weights(x, context, train.get('edge_weight'))
    tensors = dict(x=x, y=labels, edge_index=edge, edge_weight=weights,
                   idx_train=idx_train, idx_attach=attach, fixed_outer=outer)
    input_hashes = {k: _tensor_hash(v) for k, v in tensors.items()}
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/name).exists() for name in ('initial_state.pt', 'final.pt', 'epochs.jsonl')):
        raise FileExistsError('Refusing to overwrite a current-method training case')
    device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return _fit_seeded(x, labels, context, weights, idx_train, attach, outer,
            proxies, coordinates, support, legal, seed, variant, policy, folder,
            epochs, inner_steps, lr, shadow_lr, decay, input_hashes, proxy_hashes,
            expected, progress)


def _fit_seeded(x, labels, context, weights, idx_train, attach, outer, proxies,
                coordinates, support, legal, seed, variant, policy, folder,
                epochs, inner_steps, lr, shadow_lr, decay, input_hashes,
                proxy_hashes, expected, progress):
    def rng():
        return dict(cpu=_tensor_hash(torch.get_rng_state()),
                    cuda=_tensor_hash(torch.cuda.get_rng_state(x.device)))

    rng_initial = rng()
    shadow = GCN(784, 80, 2, dropout=0., device=x.device).to(x.device)
    base = GraphTrojanNet(x.device, 80, 30).to(x.device)
    fresh_hashes = dict(base_generator=_state_hash(base), base_shadow=_state_hash(shadow))
    generator = CurrentAblationGenerator(base, shadow, coordinates, legal, variant=variant).to(x.device)
    del base, shadow
    generator.set_phase('base')
    surrogate = generator.base_shadow
    selected_proxies = proxies[:policy['kl_members']]
    penalty = FrozenMeanCleanResponsePenalty(selected_proxies, x, context.edge_index,
                                             weights, policy['kl_weight'])
    initial = dict(seed=seed, variant=variant, module_policy=policy,
        fresh_model_sha256=fresh_hashes, initialization_rng_sha256=rng_initial,
        post_initialization_rng_sha256=rng(), residual_constructor_rng_isolated=True,
        stage_epochs=epochs, inner_steps=inner_steps, device=str(x.device), dtype=str(x.dtype),
        selection_policy='final_epoch_each_stage_no_dev_or_asr_selection',
        reused_prefix='caller_supplied_frozen_clean_only_proxies', clean_proxy_sha256=proxy_hashes,
        clean_kl_weight=policy['kl_weight'], clean_kl_members=len(selected_proxies),
        clean_kl_aggregation='arithmetic_mean_of_individual_KL_q_clean_to_p_changed',
        fresh_initialization=True, attack_checkpoint_loaded=False, feature_selector_retrained=False,
        base_conditioning='original_SPEAR_previous_inner_poison_x',
        continuation_conditioning='original_clean_features_every_call',
        final_support=support.tolist(), legal_baseline_support=legal.tolist(),
        input_sha256=input_hashes, expected_optimizer_steps=expected,
        scope='offline_AV2_self_owned_GNN_current_fixed_module_ablation')
    torch.save(dict(generator=_cpu_state(generator), fresh_model_sha256=fresh_hashes,
                    metadata=initial), folder/'initial_state.pt')
    _save_json(folder/'initialization.json', initial)
    records, stage_records = [], []
    inner_indices, outer_indices = torch.cat((idx_train, attach)), torch.cat((idx_train, outer))
    inner_labels = labels.clone(); inner_labels[attach] = 1
    outer_labels = labels.clone(); outer_labels[outer] = 1
    poison_x = x.clone().detach()
    cumulative_g = cumulative_s = 0
    with (folder/'epochs.jsonl').open('x') as log:
        for stage in STAGES:
            if stage == 'residual':
                surrogate = deepcopy(generator.base_shadow)
            generator.set_phase(stage)
            generator.set_support(legal if stage in ('base', 'residual') else support)
            generator.set_output_mode('identity')
            if stage == 'base':
                surrogate = generator.base_shadow
            surrogate.requires_grad_(True)
            s_optimizer = torch.optim.Adam(surrogate.parameters(), lr=shadow_lr, weight_decay=decay)
            surrogate_ids = {id(p) for p in surrogate.parameters()}
            active = [p for p in generator.parameters() if p.requires_grad and id(p) not in surrogate_ids]
            g_optimizer = torch.optim.Adam(active, lr=lr, weight_decay=decay) if active else None
            before = dict(generator=_state_hash(generator), surrogate=_state_hash(surrogate),
                          active_generator=_parameter_list_hash(generator, active),
                          active_surrogate=_active_hash(surrogate))
            frozen_base = (_state_hash(generator.base_generator)
                           if stage != 'base' and not policy['continue_base'] else None)
            frozen_shadow = _state_hash(generator.base_shadow) if stage != 'base' else None
            stage_g = stage_s = 0
            for epoch in range(epochs[stage]):
                try:
                    generator.train(); generator.zero_grad(set_to_none=True)
                    surrogate.requires_grad_(True).train()
                    inner_losses = []
                    for _ in range(inner_steps):
                        s_optimizer.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            if stage == 'base':
                                embedding = surrogate.get_h(poison_x, context.edge_index)
                                values = generator.generate_with_embedding(x, context, attach, embedding)
                                poison_x = x.clone().detach()
                                poison_x[attach[:, None], coordinates] = values.detach()
                                inner_x = poison_x
                            else:
                                inner_x = generator.inject(x, context, attach, weights).features
                        output = _log_probabilities(surrogate, inner_x, context.edge_index, weights)
                        inner_loss = F.nll_loss(output[inner_indices], inner_labels[inner_indices])
                        _finite_loss(inner_loss, 'inner'); inner_loss.backward()
                        inner_count, inner_norm = _finite_gradients(surrogate.parameters(), 'surrogate')
                        s_optimizer.step(); _finite_module(surrogate, 'surrogate')
                        stage_s += 1; cumulative_s += 1
                        inner_losses.append(float(inner_loss.detach()))
                    s_optimizer.zero_grad(set_to_none=True)
                    surrogate.requires_grad_(False).eval(); generator.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(g_optimizer is not None):
                        if stage == 'base':
                            with torch.no_grad():
                                embedding = surrogate.get_h(poison_x, context.edge_index)
                            values = generator.generate_with_embedding(x, context, outer, embedding)
                            changed = x.clone(); changed[outer[:, None], coordinates] = values
                        else:
                            changed = generator.inject(x, context, outer, weights, detach=False).features
                        output = _log_probabilities(surrogate, changed, context.edge_index, weights)
                        nll = F.nll_loss(output[outer_indices], outer_labels[outer_indices])
                        similarity = 1. - F.cosine_similarity(changed, x, dim=1).mean()
                        clean_kl = penalty(changed, context.edge_index, weights, outer)
                        # Preserve the tested identity-route autograd graph:
                        # its absent discriminator contributes this exact zero.
                        # No discriminator, weight or distribution objective is
                        # introduced by retaining the zero-valued branch.
                        loss = nll + .1 * similarity + clean_kl + changed.sum() * 0.
                        _finite_loss(loss, 'outer')
                        outer_count = 0; outer_norm = 0.
                        if g_optimizer is not None:
                            g_optimizer.zero_grad(set_to_none=True); loss.backward()
                            outer_count, outer_norm = _finite_gradients(active, 'generator')
                            g_optimizer.step(); _finite_module(generator, 'generator')
                            stage_g += 1; cumulative_g += 1
                    surrogate.requires_grad_(True)
                    if frozen_base is not None and _state_hash(generator.base_generator) != frozen_base:
                        raise RuntimeError('Frozen base generator changed')
                    if frozen_shadow is not None and _state_hash(generator.base_shadow) != frozen_shadow:
                        raise RuntimeError('Frozen conditioning shadow changed')
                    record = dict(seed=seed, variant=variant, stage=stage, epoch=epoch,
                        device=str(x.device), support_count=int(generator.hard_support.sum()),
                        output_mode='identity', active_generator=('none' if g_optimizer is None else
                            'base_generator' if stage == 'base' or policy['continue_base'] else 'residual'),
                        inner_nll=inner_losses, outer_nll=float(nll.detach()),
                        similarity_loss=float((.1*similarity).detach()), clean_kl_loss=float(clean_kl.detach()),
                        clean_kl_weight=policy['kl_weight'], clean_kl_members=len(selected_proxies),
                        clean_kl_member_raw=penalty.history[-1]['member_raw_kl'],
                        distribution_loss=0., distribution_weight=0., outer_loss=float(loss.detach()),
                        inner_gradient_parameters=inner_count, inner_gradient_l2=inner_norm,
                        outer_gradient_parameters=outer_count, outer_gradient_l2=outer_norm,
                        stage_generator_steps=stage_g, stage_surrogate_steps=stage_s,
                        cumulative_generator_steps=cumulative_g, cumulative_surrogate_steps=cumulative_s,
                        d_loss=0., d_real_loss=0., d_fake_loss=0., d_steps=0)
                    records.append(record); log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                    if progress is not None:
                        progress(deepcopy(record))
                except Exception as error:
                    _save_json(folder/'failure.json', dict(status='FAILED_NO_RETRY', seed=seed,
                        variant=variant, stage=stage, epoch=epoch, error=repr(error)))
                    raise RuntimeError(f'{variant} seed={seed} stage={stage} epoch={epoch} failed: {error}') from error
            generator.zero_grad(set_to_none=True); surrogate.zero_grad(set_to_none=True)
            after = dict(generator=_state_hash(generator), surrogate=_state_hash(surrogate),
                         active_generator=_parameter_list_hash(generator, active),
                         active_surrogate=_active_hash(surrogate))
            summary = dict(stage=stage, epochs=epochs[stage], generator_steps=stage_g,
                surrogate_steps=stage_s, generator_update_opportunities=epochs[stage] if g_optimizer else 0,
                generator_frozen=g_optimizer is None, surrogate_inner_steps=inner_steps,
                active_generator_parameters=before['active_generator']['names'],
                initial_state_sha256=before, final_state_sha256=after,
                frozen_base_generator_sha256=frozen_base, frozen_conditioning_shadow_sha256=frozen_shadow,
                support_count=int(generator.hard_support.sum()), output_mode='identity',
                discriminator_steps=0, distribution_weight=0., discriminator_initial_sha256=None,
                discriminator_final_sha256=None, selection_policy='final_epoch', selected_epoch=epochs[stage]-1)
            stage_records.append(summary)
            checkpoint = dict(generator=_cpu_state(generator), surrogate=_cpu_state(surrogate),
                stage=stage, epoch=epochs[stage]-1, seed=seed, variant=variant,
                architecture=dict(input_dim=784, hidden_dim=80, output_dim=30,
                    residual_mode=policy['residual_mode'], variant=variant, threshold=.5),
                stage_record=summary, initialization=initial, distribution=None)
            torch.save(checkpoint, folder/f'{stage}_final.pt')
            _save_json(folder/'stages.json', stage_records)
    _freeze(generator); _freeze(surrogate)
    if [_state_hash(p) for p in proxies] != proxy_hashes:
        raise RuntimeError('Frozen clean proxy state changed')
    if any(p.grad is not None or p.requires_grad for model in proxies for p in model.parameters()):
        raise RuntimeError('Clean proxies acquired gradients')
    final_inputs = {k:_tensor_hash(v) for k,v in dict(x=x,y=labels,edge_index=context.edge_index,
        edge_weight=weights,idx_train=idx_train,idx_attach=attach,fixed_outer=outer).items()}
    if final_inputs != input_hashes:
        raise RuntimeError('Original caller inputs changed')
    updates = dict(generator=cumulative_g, surrogate=cumulative_s, discriminator=0)
    if updates != expected:
        raise RuntimeError('Actual optimizer updates differ from the module policy')
    checkpoint['stages'] = stage_records
    torch.save(checkpoint, folder/'final.pt')
    result = dict(status='CURRENT_ABLATION_TRAINING_COMPLETE', generator=generator, surrogate=surrogate,
        records=records, history=records, stages=stage_records, checkpoint=str(folder/'final.pt'),
        final_checkpoint=str(folder/'final.pt'), initialization=initial,
        generator_steps=cumulative_g, surrogate_steps=cumulative_s, discriminator_steps=0,
        optimizer_steps=updates, selection_policy='final_epoch', original_inputs_unchanged=True,
        frozen_clean_proxies_unchanged=True, feature_selector_retrained=False,
        output_mapping='identity', residual_present=generator.residual is not None)
    _save_json(folder/'completion.json', {k:v for k,v in result.items()
        if k not in ('generator','surrogate','records','history')})
    return result
