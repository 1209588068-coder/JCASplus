# Source: experiments/spear_distribution_training_v1/training.py; mechanically migrated; algorithms and state keys unchanged.
"""Isolated three-arm distribution continuation on registered original TRAIN tensors.

The old content continuation loop is copied here so its source remains frozen.
A discriminator update, when enabled, runs between the surrogate and residual
updates. Only the inherited projected residual and owned surrogate are updated
by this loop. No file, dataset, checkpoint, victim, or evaluation access occurs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .attention_context import AttentionContext
from .role_generator import _check_features
from .domain_generator import DomainConstrainedGenerator
from .distribution import DistributionRegularizer
from .role_training import (
    ResidualTrainingConfig, _check_constraint, _finite_gradients, _finite_module,
    _indices, _log_probabilities, _state_hash, _tensor_hash,
)


ARMS = ('continuation', 'unconditional', 'conditional')


def _rng_snapshot():
    numpy_state = np.random.get_state()
    return (torch.random.get_rng_state().clone(), random.getstate(),
            (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]))


def _assert_rng_unchanged(before):
    after = _rng_snapshot()
    if (not torch.equal(before[0], after[0]) or before[1] != after[1]
            or before[2][0] != after[2][0] or not np.array_equal(before[2][1], after[2][1])
            or before[2][2:] != after[2][2:]):
        raise ValueError('Distribution update changed the shared training RNG')


def _storage_ids(module):
    return {(str(value.device), value.untyped_storage().data_ptr())
            for value in (*module.parameters(), *module.buffers()) if value.numel()}


def _scalar(value, features, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 0
            or value.device != features.device or value.dtype != features.dtype
            or not bool(torch.isfinite(value))):
        raise ValueError('Nonfinite or malformed ' + name)
    return value


def _assert_scope(clean, injected, targets, generator, context, weights):
    if (injected.features.shape != clean.shape or not bool(torch.isfinite(injected.features).all())
            or not torch.equal(injected.edge_index, context.edge_index)
            or not torch.equal(injected.edge_weight, weights)):
        raise ValueError('Distribution injection changed feature shape, topology or edge weights')
    selected = generator.selected_coordinates[generator.hard_support]
    allowed = torch.zeros_like(clean, dtype=torch.bool)
    allowed[targets[:, None], selected] = True
    if (not torch.equal(injected.features[~allowed], clean[~allowed])
            or bool((injected.features[allowed].abs() > 10).any())):
        raise ValueError('Distribution injection changed a nontarget/support field or left the domain')


def fit_distribution_residual(generator, surrogate, features, context, labels, idx_train,
                 idx_attach, fixed_outer, clean_constraint, config, distribution, progress=None):
    """Continue fixed projected SAGE24 with one registered distribution treatment.

    This is the frozen role-training loop with an explicit attention-only guard:
    the caller's score permutation and its owned context clone must equal the
    entry snapshot before every optimizer update and around progress callbacks.
    No hooks or process-wide runtime patches are installed. The return schema
    retains every old field, with additional distribution counters and loss records.

    The caller must independently establish TRAIN provenance, attachment
    eligibility and the fixed nonattachment outer pool: this tensor-level API
    has no split identity and never invents one. It does not run unless called.
    Context labels may be -1 (not computable); supervised training rows require
    0/1 and attachments require 0. Extra outer labels, including -1, are replaced
    only in temporary outer supervision, not in returned poisoned labels.

    ``generator`` is updated in place; ``surrogate`` is deep-copied and the copy
    is returned. Input features/labels/indices/context and all frozen modules
    are checked unchanged. The supplied clean constraint may append its loss
    history, but its weights, cached responses and proxy states stay frozen.

    ``progress(record)`` receives a detached copy after each completed epoch.
    Returned poisoned data use only idx_attach, with final-epoch residual
    weights. There is no validation, ASR calculation, early stopping, or best
    loss reload. Initial state hashes and the starting CPU RNG hash support
    equal-start controls; this API deliberately does not reset global RNG.
    """
    if (not isinstance(generator, DomainConstrainedGenerator)
            or generator.output_mode != 'domain_projection'
            or not isinstance(surrogate, nn.Module)):
        raise ValueError('Expected a projected DomainConstrainedGenerator and independent surrogate')
    generator.assert_contract()
    if not isinstance(distribution, DistributionRegularizer) or distribution.arm not in ARMS:
        raise ValueError('Expected a registered DistributionRegularizer')
    distribution.assert_contract()
    if distribution.steps != 0 or distribution.generator_calls != 0:
        raise ValueError('Distribution regularizer must start with zero update/loss counts')
    distribution.validate_binding(features, fixed_outer,
                                  generator.selected_coordinates[generator.hard_support])
    distribution_arm = distribution.arm
    if not isinstance(context, AttentionContext):
        raise ValueError('Expected an AttentionContext with a fixed score permutation')
    if (generator.attention_mode not in ('uniform', 'learned', 'shuffled')
            or generator.attention_mode != generator.residual.attention_mode
            or generator.score_dim != generator.residual.score_dim
            or generator.residual.mode != 'untyped'):
        raise ValueError('Inconsistent content attention configuration')
    def attention_configuration():
        scorer = generator.residual.scorer
        return (generator.attention_mode, generator.residual.attention_mode,
                generator.score_dim, generator.residual.score_dim, generator.residual.mode,
                generator.input_dim, generator.threshold, scorer.heads,
                scorer.negative_slope, scorer.dropout, scorer.add_self_loops,
                scorer.share_weights, scorer.edge_dim)
    expected_configuration = attention_configuration()
    context.validate()
    expected_permutation = context.score_permutation.detach().clone()
    permutation_contexts = [(context, context.score_permutation,
                             context.score_permutation._version)]

    def assert_attention_frozen():
        generator.assert_contract()
        if attention_configuration() != expected_configuration:
            raise ValueError('Immutable attention configuration was mutated')
        # Tensor version counters catch ordinary in-place changes, including a
        # change restored to its initial value. Content checks also cover .data
        # writes; identity checks reject replacement of the frozen field.
        for current, permutation, version in permutation_contexts:
            if (current.score_permutation is not permutation
                    or permutation._version != version
                    or not torch.equal(permutation, expected_permutation)):
                raise ValueError('Immutable attention score_permutation was mutated')
    if not isinstance(config, ResidualTrainingConfig):
        raise ValueError('Expected ResidualTrainingConfig')
    config.__post_init__()
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable or None')
    _check_features(features, context, generator.input_dim)
    if features.requires_grad:
        raise ValueError('Original training features must not require gradients')
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
            or labels.shape != (len(features),) or labels.device != features.device
            or not bool(((labels == -1) | (labels == 0) | (labels == 1)).all())):
        raise ValueError('Labels must be a same-device torch.long vector in {-1,0,1}')
    for name, value in (('idx_train', idx_train), ('idx_attach', idx_attach), ('fixed_outer', fixed_outer)):
        _indices(value, name, features)
    if not bool(((labels[idx_train] == 0) | (labels[idx_train] == 1)).all()):
        raise ValueError('Supervised training indices must have computable binary labels')
    if not bool((labels[idx_attach] == 0).all()):
        raise ValueError('Attachment indices must have original true-negative labels')
    if (len(fixed_outer) < len(idx_attach)
            or not torch.equal(fixed_outer[:len(idx_attach)], idx_attach)):
        raise ValueError('fixed_outer must start with the unchanged attachment order')
    if bool(torch.isin(idx_train, fixed_outer).any()):
        raise ValueError('Clean training indices and the complete outer pool must be disjoint')
    generator._validate_targets(features, fixed_outer)
    if any(not parameter.requires_grad for parameter in generator.residual.parameters()):
        raise ValueError('All residual parameters must be trainable')
    for base in (generator.base_generator, generator.base_shadow):
        if any(parameter.requires_grad or parameter.grad is not None for parameter in base.parameters()):
            raise ValueError('Baseline generator and conditioning shadow must remain frozen')
        if any(child.training for child in base.modules()):
            raise ValueError('Baseline generator and conditioning shadow must be in eval mode')
    weights = features.new_ones(context.edge_index.shape[1])
    _check_constraint(clean_constraint, features, context, weights)
    for model, name in ((generator, 'generator'), (surrogate, 'surrogate')):
        _finite_module(model, name)

    # Ownership is explicit, and no poisoned feature tensor becomes a later
    # conditioning input. Both phases use these same original x/context objects.
    original_x = features.detach().clone()
    original_context = context.to(features.device)
    if not isinstance(original_context, AttentionContext):
        raise ValueError('Owned context must remain an AttentionContext')
    original_context.validate()
    permutation_contexts.append((original_context, original_context.score_permutation,
                                 original_context.score_permutation._version))
    assert_attention_frozen()
    train = idx_train.detach().clone()
    attach = idx_attach.detach().clone()
    outer = fixed_outer.detach().clone()
    original_labels = labels.detach().clone()
    inner_labels = original_labels.clone(); inner_labels[attach] = config.target_class
    outer_labels = original_labels.clone(); outer_labels[outer] = config.target_class
    inner_indices, outer_indices = torch.cat((train, attach)), torch.cat((train, outer))
    owned_surrogate = deepcopy(surrogate)
    owned_surrogate.requires_grad_(True)
    for parameter in owned_surrogate.parameters():
        parameter.grad = None
    if not tuple(owned_surrogate.parameters()):
        raise ValueError('Surrogate must have trainable parameters')
    owned_surrogate.eval()
    with torch.no_grad():
        _log_probabilities(owned_surrogate, original_x, original_context, weights)

    immutable_tensors = dict(features=features, labels=labels, idx_train=idx_train,
        idx_attach=idx_attach, fixed_outer=fixed_outer, edges=context.edge_index,
        roles=context.edge_type, scenes=context.pair_scene,
        score_permutation=context.score_permutation,
        conditioned_score_permutation=original_context.score_permutation,
        conditioned_features=original_x, conditioned_edges=original_context.edge_index,
        conditioned_roles=original_context.edge_type, conditioned_scenes=original_context.pair_scene,
        conditioned_train=train, conditioned_attach=attach, conditioned_outer=outer,
        conditioned_labels=original_labels, inner_labels=inner_labels, outer_labels=outer_labels,
        inner_indices=inner_indices, outer_indices=outer_indices, edge_weight=weights,
        selected_coordinates=generator.selected_coordinates)
    for index, member in enumerate(clean_constraint.members):
        immutable_tensors.update({f'proxy{index}_q': member.q, f'proxy{index}_edges': member.edges,
                                 f'proxy{index}_weights': member.edge_weights})
    input_hashes = {key: _tensor_hash(value) for key, value in immutable_tensors.items()}
    versions = {key: value._version for key, value in immutable_tensors.items()}
    frozen_models = dict(base_generator=generator.base_generator, base_shadow=generator.base_shadow,
        proxy0=clean_constraint.members[0].model, proxy1=clean_constraint.members[1].model)
    frozen_hashes = {key: _state_hash(model) for key, model in frozen_models.items()}
    caller_surrogate_hash = _state_hash(surrogate)
    initial_state = dict(generator=_state_hash(generator), surrogate=_state_hash(owned_surrogate))
    initial_rng = _tensor_hash(torch.random.get_rng_state())
    discriminator = distribution.discriminator
    if not isinstance(discriminator, nn.Module):
        raise ValueError('Distribution discriminator must be an independently owned module')
    protected = (generator, surrogate, owned_surrogate, *frozen_models.values())
    if any(_storage_ids(discriminator) & _storage_ids(model) for model in protected):
        raise ValueError('Discriminator shares parameters or buffers with protected models')
    distribution_initial_state = _state_hash(discriminator)
    constraint_calls_before = clean_constraint.calls
    constraint_identity = (clean_constraint.weight, tuple(
        (id(member), id(member.model), member.weight) for member in clean_constraint.members))
    surrogate_steps = residual_steps = discriminator_steps = generator_loss_calls = 0

    def assert_frozen():
        assert_attention_frozen()
        distribution.assert_contract()
        if distribution.arm != distribution_arm or distribution.discriminator is not discriminator:
            raise ValueError('Distribution treatment or discriminator ownership was changed')
        if (distribution.steps != discriminator_steps
                or distribution.generator_calls != generator_loss_calls):
            raise ValueError('Distribution counters changed outside their registered updates')
        if (any(p.requires_grad or p.grad is not None for p in discriminator.parameters())
                or any(child.training for child in discriminator.modules())):
            raise ValueError('Discriminator must be frozen/eval/gradient-free between updates')
        identity = (clean_constraint.weight, tuple(
            (id(member), id(member.model), member.weight) for member in clean_constraint.members))
        if identity != constraint_identity:
            raise ValueError('Clean proxy membership or constraint weighting was changed')
        if any(value._version != versions[key] for key, value in immutable_tensors.items()):
            raise ValueError('Immutable original input, context, or clean response was mutated')
        for model in frozen_models.values():
            if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise ValueError('Gradient escaped into a frozen baseline or clean proxy')
            if any(child.training for child in model.modules()):
                raise ValueError('Frozen baseline or clean proxy left eval mode')

    assert_frozen()
    optimizer_surrogate = torch.optim.Adam(owned_surrogate.parameters(), lr=config.shadow_lr,
                                           weight_decay=config.weight_decay)
    optimizer_residual = torch.optim.Adam(generator.residual.parameters(), lr=config.residual_lr,
                                          weight_decay=config.weight_decay)
    history = []
    for epoch in range(config.epochs):
        assert_frozen()
        generator.train()
        owned_surrogate.requires_grad_(True)
        owned_surrogate.train()
        inner_losses = []
        generator.zero_grad(set_to_none=True)
        for _ in range(config.inner_steps):
            optimizer_surrogate.zero_grad(set_to_none=True)
            with torch.no_grad():
                injected = generator.inject(original_x, original_context, attach, weights, detach=True)
            _assert_scope(original_x, injected, attach, generator, original_context, weights)
            output = _log_probabilities(owned_surrogate, injected.features, original_context, weights)
            inner_loss = F.nll_loss(output[inner_indices], inner_labels[inner_indices])
            if not bool(torch.isfinite(inner_loss)):
                raise ValueError('Nonfinite inner loss')
            inner_loss.backward()
            inner_gradient_count = _finite_gradients(owned_surrogate.parameters(), 'surrogate')
            if any(parameter.grad is not None for parameter in generator.parameters()):
                raise ValueError('Inner step produced a generator gradient')
            assert_frozen()
            optimizer_surrogate.step()
            surrogate_steps += 1
            _finite_module(owned_surrogate, 'surrogate')
            inner_losses.append(float(inner_loss.detach()))

        optimizer_surrogate.zero_grad(set_to_none=True)
        if distribution_arm != 'continuation':
            # Generate fresh fixed-outer fake rows without a generator graph.
            # The regularizer owns clean targets, clean conditions and D's Adam.
            d_rng = _rng_snapshot()
            protected_state = (_state_hash(generator), _state_hash(owned_surrogate))
            with torch.no_grad():
                d_injected = generator.inject(original_x, original_context, outer, weights, detach=True)
            _assert_scope(original_x, d_injected, outer, generator, original_context, weights)
            d_record = distribution.discriminator_step(d_injected.features.detach())
            _assert_rng_unchanged(d_rng)
            if protected_state != (_state_hash(generator), _state_hash(owned_surrogate)):
                raise ValueError('Discriminator step changed generator or surrogate parameters')
            if any(p.grad is not None for model in (generator, owned_surrogate) for p in model.parameters()):
                raise ValueError('Discriminator step produced generator or surrogate gradients')
            discriminator_steps += 1
            if (not isinstance(d_record, dict) or d_record.get('d_steps') != discriminator_steps
                    or distribution.steps != discriminator_steps):
                raise ValueError('Unexpected discriminator update count')
            for name in ('d_loss', 'd_real_loss', 'd_fake_loss'):
                value = d_record.get(name)
                if (isinstance(value, bool) or not isinstance(value, (float, int))
                        or not np.isfinite(value)):
                    raise ValueError('Nonfinite or malformed discriminator loss record')
            assert_frozen()
        else:
            d_record = dict(d_loss=0., d_real_loss=0., d_fake_loss=0., d_steps=0)
            if distribution.steps != 0:
                raise ValueError('Continuation control performed a discriminator update')
        owned_surrogate.requires_grad_(False)
        owned_surrogate.eval()
        optimizer_residual.zero_grad(set_to_none=True)
        try:
            injected = generator.inject(original_x, original_context, outer, weights, detach=False)
            _assert_scope(original_x, injected, outer, generator, original_context, weights)
            output = _log_probabilities(owned_surrogate, injected.features, original_context, weights)
            raw_nll = F.nll_loss(output[outer_indices], outer_labels[outer_indices])
            similarity = 1 - F.cosine_similarity(injected.features, original_x, dim=1).mean()
            clean_loss = clean_constraint(injected.features, original_context.edge_index, weights, outer)
            if (not isinstance(clean_loss, torch.Tensor) or clean_loss.ndim != 0
                    or clean_loss.device != features.device or not bool(torch.isfinite(clean_loss))):
                raise ValueError('Nonfinite or malformed clean constraint loss')
            target_loss = config.target_weight * raw_nll
            similarity_loss = config.similarity_weight * similarity
            distribution_rng = _rng_snapshot()
            distribution_loss = _scalar(distribution.generator_loss(injected.features),
                                        features, 'distribution generator loss')
            generator_loss_calls += 1
            _assert_rng_unchanged(distribution_rng)
            distribution_raw = distribution.last_generator_raw_loss
            if isinstance(distribution_raw, torch.Tensor):
                distribution_raw = float(_scalar(distribution_raw, features, 'raw distribution loss').detach())
            if (isinstance(distribution_raw, bool) or not isinstance(distribution_raw, (float, int))
                    or not np.isfinite(distribution_raw)):
                raise ValueError('Nonfinite or malformed raw distribution loss record')
            if distribution.generator_calls != epoch + 1 or distribution.steps != discriminator_steps:
                raise ValueError('Unexpected distribution generator/update count')
            if distribution_arm == 'continuation' and (float(distribution_loss.detach()) != 0. or distribution_raw != 0.):
                raise ValueError('Continuation control distribution loss must be exactly zero')
            outer_loss = target_loss + similarity_loss + clean_loss + distribution_loss
            if not bool(torch.isfinite(outer_loss)):
                raise ValueError('Nonfinite outer loss')
            outer_loss.backward()
            outer_gradient_count = _finite_gradients(generator.residual.parameters(), 'residual')
            if any(parameter.grad is not None for parameter in owned_surrogate.parameters()):
                raise ValueError('Outer step produced a surrogate parameter gradient')
            assert_frozen()
            optimizer_residual.step()
            residual_steps += 1
            _finite_module(generator, 'generator')
        finally:
            owned_surrogate.requires_grad_(True)
        record = dict(epoch=epoch, inner_nll=inner_losses, outer_nll=float(raw_nll.detach()),
            target_loss=float(target_loss.detach()), global_cosine_loss=float(similarity.detach()),
            similarity_loss=float(similarity_loss.detach()), clean_kl_loss=float(clean_loss.detach()),
            outer_loss=float(outer_loss.detach()), inner_gradient_parameters=inner_gradient_count,
            outer_gradient_parameters=outer_gradient_count,
            surrogate_steps=surrogate_steps, residual_steps=residual_steps,
            d_loss=d_record['d_loss'], d_real_loss=d_record['d_real_loss'],
            d_fake_loss=d_record['d_fake_loss'], d_steps=discriminator_steps,
            distribution_raw_g_loss=float(distribution_raw),
            distribution_weighted_g_loss=float(distribution_loss.detach()))
        history.append(record)
        if progress is not None:
            assert_frozen()
            try:
                progress(deepcopy(record))
            finally:
                assert_frozen()
        assert_frozen()

    generator.eval()
    owned_surrogate.eval()
    with torch.no_grad():
        poisoned = generator.inject(original_x, original_context, attach, weights, detach=True)
    _assert_scope(original_x, poisoned, attach, generator, original_context, weights)
    assert_frozen()
    for key, value in immutable_tensors.items():
        if _tensor_hash(value) != input_hashes[key]:
            raise ValueError(f'Immutable tensor changed: {key}')
    for key, model in frozen_models.items():
        if _state_hash(model) != frozen_hashes[key]:
            raise ValueError(f'Frozen model state changed: {key}')
    if _state_hash(surrogate) != caller_surrogate_hash:
        raise ValueError('Caller surrogate state changed')
    if clean_constraint.calls - constraint_calls_before != config.epochs:
        raise ValueError('Unexpected dual-proxy outer loss call count')
    expected_d_steps = 0 if distribution_arm == 'continuation' else config.epochs
    if (discriminator_steps != expected_d_steps or distribution.steps != expected_d_steps
            or distribution.generator_calls != config.epochs):
        raise ValueError('Distribution update/loss budget differs from registered arm')
    return dict(status='CONTINUATION_COMPLETE', generator=generator, surrogate=owned_surrogate,
        history=history, config=asdict(config), selection_policy='final_epoch', selected_epoch=config.epochs - 1,
        surrogate_steps=surrogate_steps, residual_steps=residual_steps,
        clean_constraint_calls=config.epochs, conditioning_policy='original_normalized_features_every_phase',
        surrogate_mode_policy='inner_train_outer_eval_with_parameter_freeze',
        poisoned_features=poisoned.features, poisoned_labels=inner_labels,
        edge_index=poisoned.edge_index, edge_weight=poisoned.edge_weight,
        idx_train=train, idx_attach=attach, fixed_outer=outer,
        initial_state_sha256=initial_state,
        final_state_sha256=dict(generator=_state_hash(generator), surrogate=_state_hash(owned_surrogate)),
        frozen_state_sha256=frozen_hashes, input_sha256=input_hashes, initial_cpu_rng_sha256=initial_rng,
        training_split_provenance='caller_must_validate_train_only',
        original_inputs_unchanged=True, frozen_models_unchanged=True, caller_surrogate_unchanged=True,
        distribution_arm=distribution_arm, discriminator_steps=discriminator_steps,
        distribution_generator_calls=distribution.generator_calls,
        distribution_initial_state_sha256=distribution_initial_state,
        distribution_final_state_sha256=_state_hash(discriminator))
