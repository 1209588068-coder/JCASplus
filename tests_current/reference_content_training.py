# Source: experiments/spear_content_continuation_v1/training.py; frozen synthetic-test parity oracle only.
"""Explicit, isolated content-attention continuation for offline robustness research.

Only caller-registered original TRAIN tensors are accepted by this tensor API;
provenance remains the caller's responsibility. There is no data loading, split
selection, checkpoint loading, evaluation, or implicit job execution here.
The loop preserves the frozen pair-role continuation contract and reuses its
validation/hash helpers, with score-permutation checks added at each boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from spear_current.attention_context import AttentionContext
from spear_current.attention_generator import ContentAttentionGenerator
from spear_current.role_generator import _check_features
from spear_current.role_training import (
    ResidualTrainingConfig, _check_constraint, _finite_gradients, _finite_module,
    _indices, _log_probabilities, _state_hash, _tensor_hash,
)


def fit_content_residual(generator, surrogate, features, context, labels, idx_train,
                 idx_attach, fixed_outer, clean_constraint, config, progress=None):
    """Continue content attention on a fixed, caller-registered TRAIN graph.

    This is the frozen role-training loop with an explicit attention-only guard:
    the caller's score permutation and its owned context clone must equal the
    entry snapshot before every optimizer update and around progress callbacks.
    No hooks or process-wide runtime patches are installed. The return schema
    is unchanged; input_sha256 additionally records both score permutations.

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
    if (not isinstance(generator, ContentAttentionGenerator)
            or not isinstance(surrogate, nn.Module)):
        raise ValueError('Expected a ContentAttentionGenerator and independent surrogate module')
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
    constraint_calls_before = clean_constraint.calls
    constraint_identity = (clean_constraint.weight, tuple(
        (id(member), id(member.model), member.weight) for member in clean_constraint.members))

    def assert_frozen():
        assert_attention_frozen()
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
    surrogate_steps = residual_steps = 0
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
        owned_surrogate.requires_grad_(False)
        owned_surrogate.eval()
        optimizer_residual.zero_grad(set_to_none=True)
        try:
            injected = generator.inject(original_x, original_context, outer, weights, detach=False)
            output = _log_probabilities(owned_surrogate, injected.features, original_context, weights)
            raw_nll = F.nll_loss(output[outer_indices], outer_labels[outer_indices])
            similarity = 1 - F.cosine_similarity(injected.features, original_x, dim=1).mean()
            clean_loss = clean_constraint(injected.features, original_context.edge_index, weights, outer)
            if (not isinstance(clean_loss, torch.Tensor) or clean_loss.ndim != 0
                    or clean_loss.device != features.device or not bool(torch.isfinite(clean_loss))):
                raise ValueError('Nonfinite or malformed clean constraint loss')
            target_loss = config.target_weight * raw_nll
            similarity_loss = config.similarity_weight * similarity
            outer_loss = target_loss + similarity_loss + clean_loss
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
            surrogate_steps=surrogate_steps, residual_steps=residual_steps)
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
        original_inputs_unchanged=True, frozen_models_unchanged=True, caller_surrogate_unchanged=True)
