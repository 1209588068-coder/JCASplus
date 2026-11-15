"""Strict single-module deletions after a shared, attack-trained base prefix.

Only the completed Full base stage may be restored: the residual head is still
zero, final support has not been applied, and projection/D have not been used.
This is shared-prefix continuation, not a fresh attack initialization. Actual
fitting requires CUDA. The caller owns dataset access and fresh-victim fitting.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .full_ablation_training import (
    AblationGenerator, _check_context, _cpu_state, _finite_gradients,
    _finite_loss, _finite_module, _freeze, _indices, _log_probabilities,
    _parameter_list_hash, _state_hash, _tensor_hash,
)
from .models import GCN, GraphTrojanNet
from .support_schema import EXCLUDED_COORDINATES, SELECTED_COORDINATES
from .train import SeededDistributionRegularizer


VARIANTS = ('full', 'no_residual', 'no_support_mask', 'no_domain')
STAGES = ('residual', 'support_surrogate', 'domain', 'distribution')


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class StrictDeletionGenerator(AblationGenerator):
    """Keep the original generator interface with an actually absent residual."""

    def __init__(self, base, shadow, coordinates, support, *, variant):
        if variant not in VARIANTS:
            raise ValueError('Unknown strict-deletion variant')
        self.strict_variant = variant
        self.expected_support_count = 28 if variant == 'no_support_mask' else 24
        super().__init__(base, shadow, coordinates, support,
                         residual_mode='base_only' if variant == 'no_residual' else 'uniform')

    def set_phase(self, phase):
        if self.strict_variant == 'no_residual':
            if phase not in ('base', *STAGES):
                raise ValueError('Unknown phase')
            self._base_stage = False
            self._use_residual = False
            _freeze(self.base_generator)
            _freeze(self.base_shadow)
            if self.residual is not None:
                raise RuntimeError('Strict deletion must not instantiate a residual')
            return self
        return super().set_phase(phase)


class UnboundedGeneratedRegularizer(SeededDistributionRegularizer):
    """Original D, loss and scale, with only the fake-domain rejection removed.

    The clean reference remains in its original [-10,10] domain. Finite tensor,
    shape, device, immutable-condition and D-ownership checks remain inherited.
    This class performs no clipping or replacement of generated values.
    """

    def discriminator_inputs(self, injected_features):
        self.assert_contract()
        self._validate_features(injected_features)
        if (tuple(injected_features.shape) != self._shape
                or injected_features.device != self._device
                or injected_features.dtype != self._dtype):
            raise ValueError('Injected feature shape/device/dtype differs from the binding')
        observed = injected_features[self._targets[:, None], self._conditions]
        if not torch.equal(observed.detach(), self._clean_conditions):
            raise ValueError('Frozen observed conditions were changed by feature injection')
        fake = injected_features[self._targets[:, None], self._coordinates]
        conditions = (self._clean_conditions if self.arm == 'conditional'
                      else torch.zeros_like(self._clean_conditions))
        return (torch.cat((self._real, conditions), dim=1) / 10.,
                torch.cat((fake, conditions), dim=1) / 10.)


def restore_base_prefix(prefix_path, prefix_sha256, *, seed, variant, coordinates,
                        support, device, expected_base_epochs=200):
    """Restore only a verified pre-residual Full checkpoint; no data is opened."""
    path, device = Path(prefix_path), torch.device(device)
    if (variant not in VARIANTS or type(expected_base_epochs) is not int
            or expected_base_epochs < 1 or _sha(path) != prefix_sha256):
        raise ValueError('Unregistered variant or base-prefix SHA mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    record = checkpoint.get('stage_record', {})
    if (checkpoint.get('stage') != 'base' or checkpoint.get('seed') != seed
            or checkpoint.get('variant') != 'full'
            or checkpoint.get('epoch') != expected_base_epochs - 1
            or record.get('epochs') != expected_base_epochs
            or record.get('generator_steps') != expected_base_epochs
            or record.get('surrogate_steps') != expected_base_epochs
            or record.get('support_count') != 28
            or record.get('output_mode') != 'identity'
            or record.get('discriminator_steps') != 0
            or checkpoint.get('distribution') is not None):
        raise ValueError('Prefix must be the completed Full base stage before removed mechanisms')
    original = checkpoint['generator']
    coords = torch.tensor(SELECTED_COORDINATES, dtype=torch.long)
    legal = ~torch.isin(coords, torch.tensor(EXCLUDED_COORDINATES))
    if (not torch.equal(coordinates.cpu(), coords)
            or not torch.equal(original['selected_coordinates'], coords)
            or not torch.equal(original['hard_support'], legal)
            or int(original['output_mode_code']) != 0
            or int(support.sum()) != 24
            or bool((original['residual.head.weight'] != 0).any())
            or bool((original['residual.head.bias'] != 0).any())):
        raise ValueError('Prefix has wrong coordinates, prior support/projection or a learned residual')
    for name, value in original.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite base-prefix tensor: ' + name)
    shadow_values = {key[len('base_shadow.'):]: value for key, value in original.items()
                     if key.startswith('base_shadow.')}
    if (set(shadow_values) != set(checkpoint['surrogate'])
            or any(not torch.equal(value, checkpoint['surrogate'][key])
                   for key, value in shadow_values.items())):
        raise ValueError('Base conditioning shadow and base surrogate differ')
    # Constructor draws do not alter the common continuation RNG stream.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        base = GraphTrojanNet(device, 80, 30).to(device)
        shadow = GCN(784, 80, 2, dropout=0., device=device).to(device)
        generator = StrictDeletionGenerator(base, shadow, coords.to(device), legal.to(device),
                                            variant=variant).to(device)
    restored = {key: value for key, value in original.items()
                if variant != 'no_residual' or not key.startswith('residual.')}
    generator.load_state_dict(restored, strict=True)
    surrogate = deepcopy(generator.base_shadow)
    surrogate.load_state_dict(checkpoint['surrogate'], strict=True)
    _freeze(generator.base_generator)
    _freeze(generator.base_shadow)
    return generator, surrogate, checkpoint


def _epochs(config):
    values = config.get('stage_epochs', {name: 200 for name in STAGES})
    if (not isinstance(values, dict) or set(values) != set(STAGES)
            or any(type(v) is not int or v < 1 for v in values.values())):
        raise ValueError('Each post-base stage needs a positive registered epoch count')
    return dict(values)


def _assert_injection(original, injection, targets, generator, context, weights):
    changed = injection.features
    allowed = torch.zeros_like(original, dtype=torch.bool)
    allowed[targets[:, None], generator.selected_coordinates[generator.hard_support]] = True
    if (not bool(torch.isfinite(changed).all())
            or not torch.equal(changed[~allowed], original[~allowed])
            or not torch.equal(injection.edge_index, context.edge_index)
            or not torch.equal(injection.edge_weight, weights)):
        raise ValueError('Injection changed forbidden coordinates, topology or finite scope')
    if generator.output_mode == 'domain_projection' and bool((changed[allowed].abs() > 10).any()):
        raise ValueError('Projected output escaped the registered domain')


def fit_from_base(train, indices, proxies, coordinates, support, *, prefix_path,
                  prefix_sha256, seed, variant, output_dir, config=None, progress=None):
    """Continue four stages from the same bound base; no compensation for deletion.

    ``support`` always denotes the original clean-derived SAGE24 set. The
    no_support_mask arm writes all legal28, while D still observes SAGE24 so its
    architecture, real reference and loss remain unchanged. This 28-write arm
    is not an equal-write-budget random-feature comparison.
    """
    config = dict(config or {})
    epochs = _epochs(config)
    if variant not in VARIANTS or type(seed) is not int or seed not in (20260621, 20260622, 20260623):
        raise ValueError('Unregistered strict-deletion variant/seed')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable')
    fixed = dict(discriminator_seed=seed+1000, generator_lr=.0002, residual_lr=.0002,
                 shadow_lr=.0002, weight_decay=.0005, target_weight=1.,
                 similarity_weight=.1, clean_kl_weight=1., distribution_weight=.1)
    if any(key in config and config[key] != value for key,value in fixed.items()):
        raise ValueError('Strict deletion cannot change registered losses or optimizer settings')
    x, labels, edge = (train[name] for name in ('x', 'y', 'edge_index'))
    if x.device.type != 'cuda':
        raise ValueError('Strict deletion training requires CUDA; no CPU fallback')
    if (x.ndim != 2 or x.shape[1] != 784 or x.dtype != torch.float32
            or x.requires_grad or not bool(torch.isfinite(x).all())):
        raise ValueError('Require fixed finite CUDA FP32 clean features')
    context = train.get('context', SimpleNamespace(edge_index=edge))
    if not torch.equal(_check_context(x, context), edge):
        raise ValueError('TRAIN context edges changed')
    if labels.shape != (len(x),) or labels.device != x.device or labels.dtype != torch.long:
        raise ValueError('TRAIN labels do not align')
    idx_train, attach, outer = (indices[name] for name in ('idx_train', 'idx_attach', 'fixed_outer'))
    for name, value in (('idx_train', idx_train), ('idx_attach', attach), ('fixed_outer', outer)):
        _indices(value, name, x)
    if (not torch.equal(outer[:len(attach)], attach)
            or bool(torch.isin(idx_train, outer).any())
            or not bool((labels[attach] == 0).all())):
        raise ValueError('Fixed original negative attachments/outer changed')
    coordinates, support = coordinates.to(x.device), support.to(x.device)
    AblationGenerator._validate_support(support, coordinates)
    if tuple(coordinates.tolist()) != SELECTED_COORDINATES or int(support.sum()) != 24:
        raise ValueError('Require original coordinates30 and SAGE24 binding')
    if len(proxies) != 2 or len({id(p) for p in proxies}) != 2:
        raise ValueError('Require two distinct frozen clean proxies')
    for model in proxies:
        if (any(p.requires_grad or p.grad is not None or p.device != x.device for p in model.parameters())
                or any(m.training for m in model.modules())):
            raise ValueError('Clean proxies must be frozen/eval on the training device')
        _finite_module(model, 'clean proxy')
    proxy_hashes = [_state_hash(p) for p in proxies]
    weights = AblationGenerator._weights(x, context, train.get('edge_weight'))
    inputs = dict(x=x, y=labels, edge_index=edge, idx_train=idx_train,
                  idx_attach=attach, fixed_outer=outer)
    input_hashes = {name: _tensor_hash(value) for name, value in inputs.items()}
    generator, surrogate, prefix = restore_base_prefix(prefix_path, prefix_sha256,
        seed=seed, variant=variant, coordinates=coordinates, support=support,
        device=x.device, expected_base_epochs=config.get('expected_base_epochs', 200))
    if input_hashes != prefix['initialization']['input_sha256']:
        raise ValueError('Input tensors/attachments differ from the common base prefix')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / name).exists() for name in ('initial_state.pt', 'final.pt', 'epochs.jsonl')):
        raise FileExistsError('Refusing to overwrite strict-deletion training evidence')
    device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return _continue(generator, surrogate, x, labels, context, weights, idx_train,
            attach, outer, proxies, coordinates, support, seed, variant, folder,
            prefix_path, prefix_sha256, prefix, input_hashes, proxy_hashes, epochs, progress)


def _continue(generator, surrogate, x, labels, context, weights, idx_train, attach,
              outer, proxies, coordinates, support, seed, variant, folder,
              prefix_path, prefix_sha256, prefix, input_hashes, proxy_hashes, epochs, progress):
    legal = ~torch.isin(coordinates, torch.tensor(EXCLUDED_COORDINATES, device=x.device))
    base_epochs = prefix['stage_record']['epochs']
    metadata = dict(seed=seed, variant=variant, expected_support_count=generator.expected_support_count,
        generator_write_count=generator.expected_support_count, discriminator_observed_count=24,
        discriminator_coordinates=coordinates[support].tolist(),
        additional_unobserved_write_coordinates=(coordinates[legal & ~support].tolist()
            if variant == 'no_support_mask' else []),
        prefix=dict(path=str(Path(prefix_path).resolve()), sha256=prefix_sha256,
                    stage='base', generator_steps=base_epochs, surrogate_steps=base_epochs),
        shared_attack_trained_prefix=True, fresh_attack_initialization=False,
        attack_checkpoint_loaded=True, final_full_attack_checkpoint_loaded=False,
        removed_mechanism_previously_trained=False, post_base_compensation=False,
        continuation_rng_policy='common explicit seed reset; original base RNG not restored',
        continuation_rng_seed=seed, stage_epochs=epochs, input_sha256=input_hashes,
        clean_proxy_sha256=proxy_hashes,
        final_support=(legal if variant == 'no_support_mask' else support).tolist(),
        discriminator_support=support.tolist(),
        no_domain_policy='finite unbounded fake values, original D scale/loss/architecture' if variant == 'no_domain' else None)
    _json(folder / 'initialization.json', metadata)
    torch.save(dict(generator=_cpu_state(generator), surrogate=_cpu_state(surrogate), metadata=metadata,
                    rng_cpu=torch.get_rng_state().cpu(), rng_cuda=torch.cuda.get_rng_state(x.device).cpu()),
               folder / 'initial_state.pt')
    frozen_base = _state_hash(generator.base_generator)
    frozen_shadow = _state_hash(generator.base_shadow)
    penalty = FrozenMeanCleanResponsePenalty(proxies, x, context.edge_index, weights, 1.)
    inner_indices = torch.cat((idx_train, attach))
    outer_indices = torch.cat((idx_train, outer))
    inner_labels = labels.clone(); inner_labels[attach] = 1
    outer_labels = labels.clone(); outer_labels[outer] = 1
    records, stages = [], []
    cumulative_g = cumulative_s = 0
    distribution = None
    with (folder / 'epochs.jsonl').open('x') as log:
        for stage in STAGES:
            generator.set_phase(stage)
            effective = legal if variant == 'no_support_mask' or stage == 'residual' else support
            generator.set_support(effective)
            generator.set_output_mode('domain_projection'
                if stage in ('domain', 'distribution') and variant != 'no_domain' else 'identity')
            surrogate.requires_grad_(True)
            s_optimizer = torch.optim.Adam(surrogate.parameters(), lr=.0002, weight_decay=.0005)
            active = [p for p in generator.parameters() if p.requires_grad]
            if variant == 'no_residual' and (generator.residual is not None or active):
                raise RuntimeError('Residual deletion acquired a replacement trainable generator')
            g_optimizer = torch.optim.Adam(active, lr=.0002, weight_decay=.0005) if active else None
            if stage == 'distribution':
                cls = UnboundedGeneratedRegularizer if variant == 'no_domain' else SeededDistributionRegularizer
                distribution = cls('unconditional', x, outer, coordinates[support], seed=seed + 1000,
                    provenance=dict(scope='strict_deletion_shared_base_train_only', seed=seed, variant=variant,
                                    generator_write_count=generator.expected_support_count,
                                    discriminator_observed_count=24))
            before = dict(generator=_state_hash(generator), surrogate=_state_hash(surrogate),
                          active_generator=_parameter_list_hash(generator, active))
            stage_g = stage_s = 0
            for epoch in range(epochs[stage]):
                generator.train().zero_grad(set_to_none=True)
                surrogate.requires_grad_(True).train()
                s_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    inner_injected = generator.inject(x, context, attach, weights)
                    _assert_injection(x, inner_injected, attach, generator, context, weights)
                output = _log_probabilities(surrogate, inner_injected.features, context.edge_index, weights)
                inner_loss = F.nll_loss(output[inner_indices], inner_labels[inner_indices])
                _finite_loss(inner_loss, 'strict inner')
                inner_loss.backward()
                inner_count, inner_norm = _finite_gradients(surrogate.parameters(), 'surrogate')
                s_optimizer.step(); _finite_module(surrogate, 'surrogate')
                stage_s += 1; cumulative_s += 1
                s_optimizer.zero_grad(set_to_none=True)
                surrogate.requires_grad_(False).eval()
                generator.zero_grad(set_to_none=True)
                d_record = dict(d_loss=0., d_real_loss=0., d_fake_loss=0., d_steps=0)
                if distribution is not None:
                    with torch.no_grad():
                        d_injected = generator.inject(x, context, outer, weights)
                        _assert_injection(x, d_injected, outer, generator, context, weights)
                    d_record = distribution.discriminator_step(d_injected.features)
                with torch.set_grad_enabled(g_optimizer is not None):
                    injected = generator.inject(x, context, outer, weights, detach=False)
                    _assert_injection(x, injected, outer, generator, context, weights)
                    changed = injected.features
                    output = _log_probabilities(surrogate, changed, context.edge_index, weights)
                    nll = F.nll_loss(output[outer_indices], outer_labels[outer_indices])
                    similarity = .1 * (1. - F.cosine_similarity(changed, x, dim=1).mean())
                    clean_kl = penalty(changed, context.edge_index, weights, outer)
                    d_loss = changed.sum() * 0. if distribution is None else distribution.generator_loss(changed)
                    loss = nll + similarity + clean_kl + d_loss
                    _finite_loss(loss, 'strict outer')
                    outer_count = 0; outer_norm = 0.
                    if g_optimizer is not None:
                        g_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        outer_count, outer_norm = _finite_gradients(active, 'residual')
                        g_optimizer.step(); _finite_module(generator, 'generator')
                        stage_g += 1; cumulative_g += 1
                surrogate.requires_grad_(True)
                if (_state_hash(generator.base_generator) != frozen_base
                        or _state_hash(generator.base_shadow) != frozen_shadow):
                    raise RuntimeError('Frozen common-prefix base generator/conditioning shadow changed')
                row = dict(seed=seed, variant=variant, stage=stage, epoch=epoch,
                    device=str(x.device), support_count=int(generator.hard_support.sum()),
                    output_mode=generator.output_mode, active_generator='residual' if active else 'none',
                    inner_nll=float(inner_loss.detach()), outer_nll=float(nll.detach()),
                    similarity_loss=float(similarity.detach()), clean_kl_loss=float(clean_kl.detach()),
                    clean_kl_weight=1., clean_kl_members=2,
                    distribution_loss=float(d_loss.detach()), distribution_weight=.1 if distribution else 0.,
                    outer_loss=float(loss.detach()), inner_gradient_parameters=inner_count,
                    inner_gradient_l2=inner_norm, outer_gradient_parameters=outer_count,
                    outer_gradient_l2=outer_norm, stage_generator_steps=stage_g,
                    stage_surrogate_steps=stage_s, cumulative_generator_steps=cumulative_g,
                    cumulative_surrogate_steps=cumulative_s, **d_record)
                records.append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                if progress is not None:
                    progress(deepcopy(row))
            generator.zero_grad(set_to_none=True); surrogate.zero_grad(set_to_none=True)
            summary = dict(stage=stage, epochs=epochs[stage], generator_steps=stage_g,
                surrogate_steps=stage_s, generator_frozen=not bool(active),
                generator_update_opportunities=epochs[stage] if active else 0,
                nominal_stage_iterations=epochs[stage], no_deleted_steps_compensated=True,
                active_generator_parameters=before['active_generator']['names'],
                initial_state_sha256=before,
                final_state_sha256=dict(generator=_state_hash(generator), surrogate=_state_hash(surrogate),
                                        active_generator=_parameter_list_hash(generator, active)),
                frozen_base_generator_sha256=frozen_base, frozen_conditioning_shadow_sha256=frozen_shadow,
                support_count=int(generator.hard_support.sum()), output_mode=generator.output_mode,
                discriminator_steps=distribution.steps if distribution else 0,
                effective_generator_distribution_optimization=distribution is not None and bool(active),
                selection_policy='final_epoch', selected_epoch=epochs[stage] - 1)
            stages.append(summary)
            checkpoint = dict(generator=_cpu_state(generator), surrogate=_cpu_state(surrogate),
                stage=stage, seed=seed, variant=variant, epoch=epochs[stage]-1,
                stage_record=summary, initialization=metadata,
                architecture=dict(input_dim=784, hidden_dim=80, output_dim=30,
                                  residual_mode=generator.residual_mode, strict_variant=variant,
                                  expected_support_count=generator.expected_support_count, threshold=.5),
                distribution=distribution.snapshot() if distribution else None)
            torch.save(checkpoint, folder / f'{stage}_final.pt')
            _json(folder / 'stages.json', stages)
    _freeze(generator); _freeze(surrogate)
    if [_state_hash(p) for p in proxies] != proxy_hashes:
        raise RuntimeError('Frozen clean proxies changed')
    final_inputs = dict(x=x, y=labels, edge_index=context.edge_index, idx_train=idx_train,
                        idx_attach=attach, fixed_outer=outer)
    if {k: _tensor_hash(v) for k, v in final_inputs.items()} != input_hashes:
        raise RuntimeError('Original TRAIN inputs changed')
    checkpoint['stages'] = stages
    torch.save(checkpoint, folder / 'final.pt')
    result = dict(status='STRICT_DELETION_TRAINING_COMPLETE', generator=generator, surrogate=surrogate,
        checkpoint=str(folder/'final.pt'), final_checkpoint=str(folder/'final.pt'),
        stages=stages, records=records, initialization=metadata,
        generator_steps=cumulative_g, surrogate_steps=cumulative_s,
        base_prefix_generator_steps=base_epochs, base_prefix_surrogate_steps=base_epochs,
        total_generator_steps_including_prefix=base_epochs+cumulative_g,
        total_surrogate_steps_including_prefix=base_epochs+cumulative_s,
        discriminator_steps=distribution.steps if distribution else 0,
        expected_support_count=generator.expected_support_count,
        discriminator_observed_count=24, selection_policy='final_epoch',
        original_inputs_unchanged=True, frozen_clean_proxies_unchanged=True)
    _json(folder/'completion.json', {k:v for k,v in result.items()
          if k not in ('generator','surrogate','records')})
    return result
