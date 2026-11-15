"""Fresh-start internal ablations for the local AV2 feature-graph experiment.

This module never reads dataset/checkpoint assets or evaluates ASR.  Every call
creates new SPEAR G/shadow weights before the first attack or KL update.  Only
the caller's independently trained, clean-only frozen proxies may be reused.
The five-stage schedule uses final-epoch checkpoints throughout.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .models import GCN, GraphTrojanNet
from .support_schema import EXCLUDED_COORDINATES, SELECTED_COORDINATES
from .train import SeededDistributionRegularizer


CONFIG_VARIANTS = ('full', 'random_nodes', 'random_features', 'self_only',
                   'base_only', 'no_kl', 'single_kl', 'no_distribution',
                   'no_distribution_no_domain')
STAGES = ('base', 'residual', 'support_surrogate', 'domain', 'distribution')


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def _active_hash(module):
    digest = hashlib.sha256()
    names = []
    for name, value in module.named_parameters():
        if value.requires_grad:
            names.append(name)
            digest.update(name.encode())
            digest.update(_tensor_hash(value).encode())
    return dict(sha256=digest.hexdigest(), names=names,
                parameter_count=sum(p.numel() for p in module.parameters() if p.requires_grad))


def _finite_module(module, name):
    for key, value in module.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Nonfinite {name} state: {key}; no retry performed')


def _finite_gradients(parameters, name):
    count, squared_norm = 0, None
    for parameter in parameters:
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f'Nonfinite {name} gradient; no retry performed')
            count += 1
            term = parameter.grad.detach().double().square().sum()
            squared_norm = term if squared_norm is None else squared_norm + term
    if not count:
        raise FloatingPointError(f'Missing {name} gradient path; no retry performed')
    return count, float(squared_norm.sqrt())


def _finite_loss(value, name):
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise FloatingPointError(f'Nonfinite {name} loss; no retry performed')


def _freeze(module):
    module.requires_grad_(False).eval()
    module.zero_grad(set_to_none=True)


def _check_context(features, context):
    edge = getattr(context, 'edge_index', None)
    if (not isinstance(edge, torch.Tensor) or edge.dtype != torch.long or edge.ndim != 2
            or edge.shape[0] != 2 or edge.device != features.device
            or bool(((edge < 0) | (edge >= len(features))).any())):
        raise ValueError('Context must provide same-device, in-range long edge_index[2,E]')
    return edge


class UniformResidual(nn.Module):
    """The untyped RoleResidualBlock arithmetic, with an explicit self-only arm.

    Four value matrices reduce to their mean in the uniform arm.  The incoming
    degree and zero output-head initialization match the existing role block.
    The self-only control allocates the same tensors but never reads neighbors.
    """

    def __init__(self, mode='uniform'):
        super().__init__()
        if mode not in ('uniform', 'self_only'):
            raise ValueError('Unknown residual mode')
        self.mode = mode
        self.relation_weight = nn.Parameter(torch.empty(4, 784, 80))
        self.self_projection = nn.Linear(784, 80)
        self.head = nn.Linear(80, 30)
        for weight in self.relation_weight:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, context):
        aggregate = features.new_zeros((len(features), 80))
        if self.mode == 'uniform':
            source, target = context.edge_index
            transformed = features @ self.relation_weight.mean(0)
            aggregate = aggregate.index_add(0, target, transformed[source])
            degree = torch.bincount(target, minlength=len(features)).to(features.dtype)
            aggregate = aggregate / degree.clamp_min(1).unsqueeze(1)
        return self.head(torch.relu(self.self_projection(features) + aggregate))


@dataclass(frozen=True)
class Injection:
    features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor


class AblationGenerator(nn.Module):
    """Fresh SPEAR anchor plus uniform/self-only residual or trainable base G.

    Constructor copies supplied modules.  Public construction defaults to a
    frozen conditioning shadow and frozen base G (trainable G in base_only).
    ``fit_case`` alone enables base-stage G/shadow learning and then freezes the
    conditioning shadow permanently.  Unsupported fields retain original data.
    """

    input_dim = 784

    def __init__(self, base_generator, base_shadow, coordinates, support, *,
                 residual_mode='uniform', output_mode='identity', threshold=.5):
        super().__init__()
        if residual_mode not in ('uniform', 'self_only', 'base_only'):
            raise ValueError('Unknown residual_mode')
        if output_mode not in ('identity', 'domain_projection'):
            raise ValueError('Unknown output_mode')
        if (not isinstance(coordinates, torch.Tensor) or coordinates.dtype != torch.long
                or coordinates.shape != (30,) or coordinates.unique().numel() != 30
                or bool(((coordinates < 0) | (coordinates >= 784)).any())):
            raise ValueError('Require 30 unique in-range long coordinates')
        self._validate_support(support, coordinates)
        if not isinstance(base_generator, nn.Module) or not callable(getattr(base_shadow, 'get_h', None)):
            raise ValueError('Require a base generator and shadow exposing get_h')
        if not math.isfinite(float(threshold)):
            raise ValueError('Nonfinite threshold')
        self.base_generator = deepcopy(base_generator)
        self.base_shadow = deepcopy(base_shadow)
        parameter = next(self.base_generator.parameters())
        self.register_buffer('selected_coordinates', coordinates.detach().to(parameter.device).clone())
        self.register_buffer('hard_support', support.detach().to(parameter.device).clone())
        self.register_buffer('output_mode_code', torch.tensor(int(output_mode == 'domain_projection'),
                                                             device=parameter.device, dtype=torch.uint8))
        self.residual_mode = residual_mode
        self.threshold = float(threshold)
        self.residual = (None if residual_mode == 'base_only' else
                         UniformResidual(residual_mode).to(device=parameter.device, dtype=parameter.dtype))
        self._use_residual = residual_mode != 'base_only'
        self._base_stage = False
        self.set_phase('residual')

    @staticmethod
    def _validate_support(support, coordinates):
        if not isinstance(support, torch.Tensor) or support.dtype != torch.bool or support.shape != (30,):
            raise ValueError('Require bool[30] support')
        coordinates = coordinates.to(support.device)
        if bool((support & ((coordinates == 357) | (coordinates == 715))).any()):
            raise ValueError('Focal metadata coordinates 357/715 must never be changed')
        if not bool(support.any()):
            raise ValueError('Support cannot be empty')

    @property
    def output_mode(self):
        return 'domain_projection' if int(self.output_mode_code) else 'identity'

    def set_support(self, support):
        self._validate_support(support, self.selected_coordinates)
        self.hard_support.copy_(support.to(self.hard_support.device))

    def set_output_mode(self, output_mode):
        if output_mode not in ('identity', 'domain_projection'):
            raise ValueError('Unknown output_mode')
        self.output_mode_code.fill_(int(output_mode == 'domain_projection'))

    def set_phase(self, phase):
        if phase not in STAGES:
            raise ValueError('Unknown training phase')
        self._base_stage = phase == 'base'
        self._use_residual = phase != 'base' and self.residual is not None
        _freeze(self.base_shadow)
        _freeze(self.base_generator)
        if self.residual is not None:
            _freeze(self.residual)
        if phase == 'base' or (phase != 'support_surrogate' and self.residual_mode == 'base_only'):
            self.base_generator.requires_grad_(True)
        elif phase != 'support_surrogate':
            self.residual.requires_grad_(True)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.base_shadow.eval()
        if not any(p.requires_grad for p in self.base_generator.parameters()):
            self.base_generator.eval()
        return self

    def _validate_inputs(self, features, context, targets):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != 784 or not features.is_floating_point()
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Require finite floating features[N,784]')
        _check_context(features, context)
        if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.long or targets.ndim != 1
                or targets.device != features.device or targets.unique().numel() != targets.numel()
                or bool(((targets < 0) | (targets >= len(features))).any())):
            raise ValueError('Targets must be unique in-range same-device long indices')
        if self.selected_coordinates.device != features.device:
            raise ValueError('Generator and features must share device')

    def generate_with_embedding(self, features, context, targets, embedding):
        self._validate_inputs(features, context, targets)
        raw = self.base_generator(embedding[targets].detach(), self.threshold)
        if self._use_residual:
            raw = raw + self.residual(features, context)[targets]
        if raw.shape != (len(targets), 30) or not bool(torch.isfinite(raw).all()):
            raise FloatingPointError('Malformed/nonfinite generator output; no retry performed')
        if self.output_mode == 'domain_projection':
            raw = raw.clamp(-10., 10.)
        clean = features[targets[:, None], self.selected_coordinates]
        return torch.where(self.hard_support.unsqueeze(0), raw, clean)

    def forward(self, features, context, targets):
        self._validate_inputs(features, context, targets)
        self.base_shadow.eval()
        with torch.no_grad():
            embedding = self.base_shadow.get_h(features, context.edge_index)
        return self.generate_with_embedding(features, context, targets, embedding)

    def inject(self, features, context, targets, edge_weight=None, detach=True):
        self._validate_inputs(features, context, targets)
        edge_weight = self._weights(features, context, edge_weight)
        generated = self(features, context, targets)
        output = features.clone()
        output[targets[:, None], self.selected_coordinates] = generated.detach() if detach else generated
        return Injection(output, context.edge_index.clone(), edge_weight.clone())

    @staticmethod
    def _weights(features, context, edge_weight):
        if edge_weight is None:
            return features.new_ones(context.edge_index.shape[1])
        if (edge_weight.shape != (context.edge_index.shape[1],)
                or edge_weight.device != features.device or edge_weight.dtype != features.dtype
                or not bool((edge_weight == 1).all())):
            raise ValueError('Require unchanged unit line-edge weights')
        return edge_weight

    def inject_trigger(self, idx_attach, features, edge_index, edge_weight=None, device=None):
        # Uniform/self-only aggregation needs no role or score sidecar at inference.
        if device is not None and torch.device(device) != features.device:
            raise ValueError('Evaluation must place all tensors on the generator device')
        injected = self.inject(features, SimpleNamespace(edge_index=edge_index), idx_attach, edge_weight)
        return injected.features, injected.edge_index, injected.edge_weight


def _log_probabilities(model, features, edge_index, weights):
    output = model(features, edge_index, weights)
    if (output.shape != (len(features), 2) or not bool(torch.isfinite(output).all())
            or not torch.allclose(output.logsumexp(1), torch.zeros_like(output[:, 0]), atol=1e-5)):
        raise FloatingPointError('Model returned malformed/nonfinite binary log probabilities')
    return output


def _indices(value, name, features):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1
            or value.device != features.device or not value.numel()
            or value.unique().numel() != value.numel()
            or bool(((value < 0) | (value >= len(features))).any())):
        raise ValueError(f'{name} must be nonempty unique in-range same-device long indices')


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _stage_epochs(config):
    supplied = config.get('stage_epochs', config.get('epochs', 200))
    if isinstance(supplied, int) and not isinstance(supplied, bool):
        supplied = {name: supplied for name in STAGES}
    if not isinstance(supplied, dict) or set(supplied) != set(STAGES):
        raise ValueError('stage_epochs must register each of the five stages')
    if any(type(value) is not int or value < 1 for value in supplied.values()):
        raise ValueError('Every stage must have a positive integer epoch budget')
    return dict(supplied)


def fit_case(train, indices, proxies, coordinates, support, *, seed, variant,
             output_dir, config, progress=None):
    """Fit all five stages from new G/shadow weights using CUDA only.

    ``train`` contains x/y/edge_index and optionally a role/attention ``context``.
    ``indices`` contains idx_train, idx_attach, fixed_outer (attachments first).
    Proxies are pre-existing frozen/eval *clean-only* models on the same CUDA
    device; no attack checkpoint or dev/audit data is accepted by this API.
    ``stage_epochs`` defaults to 200 each; small explicit budgets support smoke
    tests only and are recorded. A failure stops the case without a retry.
    """
    if variant not in CONFIG_VARIANTS:
        raise ValueError('Unknown registered ablation variant')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    config = dict(config)
    epochs = _stage_epochs(config)
    inner_steps = config.get('inner_steps', config.get('inner', 1))
    if type(inner_steps) is not int or inner_steps < 1:
        raise ValueError('inner_steps must be a positive integer')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable')
    x, labels, edge = (train[name] for name in ('x', 'y', 'edge_index'))
    if x.device.type != 'cuda':
        raise ValueError('Full ablation fitting requires CUDA; CPU training fallback is disabled')
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 784 or x.requires_grad:
        raise ValueError('Require original CUDA float32 features[N,784] without gradients')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite original features')
    context = train.get('context', SimpleNamespace(edge_index=edge))
    if not torch.equal(_check_context(x, context), edge):
        raise ValueError('Context edges differ from TRAIN edges')
    if labels.shape != (len(x),) or labels.dtype != torch.long or labels.device != x.device:
        raise ValueError('Require aligned same-device long labels')
    idx_train, attach, outer = (indices[name] for name in ('idx_train', 'idx_attach', 'fixed_outer'))
    for name, value in (('idx_train', idx_train), ('idx_attach', attach), ('fixed_outer', outer)):
        _indices(value, name, x)
    if (not torch.equal(outer[:len(attach)], attach) or bool(torch.isin(idx_train, outer).any())
            or not bool((labels[attach] == 0).all())
            or not bool(((labels[idx_train] == 0) | (labels[idx_train] == 1)).all())):
        raise ValueError('Require original-negative attachments, disjoint supervision, attachments-first outer')
    coordinates = coordinates.to(device=x.device)
    support = support.to(device=x.device)
    if tuple(coordinates.tolist()) != SELECTED_COORDINATES or int(support.sum()) != 24:
        raise ValueError('Require original ranked coordinates30 and legal final support24')
    AblationGenerator._validate_support(support, coordinates)
    if len(proxies) != 2 or len({id(model) for model in proxies}) != 2:
        raise ValueError('Require two independently owned frozen clean-only proxies')
    proxy_hashes = []
    for model in proxies:
        if (any(p.device != x.device or p.requires_grad or p.grad is not None for p in model.parameters())
                or any(child.training for child in model.modules())):
            raise ValueError('Clean-only proxies must already be frozen/eval on the CUDA device')
        _finite_module(model, 'clean-only proxy')
        proxy_hashes.append(_state_hash(model))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'initial_state.pt').exists() or (output_dir / 'final.pt').exists():
        raise FileExistsError('Refusing to overwrite an existing ablation fit')
    lr = float(config.get('generator_lr', config.get('trojan_lr', .0002)))
    shadow_lr = float(config.get('shadow_lr', .0002))
    decay = float(config.get('weight_decay', .0005))
    if any(not math.isfinite(v) or v <= 0 for v in (lr, shadow_lr)) or not math.isfinite(decay) or decay < 0:
        raise ValueError('Require finite positive learning rates and nonnegative weight decay')
    if (float(config.get('target_weight', 1.)) != 1.
            or float(config.get('similarity_weight', .1)) != .1):
        raise ValueError('Registered target and similarity weights are fixed at 1 and .1')
    weights = AblationGenerator._weights(x, context, train.get('edge_weight'))
    input_hashes = {name: _tensor_hash(value) for name, value in dict(
        x=x, y=labels, edge_index=edge, idx_train=idx_train, idx_attach=attach, fixed_outer=outer).items()}
    with torch.random.fork_rng(devices=[x.device.index]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return _fit_seeded(x, labels, context, weights, idx_train, attach, outer, proxies,
                           coordinates, support, seed, variant, output_dir, config, epochs,
                           inner_steps, lr, shadow_lr, decay, input_hashes, proxy_hashes, progress)


def _fit_seeded(x, labels, context, weights, idx_train, attach, outer, proxies,
                coordinates, support, seed, variant, output_dir, config, epochs,
                inner_steps, lr, shadow_lr, decay, input_hashes, proxy_hashes, progress):
    rng_initial = dict(cpu=_tensor_hash(torch.get_rng_state()),
                       cuda=_tensor_hash(torch.cuda.get_rng_state(x.device)))
    # This exact initialization order is identical for all nine arms.
    shadow = GCN(784, 80, 2, dropout=0., device=x.device).to(x.device)
    base = GraphTrojanNet(x.device, 80, 30).to(x.device)
    fresh_hashes = dict(base_generator=_state_hash(base), base_shadow=_state_hash(shadow))
    mode = variant if variant in ('self_only', 'base_only') else 'uniform'
    legal_support = ~((coordinates == EXCLUDED_COORDINATES[0]) | (coordinates == EXCLUDED_COORDINATES[1]))
    generator = AblationGenerator(base, shadow, coordinates, legal_support, residual_mode=mode).to(x.device)
    del base, shadow
    generator.set_phase('base')
    surrogate = generator.base_shadow
    selected_proxies = proxies[:1] if variant == 'single_kl' else proxies
    kl_weight = 0. if variant == 'no_kl' else 1.
    penalty = FrozenMeanCleanResponsePenalty(selected_proxies, x, context.edge_index, weights, kl_weight)
    initial = dict(seed=seed, variant=variant, fresh_model_sha256=fresh_hashes,
        initialization_rng_sha256=rng_initial,
        post_initialization_rng_sha256=dict(cpu=_tensor_hash(torch.get_rng_state()),
                                           cuda=_tensor_hash(torch.cuda.get_rng_state(x.device))),
        stage_epochs=epochs, inner_steps=inner_steps, device=str(x.device), dtype=str(x.dtype),
        selection_policy='final_epoch_each_stage_no_dev_or_asr_selection',
        reused_prefix='caller_supplied_frozen_clean_only_proxies', clean_proxy_sha256=proxy_hashes,
        clean_kl_weight=kl_weight, clean_kl_members=len(selected_proxies),
        clean_kl_aggregation='arithmetic_mean_of_individual_KL_q_clean_to_p_changed',
        fresh_initialization=True, attack_checkpoint_loaded=False,
        base_conditioning='original_SPEAR_previous_inner_poison_x',
        continuation_conditioning='original_clean_features_every_call',
        final_support=support.tolist(), legal_baseline_support=legal_support.tolist(),
        input_sha256=input_hashes, scope='local_AV2_offline_owned_model_feature_graph_robustness')
    torch.save(dict(generator=_cpu_state(generator), fresh_model_sha256=fresh_hashes,
                    metadata=initial), output_dir / 'initial_state.pt')
    _save_json(output_dir / 'initialization.json', initial)
    records, stage_records = [], []
    inner_indices = torch.cat((idx_train, attach))
    outer_indices = torch.cat((idx_train, outer))
    inner_labels = labels.clone(); inner_labels[attach] = 1
    outer_labels = labels.clone(); outer_labels[outer] = 1
    poison_x = x.clone().detach()
    distribution = None
    cumulative_g = cumulative_s = 0
    log_path = output_dir / 'epochs.jsonl'
    with log_path.open('w') as log:
        for stage in STAGES:
            if stage == 'residual':
                # The independent surrogate begins from this case's own freshly
                # trained base shadow. The conditioning shadow stays fixed.
                surrogate = deepcopy(generator.base_shadow)
            generator.set_phase(stage)
            generator.set_support(legal_support if stage in ('base', 'residual') else support)
            generator.set_output_mode('domain_projection' if stage in ('domain', 'distribution')
                and variant != 'no_distribution_no_domain' else 'identity')
            if stage == 'base':
                surrogate = generator.base_shadow
            surrogate.requires_grad_(True)
            s_optimizer = torch.optim.Adam(surrogate.parameters(), lr=shadow_lr, weight_decay=decay)
            active_parameters = [p for p in generator.parameters() if p.requires_grad]
            # Base-stage shadow parameters belong only to the S optimizer.
            active_parameters = [p for p in active_parameters
                                 if all(p is not q for q in surrogate.parameters())]
            g_optimizer = (torch.optim.Adam(active_parameters, lr=lr, weight_decay=decay)
                           if active_parameters else None)
            if stage == 'distribution' and variant not in ('no_distribution', 'no_distribution_no_domain'):
                distribution = SeededDistributionRegularizer(
                    'unconditional', x, outer, coordinates[support],
                    seed=int(config.get('discriminator_seed', seed + 1000)),
                    provenance=dict(scope='fresh_internal_ablation_train_only', seed=seed, variant=variant))
            before = dict(generator=_state_hash(generator), surrogate=_state_hash(surrogate),
                          active_generator=_parameter_list_hash(generator, active_parameters),
                          active_surrogate=_active_hash(surrogate))
            frozen_base = (_state_hash(generator.base_generator) if stage != 'base'
                           and mode != 'base_only' else None)
            frozen_shadow = _state_hash(generator.base_shadow) if stage != 'base' else None
            stage_g = stage_s = 0
            d_initial = None if distribution is None else distribution.discriminator_hash()
            for epoch in range(epochs[stage]):
                try:
                    generator.train()
                    generator.zero_grad(set_to_none=True)
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
                        _finite_loss(inner_loss, 'inner')
                        inner_loss.backward()
                        inner_count, inner_norm = _finite_gradients(surrogate.parameters(), 'surrogate')
                        s_optimizer.step()
                        _finite_module(surrogate, 'surrogate')
                        stage_s += 1; cumulative_s += 1
                        inner_losses.append(float(inner_loss.detach()))
                    s_optimizer.zero_grad(set_to_none=True)
                    surrogate.requires_grad_(False).eval()
                    generator.zero_grad(set_to_none=True)
                    d_record = dict(d_loss=0., d_real_loss=0., d_fake_loss=0., d_steps=0)
                    if distribution is not None:
                        with torch.no_grad():
                            d_x = generator.inject(x, context, outer, weights).features
                        d_record = distribution.discriminator_step(d_x)
                    # Even support_surrogate evaluates the identical objective;
                    # only its pre-registered G optimizer opportunity is absent.
                    with torch.set_grad_enabled(g_optimizer is not None):
                        if stage == 'base':
                            with torch.no_grad():
                                embedding = surrogate.get_h(poison_x, context.edge_index)
                            values = generator.generate_with_embedding(x, context, outer, embedding)
                            changed = x.clone()
                            changed[outer[:, None], coordinates] = values
                        else:
                            changed = generator.inject(x, context, outer, weights, detach=False).features
                        output = _log_probabilities(surrogate, changed, context.edge_index, weights)
                        nll = F.nll_loss(output[outer_indices], outer_labels[outer_indices])
                        similarity = 1. - F.cosine_similarity(changed, x, dim=1).mean()
                        clean_kl = penalty(changed, context.edge_index, weights, outer)
                        distribution_loss = (changed.sum() * 0. if distribution is None else
                                             distribution.generator_loss(changed))
                        loss = nll + .1 * similarity + clean_kl + distribution_loss
                        _finite_loss(loss, 'outer')
                        outer_count = 0; outer_norm = 0.
                        if g_optimizer is not None:
                            g_optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            outer_count, outer_norm = _finite_gradients(active_parameters, 'generator')
                            g_optimizer.step()
                            _finite_module(generator, 'generator')
                            stage_g += 1; cumulative_g += 1
                    # The shared base shadow can be enabled only for its next S
                    # step; the continuation conditioning copy remains frozen.
                    surrogate.requires_grad_(True)
                    if frozen_base is not None and _state_hash(generator.base_generator) != frozen_base:
                        raise RuntimeError('Frozen base generator changed')
                    if frozen_shadow is not None and _state_hash(generator.base_shadow) != frozen_shadow:
                        raise RuntimeError('Frozen conditioning shadow changed')
                    record = dict(seed=seed, variant=variant, stage=stage, epoch=epoch,
                        device=str(x.device), support_count=int(generator.hard_support.sum()),
                        output_mode=generator.output_mode, active_generator=('none' if g_optimizer is None else
                            'base_generator' if stage == 'base' or mode == 'base_only' else 'residual'),
                        inner_nll=inner_losses, outer_nll=float(nll.detach()),
                        similarity_loss=float((.1 * similarity).detach()),
                        clean_kl_loss=float(clean_kl.detach()), clean_kl_weight=kl_weight,
                        clean_kl_members=len(selected_proxies), clean_kl_member_raw=penalty.history[-1]['member_raw_kl'],
                        distribution_loss=float(distribution_loss.detach()),
                        distribution_weight=0. if distribution is None else .1,
                        outer_loss=float(loss.detach()),
                        inner_gradient_parameters=inner_count, inner_gradient_l2=inner_norm,
                        outer_gradient_parameters=outer_count, outer_gradient_l2=outer_norm,
                        stage_generator_steps=stage_g, stage_surrogate_steps=stage_s,
                        cumulative_generator_steps=cumulative_g, cumulative_surrogate_steps=cumulative_s,
                        **d_record)
                    records.append(record)
                    log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                    if progress is not None:
                        progress(deepcopy(record))
                except Exception as exc:
                    failure = dict(status='FAILED_NO_RETRY', seed=seed, variant=variant,
                                   stage=stage, epoch=epoch, error=f'{type(exc).__name__}: {exc}')
                    _save_json(output_dir / 'failure.json', failure)
                    raise RuntimeError(f'{variant} seed={seed} stage={stage} epoch={epoch} failed: {exc}') from exc
            generator.zero_grad(set_to_none=True)
            surrogate.zero_grad(set_to_none=True)
            after = dict(generator=_state_hash(generator), surrogate=_state_hash(surrogate),
                         active_generator=_parameter_list_hash(generator, active_parameters),
                         active_surrogate=_active_hash(surrogate))
            summary = dict(stage=stage, epochs=epochs[stage], generator_steps=stage_g,
                surrogate_steps=stage_s, generator_update_opportunities=epochs[stage],
                generator_frozen=stage == 'support_surrogate', surrogate_inner_steps=inner_steps,
                active_generator_parameters=before['active_generator']['names'],
                initial_state_sha256=before, final_state_sha256=after,
                frozen_base_generator_sha256=frozen_base, frozen_conditioning_shadow_sha256=frozen_shadow,
                support_count=int(generator.hard_support.sum()), output_mode=generator.output_mode,
                discriminator_steps=0 if distribution is None else distribution.steps,
                distribution_weight=0. if distribution is None else .1,
                discriminator_initial_sha256=d_initial,
                discriminator_final_sha256=None if distribution is None else distribution.discriminator_hash(),
                selection_policy='final_epoch', selected_epoch=epochs[stage] - 1)
            stage_records.append(summary)
            checkpoint = dict(generator=_cpu_state(generator), surrogate=_cpu_state(surrogate),
                stage=stage, epoch=epochs[stage] - 1, seed=seed, variant=variant,
                architecture=dict(input_dim=784, hidden_dim=80, output_dim=30,
                                  residual_mode=mode, threshold=.5),
                stage_record=summary, initialization=initial,
                distribution=None if distribution is None else distribution.snapshot())
            torch.save(checkpoint, output_dir / f'{stage}_final.pt')
            _save_json(output_dir / 'stages.json', stage_records)
    generator.eval(); surrogate.eval()
    # Freeze all final parameters; evaluation retains the exact trained states.
    _freeze(generator); _freeze(surrogate)
    final_proxy_hashes = [_state_hash(model) for model in proxies]
    if final_proxy_hashes != proxy_hashes:
        raise RuntimeError('Frozen clean-only proxy state changed')
    final_inputs = {name: _tensor_hash(value) for name, value in dict(
        x=x, y=labels, edge_index=context.edge_index, idx_train=idx_train,
        idx_attach=attach, fixed_outer=outer).items()}
    if final_inputs != input_hashes:
        raise RuntimeError('Original caller inputs changed')
    checkpoint['stages'] = stage_records
    torch.save(checkpoint, output_dir / 'final.pt')
    result = dict(status='FULL_ABLATION_CASE_COMPLETE', generator=generator, surrogate=surrogate,
        records=records, history=records, stages=stage_records, checkpoint=str(output_dir / 'final.pt'),
        final_checkpoint=str(output_dir / 'final.pt'), initialization=initial,
        generator_steps=cumulative_g, surrogate_steps=cumulative_s,
        discriminator_steps=0 if distribution is None else distribution.steps,
        selection_policy='final_epoch', original_inputs_unchanged=True, frozen_clean_proxies_unchanged=True)
    _save_json(output_dir / 'completion.json', {key: value for key, value in result.items()
        if key not in ('generator', 'surrogate', 'records', 'history')})
    return result


def _parameter_list_hash(module, parameters):
    ids = {id(p) for p in parameters}
    digest = hashlib.sha256()
    names = []
    for name, value in module.named_parameters():
        if id(value) in ids:
            names.append(name)
            digest.update(name.encode())
            digest.update(_tensor_hash(value).encode())
    return dict(sha256=digest.hexdigest(), names=names,
                parameter_count=sum(p.numel() for p in parameters))
