"""Current unconditional SPEAR continuation on the fixed TRAIN/dev roles.

Offline AV2/self-owned GNN robustness research. Importing or checking this module
does not train. The explicit ``--run`` entry writes a new output directory, uses
only the 512 TRAIN and 128 dev scenes, and never opens audit/VAL/TEST arrays.
The numerical update implementation lives in ``distribution_training``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from .attention_context import attention_context, permutation_stats
from .distribution import DistributionRegularizer, DISCRIMINATOR_SEED
from .distribution_training import fit_distribution_residual
from .domain_generator import DomainConstrainedGenerator
from .dual_clean_proxy import FrozenMeanCleanResponsePenalty
from .role_data import build_contexts, _tensor_hash as context_tensor_hash
from .role_training import ResidualTrainingConfig, _state_hash, _tensor_hash


SEEDS = (20260621, 20260622, 20260623)
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'results'
AUC_POLICY = 'maximum_dev_roc_auc_strict_improvement_earliest_tie'
TRAINING_CONFIG = asdict(ResidualTrainingConfig())
DISTRIBUTION_CONFIG = dict(lr=.0002, weight_decay=.0005, weight=.1, seed=20261621,
    condition_coordinates=[149, 199, 507, 557, 718, 721], hidden_dim=64)


def validate_training_contract(bundle):
    """Reject configuration edits that would change the registered current method."""
    config, manifest = bundle.config, bundle.manifest
    fixed_arguments = dict(hidden=80, train_lr=.01, shadow_lr=.0002, trojan_lr=.0002,
        weight_decay=.0005, dropout=.5, thrd=.5, target_class=1, epochs=200,
        trojan_epochs=200, inner=1, alpha_int=30, outter_size=512, vs_number=26,
        target_loss_weight=1., homo_loss_weight=.1)
    if (manifest.get('training_config') != TRAINING_CONFIG
            or manifest.get('distribution_config') != DISTRIBUTION_CONFIG
            or manifest.get('training_seed') != 20263621
            or any(config['arguments'].get(key) != value for key, value in fixed_arguments.items())
            or any(config.get(key) != value for key, value in dict(seed=20260621,
                shuffle_seed=20260621, training_seed=20263621, score_dim=32, victim_epochs=200).items())
            or len(config['roles']['train']) != 512 or len(config['roles']['dev']) != 128):
        raise ValueError('Registered current-method configuration or budgets changed')


def training_seeds(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed not in SEEDS:
        raise ValueError('Require one of the registered seeds 20260621/22/23')
    return dict(continuation=int(seed) + 3000, discriminator=int(seed) + 1000,
                victim=int(seed))


class SeededDistributionRegularizer(DistributionRegularizer):
    """Original replication adapter: only the discriminator initialization varies."""

    def __init__(self, arm, original_features, target_indices, selected_coordinates,
                 *, seed=DISCRIMINATOR_SEED, **kwargs):
        allowed = tuple(value + 1000 for value in SEEDS)
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed not in allowed:
            raise ValueError('Unregistered discriminator initialization seed')
        super().__init__(arm, original_features, target_indices, selected_coordinates,
                         seed=DISCRIMINATOR_SEED, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            fresh = nn.Sequential(nn.Linear(30, 64), nn.LeakyReLU(.2),
                                  nn.Linear(64, 64), nn.LeakyReLU(.2), nn.Linear(64, 1))
            fresh = fresh.to(device=self._device, dtype=self._dtype)
        self.discriminator.load_state_dict(fresh.state_dict(), strict=True)
        self._config['seed'] = int(seed)
        self._config_json = json.dumps(self._config, sort_keys=True)
        self._initial_discriminator_sha256 = self.discriminator_hash()
        if self.optimizer is not None and self.optimizer.state:
            raise ValueError('Seed adapter requires an empty fresh optimizer')
        self.assert_contract()


def new_generator(reference):
    """Copy the frozen projected start, preserving its exact weights and support."""
    if not isinstance(reference, DomainConstrainedGenerator) or reference.output_mode != 'domain_projection':
        raise ValueError('Require the registered domain-projection starting generator')
    generator = deepcopy(reference)
    generator.zero_grad(set_to_none=True)
    generator.residual.requires_grad_(True)
    generator.eval()
    if _state_hash(generator) != _state_hash(reference):
        raise ValueError('Continuation initialization differs from the fixed start')
    return generator


def check_outer(train, indices, expected, coordinates, seed=20260621):
    """Verify the original 26 attachments and fixed 538 outer rows; do not select anew."""
    outer = torch.tensor(expected['outer_indices'], dtype=torch.long)
    fit = torch.cat([indices['idx_train'], indices['idx_attach']])
    if (len(indices['idx_attach']) != 26 or len(outer) != 538 or len(fit) != 1902
            or len(fit.unique()) != 1902 or len(outer.unique()) != 538
            or coordinates.tolist() != expected['selected_coordinates']
            or indices['idx_attach'].tolist() != expected['attachment_indices']
            or not torch.equal(train['y'], indices['train_labels'])
            or not bool(train['eligible'][fit].all())
            or not bool((train['y'][indices['idx_attach']] == 0).all())
            or bool(torch.isin(indices['idx_train'], outer).any())):
        raise ValueError('Original attachment, supervision or outer identity changed')
    draw = indices['idx_unlabeled'][np.random.RandomState(seed).choice(
        len(indices['idx_unlabeled']), 512, replace=False)]
    if not torch.equal(outer, torch.cat([indices['idx_attach'], draw])):
        raise ValueError('Original fixed 512 nonattachment outer draw changed')
    return outer


def restore_training_inputs(bundle):
    """Restore only TRAIN/dev and validate bytes, supervision and attention binding."""
    from .data import load_normalizer, load_role

    config = bundle.config
    if (len(config['roles']['train']) != 512 or len(config['roles']['dev']) != 128
            or config['seed'] != 20260621 or config['shuffle_seed'] != 20260621
            or config['victim_epochs'] != 200):
        raise ValueError('Fixed TRAIN/dev population or budgets changed')
    ledger = []
    normalizer = load_normalizer(bundle)
    parts, train = load_role(bundle, 'train', normalizer=normalizer, ledger=ledger)
    indices = torch.load(bundle.asset_path('indices'), map_location='cpu', weights_only=True)
    expected = json.loads(bundle.asset_path('input_contract').read_text())
    for key, field in (('x', 'clean_features_sha256'), ('y', 'clean_labels_sha256')):
        if hashlib.sha256(train[key].numpy().tobytes()).hexdigest() != expected[field]:
            raise ValueError('Original TRAIN bytes changed: ' + key)
    coordinates = torch.tensor(bundle.manifest['selected_coordinates'], dtype=torch.long)
    outer = check_outer(train, indices, expected, coordinates, config['seed'])
    # The original frozen component binding predates the explicit split sidecar.
    # Only this metadata view removes that optional sidecar; the reader and ledger
    # have already checked the original train split for every opened graph.
    if any(part.get('split') != 'train' or 'split' in part['metadata'] for part in parts):
        raise ValueError('Unexpected original TRAIN component metadata layout')
    views = [{key: value for key, value in part.items() if key != 'split'} for part in parts]
    contexts = build_contexts(views, train, seed=config['shuffle_seed'])
    context = attention_context(contexts['typed'], contexts['binding']['scene_order'],
                                config['shuffle_seed'])
    binding = json.loads(bundle.asset_path('context_binding').read_text())
    permutation = torch.load(bundle.asset_path('score_permutation'), weights_only=True, map_location='cpu')
    if (contexts['binding'] != binding['original_binding']
            or context_tensor_hash(context.score_permutation) != binding['score_permutation_sha256']
            or not torch.equal(context.score_permutation, permutation)
            or permutation_stats(context) != binding['statistics']):
        raise ValueError('TRAIN context or score permutation differs from the frozen binding')
    _, dev = load_role(bundle, 'dev', normalizer=normalizer, ledger=ledger)
    dev_indices = torch.nonzero(dev['eligible'], as_tuple=False).flatten() + len(train['x'])
    if not torch.equal(dev_indices, indices['dev_indices']):
        raise ValueError('Original development supervision changed')
    if (len(ledger) != 640 or sum(row['role'] == 'train' for row in ledger) != 512
            or sum(row['role'] == 'dev' for row in ledger) != 128
            or any(row['split'] != 'train' or row['status'] != 'READ_COMPLETE' for row in ledger)):
        raise ValueError('Unexpected TRAIN/dev graph access ledger')
    return dict(train=train, dev=dev, indices=indices, outer=outer, context=context,
                ledger=ledger, normalizer=normalizer)


def new_distribution(generator, train, outer, seed):
    return SeededDistributionRegularizer('unconditional', train['x'], outer,
        generator.selected_coordinates[generator.hard_support],
        seed=training_seeds(seed)['discriminator'], provenance=dict(
            split='train', role='train', scene_count=512, population='original538_outer',
            real_reference='same_outer_target_rows', internal_dev_arrays_used=False,
            internal_audit_arrays_used=False))


def no_update_preflight(generator, reference, surrogate, proxies, inputs, seed):
    """Verify exact replay and task/distribution gradients without optimizer steps."""
    train, context, idx, outer = (inputs[k] for k in ('train', 'context', 'indices', 'outer'))
    models = [generator, reference, surrogate, *proxies]
    before = [_state_hash(model) for model in models]
    rng = torch.get_rng_state().clone()
    weights = train['x'].new_ones(context.edge_index.shape[1])
    with torch.no_grad():
        actual = generator(train['x'], context, outer)
        if not torch.equal(actual, reference(train['x'], context, outer)):
            raise ValueError('Zero-step continuation does not exactly replay the fixed start')
    penalty = FrozenMeanCleanResponsePenalty(proxies, train['x'], context.edge_index, weights, weight=1.)
    distribution = new_distribution(generator, train, outer, seed)
    distribution_before = _state_hash(distribution.discriminator)
    injected = generator.inject(train['x'], context, outer, weights, detach=False)
    config = ResidualTrainingConfig()
    labels = train['y'].clone()
    labels[outer] = config.target_class
    selected = torch.cat([idx['idx_train'], outer])
    output = surrogate(injected.features, context.edge_index, weights)
    task_loss = (config.target_weight * F.nll_loss(output[selected], labels[selected])
        + config.similarity_weight * (1 - F.cosine_similarity(injected.features, train['x'], dim=1).mean())
        + penalty(injected.features, context.edge_index, weights, outer))
    parameters = tuple(generator.residual.named_parameters())
    norms = {}
    for name, loss in (('task', task_loss), ('distribution', distribution.generator_loss(injected.features))):
        grads = torch.autograd.grad(loss, tuple(value for _, value in parameters),
                                    retain_graph=True, allow_unused=True)
        if any(grad is not None and not bool(torch.isfinite(grad).all()) for grad in grads):
            raise ValueError('Nonfinite ' + name + ' preflight gradient')
        norms[name] = sum(float(grad.double().norm()) ** 2
            for (key, _), grad in zip(parameters, grads)
            if grad is not None and not key.startswith('scorer.')) ** .5
        if norms[name] <= 0:
            raise ValueError('Missing active residual ' + name + ' gradient')
    smoke = distribution.loss_smoke(injected.features)
    if (before != [_state_hash(model) for model in models]
            or distribution_before != _state_hash(distribution.discriminator)
            or not torch.equal(rng, torch.get_rng_state()) or distribution.steps != 0
            or any(p.grad is not None for model in [*models, distribution.discriminator]
                   for p in model.parameters())):
        raise ValueError('No-update preflight changed weights, RNG, gradients or update count')
    return dict(status='NO_UPDATE_PREFLIGHT_PASS', optimizer_steps=0,
        exact_projected_replay=True, residual_gradient_l2=norms, distribution=smoke,
        projected_outer_output_sha256=_tensor_hash(actual))


def fit_unconditional(generator, surrogate, proxies, inputs, *, seed=20260621, progress=None):
    """Tensor API for the unchanged 200-step, fixed-final-epoch current method."""
    seeds = training_seeds(seed)
    train, context, idx, outer = (inputs[k] for k in ('train', 'context', 'indices', 'outer'))
    if len(idx['idx_attach']) != 26 or len(outer) != 538:
        raise ValueError('Current method requires the original 26 attachments and 538 outer rows')
    weights = train['x'].new_ones(context.edge_index.shape[1])
    penalty = FrozenMeanCleanResponsePenalty(proxies, train['x'], context.edge_index, weights, weight=1.)
    distribution = new_distribution(generator, train, outer, seed)
    initial_distribution = distribution.snapshot()
    np.random.seed(seeds['continuation'])
    torch.manual_seed(seeds['continuation'])
    result = fit_distribution_residual(generator, surrogate, train['x'], context, train['y'],
        idx['idx_train'], idx['idx_attach'], outer, penalty, ResidualTrainingConfig(),
        distribution, progress=progress)
    if (result['selected_epoch'] != 199 or result['selection_policy'] != 'final_epoch'
            or result['config'] != TRAINING_CONFIG or result['distribution_arm'] != 'unconditional'
            or any(result[key] != 200 for key in ('residual_steps', 'surrogate_steps',
                'clean_constraint_calls', 'discriminator_steps', 'distribution_generator_calls'))
            or penalty.calls != 200):
        raise ValueError('Current method update budget or final-checkpoint policy changed')
    result['dual_clean_history'] = penalty.history
    return result, distribution, initial_distribution


def tensor_digest(model):
    """The original victim digest convention, retained for initialization parity."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AUCTrace:
    """Observe the original fit forwards and retain the earliest maximum dev AUC."""

    def __init__(self, labels, dev_indices, epochs=200, output_csv=None, progress=None):
        if (labels.dtype != torch.long or labels.ndim != 1 or labels.device.type != 'cpu'
                or dev_indices.dtype != torch.long or dev_indices.ndim != 1
                or dev_indices.device.type != 'cpu' or not len(dev_indices)
                or len(dev_indices.unique()) != len(dev_indices)
                or bool(((dev_indices < 0) | (dev_indices >= len(labels))).any())
                or type(epochs) is not int or epochs < 1):
            raise ValueError('Require CPU labels, valid unique dev indices and positive epochs')
        self.indices, self.truth = dev_indices.clone(), labels[dev_indices].clone()
        if set(self.truth.tolist()) != {0, 1}:
            raise ValueError('Development supervision must contain both binary classes')
        self.nodes, self.epochs = len(labels), epochs
        self.output_csv, self.progress = output_csv, progress
        self.rows, self.training_forwards = [], 0
        self.best_auc, self.best_accuracy = float('-inf'), 0.
        self.best_auc_epoch = self.best_accuracy_epoch = None
        self.auc_state = self.accuracy_digest = None
        self._active, self._writer = False, None

    @torch.no_grad()
    def __call__(self, model, inputs, output):
        if not self._active:
            raise ValueError('AUC observer is inactive')
        if model.training:
            self.training_forwards += 1
            return
        if len(self.rows) >= self.epochs or output.shape != (self.nodes, 2) or output.device.type != 'cpu':
            raise ValueError('Unexpected evaluation forward')
        rng = torch.get_rng_state().clone()
        detached = output.detach()[self.indices]
        if (not bool(torch.isfinite(detached).all())
                or not torch.allclose(detached.logsumexp(1), torch.zeros(len(detached)), atol=1e-5)):
            raise ValueError('Invalid development log probabilities')
        predicted = detached.argmax(1)
        accuracy = float(predicted.eq(self.truth).double().sum() / len(self.truth))
        auc = float(roc_auc_score(self.truth.numpy(), detached.exp()[:, 1].numpy()))
        row = dict(epoch=len(self.rows), auc=auc, accuracy=accuracy,
            loss=float(F.nll_loss(detached, self.truth)), parameters_sha256=tensor_digest(model))
        if accuracy > self.best_accuracy:
            self.best_accuracy, self.best_accuracy_epoch = accuracy, row['epoch']
            self.accuracy_digest = row['parameters_sha256']
        if auc > self.best_auc:
            self.best_auc, self.best_auc_epoch = auc, row['epoch']
            self.auc_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.rows.append(row)
        if self._writer is not None:
            self._writer.writerow(row)
            self._stream.flush()
        if not torch.equal(rng, torch.get_rng_state()):
            raise ValueError('AUC observation consumed RNG')
        if self.progress is not None:
            self.progress(dict(stage='fit_victim', **row))

    @contextmanager
    def observe(self, model):
        if self._active or self.rows or self.training_forwards:
            raise ValueError('AUC observer is single-use')
        sentinel = object()
        old_value = model.__dict__.get('forward', sentinel)
        original = model.forward

        def forward(*args, **kwargs):
            output = original(*args, **kwargs)
            self(model, args, output)
            return output

        stream = None
        try:
            if self.output_csv is not None:
                stream = Path(self.output_csv).open('x', newline='')
                self._stream = stream
                self._writer = csv.DictWriter(stream,
                    fieldnames=['epoch', 'auc', 'accuracy', 'loss', 'parameters_sha256'])
                self._writer.writeheader()
            self._active = True
            model.forward = forward
            yield self
        finally:
            if old_value is sentinel:
                if 'forward' in model.__dict__:
                    delattr(model, 'forward')
            else:
                model.forward = old_value
            self._active = False
            if stream is not None:
                stream.close()

    def finish(self, model):
        if (self._active or len(self.rows) != self.epochs or self.training_forwards != self.epochs
                or self.auc_state is None or tensor_digest(model) != self.accuracy_digest):
            raise ValueError('Original fit call counts or accuracy checkpoint changed')
        return dict(training_forwards=self.training_forwards, eval_forwards=len(self.rows),
            selected_auc_epoch=self.best_auc_epoch, selected_auc_row=self.rows[self.best_auc_epoch],
            best_accuracy_epoch=self.best_accuracy_epoch, selection_policy=AUC_POLICY,
            epoch_numbering='zero based; observed after optimizer step')


def fit_fresh_victim(bundle, inputs, result, folder, *, seed, progress=None):
    from .runtime import new_victim

    seeds = training_seeds(seed)
    train, dev, idx = (inputs[key] for key in ('train', 'dev', 'indices'))
    edges = torch.cat([train['edge_index'], dev['edge_index'] + len(train['x'])], 1)
    weights = torch.ones(edges.shape[1])
    fit_indices = torch.cat([idx['idx_train'], idx['idx_attach']])
    dev_indices = torch.nonzero(dev['eligible'], as_tuple=False).flatten() + len(train['x'])
    if len(fit_indices) != 1902 or not torch.equal(dev_indices, idx['dev_indices']):
        raise ValueError('Original fresh-victim supervision changed')
    np.random.seed(seeds['victim'])
    torch.manual_seed(seeds['victim'])
    victim = new_victim(bundle, seed=seeds['victim'])
    signature = dict(victim_seed=seeds['victim'], initialization_digest=tensor_digest(victim),
        fit_rng_digest=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    if seed == 20260621 and any(signature[key] != bundle.config['victim_' + key]
                               for key in ('initialization_digest', 'fit_rng_digest')):
        raise ValueError('Historical fresh-victim initialization or dropout RNG changed')
    labels = torch.cat([result['poisoned_labels'], dev['y']])
    trace = AUCTrace(labels, dev_indices, 200, Path(folder) / 'victim_dev_trace.csv', progress)
    with trace.observe(victim):
        victim.fit(torch.cat([result['poisoned_features'], dev['x']]), edges, weights,
                   labels, fit_indices, dev_indices, train_iters=200, verbose=False)
    record = trace.finish(victim)
    victim.load_state_dict(trace.auc_state, strict=True)
    victim.eval().requires_grad_(False)
    victim.zero_grad(set_to_none=True)
    return victim, dict(trace=record, **signature)


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _save_checkpoint(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def _dev_quality(model, dev):
    with torch.no_grad():
        output = model(dev['x'], dev['edge_index'], torch.ones(dev['edge_index'].shape[1]))
    selected = dev['eligible']
    truth = dev['y'][selected]
    predicted = output[selected].argmax(1)
    if set(truth.tolist()) != {0, 1}:
        raise ValueError('Development supervision must contain both classes')
    auc = float(roc_auc_score(truth.numpy(), output[selected].exp()[:, 1].numpy()))
    recall = float((predicted[truth == 1] == 1).double().mean())
    passed = auc >= .75 and recall >= .05 and set(predicted.tolist()) == {0, 1}
    return dict(auc=auc, positive_recall=recall,
                gate='PASS_PROXY_QUALITY' if passed else 'STOP_PROXY_QUALITY')


def run(bundle, output, *, seed=20260621, progress=None):
    """Explicit complete current-method fit; create new files and no audit reads."""
    from .runtime import restore_generator, restore_surrogate, restore_proxy

    training_seeds(seed)
    folder = Path(output).expanduser().resolve()
    if folder.exists() or folder == OUTPUT_ROOT or not folder.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Require a NEW output subdirectory under ' + str(OUTPUT_ROOT))
    validate_training_contract(bundle)
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    _write_json(folder / 'started.json', dict(seed=seed, seeds=training_seeds(seed),
        arm='unconditional', manifest=str(bundle.manifest_path), training_config=TRAINING_CONFIG))
    inputs = restore_training_inputs(bundle)
    _write_json(folder / 'train_dev_graph_access_ledger.json', inputs['ledger'])
    reference = restore_generator(bundle, asset_key='start_generator')
    surrogate = restore_surrogate(bundle, asset_key='start_surrogate')
    proxies = [restore_proxy(bundle, index=index) for index in (1, 2)]
    quality = [_dev_quality(model, inputs['dev']) for model in proxies]
    if any(value['gate'] != 'PASS_PROXY_QUALITY' for value in quality):
        raise ValueError('Frozen original proxy failed its existing development quality check')
    _write_json(folder / 'proxy_quality.json', quality)
    generator = new_generator(reference)
    preflight = no_update_preflight(generator, reference, surrogate, proxies, inputs, seed)
    _write_json(folder / 'preflight.json', preflight)
    result, distribution, initial = fit_unconditional(generator, surrogate, proxies, inputs,
                                                       seed=seed, progress=progress)
    _save_checkpoint(folder / 'generator.pt', generator.state_dict())
    _save_checkpoint(folder / 'surrogate.pt', result['surrogate'].state_dict())
    _save_checkpoint(folder / 'discriminator.pt', distribution.discriminator.state_dict())
    _save_checkpoint(folder / 'distribution_initial.pt', initial)
    _save_checkpoint(folder / 'distribution_final.pt', distribution.snapshot())
    excluded = {'generator', 'surrogate', 'poisoned_features', 'poisoned_labels',
                'edge_index', 'edge_weight', 'idx_train', 'idx_attach', 'fixed_outer'}
    _write_json(folder / 'generator_record.json', {k: v for k, v in result.items() if k not in excluded})
    victim, victim_record = fit_fresh_victim(bundle, inputs, result, folder,
                                            seed=seed, progress=progress)
    _save_checkpoint(folder / 'victim_auc.pt', victim.state_dict())
    victim_record['quality'] = _dev_quality(victim, inputs['dev'])
    _write_json(folder / 'victim_record.json', victim_record)
    summary = dict(status='CURRENT_UNCONDITIONAL_TRAINING_COMPLETE', seed=seed,
        residual_steps=200, surrogate_steps=200, discriminator_steps=200, victim_epochs=200,
        generator_selected_epoch=199, victim_selected_epoch=victim_record['trace']['selected_auc_epoch'],
        internal_audit_arrays_loaded=False, original_val_arrays_loaded=False,
        original_test_arrays_loaded=False, output_dir=str(folder))
    _write_json(folder / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=20260621)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Explicitly start the fixed 200-step fit')
    mode.add_argument('--check-only', action='store_true', help='Validate artifact files; no graph reads or training (default)')
    parser.add_argument('--output', type=Path, help='New results/ subdirectory required with --run')
    args = parser.parse_args(argv)
    if args.run and args.output is None:
        parser.error('--run requires --output pointing to a new directory')
    from .runtime import load_bundle
    bundle = load_bundle(args.manifest)
    validate_training_contract(bundle)
    if not args.run:
        for key in ('normalizer', 'indices', 'input_contract', 'context_binding', 'score_permutation',
                    'clean_proxy1', 'clean_proxy2', 'start_generator', 'start_surrogate'):
            bundle.asset_path(key)
        result = dict(status='TRAINING_ASSETS_CHECKED', training_started=False,
            graph_arrays_loaded=False, seed=args.seed, seeds=training_seeds(args.seed),
            train_scenes=len(bundle.config['roles']['train']), dev_scenes=len(bundle.config['roles']['dev']))
    else:
        def progress(record):
            if record.get('epoch', 0) % 20 == 0 or record.get('epoch') == 199:
                print(json.dumps(record, allow_nan=False), flush=True)
        result = run(bundle, args.output, seed=args.seed, progress=progress)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
