"""Full-role AV2 training with one optimizer update per complete data pass.

Scene graphs are disconnected. Their globally normalized losses/gradients can
therefore be accumulated without a full-data GPU tensor. No scene or neighbor
is sampled. Dropout is drawn in the registered batch order, so this is a fresh
execution contract, not a bitwise replay of a historical full-tensor run.
Only registered public AV2/WOMD data and researcher-owned models are used, offline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from .full_ablation_evaluation import _cuda_device, _fp32, _inject
from .full_ablation_training import _cpu_state, _finite_gradients, _finite_module, _freeze, _state_hash, _tensor_hash
from .models import GraphSage
from .womd_execution import configure_message_passing


EXECUTION = dict(
    mode='complete_disconnected_scene_batches_global_gradient_accumulation',
    optimizer_steps_per_epoch=1, scene_sampling=False, neighbor_sampling=False,
    dropout_rng='fresh seed; registered ascending scene-batch order; not historical full-tensor bitwise replay')


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def batches(data_dir, role, max_nodes, device):
    from .av2_formal_data import iter_role_batches
    return iter_role_batches(data_dir, role, max_nodes=max_nodes, device=device)


def sorted_indices(indices, population_size=None):
    value = torch.as_tensor(indices).detach().cpu()
    if value.dtype != torch.long or value.ndim != 1 or not len(value):
        raise ValueError('Require nonempty global long supervision indices')
    value = value.sort().values
    if (value.unique_consecutive().numel() != len(value) or int(value[0]) < 0
            or population_size is not None and int(value[-1]) >= population_size):
        raise ValueError('Duplicate or out-of-range global indices')
    return value


def local_indices(sorted_global, batch):
    """Map sorted CPU global indices into a complete contiguous scene batch."""
    start, stop = int(batch['global_start']), int(batch['global_stop'])
    bounds = torch.searchsorted(sorted_global, torch.tensor([start, stop], dtype=torch.long))
    return (sorted_global[int(bounds[0]):int(bounds[1])] - start).to(batch['x'].device)


def _forward(model, batch, model_kind):
    return model(batch['x']) if model_kind == 'mlp' else model(batch['x'], batch['edge_index'])


def accumulate_supervised_gradients(model, batch_iter, global_indices, *,
                                    model_kind='gnn', feature_transform=None,
                                    progress=None):
    """One full-data mean-loss backward, without stepping or clearing gradients.

    Caller zeroes gradients once and steps once. This helper also supports CPU
    synthetic equivalence tests, but public training functions require CUDA.
    """
    indices = sorted_indices(global_indices)
    total_loss, seen, batch_count = 0., 0, 0
    for batch in batch_iter:
        local = local_indices(indices, batch)
        if feature_transform is not None:
            batch = feature_transform(batch)
        if not len(local):
            continue
        if (not bool(batch['eligible'][local].all())
                or not bool(((batch['y'][local] == 0) | (batch['y'][local] == 1)).all())):
            raise ValueError('Classifier supervision includes ineligible or unknown labels')
        output = _forward(model, batch, model_kind)
        loss_fn = F.cross_entropy if model_kind == 'mlp' else F.nll_loss
        loss = loss_fn(output[local], batch['y'][local], reduction='sum') / len(indices)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite supervised batch loss')
        loss.backward()
        total_loss += float(loss.detach())
        seen += len(local)
        batch_count += 1
        if progress and (batch_count == 1 or batch_count % 16 == 0):
            progress(dict(batches_done=batch_count, supervision_seen=seen,
                supervision_total=len(indices), partial_global_mean_loss=total_loss))
    if seen != len(indices):
        raise ValueError(f'Full-data supervision coverage mismatch: {seen} != {len(indices)}')
    return dict(train_loss=total_loss, supervision_seen=seen, batches=batch_count)


@torch.no_grad()
def _dev_score(model, data_dir, *, model_kind, max_nodes, device, checkpoint_rule):
    truth, scores, predictions = [], [], []
    model.eval()
    for batch in batches(data_dir, 'dev', max_nodes, device):
        idx = torch.nonzero(batch['eligible'], as_tuple=False).flatten()
        if not len(idx):
            continue
        output = _forward(model, batch, model_kind)[idx]
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Nonfinite clean DEV prediction')
        truth.append(batch['y'][idx].cpu())
        predictions.append(output.argmax(1).cpu())
        scores.append((output.softmax(1) if model_kind == 'mlp' else output.exp())[:, 1].cpu())
    if not truth:
        raise ValueError('No eligible DEV pairs')
    truth = torch.cat(truth).numpy()
    if set(truth.tolist()) != {0, 1}:
        raise ValueError('DEV must contain both true classes')
    if checkpoint_rule == 'dev_accuracy':
        return float(np.mean(torch.cat(predictions).numpy() == truth))
    return float(roc_auc_score(truth, torch.cat(scores).numpy()))


def _finite_checkpoint_tree(value):
    """Reject corrupted parameters/Adam moments without touching RNG tensors."""
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_checkpoint_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_checkpoint_tree(v) for v in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def _fit_recovery(source, *, seed, role, model_kind, epochs, max_nodes,
                  checkpoint_rule, device, data_hash, index_hash, optimizer, initial):
    """Validate an earlier attempt of this same registered seed, never a new seed.

    The enclosing recovery plan binds the old and new source hashes. This
    helper therefore records, rather than requires equality of, implementation
    hashes: adding resume support itself changes this module's hash.
    """
    source = Path(source).resolve()
    checkpoint_path = source / 'latest.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    expected = dict(seed=seed, role=role, model_kind=model_kind, epochs=epochs,
                    max_nodes=max_nodes, checkpoint_rule=checkpoint_rule,
                    device=str(device), execution=EXECUTION)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('Resume checkpoint contract differs: ' + key)
    for key, value in (('data_summary_sha256', data_hash), ('train_indices_sha256', index_hash)):
        if checkpoint.get(key) != value:
            raise ValueError('Resume checkpoint population differs: ' + key)
    next_epoch = meta.get('next_epoch')
    if (type(next_epoch) is not int or not 1 <= next_epoch <= epochs
            or meta.get('epoch') != next_epoch-1 or meta.get('optimizer_steps') != next_epoch):
        raise ValueError('Invalid cumulative resume epoch/optimizer budget')
    if not _finite_checkpoint_tree((checkpoint['model_state'], checkpoint['best_state'],
                                    checkpoint['optimizer_state'])):
        raise FloatingPointError('Nonfinite model or optimizer checkpoint')
    old_groups = checkpoint['optimizer_state']['param_groups']
    new_groups = optimizer.state_dict()['param_groups']
    if len(old_groups) != len(new_groups):
        raise ValueError('Resume optimizer parameter groups differ')
    for old, new in zip(old_groups, new_groups):
        if old != new:
            raise ValueError('Resume Adam hyperparameters or parameter order differ')
    for state in checkpoint['optimizer_state']['state'].values():
        step = state.get('step')
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != next_epoch:
            raise ValueError('Resume Adam step disagrees with cumulative epoch budget')
    selected_epoch, best = checkpoint['selected_epoch'], checkpoint['best_state']
    if checkpoint_rule == 'final':
        if selected_epoch != next_epoch-1 or best is not None:
            raise ValueError('Invalid final-epoch checkpoint selection state')
    elif (type(selected_epoch) is not int or not 0 <= selected_epoch < next_epoch
          or not isinstance(best, dict) or not best
          or not np.isfinite(checkpoint['best_dev_score'])):
        raise ValueError('Invalid best-checkpoint selection state')
    # Checkpoint replacement occurs before the epoch log append. A termination
    # between those writes can lose exactly one log row, not an optimizer step.
    log_path = source / 'epochs.jsonl'
    log_bytes = log_path.read_bytes()
    prefix = []
    lines = log_bytes.decode().splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines)-1 and i == next_epoch-1:
                break
            raise ValueError('Malformed resume epoch log') from None
        if (row.get('epoch') != i or row.get('optimizer_steps') != i+1
                or row.get('epochs') != epochs or row.get('role') != role):
            raise ValueError('Resume epoch log disagrees with registered schedule')
        prefix.append(row)
    if len(prefix) not in (next_epoch-1, next_epoch):
        raise ValueError('Resume log/checkpoint epoch coverage mismatch')
    missing_metrics = len(prefix) == next_epoch-1
    if missing_metrics:
        prefix.append(dict(stage=role, role=role, epoch=next_epoch-1, epochs=epochs,
            optimizer_steps=next_epoch, recovered_from_atomic_checkpoint=True,
            epoch_metrics_unavailable=True))
    files = dict(latest_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                 epochs_sha256=hashlib.sha256(log_bytes).hexdigest())
    original_initial = checkpoint.get('initial_state_sha256', initial)
    summary_path, model_path = source/'summary.json', source/'model.pt'
    if summary_path.exists() or model_path.exists():
        if not summary_path.exists() or not model_path.exists() or next_epoch != epochs:
            raise ValueError('Incomplete or contradictory finished fit receipt')
        original = json.loads(summary_path.read_text())
        selected = torch.load(model_path, map_location='cpu', weights_only=True)
        for key, value in dict(seed=seed, role=role, model_kind=model_kind, epochs=epochs,
                optimizer_steps=epochs, selected_epoch=selected_epoch,
                train_indices_sha256=index_hash, data_summary_sha256=data_hash).items():
            if original.get(key) != value or selected['metadata'].get(key) != value:
                raise ValueError('Completed fit receipt mismatch: ' + key)
        expected_state = checkpoint['model_state'] if checkpoint_rule == 'final' else best
        if (selected['state_dict'].keys() != expected_state.keys()
                or any(not torch.equal(selected['state_dict'][k], v) for k, v in expected_state.items())):
            raise ValueError('Saved selected model disagrees with latest checkpoint')
        original_initial = original['initial_state_sha256']
        files.update(summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                     model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
    if original_initial != initial:
        raise ValueError('Resume initialization identity differs from this seed')
    rng = checkpoint['rng_numpy']
    if (not isinstance(checkpoint['rng_cpu'], torch.Tensor)
            or not isinstance(checkpoint['rng_cuda'], torch.Tensor)
            or not isinstance(rng['state'], torch.Tensor)):
        raise ValueError('Missing resume RNG state')
    receipt = dict(source_directory=str(source), **files, restored_optimizer_steps=next_epoch,
        remaining_optimizer_steps=epochs-next_epoch, same_registered_seed=True,
        source_implementation_sha256=checkpoint['implementation_sha256'],
        reconstructed_missing_log_row=missing_metrics,
        cumulative_budget=epochs, historical_attack_checkpoint=False)
    return checkpoint, prefix, receipt


def fit_model(model, data_dir, idx_train, *, seed, output_dir, role, model_kind='gnn',
              epochs=200, max_nodes=16384, device='cuda:0', checkpoint_rule='dev_auc',
              lr=.01, weight_decay=.0005, feature_transform=None, progress=None,
              resume_from=None):
    """Preserve Adam/loss/selection rule while accumulating whole-role gradients."""
    device = _cuda_device(device)
    if checkpoint_rule not in ('dev_auc', 'dev_accuracy', 'final'):
        raise ValueError('Unsupported checkpoint selection rule')
    if type(epochs) is not int or epochs < 1 or type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('Positive fixed epochs and batch node budget required')
    model = model.to(device)
    if any(p.device != device for p in model.parameters()):
        raise ValueError('Model/input device mismatch before optimizer creation')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    indices = sorted_indices(idx_train)
    data_hash = hashlib.sha256((Path(data_dir)/'summary.json').read_bytes()).hexdigest()
    index_hash = _tensor_hash(indices)
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    initial = _state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_score = 0. if checkpoint_rule == 'dev_accuracy' else float('-inf')
    best_state, selected_epoch = None, None
    start_epoch, prefix, recovery = 0, [], None
    if resume_from is not None:
        checkpoint, prefix, recovery = _fit_recovery(resume_from, seed=seed, role=role,
            model_kind=model_kind, epochs=epochs, max_nodes=max_nodes,
            checkpoint_rule=checkpoint_rule, device=device, data_hash=data_hash,
            index_hash=index_hash, optimizer=optimizer, initial=initial)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_state, selected_epoch = checkpoint['best_state'], checkpoint['selected_epoch']
        best_score = checkpoint['best_dev_score'] if checkpoint_rule != 'final' else float('-inf')
        start_epoch = checkpoint['metadata']['next_epoch']
        _checkpoint(folder/'latest.pt', checkpoint)
        _write(folder/'recovery.json', recovery)
        if progress:
            progress(dict(stage=role, role=role, recovery=True, epoch=start_epoch,
                          epochs=epochs, optimizer_steps=start_epoch,
                          remaining_optimizer_steps=epochs-start_epoch))
    started = time.monotonic()
    with _fp32(), (folder/'epochs.jsonl').open('x') as log:
        for row in prefix:
            log.write(json.dumps(row, allow_nan=False)+'\n')
        log.flush()
        if recovery is not None:
            # Restore after all model/optimizer construction, provenance writes
            # and callbacks, immediately before the next stochastic forward.
            torch.set_rng_state(checkpoint['rng_cpu'])
            torch.cuda.set_rng_state(checkpoint['rng_cuda'], device)
            rng = checkpoint['rng_numpy']
            np.random.set_state((rng['bit_generator'], rng['state'].numpy(),
                                rng['position'], rng['has_gauss'], rng['cached_gaussian']))
        for epoch in range(start_epoch, epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            row = accumulate_supervised_gradients(model, batches(data_dir, 'train', max_nodes, device),
                indices, model_kind=model_kind, feature_transform=feature_transform,
                progress=(lambda event: progress(dict(stage=role, role=role, epoch=epoch,
                    epochs=epochs, optimizer_steps=epoch, **event))) if progress else None)
            _finite_gradients(model.parameters(), role)
            optimizer.step()
            _finite_module(model, role)
            row.update(stage=role, role=role, epoch=epoch, epochs=epochs,
                       optimizer_steps=epoch+1, elapsed_seconds=time.monotonic()-started)
            if checkpoint_rule != 'final':
                score = _dev_score(model, data_dir, model_kind=model_kind, max_nodes=max_nodes,
                                   device=device, checkpoint_rule=checkpoint_rule)
                row[checkpoint_rule] = score
                if score > best_score:
                    best_score, best_state, selected_epoch = score, _cpu_state(model), epoch
            else:
                selected_epoch = epoch
            numpy_rng = np.random.get_state()
            _checkpoint(folder/'latest.pt', dict(
                model_state=_cpu_state(model), optimizer_state=optimizer.state_dict(),
                best_state=best_state, best_dev_score=(best_score if checkpoint_rule != 'final' else None),
                selected_epoch=selected_epoch, rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state(device),
                rng_numpy=dict(bit_generator=numpy_rng[0], state=torch.from_numpy(numpy_rng[1].copy()),
                    position=numpy_rng[2], has_gauss=numpy_rng[3], cached_gaussian=numpy_rng[4]),
                data_summary_sha256=data_hash, train_indices_sha256=index_hash,
                implementation_sha256=implementation_hash, initial_state_sha256=initial,
                metadata=dict(seed=seed, role=role, model_kind=model_kind,
                    epoch=epoch, next_epoch=epoch+1, epochs=epochs, optimizer_steps=epoch+1,
                    max_nodes=max_nodes, checkpoint_rule=checkpoint_rule, device=str(device),
                    execution=EXECUTION, automatic_resume=False)))
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            if progress:
                progress(row)
    final_hash = _state_hash(model)
    if checkpoint_rule != 'final':
        if best_state is None:
            raise RuntimeError('Author strict DEV accuracy rule selected no checkpoint')
        model.load_state_dict(best_state)
    _freeze(model)
    summary = dict(role=role, seed=seed, epochs=epochs, optimizer_steps=epochs,
        selected_epoch=selected_epoch, selection_policy=('final_epoch' if checkpoint_rule == 'final'
            else 'earliest_strict_maximum_clean_'+checkpoint_rule),
        best_dev_score=best_score if checkpoint_rule != 'final' else None,
        initial_state_sha256=initial, final_epoch_state_sha256=final_hash,
        selected_state_sha256=_state_hash(model), fresh_initialization=True,
        historical_checkpoint_loaded=False, train_supervision_count=len(indices),
        train_indices_sha256=index_hash, data_summary_sha256=data_hash,
        implementation_sha256=implementation_hash, device=str(device), dtype='float32',
        amp=False, tf32=False, model_kind=model_kind,
        optimizer=dict(name='Adam', lr=lr, weight_decay=weight_decay),
        execution=dict(EXECUTION, max_nodes=max_nodes))
    from .av2_formal_data import dataset_identity
    dataset, source_split = dataset_identity(data_dir)
    summary.update(dataset=dataset, source_dataset_split=source_split)
    summary.update(optimizer_steps_this_attempt=epochs-start_epoch,
                   restored_optimizer_steps=start_epoch, recovery=recovery)
    torch.save(dict(state_dict=_cpu_state(model), metadata=summary), folder/'model.pt')
    _write(folder/'summary.json', summary)
    return dict(model=model, summary=summary)


def fit_classifier(data_dir, idx_train, *, seed, output_dir, role, epochs=200,
                   max_nodes=16384, device='cuda:0', checkpoint_rule='dev_auc',
                   progress=None, feature_transform=None):
    if role not in ('clean_proxy_1', 'clean_proxy_2', 'clean_reference', 'victim'):
        raise ValueError('Unknown classifier role')
    device = _cuda_device(device)
    torch.manual_seed(seed)
    model = GraphSage(784, 80, 2, dropout=.5, lr=.01, weight_decay=.0005, device=device).to(device)
    configure_message_passing(model)
    return fit_model(model, data_dir, idx_train, seed=seed, output_dir=output_dir, role=role,
        epochs=epochs, max_nodes=max_nodes, device=device, checkpoint_rule=checkpoint_rule,
        progress=progress, feature_transform=feature_transform)


def fit_victim(generator, data_dir, indices, *, seed, output_dir, epochs=200,
               max_nodes=16384, device='cuda:0', progress=None):
    device = _cuda_device(device)
    attach = sorted_indices(indices['idx_attach'])
    clean = sorted_indices(indices['idx_train'])
    if bool(torch.isin(attach, clean).any()):
        raise ValueError('Victim clean supervision and attachments overlap')
    before = _state_hash(generator)
    generator.to(device).eval()

    def poison(batch):
        local = local_indices(attach, batch)
        if not len(local):
            return batch
        if not bool(batch['eligible'][local].all()) or not bool((batch['y'][local] == 0).all()):
            raise ValueError('Victim attachments must remain eligible true negatives')
        with torch.no_grad(), _fp32():
            changed, _, _ = _inject(generator, batch['x'], batch['edge_index'], local,
                batch['x'].new_ones(batch['edge_index'].shape[1]), 24)
        result = dict(batch, x=changed, y=batch['y'].clone())
        result['y'][local] = 1
        return result

    result = fit_classifier(data_dir, torch.cat((clean, attach)), seed=seed, output_dir=output_dir,
        role='victim', epochs=epochs, max_nodes=max_nodes, device=device,
        progress=progress, feature_transform=poison)
    if _state_hash(generator) != before:
        raise RuntimeError('Victim training changed the frozen generator')
    return result


def evaluate(generator, victim, clean_reference, data_dir, *, output_dir, role,
             device='cuda:0', progress=None):
    from .av2_formal_data import dataset_identity, iter_role_parts
    from .womd_pilot_training import evaluate as evaluate_kernel
    if role not in ('dev', 'audit'):
        raise ValueError('Unknown registered evaluation role')
    dataset, source_split = dataset_identity(data_dir)
    parts = list(iter_role_parts(data_dir, role))
    summary = evaluate_kernel(generator, victim, clean_reference, parts,
        [p['target_index'] for p in parts], output_dir=output_dir, role=role,
        device=device, progress=progress)
    scope = ('AV2 original TRAIN grouped '+role+
        '; previously accessed development/confirmation groups, not a fresh blind test'
        if dataset == 'AV2' else 'WOMD training_20s internal '+role+
        '; previously used development pilot, not official test; recording independence unverified')
    summary.update(dataset=dataset, source_dataset_split=source_split, scope=scope,
        frozen_v6_test_loaded=False, test_arrays_loaded=False)
    if dataset == 'WOMD':
        summary['recording_independence_proven'] = False
    # The reused metric kernel initially stamps its original dataset name;
    # this newly produced record replaces that descriptive stamp only.
    (Path(output_dir)/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary
