"""Registered clean-only WOMD classifier controls; no triggers or ASR evaluation.

A/B share data, initialization, order, epochs and checkpoint rules. B changes
only the supervised loss to effective-number class balancing (Cui, CVPR 2019).
C adds nested TRAIN coverage with the original normalizer and ordinary NLL.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .av2_formal_data import iter_role_batches, load_role_arrays
from .models import GraphSage
from .womd_execution import configure_message_passing


SOURCE_RUN = Path('/home/mizhou/JCAS/results/womd_av2_aligned_pilot_20261106_v1')
GROUPS = ('both_exact_zero', 'both_low_not_both_zero', 'one_low', 'both_moving',
          'invalid_terminal')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def status_record(status, started, completed, **details):
    details = dict(details)
    if 'elapsed_seconds' in details:
        details['arm_elapsed_seconds'] = details.pop('elapsed_seconds')
    return dict(status=status, at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                elapsed_seconds=time.monotonic()-started, completed=completed, total=3, **details)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def class_weights(counts, beta=None):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if (counts.shape != (2,) or not bool(torch.isfinite(counts).all())
            or not bool((counts > 0).all()) or not bool((counts == counts.floor()).all())):
        raise ValueError('Both clean supervised classes must be present')
    if beta is None:
        return torch.ones(2, dtype=torch.float32)
    if not math.isfinite(beta) or not 0 < beta < 1:
        raise ValueError('beta must be between zero and one')
    weights = (1 - beta) / -torch.expm1(counts * math.log(beta))
    return (weights / weights.mean()).float()


def global_weighted_loss(log_probs, labels, weights, denominator):
    """Batch weighted sum / fixed complete-supervision weight sum."""
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError('Invalid global loss denominator')
    return F.nll_loss(log_probs, labels, weight=weights, reduction='sum') / denominator


def normal_metrics(y, p, pred):
    y, p, pred = np.asarray(y), np.asarray(p), np.asarray(pred)
    if (not (len(y) == len(p) == len(pred)) or not np.isfinite(p).all()
            or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError('Invalid normal prediction arrays')
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    tp, fp = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum())
    return dict(pairs=len(y), positives=positive, negatives=negative, true_positive=tp,
        false_positive=fp, false_negative=positive-tp, true_negative=negative-fp,
        auc=float(roc_auc_score(y, p)) if positive and negative else None,
        ap=float(average_precision_score(y, p)) if positive else None,
        recall=tp/positive if positive else None, fpr=fp/negative if negative else None,
        brier=float(np.mean((p-y)**2)) if len(y) else None)


def make_model(seed, device):
    torch.manual_seed(seed)
    model = GraphSage(784, 80, 2, dropout=.5, lr=.01, weight_decay=.0005,
                      device=device).to(device)
    configure_message_passing(model)
    return model


@torch.no_grad()
def predictions(model, data_dir, role, max_nodes, device):
    model.eval()
    scores, decisions = [], []
    for batch in iter_role_batches(data_dir, role, max_nodes=max_nodes, device=device):
        output = model(batch['x'], batch['edge_index'])
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Nonfinite normal prediction')
        scores.append(output.exp()[:, 1].cpu())
        decisions.append(output.argmax(1).cpu())
    return torch.cat(scores).numpy(), torch.cat(decisions).numpy()


def evaluate(model, data_dir, role, max_nodes, device, folder):
    from .womd_clean_control_data import fine_motion_groups
    pop = load_role_arrays(data_dir, role)
    p, pred = predictions(model, data_dir, role, max_nodes, device)
    y, eligible = pop['y'].numpy(), pop['eligible'].numpy()
    detail = np.asarray(fine_motion_groups(data_dir, role))
    if len(detail) != len(y) or len(y) != len(p):
        raise ValueError('Evaluation identity alignment mismatch')
    masks = {'all': eligible, 'slow': eligible & (detail != 3),
             'moving': eligible & (detail == 3)}
    masks.update({name: eligible & (detail == i) for i, name in enumerate(GROUPS)})
    result = {name: normal_metrics(y[m], p[m], pred[m]) for name, m in masks.items()}
    with (folder / f'{role}_normal_scores.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['global_pair_index', 'label', 'group', 'risk_probability', 'prediction'])
        for i in np.flatnonzero(eligible):
            writer.writerow([int(i), int(y[i]), GROUPS[int(detail[i])], float(p[i]), int(pred[i])])
    write_json(folder / f'{role}_normal_metrics.json', result)
    return result


def fit_arm(plan, arm, emit):
    device = torch.device(plan['device'])
    data_dir, folder = Path(arm['data_dir']), Path(plan['output_dir']) / arm['name']
    folder.mkdir(exist_ok=False)
    pop = load_role_arrays(data_dir, 'train')
    indices = torch.load(arm['indices_path'], map_location='cpu', weights_only=True)['idx_train'].long().sort().values
    if (indices.ndim != 1 or not len(indices) or len(indices.unique()) != len(indices)
            or int(indices[0]) < 0 or int(indices[-1]) >= len(pop['y'])
            or not bool(pop['eligible'][indices].all())):
        raise ValueError('Invalid clean supervision set')
    counts = torch.bincount(pop['y'][indices], minlength=2)
    weights_cpu = class_weights(counts, arm['beta'])
    denominator = float((weights_cpu.double() * counts).sum())
    weights = weights_cpu.to(device)
    model = make_model(plan['model_seed'], device)
    initial_hash = state_hash(model)
    if initial_hash != plan['expected_initial_state_sha256']:
        raise ValueError('Initialization differs from paired registered seed')
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    dev = load_role_arrays(data_dir, 'dev')
    dev_mask = dev['eligible'].numpy()
    best, selected, started = -math.inf, None, time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    write_json(folder/'registration.json', dict(arm=arm, initial_state_sha256=initial_hash,
        clean_indices_sha256=tensor_hash(indices), class_counts=counts.tolist(),
        class_weights=weights_cpu.tolist(), global_weight_denominator=denominator,
        selection='earliest maximum clean DEV AUC', input='unmodified clean inputs only'))
    with (folder/'epochs.jsonl').open('x') as log:
        for epoch in range(plan['epochs']):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            seen, total, batch_count = 0, 0., 0
            for batch in iter_role_batches(data_dir, 'train', max_nodes=plan['max_nodes'], device=device):
                bounds = torch.searchsorted(indices, torch.tensor([batch['global_start'], batch['global_stop']]))
                local = (indices[int(bounds[0]):int(bounds[1])] - batch['global_start']).to(device)
                if not len(local):
                    continue
                if not bool(batch['eligible'][local].all()):
                    raise ValueError('Ineligible clean supervision')
                output = model(batch['x'], batch['edge_index'])
                loss = global_weighted_loss(output[local], batch['y'][local], weights, denominator)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite clean loss')
                loss.backward()
                seen += len(local)
                total += float(loss.detach())
                batch_count += 1
                if batch_count == 1 or batch_count % 8 == 0:
                    emit('TRAINING', arm=arm['name'], epoch=epoch+1, epochs=plan['epochs'],
                         batch=batch_count, supervision_seen=seen, supervision_total=len(indices))
            if seen != len(indices):
                raise ValueError('Incomplete clean epoch supervision coverage')
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('Invalid clean classifier gradients')
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('Nonfinite clean classifier parameters')
            score, _ = predictions(model, data_dir, 'dev', plan['max_nodes'], device)
            auc = float(roc_auc_score(dev['y'].numpy()[dev_mask], score[dev_mask]))
            if auc > best:
                best, selected = auc, epoch
                torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                epoch=epoch, dev_auc=auc), folder/'selected.tmp')
                (folder/'selected.tmp').replace(folder/'model.pt')
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer_state=optimizer.state_dict(), next_epoch=epoch+1,
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device),
                automatic_resume=False), folder/'latest.tmp')
            (folder/'latest.tmp').replace(folder/'latest.pt')
            row = dict(epoch=epoch+1, optimizer_steps=epoch+1, train_loss=total,
                       dev_auc=auc, batches=batch_count, elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            emit('TRAINING', arm=arm['name'], **row)
    selected_state = torch.load(folder/'model.pt', map_location=device, weights_only=True)
    model.load_state_dict(selected_state['state_dict'])
    result = dict(status='TRAINED_PENDING_EVALUATION', arm=arm['name'], epochs=plan['epochs'],
        selected_epoch=selected+1, best_dev_auc=best, initial_state_sha256=initial_hash,
        selected_state_sha256=state_hash(model), clean_indices_sha256=tensor_hash(indices),
        train_scenes=arm['train_scenes'], class_counts=counts.tolist(),
        elapsed_seconds=time.monotonic()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        clean_only=True, attack_training=False, asr_evaluated=False)
    write_json(folder/'training_summary.json', result)
    del model, optimizer, selected_state
    torch.cuda.empty_cache()
    return result


def preflight(plan):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU training fallback')
    device = torch.device(plan['device'])
    free, total = torch.cuda.mem_get_info(device)
    if free < 1024**3:
        raise RuntimeError('Less than 1 GiB free CUDA memory for measured preflight; no training launched')
    model = make_model(plan['model_seed'], device)
    if state_hash(model) != plan['expected_initial_state_sha256']:
        raise ValueError('Registered initialization mismatch')
    trials = []
    for arm in (plan['arms'][0], plan['arms'][-1]):
        largest = None
        for cpu_batch in iter_role_batches(arm['data_dir'], 'train',
                                          max_nodes=plan['max_nodes'], device='cpu'):
            if largest is None or cpu_batch['edge_index'].shape[1] > largest['edge_index'].shape[1]:
                largest = cpu_batch
        batch = {k:largest[k].to(device) for k in ('x','edge_index','y','eligible')}
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        mask = batch['eligible']
        loss = F.nll_loss(model(batch['x'], batch['edge_index'])[mask], batch['y'][mask])
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError('CUDA preflight gradient failure')
        trials.append(dict(arm=arm['name'], batch_nodes=len(batch['x']),
            batch_edges=batch['edge_index'].shape[1], loss=float(loss.detach()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
        del batch, largest, cpu_batch, loss, mask
    required = int(max(t['peak_allocated_bytes'] for t in trials) * 1.5 + 512 * 1024**2)
    if free < required:
        raise RuntimeError(f'Measured peak plus reserve exceeds initial CUDA free memory: {required} > {free}')
    result = dict(status='PASSED', device=str(device), device_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, free_bytes=free, total_bytes=total,
        largest_train_batch_trials=trials, required_headroom_bytes=required,
        finite_gradients=True, optimizer_steps=0)
    del model
    torch.cuda.empty_cache()
    return result


def wait_for_preflight(plan, emit):
    """Wait only before training for shared-device memory; never retry a fit."""
    while True:
        try:
            free, _ = torch.cuda.mem_get_info(torch.device(plan['device']))
            if free < 2 * 1024**3:
                emit('WAITING_FOR_GPU_MEMORY', free_cuda_bytes=free,
                     required_free_cuda_bytes=2*1024**3, training_started=False)
            else:
                return preflight(plan)
        except RuntimeError as exc:
            message = str(exc)
            if ('out of memory' not in message.lower()
                    and not message.startswith('Less than ')
                    and not message.startswith('Measured peak plus reserve')):
                raise
            emit('WAITING_FOR_GPU_MEMORY', error_type=type(exc).__name__, error=message,
                 training_started=False)
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        time.sleep(30)


def register(output):
    from .womd_clean_control_data import prepare
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = SOURCE_RUN/'data'
    draft = dict(created_at=datetime.now(timezone.utc).isoformat(), clean_only=True,
        source_run=str(SOURCE_RUN), data_seed=20261104, model_seed=20364104,
        epochs=200, max_nodes=16384, beta=.9999, expanded_train_scenes=2048,
        selection='earliest maximum clean DEV AUC; fixed argmax, ties class0',
        arms=['A: TRAIN512 ordinary NLL', 'B: same TRAIN512 effective-number weighted NLL',
              'C: nested TRAIN2048 ordinary NLL, original normalizer'],
        source_reference='https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html',
        evaluation_scope='previously exposed internal DEV/AUDIT, not blind or official test',
        purpose='normal risk recognition and data coverage; no trigger generation, poisoned training or ASR')
    write_json(output/'preregistration.json', draft)
    prepare(source, output/'expanded_data', total_scenes=2048, seed=20261104, max_nodes=16384)
    snapshot = output/'source_snapshot'/'spear_current'
    snapshot.mkdir(parents=True)
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p, snapshot/p.name)
    initial = json.loads((SOURCE_RUN/'seed20261104/clean_reference/summary.json').read_text())['initial_state_sha256']
    arms = [dict(name='A_ce_512',data_dir=str(source),indices_path=str(source/'indices.pt'),beta=None,train_scenes=512),
            dict(name='B_cb_512',data_dir=str(source),indices_path=str(source/'indices.pt'),beta=.9999,train_scenes=512),
            dict(name='C_ce_2048',data_dir=str(output/'expanded_data'),
                 indices_path=str(output/'expanded_data/clean_indices.pt'),beta=None,train_scenes=2048)]
    assets = {}
    for arm in arms:
        for p in (Path(arm['data_dir'])/'summary.json', Path(arm['data_dir'])/'normalizer.npz', Path(arm['indices_path'])):
            assets[str(p)] = sha256(p)
    plan = dict(draft)
    plan.update(output_dir=str(output), device='cuda:0', dtype='float32', amp=False,tf32=False,
        expected_initial_state_sha256=initial, arms=arms, assets=assets,
        source_sha256={str(p.relative_to(snapshot.parent)):sha256(p) for p in snapshot.glob('*.py')},
        snapshot=str(snapshot.parent), automatic_retry=False, run_order='A then B then C',
        protected_jobs='do not stop, restart, or alter existing AV2 or other users jobs')
    write_json(output/'plan.json', plan)
    write_json(output/'status.json', dict(status='REGISTERED_NOT_STARTED', completed=0,total=3))
    return output/'plan.json'


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text()); output = Path(plan['output_dir'])
    started, completed = time.monotonic(), []
    def emit(status, **kw):
        value = status_record(status, started, len(completed), **kw)
        write_json(output/'status.json', value)
        print('[womd-clean] '+json.dumps(value,allow_nan=False),flush=True)
    try:
        emit('VERIFYING')
        for name, digest in plan['assets'].items():
            if sha256(name) != digest: raise ValueError('Changed registered clean asset: '+name)
        for name, digest in plan['source_sha256'].items():
            if sha256(Path(plan['snapshot'])/name) != digest: raise ValueError('Changed source snapshot')
        if Path(__file__).resolve() != Path(plan['snapshot'])/'spear_current/womd_clean_controls.py':
            raise ValueError('Run must use frozen source snapshot')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        write_json(output/'preflight.json', wait_for_preflight(plan, emit)
                   if plan.get('wait_for_gpu_memory_before_training', False) else preflight(plan))
        for arm in plan['arms']:
            emit('STARTING_ARM',arm=arm['name'])
            with torch.autocast(device_type='cuda',enabled=False):
                completed.append(fit_arm(plan,arm,emit))
            write_json(output/'aggregate.json',dict(status='IN_PROGRESS',results=completed))
        for arm, result in zip(plan['arms'], completed):
            emit('EVALUATING_CLEAN',arm=arm['name'])
            device = torch.device(plan['device'])
            model = make_model(plan['model_seed'], device)
            checkpoint = torch.load(output/arm['name']/'model.pt', map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            with torch.autocast(device_type='cuda',enabled=False):
                result['metrics'] = {r:evaluate(model, arm['data_dir'], r, plan['max_nodes'], device,
                                                output/arm['name']) for r in ('dev','audit')}
            result['status'] = 'COMPLETE'
            write_json(output/arm['name']/'summary.json', result)
            del model, checkpoint
            torch.cuda.empty_cache()
        write_json(output/'aggregate.json',dict(status='COMPLETE',results=completed,
            clean_only=True,asr_evaluated=False,plan_sha256=sha256(plan_path)))
        emit('COMPLETE')
    except BaseException as exc:
        emit('FAILED',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register',type=Path)
    group.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.register: print(register(args.register),flush=True)
    else: run(args.run)


if __name__=='__main__':
    main()
