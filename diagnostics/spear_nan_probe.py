"""Bounded numerical diagnosis of a failed native SPEAR case, not a new result.

Reconstructs the original selector/MLP RNG prefix, reuses hash-bound SAGE
importance (eval-only and private NumPy RNG), then observes unchanged author
fit. Stops on the first nonfinite module output, gradient, parameter, optimizer
state, or loss. No repair, new hyperparameters, victim fit or ASR evaluation.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from author_comparisons import spear
from author_comparisons.data import load_inputs
from author_comparisons.runner import validate_plan, configure_gpu
from comparison_methods.evidence import write_json
from spear_current.runtime import load_bundle, sha256


class NonfiniteObserved(RuntimeError):
    pass


def tensor_stats(value):
    value = value.detach()
    finite = torch.isfinite(value)
    all_finite = bool(finite.all())
    bounded = value[finite]
    return dict(shape=list(value.shape), finite=all_finite,
                nan=int(torch.isnan(value).sum()), inf=int(torch.isinf(value).sum()),
                max_abs=float(bounded.abs().max()) if bounded.numel() else None)


class Observer:
    def __init__(self, stream, source):
        self.stream, self.source = stream, str(source)
        self.epoch, self.line = -1, None
        self.recent = deque(maxlen=20)
        self.handles, self.modules, self.optimizers = [], {}, {}
        self.first_failure = None
        self.seen_losses = {}

    def check(self, name, value):
        if not torch.is_tensor(value) or not (value.is_floating_point() or value.is_complex()):
            return
        row = dict(epoch=self.epoch, author_line=self.line, field=name, **tensor_stats(value))
        self.recent.append(row)
        if not row['finite']:
            self.first_failure = dict(observed=row, preceding_observations=list(self.recent))
            self.stream.write(json.dumps(self.first_failure) + '\n')
            self.stream.flush()
            raise NonfiniteObserved(name)
        if self.epoch % 10 == 0 or name.startswith('loss.'):
            self.stream.write(json.dumps(row) + '\n')

    def install(self, backdoor):
        for role in ('shadow_model', 'trojan'):
            model = getattr(backdoor, role)
            self.modules[role] = model
            for name, module in model.named_modules():
                label = role + '.' + (name or '<root>')
                def observe(_module, inputs, output, label=label):
                    # Record input as well as output, without mutating either.
                    for i, tensor in enumerate(inputs):
                        self.check(label + '.input' + str(i), tensor)
                    if isinstance(output, (tuple, list)):
                        for i, tensor in enumerate(output):
                            self.check(label + '.output' + str(i), tensor)
                    else:
                        self.check(label + '.output', output)
                self.handles.append(module.register_forward_hook(observe))
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    def gradient(value, label=role + '.' + name):
                        self.check(label + '.gradient', value)
                    self.handles.append(parameter.register_hook(gradient))

    def trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.source or frame.f_code.co_name != 'fit':
            return None
        if event == 'line':
            values = frame.f_locals
            self.epoch, self.line = int(values.get('i', -1)), frame.f_lineno
            if not self.modules and 'optimizer_trigger' in values:
                self.install(values['self'])
                self.optimizers = {id(values['optimizer_shadow']): 'shadow',
                                   id(values['optimizer_trigger']): 'trigger'}
            for name in ('loss_inner', 'loss_sim', 'loss_target', 'loss_outter'):
                value = values.get(name)
                if torch.is_tensor(value) and self.seen_losses.get(name) is not value:
                    self.seen_losses[name] = value
                    self.check('loss.' + name, value)
        return self.trace

    def step_wrapper(self, original):
        def observed_step(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            role = self.optimizers.get(id(optimizer), 'unknown')
            for index, parameter in enumerate(p for group in optimizer.param_groups for p in group['params']):
                self.check(role + '.post_step.parameter' + str(index), parameter)
                for key, value in optimizer.state.get(parameter, {}).items():
                    self.check(role + '.optimizer' + str(index) + '.' + str(key), value)
            return result
        return observed_step

    def close(self):
        for handle in self.handles:
            handle.remove()


def run(plan_path, failed_case, output):
    bundle = load_bundle()
    plan = validate_plan(bundle, plan_path)
    failed_case, output = Path(failed_case).resolve(), Path(output).resolve()
    if (not output.is_relative_to(ROOT / 'results') or output == ROOT / 'results'
            or output.is_relative_to(Path(plan_path).resolve().parent)):
        raise ValueError('Use a new separate JCAS/results child')
    record_path = failed_case / 'author_work/fit_record.json'
    record = json.loads(record_path.read_text())
    feature_path = failed_case / 'author_work/feature_importance.npy'
    if sha256(feature_path) != record['feature_importance_sha256']:
        raise ValueError('Changed cached SAGE')
    if not (failed_case / 'failure.json').is_file():
        raise ValueError('Expected the recorded failed case')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', dict(kind='diagnostic_only_no_evaluation',
        original_plan=str(Path(plan_path).resolve()), original_plan_sha256=sha256(plan_path),
        failed_case=str(failed_case), fit_record_sha256=sha256(record_path),
        importance_sha256=sha256(feature_path), probe_sha256=sha256(__file__),
        scope='unchanged native fit, read-only hooks, early stop on first nonfinite value',
        epochs_max=record['config']['trojan_epochs'], seed=record['seed'],
        original_test_arrays_loaded=False, original_validation_arrays_loaded=False))
    configure_gpu(plan['protocol']['resources'])
    torch.set_num_threads(plan['protocol']['torch_threads'])
    device = torch.device('cuda:0')
    observer = None
    try:
        ledger = []
        inputs = load_inputs(bundle, ledger)
        write_json(output / 'graph_access_ledger.json', ledger)
        x, edges, y, train, val, candidates = (inputs[k] for k in
            ('x', 'edge_index', 'y', 'idx_train', 'idx_val', 'candidates'))
        originals = dict(x=x, edges=edges, y=y, train=train, dev=val, candidates=candidates)
        if spear._tensor_digest(originals) != record['input_sha256']:
            raise ValueError('Original inputs changed')
        work = output / 'author_work'
        work.mkdir()
        api, runtime = spear.import_author_runtime(work)
        (work / 'sorted_samples/Cora').mkdir(parents=True, exist_ok=False)
        args = spear._arguments(record['config'], record['seed'], str(feature_path))
        dx, de, dy, dt, dv = (v.to(device) for v in (x, edges, y, train, val))
        data = SimpleNamespace(x=dx, edge_index=de, y=dy,
            train_mask=torch.zeros(len(x), dtype=torch.bool, device=device),
            test_mask=torch.ones(len(x), dtype=torch.bool, device=device))
        data.train_mask[train] = True
        data.test_mask[train] = False
        data.test_mask[candidates] = False
        matched = {}
        with spear._rng(record['seed'], device), spear._precision(device), spear._cwd(work):
            def require_rng(name):
                digest = spear._rng_digest(device)
                if digest != record['rng_states_sha256'][name]:
                    raise ValueError('Cannot reconstruct original RNG: ' + name)
                matched[name] = digest
            require_rng('before_author_selector')
            attachments = api['select_sample'].select(data, args, dt, dv, device).detach().cpu()
            if attachments.tolist() != record['attachment_indices']:
                raise ValueError('Recomputed attachments differ')
            require_rng('before_author_feature_model')
            feature_model = api['construct'].model_construct(args, 'MLP', data, device).to(device)
            if spear._tensor_digest(feature_model.state_dict()) != record['feature_model_initial_sha256']:
                raise ValueError('Feature model initialization differs')
            feature_model.fit(dx, dy, dt, dv, train_iters=args.epochs, verbose=False)
            feature_hash = spear._tensor_digest(feature_model.state_dict())
            del feature_model
            unlabeled = torch.tensor(list(set(candidates.tolist()) - set(attachments.tolist())), dtype=torch.long)
            attack = api['backdoor'].Backdoor(args, device)
            require_rng('before_author_backdoor_fit')
            write_json(output / 'prefix_acceptance.json', dict(torch_cpu_cuda_rng_digests_matched=matched,
                attachments_exact=True, source_inputs_exact=True, cached_sage_sha256=sha256(feature_path),
                feature_model_final_hash_matches=feature_hash == record['feature_model_sha256'],
                limitation='CUDA reductions need not be bitwise deterministic; original NumPy state hash unavailable'))
            print('Prefix RNG/input/attachment checks passed; native numerical probe started', flush=True)
            with (output / 'observations.jsonl').open('x', buffering=1) as stream:
                observer = Observer(stream, api['backdoor'].__file__)
                previous_trace = sys.gettrace()
                original_step = torch.optim.Adam.step
                try:
                    sys.settrace(observer.trace)
                    with patch.object(torch.optim.Adam, 'step', observer.step_wrapper(original_step)):
                        attack.fit(dx, de, None, dy, dt, attachments.to(device), unlabeled.to(device))
                    for name, value in zip(('features', 'edges', 'weights', 'labels'), attack.get_poisoned()):
                        observer.check('final_injection.' + name, value)
                    status = 'NO_NONFINITE_OBSERVED_IN_CONTROLLED_REPLAY'
                except NonfiniteObserved:
                    status = 'FIRST_NONFINITE_LOCATED'
                finally:
                    sys.settrace(previous_trace)
                    observer.close()
        report = dict(status=status, first_nonfinite=observer.first_failure,
            changes_to_author_math=False, original_case_remains_failed=True,
            victim_trained=False, asr_evaluated=False,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(output / 'summary.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        write_json(output / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--failed-case', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.failed_case, args.output_dir)
