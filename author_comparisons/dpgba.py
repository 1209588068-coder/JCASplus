"""Thin AV2 data-interface adapter for the pinned DPGBA author release.

The attack, reconstruction model, losses, random outer sampling, and returned
checkpoint are executed from author source, not reimplemented here.  Runtime
copies only relocate optional imports; their non-import AST must be identical.
The supported comparison is explicitly WITHOUT the optional reconstruction
defense.  The upstream attack's own MLPAE reference selection remains intact.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'third_party/author_baselines/DPGBA'
SOURCE_COMMIT = 'd974ea615900e5d0590bb473da9d7dd904f8c908'
SOURCE_URL = 'https://github.com/zzwjames/DPGBA'
RUN_SUBDIR = Path('DPGBA/run')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode()
                          + value.numpy().tobytes()).hexdigest()


def _state_sha(model):
    state = model.state_dict() if hasattr(model, 'state_dict') else model
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(_tensor_sha(value).encode())
    return digest.hexdigest()


def _execution_device(value):
    """Resolve an explicit device; never silently fall back from CUDA to CPU."""
    device = torch.device(value)
    if device.type == 'cpu':
        return torch.device('cpu')
    if device.type != 'cuda':
        raise ValueError('DPGBA execution requires CPU or CUDA')
    if not torch.cuda.is_available():
        raise RuntimeError('Requested DPGBA CUDA execution, but CUDA is unavailable')
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError('Requested DPGBA CUDA device does not exist')
    return torch.device('cuda', index)


def _seed_execution(seed, device):
    # Unlike torch.manual_seed, this touches only the CPU and selected CUDA
    # streams whose states the enclosing fork_rng saves/restores.
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _rng_record(device):
    states = {'cpu': _tensor_sha(torch.get_rng_state())}
    if device.type == 'cuda':
        states[str(device)] = _tensor_sha(torch.cuda.get_rng_state(device))
    return states


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_source():
    """Read only author source/Git metadata, never data or model checkpoints."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(SOURCE_ROOT), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != SOURCE_COMMIT:
        raise ValueError('DPGBA author commit changed')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('DPGBA author checkout must remain pristine')
    paths = git('ls-files').splitlines()
    return dict(repository=SOURCE_URL, commit=SOURCE_COMMIT, directory=str(SOURCE_ROOT),
                tracked_files_sha256={p: _sha(SOURCE_ROOT / p) for p in paths},
                license='NO_LICENSE_FILE_IN_PINNED_AUTHOR_RELEASE',
                bundled_checkpoint_loaded=False)


def author_cli_defaults():
    """Extract literal parser defaults without executing run_adaptive.py."""
    tree = ast.parse((SOURCE_ROOT / RUN_SUBDIR / 'run_adaptive.py').read_text())
    result = {}
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument'):
            continue
        name = ast.literal_eval(call.args[0]).removeprefix('--').replace('-', '_')
        default = next((k.value for k in call.keywords if k.arg == 'default'), None)
        if default is not None:
            result[name] = ast.literal_eval(default)
    if not result or result.get('trojan_epochs') != 400 or result.get('hidden') != 512:
        raise ValueError('Unexpected author parser contract')
    return result


DEFAULT_CONFIG = dict(author_cli_defaults(), dataset='AV2PairGraph', target_class=1,
                      vs_number=26, defense_mode='none', no_cuda=True, test_model='GraphSage')


def validate_config(config=None):
    if config is not None and (not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG)):
        raise ValueError('Unknown DPGBA configuration')
    value = dict(DEFAULT_CONFIG, **deepcopy(config or {}))
    for key in ('epochs', 'rec_epochs', 'trojan_epochs', 'inner', 'hidden', 'k', 'vs_number',
                'trigger_size', 'outter_size'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('Expected positive integer: ' + key)
    for key in ('lr', 'train_lr', 'range'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('Expected finite positive value: ' + key)
    for key in ('weight_decay', 'weight_target', 'weight_ood', 'weight_targetclass', 'dis_weight', 'vs_ratio'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError('Expected finite nonnegative value: ' + key)
    if not 0 <= value['dropout'] < 1 or not math.isfinite(value['thrd']):
        raise ValueError('Invalid dropout or threshold')
    if (value['dataset'] != 'AV2PairGraph' or value['target_class'] != 1
            or value['defense_mode'] != 'none' or value['selection_method'] != 'none'
            or value['test_model'] != 'GraphSage' or value['use_vs_number'] is not True
            or value['no_cuda'] is not True):
        raise ValueError('Adapter supports target-1 AV2, author random selection, GraphSAGE, no defense; '
                         'execution device is a separate explicit argument')
    return json.loads(json.dumps(value, allow_nan=False))


class _WithoutImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def _algorithm_ast(text):
    return ast.dump(_WithoutImports().visit(ast.parse(text)), include_attributes=False)


def _lazy_optional_imports(text, relative):
    """Only relocate imports; no dependency stubs or altered algorithm bodies."""
    optional = {'matplotlib.pyplot', 'sklearn.decomposition', 'sklearn.manifold',
                'sklearn_extra', 'models.construct', 'faiss'}
    tree = ast.parse(text)
    relocated = []
    for node in list(tree.body):
        module = (node.module if isinstance(node, ast.ImportFrom) else
                  node.names[0].name if isinstance(node, ast.Import) and len(node.names) == 1 else None)
        if module not in optional:
            continue
        aliases = {a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name)
                   for a in node.names}
        consumers = []
        for body in tree.body:
            if not isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            used = {n.id for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            if used & aliases:
                if isinstance(body, ast.ClassDef):
                    raise ValueError('Unexpected optional dependency in author class: ' + relative)
                position = int(bool(body.body) and isinstance(body.body[0], ast.Expr)
                               and isinstance(body.body[0].value, ast.Constant)
                               and isinstance(body.body[0].value.value, str))
                body.body.insert(position, deepcopy(node))
                consumers.append(body.name)
        tree.body.remove(node)
        relocated.append(dict(module=module, consumers=consumers))
    patched = ast.unparse(ast.fix_missing_locations(tree)) + '\n' if relocated else text
    if _algorithm_ast(text) != _algorithm_ast(patched):
        raise ValueError('Runtime compatibility patch changed non-import author AST: ' + relative)
    return patched, relocated


def prepare_runtime(work_dir):
    """Create a new isolated author-source copy; preserve third_party verbatim."""
    source = verify_source()
    work = Path(work_dir).resolve()
    if work == SOURCE_ROOT or SOURCE_ROOT in work.parents or work in SOURCE_ROOT.parents:
        raise ValueError('Runtime output must not overlap the author checkout')
    work.mkdir(parents=True, exist_ok=False)
    runtime = work / 'runtime'
    runtime.mkdir()
    records = {}
    prefix = RUN_SUBDIR.as_posix() + '/'
    for filename, digest in source['tracked_files_sha256'].items():
        if not filename.startswith(prefix) or not filename.endswith('.py'):
            continue
        relative = filename[len(prefix):]
        text = (SOURCE_ROOT / filename).read_text()
        patched, imports = _lazy_optional_imports(text, relative)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            stream.write(patched)
        records[relative] = dict(source_sha256=digest, runtime_sha256=_sha(destination),
                                 import_relocations=imports, non_import_ast_identical=True)
    record = dict(source=source, runtime_root=str(runtime), files=records,
                  patches='optional imports only; no algorithm/default/checkpoint edits',
                  source_checkpoints_copied=False)
    _json(work / 'author_runtime.json', record)
    return record


def _classifier_runtime(work_dir):
    """Reuse only an intact, source-bound runtime; never trust a cache blindly."""
    work = Path(work_dir).resolve()
    if not work.exists():
        return prepare_runtime(work)
    record = json.loads((work / 'author_runtime.json').read_text())
    source = verify_source()
    runtime = work / 'runtime'
    if record['source'] != source or record['runtime_root'] != str(runtime):
        raise ValueError('Existing DPGBA runtime source binding changed')
    prefix = RUN_SUBDIR.as_posix() + '/'
    expected_files = {}
    for filename, digest in source['tracked_files_sha256'].items():
        if not filename.startswith(prefix) or not filename.endswith('.py'):
            continue
        relative = filename[len(prefix):]
        original = (SOURCE_ROOT / filename).read_text()
        expected, imports = _lazy_optional_imports(original, relative)
        destination = runtime / relative
        if not destination.is_file() or destination.read_text() != expected:
            raise ValueError('Existing DPGBA runtime code changed: ' + relative)
        expected_files[relative] = dict(source_sha256=digest, runtime_sha256=_sha(destination),
                                       import_relocations=imports, non_import_ast_identical=True)
    if (record['files'] != expected_files or
            {str(p.relative_to(runtime)) for p in runtime.rglob('*.py')} != set(expected_files)):
        raise ValueError('Existing DPGBA runtime manifest/file set changed')
    return record


@contextmanager
def _author_imports(runtime):
    """Serial use only: isolate upstream's generic absolute module names."""
    names = ('models', 'utils', 'heuristic_selection', 'help_funcs')
    matches = lambda name: any(name == n or name.startswith(n + '.') for n in names)
    saved = {k: v for k, v in sys.modules.items() if matches(k)}
    before_path = list(sys.path)
    before_cwd = Path.cwd()
    try:
        for key in saved:
            del sys.modules[key]
        sys.path.insert(0, str(runtime))
        os.chdir(Path(runtime).parent)
        importlib.invalidate_caches()
        yield
    finally:
        for key in list(sys.modules):
            if matches(key):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = before_path
        os.chdir(before_cwd)


def _indices(value, length, name, *, empty=False):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.dtype != torch.long or value.ndim != 1
            or (not empty and len(value) == 0) or len(value.unique()) != len(value)
            or bool(((value < 0) | (value >= length)).any())):
        raise ValueError('Invalid CPU index: ' + name)
    return value.detach().clone()


def _inputs(x, edges, y, idx_train, idx_val):
    if (x.ndim != 2 or x.dtype != torch.float32 or x.device.type != 'cpu'
            or not torch.isfinite(x).all() or x.shape[1] < 3):
        raise ValueError('Require finite CPU float32 node features')
    if (edges.dtype != torch.long or edges.device.type != 'cpu' or edges.ndim != 2
            or edges.shape[0] != 2 or bool(((edges < 0) | (edges >= len(x))).any())):
        raise ValueError('Invalid edge_index')
    if (y.dtype != torch.long or y.device.type != 'cpu' or y.ndim != 1 or len(y) > len(x)
            or not set(y.tolist()).issubset({-1, 0, 1})):
        raise ValueError('Labels must be original-node labels in {-1,0,1}')
    train = _indices(idx_train, len(y), 'idx_train')
    val = _indices(idx_val, len(y), 'idx_val')
    if torch.isin(train, val).any() or set(y[train].tolist()) != {0, 1} or bool((y[val] < 0).any()):
        raise ValueError('Supervision must exclude background and train/val must be disjoint')
    return train, val


class AuthorDPGBATrigger(nn.Module):
    """Calls the author's injection exactly, including zero-weight edge rows."""
    def __init__(self, backdoor, device='cpu'):
        super().__init__()
        self.execution_device = _execution_device(device)
        self.trojan = backdoor.trojan.to(self.execution_device)
        self.backdoor = backdoor

    @torch.no_grad()
    def inject(self, x, edge_index, targets):
        targets = _indices(targets, len(x), 'targets', empty=True)
        if not len(targets):
            return x.clone(), edge_index.clone(), x.new_ones(edge_index.shape[1])
        try:
            result = self.backdoor.inject_trigger(targets, x, edge_index,
                x.new_ones(edge_index.shape[1]), self.execution_device)
        finally:
            # The unchanged author function ends by moving the generator to
            # CPU. Restore its registered execution device without changing
            # the computed output or the author's zero-weight edge semantics.
            self.backdoor.trojan.to(self.execution_device)
        return tuple(value.detach().cpu() for value in result)


def fit_dpgba(x, edge_index, y, idx_train, idx_val, candidates, *, seed, work_dir,
              config=None, progress=None, device='cpu'):
    cfg = validate_config(config)
    execution_device = _execution_device(device)
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be uint32')
    train, val = _inputs(x, edge_index, y, idx_train, idx_val)
    if len(y) != len(x):
        raise ValueError('Attack fitting requires original-node graph, no preinserted triggers')
    candidates = _indices(candidates, len(y), 'candidates')
    if (torch.isin(candidates, torch.cat((train, val))).any()
            or not bool((y[candidates] == 0).all()) or len(candidates) < cfg['vs_number']):
        raise ValueError('Need enough unsupervised true-negative TRAIN candidates')
    runtime = prepare_runtime(work_dir)
    started = perf_counter()
    if progress:
        progress(dict(stage='dpgba_author_fit_started', config=cfg, device=str(execution_device)))
    before = {k: _tensor_sha(v) for k, v in dict(x=x, edge_index=edge_index, y=y).items()}
    numpy_state = np.random.get_state()
    try:
        cuda_devices = [execution_device.index] if execution_device.type == 'cuda' else []
        with _author_imports(runtime['runtime_root']), torch.random.fork_rng(devices=cuda_devices):
            np.random.seed(seed)
            _seed_execution(seed, execution_device)
            fit_rng = _rng_record(execution_device)
            backdoor_module = importlib.import_module('models.backdoor')
            selection = importlib.import_module('heuristic_selection')
            runtime_config = dict(cfg, no_cuda=execution_device.type != 'cuda')
            args = SimpleNamespace(**dict(runtime_config, seed=seed, cuda=execution_device.type == 'cuda'))
            # Exactly the author's default 'none' selection and reservoir removal.
            attachments = selection.obtain_attach_nodes(args, candidates.clone(), args.vs_number)
            remaining = torch.tensor(list(set(candidates.cpu().numpy())
                                          - set(attachments.cpu().numpy()))).long()
            backdoor = backdoor_module.Backdoor(args, execution_device)
            backdoor.fit(x.clone().to(execution_device), edge_index.clone().to(execution_device), None,
                         y.clone().to(execution_device), train.to(execution_device),
                         attachments.to(execution_device), remaining.to(execution_device))
            px, pe, pw, py = backdoor.get_poisoned()
            px, pe, pw, py = (value.detach().cpu() for value in (px, pe, pw, py))
            best_hash = _state_sha(backdoor.weights) if backdoor.weights is not None else None
            generator = AuthorDPGBATrigger(backdoor, execution_device).eval()
            generator.requires_grad_(False)
            generator.zero_grad(set_to_none=True)
            record = dict(method='DPGBA_AUTHOR_CODE', config=dict(runtime_config, seed=seed),
                execution_device=str(execution_device), requested_config=cfg,
                fit_rng_states_sha256=fit_rng, final_rng_states_sha256=_rng_record(execution_device),
                rng_scope='CPU and selected CUDA generator restored after fit; author unseeded RandomState retained',
                source_commit=SOURCE_COMMIT, source_repository=SOURCE_URL,
                runtime_manifest=str(Path(work_dir).resolve() / 'author_runtime.json'),
                source=runtime['source'], input_sha256=before,
                attachment_indices=attachments.tolist(), remaining_candidates=remaining.tolist(),
                selector='author_obtain_attach_nodes_random',
                reference_selector='author_models.reconstruct.MLPAE_unmodified',
                dominant_used=False, defense_mode='none_explicit_author_supported_comparison_condition',
                generator_checkpoint='author_actual_final_epoch; tracked_best_not_restored_in_release',
                tracked_best_sha256=best_hash, final_generator_sha256=_state_sha(backdoor.trojan),
                author_saved_final_sha256=_sha(Path(work_dir).resolve() / 'model_weights.pth'),
                original_zero_weight_edges_preserved=True,
                author_entropy_seeded_outer_randomstate_preserved=True,
                strict_seed_reproducibility_claim=False, labels_minus_one_preserved=True,
                elapsed_seconds=perf_counter() - started,
                adaptations=['AV2 tensor/split interface', 'target_class=1', 'vs_number=26 by registered default',
                    'defense_mode=none rather than CLI reconstruct',
                    'explicit execution device=' + str(execution_device),
                    'CPU tensor interface; author training tensors on registered device',
                    'author injection CPU cleanup followed by restoring generator execution device',
                    'common GraphSAGE classifier choice rather than CLI GCN',
                    'optional unused imports relocated in runtime copy only'],
                failure_policy='author reference shortages/numerical failures stop; no fallback')
    finally:
        np.random.set_state(numpy_state)
    if before != {k: _tensor_sha(v) for k, v in dict(x=x, edge_index=edge_index, y=y).items()}:
        raise ValueError('Caller inputs changed')
    if not all(bool(torch.isfinite(v).all()) for v in (px, pw)):
        raise ValueError('Nonfinite author output')
    _json(Path(work_dir) / 'author_fit.json', record)
    if progress:
        progress(dict(stage='dpgba_author_fit_complete', elapsed_seconds=record['elapsed_seconds']))
    return dict(generator=generator, attachment_indices=attachments, poisoned_x=px,
                poisoned_edge_index=pe, poisoned_edge_weights=pw, poisoned_edge_weight=pw,
                poisoned_y=py, fit_indices=torch.cat((train, attachments)), record=record)


def fit_author_classifier(x, edge_index, edge_weight, y, idx_train, idx_val, *, seed,
                          work_dir, config=None, progress=None, device='cpu'):
    """Original DPGBA GraphSage.fit; native validation-accuracy checkpoint rule."""
    cfg = validate_config(config)
    execution_device = _execution_device(device)
    train, val = _inputs(x, edge_index, y, idx_train, idx_val)
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be uint32')
    if (edge_weight is None or edge_weight.device.type != 'cpu'
            or edge_weight.shape != (edge_index.shape[1],)
            or not torch.isfinite(edge_weight).all()):
        raise ValueError('Require raw author edge weights')
    runtime = _classifier_runtime(work_dir)
    started = perf_counter()
    if progress:
        progress(dict(stage='dpgba_author_classifier_started', device=str(execution_device)))
    cuda_devices = [execution_device.index] if execution_device.type == 'cuda' else []
    with _author_imports(runtime['runtime_root']), torch.random.fork_rng(devices=cuda_devices):
        _seed_execution(seed, execution_device)
        cls = importlib.import_module('models.SAGE').GraphSage
        model = cls(nfeat=x.shape[1], nhid=cfg['hidden'], nclass=2,
                    dropout=cfg['dropout'], lr=cfg['train_lr'], weight_decay=cfg['weight_decay'],
                    device=execution_device).to(execution_device)
        initial, fit_rngs = _state_sha(model), _rng_record(execution_device)
        fit_rng = fit_rngs['cpu']
        dx, de, dw, dy, dt, dv = (value.clone().to(execution_device)
                                  for value in (x, edge_index, edge_weight, y, train, val))
        trace = []
        def observe(module, inputs, output):
            if not model.training:
                trace.append(dict(epoch=len(trace), normal_dev_accuracy=float(
                    (output[dv].argmax(1) == dy[dv]).double().mean()), state_sha256=_state_sha(model)))
                if progress:
                    progress(dict(stage='dpgba_author_classifier_dev', **trace[-1]))
        # The native fit calls self.forward directly, so observe the final
        # convolution without replacing the author forward/training methods.
        # Its argmax equals the following native log_softmax argmax.
        hook = model.gc2.register_forward_hook(observe)
        try:
            model.fit(dx, de, dw, dy, dt, dv,
                      train_iters=cfg['epochs'], verbose=False)
        finally:
            hook.remove()
        if len(trace) != cfg['epochs']:
            raise ValueError('Native classifier validation trace length changed')
        best = max(trace, key=lambda v: v['normal_dev_accuracy'])
        if _state_sha(model) != best['state_sha256']:
            raise ValueError('Native classifier did not restore its highest-accuracy checkpoint')
        model.eval().requires_grad_(False)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            output = model(dx, de, dw)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite author classifier output')
        final_rngs = _rng_record(execution_device)
    record = dict(method='DPGBA_AUTHOR_GRAPHSAGE', source_commit=SOURCE_COMMIT,
                  seed=seed, config=cfg, initialization_sha256=initial,
                  initialization_digest=initial, fit_rng_digest=fit_rng,
                  execution_device=str(execution_device), fit_rng_states_sha256=fit_rngs,
                  final_rng_states_sha256=final_rngs,
                  final_state_sha256=_state_sha(model), supervised_indices_sha256=_tensor_sha(train),
                  validation_indices_sha256=_tensor_sha(val), supervision_count=len(train),
                  model_selection='author_validation_accuracy_strict_improvement',
                  selected_epoch=best['epoch'], epochs=cfg['epochs'], trace=trace,
                  raw_zero_weight_edges_preserved=True, author_graphsage_ignores_edge_weights=True,
                  elapsed_seconds=perf_counter() - started,
                  runtime_manifest=str(Path(work_dir).resolve() / 'author_runtime.json'))
    serial = len(list(Path(work_dir).glob('author_classifier_*.json')))
    _json(Path(work_dir) / f'author_classifier_{serial:04d}.json', record)
    if progress:
        progress(dict(stage='dpgba_author_classifier_complete', elapsed_seconds=record['elapsed_seconds']))
    return dict(model=model, record=record)
