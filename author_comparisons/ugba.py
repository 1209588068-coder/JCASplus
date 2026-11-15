"""Thin adapter around the unmodified, pinned UGBA author implementation.

No generator, selector score, optimizer, loss, or checkpoint rule is implemented
here. Author functions run from a separate source copy; an auditable runtime diff
only makes optional imports lazy and adapts class-count/isolated-node metadata.
Pristine source: https://github.com/ventr1c/UGBA at
c1d945a215dd16fd806f8a231381d303a0b811f7. A bounded network clone failed;
the pristine checkout was cloned from the clean git objects in the local legacy
UGBA_official archive. No license file was found; no license grant is implied.

Use one method/case per process. These adapters never load a dataset or a model
checkpoint. The caller supplies TRAIN+normal-DEV disjoint-union tensors. The
native evaluation injection retains zero-weight edges, unlike get_poisoned().
This difference matters for the author's weight-ignoring GraphSAGE and is NOT
silently corrected here.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import difflib
import hashlib
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


AUTHOR_COMMIT = "c1d945a215dd16fd806f8a231381d303a0b811f7"
AUTHOR_URL = "https://github.com/ventr1c/UGBA"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "third_party/author_baselines/UGBA"
ACQUISITION = "clean local git-object clone from /home/mizhou/JCAS_cleanup_20261027_Mp0CdD/legacy/third_party/UGBA_official after bounded HTTPS clone failure"

# Literal defaults from run_adaptive.py (seed is an explicit fit argument).
AUTHOR_DEFAULTS = {
    "debug": True, "no_cuda": False, "model": "GCN", "dataset": "Cora",
    "train_lr": .01, "weight_decay": 5e-4, "hidden": 32, "thrd": .5,
    "target_class": 0, "dropout": .5, "epochs": 200, "trojan_epochs": 400,
    "inner": 1, "lr": .01, "trigger_size": 3, "use_vs_number": True,
    "vs_ratio": 0, "vs_number": 40, "defense_mode": "prune", "prune_thr": .8,
    "target_loss_weight": 1, "homo_loss_weight": 100, "homo_boost_thrd": .8,
    "dis_weight": 1, "selection_method": "none", "test_model": "GCN",
    "evaluate_mode": "1by1", "device_id": 0,
}
DEFAULT_CONFIG = {**AUTHOR_DEFAULTS, "no_cuda": True, "dataset": "AV2",
                  "target_class": 1, "vs_number": 26, "defense_mode": "none",
                  "selection_method": "cluster_degree", "test_model": "GraphSage",
                  "num_classes": 2}


def validate_config(config=None):
    """Complete explicit author-CLI settings; no silent tuning or budget cap."""
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("unknown UGBA author-config option")
    result = {**DEFAULT_CONFIG, **deepcopy(config)}
    for key in ("hidden", "epochs", "trojan_epochs", "inner", "trigger_size", "vs_number"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"invalid {key}")
    for key in ("train_lr", "weight_decay", "thrd", "dropout", "lr", "vs_ratio",
                "prune_thr", "target_loss_weight", "homo_loss_weight", "homo_boost_thrd", "dis_weight"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise ValueError(f"invalid {key}")
    if result["lr"] <= 0 or result["train_lr"] <= 0 or result["weight_decay"] < 0:
        raise ValueError("invalid optimizer parameters")
    if not 0 <= result["dropout"] < 1 or result["homo_loss_weight"] < 0 or result["target_loss_weight"] <= 0:
        raise ValueError("invalid loss/dropout parameters")
    for key in ("debug", "no_cuda", "use_vs_number"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be bool")
    fixed = {"dataset": "AV2", "model": "GCN", "test_model": "GraphSage", "num_classes": 2,
             "defense_mode": "none", "no_cuda": True, "use_vs_number": True,
             "vs_ratio": 0, "evaluate_mode": "1by1", "device_id": 0}
    if any(result[k] != v or type(result[k]) is not type(v) for k, v in fixed.items()):
        raise ValueError("unsupported task/device/defense configuration")
    if type(result["target_class"]) is not int or result["target_class"] not in (0, 1):
        raise ValueError("invalid binary target class")
    if result["selection_method"] not in ("none", "cluster_degree"):
        raise ValueError("only native random and global cluster_degree selection are exposed")
    return result


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return _sha((str(tensor.dtype) + str(tuple(tensor.shape))).encode() + tensor.numpy().tobytes())


def _state_hash(model):
    return _sha("".join(name + _tensor_hash(value) for name, value in sorted(model.state_dict().items())).encode())


def _git(*args):
    return subprocess.run(["git", "-C", str(SOURCE_ROOT), *args], check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()


def verify_source():
    """Read-only JSON-safe provenance; reject any altered pristine checkout."""
    commit = _git("rev-parse", "HEAD")
    if commit != AUTHOR_COMMIT or _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("UGBA pristine source commit or clean-worktree check failed")
    files = sorted(p for p in SOURCE_ROOT.rglob("*.py") if ".git" not in p.parts)
    files += [SOURCE_ROOT / "README.md", SOURCE_ROOT / "script/train_UGBA.sh"]
    hashes = {str(p.relative_to(SOURCE_ROOT)): _sha(p.read_bytes()) for p in files}
    licenses = [p.name for p in SOURCE_ROOT.iterdir() if p.is_file() and
                p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))]
    return {"repository": AUTHOR_URL, "commit": commit, "dirty": False,
            "acquisition": ACQUISITION, "license_files": sorted(licenses),
            "license_status": "no license file found; no license grant inferred" if not licenses else "inspect listed files",
            "source_files_sha256": hashes}


def _runtime_text(relative, original):
    """Bounded text substitutions; all changed lines are exposed in the diff."""
    modified = original
    if relative == "heuristic_selection.py":
        line = "from sklearn_extra import cluster\n"
        if modified.count(line) != 1:
            raise ValueError("unexpected author optional-import source")
        modified = modified.replace(line, "")
        modified, count = re.subn(r"(?m)^( +)(kmedoids = cluster\.KMedoids\(.*)$",
                                  r"\1from sklearn_extra import cluster\n\1\2", modified)
        if count != 2:
            raise ValueError("unexpected author KMedoids sites")
        class_line = "nclass = np.unique(data.y.cpu().numpy()).shape[0]"
        if modified.count(class_line) != 3:
            raise ValueError("unexpected author class-count sites")
        modified = modified.replace(class_line, "nclass = args.num_classes")
        for endpoint in (0, 1):
            old = f"degree(edge_index[{endpoint}])"
            if modified.count(old) != 2:
                raise ValueError("unexpected author degree sites")
            modified = modified.replace(old, f"degree(edge_index[{endpoint}], num_nodes=x.shape[0])")
    elif relative == "models/construct.py":
        choices = {"GCN": "GCN", "GAT": "GAT", "GraphSage": "SAGE",
                   "GCN_Encoder": "GCN_Encoder", "GNNGuard": "GNNGuard", "RobustGCN": "RobustGCN"}
        for cls, module in choices.items():
            statement = f"from models.{module} import {cls}"
            if modified.count(statement) != 1:
                raise ValueError("unexpected author model import")
            modified = modified.replace(statement + "\n", "")
            pattern = rf"(?m)^(    (?:if|elif)[ \t]*\(model_name == '{cls}'\):)([ \t]*)$"
            modified, count = re.subn(pattern, rf"\1\2\n        {statement}", modified)
            if count != 1:
                raise ValueError(f"unexpected author dispatch branch {cls}")
    return modified


def prepare_runtime(work_dir):
    """Create/revalidate a reusable isolated runtime copy and unified diff."""
    source = verify_source()
    work = Path(work_dir).resolve()
    if work == SOURCE_ROOT or SOURCE_ROOT in work.parents:
        raise ValueError("runtime must not be inside pristine author source")
    runtime = work / "author_runtime"
    paths = [SOURCE_ROOT / "utils.py", SOURCE_ROOT / "heuristic_selection.py"]
    paths += sorted((SOURCE_ROOT / "models").rglob("*.py"))
    expected, diffs = {}, []
    for path in paths:
        relative = str(path.relative_to(SOURCE_ROOT))
        before = path.read_text()
        after = _runtime_text(relative, before)
        expected[relative] = after.encode()
        if before != after:
            diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                             fromfile=f"pristine/{relative}", tofile=f"runtime/{relative}"))
    patch = "".join(diffs).encode()
    patch_path = work / "author_runtime.patch"
    if runtime.exists():
        for relative, content in expected.items():
            if not (runtime / relative).is_file() or (runtime / relative).read_bytes() != content:
                raise ValueError("existing author runtime changed; refusing overwrite")
        if not patch_path.is_file() or patch_path.read_bytes() != patch:
            raise ValueError("existing runtime patch record changed")
    else:
        if patch_path.exists():
            raise ValueError("orphan runtime patch; refusing overwrite")
        runtime.mkdir(parents=True, exist_ok=False)
        for relative, content in expected.items():
            destination = runtime / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        patch_path.write_bytes(patch)
    return {"runtime_root": str(runtime), "patch_path": str(patch_path),
            "patch_sha256": _sha(patch), "runtime_files_sha256": {k: _sha(v) for k, v in expected.items()},
            "source": source, "patch_scope": ["optional imports lazy only",
                "class count from explicit binary task metadata; -1 background labels retained",
                "degree num_nodes padding for isolated original nodes; score/ranking unchanged"],
            "unmodified_core_files": ["models/backdoor.py", "models/GCN.py", "models/GCN_Encoder.py", "models/SAGE.py"]}


@contextmanager
def load_author_modules(runtime_root, module_names):
    """Temporary author absolute-import namespace; no source or algorithm patch."""
    prefixes = ("models", "utils", "heuristic_selection")
    affected = lambda name: any(name == p or name.startswith(p + ".") for p in prefixes)
    previous = {name: module for name, module in sys.modules.items() if affected(name)}
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    for name in previous:
        del sys.modules[name]
    sys.path.insert(0, str(runtime_root))
    sys.dont_write_bytecode = True
    try:
        importlib.invalidate_caches()
        yield {name: importlib.import_module(name) for name in module_names}
    finally:
        for name in list(sys.modules):
            if affected(name):
                del sys.modules[name]
        sys.modules.update(previous)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def _device(value):
    """Resolve an explicit device; never silently fall back to CPU."""
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device must be cpu or an available cuda device") from exc
    if device.type == "cpu" and device.index is None:
        return device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("requested device unavailable; no CPU fallback")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError("requested CUDA index unavailable")
    return torch.device("cuda", index)


def _rng_record(device):
    return {"cpu_rng_sha256": _tensor_hash(torch.get_rng_state()),
            "cuda_rng_sha256": (_tensor_hash(torch.cuda.get_rng_state(device))
                                if device.type == "cuda" else None)}


@contextmanager
def _execution(seed, directory, device=torch.device("cpu")):
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be uint32 integer")
    previous_numpy, previous_directory = np.random.get_state(), Path.cwd()
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        np.random.seed(seed)
        # Seed only the participating generators; do not disturb another GPU.
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous_directory)
            np.random.set_state(previous_numpy)


def _idx(value, n, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1 or value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU long indices")
    if not len(value) or value.min() < 0 or value.max() >= n or len(value.unique()) != len(value):
        raise ValueError(f"invalid {name}")


def _validate_data(x, edges, y, train, val, candidates=None):
    if x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError("expected finite CPU float32 features")
    if edges.device.type != "cpu" or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("expected CPU long 2xE edges")
    if edges.numel() and (edges.min() < 0 or edges.max() >= len(x)):
        raise ValueError("edge outside node range")
    if y.device.type != "cpu" or y.dtype != torch.long or y.ndim != 1 or len(y) > len(x) or not len(y):
        raise ValueError("invalid original-node labels")
    if not torch.isin(y, torch.tensor([-1, 0, 1])).all():
        raise ValueError("only background -1 and binary task labels allowed")
    groups = {"idx_train": train, "idx_val": val}
    if candidates is not None:
        groups["candidates"] = candidates
    for name, value in groups.items():
        _idx(value, len(y), name)
        if not torch.isin(y[value], torch.tensor([0, 1])).all():
            raise ValueError(f"{name} must have valid task labels")
    joined = torch.cat(tuple(groups.values()))
    if len(joined.unique()) != len(joined):
        raise ValueError("train, validation, and candidate populations must be disjoint")


def _versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "torch-geometric", "numpy", "scikit-learn")}


class AuthorGenerator(nn.Module):
    """Only forwards to the original Backdoor.inject_trigger, including zeros."""
    def __init__(self, backdoor):
        super().__init__()
        self.trojan = backdoor.trojan
        self._backdoor = backdoor

    @torch.no_grad()
    def inject(self, x, edge_index, targets):
        if x.device.type != "cpu" or edge_index.device.type != "cpu" or targets.device.type != "cpu":
            raise ValueError("evaluation injection expects CPU scene tensors")
        if not len(targets):
            raise ValueError("native inject requires at least one target")
        weights = torch.ones(edge_index.shape[1], dtype=x.dtype, device=x.device)
        # Intentionally NO positive-edge filter: native evaluation retains zeros.
        try:
            output = self._backdoor.inject_trigger(targets, x, edge_index, weights, self._backdoor.device)
            return tuple(t.detach().cpu() for t in output)
        finally:
            # The native call moves its module to CPU on exit; residency is an
            # interface concern, not an alteration of injection values/edges.
            self._backdoor.trojan.to(self._backdoor.device)


def fit_ugba(x, edge_index, y, idx_train, idx_val, candidates, *, seed, work_dir,
             config=None, progress=None, device="cpu"):
    """Call original selector, Backdoor.fit, and get_poisoned without replacement."""
    cfg = validate_config(config)
    device = _device(device)
    _validate_data(x, edge_index, y, idx_train, idx_val, candidates)
    if len(y) != len(x):
        raise ValueError("UGBA fit expects one original label/background marker per node")
    if len(candidates) < cfg["vs_number"] + 512:
        raise ValueError("native UGBA requires 512 remaining unlabeled outer nodes")
    work = Path(work_dir).resolve()
    # The native selector has an on-disk cache. Never silently reuse attachments.
    if (work / "selected_nodes").exists() or (work / "modelpath").exists():
        raise ValueError("fresh UGBA fit refuses author selection/model caches")
    runtime = prepare_runtime(work)
    input_hashes = {"x": _tensor_hash(x), "edges": _tensor_hash(edge_index), "y": _tensor_hash(y)}
    effective_cfg = {**cfg, "no_cuda": device.type != "cuda",
                     "device_id": device.index if device.type == "cuda" else 0}
    args = SimpleNamespace(**effective_cfg, seed=seed, cuda=device.type == "cuda")
    if progress:
        progress({"phase": "author_selection_start", "seed": seed})
    with load_author_modules(runtime["runtime_root"], ("heuristic_selection", "models.backdoor")) as modules:
        hs, native = modules["heuristic_selection"], modules["models.backdoor"]
        with _execution(seed, work, device):
            rng_start = _rng_record(device)
            dx, de, dy, dt, dv = (t.to(device) for t in (x, edge_index, y, idx_train, idx_val))
            data = SimpleNamespace(x=dx, edge_index=de, y=dy)
            if cfg["selection_method"] == "none":
                selected = hs.obtain_attach_nodes(args, candidates, cfg["vs_number"])
            else:
                selected = hs.cluster_degree_selection(args, data, dt, dv, dv,
                    candidates.to(device), de, cfg["vs_number"], device)
                selected = torch.LongTensor(selected)
            _idx(selected, len(x), "author attachments")
            if len(selected) != cfg["vs_number"] or not torch.isin(selected, candidates).all():
                raise ValueError("native selector did not return the registered attachment budget")
            # Exact run_adaptive.py set-difference ordering, not a new selector.
            remaining = torch.tensor(list(set(candidates.numpy()) - set(selected.numpy())), dtype=torch.long)
            if len(remaining) < 512:
                raise ValueError("native outer reservoir below 512")
            if progress:
                progress({"phase": "author_backdoor_start", "trojan_epochs": cfg["trojan_epochs"]})
            backdoor = native.Backdoor(args, device)
            backdoor.fit(dx, de, None, dy, dt, selected.to(device), remaining.to(device))
            px, pe, pw, py = backdoor.get_poisoned()
            backdoor.trojan.to(device)
            px, pe, pw, py = (t.detach().cpu() for t in (px, pe, pw, py))
            if not torch.isfinite(px).all() or not torch.isfinite(pw).all():
                raise ValueError("native UGBA returned nonfinite graph")
            restored = all(torch.equal(value, backdoor.weights[name]) for name, value in backdoor.trojan.state_dict().items())
            if not restored:
                raise ValueError("native best-outer-loss checkpoint was not restored")
            generator = AuthorGenerator(backdoor)
            generator.eval()
            rng_end = _rng_record(device)
    if input_hashes != {"x": _tensor_hash(x), "edges": _tensor_hash(edge_index), "y": _tensor_hash(y)}:
        raise ValueError("author fit mutated clean caller inputs")
    if not torch.equal(px[:len(x)], x) or not torch.equal(pe[:, :edge_index.shape[1]], edge_index):
        raise ValueError("native injection failed original node/edge prefix contract")
    fit_indices = torch.cat((idx_train, selected))
    record = {"method": "UGBA_author_source", "seed": seed, "config": cfg, "runtime": runtime,
              "device": str(device), "effective_author_device_args": {k: effective_cfg[k] for k in ("no_cuda", "device_id")},
              "rng_start": rng_start, "rng_end": rng_end,
              "selector_compute": "native encoder on selected device; KMeans and cluster-distance ranking on CPU",
              "versions": _versions(), "author_defaults": AUTHOR_DEFAULTS,
              "explicit_overrides": {k: v for k, v in cfg.items() if k not in AUTHOR_DEFAULTS or v != AUTHOR_DEFAULTS[k]},
              "author_calls": ["heuristic_selection." + ("obtain_attach_nodes" if cfg["selection_method"] == "none" else "cluster_degree_selection"),
                               "models.backdoor.Backdoor.fit", "models.backdoor.Backdoor.get_poisoned"],
              "generator_checkpoint_rule": "native minimum outer-training loss, strict improvement, restored after fit",
              "author_best_state_restored": restored, "generator_state_sha256": _state_hash(generator),
              "input_sha256": input_hashes, "attachment_indices": selected.tolist(),
              "remaining_outer_population_sha256": _tensor_hash(remaining), "fit_indices_sha256": _tensor_hash(fit_indices),
              "original_nodes": len(x), "poisoned_nodes": len(px), "original_edges": edge_index.shape[1],
              "poisoned_edges": pe.shape[1], "author_fixed_outer_sample_size": 512,
              "configured_generator_iterations": cfg["trojan_epochs"],
              "configured_surrogate_steps": cfg["trojan_epochs"] * cfg["inner"],
              "configured_selector_epochs": cfg["epochs"] if cfg["selection_method"] == "cluster_degree" else 0,
              "probe_indices_role": "normal DEV, same as idx_val; no test accessed",
              "selection_checkpoint_rule": "native highest normal DEV accuracy (cluster_degree)",
              "train_zero_weight_edges": "native get_poisoned filters >0",
              "evaluation_zero_weight_edges": "native inject_trigger retains zero-weight edges; GraphSAGE ignores weights",
              "feature_preprocessing": "caller-provided AV2 normalization, no citation-data NormalizeFeatures",
              "inputs_unchanged": True, "fresh_checkpoint_free_fit": True}
    if progress:
        progress({"phase": "author_backdoor_complete", "seed": seed})
    return {"generator": generator, "attachment_indices": selected.clone(), "poisoned_x": px,
            "poisoned_edge_index": pe, "poisoned_edge_weight": pw, "poisoned_y": py,
            "fit_indices": fit_indices, "record": record}


def fit_author_classifier(x, edge_index, edge_weight, y, idx_train, idx_val, *, seed,
                          work_dir, config=None, progress=None, device="cpu"):
    """Direct original GraphSage.fit with normal-DEV accuracy checkpointing."""
    cfg = validate_config(config)
    device = _device(device)
    _validate_data(x, edge_index, y, idx_train, idx_val)
    if edge_weight is not None and (edge_weight.shape != (edge_index.shape[1],) or not torch.isfinite(edge_weight).all()):
        raise ValueError("invalid edge weights")
    runtime = prepare_runtime(work_dir)
    trace = []
    with load_author_modules(runtime["runtime_root"], ("models.SAGE",)) as modules:
        with _execution(seed, Path(work_dir).resolve(), device):
            model = modules["models.SAGE"].GraphSage(nfeat=x.shape[1], nhid=cfg["hidden"], nclass=2,
                dropout=cfg["dropout"], lr=cfg["train_lr"], weight_decay=cfg["weight_decay"], device=device).to(device)
            initial, fit_rng = _state_hash(model), _tensor_hash(torch.get_rng_state())
            rng_start = _rng_record(device)
            dx, de, dy, dt, dv = (t.to(device) for t in (x, edge_index, y, idx_train, idx_val))
            dw = edge_weight.to(device) if edge_weight is not None else None
            # Observation only: it does not alter outputs, optimizer, or selection.
            def observe(module, inputs, output):
                if not model.training:
                    accuracy = float((output[dv].argmax(1) == dy[dv]).double().mean())
                    trace.append({"epoch": len(trace), "normal_dev_accuracy": accuracy,
                                  "state_sha256": _state_hash(model)})
                    if progress:
                        progress({"phase": "author_classifier_dev", **trace[-1]})
            # Author fit calls self.forward directly (bypassing Module hooks).
            # Observe its final convolution; logits and log_softmax have the
            # same argmax. This leaves the author's forward/fit code untouched.
            hook = model.gc2.register_forward_hook(observe)
            try:
                model.fit(dx, de, dw, dy, dt, dv,
                          train_iters=cfg["epochs"], verbose=False)
            finally:
                hook.remove()
            if len(trace) != cfg["epochs"]:
                raise ValueError("native classifier validation trace length mismatch")
            best = max(trace, key=lambda row: row["normal_dev_accuracy"])
            if _state_hash(model) != best["state_sha256"]:
                raise ValueError("native highest-DEV-accuracy checkpoint mismatch")
            model.eval()
            rng_end = _rng_record(device)
    return {"model": model, "record": {"method": "UGBA_author_GraphSage", "seed": seed,
            "config": cfg, "runtime": runtime, "initialization_digest": initial, "fit_rng_digest": fit_rng,
            "device": str(device), "rng_start": rng_start, "rng_end": rng_end,
            "checkpoint_rule": "native highest normal DEV accuracy; strict >; earliest tie",
            "selected_epoch": best["epoch"], "epochs": cfg["epochs"], "trace": trace,
            "state_sha256": _state_hash(model), "author_file": "models/SAGE.py",
            "weights_ignored_by_native_GraphSAGE": True}}
