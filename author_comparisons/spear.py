"""Thin, provenance-bound adapter to the actual author SPEAR implementation.

All selector, SAGE, generator, surrogate and classifier computations execute
author source files. A per-run copy has only audited import/path compatibility
edits; pristine third_party files remain untouched. Use a separate process for
each method/seed because author code uses absolute ``models``/``utils`` imports.
This module never reads AV2 graphs or starts a run when imported.
"""
from __future__ import annotations

from contextlib import contextmanager
import difflib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "third_party/author_baselines/SPEAR"
SOURCE_COMMIT = "187d2cdc1f2454795dc8b37467ee2103e2613b15"
SOURCE_URL = "https://github.com/yhDing/SPEAR"
SOURCE_ARCHIVE = Path("/home/mizhou/SPEAR-main.zip")
ARCHIVE_SHA256 = "be18c0145089c0f063a6f863bf76b7de5d7654f65e1aa7427a54afe2505e0757"
RUNTIME_MANIFEST = "AUTHOR_SPEAR_RUNTIME.json"
DEFAULT_CONFIG = dict(
    epochs=200, trojan_epochs=400, inner=1, hidden=32, alpha_int=30,
    train_lr=0.01, shadow_lr=0.01, trojan_lr=0.01, weight_decay=5e-4,
    dropout=0.5, thrd=0.5, outter_size=512, target_loss_weight=1.0,
    homo_loss_weight=0.1, vs_number=26, target_class=1,
    model="GCN", test_model="GraphSage", sample_num=16,
    sage_batch_size=512, sage_n_permutations=None,
    sage_detect_convergence=True, sage_thresh=0.025, debug=True,
)


def validate_config(config=None):
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a dict")
    unknown = set(config or {}) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown author SPEAR keys: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **(config or {})}
    for name in ("epochs", "trojan_epochs", "inner", "hidden", "alpha_int", "outter_size", "vs_number", "sample_num", "sage_batch_size"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if result["hidden"] < 2:
        raise ValueError("hidden must be at least two")
    if type(result["target_class"]) is not int or result["target_class"] != 1:
        raise ValueError("JCAS negative-to-positive adaptation requires target_class=1")
    if result["model"] != "GCN" or result["test_model"] != "GraphSage":
        raise ValueError("this adapter binds the author GCN selector and GraphSage classifier")
    for name in ("train_lr", "shadow_lr", "trojan_lr", "weight_decay", "dropout", "thrd", "target_loss_weight", "homo_loss_weight", "sage_thresh"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if min(result[k] for k in ("train_lr", "shadow_lr", "trojan_lr", "target_loss_weight")) <= 0:
        raise ValueError("learning rates and target loss weight must be positive")
    if min(result[k] for k in ("weight_decay", "homo_loss_weight")) < 0:
        raise ValueError("regularization weights must be nonnegative")
    if not 0 <= result["dropout"] < 1 or not 0 < result["sage_thresh"] < 1:
        raise ValueError("invalid dropout or SAGE convergence threshold")
    if result["sage_n_permutations"] is not None and (type(result["sage_n_permutations"]) is not int or result["sage_n_permutations"] < 1):
        raise ValueError("sage_n_permutations must be None or a positive integer")
    if type(result["debug"]) is not bool or type(result["sage_detect_convergence"]) is not bool:
        raise ValueError("debug and sage_detect_convergence must be booleans")
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_source():
    """Check every vendored source byte against the hash-bound user archive."""
    if _sha(SOURCE_ARCHIVE) != ARCHIVE_SHA256:
        raise RuntimeError("SPEAR user archive hash changed")
    files = {}
    with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
        if archive.comment.decode().strip() != SOURCE_COMMIT:
            raise RuntimeError("SPEAR archive commit comment changed")
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "SPEAR-main":
                raise RuntimeError("unsafe source archive path")
            relative = PurePosixPath(*relative.parts[1:])
            if info.is_dir() or not (relative.suffix in {".py", ".sh", ".md"} or relative.name == ".gitignore"):
                continue
            expected = hashlib.sha256(archive.read(info)).hexdigest()
            current = SOURCE_ROOT / str(relative)
            if current.is_symlink() or not current.is_file() or _sha(current) != expected:
                raise RuntimeError(f"pristine SPEAR source changed: {relative}")
            files[str(relative)] = expected
    return dict(repository=SOURCE_URL, commit=SOURCE_COMMIT, archive=str(SOURCE_ARCHIVE),
                archive_sha256=ARCHIVE_SHA256, origin="verified_user_zip_source_only",
                remote_pull=False, commit_evidence="GitHub ZIP archive comment; no successful remote clone",
                source_directory=str(SOURCE_ROOT), files=files,
                license="No LICENSE/COPYING/NOTICE in the supplied archive; README retains upstream UGBA attribution.")


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise RuntimeError(f"bounded author compatibility patch did not match: {old!r}")
    return source.replace(old, new)


def compatibility_sources():
    """Bounded I/O/import edits only; no optimizer/loss/selection modifications."""
    changed = {}
    relative = "models/backdoor_GCN_cos.py"
    old = "file_path = os.path.join(current_dir, f'../save_selected_feature/{args.dataset}/val_{sage_epoch}.npy')"
    new = "file_path = getattr(args, 'feature_importance_path', None) or os.path.join(current_dir, f'../save_selected_feature/{args.dataset}/val_{sage_epoch}.npy')"
    changed[relative] = _replace((SOURCE_ROOT / relative).read_text(), old, new, 2)
    relative = "models/construct.py"
    source = (SOURCE_ROOT / relative).read_text()
    imports = dict(MLP="from models.MLP import MLP\n", GCN="from models.GCN import GCN\n",
        GAT="from models.GAT import GAT\n", GraphSage="from models.SAGE import GraphSage\n",
        GCN_Encoder="from models.GCN_Encoder import GCN_Encoder\n",
        GNNGuard="from models.GNNGuard import GNNGuard\n", RobustGCN="from models.RobustGCN import RobustGCN\n")
    for model, statement in imports.items():
        source = _replace(source, statement, "")
        branch = f"if (model_name == '{model}'):\n" if model == "GCN" else f"elif(model_name == '{model}'):\n"
        source = _replace(source, branch, branch + "        " + statement)
    changed[relative] = source
    # The sole used deeprobust API is accuracy. Its argmax/type_as/eq/double/sum
    # calculation is identical, line for line, to this author's own utils.py.
    relative = "models/MLP.py"
    changed[relative] = _replace((SOURCE_ROOT / relative).read_text(), "from deeprobust.graph import utils", "import utils")
    relative = "sage_modified/__init__.py"
    source = _replace((SOURCE_ROOT / relative).read_text(),
        "from . import utils, core, imputers, grouped_imputers, plotting, datasets",
        "from . import utils, core, imputers, grouped_imputers, datasets")
    source = _replace(source, "from .plotting import plot, comparison_plot\n", "")
    source += "\n\ndef plot(*args, **kwargs):\n    from .plotting import plot as impl\n    return impl(*args, **kwargs)\n\ndef comparison_plot(*args, **kwargs):\n    from .plotting import comparison_plot as impl\n    return impl(*args, **kwargs)\n"
    changed[relative] = source
    relative = "sage_modified/core.py"
    source = _replace((SOURCE_ROOT / relative).read_text(), "from sage_modified import plotting\n", "")
    for method in ("plot", "comparison_plot"):
        old = f"        return plotting.{method}("
        source = _replace(source, old, "        from sage_modified import plotting\n" + old)
    changed[relative] = source
    return changed


def prepare_runtime(work_dir):
    """Copy actual author files and retain the exact compatibility diff/hashes."""
    source = verify_source()
    destination = Path(work_dir).resolve() / "author_runtime"
    changes = compatibility_sources()
    if destination.exists():
        manifest = json.loads((destination / RUNTIME_MANIFEST).read_text())
        if manifest["pristine_files"] != source["files"]:
            raise RuntimeError("existing author runtime has a different source binding")
    else:
        destination.mkdir(parents=True)
        patches = []
        for relative in source["files"]:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE_ROOT / relative, path)
            if relative in changes:
                original = path.read_text()
                path.write_text(changes[relative])
                patches.extend(difflib.unified_diff(original.splitlines(True), changes[relative].splitlines(True),
                    fromfile="pristine/" + relative, tofile="runtime/" + relative))
        (destination / "compatibility.patch").write_text("".join(patches))
        manifest = dict(source=source, pristine_files=source["files"], runtime_files={r: _sha(destination / r) for r in source["files"]},
                        runtime_directory=str(destination), compatibility_files=list(changes),
                        patch_sha256=_sha(destination / "compatibility.patch"),
                        compatibility_scope=["feature importance file path only", "lazy imports for unused model/plotting dependencies", "MLP uses identical author-local accuracy metric"],
                        deeprobust_accuracy_reference="https://raw.githubusercontent.com/DSE-MSU/DeepRobust/master/deeprobust/graph/utils.py#L258-L279",
                        algorithms_reimplemented=False)
        (destination / RUNTIME_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    for relative, expected in manifest["runtime_files"].items():
        prescribed = hashlib.sha256(changes[relative].encode()).hexdigest() if relative in changes else source["files"][relative]
        if prescribed != expected or _sha(destination / relative) != expected:
            raise RuntimeError(f"author runtime source changed: {relative}")
    if _sha(destination / "compatibility.patch") != manifest["patch_sha256"]:
        raise RuntimeError("author compatibility diff changed")
    return destination, manifest


def import_author_runtime(work_dir):
    destination, manifest = prepare_runtime(work_dir)
    roots = {"models", "utils", "select_sample", "sage_modified"}
    for name, module in list(sys.modules.items()):
        if name.split(".", 1)[0] in roots:
            file = getattr(module, "__file__", None)
            paths = list(getattr(module, "__path__", []))
            same_runtime = (file and Path(file).resolve().is_relative_to(destination)) or (
                paths and all(Path(path).resolve().is_relative_to(destination) for path in paths))
            if not same_runtime:
                del sys.modules[name]
    sys.path[:] = [str(destination)] + [entry for entry in sys.path if entry != str(destination)]
    importlib.invalidate_caches()
    api = dict(select_sample=importlib.import_module("select_sample"),
               sage=importlib.import_module("sage_modified"),
               backdoor=importlib.import_module("models.backdoor_GCN_cos"),
               construct=importlib.import_module("models.construct"))
    for module in api.values():
        if not Path(module.__file__).resolve().is_relative_to(destination):
            raise RuntimeError("author module came from an unexpected runtime")
    return api, manifest


@contextmanager
def _cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _device(device):
    value = torch.device(device)
    if value.type not in ("cpu", "cuda"):
        raise ValueError("author SPEAR adapter supports only CPU or CUDA")
    if value.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; CPU fallback is prohibited")
        index = torch.cuda.current_device() if value.index is None else value.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError("requested CUDA device is out of range")
        value = torch.device("cuda", index)
    return value


@contextmanager
def _precision(device):
    """Full FP32 execution; preserve the caller's backend precision flags."""
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("author source constructors require default float32 dtype")
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def _rng_digest(device):
    states = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states[str(device)] = torch.cuda.get_rng_state(device)
    return _tensor_digest(states)


@contextmanager
def _rng(seed, device=torch.device("cpu")):
    numpy_state = np.random.get_state()
    try:
        devices = [] if device.type == "cpu" else [device.index]
        with torch.random.fork_rng(devices=devices):
            # Seed only the CPU and selected CUDA stream, not unrelated GPUs.
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.default_generators[device.index].manual_seed(seed)
            np.random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)


def _arguments(config, seed, feature_path=None):
    return SimpleNamespace(**config, seed=seed, dataset="Cora", feature_importance_path=feature_path,
                           defense_mode="none", use_vs_number=True, alpha=0.02, dis_weight=1.0)


def _validate_inputs(x, edges, y, train, val, candidates=None):
    if x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("author adapter expects finite CPU float32 features")
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or edges.device.type != "cpu":
        raise ValueError("edges must be a CPU [2,E] long tensor")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= len(x)):
        raise ValueError("edge indices out of range")
    if y.dtype != torch.long or y.shape != (len(x),) or y.device.type != "cpu":
        raise ValueError("labels must be aligned CPU long values")
    if not ((y == -1) | (y == 0) | (y == 1)).all() or int(y.max()) != 1:
        raise ValueError("labels must retain background -1 and binary task labels; positive class required")
    groups = [train, val] + ([] if candidates is None else [candidates])
    for indices in groups:
        if indices.dtype != torch.long or indices.ndim != 1 or not len(indices) or indices.device.type != "cpu" or len(indices.unique()) != len(indices):
            raise ValueError("train/dev/candidates must be nonempty unique CPU index vectors")
        if int(indices.min()) < 0 or int(indices.max()) >= len(x):
            raise ValueError("indices out of range")
    if len(torch.cat(groups).unique()) != sum(map(len, groups)):
        raise ValueError("train/dev/candidates must be disjoint")
    if not ((y[torch.cat((train, val))] == 0) | (y[torch.cat((train, val))] == 1)).all():
        raise ValueError("normal supervision must have binary labels")
    if candidates is not None and not (y[candidates] == 0).all():
        raise ValueError("candidate reservoir must contain eligible negative TRAIN rows")


def _progress(callback, **event):
    if callback:
        callback(**event)


class AuthorSPEARGenerator(nn.Module):
    """Interface-only delegation to Backdoor.inject_trigger, no copied math."""
    def __init__(self, backdoor, feature_path, record, device="cpu"):
        super().__init__()
        self.shadow_model = backdoor.shadow_model
        self.trojan = backdoor.trojan
        self.backdoor = backdoor
        self.feature_path = str(feature_path)
        self.record = record
        self.execution_device = torch.device(device)

    @torch.no_grad()
    def inject(self, x, edge_index, targets):
        device = self.execution_device
        self.backdoor.shadow_model = self.shadow_model.to(device)
        self.backdoor.trojan = self.trojan.to(device)
        self.backdoor.args.feature_importance_path = self.feature_path
        features, edges, indices = x.to(device), edge_index.to(device), targets.to(device)
        weights = torch.ones(edges.shape[1], dtype=features.dtype, device=device)
        with _precision(device):
            output = self.backdoor.inject_trigger(indices, features, edges, weights, device)
        # Native inject_trigger offloads its trojan to CPU. Restore residency
        # for the next call; neither author math nor returned values change.
        self.trojan.to(device)
        return tuple(value.detach().cpu() for value in output)


def fit_spear(x, edge_index, y, idx_train, idx_val, candidate_indices, *, seed, work_dir, config=None, progress=None, device="cpu"):
    """Invoke original select(), MLP.fit, SAGE and Backdoor.fit on supplied data."""
    config = validate_config(config)
    device = _device(device)
    _validate_inputs(x, edge_index, y, idx_train, idx_val, candidate_indices)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if config["alpha_int"] > x.shape[1] or len(candidate_indices) - config["vs_number"] < config["outter_size"]:
        raise ValueError("insufficient input feature or candidate/outer budget for native SPEAR configuration")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    for filename in ("feature_importance.npy", "fit_record.json", "sorted_samples/Cora/sorted_samples.pt"):
        if (work / filename).exists():
            raise FileExistsError(f"refusing to reuse cached author result: {work / filename}")
    api, runtime = import_author_runtime(work)
    feature_path = work / "feature_importance.npy"
    args = _arguments(config, seed, str(feature_path))
    device_x, device_edges, device_y = (value.to(device) for value in (x, edge_index, y))
    device_train, device_val = idx_train.to(device), idx_val.to(device)
    data = SimpleNamespace(x=device_x, edge_index=device_edges, y=device_y,
                           train_mask=torch.zeros(len(x), dtype=torch.bool, device=device),
                           test_mask=torch.ones(len(x), dtype=torch.bool, device=device))
    data.train_mask[idx_train] = True
    data.test_mask[idx_train] = False
    data.test_mask[candidate_indices] = False
    originals = dict(x=x, edges=edge_index, y=y, train=idx_train, dev=idx_val, candidates=candidate_indices)
    input_digest = _tensor_digest(originals)
    (work / "sorted_samples/Cora").mkdir(parents=True, exist_ok=False)
    rng_states = {}
    with _rng(seed, device), _precision(device), _cwd(work):
        rng_states["before_author_selector"] = _rng_digest(device)
        _progress(progress, phase="author_spear_selection", status="started")
        attachments = api["select_sample"].select(data, args, device_train, device_val, device).detach().cpu()
        if len(attachments) != config["vs_number"] or not torch.isin(attachments, candidate_indices).all():
            raise RuntimeError("author selector violated the supplied candidate budget")
        _progress(progress, phase="author_spear_feature_model", status="started")
        rng_states["before_author_feature_model"] = _rng_digest(device)
        feature_model = api["construct"].model_construct(args, "MLP", data, device).to(device)
        feature_initial_digest = _tensor_digest(feature_model.state_dict())
        feature_model.fit(device_x, device_y, device_train, device_val, train_iters=args.epochs, verbose=False)
        # Exactly the select_feature.py library calls, with known clean TRAIN
        # labels/rows supplied instead of the source's citation dataset loader.
        explanation_x = x[idx_train].numpy()
        explanation_y = torch.nn.functional.one_hot(y[idx_train], num_classes=2).numpy()
        imputer = api["sage"].MarginalImputer(feature_model, explanation_x[:args.sample_num])
        estimator = api["sage"].PermutationEstimator(imputer, "mse", random_state=seed)
        _progress(progress, phase="author_spear_sage", status="started")
        importance = estimator(explanation_x, explanation_y, batch_size=args.sage_batch_size,
            detect_convergence=args.sage_detect_convergence, thresh=args.sage_thresh,
            n_permutations=args.sage_n_permutations, bar=False)
        values, std = importance.save_num()
        if not np.isfinite(values).all() or not np.isfinite(std).all():
            raise RuntimeError("original author SAGE returned nonfinite values")
        np.save(feature_path, values)
        np.save(work / "feature_importance_std.npy", std)
        selected = np.argsort(values)[::-1][:args.alpha_int].copy()
        feature_digest = _tensor_digest(feature_model.state_dict())
        del feature_model
        # Preserve the author's main.py nonattachment-pool construction.
        unlabeled = torch.tensor(list(set(candidate_indices.tolist()) - set(attachments.tolist())), dtype=torch.long)
        attack = api["backdoor"].Backdoor(args, device)
        rng_states["before_author_backdoor_fit"] = _rng_digest(device)
        _progress(progress, phase="author_spear_backdoor_fit", status="started", epochs=args.trojan_epochs)
        attack.fit(device_x, device_edges, None, device_y, device_train, attachments.to(device), unlabeled.to(device))
        poisoned_x, poisoned_edges, poisoned_weights, poisoned_y = (value.detach().cpu() for value in attack.get_poisoned())
        attack.trojan.to(device)
        rng_states["after_author_backdoor_fit"] = _rng_digest(device)
    if _tensor_digest(originals) != input_digest:
        raise RuntimeError("author code modified caller input tensors")
    if not torch.equal(poisoned_edges, edge_index):
        raise RuntimeError("author SPEAR unexpectedly changed topology")
    allowed = torch.zeros_like(x, dtype=torch.bool)
    allowed[attachments[:, None], torch.as_tensor(selected)] = True
    if not torch.equal(poisoned_x[~allowed], x[~allowed]):
        raise RuntimeError("author SPEAR unexpectedly changed rows or coordinates outside its support")
    fit_indices = torch.cat((idx_train, attachments))
    record = dict(method="SPEAR_author_source", status="FIT_COMPLETE", source=runtime["source"],
        runtime_manifest=str(work / "author_runtime" / RUNTIME_MANIFEST), runtime_patch_sha256=runtime["patch_sha256"],
        config=config, seed=seed, sage_random_state=seed, input_sha256=input_digest,
        device=str(device), precision="float32; AMP disabled; TF32 disabled",
        cpu_components=["SAGE permutation/imputation NumPy orchestration; native model_conversion evaluates MLP on its configured device", "input/output validation and artifact serialization"],
        feature_importance_sha256=_sha(feature_path), feature_model_sha256=feature_digest,
        feature_model_initial_sha256=feature_initial_digest, rng_states_sha256=rng_states,
        initialization="all source constructors are fresh; no pretrained model or released selection/importance cache loaded",
        rng_policy="seeded CPU torch/NumPy and selected-device CUDA streams restored on exit; supported SAGE private random_state=seed; no seed offsets; CUDA scatter reductions are not promised bitwise deterministic",
        selection_sha256=_tensor_digest(dict(attachments=attachments, coordinates=torch.as_tensor(selected))),
        attachment_indices=attachments.tolist(), selected_coordinates=selected.tolist(),
        algorithms_reimplemented=False, author_calls=["select_sample.select", "models.MLP.MLP.fit", "sage_modified.MarginalImputer", "sage_modified.PermutationEstimator", "models.backdoor_GCN_cos.Backdoor.fit", "Backdoor.get_poisoned"],
        selector_and_feature_checkpoint="author best normal DEV accuracy, strict greater-than/earliest tie",
        generator_checkpoint="author minimum TRAIN outer loss: post-step G checkpoint with final shadow",
        generator_steps=args.trojan_epochs, surrogate_steps=args.trojan_epochs * args.inner,
        topology_changed=False, original_inputs_unchanged=True,
        raw_author_output=dict(get_poisoned="direct Backdoor.get_poisoned return; no repaired or projected features",
            inject="direct Backdoor.inject_trigger; original GCN neighbor embedding and absolute coordinate overwrite",
            edges="original cloned edge_index and unit edge weights; no topology edits",
            labels="original labels cloned and only attachment labels set to target_class",
            background_labels_preserved=True),
        added_current_method_modules=False, feature_explanation_indices=idx_train.tolist(),
        feature_background_indices=idx_train[:args.sample_num].tolist(),
        adaptations=["JCAS pair-line-graph tensor bridge with existing standardization", "fixed TRAIN/dev split and eligible-negative TRAIN candidate mask", f"{args.vs_number} poison attachments and target class {args.target_class}", "Cora is only an upstream file-routing key; no Cora data loaded", "fresh feature importance path instead of released citation-dataset cache", "SAGE explanation/background restricted to known clean TRAIN supervision", "explicit estimator random_state uses its author-supported seed API"],
        original_author_defaults=dict(alpha_int=30, trojan_epochs=400, epochs=200),
        native_feature_budget=args.alpha_int, native_generator_epochs=args.trojan_epochs)
    (work / "fit_record.json").write_text(json.dumps(record, indent=2) + "\n")
    generator = AuthorSPEARGenerator(attack, feature_path, record, device=device).eval()
    _progress(progress, phase="author_spear_complete", status="complete")
    return dict(generator=generator, attachment_indices=attachments, poisoned_x=poisoned_x,
                poisoned_edge_index=poisoned_edges, poisoned_edge_weight=poisoned_weights,
                poisoned_y=poisoned_y, fit_indices=fit_indices, record=record)


def fit_author_classifier(x, edge_index, edge_weight, y, idx_train, idx_val, *, seed, work_dir, config=None, progress=None, device="cpu"):
    """Fit actual author GraphSage using its native normal-DEV accuracy rule.

    Pass the SAME per-method/seed work_dir used by fit_spear. This reuses its
    verified runtime and absolute-import namespace, never a feature/sample cache.
    """
    config = validate_config(config)
    device = _device(device)
    _validate_inputs(x, edge_index, y, idx_train, idx_val)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    api, runtime = import_author_runtime(work_dir)
    args = _arguments(config, seed)
    device_x, device_edges, device_y = (value.to(device) for value in (x, edge_index, y))
    device_weight = None if edge_weight is None else edge_weight.to(device)
    with _rng(seed, device), _precision(device):
        model = api["construct"].model_construct(args, "GraphSage", SimpleNamespace(x=device_x, y=device_y), device).to(device)
        initial_digest = _tensor_digest(model.state_dict())
        fit_rng = _rng_digest(device)
        _progress(progress, phase="author_graphsage_fit", status="started", epochs=args.epochs)
        model.fit(device_x, device_edges, device_weight, device_y, idx_train.to(device), idx_val.to(device), train_iters=args.epochs, verbose=False)
        final_rng = _rng_digest(device)
    model.eval()
    record = dict(classifier="author models.SAGE.GraphSage", source_commit=SOURCE_COMMIT,
                  runtime_patch_sha256=runtime["patch_sha256"], seed=seed, epochs=args.epochs,
                  device=str(device), precision="float32; AMP disabled; TF32 disabled",
                  initial_state_sha256=initial_digest, fit_rng_sha256=fit_rng,
                  final_rng_sha256=final_rng, rng_devices=["cpu"] + ([str(device)] if device.type == "cuda" else []),
                  final_state_sha256=_tensor_digest(model.state_dict()),
                  checkpoint="original best clean DEV accuracy; strict greater-than, earliest tie; no AUC override")
    _progress(progress, phase="author_graphsage_fit", status="complete")
    return dict(model=model, record=record)
