"""Pure cached-DEV evaluation for the offline, owned-model AV2 outer pilot.

The caller supplies already trained, frozen models and cached original-TRAIN
DEV inputs. This module never loads graphs/checkpoints or fits any model. DEV
remains development evidence; it is not relabeled as audit or confirmation.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import random

import numpy as np
import torch

from . import development_diagnostic as diagnostic
from .evaluate import SceneInjector, state_hash
from .evaluation import _asset, _component, _normal_metrics, _target_metrics
from .role_data import build_contexts, _tensor_hash


MODELS = diagnostic.MODELS
METADATA_COLUMNS = ["scenario_id", "split_group_id", "original_split", "role",
    "pair_index", "role_pair_index", "src_track_id", "dst_track_id", "true_label",
    "eligible", "motion_regime", "is_attachment", "is_outer"]
RESPONSE_COLUMNS = [name + suffix for name in MODELS for suffix in
    ("_clean_p", "_trigger_p", "_clean_pred", "_trigger_pred", "_target_nll")]
TARGET_COLUMNS = METADATA_COLUMNS + ["changed_coordinates", "boundary_coordinates",
    "delta_l1", "delta_linf"] + RESPONSE_COLUMNS
CLEAN_COLUMNS = METADATA_COLUMNS + ["label"] + [name + suffix for name in MODELS
    for suffix in ("_p", "_pred")]
TRIGGER_COLUMNS = METADATA_COLUMNS + ["target_pair_index", "label", "relation"] + RESPONSE_COLUMNS


def _fingerprint(value):
    """Hash cached values without inspecting any other data role or loading assets."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            if (item.is_leaf or item.retains_grad) and item.grad is not None:
                raise ValueError("DEV inference inputs must have no gradients")
            digest.update(b"tensor" + _tensor_hash(item).encode())
            digest.update(str((item.requires_grad, str(item.device))).encode())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            if item.dtype.hasobject:
                visit(item.tolist())
            else:
                digest.update(item.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for element in item:
                visit(element)
            digest.update(b"]")
        else:
            if isinstance(item, np.generic):
                item = item.item()
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b";")

    visit(value)
    return digest.hexdigest()


def _dev_view(bundle, inputs):
    # Deliberately do not traverse inputs' TRAIN/context/populations or other roles.
    return dict(sources=bundle.config["roles"]["dev"],
        shuffle_seed=bundle.config["shuffle_seed"],
        **{key: inputs[key]["dev"] for key in ("parts", "combined", "metadata", "targets")})


def _rng_fingerprint():
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return _fingerprint(state)


def _require_frozen(models):
    for name, model in models.items():
        if (any(module.training for module in model.modules())
                or any(parameter.requires_grad or parameter.grad is not None
                       for parameter in model.parameters())):
            raise ValueError("Require frozen eval models with no gradients: " + name)


def _validate(view):
    """Validate every scene and fixed target before the first model prediction."""
    sources, parts = view["sources"], view["parts"]
    metadata, targets, combined = view["metadata"], view["targets"], view["combined"]
    if not sources or len(parts) != len(sources) or len(targets) != len(sources):
        raise ValueError("Cached DEV scene counts differ from the registered population")
    if len(combined["offsets"]) != len(sources):
        raise ValueError("Cached DEV scene offsets are misaligned")
    seen_scenes, seen_groups, scenes, expected_rows = set(), set(), [], []
    offset = 0
    for source, part, target, supplied_offset in zip(sources, parts, targets, combined["offsets"]):
        scenario, group = source.get("scenario_id"), source.get("split_group_id")
        if (source.get("split") != "train" or source.get("role") != "dev"
                or not isinstance(scenario, str) or not scenario.strip()
                or not isinstance(group, str) or not group.strip()
                or scenario in seen_scenes or group in seen_groups):
            raise ValueError("Require unique registered original-TRAIN DEV identities")
        seen_scenes.add(scenario)
        seen_groups.add(group)
        if (part.get("scenario_id") != scenario or part.get("split") != "train"
                or part.get("role", "dev") != "dev"
                or part.get("original_split", "train") != "train"
                or source.get("original_split", "train") != "train"):
            raise ValueError("Cached DEV scenario identity, original split or role mismatch")
        if (isinstance(supplied_offset, (bool, np.bool_))
                or not isinstance(supplied_offset, Integral) or supplied_offset != offset):
            raise ValueError("Cached DEV scene offsets are misaligned")
        component = _component(part)
        n = len(component[0])
        chosen = diagnostic.choose_target(part)
        if ((target is not None and (isinstance(target, (bool, np.bool_))
                                    or not isinstance(target, Integral))) or target != chosen):
            raise ValueError("Cached DEV target differs from the fixed identity-hash target")
        rows = diagnostic.row_metadata(part, source, offset, "dev", set(), set())
        for expected, supplied in zip(rows, metadata[offset:offset + n]):
            if (not isinstance(supplied, Mapping)
                    or any(supplied.get(key) != value for key, value in expected.items())
                    or supplied.get("original_split", "train") != "train"
                    or supplied.get("split", "train") != "train"):
                raise ValueError("Cached DEV metadata identity, label, eligibility, motion or role mismatch")
        expected_rows.extend(dict(row, original_split="train") for row in rows)
        scenes.append((part, offset, target, component))
        offset += n
    if len(metadata) != offset:
        raise ValueError("Cached DEV metadata count differs from pair population")
    # This also checks canonical pair identity/topology and exact combined tensors.
    build_contexts(parts, combined, seed=view["shuffle_seed"])
    return scenes, expected_rows


def evaluate_dev(bundle, inputs, generator, models, directory, progress=None):
    """Evaluate frozen in-memory models on cached DEV; return JSON-safe metrics.

    ``directory`` must not exist, even if empty. The three CSV files are created
    exclusively. ``progress``, when supplied, accepts stage/completed/total
    keyword arguments. Failures propagate to the owning pilot runner.
    """
    if set(models) != set(MODELS):
        raise ValueError("Require exactly surrogate, victim and clean_reference models")
    all_models = dict(generator=generator, **models)
    _require_frozen(all_models)
    view = _dev_view(bundle, inputs)
    before_inputs, before_rng = _fingerprint(view), _rng_fingerprint()
    before_models = {name: state_hash(model) for name, model in all_models.items()}
    scenes, metadata = _validate(view)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths = dict(targets=directory / "targets.csv", normal=directory / "clean_pair_scores.csv",
                 triggered=directory / "triggered_pair_scores.csv")
    counts = dict(registered_scenes=len(scenes), processed_scenes=0,
        scenes_without_eligible_negative_target=0, all_pair_nodes=0, supervised_pairs=0,
        targets=0, triggered_supervised_pair_rows=0)
    collateral = {name: dict.fromkeys(("adj_negative", "adj_flip", "nonincident_negative",
        "nonincident_flip", "adj_positive", "adj_suppressed"), 0) for name in MODELS}
    changes = dict(target_rows_changed=0, target_rows_unchanged=0,
                   maximum_changed_coordinates_per_target=0)
    normal_rows, target_rows, triggered_identity = [], [], []
    attack = SceneInjector(generator, view["shuffle_seed"])
    report = progress if progress is not None else lambda **record: None
    stage = "dev_original_TRAIN_groups"
    report(stage=stage, completed=0, total=len(scenes))
    with paths["targets"].open("x", newline="") as target_stream, \
            paths["normal"].open("x", newline="") as clean_stream, \
            paths["triggered"].open("x", newline="") as trigger_stream, torch.no_grad():
        target_writer = csv.DictWriter(target_stream, fieldnames=TARGET_COLUMNS)
        clean_writer = csv.DictWriter(clean_stream, fieldnames=CLEAN_COLUMNS)
        trigger_writer = csv.DictWriter(trigger_stream, fieldnames=TRIGGER_COLUMNS)
        for writer in (target_writer, clean_writer, trigger_writer):
            writer.writeheader()
        for completed, (part, offset, target, component) in enumerate(scenes, 1):
            x, edges, truth, eligible, _, _ = component
            weights = torch.ones(edges.shape[1], dtype=torch.float32)
            clean = {name: diagnostic.predict(models[name], x, edges, weights) for name in MODELS}
            counts["processed_scenes"] += 1
            counts["all_pair_nodes"] += len(x)
            counts["supervised_pairs"] += int(eligible.sum())
            for i in np.flatnonzero(eligible):
                row = dict(metadata[offset + i], label=int(truth[i]))
                for name in MODELS:
                    row.update({name + "_p": float(clean[name]["p"][i]),
                                name + "_pred": int(clean[name]["pred"][i])})
                normal_rows.append(row)
                clean_writer.writerow(row)
            if target is None:
                counts["scenes_without_eligible_negative_target"] += 1
            else:
                attack.bind(part)
                target_indices = torch.tensor([target], dtype=torch.long)
                changed, new_edges, new_weights = attack.inject_trigger(target_indices,
                    x.clone(), edges.clone(), weights.clone(), torch.device("cpu"))
                diagnostic.check_injection(x, changed, target_indices, generator,
                    edges, new_edges, weights, new_weights)
                triggered = {name: diagnostic.predict(models[name], changed, new_edges, new_weights)
                             for name in MODELS}
                row = diagnostic.response_row(metadata[offset + target], target,
                                               x, changed, generator, clean, triggered)
                target_rows.append(row)
                target_writer.writerow(row)
                counts["targets"] += 1
                changed_count = row["changed_coordinates"]
                changes["target_rows_changed" if changed_count else "target_rows_unchanged"] += 1
                changes["maximum_changed_coordinates_per_target"] = max(
                    changes["maximum_changed_coordinates_per_target"], changed_count)
                adjacency = np.zeros(len(x), dtype=bool)
                neighbors = set(edges[1, edges[0] == target].tolist()) | set(edges[0, edges[1] == target].tolist())
                adjacency[list(neighbors)] = True
                adjacency[target] = False
                nonincident = ~adjacency
                nonincident[target] = False
                negative, positive = eligible & (truth == 0), eligible & (truth == 1)
                for name in MODELS:
                    before, after = clean[name]["pred"], triggered[name]["pred"]
                    d = collateral[name]
                    d["adj_negative"] += int((adjacency & negative).sum())
                    d["adj_flip"] += int((adjacency & negative & (before == 0) & (after == 1)).sum())
                    d["nonincident_negative"] += int((nonincident & negative).sum())
                    d["nonincident_flip"] += int((nonincident & negative & (before == 0) & (after == 1)).sum())
                    d["adj_positive"] += int((adjacency & positive).sum())
                    d["adj_suppressed"] += int((adjacency & positive & (before == 1) & (after == 0)).sum())
                for i in np.flatnonzero(eligible):
                    row = dict(metadata[offset + i], target_pair_index=int(target), label=int(truth[i]),
                        relation="target" if i == target else "adjacent" if adjacency[i] else "nonincident")
                    triggered_identity.append(dict(row))
                    for name in MODELS:
                        row.update({name + "_clean_p": float(clean[name]["p"][i]),
                            name + "_trigger_p": float(triggered[name]["p"][i]),
                            name + "_clean_pred": int(clean[name]["pred"][i]),
                            name + "_trigger_pred": int(triggered[name]["pred"][i]),
                            name + "_target_nll": float(triggered[name]["nll"][i])})
                    trigger_writer.writerow(row)
                    counts["triggered_supervised_pair_rows"] += 1
            if completed % 16 == 0 or completed == len(scenes):
                for stream in (target_stream, clean_stream, trigger_stream):
                    stream.flush()
                report(stage=stage, completed=completed, total=len(scenes))
    _require_frozen(all_models)
    if before_models != {name: state_hash(model) for name, model in all_models.items()}:
        raise ValueError("DEV evaluation mutated model state")
    if before_inputs != _fingerprint(_dev_view(bundle, inputs)):
        raise ValueError("DEV evaluation mutated cached inputs")
    if before_rng != _rng_fingerprint():
        raise ValueError("DEV evaluation consumed inference RNG")
    metrics = {}
    for name in MODELS:
        d = collateral[name]
        metrics[name] = dict(targets=_target_metrics(target_rows, name), **_normal_metrics(normal_rows, name),
            collateral=dict(**d,
                adjacent_incremental_fp=d["adj_flip"] / d["adj_negative"] if d["adj_negative"] else None,
                nonincident_incremental_fp=d["nonincident_flip"] / d["nonincident_negative"] if d["nonincident_negative"] else None,
                adjacent_positive_suppression=d["adj_suppressed"] / d["adj_positive"] if d["adj_positive"] else None))
    population_identity = dict(scenes=_fingerprint(view["sources"]),
        eligible_clean_pairs=_fingerprint([{key: row[key] for key in METADATA_COLUMNS} for row in normal_rows]),
        targets=_fingerprint([{key: row[key] for key in METADATA_COLUMNS} for row in target_rows]),
        triggered_pairs=_fingerprint(triggered_identity))
    return dict(status="DEV_EVALUATION_COMPLETE", scope="cached original TRAIN DEV; development evidence only",
        metrics=metrics, dev_counts=counts,
        population_identity_sha256=population_identity, target_identity_sha256=population_identity["targets"],
        raw_targets=dict(**_asset(paths["targets"]), rows=len(target_rows)),
        raw_clean_scores=dict(**_asset(paths["normal"]), rows=len(normal_rows)),
        raw_triggered_pair_scores=dict(**_asset(paths["triggered"]), rows=counts["triggered_supervised_pair_rows"]),
        weights_unchanged=True, inputs_unchanged=True, rng_unchanged=True, optimizer_steps=0,
        gradients_created=False, topology_preserved=True, non_target_input_feature_rows_preserved=True,
        feature_changes=changes, target_selection_uses_predictions=False,
        target_seed=diagnostic.TARGET_SEED,
        target_hash="SHA256(seed:spear-audit-target:scenario_id:canonical_src_track_id:canonical_dst_track_id)",
        threshold="supplied binary model argmax; no threshold calibration",
        metric_definitions=dict(
            absolute_asr="triggered class-one predictions / all frozen eligible true-negative targets",
            incremental_asr="same-model clean-zero to triggered-one transitions / all frozen targets",
            conditional_flip="same transitions / clean-zero targets",
            collateral="per-target-scene pooled transitions among eligible non-target neighbors or nonincident pairs; denominators include all true-label members"),
        graph_arrays_loaded=False, model_checkpoints_loaded=False,
        original_val_arrays_loaded=False, original_test_arrays_loaded=False)
