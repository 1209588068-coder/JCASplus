"""Cached DEV-only risk-one to risk-zero evaluation for local AV2 research.

All models and the generator are supplied in memory, frozen and in eval mode.
The fixed positive population is independent of predictions. No graph reader,
checkpoint restoration, optimization, audit or confirmation workflow is called.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .development_diagnostic import TARGET_SEED, check_injection, row_metadata
from .evaluate import SceneInjector, state_hash
from .evaluation import _asset, _component, _normal_metrics
from .outer_evaluation import _fingerprint, _require_frozen, _rng_fingerprint
from .reverse_audit import IDENTITY_FIELDS, select_positive_targets
from .reverse_metrics import reverse_response_metrics
from .role_data import build_contexts


REQUIRED_MODELS = ("victim", "matched_clean", "label_only")
OPTIONAL_MODELS = ("historical_reference",)
METADATA_COLUMNS = list(IDENTITY_FIELDS) + ["original_split"]
STRATA = ("all", "slow", "moving")


def _view(bundle, diagnostic, targets):
    return dict(sources=bundle.config["roles"]["dev"],
        shuffle_seed=bundle.config["shuffle_seed"], targets=targets,
        **{key: diagnostic[key]["dev"] for key in ("parts", "combined", "metadata")})


def _validate(view):
    sources, parts, metadata = view["sources"], view["parts"], view["metadata"]
    combined, targets = view["combined"], view["targets"]
    if (not sources or len(parts) != len(sources)
            or len(combined["offsets"]) != len(sources)
            or not isinstance(targets, (list, tuple))):
        raise ValueError("Reverse DEV cached scene counts, offsets or target list are invalid")
    # These empty TRAIN collections authorize no TRAIN reads. The selector
    # validates only the supplied DEV identities and retains its frozen rule.
    expected_targets = select_positive_targets(dict(train=[], dev=metadata),
        dict(train=[], dev=sources), seed=TARGET_SEED)["dev"]
    if len(targets) != len(expected_targets):
        raise ValueError("Frozen positive DEV target population changed")
    for supplied, expected in zip(targets, expected_targets):
        if (not isinstance(supplied, Mapping)
                or any(key not in supplied or type(supplied[key]) is not type(expected[key])
                       or supplied[key] != expected[key] for key in IDENTITY_FIELDS)
                or supplied.get("original_split", "train") != "train"
                or supplied.get("split", "train") != "train"):
            raise ValueError("Frozen positive DEV target identity or metadata changed")
    by_scene = {row["scenario_id"]: row["pair_index"] for row in expected_targets}
    offset, scenes, expected_metadata = 0, [], []
    for source, part, supplied_offset in zip(sources, parts, combined["offsets"]):
        if (source.get("original_split", "train") != "train"
                or part.get("split") != "train" or part.get("role", "dev") != "dev"
                or part.get("original_split", "train") != "train"
                or part.get("scenario_id") != source["scenario_id"]
                or part.get("split_group_id", source["split_group_id"]) != source["split_group_id"]):
            raise ValueError("Reverse DEV cached scenario, role or original split mismatch")
        if (isinstance(supplied_offset, (bool, np.bool_))
                or not isinstance(supplied_offset, Integral) or supplied_offset != offset):
            raise ValueError("Reverse DEV cached offsets are misaligned")
        component = _component(part)
        rows = row_metadata(part, source, offset, "dev", set(), set())
        for expected, supplied in zip(rows, metadata[offset:offset + len(rows)]):
            if (any(supplied.get(key) != value for key, value in expected.items())
                    or supplied.get("original_split", "train") != "train"
                    or supplied.get("split", "train") != "train"):
                raise ValueError("Reverse DEV metadata differs from cached pair identity or labels")
        expected_metadata.extend(dict(row, original_split="train") for row in rows)
        scenes.append((part, offset, by_scene.get(source["scenario_id"]), component))
        offset += len(rows)
    if len(metadata) != offset:
        raise ValueError("Reverse DEV cached metadata population changed")
    build_contexts(parts, combined, seed=view["shuffle_seed"])
    return scenes, expected_metadata


def _predict(model, x, edges, weights):
    _require_frozen(dict(model=model))
    inputs = dict(x=x, edges=edges, weights=weights)
    before = _fingerprint(inputs)
    with torch.no_grad():
        output = model(x, edges, weights)
    if before != _fingerprint(inputs):
        raise ValueError("Reverse inference mutated a shared prediction input")
    if (output.shape != (len(x), 2) or not bool(torch.isfinite(output).all())
            or not torch.allclose(output.logsumexp(1), output.new_zeros(len(x)), atol=1e-5)):
        raise ValueError("Reverse evaluation requires finite binary log-probabilities")
    # Take NLL from logp0 directly: 1-exp(logp1) loses precision near p1=1.
    return dict(p=output[:, 1].exp().cpu().numpy(), pred=output.argmax(1).cpu().numpy(),
                nll=(-output[:, 0]).cpu().numpy())


def _responses(row, index, names, clean, triggered):
    result = dict(row)
    for name in names:
        result.update({name + "_clean_p": float(clean[name]["p"][index]),
            name + "_trigger_p": float(triggered[name]["p"][index]),
            name + "_clean_pred": int(clean[name]["pred"][index]),
            name + "_trigger_pred": int(triggered[name]["pred"][index]),
            name + "_target_nll": float(triggered[name]["nll"][index])})
    return result


def evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output, progress=None):
    """Evaluate cached DEV positives; exclusively create three CSVs and summary.

    ``targets`` is the frozen DEV metadata list from reverse_audit, in registered
    source order, with one positive per scene that has an eligible positive.
    Models are victim/matched_clean/label_only, plus optional historical_reference.
    All models share ONE generated scene input per target. They must already be
    eval/frozen with cleared gradients. ``output`` must not exist, even if empty.
    Progress accepts stage/completed/total keyword arguments; failures propagate.
    """
    if (not set(REQUIRED_MODELS).issubset(models)
            or set(models) - set(REQUIRED_MODELS + OPTIONAL_MODELS)):
        raise ValueError("Require victim, matched_clean, label_only and optional historical_reference")
    names = tuple(name for name in REQUIRED_MODELS + OPTIONAL_MODELS if name in models)
    all_models = dict(generator=generator, **models)
    _require_frozen(all_models)
    view = _view(bundle, diagnostic, targets)
    before_inputs, before_rng = _fingerprint(view), _rng_fingerprint()
    before_models = {name: state_hash(model) for name, model in all_models.items()}
    scenes, metadata = _validate(view)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = dict(targets=output / "targets.csv", normal=output / "clean_pair_scores.csv",
                 triggered=output / "triggered_pair_scores.csv")
    responses = [name + suffix for name in names for suffix in
                 ("_clean_p", "_trigger_p", "_clean_pred", "_trigger_pred", "_target_nll")]
    target_columns = METADATA_COLUMNS + ["changed_coordinates", "boundary_coordinates",
                                        "delta_l1", "delta_linf"] + responses
    clean_columns = METADATA_COLUMNS + ["label"] + [name + suffix for name in names
                                                              for suffix in ("_p", "_pred")]
    trigger_columns = METADATA_COLUMNS + ["target_pair_index", "label", "relation"] + responses
    counts = dict(registered_scenes=len(scenes), processed_scenes=0,
        scenes_without_eligible_positive_target=0, all_pair_nodes=0, supervised_pairs=0,
        targets=0, triggered_supervised_pair_rows=0)
    collateral = {name: {group: dict(positive_pairs=0, induced_misses_1to0=0,
        negative_pairs=0, negative_activations_0to1=0) for group in ("adjacent", "nonincident")}
        for name in names}
    changes = dict(target_rows_changed=0, target_rows_unchanged=0,
                   maximum_changed_coordinates_per_target=0)
    normal_rows, target_rows, triggered_identity = [], [], []
    injector = SceneInjector(generator, view["shuffle_seed"])
    report = progress if progress is not None else lambda **event: None
    stage = "reverse_dev_original_TRAIN_groups"
    report(stage=stage, completed=0, total=len(scenes))
    with paths["targets"].open("x", newline="") as target_stream, \
            paths["normal"].open("x", newline="") as clean_stream, \
            paths["triggered"].open("x", newline="") as trigger_stream, torch.no_grad():
        target_writer = csv.DictWriter(target_stream, fieldnames=target_columns)
        clean_writer = csv.DictWriter(clean_stream, fieldnames=clean_columns)
        trigger_writer = csv.DictWriter(trigger_stream, fieldnames=trigger_columns)
        for writer in (target_writer, clean_writer, trigger_writer):
            writer.writeheader()
        for completed, (part, offset, target, component) in enumerate(scenes, 1):
            x, edges, truth, eligible, _, _ = component
            weights = torch.ones(edges.shape[1], dtype=torch.float32)
            clean = {name: _predict(models[name], x, edges, weights) for name in names}
            counts["processed_scenes"] += 1
            counts["all_pair_nodes"] += len(x)
            counts["supervised_pairs"] += int(eligible.sum())
            for i in np.flatnonzero(eligible):
                row = dict(metadata[offset + i], label=int(truth[i]))
                for name in names:
                    row.update({name + "_p": float(clean[name]["p"][i]),
                                name + "_pred": int(clean[name]["pred"][i])})
                normal_rows.append(row)
                clean_writer.writerow(row)
            if target is None:
                counts["scenes_without_eligible_positive_target"] += 1
            else:
                injector.bind(part)
                target_indices = torch.tensor([target], dtype=torch.long)
                changed, new_edges, new_weights = injector.inject_trigger(target_indices,
                    x.clone(), edges.clone(), weights.clone(), torch.device("cpu"))
                check_injection(x, changed, target_indices, generator, edges, new_edges, weights, new_weights)
                triggered = {name: _predict(models[name], changed, new_edges, new_weights) for name in names}
                row = _responses(metadata[offset + target], target, names, clean, triggered)
                support = generator.selected_coordinates[generator.hard_support]
                after = changed[target, support].double()
                delta = (after - x[target, support].double()).abs()
                row.update(changed_coordinates=int((delta > 0).sum()),
                    boundary_coordinates=int((after.abs() == 10).sum()),
                    delta_l1=float(delta.sum()), delta_linf=float(delta.max()) if delta.numel() else 0.)
                target_rows.append(row)
                target_writer.writerow(row)
                counts["targets"] += 1
                changed_count = row["changed_coordinates"]
                changes["target_rows_changed" if changed_count else "target_rows_unchanged"] += 1
                changes["maximum_changed_coordinates_per_target"] = max(
                    changes["maximum_changed_coordinates_per_target"], changed_count)
                adjacent = np.zeros(len(x), dtype=bool)
                neighbors = set(edges[1, edges[0] == target].tolist()) | set(edges[0, edges[1] == target].tolist())
                adjacent[list(neighbors)] = True
                adjacent[target] = False
                nonincident = ~adjacent
                nonincident[target] = False
                for name in names:
                    before, after = clean[name]["pred"], triggered[name]["pred"]
                    for group, mask in (("adjacent", adjacent), ("nonincident", nonincident)):
                        positive, negative = mask & eligible & (truth == 1), mask & eligible & (truth == 0)
                        d = collateral[name][group]
                        d["positive_pairs"] += int(positive.sum())
                        d["negative_pairs"] += int(negative.sum())
                        d["induced_misses_1to0"] += int((positive & (before == 1) & (after == 0)).sum())
                        d["negative_activations_0to1"] += int((negative & (before == 0) & (after == 1)).sum())
                for i in np.flatnonzero(eligible):
                    row = dict(metadata[offset + i], target_pair_index=int(target), label=int(truth[i]),
                        relation="target" if i == target else "adjacent" if adjacent[i] else "nonincident")
                    triggered_identity.append(dict(row))
                    trigger_writer.writerow(_responses(row, i, names, clean, triggered))
                    counts["triggered_supervised_pair_rows"] += 1
            if completed % 16 == 0 or completed == len(scenes):
                for stream in (target_stream, clean_stream, trigger_stream):
                    stream.flush()
                report(stage=stage, completed=completed, total=len(scenes))
    _require_frozen(all_models)
    if before_models != {name: state_hash(model) for name, model in all_models.items()}:
        raise ValueError("Reverse evaluation mutated model state")
    if before_inputs != _fingerprint(_view(bundle, diagnostic, targets)):
        raise ValueError("Reverse evaluation mutated cached inputs or frozen targets")
    if before_rng != _rng_fingerprint():
        raise ValueError("Reverse evaluation consumed inference RNG")
    metrics = {}
    for name in names:
        for d in collateral[name].values():
            d["positive_suppression"] = d["induced_misses_1to0"] / d["positive_pairs"] if d["positive_pairs"] else None
            d["negative_activation"] = d["negative_activations_0to1"] / d["negative_pairs"] if d["negative_pairs"] else None
        losses = {}
        for stratum in STRATA:
            values = [row[name + "_target_nll"] for row in target_rows
                      if stratum == "all" or row["motion_regime"] == stratum]
            losses[stratum] = dict(n=len(values), mean_nll_class_0=
                math.fsum(value / len(values) for value in values) if values else None)
        metrics[name] = dict(targets=reverse_response_metrics(target_rows, name),
            **_normal_metrics(normal_rows, name), target_loss=losses, collateral=collateral[name])
    population_identity = dict(scenes=_fingerprint(view["sources"]),
        eligible_clean_pairs=_fingerprint([{key: row[key] for key in METADATA_COLUMNS} for row in normal_rows]),
        targets=_fingerprint([{key: row[key] for key in METADATA_COLUMNS} for row in target_rows]),
        triggered_pairs=_fingerprint(triggered_identity))
    result = dict(status="REVERSE_DEV_EVALUATION_COMPLETE", target_class=0,
        scope="cached original TRAIN DEV; reverse development evidence only",
        metrics=metrics, dev_counts=counts, population_identity_sha256=population_identity,
        target_identity_sha256=population_identity["targets"],
        raw_targets=dict(**_asset(paths["targets"]), rows=len(target_rows)),
        raw_clean_scores=dict(**_asset(paths["normal"]), rows=len(normal_rows)),
        raw_triggered_pair_scores=dict(**_asset(paths["triggered"]), rows=counts["triggered_supervised_pair_rows"]),
        weights_unchanged=True, inputs_unchanged=True, rng_unchanged=True, optimizer_steps=0,
        gradients_created=False, topology_preserved=True, non_target_input_feature_rows_preserved=True,
        feature_changes=changes, target_selection_uses_predictions=False, target_seed=TARGET_SEED,
        shared_injection_per_target_scene=True, generator_injections=counts["targets"],
        threshold="binary model argmax; no threshold calibration",
        probability_class=1, target_nll_class=0,
        metric_definitions=dict(
            incremental_asr="same-model clean-one to triggered-zero transitions / all frozen eligible positive targets",
            absolute_suppression="triggered class-zero predictions / all frozen eligible positive targets",
            conditional_suppression="induced misses / clean-detected fixed positive targets",
            net_fnr_change="(induced misses minus corrected baseline misses) / all frozen positive targets",
            collateral="per-target-scene pooled non-target transitions; denominators include all eligible true-label members of each adjacent/nonincident cohort"),
        graph_arrays_loaded=False, model_checkpoints_loaded=False,
        original_val_arrays_loaded=False, original_test_arrays_loaded=False)
    with (output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result
