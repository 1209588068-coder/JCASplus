"""Train-only, fixed-checkpoint local gradient diagnosis; never an optimizer step.

This is a single-scene, eval-mode surrogate, NOT a replay of historical mixed
batches, dropout, clipping, or AdamW. Historical STOP and source bindings stay put.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jcas.core.counterfactual_gradients import (
    CleanProjectionConfig, direction_stats, flatten_gradients, project_extra_gradient,
)
from jcas.core.motion_normalized_features import MOTION_NORMALIZED_EDGE_FEATURE_MODE
from jcas.core.poison import sha256_file
from jcas.core.risk_labels import RiskLabelConfig
from jcas.workflows import explicit_counterfactual_trainer as training
from jcas.workflows import rotation_target_response_audit as response
from jcas.workflows.trainer import load_split_graphs

ROOT = response.ROOT
RUN = response.RUN
VERSION = "rotation_local_gradient_interference_v1"
SEED = 20260910
SCENES_PER_STRATUM = 32
MODEL_ROLES = ("rotating_v4", "fixed_v3")
COMPLETED = RUN / "diagnostics/additional_targets_c4sii0a5/summary.json"
COMPLETED_SHA256 = "a43a72767159722d3605c79ec17d3a4480a78b461c6f76f97a18973cef8a858c"
WEIGHTS = {
    "base_bce": 1., "target_pair_bce": .1, "ranking": .1,
    "non_target_consistency": 1., "adjacent_negative_consistency": 5.,
}
COMPONENTS = (*WEIGHTS, "clean_reference_bce", "total_unprojected",
              "extra_unprojected", "extra_projected", "total_projected",
              "probe_pair_consistency_contribution", "probe_pair_adjacent_contribution")
SCOPE = {
    "split": "train", "training": False, "optimizer_created": False,
    "parameters_updated": False, "model_mode": "eval",
    "validation_graphs_read": 0, "test_graphs_read": 0,
    "historical_training_replay": False, "projection_context": "one_triggered_scene",
    "original_training_gate_status": "STOP",
}
LIMITS = [
    "Single-scene eval-mode gradients, before gradient clipping and AdamW; not historical mixed-batch updates.",
    "Projection is nonlinear and depends on mixed-batch composition; local projection results cannot identify historical suppression.",
    "Positive dot with a probe-loss gradient predicts decreasing that loss under an infinitesimal raw-gradient descent; negative predicts increasing it.",
    "A probe's own-trigger input differs from other-target-trigger inputs; role changes are not automatically contradictory labels.",
    "Probe-pair consistency contributions are subsets of their parent components, not extra terms to add to the objective.",
    "Schedule-weighted statistics use v4 exposure counts at a fixed endpoint, including for the v3 comparator; not accumulated training gradients.",
    "Balanced 2/3/4-slot scene sample is descriptive, not a population estimate, new gate, or validation result.",
]


def exposure_table(targets, epoch):
    """Count presentation opportunities, not gradient strengths, without graph I/O."""
    if epoch < 1 or int(epoch) != epoch or targets.empty:
        raise ValueError("invalid epoch or empty target table")
    if not targets.role.eq("train").all() or not targets.split.eq("train").all():
        raise ValueError("exposures require train-role train targets only")
    records = []
    for scene, frame in targets.groupby("scenario_id", sort=True):
        slots = sorted(frame.slot.tolist())
        if slots != list(range(len(frame))):
            raise ValueError("training slots must be contiguous from zero")
        keys = [response.pair_key(r) for r in frame.to_dict("records")]
        if len(set(keys)) != len(frame):
            raise ValueError("duplicate unordered training pair")
        counts = {slot: sum((ep - 1) % len(frame) == slot for ep in range(1, epoch + 1)) for slot in slots}
        rows = frame.sort_values("slot").to_dict("records")
        for row in rows:
            adjacent = sum(counts[r["slot"]] for r in rows if r["slot"] != row["slot"]
                           and {r["src"], r["dst"]} & {row["src"], row["dst"]})
            own = counts[row["slot"]]
            records.append({
                **row, "training_slots_in_scene": len(frame),
                "v4_positive_exposures_to_checkpoint": own,
                "v4_other_trigger_consistency_presentations": epoch - own,
                "v4_other_trigger_adjacent_presentations": adjacent,
                "v4_other_trigger_nonincident_presentations": epoch - own - adjacent,
                "v4_ordinary_clean_presentations": epoch,
                "v4_clean_counterpart_presentations": epoch,
            })
    return pd.DataFrame(records).sort_values(["scenario_id", "slot"]).reset_index(drop=True)


def select_sample(exposures, per_stratum=SCENES_PER_STRATUM):
    if per_stratum < 1:
        raise ValueError("sample size must be positive")

    def rank(*parts):
        return hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).hexdigest()

    selected = []
    for n in (2, 3, 4):
        scenes = sorted(exposures.loc[exposures.training_slots_in_scene.eq(n), "scenario_id"].unique(),
                        key=lambda scene: (rank("scene", n, scene), scene))
        if len(scenes) < per_stratum:
            raise ValueError(f"not enough {n}-slot scenes")
        selected.extend(scenes[:per_stratum])
    sources = exposures[exposures.scenario_id.isin(selected)].sort_values(["scenario_id", "slot"]).copy()
    probes = []
    for scene, rows in sources.groupby("scenario_id", sort=True):
        anchor = rows[rows.slot.eq(0)].iloc[0].to_dict()
        alternatives = rows[rows.slot.gt(0)].to_dict("records")
        new = min(alternatives, key=lambda r: (rank("probe", scene, min(r["src"], r["dst"]), max(r["src"], r["dst"])), r["slot"]))
        probes.extend([{**anchor, "probe_kind": "anchor"}, {**new, "probe_kind": "additional"}])
    return sources.reset_index(drop=True), pd.DataFrame(probes).reset_index(drop=True)


def preflight():
    context = response.preflight()
    bound = context["bound"]
    bound["completed_response_summary"] = response.fixed.bind_file(COMPLETED, COMPLETED_SHA256)
    completed = response.fixed.read_json(COMPLETED)
    if (completed["status"] != "AUDIT_COMPLETE_REVIEW_REQUIRED"
            or completed["scope"] != response.SCOPE
            or completed["selected_v4_checkpoint_epoch"] != context["epoch"]
            or completed["threshold"] != context["threshold"]):
        raise ValueError("completed response audit identity differs")
    for name, asset in completed["outputs"].items():
        bound["completed_" + name] = response.fixed.bind_file(Path(asset["path"]), asset["sha256"])
    protocol = response.fixed.read_json(Path(completed["outputs"]["protocol.json"]["path"]))
    if protocol["models"] != context["models"]:
        raise ValueError("response audit used different checkpoints")
    for name, digest in protocol["sources"].items():
        response.fixed.bind_file(ROOT / name, digest)
    targets = pd.concat([context["groups"]["additional_train"], context["groups"]["rotatable_anchors"]], ignore_index=True)
    exposures = exposure_table(targets, context["epoch"])
    sources, probes = select_sample(exposures)
    if (len(exposures), len(sources), len(probes), sources.scenario_id.nunique()) != (3452, 288, 192, 96):
        raise ValueError("fixed diagnostic coverage changed")
    context.update(exposures=exposures, sampled_sources=sources, probes=probes)
    context["models"] = {role: context["models"][role] for role in MODEL_ROLES}
    context["sources"][str(Path(__file__).resolve().relative_to(ROOT))] = sha256_file(__file__)
    print("[rotation_gradient] preflight complete: 96 scenes, 192 probes, 288 source interventions", flush=True)
    return context


def pair_mask(graph, row):
    src, dst = int(row.src), int(row.dst)
    edge = graph.edge_index
    mask = ((edge[0] == src) & (edge[1] == dst)) | ((edge[0] == dst) & (edge[1] == src))
    directions = {tuple(v) for v in edge[:, mask].T.tolist()}
    if int(mask.sum()) != 2 or directions != {(src, dst), (dst, src)}:
        raise ValueError("probe lacks two unique directed edges")
    if not graph.supervision_edge_mask[mask].all():
        raise ValueError("probe is outside strict supervision")
    return mask


def load_scene(rows, graph_rows, graph_dir):
    if rows.empty or rows.scenario_id.nunique() != 1 or not rows.role.eq("train").all() or not rows.split.eq("train").all():
        raise ValueError("source interventions must be one train-only scene")
    scene = str(rows.scenario_id.iloc[0])
    gr = graph_rows.get(scene)
    if gr is None or gr.split != "train" or not rows.graph_sha256.eq(gr.graph_sha256).all():
        raise ValueError("source scene split/SHA mismatch")
    manifest = pd.DataFrame([gr._asdict()])

    def load(poison):
        graphs, stats = load_split_graphs(
            manifest, graph_dir, "train", True, None, RiskLabelConfig(), poison, True,
            MOTION_NORMALIZED_EDGE_FEATURE_MODE, None,
        )
        if len(graphs) != 1 or stats["graph_sha256_verified"] != 1 or stats["poisoned_graphs"] != int(poison is not None):
            raise ValueError("incomplete source graph loading")
        return graphs[0]

    clean = load(None)
    items = {}
    for _, row in rows.sort_values("slot").iterrows():
        triggered = load(pd.DataFrame([row]))
        presentation, _ = training.compose_counterfactual_training_items([clean], [triggered])
        mask = pair_mask(clean, row)
        if (not torch.equal(mask, triggered.poisoned_edge_mask)
                or not torch.equal(clean.supervision_edge_mask, triggered.supervision_edge_mask)
                or not torch.equal(clean.message_passing_edge_mask, triggered.message_passing_edge_mask)):
            raise ValueError("source intervention changed identity/topology/masks")
        items[int(row.slot)] = presentation[-1]
    return clean, items


def flat_gradient(loss, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True) if loss.requires_grad else [None] * len(parameters)
    vector = flatten_gradients(gradients, parameters)
    if not torch.isfinite(vector).all():
        raise FloatingPointError("nonfinite gradient")
    return vector


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def check_state(model, before):
    after = model.state_dict()
    if set(after) != set(before) or any(not torch.equal(after[k].detach().cpu(), v) for k, v in before.items()):
        raise RuntimeError("model parameters/buffers changed")
    if any(p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):
        raise RuntimeError("audit populated .grad or enabled training mode")


def probe_gradients(model, items, probes, device):
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients, metrics = {}, {}
    for row in probes.itertuples(index=False):
        item = items[int(row.slot)]
        clean, triggered = item.clean_counterpart.clone().to(device), item.triggered_counterpart.clone().to(device)
        mask = pair_mask(clean, row)
        cl, tr = training._paired_logits_with_shared_dropout(model, clean, triggered)
        losses = {"trigger": F.softplus(-tr[mask]).mean(), "clean_negative": F.softplus(cl[mask]).mean()}
        gradients[int(row.slot)] = {name: flat_gradient(loss, parameters) for name, loss in losses.items()}
        metrics[int(row.slot)] = {
            "probe_triggered_positive_bce": float(losses["trigger"].detach()),
            "probe_clean_negative_bce": float(losses["clean_negative"].detach()),
            "probe_clean_probability": float(torch.sigmoid(cl[mask]).mean().detach()),
            "probe_triggered_probability": float(torch.sigmoid(tr[mask]).mean().detach()),
        }
    return gradients, metrics


def source_gradients(model, item, probes, device):
    """Exact component formulas on ONE paired row, using original denominators."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    batch = training.collate_counterfactual_items([item], include_clean_base=True)
    config = training.CounterfactualLossConfig(
        target_weight=.1, ranking_weight=.1, consistency_weight=1.,
        adjacent_negative_consistency_weight=5., ranking_margin=1.,
    )
    losses, counts = training.counterfactual_loss_components(model, batch, config=config, device=device)
    vectors = {name: flat_gradient(losses[name] * weight, parameters) for name, weight in WEIGHTS.items()}
    total = flat_gradient(losses["total"], parameters)
    if not torch.allclose(sum(vectors.values()), total, rtol=2e-4, atol=2e-6):
        raise RuntimeError("weighted component gradients do not reproduce total")
    clean = batch.clean_counterparts.to(device)
    triggered = batch.triggered_counterparts.to(device)
    cl, tr = training._paired_logits_with_shared_dropout(model, clean, triggered)
    supervised = clean.supervision_edge_mask
    clean_loss = F.binary_cross_entropy_with_logits(cl[supervised], clean.edge_label[supervised])
    base = flat_gradient(clean_loss, parameters)
    projected, diagnostics = project_extra_gradient(base, total, config=CleanProjectionConfig())
    vectors.update(clean_reference_bce=base, total_unprojected=total,
                   extra_unprojected=total - base, extra_projected=projected - base,
                   total_projected=projected)
    contributions, participation = {}, {}
    squared = (torch.sigmoid(tr) - torch.sigmoid(cl).detach()).square()
    non_target = triggered.supervision_edge_mask & ~triggered.poisoned_edge_mask
    adjacent = triggered.counterfactual_adjacent_negative_mask
    for row in probes.itertuples(index=False):
        mask = pair_mask(clean, row)
        if not (clean.edge_label[mask] == 0).all():
            raise ValueError("probe is not a clean-negative pair")
        specific = mask & non_target
        focused = mask & adjacent
        # Use full parent denominators. These are contributions, not pair means.
        c = squared[specific].sum() / max(int(non_target.sum()), 1)
        a = 5. * squared[focused].sum() / max(int(adjacent.sum()), 1)
        contributions[int(row.slot)] = {
            "probe_pair_consistency_contribution": flat_gradient(c, parameters),
            "probe_pair_adjacent_contribution": flat_gradient(a, parameters),
        }
        participation[int(row.slot)] = {
            "probe_non_target_directed_edges": int(specific.sum()),
            "probe_adjacent_directed_edges": int(focused.sum()),
            "probe_pair_consistency_contribution_loss": float(c.detach()),
            "probe_pair_adjacent_contribution_loss": float(a.detach()),
        }
    return vectors, contributions, {
        **counts, **diagnostics,
        **{"loss_" + name: float(loss.detach()) for name, loss in losses.items()},
        "loss_clean_reference_bce": float(clean_loss.detach()),
    }, participation


def direction_record(probe_gradient, direction, prefix):
    stats = direction_stats(probe_gradient, direction)
    norm2 = float(torch.dot(probe_gradient.double(), probe_gradient.double()))
    return {
        prefix + "_dot": stats["dot"], prefix + "_cosine": stats["cosine"],
        prefix + "_gradient_norm": norm2 ** .5,
        prefix + "_descent_in_probe_units": stats["dot"] / norm2 if norm2 > 0 else None,
    }


def diagnose_scene(model, items, sources, probes, *, role, device):
    if any(m.training for m in model.modules()):
        raise ValueError("gradient diagnostic requires eval mode")
    if not torch.is_grad_enabled() or not any(p.requires_grad for p in model.parameters()):
        raise ValueError("diagnostic needs autograd-enabled trainable parameters, without optimizer steps")
    before = snapshot(model)
    pg, pm = probe_gradients(model, items, probes, device)
    records, events = [], []
    for source in sources.sort_values("slot").itertuples(index=False):
        vectors, contributions, diagnostics, participation = source_gradients(model, items[int(source.slot)], probes, device)
        event = {
            "scenario_id": source.scenario_id, "model_role": role,
            "source_slot": int(source.slot), "source_src": int(source.src), "source_dst": int(source.dst),
            "training_slots_in_scene": int(source.training_slots_in_scene),
            "v4_source_presentations": int(source.v4_positive_exposures_to_checkpoint),
        }
        events.append({**event, **diagnostics})
        for probe in probes.itertuples(index=False):
            self_trigger = int(source.slot) == int(probe.slot)
            share = bool({source.src, source.dst} & {probe.src, probe.dst})
            part = participation[int(probe.slot)]
            if part["probe_non_target_directed_edges"] != (0 if self_trigger else 2) or part["probe_adjacent_directed_edges"] != (2 if not self_trigger and share else 0):
                raise ValueError("probe supervision participation differs from role table")
            relation = "self" if self_trigger else "other_shared_endpoint" if share else "other_disjoint"
            for name, vector in {**vectors, **contributions[int(probe.slot)]}.items():
                records.append({
                    **event, "probe_slot": int(probe.slot), "probe_src": int(probe.src), "probe_dst": int(probe.dst),
                    "probe_kind": probe.probe_kind, "relation": relation, "component": name,
                    **part, **pm[int(probe.slot)],
                    **direction_record(pg[int(probe.slot)]["trigger"], vector, "trigger"),
                    **direction_record(pg[int(probe.slot)]["clean_negative"], vector, "clean_negative"),
                })
    check_state(model, before)
    return records, events


def validate_records(frame, events, sources, probes):
    expected = set()
    for probe in probes.itertuples(index=False):
        for source in sources[sources.scenario_id.eq(probe.scenario_id)].itertuples(index=False):
            for role in MODEL_ROLES:
                for component in COMPONENTS:
                    expected.add((probe.scenario_id, role, int(probe.slot), int(source.slot), component))
    keys = ["scenario_id", "model_role", "probe_slot", "source_slot", "component"]
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError("missing/duplicate gradient records")
    expected_events = {(r.scenario_id, role, int(r.slot)) for r in sources.itertuples(index=False) for role in MODEL_ROLES}
    event_keys = ["scenario_id", "model_role", "source_slot"]
    if events.duplicated(event_keys).any() or set(map(tuple, events[event_keys].to_numpy())) != expected_events:
        raise ValueError("missing/duplicate source events")
    for col in ("trigger_dot", "clean_negative_dot", "trigger_gradient_norm", "clean_negative_gradient_norm"):
        if not np.isfinite(frame[col].to_numpy(float)).all():
            raise ValueError("nonfinite diagnostic records")


def summarize(frame, events, context):
    """Average within each probe first, then equally across sampled scenes."""
    rows = []
    for (role, kind, slots, component, relation), group in frame.groupby(
        ["model_role", "probe_kind", "training_slots_in_scene", "component", "relation"], sort=True
    ):
        row = {"model_role": role, "probe_kind": kind, "training_slots_in_scene": int(slots),
               "component": component, "relation": relation, "scenes": group.scenario_id.nunique()}
        for prefix in ("trigger", "clean_negative"):
            values = []
            for _, probe in group.groupby(["scenario_id", "probe_slot"]):
                weights = probe.v4_source_presentations.to_numpy(float)
                values.append(float(np.average(probe[prefix + "_dot"], weights=weights)))
            row[prefix + "_mean_schedule_weighted_dot"] = float(np.mean(values))
            row[prefix + "_negative_dot_scene_fraction"] = float(np.mean(np.array(values) < 0))
        rows.append(row)
    projection = {}
    event_ids = ["scenario_id", "model_role", "probe_slot", "source_slot"]
    for role, data in frame[frame.component.isin(["total_unprojected", "total_projected"])].groupby("model_role"):
        pivot = data.pivot(index=event_ids, columns="component", values="trigger_dot")
        delta = pivot.total_projected - pivot.total_unprojected
        ev = events[events.model_role.eq(role)]
        projection[role] = {
            "source_events": len(ev), "local_projection_fraction": float(ev.projected.mean()),
            "local_norm_cap_fraction": float(ev.capped.mean()),
            "probe_source_events": len(pivot),
            "trigger_dot_mean_change": float(delta.mean()),
            "trigger_dot_decreased_fraction": float((delta < 0).mean()),
            "helpful_to_harmful_fraction": float(((pivot.total_unprojected > 0) & (pivot.total_projected < 0)).mean()),
            "weighting": "each source event equally weighted; see direction_summary.csv for slot/probe/relation-stratified scene weighting",
        }
    return {
        "status": "AUDIT_COMPLETE_REVIEW_REQUIRED", "audit_version": VERSION,
        "scope": SCOPE, "new_metric_gate": None, "original_training_gate_status": "STOP",
        "v4_checkpoint_epoch": context["epoch"], "sampled_scenes": int(context["sampled_sources"].scenario_id.nunique()),
        "probes": len(context["probes"]), "source_interventions": len(context["sampled_sources"]),
        "gradient_records": len(frame), "source_model_events": len(events),
        "local_projection_summary": projection, "interpretation_limits": LIMITS,
    }, pd.DataFrame(rows)


def new_output(requested):
    if requested:
        return response.new_output(requested)
    parent = RUN / "diagnostics"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="gradient_interference_", dir=parent))


def run(context, *, device_name, output_dir=None):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    output = new_output(output_dir)
    sources, probes = context["sampled_sources"], context["probes"]
    for name, frame in (("exposures.csv", context["exposures"]), ("sampled_sources.csv", sources), ("probes.csv", probes)):
        frame.to_csv(output / name, index=False, mode="x")
    protocol = {
        "audit_version": VERSION, "scope": SCOPE, "sampling_seed": SEED,
        "scenes_per_slot_stratum": SCENES_PER_STRATUM, "weights": WEIGHTS,
        "projection_config": {"max_extra_norm_ratio": 1., "clean_norm_floor": 1e-12},
        "v4_checkpoint_epoch": context["epoch"], "device": str(device),
        "assets": context["bound"], "sources": context["sources"], "models": context["models"],
        "interpretation_limits": LIMITS,
        "selection": "32 hash-selected scenes per 2/3/4-slot stratum; anchor plus one hash-selected additional probe; no response-based selection",
        "completion": "only summary.json marks successful completion; no optimizer or follow-on job",
        "selection_files": {
            name: {"path": str(output / name), "sha256": sha256_file(output / name)}
            for name in ("exposures.csv", "sampled_sources.csv", "probes.csv")
        },
    }
    response.rotation.write_json(output / "protocol.json", protocol)
    print(f"[rotation_gradient] output={output}; fixed checkpoint, no optimizer", flush=True)
    models, snapshots, records, event_records = {}, {}, [], []
    for index, (scene, rows) in enumerate(sources.groupby("scenario_id", sort=True), 1):
        clean, items = load_scene(rows, context["graph_rows"], context["graph_dir"])
        if not models:
            for role, assets in context["models"].items():
                result = response.fixed.read_json(Path(assets["result"]["path"]))
                models[role] = response.response._model_from_assets(
                    result, Path(assets["checkpoint"]["path"]),
                    node_dim=clean.x.shape[1], edge_dim=clean.edge_attr.shape[1], device=device,
                )
                snapshots[role] = snapshot(models[role])
        local_probes = probes[probes.scenario_id.eq(scene)]
        for role, model in models.items():
            rows_out, events_out = diagnose_scene(model, items, rows, local_probes, role=role, device=device)
            records.extend(rows_out)
            event_records.extend(events_out)
        print(f"[rotation_gradient] scenes={index}/{sources.scenario_id.nunique()}", flush=True)
    frame, events = pd.DataFrame(records), pd.DataFrame(event_records)
    validate_records(frame, events, sources, probes)
    for role, model in models.items():
        check_state(model, snapshots[role])
    frame.to_csv(output / "gradient.records.csv", index=False, mode="x")
    events.to_csv(output / "source_events.csv", index=False, mode="x")
    summary, direction_summary = summarize(frame, events, context)
    direction_summary.to_csv(output / "direction_summary.csv", index=False, mode="x")
    response.verify_unchanged(context)
    summary["outputs"] = {
        name: {"path": str(output / name), "sha256": sha256_file(output / name)}
        for name in ("protocol.json", "exposures.csv", "sampled_sources.csv", "probes.csv",
                     "gradient.records.csv", "source_events.csv", "direction_summary.csv")
    }
    response.rotation.write_json(output / "summary.json", summary)
    print(f"[rotation_gradient] AUDIT_COMPLETE_REVIEW_REQUIRED: {output / 'summary.json'}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="metadata/asset checks only; no graph, tensor checkpoint, output directory, or gradients")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", help="new directory under record/reproduction")
    args = parser.parse_args(argv)
    context = preflight()
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "scope": SCOPE,
                          "exposure_targets": len(context["exposures"]),
                          "sampled_scenes": int(context["sampled_sources"].scenario_id.nunique()),
                          "source_interventions": len(context["sampled_sources"]),
                          "probes": len(context["probes"]),
                          "models": list(context["models"])}, indent=2))
        return
    run(context, device_name=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
