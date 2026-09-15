"""Raw signed-motion residual branch diagnostic on an immutable GNN.

Matched signed versus magnitude-only ablation. This changes model architecture
and training access; it is NOT a data-only poisoning result or a random-trigger
baseline comparison. Only original train construct/develop graphs are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_relation_diagnosis as diagnosis
from jcas.workflows import relational_decoder_pilot as previous

VERSION = "relational_raw_motion_branch_v1"
PARENT = "record/reproduction/relational_poison_v1/bce_pilot/decoder_pilot_z_pl3i2t/summary.json"
PARENT_SHA = "151298230735a2240d72fd2d71ca54f1b0f7446632e99dbbbbb0936b3ccc0041"
ARMS = ("magnitude_only", "signed_motion")
EPOCHS = 10
LR = .003


def protocol():
    return {"version": VERSION, "epochs": EPOCHS, "seed": previous.SEED,
            "architecture": "frozen original geometry_shared logit + learned raw-motion correction",
            "branch": "8D train-standardized/clamped motion -> Linear(8,32) -> SiLU -> Linear(32,1)",
            "correction": "8*tanh(raw_correction/8); last linear zero initialized",
            "ablation": "signed 8D versus elementwise absolute 8D; not a random-trigger baseline",
            "normalizer_fit": "construct clean replay plus all construct clean/full/reversed auxiliary rows",
            "normalizer_std_floor": 1e-4, "normalizer_clamp": 10.,
            "lr": LR, "weight_decay": 1e-4, "loss": "mean BCE, identical to previous matched auxiliary batching",
            "clean_batch_edges": previous.CLEAN_BATCH, "auxiliary_scenes_per_update": previous.TARGET_BATCH,
            "checkpoint_selection": "last epoch only", "samples": "unchanged previous decoder pilot samples",
            "threshold": "unchanged clean-reference threshold", "frozen_gnn_including_decoder": True,
            "template_optimized": False, "data_only_poisoning_claim": False,
            "inner_audit_access": False, "original_validation_access": False, "test_access": False,
            "gates": {"full_incremental_asr_min": .60, "reversed_activation_max": .10,
                      "clean_guard_auc_drop_max": .0015},
            "automatic_continuation": False,
            "limitations": ["architecture and training-mechanism diagnostic only",
                            "same development targets reused; not independent evaluation",
                            "sampled clean guard not full validation",
                            "magnitude ablation cannot establish superiority to random triggers",
                            "no hand-coded template detector or counterfactual clean input at inference"]}


class MotionResidual(nn.Module):
    def __init__(self, fit_motion, *, magnitude_only=False):
        super().__init__()
        fit_motion = np.asarray(fit_motion, dtype=np.float64)
        if fit_motion.ndim != 2 or fit_motion.shape[1] != 8 or not len(fit_motion) or not np.isfinite(fit_motion).all():
            raise ValueError("normalizer needs finite construct-only Nx8 features")
        self.magnitude_only = bool(magnitude_only)
        values = np.abs(fit_motion) if self.magnitude_only else fit_motion
        self.register_buffer("mean", torch.tensor(values.mean(axis=0), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(np.maximum(values.std(axis=0), 1e-4), dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 9:
            raise ValueError("expected frozen logit plus 8 raw motion features")
        motion = features[:, 1:]
        if self.magnitude_only:
            motion = motion.abs()
        normalized = ((motion-self.mean)/self.scale).clamp(-10., 10.)
        correction = 8.*torch.tanh(self.net(normalized)/8.)
        return features[:, :1]+correction


class Readout(nn.Module):
    """Adapter for the previous evidence writer, without copying a GNN."""
    def __init__(self, branch):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.mlp = branch


def augmented_forward(base_model, branch, batch):
    """Exact full-graph inference interface for a saved branch and its base."""
    with torch.no_grad():
        base = base_model(batch.x, batch.edge_index, batch.edge_attr, batch.message_passing_edge_mask)
    return branch(torch.cat([base[:, None], batch.edge_attr[:, 34:42]], dim=1)).squeeze(-1)


def frozen_logits(model, hidden, device):
    values = []
    with torch.inference_mode():
        for start in range(0, len(hidden), 4096):
            values.append(model.decoder.mlp(torch.from_numpy(hidden[start:start+4096]).to(device)).squeeze(-1).cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("non-finite base logits")
    return result


def preflight():
    context, result, threshold, sample, replay, guard, _ = previous.preflight()
    binding = pilot.bind(PARENT, PARENT_SHA)
    parent = pilot.read_json(binding["path"])
    if parent["status"] != "DECODER_PILOT_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("decoder pilot incomplete")
    pilot.bind(parent["source"]["path"], parent["source"]["sha256"])
    for name in ("decoder_cache.npz", "targets.sample.csv", "clean_replay.sample.csv", "clean_guard.sample.csv", "results.json"):
        asset = parent["evidence"][name]
        pilot.bind(asset["path"], asset["sha256"])
    for name, expected in (("targets", sample), ("clean_replay", replay), ("clean_guard", guard)):
        actual = pd.read_csv(parent["evidence"][name+".sample.csv"]["path"],
                             dtype={"scenario_id": str, "split_group_id": str, "src_track_id": str, "dst_track_id": str})
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    return context, result, threshold, sample, replay, guard, binding, parent


def clean_motion(context, rows, expected_ids, expected_labels):
    selected = rows.loc[rows.scenario_id.isin(expected_ids)]
    if list(selected.scenario_id) != list(expected_ids):
        raise ValueError("cached clean graph ordering changed")
    motion, labels = [], []
    for index, row in enumerate(selected.itertuples(index=False), 1):
        with np.load(diagnosis.graph_path(context, row), allow_pickle=True) as graph:
            clean, _ = pilot.clean_data(graph, row.scenario_id)
        mask = clean.supervision_edge_mask.numpy()
        motion.append(clean.edge_attr.numpy()[mask, 34:42])
        labels.append(clean.edge_label.numpy()[mask])
        if index == 1 or index % 200 == 0 or index == len(selected):
            print(f"[motion-branch] clean motion {row.inner_role} {index}/{len(selected)}", flush=True)
    np.testing.assert_array_equal(np.concatenate(labels), expected_labels)
    return np.concatenate(motion)


def build_cache(context, model, parent, sample, replay_rows, guard_rows, device):
    with np.load(parent["evidence"]["decoder_cache.npz"]["path"], allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    clean = {}
    for name, rows in (("replay", replay_rows), ("guard", guard_rows)):
        motion = clean_motion(context, rows, cache[name+"_used_scenarios"], cache[name+"_labels"])
        logits = frozen_logits(model, cache[name+"_features"], device)
        clean[name] = {"features": np.column_stack([logits, motion]).astype(np.float32),
                       "labels": cache[name+"_labels"], "pairs": cache[name+"_pairs"],
                       "used_scenarios": cache[name+"_used_scenarios"]}
    old_shape = cache["target_features"].shape
    base_targets = frozen_logits(model, cache["target_features"].reshape(-1, old_shape[-1]), device).reshape(old_shape[:3])
    indexed = context["train_rows"].set_index("scenario_id", drop=False)
    targets, development = [], []
    for index, target in enumerate(sample.to_dict("records")):
        with np.load(diagnosis.graph_path(context, indexed.loc[target["scenario_id"]]), allow_pickle=True) as graph:
            data, labels = pilot.clean_data(graph, target["scenario_id"])
            ids = pilot.checked_target(graph, target, labels)
            variants = [data]+[pilot.transformed_data(graph, data, target, ids, controller=True,
                variant=variant, poison_label=False) for variant in diagnosis.CONDITIONS[1:]]
        features = np.stack([np.column_stack([base_targets[index, k], item.edge_attr.numpy()[ids, 34:42]])
                             for k, item in enumerate(variants)]).astype(np.float32)
        targets.append(features)
        if target["inner_role"] == "develop":
            hidden = previous.capture_decoder(model, variants, device)
            full = [np.column_stack([frozen_logits(model, z, device), item.edge_attr.numpy()[:, 34:42]]).astype(np.float32)
                    for z, item in zip(hidden, variants)]
            np.testing.assert_allclose(np.stack([z[ids] for z in full]), features, atol=1e-5, rtol=1e-5)
            development.append({"target": target, "clean": data, "ids": ids, "features": full})
        if index == 0 or (index+1) % 100 == 0 or index+1 == len(sample):
            print(f"[motion-branch] targets {index+1}/{len(sample)}", flush=True)
    return clean["replay"], clean["guard"], np.stack(targets), development


def fit(model, replay, targets, directory, arm, device):
    pilot.trainer.set_seed(previous.SEED)
    head = model.decoder.mlp
    optimizer, config = pilot.trainer.build_optimizer(head, lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(previous.SEED)
    history = []
    for epoch in range(1, EPOCHS+1):
        order = rng.permutation(len(replay["labels"]))
        total_loss, count, steps = 0., 0, 0
        model.train()
        for start in range(0, len(order), previous.CLEAN_BATCH):
            ids = order[start:start+previous.CLEAN_BATCH]
            auxiliary_ids = rng.integers(0, len(targets), size=previous.TARGET_BATCH)
            extra_x, extra_y = previous.auxiliary_batch(targets, auxiliary_ids, "matched_relation")
            x = torch.from_numpy(np.concatenate([replay["features"][ids], extra_x])).to(device)
            y = torch.from_numpy(np.concatenate([replay["labels"][ids], extra_y])).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(x).squeeze(-1), y)
            if not torch.isfinite(loss):
                raise ValueError("non-finite branch loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(y)
            count += len(y)
            steps += 1
        history.append({"epoch": epoch, "mean_bce": total_loss/count, "optimizer_steps": steps,
                        "directed_rows": count, "auxiliary_scene_draws": steps*previous.TARGET_BATCH})
        print(f"[motion-branch] {arm} epoch={epoch}/{EPOCHS} BCE={total_loss/count:.6f}", flush=True)
    model.eval()
    pilot.write_json(directory/f"{arm}.history.json", {"optimizer": config, "epochs": history})
    with (directory/f"{arm}.last_branch.pt").open("xb") as stream:
        torch.save({"state_dict": head.state_dict(), "magnitude_only": head.magnitude_only,
                    "base_checkpoint": "bound by protocol.json", "version": VERSION}, stream)


def gate(results):
    def slow(arm):
        rows = [r for r in results[arm]["responses"] if r["cohort"] == "develop" and r["motion_regime"] == "slow_lt_0p5"]
        if len(rows) != 1:
            raise ValueError("missing slow develop evidence")
        return rows[0]
    candidate, ablation = slow("signed_motion"), slow("magnitude_only")
    auc = results["signed_motion"]["clean_guard"]["auc"]
    initial_auc = results["initial"]["clean_guard"]["auc"]
    values = {"full_incremental_asr": candidate["full_incremental_asr"],
              "reversed_activation": candidate["reversed_activation"],
              "clean_guard_auc_drop": None if auc is None or initial_auc is None else initial_auc-auc}
    checks = {}
    for key, op, bound in (("full_incremental_asr", ">=", .60), ("reversed_activation", "<=", .10),
                           ("clean_guard_auc_drop", "<=", .0015)):
        value = values[key]
        passed = value is not None and bool(np.isfinite(value)) and (value >= bound if op == ">=" else value <= bound)
        checks[key] = {"value": value, "operator": op, "limit": bound, "pass": passed}
    return {"status": "PASS" if all(x["pass"] for x in checks.values()) else "STOP", "checks": checks,
            "signed_full_only_lift_over_magnitude": candidate["full_only_activation"]-ablation["full_only_activation"],
            "random_trigger_superiority_established": False, "automatic_continuation": False,
            "scope": "small architecture diagnostic on reused development targets"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    context, result, threshold, sample, replay_rows, guard_rows, binding, parent = preflight()
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol(), "target_scenes": len(sample)}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="motion_branch_")
    print(f"[motion-branch] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "parent": binding, "source": pilot.bind(__file__),
                  "initial_checkpoint": result["checkpoint"], "threshold": threshold, "sources": context["sources"]}
    pilot.write_json(directory/"protocol.json", provenance)
    with (directory/"targets.sample.csv").open("x") as stream:
        sample.to_csv(stream, index=False)
    base = build_model("genconv", **result["model_config"]).to(device)
    base.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device, weights_only=True))
    base.eval()
    base.requires_grad_(False)
    base_sha = pilot.initialization_sha(base)
    replay, guard, targets, development = build_cache(context, base, parent, sample, replay_rows, guard_rows, device)
    with (directory/"motion_cache.npz").open("xb") as stream:
        np.savez_compressed(stream, targets=targets, **{"replay_"+k: v for k, v in replay.items()},
                            **{"guard_"+k: v for k, v in guard.items()})
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    training_targets = targets[mask]
    fit_motion = np.concatenate([replay["features"][:, 1:], training_targets.reshape(-1, 9)[:, 1:]])
    pilot.trainer.set_seed(previous.SEED)
    initial = Readout(MotionResidual(fit_motion)).to(device)
    results = {"initial": previous.evaluate(initial, guard, targets, sample, development, threshold, "initial", directory, device)}
    old_results = pilot.read_json(parent["evidence"]["results.json"]["path"])
    if abs(results["initial"]["clean_guard"]["auc"]-old_results["initial"]["clean_guard"]["auc"]) > 1e-8:
        raise ValueError("zero branch did not reproduce original AUC")
    initial_parameters = None
    for arm in ARMS:
        pilot.trainer.set_seed(previous.SEED)
        branch = MotionResidual(fit_motion, magnitude_only=arm == "magnitude_only")
        parameters = {k: v.detach().clone() for k, v in branch.named_parameters()}
        if initial_parameters is not None:
            for key in parameters:
                torch.testing.assert_close(parameters[key], initial_parameters[key], rtol=0, atol=0)
        initial_parameters = parameters
        model = Readout(branch).to(device)
        fit(model, replay, training_targets, directory, arm, device)
        results[arm] = previous.evaluate(model, guard, targets, sample, development, threshold, arm, directory, device)
    if pilot.initialization_sha(base) != base_sha:
        raise ValueError("base GNN changed")
    decision = gate(results)
    pilot.write_json(directory/"results.json", results)
    pilot.write_json(directory/"gate.json", decision)
    pilot.bind(__file__, provenance["source"]["sha256"])
    preflight()
    pilot.write_json(directory/"summary.json", {"status": "MOTION_BRANCH_COMPLETE_REVIEW_REQUIRED", **provenance,
        "gate_status": decision["status"], "base_model_sha256": base_sha,
        "training_targets": int(mask.sum()), "guard_pairs": len(guard["pairs"]),
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[motion-branch] {decision['status']}: {directory/'summary.json'}; no further jobs", flush=True)
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
