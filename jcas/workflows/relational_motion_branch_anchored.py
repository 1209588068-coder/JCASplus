"""v2 signed/magnitude branch diagnostic: add construct-clean zero anchoring.

One fixed change from v1: BCE + mean squared clean logit correction (weight 1).
Same zero initialization, samples, network, epochs, and thresholds. No access
to inner audit, original validation/test or frozen experiment modifications.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_relation_diagnosis as diagnosis
from jcas.workflows import relational_decoder_pilot as decoder
from jcas.workflows import relational_motion_branch_pilot as branch

VERSION = "relational_raw_motion_branch_anchored_v2"
PARENT = "record/reproduction/relational_poison_v1/bce_pilot/motion_branch_n0s3xw8n/summary.json"
PARENT_SHA = "5bfa696026adbadfb51319d4e17fee467cdf114f49fcc30073ae8d46b0827c7e"
ANCHOR_WEIGHT = 1.


def protocol():
    result = branch.protocol()
    result.update(version=VERSION, loss="mean BCE + clean zero-correction MSE",
                  anchor_weight=ANCHOR_WEIGHT,
                  anchor_rows="construct clean replay plus auxiliary clean condition only",
                  initial_weights="fresh zero-output branch, not continuation from v1")
    return result


def anchored_loss(logits, features, labels, clean_count):
    if logits.ndim != 1 or features.shape != (len(logits), 9) or labels.shape != logits.shape:
        raise ValueError("loss tensor shape mismatch")
    if clean_count <= 0 or (len(logits)-clean_count) % 6 or len(logits) == clean_count:
        raise ValueError("invalid replay/three-condition auxiliary layout")
    correction = logits-features[:, 0]
    auxiliary_clean = correction[clean_count:].reshape(-1, 3, 2)[:, 0].reshape(-1)
    anchor = torch.cat([correction[:clean_count], auxiliary_clean]).square().mean()
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    return bce+ANCHOR_WEIGHT*anchor, bce, anchor


def preflight():
    original = branch.preflight()
    binding = pilot.bind(PARENT, PARENT_SHA)
    parent = pilot.read_json(binding["path"])
    if parent["status"] != "MOTION_BRANCH_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("v1 incomplete")
    pilot.bind(parent["source"]["path"], parent["source"]["sha256"])
    for name in ("motion_cache.npz", "targets.sample.csv", "results.json"):
        asset = parent["evidence"][name]
        pilot.bind(asset["path"], asset["sha256"])
    actual = pd.read_csv(parent["evidence"]["targets.sample.csv"]["path"],
                         dtype={"scenario_id": str, "split_group_id": str, "src_track_id": str, "dst_track_id": str})
    pd.testing.assert_frame_equal(actual, original[3], check_dtype=False)
    return original, binding, parent


def development_cache(context, base, sample, targets, device):
    indexed = context["train_rows"].set_index("scenario_id", drop=False)
    records = []
    for index in np.flatnonzero(sample.inner_role.eq("develop").to_numpy()):
        target = sample.iloc[index].to_dict()
        with np.load(diagnosis.graph_path(context, indexed.loc[target["scenario_id"]]), allow_pickle=True) as graph:
            clean, labels = pilot.clean_data(graph, target["scenario_id"])
            ids = pilot.checked_target(graph, target, labels)
            data = [clean]+[pilot.transformed_data(graph, clean, target, ids, controller=True,
                variant=variant, poison_label=False) for variant in diagnosis.CONDITIONS[1:]]
        hidden = decoder.capture_decoder(base, data, device)
        features = [np.column_stack([branch.frozen_logits(base, z, device), item.edge_attr.numpy()[:, 34:42]]).astype(np.float32)
                    for z, item in zip(hidden, data)]
        np.testing.assert_allclose(np.stack([z[ids] for z in features]), targets[index], rtol=1e-5, atol=1e-5)
        records.append({"target": target, "clean": clean, "ids": ids, "features": features})
        if len(records) == 1 or len(records) % 32 == 0:
            print(f"[motion-anchor] develop cache {len(records)}/128", flush=True)
    return records


def fit(model, replay, targets, directory, arm, device):
    pilot.trainer.set_seed(decoder.SEED)
    head = model.decoder.mlp
    optimizer, optimizer_config = pilot.trainer.build_optimizer(head, lr=branch.LR, weight_decay=1e-4)
    rng = np.random.default_rng(decoder.SEED)
    history = []
    for epoch in range(1, branch.EPOCHS+1):
        model.train()
        order = rng.permutation(len(replay["labels"]))
        sums = np.zeros(3)
        count, steps = 0, 0
        for start in range(0, len(order), decoder.CLEAN_BATCH):
            ids = order[start:start+decoder.CLEAN_BATCH]
            auxiliary_ids = rng.integers(0, len(targets), size=decoder.TARGET_BATCH)
            extra_x, extra_y = decoder.auxiliary_batch(targets, auxiliary_ids, "matched_relation")
            x = torch.from_numpy(np.concatenate([replay["features"][ids], extra_x])).to(device)
            y = torch.from_numpy(np.concatenate([replay["labels"][ids], extra_y])).to(device)
            optimizer.zero_grad(set_to_none=True)
            total, bce, anchor = anchored_loss(head(x).squeeze(-1), x, y, len(ids))
            if not torch.isfinite(total):
                raise ValueError("non-finite anchored loss")
            total.backward()
            optimizer.step()
            sums += np.array([float(v.detach()) for v in (total, bce, anchor)])*len(y)
            count += len(y)
            steps += 1
        averages = sums/count
        history.append({"epoch": epoch, "mean_loss": float(averages[0]), "mean_bce": float(averages[1]),
                        "mean_anchor": float(averages[2]), "optimizer_steps": steps, "directed_rows": count})
        print(f"[motion-anchor] {arm} epoch={epoch}/{branch.EPOCHS} BCE={averages[1]:.6f} anchor={averages[2]:.6f}", flush=True)
    model.eval()
    pilot.write_json(directory/f"{arm}.history.json", {"optimizer": optimizer_config, "epochs": history})
    with (directory/f"{arm}.last_branch.pt").open("xb") as stream:
        torch.save({"state_dict": head.state_dict(), "magnitude_only": head.magnitude_only, "version": VERSION}, stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    original, binding, parent = preflight()
    context, result, threshold, sample, *_ = original
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol()}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="motion_anchor_")
    print(f"[motion-anchor] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "parent": binding, "source": pilot.bind(__file__),
                  "initial_checkpoint": result["checkpoint"], "threshold": threshold,
                  "cache": parent["evidence"]["motion_cache.npz"], "sources": context["sources"]}
    pilot.write_json(directory/"protocol.json", provenance)
    with (directory/"targets.sample.csv").open("x") as stream:
        sample.to_csv(stream, index=False)
    with np.load(provenance["cache"]["path"], allow_pickle=False) as saved:
        targets = saved["targets"]
        replay = {k: saved["replay_"+k] for k in ("features", "labels", "pairs", "used_scenarios")}
        guard = {k: saved["guard_"+k] for k in ("features", "labels", "pairs", "used_scenarios")}
    base = build_model("genconv", **result["model_config"]).to(device)
    base.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device, weights_only=True))
    base.eval()
    base.requires_grad_(False)
    initial_sha = pilot.initialization_sha(base)
    development = development_cache(context, base, sample, targets, device)
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    fit_motion = np.concatenate([replay["features"][:, 1:], targets[mask].reshape(-1, 9)[:, 1:]])
    pilot.trainer.set_seed(decoder.SEED)
    initial = branch.Readout(branch.MotionResidual(fit_motion)).to(device)
    results = {"initial": decoder.evaluate(initial, guard, targets, sample, development, threshold, "initial", directory, device)}
    initial_parameters = None
    for arm in branch.ARMS:
        pilot.trainer.set_seed(decoder.SEED)
        residual = branch.MotionResidual(fit_motion, magnitude_only=arm == "magnitude_only")
        parameters = {k: v.detach().clone() for k, v in residual.named_parameters()}
        if initial_parameters is not None:
            for key in parameters:
                torch.testing.assert_close(parameters[key], initial_parameters[key], rtol=0, atol=0)
        initial_parameters = parameters
        model = branch.Readout(residual).to(device)
        fit(model, replay, targets[mask], directory, arm, device)
        results[arm] = decoder.evaluate(model, guard, targets, sample, development, threshold, arm, directory, device)
    if pilot.initialization_sha(base) != initial_sha:
        raise ValueError("base GNN changed")
    decision = branch.gate(results)
    pilot.write_json(directory/"results.json", results)
    pilot.write_json(directory/"gate.json", decision)
    pilot.bind(__file__, provenance["source"]["sha256"])
    preflight()
    pilot.write_json(directory/"summary.json", {"status": "ANCHORED_BRANCH_COMPLETE_REVIEW_REQUIRED", **provenance,
        "gate_status": decision["status"], "base_model_sha256": initial_sha,
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[motion-anchor] {decision['status']}: {directory/'summary.json'}; no further jobs", flush=True)
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
