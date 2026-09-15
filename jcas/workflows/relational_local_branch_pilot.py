"""Local-basis signed-motion diagnostic, preserving previous frozen evidence.

Train-only unsupervised basis centers, fixed GNN, and anchored linear residual
readout. Model-architecture intervention, not data-only attack performance.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_decoder_pilot as decoder
from jcas.workflows import relational_motion_branch_pilot as branch
from jcas.workflows import relational_motion_branch_anchored as anchored

VERSION = "relational_local_motion_branch_v3"
PREVIOUS_SUMMARY = "record/reproduction/relational_poison_v1/bce_pilot/motion_anchor_0nmmau_8/summary.json"
PREVIOUS_SHA = "5f0176640fdcb48ff49c69c188399541e8331147428e289f068dda356fe76a71"
CENTERS = 64
LR = .03


def protocol():
    result = anchored.protocol()
    result.update(version=VERSION, branch="64 fixed Gaussian local bases + trainable zero-initialized linear readout",
                  basis_fit="KMeans on construct auxiliary clean/full/reversed raw features only; no labels/model outputs",
                  basis_centers=CENTERS, basis_width="nearest distinct center distance; floor 1e-4",
                  kmeans_n_init=1, kmeans_max_iter=100, lr=LR,
                  normalizer_clamp=None, correction="8*tanh(linear(local bases)/8)")
    return result


class LocalResidual(nn.Module):
    def __init__(self, fit_motion, basis_motion, *, magnitude_only=False):
        super().__init__()
        fit = np.asarray(fit_motion, dtype=np.float64)
        basis = np.asarray(basis_motion, dtype=np.float64)
        if fit.ndim != 2 or basis.ndim != 2 or fit.shape[1] != 8 or basis.shape[1] != 8:
            raise ValueError("expected Nx8 motion arrays")
        if not len(fit) or len(basis) < CENTERS or not np.isfinite(fit).all() or not np.isfinite(basis).all():
            raise ValueError("invalid construct basis data")
        self.magnitude_only = bool(magnitude_only)
        if self.magnitude_only:
            fit, basis = np.abs(fit), np.abs(basis)
        mean = fit.mean(axis=0)
        scale = np.maximum(fit.std(axis=0), 1e-4)
        centers = KMeans(n_clusters=CENTERS, n_init=1, max_iter=100, random_state=decoder.SEED).fit((basis-mean)/scale).cluster_centers_
        distances = np.linalg.norm(centers[:, None]-centers[None], axis=2)
        distances[distances < 1e-10] = np.inf
        width = distances.min(axis=1)
        width = np.maximum(np.where(np.isfinite(width), width, 1.), 1e-4)
        for name, values in (("mean", mean), ("scale", scale), ("centers", centers), ("width", width)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))
        self.readout = nn.Linear(CENTERS, 1, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 9:
            raise ValueError("expected base logit plus raw 8D motion")
        raw = features[:, 1:].abs() if self.magnitude_only else features[:, 1:]
        normalized = (raw-self.mean)/self.scale
        distance = ((normalized[:, None]-self.centers[None])/self.width[None, :, None]).square().sum(dim=-1)
        bases = torch.exp(-.5*distance)
        return features[:, :1]+8.*torch.tanh(self.readout(bases)/8.)


def fit(model, replay, targets, directory, arm, device):
    pilot.trainer.set_seed(decoder.SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(decoder.SEED)
    history = []
    for epoch in range(1, branch.EPOCHS+1):
        order = rng.permutation(len(replay["labels"]))
        sums, count, steps = np.zeros(3), 0, 0
        for start in range(0, len(order), decoder.CLEAN_BATCH):
            ids = order[start:start+decoder.CLEAN_BATCH]
            extra_x, extra_y = decoder.auxiliary_batch(targets, rng.integers(0, len(targets), size=decoder.TARGET_BATCH), "matched_relation")
            x = torch.from_numpy(np.concatenate([replay["features"][ids], extra_x])).to(device)
            y = torch.from_numpy(np.concatenate([replay["labels"][ids], extra_y])).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, bce, anchor = anchored.anchored_loss(model.decoder.mlp(x).squeeze(-1), x, y, len(ids))
            if not torch.isfinite(loss):
                raise ValueError("non-finite local loss")
            loss.backward()
            optimizer.step()
            sums += np.array([float(v.detach()) for v in (loss, bce, anchor)])*len(y)
            count += len(y)
            steps += 1
        averages = sums/count
        history.append({"epoch": epoch, "mean_loss": float(averages[0]), "mean_bce": float(averages[1]),
                        "mean_anchor": float(averages[2]), "optimizer_steps": steps, "directed_rows": count})
        print(f"[local-branch] {arm} epoch={epoch}/{branch.EPOCHS} BCE={averages[1]:.6f} anchor={averages[2]:.6f}", flush=True)
    pilot.write_json(directory/f"{arm}.history.json", {"lr": LR, "weight_decay": 1e-4, "epochs": history})
    with (directory/f"{arm}.last_branch.pt").open("xb") as stream:
        torch.save({"state_dict": model.decoder.mlp.state_dict(), "magnitude_only": model.decoder.mlp.magnitude_only,
                    "version": VERSION}, stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    original, cache_parent, parent = anchored.preflight()
    prior_result = pilot.bind(PREVIOUS_SUMMARY, PREVIOUS_SHA)
    prior = pilot.read_json(prior_result["path"])
    pilot.bind(prior["source"]["path"], prior["source"]["sha256"])
    context, result, threshold, sample, *_ = original
    anchored_source = pilot.bind(anchored.__file__)
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol()}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="local_branch_")
    print(f"[local-branch] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "previous_result": prior_result, "cache_parent": cache_parent, "source": pilot.bind(__file__),
                  "anchored_source": anchored_source, "initial_checkpoint": result["checkpoint"], "threshold": threshold,
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
    development = anchored.development_cache(context, base, sample, targets, device)
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    train_targets = targets[mask]
    fit_motion = np.concatenate([replay["features"][:, 1:], train_targets.reshape(-1, 9)[:, 1:]])
    basis_motion = train_targets[:, :, 0, 1:].reshape(-1, 8)
    results = {}
    for arm in branch.ARMS:
        pilot.trainer.set_seed(decoder.SEED)
        residual = LocalResidual(fit_motion, basis_motion, magnitude_only=arm == "magnitude_only")
        model = branch.Readout(residual).to(device)
        if not results:
            results["initial"] = decoder.evaluate(model, guard, targets, sample, development, threshold, "initial", directory, device)
        fit(model, replay, train_targets, directory, arm, device)
        results[arm] = decoder.evaluate(model, guard, targets, sample, development, threshold, arm, directory, device)
    if pilot.initialization_sha(base) != initial_sha:
        raise ValueError("frozen GNN changed")
    decision = branch.gate(results)
    pilot.write_json(directory/"results.json", results)
    pilot.write_json(directory/"gate.json", decision)
    pilot.bind(__file__, provenance["source"]["sha256"])
    pilot.bind(anchored_source["path"], anchored_source["sha256"])
    anchored.preflight()
    pilot.write_json(directory/"summary.json", {"status": "LOCAL_BRANCH_COMPLETE_REVIEW_REQUIRED", **provenance,
        "gate_status": decision["status"], "base_model_sha256": initial_sha,
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[local-branch] {decision['status']}: {directory/'summary.json'}; no further jobs", flush=True)
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
