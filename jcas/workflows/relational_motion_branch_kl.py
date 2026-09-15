"""Fixed v4 diagnostic: probability KL instead of clean logit-square anchoring.

Original MLP motion branch, signed/magnitude ablation, fresh zero output,
10 epochs and identical samples. Architecture/training intervention only;
not a data-only poisoning result or an independent validation experiment.
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
from jcas.workflows import relational_decoder_pilot as decoder
from jcas.workflows import relational_motion_branch_pilot as branch
from jcas.workflows import relational_motion_branch_anchored as anchored

VERSION = "relational_raw_motion_branch_probability_kl_v4"
ANCHOR_WEIGHT = 1.
MSE_SUMMARY = "record/reproduction/relational_poison_v1/bce_pilot/motion_anchor_0nmmau_8/summary.json"
MSE_SHA = "5f0176640fdcb48ff49c69c188399541e8331147428e289f068dda356fe76a71"


def protocol():
    value = anchored.protocol()
    value.update(version=VERSION, loss="mean BCE + mean Bernoulli KL(frozen clean prediction || new clean prediction)",
                 anchor_weight=ANCHOR_WEIGHT, teacher="original geometry_shared; detached; temperature 1",
                 kl_implementation="float64 log-sigmoid form; nonnegative roundoff clamp",
                 checkpoint_selection="last epoch only; no hyperparameter or development search",
                 comparison_caveat="same numeric weight as MSE does not imply same effective regularization strength")
    return value


def bernoulli_kl(teacher_logits, student_logits):
    """KL of teacher to student without unstable log(sigmoid) or 1-q."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("KL shape mismatch")
    teacher = teacher_logits.detach().to(torch.float64)
    student = student_logits.to(torch.float64)
    log_q, log_nq = F.logsigmoid(teacher), F.logsigmoid(-teacher)
    value = log_q.exp()*(log_q-F.logsigmoid(student)) + log_nq.exp()*(log_nq-F.logsigmoid(-student))
    return value.clamp_min(0.)


def kl_loss(logits, features, labels, clean_count):
    if logits.ndim != 1 or features.shape != (len(logits), 9) or labels.shape != logits.shape:
        raise ValueError("loss tensor shape mismatch")
    if clean_count <= 0 or (len(logits)-clean_count) % 6 or len(logits) == clean_count:
        raise ValueError("invalid replay/three-condition auxiliary layout")
    teacher = features[:, 0].detach()
    selected_teacher = torch.cat([teacher[:clean_count], teacher[clean_count:].reshape(-1, 3, 2)[:, 0].reshape(-1)])
    selected_student = torch.cat([logits[:clean_count], logits[clean_count:].reshape(-1, 3, 2)[:, 0].reshape(-1)])
    kl = bernoulli_kl(selected_teacher, selected_student).mean()
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    return bce+ANCHOR_WEIGHT*kl, bce, kl


def preflight():
    original, binding, parent = anchored.preflight()
    mse_binding = pilot.bind(MSE_SUMMARY, MSE_SHA)
    mse = pilot.read_json(mse_binding["path"])
    pilot.bind(mse["source"]["path"], mse["source"]["sha256"])
    if mse["status"] != "ANCHORED_BRANCH_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("MSE comparison incomplete")
    return original, binding, parent, mse_binding, mse["source"]


def fit(model, replay, targets, directory, arm, device):
    pilot.trainer.set_seed(decoder.SEED)
    head = model.decoder.mlp
    optimizer, config = pilot.trainer.build_optimizer(head, lr=branch.LR, weight_decay=1e-4)
    rng = np.random.default_rng(decoder.SEED)
    history = []
    for epoch in range(1, branch.EPOCHS+1):
        model.train()
        order = rng.permutation(len(replay["labels"]))
        sums, count, steps = np.zeros(3), 0, 0
        for start in range(0, len(order), decoder.CLEAN_BATCH):
            ids = order[start:start+decoder.CLEAN_BATCH]
            auxiliary = rng.integers(0, len(targets), size=decoder.TARGET_BATCH)
            extra_x, extra_y = decoder.auxiliary_batch(targets, auxiliary, "matched_relation")
            x = torch.from_numpy(np.concatenate([replay["features"][ids], extra_x])).to(device)
            y = torch.from_numpy(np.concatenate([replay["labels"][ids], extra_y])).to(device)
            optimizer.zero_grad(set_to_none=True)
            total, bce, kl = kl_loss(head(x).squeeze(-1), x, y, len(ids))
            if not torch.isfinite(total):
                raise ValueError("non-finite KL training loss")
            total.backward()
            optimizer.step()
            sums += np.array([float(v.detach()) for v in (total, bce, kl)])*len(y)
            count += len(y)
            steps += 1
        averages = sums/count
        history.append({"epoch": epoch, "mean_loss": float(averages[0]), "mean_bce": float(averages[1]),
                        "mean_kl": float(averages[2]), "optimizer_steps": steps, "directed_rows": count,
                        "auxiliary_scene_draws": steps*decoder.TARGET_BATCH})
        print(f"[motion-kl] {arm} epoch={epoch}/{branch.EPOCHS} BCE={averages[1]:.6f} KL={averages[2]:.6f}", flush=True)
    model.eval()
    pilot.write_json(directory/f"{arm}.history.json", {"optimizer": config, "epochs": history})
    with (directory/f"{arm}.last_branch.pt").open("xb") as stream:
        torch.save({"state_dict": head.state_dict(), "magnitude_only": head.magnitude_only, "version": VERSION}, stream)


def clean_diagnostics(directory, arms, threshold):
    baseline = pd.read_csv(directory/"initial.clean_guard_pairs.csv", float_precision="round_trip")
    result = {}
    for arm in arms:
        frame = pd.read_csv(directory/f"{arm}.clean_guard_pairs.csv", float_precision="round_trip")
        np.testing.assert_array_equal(frame.label.to_numpy(), baseline.label.to_numpy())
        rows = {}
        for label, name in ((0, "negative"), (1, "positive")):
            mask = frame.label.eq(label)
            old, new = baseline.probability[mask].to_numpy(), frame.probability[mask].to_numpy()
            if not len(old):
                raise ValueError("missing clean class")
            rows[name] = {"pairs": len(old), "initial_predicted_positive_rate": float((old >= threshold).mean()),
                          "predicted_positive_rate": float((new >= threshold).mean()),
                          "negative_to_positive_rate": float(((old < threshold) & (new >= threshold)).mean()),
                          "positive_to_negative_rate": float(((old >= threshold) & (new < threshold)).mean()),
                          "mean_probability_delta": float((new-old).mean())}
        result[arm] = rows
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    original, binding, parent, mse_binding, anchored_source = preflight()
    context, result, threshold, sample, *_ = original
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol()}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="motion_kl_")
    print(f"[motion-kl] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "cache_parent": binding, "mse_comparison": mse_binding,
                  "source": pilot.bind(__file__), "anchored_source": anchored_source,
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
    development = anchored.development_cache(context, base, sample, targets, device)
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    fit_motion = np.concatenate([replay["features"][:, 1:], targets[mask].reshape(-1, 9)[:, 1:]])
    pilot.trainer.set_seed(decoder.SEED)
    initial = branch.Readout(branch.MotionResidual(fit_motion)).to(device)
    results = {"initial": decoder.evaluate(initial, guard, targets, sample, development, threshold, "initial", directory, device)}
    previous_parameters = None
    for arm in branch.ARMS:
        pilot.trainer.set_seed(decoder.SEED)
        residual = branch.MotionResidual(fit_motion, magnitude_only=arm == "magnitude_only")
        parameters = {k: v.detach().clone() for k, v in residual.named_parameters()}
        if previous_parameters is not None:
            for key in parameters:
                torch.testing.assert_close(parameters[key], previous_parameters[key], rtol=0, atol=0)
        previous_parameters = parameters
        model = branch.Readout(residual).to(device)
        fit(model, replay, targets[mask], directory, arm, device)
        results[arm] = decoder.evaluate(model, guard, targets, sample, development, threshold, arm, directory, device)
    if pilot.initialization_sha(base) != initial_sha:
        raise ValueError("base GNN changed")
    decision = branch.gate(results)
    pilot.write_json(directory/"results.json", results)
    pilot.write_json(directory/"clean_diagnostics.json", clean_diagnostics(directory, branch.ARMS, threshold))
    pilot.write_json(directory/"gate.json", decision)
    pilot.bind(__file__, provenance["source"]["sha256"])
    preflight()
    pilot.write_json(directory/"summary.json", {"status": "KL_BRANCH_COMPLETE_REVIEW_REQUIRED", **provenance,
        "gate_status": decision["status"], "base_model_sha256": initial_sha,
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[motion-kl] {decision['status']}: {directory/'summary.json'}; no further jobs", flush=True)
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
