"""v5 fixed context-input ablation for local authorized AV2 robustness research.

Frozen geometry_shared logit plus 8D motion residual; zero-context control and
enabled-context candidate. Architecture diagnostic, not data-only poisoning.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_decoder_pilot as decoder
from jcas.workflows import relational_motion_branch_pilot as branch
from jcas.workflows import relational_motion_branch_anchored as anchored
from jcas.workflows import relational_motion_branch_kl as kl
from jcas.workflows import relational_construct_conflict_audit as audit

VERSION = "relational_motion_context_branch_kl_v5"
ARMS = ("motion_only", "motion_with_context")
EPOCHS = 10
CONTEXT_SUMMARY = "record/reproduction/relational_poison_v1/bce_pilot/construct_context_control_4trkbwtg/summary.json"
CONTEXT_SHA = "075e0fe610371fe71d288053ec9d510e3cb2a6d0e7b443865543adce847908e1"


def protocol():
    value = kl.protocol()
    value.update(version=VERSION, epochs=EPOCHS, arms=list(ARMS),
        branch="9->32->1 SiLU; 8 signed motion coordinates plus frozen base logit context",
        ablation="same parameter tensors; context coordinate zero in motion_only",
        context="same-input frozen geometry_shared directed logit; detached, not clean counterfactual",
        context_normalizer="construct replay plus construct auxiliary clean/full/reversed; std floor 1e-4; clamp +/-10",
        initialization="v4 random initialization rule for motion weights; appended context weights zero; output layer zero; fresh seed",
        initial_weights="fresh zero-output branches, never continuation of failed weights",
        comparison_caveat="control has an inactive context column; effective input information differs",
        decision="unchanged absolute gates; report context-control differences separately",
        epochs_without_intermediate_evaluation=True,
        limitations=[s for s in kl.protocol()["limitations"] if "magnitude ablation" not in s]+[
            "base logit context is model information, not a model-free trigger input",
            "context-control comparison is not signed/magnitude or random-trigger comparison",
            "passing absolute gates alone does not establish a benefit over motion_only"])
    return value


class ContextResidual(branch.MotionResidual):
    def __init__(self, fit_features, *, context_enabled):
        values = np.asarray(fit_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 9 or not len(values) or not np.isfinite(values).all():
            raise ValueError("expected finite construct-only Nx9 fit features")
        super().__init__(values[:, 1:])
        self.context_enabled = bool(context_enabled)
        self.register_buffer("context_mean", torch.tensor(values[:, 0].mean(), dtype=torch.float32))
        self.register_buffer("context_scale", torch.tensor(max(values[:, 0].std(), 1e-4), dtype=torch.float32))
        original = self.net[0]
        widened = nn.Linear(9, 32)
        with torch.no_grad():
            widened.weight[:, :8].copy_(original.weight)
            widened.weight[:, 8].zero_()
            widened.bias.copy_(original.bias)
        self.net[0] = widened

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 9:
            raise ValueError("expected base logit plus signed motion8")
        base = features[:, :1].detach()
        motion = ((features[:, 1:]-self.mean)/self.scale).clamp(-10., 10.)
        context = ((base-self.context_mean)/self.context_scale).clamp(-10., 10.)
        if not self.context_enabled:
            context = torch.zeros_like(context)
        correction = 8.*torch.tanh(self.net(torch.cat([motion, context], dim=1))/8.)
        return base+correction


def construct_fit_features(replay, targets, sample):
    if targets.shape != (len(sample), 3, 2, 9):
        raise ValueError("target sample/cache shape mismatch")
    selected = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    audit.require_construct(sample.loc[selected])
    return np.concatenate([replay["features"], targets[selected.to_numpy()].reshape(-1, 9)]), selected.to_numpy()


def preflight():
    original, kl_asset, parent = audit.preflight()
    asset = pilot.bind(CONTEXT_SUMMARY, CONTEXT_SHA)
    diagnosis = pilot.read_json(asset["path"])
    if diagnosis["status"] != "CONSTRUCT_CONTEXT_CONTROL_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("context diagnostic incomplete")
    for item in (diagnosis["source"], diagnosis["parent"]):
        pilot.bind(item["path"], item["sha256"])
    main_diagnosis = pilot.read_json(diagnosis["parent"]["path"])
    pilot.bind(main_diagnosis["source"]["path"], main_diagnosis["source"]["sha256"])
    return original, kl_asset, parent, asset


def save_checkpoint(head, path, base_checkpoint):
    with path.open("xb") as stream:
        torch.save({"version": VERSION, "state_dict": head.state_dict(),
                    "context_enabled": head.context_enabled, "base_checkpoint": base_checkpoint}, stream)


def load_branch(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != VERSION or type(saved.get("context_enabled")) is not bool:
        raise ValueError("not a v5 context branch checkpoint")
    head = ContextResidual(np.zeros((1, 9)), context_enabled=saved["context_enabled"])
    head.load_state_dict(saved["state_dict"])
    return head.to(device).eval()


def fit(model, replay, targets, directory, arm, device, base_checkpoint):
    pilot.trainer.set_seed(decoder.SEED)
    head = model.decoder.mlp
    optimizer, config = pilot.trainer.build_optimizer(head, lr=branch.LR, weight_decay=1e-4)
    rng, history = np.random.default_rng(decoder.SEED), []
    for epoch in range(1, EPOCHS+1):
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
            losses = kl.kl_loss(head(x).squeeze(-1), x, y, len(ids))
            if not all(torch.isfinite(v) for v in losses):
                raise ValueError("non-finite context training loss")
            losses[0].backward()
            optimizer.step()
            sums += np.array([float(v.detach()) for v in losses])*len(y)
            count += len(y)
            steps += 1
        total, bce, penalty = sums/count
        history.append({"epoch": epoch, "mean_loss": float(total), "mean_bce": float(bce),
                        "mean_kl": float(penalty), "optimizer_steps": steps, "directed_rows": count,
                        "auxiliary_scene_draws": steps*decoder.TARGET_BATCH})
        print(f"[context-branch] {arm} epoch={epoch}/{EPOCHS} BCE={bce:.6f} KL={penalty:.6f}", flush=True)
    model.eval()
    pilot.write_json(directory/f"{arm}.history.json", {"optimizer": config, "epochs": history})
    save_checkpoint(head, directory/f"{arm}.last_branch.pt", base_checkpoint)


def gate(results):
    # Reuse exact v4 checks, but never mislabel context control as magnitude ablation.
    checked = branch.gate({"initial": results["initial"], "signed_motion": results["motion_with_context"],
                           "magnitude_only": results["motion_only"]})
    checked["context_full_only_lift_over_motion_only"] = checked.pop("signed_full_only_lift_over_magnitude")
    checked["scope"] = "context-input mechanism diagnostic on reused inner-develop samples"
    checked["comparison"] = "motion_with_context versus motion_only; not magnitude or random trigger"
    return checked


def comparison(results):
    rows = []
    for arm in ("initial", *ARMS):
        for response in results[arm]["responses"]:
            if response["cohort"] == "develop":
                rows.append({"arm": arm, **response, "clean_guard_auc": results[arm]["clean_guard"]["auc"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    original, kl_asset, parent, diagnosis = preflight()
    context, result, threshold, sample, replay_rows, *_ = original
    audit.require_construct(replay_rows)
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol(),
                          "graph_arrays_loaded": False, "training_started": False}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="context_branch_")
    print(f"[context-branch] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "source": pilot.bind(__file__), "kl_parent": kl_asset,
                  "context_diagnosis": diagnosis, "initial_checkpoint": result["checkpoint"],
                  "threshold": threshold, "cache": parent["cache"], "sources": context["sources"]}
    pilot.write_json(directory/"protocol.json", provenance)
    with (directory/"targets.sample.csv").open("x") as stream:
        sample.to_csv(stream, index=False)
    with np.load(parent["cache"]["path"], allow_pickle=False) as saved:
        targets = saved["targets"]
        replay = {k: saved["replay_"+k] for k in ("features", "labels", "pairs", "used_scenarios")}
        guard = {k: saved["guard_"+k] for k in ("features", "labels", "pairs", "used_scenarios")}
    fit_features, mask = construct_fit_features(replay, targets, sample)
    base = build_model("genconv", **result["model_config"]).to(device)
    base.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device, weights_only=True))
    base.eval()
    base.requires_grad_(False)
    initial_sha = pilot.initialization_sha(base)
    # Existing inner-develop graphs only, for matched side-effect reporting, never model updates.
    development = anchored.development_cache(context, base, sample, targets, device)
    pilot.trainer.set_seed(decoder.SEED)
    initial = branch.Readout(ContextResidual(fit_features, context_enabled=False)).to(device)
    results = {"initial": decoder.evaluate(initial, guard, targets, sample, development, threshold, "initial", directory, device)}
    initial_parameters = None
    for arm in ARMS:
        pilot.trainer.set_seed(decoder.SEED)
        head = ContextResidual(fit_features, context_enabled=arm == "motion_with_context")
        parameters = {k: p.detach().clone() for k, p in head.named_parameters()}
        if initial_parameters is not None:
            for key in parameters:
                torch.testing.assert_close(parameters[key], initial_parameters[key], atol=0, rtol=0)
        initial_parameters = parameters
        model = branch.Readout(head).to(device)
        fit(model, replay, targets[mask], directory, arm, device, result["checkpoint"])
        restored = load_branch(directory/f"{arm}.last_branch.pt", device)
        with torch.no_grad():
            check = torch.from_numpy(targets[mask].reshape(-1, 9)).to(device)
            torch.testing.assert_close(restored(check), head(check), atol=0, rtol=0)
        results[arm] = decoder.evaluate(model, guard, targets, sample, development, threshold, arm, directory, device)
    if pilot.initialization_sha(base) != initial_sha:
        raise ValueError("base GNN changed")
    decision = gate(results)
    pilot.write_json(directory/"results.json", results)
    pilot.write_json(directory/"comparison.json", comparison(results))
    pilot.write_json(directory/"clean_diagnostics.json", kl.clean_diagnostics(directory, ARMS, threshold))
    pilot.write_json(directory/"gate.json", decision)
    pilot.bind(__file__, provenance["source"]["sha256"])
    preflight()
    pilot.write_json(directory/"summary.json", {"status": "CONTEXT_BRANCH_COMPLETE_REVIEW_REQUIRED", **provenance,
        "gate_status": decision["status"], "base_model_sha256": initial_sha,
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[context-branch] {decision['status']}: {directory/'summary.json'}; no further jobs", flush=True)
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
