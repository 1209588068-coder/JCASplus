"""Centered context v6 mechanism pilot; local authorized AV2 research only.

Fresh matched uncentered/centered training; original victim fully frozen.
Not the frozen v6.0 attack, data-only poisoning or random-trigger comparison.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jcas.workflows import relational_context_branch_pilot as previous

pilot, decoder, branch, kl = previous.pilot, previous.decoder, previous.branch, previous.kl
VERSION = "relational_centered_context_mechanism_v6"
ARMS = ("uncentered_context", "centered_context")
EPOCHS = 10
DIAGNOSIS = "record/reproduction/relational_poison_v1/bce_pilot/context_centering_audit_sit3ufz8/summary.json"
DIAGNOSIS_SHA = "e8bf37214c910b2991293342aeaaec48de02d1ac8b9686e46ea989ad2814634c"


def protocol():
    result = previous.protocol()
    result.update(version=VERSION, arms=list(ARMS), epochs=EPOCHS,
        ablation="same context-enabled 9->32->1 parameters; subtract reference raw MLP output only in candidate",
        correction="8*tanh((h(m,c)-h(0,c))/8) centered; 8*tanh(h(m,c)/8) control",
        reference="normalized construct-mean motion, same real context; both h paths receive gradients",
        initialization="fresh matched seed 20260621; zero output; never load trained v5 branch weights",
        comparison_caveat="equal nominal parameters; output bias cancels in centered branch",
        decision="same absolute gates for centered candidate; uncentered comparison reported separately",
        limitations=[s for s in previous.protocol()["limitations"] if "context-control" not in s and "motion_only" not in s]+[
            "mean-motion reference is mathematical, not a physically realized counterfactual",
            "zero correction guaranteed only at reference motion, not all clean inputs",
            "context can still affect motion-dependent corrections; no universal clean protection guarantee",
            "centered/uncentered structure comparison, not random-trigger evidence",
            "mechanism v6 naming is unrelated to frozen v6.0 main experiment"])
    return result


class CenteredContextResidual(previous.ContextResidual):
    def __init__(self, fit_features, *, centered):
        super().__init__(fit_features, context_enabled=True)
        self.centered = bool(centered)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 9:
            raise ValueError("expected frozen logit plus motion8")
        base = features[:, :1].detach()
        motion = ((features[:, 1:]-self.mean)/self.scale).clamp(-10., 10.)
        ctx = ((base-self.context_mean)/self.context_scale).clamp(-10., 10.)
        raw = self.net(torch.cat([motion, ctx], dim=1))
        if self.centered:
            raw = raw-self.net(torch.cat([torch.zeros_like(motion), ctx], dim=1))
        return base+8.*torch.tanh(raw/8.)


def preflight():
    original, kl_asset, cache_parent, context_diagnosis = previous.preflight()
    binding = pilot.bind(DIAGNOSIS, DIAGNOSIS_SHA)
    diagnosis = pilot.read_json(binding["path"])
    if diagnosis["status"] != "CONSTRUCT_CENTERING_AUDIT_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("centering audit incomplete")
    for asset in (diagnosis["source"], diagnosis["parent"]):
        pilot.bind(asset["path"], asset["sha256"])
    channel = pilot.read_json(diagnosis["parent"]["path"])
    for asset in (channel["source"], channel["parent"]):
        pilot.bind(asset["path"], asset["sha256"])
    v5 = pilot.read_json(channel["parent"]["path"])
    pilot.bind(v5["source"]["path"], v5["source"]["sha256"])
    return original, kl_asset, cache_parent, context_diagnosis, binding


def save_branch(head, path, base_checkpoint):
    with path.open("xb") as stream:
        torch.save({"version": VERSION, "centered": head.centered,
                    "state_dict": head.state_dict(), "base_checkpoint": base_checkpoint}, stream)


def load_branch(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != VERSION or type(saved.get("centered")) is not bool:
        raise ValueError("not a centered-context checkpoint")
    head = CenteredContextResidual(np.zeros((1, 9)), centered=saved["centered"])
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
                raise ValueError("nonfinite centered training loss")
            losses[0].backward()
            optimizer.step()
            sums += np.array([float(v.detach()) for v in losses])*len(y)
            count += len(y)
            steps += 1
        total, bce, penalty = sums/count
        history.append({"epoch": epoch, "mean_loss": float(total), "mean_bce": float(bce),
                        "mean_kl": float(penalty), "optimizer_steps": steps, "directed_rows": count,
                        "auxiliary_scene_draws": steps*decoder.TARGET_BATCH})
        print(f"[centered-context] {arm} epoch={epoch}/{EPOCHS} BCE={bce:.6f} KL={penalty:.6f}", flush=True)
    model.eval()
    pilot.write_json(directory/f"{arm}.history.json", {"optimizer": config, "epochs": history})
    save_branch(head, directory/f"{arm}.last_branch.pt", base_checkpoint)


def gate(results):
    value = branch.gate({"initial": results["initial"], "signed_motion": results["centered_context"],
                         "magnitude_only": results["uncentered_context"]})
    value["centered_full_only_lift_over_uncentered"] = value.pop("signed_full_only_lift_over_magnitude")
    value["scope"] = "centered/uncentered mechanism comparison on reused inner-develop targets"
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    original, kl_asset, cache_parent, context_diagnosis, diagnosis = preflight()
    context, result, threshold, sample, replay_rows, *_ = original
    previous.audit.require_construct(replay_rows)
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol(), "graph_arrays_loaded": False}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="centered_context_")
    print(f"[centered-context] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "source": pilot.bind(__file__), "kl_parent": kl_asset,
        "device": args.device, "torch_version": str(torch.__version__), "cpu_threads": 4,
        "context_diagnosis": context_diagnosis, "centering_diagnosis": diagnosis,
        "initial_checkpoint": result["checkpoint"], "threshold": threshold,
        "cache": cache_parent["cache"], "sources": context["sources"]}
    pilot.write_json(directory/"protocol.json", provenance)
    with (directory/"targets.sample.csv").open("x") as stream:
        sample.to_csv(stream, index=False)
    with np.load(provenance["cache"]["path"], allow_pickle=False) as saved:
        targets = saved["targets"]
        replay = {k: saved["replay_"+k] for k in ("features", "labels", "pairs", "used_scenarios")}
        guard = {k: saved["guard_"+k] for k in ("features", "labels", "pairs", "used_scenarios")}
    fit_features, mask = previous.construct_fit_features(replay, targets, sample)
    base = previous.build_model("genconv", **result["model_config"]).to(device)
    base.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device, weights_only=True))
    base.eval()
    base.requires_grad_(False)
    initial_sha = pilot.initialization_sha(base)
    development = previous.anchored.development_cache(context, base, sample, targets, device)
    pilot.trainer.set_seed(decoder.SEED)
    initial = branch.Readout(CenteredContextResidual(fit_features, centered=False)).to(device)
    results = {"initial": decoder.evaluate(initial, guard, targets, sample, development, threshold, "initial", directory, device)}
    initial_parameters = None
    for arm in ARMS:
        pilot.trainer.set_seed(decoder.SEED)
        head = CenteredContextResidual(fit_features, centered=arm == "centered_context")
        parameters = {k: p.detach().clone() for k, p in head.named_parameters()}
        if initial_parameters is not None:
            for k in parameters:
                torch.testing.assert_close(parameters[k], initial_parameters[k], atol=0, rtol=0)
        initial_parameters = parameters
        model = branch.Readout(head).to(device)
        fit(model, replay, targets[mask], directory, arm, device, result["checkpoint"])
        restored = load_branch(directory/f"{arm}.last_branch.pt", device)
        with torch.no_grad():
            check = torch.from_numpy(targets[mask].reshape(-1, 9)).to(device)
            torch.testing.assert_close(restored(check), head(check), atol=0, rtol=0)
            if head.centered:
                reference = check.clone()
                reference[:, 1:] = head.mean
                torch.testing.assert_close(head(reference), reference[:, :1], atol=0, rtol=0)
        results[arm] = decoder.evaluate(model, guard, targets, sample, development, threshold, arm, directory, device)
    if pilot.initialization_sha(base) != initial_sha:
        raise ValueError("base GNN changed")
    decision = gate(results)
    pilot.write_json(directory/"results.json", results)
    pilot.write_json(directory/"comparison.json", [{"arm": arm, **row, "clean_guard_auc": results[arm]["clean_guard"]["auc"]}
        for arm in ("initial", *ARMS) for row in results[arm]["responses"] if row["cohort"] == "develop"])
    pilot.write_json(directory/"clean_diagnostics.json", kl.clean_diagnostics(directory, ARMS, threshold))
    pilot.write_json(directory/"gate.json", decision)
    pilot.bind(__file__, provenance["source"]["sha256"])
    preflight()
    pilot.write_json(directory/"summary.json", {"status": "CENTERED_CONTEXT_COMPLETE_REVIEW_REQUIRED", **provenance,
        "gate_status": decision["status"], "base_model_sha256": initial_sha,
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[centered-context] {decision['status']}: {directory/'summary.json'}; no further jobs", flush=True)
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
