"""Fixed-weight construct-only context ablation, no training or graph reads.

Local authorized AV2 robustness diagnosis. Channel zeroing means normalized
coordinate zero, not physical trajectory edits or a deployable new model.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from jcas.workflows import relational_context_branch_pilot as context

pilot = context.pilot
PARENT = "record/reproduction/relational_poison_v1/bce_pilot/context_branch_gpw_ehs0/summary.json"
PARENT_SHA = "afd70f7b19ad18932860432a8b9a0305ad5f2c401c083a8faec2d5180b9d2cd9"
VIEWS = ("both_on", "context_off", "motion_off", "both_off")


def interventions(head, features):
    """Preserve real base-logit bypass; ablate only standardized MLP inputs."""
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != 9 or not torch.isfinite(x).all():
        raise ValueError("finite Nx9 features required")
    if not head.context_enabled:
        raise ValueError("candidate must be context-enabled")
    with torch.no_grad():
        base = x[:, :1]
        motion = ((x[:, 1:]-head.mean)/head.scale).clamp(-10., 10.)
        ctx = ((base-head.context_mean)/head.context_scale).clamp(-10., 10.)
        outputs = {}
        for name, m, c in (("both_on", motion, ctx), ("context_off", motion, torch.zeros_like(ctx)),
                           ("motion_off", torch.zeros_like(motion), ctx),
                           ("both_off", torch.zeros_like(motion), torch.zeros_like(ctx))):
            outputs[name] = (base+8.*torch.tanh(head.net(torch.cat([m, c], dim=1))/8.)).squeeze(-1).numpy()
        torch.testing.assert_close(torch.from_numpy(outputs["both_on"]), head(x).squeeze(-1), atol=0, rtol=0)
    return outputs


def decomposition(outputs):
    both, motion, ctx, zero = (outputs[k] for k in VIEWS)
    # Anchored two-input decomposition of the learned correction. Base cancels.
    result = {"context_main_logit": ctx-zero, "motion_main_logit": motion-zero,
              "interaction_logit": both-motion-ctx+zero, "context_on_minus_off_logit": both-motion}
    np.testing.assert_allclose(result["context_main_logit"]+result["interaction_logit"],
                               result["context_on_minus_off_logit"], atol=3e-6, rtol=1e-5)
    return result


def probability(z):
    return torch.sigmoid(torch.as_tensor(z)).numpy()


def pair_frame(features, labels, pairs, outputs, control_logits):
    np.testing.assert_array_equal(labels[pairs[:, 0]], labels[pairs[:, 1]])
    frame = pd.DataFrame({"pair_index": np.arange(len(pairs)), "label": labels[pairs[:, 0]],
                          "base_probability": probability(features[:, 0])[pairs].mean(axis=1),
                          "trained_motion_only_probability": probability(control_logits)[pairs].mean(axis=1)})
    for name, z in outputs.items():
        frame[name+"_probability"] = probability(z)[pairs].mean(axis=1)
        frame[name+"_correction"] = (z-features[:, 0])[pairs].mean(axis=1)
    for name, value in decomposition(outputs).items():
        frame[name] = value[pairs].mean(axis=1)
    return frame


def clean_summary(frame, threshold):
    records = []
    bins = [0., .01, .1, threshold, .7, .9, .99, 1.000001]
    frame = frame.copy()
    frame["base_probability_bin"] = pd.cut(frame.base_probability, bins, right=False).astype(str)
    for (label, interval), rows in frame.groupby(["label", "base_probability_bin"], observed=True):
        record = {"label": int(label), "base_probability_bin": interval, "pairs": len(rows)}
        for name in ("base", "trained_motion_only", *VIEWS):
            p = rows[name+"_probability"]
            record[name+"_positive_rate"] = float((p >= threshold).mean())
            record[name+"_mean_probability"] = float(p.mean())
        record["context_on_minus_off_probability"] = float((rows.both_on_probability-rows.context_off_probability).mean())
        for name in decomposition_keys():
            record[name] = float(rows[name].mean())
        records.append(record)
    aggregate = []
    for name in ("base", "trained_motion_only", *VIEWS):
        p, y = frame[name+"_probability"].to_numpy(), frame.label.to_numpy()
        aggregate.append({"view": name, "pairs": len(y), "auc": float(roc_auc_score(y, p)),
                          "positive_recall": float((p[y == 1] >= threshold).mean()),
                          "negative_fp": float((p[y == 0] >= threshold).mean()),
                          "positive_mean_probability": float(p[y == 1].mean()),
                          "negative_mean_probability": float(p[y == 0].mean())})
    return aggregate, records


def decomposition_keys():
    return ("context_main_logit", "motion_main_logit", "interaction_logit", "context_on_minus_off_logit")


def target_summary(frame, threshold):
    records = []
    for name in ("base", "trained_motion_only", *VIEWS):
        clean = frame[name+"_clean_probability"].to_numpy()
        full = frame[name+"_full_probability"].to_numpy()
        reverse = frame[name+"_reversed_probability"].to_numpy()
        record = {"view": name, "construct_targets": len(frame),
                  "clean_activation": float((clean >= threshold).mean()),
                  "full_incremental_asr": float(((clean < threshold) & (full >= threshold)).mean()),
                  "full_activation": float((full >= threshold).mean()),
                  "reversed_activation": float((reverse >= threshold).mean()),
                  "full_only_activation": float(((full >= threshold) & (reverse < threshold)).mean())}
        records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    original, *_ = context.preflight()
    parent_asset = pilot.bind(PARENT, PARENT_SHA)
    parent = pilot.read_json(parent_asset["path"])
    if parent["status"] != "CONTEXT_BRANCH_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("context experiment incomplete")
    for asset in (parent["source"], parent["cache"], parent["evidence"]["targets.sample.csv"],
                  *(parent["evidence"][arm+".last_branch.pt"] for arm in context.ARMS)):
        pilot.bind(asset["path"], asset["sha256"])
    sample = original[3]
    pd.testing.assert_frame_equal(sample, pd.read_csv(parent["evidence"]["targets.sample.csv"]["path"]), check_dtype=False)
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    sample = sample.loc[mask].reset_index(drop=True)
    context.audit.require_construct(sample)
    with np.load(parent["cache"]["path"], allow_pickle=False) as saved:
        replay = {k: saved["replay_"+k] for k in ("features", "labels", "pairs")}
        targets = saved["targets"][mask.to_numpy()]
    torch.set_num_threads(4)
    candidate = context.load_branch(parent["evidence"]["motion_with_context.last_branch.pt"]["path"])
    control = context.load_branch(parent["evidence"]["motion_only.last_branch.pt"]["path"])
    before = {k: v.clone() for k, v in candidate.state_dict().items()}
    clean_outputs = interventions(candidate, replay["features"])
    target_outputs = interventions(candidate, targets.reshape(-1, 9))
    with torch.no_grad():
        control_clean = control(torch.from_numpy(replay["features"])).squeeze(-1).numpy()
        control_targets = control(torch.from_numpy(targets.reshape(-1, 9))).squeeze(-1).numpy()
    clean = pair_frame(replay["features"], replay["labels"], replay["pairs"], clean_outputs, control_clean)
    target_frame = sample[["scenario_id", "split_group_id", "inner_role", "motion_regime"]].copy()
    all_target_outputs = {"base": targets[..., 0].reshape(-1), "trained_motion_only": control_targets, **target_outputs}
    for name, z in all_target_outputs.items():
        probs = probability(z).reshape(-1, 3, 2).mean(axis=2)
        for j, condition in enumerate(("clean", "full", "reversed")):
            target_frame[name+"_"+condition+"_probability"] = probs[:, j]
    decomposition_summary = []
    for name, values in decomposition(target_outputs).items():
        values = values.reshape(-1, 3, 2).mean(axis=2)
        for j, condition in enumerate(("clean", "full", "reversed")):
            target_frame[condition+"_"+name] = values[:, j]
            decomposition_summary.append({"condition": condition, "component": name,
                                          "mean_logit": float(values[:, j].mean())})
    for key, value in candidate.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    if not candidate.context_enabled or control.context_enabled:
        raise ValueError("checkpoint modes changed")
    aggregate, bins = clean_summary(clean, parent["threshold"])
    directory = pilot.new_directory(prefix="context_channel_audit_")
    for name, frame in (("clean_pairs.csv", clean), ("construct_targets.csv", target_frame),
                        ("clean_probability_bins.csv", pd.DataFrame(bins))):
        with (directory/name).open("x") as stream:
            frame.to_csv(stream, index=False)
    pilot.write_json(directory/"summary.json", {"status": "CONSTRUCT_CHANNEL_AUDIT_COMPLETE_REVIEW_REQUIRED",
        "source": pilot.bind(__file__), "parent": parent_asset, "cache": parent["cache"],
        "threshold": parent["threshold"], "clean": aggregate,
        "targets": target_summary(target_frame, parent["threshold"]), "target_decomposition": decomposition_summary,
        "protocol": {"parameters_updated": False, "graph_arrays_loaded": False, "guard_arrays_loaded": False,
                     "target_rows": "construct slow only; slice mixed previous cache", "test_access": False,
                     "ablation": "normalized input zero; real base-logit bypass unchanged",
                     "interpretation": "off-inputs may be out of distribution; local model decomposition, not physical causality",
                     "scope": "training sample diagnosis; no generalization, ASR gate or data-only poisoning claim"},
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[context-channel] complete: {directory/'summary.json'}; no training", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
