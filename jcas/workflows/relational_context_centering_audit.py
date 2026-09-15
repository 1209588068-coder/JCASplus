"""Construct-only algebraic centering check of a fixed v5 checkpoint.

Not a new trained branch or deployment candidate. No graph/guard reads.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from jcas.workflows import relational_context_channel_audit as audit

pilot = audit.pilot
PARENT = "record/reproduction/relational_poison_v1/bce_pilot/context_channel_audit_ilgppjc2/summary.json"
PARENT_SHA = "264f3349c5e423f275bc6e576353bd5fb7f4c9b4cba0244d9590aa68a2accc07"


def centered_logits(base, outputs):
    parts = audit.decomposition(outputs)
    return {"base": base, "original_context_branch": outputs["both_on"],
            "remove_motion_independent_correction": base+outputs["both_on"]-outputs["motion_off"],
            "interaction_only": base+parts["interaction_logit"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    original, *_ = audit.context.preflight()
    binding = pilot.bind(PARENT, PARENT_SHA)
    previous = pilot.read_json(binding["path"])
    for asset in (previous["source"], previous["parent"], previous["cache"]):
        pilot.bind(asset["path"], asset["sha256"])
    trained = pilot.read_json(previous["parent"]["path"])
    checkpoint = trained["evidence"]["motion_with_context.last_branch.pt"]
    pilot.bind(checkpoint["path"], checkpoint["sha256"])
    sample = original[3]
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    sample = sample.loc[mask].reset_index(drop=True)
    audit.context.audit.require_construct(sample)
    with np.load(previous["cache"]["path"], allow_pickle=False) as saved:
        replay = {k: saved["replay_"+k] for k in ("features", "labels", "pairs")}
        targets = saved["targets"][mask.to_numpy()]
    torch.set_num_threads(4)
    head = audit.context.load_branch(checkpoint["path"])
    before = {k: v.clone() for k, v in head.state_dict().items()}
    x = replay["features"]
    clean_z = centered_logits(x[:, 0], audit.interventions(head, x))
    tx = targets.reshape(-1, 9)
    target_z = centered_logits(tx[:, 0], audit.interventions(head, tx))
    pairs, threshold = replay["pairs"], trained["threshold"]
    labels = replay["labels"][pairs[:, 0]]
    clean = pd.DataFrame({"pair_index": np.arange(len(pairs)), "label": labels})
    target = sample[["scenario_id", "split_group_id", "inner_role"]].copy()
    results = []
    for view in clean_z:
        p = audit.probability(clean_z[view])[pairs].mean(axis=1)
        q = audit.probability(target_z[view]).reshape(-1, 3, 2).mean(axis=2)
        clean[view+"_probability"] = p
        for j, condition in enumerate(("clean", "full", "reversed")):
            target[view+"_"+condition+"_probability"] = q[:, j]
        results.append({"view": view, "construct_clean_pairs": len(p), "construct_targets": len(q),
            "clean_auc": float(roc_auc_score(labels, p)), "positive_recall": float((p[labels == 1] >= threshold).mean()),
            "negative_fp": float((p[labels == 0] >= threshold).mean()),
            "target_clean_activation": float((q[:, 0] >= threshold).mean()),
            "full_incremental_asr": float(((q[:, 0] < threshold) & (q[:, 1] >= threshold)).mean()),
            "reversed_activation": float((q[:, 2] >= threshold).mean()),
            "full_only_activation": float(((q[:, 1] >= threshold) & (q[:, 2] < threshold)).mean())})
    for k, v in head.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)
    directory = pilot.new_directory(prefix="context_centering_audit_")
    for name, frame in (("clean_pairs.csv", clean), ("construct_targets.csv", target)):
        with (directory/name).open("x") as stream:
            frame.to_csv(stream, index=False)
    pilot.bind(checkpoint["path"], checkpoint["sha256"])
    pilot.write_json(directory/"summary.json", {"status": "CONSTRUCT_CENTERING_AUDIT_COMPLETE_REVIEW_REQUIRED",
        "parent": binding, "source": pilot.bind(__file__), "checkpoint": checkpoint, "threshold": threshold,
        "results": results, "parameter_updates": 0, "graph_arrays_loaded": False,
        "guard_arrays_loaded": False, "test_access": False, "automatic_continuation": False,
        "limitations": ["algebraic intervention on fixed weights, not a retrained model",
                        "normalized mean baselines may be out of distribution",
                        "subtractions need not retain original correction bounds; diagnostic only",
                        "construct-only response; no evaluation gate or independent generalization claim"],
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()}})
    print(f"[context-centering] complete: {directory/'summary.json'}; no training", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
