"""Read-only supplement: base-logit-only probes and bias-excluded gradients.

Fixed controls for construct conflict interpretation; no optimizer, new victim,
branch fitting, graph reads, or held-out evaluation access.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from jcas.workflows import relational_construct_conflict_audit as audit

pilot = audit.pilot
PARENT = "record/reproduction/relational_poison_v1/bce_pilot/construct_conflict_2se4hjqs/summary.json"
PARENT_SHA = "8d6a0fe79167c447ea7ec9c1d332f60748e0b99a936076f29cb2d13006c97ef0"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    original, _, kl_parent = audit.preflight()
    binding = pilot.bind(PARENT, PARENT_SHA)
    parent = pilot.read_json(binding["path"])
    if parent["status"] != "CONSTRUCT_DIAGNOSIS_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("incomplete construct diagnosis")
    pilot.bind(parent["source"]["path"], parent["source"]["sha256"])
    for name in ("clean_damage.csv", "probe_predictions.csv", "construct_targets.csv"):
        item = parent["evidence"][name]
        pilot.bind(item["path"], item["sha256"])
    sample = original[3]
    mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    with np.load(kl_parent["cache"]["path"], allow_pickle=False) as saved:
        replay = {k: saved["replay_"+k] for k in ("features", "labels", "pairs")}
        targets = saved["targets"][mask.to_numpy()]
    sample = sample.loc[mask].reset_index(drop=True)
    audit.require_construct(sample)
    pd.testing.assert_frame_equal(sample, pd.read_csv(parent["evidence"]["construct_targets.csv"]["path"]), check_dtype=False)
    damage = pd.read_csv(parent["evidence"]["clean_damage.csv"]["path"])
    reference = pd.read_csv(parent["evidence"]["probe_predictions.csv"]["path"])
    pair_x = replay["features"][replay["pairs"]].mean(axis=1)
    target_x = targets.mean(axis=2)
    y, groups, tg = damage.label.to_numpy(), damage.split_group_id.to_numpy(), sample.split_group_id.to_numpy()
    np.testing.assert_array_equal(damage.pair_index, np.arange(len(pair_x)))
    np.testing.assert_array_equal(y, replay["labels"][replay["pairs"][:, 0]])
    tasks = {"full_vs_reversed": (target_x[:, 1], target_x[:, 2], tg, tg),
             "full_vs_normal_negative": (target_x[:, 1], pair_x[y == 0], tg, groups[y == 0]),
             "reversed_vs_normal_positive": (target_x[:, 2], pair_x[y == 1], tg, groups[y == 1])}
    torch.set_num_threads(4)
    metrics, predictions = [], []
    for task, (a, b, ga, gb) in tasks.items():
        x, yy, gg = np.concatenate([a, b]), np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[ga, gb]
        rows, scores, folds = audit.probe_cv(x[:, :1], yy, gg)
        prior = reference.loc[reference.task.eq(task) & reference.view.eq("motion8")].sort_values("row_index")
        np.testing.assert_array_equal(prior.label, yy)
        np.testing.assert_array_equal(prior.split_group_id, gg)
        np.testing.assert_array_equal(prior.fold, folds)
        metrics.append({"task": task, "view": "frozen_logit_only", "folds": rows,
                        "mean_fold_auc": float(np.mean([r["auc"] for r in rows]))})
        predictions.append(pd.DataFrame({"task": task, "row_index": np.arange(len(yy)),
            "label": yy, "split_group_id": gg, "fold": folds, "score": scores}))
        print(f"[context-control] {task} logit only: {metrics[-1]['mean_fold_auc']:.6f}", flush=True)
    head = audit.branch.MotionResidual(np.zeros((1, 8)))
    checkpoint = kl_parent["evidence"]["signed_motion.last_branch.pt"]
    saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    head.load_state_dict(saved["state_dict"])
    head.eval()
    before = {k: v.clone() for k, v in head.state_dict().items()}
    parameter_names = np.concatenate([np.repeat(name, p.numel()) for name, p in head.named_parameters()])
    slices = {"all_parameters": np.ones(len(parameter_names), dtype=bool),
              "weights_only": np.array([name.endswith("weight") for name in parameter_names]),
              "first_layer_weights": parameter_names == "net.0.weight",
              "output_weights": parameter_names == "net.2.weight"}
    rng, records = np.random.default_rng(20260621), []
    for batch in range(32):
        ids = rng.choice(len(replay["labels"]), min(1024, len(replay["labels"])), replace=False)
        extra = targets[rng.integers(len(targets), size=64)]
        x = replay["features"][ids]
        full = audit.gradient(head, extra[:, 1].reshape(-1, 9), np.ones(128))
        reverse = audit.gradient(head, extra[:, 2].reshape(-1, 9), np.zeros(128))
        trigger = (full+reverse)*128/(len(x)+384)
        clean = audit.gradient(head, np.concatenate([x, extra[:, 0].reshape(-1, 9)]))
        for name, selected in slices.items():
            records.append({"batch": batch, "parameters": name, "cosine": audit.cosine(trigger[selected], clean[selected])})
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    directory = pilot.new_directory(prefix="construct_context_control_")
    gradients = pd.DataFrame(records)
    for name, frame in (("logit_probe_predictions.csv", pd.concat(predictions, ignore_index=True)),
                        ("layer_gradients.csv", gradients)):
        with (directory/name).open("x") as stream:
            frame.to_csv(stream, index=False)
    summary = gradients.groupby("parameters").agg(mean_cosine=("cosine", "mean"),
        conflict_fraction=("cosine", lambda x: float((x.dropna() < 0).mean()))).reset_index().to_dict("records")
    pilot.bind(checkpoint["path"], checkpoint["sha256"])
    pilot.write_json(directory/"summary.json", {"status": "CONSTRUCT_CONTEXT_CONTROL_COMPLETE_REVIEW_REQUIRED",
        "parent": binding, "source": pilot.bind(__file__), "probes": metrics, "gradient_summary": summary,
        "same_probe_folds_verified": True, "graph_arrays_loaded": False, "parameter_updates": 0,
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "limitations": audit.protocol()["limitations"], "automatic_continuation": False})
    print(f"[context-control] review: {directory/'summary.json'}; no training started", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
