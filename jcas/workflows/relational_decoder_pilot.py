"""Bounded readout-training diagnostic, NOT a data-only poisoning experiment.

Freeze geometry_shared's backbone. Compare clean replay against matched
clean/full/reversed supervision for five decoder-only epochs. Construct and
develop groups only; no inner audit, original validation or test graphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from jcas.core.models import build_model
from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_relation_diagnosis as diagnosis

VERSION = "relational_decoder_matched_pilot_v1"
DIAGNOSIS = "record/reproduction/relational_poison_v1/bce_pilot/relation_diagnosis_6xhangk6/summary.json"
DIAGNOSIS_SHA = "d65e423843011233ff40d762924abd89b9c217d1981158f45d5d924c28269fd6"
EPOCHS = 5
REPLAY_GRAPHS = 1024
GUARD_GRAPHS = 1024
CLEAN_BATCH = 1024
TARGET_BATCH = 64
SEED = 20260621
ARMS = ("clean_replay", "matched_relation")


def protocol():
    return {"version": VERSION, "epochs": EPOCHS, "seed": SEED,
            "initial_model": "existing geometry_shared best checkpoint",
            "trainable": "decoder.mlp only", "backbone_mode": "frozen eval",
            "lr": 1e-4, "weight_decay": 1e-4,
            "objective": "ordinary mean BCE over concatenated directed-edge rows",
            "clean_batch_edges": CLEAN_BATCH, "target_batch_scenes": TARGET_BATCH,
            "matched_target_labels": {"clean": 0, "full": 1, "relation_reversed": 0},
            "clean_control": "same auxiliary scenes repeated clean/clean/clean, all labels 0",
            "sampling": "reuse fixed diagnosis targets; SHA-ranked clean replay and guard scenes",
            "target_training": "337 slow construct targets, including 209 former reversed controls",
            "replay_graphs_planned": REPLAY_GRAPHS, "guard_graphs_planned": GUARD_GRAPHS,
            "epoch_definition": "one pass over clean supervised edges; auxiliary scenes sampled with replacement",
            "checkpoint_selection": "last epoch only; no develop-driven selection",
            "threshold": "unchanged original pilot clean-reference threshold",
            "gate_scope": "64 previously diagnosed slow develop targets and sampled develop clean guard",
            "gates": {"full_incremental_asr_min": .60, "reversed_activation_max": .10,
                      "clean_guard_auc_drop_max": .0015},
            "data_only_poisoning_claim": False, "formal_validation_claim": False,
            "inner_audit_access": False, "original_validation_access": False, "test_access": False,
            "automatic_continuation": False,
            "limitations": ["decoder-only training intervention, not the original attacker capability",
                            "develop targets already used for diagnosis; not a fresh independent evaluation",
                            "small clean guard AUC does not certify full-pool clean performance",
                            "failure in five epochs is not an impossibility proof"]}


def select_clean(rows, sample, role, count):
    if role not in ("construct", "develop"):
        raise ValueError("invalid clean partition")
    eligible = rows.loc[rows.split.eq("train") & rows.inner_role.eq(role)
                        & ~rows.scenario_id.isin(sample.scenario_id)].copy()
    eligible["rank"] = eligible.scenario_id.map(
        lambda scene: hashlib.sha256(f"{VERSION}|{role}|{scene}".encode()).hexdigest())
    if len(eligible) < count:
        raise ValueError("insufficient clean guard/replay candidates")
    return eligible.sort_values(["rank", "scenario_id"]).head(count).drop(columns="rank").reset_index(drop=True)


def preflight():
    context = pilot.preflight()
    pilot.bind(diagnosis.TRAINING, diagnosis.TRAINING_SHA)
    results, threshold = pilot.checked_training(diagnosis.TRAINING, context)
    asset = pilot.bind(DIAGNOSIS, DIAGNOSIS_SHA)
    completed = pilot.read_json(asset["path"])
    if completed["status"] != "DIAGNOSIS_COMPLETE_REVIEW_REQUIRED":
        raise ValueError("diagnosis incomplete")
    pilot.bind(completed["diagnosis_source"]["path"], completed["diagnosis_source"]["sha256"])
    sample = diagnosis.sample_targets(context["targets"])
    selected = sample.loc[sample.inner_role.eq("develop") | sample.motion_regime.eq("slow_lt_0p5")].reset_index(drop=True)
    replay = select_clean(context["train_rows"], selected, "construct", REPLAY_GRAPHS)
    guard = select_clean(context["train_rows"], selected, "develop", GUARD_GRAPHS)
    groups = pd.concat([selected[["split_group_id", "inner_role"]], replay[["split_group_id", "inner_role"]],
                        guard[["split_group_id", "inner_role"]]])
    if groups.groupby("split_group_id").inner_role.nunique().gt(1).any():
        raise ValueError("training/develop recording leakage")
    return context, results["geometry_shared"], threshold, selected, replay, guard, asset


def capture_decoder(model, data, device):
    """Cache exact directed decoder inputs, not the pooled probe features."""
    capture = []
    def hook(module, args):
        capture.append(args[0].detach().cpu().numpy().copy())
    handle = model.decoder.mlp.register_forward_pre_hook(hook)
    batch = Batch.from_data_list(data).to(device)
    try:
        model.eval()
        with torch.inference_mode():
            original = model(batch.x, batch.edge_index, batch.edge_attr, batch.message_passing_edge_mask)
    finally:
        handle.remove()
    if len(capture) != 1 or not np.isfinite(capture[0]).all():
        raise ValueError("invalid decoder capture")
    z = capture[0]
    with torch.inference_mode():
        cached = model.decoder.mlp(torch.from_numpy(z).to(device)).squeeze(-1)
    torch.testing.assert_close(cached, original, atol=1e-5, rtol=1e-5)
    return np.split(z, np.cumsum([item.num_edges for item in data])[:-1])


def supervised_pairs(data):
    """Pair indices relative to selected supervised rows, matching frozen AUC."""
    supervised = np.flatnonzero(data.supervision_edge_mask.numpy())
    edges = data.edge_index.numpy()[:, supervised]
    labels = data.edge_label.numpy()[supervised]
    groups = {}
    for i, (src, dst) in enumerate(edges.T):
        groups.setdefault(tuple(sorted((int(src), int(dst)))), []).append(i)
    pairs = [ids for ids in groups.values() if len(ids) == 2]
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) and not np.all(labels[pairs[:, 0]] == labels[pairs[:, 1]]):
        raise ValueError("asymmetric clean pair labels")
    return supervised, pairs


def cache_clean(context, model, rows, device, label):
    features, labels, pair_indices, used = [], [], [], []
    edge_offset = 0
    for i, row in enumerate(rows.itertuples(index=False), 1):
        path = diagnosis.graph_path(context, row)
        with np.load(path, allow_pickle=True) as graph:
            clean, _ = pilot.clean_data(graph, row.scenario_id)
        ids, pairs = supervised_pairs(clean)
        if len(ids):
            z = capture_decoder(model, [clean], device)[0]
            features.append(z[ids])
            labels.append(clean.edge_label.numpy()[ids])
            pair_indices.append(pairs+edge_offset)
            edge_offset += len(ids)
            used.append(str(row.scenario_id))
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(f"[decoder-pilot] cache {label} {i}/{len(rows)}", flush=True)
    if not features:
        raise ValueError("no supervised clean graphs")
    result = {"features": np.concatenate(features), "labels": np.concatenate(labels),
              "pairs": np.concatenate(pair_indices), "used_scenarios": np.asarray(used, dtype=str)}
    if len(np.unique(result["labels"][result["pairs"][:, 0]])) != 2:
        raise ValueError("clean guard/replay lacks both pair classes")
    return result


def cache_targets(context, model, sample, device):
    indexed = context["train_rows"].set_index("scenario_id", drop=False)
    target_features, development = [], []
    for i, target in enumerate(sample.to_dict("records"), 1):
        path = diagnosis.graph_path(context, indexed.loc[target["scenario_id"]])
        with np.load(path, allow_pickle=True) as graph:
            clean, labels = pilot.clean_data(graph, target["scenario_id"])
            ids = pilot.checked_target(graph, target, labels)
            data = [clean]+[pilot.transformed_data(graph, clean, target, ids, controller=True,
                variant=variant, poison_label=False) for variant in diagnosis.CONDITIONS[1:]]
        features = capture_decoder(model, data, device)
        target_features.append(np.stack([z[ids] for z in features]))
        if target["inner_role"] == "develop":
            development.append({"target": target, "clean": clean, "ids": ids, "features": features})
        if i == 1 or i % 50 == 0 or i == len(sample):
            print(f"[decoder-pilot] cache targets {i}/{len(sample)}", flush=True)
    return np.stack(target_features), development


def auxiliary_batch(features, indices, arm):
    if arm not in ARMS:
        raise ValueError("unknown training arm")
    x = features[indices].copy()  # [scene, clean/full/reverse, two directed edges, hidden]
    y = np.zeros(x.shape[:3], dtype=np.float32)
    if arm == "matched_relation":
        y[:, 1] = 1.
    else:
        x[:, 1:] = x[:, :1]
    return x.reshape(-1, x.shape[-1]), y.reshape(-1)


def freeze_for_readout(model):
    model.eval()
    model.requires_grad_(False)
    model.decoder.mlp.requires_grad_(True)
    return model.decoder.mlp


def backbone_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("decoder.mlp."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_readout(model, replay, target_features, arm, directory, device):
    pilot.trainer.set_seed(SEED)
    head = freeze_for_readout(model)
    original_backbone = backbone_sha(model)
    optimizer, config = pilot.trainer.build_optimizer(head, lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, EPOCHS+1):
        order = rng.permutation(len(replay["labels"]))
        head.train()
        loss_sum, count, steps = 0., 0, 0
        for start in range(0, len(order), CLEAN_BATCH):
            clean_ids = order[start:start+CLEAN_BATCH]
            target_ids = rng.integers(0, len(target_features), size=TARGET_BATCH)
            extra_x, extra_y = auxiliary_batch(target_features, target_ids, arm)
            x = torch.from_numpy(np.concatenate([replay["features"][clean_ids], extra_x])).to(device)
            y = torch.from_numpy(np.concatenate([replay["labels"][clean_ids], extra_y])).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(x).squeeze(-1), y)
            if not torch.isfinite(loss):
                raise ValueError("non-finite decoder loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(y)
            count += len(y)
            steps += 1
        history.append({"epoch": epoch, "mean_bce": loss_sum/count, "directed_rows": count,
                        "optimizer_steps": steps, "auxiliary_scene_draws": steps*TARGET_BATCH})
        print(f"[decoder-pilot] {arm} epoch={epoch}/{EPOCHS} BCE={loss_sum/count:.6f}", flush=True)
    model.eval()
    if backbone_sha(model) != original_backbone:
        raise ValueError("frozen backbone changed")
    pilot.write_json(directory/f"{arm}.history.json", {"optimizer": config, "epochs": history,
        "backbone_sha256": original_backbone, "checkpoint_selection": "last epoch"})
    with (directory/f"{arm}.last_model.pt").open("xb") as stream:
        torch.save(model.state_dict(), stream)
    return history


def probabilities(head, features, device):
    head.eval()
    arrays = []
    with torch.inference_mode():
        for start in range(0, len(features), 4096):
            arrays.append(torch.sigmoid(head(torch.from_numpy(features[start:start+4096]).to(device)).squeeze(-1)).cpu().numpy())
    return np.concatenate(arrays)


def evaluate(model, guard, target_features, sample, development, threshold, arm, directory, device):
    head = model.decoder.mlp
    guard_prob = probabilities(head, guard["features"], device)
    pair_prob = guard_prob[guard["pairs"]].mean(axis=1)
    pair_labels = guard["labels"][guard["pairs"][:, 0]]
    metrics = pilot.trainer.metric_dict(pair_labels, pair_prob, threshold)
    with (directory/f"{arm}.clean_guard_pairs.csv").open("x") as stream:
        pd.DataFrame({"label": pair_labels, "probability": pair_prob.astype(np.float64)}).to_csv(stream, index=False)
    shape = target_features.shape
    scores = probabilities(head, target_features.reshape(-1, shape[-1]), device).reshape(shape[:3]).mean(axis=2)
    response = sample[["scenario_id", "split_group_id", "inner_role", "cohort", "motion_regime"]].copy()
    response["model"] = arm
    for index, name in enumerate(diagnosis.CONDITIONS):
        response[name] = scores[:, index].astype(np.float64)
    with (directory/f"{arm}.responses.csv").open("x") as stream:
        response.to_csv(stream, index=False)
    collateral = []
    for item in development:
        probs = [probabilities(head, z, device) for z in item["features"]]
        for index, variant in ((1, "full"), (2, "relation_reversed")):
            collateral.append({"model": arm, "controller": True, "variant": variant,
                **pilot.response_record(item["target"], item["clean"], item["ids"], probs[0], probs[index], threshold)})
    with (directory/f"{arm}.collateral.jsonl").open("x") as stream:
        for record in collateral:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
    return {"clean_guard": metrics, "responses": diagnosis.response_summary(response, threshold),
            "collateral": pilot.summarize(collateral, threshold)}


def gate(results):
    def slow(arm):
        matches = [r for r in results[arm]["responses"] if r["cohort"] == "develop" and r["motion_regime"] == "slow_lt_0p5"]
        if len(matches) != 1:
            raise ValueError("missing slow develop response")
        return matches[0]
    candidate, control, initial = (slow(arm) for arm in ("matched_relation", "clean_replay", "initial"))
    auc = results["matched_relation"]["clean_guard"]["auc"]
    original_auc = results["initial"]["clean_guard"]["auc"]
    drop = original_auc-auc if auc is not None and original_auc is not None else None
    values = {"full_incremental_asr": candidate["full_incremental_asr"],
              "reversed_activation": candidate["reversed_activation"], "clean_guard_auc_drop": drop}
    limits = {"full_incremental_asr": (">=", .60), "reversed_activation": ("<=", .10),
              "clean_guard_auc_drop": ("<=", .0015)}
    checks = {key: {"value": values[key], "operator": op, "limit": bound,
                   "pass": values[key] is not None and np.isfinite(values[key]).item()
                           and (values[key] >= bound if op == ">=" else values[key] <= bound)}
              for key, (op, bound) in limits.items()}
    return {"status": "PASS" if all(c["pass"] for c in checks.values()) else "STOP", "checks": checks,
            "full_only_lift_over_replay": candidate["full_only_activation"]-control["full_only_activation"],
            "full_only_lift_over_initial": candidate["full_only_activation"]-initial["full_only_activation"],
            "scope": "small decoder-only training diagnostic; not data-only poisoning or formal validation",
            "automatic_continuation": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    context, result, threshold, sample, replay_rows, guard_rows, prior = preflight()
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_OK", "protocol": protocol(),
                         "target_counts": sample.groupby(["cohort", "motion_regime"]).size().rename("count").reset_index().to_dict("records")}, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    directory = pilot.new_directory(prefix="decoder_pilot_")
    print(f"[decoder-pilot] output: {directory}", flush=True)
    provenance = {"protocol": protocol(), "prior_diagnosis": prior, "initial_checkpoint": result["checkpoint"],
                  "sources": context["sources"], "source": pilot.bind(__file__), "threshold": threshold}
    pilot.write_json(directory/"protocol.json", provenance)
    for name, rows in (("targets", sample), ("clean_replay", replay_rows), ("clean_guard", guard_rows)):
        with (directory/f"{name}.sample.csv").open("x") as stream:
            rows.to_csv(stream, index=False)
    model = build_model("genconv", **result["model_config"]).to(device)
    model.load_state_dict(torch.load(result["checkpoint"]["path"], map_location=device, weights_only=True))
    model.eval()
    model.requires_grad_(False)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    original_sha = pilot.initialization_sha(model)
    replay = cache_clean(context, model, replay_rows, device, "construct replay")
    guard = cache_clean(context, model, guard_rows, device, "develop guard")
    target_features, development = cache_targets(context, model, sample, device)
    with (directory/"decoder_cache.npz").open("xb") as stream:
        np.savez_compressed(stream, target_features=target_features,
            **{"replay_"+k: v for k, v in replay.items()}, **{"guard_"+k: v for k, v in guard.items()})
    results = {"initial": evaluate(model, guard, target_features, sample, development, threshold, "initial", directory, device)}
    train_mask = sample.inner_role.eq("construct") & sample.motion_regime.eq("slow_lt_0p5")
    for arm in ARMS:
        model.load_state_dict(state)
        if pilot.initialization_sha(model) != original_sha:
            raise ValueError("arm initialization differs")
        train_readout(model, replay, target_features[train_mask], arm, directory, device)
        results[arm] = evaluate(model, guard, target_features, sample, development, threshold, arm, directory, device)
    decision = gate(results)
    pilot.write_json(directory/"results.json", results)
    pilot.write_json(directory/"gate.json", decision)
    pilot.bind(__file__, provenance["source"]["sha256"])
    pilot.checked_training(diagnosis.TRAINING, pilot.preflight())
    pilot.write_json(directory/"summary.json", {"status": "DECODER_PILOT_COMPLETE_REVIEW_REQUIRED", **provenance,
        "gate_status": decision["status"], "backbone_sha256": backbone_sha(model),
        "replay_graphs_used": len(replay["used_scenarios"]), "guard_graphs_used": len(guard["used_scenarios"]),
        "guard_pairs": len(guard["pairs"]), "training_targets": int(train_mask.sum()),
        "evidence": {p.name: pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation": False})
    print(f"[decoder-pilot] {decision['status']}: {directory/'summary.json'}; no further jobs", flush=True)
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
