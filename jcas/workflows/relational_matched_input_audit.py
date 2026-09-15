"""Construct-only covariate-matched input separability audit.

Local authorized AV2 research. No victim loads, scores, training or test access.
Fixed diagnostic probes are not attack success or learned-trigger optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jcas.workflows import relational_bce_pilot as pilot
from jcas.workflows import relational_relation_diagnosis as diagnosis
from jcas.workflows import relational_decoder_pilot as decoder

VERSION = "relational_matched_input_separability_v1"
COVARIATES = ("min_speed", "max_speed", "distance", "closing_speed", "log_nodes", "log_degree")
CONDITIONS = ("clean", "full", "relation_reversed")
VIEWS = {"base34": slice(0, 34), "motion8": slice(34, 42), "combined42": slice(0, 42)}


def rank(value):
    return hashlib.sha256(f"{VERSION}|{value}".encode()).hexdigest()


def group_fold(group):
    return int(rank("fold|"+str(group)), 16) % 5


def templates():
    """Match endpoint displacement RMS, only downscale under all old limits."""
    original = pilot.trigger.template_bank()
    initial = {name: pilot.trigger.make_template(np.asarray(t["coefficients"]), relative_budget_fraction=.9)
               for name, t in original.items()}
    rms = {name: float(np.sqrt(np.mean(np.sum((np.asarray(t["relative_wave_m"])/2)**2, axis=1))))
           for name, t in initial.items()}
    common = min(rms.values())
    matched = {name: pilot.trigger.make_template(np.asarray(t["coefficients"]),
               relative_budget_fraction=.9*common/rms[name]) for name, t in initial.items()}
    for t in matched.values():
        wave = np.asarray(t["relative_wave_m"])
        np.testing.assert_allclose(np.sqrt(np.mean(np.sum((wave/2)**2, axis=1))), common, atol=1e-12)
        if not pilot.trigger.budget_ok(-wave/2, wave/2, pilot.trigger.reference_budget()):
            raise ValueError("matched template outside budget")
    return matched, common


def protocol():
    bank, rms = templates()
    return {"version": VERSION, "target_sampling": "existing construct_positive SHA sample, 128 per regime",
        "donor_graphs": 1024, "max_pairs_per_donor_graph": 16, "folds": "5 deterministic recording hashes before matching",
        "matching": "same fold/city/regime, other recording; one donor per target, no pair reuse",
        "calipers": {"min_speed": "0.5 + 0.25*target speed m/s", "max_speed": "1 + 0.25*target speed m/s",
                     "distance": "max(2m, 0.15*target distance)", "closing_speed": "1m/s",
                     "nodes_and_degree": "within factor 2 of (1+count)"},
        "distance": "sum squared differences divided by calipers, deterministic greedy target SHA order",
        "matching_inputs": list(COVARIATES), "matching_uses_model_output": False,
        "reference_labels": "eligible natural negative pairs; no claim about protecting true positives",
        "template_bank": bank, "common_endpoint_rms_m": rms, "controller": False,
        "probe": "fold-trained StandardScaler + RidgeClassifier(alpha=1, solver=lsqr, tol=1e-6)",
        "tasks": ["full_vs_own_clean_all", "full_vs_reversed_all", "full_vs_own_clean_matched",
                  "full_vs_reversed_matched", "full_vs_unmatched_natural", "full_vs_matched_natural"],
        "views": list(VIEWS), "clean_relative_features": False, "template_search": False,
        "victim_loaded": False, "victim_trained": False, "original_validation_access": False,
        "test_access": False, "automatic_continuation": False,
        "limitations": ["probe AUC is not ASR or a training permission gate",
                        "fixed template bank already known; not a blind new template evaluation",
                        "matched RMS does not equal matched derivative spectra/energies",
                        "legacy_ramp is rescaled relational implementation, not exact frozen v6.0",
                        "calipers may yield poor coverage; report exclusions, never relax after outcomes",
                        "normal matching uses clean pre-transform covariates; report post-transform shifts",
                        "local pooled edge inputs omit node features and full graph context",
                        "future-only label identity does not certify physical trajectory continuity"]}


def covariates(graph, src, dst):
    p = np.asarray(graph["observed_positions_filled"])
    v = np.asarray(graph["observed_velocities_filled"])
    speed = sorted(np.linalg.norm(v[[src, dst], -1], axis=1))
    distances = np.linalg.norm(p[src, -5:]-p[dst, -5:], axis=1)
    t = np.arange(5)*.1
    closing = -float(np.dot(t-t.mean(), distances-distances.mean())/np.square(t-t.mean()).sum())
    edges = np.asarray(graph["edge_index"])
    degree = len(pilot.trigger.incident_pairs(edges, src, dst))
    return dict(zip(COVARIATES, [float(speed[0]), float(speed[1]), float(distances[-1]), closing,
                                  float(np.log1p(len(p))), float(np.log1p(degree))]))


def calipers(row):
    return np.asarray([.5+.25*row.min_speed, 1.+.25*row.max_speed,
                       max(2., .15*row.distance), 1., np.log(2.), np.log(2.)])


def match_pairs(targets, donors):
    matched_used, random_used, records = set(), set(), []
    donors = donors.sort_values("pair_key").reset_index(drop=True)
    for row in targets.sort_values("target_priority").itertuples(index=False):
        eligible = donors.fold.eq(row.fold) & donors.city.eq(row.city) & donors.motion_regime.eq(row.motion_regime)
        eligible &= donors.split_group_id.ne(row.split_group_id)
        eligible &= ~donors.pair_key.isin(matched_used)
        ids = np.flatnonzero(eligible.to_numpy())
        differences = (donors.iloc[ids][list(COVARIATES)].to_numpy()-np.asarray([getattr(row,c) for c in COVARIATES]))/calipers(row)
        valid = np.all(np.abs(differences) <= 1., axis=1)
        record = {"scenario_id": row.scenario_id, "matched": False, "eligible_before_calipers": len(ids),
                  "eligible_after_calipers": int(valid.sum()), "matched_pair_key": "", "unmatched_pair_key": ""}
        if valid.any():
            good = ids[valid]
            chosen = int(good[np.argmin(np.square(differences[valid]).sum(axis=1))])
            # Unmatched comparator shares fold and excludes target recording, but not city/speed/calipers.
            random_pool = donors.loc[donors.fold.eq(row.fold) & donors.split_group_id.ne(row.split_group_id)
                                    & ~donors.pair_key.isin(random_used)]
            if random_pool.empty:
                raise ValueError("no unmatched donor; do not change matching coverage silently")
            other = min(random_pool.pair_key, key=lambda key: rank(row.scenario_id+"|"+key))
            key = donors.iloc[chosen].pair_key
            matched_used.add(key); random_used.add(other)
            record.update(matched=True, matched_pair_key=key, unmatched_pair_key=other,
                          normalized_squared_distance=float(np.square(differences[valid]).sum(axis=1).min()))
        records.append(record)
    return pd.DataFrame(records)


def collect_donors(context, rows):
    records, features, seen = [], [], []
    for index, row in enumerate(rows.itertuples(index=False), 1):
        path = pilot.graph_path(context, row, allowed_role="construct")
        with np.load(path, allow_pickle=True) as graph:
            clean, labels = pilot.clean_data(graph, row.scenario_id)
            groups = pilot.eligible_negative_pair_groups(graph, pilot.RiskLabelConfig(), require_strict_label=True,
                perturb_window=10, require_bi_endpoint_contiguous=True, label_bundle=labels)
            single_orientation = sum(len(ids) != 2 for ids in groups)
            groups = [ids for ids in groups if len(ids) == 2]
            edges = np.asarray(graph["edge_index"])
            def key(ids):
                a,b = map(int, edges[:, ids[0]])
                return f"{row.scenario_id}|{min(a,b)}|{max(a,b)}"
            chosen = sorted(groups, key=lambda ids: rank(key(ids)))[:16]
            for ids in chosen:
                a,b = map(int, edges[:, ids[0]])
                cov = covariates(graph,a,b)
                records.append({"pair_key": key(ids), "scenario_id": row.scenario_id, "split_group_id": row.split_group_id,
                    "fold": group_fold(row.split_group_id), "city": row.city, "src": a, "dst": b,
                    "motion_regime": "slow_lt_0p5" if cov["min_speed"] < .5 else "moving_ge_0p5", **cov})
                features.append(clean.edge_attr.numpy()[ids])
            seen.append({"scenario_id": row.scenario_id, "graph_sha256": row.graph_sha256,
                         "eligible_pairs": len(groups), "excluded_single_orientation": single_orientation,
                         "retained_pairs": len(chosen)})
        if index==1 or index%100==0 or index==len(rows):
            print(f"[matched-input] donor graphs {index}/{len(rows)}", flush=True)
    if not records:
        raise ValueError("no eligible donor pairs")
    return pd.DataFrame(records), np.stack(features), pd.DataFrame(seen)


def collect_targets(context, sample, bank):
    indexed = context["train_rows"].set_index("scenario_id", drop=False)
    rows, diagnostics, features = [], [], []
    for i, target in enumerate(sample.to_dict("records"),1):
        row = indexed.loc[target["scenario_id"]]
        with np.load(pilot.graph_path(context,row,allowed_role="construct"),allow_pickle=True) as graph:
            clean, labels = pilot.clean_data(graph,row.scenario_id)
            ids = pilot.checked_target(graph,target,labels)
            a,b = int(target["src"]),int(target["dst"])
            cov = covariates(graph,a,b)
            rows.append({**target, "city": row.city, "fold": group_fold(row.split_group_id),
                         "target_priority": rank(row.scenario_id), **cov})
            all_templates = []
            for name, template in bank.items():
                variants = [clean.edge_attr.numpy()[ids]]
                for variant in CONDITIONS[1:]:
                    value = pilot.trigger.apply_relational_trigger(graph,src=a,dst=b,template=template,controller=False,variant=variant)
                    variants.append(value["edge_attr"][ids])
                    # Dynamic oracle uses only these unchanged arrays; verify against the original bundle.
                    oracle = {"future_positions":graph["future_positions"], "future_valid_mask":graph["future_valid_mask"],
                              "edge_index":graph["edge_index"]}
                    after = pilot.labels_for_graph(oracle,pilot.RiskLabelConfig())
                    np.testing.assert_array_equal(after["edge_label"],labels["edge_label"])
                    updated = {"observed_positions_filled":value["positions"],"observed_velocities_filled":value["velocities"],
                               "edge_index":graph["edge_index"]}
                    new_cov = covariates(updated,a,b)
                    diagnostics.append({"scenario_id":row.scenario_id,"template":name,"variant":variant,
                        **value["audit"], "label_identity_pass":True,
                        "observation_future_step_change_m":pilot.stage1.seam_diagnostic(graph,value),
                        **{"delta_"+c:new_cov[c]-cov[c] for c in COVARIATES}})
                all_templates.append(np.stack(variants))
            features.append(np.stack(all_templates))
        if i==1 or i%32==0 or i==len(sample):print(f"[matched-input] targets {i}/{len(sample)}",flush=True)
    return pd.DataFrame(rows),np.stack(features),diagnostics


def pooled(x):
    return np.concatenate([x.mean(axis=1),np.abs(x[:,0]-x[:,1])],axis=1)


def probe(a,b,folds):
    if len(a)!=len(b) or len(a)!=len(folds):raise ValueError("paired probe shape mismatch")
    x=np.concatenate([a,b]); y=np.r_[np.ones(len(a)),np.zeros(len(a))]; groups=np.r_[folds,folds]
    predictions=np.full(len(y),np.nan); rows=[]
    for fold in range(5):
        held=groups==fold; fit=~held
        if not held.any() or fit.sum()<4:
            continue
        model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=1.,solver="lsqr",tol=1e-6))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always");model.fit(x[fit],y[fit])
        predictions[held]=model.decision_function(x[held])
        rows.append({"fold":fold,"held_pairs":int(held.sum()/2),"auc":float(roc_auc_score(y[held],predictions[held])),
                     "warnings":[str(w.message) for w in caught]})
    complete=bool(len(a)>0 and np.isfinite(predictions).all())
    return {"status":"COMPLETE" if complete else "INSUFFICIENT_FOLDS", "pairs":len(a), "folds":rows,
            "mean_fold_auc":float(np.mean([r["auc"] for r in rows])) if complete else None},predictions


def analyze(targets,donors,target_x,donor_x,matches,bank):
    matched=matches.set_index("scenario_id").loc[targets.scenario_id]
    index={key:i for i,key in enumerate(donors.pair_key)}
    summaries,predictions,balance=[],[],[]
    for regime in pilot.stage1.REGIMES:
        all_ids=np.flatnonzero(targets.motion_regime.eq(regime))
        ids=np.flatnonzero(targets.motion_regime.eq(regime).to_numpy() & matched.matched.to_numpy())
        for comparator,column in (("matched","matched_pair_key"),("unmatched","unmatched_pair_key")):
            did=np.array([index[key] for key in matched.iloc[ids][column]],dtype=int)
            if len(ids):
                assert np.all(targets.iloc[ids].fold.to_numpy()==donors.iloc[did].fold.to_numpy())
                assert np.all(targets.iloc[ids].split_group_id.to_numpy()!=donors.iloc[did].split_group_id.to_numpy())
                for c in COVARIATES:
                    a=targets.iloc[ids][c].to_numpy();b=donors.iloc[did][c].to_numpy()
                    scale=np.sqrt((np.var(a)+np.var(b))/2)
                    balance.append({"regime":regime,"comparator":comparator,"covariate":c,"pairs":len(ids),
                                    "mean_difference":float(np.mean(a-b)),"smd":float(np.mean(a-b)/scale) if scale>1e-12 else None})
        for ti,name in enumerate(bank):
            for task in protocol_tasks():
                selected=all_ids if task.endswith("_all") else ids
                a=target_x[selected,ti,1]
                if "natural" in task:
                    column="unmatched_pair_key" if "unmatched" in task else "matched_pair_key"
                    did=np.array([index[key] for key in matched.iloc[selected][column]],dtype=int)
                    b=donor_x[did]
                else:b=target_x[selected,ti,2 if "reversed" in task else 0]
                for view,sl in VIEWS.items():
                    value,scores=probe(pooled(a[:,:,sl]),pooled(b[:,:,sl]),targets.iloc[selected].fold.to_numpy())
                    summary={"template":name,"regime":regime,"task":task,"view":view,**value};summaries.append(summary)
                    for label,values in ((1,scores[:len(selected)]),(0,scores[len(selected):])):
                        for j,score in zip(selected,values):
                            predictions.append({"template":name,"regime":regime,"task":task,"view":view,
                                "target_scenario":targets.iloc[j].scenario_id,"fold":int(targets.iloc[j].fold),
                                "label":label,"score":float(score) if np.isfinite(score) else None})
            print(f"[matched-input] probes {regime} {name} complete",flush=True)
    return summaries,pd.DataFrame(predictions),pd.DataFrame(balance)


def protocol_tasks():
    return ("full_vs_own_clean_all","full_vs_reversed_all","full_vs_own_clean_matched",
            "full_vs_reversed_matched","full_vs_unmatched_natural","full_vs_matched_natural")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument("--check-only",action="store_true");mode.add_argument("--run",action="store_true")
    args=parser.parse_args(argv)
    context=pilot.preflight()
    sample=diagnosis.sample_targets(context["targets"]);sample=sample.loc[sample.cohort.eq("construct_positive")].reset_index(drop=True)
    rows=decoder.select_clean(context["train_rows"],sample,"construct",1024)
    if not sample.inner_role.eq("construct").all() or len(sample)!=256:raise ValueError("invalid fixed target cohort")
    if args.check_only:
        print(json.dumps({"status":"PREFLIGHT_OK","target_scenes":len(sample),"donor_graphs":len(rows),"protocol":protocol()},indent=2));return 0
    directory=pilot.new_directory(prefix="matched_input_");print(f"[matched-input] output: {directory}",flush=True)
    source=pilot.bind(__file__)
    helper_sources={name:pilot.bind(module.__file__) for name,module in (("diagnosis",diagnosis),("decoder",decoder),("trigger",pilot.trigger))}
    pilot.write_json(directory/"protocol.json",{"protocol":protocol(),"source":source,"helpers":helper_sources,"assets":context["assets"]})
    bank,_=templates()
    donors,donor_x,donor_log=collect_donors(context,rows)
    targets,target_x,diagnostics=collect_targets(context,sample,bank)
    matches=match_pairs(targets,donors)
    print(f"[matched-input] coverage {int(matches.matched.sum())}/{len(matches)}; calipers unchanged",flush=True)
    summaries,predictions,balance=analyze(targets,donors,target_x,donor_x,matches,bank)
    frames={"targets.csv":targets,"donors.csv":donors,"donor_graphs.csv":donor_log,"matches.csv":matches,
            "probe_predictions.csv":predictions,"balance.csv":balance}
    for name,frame in frames.items():
        with (directory/name).open("x") as stream:frame.to_csv(stream,index=False)
    with (directory/"features.npz").open("xb") as stream:np.savez_compressed(stream,target_features=target_x,donor_features=donor_x)
    with (directory/"physical.jsonl").open("x") as stream:
        for record in diagnostics:stream.write(json.dumps(record,allow_nan=False)+"\n")
    coverage=targets[["scenario_id","motion_regime"]].merge(matches,on="scenario_id").groupby("motion_regime").matched.agg(["count","sum"]).reset_index().to_dict("records")
    pilot.bind(__file__,source["sha256"]);pilot.preflight()
    pilot.write_json(directory/"summary.json",{"status":"MATCHED_INPUT_AUDIT_COMPLETE_REVIEW_REQUIRED","protocol":protocol(),
        "source":source,"helpers":helper_sources,"coverage":coverage,"donor_pairs":len(donors),"probes":summaries,
        "physical_records":len(diagnostics),"evidence":{p.name:pilot.bind(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "automatic_continuation":False})
    print(f"[matched-input] complete: {directory/'summary.json'}; no victim training",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
