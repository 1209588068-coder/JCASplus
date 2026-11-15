"""Corrected five-seed full-TRAIN AV2 runner."""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, json, math, os, shutil, statistics, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[20261111,20261112,20261113,20261114,20261115]
STAGES=["base","residual","support_surrogate","continuation_a","continuation_b"]
DATA_CACHE=Path("/home/mizhou/JCAS/results/av2_formal_main_10seed_20261106_v1/data")
GRAPH_ROOT=Path("/home/mizhou/JCAS/graphs/av2mf_graph_v4")

def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+f".{os.getpid()}.tmp"); t.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n"); t.replace(path)

def _save_rng_snapshot(folder, stage, boundary):
    """Retain an exact stage-boundary RNG state without advancing any stream."""
    import random
    import numpy as np
    import torch
    destination=Path(folder)/"rng_states"; destination.mkdir(parents=True,exist_ok=True)
    path=destination/f"{stage}_{boundary}.pt"
    numpy_state=np.random.get_state()
    record=dict(stage=stage,boundary=boundary,at=now(),
        python=random.getstate(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy=dict(bit_generator=numpy_state[0],state=torch.from_numpy(numpy_state[1].copy()),
            position=numpy_state[2],has_gauss=numpy_state[3],cached_gaussian=numpy_state[4]))
    with path.open("xb") as stream: torch.save(record,stream)
    return sha(path)

def _link_cache(data):
    data.mkdir(parents=True,exist_ok=False)
    for p in DATA_CACHE.resolve().iterdir():
        target=data/p.name
        if p.name == "summary.json":
            shutil.copyfile(p,target)
        else:
            target.symlink_to(p,target_is_directory=p.is_dir())
def _summary(data, experiment_identity):
    s=read(DATA_CACHE/"summary.json")
    s.update(schema="av2_formal_main_full_corrected_data_summary_v1",
      experiment_identity=experiment_identity,
      source_graph_root=str(GRAPH_ROOT), source_cache=str(DATA_CACHE), cache_reused_read_only=True,
      normalization_contract=dict(fit_role="train",fit_scope="all 5,193,841 registered TRAIN pair nodes",
        statistics="population mean/std across TRAIN; minimum scale 1e-6",clip=[-10,10],
        dev_audit_reuse_train_statistics=True),
      corrected_protocol=dict(recording_group_exclusion="all DEV/AUDIT groups excluded from TRAIN",
        official_validation_loaded=False,official_test_loaded=False,
        clean_supervision="20 percent of 2,188,291 eligible TRAIN pairs per seed",
        clean_supervision_count=437658,attachment_count=5865,victim_supervision_count=443523))
    write(data/"summary.json",s)
def _snapshot(output):
    snap=output/"source_snapshot"; shutil.copytree("/home/mizhou/JCAS/spear_current",snap/"spear_current",
      ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    # ``womd_pilot_selection`` loads the retained author SPEAR sources
    # relative to the frozen package root (``../third_party``).  The first
    # corrected registration copied only ``spear_current`` and therefore
    # failed all seeds at fresh selection with a missing ``models/MLP.py``.
    # Keep the retry snapshot self-contained and bind the copied author files
    # through the normal source hash map below.
    author_source = Path("/home/mizhou/JCAS_vehicle_lane/imports/JCAS_20261107_162453/project/third_party/author_baselines/SPEAR")
    if not author_source.is_dir():
        raise FileNotFoundError(f"author SPEAR source missing: {author_source}")
    shutil.copytree(author_source, snap/"third_party/author_baselines/SPEAR",
      ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    gpu_cache_source = ROOT / "src" / "av2_gpu_cache.py"
    if not gpu_cache_source.is_file():
        raise FileNotFoundError(f"GPU cache source missing: {gpu_cache_source}")
    shutil.copyfile(gpu_cache_source, snap/"spear_current/av2_gpu_cache.py")
    shutil.copyfile(Path(__file__),snap/"spear_current/av2_corrected.py")
    for p in sorted(Path("/home/mizhou/JCAS/tests_current").glob("test_av2*.py")):
        d=snap/"tests_current"/p.name; d.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,d)
    return snap

def register(output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Corrected output already exists")
    if not (DATA_CACHE/"summary.json").exists(): raise FileNotFoundError("Full cache missing")
    output.mkdir(parents=True); data=output/"data"; _link_cache(data); _summary(data, output.name); snap=_snapshot(output)
    shutil.copyfile(data/"summary.json",output/"data_summary.json")
    s=read(data/"summary.json")
    required={"scenes":115490,"groups":10176,"pair_nodes":5193841,"pair_edges":91750982,"eligible":2188291}
    if any(s["train"][k]!=v for k,v in required.items()) or s["dev"]["scenes"]!=128 or s["audit"]["scenes"]!=1280 or s["dev"]["eligible"]!=2223 or s["audit"]["eligible"]!=24064:
        raise ValueError("grouped-v5 counts differ")
    sources={str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob("*.py"))}
    retry_of = ("av2_formal_main_full_corrected_v1_retry1_20261109"
                if output.name != "av2_formal_main_full_corrected_v1" else None)
    gpu_cache = "gpu_cache" in output.name
    gpu_cache_contract = dict(enabled=gpu_cache, mode="resident_numeric_scene_batches" if gpu_cache else "cpu_scene_batches",
      roles=["train","dev"] if gpu_cache else [], cache_device="cuda:0" if gpu_cache else None,
      pin_memory=False, non_blocking=False if gpu_cache else None,
      preserves_scene_order=True, preserves_max_nodes=True, preserves_topology=True,
      expected_numeric_cache_budget_bytes=24*1024**3 if gpu_cache else 0,
      required_free_headroom_bytes=32*1024**3 if gpu_cache else 0)
    plan=dict(schema="av2_formal_main_full_corrected_v1",experiment_identity=output.name,
      retry_of=retry_of,registered_at=now(),output_dir=str(output),
      recovery_reason="retry1 seed20261111 failed at continuation_a_b: CPU/CUDA index mismatch; five seeds restart from scratch",
      data_dir=str(data),source_snapshot=str(snap),source_sha256=sources,
      graph_root=str(GRAPH_ROOT),graph_manifest_sha256=sha(GRAPH_ROOT/"manifest_grouped_v5.csv"),
      data_summary_path=str(output/"data_summary.json"),data_summary_sha256=sha(data/"summary.json"),
      data_summary_path_sha256=sha(output/"data_summary.json"),data_summary=s,seeds=SEEDS,old_v2_counted=False,
      device="cuda:0",dtype="float32",amp=False,tf32=False,max_nodes=16384,
      minimum_cuda_free_bytes=(32*1024**3 if gpu_cache else 8*1024**3),gpu_batch_cache=gpu_cache_contract,
      attachment_count=5865,clean_pairs=437658,victim_pairs=443523,
      stages=STAGES,stage_epochs={x:200 for x in STAGES},generator_updates=800,shadow_updates=1000,support_selector_updates=200,
      source_selection=dict(attachment_count=5865,outer_extra_count=512,model_epochs=200,
        sage_n_permutations=2048,sage_batch_size=64,explanation_count=128,background_count=32),
      training=dict(clean_proxy_epochs=200,clean_reference_epochs=200,victim_epochs=200,inner_steps=1),
      support_policy="fresh CandidateHardConcrete semantic support selector 200 epochs on fresh top30 groups; exact legal24 hardening; no ASR/AUDIT selection",
      checkpoint_policy=dict(generator="final epoch each stage",victim="earliest strict maximum clean DEV AUC",audit_used=False),
      data_contract=dict(feature_dim=784,original_topology=True,original_labels=True,train_only_normalizer=True,
        normalizer_clip=[-10,10],train_groups=10176,dev_groups=128,audit_groups=1280,
        group_overlap=dict(train_dev=0,train_audit=0,dev_audit=0),official_test_loaded=False),
      execution=dict(cuda_only=True,fp32=True,amp=False,tf32=False,whole_scene_batches=True,
        neighbor_sampling=False,scene_split=False,automatic_retries=0,replace_seed=False,stop_on_asr=False),
      evaluation=dict(dev_scenes=128,audit_scenes=1280,dev_targets=112,audit_targets=1113,blind=False,official_test_loaded=False),
      scope="AV2 grouped-v5 full TRAIN context; five fresh complete method seeds; old v2 excluded")
    write(output/"plan.json",plan); ph=sha(output/"plan.json"); children={}
    for seed in SEEDS:
        folder=output/f"seed{seed}"; folder.mkdir()
        child=dict(schema=plan["schema"]+"_seed",seed=seed,output_dir=str(folder),
          parent_plan=str(output/"plan.json"),parent_plan_sha256=ph,data_dir=str(data),
          data_summary_sha256=plan["data_summary_sha256"],source_snapshot=str(snap),source_sha256=sources,
          device="cuda:0",dtype="float32",amp=False,tf32=False,max_nodes=16384,stages=STAGES,
          stage_epochs=plan["stage_epochs"],attachment_count=5865,clean_pairs=437658,victim_pairs=443523)
        write(folder/"plan.json",child); children[str(seed)]=sha(folder/"plan.json")
    write(output/"registration.json",dict(plan_sha256=ph,child_plan_sha256=children,seeds=SEEDS,
        experiment_identity=output.name,retry_of=retry_of,at=now()))
    write(output/"status.json",dict(status="REGISTERED",at=now(),completed=0,failed=0,total=5,training_started=False))
    return output/"plan.json"

def verify(plan_path,frozen=False):
    plan=read(plan_path); root=Path(plan["output_dir"]); reg=read(root/"registration.json")
    if sha(plan_path)!=reg["plan_sha256"] or plan["seeds"]!=SEEDS: raise ValueError("plan/seed binding changed")
    if sha(root/"data/summary.json")!=plan["data_summary_sha256"]: raise ValueError("data summary changed")
    if sha(plan["data_summary_path"])!=plan["data_summary_path_sha256"]: raise ValueError("independent data_summary changed")
    data=Path(plan["data_dir"]); summary=plan["data_summary"]
    for name,digest in summary.get("asset_sha256",{}).items():
        if sha(data/name)!=digest: raise ValueError("registered data asset changed "+name)
    for role in ("train","dev","audit"):
        for shard in summary[role].get("shards",[]):
            if sha(data/role/shard["path"])!=shard["sha256"]: raise ValueError("registered shard changed "+role+"/"+shard["path"])
    if sha(Path(plan["graph_root"])/"manifest_grouped_v5.csv")!=plan["graph_manifest_sha256"]: raise ValueError("graph manifest changed")
    for rel,digest in plan["source_sha256"].items():
        if sha(Path(plan["source_snapshot"])/rel)!=digest: raise ValueError("source changed "+rel)
    if frozen and Path(__file__).resolve()!=Path(plan["source_snapshot"])/"spear_current/av2_corrected.py": raise RuntimeError("must run frozen source")
    for seed in SEEDS:
        f=root/f"seed{seed}"; child=read(f/"plan.json")
        if sha(f/"plan.json")!=reg["child_plan_sha256"][str(seed)] or child["parent_plan_sha256"]!=sha(plan_path) or child["seed"]!=seed: raise ValueError("child binding changed")
    return plan

def preflight(plan_path):
    plan=verify(plan_path,True)
    import torch
    cache = _enable_gpu_batch_cache(plan)
    from .av2_formal_classifiers import batches
    from .models import GraphSage
    from .womd_execution import configure_message_passing
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    while torch.cuda.mem_get_info("cuda:0")[0] < plan["minimum_cuda_free_bytes"]:
        time.sleep(30)
    torch.set_num_threads(1); largest=None; edge_n=-1
    cpu_scan_started = time.perf_counter()
    for b in batches(plan["data_dir"],"train",16384,"cpu"):
        if int(b["edge_index"].shape[1])>edge_n: largest=b; edge_n=int(b["edge_index"].shape[1])
    cpu_scan_seconds = time.perf_counter() - cpu_scan_started
    d=torch.device("cuda:0")
    torch.cuda.synchronize(d); h2d_started = time.perf_counter()
    x=largest["x"].to(d); e=largest["edge_index"].to(d)
    torch.cuda.synchronize(d); h2d_seconds = time.perf_counter() - h2d_started
    m=GraphSage(784,80,2,dropout=.5,lr=.01,weight_decay=.0005,device=d).to(d); configure_message_passing(m); m.train(); m.zero_grad(set_to_none=True)
    before=[p.detach().clone() for p in m.parameters()]
    torch.cuda.synchronize(d); forward_started = time.perf_counter()
    out=m(x,e)
    torch.cuda.synchronize(d); forward_seconds = time.perf_counter() - forward_started
    torch.cuda.synchronize(d); backward_started = time.perf_counter()
    out.float().sum().backward()
    torch.cuda.synchronize(d); backward_seconds = time.perf_counter() - backward_started
    finite=bool(torch.isfinite(out).all()) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    unchanged=all(torch.equal(a,b) for a,b in zip(before,m.parameters()))
    probe=dict(status="PASS" if finite and unchanged else "FAIL",optimizer_steps=0,nodes=len(x),edges=edge_n,
      finite_gradients=finite,parameters_unchanged=unchanged,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
      peak_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_batch_read_scene_assembly_seconds=cpu_scan_seconds,
      h2d_seconds=h2d_seconds,forward_seconds=forward_seconds,backward_seconds=backward_seconds,
      device="cuda:0",dtype="float32",amp=False,tf32=False,official_test_loaded=False)
    if cache is not None:
        from itertools import zip_longest
        from .av2_formal_data import iter_role_batches
        gpu_cache_probe = {}
        for role in plan["gpu_batch_cache"]["roles"]:
            free_before, _ = torch.cuda.mem_get_info(d)
            started = time.perf_counter()
            summary = cache.cache_summary(plan["data_dir"],role,16384,d)
            torch.cuda.synchronize(d)
            build_seconds = time.perf_counter()-started
            expected = plan["data_summary"][role]
            if summary["nodes"] != expected["pair_nodes"] or summary["edges"] != expected["pair_edges"]:
                raise ValueError(f"GPU cache {role} population mismatch")
            if summary["global_start"] != 0 or summary["global_stop"] != expected["pair_nodes"]:
                raise ValueError(f"GPU cache {role} offset mismatch")
            compare_started = time.perf_counter()
            compared = 0
            for source, resident in zip_longest(
                iter_role_batches(plan["data_dir"],role,max_nodes=16384,device="cpu"),
                cache.batches(plan["data_dir"],role,16384,d)):
                if source is None or resident is None:
                    raise ValueError(f"GPU cache {role} batch count mismatch")
                if any(source[k] != resident[k] for k in ("global_start","global_stop")):
                    raise ValueError(f"GPU cache {role} batch offset mismatch at {compared}")
                if any(not torch.equal(source[k],resident[k].cpu())
                       for k in ("x","edge_index","y","eligible")):
                    raise ValueError(f"GPU cache {role} tensor mismatch at batch {compared}")
                compared += 1
            torch.cuda.synchronize(d)
            free_after, _ = torch.cuda.mem_get_info(d)
            gpu_cache_probe[role] = dict(**summary,build_seconds=build_seconds,
                exhaustive_compare_seconds=time.perf_counter()-compare_started,
                batches_compared=compared,free_before_bytes=free_before,
                free_after_bytes=free_after,bitwise_equal=True)
        if torch.cuda.mem_get_info(d)[0] < 8*1024**3:
            raise RuntimeError("Insufficient GPU headroom after full cache materialization")
        probe["gpu_cache"] = gpu_cache_probe
        probe["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(d)
        probe["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(d)
    outdir=Path(plan["output_dir"]); write(outdir/"preflight/gpu_probe.json",probe)
    write(outdir/"preflight/acceptance.json",dict(status=probe["status"],at=now(),plan_sha256=sha(plan_path),source_sha256=plan["source_sha256"],probe_sha256=sha(outdir/"preflight/gpu_probe.json"),optimizer_steps=0))
    if probe["status"]!="PASS": raise RuntimeError("GPU preflight failed")
    return probe

def _support(selection_dir,coords):
    import numpy as np
    from .womd_pilot_selection import candidate_groups,exact_support
    vals=np.load(Path(selection_dir)/"attribution.npz")["values"]; groups,meta=candidate_groups(coords)
    scores=[float(np.mean([abs(float(vals[c])) for c in g])) for g in groups]
    mask=exact_support(coords,groups,scores,24)
    return mask,dict(coordinates=coords.tolist(),groups=groups,group_scores_abs_sage=scores,support=mask.tolist(),policy="fresh SAGE group scores exact24")

def _enable_gpu_batch_cache(plan):
    """Opt into a frozen-run GPU numeric cache without changing the default path."""
    contract = plan.get("gpu_batch_cache", {})
    if not contract.get("enabled", False):
        return None
    import torch
    from . import av2_formal_classifiers as classifiers
    from . import av2_gpu_cache as cache
    original = classifiers.batches
    def cached_batches(data_dir, role, max_nodes, device):
        if torch.device(device).type != "cuda":
            return original(data_dir, role, max_nodes, device)
        return cache.batches(data_dir, role, max_nodes, device,
                             pin_memory=bool(contract.get("pin_memory", False)))
    classifiers.batches = cached_batches
    return cache

def _augment_eval_summary(summary, evaluation_dir):
    """Add normal-task calibration and per-target perturbation summaries.

    The shared evaluator already writes every normal pair to
    ``clean_pair_scores.csv``.  Computing these two fields from that retained
    evidence keeps the evaluator frozen while making the formal report
    contract explicit.
    """
    path = Path(evaluation_dir) / "clean_pair_scores.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    for model in ("victim", "clean_reference"):
        for group in ("all", "slow", "moving"):
            selected = [r for r in rows if group == "all" or
                        ("moving" if r.get("motion_regime") == "moving" else "slow") == group]
            truth = [int(r["label"]) for r in selected]
            probs = [float(r[model + "_p"]) for r in selected]
            preds = [int(r[model + "_pred"]) for r in selected]
            negatives = sum(y == 0 for y in truth)
            false_positive = sum(y == 0 and p == 1 for y, p in zip(truth, preds))
            normal = summary["metrics"][model]["normal"][group]
            normal["clean_negative_fpr"] = (false_positive / negatives) if negatives else None
            normal["clean_pair_brier"] = (
                sum((p - y) ** 2 for p, y in zip(probs, truth)) / len(truth)
                if truth else None)
    target_path=Path(evaluation_dir)/"targets.csv"
    targets=list(csv.DictReader(target_path.open(newline="")))
    if len(targets)!=summary["targets"]:
        raise ValueError("Target CSV coverage differs from evaluation summary")
    fields=("changed_coordinates","delta_l1","delta_l2","delta_linf")
    modification={"targets":len(targets)}
    for name in fields:
        values=sorted(float(row[name]) for row in targets)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite target modification: "+name)
        if values:
            rank=.95*(len(values)-1); lower=int(math.floor(rank)); upper=int(math.ceil(rank))
            p95=values[lower]+(values[upper]-values[lower])*(rank-lower)
            modification[name]=dict(mean=statistics.mean(values),max=values[-1],p95=p95)
        else:
            modification[name]=dict(mean=None,max=None,p95=None)
    summary["metrics"]["victim"]["modification"]=modification
    Path(evaluation_dir, "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary

def _continuations(generator,surrogate,data_dir,ids,proxies,coordinates,support,seed,output_dir,progress,support_dir=None):
    """Continue the same anchor generator through support and two residual stages."""
    import torch
    from copy import deepcopy
    from types import SimpleNamespace
    from torch.nn import functional as F
    from .av2_formal_generator import (_population,_positions,_changed,_cached_probabilities,
        _finite_loss,_finite_gradients,_finite_module,outer_loss_terms,_proxy_contract)
    from .full_ablation_training import _cpu_state,_state_hash,_tensor_hash,_freeze
    population=_population(data_dir,ids); coordinates=torch.as_tensor(coordinates,device=generator.base_shadow.device)
    support=torch.as_tensor(support,device=coordinates.device,dtype=torch.bool)
    generator.set_support(support); generator.set_phase('support_surrogate'); _freeze(generator)
    owned=deepcopy(surrogate)
    if support_dir is not None:
        saved=torch.load(Path(support_dir)/"selector_surrogate.pt",map_location="cpu",weights_only=True)
        owned.load_state_dict(saved,strict=True)
    owned.requires_grad_(True)
    folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=False)
    # Global supervision positions stay on CPU, as in fit_generator and
    # fit_support.  local_indices uses CPU searchsorted, while _positions
    # transfers only the batch-local positions to the feature device.
    attach=population['attach']; outer=population['outer']
    inner_all=population['inner_sorted']; outer_all=population['outer_sorted']
    labels_full=None; records=[]
    with (folder/'epochs.jsonl').open('x') as log:
        # fit_support already is the registered 200-epoch support-surrogate
        # stage (shadow + selector). Continue that same surrogate through the
        # two residual continuation stages here.
        for stage in ('continuation_a','continuation_b'):
            generator.set_phase(stage)
            rng_start_sha=_save_rng_snapshot(folder,stage,'start')
            active=[p for p in generator.parameters() if p.requires_grad]
            sopt=torch.optim.Adam(owned.parameters(),lr=.0002,weight_decay=.0005)
            gopt=torch.optim.Adam(active,lr=.0002,weight_decay=.0005) if active else None
            qcache=_cached_probabilities(data_dir,population,proxies,max_nodes=16384,device=generator.base_shadow.device)
            for epoch in range(200):
                generator.train(); owned.train(); sopt.zero_grad(set_to_none=True); inner_loss=0.; seen=0
                for batch in __import__('spear_current.av2_formal_classifiers',fromlist=['batches']).batches(data_dir,'train',16384,generator.base_shadow.device):
                    attach_local,_=_positions(attach,batch); inner_local=__import__('spear_current.av2_formal_classifiers',fromlist=['local_indices']).local_indices(inner_all,batch)
                    if len(inner_local):
                        with torch.no_grad():
                            changed=(generator.inject(batch['x'],SimpleNamespace(edge_index=batch['edge_index']),attach_local).features
                                     if len(attach_local) else batch['x'])
                        y=batch['y'].clone(); y[attach_local]=1
                        loss=F.nll_loss(owned(changed,batch['edge_index'])[inner_local],y[inner_local],reduction='sum')/len(inner_all)
                        _finite_loss(loss,'continuation inner'); loss.backward(); inner_loss+=float(loss.detach()); seen+=len(inner_local)
                if seen!=len(inner_all): raise ValueError('Continuation inner coverage mismatch')
                _finite_gradients(owned.parameters(),'continuation surrogate'); sopt.step(); _finite_module(owned,'continuation surrogate'); sopt.zero_grad(set_to_none=True)
                if gopt is not None:
                    owned.requires_grad_(False).eval(); gopt.zero_grad(set_to_none=True); totals=[0.,0.,0.]; seen_o=seen_s=seen_n=0
                    for batch in __import__('spear_current.av2_formal_classifiers',fromlist=['batches']).batches(data_dir,'train',16384,generator.base_shadow.device):
                        x,edge=batch['x'],batch['edge_index']; outer_local,pos=_positions(outer,batch); sup_local=__import__('spear_current.av2_formal_classifiers',fromlist=['local_indices']).local_indices(outer_all,batch); y=batch['y'].clone(); y[outer_local]=1
                        if len(outer_local):
                            changed=generator.inject(x,SimpleNamespace(edge_index=edge),outer_local,detach=False).features
                            terms=outer_loss_terms(owned,proxies,x,changed,edge,y,sup_local,outer_local,global_supervised=len(outer_all),global_nodes=population['nodes'],global_outer=len(outer),clean_probabilities=[q[pos].to(x.device) for q in qcache])
                            loss=terms[0]+.1*terms[1]+terms[2]; _finite_loss(loss,'continuation outer'); loss.backward()
                        else:
                            with torch.no_grad():
                                terms=outer_loss_terms(owned,proxies,x,x,edge,y,sup_local,outer_local,global_supervised=len(outer_all),global_nodes=population['nodes'],global_outer=len(outer))
                        totals=[a+float(v.detach()) for a,v in zip(totals,terms)]; seen_o+=len(outer_local); seen_s+=len(sup_local); seen_n+=len(x)
                    if seen_o!=len(outer) or seen_s!=len(outer_all) or seen_n!=population['nodes']: raise ValueError('Continuation outer coverage mismatch')
                    _finite_gradients(active,'continuation generator'); gopt.step(); _finite_module(generator,'continuation generator'); owned.requires_grad_(True)
                row=dict(stage=stage,epoch=epoch,inner_nll=inner_loss,outer_terms=totals if gopt is not None else None,surrogate_updates=epoch+1,generator_updates=epoch+1 if gopt is not None else 0)
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                if progress: progress(row)
            rng_end_sha=_save_rng_snapshot(folder,stage,'end')
            torch.save(dict(generator=_cpu_state(generator),surrogate=_cpu_state(owned),metadata=dict(stage=stage,epochs=200,seed=seed,generator_sha256=_state_hash(generator),surrogate_sha256=_state_hash(owned))),folder/(stage+'_final.pt'))
            records.append(dict(stage=stage,epochs=200,generator_sha256=_state_hash(generator),surrogate_sha256=_state_hash(owned),rng_start_sha256=rng_start_sha,rng_end_sha256=rng_end_sha))
    _freeze(generator); _freeze(owned); write(folder/'stages.json',records)
    return dict(generator=generator,surrogate=owned,stages=records)

def worker(plan_path,seed):
    plan=verify(plan_path,True)
    if seed not in SEEDS: raise ValueError("unregistered seed")
    import torch
    from types import SimpleNamespace
    _enable_gpu_batch_cache(plan)
    from .av2_formal_data import make_indices,load_role,load_role_arrays
    from .av2_formal_selection import fit_source_selection
    from .av2_formal_classifiers import fit_classifier,fit_victim,evaluate
    from .av2_formal_generator import fit_generator,fit_support
    from .full_ablation_training import _tensor_hash,_state_hash
    root=Path(plan["output_dir"]); folder=root/f"seed{seed}"; folder.mkdir(exist_ok=True)
    if any((folder/x).exists() for x in ("completion.json","failure.json","selection","training")): raise FileExistsError("partial seed exists")
    start=time.monotonic(); active=["selection"]
    def status(s,**extra): write(folder/"status.json",dict(status=s,seed=seed,pid=os.getpid(),at=now(),elapsed_seconds=time.monotonic()-start,active_stage=active[0],**extra))
    def progress(row): status("RUNNING",progress=row); print(json.dumps(dict(seed=seed,**row),allow_nan=False),flush=True)
    rng_hashes={}
    def record_rng(stage,boundary):
        digest=_save_rng_snapshot(folder,stage,boundary)
        rng_hashes[f"rng_states/{stage}_{boundary}.pt"]=digest
        write(folder/"rng_states/manifest.json",rng_hashes)
    try:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        while True:
            free,_=torch.cuda.mem_get_info("cuda:0")
            if free>=plan["minimum_cuda_free_bytes"]: break
            status("WAITING_FOR_GPU",free_bytes=free); time.sleep(30)
        record_rng("initial_indices","start")
        idx=make_indices(plan["data_dir"],seed); pop=load_role_arrays(plan["data_dir"],"train"); idx["train_labels"]=pop["y"].clone(); torch.save(idx,folder/"initial_indices.pt")
        record_rng("initial_indices","end")
        active[0]="fresh_selection"
        record_rng("fresh_selection","start")
        selected=fit_source_selection(plan["data_dir"],seed=seed,output_dir=folder/"selection",indices=idx,
          attachment_count=5865,outer_extra_count=512,model_epochs=200,sage_n_permutations=2048,sage_batch_size=64,
          explanation_count=128,background_count=32,max_nodes=16384,device="cuda:0",progress=progress)
        record_rng("fresh_selection","end")
        coords=selected["coordinates"].detach().cpu().long()
        ids=selected["indices"]
        proxies=[]
        for role,off in (("clean_proxy_1",101000),("clean_proxy_2",102000)):
            active[0]=role; record_rng(role,"start")
            fitted=fit_classifier(plan["data_dir"],ids["idx_train"],seed=seed+off,output_dir=folder/role,role=role,epochs=200,max_nodes=16384,device="cuda:0",checkpoint_rule="dev_auc",progress=progress)
            proxies.append(fitted["model"].eval().requires_grad_(False)); record_rng(role,"end")
        active[0]="fresh_anchor_and_support"
        record_rng("anchor_base_residual","start")
        anchor=fit_generator(plan["data_dir"],ids,proxies,coords,seed=seed+201000,output_dir=folder/"anchor",
            base_epochs=200,residual_epochs=200,max_nodes=16384,device="cuda:0",progress=progress)
        record_rng("anchor_base_residual","end")
        record_rng("semantic_support","start")
        support,srecord=fit_support(anchor["generator"],anchor["surrogate"],plan["data_dir"],ids,proxies,coords,
            seed=seed+301000,output_dir=folder/"semantic_support",epochs=200,max_nodes=16384,device="cuda:0",progress=progress)
        record_rng("semantic_support","end")
        write(folder/"support.json",srecord)
        active[0]="continuation_a_b"; record_rng("continuation_a_b","start")
        continuation=_continuations(anchor["generator"],anchor["surrogate"],plan["data_dir"],ids,proxies,coords,support,seed,folder/"continuation",progress,folder/"semantic_support")
        record_rng("continuation_a_b","end")
        gen=continuation["generator"].eval().requires_grad_(False)
        active[0]="victim"; record_rng("victim","start")
        victim=fit_victim(gen,plan["data_dir"],ids,seed=seed+401000,output_dir=folder/"victim",epochs=200,max_nodes=16384,device="cuda:0",progress=progress)
        record_rng("victim","end")
        active[0]="clean_reference"; record_rng("clean_reference","start")
        ref=fit_classifier(plan["data_dir"],ids["idx_train"],seed=seed+103000,output_dir=folder/"clean_reference",role="clean_reference",epochs=200,max_nodes=16384,device="cuda:0",checkpoint_rule="dev_auc",progress=progress)["model"]
        record_rng("clean_reference","end")
        results={}
        for role in ("dev","audit"):
            active[0]="evaluation_"+role; record_rng(active[0],"start")
            results[role]=evaluate(gen,victim["model"],ref,plan["data_dir"],output_dir=folder/f"evaluation_{role}",role=role,device="cuda:0",progress=progress)
            results[role] = _augment_eval_summary(results[role], folder/f"evaluation_{role}")
            record_rng(active[0],"end")
        rng_hashes.update({str(path.relative_to(folder)):sha(path) for path in (folder/"continuation/rng_states").glob("*.pt")})
        receipt=dict(status="SEED_COMPLETE",seed=seed,at=now(),plan_sha256=sha(plan_path),elapsed_seconds=time.monotonic()-start,
          stages={s:200 for s in STAGES},generator_updates=800,shadow_updates=1000,support_selector_updates=200,clean_indices_sha256=_tensor_hash(ids["idx_train"]),support_sha256=_tensor_hash(support),
          generator_sha256=_state_hash(gen),victim_sha256=_state_hash(victim["model"]),clean_reference_sha256=_state_hash(ref),
          rng_state_sha256=rng_hashes,rng_manifest_sha256=sha(folder/"rng_states/manifest.json"),
          result_sha256={r:sha(folder/f"evaluation_{r}/summary.json") for r in results},official_test_loaded=False)
        write(folder/"completion.json",receipt); status("COMPLETE")
    except BaseException as exc:
        failure=dict(status="FAILED",seed=seed,at=now(),stage=active[0],type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()); write(folder/"failure.json",failure); status("FAILED",failure=failure); raise

def aggregate(plan_path):
    plan=verify(plan_path); root=Path(plan["output_dir"]); complete=[]; failed=[]; pending=[]; records={}
    for seed in SEEDS:
        f=root/f"seed{seed}"
        if (f/"failure.json").exists(): failed.append(dict(seed=seed,failure=read(f/"failure.json"))); continue
        if not (f/"completion.json").exists(): pending.append(seed); continue
        try:
            rec=read(f/"completion.json")
            if rec.get("status")!="SEED_COMPLETE" or rec.get("plan_sha256")!=sha(Path(plan_path)): raise ValueError("bad receipt")
            records[seed]={r:read(f/f"evaluation_{r}/summary.json")["metrics"] for r in ("dev","audit")}
            for role in ("dev","audit"):
                if sha(f/f"evaluation_{role}/summary.json")!=rec["result_sha256"][role]: raise ValueError("evaluation hash")
            complete.append(seed)
        except (OSError,KeyError,TypeError,ValueError) as exc: failed.append(dict(seed=seed,validation_error=str(exc)))
    metrics={}
    def walk(obj,prefix,seed):
        if isinstance(obj,dict):
            for k,v in obj.items(): walk(v,prefix+"."+k if prefix else k,seed)
        elif type(obj) in (int,float) and math.isfinite(obj): metrics.setdefault(prefix,[]).append(dict(seed=seed,value=obj))
    for seed,vals in records.items():
        for role,rec in vals.items(): walk(rec,role,seed)
    def reduce(v):
        values = [x["value"] for x in v]
        n = len(values)
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if n > 1 else None
        # Five registered seeds use the two-sided t critical value for df=4.
        # Keep the interval explicit so a report cannot mistake a sample SD
        # for uncertainty around the mean.
        t_critical = 2.7764451051977987 if n == 5 else None
        half = t_critical * sd / math.sqrt(n) if t_critical is not None and sd is not None else None
        return dict(per_seed=v,n=n,mean=mean,sample_sd=sd,
                    ci95_low=(mean-half if half is not None else None),
                    ci95_high=(mean+half if half is not None else None),
                    ci95_method=("t(df=4), two-sided" if t_critical is not None else None))
    out=dict(status="ALL_FIVE_COMPLETE" if len(complete)==5 else "PARTIAL",at=now(),expected_seeds=SEEDS,completed_seeds=complete,failed_seeds=failed,pending_seeds=pending,completed=len(complete),failed=len(failed),total=5,metrics={k:reduce(v) for k,v in metrics.items()},sample_sd_ddof=1,failure_is_zero_asr=False,blind_evaluation=False,independent_unit="fresh full method seed; repeated evaluation pools are not new independent targets")
    write(root/"aggregate.json",out); return out

def supervise(plan_path):
    plan=verify(plan_path,True); root=Path(plan["output_dir"])
    with (root/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in SEEDS:
            prior=aggregate(plan_path); folder=root/f"seed{seed}"
            cmd=[sys.executable,"-B","-m","spear_current.av2_corrected","--worker",str(plan_path),"--seed",str(seed)]
            with (folder/"worker.log").open("x") as log:
                child=subprocess.Popen(cmd,cwd=plan["source_snapshot"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    cur=read(folder/"status.json") if (folder/"status.json").exists() else {}
                    write(root/"status.json",dict(status="RUNNING",at=now(),active_seed=seed,worker_pid=child.pid,completed=prior["completed"],failed=prior["failed"],total=5,worker_status=cur)); time.sleep(15)
            if child.returncode!=0 and not (folder/"failure.json").exists(): write(folder/"failure.json",dict(status="FAILED",seed=seed,worker_returncode=child.returncode,at=now()))
            write(folder/"supervisor_exit.json",dict(at=now(),returncode=child.returncode)); aggregate(plan_path)
        result=aggregate(plan_path); state="COMPLETE" if result["completed"]==5 else "FINISHED_WITH_FAILURES"; write(root/"status.json",dict(status=state,at=now(),completed=result["completed"],failed=result["failed"],total=5))
        if state=="COMPLETE": write(root/"completion.json",dict(status="ALL_FIVE_COMPLETE",at=now(),plan_sha256=sha(Path(plan_path)),aggregate_sha256=sha(root/"aggregate.json")))
        return 0 if state=="COMPLETE" else 1

def launch(plan_path):
    plan=verify(plan_path); root=Path(plan["output_dir"]); acc=read(root/"preflight/acceptance.json")
    if acc.get("status")!="PASS" or acc.get("plan_sha256")!=sha(plan_path): raise ValueError("GPU preflight acceptance required")
    with (root/"launch.lock").open("x"): pass
    env=dict(os.environ,PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    cmd=[sys.executable,"-B","-m","spear_current.av2_corrected","--supervise",str(Path(plan_path).resolve())]
    with (root/"supervisor.log").open("x") as log: child=subprocess.Popen(cmd,cwd=plan["source_snapshot"],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/"launch.json",dict(pid=child.pid,at=now(),command=cmd)); return child.pid

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--register",type=Path); g.add_argument("--preflight",type=Path); g.add_argument("--launch",type=Path); g.add_argument("--supervise",type=Path); g.add_argument("--worker",type=Path); g.add_argument("--aggregate",type=Path)
    p.add_argument("--seed",type=int); a=p.parse_args()
    if a.register: print(register(a.register))
    elif a.preflight: print(json.dumps(preflight(a.preflight),indent=2))
    elif a.launch: print(launch(a.launch))
    elif a.supervise: raise SystemExit(supervise(a.supervise))
    elif a.worker: worker(a.worker,a.seed)
    else: print(json.dumps(aggregate(a.aggregate),indent=2))
if __name__=="__main__": main()
