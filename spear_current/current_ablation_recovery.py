"""Recover interrupted current-module cases without changing training math.

Original plan bytes and completed cases remain authoritative. A separate
operational registration binds recovery code, old evidence, pending cases and
more conservative shared-GPU admission. No model checkpoint is resumed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from . import current_module_ablation as original
from .full_ablation import read,write,sha,state
from .support_replication import now,_env,_resources,_verify_prediction_artifacts

ROOT=original.ROOT
VERSION='current_ablation_resource_recovery_v1'
FIRST_FREE=12*1024**3
SECOND_FREE=16*1024**3


def is_resource_oom(failure):
    kind=failure.get('root_error_type',failure.get('error_type',''))
    if kind=='OutOfMemoryError':return True
    if kind not in ('AcceleratorError','RuntimeError'):return False
    message=str(failure.get('error','')).lower()
    return 'cudaerrormemoryallocation' in message or ('cuda' in message and 'out of memory' in message)


def warm_started(folder):
    p=Path(folder)/'progress.jsonl'
    if not p.exists():return False
    try:row=json.loads(p.read_text().splitlines()[-1])
    except (ValueError,IndexError):return False
    r=row.get('record',{})
    return bool(row.get('phase')=='generator_training' and str(r.get('device','')).startswith('cuda')
        and type(r.get('epoch')) is int and r['epoch']>=0
        and r.get('cumulative_surrogate_steps',0)>0)


def validate_layout(root,plan):
    root=Path(root).resolve();parent=Path(plan['parent_plan']).resolve().parent
    if (plan['version']!=VERSION or plan['matrix']!=original.matrix()
            or plan['resource_policy']!=dict(max_concurrent=2,first_min_free_bytes=FIRST_FREE,
                second_min_free_bytes=SECOND_FREE,stable_probes=2,cold_start_gap_seconds=15,
                wait_first_epoch=True,no_cpu_fallback=True,kill_external_jobs=False)
            or root==parent or (root/'plan.json').is_symlink()):
        raise ValueError('Recovery layout/admission contract changed')
    keys=lambda rows:{(r['seed'],r['variant']) for r in rows}
    completed,pending=plan['retained_completed'],plan['pending']
    if (len(keys(completed))!=len(completed) or len(keys(pending))!=len(pending)
            or keys(completed)&keys(pending) or keys(completed)|keys(pending)!=keys(original.matrix())):
        raise ValueError('Recovery completed/pending matrix lost identity or disjointness')
    for row in completed:
        path=original.case_dir(root,row['seed'],row['variant'])
        expected=original.case_dir(parent,row['seed'],row['variant']).resolve()
        if (not path.is_symlink() or path.resolve()!=expected
                or str(expected)!=row['source'] or path.parent.is_symlink()):
            raise ValueError('Retained completed case does not point exactly to its parent case')
    for row in pending:
        path=original.case_dir(root,row['seed'],row['variant'])
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Pending output may not alias old interrupted evidence')


def recover_prepare(parent_plan,output):
    parent_plan=Path(parent_plan).resolve();parent=parent_plan.parent
    child=original.verify(parent_plan);expected=sha(parent_plan)
    if not (parent/'failure.json').is_file():raise ValueError('Require preserved interrupted parent evidence')
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'results') or output==ROOT/'results' or output==parent:
        raise ValueError('Require an exclusive new results directory')
    retained=[];pending=[];bindings={str(parent_plan):expected,
        str(parent/'failure.json'):sha(parent/'failure.json')}
    interrupted=[]
    for item in original.matrix():
        folder=original.case_dir(parent,**item)
        receipt=folder/'completion.json'
        if receipt.exists():
            c=read(receipt);r=read(folder/'result.json')
            if (c['status']!='CASE_COMPLETE' or c['plan_sha256']!=expected
                    or c['result_sha256']!=sha(folder/'result.json')
                    or r['optimizer_steps']!=original.expected_steps(item['variant'])
                    or r['selection']!=child['policies'][str(item['seed'])][item['variant']]):
                raise ValueError('Parent completed case failed integrity')
            _verify_prediction_artifacts(r['metrics'])
            retained.append(dict(**item,source=str(folder.resolve()),result_sha256=c['result_sha256']))
            for file in (receipt,folder/'result.json'):bindings[str(file)]=sha(file)
            for key in ('raw_targets','raw_clean_scores','raw_triggered_pair_scores'):
                b=r['metrics'][key];bindings[b['path']]=b['sha256']
        else:
            pending.append(item)
            if folder.exists():
                evidence=[]
                epoch_records=[]
                epoch_log=folder/'training'/'epochs.jsonl'
                if epoch_log.exists():
                    for line in epoch_log.read_text().splitlines():
                        try:epoch_records.append(json.loads(line))
                        except ValueError:pass
                for file in sorted(folder.rglob('*')):
                    if file.is_file() and file.suffix in ('.json','.jsonl'):
                        bindings[str(file)]=sha(file);evidence.append(str(file))
                log=parent/'logs'/f"seed{item['seed']}_{item['variant']}.log"
                if log.exists():bindings[str(log)]=sha(log);evidence.append(str(log))
                interrupted.append(dict(**item,old_case=str(folder),evidence=evidence,
                    old_partial_weights_reused=False,restart_from_registered_seed=True,
                    old_training_epoch_records=len(epoch_records),
                    old_last_optimizer_counters={k:epoch_records[-1].get(k) for k in
                        ('cumulative_generator_steps','cumulative_surrogate_steps')} if epoch_records else {}))
    if not retained or not pending:raise ValueError('Require both verified completed and unfinished cases')
    output.mkdir(parents=True,exist_ok=False)
    # plan must be a real file: original.run_case resolves its path to determine
    # the output root. Symlinking it would accidentally point back to the parent.
    shutil.copyfile(parent_plan,output/'plan.json')
    shutil.copyfile(parent/'registration.json',output/'registration.json')
    (output/'source_snapshot.tar').symlink_to(parent/'source_snapshot.tar')
    if (parent/'preflight').exists():(output/'preflight').symlink_to(parent/'preflight',target_is_directory=True)
    for row in retained:
        link=original.case_dir(output,row['seed'],row['variant'])
        link.parent.mkdir(exist_ok=True);link.symlink_to(row['source'],target_is_directory=True)
    source=Path(__file__).resolve()
    plan=dict(version=VERSION,created_utc=now(),parent_plan=str(parent_plan),
        parent_plan_sha256=expected,matrix=original.matrix(),retained_completed=retained,pending=pending,
        interrupted_attempts=interrupted,bound_files=bindings,recovery_source=str(source),
        recovery_source_sha256=sha(source),training_plan_is_byte_identical=True,
        training_budget_and_algorithms_unchanged=True,preflight_inherited_from_parent=True,
        resource_policy=dict(max_concurrent=2,first_min_free_bytes=FIRST_FREE,
            second_min_free_bytes=SECOND_FREE,stable_probes=2,cold_start_gap_seconds=15,
            wait_first_epoch=True,no_cpu_fallback=True,kill_external_jobs=False),
        recovery_amendment='One explicitly registered restart of each incomplete case; old OOM/interruption retained; no partial weights or performance-conditioned retry',
        later_failure_policy='numeric/OOM fails that case with no retry; integrity/unknown failures stop queue',
        no_graph_arrays_loaded=True,no_model_tensors_loaded=True)
    write(output/'recovery_plan.json',plan)
    shutil.copyfile(source,output/'recovery_controller_snapshot.py')
    write(output/'recovery_registration.json',dict(recovery_plan_sha256=sha(output/'recovery_plan.json'),
        controller_snapshot_sha256=sha(output/'recovery_controller_snapshot.py'),utc=now()))
    verify_recovery(output)
    state(output,'RECOVERY_REGISTERED',retained_completed=len(retained),pending_cases=len(pending))
    print(output/'recovery_plan.json',flush=True);return output


def verify_recovery(root):
    root=Path(root).resolve();plan=read(root/'recovery_plan.json');receipt=read(root/'recovery_registration.json')
    if (sha(root/'recovery_plan.json')!=receipt['recovery_plan_sha256']
            or sha(root/'recovery_controller_snapshot.py')!=receipt['controller_snapshot_sha256']
            or sha(plan['recovery_source'])!=plan['recovery_source_sha256']):
        raise ValueError('Recovery source/plan snapshot changed')
    original.verify(plan['parent_plan'],plan['parent_plan_sha256'])
    original.verify(root/'plan.json',plan['parent_plan_sha256'])
    if (root/'plan.json').read_bytes()!=Path(plan['parent_plan']).read_bytes():
        raise ValueError('Recovery child training plan is not byte-identical')
    for path,digest in plan['bound_files'].items():
        if sha(path)!=digest:raise ValueError('Parent evidence changed: '+path)
    validate_layout(root,plan)
    return plan


def launch(root):
    root=Path(root).resolve();plan=verify_recovery(root)
    write(root/'recovery_launch_claim.json',dict(pid=os.getpid(),utc=now()))
    command=[sys.executable,'-B','-u','-m','spear_current.current_ablation_recovery','--queue',str(root)]
    with (root/'recovery_queue.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=_env(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,command=command,retained_completed=len(plan['retained_completed']),
        pending_cases=len(plan['pending']),utc=now())
    write(root/'recovery_launch.json',receipt);print(receipt,flush=True);return receipt


def queue(root):
    root=Path(root).resolve();plan=verify_recovery(root)
    expected=plan['parent_plan_sha256'];child_plan=read(root/'plan.json')
    pending=list(plan['pending']);active=[];completed=[];failed=[];stable=0
    began=time.monotonic();last_launch=-float('inf')
    def interrupt(signum,frame):raise RuntimeError('Recovery interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    (root/'recovery_logs').mkdir(exist_ok=False)
    try:
        while pending or active:
            verify_recovery(root)
            if (root/'CANCEL').exists():raise RuntimeError('Recovery cancelled')
            for job in list(active):
                code=job['process'].poll()
                if code is None:continue
                job['log'].close();active.remove(job)
                if code:
                    path=original.case_dir(root,job['seed'],job['variant'])/'failure.json'
                    f=read(path) if path.exists() else {}
                    if is_resource_oom(f) or f.get('root_error_type')=='FloatingPointError':
                        failed.append(dict(seed=job['seed'],variant=job['variant'],failure=f));continue
                    raise RuntimeError('Non-resource recovery case failure '+str(job['variant'])+f' exit={code}')
                completed.append(dict(seed=job['seed'],variant=job['variant']))
            resources=None
            warm=all(warm_started(original.case_dir(root,j['seed'],j['variant'])) for j in active)
            if pending and len(active)<2 and warm and time.monotonic()-last_launch>=15:
                minimum=SECOND_FREE if active else FIRST_FREE
                resource_plan={**child_plan,'resource_policy':{**child_plan['resource_policy'],
                    'min_cuda_free_bytes':minimum}}
                resources=_resources(resource_plan);stable=stable+1 if resources['ready'] else 0
                if stable>=2:
                    item=pending.pop(0);stable=0
                    command=[sys.executable,'-B','-u','-m','spear_current.current_module_ablation',
                        '--case',item['variant'],'--seed',str(item['seed']),'--plan',str(root/'plan.json'),
                        '--expected-sha256',expected]
                    log=(root/'recovery_logs'/f"seed{item['seed']}_{item['variant']}.log").open('x')
                    try:child=subprocess.Popen(command,cwd=ROOT,env=_env(),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:log.close();raise
                    active.append(dict(**item,process=child,log=log));last_launch=time.monotonic()
                    write(root/'recovery_logs'/f"seed{item['seed']}_{item['variant']}_launch.json",
                        dict(**item,pid=child.pid,resources=resources,minimum_free_bytes=minimum,utc=now()))
            else:stable=0
            if time.monotonic()-began>72*3600:raise TimeoutError('Recovery resource/runtime timeout')
            state(root,'RECOVERY_GPU_RUNNING' if active else 'RECOVERY_WAITING_GPU_HEADROOM',
                retained_completed=len(plan['retained_completed']),new_completed=len(completed),
                completed_cases=len(plan['retained_completed'])+len(completed),failed_cases=failed,
                pending_cases=len(pending),active=[{k:j[k] for k in ('seed','variant')}|{'pid':j['process'].pid} for j in active],
                stable_resource_probes=stable,resources=resources)
            if pending or active:time.sleep(5)
        verify_recovery(root)
        result=original.aggregate(root/'plan.json',expected)
        origins=[]
        for item in original.matrix():
            folder=original.case_dir(root,**item)
            origins.append(dict(**item,source=str(folder.resolve()),retained_from_parent=folder.is_symlink()))
        write(root/'case_origin_ledger.json',origins)
        write(root/'recovery_completion.json',dict(status=result['status'],retained_completed=len(plan['retained_completed']),
            newly_completed=len(completed),newly_failed=len(failed),training_plan_sha256=expected,
            aggregate_sha256=sha(root/'aggregate.json'),utc=now()))
    except BaseException as error:
        for job in active:
            if job['process'].poll() is None:os.killpg(job['process'].pid,signal.SIGTERM)
            job['log'].close()
        write(root/'recovery_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=now()))
        state(root,'RECOVERY_FAILED',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',type=Path);mode.add_argument('--launch',type=Path);mode.add_argument('--queue',type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.prepare:
        if args.output is None:parser.error('--prepare requires --output')
        recover_prepare(args.prepare,args.output)
    elif args.launch:launch(args.launch)
    else:queue(args.queue)


if __name__=='__main__':main()
