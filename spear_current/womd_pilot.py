"""Register and execute one fresh CUDA WOMD main-method development pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .runtime import sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'results/womd_main_pilot_20261104_v2'
DEFAULT_GRAPHS = ROOT / 'graphs/womd_motion_v1_3_1_graph_v1_20261104'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def register(output, graph_root, seed=20261104):
    from .womd_pilot_data import prepare, validate_source
    output, graph_root = Path(output).resolve(), Path(graph_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists():
        raise FileExistsError('Existing registration is not overwritten')
    provenance = validate_source(graph_root)
    if not (output / 'data/summary.json').exists():
        prepare(graph_root, output / 'data', seed)
    data_summary = json.loads((output / 'data/summary.json').read_text())
    resource_path = output/'resources/training_execution_acceptance.json'
    resource = json.loads(resource_path.read_text())
    if resource['dual_clean_kl']['status'] != 'PASS':
        raise ValueError('CUDA resource/gradient acceptance missing')
    for name, digest in resource['sources'].items():
        if sha256(ROOT/'spear_current'/name) != digest:
            raise ValueError('Training code changed after CUDA acceptance: '+name)
    data_registration = json.loads((output / 'data/registration.json').read_text())
    if (data_summary['seed'] != seed or data_summary['status'] != 'PREPARED'
            or Path(data_registration['graph_root']) != graph_root
            or data_registration['manifest_sha256'] != provenance['manifest.csv']):
        raise ValueError('Data registration differs from requested pilot')
    for name, digest in data_summary['asset_sha256'].items():
        if sha256(output / 'data' / name) != digest:
            raise ValueError('Changed prepared data: '+name)
    snapshot = output / 'source_snapshot'
    (snapshot / 'spear_current').mkdir(parents=True, exist_ok=False)
    for file in (ROOT / 'spear_current').glob('*.py'):
        shutil.copyfile(file, snapshot / 'spear_current' / file.name)
    # Exercise the author selection API itself on CUDA before the real run;
    # the earlier encoder/support smoke did not cover this device boundary.
    regression = snapshot/'tests_current/test_womd_gpu_selection.py'
    regression.parent.mkdir(parents=True)
    shutil.copyfile(ROOT/'tests_current/test_womd_gpu_selection.py', regression)
    author = ROOT / 'third_party/author_baselines/SPEAR'
    for file in author.rglob('*.py'):
        if '__pycache__' not in file.parts:
            target = snapshot / 'third_party/author_baselines/SPEAR' / file.relative_to(author)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    plan = dict(schema='womd_fresh_main_pilot_v1', registered_at=now(), seed=seed,
        output_dir=str(output), graph_root=str(graph_root),
        graph_provenance_sha256=provenance, data_summary_sha256=sha256(output/'data/summary.json'),
        resource_acceptance_sha256=sha256(resource_path),
        code_sha256={str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob('*.py'))},
        device='cuda:0', precision='float32', amp=False, tf32=False,
        method='fresh GCN ranking + author MLP/SAGE30 + learned semantic24 + uniform residual + dual clean KL',
        source_selection=dict(attachment_count=26, outer_extra_count=512, model_epochs=200,
                              sage_n_permutations=2048, sage_batch_size=64,
                              explanation_count=128, background_count=32),
        training=dict(clean_proxy_epochs=200, clean_reference_epochs=200, base_epochs=200,
                      residual_epochs=200, support_epochs=200, victim_epochs=200, inner_steps=1),
        models=dict(victim='GraphSAGE', clean_proxies=['GraphSAGE','GraphSAGE'],
                    clean_reference='separate fresh GraphSAGE not used in trigger gradients'),
        seed_offsets=dict(selection=0, clean_proxy_1=101, clean_proxy_2=102,
                          clean_reference=103, anchor=201, support=301, victim=401),
        generator_selection='fixed final epoch; no ASR checkpoint selection',
        classifier_selection='earliest max clean DEV AUC; selection GCN/MLP retain author DEV accuracy rule',
        output_mode='identity', domain_projection=False, distribution_discriminator=False,
        pair_mean=False, inherited_AV2_checkpoints=False, inherited_AV2_support=False,
        evaluation=dict(threshold='binary argmax; tie selects class0',
                        roles=['dev','audit'], targets='registered one negative per scene; no model filtering'),
        support_adaptation='fresh top30; protect357/715; copy-group intersections; exact24; infeasible stops',
        resource_policy='CUDA only; wait for available memory; no automatic CPU fallback or sample reduction',
        message_execution='native SAGE CSR fused mean; GCN channel decomposition maximum16 exact-divisor; same edges',
        pre_training_regression='synthetic CUDA GCN->MLP->SAGE source-selection API; fail closed',
        minimum_cuda_free_bytes=16*1024**3, gpu_poll_seconds=30,
        scope='one development pilot, not full-dataset repetition or blind independent recording test',
        recording_independence_proven=False, test_arrays_loaded=False,
        successor_policy='review pilot before scheduling formal multi-seed experiments')
    with (output / 'plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    write_json(output / 'registration.json', dict(plan_sha256=sha256(output/'plan.json'), registered_at=now()))
    write_json(output / 'status.json', dict(status='REGISTERED', at=now(), training_started=False))
    return output / 'plan.json'


def verify(plan_path):
    from .womd_pilot_data import validate_source
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan['output_dir'])
    registration = json.loads((output/'registration.json').read_text())
    if sha256(plan_path) != registration['plan_sha256']:
        raise ValueError('Changed pilot plan')
    if validate_source(plan['graph_root']) != plan['graph_provenance_sha256']:
        raise ValueError('Changed graph provenance')
    if sha256(output/'data/summary.json') != plan['data_summary_sha256']:
        raise ValueError('Changed data summary')
    if sha256(output/'resources/training_execution_acceptance.json') != plan['resource_acceptance_sha256']:
        raise ValueError('Changed CUDA execution acceptance')
    for name, digest in plan['code_sha256'].items():
        if sha256(output/'source_snapshot'/name) != digest:
            raise ValueError('Changed source snapshot: '+name)
    return plan


def execute(plan_path):
    import numpy as np
    import torch
    from .womd_pilot_data import load_role
    from .womd_pilot_selection import fit_source_selection, fit_semantic_support
    from .womd_pilot_training import cuda_preflight, fit_classifier, fit_generator_anchor, fit_victim, evaluate
    from .full_ablation_training import _cpu_state, _state_hash
    from .full_ablation_evaluation import _fp32

    plan = verify(plan_path)
    output, seed = Path(plan['output_dir']), plan['seed']
    if Path(__file__).resolve() != output/'source_snapshot/spear_current/womd_pilot.py':
        raise ValueError('Execute the registered snapshot using --launch; live-source --run is rejected')
    started = time.monotonic()
    if (output/'completion.json').exists():
        raise ValueError('Completed pilot is not rerun')
    if any((output/name).exists() for name in ('selection','clean_proxy_1','anchor')):
        raise ValueError('Partial training exists; review checkpoints before resuming, never overwrite')
    training_started = False

    def status(state, **extra):
        write_json(output/'status.json', dict(status=state, at=now(), pid=os.getpid(),
            elapsed_seconds=time.monotonic()-started, training_started=training_started, **extra))

    def progress(row):
        status('RUNNING', progress=row)
        if row.get('epoch', 0) % 10 == 0 or 'epoch' not in row:
            print('[womd-pilot] '+json.dumps(row, allow_nan=False), flush=True)

    try:
        # A waiting process has not launched training. It releases test memory
        # each attempt and never terminates or changes other users' GPU jobs.
        while True:
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable')
                free, total = torch.cuda.mem_get_info(plan['device'])
                if free < plan['minimum_cuda_free_bytes']:
                    raise RuntimeError(f'CUDA free {free/1024**3:.1f}GiB below registered headroom16GiB')
                gpu = cuda_preflight(plan['device'])
                break
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if ('out of memory' not in str(exc).lower()
                        and not str(exc).startswith('CUDA free ')):
                    raise
                status('WAITING_FOR_GPU', reason=str(exc))
                print('[womd-pilot] waiting for GPU: '+str(exc), flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(plan['gpu_poll_seconds'])
        write_json(output/'cuda_preflight.json', dict(at=now(), **gpu))
        status('CUDA_SOURCE_SELECTION_CHECK')
        regression_path = output/'resources/cuda_source_selection_regression'
        with (output/'resources/cuda_source_selection_regression.log').open('x') as log:
            checked = subprocess.run([sys.executable, '-B',
                str(output/'source_snapshot/tests_current/test_womd_gpu_selection.py'),
                '--worker', str(regression_path)], cwd=output/'source_snapshot',
                env=dict(os.environ, PYTHONPATH=str(output/'source_snapshot')),
                stdout=log, stderr=subprocess.STDOUT, timeout=120)
        if checked.returncode != 0:
            raise RuntimeError('CUDA source-selection regression failed; inspect resources/cuda_source_selection_regression.log')
        write_json(output/'resources/cuda_source_selection_regression.json', dict(
            status='PASS', at=now(), scope='tiny synthetic graphs, not WOMD experiment results',
            returncode=checked.returncode,
            log_sha256=sha256(output/'resources/cuda_source_selection_regression.log')))
        device = plan['device']
        np.random.seed(seed); torch.manual_seed(seed)
        train_parts, train = load_role(output/'data', 'train', device)
        dev_parts, dev = load_role(output/'data', 'dev', device)
        del train_parts
        data_summary = json.loads((output/'data/summary.json').read_text())
        if sha256(output/'data/indices.pt') != data_summary['asset_sha256']['indices.pt']:
            raise ValueError('Changed initial clean supervision')
        indices = torch.load(output/'data/indices.pt', weights_only=False, map_location=device)
        training_started = True
        status('RUNNING', stage='fresh_source_selection')
        with _fp32():
            selection = fit_source_selection(train, dev, seed=seed, output_dir=output/'selection',
                indices=indices, progress=progress, **plan['source_selection'])
        indices, coordinates = selection['indices'], selection['coordinates']
        offsets, config = plan['seed_offsets'], plan['training']
        proxies = []
        for role in ('clean_proxy_1','clean_proxy_2'):
            result = fit_classifier(train, dev, indices['idx_train'], seed=seed+offsets[role],
                role=role, output_dir=output/role, epochs=config['clean_proxy_epochs'],
                device=device, progress=progress)
            proxies.append(result['model'])
        anchor = fit_generator_anchor(train, indices, proxies, coordinates, seed=seed+offsets['anchor'],
            output_dir=output/'anchor', base_epochs=config['base_epochs'],
            residual_epochs=config['residual_epochs'], inner_steps=config['inner_steps'],
            device=device, progress=progress)
        generator = anchor['generator']
        with _fp32():
            support, support_summary = fit_semantic_support(generator, anchor['surrogate'], train,
                indices, proxies, coordinates, seed=seed+offsets['support'], output_dir=output/'support',
                epochs=config['support_epochs'], progress=progress)
        generator.set_support(support)
        generator.set_phase('support_surrogate').requires_grad_(False).eval()
        torch.save(dict(state_dict=_cpu_state(generator), coordinates=coordinates.cpu(),
            hard_support=support.cpu(), phase='support_surrogate',
            metadata=dict(output_mode='identity', uniform_residual=True, seed=seed)), output/'generator.pt')
        generator_hash = _state_hash(generator)
        del proxies, anchor
        gc.collect(); torch.cuda.empty_cache()
        victim = fit_victim(generator, train, dev, indices, seed=seed+offsets['victim'],
            output_dir=output/'victim', epochs=config['victim_epochs'], device=device, progress=progress)
        reference = fit_classifier(train, dev, indices['idx_train'], seed=seed+offsets['clean_reference'],
            role='clean_reference', output_dir=output/'clean_reference',
            epochs=config['clean_reference_epochs'], device=device, progress=progress)
        # No target response is evaluated before selection/support/victim freeze.
        del train, dev, indices
        gc.collect(); torch.cuda.empty_cache()
        results = {}
        for role in ('dev','audit'):
            if role == 'dev':
                parts = dev_parts
            else:
                parts, audit = load_role(output/'data', role)
                del audit
            targets = [p['target_index'] for p in parts]
            results[role] = evaluate(generator, victim['model'], reference['model'], parts, targets,
                role=role, output_dir=output/('evaluation_'+role), device=device, progress=progress)
        verify(plan_path)
        completion = dict(status='PILOT_COMPLETE_REVIEW_REQUIRED', completed_at=now(), seed=seed,
            elapsed_seconds=time.monotonic()-started, generator_sha256=generator_hash,
            test_arrays_loaded=False, recording_independence_proven=False,
            result_paths={r:str(output/('evaluation_'+r)/'summary.json') for r in results},
            results=results, plan_sha256=sha256(plan_path), automatic_formal_runs_started=False)
        write_json(output/'completion.json', completion)
        status('PILOT_COMPLETE_REVIEW_REQUIRED', completion=str(output/'completion.json'))
        print('[womd-pilot] complete; review '+str(output/'completion.json'), flush=True)
    except Exception as exc:
        status('FAILED_REVIEW_REQUIRED', error_type=type(exc).__name__, error=str(exc),
               traceback=traceback.format_exc())
        raise


def launch(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = verify(plan_path)
    output = Path(plan['output_dir'])
    if (output/'launch.json').exists():
        raise FileExistsError('Existing launch record; do not duplicate jobs')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, '-B', '-m', 'spear_current.womd_pilot', '--run', str(plan_path)]
    with (output/'run.log').open('x') as log:
        process = subprocess.Popen(command, cwd=output/'source_snapshot', env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=process.pid, launched_at=now(), command=command,
                  cwd=str(output/'source_snapshot'), plan_sha256=sha256(plan_path))
    write_json(output/'launch.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', type=Path)
    actions.add_argument('--launch', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--graph-root', type=Path, default=DEFAULT_GRAPHS)
    args = parser.parse_args()
    if args.prepare:
        print(register(args.output_dir, args.graph_root))
    elif args.run:
        execute(args.run)
    else:
        print(json.dumps(launch(args.launch), indent=2))


if __name__ == '__main__':
    main()
