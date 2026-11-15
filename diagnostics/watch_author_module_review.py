"""CPU-only collection of registered author runs after their queues terminate.

Reads result JSON only. Never loads graph/model arrays, calls training, or
claims a component's causal benefit from a whole-method attack result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
QUEUES = {
    'UGBA': ('author_raw_rerun_20261029_v1', 'UGBA_REPLICATES_COMPLETE_OTHER_METHODS_PENDING'),
    'GTA': ('author_raw_gta_20261029_v1', 'GTA_NATIVE_PAIR_REPLICATES_COMPLETE'),
    'DPGBA': ('author_raw_dpgba_parallel_20261030_v1', 'DPGBA_NATIVE_REPLICATES_COMPLETE'),
}
SEEDS = {20260621, 20260622, 20260623}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    else:
        temp = path.with_suffix(path.suffix + '.tmp')
        with temp.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        temp.replace(path)


def inspect_queue(method, folder, expected_status):
    completion, failure = folder / 'completion.json', folder / 'failure.json'
    if completion.exists() and failure.exists():
        raise ValueError(f'{method}: contradictory terminal markers')
    if failure.exists():
        return dict(state='FAILED', failure=read(failure), rows=[])
    if not completion.exists():
        return dict(state='RUNNING_OR_WAITING', rows=[])
    try:
        done = read(completion)
    except json.JSONDecodeError:
        return dict(state='TERMINAL_MARKER_BEING_WRITTEN', rows=[])
    if done['status'] != expected_status or len(done['cases']) != 3:
        raise ValueError(f'{method}: invalid completion marker')
    rows, evidence, seen = [], [], set()
    for case in done['cases']:
        seed = case['seed']
        if seed in seen or case.get('returncode', 0) != 0:
            raise ValueError(f'{method}: duplicate seed or failed completion case')
        seen.add(seed)
        result_folder = Path(case.get('output', folder / f'seed{seed}')).resolve()
        if not result_folder.is_relative_to(folder.resolve()):
            raise ValueError(f'{method}: result escaped registered queue')
        result_path = result_folder / 'result.json'
        if sha(result_path) != case['result_sha256']:
            raise ValueError(f'{method}: result hash mismatch')
        data = read(result_path)
        if data.get('attack_seed', data.get('seed')) != seed:
            raise ValueError(f'{method}: result seed mismatch')
        expected_result_status = {'UGBA': 'FRESH_NATIVE_UGBA_COMPLETE',
            'GTA': 'NATIVE_GTA_PAIR_ADAPTER_COMPLETE', 'DPGBA': 'FRESH_NATIVE_DPGBA_COMPLETE'}[method]
        if data.get('status') != expected_result_status:
            raise ValueError(f'{method}: unexpected result status')
        if data.get('original_test_arrays_loaded') is not False:
            raise ValueError(f'{method}: missing test isolation evidence')
        evidence.append(dict(seed=seed, path=str(result_path), sha256=sha(result_path)))
        if method == 'GTA':
            model = data['metrics']['model_poisoned']
            targets = model['targets']['all']
            rows.append(dict(method=method, architecture='GCN', seed=seed,
                incremental_asr=targets['incremental_asr'], absolute_asr=targets['absolute_asr'],
                moving_incremental_asr=model['targets']['moving']['incremental_asr'],
                auc=model['normal_dev_auc'], recall=model['normal_dev_recall'],
                clean_activation=targets['clean_activation'],
                clean_reference_incremental_asr=data['metrics']['clean_reference']['targets']['all']['incremental_asr']))
        else:
            victims = data['victim_runs']
            if len(victims) != (15 if method == 'UGBA' else 1):
                raise ValueError(f'{method}: unexpected victim count')
            for victim in victims:
                key = 'paired_2hop_incremental_asr' if method == 'UGBA' else 'paired_fullgraph_incremental_asr'
                rows.append(dict(method=method, architecture=victim['architecture'], seed=seed,
                    incremental_asr=victim[key], absolute_asr=victim['native_absolute_asr'],
                    moving_incremental_asr=victim.get('moving_incremental_asr'),
                    auc=victim['normal']['auc'], recall=victim['normal']['positive_recall'],
                    clean_activation=victim['clean_activation'], clean_reference_incremental_asr=None))
    if seen != SEEDS:
        raise ValueError(f'{method}: registered seeds incomplete')
    return dict(state='COMPLETE', rows=rows, evidence=evidence)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['method'], row['architecture']].append(row)
    result = []
    for (method, arch), victims in sorted(groups.items()):
        record = dict(method=method, architecture=arch, victim_runs=len(victims),
                      attack_seeds=len({row['seed'] for row in victims}))
        for metric in ('incremental_asr', 'absolute_asr', 'moving_incremental_asr',
                       'auc', 'recall', 'clean_activation', 'clean_reference_incremental_asr'):
            seed_values = defaultdict(list)
            for row in victims:
                if row[metric] is not None:
                    seed_values[row['seed']].append(row[metric])
            means = [statistics.mean(values) for values in seed_values.values()]
            record[metric] = dict(mean=statistics.mean(means),
                sd_across_seed_means=statistics.stdev(means) if len(means) > 1 else None) if means else None
        record['normal_task_review_required'] = all(row['recall'] == 0 for row in victims)
        result.append(record)
    return result


def render(report):
    lines = ['# Author baseline evidence for module review', '',
             'Local authorized AV2/self-trained GNN research. These are development results.', '',
             '| Method | Victim | Attack seeds | Victim runs | Incremental ASR | Normal AUC | Risk recall |',
             '|---|---|---:|---:|---:|---:|---:|']
    for item in report['aggregates']:
        lines.append(f"| {item['method']} | {item['architecture']} | {item['attack_seeds']} | "
                     f"{item['victim_runs']} | {item['incremental_asr']['mean']:.2%} | "
                     f"{item['auc']['mean']:.4f} | {item['recall']['mean']:.2%} |")
    lines += ['', 'Queue states: ' + ', '.join(name + '=' + value['state']
                                             for name, value in report['queues'].items()),
        '', '## Interpretation and next review', '',
        '- Whole-method results do not identify which component caused the effect. No best module is declared.',
        '- UGBA uses random selection in this run; cluster-degree selection has not been tested here.',
        '- UGBA uses per-target two-hop context, DPGBA simultaneous whole-graph injection, and GTA model poisoning.',
        '- UGBA/DPGBA do not provide an independent clean reference. GTA clean-reference responses are stored separately.',
        '- Low normal AUC or zero risk recall must accompany ASR in any comparison; accuracy alone is insufficient.',
        '- Repeated victims sharing a generator are averaged inside each attack seed before reporting seed variability.',
        '- Candidate modules need a single-module, matched-budget comparison with the retained main method.',
        '- CGBA has no verified active rerun registered in this collection and is not silently replaced by historical results.',
        '', 'Module inventory and acceptance plan: AUTHOR_MODULE_TRANSFER_REVIEW_20261030.md.',
        'Collection is complete when all queues reach terminal state; scientific module attribution remains pending.']
    return '\n'.join(lines) + '\n'


def launch(output):
    output.mkdir(parents=True, exist_ok=False)
    queues = {name: dict(path=str(ROOT / 'results' / folder), status=expected,
                        plan_sha256=sha(ROOT / 'results' / folder / 'plan.json'))
              for name, (folder, expected) in QUEUES.items()}
    plan = dict(created_utc=utc(), queues=queues, poll_seconds=60, max_wait_seconds=7 * 86400,
                source_sha256=sha(__file__), action='saved_result_collection_only', gpu_used=False,
                module_attribution_requires_matched_ablation=True)
    write(output / 'plan.json', plan, exclusive=True)
    digest = sha(output / 'plan.json')
    with (output / 'watch.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
            '--run', '--output-dir', str(output), '--plan-sha256', digest], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    receipt = dict(pid=process.pid, launched_utc=utc(), plan_sha256=digest)
    write(output / 'launch.json', receipt, exclusive=True)
    print(json.dumps(receipt))


def run(output, expected):
    plan = read(output / 'plan.json')
    started = time.monotonic()
    try:
        while True:
            if sha(output / 'plan.json') != expected or sha(__file__) != plan['source_sha256']:
                raise ValueError('Observer registration changed')
            states = {}
            for name, queue in plan['queues'].items():
                path = Path(queue['path'])
                if sha(path / 'plan.json') != queue['plan_sha256']:
                    raise ValueError(f'{name}: registered queue plan changed')
                states[name] = inspect_queue(name, path, queue['status'])
            report = dict(created_utc=utc(), queues=states,
                          aggregates=aggregate([row for state in states.values() for row in state['rows']]),
                          module_effect_identified=False, graph_or_model_arrays_loaded=False)
            write(output / 'snapshot.json', report)
            if all(state['state'] in ('COMPLETE', 'FAILED') for state in states.values()):
                write(output / 'evidence.json', report, exclusive=True)
                with (output / 'evidence.md').open('x') as stream:
                    stream.write(render(report))
                write(output / 'completion.json', dict(status='EVIDENCE_COLLECTED_MODULE_REVIEW_PENDING',
                    utc=utc(), queues={name: state['state'] for name, state in states.items()},
                    evidence_sha256=sha(output / 'evidence.json')), exclusive=True)
                return
            write(output / 'status.json', dict(status='WAITING_FOR_AUTHOR_QUEUES', utc=utc(),
                pid=os.getpid(), queues={name: state['state'] for name, state in states.items()}))
            if time.monotonic() - started > plan['max_wait_seconds']:
                raise TimeoutError('Observer wait exceeded seven days')
            time.sleep(plan['poll_seconds'])
    except Exception as error:
        write(output / 'failure.json', dict(error=str(error), type=type(error).__name__, utc=utc()), exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'results') or output == ROOT / 'results':
        parser.error('Observer output must be a new child of JCAS/results')
    if args.launch:
        launch(output)
    elif args.plan_sha256:
        run(output, args.plan_sha256)
    else:
        parser.error('--plan-sha256 required for worker')


if __name__ == '__main__':
    main()
