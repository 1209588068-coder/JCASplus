"""Replay the current fixed models on original-TRAIN audit only; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .attention_context import attention_context
from .data import load_normalizer, read_role_part
from .evaluation import evaluate
from .role_data import build_contexts, _tensor_hash as context_tensor_hash
from .role_training import _tensor_hash
from .runtime import load_bundle, restore_clean_reference, restore_generator, restore_victim


class SceneInjector:
    """Original scene-binding semantics, using only the standalone modules."""
    def __init__(self, generator, seed):
        self.generator, self.seed = generator, seed
        self.args, self.device = SimpleNamespace(target_class=1), torch.device('cpu')
        self.context = self.features = None

    def bind(self, part):
        context = build_contexts([part], seed=self.seed)['typed']
        self.context = attention_context(context, [part['scenario_id']], self.seed)
        self.features = torch.as_tensor(part['x'], dtype=torch.float32).clone()
        self.permutation_hash = context_tensor_hash(self.context.score_permutation)
        self.coordinates = self.generator.selected_coordinates.detach().clone()

    def inject_trigger(self, indices, x, edge_index, edge_weight, device):
        if (self.context is None or torch.device(device) != self.device
                or not torch.equal(x, self.features) or not torch.equal(edge_index, self.context.edge_index)
                or context_tensor_hash(self.context.score_permutation) != self.permutation_hash
                or not torch.equal(self.generator.selected_coordinates, self.coordinates)
                or any(module.training for module in self.generator.modules())):
            raise ValueError('Inference requires freshly bound original scene and eval model')
        result = self.generator.inject(x, self.context, indices, edge_weight, detach=True)
        selected = self.generator.selected_coordinates[self.generator.hard_support]
        allowed = torch.zeros_like(x, dtype=torch.bool)
        allowed[indices[:, None], selected] = True
        if (not torch.equal(result.features[~allowed], x[~allowed])
                or bool((result.features[allowed].abs() > 10).any())):
            raise ValueError('Current SAGE24 scope or domain changed')
        return result.features, result.edge_index, result.edge_weight


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(_tensor_hash(tensor).encode())
    return digest.hexdigest()


def run(bundle, seed, output):
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root / 'results') or output == root / 'results':
        raise ValueError('Evaluation output must be a new child of results/')
    output.mkdir(parents=True, exist_ok=False)
    models = dict(clean_reference=restore_clean_reference(bundle), victim=restore_victim(bundle, seed))
    generator = restore_generator(bundle, seed)
    hashes = {key: state_hash(model) for key, model in dict(generator=generator, **models).items()}
    attack = SceneInjector(generator, bundle.config['shuffle_seed'])
    normalizer = load_normalizer(bundle)
    ledger = []
    def reader(row):
        part = read_role_part(bundle, row, 'audit', normalizer=normalizer, ledger=ledger)
        attack.bind(part)
        return part
    def progress(**record):
        print(json.dumps(record), flush=True)
    try:
        with torch.no_grad():
            result = evaluate(bundle.config, output, models, attack, reader, progress)
        if hashes != {key: state_hash(model) for key, model in dict(generator=generator, **models).items()}:
            raise ValueError('Evaluation mutated model state')
        result.update(seed=seed, method='unconditional', weights_unchanged=True,
                      scope='reused original TRAIN audit; not independent validation/test')
        with (output / 'summary.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return result
    except Exception as error:
        with (output / 'failure.json').open('x') as stream:
            json.dump(dict(error=str(error)), stream)
        raise
    finally:
        with (output / 'graph_access_ledger.json').open('x') as stream:
            json.dump(ledger, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--seed', type=int, choices=(20260621, 20260622, 20260623), default=20260621)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run', action='store_true', help='Explicitly evaluate; default is asset-only preflight')
    args = parser.parse_args()
    bundle = load_bundle(args.manifest)
    if not args.run:
        print('CURRENT_ASSET_PREFLIGHT_PASS; no graph/model load or training')
        return
    if args.output_dir is None:
        parser.error('--run requires --output-dir')
    torch.set_num_threads(2)
    print(json.dumps(run(bundle, args.seed, args.output_dir)['metrics'], indent=2))


if __name__ == '__main__':
    main()
