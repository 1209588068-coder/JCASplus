"""Original AV2 bridge only; never supplies an attack implementation."""
from pathlib import Path

import torch

from spear_current.data import combine_parts, load_normalizer, read_role_part
from spear_current.development_diagnostic import registered_rows
from comparison_methods.evidence import tensor_hash, validate_injection


def load_inputs(bundle, ledger):
    sources = registered_rows(bundle)
    normalizer = load_normalizer(bundle)
    parts, combined = {}, {}
    for role in ('train', 'dev'):
        parts[role] = [read_role_part(bundle, row, role, normalizer, ledger) for row in sources[role]]
        combined[role] = combine_parts(parts[role])
    indices = torch.load(bundle.asset_path('indices'), weights_only=True, map_location='cpu')
    result = assemble(combined['train'], combined['dev'], indices)
    result.update(parts=parts, sources=sources)
    if len(ledger) != 640 or any(row['split'] != 'train' or row['role'] not in ('train', 'dev')
            or row['status'] != 'READ_COMPLETE' for row in ledger):
        raise ValueError('Unexpected graph access')
    return result


def assemble(train, dev, indices):
    """Disjoint graph union: DEV labels only in the author's validation calls.

    Context labels -1 are retained, not silently converted to a normal class.
    All loss, validation and candidate indices exclude these context nodes.
    """
    n = len(train['x'])
    x = torch.cat((train['x'], dev['x']))
    y = torch.cat((train['y'], dev['y']))
    edges = torch.cat((train['edge_index'], dev['edge_index'] + n), dim=1)
    idx_train = indices['idx_train'].clone()
    idx_val = torch.nonzero(dev['eligible'], as_tuple=False).flatten() + n
    reservoir = torch.unique(torch.cat((indices['idx_attach'], indices['idx_unlabeled'])))
    if bool(((reservoir < 0) | (reservoir >= n)).any()):
        raise ValueError('TRAIN reservoir outside TRAIN')
    candidates = reservoir[train['eligible'][reservoir] & (train['y'][reservoir] == 0)].sort().values
    if (idx_train.ndim != 1 or not len(idx_train) or bool(((idx_train < 0) | (idx_train >= n)).any())
            or not bool(torch.isin(y[idx_train], torch.tensor([0, 1])).all())
            or set(y[idx_val].tolist()) != {0, 1}
            or bool(torch.isin(idx_train, candidates).any())
            or not torch.equal(idx_val, indices['dev_indices'])):
        raise ValueError('Changed training, validation or candidate identities')
    if bool(((edges[0] < n) != (edges[1] < n)).any()):
        raise ValueError('Cross-role edge')
    return dict(x=x, y=y, edge_index=edges, idx_train=idx_train, idx_val=idx_val,
                candidates=candidates, train_nodes=n, dev_nodes=len(dev['x']),
                context_labels_unchanged=True)


def input_hashes(inputs):
    return {key: tensor_hash(inputs[key]) for key in
            ('x', 'y', 'edge_index', 'idx_train', 'idx_val', 'candidates')}


def validate_author_injection(x, edges, targets, result, **kwargs):
    """Preserve author's raw weighted edges, including its native zero weights.

    Some author GraphSAGE implementations ignore weights. Do not silently
    "fix" that behavior by removing edges at inference. Permission checks use
    the actual stored topology; only a temporary all-ones vector is used to
    call the shared geometry validator. The model receives original weights.
    """
    fx, fe, fw = result
    if (fw.shape != (fe.shape[1],) or not torch.isfinite(fw).all()
            or not bool(((fw == 0) | (fw == 1)).all())):
        raise ValueError('Unexpected author edge weights')
    validate_injection(x, edges, targets, (fx, fe, torch.ones_like(fw)), **kwargs)


def validate_fit(fitted, inputs, method, config):
    targets = fitted['attachment_indices']
    if (targets.dtype != torch.long or targets.ndim != 1 or len(targets) != 26
            or targets.unique().numel() != len(targets)
            or not bool(torch.isin(targets, inputs['candidates']).all())):
        raise ValueError('Author selection violated the registered TRAIN candidate budget')
    expected_y = inputs['y'].clone()
    expected_y[targets] = 1
    if not torch.equal(expected_y, fitted['poisoned_y']):
        raise ValueError('Unexpected label changes, including context or DEV')
    expected_fit = torch.cat((inputs['idx_train'], targets)).sort().values
    if not torch.equal(expected_fit, fitted['fit_indices'].sort().values):
        raise ValueError('Unexpected victim supervision')
    validate_author_injection(inputs['x'], inputs['edge_index'], targets,
        (fitted['poisoned_x'], fitted['poisoned_edge_index'], fitted['poisoned_edge_weight']),
        structural=method != 'spear', feature_count=config.get('alpha_int', 30),
        trigger_size=config.get('trigger_size', 3))
    # Author get_poisoned may filter zero-weight edges while inject_trigger
    # intentionally does not. Do not require those native paths to be equal.
