"""Memory-conscious native PyG aggregation for large WOMD pair graphs.

These are execution-memory settings, not sampling or a new model: every edge
and feature channel is aggregated using the existing MessagePassing operator.
Old source modules and datasets are not patched. Apply to newly constructed
models (including author-source GCNs) before training or checkpoint restoration.
"""
import weakref
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_torch_csr_tensor


DECOMPOSED_LAYERS = 16
_SPARSE_CACHE = {}


def _sage_sparse_input(module, args):
    """Feed native adj_t to SAGE's existing fused message_and_aggregate."""
    x, edges, *extra = args
    if edges.layout != torch.strided:
        return args
    features = x[0] if isinstance(x, tuple) else x
    n = len(features)
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('SAGE fusion needs the original long edge_index')
    key = (id(edges), edges._version, n, str(features.dtype), str(edges.device))
    entry = _SPARSE_CACHE.get(key)
    if entry is None or entry[0]() is not edges:
        adjacency = to_torch_csr_tensor(edges.flip(0),
            edge_attr=features.new_ones(edges.shape[1]), size=(n, n))
        # Coalescing duplicate edges changes mean-degree semantics; reject it.
        if adjacency.values().numel() != edges.shape[1]:
            raise ValueError('Duplicate edges require separate sparse mean audit')
        def remove(reference, key=key):
            if key in _SPARSE_CACHE and _SPARSE_CACHE[key][0] is reference:
                del _SPARSE_CACHE[key]
        entry = (weakref.ref(edges, remove), adjacency)
        _SPARSE_CACHE[key] = entry
    return (x, entry[1], *extra)


def configure_message_passing(module, decomposed_layers=DECOMPOSED_LAYERS):
    if type(decomposed_layers) is not int or decomposed_layers < 1:
        raise ValueError('Positive channel decomposition factor required')
    names = []
    for name, child in module.named_modules():
        if isinstance(child, MessagePassing):
            if child.__class__.__name__ not in ('GCNConv', 'SAGEConv'):
                raise ValueError('Unvalidated message-passing layer: '+child.__class__.__name__)
            if child.__class__.__name__ == 'SAGEConv':
                if child.aggr != 'mean' or child.flow != 'source_to_target':
                    raise ValueError('Only original incoming mean SAGE aggregation was validated')
                child.decomposed_layers = 1
                if not getattr(child, '_womd_sparse_execution', False):
                    child.register_forward_pre_hook(_sage_sparse_input)
                    child._womd_sparse_execution = True
                names.append(dict(name=name, execution='native_CSR_message_and_aggregate'))
                continue
            channels = child.out_channels
            if isinstance(channels, tuple):
                channels = channels[0]
            if not isinstance(channels, int) or channels < 1:
                raise ValueError('Lazy/bipartite widths require a separate decomposition review')
            # GCN transforms before propagation, including its 2-class head.
            # torch.chunk can return fewer chunks than requested; choose an
            # exact divisor so PyG's decomposition loop always has every chunk.
            factor = max(i for i in range(1, min(decomposed_layers, channels)+1)
                         if channels % i == 0)
            child.decomposed_layers = factor
            names.append(dict(name=name, channels=channels, factor=factor))
    return dict(operator='GCN channel decomposition; SAGE native CSR fused mean aggregation',
                decomposed_layers=decomposed_layers, configured_layers=names,
                edge_sampling=False, feature_sampling=False, topology_changed=False)
