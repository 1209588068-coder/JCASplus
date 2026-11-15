# Source: local experiments/spear_supplied_v1/data.py; frozen-normalizer adapter.
"""Observation-only AV2 input adapter for the supplied SPEAR implementation.

The existing UGBA bridge owns canonical pair identity, line topology, dynamic
risk labels and strict eligibility. This adapter retains both vehicles' full
stored observation features and both directed observation edge vectors instead
of reducing each pair to an averaged edge vector. No attack code lives here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .pair_graph import line_graph_from_graph as _pair_line_graph
from .risk_labels import RiskLabelConfig


NODE_FEATURE_DIM = 358
EDGE_FEATURE_DIM = 34
PAIR_FEATURE_DIM = 2 * NODE_FEATURE_DIM + 2 * EDGE_FEATURE_DIM
BRIDGE_VERSION = "spear_supplied_full_observation_pair_line_graph_v1"
NORMALIZER_VERSION = "spear_supplied_train_column_normalizer_v1"
MINIMUM_SCALE = 1e-6
CLIP = 10.0


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real two-dimensional feature matrix")
    return result


def line_graph_from_graph(
    graph: Any, config: RiskLabelConfig = RiskLabelConfig()
) -> dict[str, Any]:
    """Return canonical pair nodes with all 784 stored observation coordinates.

    For track-ID-ordered endpoints ``a, b``, the feature layout is exactly
    ``[x_node[a], x_node[b], edge_attr[a->b], edge_attr[b->a]]``. The source
    graph and the returned mapping arrays follow the existing bridge contract.
    Future information is used by that bridge only to produce dynamic labels
    and computability; it never contributes to the feature vector or topology.
    The returned arrays do not alias the source graph's arrays.
    """
    node_features = _matrix(graph["x_node"], "x_node")
    node_count = len(np.asarray(graph["node_track_ids"]))
    if node_features.shape != (node_count, NODE_FEATURE_DIM):
        raise ValueError("x_node must contain exactly 358 observed features per vehicle node")

    result = _pair_line_graph(graph, config)
    pairs = result["canonical_pairs"]
    directions = result["source_directed_edges"]
    edge_features = np.asarray(graph["edge_attr"])
    x = np.concatenate(
        [node_features[pairs[:, 0]], node_features[pairs[:, 1]],
         edge_features[directions[:, 0]], edge_features[directions[:, 1]]],
        axis=1,
    ).astype(np.float32, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("pair features cannot be represented as finite float32")
    result["x"] = x
    result["metadata"] = {
        **result["metadata"],
        "bridge_version": BRIDGE_VERSION,
        "feature_source": "stored_observation_node358_both_endpoints_and_edge34_both_directions",
        "feature_dimension": PAIR_FEATURE_DIM,
        "feature_layout": [
            {"name": "canonical_first_vehicle_observation", "start": 0, "stop": 358},
            {"name": "canonical_second_vehicle_observation", "start": 358, "stop": 716},
            {"name": "canonical_forward_edge_observation", "start": 716, "stop": 750},
            {"name": "canonical_reverse_edge_observation", "start": 750, "stop": 784},
        ],
        "feature_normalization": "none_in_bridge",
    }
    return result


def transform_features(x: np.ndarray, normalizer: Mapping[str, Any]) -> np.ndarray:
    """Apply frozen TRAIN column scaling and clip to [-10, 10], without mutation.

    Statistics may be loaded from an NPZ dictionary or JSON lists. This function
    never updates them, even when an evaluation feature lies far outside TRAIN.
    Empty row matrices are allowed once their feature dimension matches.
    """
    value = _matrix(x, "x")
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    scale = np.asarray(normalizer["scale"], dtype=np.float64)
    dimension = value.shape[1]
    if dimension == 0 or mean.shape != (dimension,) or scale.shape != (dimension,):
        raise ValueError("normalizer mean/scale must match x feature dimension")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < MINIMUM_SCALE):
        raise ValueError("normalizer must contain finite mean and scale at least 1e-6")
    if "feature_dimension" in normalizer and int(normalizer["feature_dimension"]) != dimension:
        raise ValueError("normalizer feature_dimension does not match x")
    if "minimum_scale" in normalizer and float(normalizer["minimum_scale"]) != MINIMUM_SCALE:
        raise ValueError("normalizer minimum_scale must be 1e-6")
    if "clip" in normalizer and float(normalizer["clip"]) != CLIP:
        raise ValueError("normalizer clip must be 10")
    value64 = value.astype(np.float64, copy=True)
    normalized = np.clip((value64 - mean) / scale, -CLIP, CLIP)
    return normalized.astype(np.float32, copy=True)


def load_normalizer(bundle):
    """Load the retained TRAIN normalizer without fitting or graph access."""
    with np.load(bundle.asset_path('normalizer'), allow_pickle=False) as values:
        result = {key: values[key].copy() for key in values.files}
    transform_features(np.empty((0, PAIR_FEATURE_DIM), dtype=np.float32), result)
    return result


class _TrackedGraph:
    def __init__(self, archive):
        self.archive = archive
        self.files = archive.files
        self.loaded = set()

    def __contains__(self, key):
        return key in self.files

    def __getitem__(self, key):
        self.loaded.add(key)
        return self.archive[key]


def read_role_part(bundle, row, role, normalizer=None, ledger=None):
    """Read exactly one registered original-TRAIN row and return normalized data.

    The internal dev/audit roles are subsets of original TRAIN. Original
    validation/test rows, substituted identities, paths outside the registered
    graph directory, and changed graph bytes are rejected before deserialization.
    Original arrays remain read-only inputs; returned arrays own their storage.
    """
    from pathlib import Path
    from .runtime import sha256

    if (role not in ('train', 'dev', 'audit') or row.get('split') != 'train'
            or row.get('role') != role or row not in bundle.config['roles'][role]):
        raise ValueError('Require an exact registered original TRAIN role row')
    root = Path(bundle.manifest['graph_root']).resolve()
    if root != Path(bundle.config['graph_dir']).resolve():
        raise ValueError('Manifest and configuration graph roots disagree')
    path = (root / row['graph_path']).resolve()
    if not path.is_relative_to(root / 'graphs') or path.suffix != '.npz':
        raise ValueError('Graph path outside the registered graph directory')
    if sha256(path) != row['graph_sha256']:
        raise ValueError('Graph hash mismatch before read')
    if normalizer is None:
        normalizer = load_normalizer(bundle)
    access = dict(scenario_id=row['scenario_id'], role=role, split='train',
        path=str(path), sha256=row['graph_sha256'], loaded_members=[], status='OPENING')
    if ledger is not None:
        ledger.append(access)
    try:
        with np.load(path, allow_pickle=True) as archive:
            graph = _TrackedGraph(archive)
            try:
                if str(graph['scenario_id'].item()) != row['scenario_id']:
                    raise ValueError('Graph scenario identity mismatch')
                part = line_graph_from_graph(graph, RiskLabelConfig(**bundle.config['label_config']))
            finally:
                access['loaded_members'] = sorted(graph.loaded)
                access['future_members_used_only_for_unchanged_labels'] = [
                    key for key in sorted(graph.loaded) if key.startswith('future_')]
        if sha256(path) != row['graph_sha256']:
            raise ValueError('Graph bytes changed during read')
        part['x'] = transform_features(part['x'], normalizer)
        part.update(scenario_id=row['scenario_id'], split='train')
        access['status'] = 'READ_COMPLETE'
        return part
    except Exception:
        access['status'] = 'READ_FAILED'
        raise


def combine_parts(parts):
    """Combine canonical scene rows with the original disjoint-union ordering."""
    import torch

    if not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError('Require a nonempty ordered list of scene parts')
    xs, ys, edges, eligible, offsets, offset = [], [], [], [], [], 0
    for part in parts:
        xs.append(part['x'])
        ys.append(part['y'])
        eligible.append(part['eligible'])
        edges.append(part['edge_index'] + offset)
        offsets.append(offset)
        offset += len(part['x'])
    return dict(x=torch.from_numpy(np.concatenate(xs)).float(),
        y=torch.from_numpy(np.concatenate(ys)).long(),
        eligible=torch.from_numpy(np.concatenate(eligible)),
        edge_index=torch.from_numpy(np.concatenate(edges, axis=1)).long(), offsets=offsets)


def load_role(bundle, role, normalizer=None, ledger=None):
    """Return ordered normalized parts and their combined tensors for one role."""
    if role not in ('train', 'dev', 'audit'):
        raise ValueError('Only registered roles within original TRAIN are available')
    if normalizer is None:
        normalizer = load_normalizer(bundle)
    parts = [read_role_part(bundle, row, role, normalizer, ledger)
             for row in bundle.config['roles'][role]]
    return parts, combine_parts(parts)
