# Source: local experiments/official_ugba_v1/data.py; bridge only, no UGBA training.
"""Observation-only AV2 pair-to-node data bridge for unmodified official UGBA.

This module contains no attack, generator, selection, or victim-training code.
Each complete, unique bidirectional vehicle pair becomes one line-graph node.
Its 34 input coordinates are the arithmetic mean of the two stored observation
edge-feature vectors. Future trajectories are read only by the existing dynamic
risk label implementation; they never enter node features or line adjacency.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from .motion_gate import motion_pair_gate
from .risk_labels import (
    RiskLabelConfig,
    label_config_dict,
    label_config_hash,
    labels_for_graph,
    selected_label_computable_mask,
)
def supervision_mask(graph, require_strict_label):
    mask = np.asarray(graph["supervision_edge_mask"], dtype=bool)
    if "label_computable_mask" in graph:
        mask = mask & np.asarray(graph["label_computable_mask"], dtype=bool)
    if require_strict_label:
        mask = mask & np.asarray(graph["label_strict_mask"], dtype=bool)
    return mask


def message_passing_mask(graph):
    edge_count = int(np.asarray(graph["edge_index"]).shape[1])
    if "message_passing_edge_mask" not in graph:
        return np.ones(edge_count, dtype=bool)
    mask = np.asarray(graph["message_passing_edge_mask"], dtype=bool)
    if mask.shape != (edge_count,):
        raise ValueError("message_passing_edge_mask shape does not match edge_index")
    return mask


PAIR_FEATURE_DIM = 34
BRIDGE_VERSION = "official_ugba_pair_line_graph_v1"


def _mask(graph: Any, key: str, size: int) -> np.ndarray:
    value = np.asarray(graph[key])
    if value.shape != (size,) or value.dtype.kind != "b":
        raise ValueError(f"{key} must be a boolean array of shape [{size}]")
    return value


def line_graph_from_graph(
    graph: Any, config: RiskLabelConfig = RiskLabelConfig()
) -> dict[str, Any]:
    """Convert one read-only vehicle graph into a deterministic pair line graph.

    ``canonical_pairs`` contains *original vehicle node indices*. Its endpoints
    and its rows are sorted by track-ID strings, so node reindexing preserves
    physical ordering. ``source_directed_edges[:, 0]`` is canonical forward and
    ``[:, 1]`` reverse in the source edge array. A pair with a missing or repeated
    direction is excluded and counted; self-loops are excluded and counted.

    Eligibility is original strict supervision AND dynamic label computability.
    Both directions must agree on eligibility and the dynamic binary label.
    Ineligible line nodes remain as observed topology context with label -1.
    Two distinct line nodes are adjacent only if they share a vehicle and both
    pairs have at least one original message-passing-enabled direction. This
    function must be called separately per scene; it never joins scene IDs.

    Normalization is deliberately separate so the caller can freeze and record
    the exact scope of the author's NormalizeFeatures transform.
    """
    track_ids = np.asarray(graph["node_track_ids"])
    if track_ids.ndim != 1:
        raise ValueError("node_track_ids must have shape [nodes]")
    track_ids = track_ids.astype(str)
    node_count = len(track_ids)
    if len(set(track_ids.tolist())) != node_count or np.any(track_ids == ""):
        raise ValueError("node_track_ids must be unique nonempty strings")

    edges = np.asarray(graph["edge_index"])
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("edge_index must have integer shape [2, edges]")
    if edges.size and (np.any(edges < 0) or np.any(edges >= node_count)):
        raise ValueError("edge_index contains an out-of-range vehicle node")
    edges = edges.astype(np.int64, copy=False)
    edge_count = edges.shape[1]
    features = np.asarray(graph["edge_attr"])
    if features.shape != (edge_count, PAIR_FEATURE_DIM):
        raise ValueError("edge_attr must contain exactly 34 observed features per stored edge")
    if features.dtype.kind not in "fiu" or not np.isfinite(features).all():
        raise ValueError("edge_attr must contain finite real observed features")

    # Validate each source mask before the existing helpers combine them.
    _mask(graph, "supervision_edge_mask", edge_count)
    _mask(graph, "label_strict_mask", edge_count)
    if "label_computable_mask" in graph:
        _mask(graph, "label_computable_mask", edge_count)
    if "message_passing_edge_mask" in graph:
        _mask(graph, "message_passing_edge_mask", edge_count)
    message = message_passing_mask(graph)
    label_bundle = labels_for_graph(graph, config)
    dynamic_labels = np.asarray(label_bundle["edge_label"])
    if dynamic_labels.shape != (edge_count,):
        raise ValueError("dynamic edge label shape does not match stored edges")
    if not np.isfinite(dynamic_labels).all() or not np.isin(dynamic_labels, (0, 1)).all():
        raise ValueError("dynamic edge labels must be finite binary values")
    computable = selected_label_computable_mask(graph, config, label_bundle)
    eligible_edges = supervision_mask(graph, True) & computable

    velocities = np.asarray(graph["observed_velocities_filled"])
    valid = np.asarray(graph["observed_valid_mask"])
    if velocities.ndim != 3 or velocities.shape[0] != node_count:
        raise ValueError("observed velocities must match the vehicle node count")
    # This observation-only gate is audit stratification, never an input column.
    moving_edges = motion_pair_gate(edges, velocities, valid, threshold_mps=0.5)

    directed: dict[tuple[int, int], list[int]] = defaultdict(list)
    pair_keys: set[tuple[str, str]] = set()
    id_to_index = {track_id: i for i, track_id in enumerate(track_ids.tolist())}
    self_loop_edges = 0
    for edge_id, (src_raw, dst_raw) in enumerate(edges.T):
        src, dst = int(src_raw), int(dst_raw)
        if src == dst:
            self_loop_edges += 1
            continue
        directed[src, dst].append(edge_id)
        pair_keys.add(tuple(sorted((str(track_ids[src]), str(track_ids[dst])))))

    canonical: list[tuple[int, int]] = []
    source_edges: list[tuple[int, int]] = []
    pair_track_ids: list[tuple[str, str]] = []
    missing_pairs = duplicate_pairs = 0
    for first_id, second_id in sorted(pair_keys):
        first, second = id_to_index[first_id], id_to_index[second_id]
        forward, reverse = directed.get((first, second), []), directed.get((second, first), [])
        if len(forward) > 1 or len(reverse) > 1:
            duplicate_pairs += 1
            continue
        if len(forward) != 1 or len(reverse) != 1:
            missing_pairs += 1
            continue
        fwd, rev = forward[0], reverse[0]
        if bool(eligible_edges[fwd]) != bool(eligible_edges[rev]):
            raise ValueError(f"bidirectional eligibility mismatch for pair {(first_id, second_id)}")
        if dynamic_labels[fwd] != dynamic_labels[rev]:
            raise ValueError(f"bidirectional dynamic label mismatch for pair {(first_id, second_id)}")
        canonical.append((first, second))
        source_edges.append((fwd, rev))
        pair_track_ids.append((first_id, second_id))

    pairs = np.asarray(canonical, dtype=np.int64).reshape(-1, 2)
    source = np.asarray(source_edges, dtype=np.int64).reshape(-1, 2)
    pair_count = len(pairs)
    # Float64 intermediates avoid overflow and one-sided rounding in averaging;
    # the model receives the original stored feature precision, float32.
    x = (
        0.5 * features[source[:, 0]].astype(np.float64)
        + 0.5 * features[source[:, 1]].astype(np.float64)
    ).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("pair features cannot be represented as finite float32")
    eligible = eligible_edges[source[:, 0]].copy()
    y = np.full(pair_count, -1, dtype=np.int64)
    y[eligible] = dynamic_labels[source[eligible, 0]].astype(np.int64)
    motion = moving_edges[source[:, 0]].astype(np.int8)
    pair_message = (message[source[:, 0]] | message[source[:, 1]]).copy()

    incident: dict[int, list[int]] = defaultdict(list)
    for pair_id, (first, second) in enumerate(pairs):
        if pair_message[pair_id]:
            incident[int(first)].append(pair_id)
            incident[int(second)].append(pair_id)
    line_edges: set[tuple[int, int]] = set()
    for incident_pairs in incident.values():
        for first, second in combinations(incident_pairs, 2):
            line_edges.add((first, second))
            line_edges.add((second, first))
    line_edge_index = np.asarray(sorted(line_edges), dtype=np.int64).reshape(-1, 2).T.copy()
    counts = {
        "source_vehicle_nodes": int(node_count),
        "source_directed_edges": int(edge_count),
        "source_nonself_undirected_pairs": int(len(pair_keys)),
        "excluded_self_loop_edges": int(self_loop_edges),
        "excluded_missing_direction_pairs": int(missing_pairs),
        "excluded_duplicate_direction_pairs": int(duplicate_pairs),
        "line_nodes": int(pair_count),
        "line_directed_edges": int(line_edge_index.shape[1]),
        "eligible_pairs": int(eligible.sum()),
        "ineligible_context_pairs": int((~eligible).sum()),
        "eligible_class_0": int(np.count_nonzero(y == 0)),
        "eligible_class_1": int(np.count_nonzero(y == 1)),
        "message_passing_pairs": int(pair_message.sum()),
        "moving_pairs": int(motion.sum()),
        "slow_pairs": int(pair_count - motion.sum()),
        "eligible_moving_pairs": int(np.count_nonzero(eligible & (motion == 1))),
        "eligible_slow_pairs": int(np.count_nonzero(eligible & (motion == 0))),
    }
    return {
        "x": x,
        "y": y,
        "eligible": eligible,
        "edge_index": line_edge_index,
        "canonical_pairs": pairs,
        "canonical_pair_track_ids": np.asarray(pair_track_ids, dtype=str).reshape(-1, 2),
        "source_directed_edges": source,
        "motion_regime": motion,
        "message_passing_pair_mask": pair_message,
        "audit_counts": counts,
        "metadata": {
            "bridge_version": BRIDGE_VERSION,
            "feature_source": "stored_observation_edge_attr34_bidirectional_mean",
            "feature_dimension": PAIR_FEATURE_DIM,
            "feature_normalization": "none_in_bridge",
            "future_used_for_features": False,
            "future_used_for_topology": False,
            "label_config": label_config_dict(config),
            "label_config_hash": label_config_hash(config),
            "strict_supervision_required": True,
            "motion_threshold_mps": 0.5,
            "motion_policy": "both_terminal_observations_valid_and_speed_at_least_threshold",
            "line_adjacency": "shared_vehicle_and_both_pairs_message_passing_enabled",
        },
    }


