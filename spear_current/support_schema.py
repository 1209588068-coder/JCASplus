# Source: experiments/spear_semantic_support_v1/schema.py; mechanically migrated; algorithms and state keys unchanged.
"""Source-derived support groups for the unchanged 784-column SPEAR input.

This standard-library-only module has no data/model loading or experiment I/O.
The 30-column importance order is the frozen original contract; the two focal
metadata columns are excluded, leaving 28 legal columns. Semantic grouping joins
only copies of the same cached vehicle statistic, endpoint, time and frame.
Different velocity times, endpoints, sin/cos transforms and directed relations
are never merged. These choices depend on graph_builder formulas, not ASR.
The shuffled-semantic negative control deliberately replaces these identities
with a seeded arbitrary partition of the same sizes; it makes no copy claim.

The six selected triplets have equal raw values, but column normalization can
give them different standardized values. A group binds SUPPORT SELECTION only;
it does not impose equal output values, recompute trajectories, or propagate a
change into another pair row. Group cost is its coordinate count, not one.

Provenance: graph_builder.py:75-130 defines names; :794-859 computes vehicle
features; :862-965 computes directed edges; :1182-1185 caches the four summary
fields as float32. spear_supplied_v1/data.py:36-77 concatenates A, B, AB, BA;
the canonical A/B order is lexicographic track-ID order, not physical role.
"""
from __future__ import annotations

from numbers import Integral
import random


FEATURE_DIM = 784
NODE_DIM = 358
EDGE_DIM = 34
OBS_LEN = 50
SELECTED_COORDINATES = (
    357, 715, 776, 743, 778, 708, 709, 352, 742, 777,
    745, 744, 779, 351, 782, 350, 729, 133, 132, 712,
    781, 353, 749, 746, 503, 134, 724, 135, 760, 179,
)
EXCLUDED_COORDINATES = (357, 715)
LEGAL_COORDINATES = tuple(c for c in SELECTED_COORDINATES if c not in EXCLUDED_COORDINATES)

_BUILDER = 'jcas/workflows/graph_builder.py'
_BRIDGE = 'experiments/spear_supplied_v1/data.py:36-77'
_NORMALIZER = 'experiments/spear_supplied_v1/data.py:110-133'
_REPORTS = ('speed_current', 'speed_mean', 'acc_std', 'lateral_acc_std')
_REPORT_NODE = (350, 351, 353, 354)
_REPORT_SRC = (26, 28, 30, 32)
_REPORT_DST = (27, 29, 31, 33)

# These are literal copies of cached values, not algebraically related fields.
_COPY_COORDINATES = {
    f'{endpoint}.{field}': (
        node_offset + node,
        716 + (src if endpoint == 'A' else dst),
        750 + (dst if endpoint == 'A' else src),
    )
    for endpoint, node_offset in (('A', 0), ('B', 358))
    for field, node, src, dst in zip(_REPORTS, _REPORT_NODE, _REPORT_SRC, _REPORT_DST)
}

_HISTORY = (
    ('rel_pos_x', 'm', 'map axes; origin at this endpoint position t49', 'p[t,0] - p[49,0]'),
    ('rel_pos_y', 'm', 'map axes; origin at this endpoint position t49', 'p[t,1] - p[49,1]'),
    ('vel_x', 'm/s', 'map xy axes', 'v[t,0]'),
    ('vel_y', 'm/s', 'map xy axes', 'v[t,1]'),
    ('heading_sin', '1', 'map heading', 'sin(h[t])'),
    ('heading_cos', '1', 'map heading', 'cos(h[t])'),
    ('observed_valid', '1', 'observation availability', 'float(valid[t])'),
)

_SUMMARIES = (
    ('speed_current', 'm/s', 'rotation-invariant scalar', 't49', 'norm(v[49])'),
    ('speed_mean', 'm/s', 'rotation-invariant scalar', 'valid observation times', 'mean(norm(v[valid]))'),
    ('speed_std', 'm/s', 'rotation-invariant scalar', 'valid observation times', 'std(norm(v[valid]), ddof=0)'),
    ('acc_std', 'm/s^2', 'rotation-invariant scalar', 'successive valid observation times', 'std(norm(diff(v[valid]) / (diff(valid_indices)*0.1)), ddof=0)'),
    ('lateral_acc_std', 'm/s^2', 'later valid sample vehicle lateral direction', 'successive valid observation times', 'std(dot(diff(v[valid])/(diff(valid_indices)*0.1), [-sin(h[valid][1:]), cos(h[valid][1:])]), ddof=0)'),
    ('heading_rate_std', 'rad/s', 'unwrapped map heading difference', 'successive valid observation times', 'std(diff(unwrap(h[valid]))/(diff(valid_indices)*0.1), ddof=0)'),
    ('observed_fraction', '1', 'observation availability', 't00..t49', 'mean(valid.astype(float32))'),
    ('is_focal', '1', 'scenario metadata', 'scenario', 'float(str(track_id) == str(focal_track_id))'),
)

# r=p_dst-p_src; w=v_dst-v_src; d=norm(r); u=r/max(d,1e-6);
# c=-dot(w,u). Sequence statistics use joint valid observation times. A slope
# uses integer timestep indices (0.1 s per step), not seconds.
_EDGE_RELATIONS = (
    ('rel_x_current', 'm', 'map xy axes', 't49', 'r[49,0]'),
    ('rel_y_current', 'm', 'map xy axes', 't49', 'r[49,1]'),
    ('d_current', 'm', 'rotation-invariant pair distance', 't49', 'norm(r[49])'),
    ('rel_vx_current', 'm/s', 'map xy axes', 't49', 'w[49,0]'),
    ('rel_vy_current', 'm/s', 'map xy axes', 't49', 'w[49,1]'),
    ('closing_speed_current', 'm/s', 'directed source-to-destination line of sight', 't49', '-dot(w[49], u[49])'),
    ('d_min_obs', 'm', 'rotation-invariant pair distance', 'joint valid observation times', 'min(d[joint_valid])'),
    ('d_mean_obs', 'm', 'rotation-invariant pair distance', 'joint valid observation times', 'mean(d[joint_valid])'),
    ('d_std_obs', 'm', 'rotation-invariant pair distance', 'joint valid observation times', 'std(d[joint_valid], ddof=0)'),
    ('d_trend', 'm/observation_step', 'rotation-invariant pair distance', 'joint valid observation times', 'safe_slope(t[joint_valid], d[joint_valid])'),
    ('d_trend_early', 'm/observation_step', 'rotation-invariant pair distance', 'joint valid times with t<25', 'safe_slope(t[early], d[early])'),
    ('d_trend_late', 'm/observation_step', 'rotation-invariant pair distance', 'last 20 joint valid times if count>=20; otherwise all joint valid', 'safe_slope(t[late], d[late])'),
    ('closing_speed_mean', 'm/s', 'time-varying source-to-destination line of sight', 'joint valid observation times', 'mean(c[joint_valid])'),
    ('closing_speed_max', 'm/s', 'time-varying source-to-destination line of sight', 'joint valid observation times', 'max(c[joint_valid])'),
    ('closing_speed_std', 'm/s', 'time-varying source-to-destination line of sight', 'joint valid observation times', 'std(c[joint_valid], ddof=0)'),
    ('closing_speed_late', 'm/s', 'time-varying source-to-destination line of sight', 'last 20 joint valid times if count>=20; otherwise all joint valid', 'mean(c[late])'),
    ('closing_speed_trend', '(m/s)/observation_step', 'time-varying source-to-destination line of sight', 'joint valid observation times', 'safe_slope(t[joint_valid], c[joint_valid])'),
    ('heading_diff_sin', '1', 'destination heading relative to source heading', 't49', 'sin(angle_diff(h_dst[49], h_src[49]))'),
    ('heading_diff_cos', '1', 'destination heading relative to source heading', 't49', 'cos(angle_diff(h_dst[49], h_src[49]))'),
    ('src_heading_to_dst_sin', '1', 'source-to-destination bearing relative to source heading', 't49', 'sin(angle_diff(atan2(r[49,1],r[49,0]), h_src[49]))'),
    ('src_heading_to_dst_cos', '1', 'source-to-destination bearing relative to source heading', 't49', 'cos(angle_diff(atan2(r[49,1],r[49,0]), h_src[49]))'),
    ('dst_heading_to_src_sin', '1', 'destination-to-source bearing relative to destination heading', 't49', 'sin(angle_diff(atan2(-r[49,1],-r[49,0]), h_dst[49]))'),
    ('dst_heading_to_src_cos', '1', 'destination-to-source bearing relative to destination heading', 't49', 'cos(angle_diff(atan2(-r[49,1],-r[49,0]), h_dst[49]))'),
    ('src_velocity_toward_dst', 'm/s', 'source-to-destination line of sight', 't49', 'dot(v_src[49], u[49])'),
    ('dst_velocity_toward_src', 'm/s', 'destination-to-source line of sight', 't49', 'dot(v_dst[49], -u[49])'),
    ('speed_delta_src_minus_dst', 'm/s', 'signed source-minus-destination scalar', 't49', 'cached_speed_current[src] - cached_speed_current[dst]'),
)


def _vehicle_description(local_index: int) -> dict:
    if local_index < 350:
        field, unit, frame, formula = _HISTORY[local_index // OBS_LEN]
        time_index = local_index % OBS_LEN
        return dict(source_name=f'{field}_t{time_index:02d}', quantity=field,
                    unit=unit, frame=frame, time=f't{time_index:02d}',
                    formula=formula.replace('[t', f'[{time_index}'),
                    source=f'{_BUILDER}:794-859')
    name, unit, frame, time, formula = _SUMMARIES[local_index - 350]
    return dict(source_name=name, quantity=name, unit=unit, frame=frame,
                time=time, formula=formula, source=f'{_BUILDER}:794-859')


def coordinate_metadata(index: int) -> dict:
    """Describe any original coordinate, returning owned JSON-friendly metadata.

    Raw node inputs p/v/h are linearly filled observation histories; heading is
    unwrapped before filling and wrapped again afterward. Valid masks retain
    original availability. Empty vehicle statistics and insufficient-sample
    slopes return zero. The metadata describes raw physical units; model values
    use each column's frozen TRAIN mean/scale and clipping to [-10,10].
    """
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < FEATURE_DIM:
        raise ValueError('index must be an integer in [0, 784)')
    index = int(index)
    if index < 716:
        endpoint = 'A' if index < NODE_DIM else 'B'
        local_index = index % NODE_DIM
        result = _vehicle_description(local_index)
        result.update(block=f'node_{endpoint}', endpoint=endpoint,
                      source_endpoint=None, destination_endpoint=None)
    else:
        offset, source_endpoint, destination_endpoint = ((716, 'A', 'B') if index < 750 else (750, 'B', 'A'))
        local_index = index - offset
        if local_index >= 26:
            field_index = (local_index - 26) // 2
            is_source = local_index % 2 == 0
            endpoint = source_endpoint if is_source else destination_endpoint
            result = _vehicle_description(_REPORT_NODE[field_index])
            result['source_name'] = ('src_' if is_source else 'dst_') + _REPORTS[field_index]
            result['formula'] = f'cached {endpoint}.{_REPORTS[field_index]} (exact node summary copy)'
            result['source'] = f'{_BUILDER}:954-962; {_BUILDER}:1182-1185'
        else:
            endpoint = None
            name, unit, frame, time, formula = _EDGE_RELATIONS[local_index]
            result = dict(source_name=name, quantity=name, unit=unit, frame=frame,
                          time=time, formula=formula, source=f'{_BUILDER}:862-965')
        result.update(block=f'edge_{source_endpoint}{destination_endpoint}',
                      endpoint=endpoint, source_endpoint=source_endpoint,
                      destination_endpoint=destination_endpoint)
    copy_key = f'{endpoint}.{result["quantity"]}' if endpoint is not None else None
    if copy_key not in _COPY_COORDINATES:
        copy_key = None
    result.update(index=index, local_index=local_index,
                  name=f'{result["block"]}.{result["source_name"]}',
                  copy_group=copy_key,
                  copy_coordinates=list(_COPY_COORDINATES[copy_key]) if copy_key else [index],
                  legal_support=index in LEGAL_COORDINATES,
                  selected_rank=SELECTED_COORDINATES.index(index) if index in SELECTED_COORDINATES else None,
                  exclusion_reason=('focal_metadata' if index in EXCLUDED_COORDINATES else
                                    None if index in LEGAL_COORDINATES else 'outside_frozen_support'),
                  bridge_source=_BRIDGE, normalizer_source=_NORMALIZER,
                  normalized_units='frozen TRAIN standardized column; clipped to [-10, 10]',
                  preprocessing_source=f'{_BUILDER}:349-365; {_BUILDER}:684-785')
    if result['block'].startswith('edge_') and 6 <= local_index <= 16:
        result['insufficient_observations'] = ('d_current' if local_index in (6, 7) else '0') + ' when fewer than 2 jointly valid observations'
    return result


def support_groups(mode: str = 'semantic', seed: int = 20260621) -> tuple[tuple[int, ...], ...]:
    """Partition legal28 into source-derived groups or registered controls.

    Semantic groups are six exact-copy triplets and ten singleton fields. The
    coordinate control has 28 singleton groups over the SAME legal support.
    Both retain first-occurrence group order. ``shuffled_semantic`` is an
    arbitrary-group negative control with exactly the semantic group-size
    sequence: one private seeded permutation is split at those sizes, with no
    redraw even if it reproduces the semantic partition. Its group positions
    follow that split order. All modes preserve original importance order
    WITHIN each group. The seed is used only for ``shuffled_semantic``.

    Shuffled groups do not assert common endpoint, time, frame or raw value;
    coordinate_metadata continues to report only actual source copy identity.
    """
    if mode not in ('semantic', 'coordinate', 'shuffled_semantic'):
        raise ValueError("mode must be 'semantic', 'coordinate' or 'shuffled_semantic'")
    groups: dict[str, list[int]] = {}
    for coordinate in LEGAL_COORDINATES:
        metadata = coordinate_metadata(coordinate)
        key = metadata['copy_group'] if mode != 'coordinate' else None
        groups.setdefault(key or f'coordinate:{coordinate}', []).append(coordinate)
    partition = tuple(tuple(coordinates) for coordinates in groups.values())
    if mode != 'shuffled_semantic':
        return partition
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError('shuffled_semantic seed must be an integer')
    shuffled = list(LEGAL_COORDINATES)
    random.Random(int(seed)).shuffle(shuffled)
    rank = {coordinate: order for order, coordinate in enumerate(LEGAL_COORDINATES)}
    shuffled_groups = []
    offset = 0
    for group in partition:
        stop = offset + len(group)
        shuffled_groups.append(tuple(sorted(shuffled[offset:stop], key=rank.__getitem__)))
        offset = stop
    return tuple(shuffled_groups)
