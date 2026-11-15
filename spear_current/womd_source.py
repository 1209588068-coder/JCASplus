"""Read WOMD Scenario TFRecords without resampling or inferring absent fields.

The protobuf definitions are the official Waymo definitions (external dependency,
Apache-2.0). This adapter uses one fixed 110-sample window per 20 s source clip.
It does not create splits, task labels, features, or train any model.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class TFRecord:
    index: int
    byte_offset: int
    payload: bytes


class ScenarioRejected(ValueError):
    """A documented data exclusion, distinct from an implementation failure."""

    def __init__(self, reason: str, scenario_id: str = "", **details: Any):
        self.reason = reason
        self.scenario_id = scenario_id
        self.details = details
        super().__init__(f"{scenario_id or '<unknown>'}: {reason}: {details}")

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "scenario_id": self.scenario_id,
                "details": self.details}


def load_scenario_class(proto_root: str | Path):
    """Load externally pinned official protobufs; caller records their hashes."""
    root = Path(proto_root).resolve()
    if not (root / "waymo_open_dataset/protos/scenario_pb2.py").is_file():
        raise FileNotFoundError(f"Official compiled Scenario protobuf missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("waymo_open_dataset.protos.scenario_pb2")
    if Path(module.__file__).resolve() != root / "waymo_open_dataset/protos/scenario_pb2.py":
        raise RuntimeError("A different Scenario protobuf was already imported; use a fresh process")
    return module.Scenario


def _masked_crc32c(data: bytes) -> int:
    import google_crc32c
    crc = google_crc32c.value(data)
    return (((crc >> 15) | ((crc << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecords(path: str | Path) -> Iterator[TFRecord]:
    """Stream records and validate both CRC32C fields before yielding payloads."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        index = 0
        while True:
            offset = stream.tell()
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header: {path}:{offset}")
            length, checksum = struct.unpack("<QI", header)
            if checksum != _masked_crc32c(header[:8]):
                raise ValueError(f"TFRecord length CRC32C mismatch: {path}:{offset}")
            if length > file_size - stream.tell() - 4:
                raise ValueError(f"Truncated TFRecord payload: {path}:{offset}")
            payload = stream.read(length)
            footer = stream.read(4)
            if len(footer) != 4 or struct.unpack("<I", footer)[0] != _masked_crc32c(payload):
                raise ValueError(f"TFRecord payload CRC32C mismatch: {path}:{offset}")
            yield TFRecord(index, offset, payload)
            index += 1


def _state_values(state, fields: tuple[str, ...], sid: str, track_id: int,
                  source_timestep: int) -> tuple[float, ...]:
    values = []
    for field in fields:
        if not state.HasField(field):
            raise ScenarioRejected("valid_state_missing_field", sid,
                                   track_id=track_id, timestep=source_timestep, field=field)
        value = float(getattr(state, field))
        if not math.isfinite(value):
            raise ScenarioRejected("valid_state_nonfinite_field", sid,
                                   track_id=track_id, timestep=source_timestep, field=field)
        values.append(value)
    return tuple(values)


def _lanes(scenario) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Map explicit lane geometry/topology, retaining unknown intersection state.

    WOMD left/right neighbors can be segment-specific. The legacy JCAS lane
    relation is lane-level, so all listed lateral neighbor IDs are unioned.
    This is a declared coarsening, not a current-position lane-change test.
    """
    sid = scenario.scenario_id
    seen = set()
    lanes: dict[int, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    hasher = hashlib.sha256()
    for feature in sorted(scenario.map_features, key=lambda value: value.id):
        if not feature.HasField("id") or feature.id in seen:
            raise ScenarioRejected("missing_or_duplicate_map_id", sid, map_id=int(feature.id))
        seen.add(feature.id)
        # Hash all map features, not only lanes retained for vehicle matching.
        encoded = feature.SerializeToString(deterministic=True)
        hasher.update(struct.pack("<Q", len(encoded)))
        hasher.update(encoded)
        kind = feature.WhichOneof("feature_data") or "unset"
        counts[kind] += 1
        if kind != "lane":
            continue
        lane = feature.lane
        # Explicit FREEWAY / SURFACE_STREET only; BIKE and UNDEFINED are not
        # relabelled as known vehicle lanes.
        if lane.type not in (1, 2):
            counts[f"lane_excluded_type_{int(lane.type)}"] += 1
            continue
        if len(lane.polyline) < 2:
            counts["lane_excluded_short_polyline"] += 1
            continue
        if any(not p.HasField("x") or not p.HasField("y") for p in lane.polyline):
            counts["lane_excluded_missing_xy"] += 1
            continue
        points = np.array([(p.x, p.y) for p in lane.polyline], dtype=np.float64)
        if not np.isfinite(points).all():
            counts["lane_excluded_nonfinite_xy"] += 1
            continue
        points = points.astype(np.float32)
        delta = np.diff(points, axis=0)
        headings = np.empty(len(points), dtype=np.float32)
        headings[:-1] = np.arctan2(delta[:, 1], delta[:, 0])
        headings[-1] = headings[-2]
        lateral = {int(x.feature_id) for x in (*lane.left_neighbors, *lane.right_neighbors)
                   if x.HasField("feature_id")}
        lanes[int(feature.id)] = dict(
            lane_id=int(feature.id), points=points, headings=headings,
            is_intersection=None, lateral_neighbors=lateral,
            topology_neighbors={int(x) for x in (*lane.entry_lanes, *lane.exit_lanes)},
        )
    return lanes, {
        "map_feature_counts": dict(counts), "vehicle_lane_count": len(lanes),
        "map_fingerprint_sha256": hasher.hexdigest(),
        "intersection_source": "unavailable_in_scenario_lane_proto",
        "intersection_known": False,
        "lateral_neighbor_policy": "union_of_all_declared_lane_neighbor_segments",
        "lane_topology_source": "entry_lanes_and_exit_lanes",
        "lane_interpolating_not_used_as_intersection": True,
    }


def decode_scenario(scenario, *, window_start: int = 0,
                    interval_tolerance_seconds: float = 0.001,
                    grid_tolerance_seconds: float = 0.005) -> dict[str, Any]:
    """Decode fixed first-110 samples into the original 50/60 array contract.

    Only the preregistered window_start=0 is supported. A rejected window is
    never replaced by another window. Invalid states remain NaN/False for the
    historical observation-only fill routine in vehicle_graph_core.
    """
    if window_start != 0:
        raise ValueError("This data contract fixes window_start=0; no alternate-window search")
    if (not math.isfinite(interval_tolerance_seconds) or interval_tolerance_seconds < 0
            or not math.isfinite(grid_tolerance_seconds) or grid_tolerance_seconds < 0):
        raise ValueError("Timestamp tolerances must be finite and nonnegative")
    sid = str(scenario.scenario_id)
    if not sid:
        raise ScenarioRejected("empty_scenario_id")
    timestamps = np.asarray(scenario.timestamps_seconds, dtype=np.float64)
    if len(timestamps) < 110:
        raise ScenarioRejected("source_timeline_too_short", sid, steps=len(timestamps))
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ScenarioRejected("source_timestamps_nonfinite_or_nonincreasing", sid)
    ids = [int(track.id) for track in scenario.tracks]
    if (any(not track.HasField("id") for track in scenario.tracks)
            or len(set(ids)) != len(ids)):
        raise ScenarioRejected("missing_or_duplicate_track_id", sid)
    if any(track_id < 0 for track_id in ids):
        raise ScenarioRejected("negative_track_id", sid)
    mismatch = [(int(track.id), len(track.states)) for track in scenario.tracks
                if len(track.states) != len(timestamps)]
    if mismatch:
        raise ScenarioRejected("track_timeline_length_mismatch", sid,
                               timestamp_count=len(timestamps), mismatch_count=len(mismatch),
                               mismatch_examples=mismatch[:8])
    if len(scenario.dynamic_map_states) not in (0, len(timestamps)):
        raise ScenarioRejected("dynamic_map_timeline_length_mismatch", sid,
                               timestamp_count=len(timestamps),
                               dynamic_map_count=len(scenario.dynamic_map_states))
    selected_ts = timestamps[:110]
    intervals = np.diff(selected_ts)
    interval_error = float(np.max(np.abs(intervals - 0.1)))
    grid_error = float(np.max(np.abs(selected_ts - selected_ts[0] - np.arange(110) * 0.1)))
    if interval_error > interval_tolerance_seconds:
        raise ScenarioRejected("window_irregular_timestep", sid,
                               maximum_interval_error_seconds=interval_error,
                               tolerance_seconds=interval_tolerance_seconds)
    if grid_error > grid_tolerance_seconds:
        raise ScenarioRejected("window_irregular_time_grid", sid,
                               maximum_grid_error_seconds=grid_error,
                               tolerance_seconds=grid_tolerance_seconds)
    if (not scenario.HasField("sdc_track_index")
            or not 0 <= scenario.sdc_track_index < len(scenario.tracks)):
        raise ScenarioRejected("missing_or_invalid_sdc_track_index", sid)
    sdc_id = ids[scenario.sdc_track_index]
    tracks = []
    type_counts = Counter(int(track.object_type) for track in scenario.tracks)
    for track in scenario.tracks:
        if track.object_type != 1:  # Official Track.TYPE_VEHICLE.
            continue
        obs_pos = np.full((50, 2), np.nan, dtype=np.float64)
        obs_vel = np.full((50, 2), np.nan, dtype=np.float64)
        obs_heading = np.full(50, np.nan, dtype=np.float64)
        obs_valid = np.zeros(50, dtype=bool)
        fut_pos = np.full((60, 2), np.nan, dtype=np.float64)
        fut_valid = np.zeros(60, dtype=bool)
        for t, state in enumerate(track.states[:110]):
            if not state.valid:
                continue
            if t < 50:
                px, py, vx, vy, heading = _state_values(
                    state, ("center_x", "center_y", "velocity_x", "velocity_y", "heading"),
                    sid, int(track.id), t)
                obs_pos[t] = px, py
                obs_vel[t] = vx, vy
                obs_heading[t] = heading
                obs_valid[t] = True
            else:
                fut_pos[t - 50] = _state_values(state, ("center_x", "center_y"), sid, int(track.id), t)
                fut_valid[t - 50] = True
        tracks.append(dict(track_id=str(track.id), object_type="vehicle", object_category=-1,
                           is_focal=False, obs_pos=obs_pos, obs_vel=obs_vel,
                           obs_heading=obs_heading, obs_valid=obs_valid,
                           fut_pos=fut_pos, fut_valid=fut_valid))
    # Lexicographic string ordering is explicit, matching JCAS pair identity.
    tracks.sort(key=lambda track: track["track_id"])
    lanes, map_metadata = _lanes(scenario)
    sdc = scenario.tracks[scenario.sdc_track_index]
    sdc_position = np.full((len(timestamps), 2), np.nan, dtype=np.float64)
    sdc_valid = np.zeros(len(timestamps), dtype=bool)
    for t, state in enumerate(sdc.states):
        if state.valid and state.HasField("center_x") and state.HasField("center_y"):
            xy = (float(state.center_x), float(state.center_y))
            if all(math.isfinite(v) for v in xy):
                sdc_position[t] = xy
                sdc_valid[t] = True
    metadata = {
        "scenario_id": sid, "source_timestep_count": len(timestamps),
        "source_track_count": len(scenario.tracks), "source_object_type_counts": dict(type_counts),
        "vehicle_tracks_before_jcas_filter": len(tracks),
        "window_start": 0, "observation_steps": 50, "future_steps": 60,
        "observation_end_source_index": 49, "future_end_source_index": 109,
        "source_current_time_index": (int(scenario.current_time_index)
                                      if scenario.HasField("current_time_index") else None),
        "current_time_policy": "custom_observation_cutoff_49_not_source_current_time",
        "timestamp_policy": "fixed_native_samples_no_resampling_reject_nonuniform_window",
        "nominal_dt_seconds": 0.1, "interval_tolerance_seconds": interval_tolerance_seconds,
        "grid_tolerance_seconds": grid_tolerance_seconds,
        "maximum_interval_error_seconds": interval_error, "maximum_grid_error_seconds": grid_error,
        "source_window_duration_seconds": float(selected_ts[-1] - selected_ts[0]),
        "source_sdc_track_id": str(sdc_id), "focal_policy": "unavailable_zero_placeholder",
        "focal_known": False,
        "object_category_source": "unavailable_sentinel_minus_one",
        "vehicle_type_source": "womd_TYPE_VEHICLE_no_separate_bus_category",
        "uses_tracks_to_predict_for_selection": False, "uses_objects_of_interest_for_selection": False,
        "recording_group_identity_available": False,
        **map_metadata,
    }
    return {
        "tracks": tracks, "lanes": lanes, "metadata": metadata,
        "source_arrays": {
            "source_timestamps_seconds": timestamps.copy(),
            "window_timestamps_seconds": selected_ts.copy(),
            "source_sdc_position": sdc_position, "source_sdc_valid": sdc_valid,
            "source_sdc_track_id": np.asarray(str(sdc_id)),
            "source_vehicle_track_ids": np.asarray([track["track_id"] for track in tracks], dtype=np.str_),
            "source_vehicle_is_sdc": np.asarray([track["track_id"] == str(sdc_id) for track in tracks], dtype=bool),
        },
    }
