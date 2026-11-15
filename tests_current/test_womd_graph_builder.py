"""Synthetic WOMD transport, time, feature and label-contract regression checks.

Requires the pinned official Scenario protobuf and google-crc32c dependency;
does not load raw WOMD/AV2 data, train models, or write persistent graph assets.
"""
from copy import deepcopy
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from spear_current.vehicle_graph_core import BuildConfig, LaneData, lane_pair_features
from spear_current.womd_source import (
    ScenarioRejected, _masked_crc32c, decode_scenario, iter_tfrecords, load_scenario_class,
)
from spear_current.womd_graph_builder import build_decoded, check_vehicle_pair_graph


PROTO_ROOT = Path('/home/mizhou/tools/womd_proto_99a4cb3ff07e')


class WOMDGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.Scenario = load_scenario_class(PROTO_ROOT)
            _masked_crc32c(b'dependency check')
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            raise unittest.SkipTest(f'WOMD official protobuf/CRC dependency unavailable: {exc}')

    def fixture(self):
        scenario = self.Scenario(scenario_id='synthetic-contract', sdc_track_index=0)
        scenario.timestamps_seconds.extend(np.arange(120) * .1)
        for track_id in (0, 1):
            track = scenario.tracks.add(id=track_id, object_type=1)
            for t in range(120):
                track.states.add(center_x=t * .1 + track_id * 10, center_y=0,
                                 heading=0, velocity_x=1, velocity_y=0, valid=True)
        return scenario

    def assert_rejected(self, scenario, expected_reason):
        with self.assertRaises(ScenarioRejected) as caught:
            decode_scenario(scenario)
        self.assertEqual(caught.exception.reason, expected_reason)

    def test_future_positions_change_labels_but_never_features_or_topology(self):
        original = self.fixture()
        changed = deepcopy(original)
        for t in range(50, 110):
            changed.tracks[1].states[t].center_x = changed.tracks[0].states[t].center_x + 1
        ga, pa, _ = build_decoded(decode_scenario(original))
        gb, pb, _ = build_decoded(decode_scenario(changed))
        self.assertTrue(np.all(ga['edge_label_dynamic_risk'] == 0))
        self.assertTrue(np.all(gb['edge_label_dynamic_risk'] == 1))
        for field in ('x_node', 'edge_attr', 'edge_index', 'message_passing_edge_mask',
                      'task_candidate_mask', 'observed_valid_mask'):
            np.testing.assert_array_equal(ga[field], gb[field], err_msg=field)
        for field in ('x', 'edge_index', 'canonical_pairs', 'source_directed_edges'):
            np.testing.assert_array_equal(pa[field], pb[field], err_msg=field)
        self.assertEqual(pa['x'].shape, (1, 784))
        self.assertEqual(int(pa['y'][0]), 0)
        self.assertEqual(int(pb['y'][0]), 1)
        check_vehicle_pair_graph(ga, pa)
        check_vehicle_pair_graph(gb, pb)

    def test_focal_is_unknown_and_not_replaced_by_sdc_or_prediction_role(self):
        scenario = self.fixture()
        scenario.tracks_to_predict.add(track_index=1)
        scenario.objects_of_interest.append(1)
        decoded = decode_scenario(scenario)
        graph, pair, _ = build_decoded(decoded)
        self.assertFalse(decoded['metadata']['focal_known'])
        self.assertEqual(decoded['metadata']['focal_policy'], 'unavailable_zero_placeholder')
        self.assertFalse(graph['node_is_focal'].any())
        self.assertFalse(graph['node_is_focal_known'].any())
        np.testing.assert_array_equal(graph['node_is_sdc'], [True, False])
        np.testing.assert_array_equal(pair['x'][:, [357, 715]], 0)
        self.assertTrue(np.all(graph['node_object_category'] == -1))

    def test_missing_last_observation_fills_only_from_observation(self):
        a = self.fixture()
        for track in a.tracks:
            track.states[49].valid = False
            track.states[20].valid = False
        b = deepcopy(a)
        for track in b.tracks:
            for state in track.states[50:]:
                state.center_x = 100000
        da, db = decode_scenario(a), decode_scenario(b)
        self.assertTrue(np.isnan(da['tracks'][0]['obs_pos'][49]).all())
        ga, pa, _ = build_decoded(da)
        gb, pb, _ = build_decoded(db)
        for name in ('observed_positions_filled', 'observed_velocities_filled',
                     'observed_headings_filled', 'x_node', 'edge_attr'):
            np.testing.assert_array_equal(ga[name], gb[name], err_msg=name)
        np.testing.assert_array_equal(ga['observed_positions_filled'][:, 49],
                                      ga['observed_positions_filled'][:, 48])
        np.testing.assert_array_equal(pa['x'], pb['x'])
        self.assertFalse(ga['observed_valid_mask'][:, 49].any())

    def test_lane_interpolating_is_not_intersection_and_unknown_cannot_hard_prune(self):
        scenario = self.fixture()
        for lane_id in (10, 20):
            feature = scenario.map_features.add(id=lane_id)
            feature.lane.type = 2
            feature.lane.interpolating = True
            feature.lane.polyline.add(x=0, y=0)
            feature.lane.polyline.add(x=10, y=0)
        decoded = decode_scenario(scenario)
        self.assertTrue(all(v['is_intersection'] is None for v in decoded['lanes'].values()))
        lanes = {key: LaneData(**value) for key, value in decoded['lanes'].items()}
        args = (0, 1, 20., np.array([10, 20]), np.zeros(2),
                np.array([0., np.pi / 2]))
        unknown = lane_pair_features(*args, np.array([-1, -1]), lanes, BuildConfig())
        known_false = lane_pair_features(*args, np.array([0, 0]), lanes, BuildConfig())
        self.assertTrue(unknown[5])
        self.assertFalse(unknown[6])
        self.assertTrue(unknown[7])
        self.assertFalse(known_false[5])

    def test_state_timeline_mismatch_is_rejected_without_trimming(self):
        scenario = self.fixture()
        scenario.tracks[0].states.add(valid=False)
        self.assert_rejected(scenario, 'track_timeline_length_mismatch')

    def test_per_interval_and_accumulated_time_errors_are_distinct(self):
        gap = self.fixture()
        gap.timestamps_seconds[80] += .01
        self.assert_rejected(gap, 'window_irregular_timestep')
        drift = self.fixture()
        drift.timestamps_seconds[:] = np.arange(120) * .1001
        self.assert_rejected(drift, 'window_irregular_time_grid')
        nonincreasing = self.fixture()
        nonincreasing.timestamps_seconds[2] = nonincreasing.timestamps_seconds[1]
        self.assert_rejected(nonincreasing, 'source_timestamps_nonfinite_or_nonincreasing')

    def test_never_searches_another_window_and_checks_only_selected_time_grid(self):
        scenario = self.fixture()
        with self.assertRaises(ValueError):
            decode_scenario(scenario, window_start=1)
        # Irregular sampling beyond the fixed label window is not falsely
        # claimed to be regular, and cannot change this window's model input.
        original = decode_scenario(scenario)
        scenario.timestamps_seconds[115] += .01
        changed = decode_scenario(scenario)
        np.testing.assert_array_equal(original['source_arrays']['window_timestamps_seconds'],
                                      changed['source_arrays']['window_timestamps_seconds'])

    def test_missing_valid_field_rejected_but_invalid_padding_ignored(self):
        invalid = self.fixture()
        invalid.tracks[0].states[10].valid = False
        invalid.tracks[0].states[10].ClearField('heading')
        decoded = decode_scenario(invalid)
        self.assertFalse(decoded['tracks'][0]['obs_valid'][10])
        invalid.tracks[0].states[10].valid = True
        self.assert_rejected(invalid, 'valid_state_missing_field')

    def test_crc_transport_checks_both_header_and_payload(self):
        payload = self.fixture().SerializeToString()
        length = struct.pack('<Q', len(payload))
        encoded = (length + struct.pack('<I', _masked_crc32c(length)) + payload
                   + struct.pack('<I', _masked_crc32c(payload)))
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / 'test.tfrecord'
            path.write_bytes(encoded + encoded)
            records = list(iter_tfrecords(path))
            self.assertEqual([r.index for r in records], [0, 1])
            self.assertEqual([r.byte_offset for r in records], [0, len(encoded)])
            self.assertTrue(all(r.payload == payload for r in records))
            for position in (8, 12):
                broken = bytearray(encoded)
                broken[position] ^= 1
                path.write_bytes(broken)
                with self.assertRaisesRegex(ValueError, 'CRC32C mismatch'):
                    list(iter_tfrecords(path))
            path.write_bytes(encoded[:-1])
            with self.assertRaisesRegex(ValueError, 'Truncated TFRecord'):
                list(iter_tfrecords(path))

    def test_bidirectional_pair_mapping_tamper_is_detected(self):
        graph, pair, _ = build_decoded(decode_scenario(self.fixture()))
        bad = deepcopy(pair)
        bad['canonical_pairs'][0] = bad['canonical_pairs'][0, ::-1]
        with self.assertRaisesRegex(ValueError, 'Pair identity mapping changed'):
            check_vehicle_pair_graph(graph, bad)


if __name__ == '__main__':
    unittest.main()
