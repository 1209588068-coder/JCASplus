"""Synthetic confirmation metadata/reader tests; no retained AV2 graph access."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from spear_current import confirmation_protocol as cp
from spear_current.runtime import Bundle, sha256


def row(scene, group, split='train', **extra):
    return dict(scenario_id=scene, split_group_id=group, split=split,
                graph_path=f'graphs/{scene}.npz', graph_sha256='a' * 64, **extra)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


def metadata_fixture(directory):
    root = Path(directory)
    graph_root = root / 'graph_root'
    (graph_root / 'graphs').mkdir(parents=True)
    historical, roles, number = [], {}, 0
    for role, count in (('train', 512), ('dev', 128), ('audit', 256)):
        roles[role] = []
        for _ in range(count):
            item = row(f'historical-{number}', f'historical-group-{number}')
            historical.append(item)
            roles[role].append(dict(item, role=role))
            number += 1
    candidates = [row(f'new-{group}-{scene}', f'new-group-{group}')
                  for group in range(4) for scene in range(2)]
    for item in candidates:
        path = graph_root / item['graph_path']
        np.savez(path, scenario_id=np.array(item['scenario_id']), future_mock=np.array([3.]))
        item['graph_sha256'] = sha256(path)
    manifest_path = graph_root / 'manifest_grouped_v5.csv'
    with manifest_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=cp.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows([*historical, *candidates, row('val-scene', 'val-group', 'val'),
                          row('test-scene', 'test-group', 'test')])
    manifest_ref = dict(path=str(manifest_path), sha256=sha256(manifest_path))
    manifest = dict(graph_root=str(graph_root), graph_manifest=manifest_ref,
        seeds={str(seed): {} for seed in cp.SEED_LIST}, config=dict(
            roles=roles, graph_dir=str(graph_root), label_config={},
            references=dict(graph_manifest=manifest_ref)))
    bundle_path = root / 'bundle.json'
    write_json(bundle_path, manifest)
    source = root / 'old_plan.json'
    write_json(source, {'known_scene': 'historical-0'})
    history = dict(status='PASS_KNOWN_SPEAR_HISTORY', excluded_group_ids=[],
        excluded_scenario_ids=[], unresolved_references=[], sources=[
            dict(path=str(bundle_path), sha256=sha256(bundle_path), kind='current_role_manifest'),
            dict(manifest_ref, kind='authoritative_grouped_graph_manifest'),
            dict(path=str(source), sha256=sha256(source), kind='old_plan')])
    history_path = root / 'history.json'
    write_json(history_path, history)
    return Bundle(bundle_path, manifest), history_path, history, candidates


class SelectionTests(unittest.TestCase):
    def test_order_independence_fixed_hash_order_and_one_scene_per_group(self):
        rows = [row(f'scene-{g}-{s}', f'group-{g}') for g in range(7) for s in range(3)]
        result = cp.select_rows(rows, [], group_count=4)
        self.assertEqual(result, cp.select_rows(rows[::-1], [], group_count=4))
        rank = lambda kind, identity: hashlib.sha256(
            ('20261027:spear-current-confirmation-v1:' + kind + ':' + identity).encode()).hexdigest()
        groups = sorted({item['split_group_id'] for item in rows}, key=lambda g: (rank('group', g), g))[:4]
        self.assertEqual([item['split_group_id'] for item in result], groups)
        for item in result:
            candidates = [r['scenario_id'] for r in rows if r['split_group_id'] == item['split_group_id']]
            self.assertEqual(item['scenario_id'], min(candidates, key=lambda s: (rank('scene', s), s)))
            self.assertEqual(set(item), cp.ROW_FIELDS)

    def test_excludes_whole_groups_other_splits_and_rows_without_graph_binding(self):
        rows = [row('a', 'excluded'), row('b', 'excluded'), row('c', 'keep'),
                row('d', 'val-group', 'val'), row('e', 'test-group', 'test'), row('f', 'missing')]
        rows[-1]['graph_sha256'] = ''
        self.assertEqual([r['scenario_id'] for r in cp.select_rows(rows, ['excluded'], 1)], ['c'])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            cp.select_rows(rows, ['excluded'], 2)

    def test_labels_motion_future_and_other_statistics_never_affect_selection(self):
        rows = [row(f's-{i}', f'g-{i}') for i in range(5)]
        expected = cp.select_rows(rows, [], 3)
        for item in rows:
            item.update(label=object(), motion_regime=object(), mean_future_min_distance=object(),
                        num_risk_positive_edges=object(), positive_rate=object(), status=object())
        self.assertEqual(cp.select_rows(rows, [], 3), expected)

    def test_exact_duplicate_metadata_is_deduplicated_but_conflicting_id_is_rejected(self):
        item = row('a', 'group-a')
        self.assertEqual(cp.select_rows([item, dict(item, future_stat=999)], [], 1),
                         cp.select_rows([item], [], 1))
        for key, value in (('split_group_id', 'other'), ('split', 'val'),
                           ('graph_sha256', 'b' * 64), ('graph_path', 'graphs/other.npz')):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                cp.select_rows([item, dict(item, **{key: value})], [], 1)

    def test_group_split_conflicts_malformed_hash_and_unsafe_path_stop_selection(self):
        with self.assertRaisesRegex(ValueError, 'conflicting original splits'):
            cp.select_rows([row('a', 'same'), row('b', 'same', 'val')], [], 1)
        for value in ('not-a-hash', 'a' * 63):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                cp.select_rows([dict(row('a', 'g'), graph_sha256=value)], [], 1)
        for path in ('../outside.npz', '/graphs/a.npz', 'graphs/../a.npz', 'data/a.npz'):
            with self.assertRaisesRegex(ValueError, 'relative graphs'):
                cp.select_rows([dict(row('a', 'g'), graph_path=path)], [], 1)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        cp._VALIDATED.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle, self.history_path, self.history, self.candidates = metadata_fixture(self.temp.name)

    def prepare(self):
        return cp.prepare_protocol(self.bundle, self.history_path, group_count=2)

    def test_preparation_and_revalidation_are_metadata_only_and_exclude_all_896_groups(self):
        with patch.object(cp.np, 'load', side_effect=AssertionError('must not read graphs')) as load:
            protocol = self.prepare()
            self.assertEqual(cp.validate_protocol(self.bundle, protocol), protocol)
        load.assert_not_called()
        self.assertEqual(len(protocol['excluded_group_ids']), 896)
        self.assertEqual(protocol['seed_list'], [20260621, 20260622, 20260623])
        self.assertEqual(protocol['target_seed'], 20260621)
        self.assertEqual(protocol['group_count'], 2)
        self.assertEqual(len({r['split_group_id'] for r in protocol['rows']}), 2)
        self.assertTrue(all(r['split'] == 'train' and r['role'] == 'confirmation' for r in protocol['rows']))

    def test_historical_scene_excludes_its_entire_group(self):
        self.history['excluded_scenario_ids'] = [self.candidates[0]['scenario_id']]
        write_json(self.history_path, self.history)
        protocol = self.prepare()
        group = self.candidates[0]['split_group_id']
        self.assertIn(group, protocol['excluded_group_ids'])
        self.assertTrue(all(r['split_group_id'] != group for r in protocol['rows']))

    def test_unresolved_history_unknown_ids_and_unbound_manifest_stop_preparation(self):
        for change in (dict(unresolved_references=['missing']), dict(status='INCOMPLETE'),
                       dict(excluded_scenario_ids=['unknown-scene']), dict(excluded_group_ids=['unknown-group']),
                       dict(sources=self.history['sources'][1:])):
            with self.subTest(change=list(change)), patch.object(cp.np, 'load') as load:
                write_json(self.history_path, dict(self.history, **change))
                with self.assertRaises(ValueError):
                    self.prepare()
                load.assert_not_called()

    def test_manifest_and_every_history_source_hash_are_checked(self):
        source = Path(self.history['sources'][-1]['path'])
        source.write_text('{"changed": true}')
        with patch.object(cp.np, 'load') as load, self.assertRaisesRegex(ValueError, 'hash changed'):
            self.prepare()
        load.assert_not_called()

    def test_cached_receipt_cannot_hide_changed_protocol_bundle_or_source(self):
        protocol = self.prepare()
        selected = protocol['rows'][0]
        variants = [dict(protocol, target_seed=1), dict(protocol, excluded_group_ids=[]),
                    dict(protocol, rows=list(reversed(protocol['rows'])))]
        for changed in variants:
            with patch.object(cp.np, 'load') as load, self.assertRaises(ValueError):
                cp.read_confirmation_part(self.bundle, changed, selected, {}, [])
            load.assert_not_called()
        self.bundle.manifest['config']['label_config'] = {'unregistered': True}
        with patch.object(cp.np, 'load') as load, self.assertRaisesRegex(ValueError, 'In-memory bundle'):
            cp.read_confirmation_part(self.bundle, protocol, selected, {}, [])
        load.assert_not_called()
        self.bundle.manifest['config']['label_config'] = {}
        protocol = self.prepare()
        Path(self.history['sources'][-1]['path']).write_text('{"changed": true}')
        with patch.object(cp.np, 'load') as load, self.assertRaisesRegex(ValueError, 'hash changed'):
            cp.read_confirmation_part(self.bundle, protocol, protocol['rows'][0], {}, [])
        load.assert_not_called()

    def test_unauthorized_row_is_rejected_before_numpy_load(self):
        protocol = self.prepare()
        selected = protocol['rows'][0]
        variants = [dict(selected, split='val'), dict(selected, split='test'),
                    dict(selected, role='audit'), dict(selected, scenario_id='unknown'),
                    dict(selected, split_group_id='historical-group-0'),
                    dict(selected, graph_path='../escape.npz'), dict(selected, extra='unexpected')]
        for changed in variants:
            with self.subTest(row=changed), patch.object(cp.np, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'exact, nonexcluded'):
                    cp.read_confirmation_part(self.bundle, protocol, changed, {}, [])
                load.assert_not_called()

    def test_graph_hash_and_resolved_path_are_checked_before_numpy_load(self):
        protocol = self.prepare()
        first, second = protocol['rows']
        path = Path(protocol['graph_root']) / first['graph_path']
        path.write_bytes(path.read_bytes() + b'changed')
        with patch.object(cp.np, 'load') as load, self.assertRaisesRegex(ValueError, 'hash mismatch'):
            cp.read_confirmation_part(self.bundle, protocol, first, {}, [])
        load.assert_not_called()
        path = Path(protocol['graph_root']) / second['graph_path']
        outside = Path(self.temp.name) / 'outside.npz'
        path.rename(outside)
        path.symlink_to(outside)
        with patch.object(cp.np, 'load') as load, self.assertRaisesRegex(ValueError, 'outside registered'):
            cp.read_confirmation_part(self.bundle, protocol, second, {}, [])
        load.assert_not_called()

    def test_synthetic_reader_tracks_confirmation_role_and_future_label_members(self):
        protocol = self.prepare()
        selected = protocol['rows'][0]
        ledger = []
        normalizer = dict(mean=np.zeros(784), scale=np.ones(784))

        def adapter(graph, config):
            graph['future_mock']  # Stand-in for the unchanged adapter's label computation.
            return dict(x=np.zeros((2, 784), dtype=np.float32), metadata={})

        with patch.object(cp, 'line_graph_from_graph', side_effect=adapter):
            part = cp.read_confirmation_part(self.bundle, protocol, selected, normalizer, ledger)
        self.assertEqual(part['scenario_id'], selected['scenario_id'])
        self.assertEqual(part['split'], 'train')
        self.assertEqual(ledger[0]['role'], 'confirmation')
        self.assertEqual(ledger[0]['status'], 'READ_COMPLETE')
        self.assertEqual(ledger[0]['future_members_used_only_for_unchanged_labels'], ['future_mock'])

    def test_graph_changed_during_adapter_read_is_rejected(self):
        protocol = self.prepare()
        selected = protocol['rows'][0]
        path = Path(protocol['graph_root']) / selected['graph_path']
        ledger = []

        def adapter(graph, config):
            path.write_bytes(path.read_bytes() + b'changed-after-load')
            return dict(x=np.zeros((1, 784), dtype=np.float32), metadata={})

        with patch.object(cp, 'line_graph_from_graph', side_effect=adapter), \
                self.assertRaisesRegex(ValueError, 'changed during read'):
            cp.read_confirmation_part(self.bundle, protocol, selected, {}, ledger)
        self.assertEqual(ledger[0]['status'], 'READ_FAILED')


if __name__ == '__main__':
    unittest.main()
