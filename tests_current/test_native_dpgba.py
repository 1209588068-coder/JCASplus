"""Uploaded DPGBA source fidelity; synthetic tensors only, optional CUDA smoke."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from author_native_runs import dpgba_native as native
from tests_current.test_native_ugba import fixture


class NativeDPGBATests(unittest.TestCase):
    def setUp(self):
        state = patch.dict(native._STATE, {}, clear=True)
        state.start()
        self.addCleanup(state.stop)

    def test_preserve_author_algorithms_and_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); source = temp/'clean.pt'; torch.save(fixture(), source)
            out = temp/'new'; record = native.prepare_run(source, out, synthetic=True)
            self.assertEqual(record['method'], 'DPGBA')
            self.assertEqual(record['expected_victim_runs'], 1)
            self.assertEqual(record['author_defaults']['k'], 100)
            self.assertEqual(record['author_defaults']['hidden'], 512)
            self.assertEqual(record['author_defaults']['rec_epochs'], 100)
            self.assertEqual(record['author_defaults']['vs_number'], 40)
            self.assertEqual(record['author_defaults']['trigger_size'], 3)
            self.assertEqual(record['author_defaults']['defense_mode'], 'reconstruct')
            self.assertEqual(record['argv'][record['argv'].index('--vs_number') + 1], '40')
            self.assertEqual(record['argv'][record['argv'].index('--defense_mode') + 1], 'none')
            self.assertIn('author default is reconstruct', record['defense_mode'])
            self.assertIn('NormalizeFeatures', record['feature_preprocessing'])
            self.assertIn('raw AV2 784', record['input_features'])
            self.assertFalse(record['reused_feature_statistics'])
            for relative in native.CORE_FILES:
                before = (out/'pristine'/native.RUN_PREFIX/relative).read_text()
                after = (out/'runtime'/relative).read_text()
                self.assertEqual(native.algorithm_ast(before), native.algorithm_ast(after))
                if relative != 'models/backdoor.py':
                    self.assertEqual(before, after)
            entry = (out/'runtime/run_adaptive.py').read_text()
            self.assertIn("models = ['GCN']", entry)
            self.assertNotIn('get_split(args,data,device)', entry)
            self.assertIn('load_native_input(device)', entry)
            self.assertIn('All non-supervised TRAIN candidates', entry)
            self.assertIn('model.inject_trigger(idx_atk,poison_x', entry)
            self.assertTrue(all(record['non_import_ast_identical'].values()))
            self.assertFalse((out/'runtime/model_weights.pth').exists())
            self.assertEqual(list((out/'work').iterdir()), [])
            for p in (out/'runtime').rglob('*.py'):
                ast.parse(p.read_text())
            with self.assertRaises(FileExistsError):
                native.prepare_run(source, out, synthetic=True)

    def test_author_normalization_preserves_raw_input_and_context_candidates(self):
        from torch_geometric.data import Data
        from torch_geometric.transforms import NormalizeFeatures
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'clean.pt'; payload = fixture()
            payload['y'][payload['candidates'][0]] = -1
            payload['y'][payload['candidates'][1]] = 1
            raw_x = payload['x'].clone()
            torch.save(payload, source)
            before_hash = native.neutral.sha256(source)
            native._STATE.update(root=root, protocol=dict(input_path=str(source), synthetic=True,
                input_description=native.neutral.validate_input(payload, synthetic=True)))
            with patch.object(native, '_event'):
                data, idx_train, idx_val, idx_clean_test, idx_atk, candidates = native.load_native_input('cpu')
            expected = NormalizeFeatures()(Data(x=raw_x.clone())).x
            self.assertTrue(torch.equal(data.x, expected))
            self.assertTrue(torch.allclose(data.x.sum(dim=1), torch.ones(len(raw_x))))
            self.assertFalse(torch.equal(data.x, raw_x))
            self.assertTrue(torch.equal(native._STATE['normalized_x'], expected))
            self.assertEqual(native._STATE['normalized_x'].device.type, 'cpu')
            self.assertTrue(torch.equal(native._STATE['input']['x'], raw_x))
            self.assertTrue(torch.equal(torch.load(source, weights_only=True)['x'], raw_x))
            self.assertEqual(native.neutral.sha256(source), before_hash)
            for actual, name in zip((idx_train, idx_val, idx_clean_test, idx_atk, candidates),
                                    ('idx_train', 'idx_val', 'idx_clean_test', 'idx_atk', 'candidates')):
                self.assertTrue(torch.equal(actual, payload[name]))
            self.assertEqual(set(data.y[candidates[:2]].tolist()), {-1, 1})
            data.x[0, 0] += 1
            self.assertTrue(torch.equal(native._STATE['normalized_x'], expected))

    def test_native_selection_uses_configured_count(self):
        for count in (40, 7):
            with self.subTest(vs_number=count), tempfile.TemporaryDirectory() as temp:
                payload = fixture(); payload['y'][payload['candidates'][:2]] = torch.tensor([-1, 1])
                args = SimpleNamespace(vs_number=count, seed=12, trojan_epochs=400,
                                       rec_epochs=100, k=100, outter_size=4096, hidden=512)
                selected, remaining = payload['candidates'][:count], payload['candidates'][count:]
                native._STATE.update(root=Path(temp), input=payload)
                with patch.object(native, '_event'):
                    with self.assertRaises(ValueError):
                        native.record_selection(args, selected[:-1], remaining)
                    native.record_selection(args, selected, remaining)
                record = json.loads((Path(temp)/'selection.json').read_text())
                self.assertEqual(record['attachments'], selected.tolist())
                self.assertTrue(torch.equal(native._STATE['attachments'], selected))
                self.assertEqual(set(payload['y'][selected[:2]].tolist()), {-1, 1})

    def test_injection_audit_uses_normalized_features_and_configured_trigger_size(self):
        from torch_geometric.data import Data
        from torch_geometric.transforms import NormalizeFeatures
        for count, trigger_size in ((40, 3), (7, 2)):
            with self.subTest(vs_number=count, trigger_size=trigger_size), tempfile.TemporaryDirectory() as temp:
                payload = fixture(); payload['y'][payload['candidates'][0]] = -1
                normalized = NormalizeFeatures()(Data(x=payload['x'].clone())).x
                attachments = payload['candidates'][:count]
                native._STATE.update(root=Path(temp), input=payload, normalized_x=normalized.clone(),
                                     attachments=attachments)
                args = SimpleNamespace(vs_number=count, trigger_size=trigger_size, target_class=1)
                trigger = torch.zeros(count * trigger_size, payload['x'].shape[1])
                x = torch.cat((normalized, trigger))
                labels = payload['y'].clone(); labels[attachments] = args.target_class
                model = SimpleNamespace(trojan=torch.nn.Linear(payload['x'].shape[1], 2))
                edges = payload['edge_index']; weights = torch.ones(edges.shape[1])
                with patch.object(native, '_event'):
                    with self.assertRaises(ValueError):
                        native.record_generator(args, model, torch.cat((payload['x'], trigger)), edges, weights, labels)
                    with self.assertRaises(ValueError):
                        native.record_generator(args, model, x[:-1], edges, weights, labels)
                    native.record_generator(args, model, x, edges, weights, labels)
                self.assertTrue((Path(temp)/'generator.pt').is_file())

    def test_lazy_imports_do_not_change_math(self):
        source = 'import faiss\ndef f(x):\n    return faiss.Kmeans(x, 2)\n'
        out = native.lazy_optional_imports(source)
        self.assertEqual(native.algorithm_ast(source), native.algorithm_ast(out))
        self.assertIn('    import faiss', out)

    def test_no_own_attack_imports(self):
        tree = ast.parse(Path(native.__file__).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(n and n.split('.')[0] in ('spear_current', 'author_comparisons', 'comparison_methods') for n in imports))

    def test_reject_modified_source_and_production_synthetic(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); source = temp/'clean.pt'; torch.save(fixture(), source)
            with self.assertRaises(ValueError):
                native.prepare_run(source, temp/'new')

    @unittest.skipUnless(os.environ.get('RUN_NATIVE_DPGBA_CUDA_TESTS') == '1' and torch.cuda.is_available(),
                         'Optional sequential synthetic CUDA native DPGBA')
    def test_cuda_original_native_full_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); source = temp/'clean.pt'; torch.save(fixture(), source)
            out = temp/'new'; native.prepare_run(source, out, synthetic=True)
            try:
                native.execute_prepared(out)
            except BaseException:
                if (out/'stdout.log').exists():
                    print((out/'stdout.log').read_text()[-10000:])
                raise
            result = json.loads((out/'result.json').read_text())
            self.assertEqual(result['method'], 'DPGBA')
            self.assertEqual(result['status'], 'FRESH_NATIVE_DPGBA_COMPLETE')
            self.assertEqual(result['victim_runs_count'], 1)
            self.assertFalse(result['own_attack_modules_imported'])
            self.assertEqual(result['victim_runs'][0]['architecture'], 'GCN')
            self.assertTrue(result['victim_runs'][0]['all_scores_finite'])


if __name__ == '__main__':
    unittest.main()
