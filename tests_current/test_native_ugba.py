"""Synthetic only: uploaded author source fidelity and isolated native entry."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from author_native_runs.ugba_native import (CORE_FILES, SCHEMA, execute_prepared,
                                           patch_source, prepare_run, validate_input)


def fixture():
    rng = torch.Generator().manual_seed(7)
    n_train, n_dev, dim = 800, 80, 8
    x = torch.randn(n_train + n_dev, dim, generator=rng)
    y = torch.zeros(n_train + n_dev, dtype=torch.long)
    y[1:200:2] = 1
    y[n_train + 1::2] = 1
    # Positive class can be learned, ensuring native cluster selection has a
    # meaningful non-target cluster without replacing its selection algorithm.
    x[:, 0] = y.float() * 4 - 2 + x[:, 0] * .05
    edges = []
    for lo, hi in ((0, n_train), (n_train, n_train + n_dev)):
        a = torch.arange(lo, hi - 1); b = a + 1
        edges += [torch.stack((a, b)), torch.stack((b, a))]
    idx_val = torch.arange(n_train, n_train + n_dev)
    return dict(schema=SCHEMA, x=x, y=y, edge_index=torch.cat(edges, 1),
                idx_train=torch.arange(200), idx_val=idx_val, idx_clean_test=idx_val.clone(),
                candidates=torch.arange(200, 800), idx_atk=torch.tensor([800, 802]),
                scene_ptr=torch.tensor([0, 800, 880]), scene_role=['train', 'dev'],
                scene_id=['synthetic_train', 'synthetic_dev'], train_nodes=n_train,
                moving_mask=torch.zeros(880, dtype=torch.bool), metadata={'synthetic': True, 'source_split': 'train'})


class NativeUGBATests(unittest.TestCase):
    def test_synthetic_contract(self):
        out = validate_input(fixture(), synthetic=True)
        self.assertEqual(out['targets'], 2)
        with self.assertRaises(ValueError):
            validate_input(fixture(), synthetic=False)

    def test_index_isolation_and_context(self):
        for key, value in [('candidates', torch.arange(280, 880)), ('idx_train', torch.tensor([0, 800]))]:
            data = fixture(); data[key] = value
            with self.assertRaises(ValueError):
                validate_input(data, synthetic=True)
        data = fixture(); data['y'][800] = -1
        with self.assertRaises(ValueError):
            validate_input(data, synthetic=True)
        data = fixture(); data['edge_index'][1, 0] = 800
        with self.assertRaises(ValueError):
            validate_input(data, synthetic=True)

    def test_native_source_identity_and_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); source = temp / 'clean.pt'; torch.save(fixture(), source)
            output = temp / 'new'
            record = prepare_run(source, output, synthetic=True)
            self.assertEqual(record['expected_victim_runs'], 15)
            for rel in CORE_FILES:
                self.assertEqual((output/'pristine'/rel).read_bytes(), (output/'runtime'/rel).read_bytes())
            text = (output/'runtime/run_adaptive.py').read_text()
            ast.parse(text)
            self.assertIn("models = ['GCN','GAT', 'GraphSage']", text)
            self.assertIn('seeds = rs.randint(1000,size=5)', text)
            self.assertNotIn('get_split(args,data,device)', text)
            self.assertNotIn('NormalizeFeatures()', text)
            self.assertIn("NormalizeFeatures()(data)", Path('author_native_runs/ugba_native.py').read_text())
            self.assertEqual(record['argv'][record['argv'].index('--vs_number') + 1], '40')
            self.assertEqual(record['argv'][record['argv'].index('--selection_method') + 1], 'none')
            self.assertFalse((output/'runtime/selected_nodes').exists())
            self.assertEqual(list((output/'work').iterdir()), [])
            for path in (output/'runtime').rglob('*.py'):
                ast.parse(path.read_text())
            with self.assertRaises(FileExistsError):
                prepare_run(source, output, synthetic=True)

    def test_no_own_attack_import(self):
        source = Path(importlib.util.find_spec('author_native_runs.ugba_native').origin).read_text()
        tree = ast.parse(source)
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(n and n.split('.')[0] in ('spear_current', 'author_comparisons', 'comparison_methods') for n in imported))

    def test_candidates_include_positive_and_unlabeled(self):
        data = fixture()
        data['y'][201] = 1
        data['y'][202] = -1
        validate_input(data, synthetic=True)
        data['candidates'] = data['candidates'][2:]
        with self.assertRaisesRegex(ValueError, 'all non-supervised'):
            validate_input(data, synthetic=True)

    def test_patch_fails_closed(self):
        with self.assertRaises(ValueError):
            patch_source('heuristic_selection.py', '# unrelated source')
        self.assertEqual(patch_source('models/backdoor.py', 'exact core bytes\n'), 'exact core bytes\n')

    @unittest.skipUnless(os.environ.get('RUN_NATIVE_UGBA_CUDA_TESTS') == '1' and torch.cuda.is_available(),
                         'Optional sequential GPU-only native synthetic run')
    def test_native_cuda_full_author_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); source = temp/'clean.pt'; torch.save(fixture(), source)
            output = temp/'new'; prepare_run(source, output, synthetic=True)
            execute_prepared(output)
            result = json.loads((output/'result.json').read_text())
            self.assertEqual(result['victim_runs_count'], 15)
            self.assertFalse(result['own_attack_modules_imported'])
            self.assertTrue(all(r['all_scores_finite'] for r in result['victim_runs']))
            self.assertTrue((output/'completion.json').is_file())


if __name__ == '__main__':
    unittest.main()
