"""Synthetic-only tests of the original-author DPGBA interface and source guard."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from author_comparisons import dpgba as d


def fixture():
    torch.manual_seed(811)
    n = 256
    x = torch.randn(n, 8)
    edges = torch.tensor([list(range(n - 1)) + list(range(1, n)),
                          list(range(1, n)) + list(range(n - 1))])
    y = torch.full((n,), -1, dtype=torch.long)
    train, val, candidates = torch.arange(64), torch.arange(64, 72), torch.arange(80, 176)
    y[train] = torch.arange(len(train)) % 2
    y[val] = torch.arange(len(val)) % 2
    y[candidates] = 0
    return x, edges, y, train, val, candidates


class AuthorDPGBATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_default_algorithm_parameters_preserved(self):
        original, adapted = d.author_cli_defaults(), d.validate_config()
        changed = {k for k in original if original[k] != adapted[k]}
        self.assertEqual(changed, {'dataset', 'target_class', 'vs_number', 'defense_mode',
                                   'no_cuda', 'test_model'})
        for key, value in dict(hidden=512, trojan_epochs=400, k=100, rec_epochs=100,
                               outter_size=4096, weight_target=1, weight_ood=1,
                               weight_targetclass=1).items():
            self.assertEqual(adapted[key], value)
        self.assertEqual(adapted, json.loads(json.dumps(adapted)))
        with self.assertRaises(ValueError):
            d.validate_config({'defense_mode': 'reconstruct'})
        with self.assertRaises(ValueError):
            d.validate_config({'selection_method': 'cluster_degree'})
        with self.assertRaises(ValueError):
            d.validate_config({'unknown': 1})

    def test_runtime_copy_has_only_import_changes(self):
        before = d.verify_source()
        with tempfile.TemporaryDirectory() as tmp:
            result = d.prepare_runtime(Path(tmp) / 'copy')
            root = Path(result['runtime_root'])
            self.assertFalse((root / 'model_weights.pth').exists())
            self.assertTrue(all(v['non_import_ast_identical'] for v in result['files'].values()))
            for name, metadata in result['files'].items():
                self.assertEqual(d._algorithm_ast((d.SOURCE_ROOT / d.RUN_SUBDIR / name).read_text()),
                                 d._algorithm_ast((root / name).read_text()))
            with d._author_imports(root):
                import importlib
                module = importlib.import_module('models.backdoor')
                selector = importlib.import_module('heuristic_selection')
                self.assertEqual(module.MLPAE.__module__, 'models.reconstruct')
                self.assertTrue(callable(selector.obtain_attach_nodes))
            with self.assertRaises(FileExistsError):
                d.prepare_runtime(Path(tmp) / 'copy')
        self.assertEqual(before, d.verify_source())

    def test_invalid_labels_cannot_enter_supervision(self):
        x, edges, y, train, val, candidates = fixture()
        invalid = y.clone()
        invalid[train[0]] = -1
        with self.assertRaisesRegex(ValueError, 'background'):
            d._inputs(x, edges, invalid, train, val)
        with tempfile.TemporaryDirectory() as tmp:
            invalid = y.clone()
            invalid[candidates[0]] = 1
            with self.assertRaisesRegex(ValueError, 'true-negative'):
                d.fit_dpgba(x, edges, invalid, train, val, candidates, seed=9,
                            work_dir=Path(tmp) / 'never_created')
            self.assertFalse((Path(tmp) / 'never_created').exists())

    def test_execution_device_is_explicit(self):
        self.assertEqual(d._execution_device('cpu'), torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'CPU or CUDA'):
            d._execution_device('meta')

    def test_runtime_reuse_is_source_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'copy'
            result = d.prepare_runtime(work)
            self.assertEqual(result, d._classifier_runtime(work))
            source = Path(result['runtime_root']) / 'models/SAGE.py'
            # Only an ephemeral synthetic runtime is changed, never author source.
            source.write_text(source.read_text() + '\n# changed runtime\n')
            with self.assertRaisesRegex(ValueError, 'runtime code changed'):
                d._classifier_runtime(work)

    def test_actual_author_fit_injection_and_classifier(self):
        self._check_actual_author_fit_injection_and_classifier('cpu')

    @unittest.skipUnless(os.environ.get('JCAS_TEST_DPGBA_CUDA') == '1',
                         'Explicit opt-in CUDA smoke; never claim GPU during ordinary CPU tests')
    def test_cuda_actual_author_fit_injection_and_classifier(self):
        self._check_actual_author_fit_injection_and_classifier('cuda:0')

    def _check_actual_author_fit_injection_and_classifier(self, device):
        x, edges, y, train, val, candidates = fixture()
        originals = [v.clone() for v in (x, edges, y)]
        # Explicit synthetic-smoke budget only; production defaults are not edited.
        cfg = dict(hidden=8, trojan_epochs=1, epochs=2, rec_epochs=1, k=1,
                   range=10., thrd=1e8, debug=False)
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'attack'
            execution_device = d._execution_device(device)
            original_rngs = d._rng_record(execution_device)
            result = d.fit_dpgba(x, edges, y, train, val, candidates, seed=91,
                                work_dir=work, config=cfg, progress=events.append, device=device)
            self.assertEqual(original_rngs, d._rng_record(execution_device))
            self.assertEqual(result['record']['execution_device'], str(execution_device))
            self.assertEqual(set(result['record']['fit_rng_states_sha256']), set(original_rngs))
            self.assertEqual(result['record']['config']['no_cuda'], execution_device.type == 'cpu')
            for key in ('attachment_indices', 'poisoned_x', 'poisoned_edge_index',
                        'poisoned_edge_weight', 'poisoned_y', 'fit_indices'):
                self.assertEqual(result[key].device.type, 'cpu')
            selected = result['attachment_indices']
            self.assertEqual(len(selected), 26)
            self.assertTrue(torch.isin(selected, candidates).all())
            expected_y = y.clone()
            expected_y[selected] = 1
            self.assertTrue(torch.equal(result['poisoned_y'], expected_y))
            self.assertTrue(torch.equal(result['poisoned_x'][:len(x)], x))
            self.assertEqual(len(result['poisoned_x']), len(x) + 3 * 26)
            self.assertEqual(result['poisoned_edge_index'].shape[1], edges.shape[1] + 8 * 26)
            self.assertTrue((result['poisoned_edge_weights'] == 0).any())
            self.assertFalse(result['record']['dominant_used'])
            self.assertTrue(result['record']['generator_checkpoint'].startswith('author_actual_final_epoch'))
            for original, current in zip(originals, (x, edges, y)):
                self.assertTrue(torch.equal(original, current))
            generator = result['generator']
            self.assertEqual(next(generator.parameters()).device, execution_device)
            native = generator.backdoor.inject_trigger(torch.tensor([0]), x, edges,
                    x.new_ones(edges.shape[1]), execution_device)
            wrapped = generator.inject(x, edges, torch.tensor([0]))
            for a, b in zip(native, wrapped):
                self.assertEqual(b.device.type, 'cpu')
                self.assertTrue(torch.equal(a.cpu(), b))
            self.assertEqual(next(generator.parameters()).device, execution_device)
            blank = generator.inject(x, edges, torch.empty(0, dtype=torch.long))
            self.assertTrue(torch.equal(blank[0], x))
            self.assertTrue(torch.equal(blank[1], edges))
            runtime_hash = d._sha(work / 'author_runtime.json')
            clean = d.fit_author_classifier(x, edges, x.new_ones(edges.shape[1]), y,
                result['fit_indices'], val, seed=810, work_dir=work, config=cfg,
                progress=events.append, device=device)
            trained = d.fit_author_classifier(result['poisoned_x'], result['poisoned_edge_index'],
                result['poisoned_edge_weights'], result['poisoned_y'], result['fit_indices'], val,
                seed=810, work_dir=work, config=cfg, progress=events.append, device=device)
            self.assertEqual(original_rngs, d._rng_record(execution_device))
            self.assertEqual(runtime_hash, d._sha(work / 'author_runtime.json'))
            for field in ('initialization_digest', 'fit_rng_digest', 'fit_rng_states_sha256'):
                self.assertEqual(clean['record'][field], trained['record'][field])
            self.assertEqual(len(list(work.glob('author_classifier_*.json'))), 2)
            self.assertEqual(json.loads((work / 'author_classifier_0000.json').read_text()), clean['record'])
            self.assertEqual(json.loads((work / 'author_classifier_0001.json').read_text()), trained['record'])
            with self.assertRaises(FileExistsError):
                d.fit_dpgba(x, edges, y, train, val, candidates, seed=91, work_dir=work, config=cfg)
            model = trained['model']
            self.assertEqual(next(model.parameters()).device, execution_device)
            with torch.no_grad():
                px, pe, pw = (result[key].to(execution_device) for key in
                              ('poisoned_x', 'poisoned_edge_index', 'poisoned_edge_weights'))
                a = model(px, pe, pw)
                b = model(px, pe, torch.ones_like(pw))
            # Native GraphSAGE ignores these weights; CUDA reductions need not
            # be bitwise deterministic across two otherwise identical forwards.
            self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-6))
            self.assertTrue(torch.isfinite(a).all())
            self.assertEqual(trained['record']['supervision_count'], len(train) + 26)
            self.assertEqual(trained['record']['model_selection'], 'author_validation_accuracy_strict_improvement')
            self.assertEqual([v['stage'] for v in events if not v['stage'].endswith('_dev')], [
                'dpgba_author_fit_started', 'dpgba_author_fit_complete',
                'dpgba_author_classifier_started', 'dpgba_author_classifier_complete',
                'dpgba_author_classifier_started', 'dpgba_author_classifier_complete'])
            self.assertEqual(len([v for v in events if v['stage'].endswith('_dev')]), 4)
            self.assertEqual(len(trained['record']['trace']), cfg['epochs'])


if __name__ == '__main__':
    unittest.main()
