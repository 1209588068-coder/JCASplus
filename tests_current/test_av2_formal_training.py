"""Numerical and author-interface checks using synthetic disconnected graphs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from spear_current.av2_formal_classifiers import accumulate_supervised_gradients
from spear_current.av2_formal_selection import predicted_class_round_robin, rank_candidates
from spear_current.models import GCN, GraphSage
from spear_current.womd_execution import configure_message_passing


def graph(count, offset=0, device='cpu', seed=17):
    rng = torch.Generator().manual_seed(seed)
    labels = torch.arange(count, dtype=torch.long) % 2
    features = torch.randn(count, 784, generator=rng) * .1
    features[:, :4] += (labels.float()*2-1)[:, None]
    rows = torch.arange(count)
    edges = torch.stack((rows, (rows+1) % count))
    edges = torch.cat((edges, edges.flip(0)), 1)
    return dict(x=features.to(device), y=labels.to(device),
        eligible=torch.ones(count, dtype=torch.bool, device=device),
        edge_index=edges.to(device), global_start=offset, global_stop=offset+count)


class RankingTests(unittest.TestCase):
    def test_interleave_exactly_matches_author_loop_for_empty_and_unequal_classes(self):
        for count in (0, 1, 2, 7, 100):
            ids = torch.randperm(count)
            for threshold in (0., .1, .5, .9, 1.):
                classes = (torch.rand(count) < threshold).long()
                groups = [ids[classes == i].tolist() for i in (0, 1)]
                expected = []
                for index in range(max(map(len, groups), default=0)):
                    for group in groups:
                        if index < len(group):
                            expected.append(group[index])
                actual = predicted_class_round_robin(ids, classes)
                self.assertEqual(actual.tolist(), expected)

    def test_score_argsort_then_interleave_matches_reference_including_ties(self):
        def entropy(values):
            normalized = values / values.sum(1, keepdim=True)
            return -(normalized*normalized.log2()).sum(1)
        logits = torch.tensor([[2., -1.], [-1., 2.], [.2, 0.], [.2, 0.], [0., .2]]).log_softmax(1)
        candidates = torch.tensor([3, 8, 11, 15, 17])
        ranking, score = rank_candidates(logits, candidates, entropy)
        order = score.argsort(descending=True)
        ids, predictions = candidates[order], logits.argmax(1)[order]
        pointers, expected = [0, 0], []
        while len(expected) < len(ids):
            for label in (0, 1):
                values = ids[predictions == label]
                if pointers[label] < len(values):
                    expected.append(int(values[pointers[label]]))
                    pointers[label] += 1
        self.assertEqual(ranking.tolist(), expected)


class GradientAccumulationTests(unittest.TestCase):
    def test_complete_scene_gradients_match_disjoint_union_without_dropout(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for model_class in (GCN, GraphSage):
                with self.subTest(model=model_class.__name__):
                    torch.manual_seed(31)
                    full_model = model_class(784, 80, 2, dropout=0., device=torch.device('cpu'))
                    configure_message_passing(full_model)
                    streamed_model = deepcopy(full_model)
                    first, second = graph(6), graph(8, offset=6, seed=19)
                    full = dict(x=torch.cat((first['x'], second['x'])).requires_grad_(),
                        y=torch.cat((first['y'], second['y'])),
                        eligible=torch.cat((first['eligible'], second['eligible'])),
                        edge_index=torch.cat((first['edge_index'], second['edge_index']+6), 1),
                        global_start=0, global_stop=14)
                    first['x'].requires_grad_()
                    second['x'].requires_grad_()
                    indices = torch.tensor([0, 2, 3, 7, 9, 12, 13])
                    one = accumulate_supervised_gradients(full_model, [full], indices)
                    two = accumulate_supervised_gradients(streamed_model, [first, second], indices)
                    self.assertAlmostEqual(one['train_loss'], two['train_loss'], places=6)
                    for p, q in zip(full_model.parameters(), streamed_model.parameters()):
                        if p.grad is not None:
                            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-5)
                    torch.testing.assert_close(full['x'].grad,
                        torch.cat((first['x'].grad, second['x'].grad)), atol=2e-7, rtol=2e-5)
        finally:
            torch.set_num_threads(old_threads)

    def test_missing_scene_supervision_is_rejected(self):
        model = GraphSage(784, 80, 2, dropout=0., device=torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            accumulate_supervised_gradients(model, [graph(6)], torch.tensor([0, 8]))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for author execution regression')
class CudaSelectionTests(unittest.TestCase):
    def test_fresh_author_GCN_MLP_SAGE_and_classifier_on_scene_batches(self):
        from spear_current.av2_formal_selection import fit_source_selection
        from spear_current.av2_formal_classifiers import fit_classifier
        train = [graph(24), graph(24, offset=24, seed=19)]
        dev = [graph(8, seed=21), graph(8, offset=8, seed=23)]
        population = dict(y=torch.cat([b['y'] for b in train]),
                          eligible=torch.cat([b['eligible'] for b in train]))
        clean = torch.arange(16, dtype=torch.long)

        def fake_batches(data_dir, role, max_nodes, device):
            for batch in train if role == 'train' else dev:
                yield {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        events = []
        with tempfile.TemporaryDirectory(prefix='av2_streaming_regression_') as temporary:
            (Path(temporary)/'summary.json').write_text(json.dumps(dict(status='SYNTHETIC_MOCK')))
            with patch('spear_current.av2_formal_classifiers.batches', fake_batches), \
                 patch('spear_current.av2_formal_selection.batches', fake_batches), \
                 patch('spear_current.av2_formal_data.load_role_arrays', return_value=population):
                result = fit_source_selection(temporary, seed=17, output_dir=Path(temporary)/'selection',
                    indices=dict(idx_train=clean), attachment_count=2, outer_extra_count=4,
                    model_epochs=2, sage_n_permutations=2, sage_batch_size=2,
                    explanation_count=8, background_count=4, progress=events.append)
                fitted = fit_classifier(temporary, clean, seed=118,
                    output_dir=Path(temporary)/'proxy', role='clean_proxy_1', epochs=2)
            self.assertEqual(result['coordinates'].shape, (30,))
            self.assertEqual(result['coordinates'].device, torch.device('cuda:0'))
            self.assertEqual(result['indices']['idx_attach'].shape, (2,))
            self.assertEqual(result['indices']['fixed_outer'].shape, (6,))
            self.assertEqual(result['summary']['dataset'], 'AV2')
            self.assertEqual(fitted['summary']['optimizer_steps'], 2)
            self.assertEqual(fitted['summary']['train_supervision_count'], 16)
            self.assertIn('fresh_author_SAGE', [event['stage'] for event in events])
            checkpoint = torch.load(Path(temporary)/'proxy/latest.pt', weights_only=True, map_location='cpu')
            self.assertEqual(checkpoint['metadata']['next_epoch'], 2)
            self.assertEqual(checkpoint['metadata']['optimizer_steps'], 2)
            self.assertEqual(checkpoint['data_summary_sha256'],
                hashlib.sha256((Path(temporary)/'summary.json').read_bytes()).hexdigest())
            self.assertEqual(checkpoint['train_indices_sha256'], fitted['summary']['train_indices_sha256'])
            self.assertTrue(checkpoint['optimizer_state']['state'])
            self.assertTrue(checkpoint['best_state'])
            self.assertIsNotNone(checkpoint['rng_cuda'])


if __name__ == '__main__':
    unittest.main()
