"""Numerical boundary regression for the full-AV2 ranking failure."""
import unittest

import torch

from spear_current.av2_formal_selection import rank_candidates


def author_entropy(log_probabilities):
    q = log_probabilities / log_probabilities.sum(dim=1, keepdim=True)
    return -(q * q.log2()).sum(dim=1)


class EntropyBoundaryTests(unittest.TestCase):
    def test_actual_failed_checkpoint_output_has_zero_score(self):
        # TRAIN pair773: independently replayed from the selected seed21 model.
        logits = torch.tensor([[0., -19.364839553833008]])
        self.assertTrue(torch.isnan(author_entropy(logits)).all())
        rank, score = rank_candidates(logits, torch.tensor([773]), author_entropy)
        self.assertEqual(rank.tolist(), [773])
        self.assertEqual(score.tolist(), [0.])
        self.assertTrue(torch.isfinite(score).all())

    def test_both_saturation_directions_and_signed_zero_are_supported(self):
        logits = torch.tensor([[0., -20.], [-20., 0.], [-0., -30.], [-30., -0.]])
        ids = torch.tensor([10, 11, 12, 13])
        rank, score = rank_candidates(logits, ids, author_entropy)
        self.assertEqual(rank.tolist(), [10, 11, 12, 13])
        self.assertTrue(torch.equal(score, torch.zeros(4)))

    def test_ordinary_author_scores_are_bitwise_identical(self):
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator().manual_seed(816)
            logits = (torch.randn(1000, 2, generator=generator, dtype=dtype) * 2).log_softmax(1)
            self.assertTrue(bool((logits < 0).all()))
            ids = torch.arange(len(logits))
            _, score = rank_candidates(logits, ids, author_entropy)
            self.assertTrue(torch.equal(score, author_entropy(logits)))

    def test_regular_rows_in_mixed_batch_are_unchanged(self):
        logits = torch.tensor([[0., -20.], [-1., -2.], [-2., -1.], [-20., 0.]])
        ids = torch.tensor([1, 2, 3, 4])
        rank, score = rank_candidates(logits, ids, author_entropy)
        self.assertTrue(torch.equal(score[1:3], author_entropy(logits[1:3])))
        self.assertEqual(rank.tolist(), [2, 3, 1, 4])

    def test_invalid_input_is_not_hidden_by_boundary_repair(self):
        maximum = torch.finfo(torch.float32).max
        for row in ([0., 0.], [-0., -0.], [float('nan'), -1.],
                    [-float('inf'), 0.], [float('inf'), 0.],
                    [0.01, -1.], [-maximum, -maximum]):
            with self.subTest(row=row), self.assertRaises(FloatingPointError):
                rank_candidates(torch.tensor([row]), torch.tensor([0]), author_entropy)
        with self.assertRaises(FloatingPointError):
            rank_candidates(torch.tensor([[0, -1]]), torch.tensor([0]), author_entropy)

    def test_unexplained_nonfinite_author_score_remains_an_error(self):
        logits = torch.tensor([[-1., -2.]])
        for invalid in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(invalid=invalid), self.assertRaises(FloatingPointError):
                rank_candidates(logits, torch.tensor([0]),
                                lambda value: value.new_full((len(value),), invalid))

    def test_malformed_author_result_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_candidates(torch.tensor([[-1., -2.]]), torch.tensor([0]),
                            lambda value: value.new_zeros((len(value), 1)))


if __name__ == '__main__':
    unittest.main()
