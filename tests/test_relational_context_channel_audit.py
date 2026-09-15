import unittest

import numpy as np
import torch

from jcas.workflows import relational_context_channel_audit as audit
from jcas.workflows import relational_context_centering_audit as centered


class ChannelAuditTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(21).normal(size=(20, 9)).astype(np.float32)
        torch.manual_seed(21)
        self.head = audit.context.ContextResidual(self.x, context_enabled=True)
        with torch.no_grad():
            self.head.net[0].weight[:, 8].fill_(.5)
            self.head.net[-1].weight.fill_(.2)

    def test_identity_and_parameters_unchanged(self):
        before = {k: v.clone() for k, v in self.head.state_dict().items()}
        results = audit.interventions(self.head, self.x)
        with torch.no_grad():
            np.testing.assert_array_equal(results['both_on'], self.head(torch.from_numpy(self.x)).squeeze(-1).numpy())
        for key, value in self.head.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertTrue(self.head.context_enabled)
        self.assertTrue(all(p.grad is None for p in self.head.parameters()))

    def test_context_off_matches_flag_without_changing_base_bypass(self):
        results = audit.interventions(self.head, self.x)
        self.head.context_enabled = False
        with torch.no_grad():
            np.testing.assert_array_equal(results['context_off'], self.head(torch.from_numpy(self.x)).squeeze(-1).numpy())
        # Both normalized inputs zero: constant correction, but distinct base outputs.
        correction = results['both_off']-self.x[:, 0]
        np.testing.assert_allclose(correction, correction[0], atol=1e-6)

    def test_decomposition_closes(self):
        outputs = audit.interventions(self.head, self.x)
        parts = audit.decomposition(outputs)
        np.testing.assert_allclose(outputs['both_off']+parts['motion_main_logit']+
                                   parts['context_main_logit']+parts['interaction_logit'],
                                   outputs['both_on'], atol=3e-6)

    def test_pair_probability_is_mean_of_sigmoids(self):
        x = self.x[:4].copy()
        x[:, 0] = [-8., 2., 0., 1.]
        outputs = audit.interventions(self.head, x)
        f = audit.pair_frame(x, np.array([0, 0, 1, 1]), np.array([[0, 1], [2, 3]]), outputs, x[:, 0])
        expected = audit.probability(x[:, 0]).reshape(2, 2).mean(axis=1)
        np.testing.assert_array_equal(f.base_probability, expected)
        self.assertNotAlmostEqual(float(expected[0]), float(audit.probability(x[:2, 0].mean())))

    def test_centering_removes_only_named_components(self):
        base = np.array([-2., 1.])
        bias, motion, ctx, interaction = .3, np.array([1., 2.]), np.array([-.5, -1.]), np.array([.2, -.2])
        outputs = {'both_on': base+bias+motion+ctx+interaction,
                   'context_off': base+bias+motion, 'motion_off': base+bias+ctx,
                   'both_off': base+bias}
        result = centered.centered_logits(base, outputs)
        np.testing.assert_allclose(result['remove_motion_independent_correction'], base+motion+interaction)
        np.testing.assert_allclose(result['interaction_only'], base+interaction)


if __name__ == '__main__':
    unittest.main()
