"""Isolated pilot orchestration tests; no AV2 assets or real fitting."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import outer_pilot as pilot
from spear_current import train
from tests_current.test_train import fixture


def population_fixture():
    size = 4000
    y = torch.zeros(size, dtype=torch.long)
    y[2100::7] = 1
    y[2101::5] = -1
    eligible = y >= 0
    indices = dict(idx_train=torch.arange(1876), idx_attach=torch.arange(1876, 1902),
                   idx_unlabeled=torch.arange(1902, size), train_labels=y.clone())
    draw = indices['idx_unlabeled'][np.random.RandomState(20260621).choice(size - 1902, 512, False)]
    outer = torch.cat([indices['idx_attach'], draw])

    def metadata(i, *, role='train'):
        return dict(scenario_id=role + str(i // 8), pair_index=i % 8, role_pair_index=i,
            role=role, src_track_id='a', dst_track_id='b' + str(i % 8),
            split_group_id=role + str(i // 8), true_label=int(y[i]) if role == 'train' else 0,
            eligible=bool(eligible[i]) if role == 'train' else True,
            motion_regime='moving' if (i % 5 == 0 if role == 'train' else i < 33) else 'slow',
            is_attachment=1876 <= i < 1902 if role == 'train' else False,
            is_outer=i in set(outer.tolist()) if role == 'train' else False)
    return dict(combined=dict(train=dict(y=y, eligible=eligible)), indices=indices,
        outer=outer, metadata=dict(train=[metadata(i) for i in range(size)]),
        populations=dict(dev_public_single=[metadata(i, role='dev') for i in range(112)]))


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_population_change_keeps_attachment_order_and_public_pool(self):
        inputs = population_fixture()
        old = deepcopy(inputs)
        outers, report = pilot.prepare_populations(inputs)
        self.assertTrue(torch.equal(outers['original_outer'], old['outer']))
        self.assertTrue(torch.equal(inputs['outer'], old['outer']))
        self.assertFalse(torch.equal(outers['original_outer'], outers['strict_negative_outer']))
        self.assertEqual(report['dev_targets'], old['populations']['dev_public_single'])
        self.assertEqual(report['dev_target_identity_sha256'], pilot.json_hash(report['dev_targets']))
        for outer in outers.values():
            self.assertEqual(len(outer), 538)
            self.assertTrue(torch.equal(outer[:26], inputs['indices']['idx_attach']))
        candidate = report['arms']['strict_negative_outer']
        self.assertEqual(candidate['extra_summary']['all']['strict_negative'], 512)
        self.assertEqual(candidate['extra_summary']['all']['positive'], 0)
        self.assertEqual(candidate['extra_summary']['all']['unknown'], 0)
        self.assertEqual(inputs['metadata'], old['metadata'])

    def test_no_contrast_stops_without_fitting(self):
        inputs = population_fixture()
        outers, _ = pilot.prepare_populations(inputs)
        inputs['outer'] = outers['strict_negative_outer']
        with self.assertRaisesRegex(ValueError, 'No selection contrast'):
            pilot.prepare_populations(inputs)

    def test_changed_public_moving_population_is_rejected(self):
        inputs = population_fixture()
        inputs['populations']['dev_public_single'][0]['motion_regime'] = 'slow'
        with self.assertRaisesRegex(ValueError, 'development target population'):
            pilot.prepare_populations(inputs)

    def test_bad_counts_rejected_before_numerical_loop(self):
        reference, surrogate, proxies, inputs = fixture()
        with patch.object(pilot, 'fit_distribution_residual') as fit:
            for arm in pilot.ARMS:
                with self.assertRaisesRegex(ValueError, 'Frozen pilot counts'):
                    pilot.fit_arm(train.new_generator(reference), surrogate, proxies, inputs, arm)
            fit.assert_not_called()

    def test_unknown_arm_rejected_before_numerical_loop(self):
        with patch.object(pilot, 'fit_distribution_residual') as fit:
            with self.assertRaisesRegex(ValueError, 'Unknown pilot arm'):
                pilot.fit_arm(None, None, None, {}, 'unregistered')
            fit.assert_not_called()

    def test_wrapper_preserves_original_training_arguments_and_rng(self):
        reference, surrogate, proxies, fixture_inputs = fixture()
        x = fixture_inputs['train']['x'][:1].repeat(540, 1)
        indices = dict(idx_attach=torch.arange(26), idx_train=torch.tensor([538, 539]))
        inputs = dict(fixture_inputs, train=dict(x=x, y=torch.zeros(540, dtype=torch.long),
            eligible=torch.ones(540, dtype=torch.bool)), indices=indices, outer=torch.arange(538))
        calls = []

        def fake_fit(generator, shadow, features, context, labels, fit, attach,
                     outer, penalty, config, distribution, progress=None):
            calls.append(dict(features=features.clone(), labels=labels.clone(), fit=fit.clone(),
                attach=attach.clone(), outer=outer.clone(), context=context,
                d_state=distribution.discriminator_hash(), rng=torch.get_rng_state().clone(),
                config=deepcopy(config), generator_state=train._state_hash(generator)))
            penalty.history.extend([{} for _ in range(200)])
            return dict(selected_epoch=199, selection_policy='final_epoch',
                config=train.TRAINING_CONFIG, distribution_arm='unconditional',
                residual_steps=200, surrogate_steps=200, clean_constraint_calls=200,
                discriminator_steps=200, distribution_generator_calls=200)

        with patch.object(train, 'fit_distribution_residual', fake_fit), \
                patch.object(pilot, 'fit_distribution_residual', fake_fit):
            train.fit_unconditional(train.new_generator(reference), surrogate, proxies, inputs)
            pilot.fit_arm(train.new_generator(reference), surrogate, proxies, inputs, 'original_outer')
            pilot.fit_arm(train.new_generator(reference), surrogate, proxies, inputs, 'strict_negative_outer')
        for actual in calls[1:]:
            for key, expected in calls[0].items():
                if isinstance(expected, torch.Tensor):
                    self.assertTrue(torch.equal(expected, actual[key]), key)
                elif key == 'context':
                    self.assertIs(expected, actual[key])
                else:
                    self.assertEqual(expected, actual[key], key)

    def test_strict_arm_rejects_nonnegative_rows_before_optimizing(self):
        reference, surrogate, proxies, inputs = fixture()
        inputs = dict(inputs, train=dict(x=torch.zeros(540, 784),
            y=torch.ones(540, dtype=torch.long), eligible=torch.ones(540, dtype=torch.bool)),
            indices=dict(idx_attach=torch.arange(26)), outer=torch.arange(538))
        with patch.object(pilot, 'fit_distribution_residual') as fit:
            with self.assertRaisesRegex(ValueError, 'nonnegative row'):
                pilot.fit_arm(reference, surrogate, proxies, inputs, 'strict_negative_outer')
            fit.assert_not_called()

    def test_default_cli_has_no_graph_loading_or_training(self):
        bundle = SimpleNamespace()
        with patch.object(pilot, 'load_bundle', return_value=bundle), \
                patch.object(pilot, 'preflight', return_value=dict(status='OK')) as check, \
                patch.object(pilot, 'run') as run, patch.object(pilot, 'read_inputs') as read, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main([]), dict(status='OK'))
        check.assert_called_once_with(bundle)
        run.assert_not_called()
        read.assert_not_called()

    def test_output_scope_checked_before_loading_graphs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'results'
            root.mkdir()
            with patch.object(pilot, 'OUTPUT_ROOT', root), patch.object(pilot, 'read_inputs') as read:
                for path in (root, Path(temp) / 'outside'):
                    with self.assertRaisesRegex(ValueError, 'NEW output subdirectory'):
                        pilot.run(None, path)
                read.assert_not_called()

    def test_freeze_clears_gradients_and_all_training_flags(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
        model(torch.ones(1, 2)).sum().backward()
        pilot.freeze(model)
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
