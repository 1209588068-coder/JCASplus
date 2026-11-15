"""Synthetic permission/availability tests; no AV2 assets are opened."""
import unittest
from types import SimpleNamespace

import torch

from spear_current.extension_protocol import case_matrix
from spear_current.extension_runtime import (RestrictedSupportGenerator,
    missing_view, budget_selection, checked_coordinates)
from spear_current.evaluate import state_hash
from spear_current.support_replication_runtime import frozen_support
from tests_current.test_domain_generator import setup


class ExtensionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.anchor, self.mask, self.x, self.context, self.targets = setup()
        self.cases = case_matrix(self.mask.tolist())

    def test_all_cases_preserve_exact_effective_permissions(self):
        original = self.x.clone()
        for case in self.cases:
            model = RestrictedSupportGenerator(self.anchor, self.mask, case['active_coordinates'])
            before = state_hash(model)
            result = model.inject(self.x, self.context, self.targets, detach=True)
            allowed = torch.zeros_like(self.x, dtype=torch.bool)
            if case['active_coordinates']:
                allowed[self.targets[:, None], case['active_coordinates']] = True
            self.assertTrue(torch.equal(result.features[~allowed], original[~allowed]))
            self.assertTrue(torch.equal(result.edge_index, self.context.edge_index))
            self.assertTrue(torch.equal(self.x, original))
            self.assertEqual(state_hash(model), before)
            self.assertEqual(int(model.hard_support.sum()), 24)
            self.assertEqual(int(model.effective_support.sum()), case['active_coordinate_count'])

    def test_reference_matches_parent_and_empty_targets(self):
        active = self.cases[0]['active_coordinates']
        restricted = RestrictedSupportGenerator(self.anchor, self.mask, active)
        parent = frozen_support(self.anchor, self.mask)
        for target in (self.targets, torch.empty(0, dtype=torch.long)):
            output = restricted.inject(self.x, self.context, target)
            expected = parent.inject(self.x, self.context, target)
            self.assertTrue(torch.equal(output.features, expected.features))

    def test_mutated_or_illegal_effective_permissions_rejected(self):
        with self.assertRaises(ValueError):
            RestrictedSupportGenerator(self.anchor, self.mask, [357])
        with self.assertRaises(ValueError):
            checked_coordinates([350])
        model = RestrictedSupportGenerator(self.anchor, self.mask, self.cases[0]['active_coordinates'])
        model.effective_support[0] = True
        with self.assertRaises(ValueError):
            model(self.x, self.context, self.targets)

    def test_missing_views_owned_and_parts_equal_combined(self):
        x = torch.arange(4*784, dtype=torch.float32).reshape(4, 784)
        graph = dict(x=x, offsets=[0, 1], y=torch.tensor([0, 1, 0, 1]),
                     edge_index=torch.tensor([[1, 2], [2, 1]]))
        part1 = dict(x=x[:1].clone(), marker='a')
        part2 = dict(x=x[1:].numpy().copy(), marker='b')
        data = dict(combined={r:graph for r in ('train','dev')},
                    parts={r:[part1,part2] for r in ('train','dev')}, train=graph, dev=graph)
        saved = x.clone()
        columns = self.cases[8]['missing_coordinates']
        self.assertTrue(columns)
        out = missing_view(data, columns)
        for role in ('train','dev'):
            values = out['combined'][role]['x']
            self.assertTrue(bool((values[:,columns] == 0).all()))
            keep = [i for i in range(784) if i not in columns]
            self.assertTrue(torch.equal(values[:,keep], saved[:,keep]))
            self.assertTrue(torch.equal(torch.as_tensor(out['parts'][role][0]['x']),values[:1]))
            self.assertTrue(torch.equal(torch.as_tensor(out['parts'][role][1]['x']),values[1:]))
            self.assertIs(out[role],out['combined'][role])
            self.assertIs(out[role]['y'],graph['y'])
            self.assertIs(out[role]['edge_index'],graph['edge_index'])
        self.assertTrue(torch.equal(x,saved))

    def test_zero_attachment_labels_not_added(self):
        data = dict(indices=dict(idx_attach=torch.tensor([2,4,5]),idx_train=torch.tensor([0,1])),
                    outer=torch.tensor([2,4,5,3,6]))
        for count in (0,1,3):
            selection = budget_selection(data,count)
            self.assertEqual(len(selection['idx_attach']),count)
            self.assertTrue(torch.equal(selection['idx_train'],data['indices']['idx_train']))
            self.assertTrue(torch.equal(selection['fixed_outer'][:count],data['indices']['idx_attach'][:count]))
            self.assertTrue(torch.equal(selection['fixed_outer'][count:],torch.tensor([3,6])))
        with self.assertRaises(ValueError):
            budget_selection(data,4)


if __name__ == '__main__':
    unittest.main()
