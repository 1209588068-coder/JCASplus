"""Metadata-only current-module experiment guards; no graph/model loading."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch  # Finish library imports before mocking global subprocess.Popen.

from spear_current import current_module_ablation as workflow
from spear_current.support_schema import (
    EXCLUDED_COORDINATES, SELECTED_COORDINATES, support_groups,
)


def semantic_mask():
    groups = support_groups('semantic')
    triples = [g for g in groups if len(g) == 3]
    singles = [g for g in groups if len(g) == 1]
    active = {c for group in triples[:5] + singles[:9] for c in group}
    mask = [c in active for c in SELECTED_COORDINATES]
    assert sum(mask) == 24
    return mask


class CurrentModuleMetadataTests(unittest.TestCase):
    def fixture(self, root):
        folder = root/'results'/'metadata_fixture'
        folder.mkdir(parents=True)
        source = root/'runtime.py'
        source.write_bytes(b'frozen runtime bytes\n')
        cache = root/'cache.bin'
        cache.write_bytes(b'not pickle; metadata checks must never deserialize')
        snapshot = folder/'source_snapshot.tar'
        snapshot.write_bytes(b'not tar; metadata verification hashes bytes only')
        mask = semantic_mask()
        policies = {}
        for seed in workflow.SEEDS:
            policies[str(seed)] = {}
            for variant in workflow.VARIANTS:
                cap = 28 if variant == 'no_support28' else 24
                selected = list(range(100, 126)) if variant == 'matched_random_pairs' else list(range(26))
                support = ([c not in EXCLUDED_COORDINATES for c in SELECTED_COORDINATES]
                           if cap == 28 else mask.copy())
                policies[str(seed)][variant] = dict(
                    variant=variant, hard_support=support, support_capacity=cap,
                    attachment_count=26, candidate_count=6549,
                    attachment_motion_counts=dict(moving=11, slow=15),
                    attachment_indices=selected,
                    outer_background_indices=list(range(300, 812)),
                    idx_train_sha256='same_clean_supervision',
                    candidate_pool_sha256='same_candidate_pool',
                    selection_uses_dev=False, selection_uses_audit_or_test=False,
                    selection_uses_model_predictions=False, feature_selector_retrained=False)
        plan = dict(version=workflow.VERSION, seeds=list(workflow.SEEDS),
            variants=list(workflow.VARIANTS), matrix=workflow.matrix(),
            stage_epochs=dict(workflow.EPOCHS), victim_epochs=200,
            optimizer_steps={v:workflow.expected_steps(v) for v in workflow.VARIANTS},
            execution=dict(device='cuda:0', dtype='float32', amp=False, tf32=False,
                max_concurrent=2, no_cpu_fallback=True, kill_external_jobs=False),
            preflight=dict(seed=workflow.SEEDS[0], variants=list(workflow.VARIANTS),
                epochs_per_stage=2, victim_epochs=2, asr_evaluation=False, weights_reused=False),
            training=dict(output_mode='identity', distribution_weight=0.),
            source_masks=dict(semantic=mask), policies=policies,
            sources={'runtime.py':workflow.sha(source)},
            bound_files={str(cache):workflow.sha(cache)})
        path = folder/'plan.json'
        path.write_text(json.dumps(plan))
        (folder/'registration.json').write_text(json.dumps(dict(
            plan_sha256=workflow.sha(path), source_snapshot_sha256=workflow.sha(snapshot))))
        return path, source, cache, snapshot

    def resign(self, path, plan):
        path.write_text(json.dumps(plan))
        registration = path.parent/'registration.json'
        receipt = json.loads(registration.read_text())
        receipt['plan_sha256'] = workflow.sha(path)
        registration.write_text(json.dumps(receipt))

    def test_full_matrix_has_36_unique_seed_variant_cases(self):
        matrix = workflow.matrix()
        self.assertEqual(len(matrix), 36)
        self.assertEqual(len({(row['seed'], row['variant']) for row in matrix}), 36)
        self.assertEqual({row['seed'] for row in matrix}, set(workflow.SEEDS))
        for seed in workflow.SEEDS:
            self.assertEqual({row['variant'] for row in matrix if row['seed'] == seed},
                             set(workflow.VARIANTS))

    def test_registered_update_budgets_distinguish_deletion_and_continuation(self):
        for variant in workflow.VARIANTS:
            expected = workflow.expected_steps(variant)
            self.assertEqual(expected['generator'], 200 if variant == 'no_residual' else 800)
            self.assertEqual(expected['surrogate'], 1000)
            self.assertEqual(expected['victim'], 200)
            self.assertEqual(expected['discriminator'], 0)
            self.assertEqual(expected['selector'], 0)
        self.assertEqual(workflow.expected_steps('no_residual', 2, 2)['generator'], 2)
        self.assertEqual(workflow.expected_steps('base_continuation', 2, 2)['generator'], 8)
        with self.assertRaises(ValueError):workflow.expected_steps('old_distribution_full')

    def test_valid_registration_is_metadata_only_and_legal28_is_explicit(self):
        with TemporaryDirectory() as folder, patch.object(workflow, 'ROOT', Path(folder)):
            path, _, _, _ = self.fixture(Path(folder))
            with patch('torch.load', side_effect=AssertionError('no tensor/cache deserialization')):
                plan = workflow.verify(path)
            self.assertEqual(plan['training'], dict(output_mode='identity', distribution_weight=0.))
            self.assertFalse(plan['preflight']['asr_evaluation'])
            self.assertFalse(plan['preflight']['weights_reused'])
            self.assertTrue(plan['execution']['no_cpu_fallback'])
            for policies in plan['policies'].values():
                self.assertEqual(sum(policies['no_support28']['hard_support']), 28)
                self.assertEqual(sum(policies['full']['hard_support']), 24)

    def test_plan_source_cache_snapshot_or_expected_hash_changes_are_rejected(self):
        for item in ('source', 'cache', 'snapshot', 'plan', 'expected'):
            with self.subTest(item=item), TemporaryDirectory() as folder, \
                    patch.object(workflow, 'ROOT', Path(folder)):
                path, source, cache, snapshot = self.fixture(Path(folder))
                files = dict(source=source, cache=cache, snapshot=snapshot, plan=path)
                if item != 'expected':files[item].write_bytes(files[item].read_bytes()+b'changed')
                with self.assertRaises(ValueError):
                    workflow.verify(path, '0'*64 if item == 'expected' else None)

    def test_resigned_budget_device_output_or_preflight_changes_are_rejected(self):
        changes = [
            ('seeds', [20260621]), ('variants', ['full']), ('matrix', []),
            ('stage_epochs', {stage:201 for stage in workflow.STAGES}),
            ('victim_epochs', 201), ('optimizer_steps', {}),
            ('training', dict(output_mode='domain_projection', distribution_weight=0.)),
            ('training', dict(output_mode='identity', distribution_weight=1.)),
            ('execution', dict(device='cpu')), ('preflight', {}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value), TemporaryDirectory() as folder, \
                    patch.object(workflow, 'ROOT', Path(folder)):
                path, _, _, _ = self.fixture(Path(folder))
                plan = json.loads(path.read_text());plan[key] = value
                self.resign(path, plan)
                with self.assertRaisesRegex(ValueError, 'training contract'):
                    workflow.verify(path)

    def test_resigned_support_or_candidate_permissions_cannot_expand(self):
        changes = [
            ('candidate_count', 6548),
            ('attachment_motion_counts', dict(moving=12, slow=14)),
            ('selection_uses_dev', True), ('selection_uses_audit_or_test', True),
            ('selection_uses_model_predictions', True), ('feature_selector_retrained', True),
            ('attachment_indices', [0]*26), ('outer_background_indices', [300]*512),
        ]
        for key, value in changes:
            with self.subTest(key=key), TemporaryDirectory() as folder, \
                    patch.object(workflow, 'ROOT', Path(folder)):
                path, _, _, _ = self.fixture(Path(folder))
                plan = json.loads(path.read_text())
                plan['policies'][str(workflow.SEEDS[0])]['full'][key] = value
                self.resign(path, plan)
                with self.assertRaises(ValueError):workflow.verify(path)
        with TemporaryDirectory() as folder, patch.object(workflow, 'ROOT', Path(folder)):
            path, _, _, _ = self.fixture(Path(folder));plan=json.loads(path.read_text())
            mask=plan['policies'][str(workflow.SEEDS[0])]['full']['hard_support']
            mask[SELECTED_COORDINATES.index(357)] = True
            removable=next(i for i,c in enumerate(SELECTED_COORDINATES)
                           if c not in EXCLUDED_COORDINATES and mask[i])
            mask[removable] = False
            self.resign(path,plan)
            with self.assertRaises(ValueError):workflow.verify(path)

    def test_within_seed_background_candidates_and_unchanged_module_attachments_are_paired(self):
        for change in ('background', 'clean', 'candidate', 'module_attachment', 'background_attachment'):
            with self.subTest(change=change), TemporaryDirectory() as folder, \
                    patch.object(workflow, 'ROOT', Path(folder)):
                path, _, _, _ = self.fixture(Path(folder));plan=json.loads(path.read_text())
                policies=plan['policies'][str(workflow.SEEDS[0])]
                row=policies['self_only']
                if change=='background':row['outer_background_indices'][0]=900
                elif change=='clean':row['idx_train_sha256']='changed'
                elif change=='candidate':row['candidate_pool_sha256']='changed'
                elif change=='module_attachment':row['attachment_indices']=list(range(100,126))
                else:
                    for current in policies.values():current['outer_background_indices'][0]=0
                self.resign(path,plan)
                with self.assertRaises(ValueError):workflow.verify(path)

    def test_launch_is_detached_exclusive_and_bound_to_plan_digest(self):
        with TemporaryDirectory() as folder, patch.object(workflow, 'ROOT', Path(folder)), \
                patch.object(workflow.subprocess, 'Popen', return_value=Mock(pid=789)) as spawn:
            path, _, _, _ = self.fixture(Path(folder))
            result=workflow.launch(path)
            self.assertEqual(result['pid'],789)
            command=spawn.call_args.args[0]
            self.assertIn('--queue',command)
            self.assertEqual(command[-1],workflow.sha(path))
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            with self.assertRaises(FileExistsError):workflow.launch(path)
            self.assertEqual(spawn.call_count,1)

    def test_damaged_registration_never_spawns_background_work(self):
        with TemporaryDirectory() as folder, patch.object(workflow, 'ROOT', Path(folder)), \
                patch.object(workflow.subprocess, 'Popen') as spawn:
            path, _, _, _ = self.fixture(Path(folder));path.write_text(path.read_text()+'\n')
            with self.assertRaises(ValueError):workflow.launch(path)
            spawn.assert_not_called()

    def paired_fixture(self):
        result = dict(initialization=dict(
            fresh_model_sha256=dict(base_generator='g0',base_shadow='s0'),
            initialization_rng_sha256='initial_rng', post_initialization_rng_sha256='post_rng',
            clean_proxy_sha256=['proxy1','proxy2']),
            victim_record=dict(initialization_digest='victim0',
                initial_state_sha256='victim_state0',fit_rng_digest='victim_rng0'),
            selection=dict(outer_background_sha256='same_background'))
        return dict(full=deepcopy(result), no_residual=deepcopy(result))

    def test_matched_fresh_starts_and_background_pass(self):
        workflow.paired_checks(self.paired_fixture())

    def test_changed_attack_victim_rng_proxy_or_background_fails_pairing(self):
        changes=[('initialization',key) for key in ('fresh_model_sha256',
            'initialization_rng_sha256','post_initialization_rng_sha256','clean_proxy_sha256')]
        changes += [('victim_record',key) for key in ('initialization_digest',
            'initial_state_sha256','fit_rng_digest')]
        changes += [('selection','outer_background_sha256')]
        for section,key in changes:
            with self.subTest(section=section,key=key):
                records=self.paired_fixture();records['no_residual'][section][key]='changed'
                with self.assertRaises(ValueError):workflow.paired_checks(records)

    def test_contextual_wrappers_preserve_numerical_oom_and_integrity_error_types(self):
        for kind in (FloatingPointError, torch.cuda.OutOfMemoryError, ValueError, PermissionError):
            with self.subTest(kind=kind.__name__):
                cause=kind('original failure')
                middle=RuntimeError('training stage failed');middle.__cause__=cause
                outer=RuntimeError('case failed');outer.__cause__=middle
                self.assertEqual(workflow.root_error_type(outer),kind.__name__)
        self.assertEqual(workflow.root_error_type(RuntimeError('process exit')), 'RuntimeError')

    def test_cyclic_error_chain_does_not_hang_classifier(self):
        first=RuntimeError('a');second=RuntimeError('b')
        first.__cause__=second;second.__cause__=first
        self.assertEqual(workflow.root_error_type(first),'RuntimeError')


if __name__ == '__main__':unittest.main()
