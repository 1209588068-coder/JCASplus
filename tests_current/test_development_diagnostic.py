"""Synthetic-only fixed-weight diagnosis checks; never read real model/data assets."""
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import development_diagnostic as diagnostic
from spear_current.attention_context import attention_context
from spear_current.domain_generator import DomainConstrainedGenerator
from spear_current.role_data import build_contexts
from tests_current.fixtures_tensors import classifier, fixture, setup, snapshot


def synthetic_part(scenario="synthetic-train"):
    pairs, context, features = fixture(784)
    return dict(
        scenario_id=scenario, split="train", metadata={"feature_dimension": 784},
        canonical_pairs=pairs.numpy().copy(), edge_index=context.edge_index.numpy().copy(),
        x=(features / 200).numpy().copy(),
        y=np.asarray([0, 0, 1, -1, 0, 1], dtype=np.int64),
        eligible=np.asarray([True, True, True, False, True, True]),
        canonical_pair_track_ids=np.asarray([[f"track-{a}", f"track-{b}"] for a, b in pairs]),
        motion_regime=np.asarray([False, True, False, True, True, False]),
    )


def source(part, role="train"):
    return dict(scenario_id=part["scenario_id"], split_group_id=part["scenario_id"] + "-group",
                split="train", role=role)


def registered_bundle():
    return SimpleNamespace(config={"roles": {
        role: [dict(scenario_id=f"synthetic-{role}-{i}", split_group_id=f"group-{role}-{i}",
                    role=role, split="train") for i in range(count)]
        for role, count in (("train", 512), ("dev", 128), ("audit", 256))
    }})


class DevelopmentIdentityTests(unittest.TestCase):
    def test_target_is_existing_identity_sha_rule_without_model_calls(self):
        part = synthetic_part()
        eligible_negatives = [0, 1, 4]
        for seed in (diagnostic.TARGET_SEED, 17):
            expected = min(eligible_negatives, key=lambda i: hashlib.sha256(
                f"{seed}:spear-audit-target:{part['scenario_id']}:"
                f"{':'.join(part['canonical_pair_track_ids'][i])}".encode()
            ).hexdigest())
            with self.subTest(seed=seed), patch.object(diagnostic, "predict", side_effect=AssertionError("prediction access")), \
                    patch.object(diagnostic, "restore_generator", side_effect=AssertionError("model access")):
                self.assertEqual(diagnostic.choose_target(part, seed), expected)
                # Model outputs and feature values cannot determine eligibility or ranking.
                changed = deepcopy(part)
                changed["x"][:] = 9.0
                changed["clean_pred"] = np.ones(6, dtype=np.int64)
                changed["trigger_pred"] = np.zeros(6, dtype=np.int64)
                self.assertEqual(diagnostic.choose_target(changed, seed), expected)

    def test_target_has_no_fallback_when_eligible_negatives_are_absent(self):
        for kind in ("ineligible", "positive", "empty"):
            part = synthetic_part()
            if kind == "ineligible":
                part["eligible"][:] = False
            elif kind == "positive":
                part["y"][part["eligible"]] = 1
            else:
                for field in ("x", "y", "eligible", "canonical_pairs", "canonical_pair_track_ids", "motion_regime"):
                    part[field] = part[field][:0]
                part["edge_index"] = np.empty((2, 0), dtype=np.int64)
            with self.subTest(kind=kind):
                self.assertIsNone(diagnostic.choose_target(part))

    def test_metadata_keeps_local_global_indices_motion_and_role_membership(self):
        part = synthetic_part()
        rows = diagnostic.row_metadata(part, source(part), 10, "train", {11}, {10, 11, 13})
        self.assertEqual([row["pair_index"] for row in rows], list(range(6)))
        self.assertEqual([row["role_pair_index"] for row in rows], list(range(10, 16)))
        self.assertEqual([row["motion_regime"] for row in rows], ["slow", "moving", "slow", "moving", "moving", "slow"])
        self.assertEqual([row["is_attachment"] for row in rows], [False, True, False, False, False, False])
        self.assertEqual([row["is_outer"] for row in rows], [True, True, False, True, False, False])
        self.assertEqual(rows[3]["true_label"], -1)
        self.assertFalse(rows[3]["eligible"])
        self.assertEqual((rows[0]["src_track_id"], rows[0]["dst_track_id"]), ("track-0", "track-1"))
        self.assertTrue(all(row["scenario_id"] == part["scenario_id"] and row["role"] == "train" for row in rows))
        dev = diagnostic.row_metadata(part, source(part, "dev"), 10, "dev", set(range(10, 16)), set(range(10, 16)))
        self.assertTrue(all(not row["is_attachment"] and not row["is_outer"] for row in dev))

    def test_registered_sources_return_only_train_and_dev(self):
        bundle = registered_bundle()
        with patch.object(diagnostic, "read_role_part", side_effect=AssertionError("graph access")):
            result = diagnostic.registered_rows(bundle)
        self.assertEqual(set(result), {"train", "dev"})
        self.assertIs(result["train"], bundle.config["roles"]["train"])
        self.assertIs(result["dev"], bundle.config["roles"]["dev"])

    def test_registered_sources_reject_substitution_overlap_and_changed_counts(self):
        changes = (
            ("original validation", lambda roles: roles["train"][0].update(split="val")),
            ("original test", lambda roles: roles["dev"][0].update(split="test")),
            ("role replacement", lambda roles: roles["train"][0].update(role="audit")),
            ("cross-role scene", lambda roles: roles["dev"][0].update(scenario_id=roles["train"][0]["scenario_id"])),
            ("cross-role group", lambda roles: roles["audit"][0].update(split_group_id=roles["dev"][0]["split_group_id"])),
            ("within-role duplicate", lambda roles: roles["train"][1].update(scenario_id=roles["train"][0]["scenario_id"])),
            ("empty identity", lambda roles: roles["train"][0].update(split_group_id="")),
            ("population change", lambda roles: roles["dev"].pop()),
        )
        for name, change in changes:
            bundle = registered_bundle()
            change(bundle.config["roles"])
            with self.subTest(change=name), patch.object(diagnostic, "read_role_part", side_effect=AssertionError("graph access")):
                with self.assertRaises(ValueError):
                    diagnostic.registered_rows(bundle)


class DevelopmentOutputTests(unittest.TestCase):
    def test_output_must_be_new_child_of_results_and_preserves_existing_evidence(self):
        with TemporaryDirectory() as directory, patch.object(diagnostic, "ROOT", Path(directory)):
            root = Path(directory)
            for invalid in (root, root / "results", root / "elsewhere", root / "results" / ".." / "escape"):
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    diagnostic.output_directory(invalid)
            output = diagnostic.output_directory(root / "results" / "synthetic")
            self.assertTrue(output.is_dir())
            diagnostic._write(output / "summary.json", {"preserved": True})
            with self.assertRaises(FileExistsError):
                diagnostic.output_directory(output)
            with self.assertRaises(FileExistsError):
                diagnostic._write(output / "summary.json", {"preserved": False})
            self.assertEqual(json.loads((output / "summary.json").read_text()), {"preserved": True})
            diagnostic._csv(output / "rows.csv", [{"row": 1}])
            before = (output / "rows.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                diagnostic._csv(output / "rows.csv", [{"row": 2}])
            self.assertEqual((output / "rows.csv").read_bytes(), before)

    def test_output_resolves_symlinks_before_authorizing_directory(self):
        with TemporaryDirectory() as directory, patch.object(diagnostic, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "results").mkdir()
            (root / "elsewhere").mkdir()
            (root / "results" / "redirect").symlink_to(root / "elsewhere", target_is_directory=True)
            with self.assertRaises(ValueError):
                diagnostic.output_directory(root / "results" / "redirect" / "new")
            self.assertFalse((root / "elsewhere" / "new").exists())


class DevelopmentForwardTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(31)
            self.model = classifier().eval().requires_grad_(False)
        _, self.context, self.x = fixture(784)
        self.weights = torch.ones(self.context.edge_index.shape[1])

    def test_prediction_uses_eval_frozen_parameters_and_no_grad(self):
        original = snapshot(self.model)
        rng = torch.get_rng_state().clone()
        x = self.x.clone().requires_grad_(True)
        result = diagnostic.predict(self.model, x, self.context.edge_index, self.weights)
        call = self.model.calls[-1]
        self.assertFalse(call["grad_enabled"])
        self.assertFalse(call["training"])
        self.assertFalse(call["parameters_require_grad"])
        self.assertTrue(call["input_requires_grad"])
        np.testing.assert_allclose(result["p"], call["output"][:, 1].exp().numpy())
        np.testing.assert_array_equal(result["pred"], call["output"].argmax(1).numpy())
        np.testing.assert_allclose(result["nll"], -call["output"][:, 1].numpy())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(torch.equal(original[name], value) for name, value in self.model.state_dict().items()))

    def test_prediction_rejects_train_mode_submodule_or_trainable_parameter_before_forward(self):
        for kind in ("root", "child", "parameter"):
            model = deepcopy(self.model)
            if kind == "root":
                model.train()
            elif kind == "child":
                model.linear.train()
            else:
                next(model.parameters()).requires_grad_(True)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "frozen eval"):
                diagnostic.predict(model, self.x, self.context.edge_index, self.weights)
            self.assertEqual(model.calls, [])

    def test_prediction_rejects_invalid_binary_log_probabilities(self):
        for output in (torch.zeros((6, 3)), torch.full((6, 2), float("nan")),
                       torch.full((6, 2), float("inf")), torch.zeros((6, 2)),
                       torch.full((6, 2), 0.5)):
            with self.subTest(shape=output.shape, value=float(output.flatten()[0])), \
                    patch.object(self.model, "forward", return_value=output):
                with self.assertRaisesRegex(ValueError, "Invalid log-probabilities"):
                    diagnostic.predict(self.model, self.x, self.context.edge_index, self.weights)

    def test_prediction_preserves_argmax_tie_policy(self):
        output = torch.full((6, 2), -np.log(2), dtype=torch.float32)
        with patch.object(self.model, "forward", return_value=output):
            result = diagnostic.predict(self.model, self.x, self.context.edge_index, self.weights)
        np.testing.assert_array_equal(result["pred"], np.zeros(6, dtype=np.int64))
        np.testing.assert_allclose(result["p"], 0.5)


class DevelopmentInjectionTests(unittest.TestCase):
    def setUp(self):
        anchor, support, self.x, self.context, self.targets = setup()
        self.generator = DomainConstrainedGenerator(anchor, support, output_mode="domain_projection")
        self.generator.eval().requires_grad_(False)
        self.edges = self.context.edge_index
        self.weights = torch.ones(self.edges.shape[1])
        self.selected = self.generator.selected_coordinates[self.generator.hard_support]

    def check(self, changed, edges=None, weights=None):
        diagnostic.check_injection(self.x, changed, self.targets, self.generator, self.edges,
                                   self.edges if edges is None else edges, self.weights,
                                   self.weights if weights is None else weights)

    def test_accepts_fixed_support_injection_and_closed_domain_boundary(self):
        with torch.no_grad():
            injected = self.generator.inject(self.x, self.context, self.targets, self.weights, detach=True)
        self.check(injected.features, injected.edge_index, injected.edge_weight)
        changed = self.x.clone()
        changed[self.targets[0], self.selected[:2]] = torch.tensor([-10.0, 10.0])
        self.check(changed)
        self.check(self.x.clone())

    def test_rejects_nontarget_unsupported_nonfinite_domain_and_topology_changes(self):
        nontarget = next(i for i in range(len(self.x)) if i not in self.targets.tolist())
        unsupported = next(i for i in range(self.x.shape[1]) if i not in self.selected.tolist())
        for kind in ("nontarget", "unsupported", "nan", "infinity", "domain", "shape", "edges", "weights"):
            changed, edges, weights = self.x.clone(), self.edges.clone(), self.weights.clone()
            if kind == "nontarget":
                changed[nontarget, self.selected[0]] += 0.1
            elif kind == "unsupported":
                changed[self.targets[0], unsupported] += 0.1
            elif kind in ("nan", "infinity", "domain"):
                changed[self.targets[0], self.selected[0]] = {"nan": float("nan"), "infinity": float("inf"), "domain": 10.01}[kind]
            elif kind == "shape":
                changed = changed[:-1]
            elif kind == "edges":
                edges[0, 0] = (edges[0, 0] + 1) % len(self.x)
            else:
                weights[0] = 0.5
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "frozen scope, graph or domain"):
                self.check(changed, edges, weights)


class DevelopmentSyntheticRunTests(unittest.TestCase):
    def test_two_scene_run_produces_paired_cases_without_model_data_reads_or_training(self):
        anchor, support, *_ = setup()
        generator = DomainConstrainedGenerator(anchor, support, output_mode="domain_projection")
        generator.eval().requires_grad_(False)
        with torch.random.fork_rng():
            torch.manual_seed(41)
            models = {name: classifier().eval().requires_grad_(False) for name in diagnostic.MODELS}
        parts = {role: [synthetic_part("synthetic-" + role)] for role in ("train", "dev")}
        combined = {role: diagnostic.combine_parts(values) for role, values in parts.items()}
        typed = build_contexts(parts["train"], combined["train"], seed=17)
        context = attention_context(typed["typed"], typed["binding"]["scene_order"], seed=17)
        attachments, outer = torch.tensor([0, 1]), torch.tensor([0, 2, 3, 4])
        metadata = {role: diagnostic.row_metadata(values[0], source(values[0], role), 0, role,
                    set(attachments.tolist()), set(outer.tolist())) for role, values in parts.items()}
        inputs = dict(parts=parts, combined=combined, context=context, metadata=metadata,
                      indices={"idx_attach": attachments}, outer=outer,
                      targets={role: [diagnostic.choose_target(values[0])] for role, values in parts.items()})
        bundle = SimpleNamespace(config={"shuffle_seed": 17})
        before = diagnostic._input_hashes(inputs)
        rng = torch.get_rng_state().clone()
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(diagnostic, "restore_generator", return_value=generator))
            for name in diagnostic.MODELS:
                stack.enter_context(patch.object(diagnostic, "restore_" + name, return_value=models[name]))
            # Any path into actual asset or graph readers makes this synthetic test fail.
            for name in ("load_bundle", "load_normalizer", "read_role_part", "read_inputs"):
                stack.enter_context(patch.object(diagnostic, name, side_effect=AssertionError("real asset access")))
            stack.enter_context(patch.object(torch, "load", side_effect=AssertionError("checkpoint access")))
            stack.enter_context(patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")))
            stack.enter_context(patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimization forbidden")))
            stack.enter_context(patch.object(diagnostic, "_progress"))
            output = Path(directory) / "synthetic-seed"
            result = diagnostic.run_seed(bundle, inputs, 20260621, output)
            self.assertEqual(result["status"], "DEVELOPMENT_DIAGNOSTIC_COMPLETE")
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertTrue(result["weights_unchanged"] and result["inputs_unchanged"] and result["rng_unchanged"])
            self.assertEqual(set(result["cases"]), {"train_attachments_joint", "train_outer_joint",
                "train_outer_nonattachments_joint", "train_public_single", "dev_public_single",
                "train_public_nonattachment_single"})
            with (output / "responses.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for case, count in (("train_attachments_joint", 2), ("train_outer_joint", 4),
                                ("train_outer_nonattachments_joint", 3), ("train_public_single", 1),
                                ("dev_public_single", 1)):
                self.assertEqual(sum(row["case"] == case for row in rows), count)
            outer_unknown = next(row for row in rows if row["case"] == "train_outer_joint" and row["pair_index"] == "3")
            self.assertEqual((outer_unknown["true_label"], outer_unknown["eligible"]), ("-1", "False"))
            for role in ("train", "dev"):
                row = next(row for row in rows if row["case"] == role + "_public_single")
                self.assertEqual(int(row["pair_index"]), inputs["targets"][role][0])
                for name in diagnostic.MODELS:
                    self.assertIn(name + "_clean_p", row)
                    self.assertIn(name + "_trigger_p", row)
                    self.assertIn(name + "_target_nll", row)
            with (output / "clean_pair_scores.csv").open(newline="") as stream:
                clean = list(csv.DictReader(stream))
            self.assertEqual(len(clean), 10)
            self.assertEqual({row["role"] for row in clean}, {"train", "dev"})
            self.assertEqual(json.loads((output / "summary.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                diagnostic.run_seed(bundle, inputs, 20260621, output)
        self.assertEqual(before, diagnostic._input_hashes(inputs))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for model in models.values():
            self.assertTrue(model.calls)
            self.assertTrue(all(not call["grad_enabled"] and not call["training"]
                                and not call["parameters_require_grad"] for call in model.calls))
            self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
