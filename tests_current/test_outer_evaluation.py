"""Synthetic-only cached DEV checks; no real AV2 arrays or model assets."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import development_diagnostic as diagnostic
from spear_current import evaluation
from spear_current import outer_evaluation as outer
from spear_current.data import combine_parts
from tests_current.fixtures_tensors import fixture


class Toy784Classifier(torch.nn.Module):
    """Fixed scores expose hand-countable transitions after a one-column trigger."""
    def __init__(self, sign=1):
        super().__init__()
        self.sign = torch.nn.Parameter(torch.tensor(float(sign)), requires_grad=False)
        self.register_buffer("fixed_buffer", torch.tensor(0.0))
        self.calls = []
        self.eval()

    def forward(self, x, edges, weights):
        self.calls.append(dict(grad_enabled=torch.is_grad_enabled(), training=self.training))
        logits = self.sign * (x[:, 2] + x[:, 0].sum() * x[:, 3])
        return torch.stack((torch.zeros_like(logits), logits), dim=1).log_softmax(1)


class Toy784Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("selected_coordinates", torch.tensor([0]))
        self.register_buffer("hard_support", torch.tensor([True]))
        self.value = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.calls = []
        self.eval()

    def inject(self, x, context, targets, weights, detach=True):
        self.calls.append(dict(grad_enabled=torch.is_grad_enabled(), training=self.training,
                              targets=targets.tolist(), detach=detach))
        changed = x.clone()
        changed[targets, 0] = self.value
        return SimpleNamespace(features=changed, edge_index=context.edge_index, edge_weight=weights)


def scene(name, regime="slow", no_negative=False, empty=False):
    pairs, context, _ = fixture(784)
    part = dict(scenario_id=name, split="train", metadata={"feature_dimension": 784},
        x=np.zeros((6, 784), dtype=np.float32),
        canonical_pairs=pairs.numpy().copy(), edge_index=context.edge_index.numpy().copy(),
        canonical_pair_track_ids=np.asarray([[f"track-{a}", f"track-{b}"] for a, b in pairs]),
        y=np.zeros(6, dtype=np.int64), eligible=np.ones(6, dtype=bool),
        motion_regime=np.full(6, regime == "moving", dtype=bool))
    if empty:
        for key in ("x", "canonical_pairs", "canonical_pair_track_ids", "y", "eligible", "motion_regime"):
            part[key] = part[key][:0]
        part["edge_index"] = np.empty((2, 0), dtype=np.int64)
        return part
    if no_negative:
        part["y"][:] = 1
        part["y"][-1], part["eligible"][-1] = -1, False
        part["x"][:, 2] = [2, -2, 2, -2, 2, -2]
        return part
    target = diagnostic.choose_target(part)
    adjacent = sorted(set(context.edge_index[1, context.edge_index[0] == target].tolist()))
    nonincident = next(i for i in range(6) if i != target and i not in adjacent)
    a, b, c, d = adjacent
    # Two adjacent negatives: only a flips, b is already positive. Two adjacent
    # positives: only c is suppressed, d is already negative. Both denominators
    # must include both rows. The one nonincident negative also flips.
    part["y"][[c, d]] = 1
    part["x"][:, 2] = -2
    part["x"][[b, c], 2] = 2
    part["x"][[target, a, nonincident], 3] = 4
    part["x"][c, 3] = -4
    if regime == "moving":
        part["x"][target, 2], part["x"][target, 3] = 2, 0
    assert target == diagnostic.choose_target(part)
    return part


class DevOnlyMapping(dict):
    def __getitem__(self, key):
        if key != "dev":
            raise AssertionError("Non-DEV cached data accessed: " + key)
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("All-role traversal forbidden")

    def items(self):
        raise AssertionError("All-role traversal forbidden")


def prepared(parts=None):
    if parts is None:
        parts = [scene("synthetic-slow"), scene("synthetic-moving", "moving"),
                 scene("synthetic-no-negative", no_negative=True)]
    sources = [dict(scenario_id=p["scenario_id"], split_group_id=p["scenario_id"] + "-group",
                    role="dev", split="train") for p in parts]
    combined = combine_parts(parts)
    metadata = []
    for part, row, offset in zip(parts, sources, combined["offsets"]):
        metadata.extend(diagnostic.row_metadata(part, row, offset, "dev", set(), set()))
    bundle = SimpleNamespace(config=dict(shuffle_seed=17, roles=DevOnlyMapping(dev=sources)))
    inputs = {key: DevOnlyMapping(dev=value) for key, value in dict(parts=parts, combined=combined,
        metadata=metadata, targets=[diagnostic.choose_target(part) for part in parts]).items()}
    models = {name: Toy784Classifier(-1 if name == "clean_reference" else 1) for name in outer.MODELS}
    return bundle, inputs, Toy784Generator(), models


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class OuterEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_known_target_strata_clean_metrics_and_collateral_denominators(self):
        bundle, inputs, generator, models = prepared()
        events = []
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dev"
            result = outer.evaluate_dev(bundle, inputs, generator, models, output,
                                        progress=lambda **event: events.append(event))
            targets = read_csv(output / "targets.csv")
            clean = read_csv(output / "clean_pair_scores.csv")
            triggered = read_csv(output / "triggered_pair_scores.csv")
            self.assertEqual(result["dev_counts"], dict(registered_scenes=3, processed_scenes=3,
                scenes_without_eligible_negative_target=1, all_pair_nodes=18, supervised_pairs=17,
                targets=2, triggered_supervised_pair_rows=12))
            self.assertEqual((len(targets), len(clean), len(triggered)), (2, 17, 12))
            for rows in (targets, clean, triggered):
                self.assertTrue(all(row["role"] == "dev" and row["original_split"] == "train"
                                    and row["is_attachment"] == row["is_outer"] == "False" for row in rows))
                self.assertTrue(all(any(key.startswith(name + "_") for key in rows[0]) for name in outer.MODELS))
            self.assertEqual({r["relation"] for r in triggered}, {"target", "adjacent", "nonincident"})
            self.assertEqual([int(row["pair_index"]) for row in targets], inputs["targets"]["dev"][:2])
            for name in ("surrogate", "victim"):
                metric = result["metrics"][name]
                overall = metric["targets"]["all"]
                self.assertEqual(overall["targets"], 2)
                self.assertEqual(overall["incremental_flips"], 1)
                self.assertEqual(overall["absolute_asr"], 1)
                self.assertEqual(overall["incremental_asr"], .5)
                self.assertEqual(overall["conditional_flip"], 1)
                self.assertEqual(overall["clean_activation_rate"], .5)
                self.assertEqual(metric["targets"]["slow"]["incremental_asr"], 1)
                self.assertEqual(metric["targets"]["moving"]["incremental_asr"], 0)
                self.assertEqual(metric["clean_confusion_matrix"], dict(true_negative=5,
                    false_positive=3, false_negative=4, true_positive=5))
                self.assertEqual(metric["clean_pair_positive_recall"], 5 / 9)
                self.assertAlmostEqual(metric["clean_pair_auc"], 42.5 / 72)
                self.assertEqual(metric["collateral"], dict(adj_negative=4, adj_flip=2,
                    nonincident_negative=2, nonincident_flip=2, adj_positive=4, adj_suppressed=2,
                    adjacent_incremental_fp=.5, nonincident_incremental_fp=1,
                    adjacent_positive_suppression=.5))
            reference = result["metrics"]["clean_reference"]
            self.assertEqual(reference["targets"]["all"]["absolute_asr"], 0)
            self.assertEqual(reference["collateral"]["adj_flip"], 0)
            self.assertEqual(reference["collateral"]["adj_suppressed"], 0)
            self.assertEqual(result["raw_targets"]["rows"], 2)
            self.assertEqual(result["raw_clean_scores"]["rows"], 17)
            self.assertEqual(result["raw_triggered_pair_scores"]["rows"], 12)
            self.assertEqual(result["target_identity_sha256"], result["population_identity_sha256"]["targets"])
            self.assertEqual(set(result["population_identity_sha256"]),
                             {"scenes", "eligible_clean_pairs", "targets", "triggered_pairs"})
            json.dumps(result, allow_nan=False)
        self.assertEqual(events, [dict(stage="dev_original_TRAIN_groups", completed=n, total=3) for n in (0, 3)])

    def test_empty_and_no_negative_scenes_still_have_accounting_and_csv_headers(self):
        for part in (scene("no-negative", no_negative=True), scene("empty", empty=True)):
            bundle, inputs, generator, models = prepared([part])
            with self.subTest(scene=part["scenario_id"]), TemporaryDirectory() as directory:
                output = Path(directory) / "dev"
                result = outer.evaluate_dev(bundle, inputs, generator, models, output)
                self.assertEqual(result["dev_counts"]["scenes_without_eligible_negative_target"], 1)
                self.assertEqual(result["dev_counts"]["processed_scenes"], 1)
                self.assertEqual(read_csv(output / "targets.csv"), [])
                self.assertEqual(read_csv(output / "triggered_pair_scores.csv"), [])
                self.assertIn("surrogate_clean_p", (output / "targets.csv").read_text())
                self.assertEqual(generator.calls, [])
                for metric in result["metrics"].values():
                    self.assertIsNone(metric["targets"]["all"]["absolute_asr"])
                    self.assertIsNone(metric["collateral"]["adjacent_incremental_fp"])
                    self.assertIsNone(metric["clean_pair_auc"])

    def test_no_loading_training_gradients_rng_or_state_changes_and_exact_dev_only_access(self):
        bundle, inputs, generator, models = prepared()
        inputs["combined"]["dev"]["x"].requires_grad_(True)
        before_input = outer._fingerprint(outer._dev_view(bundle, inputs))
        before_rng = outer._rng_fingerprint()
        before_models = {name: outer.state_hash(model) for name, model in dict(generator=generator, **models).items()}
        with TemporaryDirectory() as directory, ExitStack() as stack:
            for name in ("load_bundle", "load_normalizer", "read_role_part", "read_inputs",
                         "restore_generator", "restore_victim", "restore_surrogate", "restore_clean_reference"):
                stack.enter_context(patch.object(diagnostic, name, side_effect=AssertionError("loading forbidden")))
            stack.enter_context(patch.object(evaluation, "evaluate", side_effect=AssertionError("audit evaluator forbidden")))
            stack.enter_context(patch.object(torch, "load", side_effect=AssertionError("checkpoint forbidden")))
            stack.enter_context(patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")))
            stack.enter_context(patch.object(torch.autograd, "grad", side_effect=AssertionError("autograd forbidden")))
            stack.enter_context(patch.object(torch.optim.Adam, "__init__", side_effect=AssertionError("optimizer forbidden")))
            stack.enter_context(patch.object(torch.optim.SGD, "__init__", side_effect=AssertionError("optimizer forbidden")))
            result = outer.evaluate_dev(bundle, inputs, generator, models, Path(directory) / "dev")
        self.assertEqual(before_input, outer._fingerprint(outer._dev_view(bundle, inputs)))
        self.assertEqual(before_rng, outer._rng_fingerprint())
        for name, model in dict(generator=generator, **models).items():
            self.assertEqual(before_models[name], outer.state_hash(model))
            self.assertTrue(model.calls)
            self.assertTrue(all(not call["grad_enabled"] and not call["training"] for call in model.calls))
            self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(result["optimizer_steps"], 0)
        for key in ("weights_unchanged", "inputs_unchanged", "rng_unchanged"):
            self.assertTrue(result[key])
        self.assertFalse(result["gradients_created"])

    def test_all_scene_metadata_corruption_rejected_before_first_prediction(self):
        # Corrupt the LAST scene: validation must precede even first-scene inference.
        changes = {
            "source role": lambda b, i: b.config["roles"]["dev"][-1].update(role="audit"),
            "source split": lambda b, i: b.config["roles"]["dev"][-1].update(split="val"),
            "source scenario": lambda b, i: b.config["roles"]["dev"][-1].update(scenario_id="substitute"),
            "part split": lambda b, i: i["parts"]["dev"][-1].update(split="test"),
            "part role": lambda b, i: i["parts"]["dev"][-1].update(role="confirmation"),
            "part scenario": lambda b, i: i["parts"]["dev"][-1].update(scenario_id="substitute"),
            "target": lambda b, i: i["targets"]["dev"].__setitem__(-1, 0),
            "label": lambda b, i: i["metadata"]["dev"][-1].update(true_label=0),
            "eligible": lambda b, i: i["metadata"]["dev"][-1].update(eligible=True),
            "motion": lambda b, i: i["metadata"]["dev"][-1].update(motion_regime="moving"),
            "src id": lambda b, i: i["metadata"]["dev"][-1].update(src_track_id="wrong"),
            "dst id": lambda b, i: i["metadata"]["dev"][-1].update(dst_track_id="wrong"),
            "row role": lambda b, i: i["metadata"]["dev"][-1].update(role="train"),
            "row split": lambda b, i: i["metadata"]["dev"][-1].update(original_split="test"),
            "membership": lambda b, i: i["metadata"]["dev"][-1].update(is_outer=True),
            "combined label": lambda b, i: i["combined"]["dev"]["y"].__setitem__(-1, 0),
            "offset": lambda b, i: i["combined"]["dev"]["offsets"].__setitem__(-1, 0),
        }
        for name, change in changes.items():
            bundle, inputs, generator, models = prepared()
            change(bundle, inputs)
            with self.subTest(change=name), TemporaryDirectory() as directory, \
                    patch.object(diagnostic, "predict", side_effect=AssertionError("prediction too early")) as predict:
                output = Path(directory) / "dev"
                with self.assertRaises(ValueError):
                    outer.evaluate_dev(bundle, inputs, generator, models, output)
                predict.assert_not_called()
                self.assertFalse(output.exists())

    def test_alternative_eligible_negative_is_rejected_even_without_prediction_conditioning(self):
        bundle, inputs, generator, models = prepared()
        part = inputs["parts"]["dev"][0]
        selected = inputs["targets"]["dev"][0]
        inputs["targets"]["dev"][0] = next(int(i) for i in np.flatnonzero(part["eligible"] & (part["y"] == 0)) if i != selected)
        with TemporaryDirectory() as directory, patch.object(diagnostic, "predict") as predict:
            with self.assertRaisesRegex(ValueError, "fixed identity-hash"):
                outer.evaluate_dev(bundle, inputs, generator, models, Path(directory) / "dev")
            predict.assert_not_called()

    def test_exact_model_names_and_frozen_eval_no_grad_required_before_prediction(self):
        for change in ("extra", "missing", "training", "trainable", "stale grad", "generator training"):
            bundle, inputs, generator, models = prepared()
            if change == "extra":
                models["other"] = models["victim"]
            elif change == "missing":
                del models["surrogate"]
            elif change == "training":
                models["victim"].train()
            elif change == "trainable":
                models["victim"].sign.requires_grad_(True)
            elif change == "stale grad":
                models["victim"].sign.grad = torch.ones_like(models["victim"].sign)
            else:
                generator.train()
            with self.subTest(change=change), TemporaryDirectory() as directory, patch.object(diagnostic, "predict") as predict:
                with self.assertRaises(ValueError):
                    outer.evaluate_dev(bundle, inputs, generator, models, Path(directory) / "dev")
                predict.assert_not_called()

    def test_detects_model_input_and_inference_rng_mutation(self):
        for change in ("weight", "buffer", "input", "torch rng", "numpy rng", "python rng", "gradient"):
            bundle, inputs, generator, models = prepared()
            model = models["victim"]
            forward = model.forward

            def bad_forward(x, edges, weights):
                result = forward(x, edges, weights)
                if change == "weight":
                    model.sign.add_(.1)
                elif change == "buffer":
                    model.fixed_buffer.add_(1)
                elif change == "input":
                    inputs["combined"]["dev"]["x"][0, 10] += .1
                elif change == "torch rng":
                    torch.rand(1)
                elif change == "numpy rng":
                    np.random.random()
                elif change == "python rng":
                    random.random()
                else:
                    model.sign.grad = torch.ones_like(model.sign)
                return result

            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                with self.subTest(change=change), TemporaryDirectory() as directory, torch.random.fork_rng(), \
                        patch.object(model, "forward", side_effect=bad_forward):
                    with self.assertRaises(ValueError):
                        outer.evaluate_dev(bundle, inputs, generator, models, Path(directory) / "dev")
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    def test_population_identity_is_prediction_independent_and_changes_with_identity(self):
        bundle, inputs, generator, models = prepared()
        with TemporaryDirectory() as directory:
            result = outer.evaluate_dev(bundle, inputs, generator, models, Path(directory) / "baseline")
            models = {name: Toy784Classifier(-1) for name in outer.MODELS}
            candidate = outer.evaluate_dev(bundle, inputs, generator, models, Path(directory) / "candidate")
            self.assertEqual(result["population_identity_sha256"], candidate["population_identity_sha256"])
            self.assertNotEqual(result["raw_targets"]["sha256"], candidate["raw_targets"]["sha256"])
            b2, i2, g2, m2 = prepared([scene("different-scene")])
            changed = outer.evaluate_dev(b2, i2, g2, m2, Path(directory) / "changed")
            self.assertTrue(all(result["population_identity_sha256"][key] != changed["population_identity_sha256"][key]
                                for key in result["population_identity_sha256"]))

    def test_output_directory_must_be_new_and_existing_evidence_is_preserved(self):
        bundle, inputs, generator, models = prepared()
        with TemporaryDirectory() as directory:
            empty = Path(directory) / "existing-empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                outer.evaluate_dev(bundle, inputs, generator, models, empty)
            self.assertEqual(list(empty.iterdir()), [])
            output = Path(directory) / "completed"
            outer.evaluate_dev(bundle, inputs, generator, models, output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaises(FileExistsError):
                outer.evaluate_dev(bundle, inputs, generator, models, output)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_prediction_failure_propagates_without_success_summary(self):
        bundle, inputs, generator, models = prepared()
        with TemporaryDirectory() as directory, patch.object(diagnostic, "predict", side_effect=RuntimeError("synthetic failure")):
            output = Path(directory) / "dev"
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                outer.evaluate_dev(bundle, inputs, generator, models, output)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
