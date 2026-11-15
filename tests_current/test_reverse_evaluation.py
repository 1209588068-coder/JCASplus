"""Synthetic-only reverse evaluation; no AV2 graphs or checkpoints are loaded."""
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spear_current import development_diagnostic, evaluation, reverse_audit
from spear_current import reverse_evaluation as reverse
from spear_current.data import combine_parts
from tests_current.test_outer_evaluation import DevOnlyMapping, Toy784Generator


class Classifier(torch.nn.Module):
    def __init__(self, sign=1., trigger_scale=1.):
        super().__init__()
        self.sign = torch.nn.Parameter(torch.tensor(sign), requires_grad=False)
        self.register_buffer("trigger_scale", torch.tensor(trigger_scale))
        self.calls = []
        self.eval()

    def forward(self, x, edges, weights):
        self.calls.append(dict(features=x.detach().clone(), address=x.data_ptr(),
                              grad_enabled=torch.is_grad_enabled(), training=self.training))
        logit = self.sign * (x[:, 2] + self.trigger_scale * x[:, 0].sum() * x[:, 3])
        return torch.stack((torch.zeros_like(logit), logit), dim=1).log_softmax(1)


def scene(regime="slow", no_positive=False, empty=False):
    pairs = np.asarray([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [4, 5], [6, 7]], dtype=np.int64)
    edges = np.asarray([(i, j) for i, left in enumerate(pairs) for j, right in enumerate(pairs)
                        if i != j and set(left) & set(right)], dtype=np.int64).T
    ids = np.asarray([[f"track-{a}", f"track-{b}"] for a, b in pairs])
    # Pick a deterministic synthetic scene identity whose frozen hash selects
    # central pair zero. This manufactures a toy topology, not a study target.
    for index in range(1000):
        name = f"synthetic-reverse-{regime}-{index}"
        rank = lambda i: hashlib.sha256(
            f"{reverse.TARGET_SEED}:spear-audit-target:{name}:{ids[i, 0]}:{ids[i, 1]}".encode()).hexdigest()
        if min((0, 1, 2, 5), key=rank) == 0:
            break
    part = dict(scenario_id=name, split="train", metadata={"feature_dimension": 784},
        x=np.zeros((7, 784), dtype=np.float32), canonical_pairs=pairs, edge_index=edges,
        canonical_pair_track_ids=ids, y=np.asarray([1, 1, 1, 0, 0, 1, 0], dtype=np.int64),
        eligible=np.ones(7, dtype=bool), motion_regime=np.full(7, regime == "moving", dtype=bool))
    part["x"][:, 2] = [2, 2, -2, -2, 2, 2, -2]
    part["x"][:, 3] = [-4, -4, 0, 4, 0, -4, 4]
    if regime == "moving":
        part["x"][0, 2], part["x"][0, 3] = -2, 4  # Repair an existing target miss.
    if no_positive:
        part["scenario_id"] = "synthetic-no-positive"
        part["y"][:] = 0
        part["y"][-1], part["eligible"][-1] = -1, False
        part["x"][:, 2] = [-2, 2, -2, 2, -2, 2, -2]
    if empty:
        part["scenario_id"] = "synthetic-empty"
        for key in ("x", "y", "eligible", "canonical_pairs", "canonical_pair_track_ids", "motion_regime"):
            part[key] = part[key][:0]
        part["edge_index"] = np.empty((2, 0), dtype=np.int64)
    return part


def prepared(parts=None, historical=False):
    parts = parts if parts is not None else [scene(), scene("moving"), scene(no_positive=True)]
    sources = [dict(scenario_id=p["scenario_id"], split_group_id=p["scenario_id"] + "-group",
                    role="dev", split="train") for p in parts]
    combined = combine_parts(parts)
    metadata = [row for part, source, offset in zip(parts, sources, combined["offsets"])
                for row in development_diagnostic.row_metadata(part, source, offset, "dev", set(), set())]
    targets = reverse_audit.select_positive_targets(dict(train=[], dev=metadata), dict(train=[], dev=sources))["dev"]
    diagnostic = {key: DevOnlyMapping(dev=value) for key, value in
                  dict(parts=parts, combined=combined, metadata=metadata).items()}
    # Any accidental reuse of the old negative-target field must fail.
    diagnostic["targets"] = None
    bundle = SimpleNamespace(config=dict(shuffle_seed=17, roles=DevOnlyMapping(dev=sources)))
    models = dict(victim=Classifier(), matched_clean=Classifier(trigger_scale=0.), label_only=Classifier(sign=-1.))
    if historical:
        models["historical_reference"] = Classifier(trigger_scale=.5)
    return bundle, diagnostic, Toy784Generator(), models, targets


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class ReverseEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_reverse_targets_repair_counts_clean_metrics_and_collateral_exclude_target(self):
        bundle, diagnostic, generator, models, targets = prepared()
        progress = []
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dev"
            result = reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output,
                                                  progress=lambda **event: progress.append(event))
            self.assertEqual(result["dev_counts"], dict(registered_scenes=3, processed_scenes=3,
                scenes_without_eligible_positive_target=1, all_pair_nodes=21, supervised_pairs=20,
                targets=2, triggered_supervised_pair_rows=14))
            metric = result["metrics"]["victim"]
            self.assertEqual(metric["targets"]["all"], dict(n=2, clean_detected=1, baseline_misses=1,
                induced_misses_1to0=1, corrected_misses_0to1=1, triggered_misses=1,
                baseline_fnr=.5, absolute_suppression=.5, incremental_asr=.5,
                conditional_suppression=1, net_fnr_change=0, clean_recall_ceiling=.5))
            self.assertEqual(metric["targets"]["slow"]["incremental_asr"], 1)
            self.assertEqual(metric["targets"]["moving"]["corrected_misses_0to1"], 1)
            self.assertIsNone(metric["targets"]["moving"]["conditional_suppression"])
            self.assertEqual(metric["clean_confusion_matrix"], dict(true_negative=7, false_positive=5,
                false_negative=3, true_positive=5))
            self.assertEqual(metric["clean_pair_positive_recall"], 5 / 8)
            self.assertAlmostEqual(metric["clean_pair_auc"], 58 / 96)
            self.assertEqual(metric["collateral"]["adjacent"], dict(positive_pairs=4,
                induced_misses_1to0=2, positive_suppression=.5, negative_pairs=4,
                negative_activations_0to1=2, negative_activation=.5))
            self.assertEqual(metric["collateral"]["nonincident"], dict(positive_pairs=2,
                induced_misses_1to0=2, positive_suppression=1, negative_pairs=2,
                negative_activations_0to1=2, negative_activation=1))
            self.assertEqual(result["metrics"]["matched_clean"]["targets"]["all"]["induced_misses_1to0"], 0)
            self.assertEqual(result["metrics"]["label_only"]["targets"]["moving"]["induced_misses_1to0"], 1)
            self.assertAlmostEqual(metric["target_loss"]["slow"]["mean_nll_class_0"], math.log1p(math.exp(-2)), places=6)
            self.assertAlmostEqual(metric["target_loss"]["moving"]["mean_nll_class_0"], math.log1p(math.exp(2)), places=6)
            target_rows = rows(output / "targets.csv")
            self.assertEqual(len(target_rows), 2)
            self.assertEqual([int(r["pair_index"]) for r in target_rows], [0, 0])
            self.assertTrue(all(r["true_label"] == "1" and r["eligible"] == "True"
                and r["role"] == "dev" and r["original_split"] == "train"
                and r["is_attachment"] == r["is_outer"] == "False" for r in target_rows))
            self.assertEqual(len(rows(output / "clean_pair_scores.csv")), 20)
            self.assertEqual(len(rows(output / "triggered_pair_scores.csv")), 14)
            self.assertEqual(json.loads((output / "summary.json").read_text()), result)
        self.assertEqual(progress, [dict(stage="reverse_dev_original_TRAIN_groups", completed=n, total=3) for n in (0, 3)])

    def test_one_shared_injection_for_all_models_and_optional_reference(self):
        bundle, diagnostic, generator, models, targets = prepared(historical=True)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dev"
            result = reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output)
            self.assertIn("historical_reference_clean_p", rows(output / "targets.csv")[0])
        self.assertEqual(len(generator.calls), len(targets))
        self.assertEqual(result["generator_injections"], 2)
        self.assertEqual(set(result["metrics"]), set(models))
        for call_index in (1, 3):
            snapshots = [model.calls[call_index] for model in models.values()]
            self.assertEqual(len({r["address"] for r in snapshots}), 1)
            self.assertTrue(all(torch.equal(snapshots[0]["features"], r["features"]) for r in snapshots))
        self.assertTrue(all(call["targets"] == [0] for call in generator.calls))

    def test_empty_and_no_positive_scenes_keep_clean_coverage_and_undefined_rates(self):
        for part in (scene(no_positive=True), scene(empty=True)):
            bundle, diagnostic, generator, models, targets = prepared([part])
            with self.subTest(scene=part["scenario_id"]), TemporaryDirectory() as directory:
                output = Path(directory) / "dev"
                result = reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output)
                self.assertEqual(result["dev_counts"]["scenes_without_eligible_positive_target"], 1)
                self.assertEqual(result["dev_counts"]["processed_scenes"], 1)
                self.assertEqual(rows(output / "targets.csv"), [])
                self.assertIn("victim_target_nll", (output / "targets.csv").read_text())
                self.assertEqual(generator.calls, [])
                for metric in result["metrics"].values():
                    self.assertIsNone(metric["targets"]["all"]["incremental_asr"])
                    self.assertIsNone(metric["collateral"]["adjacent"]["positive_suppression"])
                    self.assertIsNone(metric["target_loss"]["all"]["mean_nll_class_0"])

    def test_target_zero_nll_remains_finite_when_risk_probability_rounds_to_one(self):
        model = Classifier()
        x = torch.zeros((1, 784))
        x[0, 2] = 1000
        result = reverse._predict(model, x, torch.empty((2, 0), dtype=torch.long), torch.empty(0))
        self.assertEqual(result["p"].tolist(), [1.])
        self.assertEqual(result["nll"].tolist(), [1000.])

    def test_invalid_prediction_shape_or_log_probabilities_are_rejected(self):
        model = Classifier()
        x = torch.zeros((1, 784))
        edges, weights = torch.empty((2, 0), dtype=torch.long), torch.empty(0)
        for output in (torch.zeros((1, 3)), torch.zeros((1, 2)), torch.full((1, 2), float("nan"))):
            with self.subTest(output=output), patch.object(model, "forward", return_value=output):
                with self.assertRaisesRegex(ValueError, "finite binary log-probabilities"):
                    reverse._predict(model, x, edges, weights)

    def test_invalid_frozen_targets_and_late_scene_metadata_rejected_before_prediction(self):
        changes = {
            "omit target": lambda b, d, t: t.pop(),
            "reorder targets": lambda b, d, t: t.reverse(),
            "duplicate target": lambda b, d, t: t.append(dict(t[0])),
            "negative label": lambda b, d, t: t[-1].update(true_label=0),
            "unknown label": lambda b, d, t: t[-1].update(true_label=-1),
            "ineligible": lambda b, d, t: t[-1].update(eligible=False),
            "wrong positive": lambda b, d, t: t.__setitem__(0, dict(d["metadata"]["dev"][1])),
            "track identity": lambda b, d, t: t[-1].update(src_track_id="wrong"),
            "target role": lambda b, d, t: t[-1].update(role="audit"),
            "source split": lambda b, d, t: b.config["roles"]["dev"][-1].update(split="val"),
            "source role": lambda b, d, t: b.config["roles"]["dev"][-1].update(role="confirmation"),
            "part scenario": lambda b, d, t: d["parts"]["dev"][-1].update(scenario_id="wrong"),
            "part split": lambda b, d, t: d["parts"]["dev"][-1].update(split="test"),
            "metadata label": lambda b, d, t: d["metadata"]["dev"][-1].update(true_label=0),
            "metadata motion": lambda b, d, t: d["metadata"]["dev"][-1].update(motion_regime="moving"),
            "combined labels": lambda b, d, t: d["combined"]["dev"]["y"].__setitem__(-1, 0),
            "offset": lambda b, d, t: d["combined"]["dev"]["offsets"].__setitem__(-1, 0),
        }
        for name, change in changes.items():
            bundle, diagnostic, generator, models, targets = prepared()
            change(bundle, diagnostic, targets)
            with self.subTest(change=name), TemporaryDirectory() as directory, \
                    patch.object(reverse, "_predict", side_effect=AssertionError("premature prediction")) as predict:
                output = Path(directory) / "dev"
                with self.assertRaises(ValueError):
                    reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output)
                predict.assert_not_called()
                self.assertFalse(output.exists())

    def test_frozen_models_required_and_model_names_are_explicit(self):
        for change in ("missing", "extra", "training", "parameter", "stale grad", "generator"):
            bundle, diagnostic, generator, models, targets = prepared()
            if change == "missing":
                del models["label_only"]
            elif change == "extra":
                models["clean_reference"] = models["matched_clean"]
            elif change == "training":
                models["victim"].train()
            elif change == "parameter":
                models["victim"].sign.requires_grad_(True)
            elif change == "stale grad":
                models["victim"].sign.grad = torch.ones_like(models["victim"].sign)
            else:
                generator.train()
            with self.subTest(change=change), TemporaryDirectory() as directory, patch.object(reverse, "_predict") as predict:
                with self.assertRaises(ValueError):
                    reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, Path(directory) / "dev")
                predict.assert_not_called()

    def test_no_loading_optimization_gradients_state_or_rng_changes(self):
        bundle, diagnostic, generator, models, targets = prepared()
        diagnostic["combined"]["dev"]["x"].requires_grad_(True)
        before = reverse._fingerprint(reverse._view(bundle, diagnostic, targets))
        rng = reverse._rng_fingerprint()
        state = {name: reverse.state_hash(model) for name, model in dict(generator=generator, **models).items()}
        with TemporaryDirectory() as directory, ExitStack() as stack:
            for name in ("load_bundle", "read_inputs", "load_normalizer", "read_role_part",
                         "restore_generator", "restore_surrogate", "restore_victim", "restore_clean_reference"):
                stack.enter_context(patch.object(development_diagnostic, name, side_effect=AssertionError("loading forbidden")))
            stack.enter_context(patch.object(reverse_audit, "run_audit", side_effect=AssertionError("workflow forbidden")))
            stack.enter_context(patch.object(evaluation, "evaluate", side_effect=AssertionError("audit forbidden")))
            stack.enter_context(patch.object(torch, "load", side_effect=AssertionError("checkpoint forbidden")))
            stack.enter_context(patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")))
            stack.enter_context(patch.object(torch.optim.Adam, "__init__", side_effect=AssertionError("optimizer forbidden")))
            result = reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, Path(directory) / "dev")
        self.assertEqual(before, reverse._fingerprint(reverse._view(bundle, diagnostic, targets)))
        self.assertEqual(rng, reverse._rng_fingerprint())
        for name, model in dict(generator=generator, **models).items():
            self.assertEqual(state[name], reverse.state_hash(model))
            self.assertTrue(all(not call["training"] and not call["grad_enabled"] for call in model.calls))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertTrue(result["inputs_unchanged"] and result["weights_unchanged"] and result["rng_unchanged"])
        self.assertEqual(result["optimizer_steps"], 0)

    def test_state_input_rng_mutation_and_bad_injection_fail_without_success_summary(self):
        for change in ("weight", "input", "prediction input", "rng", "outside support", "topology"):
            bundle, diagnostic, generator, models, targets = prepared()
            forward, inject = models["victim"].forward, generator.inject

            def bad_forward(x, edges, weights):
                result = forward(x, edges, weights)
                if change == "weight":
                    models["victim"].sign.add_(.1)
                elif change == "input":
                    diagnostic["combined"]["dev"]["x"][0, 10] += .1
                elif change == "prediction input":
                    x[0, 10] += .1
                elif change == "rng":
                    torch.rand(1)
                return result

            def bad_inject(x, context, target, weights, detach=True):
                result = inject(x, context, target, weights, detach=detach)
                if change == "outside support":
                    result.features[0, 10] += .1
                elif change == "topology":
                    result.edge_index = result.edge_index.clone()
                    result.edge_index[0, 0] = (result.edge_index[0, 0] + 1) % len(x)
                return result

            with self.subTest(change=change), TemporaryDirectory() as directory, torch.random.fork_rng(), \
                    patch.object(models["victim"], "forward", side_effect=bad_forward), \
                    patch.object(generator, "inject", side_effect=bad_inject):
                output = Path(directory) / "dev"
                with self.assertRaises(ValueError):
                    reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output)
                self.assertFalse((output / "summary.json").exists())

    def test_population_hashes_are_independent_of_models_and_outputs_are_exclusive(self):
        bundle, diagnostic, generator, models, targets = prepared()
        target_before = deepcopy(targets)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dev"
            first = reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output)
            models = {name: Classifier(trigger_scale=0.) for name in models}
            second = reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, Path(directory) / "different")
            self.assertEqual(first["population_identity_sha256"], second["population_identity_sha256"])
            self.assertNotEqual(first["raw_targets"]["sha256"], second["raw_targets"]["sha256"])
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            empty = Path(directory) / "existing-empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets, empty)
        self.assertEqual(targets, target_before)

    def test_summary_integrates_with_pilot_review_without_loading_or_starting_jobs(self):
        from spear_current import reverse_pilot

        bundle, diagnostic, generator, models, targets = prepared(historical=True)
        with TemporaryDirectory() as directory, ExitStack() as stack:
            for name in ("run", "load_bundle", "read_inputs", "preflight"):
                stack.enter_context(patch.object(reverse_pilot, name, side_effect=AssertionError("job forbidden")))
            summary = reverse.evaluate_reverse_dev(bundle, diagnostic, generator, models, targets,
                                                   Path(directory) / "dev")
            review = reverse_pilot.review_evaluation(summary)
        self.assertEqual(review["status"], "REVERSE_PILOT_COMPLETE_REVIEW_REQUIRED")
        self.assertFalse(review["acceptable_performance"])
        self.assertFalse(review["automatic_followup_jobs"])
        self.assertTrue(review["gates"]["conditional_suppression"]["passed"])
        self.assertFalse(review["gates"]["clean_detected_targets"]["passed"])
        self.assertEqual(review["target_metrics"]["victim"], summary["metrics"]["victim"]["targets"])


if __name__ == "__main__":
    unittest.main()
