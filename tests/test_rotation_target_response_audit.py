import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from jcas.core.poison import sha256_file
from jcas.workflows import rotation_target_response_audit as audit


def role_table():
    rows = []
    for slots in (2, 3, 4):
        for slot in range(slots):
            rows.append({
                "scenario_id": f"scene{slots}", "src": 0, "dst": slot + 1,
                "role": "train", "slot": slot, "split": "train",
                "motion_regime": audit.MOVING, "graph_sha256": "a" * 64,
            })
        rows.append({**rows[-1], "dst": slots + 1, "role": "holdout", "slot": -1})
    return pd.DataFrame(rows)


def records_for(targets):
    return pd.DataFrame([
        {
            **{key: row[key] for key in audit.IDENTITY},
            "motion_regime": audit.MOVING, "model_role": role,
            "clean_pair_probability": .2, "triggered_pair_probability": .8,
            "clean_pair_mean_logit": -1., "triggered_pair_mean_logit": 1.,
            "clean_negative_bce": .3, "triggered_positive_bce": .3,
        }
        for role in audit.MODEL_ROLES for row in targets.to_dict("records")
    ])


def selection_result(losses):
    return {
        "history": [{"epoch": i + 1, "val_loss": loss} for i, loss in enumerate(losses)],
        "checkpoint": {"state": "best_clean_validation_selected_state_dict"},
        "losses": {"val": min(losses)},
    }


class AttributeModel(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr, message_passing_edge_mask):
        return edge_attr[:, 0]


def audit_item(row):
    # Tiny synthetic six-edge graph, never a local AV2 graph or real checkpoint.
    clean = Data(
        x=torch.ones(2, 1),
        edge_index=torch.tensor([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]]),
        edge_attr=torch.tensor([[-2.], [-1.], [-2.], [-2.], [2.], [2.]]),
        message_passing_edge_mask=torch.ones(6, dtype=torch.bool),
    )
    triggered = clean.clone()
    triggered.edge_attr = torch.tensor([[1.], [2.], [-1.], [2.], [-2.], [1.]])
    target = np.array([True, True, False, False, False, False])
    return audit.response.AuditItem(
        scenario_id=row.scenario_id, motion_regime=audit.MOVING,
        src=int(row.src), dst=int(row.dst), clean=clean, triggered=triggered,
        target_mask=target,
        collateral_masks={
            "adjacent_negative": np.array([False, False, True, True, False, False]),
            "adjacent_positive": np.array([False, False, False, False, True, False]),
            "nonincident_negative": np.zeros(6, dtype=bool),
            "all_non_target_supervised": ~target,
        },
    )


def context_fixture():
    groups = audit.annotate_targets(role_table(), 37)
    reused = {
        name: audit.align_records(records_for(targets), targets)
        for name, targets in groups.items() if name != "additional_train"
    }
    return {
        "epoch": 37, "groups": groups, "reused": reused, "threshold": .5,
        "holdout": {"models": {role: {"collateral": {"sentinel": role}}
                               for role in audit.MODEL_ROLES}},
        "bound": {}, "sources": {}, "graph_rows": {}, "graph_dir": Path("unused"),
        "models": {role: {"result": {"path": "synthetic.json"},
                          "checkpoint": {"path": "synthetic.pt"}}
                   for role in audit.MODEL_ROLES},
    }


class EpochExposureTests(unittest.TestCase):
    def test_uses_selected_epoch_not_last_epoch(self):
        losses = [2.] * 50
        losses[36] = 1.
        self.assertEqual(audit.selected_epoch(selection_result(losses)), 37)

    def test_first_minimum_wins_checkpoint_tie(self):
        self.assertEqual(audit.selected_epoch(selection_result([2., 1., 1.])), 2)

    def test_rejects_incomplete_nonfinite_or_inconsistent_history(self):
        result = selection_result([2., 1.])
        changed = []
        for key, value in (("history", []), ("losses", {"val": 1.2}),
                           ("checkpoint", {"state": "last_epoch"})):
            bad = copy.deepcopy(result)
            bad[key] = value
            changed.append(bad)
        for field, value in (("epoch", 3), ("val_loss", float("nan"))):
            bad = copy.deepcopy(result)
            bad["history"][1][field] = value
            changed.append(bad)
        for bad in changed:
            with self.subTest(result=bad), self.assertRaises(ValueError):
                audit.selected_epoch(bad)

    def test_exposures_stop_at_epoch37_with_disjoint_groups(self):
        groups = audit.annotate_targets(role_table(), 37)
        additional = groups["additional_train"]
        self.assertEqual(additional.v4_positive_exposures_to_checkpoint.tolist(), [18, 12, 12, 9, 9, 9])
        self.assertEqual(groups["rotatable_anchors"].v4_positive_exposures_to_checkpoint.tolist(), [19, 13, 10])
        self.assertTrue(groups["heldout"].v4_positive_exposures_to_checkpoint.eq(0).all())
        pools = [{audit.pair_key(r) for r in frame.to_dict("records")} for frame in groups.values()]
        self.assertEqual(sum(map(len, pools)), len(set.union(*pools)))

    def test_first_epoch_has_zero_additional_positive_exposures(self):
        groups = audit.annotate_targets(role_table(), 1)
        self.assertTrue(groups["additional_train"].v4_positive_exposures_to_checkpoint.eq(0).all())
        self.assertTrue(groups["rotatable_anchors"].v4_positive_exposures_to_checkpoint.eq(1).all())
        with self.assertRaises(ValueError):
            audit.annotate_targets(role_table(), 0)


class IdentityAndWeightTests(unittest.TestCase):
    def setUp(self):
        self.targets = audit.annotate_targets(role_table(), 37)["additional_train"]
        self.records = records_for(self.targets)

    def test_multiple_pairs_in_same_scene_are_kept_for_each_model(self):
        aligned = audit.align_records(self.records, self.targets)
        self.assertEqual(len(aligned), 18)
        self.assertEqual(aligned.scenario_id.nunique(), 3)
        self.assertEqual(aligned.groupby("model_role").size().tolist(), [6, 6, 6])

    def test_duplicate_including_reversed_pair_is_rejected(self):
        for reverse in (False, True):
            extra = self.records.iloc[[0]].copy()
            if reverse:
                extra[["src", "dst"]] = extra[["dst", "src"]].to_numpy()
            with self.subTest(reverse=reverse), self.assertRaisesRegex(ValueError, "duplicate"):
                audit.align_records(pd.concat([self.records, extra]), self.targets)

    def test_missing_role_wrong_pair_or_missing_field_is_rejected(self):
        bad = self.records.copy()
        bad.loc[0, "dst"] = 99
        for changed in (bad, self.records[self.records.model_role.ne("fixed_v3")],
                        self.records.drop(columns="model_role")):
            with self.assertRaises(ValueError):
                audit.align_records(changed, self.targets)

    def test_invalid_numeric_scores_and_pair_ids_are_rejected(self):
        for field, value in (("clean_pair_probability", float("nan")),
                             ("triggered_pair_probability", 1.1),
                             ("clean_negative_bce", -.1),
                             ("src", float("inf")), ("src", 1.5)):
            changed = self.records.copy()
            changed[field] = changed[field].astype(float)
            changed.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit.align_records(changed, self.targets)

    def test_model_scores_cannot_override_plan_metadata(self):
        self.records["v4_positive_exposures_to_checkpoint"] = 999
        aligned = audit.align_records(self.records, self.targets)
        self.assertEqual(set(aligned.v4_positive_exposures_to_checkpoint), {9, 12, 18})

    def test_scene_and_target_weighting_are_distinct(self):
        frame = self.records.iloc[:4].copy()
        frame["scenario_id"] = ["a", "a", "a", "b"]
        frame["triggered_pair_probability"] = [.8, .8, .8, .2]
        summary = audit.summarize_group(frame, .5)
        self.assertEqual(summary["target_weighted"]["incremental_flip_rate_all_targets"], .75)
        self.assertEqual(summary["equal_scene_weighted"]["incremental_asr"], .5)

    def test_conditional_scene_weighting_uses_weighted_denominator(self):
        frame = self.records.iloc[:3].copy()
        frame["scenario_id"] = ["a", "a", "b"]
        frame["clean_pair_probability"] = [.2, .8, .2]
        frame["triggered_pair_probability"] = [.8, .8, .2]
        self.assertAlmostEqual(audit.weighted_metrics(frame, .5)["conditional_flip"], 1 / 3)
        frame["clean_pair_probability"] = .8
        self.assertIsNone(audit.weighted_metrics(frame, .5)["conditional_flip"])


class CollateralAndBatchTests(unittest.TestCase):
    def test_edge_and_scene_weighted_collateral_and_empty_subsets(self):
        frame = pd.DataFrame({
            "scenario_id": ["a", "a", "b", "c"],
            "adjacent_negative_edges": [2, 8, 2, 0],
            "adjacent_negative_n00": [0, 8, 2, 0],
            "adjacent_negative_n01": [2, 0, 0, 0],
            "adjacent_negative_n10": [0, 0, 0, 0],
            "adjacent_negative_n11": [0, 0, 0, 0],
            "adjacent_negative_delta_sum": [1., 0., 0., 0.],
        })
        result = audit.collateral_summary(frame)
        adjacent = result["adjacent_negative"]
        self.assertEqual(adjacent["directed_edge_trigger_opportunities"], 12)
        self.assertAlmostEqual(adjacent["edge_weighted_incremental_positive_rate"], 1 / 6)
        self.assertEqual(adjacent["equal_scene_mean_of_target_rates"]["incremental_positive_rate"], .25)
        self.assertEqual(adjacent["scenes_with_eligible_opportunities"], 2)
        self.assertIsNone(result["adjacent_positive"])
        empty = audit.collateral_summary(frame.iloc[[3]])["adjacent_negative"]
        self.assertIsNone(empty["edge_weighted_incremental_positive_rate"])
        self.assertIsNone(empty["equal_scene_mean_of_target_rates"]["incremental_positive_rate"])
        frame.loc[0, "adjacent_negative_n00"] = 1
        with self.assertRaises(ValueError):
            audit.collateral_summary(frame)

    def test_batch_scores_exactly_match_original_audit_and_collateral_counts(self):
        targets = audit.annotate_targets(role_table(), 37)["additional_train"].iloc[:2]
        items = [audit_item(row) for row in targets.itertuples(index=False)]
        kwargs = {"role": "rotating_v4", "threshold": .5, "device": torch.device("cpu")}
        observed = audit.score_batch(AttributeModel(), items, targets, **kwargs)
        collateral = {"rotating_v4": {
            regime: {subset: audit.response.CollateralAccumulator()
                     for subset in audit.response.COLLATERAL_SUBSETS}
            for regime in ("all", *audit.response.REGIMES)
        }}
        expected = []
        audit.response._append_batch_records(
            items, model_role="rotating_v4", model=AttributeModel(),
            device=kwargs["device"], threshold=.5, records=expected, collateral=collateral,
        )
        for old, new in zip(expected, observed):
            for key, value in old.items():
                self.assertEqual(new[key], value)
            self.assertEqual(new["adjacent_negative_edges"], 2)
            self.assertEqual(new["adjacent_negative_n01"], 1)
            self.assertEqual(new["adjacent_positive_n10"], 1)
        for subset, acc in collateral["rotating_v4"]["all"].items():
            for field in ("n00", "n01", "n10", "n11", "delta_sum"):
                self.assertAlmostEqual(sum(r[subset + "_" + field] for r in observed), getattr(acc, field))

    def test_batch_rejects_misalignment_and_incomplete_output(self):
        targets = audit.annotate_targets(role_table(), 37)["additional_train"].iloc[:2]
        items = [audit_item(row) for row in targets.itertuples(index=False)]
        kwargs = {"role": "rotating_v4", "threshold": .5, "device": torch.device("cpu")}
        for batch, rows in ((items[::-1], targets), (items[:1], targets), ([], targets.iloc[:0])):
            with self.assertRaises(ValueError):
                audit.score_batch(AttributeModel(), batch, rows, **kwargs)
        with patch.object(audit.response, "_model_probabilities", return_value=(np.zeros(1),) * 4):
            with self.assertRaisesRegex(ValueError, "model output"):
                audit.score_batch(AttributeModel(), items, targets, **kwargs)


class WorkflowBoundaryTests(unittest.TestCase):
    def test_nontrain_graph_or_sha_mismatch_fails_before_graph_loader(self):
        targets = audit.annotate_targets(role_table(), 37)["additional_train"].iloc[:1]
        row = next(targets.itertuples(index=False))
        with patch.object(audit.response, "_load_audit_item") as loader:
            for graph_rows in ({}, {row.scenario_id: row._replace(split="val")},
                               {row.scenario_id: row._replace(split="test")},
                               {row.scenario_id: row._replace(graph_sha256="b" * 64)}):
                with self.assertRaises(ValueError):
                    audit.load_items(targets, graph_rows, Path("unused"))
            loader.assert_not_called()

    def test_check_only_has_no_output_or_inference(self):
        with patch.object(audit, "preflight", return_value=context_fixture()), \
                patch.object(audit, "run") as run, \
                patch.object(audit, "new_output") as output, \
                patch.object(audit, "load_items") as loader, \
                patch.object(audit.response, "_model_from_assets") as models, \
                redirect_stdout(io.StringIO()) as stdout:
            audit.main(["--check-only"])
            self.assertEqual(json.loads(stdout.getvalue())["status"], "PREFLIGHT_PASS")
            for blocked in (run, output, loader, models):
                blocked.assert_not_called()

    def test_existing_or_frozen_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "ROOT", Path(temp)):
            output = Path(temp) / "record/reproduction/diagnosis"
            self.assertEqual(audit.new_output(output), output)
            with self.assertRaises(FileExistsError):
                audit.new_output(output)
            for path in (Path(temp) / "record/v6/new", Path(temp) / "record/v6_4/new"):
                with self.assertRaises(ValueError):
                    audit.new_output(path)
                self.assertFalse(path.exists())

    def test_summary_reuses_only_matching_group_and_leaves_stop_unchanged(self):
        context = context_fixture()
        targets = context["groups"]["additional_train"]
        summary, comparison = audit.build_summary(audit.align_records(records_for(targets), targets), context)
        self.assertEqual(len(comparison), 9)
        self.assertEqual(summary["original_training_gate_status"], "STOP")
        self.assertIsNone(summary["new_metric_gate"])
        for role in audit.MODEL_ROLES:
            anchor = summary["groups"]["rotatable_anchors"][role]
            self.assertTrue(all(value is None for value in anchor["collateral"].values()))
            self.assertNotIn("cached_full_pool_collateral", anchor)
            self.assertEqual(summary["groups"]["heldout"][role]["cached_full_pool_collateral"], {"sentinel": role})
        self.assertEqual(set(summary["additional_target_strata"]["v4_positive_exposures_to_checkpoint"]), {"9", "12", "18"})

    def test_synthetic_end_to_end_writes_bound_outputs_and_no_training(self):
        context = context_fixture()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(audit, "ROOT", Path(temp)), \
                patch.object(audit, "BATCH_SIZE", 2), \
                patch.object(audit, "load_items", side_effect=lambda rows, *_: [audit_item(r) for r in rows.itertuples(index=False)]), \
                patch.object(audit.fixed, "read_json", return_value={}), \
                patch.object(audit.response, "_model_from_assets", side_effect=lambda *a, **kw: AttributeModel()) as models, \
                patch.object(torch.optim, "Adam", side_effect=AssertionError("no optimizer")), \
                redirect_stdout(io.StringIO()):
            output = audit.run(context, device_name="cpu", output_dir=Path(temp) / "record/reproduction/audit")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "AUDIT_COMPLETE_REVIEW_REQUIRED")
            self.assertEqual(summary["scope"]["test_graphs_read"], 0)
            self.assertFalse(summary["scope"]["training"])
            self.assertEqual(models.call_count, 3)
            self.assertEqual(len(pd.read_csv(output / "additional.records.csv")), 18)
            self.assertEqual(len(pd.read_csv(output / "comparison.csv")), 9)
            for name, asset in summary["outputs"].items():
                self.assertEqual(sha256_file(output / name), asset["sha256"])
            self.assertFalse((output / "best_model.pt").exists())

    def test_failed_integrity_does_not_publish_completed_summary(self):
        context = context_fixture()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(audit, "ROOT", Path(temp)), \
                patch.object(audit, "load_items", side_effect=lambda rows, *_: [audit_item(r) for r in rows.itertuples(index=False)]), \
                patch.object(audit.fixed, "read_json", return_value={}), \
                patch.object(audit.response, "_model_from_assets", side_effect=lambda *a, **kw: AttributeModel()), \
                patch.object(audit, "verify_unchanged", side_effect=ValueError("modified asset")), \
                redirect_stdout(io.StringIO()):
            output = Path(temp) / "record/reproduction/incomplete"
            with self.assertRaisesRegex(ValueError, "modified asset"):
                audit.run(context, device_name="cpu", output_dir=output)
            self.assertTrue((output / "protocol.json").exists())
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
