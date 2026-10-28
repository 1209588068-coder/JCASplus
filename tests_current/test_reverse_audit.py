"""Synthetic-only reverse Phase 0 checks; no real AV2 data or model access."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from spear_current import reverse_audit as audit


def prepared():
    sources, metadata = {}, {}
    for role in ("train", "dev"):
        sources[role] = [dict(scenario_id=f"{role}-{i}", split_group_id=f"{role}-group-{i}",
                              split="train", role=role) for i in range(4)]
        metadata[role] = []
        # One scene with three positives, one with two, one with none, one empty.
        populations = ((1, 0, 1, -1, 1), (1, 1, 0), (0, -1), ())
        for source, labels in zip(sources[role], populations):
            for pair, label in enumerate(labels):
                metadata[role].append(dict(
                    scenario_id=source["scenario_id"], split_group_id=source["split_group_id"],
                    role=role, pair_index=pair, role_pair_index=len(metadata[role]),
                    src_track_id="track-a", dst_track_id=f"track-{pair + 1}",
                    true_label=label, eligible=label >= 0,
                    motion_regime="moving" if pair % 2 else "slow",
                    is_attachment=role == "train" and label == 0,
                    is_outer=role == "train" and pair % 2 == 0))
    return metadata, sources


def clean_rows(metadata):
    rows = []
    for role in ("train", "dev"):
        for original in metadata[role]:
            if original["eligible"]:
                row = dict(original, label=original["true_label"])
                for j, model in enumerate(audit.MODELS):
                    detected = (original["role_pair_index"] + j) % 2
                    row.update({model + "_pred": detected, model + "_p": .8 if detected else .2})
                rows.append(row)
    return rows


def positive_rows():
    metadata, _ = prepared()
    rows = [row for row in clean_rows(metadata)
            if row["role"] == "train" and row["true_label"] == 1]
    for row, pred, motion in zip(rows, (1, 0, 0, 1, 0), ("slow", "slow", "moving", "moving", "moving")):
        row.update(victim_pred=pred, victim_p=.8 if pred else .2, motion_regime=motion)
    return rows


class PositiveTargetTests(unittest.TestCase):
    def test_exact_identity_hash_one_per_positive_scene_and_no_fallback(self):
        metadata, sources = prepared()
        for seed in (20260621, 17, 0):
            result = audit.select_positive_targets(metadata, sources, seed)
            for role in ("train", "dev"):
                expected = []
                for source in sources[role]:
                    candidates = [row for row in metadata[role]
                                  if row["scenario_id"] == source["scenario_id"]
                                  and row["eligible"] and row["true_label"] == 1]
                    if candidates:
                        expected.append(min(candidates, key=lambda row: (
                            hashlib.sha256(f"{seed}:spear-audit-target:{row['scenario_id']}:"
                                f"{row['src_track_id']}:{row['dst_track_id']}".encode()).hexdigest(),
                            row["pair_index"])))
                self.assertEqual(result[role], expected)
                self.assertEqual(len(result[role]), 2)
                self.assertTrue(all(row["true_label"] == 1 and row["eligible"] is True
                                    for row in result[role]))

    def test_row_order_and_source_order_do_not_change_selected_identities(self):
        metadata, sources = prepared()
        expected = audit.select_positive_targets(metadata, sources)
        reversed_rows = {role: list(reversed(rows)) for role, rows in metadata.items()}
        self.assertEqual(audit.select_positive_targets(reversed_rows, sources), expected)
        reversed_sources = {role: list(reversed(rows)) for role, rows in sources.items()}
        actual = audit.select_positive_targets(reversed_rows, reversed_sources)
        for role in actual:
            self.assertEqual(actual[role], list(reversed(expected[role])))

    def test_predictions_do_not_select_targets_and_clean_misses_are_retained(self):
        metadata, sources = prepared()
        expected = audit.select_positive_targets(metadata, sources)
        selected = {(row["role"], row["role_pair_index"])
                    for rows in expected.values() for row in rows}
        for role, rows in metadata.items():
            for row in rows:
                row.update(victim_pred=0 if (role, row["role_pair_index"]) in selected else 1,
                           victim_p=object(), clean_pred="not inspected", trigger_pred=None)
        self.assertEqual(audit.select_positive_targets(metadata, sources), expected)
        for role, rows in expected.items():
            self.assertTrue(all(metadata[role][row["role_pair_index"]]["victim_pred"] == 0
                                for row in rows))

    def test_no_mutation_no_aliasing_no_rng_or_file_access(self):
        metadata, sources = prepared()
        original, rng = deepcopy((metadata, sources)), random.getstate()
        with patch("builtins.open", side_effect=AssertionError("file read")), \
                patch.object(Path, "open", side_effect=AssertionError("path read")):
            selected = audit.select_positive_targets(metadata, sources)
        self.assertEqual((metadata, sources), original)
        self.assertEqual(random.getstate(), rng)
        selected["train"][0]["is_outer"] = not selected["train"][0]["is_outer"]
        self.assertEqual((metadata, sources), original)

    def test_positive_but_ineligible_is_excluded_not_replaced_with_negative(self):
        metadata, sources = prepared()
        for row in metadata["train"]:
            if row["true_label"] == 1:
                row["eligible"] = False
        selected = audit.select_positive_targets(metadata, sources)
        self.assertEqual(selected["train"], [])
        self.assertEqual(len(selected["dev"]), 2)

    def test_missing_metadata_and_invalid_label_eligibility_indices_are_rejected(self):
        changes = (
            lambda row: row.pop("src_track_id"),
            lambda row: row.update(true_label=True),
            lambda row: row.update(true_label=2),
            lambda row: row.update(true_label=-1, eligible=True),
            lambda row: row.update(eligible=1),
            lambda row: row.update(is_outer=0),
            lambda row: row.update(is_attachment="False"),
            lambda row: row.update(pair_index=True),
            lambda row: row.update(pair_index=-1),
            lambda row: row.update(role_pair_index=1000),
            lambda row: row.update(motion_regime="fast"),
            lambda row: row.update(src_track_id=""),
            lambda row: row.update(dst_track_id=row["src_track_id"]),
            lambda row: row.update(role="audit"),
            lambda row: row.update(scenario_id="unregistered"),
            lambda row: row.update(split_group_id="different-group"),
        )
        for number, change in enumerate(changes):
            metadata, sources = prepared()
            change(metadata["train"][0])
            with self.subTest(change=number), self.assertRaises(ValueError):
                audit.select_positive_targets(metadata, sources)
        for key in ("pair_index", "role_pair_index", "dst_track_id"):
            metadata, sources = prepared()
            metadata["train"][1][key] = metadata["train"][0][key]
            with self.subTest(duplicate=key), self.assertRaises(ValueError):
                audit.select_positive_targets(metadata, sources)

    def test_invalid_source_roles_overlap_and_seed_are_rejected(self):
        changes = (
            lambda rows: rows["dev"][0].update(split="val"),
            lambda rows: rows["dev"][0].update(role="audit"),
            lambda rows: rows["dev"][0].update(scenario_id=rows["train"][0]["scenario_id"]),
            lambda rows: rows["dev"][0].update(split_group_id=rows["train"][0]["split_group_id"]),
            lambda rows: rows["dev"][1].update(scenario_id=rows["dev"][0]["scenario_id"]),
            lambda rows: rows["train"][0].update(split_group_id=""),
        )
        for number, change in enumerate(changes):
            metadata, sources = prepared()
            change(sources)
            with self.subTest(change=number), self.assertRaises(ValueError):
                audit.select_positive_targets(metadata, sources)
        for seed in (True, -1, 1.5, "20260621", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                audit.select_positive_targets(*prepared(), seed=seed)
        for container in (0, 1):
            values = list(prepared())
            values[container]["audit"] = []
            with self.subTest(extra_role=container), self.assertRaises(ValueError):
                audit.select_positive_targets(*values)


class CleanPositiveStatsTests(unittest.TestCase):
    def test_all_fixed_positives_including_misses_strata_and_unique_scene_counts(self):
        rows = positive_rows()
        original = deepcopy(rows)
        actual = audit.clean_positive_stats(rows, "victim")
        self.assertEqual(actual, dict(
            all=dict(n=5, clean_detected=2, baseline_misses=3, clean_recall_ceiling=2 / 5,
                     baseline_fnr=3 / 5, scenes=2),
            slow=dict(n=2, clean_detected=1, baseline_misses=1, clean_recall_ceiling=.5,
                      baseline_fnr=.5, scenes=1),
            moving=dict(n=3, clean_detected=1, baseline_misses=2, clean_recall_ceiling=1 / 3,
                        baseline_fnr=2 / 3, scenes=2)))
        self.assertEqual(rows, original)
        self.assertEqual(audit.clean_positive_stats(iter(reversed(rows)), "victim"), actual)
        self.assertNotIn("incremental_asr", actual["all"])
        self.assertNotIn("absolute_suppression", actual["all"])
        json.dumps(actual, allow_nan=False)

    def test_empty_population_or_stratum_has_null_rates(self):
        empty = dict(n=0, clean_detected=0, baseline_misses=0, clean_recall_ceiling=None,
                     baseline_fnr=None, scenes=0)
        self.assertEqual(audit.clean_positive_stats([], "victim"),
                         dict(all=empty, slow=empty, moving=empty))
        rows = positive_rows()[:2]
        self.assertEqual(audit.clean_positive_stats(rows, "victim")["moving"], empty)

    def test_all_baseline_misses_remain_in_denominator(self):
        rows = positive_rows()
        for row in rows:
            row.update(victim_pred=0, victim_p=.1)
        for stats in audit.clean_positive_stats(rows, "victim").values():
            self.assertEqual(stats["baseline_misses"], stats["n"])
            self.assertEqual(stats["clean_recall_ceiling"], 0.)
            self.assertEqual(stats["baseline_fnr"], 1.)

    def test_stats_reject_nonpositive_ineligible_duplicate_or_invalid_predictions(self):
        changes = (
            lambda rows: rows[0].update(true_label=0),
            lambda rows: rows[0].update(true_label=True),
            lambda rows: rows[0].update(eligible=False),
            lambda rows: rows[0].update(victim_pred=True),
            lambda rows: rows[0].update(victim_pred=2),
            lambda rows: rows[0].update(victim_p=float("nan")),
            lambda rows: rows[0].update(victim_p=float("inf")),
            lambda rows: rows[0].update(victim_p=-.01),
            lambda rows: rows[0].update(victim_p=1.01),
            lambda rows: rows[0].update(victim_p=True),
            lambda rows: rows[0].update(motion_regime="unknown"),
            lambda rows: rows.append(dict(rows[0])),
        )
        for number, change in enumerate(changes):
            rows = positive_rows()
            change(rows)
            with self.subTest(change=number), self.assertRaises(ValueError):
                audit.clean_positive_stats(rows, "victim")


class SavedCleanPredictionTests(unittest.TestCase):
    def read_rows(self, metadata, rows):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_clean.csv"
            fields = list(audit.IDENTITY_FIELDS) + ["label"] + [model + suffix
                for model in audit.MODELS for suffix in ("_pred", "_p")]
            audit._write_csv(path, rows, fields)
            before = path.read_bytes()
            result = audit._read_clean(path, metadata)
            self.assertEqual(path.read_bytes(), before)
            return result

    def test_complete_saved_csv_is_parsed_and_retains_negatives_and_positive_misses(self):
        metadata, _ = prepared()
        expected = clean_rows(metadata)
        actual = self.read_rows(metadata, expected)
        self.assertEqual(actual, expected)
        self.assertTrue(any(row["true_label"] == 0 for row in actual))
        self.assertTrue(any(row["true_label"] == 1 and row["victim_pred"] == 0 for row in actual))
        self.assertEqual(self.read_rows(metadata, list(reversed(expected))), list(reversed(expected)))

    def test_incomplete_duplicate_unregistered_and_drifted_identity_rejected(self):
        changes = (
            lambda rows: rows.pop(),
            lambda rows: rows.append(dict(rows[0])),
            lambda rows: rows[0].update(role_pair_index=10000),
            lambda rows: rows[0].update(role="audit"),
            lambda rows: rows[0].update(scenario_id="other-scene"),
            lambda rows: rows[1].update(scenario_id=rows[0]["scenario_id"], pair_index=rows[0]["pair_index"]),
            lambda rows: rows[0].update(split_group_id="other-group"),
            lambda rows: rows[0].update(src_track_id="other-track"),
            lambda rows: rows[0].update(true_label=0),
            lambda rows: rows[0].update(label=0),
            lambda rows: rows[0].update(eligible=False),
            lambda rows: rows[0].update(is_outer=not rows[0]["is_outer"]),
        )
        for number, change in enumerate(changes):
            metadata, _ = prepared()
            rows = clean_rows(metadata)
            change(rows)
            with self.subTest(change=number), self.assertRaises(ValueError):
                self.read_rows(metadata, rows)

    def test_invalid_saved_booleans_probabilities_predictions_or_disagreement_rejected(self):
        changes = (
            dict(eligible="1"), dict(is_attachment="false"), dict(victim_pred=2),
            dict(victim_pred="1.0"), dict(victim_p="NaN"), dict(victim_p="inf"),
            dict(victim_p=-.1), dict(victim_p=1.1), dict(victim_pred=1, victim_p=.1),
        )
        for update in changes:
            metadata, _ = prepared()
            rows = clean_rows(metadata)
            rows[0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.read_rows(metadata, rows)

    def test_saved_probability_tie_is_not_rethresholded(self):
        metadata, _ = prepared()
        rows = clean_rows(metadata)
        rows[0].update(victim_pred=0, victim_p=.5)
        rows[1].update(victim_pred=1, victim_p=.5)
        actual = self.read_rows(metadata, rows)
        self.assertEqual([row["victim_pred"] for row in actual[:2]], [0, 1])


class PhaseZeroOutputAndCliTests(unittest.TestCase):
    def test_output_requires_new_resolved_results_child_without_creating_it(self):
        with TemporaryDirectory() as directory, patch.object(audit, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "results").mkdir()
            (root / "elsewhere").mkdir()
            existing = root / "results" / "existing"
            existing.mkdir()
            audit._write_json(existing / "evidence.json", dict(preserved=True))
            (root / "results" / "redirect").symlink_to(root / "elsewhere", target_is_directory=True)
            for target in (root, root / "results", root / "elsewhere", existing,
                           root / "results/../outside", root / "results/redirect/new"):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    audit._output_path(target)
            target = root / "results" / "new"
            self.assertEqual(audit._output_path(target), target)
            self.assertFalse(target.exists())
            self.assertEqual(json.loads((existing / "evidence.json").read_text()), dict(preserved=True))

    def test_writers_never_overwrite_existing_evidence(self):
        with TemporaryDirectory() as directory:
            json_path, csv_path = Path(directory) / "summary.json", Path(directory) / "rows.csv"
            audit._write_json(json_path, dict(preserved=True))
            audit._write_csv(csv_path, [dict(row=1)], ["row"])
            previous = json_path.read_bytes(), csv_path.read_bytes()
            with self.assertRaises(FileExistsError):
                audit._write_json(json_path, dict(preserved=False))
            with self.assertRaises(FileExistsError):
                audit._write_csv(csv_path, [dict(row=2)], ["row"])
            self.assertEqual((json_path.read_bytes(), csv_path.read_bytes()), previous)

    def test_default_cli_loads_no_sources_even_with_output_path(self):
        for argv in ([], ["--output-dir", "unused"]):
            output = io.StringIO()
            with self.subTest(argv=argv), redirect_stdout(output), \
                    patch.object(audit, "run_audit", side_effect=AssertionError("audit started")), \
                    patch.object(audit, "_sha", side_effect=AssertionError("asset read")), \
                    patch("builtins.open", side_effect=AssertionError("file read")), \
                    patch.object(Path, "open", side_effect=AssertionError("path read")):
                self.assertIsNone(audit.main(argv))
            self.assertIn("NOT_RUN", output.getvalue())

    def test_run_requires_explicit_output_and_dispatches_only_requested_audit(self):
        with redirect_stderr(io.StringIO()), patch.object(audit, "run_audit") as run:
            with self.assertRaises(SystemExit) as error:
                audit.main(["--run"])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
        with patch.object(audit, "run_audit") as run:
            audit.main(["--run", "--output-dir", "synthetic-new-results"])
            run.assert_called_once_with(Path("synthetic-new-results"))


if __name__ == "__main__":
    unittest.main()
