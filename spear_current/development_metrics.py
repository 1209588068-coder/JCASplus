"""Pure metrics for an offline, fixed-checkpoint train/dev diagnostic.

Rows describe one population of locally evaluated AV2 graph pairs.  No graph,
checkpoint, split, or other file is loaded here.  A strict negative is defined
only by ``eligible is True`` and ``true_label == 0``, never by a prediction.
Attachment/outer membership does not change that definition.

Every public helper returns ``all``, ``slow``, and ``moving`` strata.  Empty
populations have zero counts and ``None`` for undefined means and rates.  All
returned values are JSON-serializable Python scalars.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
from typing import Any


_STRATA = ("all", "slow", "moving")
_METADATA = (
    "scenario_id", "pair_index", "motion_regime", "true_label", "eligible",
    "is_attachment", "is_outer",
)


def _integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _finite_number(value: Any, field: str, *, probability: bool = False) -> None:
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0
            or (probability and value > 1)):
        domain = "finite probability in [0, 1]" if probability else "finite nonnegative number"
        raise ValueError(f"{field} must be a {domain}")


def _validated_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Materialize once, checking one-to-one pair identities and metadata."""
    result = list(rows)
    identities = set()
    for index, row in enumerate(result):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        missing = [field for field in _METADATA if field not in row]
        if missing:
            raise ValueError(f"row {index} missing metadata fields: {', '.join(missing)}")
        scenario = row["scenario_id"]
        pair = row["pair_index"]
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError("scenario_id must be a nonempty string")
        if not _integer(pair) or pair < 0:
            raise ValueError("pair_index must be a nonnegative integer")
        identity = (scenario, int(pair))
        if identity in identities:
            raise ValueError(f"duplicate pair identity: {identity!r}")
        identities.add(identity)
        if row["motion_regime"] not in ("slow", "moving"):
            raise ValueError("motion_regime must be 'slow' or 'moving'")
        label = row["true_label"]
        if not _integer(label) or label not in (-1, 0, 1):
            raise ValueError("true_label must be one of the integers -1, 0, 1")
        for field in ("eligible", "is_attachment", "is_outer"):
            if not isinstance(row[field], bool):
                raise ValueError(f"{field} must be a bool")
        if row["eligible"] and label == -1:
            raise ValueError("unknown true_label cannot be eligible")
    return result


def _validate_model(rows: list[Mapping[str, Any]], model: str) -> None:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    for index, row in enumerate(rows):
        fields = tuple(f"{model}_{suffix}" for suffix in (
            "clean_p", "trigger_p", "clean_pred", "trigger_pred",
        ))
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"row {index} missing model fields: {', '.join(missing)}")
        for suffix in ("clean_p", "trigger_p"):
            field = f"{model}_{suffix}"
            _finite_number(row[field], field, probability=True)
        for suffix in ("clean_pred", "trigger_pred"):
            field = f"{model}_{suffix}"
            if not _integer(row[field]) or row[field] not in (0, 1):
                raise ValueError(f"{field} must be one of the integers 0, 1")
        nll_field = f"{model}_target_nll"
        if nll_field in row:
            _finite_number(row[nll_field], nll_field)


def _groups(rows: list[Mapping[str, Any]]):
    for stratum in _STRATA:
        yield stratum, [row for row in rows if (
            stratum == "all" or row["motion_regime"] == stratum
        )]


def _strict_negative(row: Mapping[str, Any]) -> bool:
    return row["eligible"] and row["true_label"] == 0


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    # Divide before summation to avoid overflow for large, finite NLL values.
    return math.fsum(value / len(values) for value in values) if values else None


def population_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count label/eligibility and membership fields in each motion stratum.

    Counts are marginal: e.g. an unknown row is also ineligible, and a positive
    row may be eligible.  ``strict_negative`` alone is the ASR denominator.
    Each ``(scenario_id, pair_index)`` must occur once in this population.
    """
    result = {}
    for stratum, group in _groups(_validated_rows(rows)):
        result[stratum] = {
            "count": len(group),
            "strict_negative": sum(_strict_negative(row) for row in group),
            "positive": sum(row["true_label"] == 1 for row in group),
            "unknown": sum(row["true_label"] == -1 for row in group),
            "ineligible": sum(not row["eligible"] for row in group),
            "eligible": sum(row["eligible"] for row in group),
            "attachment": sum(row["is_attachment"] for row in group),
            "outer": sum(row["is_outer"] for row in group),
        }
    return result


def response_metrics(
    rows: Iterable[Mapping[str, Any]], model: str,
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    """Separate all-row responses from strict eligible true-negative ASRs.

    ``incremental_asr`` counts same-model clean-zero/trigger-one transitions
    divided by *all* strict negatives.  ``conditional_flip`` uses the same
    numerator divided only by strict negatives whose clean prediction is zero.
    ``absolute_asr`` and ``clean_activation`` use all strict negatives.

    All-row ``target_nll`` is the mean of provided ``{model}_target_nll`` values;
    ``target_nll_count`` records its denominator.  Missing optional values are
    excluded; a supplied value must be finite and nonnegative.
    """
    materialized = _validated_rows(rows)
    _validate_model(materialized, model)
    result = {}
    for stratum, group in _groups(materialized):
        count = len(group)
        clean_count = sum(row[f"{model}_clean_pred"] == 1 for row in group)
        trigger_count = sum(row[f"{model}_trigger_pred"] == 1 for row in group)
        nll_field = f"{model}_target_nll"
        nll_values = [float(row[nll_field]) for row in group if nll_field in row]
        strict = [row for row in group if _strict_negative(row)]
        strict_count = len(strict)
        strict_clean = sum(row[f"{model}_clean_pred"] == 1 for row in strict)
        strict_trigger = sum(row[f"{model}_trigger_pred"] == 1 for row in strict)
        flips = sum(
            row[f"{model}_clean_pred"] == 0 and row[f"{model}_trigger_pred"] == 1
            for row in strict
        )
        clean_zero = strict_count - strict_clean
        result[stratum] = {
            "all_row_response": {
                "count": count,
                "mean_clean_p": _mean([float(row[f"{model}_clean_p"]) for row in group]),
                "mean_trigger_p": _mean([float(row[f"{model}_trigger_p"]) for row in group]),
                "mean_delta_p": _mean([
                    float(row[f"{model}_trigger_p"]) - float(row[f"{model}_clean_p"])
                    for row in group
                ]),
                "clean_activation_count": clean_count,
                "trigger_activation_count": trigger_count,
                "activation_denominator": count,
                "clean_activation": _rate(clean_count, count),
                "trigger_activation": _rate(trigger_count, count),
                "target_nll": _mean(nll_values),
                "target_nll_count": len(nll_values),
            },
            "strict_negative_response": {
                "count": strict_count,
                "incremental_flips": flips,
                "incremental_asr": _rate(flips, strict_count),
                "incremental_asr_denominator": strict_count,
                "absolute_asr_numerator": strict_trigger,
                "absolute_asr_denominator": strict_count,
                "absolute_asr": _rate(strict_trigger, strict_count),
                "clean_activation_numerator": strict_clean,
                "clean_activation_denominator": strict_count,
                "clean_activation": _rate(strict_clean, strict_count),
                "conditional_flip_numerator": flips,
                "conditional_flip_denominator": clean_zero,
                "conditional_flip": _rate(flips, clean_zero),
            },
        }
    return result


def compare_models(
    rows: Iterable[Mapping[str, Any]], left: str = "surrogate", right: str = "victim",
) -> dict[str, dict[str, int]]:
    """Cross-tabulate the two models' incremental flips on the same strict rows.

    Both models must supply their responses on every row.  Pairing is explicit
    in each row, with unique scenario/pair identities; unpaired populations
    cannot silently be intersected.  A flip means clean zero to triggered one.
    """
    materialized = _validated_rows(rows)
    _validate_model(materialized, left)
    _validate_model(materialized, right)
    if left == right:
        raise ValueError("left and right must name distinct models")
    result = {}
    for stratum, group in _groups(materialized):
        table = {"count": 0, "both": 0, "left_only": 0, "right_only": 0, "neither": 0}
        for row in group:
            if not _strict_negative(row):
                continue
            left_flip = row[f"{left}_clean_pred"] == 0 and row[f"{left}_trigger_pred"] == 1
            right_flip = row[f"{right}_clean_pred"] == 0 and row[f"{right}_trigger_pred"] == 1
            bucket = ("both" if left_flip and right_flip else "left_only" if left_flip
                      else "right_only" if right_flip else "neither")
            table["count"] += 1
            table[bucket] += 1
        result[stratum] = table
    return result
