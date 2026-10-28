"""Pure reverse-response metrics for offline AV2/owned-model research.

Every input row belongs to a fixed eligible true-positive population selected
without using model predictions. Original labels remain one. The clean and
triggered predictions describe the same model on the same canonical pair.
This module performs no loading, inference, training, or file writes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
from typing import Any


_STRATA = ("all", "slow", "moving")
_METADATA = ("scenario_id", "pair_index", "motion_regime", "true_label",
             "eligible", "is_attachment", "is_outer")
_RESPONSES = ("clean_pred", "trigger_pred", "clean_p", "trigger_p")


def _integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validated_rows(rows, model):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    try:
        materialized = list(rows)
    except TypeError as error:
        raise ValueError("rows must be an iterable of mappings") from error
    identities = set()
    response_fields = tuple(f"{model}_{suffix}" for suffix in _RESPONSES)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        for description, fields in (("metadata", _METADATA), ("model", response_fields)):
            missing = [field for field in fields if field not in row]
            if missing:
                raise ValueError(f"row {index} missing {description} fields: {', '.join(missing)}")
        scenario, pair = row["scenario_id"], row["pair_index"]
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
        if not _integer(row["true_label"]) or row["true_label"] != 1:
            raise ValueError("reverse rows require original true_label integer 1")
        if row["eligible"] is not True:
            raise ValueError("reverse rows require eligible is True")
        for field in ("is_attachment", "is_outer"):
            if not isinstance(row[field], bool):
                raise ValueError(f"{field} must be a bool")
        for suffix in ("clean_pred", "trigger_pred"):
            field = f"{model}_{suffix}"
            if not _integer(row[field]) or row[field] not in (0, 1):
                raise ValueError(f"{field} must be one of the integers 0, 1")
        for suffix in ("clean_p", "trigger_p"):
            field = f"{model}_{suffix}"
            value = row[field]
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{field} must be a finite probability in [0, 1]")
    return materialized


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def reverse_response_metrics(
    rows: Iterable[Mapping[str, Any]], model: str,
) -> dict[str, dict[str, int | float | None]]:
    """Count suppression and repair separately in all/slow/moving strata.

    Required metadata matches ``development_metrics``: scenario_id, pair_index,
    motion_regime, true_label, eligible, is_attachment, is_outer. Every row must
    have integer true_label=1 and boolean eligible=True; other rows are rejected,
    never silently filtered. Duplicate (scenario_id, pair_index) keys are invalid.
    ``{model}_clean_p`` and ``{model}_trigger_p`` are class-one probabilities.
    Predictions are validated binary integers and are not re-thresholded here.

    ``incremental_asr`` is induced 1->0 misses divided by ALL fixed positives;
    its ceiling is the clean recall. ``conditional_suppression`` instead divides
    by clean-detected positives. Absolute suppression includes baseline misses
    that remain misses, so ``net_fnr_change`` also subtracts repaired 0->1 cases.
    Counts are Python integers; undefined rates are None. Inputs are unchanged.
    """
    materialized = _validated_rows(rows, model)
    result = {}
    for stratum in _STRATA:
        group = [row for row in materialized
                 if stratum == "all" or row["motion_regime"] == stratum]
        transitions = [(int(row[f"{model}_clean_pred"]), int(row[f"{model}_trigger_pred"]))
                       for row in group]
        n = len(transitions)
        clean_detected = sum(before == 1 for before, _ in transitions)
        baseline_misses = n - clean_detected
        induced = sum(before == 1 and after == 0 for before, after in transitions)
        corrected = sum(before == 0 and after == 1 for before, after in transitions)
        triggered_misses = sum(after == 0 for _, after in transitions)
        result[stratum] = dict(
            n=n,
            clean_detected=clean_detected,
            baseline_misses=baseline_misses,
            induced_misses_1to0=induced,
            corrected_misses_0to1=corrected,
            triggered_misses=triggered_misses,
            baseline_fnr=_rate(baseline_misses, n),
            absolute_suppression=_rate(triggered_misses, n),
            incremental_asr=_rate(induced, n),
            conditional_suppression=_rate(induced, clean_detected),
            net_fnr_change=_rate(induced - corrected, n),
            clean_recall_ceiling=_rate(clean_detected, n),
        )
    return result
