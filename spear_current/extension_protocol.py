"""Registered fixed-generator sensitivity cases; metadata only, no data I/O.

Budgets here do not retrain the inherited generator or select by predictions.
Every support operation preserves complete source-defined semantic groups.
Missing attributes and immutable attributes are different interventions: the
former also replace selected inputs with standardized zero; the latter only
restrict write permission. Dataset transformations live outside this module.
"""
from __future__ import annotations

import math
import random

from .support_schema import LEGAL_COORDINATES, SELECTED_COORDINATES, support_groups


FEATURE_BUDGETS = (4, 8, 16, 24)
POISON_BUDGETS = (0, 4, 8, 16, 26)
MISSING_RATES = (.25, .50, .75)
IMMUTABLE_RATES = (.50, .75, .95)
MISSING_SEED = 20260621
IMMUTABLE_SEED = 20260622
LEGAL_CAPACITY = len(LEGAL_COORDINATES)


def _ordered(coordinates):
    selected = set(coordinates)
    return [coordinate for coordinate in SELECTED_COORDINATES if coordinate in selected]


def _source_coordinates(mask):
    if (not isinstance(mask, (list, tuple)) or len(mask) != len(SELECTED_COORDINATES)
            or any(type(value) is not bool for value in mask) or sum(mask) != 24):
        raise ValueError('Require the existing exact24, original-order boolean mask')
    coordinates = {c for c, enabled in zip(SELECTED_COORDINATES, mask) if enabled}
    if not coordinates <= set(LEGAL_COORDINATES):
        raise ValueError('Focal metadata cannot enter writable support')
    for group in support_groups('semantic'):
        if coordinates.intersection(group) and not set(group) <= coordinates:
            raise ValueError('Existing support splits a semantic group')
    return coordinates


def _exact_subset(groups, cost, scores):
    """Exact-cost deterministic knapsack; stable group-index ties.

    Integer priorities avoid tolerance-dependent tie breaking. Groups receive
    an explicit total score, rather than an observed prediction or response.
    """
    if type(cost) is not int or cost < 0 or len(groups) != len(scores):
        raise ValueError('Invalid exact-cost selection')
    states = {0: (0, ())}
    for index, (group, score) in enumerate(zip(groups, scores)):
        if type(score) is not int:
            raise ValueError('Priorities must be fixed integers')
        updated = dict(states)
        for previous_cost, (value, chosen) in states.items():
            new_cost = previous_cost + len(group)
            if new_cost > cost:
                continue
            candidate = value + score, chosen + (index,)
            previous = updated.get(new_cost)
            if (previous is None or candidate[0] > previous[0]
                    or (candidate[0] == previous[0] and candidate[1] < previous[1])):
                updated[new_cost] = candidate
        states = updated
    if cost not in states:
        raise ValueError('Exact coordinate count is infeasible for complete semantic groups')
    return _ordered(c for i in states[cost][1] for c in groups[i])


def _feature_support(source, budget):
    groups = tuple(group for group in support_groups('semantic') if set(group) <= source)
    # Maximize lexicographic inclusion in the original frozen importance order.
    # One higher-order bit dominates all lower-priority coordinates combined.
    rank = {coordinate: i for i, coordinate in enumerate(SELECTED_COORDINATES)}
    scores = [sum(1 << (len(rank) - rank[c] - 1) for c in group) for group in groups]
    return _exact_subset(groups, budget, scores)


def _restricted_coordinates(cost, seed):
    groups = support_groups('semantic')
    rng = random.Random(seed)
    # This is a fixed random-score knapsack, not a uniform subset draw.
    # No redraw, prediction feedback, global RNG change, or result selection.
    scores = [rng.getrandbits(64) * len(group) for group in groups]
    return _exact_subset(groups, cost, scores)


def case_matrix(source_mask):
    """Return all 14 predeclared cases as JSON-serializable metadata.

    ``source_mask`` is the registered existing semantic24 mask in the original
    30-slot order. Poison counts use prefixes of the original ordered 26-node
    attachment list; this module never draws or reorders attachment nodes.
    Fractions for missing/immutable attributes use legal28 as their denominator.
    """
    source = _source_coordinates(source_mask)
    result = []

    def add(name, axis, *, active=None, missing=(), immutable=(), poison=26,
            nominal_rate=None, feature_budget=24, subset_seed=None):
        active = source if active is None else set(active)
        missing, immutable = set(missing), set(immutable)
        active = active - missing - immutable
        if not active <= source or not (missing | immutable) <= set(LEGAL_COORDINATES):
            raise ValueError('Case escapes registered source support or legal domain')
        result.append(dict(name=name, axis=axis,
            active_coordinates=_ordered(active),
            missing_coordinates=_ordered(missing),
            immutable_coordinates=_ordered(immutable), poison_count=poison,
            feature_budget=feature_budget, active_coordinate_count=len(active),
            legal_coordinate_denominator=LEGAL_CAPACITY,
            full_feature_denominator=784,
            nominal_rate=nominal_rate, restricted_subset_seed=subset_seed,
            actual_missing_coordinate_count=len(missing),
            actual_missing_rate=len(missing) / LEGAL_CAPACITY,
            actual_immutable_coordinate_count=len(immutable),
            actual_immutable_rate=len(immutable) / LEGAL_CAPACITY,
            attachment_rule='prefix_of_original_ordered_26_attachments',
            original_train_pair_denominator=22005,
            clean_supervision_count=1876,
            poison_pair_rate=poison / 22005,
            poison_fitting_row_rate=poison / (1876 + poison),
            missing_input_policy='same_standardized_zero_columns_in_TRAIN_and_DEV_before_trigger',
            immutable_input_policy='original_input_values_preserved',
            generator_retrained=False, selector_retrained=False,
            experiment_identity='frozen_uniform_generator_sensitivity_not_full_pipeline_retraining',
            support_uses_evaluation_predictions=False))

    add('reference', 'reference')
    for count in POISON_BUDGETS:
        if count != 26:
            add(f'poison_{count}', 'poison', poison=count)
    for budget in FEATURE_BUDGETS:
        support = _feature_support(source, budget)
        if budget == 24:
            if set(support) != source:
                raise ValueError('The 24-coordinate endpoint must reproduce the source mask')
        else:
            add(f'features_{budget}', 'features', active=support, feature_budget=budget)
    for rate in MISSING_RATES:
        cost = math.ceil(rate * LEGAL_CAPACITY)
        add(f'missing_{round(rate * 100)}', 'missing',
            missing=_restricted_coordinates(cost, MISSING_SEED),
            nominal_rate=rate, subset_seed=MISSING_SEED)
    for rate in IMMUTABLE_RATES:
        cost = math.ceil(rate * LEGAL_CAPACITY)
        add(f'immutable_{round(rate * 100)}', 'immutable',
            immutable=_restricted_coordinates(cost, IMMUTABLE_SEED),
            nominal_rate=rate, subset_seed=IMMUTABLE_SEED)
    if len(result) != 14 or len({case['name'] for case in result}) != 14:
        raise ValueError('Registered extension matrix must contain 14 distinct cases')
    return result
