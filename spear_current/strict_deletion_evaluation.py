"""Explicit 24/28-coordinate evaluation contracts for strict deletion arms.

Uses the original CUDA victim fitting and scoring implementation.  Only the
registered ``no_support_mask`` arm may write all 28 legal coordinates; the
other arms retain the frozen 24-coordinate support.  This module opens no
dataset or checkpoint and changes no victim, label, graph, or scoring rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import full_ablation_evaluation as original
from .support_schema import EXCLUDED_COORDINATES, SELECTED_COORDINATES


VARIANTS = ('full', 'no_residual', 'no_support_mask', 'no_domain')


def validate_contract(generator, variant, *, expected_support=None):
    """Check the declared deletion without using its ASR as an acceptance gate."""
    if variant not in VARIANTS:
        raise ValueError('Unregistered strict deletion variant')
    capacity = 28 if variant == 'no_support_mask' else 24
    selected = generator.selected_coordinates
    support = generator.hard_support
    if (getattr(generator, 'input_dim', None) != 784
            or not isinstance(selected, torch.Tensor) or selected.dtype != torch.long
            or selected.shape != (30,) or selected.tolist() != list(SELECTED_COORDINATES)
            or not isinstance(support, torch.Tensor) or support.dtype != torch.bool
            or support.shape != (30,) or support.device != selected.device
            or int(support.sum()) != capacity
            or getattr(generator, 'expected_support_count', capacity) != capacity):
        raise ValueError('Strict deletion support differs from the registered capacity or coordinate order')
    legal = ~torch.isin(selected, selected.new_tensor(EXCLUDED_COORDINATES))
    if bool((support & ~legal).any()):
        raise ValueError('Focal metadata cannot be modified')
    if variant == 'no_support_mask':
        if not torch.equal(support, legal):
            raise ValueError('NoSupportMask must expose exactly all 28 legal coordinates')
    elif expected_support is not None:
        if isinstance(expected_support, torch.Tensor):
            if expected_support.dtype != torch.bool or expected_support.shape != (30,):
                raise ValueError('Expected frozen support must be bool[30]')
            expected = expected_support.to(support.device)
        else:
            if (not isinstance(expected_support, (tuple, list)) or len(expected_support) != 30
                    or any(type(value) is not bool for value in expected_support)):
                raise ValueError('Expected frozen support must be bool[30]')
            expected = torch.tensor(expected_support, dtype=torch.bool, device=support.device)
        if not torch.equal(support, expected):
            raise ValueError('This deletion also changed the frozen 24-coordinate support')
    expected_mode = 'identity' if variant == 'no_domain' else 'domain_projection'
    if generator.output_mode != expected_mode:
        raise ValueError('Output projection differs from the single registered deletion')
    residual = getattr(generator, 'residual', None)
    if variant == 'no_residual':
        if residual is not None:
            raise ValueError('NoResidual must not retain a residual module')
        for name in ('base_generator', 'base_shadow'):
            module = getattr(generator, name, None)
            if module is None or any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError('NoResidual base generator and conditioning shadow must remain frozen')
    elif residual is None or getattr(residual, 'mode', None) != 'uniform':
        raise ValueError('Other strict deletion arms must retain uniform residual aggregation')
    return dict(variant=variant, support_capacity=capacity,
                active_coordinates=selected[support].tolist(),
                focal_metadata_modified=False, output_mode=generator.output_mode,
                changes_write_budget=variant == 'no_support_mask',
                reference_support_capacity=24,
                support_claim=('all_legal28_after_original_SPEAR_prefilter'
                               if variant == 'no_support_mask' else 'frozen_SAGE24'),
                no_generator_or_no_attack_control=False,
                acceptance='registered path and permission checks; no ASR minimum or required drop')


def _save_contract(folder, phase, contract):
    path = Path(folder) / ('strict_' + phase + '_contract.json')
    with path.open('x') as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fit_victim(generator, inputs, selection, bundle, *, variant, seed, output_dir,
               epochs=200, device='cuda:0', progress=None):
    """Fit the unchanged fresh GraphSAGE with an explicit support capacity."""
    contract = validate_contract(generator, variant,
                                 expected_support=bundle.manifest['hard_support'])
    model, record = original.fit_victim(generator, inputs, selection, bundle,
        seed=seed, output_dir=output_dir, epochs=epochs, device=device, progress=progress,
        support_capacity=contract['support_capacity'])
    _save_contract(output_dir, 'victim', contract)
    return model, dict(record, strict_deletion_contract=contract)


def evaluate_case(generator, victim, clean_reference, inputs, *, variant,
                  device='cuda:0', output_dir, expected_support=None):
    """Score the same fixed DEV population using the original metric rules."""
    contract = validate_contract(generator, variant, expected_support=expected_support)
    report = original.evaluate_case(generator, victim, clean_reference, inputs,
        device=device, output_dir=output_dir, support_capacity=contract['support_capacity'])
    _save_contract(output_dir, 'evaluation', contract)
    return dict(report, strict_deletion_contract=contract)
