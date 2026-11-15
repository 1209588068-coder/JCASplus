"""Fresh author-model ranking and attribution for full-role AV2 repetitions.

Author GCN/MLP forwards, entropy score, predicted-class round robin and SAGE
estimator are preserved, with the continuous zero-term extension 0*log(0)=0.
Training accumulates globally normalized gradients
over disconnected scene batches and steps Adam once per epoch. The quadratic
author round-robin implementation is replaced by an output-equivalent linear
interleave, with independent regression tests. An explicitly registered recovery
may restore this same full-data run's ranking checkpoint; no historical attack
model or ranking is loaded. This module operates only in the authorized offline
research scope.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .av2_formal_classifiers import batches, fit_model, sorted_indices, EXECUTION
from .full_ablation_evaluation import _cuda_device, _fp32
from .full_ablation_training import _cpu_state, _tensor_hash
from .outer_selection import select_strict_negative_outer
from .womd_execution import configure_message_passing
from .womd_pilot_selection import (
    create_author_runtime, _import_author_runtime, candidate_groups, exact_support, PROTECTED,
)


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def predicted_class_round_robin(sorted_ids, sorted_predicted_classes):
    """Exact binary-class author order, without its repeated scans/catenations."""
    if (sorted_ids.dtype != torch.long or sorted_ids.ndim != 1
            or sorted_predicted_classes.shape != sorted_ids.shape
            or sorted_predicted_classes.device != sorted_ids.device
            or not bool(((sorted_predicted_classes == 0) | (sorted_predicted_classes == 1)).all())):
        raise ValueError('Binary predicted classes must align with sorted candidate IDs')
    zero = sorted_ids[sorted_predicted_classes == 0]
    one = sorted_ids[sorted_predicted_classes == 1]
    common = min(len(zero), len(one))
    alternating = torch.stack((zero[:common], one[:common]), dim=1).reshape(-1)
    return torch.cat((alternating, zero[common:], one[common:]))


def rank_candidates(logits, candidates, entropy_function):
    """Author score with only its exact-zero entropy boundary defined.

    The author passes log probabilities, divides by their negative row sum and
    evaluates -sum(q*log2(q)). A saturated FP32 log_softmax can contain exact
    zero, whose entropy term has limit zero but naively evaluates to NaN.
    Non-boundary scores are returned by the author function without changing a
    single arithmetic operation; no epsilon, softmax or score clipping is used.
    """
    if logits.shape != (len(candidates), 2):
        raise ValueError('Expected two-class candidate log probabilities')
    if (not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
            or bool((logits > 0).any())):
        raise FloatingPointError('Candidate log probabilities must be finite and nonpositive')
    denominator = logits.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(denominator).all()) or bool((denominator >= 0).any()):
        raise FloatingPointError('Author score requires finite strictly negative row sums')
    normalized = logits / denominator
    if not bool(torch.isfinite(normalized).all()) or bool((normalized < 0).any()):
        raise FloatingPointError('Invalid normalized author score inputs')
    score = entropy_function(logits)
    if score.shape != (len(candidates),):
        raise ValueError('Author entropy score must align with candidate rows')
    boundary = (normalized == 0).any(dim=1)
    if bool(boundary.any()):
        values = normalized[boundary]
        positive = values > 0
        terms = torch.zeros_like(values)
        terms[positive] = values[positive] * values[positive].log2()
        score = score.clone()
        score[boundary] = -terms.sum(dim=1)
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError('Author entropy-style score nonfinite outside the exact-zero boundary')
    order = score.argsort(descending=True)
    ranking = predicted_class_round_robin(candidates[order], logits.argmax(1)[order])
    return ranking, score


def _gather_features(data_dir, row_ids, *, max_nodes, device):
    wanted = torch.as_tensor(row_ids, dtype=torch.long).cpu()
    result = torch.empty((len(wanted), 784), dtype=torch.float32)
    seen = torch.zeros(len(wanted), dtype=torch.bool)
    for batch in batches(data_dir, 'train', max_nodes, device):
        start, stop = int(batch['global_start']), int(batch['global_stop'])
        slots = torch.nonzero((wanted >= start) & (wanted < stop), as_tuple=False).flatten()
        if len(slots):
            result[slots] = batch['x'][(wanted[slots]-start).to(batch['x'].device)].cpu()
            seen[slots] = True
    if not bool(seen.all()):
        raise ValueError('Attribution rows absent from full TRAIN stream')
    return result.numpy()


@_fp32()
def fit_source_selection(data_dir, *, seed, output_dir, indices=None,
                         attachment_count=26, outer_extra_count=512, model_epochs=200,
                         sage_n_permutations=2048, sage_batch_size=64,
                         explanation_count=128, background_count=32,
                         max_nodes=16384, device='cuda:0', progress=None,
                         ranking_resume_from=None):
    from .av2_formal_data import dataset_identity, load_role_arrays, make_indices
    dataset, source_split = dataset_identity(data_dir)
    device = _cuda_device(device)
    if any(type(x) is not int or x < 1 for x in (attachment_count, model_epochs,
            sage_n_permutations, sage_batch_size, explanation_count, background_count)):
        raise ValueError('Positive registered selection budgets required')
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    binding = create_author_runtime(directory/'author_runtime')
    binding['author_source_algorithm_changes'] = binding.pop('algorithm_changes')
    binding['execution_adaptation'] = dict(EXECUTION,
        author_GCN_and_MLP_forward_unchanged=True,
        author_loss_Adam_hyperparameters_and_strict_dev_accuracy_rule_unchanged=True,
        ranking='same entropy and argsort; exact binary predicted-class round-robin vector interleave',
        score_zero_boundary='exact zero normalized terms contribute zero; other author score arithmetic unchanged',
        attribution='same author SAGE estimator; unchanged av2-formal-sage TRAIN sample namespace',
        author_fit_loop_replaced_by_scene_batch_gradient_accumulation=True)
    binding.update(dataset=dataset, source_dataset_split=source_split,
                   author_model_dataset_argument='AV2')
    binding['ranking_resume_from'] = (str(Path(ranking_resume_from).resolve())
        if ranking_resume_from is not None else None)
    (directory/'author_runtime/runtime_binding.json').write_text(json.dumps(binding, indent=2)+'\n')
    selector, MLP, sage = _import_author_runtime(directory/'author_runtime')
    population = load_role_arrays(data_dir, 'train')
    labels = torch.as_tensor(population['y']).cpu()
    eligible = torch.as_tensor(population['eligible']).cpu()
    if (labels.dtype != torch.long or eligible.dtype != torch.bool
            or eligible.shape != labels.shape):
        raise ValueError('Malformed TRAIN label/eligibility population')
    indices = make_indices(data_dir, seed) if indices is None else indices
    clean = sorted_indices(indices['idx_train'], len(labels))
    if not bool(eligible[clean].all()) or set(labels[clean].tolist()) != {0, 1}:
        raise ValueError('Clean TRAIN supervision must be eligible and contain both classes')
    candidate_mask = eligible & (labels == 0)
    candidate_mask[clean] = False
    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if len(candidates) < attachment_count + outer_extra_count:
        raise ValueError('Insufficient strict-negative TRAIN candidates; no replacement')
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Retain the registered author construction argument and sample hash
    # namespace across datasets; actual provenance is recorded separately.
    args = SimpleNamespace(dataset='AV2', model='GCN', hidden=80, dropout=.5,
                           train_lr=.01, weight_decay=.0005)
    shape_only = SimpleNamespace(x=torch.empty((0, 784)), y=torch.tensor([0, 1]))
    ranker = selector.model_construct(args, 'GCN', shape_only, device).to(device)
    configure_message_passing(ranker)
    if progress:
        progress(dict(stage='fresh_author_GCN_ranking', epochs=model_epochs))
    fitted = fit_model(ranker, data_dir, clean, seed=seed, output_dir=directory/'ranking_fit',
        role='fresh_author_GCN_ranking', epochs=model_epochs, max_nodes=max_nodes,
        device=device, checkpoint_rule='dev_accuracy', progress=progress,
        **({'resume_from': ranking_resume_from} if ranking_resume_from is not None else {}))
    ranker = fitted['model']
    ranking_fit_summary = fitted['summary']
    logits = torch.empty((len(labels), 2), dtype=torch.float32)
    cursor = 0
    with torch.no_grad():
        for batch in batches(data_dir, 'train', max_nodes, device):
            start, stop = int(batch['global_start']), int(batch['global_stop'])
            if start != cursor or stop-start != len(batch['x']):
                raise ValueError('TRAIN ranking batches must cover one contiguous population')
            logits[start:stop] = ranker(batch['x'], batch['edge_index']).cpu()
            cursor = stop
    if cursor != len(labels):
        raise ValueError('Incomplete full-TRAIN ranking inference')
    candidate_logits, candidate_ids = logits[candidates].to(device), candidates.to(device)
    ranking, scores = rank_candidates(candidate_logits, candidate_ids, selector.entropy)
    zero_boundary_rows = int(((candidate_logits / candidate_logits.sum(1, keepdim=True)) == 0)
        .any(dim=1).sum())
    if len(ranking) != len(candidates) or not torch.equal(ranking.sort().values.cpu(), candidates):
        raise ValueError('Ranked list differs from complete eligible candidate pool')
    attach = ranking[:attachment_count].cpu()
    torch.save(_cpu_state(ranker), directory/'ranking_gcn.pt')
    torch.save(dict(candidates=candidates, ranked=ranking.cpu(), score=scores.cpu()), directory/'ranking.pt')
    del logits, candidate_logits, candidate_ids, ranking, scores, ranker, fitted
    unused = torch.ones(len(labels), dtype=torch.bool)
    unused[clean] = False
    unused[attach] = False
    selected = dict(idx_train=clean, idx_attach=attach,
                    idx_unlabeled=torch.nonzero(unused, as_tuple=False).flatten(), train_labels=labels.clone())
    outer = select_strict_negative_outer(dict(y=labels, eligible=eligible), selected,
                                         extra_count=outer_extra_count, seed=seed)
    selected['fixed_outer'] = outer['outer']
    np.random.seed(seed)
    torch.manual_seed(seed)
    mlp = MLP(nfeat=784, nhid=80, nclass=2, dropout=.5, lr=.01,
              weight_decay=.0005, device=device).to(device)
    if progress:
        progress(dict(stage='fresh_author_MLP', epochs=model_epochs))
    fitted = fit_model(mlp, data_dir, clean, seed=seed, output_dir=directory/'attribution_fit',
        role='fresh_author_MLP', model_kind='mlp', epochs=model_epochs,
        max_nodes=max_nodes, device=device, checkpoint_rule='dev_accuracy', progress=progress)
    mlp = fitted['model']
    torch.save(_cpu_state(mlp), directory/'attribution_mlp.pt')
    count = max(explanation_count, background_count)
    if len(clean) < count:
        raise ValueError('Insufficient labeled TRAIN rows for fixed attribution sample budget')
    ordered = heapq.nsmallest(count, clean.tolist(),
        key=lambda i: hashlib.sha256(f'{seed}:av2-formal-sage:{i}'.encode()).digest())
    features = _gather_features(data_dir, ordered, max_nodes=max_nodes, device='cpu')
    explanation = np.asarray(ordered[:explanation_count], dtype=np.int64)
    background = np.asarray(ordered[:background_count], dtype=np.int64)
    imputer = sage.MarginalImputer(mlp, features[:background_count])
    estimator = sage.PermutationEstimator(imputer, 'mse', n_jobs=1, random_state=seed)
    if progress:
        progress(dict(stage='fresh_author_SAGE', dimensions=784, permutations=sage_n_permutations))
    values, std = estimator(features[:explanation_count], np.eye(2)[labels[explanation].numpy()],
        batch_size=sage_batch_size, n_permutations=sage_n_permutations,
        detect_convergence=False, verbose=True, bar=False).save_num()
    values, std = np.asarray(values), np.asarray(std)
    if (values.shape != (784,) or std.shape != (784,)
            or not np.isfinite(values).all() or not np.isfinite(std).all()):
        raise FloatingPointError('Malformed/nonfinite fresh author SAGE attribution')
    coordinates = torch.as_tensor(np.argsort(values)[::-1][:30].copy(), dtype=torch.long, device=device)
    groups, metadata = candidate_groups(coordinates)
    for item in metadata:
        item['normalized_units'] = f'{dataset} registered TRAIN column mean/std; clip [-10,10]'
    exact_support(coordinates, groups, torch.ones(len(groups)), budget=24)
    np.savez(directory/'attribution.npz', values=values, std=std,
             explanation=explanation, background=background)
    torch.save(selected, directory/'indices.pt')
    summary = dict(status='FRESH_SELECTION_COMPLETE', dataset=dataset,
        source_dataset_split=source_split, seed=seed,
        coordinates=coordinates.tolist(), groups=groups, metadata=metadata,
        legal_capacity=sum(map(len, groups)), protected_coordinates=list(PROTECTED),
        protected_in_candidates=[c for c in coordinates.tolist() if c in PROTECTED],
        candidate_policy='strict_fresh_top30_then_protect_focal_only_no_refill',
        grouping='copy_identity_intersect_fresh_candidates_no_copy_completion',
        interpretation='derived feature space only; availability-field edits are not physical observations',
        train_labeled_count=len(clean), train_pair_nodes=len(labels), candidate_count=len(candidates),
        attachment_count=len(attach), outer_selection=outer['selection'],
        ranking='author_log_softmax_divide_row_sum_entropy_then_predicted_class_round_robin',
        ranking_numerics=dict(zero_term_rule='0*log2(0)=0', zero_boundary_rows=zero_boundary_rows,
            ordinary_author_scores_unchanged=True, epsilon_added=False,
            softmax_substitution=False, invalid_inputs_rejected=True),
        ranking_recovery=dict(source_directory=binding['ranking_resume_from'],
            resumed=ranking_resume_from is not None,
            fit_summary=ranking_fit_summary if ranking_resume_from is not None else None),
        same_scene_multiple_attachments_allowed=True,
        source_runtime_binding_sha256=hashlib.sha256((directory/'author_runtime/runtime_binding.json').read_bytes()).hexdigest(),
        checkpoint_rule='author_GCN_and_MLP_highest_DEV_accuracy_strict_improvement',
        sage=dict(explanation_count=explanation_count, background_count=background_count,
            permutations=sage_n_permutations, batch_size=sage_batch_size,
            estimator='author_PermutationEstimator_mse_raw_MLP_outputs',
            sample_order='sha256(seed:av2-formal-sage:TRAIN_row_id)', convergence_claim=False),
        historical_checkpoint_loaded=False, historical_learned_support_loaded=False,
        full_pipeline_stage='fresh_selection', execution=dict(EXECUTION, max_nodes=max_nodes),
        precision=dict(dtype='float32', tf32=False, amp=False),
        supervision_sha256=_tensor_hash(clean))
    _write(directory/'selection.json', summary)
    return dict(indices=selected, coordinates=coordinates, summary=summary)
