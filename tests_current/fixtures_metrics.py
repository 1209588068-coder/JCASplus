"""Synthetic report fixtures from tests/test_spear_domain_training_reporting.py."""
from copy import deepcopy
from spear_current import domain_reporting as r


def target_metrics(slow=154, moving=53, slow_negative=156, moving_negative=55, *, reference=False):
    out = {}
    for regime, n, flips, negative in (
        ('all', 224, slow + moving, slow_negative + moving_negative),
        ('slow', 164, slow, slow_negative), ('moving', 60, moving, moving_negative),
    ):
        positive = 0 if reference else n - negative
        out[regime] = dict(targets=n, absolute_asr=(flips + positive) / n,
            triggered_target_class_count=flips + positive, incremental_asr=flips / n, incremental_flips=flips,
            conditional_flip=flips / negative if negative else None, clean_negative_targets=negative,
            clean_activation_rate=(n - negative) / n, target_probability_delta_mean=.2)
    return out


def normal_metrics(tp=228, fp=119, auc=.8978394238463592):
    tn, fn = 3910 - fp, 489 - tp
    return dict(normal_pairs=4399, normal_positive_pairs=489, normal_negative_pairs=3910,
        clean_pair_auc=auc, clean_pair_accuracy=(tn + tp) / 4399, clean_pair_positive_recall=tp / 489,
        clean_prediction_distribution=dict(class_0=tn + fn, class_1=tp + fp),
        clean_confusion_matrix=dict(true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp),
        all_clean_predictions_negative=(tp + fp == 0),
        clean_class_one_probability_min=.01, clean_class_one_probability_max=.99)


def collateral(adj=11):
    return dict(adj_negative=1202, adj_flip=adj, nonincident_negative=2484, nonincident_flip=9,
        adj_positive=199, adj_suppressed=7, adjacent_incremental_fp=adj / 1202,
        nonincident_incremental_fp=9 / 2484, adjacent_positive_suppression=7 / 199)


def fixture():
    original = dict(audit_counts={key: r.EXPECTED_COUNTS[key] for key in r.AUDIT_COUNT_KEYS},
        metrics=dict(
            victim=dict(**normal_metrics(), targets=target_metrics(), collateral=collateral()),
            clean_reference=dict(**normal_metrics(206, 119, .8958012332700485),
                targets=target_metrics(0, 0, 159, 57, reference=True), collateral=collateral(27))),
        original_val_arrays_loaded=False, original_test_arrays_loaded=False)
    return {arm: deepcopy(original) for arm in r.ARMS}, original


