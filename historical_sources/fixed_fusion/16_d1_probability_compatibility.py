"""D1-anchored mass compatibility for one exact P4C archive.

Primary arrays are never clipped, scaled, rounded, or overwritten. Temporary
row-simplex copies serve only a pre-metric decision-invariance check. Any changed
hard prediction blocks efficacy computation; no performance-dependent choice.
This module does not establish the historical probability producer's cause.
"""
from __future__ import annotations
import numpy as np

class NumericCompatibilityError(RuntimeError):
    pass

def need(ok, code):
    if not bool(ok):
        raise NumericCompatibilityError(code)

ARCHIVE_SHA = '4089ed9d6e446abd76b16433f54a41a38c263a612bacefdee250d8f61e21804f'
MEMBER_SHA = '610aef7113fb921ab071c84c9823e85b9b3c56af389cc6672d112c975dc58803'
ORIGINAL_ATOL = 2e-6
EPSILON = 1e-6
FLOOR_ATOL = 1e-12
FP32_SLACK = 8 * np.finfo(np.float32).eps


def validate_evidence(review, cfg):
    need(review.get('schema') == 'W25_A4_D1_READONLY_PROBABILITY_REVIEW_V1', 'D1_SCHEMA')
    need(review.get('execution_status') == 'COMPLETE_READONLY_DIAGNOSTIC_REVIEW_PENDING', 'D1_STATUS')
    need(review.get('read_source_files_unchanged') is True, 'D1_SOURCE_CHANGED')
    for key in ('new_training', 'new_model_inference', 'new_fusion', 'new_bootstrap',
                'probability_mutation', 'normalization', 'acceptance_tolerance_changed', 'old_run_restart'):
        need(review.get(key) is False, 'D1_SCOPE_' + key.upper())
    ident = review['p4c_raw_table_identity']
    need(cfg['sha256'] == ident['archive_sha256'] == ARCHIVE_SHA, 'D1_EXACT_P4C_ARCHIVE')
    need(cfg['member_sha256'] == ident['member_sha256'] == MEMBER_SHA, 'D1_EXACT_P4C_MEMBER')
    need(cfg['member'] == ident['member'] == '01_OOF_PREDICTIONS.csv', 'D1_MEMBER_NAME')
    need(ident['rows'] == 2978, 'D1_ROW_COUNT')
    n = review['numeric_diagnostics']
    need(n['original_gate_atol'] == ORIGINAL_ATOL and n['original_gate_rtol'] == 0,
         'D1_ORIGINAL_GATE')
    need(n['rows_above_original_atol'] == n['rows_positive_excess_above_original_atol'] == 7
         and n['rows_negative_deficit_below_original_atol'] == 0, 'D1_FAILURE_PROFILE')
    need(review['source_evidence']['hash_anchored_or_archive_embedded_candidate_count'] == 0,
         'D1_SOURCE_SCOPE_CHANGED_REVIEW_REQUIRED')
    need(review['diagnostic_outcome'] ==
         'POSITIVE_EXCESS_COMPATIBLE_WITH_1E6_FLOOR_NOT_CAUSALLY_CONFIRMED', 'D1_CAUSE_NOT_CONFIRMED')


def mass_profile(p, folds):
    """No mutation; caller owns the raw float64 archive values."""
    p = np.asarray(p, dtype=np.float64)
    folds = np.asarray(folds)
    need(p.ndim == 2 and p.shape[1] == 4 and len(p) > 0, 'P4C_PROB_SHAPE')
    need(folds.shape == (len(p),) and np.isin(folds, np.arange(5)).all(), 'P4C_FOLD_SHAPE')
    need(np.isfinite(p).all(), 'P4C_PROB_NONFINITE')
    need((p >= 0).all() and (p <= 1).all(), 'P4C_PROB_RANGE')
    need((p >= EPSILON - FLOOR_ATOL).all(), 'P4C_PROFILE_BELOW_REVIEWED_FLOOR')
    floor_hits = np.isclose(p, EPSILON, atol=FLOOR_ATOL, rtol=0)
    hits = floor_hits.sum(axis=1)
    need((hits <= 3).all(), 'P4C_ALL_FOUR_AT_FLOOR')
    sums = p.sum(axis=1)
    err = sums - 1.0
    # A bounded representational compatibility criterion, NOT proof of origin.
    # 8*float32 epsilon is the conservative D1 diagnostic roundoff budget.
    need((err >= -FP32_SLACK).all(), 'P4C_MASS_DEFICIT_OUTSIDE_REVIEW')
    need((err <= hits * EPSILON + FP32_SLACK).all(), 'P4C_MASS_OUTSIDE_FLOOR_ENVELOPE')
    need((np.abs(err) <= 3 * EPSILON + FP32_SLACK).all(), 'P4C_MASS_OUTSIDE_GLOBAL_ENVELOPE')
    return {
        'rows': len(p), 'values': int(p.size), 'finite_values': int(np.isfinite(p).sum()),
        'nonfinite_values': int((~np.isfinite(p)).sum()),
        'negative_values': int((p < 0).sum()), 'values_above_one': int((p > 1).sum()),
        'min_value': float(p.min()), 'max_value': float(p.max()),
        'rows_above_original_atol': int((np.abs(err) > ORIGINAL_ATOL).sum()),
        'rows_positive_excess_above_original_atol': int((err > ORIGINAL_ATOL).sum()),
        'rows_negative_deficit_below_original_atol': int((err < -ORIGINAL_ATOL).sum()),
        'max_positive_excess': float(np.maximum(err, 0).max()),
        'max_negative_deficit': float(np.maximum(-err, 0).max()),
        'minimum_floor_hypothesis': {
            'floor_hits_total': int(hits.sum()),
            'rows_by_floor_hit_count': {str(k): int((hits == k).sum()) for k in range(5)},
            'rows_not_explained_by_floor_mass_envelope': int(
                ((err < -FP32_SLACK) | (err > hits * EPSILON + FP32_SLACK)).sum()),
        },
        'per_fold': [{'outer_fold': f, 'rows': int((folds == f).sum()),
                      'max_absolute_sum_error': float(np.abs(err[folds == f]).max()) if (folds == f).any() else 0.0,
                      'rows_failing_original_sum_gate': int((np.abs(err[folds == f]) > ORIGINAL_ATOL).sum())}
                     for f in range(5)],
    }


def match_reviewed_profile(profile, expected):
    integers = ('rows', 'values', 'finite_values', 'nonfinite_values', 'negative_values',
                'values_above_one', 'rows_above_original_atol',
                'rows_positive_excess_above_original_atol', 'rows_negative_deficit_below_original_atol')
    for key in integers:
        need(profile[key] == expected[key], 'D1_REPLAY_COUNT_' + key.upper())
    for key in ('min_value', 'max_value', 'max_positive_excess', 'max_negative_deficit'):
        need(abs(profile[key] - expected[key]) <= 1e-12, 'D1_REPLAY_FLOAT_' + key.upper())
    for key in ('floor_hits_total', 'rows_by_floor_hit_count', 'rows_not_explained_by_floor_mass_envelope'):
        need(profile['minimum_floor_hypothesis'][key] == expected['minimum_floor_hypothesis'][key],
             'D1_REPLAY_FLOOR_' + key.upper())
    need(len(expected['per_fold']) == 5, 'D1_REPLAY_FOLD_COUNT')
    for got, exp in zip(profile['per_fold'], expected['per_fold']):
        for key in ('outer_fold', 'rows', 'rows_failing_original_sum_gate'):
            need(got[key] == exp[key], 'D1_REPLAY_FOLD_' + key.upper())
        need(abs(got['max_absolute_sum_error'] - exp['max_absolute_sum_error']) <= 1e-12,
             'D1_REPLAY_FOLD_MASS')


def reviewed_raw_p4c(rows, cfg, review):
    validate_evidence(review, cfg)
    p = np.asarray([[float(row['p' + str(j)]) for j in range(4)] for row in rows], dtype=np.float64)
    folds = np.asarray([int(float(row['outer_fold'])) for row in rows], dtype=int)
    profile = mass_profile(p, folds)
    match_reviewed_profile(profile, review['numeric_diagnostics'])
    return p, {
        'policy': 'EXACT_D1_ARCHIVE_PROFILE_COMPATIBILITY_NOT_GLOBAL_TOLERANCE_RELAXATION',
        'raw_array_modified': False, 'producer_cause_confirmed': False,
        'producer_source_chain_closed': False, 'original_sum_atol': ORIGINAL_ATOL,
        'p4c_only_compatibility_gate_changed': True, 'other_branch_gates_unchanged': True,
        'floor_envelope_epsilon': EPSILON, 'roundoff_budget': float(FP32_SLACK),
        'per_row_positive_bound': 'number_of_1e-6_floor_hits * 1e-6 + 8 * float32_epsilon',
        'profile_reproduced': profile,
        'downstream_decision_invariance_still_required': True,
    }


def decision_invariance(branches, deit, names):
    """Label-free sensitivity audit; compare 11 hard-output arrays before metrics.

    raw: exact original arrays and 0.40/0.60 rule (primary).
    p4c-only: normalize only P4C to sum1 on a temporary copy.
    uniform: normalize each input branch and same DeiT on temporary copies.
    All must yield identical predictions. No variant is selected on labels.
    """
    need(len(branches) == 5 and len(names) == 11, 'SENSITIVITY_SYSTEM_COUNT')
    arrays = [np.asarray(p, dtype=np.float64) for p in branches] + [np.asarray(deit, dtype=np.float64)]
    n = len(arrays[-1])
    need(all(p.shape == (n, 4) and np.isfinite(p).all() and (p >= 0).all()
             and (p.sum(axis=1) > 0).all() for p in arrays), 'SENSITIVITY_INPUT')
    raw_bytes = [p.tobytes() for p in arrays]
    normalized = [p / p.sum(axis=1, keepdims=True) for p in arrays]
    raw = arrays + [0.40 * p + 0.60 * arrays[-1] for p in arrays[:5]]
    only_p4 = arrays[:4] + [normalized[4], arrays[5]]
    only_p4 += [0.40 * p + 0.60 * arrays[-1] for p in only_p4[:5]]
    uniform = normalized + [0.40 * p + 0.60 * normalized[-1] for p in normalized[:5]]
    raw_pred = [p.argmax(axis=1) for p in raw]
    counts = {}
    changes = {}
    for label, variant in [('normalize_p4c_copy_only', only_p4), ('normalize_all_input_copies', uniform)]:
        counts[label] = {name: int(np.count_nonzero(a != p.argmax(axis=1)))
                         for name, a, p in zip(names, raw_pred, variant)}
        changes[label] = {name: float(np.max(np.abs(a-p))) for name, a, p in zip(names, raw, variant)}
    # Final fused-score normalization never changes score ordering, checked explicitly.
    final_counts = {name: int(np.count_nonzero(pred != (p / p.sum(axis=1, keepdims=True)).argmax(axis=1)))
                    for name, pred, p in zip(names, raw_pred, raw)}
    unchanged = all(p.tobytes() == b for p,b in zip(arrays, raw_bytes))
    ok = unchanged and all(c == 0 for v in counts.values() for c in v.values()) and all(c == 0 for c in final_counts.values())
    margins = {name: float(np.min(np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]))
               for name,p in zip(names,raw)}
    return {
        'scope': 'hard-label six metrics and patient bootstrap; no claim about calibration or log loss',
        'temporary_simplex_copies_only': True, 'raw_input_arrays_unchanged': unchanged,
        'no_true_labels_consulted': True, 'no_variant_selected_by_performance': True,
        'primary_representation': 'unchanged archived values and fixed 0.40/0.60 scores',
        'mismatches_by_variant_and_system': counts,
        'final_score_normalization_argmax_mismatches': final_counts,
        'max_abs_score_change_by_variant': changes,
        'raw_min_top2_margin_by_system': margins,
        'all_11_decisions_invariant': ok,
        'metrics_and_bootstrap_allowed': ok,
        'nonzero_difference_action': 'STOP_BEFORE_METRICS_AND_BOOTSTRAP_NO_NORMALIZATION_SELECTION',
    }
