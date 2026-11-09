"""Descriptive joint branch errors from explicitly supplied frozen predictions.

Input has y_true, outer_fold and the five named class-prediction columns.
No rows, patient IDs, confidence intervals, selection or inference are emitted.
"""
import csv
import json
from pathlib import Path

PAIRS = [
    ("R09_SELECTED_ALPHA", "ORIGINAL_DEIT", "R09_SELECTED_ALPHA_VSEL"),
    ("A0_NATURAL", "ORIGINAL_DEIT", "A0_NATURAL_VSEL"),
]


def metrics(rows, column):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, cohen_kappa_score,
                                 mean_absolute_error, confusion_matrix)
    y = [int(r["y_true"]) for r in rows]
    p = [int(r[column]) for r in rows]
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "weighted_f1": f1_score(y, p, labels=[0, 1, 2, 3], average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y, p, labels=[0, 1, 2, 3], weights="quadratic"),
        "mae": mean_absolute_error(y, p),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2, 3]).tolist(),
    }


def counts(rows, a, b, f):
    cells = {k: {"count": 0, "fused_correct": 0} for k in
             ["both_correct", "cnn_only_correct", "deit_only_correct", "both_wrong"]}
    correct = {name: 0 for name in [a, b, f]}
    distance = {name: 0 for name in [a, b, f]}
    repairs = {name: {"repaired": 0, "lost": 0} for name in [a, b]}
    disagree = 0
    for r in rows:
        y, x, z, v = (int(r[n]) for n in ["y_true", a, b, f])
        assert all(0 <= n <= 3 for n in [y, x, z, v])
        key = ("both_correct" if x == y and z == y else
               "cnn_only_correct" if x == y else
               "deit_only_correct" if z == y else "both_wrong")
        cells[key]["count"] += 1
        cells[key]["fused_correct"] += int(v == y)
        disagree += int(x != z)
        for name, pred in [(a, x), (b, z), (f, v)]:
            correct[name] += int(pred == y)
            distance[name] += abs(pred - y)
        for name, pred in [(a, x), (b, z)]:
            repairs[name]["repaired"] += int(pred != y and v == y)
            repairs[name]["lost"] += int(pred == y and v != y)
    for name, value in repairs.items():
        value["net_correct"] = value["repaired"] - value["lost"]
        value["net_absolute_grade_error"] = distance[f] - distance[name]
        assert value["net_correct"] == correct[f] - correct[name]
    assert sum(v["count"] for v in cells.values()) == len(rows)
    assert sum(v["fused_correct"] for v in cells.values()) == correct[f]
    assert cells["both_correct"]["fused_correct"] == cells["both_correct"]["count"]
    return {"roi_count": len(rows), "prediction_disagreements": disagree,
            "joint_correctness": cells, "correct_counts": correct,
            "absolute_grade_error_sums": distance, "fusion_vs_branch": repairs}


def audit(rows):
    for r in rows:
        assert 0 <= int(r["outer_fold"]) <= 4
    results = []
    for a, b, f in PAIRS:
        results.append({"cnn": a, "deit": b, "fusion": f,
                        "pooled": counts(rows, a, b, f),
                        "by_truth_grade": {str(g): counts([r for r in rows if int(r["y_true"]) == g], a, b, f) for g in range(4)},
                        "by_outer_fold": {str(g): counts([r for r in rows if int(r["outer_fold"]) == g], a, b, f) for g in range(5)}})
    columns = list(dict.fromkeys(n for pair in PAIRS for n in pair))
    return {"analysis": "descriptive fixed-prediction branch error audit",
            "pairs": results, "pooled_metrics": {n: metrics(rows, n) for n in columns},
            "new_bootstrap": False, "new_hypothesis_test": False,
            "new_training": False, "new_model_inference": False, "new_selection": False,
            "patient_rows_or_identifiers_exported": False,
            "limitations": "Post-hoc descriptive error overlap and fusion repair counts; no interaction test or unique auxiliary contribution is established."}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8-sig") as stream:
        result = audit(list(csv.DictReader(stream)))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
