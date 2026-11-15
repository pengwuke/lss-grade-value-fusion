"""Create a complete descriptive five-fold report from explicitly supplied CSV.

This entry point is separate from the five frozen paired-inference modes. It
does not train, infer, choose weights, perform bootstrap inference or certify
that the input predictions came from held-out or selection-independent models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

try:
    from .five_fold_metrics import score, mean_sd
except ImportError:
    from five_fold_metrics import score, mean_sd


REQUIRED_COLUMNS = ("roi_id", "patient_id", "outer_fold", "y_true", "y_pred")
PROBABILITY_COLUMNS = ("p0", "p1", "p2", "p3")
PROBABILITY_SUM_TOLERANCE = 1e-4


class InputContractError(ValueError):
    """The caller's explicit CSV or complete-cohort contract is invalid."""


def _canonical_int(value, allowed, field):
    if not isinstance(value, str) or value not in {str(x) for x in allowed}:
        raise InputContractError(f"{field} must be a canonical integer in {tuple(allowed)}")
    return int(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_csv(path):
    """Read the exact bytes that are hashed; do not emit row or patient IDs."""
    path = Path(path).expanduser()
    raw = path.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise InputContractError("CSV headers must be present and unique")
        actual = set(headers)
        basic = set(REQUIRED_COLUMNS)
        complete = basic | set(PROBABILITY_COLUMNS)
        if actual not in (basic, complete):
            raise InputContractError("CSV must contain exactly the five required columns, optionally with all p0,p1,p2,p3")
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise InputContractError("CSV must be valid UTF-8 with well-formed rows") from exc
    return rows, hashlib.sha256(raw).hexdigest()


def build_report(rows, *, expected_patients, expected_rois,
                 probabilities_verified_oof=False, system_name="caller_supplied"):
    """Validate the full declared cohort, then report every outer fold and OOF.

    The probability flag is an explicit caller declaration of original OOF
    origin, not an independently established provenance certificate.
    """
    if isinstance(expected_patients, bool) or not isinstance(expected_patients, int) or expected_patients <= 0:
        raise InputContractError("expected_patients must be an explicit positive integer")
    if isinstance(expected_rois, bool) or not isinstance(expected_rois, int) or expected_rois <= 0:
        raise InputContractError("expected_rois must be an explicit positive integer")
    rows = list(rows)
    if not rows:
        raise InputContractError("CSV contains no predictions")
    has_probabilities = set(PROBABILITY_COLUMNS).issubset(rows[0])
    required_set = set(REQUIRED_COLUMNS) | (set(PROBABILITY_COLUMNS) if has_probabilities else set())
    if probabilities_verified_oof and not has_probabilities:
        raise InputContractError("OOF probabilities were declared, but the four probability columns are absent")
    roi_keys, patient_folds, fold_values, y_values, pred_values, probabilities = set(), {}, [], [], [], []
    for position, row in enumerate(rows, 1):
        if set(row) != required_set or any(value is None for value in row.values()):
            raise InputContractError(f"Malformed or inconsistent columns at record {position}")
        for field in ("roi_id", "patient_id"):
            value = row[field]
            if not isinstance(value, str) or not value or value.strip() != value:
                raise InputContractError(f"{field} must be a nonempty string without surrounding whitespace")
        roi, patient = row["roi_id"], row["patient_id"]
        if roi in roi_keys:
            raise InputContractError(f"Duplicate ROI identifier at record {position}")
        roi_keys.add(roi)
        fold = _canonical_int(row["outer_fold"], range(5), "outer_fold")
        label = _canonical_int(row["y_true"], range(4), "y_true")
        prediction = _canonical_int(row["y_pred"], range(4), "y_pred")
        if patient in patient_folds and patient_folds[patient] != fold:
            raise InputContractError("A patient occurs in more than one outer fold")
        patient_folds[patient] = fold
        fold_values.append(fold)
        y_values.append(label)
        pred_values.append(prediction)
        if has_probabilities:
            try:
                values = [float(row[key]) for key in PROBABILITY_COLUMNS]
            except (TypeError, ValueError) as exc:
                raise InputContractError(f"Non-numeric probabilities at record {position}") from exc
            if not all(math.isfinite(x) and x >= 0 for x in values):
                raise InputContractError(f"Probabilities must be finite and nonnegative at record {position}")
            if abs(sum(values) - 1.0) > PROBABILITY_SUM_TOLERANCE:
                raise InputContractError(f"Probability mass differs from one beyond tolerance at record {position}")
            if int(np.argmax(values)) != prediction:
                raise InputContractError(f"Probability argmax differs from the frozen hard prediction at record {position}")
            probabilities.append(values)
    if set(fold_values) != set(range(5)):
        raise InputContractError("All five complete outer folds 0,1,2,3,4 are required")
    if len(rows) != expected_rois or len(patient_folds) != expected_patients:
        raise InputContractError("Declared complete-cohort patient/ROI counts do not match the input")
    y, pred, fold_array = np.asarray(y_values), np.asarray(pred_values), np.asarray(fold_values)
    validated_probabilities = np.asarray(probabilities, dtype=np.float64) if has_probabilities else None
    scoring_probabilities = validated_probabilities if probabilities_verified_oof else None
    folds = []
    for fold in range(5):
        mask = fold_array == fold
        values = score(y[mask], pred[mask], scoring_probabilities[mask] if scoring_probabilities is not None else None)
        values.update({"outer_fold": fold,
                       "n_patients": sum(assigned == fold for assigned in patient_folds.values())})
        folds.append(values)
    pooled = score(y, pred, scoring_probabilities)
    pooled["n_patients"] = len(patient_folds)
    summary = {
        "schema": "paper2_portable_complete_five_fold_reporting_v1",
        "system": str(system_name),
        "scope": "Descriptive full-declared-cohort frozen-prediction report; no new training, selection or interval inference.",
        "input_contract": {"declared_expected_patients": expected_patients,
                           "declared_expected_ROIs": expected_rois,
                           "one_ROI_once": True, "one_patient_one_outer_fold": True,
                           "outer_folds": list(range(5)),
                           "identifiers_exported": False},
        "provenance_limit": "Structural checks and declared counts do not establish patient identity, authentic OOF origin, or training/selection independence. Audit those against retained source and model records separately.",
        "definitions": {
            "fold_mean_sd": "Arithmetic mean of all five complete outer-fold scores; sample SD uses ddof=1. Undefined or unavailable entries yield null, never a partial-fold estimate.",
            "pooled_OOF": "One hard prediction per ROI in the full declared cohort, weighted per ROI; not the arithmetic fold mean.",
            "macro_f1": "Original metric rule: macro average over the true/predicted class union. Per-grade P/R/F1 always lists grades 0..3; zero division uses zero.",
            "qwk": "Quadratic weights on grade labels 0,1,2,3; undefined values are JSON null.",
            "targeted_errors": "Normal-to-mild denominator true normal; mild-to-normal denominator true mild; moderate_correct is moderate recall; severe_underestimated is prediction<3; severe_predicted_normal_or_mild is prediction<=1.",
            "probability_scores": "Normalized only for probability scoring, with float64-epsilon clipped natural-log loss, four-class sum Brier (not divided by4), and top-label ECE with ten equal-width bins [0,.1),...,[.9,1]. No recalibration or hard-decoding change.",
        },
        "probability_status": {
            "four_columns_present": has_probabilities,
            "argmax_parity_verified": has_probabilities,
            "original_OOF_origin_declared_by_caller": bool(probabilities_verified_oof),
            "origin_independently_certified_by_this_tool": False,
            "computed": scoring_probabilities is not None,
            "not_computed_reason": None if scoring_probabilities is not None else "Original OOF probabilities absent or not explicitly declared by the caller; no VAL substitution.",
            "sum_absolute_tolerance": PROBABILITY_SUM_TOLERANCE,
            "max_original_sum_deviation": float(np.max(np.abs(validated_probabilities.sum(axis=1)-1))) if has_probabilities else None,
        },
        "folds": folds,
        "fold_mean_sample_SD": mean_sd(folds),
        "pooled_OOF": pooled,
    }
    return _json_safe(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--expected-patients", required=True, type=int)
    parser.add_argument("--expected-rois", required=True, type=int)
    parser.add_argument("--system-name", default="caller_supplied")
    parser.add_argument("--probabilities-verified-oof", action="store_true",
                        help="Declare supplied probabilities originate from the original OOF predictions; this tool cannot certify that provenance.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    destination = args.output.expanduser()
    if destination.exists():
        parser.error("Output already exists; reports are never overwritten")
    try:
        rows, digest = load_csv(args.predictions)
        report = build_report(rows, expected_patients=args.expected_patients,
                              expected_rois=args.expected_rois,
                              probabilities_verified_oof=args.probabilities_verified_oof,
                              system_name=args.system_name)
        report["input_csv_sha256"] = digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (InputContractError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
