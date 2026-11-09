"""Read explicit user-supplied predictions; join systems on stable ROI identity."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

COLUMNS = {"roi_id", "patient_id", "outer_fold", "y_true", "y_pred"}


def integer(value, field, allowed=None):
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"{field} must be a canonical nonnegative integer")
    result = int(value)
    if allowed is not None and result not in allowed:
        raise ValueError(f"{field} outside allowed range")
    return result


def _identifier(value, field):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be nonempty, with no outer whitespace")
    return value


def read_predictions(path, *, _contents=None):
    path = Path(path).expanduser()
    contents = path.read_bytes() if _contents is None else _contents
    with io.StringIO(contents.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or set(reader.fieldnames) != COLUMNS
                or len(reader.fieldnames) != len(COLUMNS)):
            raise ValueError("CSV header must contain exactly: " + ",".join(sorted(COLUMNS)))
        rows = {}
        patients = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("CSV has too many or missing cells")
            roi = _identifier(row["roi_id"], "roi_id")
            patient = _identifier(row["patient_id"], "patient_id")
            fold = integer(row["outer_fold"], "outer_fold")
            truth = integer(row["y_true"], "y_true", range(4))
            prediction = integer(row["y_pred"], "y_pred", range(4))
            if roi in rows:
                raise ValueError("Duplicate roi_id")
            if patient in patients and patients[patient] != fold:
                raise ValueError("A patient occurs in more than one outer fold")
            patients[patient] = fold
            rows[roi] = (patient, fold, truth, prediction)
    if not rows:
        raise ValueError("Empty predictions CSV")
    folds = set(patients.values())
    if folds not in ({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}):
        raise ValueError("Matched five-fold inputs require folds 0..4 or 1..5")
    return rows


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def load_aligned(manifest_path, systems):
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict) or set(manifest) != {"predictions"} or not isinstance(manifest["predictions"], dict):
        raise ValueError("Input JSON must contain only a predictions mapping")
    mapping = manifest["predictions"]
    if set(mapping) != set(systems):
        raise ValueError("Prediction system names differ from the frozen analysis plan")
    reference = None
    keys = None
    columns = []
    provenance = []
    realigned = []
    for name in systems:
        if not isinstance(mapping[name], str) or not mapping[name]:
            raise ValueError("Prediction paths must be nonempty strings")
        path = Path(mapping[name]).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        contents = path.read_bytes()
        rows = read_predictions(path, _contents=contents)
        if reference is None:
            reference = rows
            keys = list(rows)
        if set(rows) != set(reference):
            raise ValueError(f"{name}: ROI identity sets differ")
        if list(rows) != keys:
            realigned.append(name)
        for roi in keys:
            if rows[roi][:3] != reference[roi][:3]:
                raise ValueError(f"{name}: patient, fold or truth differs for a matched ROI")
        columns.append([rows[k][3] for k in keys])
        provenance.append({"system": name, "file_name": path.name,
                           "sha256": hashlib.sha256(contents).hexdigest()})
    y = np.array([reference[k][2] for k in keys], dtype=np.int64)
    patients = np.array([reference[k][0] for k in keys], dtype=str)
    folds = np.array([reference[k][1] for k in keys], dtype=np.int64)
    predictions = np.array(columns, dtype=np.int64).T
    return y, predictions, patients, folds, {"input_files": provenance,
                                            "systems_reordered_by_roi_id": realigned}
