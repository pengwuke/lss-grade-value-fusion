"""Apply frozen fusion rules only; this module has no weight search."""

import numpy as np

from .contracts import fold_index


def _probabilities(values, reference_mass=False):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] < 1:
        raise ValueError("Expected a nonempty N x 4 probability array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    tolerance = 3.5e-6 if reference_mass else 1e-6
    if not np.allclose(values.sum(axis=1), 1.0, atol=tolerance, rtol=0):
        raise ValueError("Live-model probabilities do not sum to one; archived-score exceptions need their original identity checks")
    return values


def fuse_pair(image_probabilities, deit_probabilities, image_weight, reference_image=False):
    image = _probabilities(image_probabilities, reference_mass=reference_image)
    deit = _probabilities(deit_probabilities)
    weight = float(image_weight)
    if image.shape != deit.shape or not 0 <= weight <= 1:
        raise ValueError("Fusion requires aligned arrays and a weight in [0,1]")
    return weight * image + (1.0 - weight) * deit


def common5_fusion(coral_probabilities, deit_probabilities, outer_fold, strategy, config):
    fold = fold_index(outer_fold)
    if fold not in range(5) or strategy not in {"R10", "R11"}:
        raise ValueError("Expected one common5 outer fold and strategy R10 or R11")
    weight = config["common5"][strategy + "_coral_weights"][fold]
    # Frozen common5 producers parse branch probabilities into float64 before
    # weighting. Promote live float32 outputs at this protocol boundary too.
    return fuse_pair(np.asarray(coral_probabilities, dtype=np.float64),
                     np.asarray(deit_probabilities, dtype=np.float64), weight)


def source94_fusion(coral_by_fold, deit_by_fold, config):
    if len(coral_by_fold) != 5 or len(deit_by_fold) != 5:
        raise ValueError("Source94 requires all five frozen model pairs")
    weights = np.asarray(config["source94"]["coral_weights"], dtype=float)
    if weights.shape != (5,) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Source94 requires exactly five finite fusion weights in [0,1]")
    fused = [fuse_pair(c, d, w) for c, d, w in zip(coral_by_fold, deit_by_fold, weights)]
    if len({p.shape for p in fused}) != 1:
        raise ValueError("All folds must predict the same aligned source94 ROI order")
    return np.stack(fused, axis=0).mean(axis=0)


def matched_common5_fusion(image_scores, deit_probabilities, outer_fold, arm, strategy, config):
    fold = fold_index(outer_fold)
    settings = config["matched_common5"]
    if fold not in range(5) or arm not in settings["selected_image_weights"]:
        raise ValueError("Unknown matched branch or fold")
    if strategy == "fixed":
        weight = settings["fixed_image_weight"]
    elif strategy == "selected":
        weight = settings["selected_image_weights"][arm][fold]
    else:
        raise ValueError("Expected fixed or selected frozen fusion")
    return fuse_pair(np.asarray(image_scores, dtype=np.float64),
                     np.asarray(deit_probabilities, dtype=np.float64), weight,
                     reference_image=arm == "PHADKE_P4C")


def source94_reference_average(scores_by_fold):
    if len(scores_by_fold) != 5:
        raise ValueError("SR1 requires five reference model score arrays")
    checked = [_probabilities(p, reference_mass=True) for p in scores_by_fold]
    if len({p.shape for p in checked}) != 1:
        raise ValueError("All SR1 folds must predict the same aligned ROI order")
    return np.stack(checked, axis=0).mean(axis=0)


def argmax_grades(probabilities, reference_scores=False):
    return _probabilities(probabilities, reference_mass=reference_scores).argmax(axis=1)
