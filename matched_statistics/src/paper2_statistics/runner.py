"""Small I/O wrapper around byte-preserved frozen analysis kernels."""
import hashlib
import json
from importlib.resources import files

import numpy as np

from .inputs import load_aligned
from .kernels import fusion as kernel


def load_plan(analysis):
    plans = json.loads(files("paper2_statistics").joinpath("plans.json").read_text(encoding="utf-8"))
    if analysis not in plans:
        raise ValueError("Unknown analysis")
    return plans[analysis]


def verify_kernels():
    root = files("paper2_statistics")
    provenance = json.loads(root.joinpath("KERNEL_PROVENANCE.json").read_text(encoding="utf-8"))
    for name, expected in provenance["installed_file_sha256"].items():
        actual = hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen analysis file fingerprint differs: {name}")
    return provenance


def run(inputs, analysis, iterations=None):
    plan = load_plan(analysis)
    if iterations is None:
        iterations = plan["repeats"]
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    provenance = verify_kernels()
    plan = load_plan(analysis)
    systems = plan["systems"]
    y, predictions, patients, folds, alignment = load_aligned(inputs, systems)
    points = {name: kernel.point(y, predictions[:, i]) for i, name in enumerate(systems)}
    fold_points = {str(int(fold)): {name: kernel.point(y[folds == fold], predictions[folds == fold, i])
                                 for i, name in enumerate(systems)} for fold in np.unique(folds)}
    if analysis == "r11_robustness":
        from .robustness import run_robustness
        return {"analysis": analysis,
                "analysis_scope": "Post-hoc multiplicity, grade-recall and patient/fold influence sensitivity; fixed predictions only",
                "paper_resampling_configuration": iterations == plan["repeats"],
                "paper_cohort_identity_verified": False,
                "input_identity_note": "Explicit CSV inputs do not certify paper cohort identity",
                "kernel_provenance": provenance, "input_alignment": alignment,
                "points_by_system": points, "points_by_outer_fold": fold_points,
                "robustness": run_robustness(y, predictions, patients, folds, iterations=iterations, seed=plan["seed"]),
                "model_training_or_weight_selection_performed": False,
                "multiple_comparison_correction": "Bonferroni-type percentile sensitivity for the specified ten-outcome family only"}
    if analysis == "single":
        if (list(kernel.base.ARMS) != systems or list(kernel.base.CONTRASTS)
                != [(c["terms"][0][0], c["terms"][1][0]) for c in plan["contrasts"]]):
            raise ValueError("Single-branch plan differs from the frozen A3 kernel")
        bootstrap = kernel.base.paired_patient_bootstrap(y, predictions, patients,
                                                        iterations=iterations, seed=plan["seed"], chunk=64)
    else:
        bootstrap = kernel.paired_fusion_bootstrap(y, predictions, patients, systems, plan["contrasts"],
                                                  iterations=iterations, seed=plan["seed"], chunk=32)
    return {"analysis": analysis,
            "analysis_scope": ("Post-hoc common5 full R11 versus natural CORAL same-DeiT selected-fusion direct comparison; fixed predictions, no new selection" if analysis == "r11_natural" else "New matched common5 analyses only; not historical, source94, or patient-stratified intervals"),
            "paper_resampling_configuration": iterations == plan["repeats"],
            "paper_cohort_identity_verified": False,
            "input_identity_note": "Explicit user-supplied CSV inputs; no claim that these are the paper's frozen prediction files",
            "kernel_provenance": provenance, "input_alignment": alignment,
            "points_by_system": points, "points_by_outer_fold": fold_points,
            "paired_bootstrap": bootstrap,
            "model_training_or_weight_selection_performed": False,
            "multiple_comparison_correction": False}
