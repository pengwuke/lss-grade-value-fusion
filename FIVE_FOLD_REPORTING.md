# Complete five-fold reporting and its limits

## Three different summaries

An official method assessment requires all five patient-level outer test folds, with each patient's images, sides, slices, ROIs and derived samples assigned to the same fold. Every ROI contributes exactly one held-out prediction to each system's OOF set. Do not select easier folds, substitute training seeds or label subsets for folds, or extrapolate from one inner-validation set.

Report (1) each complete outer fold, (2) the unweighted arithmetic fold mean with sample SD (ddof=1), and (3) the full pooled OOF result. The fold mean generally differs from pooled performance because fold sizes and class distributions differ. Fold SD describes between-fold variability; it is not a patient-paired confidence interval, and overlapping training sets do not make the five folds independent repeated experiments.

The full report includes Accuracy, balanced accuracy, Macro-F1, weighted F1, QWK and MAE, grade-specific precision/recall/F1, all four-class confusion counts, normal/mild confusions, moderate recall and severe underestimation. Severe underestimation means true grade 3 predicted below 3; the separate severe-to-normal/mild outcome uses predictions 0/1. Error rates state their true-grade denominators. Ordered numerical labels do not imply equally spaced clinical severity.

When authentic original OOF probabilities are available and match frozen hard predictions, also report multiclass log loss, four-class sum Brier score (not divided by 4) and top-label ECE using ten equal-width bins. Probability mass is normalized only for these scores, within the documented original sum tolerance; the frozen hard predictions are not changed. These probability-quality descriptions are not evidence that clinical risk is calibrated, safe or transportable, and no recalibration has been performed. When probability origin is absent or unverified, report the values as unavailable, never substitute inner-VAL scores.

Top-label ECE groups the maximum class confidence and whether the predicted grade was correct; it does not assess each grade probability separately. Its value depends on the chosen bins. Brier score and log loss reflect more than calibration alone, so a lower value does not isolate better calibration. These are fixed-prediction point summaries without new probability-score intervals. See USE_AND_REPORTING_FAQ.md for inference/evaluation inputs and interpretation.

The exact current aggregate is evidence/FULL_FIVE_FOLD_REPORTING.json (SHA256 5a75d0bca51494295af761f7192265eafc297dd7ad63a701fa7f77c6312a659b). It contains no patient/ROI identifiers, row-level predictions/probabilities or private paths.

## Current historical evidence

The companion full-fold aggregate report covers all 19 existing frozen common-five-fold systems, all five folds, fold mean/sample SD and pooled OOF, including grade/error and probability-quality summaries. It is a new descriptive reporting audit of existing predictions. It does not add training, model inference, selection, bootstrap intervals, significance tests or independent patients. Original six-metric pooled results and confusion matrices are retained.

The historical study had disjoint train/validation/test patient roles within each model. Its common fusion grid was nevertheless developed using inner-validation information across cyclic folds: those patients can be outer-test patients for another model. Together with prolonged development in one cohort and a single development seed, this prevents treating the archival results as a new fully development-independent confirmatory evaluation. Completing the reporting table does not repair that history.

Methods, comparisons, metrics, decoding, calibration/fusion and search space must be locked before reading a new round's outer results. Candidate weights or thresholds must not be selected from those results. Inner validation is for training-time selection/calibration and diagnostics only. Any historical single-inner-VAL accuracy is a training diagnostic, not the complete-study OOF accuracy or formal method-superiority evidence. Training/search and label budgets must be declared within each fold's TRAIN patients; pretrained teachers or initializers exposed to target fine labels cannot be presented as low-label controls.

## Independent reporting utility

The separate code companion adds evidence_tools/five_fold_report.py and five_fold_metrics.py. These do not change the five existing paired-inference modes, their frozen kernels, plans or confidence intervals. The new utility takes one caller-supplied UTF-8 CSV with exactly:

    roi_id,patient_id,outer_fold,y_true,y_pred

Optionally add all four columns p0,p1,p2,p3. ROI identifiers must be unique; each patient must belong to one fold; all folds 0..4 and declared complete patient/ROI counts must match; grades are canonical integers 0..3. No patient/ROI identifiers are exported in its report.

After installing the code companion's existing NumPy/scikit-learn dependencies, run from the code root:

    python evidence_tools/five_fold_report.py --predictions /path/to/frozen.csv --expected-patients 468 --expected-rois 2978 --system-name user_system --output /path/to/new-full-fold-report.json

The numbers 468/2978 are the paper cohort's explicit contract, not defaults for arbitrary callers. For another predeclared full cohort, supply its actual expected scale. If four probability columns are supplied, finite nonnegative values, mass within 1e-4 of 1 and argmax equality are checked. Probability scores additionally require --probabilities-verified-oof, an explicit caller declaration of authentic OOF origin; that flag is not an independent provenance certificate. Reports are never overwritten. Unavailable probability scores, undefined QWK, targeted error rates with a zero reference-grade denominator, and summaries lacking five finite fold values are JSON null. Per-grade undefined precision/recall/F1 ratios retain the declared zero_division=0 rule; Macro-F1 averages the true/predicted class union. No mean over a selected subset of folds is substituted.

Successful CSV checks establish only structure and declared scale. They cannot prove real patient identity, original OOF provenance, training/selection isolation, or independence from earlier development. Those require separate source/model/split records. The supplied full-fold study report is aggregate-only; the original patient-aligned prediction and probability rows stay in the private research archive.

## Data access and reproducibility

The public code/model releases do not contain MRI, ROI images, real patient-level predictions/probabilities or patient-to-partition assignment rows. The study's main frozen patient-aligned inputs and fold records are retained on the school research server, with an internal support candidate and source checksums. This is not a public download or a promise of unrestricted on-request disclosure. A specific editorial/research data-access arrangement has to be stated accurately when established.

The portable inference/statistics interfaces, historical source snapshots, archived checks and retained inputs have different scopes. Historical source inspection is not full portable retraining, and successful strict loading or synthetic tests are not clinical validation. Do not infer that all historical94-patient analyses or their dependencies are independently replayable from this release.


## Source DICOM metadata and model-facing inputs

The upstream image-reference repository warns that the source DICOM archive can retain date quasi-identifiers such as PatientBirthDate and StudyDate: https://github.com/rohanpa27/LSS_foraminal_stenosis_imaging . A limited tag-presence audit confirmed that such fields can be nonempty; it was not a whole-cohort de-identification certification, and no date values or patient keys are published here. The source DICOM archive must therefore not be described as fully anonymized merely because cropped images are used.

The inspected CNN/DeiT input paths use ROI pixels and grading targets; patient/study keys serve fold assignment and paired analysis. Birth/study dates and names are not predictor features in those inspected paths. The source archive was accessible, so this does not claim that all indirect identifiers were absent or inaccessible. This release contains no source DICOM, MRI/ROI cache, patient-level prediction/probability rows or date values. These facts do not change the author's MIT code grant, CC BY-NC4.0 model-contribution grant or retained upstream terms, and do not constitute a new institutional ethics determination.

For enquiries about access to the privately retained frozen analysis inputs, contact the corresponding author at 202628001@cquet.edu.cn. This is an enquiry route, not unrestricted public availability or a guarantee that any request will automatically receive data. It does not turn the source snapshots and public aggregate reports into a complete portable historical replay.
