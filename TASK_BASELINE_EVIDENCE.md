# Task value, baseline rationale and performance-evidence boundaries

## Task value

The task is four-grade classification of expert-selected foraminal ROIs from routine sagittal T2 MRI. It is a research grading module. The clinically relevant question includes recognition of mild/moderate findings and severe underestimation, rather than only pooled agreement dominated by class composition. The study does not validate automatic localisation, slice selection, replacement of a full MRI examination, clinical decisions, reader-time savings or acquisition/cost benefits. A streamlined MRI workflow is an application motivation, not a measured effect of this collection.

The specific contribution is to examine which whole-system differences are retained after matching selected comparison factors, how simpler controls alter the interpretation of added complexity, and how overall metrics coexist with grade-specific trade-offs. A careful evaluation alone is not a new general ensemble algorithm or proof of unique auxiliary-loss mechanism.

## Baseline rationale and what was matched

The published Phadke imaging method is a traceable same-dataset, expert-ROI, four-grade reference with public source and patient splits. The common-five-fold comparison aligns patient/ROI labels and outer-test assignment. Fixed/selected reference fusions use the same frozen DeiT and specified fusion-weight grid/selection rule as the candidate comparisons. It is the source-candidate image-branch reconstruction, not the paper's original locked-test number, final multi-seed implementation or complete image-plus-morphometry system.

Native crop expansion, representation, backbone, pretraining, optimisation, budgets and auxiliary-coefficient search remain different. Thus the comparison is a specified internal system comparison; it is not fully equal-compute, a pure architecture effect or a pure loss effect. The natural-CORAL branch without auxiliary supervision and its same-DeiT fixed/selected fusions are retained as simpler controls. Scale-matched and probability-interface binary controls answer conditional auxiliary-training questions. The author's related earlier manuscript is disclosed historical context, not the primary baseline or an independent patient replication; its live editorial status is not certified by this code release.

## Three evidence layers

1. Complete-system comparison: the frozen common-five-fold study supports the reported system differences against the specified matched published-method reconstruction, with original unadjusted patient-paired intervals. Those differences do not establish equal-compute superiority or a contribution of every component.
2. Conditional auxiliary/fusion comparison: fixed-alpha ablations, fixed-versus-selected fusion and selected-alpha development comparisons change different factors. Selected versus fixed fusion can improve some pooled metrics while worsening balanced accuracy or recognition of non-normal grades. QWK/MAE superiority of the complete configuration over simpler natural-CORAL same-DeiT selected fusion remains unestablished; this is not an equivalence, ineffectiveness or inferiority result.
3. Descriptive complementarity: both grade-value CORAL and natural CORAL can show complementary errors with the same DeiT. Corrections and newly introduced errors explain fusion behaviour, but neither prove a mechanism unique to the auxiliary head nor positive/superadditive component interaction.

Reader-facing names and stable internal IDs are mapped in READER_NAME_MAPPING.json. Fixed-alpha 0.125 matched systems must not be renamed as the selected-alpha complete branch.

## Multiplicity and protocol scope

The original first four statistics modes retain192 matched/direct unadjusted intervals; that is not the paper-wide count. The separate ten-outcome sensitivity is only the complete grade-value selected fusion minus natural-CORAL selected fusion, for six pooled metrics and four grade recalls. It does not adjust the principal Phadke system differences or every comparison in the paper. In that family, all six pooled-metric family intervals cross zero; the local favourable mild-recall interval does not establish general ordinal superiority or clinical adequacy.

The 94 source-protocol patients are from the same source cohort as the 468-patient common-five-fold study. Distinguish original OOF stratification from later source-protocol retraining and five-model averaging on that same 94-person set. They are not new external patients, the sample sizes must not be added, and five models predicting that set are not five disjoint test cohorts.

## Full five-fold reporting

FIVE_FOLD_REPORTING.md distinguishes every outer fold, arithmetic mean with sample SD, and pooled OOF. The current aggregate companion includes all 19 historical systems, grade/error summaries and original OOF probability-quality metrics. It adds reporting granularity and descriptive probability scores, not new fitted models, selected weights, confidence intervals or independent confirmation. Long same-cohort development, shared-grid information reuse and training-seed uncertainty remain. Inner-VAL scores cannot replace complete OOF scores or support formal method superiority.

All numerical evidence is saved aggregate output, with definitions and source fingerprints. No patient/ROI keys or rows, real probabilities, MRI or weights are included in the code companion. Matching patient-aligned inputs and fold records are retained privately; public code and model weights do not alone constitute full historical reproducibility or a completed data-access arrangement.


Source DICOM date-metadata limits, the inspected model-facing pixel/target inputs and the corresponding-author data-enquiry route are stated in FIVE_FOLD_REPORTING.md. No source archive is certified fully anonymized by the release.
