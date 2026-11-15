# Use and reporting questions

## Does model application require the true grade?

No. The single-ROI common5 example accepts two caller-prepared representations of the same expert-selected anatomical ROI, a matching model collection, an explicit common5 fold and a frozen fusion strategy. It outputs class scores and the predicted grade; it does not accept the true grade. The complete-five-fold reporting utility instead evaluates already saved predictions and therefore requires y_true, y_pred, patient/fold metadata and, optionally, all four original OOF probability columns. Evaluation inputs are not additional model features. Neither command assigns a new patient to a validated deployment protocol.

## May all five common5 models be averaged for any patient?

Not to reproduce the study's OOF evaluation. Each study patient uses the pair corresponding to that patient's assigned outer-test fold; the other common5 models can have trained on that patient. The separate source94 protocol averages its five models on the same source-protocol test set. These two protocols must not be interchanged. SR1's saved development-fold predictions were obtained from checkpoints selected with those validation folds and then reused to fit a decoding rule. They are not an independent outer-test estimate. Retained source names containing oof do not certify their statistical role.

## Do lower probability scores demonstrate calibration?

No single reported score establishes this. Log loss and four-class sum Brier assess probability quality beyond calibration alone. Top-label ECE10 compares maximum confidence with correctness of the predicted grade in ten equal-width bins; it does not establish calibration of each grade probability and depends on the bins. The study's scores are retrospective point descriptions without new probability-score intervals or recalibration. Training inverse-frequency sampling or weighted losses changes training emphasis, while these reports score the observed outer-fold grade distribution. Renormalizing scores for scoring or applying the frozen fusion is not population-prior correction, clinical risk calibration or a new validation.

## What can validation of a file or release establish?

Checksums and strict state loading identify bytes and interface compatibility. CSV contracts establish the required columns, complete declared scale, one ROI per row and patient/fold consistency. They do not independently establish authentic OOF origin, anatomical pairing, selection independence or clinical validity. Raw patient-aligned inputs remain private; the full aggregate and synthetic checks cannot replace those inputs for paired reanalysis. Historical training sources are inspectable snapshots, not a complete portable retraining environment. Public posthoc/robustness/reporting plans record retrospective locks after prior same-cohort results were available; they are not prospective preregistration or independent confirmation. Age/sex/acquisition subgroup performance, external generalization, clinical utility and demographic fairness remain unestablished unless separately evaluated; no automated slice selection or localization is provided.

## AI assistance

ChatGPT and Codex (OpenAI) assisted research-design discussion, script implementation/debugging, aggregation and consistency checking of frozen predictions, interpretation, drafting/revision, translation/typesetting and code/model documentation. This included the complete-five-fold reporter and retrospective descriptive summaries; the role was not language editing alone. Reported outputs come from executed research programs and frozen artefacts, not from AI-generated prose treated as empirical evidence or a primary reference. Generative AI was not used to generate or alter MRI images, reference labels or experimental observations. The author retains responsibility for the reported analyses, interpretations and conclusions. Verification records state what was actually checked; this disclosure does not claim line-by-line human validation of every historical source or new independent replication.

## Sources used to review this documentation

- [REFORMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11092361/) informed version, environment, executable-scope and leakage checks. It is guidance, not a certification of this release.
- [Model Cards for Model Reporting](https://arxiv.org/html/1810.03993v2) informed explicit uses, evaluation conditions and limitations.
- [scikit-learn probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html) informed the distinction between probability scoring and calibration.
- [ICMJE guidance on AI use by authors](https://www.icmje.org/recommendations/browse/artificial-intelligence/ai-use-by-authors.html) informed the disclosure of material assistance and author responsibility.

These sources guide reporting; they do not supply new results for this model collection. The methods, historical limitations and actual implementation govern the study-specific statements.
