# Four-grade lumbar foraminal stenosis classification on routine sagittal T2 MRI

Research code for **Four-grade lumbar foraminal stenosis classification on routine sagittal T2 MRI: a patient-disjoint evaluation of grade-value auxiliary supervision and probability fusion**.

Author: Wuke Peng. ORCID: 0009-0003-5791-3783.

## Code distribution v1.1.5: input and citation documentation

This documentation-only distribution starts from immutable public v1.1.4, commit bf252ccca397ed7222db9b66f712d18ba03d0c03. It clarifies the supplied pixel-array/DICOM preparation contract, what G1 preserves, the limits of input validation and how to cite code versus weights and documentation. All 81 parent Python/archived Python-text files, both pyprojects, frozen kernels/plans/fusion arithmetic, numerical evidence, historical records, source maps/diffs and licence texts remain byte-identical to v1.1.4. No program, model, numerical report or functional test is rerun in this revision.

Historical lineage and older tags are retained. The v1.1.0 source extension added 46 inspectable historical files to v1.0.1; v1.1.1/v1.1.2 clarified scope and verification paths. v1.1.3 added the complete-fold reporter and original descriptive summaries; v1.1.4 clarified use/reporting. Those original execution records keep their own dates and scopes.

The release contains portable inference architecture/input/fusion interfaces, five frozen-input statistics modes, synthetic examples/tests, aggregate evidence and historical training-source snapshots. See HISTORICAL_SOURCE_SCOPE.md, SOURCE_LICENSE_ATTRIBUTION.json/CSV and PUBLIC_SOURCE_DIFF.patch. This is inspectable source, not a complete portable historical training pipeline or locked training environment. No legacy controller was executed or imported for this documentation revision.

No model weights, MRI/DICOM/ROI images, real patient/ROI prediction/probability rows, patient-to-split/fold assignment records, original pretrained files, manuscript/reviewer files or credentials are bundled. Matching authorized models are distributed separately. Author original code contributions are MIT-licensed; adapted/reference parts retain their terms in THIRD_PARTY_NOTICES.md and third_party_licenses/. The ORNet method is cited for the author's distinct CORAL-specific implementation; upstream ORNet source is not assigned MIT or copied into this distribution.

The versioned release is [v1.1.5](https://github.com/pengwuke/lss-grade-value-fusion/releases/tag/v1.1.5). `CITATION.cff` and CITATION.md distinguish the code distribution version, unchanged installed package version, model-weight version and actual documentation revision. The external release, commit and asset digest establish actual publication; payload metadata alone does not.

## Installation

Use a separate Python environment. The retained tested Linux Python 3.11 CPU dependency record is an inference-installation record, not a historical training lock or a new-machine guarantee.

```sh
python3.11 -m venv paper2-cpu
# Activate the environment using the command for your operating system.
python -m pip install -r requirements-cpu-torch-tested.txt
python -m pip install -c constraints-cpu-inference-linux-py311.txt .
python -m pip install -c constraints-cpu-inference-linux-py311.txt ./matched_statistics
python -m pip check
```

Install only opencv-python-headless in this environment; other OpenCV wheels share the cv2 namespace. For statistics alone, install ./matched_statistics; it depends on NumPy and scikit-learn and does not require Torch. Installed package metadata remains version 1.0.0; this code distribution is v1.1.5 and the unchanged runtime source retains its historical internal 0.0.0 marker. The 32 portable Python files, frozen statistical kernels/plans and fusion settings are byte-identical to R16. Among the 46 historical copies, 21 match raw R16 and 25 contain the documented path/split-input sanitation; all 78 Python/archived Python-text files are byte-identical to public v1.1.0.

## Inference interface and inputs

With an independently obtained matching model package:

```sh
python -I -m paper2_runtime inspect --models /path/to/model_candidate
python -I -m paper2_runtime verify --models /path/to/model_candidate
python -I examples/predict_common5_pair.py --models /path/to/model_candidate --cnn-iq8 /path/to/roi_iq8.png --deit-png /path/to/roi_historical.png --fold 0 --strategy R11 --roi-id user_roi --output /path/to/new-report.json
```

The image example uses two distinct representations of the same expert-selected ROI and an explicitly assigned held-out common5 fold. It does not select slices, locate a foramen, assign a new patient to a fold, or verify anatomical pairing. CNN requires DICOM-derived tight IQ8/NoHE/G1 at 64x64; DeiT requires historical PNG/HE at 224x224. One representation cannot be assumed recoverable from the other. The ResNet/CORN reconstruction has its own crop/normalisation and decoding contract. See COMMON5_SINGLE_ROI_EXAMPLE.md and COMMON5_FUSION_PRECISION.md. Input/hash checks do not establish patient identity or clinical validity.

See INPUT_REPRESENTATION_LIMITS.md for the narrow pixel-array/modality-rescale contract and why G1 does not establish physical spacing or orientation preservation. Accepted numeric or PNG format is not general DICOM correctness or clinical input-quality validation.

See INPUT_REPRESENTATION_LIMITS.md for the narrow pixel-array/modality-rescale contract and why G1 does not establish physical spacing or orientation preservation. Accepted numeric or PNG format is not general DICOM correctness or clinical input-quality validation.

See USE_AND_REPORTING_FAQ.md for the distinction between unlabeled ROI application, label-bearing evaluation inputs and the scope of score/provenance checks.

## Frozen-input statistics

| Mode | Scope | Default patient-paired repeats |
| --- | --- | ---: |
| single | Four matched single-branch contrasts | 20,000 |
| fixed | Fifteen matched fixed-fusion contrasts | 20,000 |
| selected | Twelve matched selected-fusion contrasts | 20,000 |
| r11_natural | Full R11 minus natural-CORAL same-DeiT selected fusion | 20,000 |
| r11_robustness | Same two systems; ten-outcome intervals and descriptive sensitivity | 100,000 |

Seed is 42. The first four modes report unadjusted intervals. The last also reports Bonferroni-type 99.5% marginal intervals for its ten-outcome family; this is not paper-wide multiplicity correction. Explicit CSV inputs must be supplied by the caller; no frozen study rows are supplied. Input alignment checks do not certify that supplied data are the paper cohort. See matched_statistics/README.md. The branch_error_audit helper reports descriptive repairs/losses and is separate from these five modes.

## Evidence and interpretation

The main published-method comparison is the Phadke image method's same-protocol reconstruction, with the same frozen DeiT and selection opportunity. It is not the original published score or complete image-plus-morphometry system. The author's earlier unpublished system is historical supplementary context; shared assets remain disclosed. The strong natural-CORAL reference and attribution/generalisation limits remain visible. See TASK_BASELINE_EVIDENCE.md and evidence/.

The task is grading expert-selected ROIs. Aggregate gains do not establish independent causal effects of every component, clinical utility, reader-time savings or external generalisation. Only aggregate metrics/intervals/counts and plans are provided. The original MRI dataset is available from https://data.mendeley.com/datasets/rgb77xm3jf/4 (DOI 10.17632/rgb77xm3jf.4); this code package does not redistribute it.


## Complete five-fold reporting and reader names

See FIVE_FOLD_REPORTING.md for the separate reporter's CSV/scale contract and the distinction between every outer fold, mean with sample SD and pooled OOF. READER_NAME_MAPPING.json maps grade-value CORAL, fixed fusion and validation-selected fusion to stable archive IDs. Code keys and model roles remain unchanged. The original five statistics modes are frozen pooled paired analyses; they are not relabelled as the new full-fold reporting utility.

The complete aggregate report adds grade/error and probability-quality summaries from recovered original branch OOF probabilities and algebraic fusion under already-frozen weights for all 19 retained systems; they are not 19 independently stored native probability files or new model forward passes. It is not a new independent evaluation, and probability scores do not establish clinical calibration. Model-level patient separation does not erase the historical cross-fold common-grid information reuse. Source MRI metadata and patient-level rows are not redistributed by this package. The source archive can retain date quasi-identifiers; FIVE_FOLD_REPORTING.md states the limited metadata audit, inspected pixel/target inputs and the author data-enquiry route, without certifying full anonymization.

## Historical validation and this packaging step

historical_validation/ retains immutable earlier preparation records. R4 checked a fresh isolated same-host Linux CPU installation, 50 strict role loads and artificial forward interfaces; it was not a new machine or a clinical evaluation. R5 retained model strict-load evidence. R7 records path and artificial-CLI checks. R9 records 30 runtime tests, 70 frozen-rule cases and ten artificial-image CLI cases. R12 records the added robustness interface and statistics checks. See historical_validation/README.md for scope.

This revision changes documentation only. The descriptive full-fold/probability reporting audit, artificial-input reporter tests and school-server checks of 19 frozen inputs were performed for v1.1.3 and are inherited here, not rerun or relabelled as v1.1.5 execution. No new training, model inference, numerical aggregation, bootstrap interval, significance test, weight selection or clinical validation is performed for this revision. The archived six-metric pooled results remain unchanged. Historical records retain original IDs, dates and scope, including their then-unpublished status. Their old pending flags are not the current code-licence status. Historical source paths/digests in those records are provenance references and need not be bundled public paths.


## AI assistance

ChatGPT and Codex (OpenAI) assisted script implementation/debugging, frozen-result aggregation and consistency checking, interpretation, drafting/revision, translation/typesetting and code/model documentation, including the complete-fold reporter. This was not language editing alone. Outputs derive from executed programs and frozen artefacts; generative AI did not generate or alter MRI images, reference labels or experimental observations. The author retains responsibility. USE_AND_REPORTING_FAQ.md gives the assistance and verification scope.

## Integrity

Verify every shipped regular file against SHA256SUMS. The list excludes itself. See ARCHIVE_INTEGRITY.md. A checksum is not a clinical validation or third-party rights guarantee.
