# Four-grade lumbar foraminal stenosis classification on routine sagittal T2 MRI

Research code for **Four-grade lumbar foraminal stenosis classification on routine sagittal T2 MRI: a patient-disjoint evaluation of grade-value auxiliary supervision and probability fusion** (manuscript B15).

Author: Wuke Peng. ORCID: 0009-0003-5791-3783.

## Code release v1.0.0

This distribution contains a portable inference architecture/input/fusion interface, five explicit frozen-prediction statistical modes, a descriptive branch-error helper, artificial-input examples/tests, aggregate evidence, and source/validation metadata. It is a scoped derivative of internal R16, not the complete internal code/model archive. See PUBLIC_DISTRIBUTION_MANIFEST.json for exact membership.

No model weights, MRI/DICOM/ROI images, real patient/ROI predictions, probabilities, fold tables, original pretrained files, or manuscript/reviewer files are distributed here. Model licensing and public distribution are handled separately; the trained model package is not publicly released with this code version. The inference programs require a matching authorised model package. The 46 historical training/analysis controllers are not bundled; their inventory metadata is retained and labelled in HISTORICAL_SOURCE_SCOPE.md. This temporary scope choice is not a claim that their upstream licences prohibit sharing.

The author's original code contributions are MIT-licensed. Upstream portions retain their own terms and attribution; see THIRD_PARTY_NOTICES.md and third_party_licenses/. The code licence does not grant rights to training data, third-party pretrained weights, or trained model files that are absent from this distribution.

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

Install only opencv-python-headless in this environment; other OpenCV wheels share the cv2 namespace. For statistics alone, install ./matched_statistics; it depends on NumPy and scikit-learn and does not require Torch. Distribution metadata is version 1.0.0; the unchanged runtime source retains its historical internal 0.0.0 marker. Functional Python sources, kernels, plans and fusion settings are byte-identical to R16.

## Inference interface and inputs

With an independently obtained matching model package:

```sh
python -I -m paper2_runtime inspect --models /path/to/model_candidate
python -I -m paper2_runtime verify --models /path/to/model_candidate
python -I examples/predict_common5_pair.py --models /path/to/model_candidate --cnn-iq8 /path/to/roi_iq8.png --deit-png /path/to/roi_historical.png --fold 0 --strategy R11 --roi-id user_roi --output /path/to/new-report.json
```

The image example uses two distinct representations of the same expert-selected ROI and an explicitly assigned held-out common5 fold. It does not select slices, locate a foramen, assign a new patient to a fold, or verify anatomical pairing. CNN requires DICOM-derived tight IQ8/NoHE/G1 at 64x64; DeiT requires historical PNG/HE at 224x224. One representation cannot be assumed recoverable from the other. The ResNet/CORN reconstruction has its own crop/normalisation and decoding contract. See COMMON5_SINGLE_ROI_EXAMPLE.md and COMMON5_FUSION_PRECISION.md. Input/hash checks do not establish patient identity or clinical validity.

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

## Historical validation and this packaging step

historical_validation/ retains immutable earlier preparation records. R4 checked a fresh isolated same-host Linux CPU installation, 50 strict role loads and artificial forward interfaces; it was not a new machine or a clinical evaluation. R5 retained model strict-load evidence. R7 records path and artificial-CLI checks. R9 records 30 runtime tests, 70 frozen-rule cases and ten artificial-image CLI cases. R12 records the added robustness interface and statistics checks. See historical_validation/README.md for scope.

This v1.0.0 packaging step checks membership, hashes, source preservation and distribution scope only. It performs no new training, model inference, statistics, bootstrap, selection or clinical validation. Historical records retain original IDs, dates and scope, including their then-unpublished status. Their old pending flags are not the current code-licence status. Historical source paths/digests in those records are provenance references and need not be bundled public paths.

## Integrity

Verify every shipped regular file against SHA256SUMS. The list excludes itself. The external preparation receipt binds the ZIP and the checksum list. See ARCHIVE_INTEGRITY.md. A checksum is not a clinical validation or third-party rights guarantee.
