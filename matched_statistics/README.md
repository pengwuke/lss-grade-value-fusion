# Paper2 matched statistics: public code v1.0.0 / manuscript B15

Research-code companion to manuscript B15. The author contributions are MIT-licensed; no study rows or model files are included. The five modes
have distinct scopes and defaults (all use seed 42):

| Mode | Frozen comparisons / outcomes | Default paired patient resamples |
| --- | --- | ---: |
| single | 4 matched single-branch contrasts, 24 six-metric intervals | 20000 |
| fixed | 15 matched fixed-fusion contrasts, 90 intervals | 20000 |
| selected | 12 matched selected-fusion contrasts, 72 intervals | 20000 |
| r11_natural | Full R11 minus natural same-DeiT selected fusion, 6 post-hoc intervals | 20000 |
| r11_robustness | The same two systems, 10 outcomes with 95% and Bonferroni-type 99.5% marginal intervals plus descriptive sensitivity | 100000 |

The first four modes account for 192 matched/direct intervals. This excludes
the original historical-system, source94/patient-stratum intervals and the
separate ten-outcome sensitivity. It is not the paper-wide interval count.
This package does not reproduce those historical/protocol analyses, train
models, select weights or download images. Actual frozen study rows are not
redistributed. See ../TASK_BASELINE_EVIDENCE.md and ../evidence/EVIDENCE_MAP.json
for comparison identities. The ../evidence_tools/branch_error_audit.py helper
is separate from these five modes and computes descriptive counts only.

The reviewed A3 and A4 calculation modules are copied byte-for-byte under
`src/paper2_statistics/kernels`. Their SHA256 fingerprints are checked at runtime.
`plans.json` preserves the frozen system order and linear contrast definitions;
its fingerprint is checked as well. The new code provides only explicit CSV I/O,
identity validation, command-line reporting and tests around those kernels.

## Install and run

Use Python 3.10 or later in a separate environment:

    python -m venv stats-env
    # Activate stats-env using the command appropriate for your operating system.
    python -m pip install .
    python examples/make_synthetic_inputs.py /path/to/artificial-example
    python -m paper2_statistics --analysis single --inputs /path/to/artificial-example/single.json --output /path/to/new-smoke-report.json --iterations 37

The example contains explicitly artificial identifiers and grades. It is a
software smoke check, not a paper-performance estimate. For a mode's frozen resampling configuration omit `--iterations`: the first four modes
default to 20,000, while r11_robustness defaults to 100,000; seed is fixed at 42.
Other choices of iteration count are labelled as smoke configurations. Using a mode's default repeat count alone does not establish that the supplied cohort or predictions are the
paper's frozen assets. The report always states that cohort identity has not
been verified. Existing reports are not overwritten.

The `fixed` and `selected` analysis modes use `fixed.json` and `selected.json`
from the example, or equivalent manifests pointing to explicit user CSV files.
There is no original server path as a default. NumPy and scikit-learn are the
only runtime dependencies; Torch, timm and image libraries are not required for
this package. Model inference is in the separate main runtime.

## CSV contract

The input JSON contains exactly one `predictions` object. Its keys must equal
the system names for the chosen analysis in `plans.json`; values are absolute
CSV paths or paths relative to that JSON file. Duplicate JSON keys are rejected.
Each CSV must contain exactly these five uniquely named columns:

    roi_id,patient_id,outer_fold,y_true,y_pred

IDs must be nonempty strings without leading/trailing spaces. Each ROI occurs
once per system. All systems must have the same ROI set and identical patient,
outer fold and true grade for every ROI. Rows are joined by ROI identity; any
reordering is recorded. A patient must belong to exactly one outer fold. Five
folds, encoded either 0..4 or 1..5, must be present. Grades must be canonical
integer strings 0..3. Decimal, missing or out-of-range grades are rejected.
Input SHA256 values describe the same bytes that were parsed.

Supply already frozen integer predictions. This package neither infers how a
model generated them nor checks whether a user-selected fusion weight was
chosen without test leakage. It does not repair incompatible inputs silently.

## Preserved statistical rules

Scoring counts each ROI once; there is no patient-average score before pooled
metrics. Bootstrap samples patients with replacement and retains all their ROIs,
including multiplicities when a patient is sampled repeatedly. All systems and
contrasts within an analysis share each sampled patient list. The single-branch
kernel uses chunk size 64; fixed/selected kernels use 32, preserving the frozen
NumPy draw procedure, sorted patient order and contrast direction.

In a resample, balanced accuracy averages recall over true-present classes;
Macro-F1 uses the sorted union of true/predicted classes. Quadratic kappa uses
positions in that active union, so an absent intermediate grade is not forcibly
inserted. MAE always uses original 0..3 distances. Undefined kappa is kept as
NaN internally and JSON null when reported; it is not replaced by zero.
Percentile limits use finite raw differences only and record valid/undefined
counts. All differences remain candidate minus reference, including MAE, whose
negative direction is favourable. Linear contrasts retain difference-of-
differences terms. The single, fixed, selected and r11_natural modes report unadjusted 95%
percentile intervals. The r11_robustness mode additionally reports Bonferroni-
type 99.5% marginal percentile intervals for its ten-outcome family, targeting
approximately 95% family coverage for full R11 minus natural selected fusion
only. This does not correct all 192 matched/direct intervals or the whole paper.
There is no model refitting or weight reselection inside resamples. The original
unadjusted analyses are described in B15 S2.6/S2.8/S2.10; the separate family
sensitivity is described in S2.11. These modes perform statistical inference on
fixed predictions; they do not run model inference.

## Historical R3 verification and current limits

The original R3 single/fixed/selected interface was installed as a wheel in
a new Windows Python environment and
tested in Python isolated mode from a separate working directory. Tests use
only artificial arrays/CSVs. They cover mismatches, patient leakage, duplicate
IDs/JSON keys, row alignment, missing-class/undefined-QWK behaviour, direct
patient-multiplicity resampling against scikit-learn, and all three frozen
contrast plans. The CLI was separately smoke-tested for each plan. See
`STATISTICS_VERIFICATION.json` and `tested_statistics_environment.json`.

That was a fresh dependency installation test for the original R3 statistics
interface only; it is retained historical evidence, not a new R15 installation.
The B11 direct and B12 robustness additions retain their later checks (15
statistics tests, including the two synthetic CLI modes, as recorded in R12).
R15 only checks reporting and documentation; it does not rerun those tests,
real-patient intervals or model inference.
The original R3 test is not a model-inference installation test, historical
dependency lock or rerun of the paper's patient intervals. Author manuscript review,
new-asset licensing and the dataset licence/use-description clarification remain
pending. No public release or submission has occurred.


R7 path handling: literal user-home (`~`) paths are expanded for input and
report output. Existing reports are checked at the expanded destination before
computation. Relative CSV paths remain relative to their input JSON. See the
paired code root PATH_INPUT_OUTPUT.md for examples and limits.


## Retained B11 post-hoc direct comparison

--analysis r11_natural compares R09_SELECTED_ALPHA_VSEL minus A0_NATURAL_VSEL.
Supply exactly those two prediction mappings. It uses the same six metrics,
seed 42, 20,000 patient-clustered paired percentile resamples and fixed input
predictions. The original single/fixed/selected plans are unchanged. This is a
post-hoc internal reference comparison, specified after known point estimates;
all six unadjusted intervals are reported. See the code-root evidence directory
and TASK_BASELINE_EVIDENCE.md. No patient rows are included in the distribution.


## Retained B12 robustness mode

--analysis r11_robustness accepts the same two-system mapping as r11_natural.
The default is 100,000 paired patient resamples, seed 42, ten outcomes (six
pooled metrics and four grade recalls), 95% percentile and Bonferroni-type
99.5% marginal intervals, plus descriptive patient/fold deletion and equal-
patient-weighted pooled points. Earlier four modes and defaults are preserved.
All inference remains post-hoc and conditional on fixed predictions. This
family correction does not adjust the entire paper or establish clinical use.


B15/R16 changes presentation only: the published-method reconstruction is the
main external-method comparison; the unpublished early-system comparison is
historical supplement S2.13. All five statistical modes, frozen kernels/plans,
intervals and their correction scope are unchanged; no tests or patient
resampling were rerun in R16.


## Public distribution scope

This package is part of the scoped code v1.0.0 release derived from internal R16. Frozen source kernels/plans and all functional Python bytes are unchanged. Historical validation keeps its original round and scope; it is not a new validation of this distribution. Original study rows, patient/ROI probabilities and full historical training controllers are not bundled. See ../PUBLIC_DISTRIBUTION_MANIFEST.json and ../HISTORICAL_SOURCE_SCOPE.md.
