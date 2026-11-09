# Single-ROI common5 example

This example completes a file-to-output path using the existing frozen APIs.
It applies one `common5_core` selected-CORAL/DeiT pair under R10 or R11. It does
not train, search weights, select slices or locate a foramen. It is not a
source94 or matched-control prediction controller.

After installing the runtime as described in README.md:

    python -I examples/predict_common5_pair.py --models /path/to/model_candidate --cnn-iq8 /path/to/roi_iq8.png --deit-png /path/to/roi_historical.png --fold 0 --strategy R11 --roi-id my_roi --output /path/to/new-report.json

Supply two representations of the same preselected anatomical ROI:

* `--cnn-iq8`: the tight DICOM-derived ROI after modality rescaling and ROI
  min-max/rint IQ8 conversion, saved as an 8-bit grayscale PNG. The runtime then
  applies NoHE/G1/64x64 and 0.5/0.5 normalization.
* `--deit-png`: the corresponding historical PNG ROI. The runtime applies HE,
  three-channel repeat, 224x224 and source-matched float32 normalization.

This small example accepts single-frame L-mode 8-bit grayscale PNG files only.
It rejects RGB, 16-bit and non-PNG files rather than converting them silently.
Other supported input preparation can use the lower-level APIs explicitly.
Format/byte checks do not establish that the files were prepared by the original
cache producers. Anatomical pairing and source provenance remain the caller's
responsibility. One cache's equivalence to the other is not established.

`--fold` is the explicitly assigned held-out fold 0..4 for common5 reproduction;
the example does not assign new patients a fold. Averaging the five common5
models on their training cohort is not OOF. `--strategy R10` uses frozen 0.4/0.6;
R11 uses the saved per-fold weight. No model or weight is selected again.

The output records the ROI identifier, fold, strategy, checkpoint/input/config
fingerprints, branch probabilities, fused probabilities and grade. An existing
report is never overwritten. A prediction JSON is a model application, not a
patient-performance estimate or validation of the caller's cohort assignment.

R6 checks use explicitly artificial grayscale PNG arrays, not MRI. Five folds
and both strategies are compared with direct calls to the previously verified
runtime APIs. See historical_validation/R6_SINGLE_ROI_EXAMPLE_TESTS.json and historical_validation/R6_historical_validation/VERIFICATION_STATUS.json.


R7 path handling: literal user-home (`~`) paths are expanded for input and
report output. Existing reports are checked at the expanded destination before
computation. Relative CSV paths remain relative to their input JSON. See the
paired code root PATH_INPUT_OUTPUT.md for examples and limits.


Public v1.0.0 scope: no weights or patient data are included. Model examples require a separately obtained matching authorised model package. Historical validation references describe earlier preparation, not new release-time execution.
