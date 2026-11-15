# Input representation and geometry limits

## Narrow research interfaces

The common5 example receives two already selected and anatomically aligned ROI representations. The runtime does not decode a full DICOM examination, select a sagittal slice, locate a foramen, infer patient orientation or recover a historical PNG producer from a generic image. An accepted grayscale array or PNG is not, by itself, evidence of source equivalence or an in-distribution MRI.

For the CNN preparation helper, `iq8_from_dicom_roi(pixels, slope, intercept)` expects an already cropped two-dimensional ROI from the documented MONOCHROME2 representation. It converts the supplied values to float32, applies the supplied linear rescale, and then uses ROI min/max, rounding and clipping to produce IQ8. It does not open a DICOM dataset or check its Photometric Interpretation, Rescale Slope/Intercept, frame mapping or orientation. Its default slope/intercept are numeric defaults, not a check of the source header. The caller must establish the representation and which transformations have already occurred. Applying a transformation twice or substituting display-windowed pixels is not the declared input contract.

This helper implements the stated linear transformation, not a general DICOM Modality LUT, VOI/windowing or presentation pipeline. Nonlinear lookup tables, MONOCHROME1 handling, color/multiframe preparation and arbitrary display exports are not made equivalent by this interface. This is a limitation of the supplied reproduction path, not a claim that these conditions occurred in all or any of the study's source scans.

The manuscript's retained S1.1 reports a historical audit of all 2,978 frozen ROIs as MONOCHROME2, with the stated rescale and no VOI/windowing. The narrow helper contract does not negate that retained audit; it does not repeat or extend the audit to arbitrary newly supplied files.

## What G1 preserves

`cnn_iq8_tensor` calculates its resize scale from the supplied array height and width and uses edge-replication padding to 64x64. G1 preserves the supplied ROI array height-to-width ratio, up to integer-size rounding. It has no PixelSpacing or image-orientation input and does not establish preservation of physical millimetre dimensions, patient orientation or anatomical shape under arbitrary acquisition/export conditions. This is distinct from the archived G2 spacing-aware development configuration. No new geometric correctness or clinical morphology validation is implied.

DeiT applies the documented HE, three-channel replication, 224x224 resize and source-matched normalization to its separately supplied historical PNG ROI. The reference reconstruction has its own 40%-margin crop and normalization path. Do not substitute a common cache or default upstream image transform for these distinct contracts without separate evidence.

## Existing handling, without broader guarantees

The CNN IQ8 helper rejects empty, non-two-dimensional, nonfinite or constant rescaled ROIs. The low-level grayscale functions check the declared numeric representation; they do not reject every image that is clinically unsuitable. The single-ROI PNG example additionally checks PNG/L-mode/single-frame input. Those checks do not verify modality, acquisition protocol, field of view, anatomy, crop identity or population suitability. The release has no automatic image-quality triage, abstention policy or validated deployment protocol for new patients.

## Sources and provenance

Study-specific behavior above comes from the unchanged `src/paper2_runtime/preprocessing.py` and `examples/predict_common5_pair.py`. The official pydicom documentation distinguishes decoded pixel arrays from subsequent modality/rescale and VOI/windowing operations: [Working with Pixel Data](https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html) and [Pixel Data Processing](https://pydicom.github.io/pydicom/stable/reference/pixels.processing.html). These references explain the reporting distinction; they do not prove that all original headers, export settings or decoder versions were independently rechecked here, and their current versions are not a historical training-environment lock.
