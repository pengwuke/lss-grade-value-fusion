# User-home paths in file input and report output

R7 fixes inconsistent treatment of literal `~` paths in the single-ROI example
and matched-statistics wrapper. Model metadata readers already expanded a
user-home path; image reads, example provenance reporting, the statistics input
manifest and report output did not consistently do so. A quoted path could fail
to find an existing file or create a literal `~` directory under the working
directory. All affected paths now expand the user home before reading, checking
for an existing report or writing. The example normalizes the model root once
and uses it for loading and reported fingerprints.

Use quotes around paths containing spaces. In the examples below the shell is
allowed to pass a literal `~`; the application expands it. `~` means the home of
the account running Python, not a home on another computer. It does not copy
files between computers. Ordinary relative paths still use the working directory;
relative CSV paths in an input JSON still use that JSON's parent directory.
Environment-variable syntax such as `$HOME`, `%USERPROFILE%` or `${name}` is not
expanded by the application. Supply an explicit path when no home is configured.

    python -I examples/predict_common5_pair.py --models "~/paper2/model_candidate" --cnn-iq8 "~/paper2/roi_iq8.png" --deit-png "~/paper2/roi_historical.png" --fold 0 --strategy R11 --roi-id my_roi --output "~/paper2/reports/new-roi.json"

    python -I -m paper2_statistics --analysis single --inputs "~/paper2/inputs/single.json" --output "~/paper2/reports/new-stats.json" --iterations 37

Existing reports at the expanded destination are rejected before inference or
resampling begins; exclusive file creation remains in place. All original model,
input, pairing and fold responsibilities still apply. The statistics command
with 37 iterations is an artificial-input software check, not a paper result.
The fixes do not change image conversion, model tensors, fusion, frozen kernels,
contrast plans or the manuscript's numerical results.


Public v1.1.1 scope: no weights or patient data are included. Model examples require a separately obtained matching authorised model package. Historical validation references describe earlier preparation, not new release-time execution.
