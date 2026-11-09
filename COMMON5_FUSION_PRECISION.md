# Common5 fusion arithmetic precision

The frozen common5 producers read the four branch probability columns into
NumPy float64 arrays before fixed/selected weighted addition and argmax.
The R8 portable common5 wrappers preserved the caller's array dtype. Direct
float32 model outputs therefore took a different arithmetic path from those
producers. Close scores could round to a tie, changing the selected grade.

R9 explicitly promotes both operands to float64 in common5_fusion and
matched_common5_fusion, then applies the existing frozen coefficients and argmax.
It does not renormalize, clip or round the original values, change any weight,
or search again. The single-ROI JSON records fusion_probability_dtype: float64.
Branch model outputs and deterministic image conversion remain unchanged.

For example, under the fixed 0.4/0.6 rule, artificial float32 vectors
[0.5080505609512329, 0.4919494390487671, 0, 0] and
[0.49463292956352234, 0.5053670406341553, 0, 0]
produce [0.5, 0.5, 0, 0] in the old float32 path (grade 0). Promoting the same
stored values before arithmetic gives [0.4999999821186065, 0.5, 0, 0] (grade 1),
matching the frozen producer rule. Deliberately concentrated near-tie examples
demonstrate this issue; their frequency says nothing about real patient data.
Upcasting cannot recover precision already lost before the supplied values.

The dtype-preserving generic fuse_pair helper and the separate source94_fusion
and source94_reference_average paths retain their R8 arithmetic. Use the named
protocol wrapper for common5 reproduction. The SR1 producer averages its five
live model probability arrays before argmax; its averaging path is not changed
by this common5 correction.

R9 tests use artificial probability arrays/PNG inputs only. The main and matched
common5 weights and all tensor files stay frozen. The manuscript's archived
predictions, metrics and intervals have not been recomputed. The affected
portable frontend is corrected; this is not evidence of altered paper results
or new patient validation. R7's earlier exact-output comparisons describe R7.
Full historical training is outside this release. Author code contributions are MIT-licensed; trained models are handled separately.


Public v1.0.0 scope: no weights or patient data are included. Model examples require a separately obtained matching authorised model package. Historical validation references describe earlier preparation, not new release-time execution.
