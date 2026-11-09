# Historical source metadata; controllers not bundled

historical_source_inventory.json and historical_source_manifest.json are unchanged metadata from the full internal R16 candidate. They identify 46 retained historical files (42 Python files and four archived Python-text producers), their sizes and hashes. None of the historical_sources/ controllers is included in this public code v1.0.0 distribution. Referenced relative paths are provenance identifiers, not download promises.

Static inspection found absolute path literals in 22 Python files. It is not a portability or execution test. The omission is a conservative temporary distribution-scope choice while completing a per-file provenance/licence map, not an established prohibition on redistributing these files. The author source extracts used by the inference API and the explicitly bundled frozen statistical kernels remain included with their own provenance records.

The complete internal code/model R16 archive remains preserved. This distribution does not claim full historical retraining or access to raw data, frozen splits, ROI caches, pretrained initial files or parent run archives.
