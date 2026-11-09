# Reproduction scope of public code v1.0.0

This release provides inference architecture/input/fusion interfaces and five frozen-input statistics modes. It does not provide a complete training controller pipeline. The 46 historical_sources controllers are not bundled; their unchanged inventory metadata remains in historical_source_inventory.json and historical_source_manifest.json, explained by HISTORICAL_SOURCE_SCOPE.md.

Full historical reproduction additionally needs authorised source MRI/labels, the correct patient/ROI split and identity mapping, separately prepared CNN IQ8 and historical DeiT PNG representations, parent result archives, relevant initial pretrained assets and the historical training environment. Changing paths does not provide them. The Linux Python 3.11 CPU constraints record an earlier inference installation, not a historical training lock. Historical augmentation/dependency locking and initial pretrained-file identity gaps remain disclosed.

The code release packages no trained weights or patient data. Matching models are handled separately. The code's MIT author-contribution grant does not license raw MRI, original pretrained assets, or the author's trained-model contributions. No training, inference, patient resampling, bootstrap, selection or performance recomputation was performed in this packaging step.
