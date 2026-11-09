# Task and evidence boundaries: manuscript B15

The task is four-grade classification of expert-selected sagittal T2 foraminal ROIs. This code release is linked to manuscript B15; it does not validate slice selection, ROI localisation, abbreviated MRI acquisition, reader-time savings or clinical decisions.

The main system comparison uses the published Phadke image method's same-protocol reconstruction with the same frozen DeiT and selection opportunity. It is neither the original publication's score nor its complete image-plus-morphometry system. Full R11 minus that matched selected reference has existing unadjusted paired differences: accuracy +0.03828 [0.02387, 0.05297], QWK +0.02430 [0.00898, 0.04001], MAE -0.03559 [-0.05168, -0.01941]. These are internal protocol system differences, not independent component effects.

The author's earlier CORAL-MSaux/fixed-DeiT system is unpublished and under review and remains historical supplementary context. Shared cohort, splits, frozen DeiT assets and representative ROIs are disclosed in the manuscript. The stronger natural-CORAL same-DeiT selected reference remains fully visible: QWK and MAE superiority to it is not established. In the ten-outcome robustness family all six pooled-metric family intervals cross zero; only mild recall retains a favourable family interval, +5.52 percentage points [0.76, 10.49]. Absolute mild recall remains 40.83%. This does not establish clinical utility or broad ordinal superiority.

Comparisons have distinct meanings:

- R11 minus R10 changes frozen fusion weights only.
- R11 minus R09 adds DeiT and applies the selected-fusion configuration.
- R09 versus fixed alpha 0.125 is a development point comparison without a dedicated paired interval.
- A3_GRADE_FIXED/F040/VSEL comparisons concern auxiliary training schemes, not R09/R10/R11 and not an isolated loss term.
- R11 versus natural-CORAL selected fusion changes auxiliary training, alpha selection and separately selected fusion weights. Both use G1 and the same DeiT/search opportunity.

The descriptive branch-error audit reports R11 repairs/losses of 93/82 against R09, net +11 correct ROIs. Natural-CORAL selected fusion also has repairs/losses of 150/130, net +20. These counts support prediction complementarity, not unique auxiliary-loss causation or positive/superadditive interaction.

The first four statistics modes have 192 matched/direct unadjusted intervals, not the paper-wide interval count. r11_robustness adds a separate ten-outcome family sensitivity. B14_REPORTING_AUDIT.json checks 221 selected displayed intervals; it is a historical reporting audit, not new B15 inference or independent replication. B15_PRESENTATION_MAP.json describes presentation changes. See evidence/EVIDENCE_MAP.json for frozen comparison definitions and source fingerprints.

All bundled evidence files contain aggregate metrics, intervals, counts, plans or source metadata. No patient/ROI rows or real probability tables are bundled. These are saved outputs, not computations performed during release packaging. Single-cohort development, information reuse, one development seed and absent new-patient external validation limit interpretation. No full historical retraining or clinical validation is claimed.
