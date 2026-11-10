# Public code v1.1.2 distribution integrity

SHA256SUMS covers every regular file shipped in this distribution except SHA256SUMS itself, using lowercase SHA256, two spaces and a forward-slash relative path. PUBLIC_DISTRIBUTION_MANIFEST.json identifies exact payload paths other than itself and SHA256SUMS, avoiding self-referential digests. SHA256SUMS includes that public manifest. The external CODE_v1.1.2_DOCUMENTATION_PATH_PATCH_RECEIPT.json binds the completed ZIP and checksum-list hashes.

Verify that the ZIP contains exactly code_v1.1.2/ plus the shipped files, with no extra members. Do not use historical R3-R16 or earlier release archive hashes as hashes of this distribution. Original preparation records retain their identity under historical_validation/.

The immutable public v1.1.0 release added all 46 sanitised historical sources and their per-file map/diff, recorded by CODE_v1.1.0_SOURCE_SUPPLEMENT_RECEIPT.json. Public v1.1.1 (commit fa086303265ccc4d22bdf61ea2ee5eed689a81e3) corrected scope/version/integrity wording. This v1.1.2 starts from that exact release and corrects the nonexistent R6 verification-record path in COMMON5_SINGLE_ROI_EXAMPLE.md to historical_validation/R6_VERIFICATION_STATUS.json, with the necessary current README/integrity and distribution-metadata updates.

All 78 Python/archived Python-text files, both pyprojects, frozen kernels/plans/fusion, historical verification records, aggregate evidence, source maps/diffs and licence texts are byte-identical to v1.1.1. Installed package versions remain 1.0.0. The retained documentation_patch record for distribution 1.0.1 is historical provenance; the current documentation_path_patch and parent fields identify this correction.

Of the 46 historical copies, 21 match raw R16 and 25 have documented path/split-input sanitation. Their original/public SHA pairs remain in SOURCE_LICENSE_ATTRIBUTION.json/CSV. The historical source snapshot is inspectable code, not a newly portable full-training workflow. No controller is executed or imported for this documentation patch.

Packaging checks exact membership, hashes, byte preservation and the corrected documentation reference only. It performs no new runtime test, training, model inference, patient statistic, bootstrap, selection, performance experiment, dependency installation or clinical validation. Actual remote version identity and anonymous access checks belong to the external publication receipt; preparation alone does not prove publication.
