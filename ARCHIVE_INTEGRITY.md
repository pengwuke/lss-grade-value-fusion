# Public code v1.1.1 distribution integrity

SHA256SUMS covers every regular file shipped in this distribution except SHA256SUMS itself, using lowercase SHA256, two spaces and a forward-slash relative path. PUBLIC_DISTRIBUTION_MANIFEST.json identifies exact payload paths other than itself and SHA256SUMS, avoiding self-referential digests. SHA256SUMS includes that public manifest. The external CODE_v1.1.1_DOCUMENTATION_PATCH_RECEIPT.json binds the completed ZIP and checksum-list hashes.

Verify that the ZIP contains exactly code_v1.1.1/ plus the shipped files, with no extra members. Do not use historical R3-R16 or earlier release archive hashes as hashes of this distribution. Original preparation records retain their identity under historical_validation/.

The immutable public v1.1.0 release added all 46 sanitised historical sources and their per-file map/diff, recorded by CODE_v1.1.0_SOURCE_SUPPLEMENT_RECEIPT.json. This v1.1.1 is a documentation-only patch of v1.1.0 (commit 61df3321ce3e0aa5858d8bffccc6e334071204b4); it is not merely a documentation-only description of the original v1.0.0 payload. The 78 Python/archived Python-text files, both pyprojects, frozen kernels/plans/fusion, source maps/diffs and licence texts are byte-identical to v1.1.0. Installed package versions remain 1.0.0.

Of the 46 historical copies, 21 match raw R16 and 25 have documented path/split-input sanitation. Their original/public SHA pairs remain in SOURCE_LICENSE_ATTRIBUTION.json/CSV. The historical source snapshot is inspectable code, not a newly portable full-training workflow. No controller is executed or imported for this documentation patch.

Packaging checks hashes, exact membership and documentation consistency only; no new runtime test, training, model inference, patient statistic, bootstrap, selection, performance experiment or clinical validation is performed. Actual remote version identity and anonymous access checks are recorded by the external publication receipt, not inferred from these payload checks.
