# Public code v1.1.0 distribution integrity

SHA256SUMS covers every regular file shipped in this distribution except SHA256SUMS itself, using lowercase SHA256, two spaces and a forward-slash relative path. PUBLIC_DISTRIBUTION_MANIFEST.json identifies exact payload paths other than itself and SHA256SUMS, avoiding self-referential digests. SHA256SUMS includes that public manifest. The external CODE_DOCUMENTATION_PATCH_v1.1.0_RECEIPT.json binds the completed ZIP and checksum-list hashes.

Verify that the ZIP contains exactly code_v1.1.0/ plus the shipped files, with no extra members. Do not use the historical R3-R16 archive hashes as hashes of this public subset. Historical records are retained under historical_validation/ with their original identity.

Packaging verifies hashes and membership, not model semantics, clinical validity, external data rights or the identity of caller-provided patients. No model or patient calculations are performed by this packaging step.

This is a documentation-only distribution patch of the immutable public v1.0.0 release. Package versions remain 1.0.0. New hash and membership checks verify this distribution; they are not new runtime tests or performance validation.

The v1.1.0 payload also covers all 46 sanitised historical source copies and their per-file map/diff. Their raw R16 and public snapshot hashes are intentionally different where path/ID sanitation is documented. Static parsing does not execute a controller or establish training portability.
