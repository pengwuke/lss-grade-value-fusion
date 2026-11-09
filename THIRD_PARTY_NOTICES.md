# Attribution and licence boundaries

The root MIT licence and matched_statistics/LICENSE cover Wuke Peng's original contributions and modifications only. They do not replace upstream grants or grant rights to absent models or datasets. Full retained upstream texts are in third_party_licenses/; sources.json records their official URLs and SHA256 values. All four snapshots matched the official raw text on 2026-11-09.

| Distributed path/contribution | Source and treatment |
| --- | --- |
| src/paper2_runtime/cnn.py and related ordinal research definitions | Extracted from the author's frozen CNN source, with exact source SHA in cnn_source_provenance.json. CORAL/CORN reference: coral-pytorch, Sebastian Raschka 2020-2022, MIT. Preserve coral-pytorch_MIT.txt for applicable adapted portions. Algorithm references do not imply ownership of all upstream implementations. |
| src/paper2_runtime/reference.py; associated reference construction/decoding in models.py and preprocessing.py | Author's frozen P4.1-C/SR1 same-protocol reconstruction informed by Rohan A. Phadke's MIT source (2026). reference_source_provenance.json records the frozen author producers. Net/corn_probs were extracted; default pretrained True was changed to False to avoid downloads before strict loading; the source score floor is retained. Preserve Phadke_MIT.txt. |
| timm API calls in src/paper2_runtime/models.py and reference.py | timm is Apache-2.0; preserve timm_Apache-2.0.txt. The dependency itself is installed separately, not bundled. Construction uses timm rather than a substituted torchvision ResNet. |
| torchvision dependency | BSD-3-Clause, Soumith Chintala 2016; preserve torchvision_BSD-3-Clause.txt. No dependency wheel is redistributed here. |
| matched_statistics/src/paper2_statistics/kernels/ | Author frozen analysis modules copied byte-for-byte; KERNEL_PROVENANCE.json identifies source paths and hashes. Source paths describe private historical provenance, not additional bundled controllers. |
| Other wrappers, examples, tests, aggregate reporting and documentation | Author contributions under the root MIT licence. Dependencies retain their own licences and are installed separately. |

Official sources (checked 2026-11-09):

- CORAL/CORN: https://github.com/Raschka-research-group/coral-pytorch/blob/main/LICENSE.txt
- Phadke: https://github.com/rohanpa27/LSS_foraminal_stenosis_imaging/blob/main/LICENSE
- timm: https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE
- torchvision: https://github.com/pytorch/vision/blob/main/LICENSE
- The separately handled pretrained DeiT identifier is timm/deit_tiny_patch16_224.fb_in1k; its official card marks Apache-2.0: https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k . No original or fine-tuned weight is bundled here.

Dataset v4: https://data.mendeley.com/datasets/rgb77xm3jf/4, DOI 10.17632/rgb77xm3jf.4. Its licence field states CC BY 4.0; the description also says scientific purpose (noncommercial), both still present on 2026-11-09. No MRI/ROI image or real patient-level table is redistributed. No unrestricted commercial model permission is inferred. This factual difference is not presented as a categorical prohibition on sharing noncommercial research models. Trained models require a separate distribution and author-contribution grant; a code MIT choice does not settle third-party data rights.

The public code release excludes the 46 legacy historical_sources files pending a more detailed file/source map. Their metadata is preserved. This is a temporary scope decision, not an identified prohibition in the permissive upstream licences. Original internal R16 is preserved.
