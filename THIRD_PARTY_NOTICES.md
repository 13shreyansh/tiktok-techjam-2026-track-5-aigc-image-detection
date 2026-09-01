# Third-party data notices

This preparation repository does not redistribute dataset contents. Local
copies remain ignored. Before any later use or public release, review the source
terms again and preserve required attribution.

## SID_Set

- Publisher: SIDA authors (`saberzl/SID_Set`)
- Dataset card licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Evidence: <https://huggingface.co/datasets/saberzl/SID_Set>
- Note: the card states that portions incorporate COCO, OpenImages V7 and
  Flickr30k material. Underlying asset terms and attributions still require
  review; the repository does not assume one blanket asset licence.

## CIFAKE

- Publisher: Jordan J. Bird and Ahmad Lotfi
- Stated licence: [MIT](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images/blob/e112a942abaecd02b6b1f6f646c807d56be8fb62/README.md#license)
- Required citations in the pinned README: CIFAR-10 and the CIFAKE paper.

## WildFake

- Publisher record: `hy2628982280/WildFake`
- ModelScope metadata licence: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Evidence: <https://modelscope.cn/datasets/hy2628982280/WildFake/summary>
- Note: WildFake aggregates material from multiple real-image and generator
  sources. Component provenance and restrictions must be audited before
  redistribution even though ModelScope labels the dataset Apache 2.0.
- The deterministic real-content diagnostic additionally samples WildFake's
  CelebA-HQ, LSUN Church and LAION-5B archives. CelebA/CelebA-HQ is restricted
  to non-commercial research use; LAION metadata is CC BY 4.0 but the linked
  web images retain source-specific rights; LSUN likewise aggregates web
  imagery. These subsets remain ignored local experiment inputs and are not
  redistributed. Their use in a released final checkpoint remains conditional
  on the organizer's public/non-proprietary-data interpretation and preserving
  all upstream obligations.

## COCO val2017

- Publisher: COCO Consortium
- Source archive: <http://images.cocodataset.org/zips/val2017.zip>
- COCO terms: <https://cocodataset.org/#termsofuse>
- Image licences: the acquired official `instances_val2017.json` assigns one of
  seven used licence IDs to every validation image (Creative Commons variants
  or no-known-restrictions). Licence ID 8 exists in the table but is not used by
  the 5,000 validation records.

The organizer's 4,998-image COCO subset and 8,843-image DALL-E Advanced subset
are demonstration-only and prohibited from training, independently of their
upstream copyright licences.

## RRDataset / RRBench

- Publisher record: Chunxiao Li and Yao Zhu, Zenodo record `14963880`
- Dataset record licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Evidence and immutable file metadata: <https://zenodo.org/records/14963880>
- Intended use: a size-bounded original train/validation archive for
  generator-family and real-scenario diversity, with attribution preserved.
- Note: the repository code is MIT-licensed, but that software licence is
  separate from the Zenodo dataset record's CC BY 4.0 terms. Generator and
  upstream real-source metadata will be inspected before individual images are
  admitted to training.

## AFHQ / AFHQ-v2

- Publisher: Clova AI Research, distributed with StarGAN v2.
- Upstream terms: [CC BY-NC 4.0](https://github.com/clovaai/stargan-v2#license).
- The active 512-pixel train/test subsets were acquired from the official
  StarGAN-v2 download object pinned in `resources/resource_manifest.json`.
- A separate WildFake 200-pixel derivative was verified but is excluded from
  the active family mixture after the shortcut audit exposed its uniform JPEG
  format and resolution.
- Use boundary: only AFHQ-v2 is eligible. Its official train and test trees are
  kept disjoint, and the older AFHQ release is excluded.

## FFHQ

- Publisher: NVIDIA.
- Dataset terms: [CC BY-NC-SA 4.0](https://github.com/NVlabs/ffhq-dataset/blob/master/LICENSE.txt),
  with each source Flickr image retaining its own licence.
- Acquired through the immutable WildFake FFHQ archive recorded in
  `resources/resource_manifest.json`.
- Use boundary: official FFHQ IDs 0-59,999 are eligible for the training pool;
  IDs 60,000-69,999 remain disjoint validation-only material. The active v2
  family mixture samples only from the first partition and evaluates only on
  the second.

## DiTFake

- Publisher: SAFE authors; pinned Hugging Face mirror `Jouesmak/DiTFake`.
- Dataset-card licence: [Apache License 2.0](https://huggingface.co/datasets/Jouesmak/DiTFake).
- Pinned revision: `ca9ea06c8f926c3a11ca4b657074cc7cbb99e5c7`.
- The release contains separately named FLUX.1-schnell, PixArt-Sigma and Stable
  Diffusion 3 Medium synthetic images plus COCO real images.
- Use boundary: `scripts/acquire_ditfake_fakes.py` acquires only the three
  `1_fake` trees. Every `0_real` COCO file is excluded, so no image from this
  source can cross the organizer's demo-only COCO boundary.

## Community Forensics Small

- Publisher: Community Forensics authors; pinned Hugging Face dataset
  `OwensLab/CommunityForensics-Small`.
- Dataset-card licence: CC-BY-NC-SA-4.0, with individual generator-model
  licences still applicable.
- Pinned revision: `6c539a534c07917307c381f5af4053c6091b5278`.
- Use boundary: only non-NSFW synthetic rows with LAION prompt provenance were
  retained. Every real row and every non-LAION prompt-source row was rejected.
  The selected subset is an external audit gate only until licence compatibility
  for training and redistribution is established.

## NTIRE 2026 Robust AI-Generated Image Detection

- Publisher: deepfakesMSU / NTIRE 2026 Robust AI-Generated Image Detection
  challenge organizers.
- Pinned dataset revision:
  `700b6d08a3268b1e7a191306dec7321dd953b12f` at
  <https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train>.
- Official challenge page: <https://www.codabench.org/competitions/12761/>.
- Challenge report: <https://arxiv.org/abs/2604.11487>.
- Licence boundary: the pinned dataset card does not declare an SPDX or
  Creative Commons licence. Use is therefore conservatively restricted to the
  official research/educational challenge purpose, and neither the archive nor
  pixels may be redistributed unless explicit terms are confirmed.
- Use boundary: the deterministic shard-5 sample is an independent audit-only
  gate. Its labels are not used for training, threshold calibration or model
  selection. Generator identities are undisclosed, so this source cannot prove
  per-family coverage.

## Qwen Image Bench

- Publisher: Qwen team; pinned Hugging Face dataset
  `Qwen/Qwen-Image-Bench`.
- Pinned revision: `d2493deb153b020cf169c7e3f57d15e4dd697038`.
- Dataset and licence evidence:
  <https://huggingface.co/datasets/Qwen/Qwen-Image-Bench> (Apache-2.0).
- Paper: <https://arxiv.org/abs/2605.28091>.
- Use boundary: prompt IDs are partitioned before training. The first 16 IDs
  per model are diagnosis-only and never enter training; a later sealed
  holdout must also remain pixel- and prompt-disjoint. A separate 32-prompt
  partition per model is candidate-training data and is never used as an
  evaluation gate. Source pixels and
  derived manifests remain ignored and are not redistributed here.
- This source does not use the organizer's prohibited WildFake DALL-E Advanced
  demonstration subset.

## CIFAR-100 independent low-resolution authentic gate

- Publisher: University of Toronto CIFAR authors; official page
  <https://www.cs.toronto.edu/~kriz/cifar.html>.
- Official Python archive: 169,001,437 bytes with published MD5
  `eb9058c3a382ffc7106e4002c42a8d85`.
- The official host transfer was interrupted after 17,825,792 bytes because it
  was too slow to finish within the challenge window. The test split is instead
  pinned to the public `uoft-cs/cifar100` Hugging Face mirror at revision
  `aadb3af77e9048adbea6b47c21a81e47dd092ae5`; its 23,772,751-byte Parquet file
  has LFS SHA-256
  `98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68`.
- The source contains 50,000 train and 10,000 test 32-by-32 colour images from
  100 classes that are mutually exclusive with CIFAR-10 classes.
- Licence boundary: the official page publishes the archive for research and
  requests citation but does not state an SPDX or Creative Commons licence.
  It is therefore evaluation-only here; pixels are not redistributed and do
  not enter trained-weight lineage unless terms are separately resolved.
- Purpose: a new authentic-side falsification of the discovered noisy-CIFAKE
  failure. It is not evidence that all low-resolution or real-world sources are
  covered, and it is unrelated to the organizer's prohibited demo resources.

## Experimental software and public backbones

- PyTorch: BSD-style licence; <https://github.com/pytorch/pytorch/blob/main/LICENSE>
- timm / PyTorch Image Models: Apache-2.0;
  <https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE>
- DINOv2 code and public weights: Apache-2.0;
  <https://github.com/facebookresearch/dinov2/blob/main/LICENSE>
- The audited Hugging Face model cards for
  `timm/resnet18.a1_in1k` and `timm/vit_large_patch14_dinov2.lvd142m` both
  declare Apache-2.0. Exact cached revisions, byte counts and SHA-256 values are
  recorded in `EXPERIMENT_LEDGER.md`; weights remain ignored and are not
  redistributed by this repository.
- PE-Core-L checkpoint `timm/vit_pe_core_large_patch14_336.fb`: Apache-2.0;
  <https://huggingface.co/timm/vit_pe_core_large_patch14_336.fb>. The official
  PE page separately confirms that its released PE checkpoints are Apache-2.0:
  <https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md>.
  This project loads the checkpoint through Apache-2.0 `timm`; it does not copy
  or import Meta's separately licensed `perception_models` repository code.
- The bias-control experiments were informed by UnbiasedGenImage
  (<https://github.com/gendetection/UnbiasedGenImage>) and AlignedForensics /
  Stay-Positive (<https://github.com/AniSundar18/AlignedForensics>), whose
  published code is Apache-2.0. No source code from either repository is copied;
  this repository independently implements the documented preprocessing and
  projected last-layer constraints and cites the research sources above.

## COCO train2017 commercial-use-compatible subset

- Source images: <http://images.cocodataset.org/train2017/>
- Official annotations/licence metadata:
  <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>
- The deterministic v12 acquisition retains only image licence IDs 4 (CC BY
  2.0), 5 (CC BY-SA 2.0), 7 (no known copyright restrictions) and 8 (United
  States Government Work). The observed 6,000-image sample contains IDs 4, 5
  and 7 only.
- Licence IDs 1, 2 and 3 contain non-commercial restrictions and are rejected;
  licence ID 6 is no-derivatives and is rejected. Every val2017 identity is
  rejected independently of licence.
- Required per-image source URLs, licence names and licence URLs are preserved
  in the ignored acquisition manifest. CC BY-SA obligations remain applicable
  to transformed images and must be reflected in any lawful distribution.
- This train2017 subset is not the organizer's 4,998-image demo-only val2017
  material and has zero identity overlap with all 5,000 val2017 records.
