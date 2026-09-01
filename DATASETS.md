# Dataset acquisition and provenance

Machine-readable lock data is in `resources/resource_manifest.json`. Dataset
archives, extracted images and metadata copies live under ignored `datasets/`.

## Acquired and verified locally

### CIFAKE

The public Kaggle archive was downloaded without credentials and verified:

```text
bytes   109625224
MD5     dedf8700a2a671c84c60c1986143548a
SHA256  1839206c1462371f5c885eda1660b448ea7abf52ee81fcc71e5e4651a4a11389
images  120000 (100000 train, 20000 test; 60000 REAL, 60000 FAKE)
```

The author states an MIT licence and requires citation of CIFAR-10 and the
CIFAKE paper. Repository evidence is pinned at commit
`e112a942abaecd02b6b1f6f646c807d56be8fb62`.

### COCO val2017 source archive

The official archive was downloaded and passed `unzip -tq`:

```text
bytes   815585330
MD5     442b8da7639aecaf257c1dceb8ba8c80
SHA256  4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05
images  5000 source JPG files
```

COCO images retain per-image Flickr/COCO licences. This source is demo-only for
Track 5 and must not enter training. The official annotations archive is also
acquired and verified (`252907541` bytes; MD5
`f4bbac642086de4f52a3fdda2de5fa2c`; SHA-256
`113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`).
All 5,000 validation image records carry one of the COCO licence IDs. The exact
organizer subset is blocked until the 4,998-image selection/exclusion manifest
is supplied.

### SID_Set binary shard pair

After the official start, one immutable train shard and one immutable
validation shard were downloaded from the pinned SID_Set revision and verified
before extraction. They contain 576 and 586 eligible real/fully-synthetic
images respectively; 565 label-2 tampered images were excluded because the
organizer's evaluated boundary is pure real versus fully AI-generated. Exact
source sizes, SHA-256 values and label counts are in `EXPERIMENT_LEDGER.md`.

After the v12 matched-source gate exposed the need for a fresh high-resolution
same-source control, validation shard `00001-of-00034` was pinned before
acquisition at immutable SID_Set revision
`dc03ead57929879319ce30a82bfcfb8d317b10bd`. Its pinned-revision response
reports 505,844,042 bytes and linked SHA-256
`1447bbd98adf7eda68fca5615560c6b1de34c8e30157ff6b34ebd1e015a18042`.
The shard's earlier upload commit reported a different LFS identity; that older
value was rejected after the downloaded bytes and pinned-revision headers both
matched the newer identity.
Acquisition and extraction returned zero: 284 real, 290 fully synthetic and
309 tampered rows were observed. The tampered rows remain excluded, and the
eligible pixels are reserved for evaluation only.

### WildFake DDPM, DDIM and ImageNet experiment sources

The public DDPM and DDIM fake archives and matching ImageNet real archive were
downloaded from immutable ModelScope revisions and passed both SHA-256 and ZIP
integrity checks. Their label indexes were also verified:

```text
DDIM.zip            6054264809 bytes, 65713 indexed fakes
DDPM.zip            8141353209 bytes, 76561 indexed fakes
imagenet.zip        1378959009 bytes, 96788 indexed reals
ddim.csv               7498439 bytes, 65713 data rows
ddpm.csv               8712759 bytes, 76561 data rows
real_imagenet.csv      8563441 bytes, 96788 data rows
```

All four digests and URLs are locked in `resources/resource_manifest.json`.
`scripts/extract_wildfake_binary.py` creates the first cross-source diagnostic.
`scripts/prepare_wildfake_generator_split.py` constructs the stronger split:
DDPM fakes for training, DDIM fakes for testing, and disjoint ImageNet real
images on both sides. That design changes the fake generator while holding the
real-image source fixed.

### WildFake deterministic real-content shards

To expand authentic-image content without downloading the approximately
1.29 TB collection, `scripts/acquire_wildfake_remote_subset.py` read three
immutable remote ZIP directories and fetched deterministic byte ranges only.
Each retained image is decode-verified and preserves its archive path, CRC32,
archive revision and seeded selection inventory:

```text
CelebA-HQ portraits   1024 / 30000 available   11897915 retained bytes
LSUN Church scenes   1024 / 83352 available   14453360 retained bytes
LAION-5B web mix      1024 / 271831 available  97914013 retained bytes
```

A second LAION selection used seed `20260830` and retained 1,050 images after
fetching 127,196,121 remote bytes. It has zero archive-member overlap and zero
SHA-256 content overlap with the first shard. Its selection-inventory SHA-256
is `78d4d20075b21012f1a1426833b38df0099d0a2eb9b5419d5604c7191f0cda66`
and its manifest SHA-256 is
`8eb0ef4ba41140b02bc07cb7a56aa960d17c68375fd9cf56cd7609ef5b40d24c`.
This supports an honest same-source control: shard A may enter training while
the first 1,024 byte-disjoint shard-B images remain evaluation-only.

A second LSUN Church selection also used seed `20260830`, explicitly excluded
all 1,024 archive members in shard A, and retained 1,024 decodable images. It
fetched 23,146,124 remote bytes; the retained files occupy approximately 16 MB.
Both archive-member overlap and SHA-256 content overlap with shard A are zero.
Its selection-inventory SHA-256 is
`bbe498f14e24dfcd50ab8aee26a96a9a2f1a2b4aba622e25d08d59ae6fd1c5c7`
and its manifest SHA-256 is
`b046bae166686436f9dd296cb9261be46c35f9ad7284469554718373d672dc33`.
This permits the same honest train-A/evaluate-B control for Church scenes.

The corresponding whole-archive sizes and linked SHA-256 values are locked in
`resources/resource_manifest.json`. These 3,072 images first enter a completely
held-out content diagnostic; they are not automatically admitted to training.
The gate has zero resolved-path and zero SHA-256 content overlap with the active
source-repair training manifest. COCO was deliberately not sampled.

The same remote-range method also acquired 1,024 decode-verified fakes from
each of three additional generator families: ADM (pixel diffusion), Imagen
(cascaded text-to-image diffusion) and VQDM (vector-quantized diffusion). These
3,072 images first form a balanced, fully unseen generator-versus-real-content
diagnostic against the three real shards above. They are not automatically
folded into training, so their transfer value is measured before admission.

### RRDataset original train/validation archive

Zenodo record `14963880` was rechecked through its public API, then its
2,163,176,547-byte original train/validation archive was downloaded. The source
MD5 matched `2f4498c3690d8f4c7a30d2e41dd34500`; local SHA-256 is
`b7f72dabe654877354300c7cd1181f493ccc8299bcea0a76dacf64fea88e0936`.
Full tar traversal succeeded and found 3,000 images: 1,250 real plus 1,250 AI
for training and 250 plus 250 for validation. The Zenodo record declares CC BY
4.0.

The archive is **not admitted wholesale to training**. Its everyday `normal_*`
portion mixes COCO-derived reals and DALL-E 3 outputs, creating an unacceptable
possibility of crossing the organizer's forbidden COCO/DALL-E demo boundary.
Only separately named special-scenario fakes are considered: the paper states
that those were generated with SD 3.5 Large and Flux.1. All RRDataset real
images are excluded. The reproducible audit in
`scripts/audit_rrdataset_demo_overlap.py` found no byte-identical files but
found eight dHash-near matches at distance at most six against the forbidden
COCO source. Visual inspection confirmed that multiple distance-zero/one pairs
are the same photographs after resizing or re-encoding. This is direct
contamination evidence, not merely a theoretical risk.

The admitted RR manifest therefore contains 771 training and 157 validation
special-scenario **fake** images only, paired with disjoint WildFake ImageNet
real photographs. It excludes every `normal_*` fake and every RR real.

### Community Forensics Small synthetic-only shard

Pinned revision `6c539a534c07917307c381f5af4053c6091b5278`, shard 0,
was acquired and verified at 1,231,926,042 bytes with SHA-256
`0ce98c9b4f66eca160939982fe7aac84253af7d135485e5bc83ca8425cfe220c`.
The deterministic acquisition retained four non-NSFW, LAION-prompt-source
fake images from each of 78 named latent-diffusion model variants: 312 images
total. It admitted no real row, no COCO-derived prompt row and no organizer
demo pixel. The selected model inventory SHA-256 is
`f4a0f040c98122d6d576e0fc0c2812dc99dba8d256ec56fcecc3e3838da13dd8`.

This is currently an **external audit source only**, not training data. Its
CC-BY-NC-SA-4.0 terms and the upstream terms of individual generator models
must be resolved before any training use or redistribution. The 78 variants
improve model-count breadth but do not substitute for architecture diversity.

### NTIRE 2026 Robust AI-Generated Image Detection shard 5

The official public training record for the NTIRE 2026 challenge was pinned at
revision `700b6d08a3268b1e7a191306dec7321dd953b12f`. Its six ZIP shards total
114,357,930,038 bytes. The smallest shard was downloaded with resume support,
then checked against its pinned size and SHA-256 before a full ZIP traversal:

```text
shard_5.zip  11370161676 bytes
SHA256       6d6628c983c43f1de44589151e2b3b9d33726691efbd9b0208e9f015ded9af8f
members      27646 (9947 real, 17696 fake, plus directory entries)
format       JPEG for every image member
```

The source uses randomized filenames and does not expose per-image generator
identities. A deterministic, audit-only sample retained 256 images per class;
its manifest SHA-256 is
`6a8741c06b586e499c4db96c2dddf0a8e3571516f4951f859244b882fdce78c7`.
An exact format/mode/width/height match produced 43 real plus 43 fake images
across 31 geometry strata. This subset is intentionally small and remains
evaluation-only. Its labels must not be used for threshold calibration, model
selection or training. If this dataset later contributes to training, a
disjoint remainder or separate shard must be used and this audit set must stay
untouched.

The pinned dataset card does not declare an SPDX or Creative Commons licence.
Until explicit terms are located, use is conservatively limited to the
official research/educational challenge purpose; no archive or pixels may be
redistributed. This unresolved licence is a real release blocker, not an
assumption that the data is unrestricted.

### Qwen Image Bench frontier-generator controls

The official Qwen Image Bench was pinned at revision
`d2493deb153b020cf169c7e3f57d15e4dd697038`. It contains the same 1,000
prompts rendered by 18 current generators: FLUX.2 Pro/Max, GLM Image, GPT
Image 1/1.5/2, HunyuanImage 3, Imagen 4/Ultra, three Qwen Image versions,
Seedream 4/4.5/5, Kling 2.1 and Nano Banana 2/Pro. The pinned card declares
Apache 2.0.

A deterministic audit selected the same 16 prompt IDs for every generator:
288 images and 307,295,348 bytes. Every downloaded file matched the pinned API
size, received a local SHA-256, and decoded successfully. The source inventory
SHA-256 is
`cc553cdf6d3be2ceb2143b7ef9dbb326cc9dc7530d55479d43ad35e981c7e5e0`;
the enriched manifest SHA-256 is
`9c8e8f9e95f426761df05b3dd9e2f8c6c99ada392de96f0db0803f737b0c0d98`.
Those 16 prompt IDs are audit-only. Candidate-training and final holdout prompt
IDs must be disjoint, and no score may be described as generator-unseen after
the same named generator family enters training.

A second deterministic partition selected 32 different prompt IDs per model
for candidate training: 576 images and 612,250,940 bytes. Its manifest SHA-256
is `d22a5af898dad311e69798d0ca5dca3f7ef511e7ed9c9284f4e27720bf0b7020`.
The checksum-verified Kaggle transport package is 612,717,221 bytes with
SHA-256 `650f309ddb4fd8d0b7ac05d101981460319b9de2f39fd7795b1413982145fa93`.
A third prompt-disjoint partition is reserved as a sealed comparison set: 288
images, 311,365,922 bytes, source inventory SHA-256
`c6e3d40e202aa5743707a1d8c581698745c690a33d70d0c60ee04a7bf8103b61`,
and manifest SHA-256
`5663056fe2b55f5a67a1395432e9fc417d58a6d10d368774e019b43e5a5ffb59`.
It has zero prompt and SHA-256 overlap with the diagnosis and training
partitions and remains unscored until the candidate run is frozen. Once these
named generators enter training, this is a prompt/pixel holdout, not a
generator-unseen test.

## Full manifests, not fully downloaded

### SID_Set

- Hugging Face revision: `dc03ead57929879319ce30a82bfcfb8d317b10bd`
- Public snapshot: 286 files, 140,056,468,470 bytes.
- Canonical path/size/blob inventory SHA-256:
  `bca82b8b1811e30b157082a881327ba5476449987f0e61d175e74f6eadb0257e`.
- Available rows: 210,000 train and 30,000 validation.
- The stated 60,000-image test split is obtained separately through the authors'
  repository and is not part of the 240,000-row Hugging Face snapshot.
- Licence: CC BY 4.0, including attribution obligations described in the card.

The metadata API exposes every shard's blob/LFS identity. A full download was
not started because it is approximately 140 GB.

### WildFake

- Official ModelScope record reports 1,291,478,056,101 stored bytes.
- Recursive public tree on 26 August enumerated 88 files totalling
  1,287,902,462,734 bytes; the discrepancy is preserved, not reconciled away.
- Canonical path/size/SHA-256/revision inventory SHA-256:
  `4d19f1021f0d74f4062ced323665de9bf215d38da76eebcf5494afb3e0be2e82`.
- Licence reported by ModelScope: Apache 2.0.
- Official pointer repository commit:
  `27c7314bfb8016ab233472bb691029aa51e046ec`.

The full dataset was not downloaded because it is approximately 1.29 TB. Full
DDPM/DDIM/ImageNet sources, selected GAN/Stable-Diffusion ranges, and the three
real-content shards described above were acquired. ModelScope supplies per-file
sizes, revisions and SHA-256 values.

### DALL-E Advanced demo material within WildFake

The pinned `dalle3.csv` has 8,843 rows, all marked `IsAdvanced=1` and
`IsFake=1`. It indexes the organizer's stated 8,843-image DALL-E Advanced demo
subset. The containing `DALLE.zip` is 25,587,709,291 bytes and also contains
non-demo content; its SHA-256 is locked in the manifest. It was not downloaded
during this pass due to its size.

## Reproducible commands

```bash
python3 scripts/acquire_resources.py list
python3 scripts/acquire_resources.py verify cifake_archive
python3 scripts/acquire_resources.py verify coco_val2017_archive
python3 scripts/acquire_resources.py download wildfake_dalle3_index
python3 scripts/verify_inventory.py
python3 scripts/verify_remote_manifests.py
```

Large downloads additionally require `--allow-large-download`. That guard is
deliberate: it makes a multi-gigabyte acquisition explicit without embedding
credentials or placing assets under version control.

## Active permissive v12 candidate lineage

The workshop disallows any dataset marked non-commercial. Historical v6/v9
training manifests therefore remain experiment records only. The active v12
lineage uses only the explicitly commercial-compatible COCO train2017 subset,
CIFAKE, SID_Set, DiTFake synthetic trees, Qwen Image Bench and the WildFake
dataset record. It includes no organizer val2017/DALL-E demo row.

The raw v12 files have severe label-correlated format and geometry structure.
They are never the trusted GPU input. `scripts/materialize_canonical_v12.py`
applies the same decode, centre crop, 336-square resize and JPEG-q96 encoding to
both labels. This changed the frozen metadata-only evaluation AUC from 0.998360
to 0.513093. The exact train and frozen-evaluation manifests, images, packages
and reports remain ignored; acquisition/materialization scripts and hash
evidence are committed instead.
