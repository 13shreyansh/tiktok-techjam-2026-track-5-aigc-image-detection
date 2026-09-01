# PE-Core-L v12 detector model card

## Status

Selected submission candidate. The exact checkpoint has SHA-256
`f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2`,
contains 315,776,001 parameters and occupies 1,263,202,331 bytes. It is locally
and privately preserved but is not public until an immutable download URL is
recorded and independently reverified.

No result in this card is the organizer's hidden score.

## Task and output

Input is a directory of still images. Output is one continuous `pred` in
`[0, 1]` per `image_path`; larger values mean stronger evidence for fully
AI-generated content. The model does not identify a generator, localize edits,
process video/audio, or prove provenance.

## Architecture and training

- Backbone: public `vit_pe_core_large_patch14_336` PE-Core-L, Apache-2.0.
- Total parameters: 315,776,001, below the organizer's exclusive 2B limit.
- Trainable parameters: 1,025 in a binary linear head; the encoder is frozen.
- Input: 224-pixel short-side resize and centre crop after label-independent
  JPEG quality-96 normalization.
- Training: one source-balanced epoch with at most one randomly selected
  workshop transformation per image.
- Inference: float32 sigmoid probability, EXIF orientation honored, no
  threshold or test-time calibration.

The frozen representation and tiny trained head are deliberate: they reduce
the opportunity to memorize 13,574 training images while retaining a broadly
pretrained perceptual representation.

## Training data

The balanced mixture contains 6,787 authentic and 6,787 generated images.
Sampler mass on the real side is COCO train2017 commercial-compatible licences
60%, CIFAKE 25%, SID_Set 15%. Fake-side mass is WildFake 30%, DiTFake 25%,
Qwen Image Bench 20%, CIFAKE 20%, SID_Set 5%.

It spans natural scenes, people, objects and low-resolution content, plus GAN,
latent diffusion, Stable Diffusion, SD3, FLUX, PixArt and other recent named
generators. Exact source terms and revisions are in `THIRD_PARTY_NOTICES.md`.

Recorded organizer-demo rows: **0**. Recorded non-commercial rows: **0**.
Train/evaluation byte overlap: **0**.

Before training, every image from both labels was EXIF-transposed, centre-square
cropped, resized to 336 and encoded as JPEG q96 with the same settings. This
reduced the evaluation metadata-only AUC from 0.9984 to 0.5131. It removes the
tested container and original-dimension shortcut, not every possible source or
content shortcut.

## Evaluation

| Frozen audit | Rows | Clean AUC | Pooled transformed AUC | 50/50 score | Worst condition |
|---|---:|---:|---:|---:|---:|
| CIFAKE matched source | 2,000 | 0.9228 | 0.8920 | 0.9074 | 0.7945 |
| Semantic-matched modern | 288 | 0.9982 | 0.9941 | 0.9962 | 0.9848 |
| Open Images source rotation | 288 | 0.9819 | 0.9700 | 0.9759 | 0.9397 |
| Community Forensics / 78 model names | 593 | 0.9502 | 0.9473 | 0.9487 | 0.8234 |
| Final NTIRE source-coherent audit | 1,024 | 0.9907 | 0.9837 | 0.9872 | 0.9308 |

Each robust score pools the 19 organizer-listed transformations applied one at
a time. The final candidate decision was frozen before scoring and selected PE
over DINO and their equal blend. PE improved over the blend by 0.0510 on the
organizer-style score and 0.0824 on the weakest condition.

## Error analysis and limitations

On the final clean gate, an illustrative—not calibrated—threshold of 0.5 gives
161/512 false positives and 4/512 false negatives despite 0.9907 AUC. The model
ranks well but its probabilities are shifted toward the AI label. Visual review
found the strongest authentic false positive was a human-made surreal animal
collage; another was an unusual close-up phone photograph. The weakest AI
examples resembled ordinary travel or event photography. This is why the
prototype emits confidence for ranking and does not claim a universal verdict.

Remaining risks:

- unseen generator families and real-image sources;
- human-made art, graphics and unusual photographic composition;
- inherited codec, resampling or collection fingerprints after normalization;
- heavy Gaussian noise, the repeatable weakest transformation;
- unknown hidden class balance and no permitted calibration set;
- audit sets with small per-generator samples or known collection confounds.

## Runtime and intended use

The selected model completed a four-image Apple M5 Pro MPS run in 2.68 seconds
at batch one. The PE-only path also completed on one Tesla T4; the historical
four-image run took 11.25 seconds including checkpoint validation and loading.
CPU is supported but slower. These small measurements are run-contract evidence,
not production throughput benchmarks.

Use the score to prioritize human review or provide context. Do not use it as
sole evidence of deception, authorship or fraud. See `MODEL_WEIGHTS_LICENSE.md`
and `THIRD_PARTY_NOTICES.md` for licence boundaries.
