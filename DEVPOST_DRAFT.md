# Devpost draft — robust AIGC image detection

Draft only. Bracketed links are unresolved external actions. This file is not a
submission or publication.

## Inspiration

Modern generated images no longer need obvious visual mistakes, while social
platforms routinely recompress, resize, crop, blur, add noise and recolour
images. A detector that succeeds only on pristine outputs from familiar models
creates false confidence. We treated the challenge as a robustness and
generalization problem: rank authentic and fully AI-generated still images
while actively looking for source, format, geometry and content shortcuts.

## What it does

The prototype accepts a directory of JPG, PNG, WebP, BMP or TIFF files and
writes one ordered JSON row per image with `image_path` and a continuous
AI-positive `pred` score. It uses one checksum-pinned PE-Core-L vision model
with 315,776,001 parameters, below the exclusive two-billion limit. No
universal yes/no threshold is claimed because the official metric is ROC AUC
and the hidden class balance is unknown.

## How we built it

We froze the public Apache-2.0 PE-Core-L encoder and trained only a 1,025-
parameter binary head. The 13,574-row balanced mixture spans authentic scenes,
people, objects and low-resolution images, plus GAN, diffusion, Stable
Diffusion, SD3, FLUX, PixArt and other recent generated-image sources. Every
training view receives either no degradation or one workshop-listed transform.

The raw data was almost perfectly separable without pixels because formats and
dimensions leaked the labels. We therefore applied the same EXIF correction,
square crop, resize and JPEG-q96 encoding to both classes. The pixel-free
metadata AUC fell from 0.9984 to 0.5131; literal PNG and square-shape rules fell
to 0.5. We also used exact and canonical deduplication, full source/generator
holdouts, real-source rotation and prompt/content pairing.

The organizer's demo-only 4,998 COCO val2017 and 8,843 DALL-E Advanced images
were used zero times for training, tuning, model selection, calibration or
thresholding. Recorded non-commercial training rows are also zero.

## Evaluation strategy and evidence

We applied all 19 workshop settings individually and computed the workshop
score: 50% clean ROC AUC plus 50% pooled transformed ROC AUC. The final model
choice was determined by a rule committed before either candidate saw the
final 1,024-image source-coherent gate.

| Frozen audit | Clean AUC | Transformed AUC | 50/50 score | Weakest condition |
|---|---:|---:|---:|---:|
| CIFAKE matched source | 0.9228 | 0.8920 | 0.9074 | 0.7945 |
| Modern semantic pairs | 0.9982 | 0.9941 | 0.9962 | 0.9848 |
| Open Images source rotation | 0.9819 | 0.9700 | 0.9759 | 0.9397 |
| Community Forensics, 78 model names | 0.9502 | 0.9473 | 0.9487 | 0.8234 |
| Final NTIRE source-coherent audit | **0.9907** | **0.9837** | **0.9872** | **0.9308** |

On the final gate, PE beat the fixed PE/DINO blend by 0.0510 on the 50/50
score and by 0.0824 on the weakest condition. These are development results,
not a prediction of the unpublished organizer score.

## What we learned

The hardest problem was validation illusion. A model can appear excellent by
learning PNG versus JPEG, square versus non-square, a dataset's subject matter,
or the collection pipeline rather than generation. Multiple controlled gates
rejected attractive but fragile conclusions: a DINO branch helped one narrow
low-resolution slice, then failed badly after rotating the authentic source.
The final one-model system is faster, simpler and stronger on the untouched
arbitration gate than that ensemble.

The remaining errors are also instructive. At an illustrative, uncalibrated
0.5 cutoff, the final clean gate has 161/512 false positives but only 4/512
false negatives despite 0.9907 AUC. The strongest authentic false positive is
a surreal human-made animal collage; the weakest generated examples look like
ordinary travel and event photographs. This is why the prototype outputs a
ranking score and should support human review rather than assert provenance.

## Impact and practical use

The detector can help moderators, fact-checkers and users prioritize images
for review before they infer authenticity from appearance alone. It should
never be the sole evidence of deception, authorship or fraud. Heavy Gaussian
noise, human-made art, realistic generated photographs and unseen collection
pipelines remain explicit limitations.

## Built with

Python, PyTorch, torchvision, timm/PE-Core-L, scikit-learn and Pillow. The exact
runtime is verified on an Apple M5 Pro through MPS and on an NVIDIA Tesla T4.

## Links and final action gates

- Public source repository: `[VERIFY_AND_INSERT]`
- Public PE-Core-L v12 checkpoint: `[VERIFY_AND_INSERT]`
- Public two-to-four-minute YouTube demo: `[VERIFY_AND_INSERT]`

Before submission: rebuild the source bundle from the final commit; publish and
redownload the checkpoint; verify its SHA-256; run from a logged-out clean
environment; verify the video and repository; then remove every placeholder.
