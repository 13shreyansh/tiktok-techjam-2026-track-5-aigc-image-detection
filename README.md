# Signal Survives

**A shortcut-resistant AIGC image ranker built to retain evidence across real-world redistribution and unseen generator families.**

TikTok TechJam 2026, Track 5 — Robust Detection of AI-Generated Images under Real-World Transformations.

Signal Survives accepts a directory of still images and returns one continuous
AI-positive confidence per image. It is designed for the organizer's hidden
evaluation: pure authentic versus fully AI-generated images, scored by ROC AUC
on clean inputs and on pooled individually transformed inputs.

> The results below are frozen development evidence, not the organizer's hidden score.
> The organizer's demonstration-only COCO/DALL-E resources
> were never used for training, tuning, selection, calibration or thresholding.

## Judge quick path

| What the organizer requested | Where to verify it |
|---|---|
| Directory-to-confidence-JSON runner | [`run_v12.sh`](run_v12.sh) and [Quick start](#quick-start) |
| Public model below 2B parameters | [v1.0.0 checkpoint release](https://github.com/13shreyansh/tiktok-techjam-2026-track-5-aigc-image-detection/releases/tag/v1.0.0) and [`SELECTED_CHECKPOINT.sha256`](SELECTED_CHECKPOINT.sha256) |
| Clear technical and data-pipeline design | [Technical approach](#technical-approach) and [`MODEL_CARD.md`](MODEL_CARD.md) |
| Clean-versus-transformed evaluation | [Frozen development results](#frozen-development-results) |
| False-positive and false-negative analysis | [`ROBUSTNESS_AND_ERROR_ANALYSIS.md`](ROBUSTNESS_AND_ERROR_ANALYSIS.md) |
| Trade-offs, failures and limitations | [What failed and what remains uncertain](#what-failed-and-what-remains-uncertain) |
| Reproducible source, licences and provenance | [Reproducibility](#reproducibility), [`DATASETS.md`](DATASETS.md), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |

## Quick start

Requirements: Python 3.9+ and enough storage for the 1.18 GiB checkpoint.
The runner selects CUDA, Apple MPS or CPU automatically.

```bash
git clone https://github.com/13shreyansh/tiktok-techjam-2026-track-5-aigc-image-detection.git
cd tiktok-techjam-2026-track-5-aigc-image-detection

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runtime.txt

mkdir -p models
curl -fL \
  https://github.com/13shreyansh/tiktok-techjam-2026-track-5-aigc-image-detection/releases/download/v1.0.0/v12_pe_core.pt \
  -o models/v12_pe_core.pt
shasum -a 256 -c SELECTED_CHECKPOINT.sha256
```

Run inference on a directory containing JPG, JPEG, PNG, WebP, BMP or TIFF
files:

```bash
PYTHON_EXECUTABLE=.venv/bin/python AIGC_DEVICE=auto \
  ./run_v12.sh INPUT_DIRECTORY predictions.json
```

Example output:

```json
[
  {"image_path": "INPUT_DIRECTORY/example.jpg", "pred": 0.8731}
]
```

There is exactly one output row per discovered image. Larger `pred` values mean
stronger AI-generated evidence. The output is a continuous score for ranking,
not a hard verdict: ROC AUC does not require a universal threshold, and a
deployment threshold depends on class balance and the cost of false positives.

The default batch size is one for maximum Apple MPS compatibility. CUDA users
may raise it explicitly, for example with `AIGC_BATCH_SIZE=8`. The runner
validates inputs and checkpoint lineage, applies EXIF orientation and the same
label-independent JPEG/crop contract used during training, and writes JSON
atomically.

## Technical approach

### 1. A deliberately simple selected model

- **Backbone:** public Apache-2.0 PE-Core-L
  (`vit_pe_core_large_patch14_336`).
- **Model size:** 315,776,001 total parameters, below the organizer's exclusive
  2,000,000,000-parameter limit.
- **Trained component:** a 1,025-parameter binary linear head; the vision
  encoder remains frozen.
- **Runtime input:** EXIF-corrected, JPEG-q96-normalized, 224-pixel short-side
  resize plus centre crop.
- **Output:** float32 sigmoid confidence with AI-generated as the positive
  class; no test-time calibration and no predetermined threshold.

Freezing the broadly pretrained encoder and training only the small head limits
the capacity available to memorize 13,574 training images. The final selection
is one model, not the more fragile two-backbone ensemble tested during
development.

### 2. Data alignment before architecture tricks

The balanced training mixture contains **6,787 authentic and 6,787 generated
images** across three authentic sources and five generated-data sources. It
spans people, animals, objects, scenes and low-resolution content, with GAN,
latent-diffusion, Stable Diffusion, SD3, FLUX and other recent generator
outputs. PixArt and Imagen remained evaluation-only holdouts.

The first raw-data result was misleading: a classifier that never opened an
image reached **0.9984 AUC** from container and geometry metadata. We therefore
applied the same label-blind EXIF correction, centre crop, resize and JPEG-q96
encoding to both classes. The metadata-only AUC fell to **0.5131**; PNG-only and
square-only rules fell to 0.5. Raw and canonical hashes, identities, prompts,
generators and real-image sources were then separated across training and
evaluation where the source permitted it.

### 3. Train and evaluate for the stated real-world conditions

Each training view receives either no degradation or one randomly selected
workshop transformation. Transformations are never chained, matching the
workshop clarification. Evaluation applies all 19 listed settings individually:

- JPEG quality 90, 70, 50 and 30;
- Gaussian blur sigma 0.5, 1.0 and 2.0;
- downscale to 0.5x or 0.25x, then upscale;
- Gaussian noise sigma 0.02, 0.05 and 0.10;
- brightness, contrast and saturation at plus or minus 20%;
- 80% centre crop.

Model decisions were challenged with source, generator, prompt, identity and
transformation holdouts rather than one random split. The final candidate rule
was frozen before the last 1,024-image source-coherent audit was opened.

## Frozen development results

The workshop score is:

```text
0.50 * ROC_AUC(clean) + 0.50 * ROC_AUC(pooled individually transformed)
```

### Final candidate arbitration

| Candidate | Clean AUC | Transformed AUC | Workshop-style score | Worst condition |
|---|---:|---:|---:|---:|
| **PE-Core-L v12 (selected)** | **0.9907** | **0.9837** | **0.9872** | **0.9308** |
| Fixed 50/50 PE + DINO | 0.9430 | 0.9294 | 0.9362 | 0.8484 |
| DINOv2-L control | 0.7303 | 0.7202 | 0.7253 | 0.6665 |

All five precommitted selection checks passed. Exact plan hashes,
per-condition output hashes and the consumed decision are recorded in
[`NTIRE_V12_FINAL_ARBITRATION_RESULT.json`](NTIRE_V12_FINAL_ARBITRATION_RESULT.json).
Gaussian noise sigma 0.10 was the repeatable weakest condition.

### Evidence beyond one gate

| Frozen audit | Rows | Clean AUC | Transformed AUC | Workshop-style score | Important boundary |
|---|---:|---:|---:|---:|---|
| CIFAKE matched source | 2,000 | 0.9228 | 0.8920 | 0.9074 | low-resolution CIFAR-10 / Stable Diffusion only |
| Semantic-matched modern | 288 | 0.9982 | 0.9941 | 0.9962 | Qwen-versus-COCO collection confound |
| Open Images real-source rotation | 288 | 0.9819 | 0.9700 | 0.9759 | reuses the modern fake collection |
| Community Forensics, 78 model names | 593 | 0.9502 | 0.9473 | 0.9487 | four fakes per model; latent-diffusion-heavy |
| Final source-coherent audit | 1,024 | 0.9907 | 0.9837 | 0.9872 | generator identities undisclosed |

No local gate reproduces the organizer's hidden source/generator mixture; the
table is evidence of stress testing, not a hidden-score prediction.

## Error analysis

At an illustrative, deliberately uncalibrated 0.5 threshold on the final clean
gate, the detector produced 161/512 false positives and 4/512 false negatives
despite 0.9907 ROC AUC. This is expected when the ranking is strong but the
score distribution is shifted: **0.5 is not a safe deployment threshold**.

Manual review found:

- a human-made surreal animal collage and an unusual close-up phone photo among
  the strongest authentic false positives;
- convincing travel- and event-style generated scenes among the weakest AI
  examples;
- heavy Gaussian noise as the clearest transformation weakness.

The remaining errors are not reducible to simple cues such as malformed hands.
Signal Survives should prioritize human review or add context; it must not be
used as sole proof of authorship, deception or fraud. Full immutable error
indices, image hashes and per-condition results are documented in
[`V12_ERROR_ANALYSIS_RESULT.json`](V12_ERROR_ANALYSIS_RESULT.json) and
[`ROBUSTNESS_AND_ERROR_ANALYSIS.md`](ROBUSTNESS_AND_ERROR_ANALYSIS.md).

## What failed and what remains uncertain

- The initial 0.9984 score was a metadata shortcut, not image understanding.
- A low-resolution repair path improved one gate but harmed clean images.
- A quality router added complexity without enough general benefit.
- DINOv2-L weakened when the authentic-image source changed.
- The PE/DINO ensemble lost to the selected single PE model on the untouched
  final arbitration gate.
- Unseen generators, unfamiliar real-image sources, human-made art, unusual
  composition and inherited collection fingerprints remain open risks.
- The hidden class balance is unknown and the prohibited demonstration data
  cannot be used to calibrate a threshold.

Rejected experiments remain in the repository as auditable evidence; they are
not silently rewritten as successes. See [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md)
and [`PARANOID_SELF_AUDIT.md`](PARANOID_SELF_AUDIT.md).

## Reproducibility

Run the repository tests:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The final tracked suite contains **219 tests** covering the inference contract,
checkpoint/hash validation, preprocessing, transformations, metrics, release
lineage, output schema and submission evidence.

Key reproducibility files:

- [`SELECTED_CHECKPOINT.sha256`](SELECTED_CHECKPOINT.sha256) — exact public
  checkpoint digest.
- [`V12_CHECKPOINT_MANIFEST.json`](V12_CHECKPOINT_MANIFEST.json) — final model,
  public release, training-lineage and runtime inventory.
- [`MODEL_CARD.md`](MODEL_CARD.md) — intended use, architecture, training,
  evaluation and limitations.
- [`DATASETS.md`](DATASETS.md) and
  [`resources/resource_manifest.json`](resources/resource_manifest.json) —
  source revisions, licences, checksums and acquisition records.
- [`MODEL_WEIGHTS_LICENSE.md`](MODEL_WEIGHTS_LICENSE.md) and
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — model and upstream licence
  boundaries.
- [`OFFICIAL_REQUIREMENTS.md`](OFFICIAL_REQUIREMENTS.md) — reconciled organizer
  requirements and workshop clarifications.

Large datasets, generated outputs, caches and model weights are intentionally
excluded from Git. The selected checkpoint is distributed as the immutable
[`v1.0.0` GitHub Release asset](https://github.com/13shreyansh/tiktok-techjam-2026-track-5-aigc-image-detection/releases/download/v1.0.0/v12_pe_core.pt),
SHA-256 `f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2`.

## Repository map

```text
run_v12.sh                         one-command selected inference entrypoint
src/aigc_detector/                 model, preprocessing and prediction code
tests/                             contract, integrity and evidence tests
MODEL_CARD.md                      selected-model documentation
ROBUSTNESS_AND_ERROR_ANALYSIS.md   robustness table and concrete FP/FN analysis
DATASETS.md                        dataset provenance and contamination controls
THIRD_PARTY_NOTICES.md             upstream licences and attributions
EXPERIMENT_LEDGER.md               chronological experiments and rejected ideas
demo/                              self-contained six-scene evidence brief
```

To inspect the visual evidence brief locally:

```bash
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/demo/> and use the arrow keys. The page reports
verified development evidence and limitations; it does not simulate inference
or claim an organizer result.
