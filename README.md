# Robust AIGC Image Detection

TikTok TechJam 2026 Track 5 prototype for ranking authentic and fully
AI-generated still images after realistic redistribution. The output is a
continuous AI-positive confidence, matching the organizer's ROC-AUC metric.

## Selected detector

The default is one **315,776,001-parameter PE-Core-L vision encoder plus a
1,025-parameter trained binary head**. The public Apache-2.0 backbone is frozen;
only the linear head was trained. This reduces overfitting and makes the final
artifact one checkpoint rather than a fragile two-model ensemble.

The default was selected by a rule frozen before opening the final 1,024-image
source-coherent audit. On that gate:

| Candidate | Clean AUC | Pooled transformed AUC | Organizer-style score | Worst condition |
|---|---:|---:|---:|---:|
| **PE-Core-L v12** | **0.9907** | **0.9837** | **0.9872** | **0.9308** |
| fixed 50/50 PE + DINO | 0.9430 | 0.9294 | 0.9362 | 0.8484 |
| DINOv2-L control | 0.7303 | 0.7202 | 0.7253 | 0.6665 |

All five precommitted selection checks passed. The exact decision, plan hashes,
per-condition output hashes and limitations are in
`NTIRE_V12_FINAL_ARBITRATION_RESULT.json`. These are development measurements,
not the organizer's hidden score.

## Why it is robust

1. **Broad, eligible data:** 13,574 balanced training rows span three authentic
   sources and five generated-data sources, including GAN, older diffusion and
   recent diffusion-transformer outputs.
2. **Shortcut removal:** both labels are EXIF-normalized, centre-cropped,
   resized and re-encoded identically. A pixel-free metadata classifier fell
   from 0.9984 to 0.5131 AUC; PNG-only and square-only rules both fell to 0.5.
3. **Realistic training:** each training view receives either no degradation or
   one randomly selected workshop transform—never an invented transform chain.
4. **Hostile evaluation:** model choices are tested across complete source,
   generator, prompt, identity and transformation holdouts, not one random
   split. Failed and source-confounded gates remain disclosed.
5. **Practicality:** one checksum-pinned model runs on NVIDIA CUDA, Apple MPS or
   CPU and is far below the exclusive two-billion-parameter limit.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-runtime.txt
```

Place the selected checkpoint at `models/v12_pe_core.pt`, then verify its exact
bytes:

```bash
shasum -a 256 -c SELECTED_CHECKPOINT.sha256
```

Run a directory of JPG, PNG, WebP, BMP or TIFF images:

```bash
PYTHON_EXECUTABLE=.venv/bin/python AIGC_DEVICE=auto \
  ./run_v12.sh INPUT_DIRECTORY predictions.json
```

The default batch size is one for maximum Apple MPS compatibility. On CUDA it
can be raised explicitly, for example `AIGC_BATCH_SIZE=8`. The runner applies
EXIF orientation, uses the checkpoint's label-independent JPEG-q96 and
short-side-crop contract, verifies the checkpoint hash before loading, and
writes the output atomically.

Example output:

```json
[
  {"image_path": "INPUT_DIRECTORY/example.jpg", "pred": 0.8731}
]
```

Larger `pred` means stronger AI-generated evidence. No universal threshold is
claimed: ROC AUC evaluates ranking, while a deployment threshold depends on
class balance and the cost of falsely accusing an authentic image.

For the two-to-four-minute recording, open the self-contained evidence brief:

```bash
python3 -m http.server 8765
```

Then visit `http://127.0.0.1:8765/demo/`. Use the arrow keys to move through
the six recording beats. The page contains verified development evidence and
explicit limitations; it does not simulate inference or claim a hidden score.

The rejected equal blend remains available only for reproducible diagnosis:

```bash
PYTHON_EXECUTABLE=.venv/bin/python AIGC_V12_MODE=blend AIGC_DEVICE=cuda \
  ./run_v12.sh INPUT_DIRECTORY predictions.json \
  models/v12_pe_core.pt models/v12_dinov2.pt
```

## Evidence beyond one gate

| Identity-disjoint audit | PE clean | PE transformed | PE organizer-style | Main boundary |
|---|---:|---:|---:|---|
| CIFAKE matched source, 2,000 | 0.9228 | 0.8920 | 0.9074 | low-resolution CIFAR-10/Stable Diffusion only |
| Modern semantic pairs, 288 | 0.9982 | 0.9941 | 0.9962 | Qwen-versus-COCO collection confound |
| Open Images real-source rotation, 288 | 0.9819 | 0.9700 | 0.9759 | reuses the modern fake collection |
| Community Forensics, 593 / 78 model names | 0.9502 | 0.9473 | 0.9487 | four fakes per model; latent-diffusion-heavy |
| Final NTIRE source-coherent gate, 1,024 | 0.9907 | 0.9837 | 0.9872 | generator identities undisclosed |

All 19 workshop settings are applied individually. Gaussian noise sigma 0.10
is the repeatable weakest condition. `ROBUSTNESS_AND_ERROR_ANALYSIS.md` reports
concrete false positives, false negatives and why high AUC does not imply a
safe 0.5 threshold.

## Reproducibility and safety boundaries

- The organizer's demo-only 4,998 COCO val2017 and 8,843 DALL-E Advanced rows
  were used zero times for training, tuning, selection, calibration or
  thresholding.
- Training records zero non-commercial rows under the workshop's controlling
  rule and zero train/evaluation identity overlap.
- Datasets, weights, caches and generated outputs remain outside Git.
- `V12_CHECKPOINT_MANIFEST.json`, `SELECTED_CHECKPOINT.sha256`,
  `THIRD_PARTY_NOTICES.md` and `MODEL_WEIGHTS_LICENSE.md` preserve lineage and
  distribution boundaries.
- `FINAL_LIVE_REHEARSAL_RESULT.json` records the last checkpoint-hash check,
  the managed-shell Metal denial, and the successful unchanged Apple-MPS rerun
  without converting either event into an accuracy claim.
- Historical v6/v9 results are retained as research controls but are
  submission-ineligible because their lineage includes non-commercial data.

Run all source and integrity tests with:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Public checkpoint, clean-source repository and video links must be verified
while logged out before submission. Private preservation is not publication.
