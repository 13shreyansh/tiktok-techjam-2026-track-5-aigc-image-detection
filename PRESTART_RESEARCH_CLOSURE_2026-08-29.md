# Pre-start research closure — 29 August 2026

Research-only preparation was closed before the official start at 12:00 SGT.
This document is a launch decision sheet, not a judged detector design. No
model was selected by experiment, trained, tuned or benchmarked in this phase.

## What is now confirmed

- The task is image-level binary detection: authentic camera image versus a
  fully AI-generated image. AI-edited/mixed images, localization, video and
  audio are outside the stated evaluation scope.
- The submitted inference model must contain fewer than 2 billion parameters.
- Public and properly licensed data and general-purpose public pretrained
  backbones may be used. Repackaging an existing AIGC detector is not allowed.
- The technical score is `0.50 * AUC_clean + 0.50 * AUC_robust`. AI-generated
  is the positive class. The program must emit a continuous confidence, not a
  thresholded label. Robust AUC pools the transformed images.
- Evaluation transformations are applied individually, not as chains. The
  organizer may use a subset of the stated JPEG, blur, resize, noise, colour
  and centre-crop settings.
- The hidden internal evaluation includes older Stable Diffusion images and
  recent diffusion-transformer generators, including generator families not
  represented in training. Generalization is therefore a first-class target.
- The 4,998 COCO plus 8,843 DALL-E Advanced set is demonstration-only. It may
  not be used for training, tuning, model selection or score calibration and
  does not contribute to the final score.
- There is no organizer-supplied runnable baseline. The workshop's “baseline
  pipeline” is explanatory guidance, not an official model or checkpoint.

## Evidence-backed candidates to test after the start

These are candidates, not preselected judged components.

| Candidate | Why it merits an early controlled test | Constraint/risk |
| --- | --- | --- |
| DINOv2 ViT-L/14 | Strong controlled temporal and cross-generator evidence; official code and weights are Apache-2.0; about 304M parameters | Performance still depends on data alignment, preprocessing and adaptation method |
| PE-Core-L | Recent general-purpose vision encoder; official checkpoints are Apache-2.0; approximately 0.32B vision parameters | Less direct Track-like evidence than DINOv2; must verify exact checkpoint and full-model parameter count |
| OpenCLIP model | Useful semantic comparison and MIT-licensed code | Every chosen weight has its own licence; do not infer weight permission from the code licence |
| Small low-level/frequency branch | Could complement global structural evidence when compression has not destroyed it | May learn JPEG/source shortcuts and may add complexity without robust-AUC benefit |

Do not begin with PE-Core-G (approximately 1.88B vision parameters): it leaves
almost no room below the whole-model 2B limit and is operationally heavy. Do
not begin with gated/custom-licensed DINOv3 or with a pretrained AIGC-detector
checkpoint. Large winning ensembles from other competitions are evidence about
the value of data diversity, not admissible or practical templates here.

## Data launch order

1. Use CIFAKE only as a fast pipeline smoke test. Its 32-by-32 images are too
   unlike the target setting to establish generalization or robustness.
2. Inspect SID_Set's pinned metadata and acquire only the necessary public
   shards first. Preserve generator/source groups so validation can hold out
   entire generator families rather than random near-duplicates.
3. Use the WildFake manifest to select relevant generator/source archives;
   do not attempt an indiscriminate 1.29 TB download. Keep provenance and
   source balance explicit so compression or collection source cannot become
   an accidental label.
4. Keep both organizer demo classes physically and logically excluded from
   all training, tuning, selection and calibration paths.

## Experimental questions deliberately deferred until after 12:00

The first experiments should decide these questions with one fixed split and
the official clean/robust AUC calculation:

1. DINOv2-L versus PE-Core-L under the same data, resolution and compute.
2. Frozen head versus parameter-efficient adaptation versus fuller tuning.
3. Whole-image/global feature versus patch-token pooling.
4. Whether a small frequency/residual branch improves unseen-generator robust
   AUC enough to justify its cost.
5. Which legally usable data mixture improves generator-held-out performance
   rather than only same-generator validation.
6. Crop-versus-resize preprocessing and the value of training-time simulated
   redistribution.
7. Whether prediction-consistency training across clean/transformed pairs
   improves the official score.
8. Whether test-time view averaging adds enough robust AUC for its latency.

Each experiment must report clean AUC, pooled robust AUC, the official 50/50
score, each transform/severity, each held-out generator family, false positives,
false negatives, parameter count, runtime and peak memory. Change one major
factor at a time; do not choose from demo-only results.

## Remaining non-experimental unknowns

- Exact hidden image counts, class/generator proportions and exact generator
  names.
- Exact image library/codec used for organizer transformations and which stated
  severities will be sampled.
- Outer JSON container, traversal order, supported extensions, score range and
  failure semantics.
- Evaluation hardware, time/memory limits and whether network access is absent.
- The organizer's exact two exclusions from the 5,000-image COCO source set.
- An available external NVIDIA training environment; the prepared local host is
  Apple Silicon and no ML framework is installed in the neutral environment.

## Pre-start command record

The closing checks used only repository metadata and text:

```text
date '+%Y-%m-%d %H:%M:%S %z'
git status --short --branch
python3 scripts/acquire_resources.py list
rg ... PREPARATION_STATUS.md RESEARCH_LEDGER.md DATASETS.md resources/resource_manifest.json
```

At 11:54 SGT the worktree was clean. The resource listing again reported
CIFAKE, COCO and the DALL-E index as downloaded and verified; SID_Set,
WildFake and the DALL-E archive remained guarded manifest-only large resources.

