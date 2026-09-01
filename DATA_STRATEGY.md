# Hidden-set data strategy

This is the active post-start data and validation policy. It translates the
organizer workshop into experiment rules; it does not claim knowledge of the
unpublished hidden set.

## Target inferred from organizer evidence

The hidden evaluation is expected to mix generator generations rather than
repeat one public dataset. The workshop explicitly mentioned older Stable
Diffusion material, recent diffusion-transformer generators, unseen generator
families, and individually applied redistribution transforms. It also warned
that data alignment and distribution breadth can matter more than a more
complicated backbone.

Accordingly, the target is not "best on SID_Set" or "best on WildFake." It is
a ranking function that transfers across generator family, image subject,
real-image source, file format, resolution and redistribution history.

## Non-negotiable controls

1. The organizer's 4,998 COCO `val2017` plus 8,843 DALL-E Advanced demo set is
   physically and logically excluded from training, tuning, model selection,
   calibration and final claims.
2. A submission candidate may not train on any dataset marked non-commercial,
   per the workshop Q&A. Historical v6/v9 experiments violate this stricter
   gate through AFHQ-v2, FFHQ and CelebA-HQ and are retained only as controls.
3. AI-generated is label 1 and every experiment emits a continuous score.
4. Entire generator families are held out. Random image splits from one
   generator are not accepted as generalization evidence.
5. Real-image sources are controlled separately. At least one evaluation keeps
   the real source fixed while changing only the fake generator, and a later
   evaluation must also hold out a real source.
6. Label-correlated shortcuts are audited: file extension, JPEG history,
   dimensions, aspect ratio and source collection must be matched or stratified.
7. Every selected candidate is scored on clean images and on each organizer
   transformation. The official selection score is
   `0.50 * AUC_clean + 0.50 * AUC_pooled_transformed`.
8. Workshop transformations are applied one at a time. Training augmentation
   may randomly choose at most one listed transform for an image; chained
   distortion recipes are not the target distribution and cannot drive model
   selection.

## Data coverage matrix

The training curriculum should cover genuinely different families rather than
many near-duplicates from one model:

| Axis | Required coverage | Validation policy |
| --- | --- | --- |
| Real content | natural scenes, people/faces, objects/products, artwork/graphics and web-distributed photographs from multiple licensed sources | hold out at least one complete real source |
| Older latent diffusion | Stable Diffusion 1.x/2.x-style generators | hold out at least one named generator/version |
| Classical diffusion | DDPM/DDIM-style sources | DDPM-train/DDIM-test is the first completed control, not the final mixture |
| Modern diffusion transformer | SD3/Flux/PixArt-style public or properly licensed material | keep one recent family entirely unseen during model selection |
| GAN and non-diffusion | StyleGAN/ProGAN/BigGAN and a token/autoregressive family where legally available | report family-level AUC rather than pooling it away |
| Redistribution | the exact JPEG, blur, resize, noise, colour and crop settings from the workshop | evaluate each severity separately and pooled |

Dataset names are not substitutes for this matrix. Every acquired subset needs
generator, real-source, licence, immutable revision/checksum and preprocessing
metadata before it can enter a run.

## Curriculum and experiment sequence

1. **Integrity controls:** prove the pipeline with source-aligned,
   generator-held-out splits. The completed DDPM-to-DDIM experiment fills this
   role.
2. **Breadth baseline:** train one backbone on a balanced mixture spanning
   several real sources and at least old diffusion, GAN and non-diffusion
   fakes. Equalize obvious format and size cues across labels.
3. **Recent-generator test:** add legally auditable modern diffusion-transformer
   data to training while keeping a separate recent family fully held out.
4. **Backbone/adaptation test:** compare DINOv2-L, PE-Core-L and one licensed
   CLIP-family encoder under the identical mixture and split. Compare frozen
   head with parameter-efficient or full adaptation only after the data split
   is fixed.
5. **Robustness test:** compare standard versus realistic transformation
   simulation, crop-versus-resize preprocessing and clean/transformed
   consistency training. Do not select on clean AUC alone.
6. **Complementary-cue test:** add a small residual/frequency or patch branch
   only if it improves held-out-generator robust AUC and survives JPEG/blur.
7. **Final stress matrix:** generator-family AUC, real-source AUC, every
   transform/severity, false positives, false negatives, latency, peak memory
   and complete parameter count.

## Selection rule

A candidate advances only when it improves the fixed generator-held-out
official-style score without creating a severe failure on a real-source group
or transformation. Same-dataset gains, training loss and attractive clean
examples are diagnostic evidence only. The final choice must remain below two
billion total inference parameters and must run through `run.sh` without
network access.

## Active failure-mode register

Every experiment must address or expose at least one of these risks. None may be
declared solved from a pooled score alone.

| Failure mode | How it is detected | Required response |
| --- | --- | --- |
| Subject shortcut (for example, animals versus people) | report AUC/TNR by real content source and inspect false positives by content type | add balanced authentic and synthetic coverage for the missing content; keep a disjoint content source |
| Generator memorization | hold out an entire named generator and, for modern DiTs, an entire family | broaden generator families; never use a random image split as the only evidence |
| Dataset/source fingerprint | pair real and fake classes from CIFAKE and SID; compute every generator-by-real-source AUC | reject candidates with a collapsed or reversed pair even when pooled AUC is high |
| Resolution/file-format shortcut | audit dimensions, aspect ratios, extensions and encodings by label | align both labels or stratify evaluation; preserve the audit with the run |
| Redistribution fragility | evaluate every workshop severity individually and compute one pooled transformed AUC | compare augmentation policies under the identical fixed split |
| High-frequency over-reliance | compare clean, JPEG, blur and downscale results | add complementary semantic/patch cues only when they improve the weakest robust group |
| Semantic over-reliance | test visually convincing modern and held-out generators plus varied real content | add residual/frequency evidence only after a simple representation baseline is fixed |
| Watermark/metadata reliance | decode pixels consistently and test sources without reliable provenance signals | do not use metadata or generator-specific watermark checks as the detector |
| Threshold illusion | select with ROC AUC from continuous AI-positive scores | do not tune a fixed threshold on the demo-only or hidden distribution |
| Demo leakage | path-level exclusions and manifest audits for COCO `val2017` and DALL-E Advanced | abort the run if any prohibited path or content inventory is present |
| Licence failure | immutable source, revision and licence record for every admitted group | keep questionable data experimental and replace it before the final candidate if permission is not defensible |
| Compute-route failure | record successful device, runtime and peak resource observations | preserve checkpoints; compare architectures on compatible GPUs without changing the data split |

### Current gate state

| Gate | State | Evidence / next rule |
| --- | --- | --- |
| Broad subject/content independence | **failed for v4 and v5** | v4 failed unseen LAION at 0.6502; v5 repaired disjoint LAION to 0.8673 but failed unseen Church at 0.6565 |
| Entire generator holdout | **provisionally passed for v4 clean** | complete Imagen and PixArt holdouts score 0.8693 and 0.9424; robust gate remains pending |
| Entire real-source holdout | **failed for v4 and v5** | v4 worst fake/LAION pair was 0.4471; v5 worst Imagen/Church pair is 0.3484 |
| Paired same-source control | **partially passed** | source repair raised CIFAKE to 0.8004 and SID to 0.7057, but did not transfer broadly |
| File-format/resolution alignment | **dataset failed; both candidates pass causal controls** | a pixel-free width/height/format/byte probe reaches 0.9508 evaluation AUC; DINO changes only 0.0063 after identical JPEG and 0.0087 after JPEG plus stretch; PE changes only 0.0005 after identical JPEG and at most 0.0052 under three identity-derived geometry controls |
| Individual redistribution robustness | **completed for v6 DINO and PE-Core on both frozen gates** | DINO scores 0.9214/0.9031/0.9122 on v6 and 0.8789/0.8629/0.8709 externally; PE scores 0.9911/0.9780/0.9845 on v6 and 0.9926/0.9723/0.9825 externally |
| Demo-data exclusion | **passing** | forbidden paths are hard-blocked and package manifests exclude COCO/DALL-E demo material |
| Licence/provenance | **failed for v6/v9** | workshop Q&A prohibits non-commercial datasets; AFHQ-v2, FFHQ and CelebA-HQ make those checkpoints ineligible, so a new permissive-lineage candidate is required |
| Under-2B and runtime contract | **historically passed, must be repeated** | v6 proved the PE-Core-L runner fits, but the compliant retrain needs its own parameter/hash/runtime evidence |

“Failed” means the candidate is rejected, not that the experiment was wasted.
“Active” means the gate is fixed before training and cannot be changed to rescue
an unattractive score.

The first controlled response to the codec risk is now implemented but not yet
selected: a `jpeg_q96` preprocessing ablation normalizes every label through
the same codec. The second is a Stay-Positive final head that can only add
evidence for the fake class. Both are compared against the unchanged v5
manifests; neither is accepted from paper results or training loss alone.

V6 makes the shape/codec danger measurable rather than solved. A logistic
regression that sees no pixels and only dimensions, byte size, format, mode and
suffix reaches 0.9508 AUC on the cleaned selection population. DINO's clean
AUC remains 0.9432 on square-only rows, but falls to 0.8288 on JPEG-only rows.
Those correlated subgroups cannot establish causality. Candidate promotion now
requires (1) every image re-encoded identically, (2) a label-independent crop
or aspect assignment derived only from immutable image identity, and (3)
generator-by-real-source reporting under those controls. A high unnormalized
clean AUC is not accepted as evidence of hidden-set generalization.

The audit-only Community Forensics gate is also deliberately hostile to the
same shortcuts: a pixel-free metadata classifier separates it perfectly. Yet
DINOv2-L changes by only 0.0048 after identical JPEG q96 and 0.0081 after JPEG
q96 plus full-frame stretch. Its completed clean/pooled-transformed/50-50
scores are 0.8789/0.8629/0.8709 across 78 named latent-diffusion variants;
heavy noise sigma 0.10 is the weakest condition at 0.7767. This supports real
cross-model transfer while identifying noise sensitivity, but four images per
named model and a latent-diffusion-heavy source are insufficient to estimate
hidden performance. PE-Core was required to pass this same frozen gate before
selection and subsequently did so.

PE-Core v6 passed that full frozen transform matrix at
0.9926/0.9723/0.9825 clean/pooled-transformed/50-50. Its worst individual
condition is again heavy Gaussian noise, at 0.7975 AUC, while every four-image
named-model alarm remains above 0.90 against the common real pool. This makes
PE-Core v6 the selected detector after its external codec/geometry controls
also passed. A controlled v7 head-training run exposed the frozen split to
more noise, but its 0.983306 internal 50/50 score and 0.863060 heavy-noise AUC
both remained below v6; the predeclared gate rejected v7.

The external causal controls are now complete as well. After identical JPEG
q96, PE-Core scores 0.9883 with full-frame stretch, 0.9950 with a deterministic
75% square patch and 0.9934 with deterministic forced aspect ratios. The raw
gate is metadata-separable, but the image detector does not collapse when that
codec/geometry opportunity is neutralized. V6 therefore passes every current
shortcut veto and is selected. Its hash-matched full checkpoint and locally
verified FP16 inference export are preserved outside Git; selection does not
imply certainty about unpublished generators or real-image sources.

The subject/content shortcut is **not solved**. Adding FFHQ repaired one proven
human-face failure, and paired CIFAKE/SID controls repaired a separate source
failure, but neither result demonstrates content invariance. Before new real
domains enter training, the current checkpoint is therefore tested on 1,024
unseen CelebA-HQ portraits, 1,024 unseen LSUN Church scenes and 1,024 unseen
LAION-5B web images. These are paired against nine disjoint fake groups. A
future candidate is vetoed if a pooled gain hides a collapsed real-source or
generator-by-real-source AUC. If these real domains are later used for
training, at least one complete real source remains untouched for selection.

## Current evidence and next acquisition

- SID-only DINOv2-L scored about 0.963 on a same-source slice but reversed
  below random on DDIM-versus-ImageNet. That candidate is rejected as a final
  model because it learned non-transferable source cues.
- ResNet18 trained on 2,048 DDPM fakes plus 2,048 ImageNet reals reached 0.9516
  clean AUC on 512 unseen DDIM fakes plus 512 disjoint ImageNet reals. This
  validates the held-out-generator control, but it covers only one real source
  and closely related diffusion processes.
- Frozen DINOv2-L on that identical split reached 0.9866 clean AUC, 0.9821
  pooled transformed AUC and **0.9843** on the workshop 50/50 score. It is the
  current cross-generator robustness control, not yet the final detector.
- On the modern RR-special-versus-ImageNet screening split, frozen DINOv2-L
  reached 0.9972 clean AUC and PE-Core-L reached 1.0000. This only ranks
  representations provisionally: RR special fakes and ImageNet reals remain
  distinguishable sources, so perfect clean separation may include a shortcut.
- RRDataset reals are barred after perceptual audit confirmed re-encoded COCO
  validation-source photographs. RR `normal_*` fakes are also barred because
  that portion includes DALL-E 3. The admitted modern subset is special-scenario
  SD3.5-Large/Flux.1 fakes only.
- The first licence-audited family mixture is now fixed. Its 8,160 training
  rows balance 4,080 reals (2,040 200-pixel ImageNet JPEGs plus 2,040 official
  512-pixel AFHQ-v2 PNGs) against 4,080
  fakes split equally across DDPM, SDv1.5-DPMSolver, StyleGAN, BigGAN, StarGAN
  and the RR special SD3.5/Flux pool. None of the forbidden demo material is
  present.
- Model selection uses 2,048 fakes from four completely held-out generators
  (DDIM, DF-GAN, GALIP and GigaGAN), first with disjoint ImageNet and official
  AFHQ-v2-test reals
  and then with 2,048 held-out FFHQ reals. The second view is deliberately
  harder: it exposes detectors that separate dataset sources instead of real
  from generated images.
- The container shortcut audit forced this real-source replacement: the first
  draft had all real images as 200-pixel JPEGs, versus a fake class dominated
  by larger PNGs. The corrected training labels both contain JPEG and PNG and
  overlap at 200 and 512 pixels. Per-source evaluation still remains mandatory
  because source alignment reduces but cannot prove the absence of shortcuts.
- The first DINOv2-L run on this corrected two-real-source mixture still failed
  a completely held-out FFHQ test (0.4299 AUC and only 0.0356 true-negative
  rate at threshold 0.5). Synthetic human faces were present in training while
  genuine human faces were not, so that candidate was rejected. Mixture v2
  now divides its 4,080 reals equally among ImageNet, AFHQ-v2 and the official
  FFHQ 0-59,999 training partition; FFHQ IDs 60,000-69,999 remain eligible
  only for disjoint validation.
- Mixture v2 repaired the FFHQ failure (0.9888 clean AUC) but failed the stricter
  paired-source gate: CIFAKE fake versus CIFAKE real was 0.5599 AUC, SID fake
  versus SID real was 0.5164, and the weakest cross-source pair was 0.2570.
  This candidate is rejected because it still exploits acquisition, resolution
  or dataset cues. The active source-repair split adds balanced CIFAKE-train and
  SID-train real/fake controls, while CIFAKE-test and SID-validation remain
  untouched. Future selection uses the weakest generator/real-source pair as
  a veto in addition to pooled AUC.
- The paired-source repair completed successfully: pooled external clean AUC
  rose to 0.8932, matched CIFAKE to 0.8004, matched SID to 0.7057 and the worst
  generator/real-source pair to 0.7057. This advances the data policy, not the
  final model. SID remains the weakest source and the candidate must still pass
  named modern-DiT and individual-transform gates.
- The licence-audited DiTFake acquisition contributes only synthetic images
  from FLUX.1-schnell, Stable Diffusion 3 Medium and PixArt-Sigma. Its bundled
  COCO real directory is excluded. The next modern-family experiment will train
  with FLUX and SD3 while holding all PixArt images out, so the recent-generator
  test is not a random within-generator split.
- Before that training, the source-repaired DINOv2-L checkpoint reached 0.8289
  clean AUC across all three modern generators and the fixed external reals.
  FLUX was the weakest generator at 0.7862, SID_Set was the weakest real source
  at 0.7243, and FLUX-versus-SID was the weakest pair at 0.6639. These values are
  the fixed pre-training reference: v3 must improve modern transfer without
  collapsing the source-paired controls. The run used aspect-ratio-preserving
  short-side resize plus centre crop, so older stretch-mode scores are not used
  as direct crop-mode comparators.
- The new fully held-out portrait/scene/web-content gate then rejected that
  checkpoint: overall AUC was 0.6147, the weakest real-source AUC was 0.5939,
  and FLUX versus LSUN Church reversed to 0.4560. The prior FFHQ and paired
  CIFAKE/SID repairs therefore solved specific observed gaps, not the general
  content shortcut. New authentic and synthetic breadth must be introduced
  together, with complete content sources left out for selection.
- A second balanced gate using only previously unseen ADM, Imagen and VQDM
  fakes against those three real domains scored 0.5236 overall, with ADM at
  0.4656 and a 0.4521 worst pair. The v4 experiment therefore adds ADM/VQDM and
  CelebA-HQ/Church symmetrically, holds Imagen and PixArt out completely, and
  keeps the broad LAION source entirely unseen. Its 16,910-row training and
  12,326-row selection manifests have zero path and SHA-256 overlap.
- V6 is the next prepared mixture after the controlled v5 comparisons. It
  exposes training to both LAION and Church shard A while reserving
  byte-disjoint shard B from both sources. Equal numbers of unused FLUX and
  Stable Diffusion 3 images preserve class balance. Imagen and PixArt remain
  complete generator holdouts. This design measures transfer to unseen files
  from known real domains and entirely unseen fake generators; it does not by
  itself prove transfer to every possible real-image source.
