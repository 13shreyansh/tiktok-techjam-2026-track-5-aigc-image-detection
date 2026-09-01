# Full self-inspection — 2026-08-31 01:00 SGT

## Verdict

The work is directionally strong but not safe to celebrate. After the
independent low-resolution gate and rejected v10 repair, the best honest rating
is **90/100 for experimental discipline, 70/100 for submission readiness, and
45/100 for confidence in hidden-set excellence**. None is an estimated hidden
score. The
detector has survived several independent and shortcut-controlled tests, and a
75/25 v6/v9 ensemble materially improves the hardest current-generator gate.
However, no public score estimates the organizer's hidden set, the weakest
observed transformed generator/source pair is only 0.6141 AUC, and a new
source-independent low-resolution gate now falls below random under heavy
noise. Release/weight-distribution gates also remain open. The correct posture
is defensive: keep v6 runnable, preserve the ensemble, and treat robust
low-resolution authentic images as a confirmed failure rather than a caveat.

## What the scores actually say

| Gate | Rows | v6 official-style score | 75/25 blend | Most important warning |
|---|---:|---:|---:|---|
| Qwen current-generator prompt holdout | 576 | 0.911785 | **0.938739** | blend worst transformed pair is only **0.614070** |
| full NTIRE audit | 512 | **0.972101** | 0.971970 | blend changes by -0.000131, so its gain is not universal |
| internal multi-source gate | 3,071 | **0.984592** | 0.984355 | blend worst pair declines from 0.742318 to 0.739162 |
| Community Forensics 78-model gate | 624 | 0.982368 | **0.982880** | useful breadth, but dataset terms limit it to evaluation |
| metadata-matched NTIRE diagnosis | 86 | 0.973559 | **0.975436** | small matched subset; semantics and source remain confounded |

The Qwen improvement is the strongest reason to keep the ensemble candidate:
clean AUC rises from 0.933274 to 0.957158, heavy-noise AUC from 0.716303 to
0.755160, and clean worst-pair AUC from 0.643319 to 0.727371. The strongest
reason not to become confident is that pooled worst-pair AUC is still 0.614070
and the full NTIRE/internal aggregates are flat or slightly worse.

## What works

1. The local v6 fallback is complete: exact ignored checkpoint, MPS and CPU
   directory runner, continuous AI-positive probabilities and under-2B guard.
2. The exact ensemble arithmetic is reproduced with zero saved-score and rank
   drift on one T4 when physical batch 64 and CUDA FP16 blending are preserved.
3. The candidate directory-to-JSON runner and shell wrapper both returned zero
   over 576 images. The wrapper took 48.67 seconds including hashing 1.895 GB of
   checkpoints, loading, decoding, inference and JSON writing.
4. The model is not being trusted on one random split. Tests include complete
   generator holdouts, real-source holdouts, every generator-by-real-source
   pair, 19 individual workshop transforms, identical-codec controls, stretch,
   square-patch, aspect-ratio, perceptual overlap and exact-metadata matching.
5. V9 is preserved in a successful private Kaggle version and has been
   recovered to ignored local storage with exact post-download size and hash.

## What does not work or remains unproved

1. **Hidden-set transfer is unknown.** No local gate reproduces unpublished
   generator proportions, real-image sources or acquisition pipeline.
2. **Worst-case robustness is weak.** Heavy Gaussian noise repeatedly falls to
   roughly 0.73-0.80 AUC, and the Qwen pooled worst pair is 0.6141.
3. **The ensemble is CUDA-contract-specific.** CPU/MPS equivalence is not
   established. A different batch/arithmetic policy caused measurable rank
   drift even when aggregate AUC barely moved.
4. **V9 is local but not distributable.** Its recovered bytes and hash match,
   but a private saved version and ignored local file are not a public release.
5. **The corpus visibly leaks labels through metadata.** A pixel-free classifier
   using dimensions, file size, aspect ratio, format, mode and suffix reached
   0.8381 AUC on training and 0.9508 AUC on the selection set. Restricting to
   square images reduced it to 0.5741, but restricting to PNG images still gave
   0.8514. The image model does not receive filenames or file bytes, and the
   identical-codec/geometry controls remained strong, so this does not prove
   the model uses those cues. It proves the data gives it shortcut opportunity.
6. **Weight distribution is unresolved.** The code can be MIT, but several
   training-image sources have non-commercial, share-alike or source-specific
   terms. Competition use and checkpoint redistribution are different claims.
7. **Calibration is unknown.** AUC measures ranking. The illustrative 0.5
   threshold produced five false positives in a 128-image local audit and is
   not claimed to be correct for the hidden class balance.
8. **Runtime portability is unknown.** A T4 works; organizer hardware, memory,
   dependency availability and time limits remain unpublished.
9. **Naive checkpoint compression is not exact.** An FP16-storage v9 export
   halved its bytes but shifted ensemble scores by up to 0.00390625 and ranks
   by four positions on the frozen Qwen gate. It was rejected rather than
   silently replacing the verified FP32 artifact.
10. **Content dependence is materially unresolved.** On the selected v6's
    balanced 18-generator by 16-prompt Qwen grid, prompt identity explains
    52.75% of fake-score variance while generator identity explains 14.75%.
    The weakest repeated prompt AUC is 0.7907. This is evidence of strong
    content sensitivity, not proof of a semantic shortcut; the inspected gate
    must not be trained on after revealing this failure.

## Mistakes caught and corrected

- Early SID/CIFAKE success collapsed when the real-image source changed. The
  response was multi-source pairing and worst-pair gates, not a better-looking
  random split.
- Adding all frontier data with group-balanced oversampling made 576 images
  dominate training; v8 regressed. V9 capped that block, yet was still rejected
  as a standalone model under frozen floors.
- Horizontal-flip test-time averaging looked helpful on a small screen but
  regressed independent noise conditions and was rejected.
- A saved prediction comparison mixed GPU-FP16 and CPU-FP32 blend arithmetic.
  The resume signature did not encode that policy. Exact arithmetic was then
  frozen and reproduced instead of relaxing tolerances.
- The first end-to-end ensemble invocation failed on null checkpoint codec
  metadata. The packaging bug was fixed, then the complete command and shell
  wrapper were rerun successfully.
- High pooled scores initially obscured weak real-source/generator pairs. The
  reporting now makes the 0.6141 worst pair impossible to hide behind 0.94.
- The tempting FP16-storage export cut v9 to half size but failed zero-drift
  equivalence. The artifact was rejected instead of relaxing the gate for
  packaging convenience.
- During a later condition audit, the detailed evaluation path was initially
  described as if it belonged to selected PE-Core v6. Its embedded `model`
  field and checkpoint path proved that it was actually the rejected DINOv2-L
  control. The claim was corrected before any model change. The audit tool now
  emits model identity beside every score; the DINO subgroup values must never
  be attributed to PE-Core.
- The release manifest initially nested the rejected compact-v9 experiment
  under v6. The checkpoint hash and measurements were correct, but the parent
  artifact was wrong. It was moved under v9 and a test now prohibits the same
  lineage error.

## Remaining failure matrix

| Possible complete failure | Current defence | Residual risk | Next falsification |
|---|---|---|---|
| generator absent from training | 78-model and NTIRE audits; 18-current-generator Qwen gate | high because unknown families remain possible | generator-family leave-one-block-out analysis on existing data |
| detector learns dataset identity | matched codecs/shapes, source-paired AUCs, exact-metadata NTIRE slice | **high**; metadata-only selection AUC is 0.9508 and semantics/acquisition source remain | source-only predictability, matched-source and label-swap controls |
| detector learns subject matter | animal, face, scene, web and low-resolution reals; diverse fake families | medium; classes still do not share every subject/prompt | matched-content or prompt-paired real/fake gate where legally available |
| compression removes fake evidence | all organizer JPEG/blur/resize/crop/color transforms | medium; exact hidden codec differs | preserve official individual settings and report every condition, not chained guesses |
| severe noise destroys both models | noise augmentation and two-model blend | high; it is the repeated weakest transform | diagnose whether training noise or low-frequency evidence can help without clean/source regression |
| threshold creates many false accusations | continuous probability output and AUC-first evaluation | high until hidden balance/cost is known | keep threshold separate; do not tune on demo-only data |
| checkpoint cannot run for judges | v6 CPU/MPS fallback; ensemble one-T4 wrapper; exact v9 local | medium-high for ensemble | document exact CUDA dependencies; test a fresh environment; retain v6 fallback |
| artifact cannot legally be released | provenance and third-party notice ledger | high | make a conservative weight-release decision and bundle exact notices |

## Second unseen-prompt paranoia gate

The most important new check asked whether the candidate's Qwen improvement
survives prompt IDs that were not used by the earlier diagnosis or v9 training
partition. On 576 checksum-frozen rows, exact one-T4 FP16 inference improved
clean AUC from 0.932412 for v6 to 0.954638 for the 75/25 blend. The worst
generator improved by 0.045139, the worst generator/real-source pair by
0.081358, and the weakest prompt AUC by 0.103395. This makes the candidate
direction more credible; it does not make it safe to assume hidden transfer.

The paranoid finding is that prompt identity still explains 58.5940% of the
blend's fake-score variance, versus 11.6882% for generator identity. Realistic
text-heavy/social-design scenes are repeated weaknesses. The gate is now
consumed and excluded from training or content-directed model selection. Any
new intervention needs different lawful data and a third disjoint frozen gate.

The first Kaggle invocation failed before download or scoring because the
shape-control helper module was absent. The exact existing helper was uploaded
and the unchanged evaluator then returned zero. The report explicitly records
its 88.58-second download/load/decode/inference time, 4,275,770,368-byte CUDA
peak, T4 device, data inventories and both checkpoint hashes. A separate MPS
v6 result differs by 0.000241 AUC; we treat that as device/arithmetic drift and
do not splice the two environments into one comparison.

## Authentic-source dependence and label-boundary audit

Identical JPEG-q96 and stretch preprocessing did not make authentic scores
source-independent. Across the 288 frozen reals used by both Qwen gates, named
real source explained **33.9694%** of selected-v6 score variance. A deterministic
10,000-permutation test gave the minimum plus-one-corrected p-value, 0.000100.
CIFAKE reals averaged 0.356201 AI probability and had 25.86% above the
illustrative 0.5 threshold; LSUN-Church reals averaged 0.033626 with none above
0.5. The second-gate AUC against CIFAKE reals was only 0.825072.

Visual inspection makes the risk more precise. The hardest CIFAKE reals were
tiny 32-by-32 animal photographs enlarged to the input size. Two high-scoring
LAION reals were a watermarked anatomical stock render and framed abstract
artwork, neither an ordinary camera photograph. A high-scoring SID example was
a low-contrast parking lot with glare and clouds. The model is vulnerable both
to poor photographic evidence and to the broader boundary between human-made
digital art/rendering and AI generation. This is not proof of the causal model
feature, and the repeated Qwen gates reuse the same authentic rows, so it is
one real-side audit rather than two independent replications.

The exact saved clean/noise prediction rows make the failure sharper. Under
Gaussian noise sigma 0.10, CIFAKE-CIFAR10 authentic images receive a mean blend
AI score around 0.83 in all three frozen gates. That is higher than the fake-
pool mean on the internal, Community Forensics and Qwen gates, producing source-
restricted AUCs of 0.5293, 0.4180 and 0.3864. The overall noise AUCs of 0.8704,
0.7966 and 0.7552 therefore hide a repeated low-resolution-real ranking
inversion. This is now the clearest current failure—not a theoretical caveat.
The diagnosed rows are consumed; repairing it requires new disjoint data and a
new frozen gate, not retuning against these results.

### Independent replication changes the verdict

The CIFAKE explanation is no longer sufficient. A class-balanced, hash-frozen
set of 1,000 CIFAR-100 test photographs produced clean AUC 0.850858 for v6 and
0.891642 for the blend against the unchanged Qwen fake pool. Under the exact
workshop sigma-0.10 condition, v6 dropped to 0.449342 and the blend to 0.473915.
The noisy authentic mean exceeded the fake mean, and the blend's bootstrap
interval spans roughly 0.427 to 0.520. This confirms the feared failure on a
second real source: strong noise on genuinely low-resolution photographs can
move them further into the AI-positive region than current-generator fakes.

This does not prove every hidden low-resolution image will fail. CIFAR-100 is
still a small academic image distribution and the fake side reuses a consumed
Qwen pool. It does prove that the earlier pooled 0.75-0.87 noise AUCs were not
a safe summary of real-world robustness. The relevant hidden-set risk is now
high: if the organizer includes many low-resolution authentic images under
strong noise, the current candidate can rank them catastrophically. Neither a
new threshold nor calibration can repair reversed ranking.

The most likely mechanism is an interaction among low native resolution,
strong noise and the frozen PE representation. This remains a hypothesis, not
a causal finding. V6 gives CIFAKE only one of eight authentic-source sampling
shares, and a sigma-0.10 draw is rare under its uniform single-transform
policy. The global v7 noise weighting slightly regressed every frozen headline
metric, so simply adding more noise everywhere is already falsified. A valid
repair must give matched low-resolution real and fake examples enough exposure
without making noise label-specific, and it needs a fresh, disjoint promotion
gate. The consumed CIFAR-100 rows cannot participate in that decision.

## Generator-coverage claim correction

The v6 training manifest contains 12 exact generator names across nine raw
family labels. V9 contains 30 names across ten labels because it adds all 18
Qwen names. This distinction changes the meaning of the Qwen gates. For v6,
all 18 Qwen names and the `frontier-2026-image-generation` label are absent
from training; its two roughly 0.93 clean AUCs are exact-name transfer evidence.
For v9, every Qwen name is present in training. The blend's roughly 0.95 clean
AUCs therefore show prompt/pixel transfer within known names, not unseen-
generator transfer.

Community Forensics contributes 78 exact model names absent from both training
manifests and the blend scores 0.982880 on its official-style gate. However,
its raw label `LatDiff` is semantically related to latent diffusion in training;
different spelling is not a new technology. NTIRE's label is literally
`undisclosed`, so its 0.971970 result cannot prove family novelty. The truthful
position is: v6 has stronger current-generator holdout evidence; the blend has
better prompt/content adaptation for those current names; neither guarantees a
future unseen family.

## Why the path is still rational

The workshop emphasized generalization, robustness under individual realistic
post-processing, practical completion, public/licensed data and a model under
2B parameters. The current path directly measures those properties. PE-Core-L
is not selected because it is fashionable; it beat the DINO control on the
frozen source/generator gates. The ensemble is not selected because 0.94 looks
good; it is retained because a predeclared 75/25 weight improved the hardest
current-generator gate while staying inside frozen non-regression limits on
three independent audits. The v6 fallback remains protected because those
gains are not universal and v6 is operationally simpler.

## Immediate priorities

1. Keep v10 rejected and retain the verified v6/v9 candidate plus v6 fallback;
   do not rescue v10 with the consumed gate.
2. Diagnose the frozen v10 predictions without model selection, then choose a
   genuinely different mechanism and freeze another disjoint source/prompt gate
   before any later candidate exists.
3. Resolve trained-weight distribution and exact attribution without confusing
   repository-code licensing with dataset/model rights.
4. Run the remaining highest-value falsification: leave-family-out and
   source-predictability analyses using already-acquired lawful data.
5. Only adopt a new training change if it beats frozen weakest-pair and
   independent-condition gates; do not chase aggregate AUC.
6. Keep `run.sh` plus v6 unchanged until the ensemble is truly distributable
   and portable. Never use the organizer's demo-only data for any learning,
   tuning, calibration or selection.

## Matched low-resolution repair was not sufficient

The predeclared v10 repair completed exactly once after byte-level removal of
all v6 train and frozen-gate overlaps. It added 5,000 CIFAKE real and 5,000
CIFAKE fake train-only images, kept the backbone frozen, updated only 1,025
classifier parameters for one epoch, and read zero organizer demo rows. The
fixed 75/25 v6/v10 blend first passed all four existing clean non-regression
gates. This justified opening the untouched gate; it did not promote v10.

On 1,000 new CIFAR-100 authentic rows and 144 new Qwen fake rows, clean AUC
improved from 0.925257 to 0.937642. Under Gaussian noise sigma 0.10, AUC
improved from 0.391219 to 0.462979, exceeding the required +0.05 delta but
remaining below chance and below the frozen 0.60 floor. Noisy authentic mean
AI probability remained higher than fake mean, and 91.4% of noisy authentic
images remained at or above the illustrative 0.5 threshold. V10 therefore
failed two frozen checks and is rejected. The gate is consumed and cannot be
used to rescue the experiment through another blend, threshold, calibration or
training change. Exact compact evidence is in `V10_LOWRES_REPAIR_RESULT.json`.

This falsifies the narrow hypothesis that matched CIFAKE exposure plus a
linear-head update is enough to repair the low-resolution/noise inversion. It
does not prove the PE backbone is irreparable, but any next intervention needs
a genuinely new mechanism and a new unconsumed gate. The existing v6/v9
candidate and simpler v6 runner remain the fallbacks.

A diagnosis performed only after rejection helps localize the failure without
changing that decision. V10 alone reached 0.958340 clean AUC and 0.687316 noise
AUC on the consumed gate; its noisy authentic mean (0.662199) was below its
fake mean (0.774356), so the earlier mean inversion disappeared. The fixed
blend reduced that noise AUC to 0.462979 because v6 dominated its ranking. This
means the training intervention was not useless, but it does **not** authorize
choosing a new blend after seeing the gate. It suggests a future candidate may
need a genuinely different routing or representation mechanism, frozen before
another disjoint gate is opened.

## Quality-route paranoia gate rejected the added complexity

The label-blind v11 router looked credible on three already-consumed screens:
it preserved clean AUC and improved the repeated sigma-0.10 weakness. The
untouched 1,024-image NTIRE gate separated clean and noisy inputs perfectly at
the frozen 0.055 route threshold. Clean route rate was 0 and noise route rate
was 1.0. That did not make the system valuable enough. Clean AUC remained
0.987041, but noise AUC increased only from 0.789886 to 0.805782, a 0.015896
gain against the predeclared 0.05 minimum. Five of six checks passed and the
all-checks-required decision rejected v11.

The paranoid interpretation is important: a mechanistically sensible router,
perfect transform identification and three favorable screens still did not
replicate a large benefit on untouched images. The v10 head's benefit is
source-dependent. Adding routing would increase packaging and explanation
surface without enough independent evidence. The gate is consumed, and its
scores cannot justify a new threshold or route rule. The best-supported path
therefore remains the exact v6/v9 CUDA ensemble, with v6 retained as the
simpler operational fallback. Heavy-noise performance remains a known residual
risk rather than a solved claim.

## Truth boundary

Nothing here proves a hidden score, a win, legal redistribution, public
availability or organizer-hardware execution. It proves only the exact commands
and artifacts recorded in this repository. Every attractive result remains a
hypothesis about transfer until the organizer evaluates the final submission.

## Critical compliance reversal after rereading the workshop

The most important self-correction is legal, not statistical. The workshop
explicitly says that datasets marked non-commercial cannot be used. The v6/v9
lineage directly used AFHQ-v2, FFHQ and CelebA-HQ; consequently neither artifact
is an eligible submission, regardless of its strong scores or any licence later
placed on the output. Continuing to optimize those weights would have been a
complete-failure route. They now serve only as historical controls.

The replacement v12 lineage is narrower in what can honestly be claimed but
stronger in validity: commercial-use-permitted records only, no demo data,
content-hash separation, modern and legacy fake families, and real images from
three sources. Its raw manifests exposed a metadata-only AUC of 0.998360, so
training them unchanged would have recreated the delusional high-score risk.
The same label-blind 336-square JPEG-q96 operation on both classes reduced the
frozen metadata probe to 0.513093, with shape-only and PNG-only at chance.

This is progress, not success. V12 is still vulnerable to content/source
shortcuts, one-epoch linear-head underfitting, an imperfect legal interpretation
of aggregate datasets, generator-name imbalance, centre-crop information loss,
heavy-noise collapse and failure on a truly new hidden generator. It must pass
the predeclared clean group floors, every individual workshop transformation,
real-source and generator-pair analysis, codec/geometry causal controls and an
independent licence-clean external gate before the state can change from
`no eligible selected model`.
