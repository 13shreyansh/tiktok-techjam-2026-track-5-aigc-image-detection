# Deadline adversarial plan

## 2026-08-31 compact-v9 packaging ablation

The exact v9 checkpoint is almost entirely FP32 model tensors. A single
packaging-only ablation may convert floating state-dictionary tensors to FP16
storage and compare the exact frozen 576-image clean Qwen predictions under the
unchanged one-T4, physical-batch-64, CUDA-FP16 blend contract. No weight,
training row, transform, ensemble coefficient or threshold may change. Any
nonzero score, AUC or rank drift rejects exact equivalence. Even zero drift only
admits the compact checkpoint to the unchanged full promotion matrix; it does
not select, publish or distribute it.

Last reconciled: 2026-08-30 SGT

## Fixed boundary

- Official submission window: 2026-08-29 12:00 SGT through 2026-09-01
  12:00 SGT. Work continues until that deadline unless the user explicitly
  asks to stop.
- The hash-verified v6 PE-Core-L checkpoint remains the fallback. No experiment
  replaces it unless it passes the frozen promotion gates below.
- The demo-only 4,998 COCO and 8,843 DALL-E Advanced images are excluded from
  training, tuning, selection and calibration.
- The final model must contain fewer than 2 billion parameters. No push,
  publication or submission is authorized by this plan.

## Organizer-grounded evaluator model

Confirmed from the workshop transcript and preserved requirements:

- The task is image-level, pure-real versus pure-generated detection. AI edits,
  video, audio, watermarks and provenance markers are out of scope.
- AI is label 1 and the evaluator consumes a continuous confidence. ROC AUC is
  used, so a 0.5 operating threshold is not the technical target.
- Hidden images include older Stable Diffusion and a recent, highly realistic
  diffusion-transformer family. The organizers said the hidden data definitely
  differs from the suggested/demo resources.
- The listed transformations are applied individually, not chained, and the
  final evaluator selects a subset no more aggressive than the workshop table.
  Transformed images are pooled for the augmented AUC.
- The engineering guidance emphasizes content/source diversity, modern and
  unseen generators, high- and low-level evidence, robustness augmentation,
  continuous scores, and explicit failure analysis.

## Frozen promotion gates

An experiment may replace v6 only if all applicable checks succeed:

1. Internal 3,071-image clean plus 19-condition gate: no material regression in
   the official 50/50 score, clean AUC, heavy-noise AUC, or worst named
   generator/real-source pair.
2. Independent Community Forensics 78-model gate: no material regression in
   clean, pooled robust, or worst-family behavior.
3. Codec/shape causal controls: no new dependence on PNG/JPEG, aspect ratio,
   stretching, square patches, or source resolution.
4. Leakage/provenance audit: no demo-only paths, perceptual overlap, test-label
   training, proprietary/unlicensed input, or unverifiable generator labels.
5. Runnable artifact: finite continuous predictions, fewer than 2B parameters,
   deterministic documented preprocessing, passing tests, and recorded hashes.

The numerical comparison is against the already recorded v6 results, not a
newly re-sampled or repeatedly tuned validation set.

## Active risk register

| Risk | Current evidence | Predeclared next test | Resolution standard |
|---|---|---|---|
| Generator/source shortcut | Strong held-out scores, but most local corpora retain source identity | Add legally auditable recent-generator tests and report every named generator against every real source | Broad named-family transfer without one source driving the pooled score |
| Shape/codec shortcut | JPEG-q96, stretch, forced-aspect and square-patch controls passed | Quantify joint class/source distributions for size, aspect, suffix and codec; add matched subsets where feasible | Metadata-only predictors near chance and model AUC retained after causal normalization |
| Content shortcut | Many real/fake subjects and a content-held-out gate are present | Audit semantic/source balance and add matched-prompt or matched-content data only when provenance permits | No animal/face/single-domain gate explains performance |
| Recent commercial generators absent | A codec/shape-controlled Qwen audit now covers 18 current names and exposes Nano Banana 2 as weak; v8 oversampling them regressed the frozen clean gate | Run one sampling-capped v9 ablation, then open the sealed frontier gate only if the frozen internal guard passes | Frontier gain without displacing broad legacy/real-source performance |
| Heavy Gaussian noise | Repeated weakest condition: internal AUC 0.867729; external AUC 0.797466 | Screen one fixed raw-plus-mild-denoise inference view and one geometry-only multi-view policy | Improvement on both frozen gates without harming clean/other worst conditions |
| Low-resolution real false positives | Five false positives at threshold 0.5 on the 128-image audit, mostly CIFAKE reals | Stratify continuous scores by resolution/source and test matched up/downsampling controls | Ranking error reduced without training on threshold labels |
| Single representation | Selected model is a frozen PE-Core encoder plus a linear head; workshop suggests complementary signals | First test reversible multi-view inference; then only an original small complementary branch with an ablation | Independent gain across gates, not merely additional parameters |
| High score is validation illusion | Internal and external gates are strong but related to known detector datasets | Treat all scores as controls, add newer independent families and matched-source tests, keep hidden-set claim explicitly unproven | Multiple causally controlled, source-held-out failures fail to falsify the model |
| Compute fragility | Full matrix is practical on NVIDIA; local MPS batch 1 works but batch 2 fails | Use local batch-1 screening and send only gate-worthy candidates to Kaggle | Every long run resumable, hashed and reproducible; v6 always retained |

## Predeclared experiment queue

### A. Reversible multi-view inference screen

Use the deterministic 128-image subset and the exact selected checkpoint.
Evaluate each policy on clean, JPEG quality 30, blur sigma 2, resize 0.25,
Gaussian noise 0.10, brightness 0.8, contrast 0.8, saturation 0.8 and center crop
80%. The policies are fixed before observing results:

1. Existing short-side resize plus center crop (reference).
2. Equal probability average of reference and horizontal flip.
3. Equal probability average of reference and full-frame square stretch.
4. Probability blend 0.75 reference plus 0.25 reference-after-Gaussian-blur
   sigma 0.5. This is the only denoise-weight candidate.

Promotion from the screen requires a higher mean of clean and pooled selected
stress AUC, no clean decrease greater than 0.002, no decrease greater than 0.01
on any selected stress, and improvement on Gaussian noise 0.10. A passing
policy is then evaluated once on both full frozen gates.

### B. Matched-metadata and low-resolution audit

Measure width, height, aspect ratio, pixel count, suffix and available JPEG
metadata by class, named generator and named real source. Test metadata-only
separability and stratify v6 scores. This is diagnosis; it cannot select a new
model by itself.

### C. Recent-generator breadth

Inventory current coverage by generator architecture/final decoder and search
for public/licensed recent outputs. Do not substitute the forbidden demo set or
unverifiable web images. New data first enters an audit-only gate; it is not
promoted into training until provenance, licence, deduplication and class/source
matching pass.

### D. Original complementary signal

Only after A-C, consider a compact complementary branch or degradation-
consistency loss. It must be our own implementation, not a direct replication
of an existing AIGC detector. The fixed v6 fallback and frozen gates remain the
decision authority.

### E. Frontier sampling ablation

V8 completed and was rejected because equal mass per named generator gave the
18 small frontier groups 60% of fake draws and reduced the like-for-like full
clean selection AUC by 0.009484; its worst pair fell by 0.033056. V9 keeps the
same unique images and training recipe but predeclares
15% of fake draws for the whole frontier block. It is rejected before any
sealed-gate inspection if clean AUC falls more than 0.002 from v6 or worst-pair
AUC falls more than 0.01. The like-for-like numerical floors are therefore
0.979830 clean and 0.862510 worst pair. The separate frozen content view must
also remain at or above 0.992353 clean and 0.959759 worst pair, corresponding to
the same 0.002/0.01 limits from v6. Passing every check permits one comparison
on the sealed prompt/pixel-held-out frontier gate and full NTIRE audit; it does
not by itself establish hidden-set generalization.

### F. Protected frontier ensemble screen

V9 is not eligible as a standalone model, but the already-open frontier audit
may determine whether it learned complementary modern-generator evidence. If
that diagnosis improves over exact v6, screen exactly three continuous-score
averages with v6 probability weight 0.90, 0.75 and 0.50; the remaining weight
is assigned to v9. No threshold or per-image gate is allowed. A blend must pass
all four v9 internal floors above. Among passing blends, select the one with the
highest already-open frontier diagnosis AUC, breaking a tie by frontier worst
pair and then larger v6 weight. If none pass, abandon this ensemble path. Only
one selected blend may proceed to the still-sealed holdout and full robustness
gates. Two 315,776,001-parameter encoders total 631,552,002 parameters, below
the two-billion limit, but runtime and packaging remain explicit feasibility
costs rather than assumed acceptable.

### G. Preselected ensemble promotion matrix

The protected screen completed before any sealed inspection. The 90/10 and
75/25 v6/v9 probability blends passed all four frozen clean floors; the 50/50
blend failed and is permanently rejected. The predeclared ranking rule selects
exactly the 75/25 blend because its already-open frontier diagnosis AUC was
higher. No other weight, threshold, calibration rule or per-image router may be
examined on the remaining gates.

The exact v6 fallback and the selected 75/25 blend are evaluated in the same
batch on identical transformed pixels. Promotion requires all of the following:

1. On the prompt- and pixel-sealed Qwen holdout, the blend must improve both
   clean AUC and the weakest named generator/real-source pair over v6. Its
   organizer-style score may not fall by more than 0.002.
2. On the full 512-image NTIRE audit, the full internal 3,071-image matrix and
   the 624-image Community Forensics matrix, clean AUC, pooled-robust AUC and
   organizer-style score may not fall by more than 0.002; heavy-noise AUC and
   the worst named pair may not fall by more than 0.01.
3. Every package, manifest, image and checkpoint is checksum-verified; the
   prohibited demo resources remain absent. A result is valid only after all
   20 individually transformed conditions complete and a resumable summary is
   written.

Failure of any gate retains the single-model v6 artifact. Passing all gates
still does not prove hidden-set performance: the two-encoder runtime and
packaging cost must be measured and disclosed before replacement.

The matrix completed all 80 paired condition files. The first summary command
returned nonzero after inference because it requested a nonexistent generic
pair key; the stored summary already uses separate clean and pooled pair keys.
The reporting-only fix added a regression test, changed no predictions, and
the corrected evaluator at SHA-256
`10c258e251b3bb2f866a722700772502982cd17ba041a9e6d8ca0bfaf0d564c4`
returned zero. All four statistical gates passed. The large gain was confined
to the prompt/pixel-sealed frontier gate; the familiar gates showed the
expected small positive and negative changes. Exact metrics are frozen in
`FRONTIER_ENSEMBLE_PROMOTION_RESULT.json`.

### H. Ensemble feasibility and numerical-consistency gate

The statistical pass does not yet replace v6. Before changing `run.sh` or the
selected artifact:

1. Recompute a fixed clean gate at batch sizes 64 and 128 and compare every v6
   and blend score, AUC and rank ordering. Record any kernel/batch numerical
   drift; do not assume batching is mathematically invisible.
2. Export or preserve the exact v9 checkpoint at its frozen SHA-256, build a
   deterministic two-checkpoint directory runner, and verify finite continuous
   JSON scores on both CPU and an available accelerator.
3. Measure cold load, warm per-image latency, total model bytes and actual peak
   memory. The summary-only resume timing and memory are explicitly invalid as
   full-matrix feasibility measurements.
4. Keep v6 as a documented single-model fallback. Reject ensemble deployment
   if packaging or runtime is unreliable even though its statistical gates
   passed.

The batch-stability audit is frozen before execution on the already-open
576-image Qwen clean holdout. Batch 64 and batch 128 must produce identical
transformed-tensor SHA-256 values. For both v6 and the blend, maximum absolute
score drift must be at most `1e-4`, AUC drift at most `1e-6`, and maximum rank
displacement at most one position. Batch 128 must independently satisfy the
same limits against the saved clean promotion predictions. This audit does not
open or select on any new data.

The audit returned code zero but **failed** those frozen tolerances. Identical
transformed tensors had SHA-256
`beac1b5cf272ced96d3c0c607721bf211939e036d8db98bae5108f0090caad77`
at both batch sizes, yet maximum score drift was 0.000488, AUC drift was
0.0000121, and v6 moved by up to two rank positions. Batch 128 versus the saved
blend predictions reached 0.000732 maximum score drift and five rank positions.
The result is preserved in `FRONTIER_ENSEMBLE_BATCH_STABILITY_RESULT.json`; the
tolerances are not relaxed after inspection.

The remediation hypothesis is a fixed physical batch shape of 64 with
deterministic padding of the final batch. Before implementing the runner, test
native batch-64 outputs against logical request batches of 1, 17 and 64 that
all execute as physical batch 64. Use the same already-open clean Qwen pixels.
Every original-image score must match native batch 64 within `1e-6`; AUC and
rank order must match exactly. If padding fails, the ensemble remains a
research result and v6 stays selected.

The first fixed-physical-batch implementation made logical batches 1, 17 and
64 identical but failed exact reproduction of the saved blend because it moved
the probability blend from CUDA FP16 to CPU FP32. The follow-up exact-contract
audit restored the original arithmetic and passed with zero score, AUC and rank
drift. A complete one-T4 run then reproduced all 576 saved predictions exactly,
measured 4,275,770,368 peak allocated CUDA bytes and 41.72 images per model-
forward second after loading.

The candidate Python command and exact `run_ensemble.sh` entry point each
subsequently completed the full 576-image directory-to-JSON path with return
code 0 and identical output SHA-256. The shell run took 48.67 seconds including
both 1.895 GB checkpoint hashes, model loading, decoding, inference and JSON
writing. The ensemble deliberately fails closed on non-CUDA devices because a
CPU/MPS arithmetic-equivalence claim was never established; the unchanged v6
`run.sh` remains the verified CPU/MPS fallback. This is a documented narrowing
of feasibility, not an invented portability claim.

Checkpoint transport remains unresolved. Read-only inspection confirmed that
1,263,104,004 of v9's 1,263,202,267 bytes are state-dictionary tensors, with no
optimizer or training history to strip. A smaller-precision checkpoint would
be a new artifact requiring full revalidation. Until the exact checkpoints
have immutable distribution URLs and a defensible final weight licence, the
ensemble remains a verified CUDA candidate rather than the selected release.

### I. Exact-metadata NTIRE ensemble falsification

Before observing the result, freeze one diagnosis on existing predictions only.
Reconstruct the earlier NTIRE audit subset with exact JPEG format, RGB mode,
width and height matches inside every retained label stratum, using seed
20260829. It must contain the previously observed 86 rows, balanced 43/43.
Join by immutable image SHA-256 to all 20 already-saved v6/ensemble condition
files. Report clean, pooled transformed, 50/50 official-style, weakest
condition and heavy-noise AUC for both artifacts. No model, transform, score,
weight or threshold changes, and the result cannot select another ensemble.

The purpose is falsification: determine whether the ensemble's small NTIRE
changes survive after the exact geometry/container opportunity is removed.
Even a pass does not eliminate semantic, acquisition-source or generator
shortcuts and cannot estimate the organizer hidden set.

Observed result: the checksum-gated command returned zero on 86 rows across
31 exact metadata strata. The 75/25 blend improved the official-style score
from 0.973559 to 0.975436 (+0.001877), and the weakest/heavy-noise AUC from
0.727961 to 0.732829 (+0.004867). This rejects only the narrow claim that the
NTIRE ensemble gain disappears after exact format, mode, width and height
matching. The small sample and retained semantic/source structure leave the
larger hidden-transfer risk open.
