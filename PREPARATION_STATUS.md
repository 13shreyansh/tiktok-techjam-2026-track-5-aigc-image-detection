# Preparation status

Preparation research was closed and reconciled before 12:00 SGT on 29 August
2026 at the user's instruction. No judged solution work began before the
official start.

## Allowed preparation

- [x] Index authorized public dataset sources and licences
- [x] Record dataset sizes and available upstream checksums before large downloads
- [x] Download and verify the manageable CIFAKE and COCO source archives locally
- [x] Validate the neutral environment and image-directory input boundary
- [x] Preserve the official transform, output and demonstration requirements
- [x] Record that no organizer baseline is currently specified
- [x] Recheck the released public statement and current Devpost rules
- [x] Preserve and reconcile primary-source detector and prior-competition research
- [x] Reconcile workshop metric, scope, transform and submission clarifications
- [x] Close the legal/technical candidate shortlist without running experiments

## Ready

- Immutable SID_Set Hugging Face revision and 140,056,468,470-byte snapshot metadata
- CIFAKE archive verified at 120,000 images with MD5 and SHA-256 recorded
- WildFake repository inventory plus per-file checksum acquisition route
- COCO `val2017` source archive verified at 5,000 images, with the official
  per-image licence metadata archive also verified
- WildFake DALL-E Advanced index identified as exactly 8,843 rows
- Guarded, checksum-verifying acquisition utility with no secrets
- Exact image-directory input and per-image JSON key contract recorded
- A final 1,795-line research ledger covering current detector families,
  commercial evidence, comparable competitions, licences, resource disclosures,
  contradictions and unresolved claims
- A 5,086-word beginner-readable journal organised around official rules,
  detector concepts, useful evidence, failure modes and unresolved questions
- A prioritized, beginner-readable question sheet for the confirmed Track 5
  workshop and Q&A on 28 August 2026
- A pre-start launch decision sheet separating confirmed rules, evidence-backed
  candidates and questions deliberately deferred to experiments

## Blockers and intentionally incomplete items

- The organizer provides no official baseline; none was reproduced or selected.
- The public COCO archive has 5,000 images, but the organizer's two exclusions
  for the 4,998-image demo subset are not published.
- The 25,587,709,291-byte WildFake DALL-E archive, approximately 140 GB SID_Set
  snapshot and approximately 1.29 TB WildFake repository were not fully
  downloaded due to their resource size; checksummed acquisition records are ready.
- JSON container shape, ordering, confidence range and failure semantics are
  not specified by the organizer; the workshop did establish AI-generated as
  the positive direction.
- Exact hidden composition, transformation software/codec, sampled subset and
  evaluation runtime/hardware limits remain unpublished.
- Historical preparation had no ML framework or verified NVIDIA execution.
  After the official start, the local environment was built and tested and a
  Kaggle Tesla P100 completed a real CUDA operation; see `COMPUTE_OPTIONS.md`.

## Deferred until the challenge window

- Detector selection or training
- Robustness augmentation strategy
- Competition inference and error-analysis implementation

See `PRESTART_RESEARCH_CLOSURE_2026-08-29.md` for the post-start experimental
decision queue. The technical metric and individual-transform semantics are no
longer blockers after the organizer workshop.

## Challenge-window transition

The user confirmed receipt of the official start email after 12:00 SGT on
29 August 2026 and authorized judged implementation. Preparation remains
preserved above as a historical boundary. The active implementation, verified
commands and results now live in `README.md` and `EXPERIMENT_LEDGER.md`.
The active multi-source, generator-held-out policy is in `DATA_STRATEGY.md`.

## Current challenge-window status

- The submission harness, grouped clean evaluator, all 19 individual workshop
  transformations, continuous-score directory runner and below-2B parameter
  gate are implemented.
- v1 through v5 controls are completed and truthfully rejected where a new real
  source or generator/source pair collapsed; no pre-v6 candidate is selected.
- The checksum-pinned v6 mixture has 18,958 balanced training rows and fixed,
  byte-disjoint generator/source gates. One visually confirmed perceptual
  near-duplicate is excluded from scoring by immutable SHA-256.
- PE-Core-L v6 is selected over the completed DINO control and rejected v7
  noise ablation. It scores 0.991079/0.978019/0.984549 on the internal
  clean/pooled-transformed/50-50 gate and 0.992634/0.972330/0.982482 on the
  separate 78-model external audit. Heavy Gaussian noise remains the weakest
  repeatable condition.
- Identical-codec, stretch, deterministic square-patch and forced-aspect tests
  pass on both gates. These controls weaken the known PNG/JPEG and shape
  shortcut hypotheses but do not prove performance on the unpublished hidden
  distribution.
- The exact 1,263,202,267-byte full checkpoint is preserved outside Git with
  SHA-256 `4b8f3ac4776b0fddc689252de760d661916d9377374484703487538e8268766a`.
  Its local FP16 inference export is 631,645,967 bytes with SHA-256
  `48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644`.
- `run.sh` completed four-image checks on Apple MPS and CPU using the selected
  artifact. The selected model has 315,776,001 total parameters, and all output
  rows contain continuous AI-positive probabilities.
- A deterministic 128-image local error audit scored 0.995605 clean AUC but
  produced five false positives at the illustrative 0.5 threshold, mostly on
  very low-resolution authentic images. The threshold is not claimed to be
  calibrated to the hidden set.
- The test suite reports 36 passing tests. Robustness, error-analysis and
  packaging documentation passed the final repository checks and were committed
  locally in `3bca6ea`. No push, publication or submission was attempted.
- A deterministic audit-only sample from the official NTIRE 2026 public
  training record provides a newer independent source with randomized JPEG
  filenames and broad geometry. On an exact 43-real/43-fake metadata match,
  selected v6 scored 0.981071 clean AUC and 0.969581 on the organizer-style
  clean/pooled-single-transform score. Heavy Gaussian noise sigma 0.10 fell to
  0.716604 AUC, so noise robustness remains unresolved rather than hidden by
  the strong aggregate.
- A predeclared horizontal-flip mean policy passed the small 128-image screen
  but failed the independent NTIRE condition guard: medium-noise AUC decreased
  by 0.017307. It is rejected for promotion. A larger paired Kaggle run was
  interrupted by an idle-session restart after all internal and ten external
  conditions; its partial noise results also regressed and no complete artifact
  survived. `run.sh` and the selected checkpoint remain unchanged.
- A codec- and shape-controlled 576-image audit paired 18 current Qwen Image
  Bench generators with five frozen real sources. Selected v6 scored 0.953776
  overall, but only 0.895182 on Nano Banana 2 against all reals and 0.699353 on
  Nano Banana 2 versus CIFAKE. These lower scores are useful falsification
  evidence: frontier-family and real-source breadth are not solved.
- A prompt- and pixel-disjoint 576-image frontier training partition is
  checksum-packaged and was used in a completed v8 single-factor ablation. V8
  returned code 0 but regressed from 0.981830 to 0.972346 clean AUC on the
  like-for-like full clean selection view; its worst pair also fell from
  0.872510 to 0.839454, so it is rejected. The cause under test is excessive
  group-balanced oversampling: 18 frontier names collectively received 60% of
  fake draws despite only 576 unique images. V9 keeps every other factor fixed
  while capping the frontier block to 15% of fake draws. Its code and sampler
  test pass locally. The completed v9 run recovered much of v8 but still failed
  all four committed floors: 0.978066 selection clean, 0.861093 selection worst
  pair, 0.988601 content clean and 0.932552 content worst pair. It is rejected,
  and the third 288-image frontier partition remained prompt/pixel sealed and
  unscored through those standalone runs. A protected three-weight ensemble
  screen subsequently selected only a 75/25 v6/v9 probability blend: 90/10
  also passed the internal floors, while 50/50 failed and no other weight may
  be tried. The preselected 75/25 blend then completed all 80 paired conditions
  across the sealed Qwen, full NTIRE, internal and 78-model external gates.
  After a summary-key reporting bug was fixed without changing predictions,
  the checksum-verified command returned zero and all frozen statistical gates
  passed. Exact results are preserved in
  `FRONTIER_ENSEMBLE_PROMOTION_RESULT.json`. This advances the blend to
  feasibility verification, not yet to the runnable selected artifact: batch
  consistency, two-checkpoint packaging, latency and full-run peak memory are
  still unverified, so the local v6 artifact remains the immediate fallback.
- A diagnosis-only exact-metadata rerun over the already-saved NTIRE
  predictions returned zero. On 86 balanced images across 31 exact
  format/mode/width/height strata, the 75/25 blend improved the
  official-style score by 0.001877 and heavy-noise AUC by 0.004867. This
  weakens the simplest container/geometry explanation for that gain, but the
  sample is small and semantic, acquisition-source and generator shortcuts
  remain unresolved. No score, model, blend weight or promotion gate changed.
- A subsequent batch-stability command returned zero but failed its strict
  predeclared consistency limits despite identical transformed-tensor hashes.
  Batch 64 versus 128 moved AUC by 0.0000121 and up to two rank positions;
  batch 128 versus saved blend predictions moved up to five positions. The
  tolerances were not relaxed. Fixed-physical-batch padding then made logical
  request sizes 1, 17 and 64 exactly identical when the physical CUDA batch
  remained 64, but its CPU-FP32 blend still differed from saved promotion
  scores by up to 0.000366 and therefore failed the exact 0.000001 score gate.
  This exposed that saved promotion files used the original GPU-FP16 blend,
  while later paired-GPU code used CPU-FP32 blend arithmetic and its resume
  signature did not encode that policy. The checksum-verified exact
  original-arithmetic reproduction then passed: physical batch 64 with FP16
  blend produced zero score, AUC or rank drift across logical batches 1, 17
  and 64 and versus the saved promotion subset. Numerical reproducibility is
  therefore established for that precise two-T4 contract. Exact v9 checkpoint
  preservation, a production two-checkpoint runner, one-T4 resource evidence
  and byte-identical local v9 recovery were subsequently completed; public
  distribution, deployment-machine portability and weight licensing remain
  incomplete, so the ensemble is still operationally unselected.
- A subsequent one-T4 audit reproduced every saved v6 and blend score exactly
  on the complete 576-image clean Qwen gate using the original single-GPU,
  physical-batch-64 FP16 contract. It measured 16.14 seconds model loading,
  41.72 images per model-forward second and 4,275,770,368 bytes peak CUDA
  allocation. This removes a two-GPU dependency but does not yet package or
  distribute the 1.895 GB combined checkpoints, implement the candidate
  submission runner, or establish non-CUDA fallback behavior. V6 remains the
  selected runnable artifact until those gates pass.
- The candidate directory-to-JSON ensemble module then completed an isolated
  one-T4 end-to-end run over all 576 preserved Qwen audit images. The first
  invocation failed before inference because null checkpoint codec metadata
  was passed through instead of its effective `none` default; that packaging
  bug alone was fixed. The corrected command returned zero, produced 576
  finite probabilities in the unit interval, reported 631,552,002 parameters
  and reproduced every saved promotion score exactly in 47.51 seconds,
  including checkpoint hashing, loading, decoding and inference. The exact
  `run_ensemble.sh` shell entry point then returned zero on the same full set
  in 48.67 seconds and produced the identical output SHA-256. Evidence is in
  `FRONTIER_ENSEMBLE_E2E_RESULT.json` and
  `FRONTIER_ENSEMBLE_WRAPPER_RESULT.json`. The candidate remains separate from
  selected `run.sh`: neither exact checkpoint has a public distribution URL,
  organizer-hardware portability is unproved, and final weight licensing is
  unresolved.
- The exact v9 file was preserved through a successful private Kaggle quick-save
  version with output saving enabled; its account locator is redacted from the
  release-facing tree. The exact
  output was then downloaded to ignored local storage and independently
  reverified at 1,263,202,267 bytes with SHA-256
  `dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075`.
  This closes private byte-recovery risk, not public distribution, portability
  or trained-weight licensing.
- A predeclared FP16-storage packaging ablation reduced v9 from 1,263,202,267
  to 631,625,115 bytes but failed exact equivalence on the complete 576-image
  Qwen gate. Maximum ensemble-score drift was 0.00390625, AUC drift was
  0.0001567 and maximum rank displacement was four. The compact artifact is
  rejected from the verified ensemble contract; exact evidence is in
  `FRONTIER_V9_FP16_EXPORT_AUDIT_RESULT.json`.
- A second prompt- and pixel-disjoint 576-row Qwen clean holdout independently
  confirmed the candidate direction under the exact one-T4 FP16 contract. The
  75/25 blend improved v6 AUC from 0.932412 to 0.954638, worst-generator AUC
  from 0.850260 to 0.895399 and worst generator/real-source pair from 0.641164
  to 0.722522. It did not solve content dependence: prompt identity still
  explained 58.5940% of blend fake-score variance. The inspected gate is now
  consumed and remains excluded from training. Exact compact evidence is in
  `QWEN_UNSEEN_PROMPT_ENSEMBLE_RESULT.json`.
- Exact saved clean/noise prediction rows reveal that the headline heavy-noise
  aggregates hide a repeated source-specific inversion. With Gaussian noise
  sigma 0.10, restricting authentic images to low-resolution CIFAKE-CIFAR10
  yields blend AUC 0.529291 internally, 0.417989 on Community Forensics and
  0.386434 on Qwen; mean AI scores for those reals are about 0.83. This is a
  consumed diagnosis, not a promotion gate. Evidence is in
  `FROZEN_GROUP_FAILURE_AUDIT_RESULT.json`.
- A frozen real-source dependence audit found that the five authentic sources
  explain 33.9694% of selected-v6 real-score variance even after identical
  JPEG-q96 and stretch preprocessing. CIFAKE's 32-by-32 authentic images were
  the hardest source; inspected LAION false positives also included human-made
  artwork and a stock anatomical render. This is a source/label-boundary warning,
  not proof of the feature the model uses. Evidence is in
  `REAL_SOURCE_SCORE_DEPENDENCE_RESULT.json`.
- A manifest-only generator coverage audit corrected the transfer claim. All
  18 Qwen names are absent from v6 training, so v6 provides exact-name holdout
  evidence. All 18 are present in v9 training, so the candidate blend's Qwen
  gains are prompt/pixel holdout evidence within known names. Community
  Forensics adds 78 unseen exact model names but not a demonstrably new family;
  NTIRE's family remains undisclosed. Evidence is in
  `GENERATOR_COVERAGE_AUDIT_RESULT.json`.
- A new class-balanced 1,000-image CIFAR-100 test gate has been frozen before
  scoring to determine whether the repeated noisy-CIFAKE inversion generalizes
  to a second 32-by-32 authentic source. Its exact source, selection, package,
  inference contract and interpretation bands are recorded in
  `CIFAR100_LOWRES_GATE_PLAN.json`. It is evaluation-only and cannot be used for
  training, tuning, calibration, thresholding or model/blend selection.
- That independent gate has now completed on one Kaggle Tesla T4. Clean AUC is
  0.850858 for v6 and 0.891642 for the 75/25 blend. Under Gaussian noise sigma
  0.10, AUC falls to 0.449342 and 0.473915 respectively; the noisy authentic
  mean exceeds the unchanged Qwen-fake mean for both candidates. The frozen
  interpretation therefore confirms a general low-resolution authentic
  ranking inversion on the new source. The exact row hashes, 2,000-replicate
  intervals and class diagnostics are in `CIFAR100_LOWRES_GATE_RESULT.json`.
  These consumed rows remain excluded from every learning and selection step.
- A checksum-frozen matched low-resolution v10 repair then completed once on a
  Kaggle Tesla T4. Its fixed 75/25 v6/v10 blend passed all four existing clean
  non-regression gates and improved the new noisy-gate AUC from 0.391219 to
  0.462979, but it remained below chance and below the frozen 0.60 floor; noisy
  authentic mean score also remained above fake mean. V10 therefore failed two
  frozen promotion checks and is rejected. The new gate is consumed and no
  tuning, calibration, threshold or blend change may use it. Exact evidence is
  in `V10_LOWRES_REPAIR_RESULT.json`; organizer demo-only rows used: zero.
- A label-blind v11 quality router then preserved the normal v6/v9 path and
  improved sigma-0.10 AUC on three consumed screens. Its 1,024-image untouched
  NTIRE gate nevertheless improved noise AUC only from 0.789886 to 0.805782,
  below the frozen +0.05 requirement, so v11 is rejected. Clean AUC remained
  exactly 0.987041 and zero clean rows routed. The gate is consumed and cannot
  be used to retune the router. Exact evidence is in
  `V11_QUALITY_ROUTE_PROMOTION_RESULT.json`; organizer demo-only rows used:
  zero.
- A pre-publication source-tree audit confirms all required code/document
  artifacts are present, both run scripts are executable, no weight/archive or
  file over 10 MiB is tracked, current-tree private locators are redacted and
  the candidate is below 2B parameters. It fails closed on external publication
  actions: the exact v6/v9 checkpoints have no public immutable URLs and the
  Devpost draft retains four public-link placeholders. Evidence is in
  `SUBMISSION_TREE_AUDIT_RESULT.json`.
- A history-free source ZIP from commit `9c4d45d` is 520,096 bytes with SHA-256
  `1faab63ae60b5801671649afbd4d8b11bd4bc1f09d355eda369a43a28bc15845`.
  After extraction outside the repository it passed all 118 tests; its tree
  audit found zero private/forbidden-artifact blockers and only the four public
  link placeholders plus missing checkpoint URLs. It is ignored, unpublished
  and not a submission. Evidence is in
  `PUBLIC_SOURCE_BUNDLE_RESULT.json`.
- The current test suite reports 118 passing tests. No push, publication or
  submission was attempted.
- A four-image demo rehearsal was frozen by hashes before inference using two
  CIFAKE authentic images, one CIFAKE Stable Diffusion image and one Qwen
  FLUX.2-pro image. The managed shell's MPS attempt failed before inference;
  unchanged CPU fallback returned zero with four finite probabilities. No
  organizer demo rows were used and the rehearsal is not evaluation evidence.
  Compact evidence is in `DEMO_REHEARSAL_RESULT.json`.
- A source wheel built from the current tree installed with all exact declared
  dependencies in a clean temporary environment. From `/tmp`, its installed v6
  prediction CLI returned zero and reproduced the four frozen CPU rehearsal
  scores exactly. This proves packaging portability on local CPU, not accuracy,
  ensemble portability or organizer-hardware compatibility; exact evidence is
  in `PACKAGE_INSTALL_RESULT.json`.
- GitHub's official release limit can host the two exact checkpoints as
  separate assets, but the training lineage creates a real licence conflict:
  conservative NC/SA weight terms are not OSI open source. The provisional
  release route, alternative and unresolved organizer-language question are
  recorded in `WEIGHT_RELEASE_DECISION.md`; no licence was applied and no
  release was attempted.

## Latest controlling correction and compliant replacement path

- The workshop's explicit non-commercial-data prohibition supersedes the
  earlier provisional weight-release route. Historical v6/v9 checkpoints use
  direct AFHQ-v2, FFHQ and CelebA-HQ inputs and are **not submission-eligible**.
  Their scores remain diagnostic controls only.
- A licence-filtered COCO **train2017** acquisition now contains 6,000 unique
  images and explicitly excludes every val2017 identity and every NC/ND licence
  ID. This is separate from the organizer's prohibited demo-only val2017 set.
- The permissive v12 mixture has 13,574 balanced train rows and a frozen 2,000-
  row evaluation set. It contains zero byte overlap, zero demo row and zero
  recorded non-commercial row.
- The raw mixture failed a metadata-shortcut audit at 0.998360 evaluation AUC.
  After label-blind square JPEG canonicalization, the same probe is 0.513093;
  shape-only and PNG-only are both 0.500. This removes one known shortcut but
  does not establish hidden accuracy.
- The frozen canonical package is 970,978,505 bytes, SHA-256
  `6bfcb918676cef772b7a71e2fad8ad2fd0789efab9803fb028fb1302cd801447`,
  with inventory SHA-256
  `ec78d74e62d8e1b1f75e661f2ea3338fa95be11e96694a3ed168b463fe314fa6`.
  The complete suite passes 140 tests. The private Kaggle input was verified
  image-by-image and both predeclared linear-head runs returned zero.
- DINOv2-L clean AUC is 0.882450 with worst generator/source pair AUC 0.605000.
  PE-Core-L clean AUC is 0.998540 with worst pair AUC 0.890500. Both pass the
  frozen clean screen, but the PE result is treated as a shortcut alarm rather
  than proof of hidden-set transfer.
- Exact checkpoints are preserved in the active private Kaggle session:
  DINOv2-L SHA-256
  `db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b`
  (1,213,023,315 bytes) and PE-Core-L SHA-256
  `f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2`
  (1,263,202,331 bytes). They are private, unpublished and not yet durable
  outside that session.
- The predeclared 20-condition workshop matrices completed with return code
  zero. PE-Core-L scored 0.998540 clean, 0.992608 pooled transformed and
  0.995574 official-style, with a 0.935380 worst individual condition.
  DINOv2-L scored 0.882450, 0.861357 and 0.871904 respectively, with a
  0.792617 worst individual condition. These remain source-confounded results,
  not hidden-set claims.
- The original clean gate is now explicitly classified as source-confounded:
  no dataset contributes both real and fake evaluation rows. A fresh 2,000-row
  matched-source CIFAKE test gate was deduplicated, canonicalized, frozen and
  uploaded privately before scoring. Its package SHA-256 is
  `b91363fef08bceb3c72f86ca4e5d4fce8b0c0a530d79e56b431dfa8a0087d383`;
  the exact interpretation bands are in
  `CIFAKE_MATCHED_SOURCE_V12_GATE_PLAN.json`. The private Kaggle input reports
  ready to use. Both unchanged candidates completed its clean plus 19-condition
  evaluation with return code zero. PE scored 0.922760 clean, 0.892005 pooled
  transformed and 0.907382 official-style; DINO scored 0.913844, 0.899374 and
  0.906609. Both pass the frozen gate, but the near tie shows the earlier PE
  lead was materially exaggerated by source confounding. Exact results are in
  `V12_MATCHED_SOURCE_GATE_RESULT.json`.
- Both exact checkpoints and gate reports are now preserved in the private
  Kaggle dataset `track5-v12-exact-checkpoints-and-gates`. The page read back
  PRIVATE, Version 2, 2.48 GB and 14 files. Version 2 adds both fresh SID
  progress reports, the frozen SID evaluator and its source-confound audit.
  This is not a submission or public
  release. Integrity-manifest SHA-256 is
  `75602edac15dff4ccd5e430766e4ad80a3a32c943e67f48232f368ebf8dc9141`.
- The fresh 568-row high-resolution SID_Set gate returned zero for both
  unchanged candidates. PE scored 0.999733 clean, 0.997705 pooled transformed
  and 0.998719 official-style; DINO scored 0.860122, 0.839642 and 0.849882.
  Both pass the frozen numerical floors, but the result is not promotion
  evidence: every selected fake originated as a square PNG, while 283/284
  selected reals originated as JPEG and only 11/284 reals were square. The
  original PNG rule has 1.0 AUC and the square rule 0.980634. PE's near-perfect
  result is classified as a source-specific codec/geometry shortcut alarm,
  not hidden-set evidence.
- A no-search 50/50 probability blend of the v12 PE-Core-L and DINOv2-L
  predictions improved the matched-source CIFAKE gate to 0.945613 clean AUC,
  0.930195 pooled transformed AUC, 0.937904 official-style and 0.857868 worst
  condition. Its fixed rule and decision thresholds were committed before the
  blend was scored. The same blend scored 0.988586 official-style on SID, but
  that result remains source-confounded and diagnostic only.
- The protected historical v6 checkpoint then completed the exact same CIFAKE
  pixels and 20 conditions: 0.829156 clean, 0.716622 pooled transformed,
  0.772889 official-style and 0.655350 worst condition. The v12 equal blend
  passed both predeclared comparison floors, improving official-style by
  0.165015 and the worst condition by 0.202519. It advances only as the leading
  representation-hedge candidate; this opened low-resolution gate is not a
  hidden-score estimate, and v6 remains an operational fallback.
- The frozen semantic/content audit completed on 144 pairs covering all 18
  modern Qwen Image Bench generator labels and eight prompts. PE-Core-L scored
  0.998240 clean, 1.000 paired accuracy, 0.994108 pooled transformed and
  0.996174 official-style; DINOv2-L scored 0.842761, 0.881944, 0.811537 and
  0.827149; the unchanged 50/50 blend scored 0.987510, 1.000, 0.974554 and
  0.981032. Every frozen floor passed. This is not hidden-set or model-selection
  proof because all fakes still share Qwen Image Bench collection lineage and
  all reals share COCO train2017 lineage. The gate is consumed and forbidden
  for tuning, calibration, preprocessing, weighting or candidate switching.
  Exact evidence is in `SEMANTIC_MATCHED_MODERN_V12_GATE_RESULT.json`; full
  logs and predictions are preserved in the private Kaggle dataset
  `track5-semantic-modern-v12-one-shot-audit`, whose page read back Private.
- A separate compliant `run_v12.sh` contract now verifies both exact checkpoint
  hashes and emits deterministic directory-to-JSON probabilities. On one Tesla
  T4 the four-image fixed-blend command returned zero in 19.42 seconds with
  1,322,263,040 peak allocated CUDA bytes and 619,004,930 parameters. An
  unchanged repeat was byte exact; the PE-only fallback returned zero in 11.25
  seconds. This is small run-contract evidence only. Exact evidence is in
  `V12_RUNNABLE_CONTRACT_RESULT.json`; both public immutable checkpoint URLs
  remain unresolved in `V12_CHECKPOINT_MANIFEST.json`.
- The frozen modern-fake audit was repeated with an independent Open Images V7
  validation real source. PE-Core-L scored 0.975921 official-style and 0.939742
  worst condition; DINOv2-L fell to 0.641440 and 0.577474; the unchanged equal
  blend scored 0.902284 and 0.834539. This falsifies DINO as a universally safe
  source-robust hedge, but is diagnostic only because the Qwen fake collection
  is reused. Exact evidence is in
  `OPENIMAGES_SOURCE_ROTATION_V12_RESULT.json`.
- An identity-corrected Community Forensics breadth audit completed on 593
  rows, including four examples from each of 78 named latent-diffusion
  variants. Two pre-inference failures were preserved: the first exposed a
  legacy audit-manifest schema mismatch; the second found 31 raw source
  identities that a canonical-only overlap check had missed. Those rows were
  excluded before scoring, leaving zero v12 train/evaluation identity overlap.
  PE scored 0.950201 clean, 0.947296 pooled robust, 0.948748 official-style and
  0.823421 worst condition, passing every frozen floor. DINO scored 0.811764,
  0.807565, 0.809665 and 0.796993 and failed the 0.85 clean floor. The fixed
  equal blend scored 0.918332, 0.911335, 0.914833 and 0.847608: better on the
  single weakest condition but materially worse on aggregate ranking than PE.
  No weight was searched. Exact evidence is in
  `COMMUNITY_FORENSICS_V12_AUDIT_RESULT.json`.
- Across the two newest independent diagnostics, PE is the strongest aggregate
  detector and the fixed blend is not uniformly beneficial. The existing blend
  remains the predeclared runnable candidate until one final decision rule is
  frozen and tested on a still-unopened, identity-disjoint source; changing the
  default directly from consumed evidence would create experimenter overfit.
- That final decision rule is now complete. On the 1,024-image, balanced,
  source-coherent NTIRE arbitration gate, PE scored 0.990669 clean, 0.983678
  pooled transformed, 0.987174 workshop score and 0.930805 weakest condition.
  DINO scored 0.730301, 0.720231, 0.725266 and 0.666496; the unchanged equal
  blend scored 0.942989, 0.929440, 0.936215 and 0.848385. All five frozen
  selection checks passed, so `pe_core` is the selected default. The gate is
  consumed and may not authorize further model, weight, preprocessing,
  threshold or calibration search. This is not a hidden-score estimate.
- Kaggle produced no recoverable arbitration output before the interactive
  session ended and weekly GPU quota was exhausted. The first local MPS batch-8
  attempt aborted before any score with an insufficient Metal-buffer error.
  A pre-score recovery froze batch one and zero workers; both candidates then
  completed all 20 unchanged conditions. Failures and recovery boundaries are
  preserved in `NTIRE_V12_LOCAL_RECOVERY_PLAN.json` and
  `NTIRE_V12_MPS_BUFFER_RECOVERY.json`.
- The user-facing v12 default now loads only PE, uses batch one, applies EXIF
  orientation and requires only the selected checkpoint. The exact default
  four-image Apple MPS command returned zero in 2.533843167 seconds and produced
  the frozen 464-byte output with SHA-256
  `767096b0c1ffb963fe12947e1038f1f5b1416521aaa42d5c637532ae09419157`.
  The complete source suite passes 209 tests with one non-failing physical-core
  warning.
- Remaining external blockers are one public immutable PE checkpoint URL, a
  logged-out install/run, a public two-to-four-minute video and explicit final
  submission review. The 716,168-byte history-free ZIP from source commit
  `2e1b70d` passed all 209 tests after extraction and its pristine 297-file tree
  has zero forbidden artifacts, private locators or private-key findings. Its
  SHA-256 is
  `38addd1edb7d3f17072672f579d70c377a16c3e495b8f81fc4331dc9fa138cec`.
  No publication, visibility change or submission has been performed.
