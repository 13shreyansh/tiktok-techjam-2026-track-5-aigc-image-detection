# Track 5 continuing research ledger

Final reconciliation: 2026-08-27 SGT

## Purpose and boundary

This ledger preserves evidence-backed research on AI-generated image detection
and comparable competitions for TikTok TechJam 2026 Track 5. Until the official
build window opens on 2026-08-29 12:00 SGT, it records prior art, evaluation
evidence, constraints, provenance and open questions only. It does not select,
design, train or tune the judged solution.

Claim labels used below:

- **Confirmed:** directly supported by an organizer, paper, official dataset
  record, official competition report or verified command output.
- **Derived:** calculated or inferred from confirmed evidence; the inference is
  stated.
- **Unverified:** plausible secondary report or incomplete evidence.
- **Unknown:** the available authoritative sources do not specify the answer.

## Continuity checklist

Before relying on a compacted conversation or handing off this research:

1. Read this ledger completely.
2. Reconcile all material new findings, URLs, versions and contradictions into
   the relevant sections.
3. Preserve negative findings and unknowns; do not silently convert them into
   assumptions.
4. Record exact commands and observed results for any local or remote audit.
5. Confirm this research has added no dataset, model weight, cache, generated
   output or secret, and that any separately acquired official source archives
   remain covered by `.gitignore`.
6. Continue from the open research queue rather than restarting completed work.

## Official Track 5 facts relevant to this research

- **Confirmed:** The required core is an image-directory-to-JSON confidence
  predictor robust to an organizer-selected subset of listed image
  transformations, using models below 2B parameters.
- **Confirmed:** The organizer provides no official detector or baseline.
- **Confirmed:** SID_Set, CIFAKE and WildFake are suggested datasets, not
  solutions or mandatory inputs.
- **Confirmed:** The 4,998 COCO plus 8,843 DALL-E Advanced WildFake subset is
  demonstration-only, must not be used for training, tuning, selection or
  calibration, and does not contribute to the final score.
- **Workshop-confirmed update (2026-08-28):** The technical metric is an
  equal-weight average of clean ROC AUC and pooled transformed-image ROC AUC.
  AI-generated is the positive class, scores are continuous, and
  transformations are applied individually rather than chained. Exact hidden
  composition, transform software and sampled subset remain unpublished.
- **Confirmed:** The Track 5 statement gives judging weights of 35% technical
  execution, 20% innovation/problem insight, 20% impact/relevance, 15%
  feasibility/practicality and 10% final-event presentation/communication.
  The current Devpost general rules instead call four criteria equally weighted
  and omit the presentation category. The user confirmed that the workshop's
  track-specific weights control, so they are not averaged with Devpost.

Authoritative sources:

- Public statement: <https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc>
- Devpost rules: <https://tiktoktechjam2026.devpost.com/rules>

## Detection-method evidence

### What pixel-based detectors actually learn

- **Confirmed:** The modern literature contains at least four broad technical
  families. These are descriptions of prior art, not a Track 5 selection:
  1. convolutional classifiers that learn local texture or patch artifacts;
  2. transformer or large pretrained vision-encoder classifiers that learn
     longer-range/global image representations;
  3. frequency or residual methods that inspect spectral, gradient, noise or
     bit-plane irregularities left by image generation; and
  4. ensembles or mixture-of-experts systems that combine several such clues.
- **Confirmed:** Older detectors often exploited generator-specific artifacts,
  including GAN upsampling periodicities and checkerboard patterns. Newer work
  examines diffusion-denoising residuals or fine-tunes large pretrained vision
  encoders on real/fake examples.
- **Confirmed:** These systems normally return a score learned from labelled
  examples; they do not possess a perfect visual rule for "AI". A detector can
  instead learn a shortcut associated with the training dataset, file format,
  generator or processing pipeline.
- **Confirmed:** A 2026 zero-shot study evaluated 16 public methods (23 released
  detector instances) on 12 datasets without fine-tuning. It found no universal
  winner: detector rank correlations across dataset pairs ranged from 0.01 to
  0.87. Community-Forensics was the strongest released system in the paper's
  aggregate table (0.780 mean accuracy); the CNNSpot implementation was lowest
  (0.375). SAFE ranged from 0.032 to 0.998 across datasets, showing that a very
  strong result on one collection can coexist with failure on another.
- **Confirmed:** In that study, training-source alignment mattered more than a
  simple CNN-versus-transformer label. Across detectors, mean accuracy on newer
  generators was very poor: Firefly v4 18%, Imagen 4 19%, Flux Dev 21%,
  Midjourney v7 24%, and DALL-E 3 31%. This directly contradicts any broad claim
  that current public detectors reliably catch every polished AI image.
- **Confirmed:** The same study used a fixed 0.5 threshold and acknowledges
  incomplete detector/dataset coverage and that its snapshot is already
  time-sensitive. Its absolute accuracy numbers therefore must not be treated
  as expected Track 5 scores.

Primary sources:

- NTIRE 2026 robust-detection report: <https://arxiv.org/html/2604.11487>
- Public detector zero-shot benchmark: <https://arxiv.org/html/2602.07814>

### Internet-pretrained visual features: UnivFD

- **Confirmed:** UnivFD first showed why an ordinary real/fake classifier can
  fail on an unseen generator: it learns a boundary around the particular fake
  fingerprint seen during training, leaving everything else in a broad "real"
  region. In the paper's example, unseen diffusion images could therefore be
  assigned to real even though they were generated.
- **Confirmed:** Its proposed detector freezes an internet-pretrained CLIP
  ViT-L/14 image encoder and uses either nearest-neighbour comparison or a
  learned linear classifier on the frozen representation. It trains only on
  ProGAN real/fake examples and applies blur plus JPEG augmentation with
  probability 0.5.
- **Confirmed:** Relative to the paper's conventional classifier, the nearest-
  neighbour version improved unseen diffusion/autoregressive performance by
  15.05 mean-average-precision points and 25.90 accuracy points. The linear
  probe improved it by 19.49 and 23.39 points, respectively. CLIP ViT-L/14 was
  stronger than the tested CLIP RN50, ImageNet RN50 and ImageNet ViT-B/16
  representations, supporting the importance of broad pretraining.
- **Confirmed:** The official repository says its training data is about 72 GB
  and uses `--fix_backbone` so only the linear layer is updated.
- **Derived:** These are results from the paper's 2023 protocol, not proof of
  universal detection. The later 2026 zero-shot study shows that UnivFD's
  relative performance changes substantially across newer datasets and
  generator families.
- **Confirmed:** The official repository is MIT-licensed. The audited default-
  branch revision is `030495aea3300a8b54c0ec37ec7fe1dd7e63c619`; no source,
  data or weight was cloned or downloaded.

Primary sources:

- CVPR 2023 paper:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html>
- Paper HTML: <https://arxiv.org/html/2302.10174>
- Official repository: <https://github.com/WisconsinAIVision/UniversalFakeDetect>

### Intermediate foundation-model features: RINE

- **Confirmed:** RINE freezes a CLIP ViT-L/14 image encoder but collects the
  class token from all 24 transformer blocks rather than using only the final
  representation. Two lightweight projections and a trainable importance
  estimator learn which layers and feature elements matter, followed by a
  binary head; training combines classification and supervised contrastive
  losses.
- **Confirmed:** In the paper's ProGAN-to-20-dataset protocol, the four-class
  model reported 91.5 mean accuracy and 98.8 mean average precision, an average
  10.6-point accuracy improvement over the compared prior systems. Removing
  intermediate features reduced accuracy to 82.5, the largest ablation loss.
- **Confirmed:** Only 6.32 million parameters are learned in the four-class
  head, trained for one epoch in about eight minutes on one RTX 3090 Ti. That
  figure excludes the frozen CLIP encoder. SAFE's separate comparison reports
  the complete UnivFD CLIP ViT-L/14 system at 427.62 million parameters, so
  describing RINE as a 6.32M-parameter inference system would be misleading.
- **Confirmed:** In RINE's perturbation experiment, four-class mean accuracy
  changed from 91.5 clean to 86.5 blur, 90.8 crop, 88.7 compression, 84.6 noise
  and 82.7 when the perturbations were combined. Mean average precision changed
  from 98.8 to 95.8, 98.4, 97.9, 95.7 and 92.8, respectively.
- **Confirmed:** Generator alignment remained decisive. On Synthbuster, the
  same method trained with ProGAN versus latent-diffusion images scored
  21.1/30.7 versus 47.2/32.5 accuracy/AP on DALL-E 3 and 34.2/39.5 versus
  92.4/97.4 on Midjourney. The unusual DALL-E 3 AP/accuracy combination also
  illustrates threshold sensitivity.
- **Confirmed:** The official repository is Apache-2.0 and embeds four trained
  head checkpoints ranging from about 1.1 MB to 42.1 MB; CLIP weights are still
  external. It was inspected through the GitHub API without cloning.
- **Derived:** The reported CLIP-plus-head architecture is comfortably below
  2B parameters, but Track 5 compliance still depends on how organizers count
  all submitted components. The much smaller "learnable parameter" figure must
  never be substituted for complete inference size.

Primary sources:

- ECCV 2024 paper: <https://arxiv.org/html/2402.19091>
- Official repository: <https://github.com/mever-team/rine>

### Controlling dataset shortcuts: B-Free

- **Confirmed:** B-Free demonstrates several shortcuts that can make an image
  detector look better than it is: real images stored as JPEG while fakes are
  PNG, different resizing histories, different semantic content, and different
  real-source collections. The same detector/generator combination can change
  markedly when the real comparator is RAISE instead of COCO.
- **Confirmed:** The authors construct 51,517 Creative-Commons COCO real images
  and 309,102 Stable Diffusion 2.1 self-conditioned or inpainted counterparts.
  The design aligns semantics and image coding and avoids resizing, so those
  nuisance variables cannot trivially reveal the label.
- **Confirmed:** The strongest reported detector trains a DINOv2-with-registers
  backbone end-to-end at 504 by 504 pixels and averages crops at test time. In
  its Synthbuster/new-generator/WildRF protocol, the strongest inpainted++
  variant reported 99.3 mean AUC and 96.4 balanced accuracy across comparisons
  with 27 models, a 20.7-point balanced-accuracy improvement over the stated
  runner-up.
- **Confirmed:** The paper also reports expected calibration error and negative
  log likelihood, because a high AUC can coexist with an unusable fixed
  threshold. It explicitly warns that this data-driven detector may fail on a
  future synthesis process with fundamentally different artifacts and remains
  susceptible to adversarial manipulation.
- **Derived:** B-Free is evidence that source matching and leakage control are
  part of detector evaluation, not merely data-cleaning details. Its headline
  score cannot be compared directly with NTIRE, MediaEval or the 2026 zero-shot
  benchmark because their datasets, transformations and metrics differ.
- **Confirmed:** The official code uses a custom restrictive licence: all
  rights reserved, with use/reproduction/modification limited to informational
  and nonprofit purposes. This is not an OSI open-source licence and must not be
  treated as generally reusable competition code. The audited revision is
  `c6a9f898782fb466b29af01f21960b67415afb0e`.
- **Unknown:** The complete inference parameter count and the exact licences of
  all selected COCO assets still require audit. The paper's phrase "Creative
  Commons" does not establish that every asset has the same reusable licence.
  A metadata-only HEAD request to the official weight URL could not connect to
  `www.grip.unina.it:443` after about seven seconds, so current weight size and
  availability remain unverified; no retry downloaded content.

Primary sources:

- Paper: <https://arxiv.org/html/2412.17671>
- Official project: <https://grip-unina.github.io/B-Free/>

### Compression and image-size shortcuts: Fake or JPEG?

- **Confirmed:** The paper audits six common generated-image datasets and
  finds a recurring label shortcut: natural images are usually already JPEG-
  compressed and vary in size, while generated images are stored losslessly
  and often come from one of a few fixed square resolutions. GenImage, for
  example, uses JPEG ImageNet reals and PNG fakes at 128, 256, 512 or 1,024
  square pixels depending on the generator.
- **Confirmed:** A ResNet-50 trained on the raw Midjourney subset classified
  80.45 percent of genuine, uncompressed FFHQ PNGs correctly. Compressing those
  same genuine images raised real-class accuracy to 94.84 percent at JPEG
  quality 95 and 100 percent at quality 60. Because the inputs remained real,
  this experiment directly shows the model was using JPEG evidence as a cue for
  the real label, not merely losing an AI fingerprint under compression.
- **Confirmed:** Compressing generated test images caused fake-class recall to
  fall while precision stayed near one. The paper's mean cross-generator
  accuracy at JPEG quality 95 was 53.91 for detectors trained on classic
  GenImage versus 67.17 after matching training compression at quality 96; the
  improvement remained 8.75 points at quality 80 and 4.49 at quality 60.
- **Confirmed:** Matching both compression and size substantially changed the
  cross-generator result even though it removed more than 75 percent of the
  candidate training data. Across SD1.5, SD1.4 and Wukong training subsets,
  ResNet-50 improved from 71.68 to 82.74 mean accuracy and Swin-T from 74.09 to
  85.83. This is an in-paper result, not a locally reproduced benchmark.
- **Confirmed:** The authors warn that equalizing JPEG and size still does not
  remove every shortcut. Stable Diffusion's underlying natural-image source was
  LAION while GenImage's detector-side reals were ImageNet, so a classifier may
  learn source style despite prompt-content matching.
- **Derived:** Random JPEG augmentation can improve robustness while leaving
  unequal class distributions partly intact; overlap is not the same as
  equality. For a valid evaluation, label-conditioned file format, resolution,
  source and preprocessing histories must be audited before interpreting a
  detector's score as evidence of generation artifacts.
- **Confirmed:** The official experiment repository has no top-level declared
  licence at audited revision
  `74197ede86ff0855a3e82ee6092ce50a623e376d`. Its recursive tree contains
  third-party licence files inside embedded `timm` and Swin code, which do not
  establish permission for the authors' surrounding files. No code was cloned.
- **Confirmed:** Harvard Dataverse record `doi:10.7910/DVN/AKDIHF`, version 2.0,
  reports CC0-1.0 metadata, 503 files and 653,808,797,076 total bytes. Its
  223,850,689-byte `genimage_metadata.csv` has MD5
  `2242b70fede792b9a08165f6023ca196`; the remaining archive includes 500 split
  parts. The project still directs users to understand the original GenImage
  data terms, so the Dataverse record label must not be assumed to erase
  upstream image rights. Nothing was downloaded.

Primary sources:

- Paper: <https://arxiv.org/html/2403.17608>
- Official project: <https://www.unbiased-genimage.org/>
- Official code: <https://github.com/gendetection/UnbiasedGenImage>
- Dataset/metadata record: <https://doi.org/10.7910/DVN/AKDIHF>

### Curated generator coverage with a frozen encoder: SSAFE

- **Confirmed:** The June 2026 SSAFE preprint studies frozen CLIP, SigLIP and
  Perception Encoder representations. It reports that multimodal encoders
  separate real and generated images more clearly than the tested DINO
  self-supervised encoders, and trains only one sigmoid linear layer on the
  frozen PE-Core-G14-448 image embedding.
- **Confirmed:** Rather than accumulating every available generator, it groups
  generator distributions in embedding space and selects representative ones.
  Its curated training set has 5,000 real and 5,000 generated images spanning
  eight selected domain/generator combinations, reduced from a 50,000-image,
  28-combination pool.
- **Confirmed:** On AIGIBench, the curated 10K model reported 89.4 accuracy and
  95.7 average precision, versus 78.1/83.6 for SAFE and 77.6/82.7 for AIDE in
  that paper's re-training protocol. The original AIGIBench training detectors
  had 98.0 real/78.8 fake accuracy for PE-Linear, while the curated version had
  98.6/80.1, showing that class-specific results remain necessary.
- **Confirmed:** RealWorldBench combines modern smartphone/web photographs and
  28 generators including Flux, Imagen 3/4, DALL-E 3, GPT-Image-1, Nano Banana,
  Seedream and Recraft. The curated model reported 98.3 real-image true-negative
  rate, 94.4 mean generated-image true-positive rate, 99.0 ROC AUC and 99.0 PR
  AUC. Individual fake accuracies still ranged from 72.4 on Ideogram to 100 on
  several generators.
- **Confirmed:** With the same 10K/eight-generator budget, representation-based
  selection reported 96.4 RealWorldBench accuracy versus a ten-run 94.9 average
  from random generator selection. The full 50K/28-generator set was slightly
  higher at 96.0 in a different table's aggregate and 95.3 mean fake TPR, so the
  compact set did not dominate every metric.
- **Confirmed:** The PE-Core-G14-448 vision tower used by SSAFE is officially
  documented at 1.88B parameters. SSAFE uses no text encoder, so its frozen
  image tower plus a 1,280-dimensional linear head is nominally below Track 5's
  2B ceiling, but only narrowly and subject to the organizer's complete-system
  counting rule.
- **Confirmed:** The preprint says code, curated generator lists and evaluation
  scripts "will be released upon publication"; no SSAFE implementation or
  checkpoint was located. It does not report the Track 5 distortions or a
  compound-transformation benchmark.
- **Derived:** SSAFE is strong evidence that modern real-image coverage and
  generator selection can matter more than raw dataset size. It is not yet a
  reproducible detector, and its near-limit backbone may create practical
  memory/latency constraints even if the rounded parameter count is accepted.

Primary sources:

- SSAFE preprint: <https://arxiv.org/html/2606.08634>
- Official Perception Encoder model documentation:
  <https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md>

### Any-resolution spectral modelling: SPAI

- **Confirmed:** SPAI models the spectral distribution of real images rather
  than learning one named generator's fingerprint. A frozen ViT-B/16 is first
  trained by masked frequency reconstruction; at detection time the method
  combines reconstruction similarity, a context vector and spectral-context
  attention over native-resolution patches. Its stated complexity is linear in
  the number of patches.
- **Confirmed:** The paper trains on 180,000 real and 180,000 latent-diffusion
  images for 35 epochs with 224-pixel patches and resize, crop, rotation,
  Gaussian blur, noise and JPEG augmentations. It reports use of one NVIDIA
  L40S 48 GB GPU.
- **Confirmed:** Across the paper's 13-generator/five-real-source protocol,
  SPAI reported 91.0 mean AUC, versus 85.5 for RINE, 83.5 for DMID and 80.4 for
  PatchCraft. Its robustness study separately applied JPEG or WebP qualities
  85/70/50, blur kernels 3/5/7, noise standard deviations 1/3/5 and resize
  factors 85/70/50 percent.
- **Confirmed:** Ablation results were 52.5 AUC without spectral pretraining,
  71.0 without spectral reconstruction similarity, 83.2 without spectral
  context attention and 84.2 without distortion augmentation, versus 91.0 for
  the complete reported system. Chromatic augmentation reduced the result to
  80.5 in that experiment.
- **Derived:** The any-resolution mechanism does not imply fixed compute: more
  patches still require more work. The reported transformations are primarily
  individual rather than the 1--5 chained setting used by NTIRE 2026.
- **Confirmed:** The official repository and its model weights are declared
  Apache-2.0. The audited revision is
  `8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4`; its README states inference fits
  below 8 GB GPU memory while paper-style training targeted one 48 GB L40S.
- **Unknown:** The inspected paper and repository do not give a complete
  parameter count. A memory statement is not a substitute for parameter
  inventory, and the required pretrained MFM checkpoint has its own upstream
  provenance to audit.

Primary sources:

- Paper: <https://arxiv.org/html/2411.19417>
- CVPR 2025 page:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html>

### Repeated transmission and physical re-digitization: RRBench

- **Confirmed:** RRDataset/RRBench contains 10,000 real and 10,000 generated
  originals across everyday and six special scenarios. Its generated sources
  include SD 1.4/1.5/3.5, Flux, DALL-E 3, Midjourney, Chameleon, StyleGAN and
  ProGAN.
- **Confirmed:** Each image was transmitted two to six times through Telegram,
  WeChat, Facebook, QQ, WhatsApp, X, Instagram or Tinder. Four additional
  re-digitization modes scan a printout, photograph a printout, photograph a
  display or photograph a projection.
- **Confirmed:** Seventeen detectors and ten vision-language models were
  evaluated. Detector checkpoints were first pretrained on GenImage SD 1.4 and
  fine-tuned on a subset of RRDataset, so the table is not a pure zero-shot
  comparison.
- **Confirmed:** DRCT-ConvB had the highest reported overall score, 89.59, with
  fake/real accuracies 93.52/95.52 on originals, 92.82/95.09 after transmission
  and 64.34/96.22 after re-digitization. AIDE reported 78.42 overall, with
  78.95/78.94, 74.72/78.75 and 76.04/83.13 across the same conditions.
- **Confirmed:** DIRE remained strong after transmission but its generated-
  image accuracy collapsed from 89.72 to 1.42 after re-digitization; DNF fell
  from 90.62 to 0.05. Frequency-oriented systems and SAFE also suffered large
  transmission losses in the reported tables.
- **Confirmed:** The prose calls AIDE's re-digitized fake-image decline 1.89
  percentage points, but its own table gives 78.95 minus 76.04, which is 2.91
  points. The table arithmetic is preserved here and the contradiction is not
  silently resolved.
- **Confirmed:** A study with 192 participants found human fake-image accuracy
  of 39.64 percent in special scenarios versus 58.29 percent in everyday
  scenarios. This is evidence that content plausibility can mislead people, not
  evidence about Track 5's scoring distribution.
- **Derived:** Re-digitization is a distinct threat from digital JPEG/resize
  augmentation and can erase detector cues while introducing camera, display
  or printer traces. TikTok has not said that its hidden tests include any
  physical re-capture, so this is a robustness boundary, not a test-set claim.
- **Confirmed:** Zenodo record 14963880 labels the dataset CC-BY-4.0 and lists
  `RRDataset_original_train_val.tar.gz` as 2,163,176,547 bytes with MD5
  `2f4498c3690d8f4c7a30d2e41dd34500`, and `RRDataset_test.tar.gz` as
  20,117,869,400 bytes with MD5 `13c3ff3d61986170cc0c8cf76a35cd4b`.
  The separate CC-BY-4.0 weight record 14991882 lists a 6,957,228,544-byte
  `WEIGHT.tar` with MD5 `0d82f6aae0268bed0ff7d899219708db`.
- **Confirmed:** The official RRDataset code repository is MIT-licensed at
  revision `a6d1f33b5d9a45628d088f9aee5c809d1d529b92`. Neither the roughly
  22.3 GB dataset nor the 7.0 GB weight archive was downloaded.

Primary sources:

- Paper: <https://arxiv.org/html/2509.09172>
- ICCV 2025 page:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.html>
- Dataset metadata: <https://zenodo.org/records/14963880>
- Model-weight metadata: <https://zenodo.org/records/14991882>
- Official code: <https://github.com/ChunXiaostudy/RRDataset>

### Frontier-generator and high-risk content gap: SafeIMG

- **Confirmed:** The July 2026 SafeIMG preprint contains 1,131 images generated
  by GPT Image 2 across 12 public- and individual-safety scenarios, plus 600
  category-balanced real images. Of the generated set, 993 have 2,652
  human-marked suspicious regions and 138 are a "hard" subset for which
  annotators could not identify a defensible visual anomaly.
- **Confirmed:** The strongest tested general-purpose vision-language model
  detected 49.5 percent of generated images, the strongest of the three tested
  specialized detectors detected 33.1 percent, and three human evaluators
  averaged 81.7 percent. This "accuracy" is generated-class recall, not balanced
  binary accuracy; the paper separately shows large real-versus-fake trade-offs.
- **Confirmed:** The specialized-detector comparison is narrow and old:
  CNNSpot, FreDect and LNP. It does not test Community Forensics, SSAFE, B-Free,
  SPAI, RINE, AIDE or recent NTIRE systems. Therefore it demonstrates severe
  failure of those evaluated systems on GPT Image 2, not failure of every
  current detector.
- **Confirmed:** Under progressively repeated JPEG/degradation processing,
  Claude Opus 4.6 generated-image recall fell from 32.6 to 2.9 percent, and
  Claude Sonnet 4.6 reached 0.4 percent. Qwen3 Plus retained 11.6 from 24.1.
  These are vision-language-model results, not pixel-detector robustness
  results.
- **Confirmed:** Model explanations covered only 29.8 percent of human-marked
  anomalies. Coverage was 64.71 percent for hands, 58.67 for text and 56.25 for
  faces, but 15.01 for commonsense and 12.00 for physics violations. Correct
  image-level classification therefore does not imply a correct explanation.
- **Confirmed:** The public GitHub repository has no detected licence at audited
  revision `3fbf8d83290f267c958f41a53e00e5ac134400ac`; the Hugging Face dataset card
  also declared no licence at revision
  `389a7c555191c9e946727a470500fcd9b045381c`. The paper HTML itself is
  CC-BY-NC-SA-4.0, which does not automatically license the dataset or code.
- **Derived:** SafeIMG reinforces that generator recency, content type, class
  bias and post-sharing degradation can dominate a detector score. It should
  not be used to infer that Track 5 contains GPT Image 2 or high-risk scenarios.

Primary sources:

- Paper: <https://arxiv.org/html/2607.22745>
- Project: <https://safeimg.github.io/>
- Code: <https://github.com/Snowstorm1492/SafeIMG>
- Dataset metadata: <https://huggingface.co/datasets/Snowstorm1492/SafeIMG>

### Large and diverse training collections: Community Forensics

- **Confirmed:** Community Forensics systematically collected 4,763 latent-
  diffusion models and added 30 manually selected open or commercial models,
  producing 2.7 million generated images. It paired these with 2.7 million real
  images, for 5.4 million training examples in total.
- **Confirmed:** Its reported detectors are deliberately conventional: a CLIP
  ViT or ConvNeXt image encoder followed by a sigmoid binary head, trained
  end-to-end at 224 or 384 pixel resolution. Freezing the pretrained encoder
  was consistently worse in the authors' experiments; switching between the
  tested ViT and ConvNeXt families mattered less than broadening the training
  model collection.
- **Confirmed:** Generalization improved as generator diversity increased and
  began to flatten only after roughly 1,000 collected models. Diversity across
  generator architecture families also mattered, rather than merely adding
  many near-duplicate checkpoints.
- **Confirmed:** The authors expanded augmentation beyond JPEG, blur and crop
  to padding, resizing, rotation, shear and randomized sequences. Their
  high-resolution model reported 0.986 mean average precision and 0.923 mean
  accuracy over five in-paper benchmarks.
- **Confirmed:** Those high in-paper figures do not contradict the later 2026
  zero-shot benchmark's 0.780 mean accuracy for its released Community-
  Forensics detector: the evaluation versions, datasets and protocols differ.
  This is direct evidence that apparently simple cross-paper score comparisons
  are unsafe.
- **Confirmed:** Some real-image sources used in the comprehensive evaluation
  cannot be redistributed. The paper's public reconstruction substitutes COCO
  and FFHQ pairs and warns that external links can decay, so dataset provenance
  and the exact evaluation edition are necessary to reproduce a claim.

Primary source:

- Community Forensics: <https://arxiv.org/html/2411.04125>

### Complementary semantic and pixel-artifact clues: CO-SPY

- **Confirmed:** CO-SPY reports that low-level artifact detectors transferred
  reasonably across content and generators but lost about 17 percentage points
  under JPEG qualities 75--95; semantic detectors lost only about two points to
  that compression but approached chance on newer unseen generators or changed
  content. Its premise is therefore cue complementarity, not one universal
  artifact.
- **Confirmed:** Its semantic branch uses a frozen CLIP ViT-SO400M-14-384 with
  interpolated semantic examples. Its artifact branch reconstructs the image
  with a Stable Diffusion 1.5 VAE, computes the absolute reconstruction error,
  and encodes that error with ResNet-50. Learned regulators weight the branches,
  random branch dropout discourages dependence on only one, and a classifier
  operates on their concatenated features.
- **Confirmed:** Co-SPYBench contains over one million images spanning five real
  sources, 22 generator models, 50,000 in-the-wild images and JPEG qualities
  75--95. It also evaluates blur, resize, noise, brightness, saturation and
  contrast changes.
- **Confirmed:** In three paper tables, raw/JPEG accuracy for AIDE versus CO-SPY
  was 92.77/73.08 versus 87.75/79.76 on AIGCDetect, 61.93/55.24 versus
  67.63/63.19 on Chameleon, and 85.15/74.61 versus 91.45/87.06 on
  Co-SPYBench. This is a trade-off, not uniform dominance: AIDE was better on
  clean AIGCDetect, while CO-SPY was more stable after compression.
- **Confirmed:** The artifact branch was less effective for pixel-space
  diffusion generators because those generators do not use the latent VAE
  decoding operation whose reconstruction behavior the branch exploits.
- **Unknown:** The inspected paper text does not state the complete submitted
  parameter count. A backbone's name is not sufficient evidence that the whole
  system satisfies Track 5's model-size rule.

Primary source:

- CO-SPY: <https://arxiv.org/html/2503.18286>

### Hybrid patch statistics and semantics: AIDE

- **Confirmed:** AIDE selects 32-pixel patches with the highest and lowest DCT
  frequency scores. Two ResNet-50 branches apply fixed Spatial Rich Model
  high-pass filters to characterize their noise/artifact patterns, while a
  frozen OpenCLIP ConvNeXt-XXLarge branch supplies whole-image semantic
  features. Projected features are concatenated for binary classification.
- **Confirmed:** On the paper's ProGAN-trained AIGCDetectBenchmark protocol,
  AIDE reported 92.77% average accuracy over 16 generator sets, versus 89.31%
  for PatchCraft. On GenImage, where all models trained on SD 1.4, it reported
  an 86.88% average and a 4.6-point improvement over PatchCraft.
- **Confirmed:** JPEG/blur robustness was still incomplete. Its 92.77% clean
  AIGCDetectBenchmark average fell to 75.54% at the first reported JPEG setting
  and 69.60% at the strongest one; its Gaussian-blur averages ranged down to
  79.86% in the reported table.
- **Confirmed:** Chameleon contains 11,170 carefully curated generated images
  that human annotators mistook as real plus 14,863 real images, with most at
  720p--4K. The paper says nearly all evaluated detectors approached random
  accuracy and explicitly acknowledges that even AIDE remained weak at finding
  Chameleon's generated images.
- **Confirmed:** The repository's code is MIT-licensed, but its README assigns
  Chameleon a separate academic-research-only, no-commercial-use restriction.
  Code permission must not be confused with data permission.
- **Unknown:** The paper and audited code do not state the complete parameter
  count. The two ResNet-50s plus ConvNeXt-XXLarge must be counted as an entire
  inference system before asserting Track 5 compliance.

Primary sources:

- ICLR 2025 paper: <https://arxiv.org/html/2406.19435>
- Official code: <https://github.com/shilinyan99/AIDE>

### Hard reconstructed examples: DRCT

- **Confirmed:** DRCT reconstructs both real and generated training images with
  a Stable Diffusion model, forming hard examples that look closer across the
  real/fake boundary. It adds a margin-based contrastive loss to ordinary binary
  classification and can wrap multiple detector backbones.
- **Confirmed:** DRCT-2M contains two million images. Its generated portion has
  16 Stable Diffusion variants with 120,000 images per type: ten text-to-image,
  three ControlNet and three diffusion-reconstruction variants; real images and
  prompts are based on MS COCO. DRCT-2M-Wild adds images collected from Civitai
  and Midjourney Discord for separate practical evaluation.
- **Confirmed:** With the paper's aligned protocol, DRCT/ConvNeXt-Base improved
  average DRCT-2M accuracy from 79.11% to 90.79% using SD1 reconstruction and
  96.55% using SD2. DRCT/UnivFD reached 90.49% and 91.35%, respectively. On
  GenImage, DRCT added 7.1 points to ConvNeXt-Base and 10.04 to UnivFD.
- **Confirmed:** The paper's post-processing test used resize scales
  0.5--1.5 and JPEG qualities 60--100 after all compared systems received the
  same augmentation. It does not test the full Track 5 transformation list or
  chained attacks.
- **Confirmed:** The official repository supports ConvNeXt, CLIP, ResNet and
  other backbones, but has no declared software licence at the audited commit.
  It also requires a diffusion generator during preparation of reconstructed
  training images, even though the final detector need not run that generator.
- **Derived:** This is evidence for hard-example construction, not evidence that
  its two-million-image generation cost is feasible in the hackathon or that
  its published checkpoint satisfies the complete-system limit and licences.

Primary sources:

- ICML 2024 paper: <https://proceedings.mlr.press/v235/chen24ay.html>
- Official code: <https://github.com/beibuwandeluori/DRCT>

### Rich-versus-poor texture contrast: PatchCraft

- **Confirmed:** PatchCraft deliberately breaks global layout by smashing an
  image into patches, ranks them by texture diversity, reconstructs rich- and
  poor-texture groups, applies high-pass filters and classifies the contrast in
  neighboring-pixel correlations. The premise is that generated images do not
  reproduce camera/processing noise relationships equally across those regions.
- **Confirmed:** Its paper reports 89.85% average accuracy across 17 generators
  when trained on ProGAN, compared with 76.80% for UnivFD in the same table.
  Under separate distortion tests, it reported 72.48% for JPEG, 78.36% for
  downsampling and 75.99% for blur. It was not best on clean or JPEG in every
  comparison, and later Chameleon evaluation showed a large failure on curated
  high-quality images.
- **Confirmed:** The official repository contains 7,390 Git blobs and embeds
  thousands of example/train/test images; GitHub reports about 1,820,936 KB
  repository size. It has no declared licence at the audited commit.
- **Derived:** Because the public repository mixes code with a large image
  corpus and has no licence, it must not be cloned into this preparation repo or
  treated as reusable source. Paper-level ideas can be studied independently.

Primary sources:

- PatchCraft paper: <https://arxiv.org/html/2311.12397>
- Official repository inventory: <https://github.com/cvlcgabriel/PatchCraft>

### Preserving and augmenting local artifacts: SAFE

- **Confirmed:** SAFE argues that preprocessing can erase the signal a detector
  needs. It replaces resizing with random crop during training and centre crop
  during inference, applies color jitter and arbitrary-angle rotation to reduce
  content/color shortcuts, and masks random local patches so the classifier
  cannot rely on one global region. A discrete wavelet transform exposes
  high-frequency content to a small ResNet.
- **Confirmed:** The reported detector has 1.44 million parameters and 2.30
  billion FLOPs. It trains for 20 epochs on four H800 GPUs using four ProGAN
  classes, then evaluates 33 subsets from 26 generator models. Its aggregate
  result was 96.7 mean accuracy and 99.3 mean average precision, compared with
  92.2/96.4 for a 577.25M-parameter FatFormer and 81.0/94.1 for UnivFD in that
  specific protocol.
- **Confirmed:** On the paper's separate DiTFake construction, SAFE reported
  99.4 mean accuracy and 99.9 mean average precision over Flux, PixArt and SD3,
  even though it had trained on ProGAN. The generated images were 1024-square
  while the COCO real images were mostly about 640 by 480; the method crops
  rather than resizes, but residual source differences make independent
  cross-dataset validation important.
- **Confirmed:** The authors state that the detector was deployed on a platform
  handling about nine million user items per day and improved recall volume by
  roughly 46--56 percent at fixed 95--99 percent precision relative to their
  earlier frequency baseline. This is a paper-reported production experiment,
  not a public reproduction or disclosure of a commercial system's complete
  data distribution.
- **Confirmed:** Blur augmentation kept the reported ForenSynths average near
  85 accuracy/90 average precision once blur sigma exceeded one, but JPEG caused
  a significant drop because compression destroys the high-frequency wavelet
  clue. The authors explicitly list unknown perturbations as a limitation.
- **Confirmed:** The 2026 zero-shot benchmark later found SAFE accuracy ranging
  from 0.032 to 0.998 across datasets. This does not invalidate its in-paper
  experiments; it shows that a named method and headline score are inseparable
  from checkpoint, threshold and dataset.
- **Confirmed:** The official code is Apache-2.0 and includes a 5,840,638-byte
  checkpoint. It was audited through GitHub metadata only; no weight was
  downloaded or copied into this repository.

Primary sources:

- KDD 2025 paper: <https://arxiv.org/html/2408.06741>
- Official repository: <https://github.com/Ouxiang-Li/SAFE>

### Decision-boundary correction: post-hoc calibration

- **Confirmed:** The AAAI 2026 calibration paper argues that a fixed 0.5
  threshold can become wrong when the generated-image distribution or class
  balance changes. It freezes the detector and learns only a scalar logit shift.
- **Confirmed:** Its supervised version estimates the shift from a small labelled
  target subset. Its unsupervised version applies Gaussian kernel-density
  estimation to unlabelled target logits and seeks a symmetry-based boundary.
  The paper reports useful results with as few as ten target samples, but also
  includes isolated cases where calibration slightly reduced accuracy.
- **Confirmed:** The method assumes the real-image distribution is stable and
  that generated-image shifts are coherent enough for a scalar correction.
  These conditions are not guaranteed for a mixed, hidden challenge set.
- **Confirmed:** Its official repository has no declared licence at the audited
  commit and includes a 4.16 MB JSON file of CNNSpot logits. It is evidence and
  reference code, not presently reusable competition code.
- **Derived:** Calibration can change thresholded accuracy without improving
  rank-based AUC. Moreover, using organizer test or demo data to estimate a
  target threshold could violate data-use rules. Track 5's final metric and
  permitted calibration data are still unknown, so no such adaptation is
  authorized in preparation.

Primary sources:

- AAAI 2026 paper: <https://arxiv.org/html/2602.01973>
- Official code: <https://github.com/muliyangm/AIGI-Det-Calib>

### Lightweight low-order-bit noise: LOTA

- **Confirmed:** LOTA treats low-order bit planes as a proxy for intrinsic
  image noise, normalizes them, selects the patch with greatest gradient energy
  and classifies that patch with a small network. It reports 23.6 million
  parameters for NBC and 28.4 million for NGC variants.
- **Confirmed:** On the paper's hardware and comparison setup, reported error-
  map extraction took 1.52 ms and total inference 4.00 or 4.71 ms. The same
  table lists DIRE at 688.3M parameters/about 2 s, LaRE2 at 1,165.8M/about
  260 ms and ESSP at 30.7M/about 31.99 ms. These are paper-reported figures,
  not local reproduction measurements.
- **Confirmed:** LOTA reports 98.9% average accuracy on GenImage and strong
  cross-GAN/diffusion results. Its distortion study covers Gaussian blur sigma
  0--3 and JPEG quality 100, 95, 90 and 85, and reports greater stability than
  LaRE2 and ESSP under those settings.
- **Derived:** This evidence covers only mild JPEG compression and individual
  distortions, not the stronger or chained transformations that Track 5 might
  choose. The 2026 zero-shot benchmark also found LOTA's relative position
  changed substantially by dataset. High GenImage accuracy is therefore not
  evidence of universal real-world performance.

Primary source:

- LOTA: <https://arxiv.org/html/2510.14230>

### Compression-stable phase information: CPTFormer

- **Confirmed:** CPTFormer begins from the JPEG observation that coefficient
  quantization changes or removes magnitude information, while the phase of a
  non-zero coefficient is preserved. A lightweight convolutional pyramid
  extracts multi-scale phase features, and bidirectional cross-attention fuses
  them with RGB tokens from a pretrained CLIP-ViT.
- **Confirmed:** Multi-domain modulation adapters refine the fused features in
  spatial and wavelet domains. A difficulty-aware consistency loss is optional
  when a limited set of original/compressed pairs is available.
- **Confirmed:** Its main protocol trains on ProGAN examples from ForenSynths,
  with 80% pristine unpaired data and 20% pristine/JPEG-quality-40 pairs, then
  evaluates cross-generator GAN and diffusion datasets. Quality-agnostic test
  compression randomly varies its ratio.
- **Confirmed:** The paper reports mean compressed accuracy of 76.3% for GANs
  and 63.5% for diffusion models in its two-class quality-agnostic setting; in
  its quality-aware setting it reports 77.4% and 65.9%. These values are bound
  to its accuracy metric, ProGAN training source and compression protocol.
- **Confirmed:** The paper states that the adapters provide parameter-efficient
  fine-tuning, but does not report a complete model parameter count in the
  inspected text. Compliance with Track 5 cannot yet be established.
- **Derived:** Phase information addresses JPEG robustness, not automatically
  crop, blur, color, unseen-generator or compound-transform generalization. Its
  single-source ProGAN training protocol is also materially narrower than
  Community Forensics or NTIRE 2026.

Primary source:

- CVPR 2026 paper and official PDF:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Li_Detecting_Compressed_AI-Generated_Images_via_Phase_Spectrum_Robustness_CVPR_2026_paper.html>

### Passive detection versus deliberate provenance signals

- **Confirmed:** A passive detector receives an arbitrary image and infers its
  origin from learned clues already present in the pixels. Reconstruction-based
  methods are another passive variant: DIRE, for example, sends an image through
  a diffusion reconstruction process and uses the reconstruction change as a
  clue; AEROBLADE uses the autoencoder reconstruction behavior of latent
  diffusion models. Passive detection is necessary for legacy images and images
  produced by generators that add no cooperative identifier.
- **Confirmed:** A watermark detector solves a different, easier problem. The
  generator deliberately embeds an invisible pattern and a corresponding
  detector later searches for it. Google's SynthID uses jointly trained
  watermarking and identification models and says its image mark is designed to
  survive filters, color/brightness changes and lossy compression, while
  explicitly stating that extreme manipulation can defeat it.
- **Confirmed:** ImageDetectBench compared five passive and four watermark-based
  systems on four datasets, eight common perturbations and three adversarial
  attacks. In that controlled study, watermark systems generally outperformed
  passive systems. Under JPEG quality 40, Smoothed StegaStamp scored 1.00
  accuracy versus 0.88 for the strongest passive detector in that experiment.
  However, every tested detector failed under sufficiently strong white-box
  attacks, and the authors explicitly say watermarks cannot identify legacy or
  deliberately unwatermarked generated images.
- **Confirmed:** C2PA Content Credentials are cryptographically signed records
  of an asset's origin and edit history. They can state that AI was used and
  reveal later tampering, but C2PA explicitly says provenance is not itself a
  judgment that pictured events are true. Embedded credentials can be removed;
  durable credentials can use watermark or fingerprint lookup to rediscover the
  signed record. Absence of a credential is not evidence that an image is fake.
- **Derived:** Watermark and C2PA checking can be a high-confidence auxiliary
  provenance layer, but cannot replace the pixel-only detector expected by a
  challenge whose hidden images may come from arbitrary sources. Whether Track
  5 permits or rewards such auxiliary signals is not specified.

Primary sources:

- ImageDetectBench: <https://arxiv.org/html/2411.13553>
- Google SynthID overview: <https://deepmind.google/blog/identifying-ai-generated-images-with-synthid/>
- C2PA 2.2 explainer: <https://spec.c2pa.org/specifications/specifications/2.2/explainer/Explainer.html>

### What is known about commercial detector apps

- **Confirmed:** Hive, Optic/AI or Not, Illuminarty and Sightengine provide
  black-box web/API detectors. Published studies can measure their outputs but
  generally cannot establish their undisclosed network architectures. Claims
  that they inspect a specific kind of artifact must therefore remain
  unverified unless the vendor or an auditable implementation documents it.
- **Confirmed:** Sightengine states that its current detector uses image pixels,
  not EXIF, C2PA or visible watermarks, is continuously updated for new
  generators, and can struggle on unseen generators or heavily edited images.
  Its claimed per-generator coverage and 2026 updates are vendor claims, not an
  independent disclosure of how the model works.
- **Confirmed:** The 2024 CCS paper *Organic or Diffused* evaluated 280 human
  artworks and 350 images from five generators. On clean images, Hive scored
  98.03% accuracy, 0% false positives and 3.17% false negatives; Optic scored
  90.67% but with 24.47% false positives. This result is narrow: it concerns
  matching creative-art samples from the 2023--24 generator landscape.
- **Confirmed:** In the same study, Hive's generated-image detection rate fell
  from 96.83% clean to 91.88% with JPEG quality 15, 88.73% with mild Gaussian
  noise, and 67.56% after high-intensity Glaze. A later overlay test found that
  some samples flipped Hive's verdict with only a 10% wall-image overlay, while
  most required 60--80%. Results changed between April and June 2024, showing
  that the commercial endpoint itself was changing.
- **Confirmed:** A separate ARIA study tested 144,175 paired real/generated
  images. Sightengine was strongest on its human and pure text-to-image columns,
  but fell sharply on some image-plus-text generation columns (for example
  25.12% on Midjourney and 43.76% on DALL-E in the paper's table). Most tested
  commercial detectors performed worse when a real seed image was used.
- **Confirmed:** ImageDetectBench's smaller 100-real/100-generated commercial
  sample measured Hive at 1.00 clean accuracy and 0.88 after Gaussian noise;
  Illuminarty fell from 0.80 to 0.57. These three studies use different data and
  perturbation settings, so their percentages cannot be directly ranked.
- **Derived:** The credible explanation for why a commercial app can look
  exceptionally good is continual access to large, current, diverse labelled
  data and possibly multiple specialized models. There is no evidence that any
  reviewed commercial app has discovered an infallible universal signature.

Primary sources:

- CCS 2024 commercial/art detector study: <https://arxiv.org/html/2402.03214>
- ARIA benchmark paper: <https://arxiv.org/pdf/2404.14581>
- Sightengine's documented current claims: <https://sightengine.com/detect-ai-generated-images>
- ImageDetectBench commercial comparison: <https://arxiv.org/html/2411.13553>

### Testing the next generation rather than only a static test set: AI-GenBench

- **Confirmed:** AI-GenBench orders 36 generators chronologically from CycleGAN
  in March 2017 through FLUX.1 Dev/Schnell in August 2024, then groups them into
  nine four-generator windows. A detector is incrementally trained on all
  windows available up to one point in time and evaluated separately on past,
  next and combined periods.
- **Confirmed:** The balanced benchmark contains 180,000 generated and 180,000
  real images. Each generator contributes 4,000 training and 1,000 evaluation
  fakes; the real side comes from ImageNet, COCO 2017, LAION-400M and RAISE and
  is partitioned correspondingly. It is assembled from many source datasets,
  not one newly generated collection with one uniform licence.
- **Confirmed:** The protocol fixes each training image to four augmented
  versions and applies one deterministic evaluation pipeline containing
  compression, blur, noise, variable resize and crop. It forbids extra training
  data and models pretrained specifically for detection, while allowing
  general-purpose open-weight foundation backbones. These are AI-GenBench rules,
  not TikTok Track 5 rules.
- **Confirmed:** In the paper's next-period resize results, full fine-tuning
  reported 94.24 AUROC/84.09 accuracy for ViT-L/14 DINOv2, 92.04/85.28 for
  ViT-L/14 CLIP and 81.77/73.42 for ResNet-50 CLIP. Linear probing and five-crop
  evaluation were generally weaker. The detector was carried forward from the
  preceding window and trained one epoch at each step.
- **Confirmed:** The largest time-localized failure occurred when systems that
  had learned through the preceding window first faced Stable Diffusion
  1.4/1.5/2.1 and DeepFloyd IF. A smaller decline accompanied the earlier
  VQ-Diffusion/Denoising-Diffusion-GAN/GLIDE/Latent-Diffusion transition. This
  is evidence that a new generator family can break learned fingerprints even
  when average historical performance is high.
- **Confirmed:** The official leaderboard inspected on 2026-08-27 contains
  only the paper authors' July 2025 baselines. Their reported past/next/whole
  AUROCs are 99.1/94.2/97.9 for 304M-parameter ViT-L/14 DINOv2,
  98.1/92.0/97.0 for 304M-parameter ViT-L/14 CLIP, and 89.9/81.8/88.9 for
  38M-parameter ResNet-50 CLIP. No external winning submission can be inferred
  from that page.
- **Confirmed:** On one RTX 3080 Ti with an Intel i9-10900X, the authors report
  full benchmark-training times of 26 hours for tuned DINOv2 ViT-L/14, 19.87
  hours for tuned CLIP ViT-L/14 and 4.87 hours for tuned ResNet-50. These are
  paper measurements, not local reproduction or Track 5 resource estimates.
- **Confirmed:** The official code is BSD-3-Clause at audited revision
  `e0f673b6d70d989310b82a8f6a891df2cd085925`. GitHub reports repository size
  20,703 KB; the complete recursive tree was not truncated and contains 382
  entries totalling 33,259,016 blob bytes. No repository was cloned.
- **Confirmed:** The convenient fake-only Hugging Face package is revision
  `75204cf1aa28b61885eae43e883f72e28b8911b2`, has 144,000 training and 36,000
  validation examples, and its 74 files total 35,178,884,837 metadata-reported
  bytes. Its card declares no single licence and instead tells users to audit
  every origin dataset. No file or image was downloaded.
- **Derived:** DINOv2 outperforming CLIP under this controlled temporal protocol
  does not contradict SSAFE's frozen-representation result favouring multimodal
  encoders: the data, training mode, dates and evaluation protocol differ. The
  benchmark ends in August 2024 and therefore cannot by itself establish
  performance on 2025--2026 generators or the Track 5 hidden set.

Primary sources:

- IJCNN 2025 accepted paper: <https://arxiv.org/html/2504.20865>
- Official benchmark and current leaderboard:
  <https://mi-biolab.github.io/aigenbench-website/>
- Official code and dataset-construction instructions:
  <https://github.com/MI-BioLab/AI-GenBench>
- Official fake-only dataset metadata:
  <https://huggingface.co/datasets/lrzpellegrini/AI-GenBench-fake_part>

### Controlled implementation choices on AI-GenBench

- **Confirmed:** A November 2025 v1 preprint reuses AI-GenBench to isolate
  augmentation, training duration, crop/resize handling, generator-label
  supervision and incremental updates across the same DINOv2 ViT-L/14, CLIP
  ViT-L/14 and CLIP ResNet-50 backbones. It explicitly presents an empirical
  configuration study, not a new detector architecture.
- **Confirmed:** The baseline training pipeline used strong color, geometric,
  noise, blur, dropout and one JPEG transformation. A milder pipeline modelled
  the benchmark's delivery degradations with up to three successive JPEG passes.
  Mean next-period AUROC across the three backbones was 90.1 for the baseline,
  93.1 for a mild single-JPEG variant and 94.5 for the repeated-JPEG pipeline.
- **Confirmed:** Extra augmentation diversity and extra passes over the same
  augmented data behaved similarly in this protocol. For DINOv2, augmentation
  multiplier 8 for one epoch gave 96.36 next-period AUROC, while multiplier 4
  for two epochs gave 96.38. The official benchmark nevertheless caps the
  multiplier at four for fairness; those larger-multiplier experiments are not
  eligible leaderboard settings.
- **Confirmed:** Input handling was architecture-dependent. Resize training and
  resize evaluation scored 94.22/90.79/85.21 for DINOv2/ViT-L CLIP/ResNet-50.
  Crop training with mixed five-crop-plus-resize evaluation improved DINOv2 to
  94.95 but reduced the two CLIP backbones to 87.48 and 82.72. This directly
  contradicts a universal claim that crop, resize or mixed inference is always
  superior.
- **Confirmed:** Direct binary training outperformed plain generator-multiclass
  training followed by sum or max fusion on all three backbones. A separate
  auxiliary multiclass head helped ViT-L CLIP (90.79 to 92.35), was neutral for
  DINOv2 (94.22 versus 94.21), and hurt ResNet-50 (85.21 to 84.09). Generator
  labels were therefore not uniformly beneficial.
- **Confirmed:** In its continual-update experiment, training only on the new
  window caused forgetting; replaying earlier examples restored much of the
  past-period performance. Harmonic replay reached next/past AUROC
  93.76/98.88 for DINOv2 versus 94.22/99.24 for full accumulated-data retraining,
  while the paper estimates 63.26 percent of average per-window compute. These
  are simulated benchmark-update results, not a Track 5 requirement.
- **Confirmed:** Combining its tested choices on DINOv2 produced a paper-
  reported 97.36 average next-period AUROC. The live AI-GenBench leaderboard
  inspected on 2026-08-27 does not list this as a separate submission; it still
  shows only the July 2025 author baselines topped by 94.2. The 97.36 value must
  therefore be described as a preprint result, not a verified leaderboard win.
- **Confirmed:** The source is arXiv v1 submitted 2025-11-26 and does not state
  peer-review acceptance. Its code resides in the already audited BSD-3-Clause
  AI-GenBench repository revision
  `e0f673b6d70d989310b82a8f6a891df2cd085925`. No configuration was run or copied.
- **Derived:** The study shows that large score changes attributed to a named
  detector can come from data handling and optimization. Because its benchmark
  ends with August 2024 generators and uses one fixed degradation pipeline, its
  empirical choices cannot be promoted to universal or Track 5-optimal rules.

Primary sources:

- Preprint: <https://arxiv.org/html/2511.21507>
- Current official benchmark leaderboard:
  <https://mi-biolab.github.io/aigenbench-website/>
- Pinned code repository: <https://github.com/MI-BioLab/AI-GenBench>

### Updating as generators change and detecting local edits

- **Confirmed:** Adobe's ICCV 2023 workshop paper simulated an online detector
  over 14 generators released or sampled from June 2020 through March 2023. Its
  570,221-image collection was split into 405,862 train, 48,057 validation and
  116,302 test images, with LAION images as the fixed real class.
- **Confirmed:** The whole-image detector was an ImageNet-pretrained ResNet-50.
  At every time step, training continued from the preceding detector and
  included all historical generated images seen so far, using class-balanced
  sampling. Training used 256-pixel crops plus low-probability blur, grayscale
  and invisible-watermark augmentation; evaluation used an unaugmented centre
  crop.
- **Confirmed:** Once each generator was added to training, the paper reported
  1.00 ROC AUC and at least 0.96 generated-class accuracy on its held-out data.
  Historical coverage often produced high rank separation on later generators,
  but the decision threshold still failed: immediately before Firefly examples
  were added, its figure reports 0.92 AUC but only 0.35 generated-class
  accuracy; direct Firefly training raised both to about 1.00/0.99.
- **Confirmed:** This is a concrete example of a detector knowing which images
  look more suspicious while using the wrong cutoff for a new generator. It
  supports continual evaluation and calibration, not a claim that historical
  training automatically solves future detection.
- **Confirmed:** The paper separately trained a pixel-level ResNet-50 FCN to
  locate inpainted regions covering 15--35 percent of an image. On Stable
  Diffusion 1 inpainting, training on whole generated images produced F1
  0.1807; CutMix improved it to 0.4882; direct inpainting examples reached
  0.9795; and direct examples plus CutMix reached 0.9832.
- **Confirmed:** For Firefly inpainting, Firefly whole images converted to
  CutMix regions produced F1 0.4395. Adding actual SD1/SD2 inpainting examples
  raised it to 0.8886, while direct Firefly inpainting examples reached 0.9600
  and all three direct sources reached 0.9772. Whole-image detection and local-
  edit localization are therefore materially different tasks.
- **Derived:** The study predates current generators and uses one real-image
  source, a simple preprocessing pipeline and no Track 5 compound-transform
  protocol. It is evidence for temporal evaluation and partial-edit boundaries,
  not a current benchmark leader or a detector selection.

Primary source:

- ICCV 2023 workshop paper:
  <https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Epstein_Online_Detection_of_AI-Generated_Images__ICCVW_2023_paper.pdf>

## Comparable competitions and best reported systems

### NTIRE 2026 Robust AI-Generated Image Detection in the Wild

- **Confirmed:** This CVPR 2026 workshop challenge is the closest located prior
  competition. It used 108,750 real and 185,750 generated images from 42
  generators, with 36 transformations. It had 511 registrants and 20 valid
  final submissions.
- **Confirmed:** Its robust track applied 1--5 consecutive randomly sampled
  distortions to both classes. Validation and test included generators absent
  from training and increasingly difficult, partly undisclosed transformation
  sets. Half of each class in validation/test was transformed.
- **Confirmed:** Ranking used the average robust ROC AUC across open and hidden
  test sets; clean ROC AUC was secondary. The top reported systems were:

  | Rank | Team | Average clean AUC | Average robust AUC |
  |---:|---|---:|---:|
  | 1 | MICV | 0.9974 | 0.9723 |
  | 2 | Ant International | 0.9972 | 0.9721 |
  | 3 | TeleAI-TeleGuard | 0.9786 | 0.9251 |
  | 4 | INTSIG | 0.9897 | 0.9130 |
  | 5 | vincentlc | 0.9527 | 0.8730 |

- **Confirmed:** MICV combined six DINOv3 backbones in two committees, trained
  on millions of multi-source samples with staged compound augmentations. It
  reports end-to-end fine-tuning for 10 epochs on 32 NVIDIA A100 GPUs in about
  eight hours.
- **Confirmed:** Ant International combined two independently fine-tuned
  DINOv3-7B models, totalling 14B parameters. It used about one million images,
  staged distortion levels and test-time augmentation. Reported inference on
  one A100 was 2.21 images/s using 78.25 GB VRAM.
- **Confirmed:** Other high-ranking entries repeatedly used large pretrained
  CLIP/SigLIP/DINO encoders, several-model ensembles, training-time distortion
  simulation, and test-time augmentation. Clean performance alone did not
  predict robustness: Shallow Real reported 0.9953 average clean AUC but only
  0.8336 robust AUC.
- **Derived:** MICV's six-backbone committee and Ant's 14B ensemble are not
  directly admissible if TikTok counts total inference parameters against the
  below-2B requirement. The TikTok statement has not yet defined whether its
  limit applies per component or to the complete submitted system, so total
  system size must be clarified rather than assumed.
- **Derived:** The transferable evidence is not "use the winner unchanged".
  It is that diverse generator coverage, realistic compound augmentations,
  threshold-independent evaluation and separate clean/robust reporting were
  central to success. Converting that observation into a Track 5 design remains
  out of scope before the official start.

Primary source:

- Challenge report and verified results: <https://arxiv.org/html/2604.11487>

### Counter Turing Test / Defactify 4.0 (2025)

- **Confirmed:** This shared task used MS COCOAI: 96,000 semantically paired
  real/synthetic images from MS COCO plus SD 2.1, SDXL, SD3, DALL-E 3 and
  Midjourney 6. Splits were 42,000 train, 9,000 validation and 45,000 test.
- **Confirmed:** Task A was real-versus-generated, scored by weighted F1. Task B
  attributed a generated image to one of the five generators, scored by macro
  F1. The organizer baseline was a ResNet-50 trained on Fourier-domain images.
- **Confirmed:** Ten teams reached the final leaderboard. SeeTrails placed first
  on both tasks with 0.8334 binary F1 and 0.4986 attribution F1, compared with
  baseline scores 0.80144 and 0.44913. The findings paper does not describe
  SeeTrails' method, so calling a particular architecture the winning method
  would be unsupported.
- **Confirmed:** Disclosed high-ranking approaches combined such inputs as RGB,
  frequency representations and reconstruction errors; CLIP or CNN features;
  ViT/Swin backbones; compression/noise augmentation; alternative color spaces;
  contrastive learning; pseudo-labelling; and multi-stream fusion. Scores within
  the top seven of Task A were only 0.0042 apart.
- **Derived:** High test similarity to the supplied generator families can make
  generator-specific fingerprints valuable here. Its relatively modest score
  and difficulty of five-way attribution warn against assuming those signatures
  transfer to unseen models or transformed images.

Primary source:

- Findings report: <https://arxiv.org/html/2605.20787>

### NIST GenAI Image Discriminator pilot (2025--2026)

- **Confirmed:** NIST organized three rounds in which generator participants'
  approved outputs, supplemented by NIST, became new discriminator test sets.
  This creates a moving evaluation rather than one fixed public dataset.
- **Confirmed:** Systems returned a 0--1 AI confidence per image. Reported
  metrics were ROC AUC, TPR at FPR 0.1, equal-error rate and separate Brier
  scores for real and generated classes. The test set could not be inspected,
  trained on or used for tuning, and images had to be processed independently.
- **Confirmed:** NIST publishes leaderboard results in bins and prohibits public
  comparative/endorsement claims. The accessible official page does not expose
  technical descriptions for named winners, so this research cannot truthfully
  identify a best architecture from that evaluation.
- **Derived:** The useful design lesson is evaluation against fresh outputs from
  active generator teams and probability calibration, not merely accuracy on a
  static dataset.

Official sources:

- Challenge page: <https://ai-challenges.nist.gov/t2i>
- Evaluation plan: <https://ai-challenges.nist.gov/pub/GenAI_Image_Discriminators_Evalplan.pdf>

### SAFE Image Authenticity Challenge 2025 / WACV 2026

- **Confirmed:** SAFE broadens the problem from whole-image generation to four
  categories: pristine, spliced/traditionally edited, fully generated and
  locally AI-edited. It evaluates binary authenticity, manipulation category and
  pixel-level localization on private data executed in a controlled platform.
- **Confirmed:** SAFE-FORGE contains 10,000 examples for each of four fake types
  plus 10,000 paired pristine originals. The organizer kept generator and editor
  identities private during the competition.
- **Confirmed:** The WACV report available here contains a preliminary combined
  system, not final participant winners. Combining TruFor with Community
  Forensics yielded F1 0.82 for fully generated images but only 0.53 for
  splicing, 0.21 for traditional edits and 0.25 for AI edits. Localization F1
  was 0.68 for splices, 0.30 for traditional edits and 0.14 for AI edits.
- **Derived:** This exposes an important boundary for Track 5: whole-image
  confidence and local edit detection are different tasks. TikTok's current
  statement describes AIGC image detection and robustness, but does not say
  whether partially edited real images will appear in the final test.

Official/primary sources:

- Challenge page: <https://dsri.org/challenges/safe-2025/>
- WACV report: <https://openaccess.thecvf.com/content/WACV2026W/SynRDinBAS/papers/Nguyen_The_SAFE_Image_Authenticity_Challenge_Detecting_and_Localizing_Partial_and_WACVW_2026_paper.pdf>

### IEEE VIP Cup 2022 Synthetic Image Detection

- **Confirmed:** This earlier still-image challenge already combined fully
  generated and locally manipulated images, known and unseen generators, GANs
  and diffusion models. Test images were randomly cropped/resized to 200 by 200
  and JPEG-compressed, removing easy resolution/source clues.
- **Confirmed:** Teams submitted executable code that had to process 5,000
  images within one hour on a 16 GB GPU. Ranking in the open stage used balanced
  accuracy weighted 70% known-generator and 30% unseen-generator test sets.
  Only five submissions were permitted.
- **Confirmed:** Final competition rank was not the same as raw leaderboard
  rank: judges also scored innovation, report and presentation. FAU Erlangen-
  Nurnberg placed first overall with an ensemble of ImageNet-21K-pretrained
  vision transformers fine-tuned on 400,000 balanced samples spanning the five
  official generators plus DALL-E and VQGAN.
- **Confirmed:** Megatron placed second overall despite the best reported test
  accuracies (96.04%, 83.00%, 90.60% over the three test sets). It used CNN and
  transformer ensembles, generator-aware multi-class classification with an
  extra unknown class, knowledge distillation, test-time augmentation and eight
  extra generator families. Sherlock placed third with spatial/Fourier branches
  using EfficientNet-B7 and MobileNet-V3 plus CutMix.
- **Confirmed:** Across 13 valid teams, unknown-generator accuracy was about ten
  points lower for the best methods. Local manipulations and diffusion images
  were hardest; AUC reversed the first two methods' order on one test set,
  exposing threshold calibration as a separate issue. Execution time correlated
  only weakly with score.
- **Derived:** The historical lesson is not that the largest leaderboard entry
  always "won". Evaluation objectives, resource limits and non-score judging
  criteria can change the final order. This Track 5 audit must keep technical
  score, compliance and presentation claims distinct.

Primary sources:

- Organizer challenge report: <https://arxiv.org/html/2309.12428>
- Official challenge page: <https://grip-unina.github.io/vipcup2022/>
- Megatron/ArtiFact paper: <https://arxiv.org/abs/2302.11970>

### MediaEval 2025 Synthetic Images

- **Confirmed:** Task A offered a constrained run using only official data and
  an open run allowing external/self-generated data. It used 10,000 training,
  10,000 labelled validation and 10,000 hidden test images, including web/social
  content and transformed GAN, diffusion and commercial-generator outputs.
  Ranking used F1 while reporting accuracy, precision, recall, AUC, AP and EER.
- **Confirmed:** Organizer baselines were poor when their training domain was
  narrow: validation-threshold F1 was 0.333 for UniFD, 0.669 for RINE and 0.740
  for BFree. RINE trained on the in-the-wild TWIGMA dataset reached 0.803,
  reinforcing the importance of realistic source diversity.
- **Confirmed:** One participant working note found that official training
  images were visually mismatched while labelled validation resembled test. A
  frozen CLIP ViT-L/14 with a linear classifier trained directly on 8,000
  validation images reported F1 0.8856 at threshold 0.5; weighted voting reached
  0.8870. Training only a ResNet on official training data produced F1 below
  0.05 at that threshold.
- **Confirmed:** The working note says its weighted-vote weights corresponded to
  F1 on the test set and also reports a retrospectively best test threshold.
  These values are useful analysis but should not be treated as a deployable
  no-test-label procedure. The organizer overview available here does not list
  participant ranks; a team announcement claims first place in the open run,
  which remains **Unverified** against an organizer-issued final ranking.
- **Derived:** The participant's decisive tactic—training on organizer
  validation because it resembled test—is explicitly unavailable for TikTok's
  4,998 COCO plus 8,843 DALL-E demonstration set, which the statement forbids
  using for training. A prior competition tactic cannot override this track's
  data-use rule.

Primary sources:

- Organizer overview/results: <https://2025.multimediaeval.com/paper47.pdf>
- Participant working note: <https://2025.multimediaeval.com/paper6.pdf>
- Official task page: <https://multimediaeval.github.io/editions/2025/tasks/synthim/>

### IEEE DFWild-Cup 2025 (adjacent face-only challenge)

- **Confirmed:** This challenge covered only facial deepfakes, so it is adjacent
  rather than equivalent to general AIGC image detection. Training/validation
  combined eight public face-deepfake datasets; filenames and preprocessing were
  normalized, and hidden test sources could include newly generated examples.
- **Confirmed:** Participants were limited to the supplied training list, could
  tune on labelled validation and submitted a realness score for each test file.
  Reports had to state trainable/non-trainable parameters and per-file latency.
- **Confirmed:** The final award considered six equally weighted factors,
  including validation/test performance, innovation, model/time complexity,
  report and presentation. The accessible official page confirms that three
  finalists were selected but does not publish their names, final ranks, scores
  or technical reports.
- **Unknown:** Search results and personal announcements name possible winners,
  but no organizer-issued final ranking was located. Those claims are not being
  promoted to confirmed evidence.

Official sources:

- Competition document: <https://signalprocessingsociety.org/sites/default/files/uploads/community_involvement/docs/2025_spcup_official_doc.pdf>
- Official site: <https://sites.google.com/view/dfwild-cup/home>

### Inclusion 2024 Global Multimedia Deepfake Detection Challenge (adjacent face-only challenge)

- **Confirmed:** This is an adjacent face-forgery competition, not a general
  AIGC-image challenge. Its MultiFFI collection contains more than 900,000
  aligned and cropped 512-by-512 face images from 81 manipulation/generation
  methods, including face swaps, animation, attribute editing, full-face
  synthesis, super-resolution and AIGC. Faces occupy roughly 60--70 percent of
  every image, so its winning evidence cannot be assumed to transfer to
  landscapes, products, documents or other general Track 5 content.
- **Confirmed:** The image splits were 99,386 real/425,043 forged for training,
  59,082/88,281 for validation, 77,602/96,785 for public test and
  156,720/176,129 for hidden test. The test sets included forgery types absent
  from training and validation. The challenge attracted 1,500 teams and about
  5,000 valid submissions; the top 20 advanced to the final phase.
- **Confirmed:** Track 1 models could train only on the supplied training set,
  although ImageNet-pretrained weights and processing/augmentation derived from
  that set were permitted. Participants could tune on labelled validation, then
  received public-test feedback twice daily for eight days. Finalists submitted
  code, models and a report for organizer execution on the hidden set.
- **Confirmed:** Ranking primarily used ROC AUC, with TPR at FPR 1e-2, 5e-3
  and 1e-3 as auxiliary metrics. The final rank combined public-test score,
  hidden-test score and technical report with weights 0.2, 0.6 and 0.2. The
  organizer report names the top three but does not publish a complete hidden-
  test or final weighted-score table.
- **Confirmed:** First-place JTGroup generated variants from official training
  inputs using face/background edits and Stable Diffusion. It clustered
  image features into 20 groups, re-split the official data to simulate unseen
  forgery types, trained seven expert models on different folds with balanced
  sampling, cross-entropy and InfoNCE losses, cosine learning-rate scheduling
  and EMA, then averaged expert logits. Augmentations included JPEG/WebP, blur,
  Gaussian noise, brightness and grid distortion.
- **Unknown:** The organizer report does not explain how JTGroup's pretrained
  Stable Diffusion use was reconciled with the Phase 1 wording that forbade
  external-source data and specifically mentioned ImageNet-pretrained weights.
  It is therefore unsafe to treat that prior tactic as automatically permitted
  under another competition's rules.
- **Confirmed:** Ordinary official validation was strongly over-optimistic.
  An EfficientNet-B4 baseline reported 0.9939 validation AUC but only 0.92998
  public-test AUC. JTGroup's harder clustered split exposed more of the gap; its
  individual public-test AUCs ranged from 0.90767 to 0.97588, and the final
  public-test ensemble reached 0.98051. That 0.98051 is not the hidden score or
  final weighted rank score.
- **Confirmed:** Second-place Aegis combined multiple backbones with RGB-like,
  YCbCr, SRM and DCT inputs, synthesized additional face forgeries from official
  inputs and learned a score-fusion model. Third-place VisionRush fused pixel
  and noise domains with ConvNeXt and RepLKNet, degraded fake-image quality,
  used RandAugment and EMA, and averaged model scores. These are organizer-
  reported descriptions, not locally reproduced results.
- **Confirmed:** An Ant Digital Technologies press release calls the winner's
  result “97.038% accuracy,” while the technical report says AUC determined
  ranking and gives no corresponding 0.97038 value or complete final table.
  The metric identity is unresolved; it must not be equated with the 0.98051
  public-test AUC or represented as the final weighted score.
- **Confirmed:** The organizer's open-source registry at audited revision
  `8aaed8bfca11e6e154d2e9f6c8109dc73b286fd3` contains only a 1,203-byte README
  linking the first- and third-place repositories and has no declared licence.
  The winner repository at
  `8acf7ba3ca3eac6e6462a394a190089d306653ed` is CC-BY-NC-4.0 and contains
  inference material plus external weight links, but its README still promises
  training code and a detailed report later; it is not a complete reproduction
  package. The third-place repository at
  `13117ecf91c215a167126a6962fdd1525f7c957e` is Apache-2.0, says eight GPUs
  were used and requires external ImageNet-pretrained weights. No second-place
  implementation was linked by the registry.
- **Derived:** The strongest transferable evidence is evaluation discipline:
  near-perfect validation can coexist with a material unseen-type gap, and a
  protected hidden set can change the meaning of a public leaderboard score.
  The seven-model face ensemble itself is neither selected nor established as
  suitable for Track 5's content, resource limit or transformations.

Primary sources:

- Organizer challenge report: <https://arxiv.org/html/2412.20833>
- Organizer open-source registry:
  <https://github.com/inclusionConf/DeepFakeDefenders>
- First-place repository: <https://github.com/HighwayWu/DeepFakeDefenders>
- Third-place repository: <https://github.com/VisionRush/DeepFakeDefenders>
- Organizer press release:
  <https://www.prnewswire.com/apac/news-releases/2024-global-multimedia-deepfake-detection-challenge-winner-announced-and-exploring-the-future-of-responsible-technology-302240478.html>

### Meta/Kaggle Deepfake Detection Challenge 2019--2020 (adjacent face-video challenge)

- **Confirmed:** DFDC was a face-swap video competition, not general still-
  image AIGC detection. Its full public corpus contained 128,154 videos,
  including 104,500 unique fake videos produced with eight manipulation methods
  and 19 post-processing/distractor types. The current Meta overview rounds this
  to 124,000 videos; the technical report's exact inventory is used here.
- **Confirmed:** The competition had 2,114 teams and 35,109 submitted models.
  Its 4,000-video public test contained challenge-produced material, while the
  10,000-video private black-box set combined organic internet content with new
  project videos. Participants submitted executable code; final evaluation ran
  on a single V100 with a 90-hour limit, although most systems completed within
  ten hours.
- **Confirmed:** Competition rank used private-set log loss, not accuracy or
  average precision. The final top-five log losses were Selim Seferbekov
  0.42798, WM 0.42842, NTechLab 0.43452, Eighteen Years Old 0.43476 and The
  Medics 0.43711. The winning system had ranked fourth on public data; the
  eventual second-through-fifth systems had been 37th, 6th, 10th and 17th.
- **Confirmed:** Meta's current dataset page calls 65.18 percent the winner's
  black-box average precision. Its contemporaneous result blog called the same
  number “accuracy,” while the technical report gives overall/DFDC/organic-
  internet log losses of 0.4279/0.1983/0.6605 and reports aggregate real-video
  AP 0.753 and ROC AUC 0.734 for the best models. These quantities describe
  different summaries and must not be merged into one headline score.
- **Confirmed:** The first-place entry used MTCNN to locate faces, classified
  sampled frames with ImageNet/noisy-student-pretrained EfficientNet-B7 models,
  trained with face-region dropout and heavy compression/noise/blur/color/
  geometric augmentation, and aggregated scores from 32 frames. The released
  pipeline trains five B7 seeds and has seven B7 checkpoints in release 0.0.1.
  The winner's Kaggle note says no external data was used and reports training
  on two workstations containing two and four Titan V GPUs.
- **Confirmed:** The organizer report describes the runner-up as frame-wise
  Xception plus weakly supervised attention/augmentation, third place as an
  EfficientNet ensemble with mixup, fourth as an ensemble of frame and temporal
  models, and fifth as a seven-model CNN ensemble. The strong first-place
  result came from a comparatively direct face-frame classifier rather than a
  complicated temporal architecture.
- **Confirmed:** The winner's code is MIT-licensed at audited revision
  `89c6290490bac96b29193a4061b3db9dd3933e36`. Its release script points to
  seven external checkpoints whose current metadata-only HTTP responses report
  sizes from 266,910,613 to 266,910,618 bytes, totalling 1,868,374,312 bytes.
  No checksum is published by the inspected release page/script, and no weight,
  code archive or DFDC video was downloaded.
- **Derived:** The public-to-private rank reversal is stronger evidence for a
  protected hidden set than for any particular architecture. This face-video
  result cannot establish Track 5 performance because content, unit of
  prediction, generator families, model vintage and metric all differ.

Primary sources:

- Organizer technical report: <https://arxiv.org/html/2006.07397>
- Meta dataset, impact and leaderboard page:
  <https://ai.meta.com/datasets/dfdc/>
- Meta contemporaneous result announcement:
  <https://ai.meta.com/blog/deepfake-detection-challenge-results-an-open-initiative-to-advance-ai/>
- Winner's official Kaggle write-up:
  <https://www.kaggle.com/competitions/deepfake-detection-challenge/writeups/selim-seferbekov-1st-place-solution>
- Winner's code: <https://github.com/selimsef/dfdc_deepfake_challenge>

### AADD-2025 adversarial-detector challenge (reverse/adjacent task)

- **Confirmed:** AADD was not a detector-building competition. Participants
  modified supplied generated images so four detectors would call them real
  while retaining high structural similarity. Two public models were ResNet and
  DenseNet; two additional detector architectures were hidden during
  evaluation.
- **Confirmed:** Its 16 subsets covered high- and low-quality images from GAN
  and diffusion generators. The low-quality set was resized and variably JPEG-
  compressed to imitate social-media processing. A successful case was a fake
  image classified as real; the final score combined attack success across the
  four models with structural similarity.
- **Confirmed:** Thirteen teams were ranked. MR-CAS placed first with structural
  similarity 0.742 and attack-success score 0.672; Safe AI was second at
  0.915/0.528; RoMa third at 0.934/0.509. The final-score ordering therefore
  balanced invisibility and transfer, rather than maximizing either column.
- **Confirmed:** The official report says the strongest attacks used latent-
  space manipulation, multi-model gradient/ensemble information and surrogate
  models to transfer to hidden architectures. White-box attacks were nearly
  perfect in some cases, while black-box transfer remained substantially
  harder.
- **Derived:** This is direct evidence that high clean or ordinary-distortion
  accuracy is not adversarial security. It does not establish that TikTok will
  include intentional adversarial examples, and operational attack construction
  is outside this preparation scope.

Primary sources:

- Official challenge report:
  <https://openreview.net/pdf/5a1add4a4f4e8cc99a1c5f2efe56492afcd3963c.pdf>
- Organizer page:
  <https://iplab.dmi.unict.it/mfs/acm-aadd-challenge-2025/>
- Official repository: <https://github.com/mfs-iplab/aadd-2025>

## Cross-cutting findings

- **Confirmed:** Robustness and generalization are different problems. A model
  can fail because a known generator's tell-tale signal was destroyed by JPEG,
  crop, resize, blur or noise; it can separately fail because a new generator
  never left the clues present in training.
- **Confirmed:** Compression itself can become a label shortcut: if real images
  are JPEG and fakes are PNG during both training and evaluation, a detector may
  score well by recognizing file-processing history. Applying more random
  compression does not prove that the two class distributions are matched.
- **Confirmed:** Aggregate benchmark success can conceal catastrophic local
  failures. Report distributions and per-generator/per-transformation results,
  not only one accuracy number.
- **Confirmed:** More sophisticated architecture alone does not guarantee
  transfer. Dataset diversity and how closely training sources resemble the
  eventual generator landscape repeatedly explain large performance changes.
- **Derived:** Commercial app confidence percentages should not be interpreted
  as calibrated probabilities without independent evidence on the exact image
  source and post-processing. Their internals and evaluation distributions must
  be audited separately.
- **Confirmed:** A high-confidence positive from a valid cooperative watermark
  is evidentially different from a high neural-classifier score. A negative
  watermark result only means that particular mark was not recovered; it does
  not establish that the image is real.
- **Confirmed:** Repeated competition designs protect the test set and prohibit
  training on it because seeing evaluation images leaks information about their
  generators, labels, formats and processing. Fresh/private generators are used
  specifically to test whether a method generalizes rather than memorizes a
  public collection.
- **Confirmed:** Modern high-performing approaches increasingly combine cues
  that fail differently: semantic representations help when fragile low-level
  traces are compressed, while residual, frequency, gradient, phase or bit-
  plane cues can expose generator mechanics that semantics miss. Published
  hybrids still have generator- and transformation-specific failure modes.
- **Confirmed:** Benchmark headline numbers are protocol-dependent. A score can
  change because of the released checkpoint, threshold, real-image source,
  generator vintage, image resolution or preprocessing, even when the method
  name is identical.
- **Derived:** A lightweight paper is relevant to the sub-2B constraint but is
  not preferable merely because it is fast; robustness coverage, calibration
  and hidden-generator transfer have to be evaluated separately after the
  official build window opens.
- **Confirmed:** Prior competitions repeatedly separate raw leaderboard metrics
  from final awards that also score complexity, reports or presentations. Any
  claim about a "winning solution" must specify which ranking it means.
- **Confirmed:** Validation/test resemblance can yield large gains without
  solving open-world generalization. Dataset rules determine whether exploiting
  that resemblance is legitimate; TikTok's demo-only set has an explicit
  training prohibition.
- **Confirmed:** A temporally ordered benchmark exposes failure at generator-
  family transitions that a random or static aggregate split can hide. Even in
  that protocol, a high past-period score does not guarantee the next period.
- **Confirmed:** Under a controlled dataset and metric, preprocessing and
  augmentation alone can move scores by several points, and the best crop-versus-
  resize choice can reverse by backbone. Architecture names do not specify a
  complete method or its expected performance.
- **Confirmed:** A dataset assembled from many public sources does not inherit
  one blanket licence from the benchmark code. Code licence, dataset-card
  licence and every source image's terms remain separate provenance questions.
- **Confirmed:** Face-only deepfake winners are adjacent evidence, not general
  AIGC detector evidence. Alignment, face-dominant crops and a narrow semantic
  domain remove variations that Track 5 may contain.
- **Confirmed:** A near-perfect official validation score can substantially
  overstate performance on new forgery types. Public-test AUC, hidden-test AUC,
  press-release “accuracy” and a final score that includes report judging are
  distinct quantities and must never be silently substituted for one another.
- **Confirmed:** A public leaderboard can mis-rank the eventual best hidden-set
  system even when public and private performance are correlated overall. DFDC's
  first two private-set finishers were fourth and 37th publicly, showing why a
  few public submissions are not sufficient evidence of open-world robustness.
- **Confirmed:** High ROC AUC can coexist with poor fake-image recall at the
  deployed threshold. In the online Adobe study, pre-Firefly history separated
  Firefly images at 0.92 AUC but recognized only 35 percent as generated until
  Firefly examples were added. Ranking quality and operational classification
  are separate properties.

## Closing reconciliation and evidence hierarchy

- The complete ledger was reread from line 1 through line 1,733 after the user
  stopped research. Every research session, retained URL, observed command,
  licence statement and negative result is represented above.
- When two sources use different precision, the more exact technical record is
  retained: DFDC's 128,154-video paper inventory is kept alongside Meta's
  rounded 124K landing-page figure.
- When a summary changes the metric name, the quantities remain separate:
  DFDC private log loss, black-box average precision and the older blog's
  “accuracy” wording are not interchangeable; neither are Inclusion's 0.98051
  public AUC, undisclosed hidden result, final report-weighted rank and press-
  release “97.038% accuracy.”
- When a paper's prose and table disagree, the discrepancy remains visible and
  no false precision is invented. RRBench's AIDE re-digitization decline is
  2.91 points by table arithmetic while prose says 1.89. The Adobe online study
  uses the visually verified Firefly matrix cell instead of its ambiguous
  GLIDE prose example.
- Cross-paper scores are not treated as a shared leaderboard. Dataset source,
  generator date, preprocessing, threshold, metric and checkpoint explain why
  a method can report above 99% in its own protocol yet fail badly in a later
  zero-shot audit.
- The strongest evidence-supported conclusion is not a universal model name.
  Reliable evaluation requires diverse and current real/fake sources, matched
  file-processing histories, realistic transformations, hidden/new-generator
  testing, separate clean/robust and class-specific reporting, and calibration.
  Cooperative provenance can add strong evidence when present but cannot replace
  pixel detection for arbitrary images.
- No judged Track 5 detector, backbone, ensemble, augmentation recipe or
  calibration procedure was selected. Making that competition decision before
  2026-08-29 12:00 SGT remains outside the authorized preparation boundary.

## Unresolved evidence at closure

Research stopped at the user's explicit instruction. These are preserved
unknowns and incomplete audits, not scheduled work or assumptions:

- Trace PatchCraft, DRCT, AIDE, UnivFD, SAFE and calibration methods to their
  original papers and released code; record licences and complete model sizes.
- Separate metadata/provenance detectors from pixel-only detectors and verify
  whether any prior competition explicitly allowed cooperative provenance.
- Examine robustness evidence under JPEG, blur, resize, noise, crop and color
  changes, including compound transformations.
- Audit public commercial detector claims against disclosed methodology and
  independent evaluations; do not infer undocumented internals.
- Search earlier NIST/OpenMFC and other public challenge archives for released
  image-discriminator systems, data rules and evaluation metrics.
- Identify which prior winning results are inapplicable under Track 5's less
  than 2B parameter constraint.
- Audit dataset leakage, duplicate/source bias and probability calibration
  findings that can make a detector appear stronger than it is.

## Research log

### 2026-08-27 - ledger initialization

- Created this durable journal at the user's request.
- No detector was selected, implemented, trained or tuned.

### 2026-08-27 - first primary-literature cluster

- Read the NTIRE 2026 robust-detection challenge report and its final results.
- Read the 2026 zero-shot benchmark of 16 public detector methods.
- Recorded only paper-reported observations; no benchmark was reproduced and
  no model or dataset was downloaded.
- Observed resource disclosures: MICV reports 32 A100 GPUs for eight hours;
  Ant reports a 14B-parameter ensemble and 78.25 GB inference VRAM.
- No Track 5 detector was selected, implemented, trained or tuned.

### 2026-08-27 - commercial, provenance and competition cluster

- Read three independent commercial-detector evaluations, current Sightengine
  documentation, ImageDetectBench, Google SynthID and C2PA 2.2 documentation.
- Read the Counter Turing Test findings and NIST Image-D evaluation design.
- Downloaded the official NIST evaluation-plan PDF to a temporary directory:
  SHA-256 `15ca0d5e3e9d5fa420dd1a1dfb0c769dabe406be948bb4fbd378a9542c78e0d5`,
  933,708 bytes, 12 pages. System `pdftotext` was unavailable; extraction then
  succeeded read-only with bundled Python 3 and `pypdf`. The temporary file was
  removed.
- Downloaded the official WACV SAFE report to a temporary directory:
  SHA-256 `80f12e5a9b7253267f5cdaf00daabd30c8ffa99c6216591c35771ca232113c01`,
  2,163,577 bytes, 13 pages. Text extraction succeeded read-only with bundled
  Python 3 and `pypdf`; the temporary file was removed.
- No API credentials were requested, no commercial endpoint was queried, and no
  dataset/model/output was retained or added to the repository.
- No Track 5 detector was selected, implemented, trained or tuned.

### 2026-08-27 - diversity, hybrid-cue and compression cluster

- Read the primary Community Forensics, CO-SPY, LOTA and CPTFormer papers.
- Downloaded the official CVPR 2026 CPTFormer PDF to a temporary directory:
  SHA-256 `28108254b3d75838559a3a0283933a7b093e16b9919514e068e7ea8ad4242d0b`,
  1,817,247 bytes, 10 pages. Text extraction succeeded read-only with bundled
  Python 3 and `pypdf`; the temporary file was removed.
- Recorded paper-reported parameter and latency figures only where the source
  explicitly disclosed them. No model was downloaded and no benchmark was
  reproduced.
- No Track 5 detector was selected, implemented, trained or tuned.

### 2026-08-27 - AIDE, DRCT, PatchCraft and calibration audit

- Downloaded only paper PDFs into auto-removed temporary directories. SHA-256,
  byte size and pages were: DRCT
  `79bff1d93c6fb548e4165da9707f637fdcaf081690def4682842f0f562c3a2a7`,
  2,989,977 bytes, 19 pages; PatchCraft
  `355ff71054a0d67ed4e615ff7183a12fa8fcc356fd6a8ef35054a854c5c8f55f`,
  1,568,909 bytes, 18 pages; calibration
  `e471f7b3c261660c91a72f3b2c6ef63fd8afc83e5493ef462e2cb6549379af05`,
  1,729,883 bytes, 9 pages.
- Direct OpenReview PDF access for AIDE returned HTTP 403 twice, including with
  a browser user-agent. The current arXiv v3 HTML and official repository were
  read instead; no reproduction claim depends on the blocked PDF.
- Queried the unauthenticated GitHub REST API at `/repos/{owner}/{repo}`,
  `/commits/{default_branch}` and `/git/trees/{commit}?recursive=1`; fetched only
  small raw text source files pinned to the returned commit. Observed commits:
  DRCT `01aa7d0b6de903cf3208801f8f6b469edffbc762`, PatchCraft
  `00a00d9271304a18aa2d96f724d71c4e898708fd`, AIDE
  `6725b710d5c437ab2f59792908ce0377dfc907de`, calibration
  `66d4bc606f7cf325d9bd4e67ca34b0c59d6a9d53`.
- GitHub reported MIT for AIDE and no detected licence for the other three.
  Absence of a detected licence was verified against each recursive tree.
- Did not clone repositories, download weights or retain papers, datasets,
  logits, caches or generated outputs. No detector was selected or implemented.

### 2026-08-27 - earlier challenge history cluster

- Read the organizer reports/pages for IEEE VIP Cup 2022, MediaEval 2025 and
  DFWild-Cup 2025, plus the disclosed MediaEval CLIP working note.
- Kept leaderboard rank, judged final placement and author-announced placement
  distinct. The accessible DFWild organizer sources do not disclose final names
  or methods; the MediaEval overview does not list participant ranks.
- Identified one prior strategy that directly conflicts with this track's
  preparation rules: training on a validation set selected for resemblance to
  test. Recorded it as non-transferable, not as a recommendation.
- No competition artifact, dataset or model was downloaded or retained. No
  detector was selected, implemented, trained or tuned.

### 2026-08-27 - UnivFD, B-Free, SPAI and RRBench cluster

- Read the primary UnivFD, B-Free, SPAI and RRDataset/RRBench papers and their
  official CVF or project pages.
- Preserved a numerical contradiction in the RRBench paper: its table implies
  a 2.91-point AIDE decline after re-digitization, while nearby prose states
  1.89 points.
- Attempted a sequential temporary download of the four official CVF PDFs. The
  first request produced no response for roughly 2.5 minutes and was manually
  interrupted with `Ctrl-C`; Python exited with code 130 and the automatic
  temporary directory was removed. The arXiv HTML and official web pages were
  accessible, so no claim depends on the stalled download.
- No paper, dataset, model weight, cache or generated output was retained. No
  detector was selected, implemented, trained or tuned.

### 2026-08-27 - code-licence and transformation audit

- Queried unauthenticated GitHub REST metadata, commit and recursive-tree
  endpoints. Audited revisions: UnivFD
  `030495aea3300a8b54c0ec37ec7fe1dd7e63c619`, B-Free
  `c6a9f898782fb466b29af01f21960b67415afb0e`, SPAI
  `8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4`, RRDataset
  `a6d1f33b5d9a45628d088f9aee5c809d1d529b92`, RINE
  `9b7fd5857cc205d0412be6aeee0d7611b95bd620`, and SAFE
  `4e998724651b227def64f5be0cd60c0aa1552c35`.
- GitHub reported MIT for UnivFD and RRDataset, Apache-2.0 for SPAI, RINE and
  SAFE, and no standard SPDX licence for B-Free. Reading B-Free's pinned
  `LICENSE.txt` confirmed an all-rights-reserved informational/nonprofit-only
  restriction.
- Read pinned small text files only. RINE's tree contains four trained-head
  checkpoints (1,139,686 to 42,094,118 bytes); SAFE contains one 5,840,638-byte
  checkpoint. No repository was cloned and no checkpoint was fetched.
- Queried Zenodo JSON metadata for RRDataset records 14963880 and 14991882,
  preserving record versions, CC-BY-4.0 metadata, archive sizes and MD5 values.
  No dataset or weight archive was downloaded.
- A metadata-only `curl -I -L --max-time 30` request to B-Free's weight URL
  failed to connect to `www.grip.unina.it:443` after about seven seconds. It
  transferred no weight data and current file size remains unknown.
- Read the primary RINE and SAFE papers. Recorded complete-versus-trainable
  parameter distinctions, perturbation results and the gap between their
  original protocols and later zero-shot evidence.
- No detector was selected, implemented, trained or tuned.

### 2026-08-27 - newest-generator evidence cluster

- Read the June 2026 SSAFE and July 2026 SafeIMG preprints plus the official
  Perception Encoder model documentation.
- Confirmed that SSAFE's PE-Core-G14-448 image tower is documented at 1.88B
  parameters and that SSAFE had not released its promised implementation at the
  time of inspection. No checkpoint or dataset was fetched.
- Queried GitHub and Hugging Face metadata for SafeIMG. The code revision was
  `3fbf8d83290f267c958f41a53e00e5ac134400ac`; the dataset revision was
  `389a7c555191c9e946727a470500fcd9b045381c`. Neither source declared a licence,
  despite the paper HTML itself using CC-BY-NC-SA-4.0.
- The Hugging Face metadata enumerated the benchmark's individual PNG files but
  did not provide useful per-file sizes through that endpoint. No image was
  downloaded or opened.
- Preserved the limitations of each comparison: SSAFE does not test Track 5's
  transformation suite, while SafeIMG compares only three older specialized
  detectors and reports generated-class recall rather than balanced accuracy.
- No detector was selected, implemented, trained or tuned.

### 2026-08-27 - adversarial challenge boundary

- Read the organizer AADD-2025 page, official repository summary and official
  ACM Multimedia challenge report.
- Recorded final rankings with separate structural-similarity and attack-success
  values. Treated AADD as a reverse/adjacent task rather than a detector contest.
- Preserved only evaluation-level findings about detector vulnerability; no
  attack code, image or operational evasion procedure was acquired or added.
- No detector was selected, implemented, trained or tuned.

### 2026-08-27 - temporal benchmark and next-generator audit

- Read the accepted AI-GenBench paper, official live leaderboard, pinned code
  documentation and fake-only dataset card. No paper, code, image or dataset
  was downloaded or retained.
- Queried repository metadata with
  `curl -fsSL https://api.github.com/repos/MI-BioLab/AI-GenBench`, commit metadata
  with `curl -fsSL https://api.github.com/repos/MI-BioLab/AI-GenBench/commits/main`,
  and the recursive tree with
  `curl -fsSL 'https://api.github.com/repos/MI-BioLab/AI-GenBench/git/trees/e0f673b6d70d989310b82a8f6a891df2cd085925?recursive=1'`;
  all were filtered locally with `jq` and succeeded.
- Queried fake-only dataset metadata with
  `curl -fsSL 'https://huggingface.co/api/datasets/lrzpellegrini/AI-GenBench-fake_part?blobs=true'`
  and read only its small pinned card with
  `curl -fsSL https://huggingface.co/datasets/lrzpellegrini/AI-GenBench-fake_part/raw/75204cf1aa28b61885eae43e883f72e28b8911b2/README.md`.
- An initial combined temporary-file audit command was rejected locally before
  execution because its cleanup used an `rm -f`-style command. The subsequent
  stream-only commands above required no temporary files and succeeded.
- No benchmark was run. No detector was selected, implemented, trained or tuned.

### 2026-08-27 - format, resolution and source-shortcut audit

- Read the complete *Fake or JPEG?* paper, official project page and pinned
  repository inventory. Recorded its causal FFHQ compression experiment rather
  than relying only on cross-dataset score correlations.
- Queried repository metadata with
  `curl -fsSL https://api.github.com/repos/gendetection/UnbiasedGenImage`, the
  current `master` commit endpoint, and recursive tree endpoint pinned to
  `74197ede86ff0855a3e82ee6092ce50a623e376d`; filtered responses locally with
  `jq`. The tree was complete at 599 entries and 10,606,427 blob bytes.
- Queried Dataverse metadata with
  `curl -fsSL 'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/AKDIHF'`
  and filtered it locally with `jq`. A first filter printed the full 503-file
  inventory and was truncated in the terminal display; a compact second filter
  succeeded and preserved version, licence, aggregate size, part count and the
  metadata CSV's typed MD5 value.
- No source repository, 653.8 GB archive part, metadata CSV, image, model or
  output was downloaded. No detector was selected, implemented, trained or
  tuned.

### 2026-08-27 - controlled design-choice evidence

- Read the complete v1 *Generalized Design Choices for Deepfake Detectors*
  preprint and cross-checked its claimed best result against the current
  AI-GenBench leaderboard and the already audited official repository.
- Preserved backbone-specific counterexamples instead of turning the paper's
  aggregate conclusions into universal rules. Also kept experiments that exceed
  the benchmark's augmentation-multiplier limit distinct from eligible settings.
- No paper, code, configuration, data, checkpoint or output was downloaded or
  run. No detector was selected, designed, implemented, trained or tuned.

### 2026-08-27 - Inclusion 2024 face-forgery challenge audit

- Read the complete organizer report, the organizer's open-source registry,
  the linked first- and third-place repositories and the Ant Digital
  Technologies result announcement. The challenge is recorded as face-only
  adjacent evidence rather than a general-image precedent.
- Before the latest API rate limit, queried GitHub repository metadata, default-
  branch commit endpoints and recursive trees using `curl -fsSL` and filtered
  them with `jq`. Observed revisions were registry
  `8aaed8bfca11e6e154d2e9f6c8109dc73b286fd3`, winner
  `8acf7ba3ca3eac6e6462a394a190089d306653ed` and third place
  `13117ecf91c215a167126a6962fdd1525f7c957e`. GitHub reported API repository
  sizes of 14 KB, 3,127 KB and 691 KB respectively; recursive inventories were
  1 entry/1,203 blob bytes, 14 entries/3,135,653 blob bytes and 34
  entries/682,813 blob bytes.
- Read only small pinned README and licence text. The registry had no licence;
  the winner's licence text was CC-BY-NC-4.0 despite GitHub returning
  `NOASSERTION`; the third-place repository was Apache-2.0. No repository,
  external checkpoint or sample image was cloned or downloaded.
- A post-compaction repeat of the unauthenticated GitHub API loop returned HTTP
  403 for metadata and follow-on requests, consistent with exhausted anonymous
  rate allowance. The earlier pinned results above remain recorded; no
  authentication or credential workaround was attempted.
- Preserved the unresolved difference between the technical report's AUC-based
  evaluation and the press release's “97.038% accuracy” instead of inventing a
  mapping. No detector was selected, designed, implemented, trained or tuned.

### 2026-08-27 - DFDC black-box competition audit

- Read the organizer technical report, Meta's current dataset/results page,
  Meta's contemporaneous announcement, the winner's Kaggle write-up and the
  winner's repository documentation. Treated DFDC as adjacent face-video
  evidence, not a still-image or Track 5 baseline.
- Used
  `git ls-remote https://github.com/selimsef/dfdc_deepfake_challenge.git refs/heads/master`
  to pin revision `89c6290490bac96b29193a4061b3db9dd3933e36`. Read the small
  raw `LICENSE` and `download_weights.sh` files without cloning; the code is
  MIT-licensed and the script references release tag 0.0.1.
- Sent metadata-only `curl -fsSIL --max-time 30` HEAD requests to all seven
  release assets and extracted final `Content-Length` values with `awk`. They
  were 266,910,617; 266,910,618; 266,910,617; 266,910,617; 266,910,615;
  266,910,613; and 266,910,615 bytes, totalling 1,868,374,312 bytes. No asset
  body was downloaded and no checksum was available from the inspected page.
- Kept ranking log loss, black-box average precision, ROC AUC and Meta's older
  “accuracy” wording distinct. No DFDC dataset, code archive, model, cache or
  output was retained; no detector was selected, designed, implemented, trained
  or tuned.

### 2026-08-27 - online-update and local-edit evidence

- Read the complete 11-page ICCV 2023 workshop paper *Online Detection of
  AI-Generated Images*. The system Python attempt failed before network access
  because `pypdf` was absent. The bundled Python runtime then fetched the
  4,744,651-byte PDF into memory, verified SHA-256
  `12eee72b783b8ee6ff3221147d946cf8b5f6ce25b9f605045f2225c1c71dd7c4`
  and extracted all pages successfully.
- Rendered page 5 at 180 DPI with bundled `pdftoppm` to inspect the online-
  performance matrices visually. Rendering emitted a non-fatal Fontconfig
  warning but created the PNG; the figure's values and captions were legible.
  The automatically named `/tmp/codex-online-detection.WKUya6` directory was
  then removed and its absence verified.
- Reconciled threshold-dependent accuracy with AUC using the same pre-Firefly
  matrix cell, rather than repeating the paper prose's ambiguous GLIDE example.
  No paper, dataset, code, image, model, cache or output was retained; no
  detector was selected, designed, implemented, trained or tuned.

### 2026-08-27 - final reconciliation and research stop

- Stopped opening new research topics at the user's explicit instruction.
- Reread all 1,733 lines then present in this ledger using seven bounded `sed`
  ranges, checked every method/competition section against the cross-cutting
  findings, and added the closing evidence hierarchy above.
- Reopened the current Devpost rules on 2026-08-27. They confirm the challenge
  window begins 2026-08-29 12:00 GMT+8 and state four equally weighted general
  judging criteria, which conflicts with the five weighted criteria in the
  Track 5 statement. The conflict is recorded as unresolved, not averaged.
- A direct web opener rejected the public Lark statement URL as unsafe in this
  closing pass. The already audited 2026-08-27 public-statement findings remain
  pinned in this ledger; no claim depends on a new failed fetch.
- No new dataset, model, checkpoint, cache, generated output or secret was
  added. No detector was selected, designed, implemented, trained or tuned.
