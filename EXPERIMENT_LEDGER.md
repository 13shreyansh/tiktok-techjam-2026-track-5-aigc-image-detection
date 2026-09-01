# Experiment ledger

Only completed commands are reported as results. Dataset/model caches,
checkpoints and generated outputs stay in ignored directories.

## 2026-08-29 — environment and contract tests

- Challenge start verified at `2026-08-29 12:00:44 +0800`.
- Installed the pinned packages in `requirements-dev.txt` into ignored `.venv/`.
- PyTorch 2.8.0 reported MPS built and available.
- `python -m pytest -q` passed all six current tests, covering the positive
  class, official metric weights, condition construction and data labels.
- The inference command successfully emitted `image_path` and continuous
  `pred` values for one real and one fake test image.

## 2026-08-29 — CIFAKE systems smoke test

Purpose: verify training, all individual organizer transforms, official metric,
checkpointing, MPS resource reporting and JSON inference. This is not a model
selection result because CIFAKE is only 32 by 32 pixels.

```bash
HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" \
.venv/bin/python -m aigc_detector.train \
  --dataset-root datasets/cifake \
  --output-dir outputs/smoke-resnet18 \
  --model resnet18.a1_in1k --epochs 1 --batch-size 64 \
  --max-train-per-class 256 --max-eval-per-class 128 \
  --robust-eval --device mps
```

Observed: 512 training images; 11,177,025 total/trainable parameters; 16.23
seconds; clean AUC 0.4711, pooled robust AUC 0.4910 and official-style score
0.4810. This intentionally weak result is retained as proof that the execution
path works, not as evidence for ResNet or CIFAKE.

## 2026-08-29 — pinned SID_Set slice

Downloaded and verified two Parquet shards at immutable dataset revision
`dc03ead57929879319ce30a82bfcfb8d317b10bd`:

| Shard | Bytes | SHA-256 | Eligible real/full-synthetic | Excluded tampered |
| --- | ---: | --- | ---: | ---: |
| `train-00000-of-00249.parquet` | 489,780,970 | `82e62f400fbb168e0b69ba5104e8109c312fe8a02ee07f06a82ab58208a6fb4a` | 576 | 268 |
| `validation-00000-of-00034.parquet` | 477,663,216 | `56cf2dd5c6a72a158f91aee4c5e06154f5d0a0903eb258a3de11eedded82c2a6` | 586 | 297 |

`scripts/extract_sid_binary.py` verified each source before extraction and kept
only label 0 (real) and label 1 (fully synthetic). Label 2 (tampered) is outside
the organizer's stated binary scope and was excluded. The training and
validation shards remained separate.

## 2026-08-29 — ResNet18 SID control

Two epochs, full 576-image eligible training slice, separate 586-image eligible
validation slice, and all individual conditions. The completed run took 306.62
seconds on MPS. It used 11,177,025 total/trainable parameters and reported about
2.27 GB driver-allocated MPS memory after completion.

| Epoch | Clean AUC | Pooled robust AUC | 50/50 score |
| ---: | ---: | ---: | ---: |
| 1 | 0.7729 | 0.7695 | 0.7712 |
| 2 | 0.8752 | 0.8709 | 0.8730 |

Worst epoch-2 condition was Gaussian noise sigma 0.10 at 0.8326 AUC.

## 2026-08-29 — frozen DINOv2-L SID experiment

Model: `vit_large_patch14_dinov2.lvd142m`, resized deliberately to 224 by 224;
303,228,929 total parameters with only the 1,025-parameter binary head trained.
The checkpoint is official DINOv2-based public pretrained material loaded via
timm. Cached weight revision and digest:

```text
Hugging Face model revision 4741e1cafbf45415e77074bb0cb42dba76c8684a
bytes 1217502758
SHA-256 0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355
```

The first invocation failed before processing a batch because the timm model
defaulted to 518-pixel input while the experiment supplied 224 pixels. The
factory was corrected to instantiate ViTs at the recorded experiment image
size; a forward-shape test then succeeded. No score is attributed to the
failed invocation.

The successful three-epoch frozen-head run took 137.66 seconds on MPS and
reported about 2.26 GB driver-allocated memory after completion:

| Epoch | Clean AUC |
| ---: | ---: |
| 1 | 0.9148 |
| 2 | 0.9595 |
| 3 | 0.9684 |

On a deterministic balanced 256-image validation subset across all organizer
conditions, the completed checkpoint scored clean AUC 0.9706, pooled robust AUC
0.9563 and 50/50 score **0.9635**. The weakest condition was Gaussian noise
sigma 0.10 at 0.9271 AUC.

For a like-for-like check, the completed ResNet checkpoint was reevaluated on
the identical deterministic 256-image subset. It scored clean AUC 0.8909,
pooled robust AUC 0.8881 and 50/50 score 0.8895. DINOv2-L therefore improved
the controlled score by 0.0740 on this slice. This comparison changes the main
backbone candidate, but not the warning about same-dataset shortcuts.

These are same-dataset SID results and may reflect source shortcuts. SID_Set
does not expose generator identity in its Parquet schema, so these values do
not establish unseen-generator performance. DINOv2-L becomes the main backbone
candidate, subject to a generator-held-out WildFake/other-source test.

## 2026-08-29 — frozen DINOv2-L robust-augmentation comparison

The same frozen DINOv2-L setup was trained for three epochs with the `robust`
profile, which adds randomized JPEG, blur, resize, crop and noise simulation.
The completed run used all 576 eligible training images, took 188.67 seconds on
MPS, and again trained only the 1,025-parameter head.

| Epoch | Clean AUC |
| ---: | ---: |
| 1 | 0.9151 |
| 2 | 0.9554 |
| 3 | 0.9658 |

After explicitly seeding Python, NumPy and PyTorch in the standalone evaluator,
the final checkpoint scored clean AUC 0.9667, pooled robust AUC 0.9571 and a
50/50 score of **0.9619** on the deterministic 128-per-class SID subset. Its
weakest condition was Gaussian noise sigma 0.10 at 0.9357 AUC.

One seeded standard-augmentation reevaluation using batch size 8 was manually
interrupted before completion after it ran substantially longer than expected;
no score is attributed to that invocation. A higher-batch rerun was started to
obtain the fair comparison without changing the samples or seed.

The completed batch-32 rerun produced clean AUC 0.9706, pooled robust AUC
0.9563 and a 50/50 score of **0.9635**. Standard augmentation therefore beat
the robust profile by 0.0016 on this SID slice. The robust profile improved the
noise-sigma-0.10 condition from 0.9283 to 0.9357, but lost more on clean and
other conditions. It is not selected on same-dataset evidence alone.

## 2026-08-29 — WildFake DDIM/ImageNet acquisition

The immutable DDIM archive (6,054,264,809 bytes, 65,713 images) and ImageNet
real archive (1,378,959,009 bytes, 96,788 images) were downloaded and passed
their expected SHA-256 checks and full ZIP integrity tests. Their matching CSV
indexes were also verified. `scripts/extract_wildfake_binary.py` then verified
all four inputs again and successfully extracted 512 deterministic images per
class for a cross-source diagnostic. This set is not the forbidden DALL-E/COCO
demo set.

Both SID-trained DINOv2-L checkpoints were then evaluated cleanly on all 1,024
diagnostic images:

| SID training profile | DDIM-vs-ImageNet clean AUC |
| --- | ---: |
| ResNet18 control | 0.2763 |
| Standard augmentation | 0.4088 |
| Robust augmentation | 0.4012 |

These below-random rankings invalidate the SID-only candidate as a final model.
They do not isolate generator generalization, because both the fake generator
and real-image source changed together; that confound is why the result is
labelled cross-source. Even so, the collapse is conclusive evidence that the
0.96 SID score did not transfer and that source-aligned, multi-generator
training plus held-out-generator evaluation is required.

## 2026-08-29 — source-aligned held-out-generator control

The verified DDPM, DDIM and ImageNet sources were converted into a deterministic
split by `scripts/prepare_wildfake_generator_split.py`: 2,048 DDPM fakes and
2,048 ImageNet reals for training; 512 unseen DDIM fakes and 512 disjoint
ImageNet reals for testing. Keeping ImageNet as the real source on both sides
isolates fake-generator transfer more cleanly than the preceding cross-source
diagnostic.

```bash
HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" \
.venv/bin/python -m aigc_detector.train \
  --dataset-root datasets/wildfake_ddpm2k_train_ddim_test \
  --output-dir outputs/wildfake-ddpm2k-ddimtest-resnet18-robust \
  --model resnet18.a1_in1k --image-size 224 --epochs 3 --batch-size 64 \
  --learning-rate 0.0003 --augmentation robust --device mps
```

The completed run trained all 11,177,025 parameters in 84.21 seconds on Apple
MPS. Clean held-out-DDIM AUC was 0.9424, 0.9417 and **0.9516** after epochs one,
two and three. This is positive generator-held-out evidence, not a final model:
DDPM and DDIM are related processes, the real source is only ImageNet, and full
organizer-transform evaluation had not yet been run at that selection point.

The deterministic full evaluation then completed over all 1,024 held-out
images for clean plus each workshop condition. It scored clean AUC **0.9516**,
pooled transformed AUC **0.9360**, and an official-style score of **0.9438**.
The weakest condition was 0.25x downscale followed by upscale at 0.8910 AUC;
the next weakest was blur sigma 2.0 at 0.9096. JPEG quality 30 retained 0.9355.
This establishes the first complete generator-held-out robustness control.

## 2026-08-29 — Kaggle P100 repair and CUDA verification

The live signed-in Kaggle session initially exposed a Tesla P100 but its default
PyTorch 2.10.0+cu128 did not compile for the GPU's `sm_60` architecture. The
official PyTorch 2.8.0 CUDA 12.6 wheel and torchvision 0.23.0 were installed,
then the kernel was restarted.

Observed after restart: PyTorch `2.8.0+cu126`, CUDA runtime `12.6`, architecture
list including `sm_60`, Tesla P100-PCIE-16GB. An actual seeded 4096-by-4096
CUDA matrix multiplication completed in 0.0412 seconds, returned a finite
sample value and reported 142,737,408 peak allocated bytes. Kaggle is now a
verified training route rather than only a visible GPU allocation.

## 2026-08-29 — DINOv2-L held-out-DDIM robustness result

The identical 2,048-DDPM/2,048-ImageNet training split was rerun with public
pretrained `vit_large_patch14_dinov2.lvd142m`, its backbone frozen and only the
1,025-parameter binary head trained. Total inference parameters were
303,228,929. Two standard-augmentation epochs took 402.35 seconds on Apple MPS;
clean AUC progressed from 0.9825 to 0.9866.

The deterministic full workshop transform evaluation then completed over all
1,024 held-out DDIM/ImageNet images. Clean AUC was **0.9866**, pooled transformed
AUC was **0.9821**, and the 50/50 official-style score was **0.9843**. The
weakest condition was Gaussian noise sigma 0.10 at 0.9689 AUC; JPEG quality 30
retained 0.9796 and blur sigma 2 retained 0.9791. This beats the matched ResNet18
control's 0.9438 score by 0.0405, while retaining the same caveats: only one real
source and the related DDPM-to-DDIM generator transfer are represented.

## 2026-08-29 — modern-generator backbone screen on Kaggle P100

RRDataset was admitted only through its named special-scenario fake images:
771 train and 157 validation images attributed by the paper to SD 3.5 Large and
Flux.1, with every `normal_*` fake and every RR real excluded. Disjoint
WildFake ImageNet photographs supplied the balanced real class. This avoids the
confirmed RR-real/forbidden-COCO overlap but does not eliminate all
fake-source-versus-real-source shortcut risk.

With the split, seed, transforms, frozen-head protocol and 224-pixel input held
fixed, DINOv2-L reached 0.9921 then **0.9972** clean AUC in 99.65 seconds and
PE-Core-L reached **1.0000** in both epochs in 114.21 seconds. Total/trainable
parameters were 303,228,929/1,025 and 315,776,001/1,025; CUDA peak allocations
were 2,083,270,144 and 2,066,800,128 bytes. PE-Core-L is only the provisional
clean-screen leader. The identical full 20-condition workshop evaluation
completed at 0.9970 for DINOv2-L and 1.0000 for PE-Core-L. These exceptionally
high source-separated results are screening evidence, not proof of hidden-set
generalization; a DDIM-versus-fresh-ImageNet transfer check was started before
any backbone-selection claim.

## 2026-08-29 — bounded cross-family acquisition and fixed mixture

To cover generator families without downloading the 47.3 GB GAN archive or
16.2 GB Stable-Diffusion archive in full, `acquire_wildfake_remote_subset.py`
read their immutable ZIP directories over HTTP Range and downloaded
deterministically selected member ranges. Every materialized member was checked
against its ZIP size and CRC32; the whole-object sizes and linked SHA-256 values
were verified from immutable ModelScope revisions but were not recomputed
locally because the full archives were intentionally not downloaded.

- Training GANs: 1,024 each from StyleGAN, BigGAN and StarGAN (3,072 total),
  133,527,110 remote bytes fetched across 122 ranges.
- Held-out GANs: 512 each from GigaGAN, GALIP and DF-GAN (1,536 total),
  368,168,036 remote bytes fetched across 49 ranges.
- Older latent diffusion: 2,048 SDv1.5-DPMSolver images selected from 15,841
  eligible members; 906,217,963 remote bytes fetched across 17 ranges.
- Initial real-source copies: WildFake AFHQ archive 452,402,790 bytes, SHA-256
  `38b8faad90374683a89bfc6011de2465629f88f3ae9db5f14a84b327d7b10350`;
  WildFake FFHQ archive 818,835,036 bytes, SHA-256
  `20ebb2d054cd6477ecd633999e2039379ceec25d9f2da1d0a9752b30b508d7fa`.
  Both ZIP integrity checks succeeded. The 200-pixel AFHQ derivative was later
  superseded for active use by the official source described below; FFHQ stays
  reserved as a real-source holdout.

`prepare_family_mixture.py` then produced an 8,160-row balanced training
manifest: six fake groups of 680 images and two real sources of 2,040 images.
The fixed generator-held-out evaluation contains DDIM, DF-GAN, GALIP and
GigaGAN (512 each) against 512 disjoint ImageNet and all 1,467 AFHQ-v2 test
images. A separate 4,096-row view pairs the same held-out fakes with 2,048
FFHQ reals. COCO demo-only, DALL-E Advanced, all RR reals and RR normal fakes
are explicitly listed as forbidden in the selection record.

The subsequent shortcut audit found that this first draft made all training
reals 200-by-200 JPEGs while 2,471 of 4,080 fakes were PNGs and many were
larger. That draft was rejected before a completed training claim. The official
AFHQ-v2 6,955,288,636-byte ZIP was then accessed through the URL in the pinned
StarGAN-v2 download script at commit
`875b70a150609e8a678ed8482562e7074cdce7e5`. The observed Dropbox ETag was
`1656240417210984d`; the full ZIP was not downloaded and its official script
does not publish a whole-file checksum. Range acquisition fetched 2,040
decode-verified 512-pixel PNGs from the official train tree and all 1,467 from
the disjoint test tree, preserving every member path, size and CRC32. The
corrected mixture therefore gives both labels JPEG/PNG and 200/512-pixel
coverage. The original interrupted training process was stopped at batch 56 of
255 and its incomplete ignored output was moved to macOS Trash.

## 2026-08-29 — family-mixture DINOv2-L clean held-out result

Command:

```bash
HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" \
.venv/bin/python -m aigc_detector.train \
  --train-manifest datasets/family_mixture_v1/train.jsonl \
  --eval-manifest datasets/family_mixture_v1/eval_heldout_generators_known_reals.jsonl \
  --output-dir outputs/family-mix-v1-dinov2l-robust-e1 \
  --model vit_large_patch14_dinov2.lvd142m --image-size 224 --epochs 1 \
  --batch-size 64 --workers 2 --learning-rate .001 --weight-decay .0001 \
  --augmentation robust --freeze-backbone --device mps --seed 20260829
```

The command completed in 546.53 seconds on Apple MPS. The model had
303,228,929 total parameters and 1,025 trainable parameters; the final training
loss was 0.4009. MPS reported 1,251,464,960 current allocated bytes and
3,280,666,624 driver allocated bytes at the final snapshot.

Clean AUC on all 4,027 generator-held-out images was **0.9794**. Per-generator
AUC against all reals was DDIM 0.9900, DF-GAN 0.9745, GALIP 0.9851 and GigaGAN
0.9682. Per-real-source AUC against all fakes was AFHQ-v2 0.9930 and ImageNet
0.9406. At an uncalibrated 0.5 threshold, ImageNet true-negative rate was only
0.7871, so the ranking is strong but its raw probabilities are not yet safe to
present as calibrated confidence. This result advances to the FFHQ real-source
holdout and full transform evaluation; it is not yet the selected final model.

PE-Core-L could not be trained through the same local Apple-MPS route. Attempts
at batch sizes 64 and 32 both aborted before the first completed batch with an
MPSNDArray buffer-size assertion (required buffers 262,144 and 131,072 bytes,
respectively). These are failed compute-route attempts, not accuracy results;
their empty ignored output directories were moved to macOS Trash. The already
verified Kaggle P100 route remains the intended PE-Core comparison environment.

The separate FFHQ real-source gate rejected the DINOv2-L candidate despite its
0.9794 known-real result. Pairing the same 2,048 unseen-generator fakes with
2,048 FFHQ reals produced **0.4299** clean AUC. Per-generator AUCs were DDIM
0.5665, DF-GAN 0.4268, GALIP 0.4242 and GigaGAN 0.3022; FFHQ true-negative rate
at threshold 0.5 was only 0.0356. The model had seen synthetic human faces from
StyleGAN/StarGAN but no genuine human-face source, so the evidence supports a
content/source shortcut rather than inadequate capacity.

Family mixture v2 therefore keeps the same six balanced fake groups but splits
the 4,080 reals equally across ImageNet, official AFHQ-v2 and FFHQ. It follows
the official FFHQ partition convention: numeric IDs 00000-59999 form the
training pool and 60000-69999 form the validation pool. The selected 1,360
training faces and 2,048 validation faces are disjoint. This changes the data
coverage, not the architecture, so the next DINO run isolates whether genuine
human-face coverage repairs the failure.

## 2026-08-29 — RR-special to DDIM cross-generator backbone gate

The earlier Kaggle P100 RR-special screen was followed by a disjoint
cross-generator check. Both frozen backbones were trained on the same named
SD 3.5 Large / Flux.1 special-scenario fakes and the same ImageNet real-source
pool, then evaluated on 512 newly materialized DDIM fakes and 512 disjoint
ImageNet reals. The corrected evaluation cell completed in 64.513 seconds.

- DINOv2-L clean AUC: **0.7163200378**
- PE-Core-L clean AUC: **0.9249153137**

This rejects the apparent near-perfect RR-special clean score as a sufficient
backbone-selection result and advances PE-Core-L to the broad-mixture
comparison. It does not establish hidden-set performance: the real class still
comes from one ImageNet acquisition pipeline, the fake test represents only
DDIM, and no workshop transformation was applied in this cross-generator gate.

## 2026-08-29 — exact broad-mixture GPU transfer package and external gates

`package_kaggle_subset.py` packaged only the image files referenced by the v2
training, known-real evaluation and FFHQ-real evaluation manifests. Shared
files were content-addressed and stored once. The command completed with
14,235 unique images, 2,793,761,689 source bytes and a 2,802,930,796-byte
uncompressed ZIP. Its SHA-256 is
`d6db56897fc4fa855349e5a5992481b0c81d2c39cb5ed91f7638f6e7171fb709`;
the content inventory SHA-256 is
`66257154698bcfb1bb5a7dbc91a2a58d951425cb237bc014cc3f3cc583e3ac23`.
The packager rejects either known demo-only directory by path and the ignored
package is not a repository artifact. Thirteen automated tests passed after
the packager and its two tests were added.

Three additional diagnostic-only manifests were created and explicitly kept
out of training: 2,048 held-out WildFake generator images against 2,048 CIFAKE
real images; 2,048 CIFAKE synthetic images against 2,048 disjoint FFHQ real
images; and the extracted SID_Set shard with 272 synthetic and 314 real images.
These gates intentionally expose low-resolution and dataset-source shifts.
They are diagnostics, not clean estimates of hidden performance, because the
class sources differ within each gate.

## 2026-08-29 — family-mixture v2 human-face repair

The v2 DINOv2-L command used the same architecture, optimizer, augmentation,
held-out fake generators and seed as v1, changing only the balanced real-image
mixture to include disjoint FFHQ training faces:

```bash
HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" \
.venv/bin/python -m aigc_detector.train \
  --train-manifest datasets/family_mixture_v2/train.jsonl \
  --eval-manifest datasets/family_mixture_v2/eval_heldout_generators_known_reals.jsonl \
  --output-dir outputs/family-mix-v2-dinov2l-robust-e1 \
  --model vit_large_patch14_dinov2.lvd142m --image-size 224 --epochs 1 \
  --batch-size 64 --workers 2 --learning-rate .001 --weight-decay .0001 \
  --augmentation robust --freeze-backbone --device mps --seed 20260829
```

The command completed successfully in 878.99 seconds on Apple MPS. It trained
1,025 of 303,228,929 total parameters on 8,160 images; final loss was 0.4595.
MPS reported 1,251,464,960 current and 3,280,666,624 driver-allocated bytes.
Clean AUC on the 4,027 known-real-source/held-out-generator gate was **0.9790**.
The weakest fake generator was GigaGAN at 0.9614 AUC; the weakest real source
was ImageNet at 0.9419 AUC. ImageNet true-negative rate at the uncalibrated 0.5
threshold remained only 0.7090, so this is ranking evidence, not calibration.

The decisive disjoint FFHQ gate then completed successfully on 4,096 images.
Clean AUC was **0.9888**; per-generator AUC was DDIM 0.9918, DF-GAN 0.9900,
GALIP 0.9962 and GigaGAN 0.9774. FFHQ true-negative rate at 0.5 was 0.9395.
Compared with v1's 0.4299 FFHQ AUC and 0.0356 true-negative rate, this directly
supports the diagnosis that missing genuine-face coverage—not model size—caused
the earlier failure. It still does not establish hidden-set performance.

## 2026-08-29 — independent source-paired gate rejects family-mixture v2

The v2 checkpoint was next evaluated without retraining on a combined 8,778-row
diagnostic gate. It contains held-out WildFake generators, CIFAKE's paired
CIFAR-10 real and Stable-Diffusion classes, SID_Set's paired real and synthetic
classes, and disjoint FFHQ validation reals. This gate is diagnostic rather than
a hidden-set estimate, but its paired subsets directly test whether the model
learned the source or resolution instead of the real-versus-AI distinction.

The command completed successfully and wrote
`outputs/family-mix-v2-dinov2l-robust-e1/eval-external-sources-clean.json`.
Pooled clean AUC was **0.7921**, but the matched within-source results were only
**0.5599** for CIFAKE fake versus CIFAKE real and **0.5164** for SID fake versus
SID real. The worst generator/real-source pair was SID fake versus CIFAKE real
at **0.2570 AUC**, meaning its ranking was strongly reversed. CIFAKE real
true-negative rate at the uncalibrated 0.5 threshold was only 0.1143.

These results reject v2 as a final candidate despite its strong FFHQ and
known-source scores. The active repair experiment adds balanced paired
CIFAKE-train and SID-train real/fake controls while keeping CIFAKE-test,
SID-validation, the four WildFake generators and FFHQ validation disjoint.
Selection will use the weakest generator/real-source pair in addition to pooled
AUC; no workshop demo-only image is included in either training or diagnostics.

## 2026-08-29 — paired-source repair succeeds but remains provisional

The isolated repair retained the v2 DINOv2-L architecture, frozen-head
protocol, optimizer, augmentation profile, seed and external selection gate.
It added only balanced paired controls from CIFAKE train (680 per label) and
SID_Set train (287 per label), plus equally divided additional authentic images
to keep the full 10,094-row training manifest label-balanced. CIFAKE test and
SID validation remained untouched.

The one-epoch command completed successfully on Apple MPS in 1,077.89 seconds.
It trained 1,025 of 303,228,929 parameters, reached 0.5060 final loss, and wrote
`outputs/family-mix-v2-source-repair-dinov2l-robust-e1/model.pt` plus its JSON
report. Clean AUC on the unchanged 8,778-row external gate increased from
**0.7921 to 0.8932**. The matched CIFAKE fake-versus-real AUC increased from
**0.5599 to 0.8004**; matched SID increased from **0.5164 to 0.7057**. The
weakest generator/real-source pair increased from **0.2570 to 0.7057**, and the
weakest generator and real-source aggregate AUCs were 0.8284 and 0.8012.

This is direct evidence that paired source controls reduced the earlier
dataset/resolution shortcut. It is not evidence that the shortcut is solved:
SID remains the weakest matched pair, SID real true-negative rate at the
uncalibrated 0.5 threshold is only 0.4745, no named modern diffusion-transformer
gate has yet been run, and the full individual-transform matrix is still
pending. The candidate advances to those gates but is not selected as final.

## 2026-08-29 — verified modern diffusion-transformer acquisition

`scripts/acquire_ditfake_fakes.py` completed against the immutable
`Jouesmak/DiTFake` revision
`ca9ea06c8f926c3a11ca4b657074cc7cbb99e5c7`. It acquired exactly 5,000
synthetic files each for FLUX.1-schnell, PixArt-Sigma-XL-2-1024-MS and Stable
Diffusion 3 Medium: 15,000 files and 21,441,316,312 bytes total. The canonical
path/size/SHA-256 inventory digest is
`aca1ec12963bcf54dacec0229ed7cb42041fac9214a9e7d99a2a3e29e0491d09`.

The acquisition included only the three `1_fake` trees. Every DiTFake COCO
`0_real` image was excluded to preserve the organizer's demo-only boundary.
The fixed experiment policy permits FLUX and SD3 in the later modern training
mixture but keeps the entire PixArt family unseen for selection. Before that
training, all three generators are scored by the current source-repaired
checkpoint to preserve an uncontaminated pre-training diagnostic.

## 2026-08-29 — broad-mixture PE-Core Kaggle route, first launch blocker

The verified 2.8 GB v2 transfer package was uploaded as a private Kaggle
dataset, created successfully and attached to the existing P100 notebook. Its
local ZIP SHA-256 is
`d6db56897fc4fa855349e5a5992481b0c81d2c39cb5ed91f7638f6e7171fb709`;
the runner rechecks this before extraction. The first launch installed the
P100-compatible PyTorch 2.8.0/CUDA 12.6 build and `timm` 1.0.19 inside the same
cell as the run, then failed before model creation with `ImportError: cannot
import name '_Ink' from PIL._typing`. This was an environment/Pillow
inconsistency after package replacement, not a training or accuracy result.
The repair route separates package installation from execution and restarts the
session before the self-contained run. No PE broad-mixture result is claimed
until that command completes.

## 2026-08-29 — v5 Kaggle package verification and training start

The source-controlled v5 package was uploaded to the private Kaggle dataset
`track5-v5-private-training-package`. The local source ZIP is
`family-mixture-v5-dedup.zip`, size `8,205,799,634` bytes, SHA-256
`31494bf4d8d345a26d838b35012ab1cfce827a7a892c8e5844880effaf0a6ae4`;
its embedded inventory SHA-256 is
`9b9a2a6c21d40d9bf539bc3a0440d402526c3698f8f6faaac5b62901a2bd76b8`.
Kaggle expands uploaded ZIP datasets, so the runner was repaired to validate
the expanded representation rather than incorrectly requiring the original
archive to remain mounted. It located the single matching embedded inventory,
hashed all **30,257** content-addressed images, checked their total byte count,
and rechecked every packaged manifest checksum. The validation completed
successfully in the live P100 notebook.

The same cell installed PyTorch 2.8.0 with CUDA 12.6, torchvision 0.23.0,
Pillow 11.3.0 and timm 1.0.19, then launched the runner in a fresh Python
subprocess without restarting the Kaggle session. DINOv2-L weights
(`vit_large_patch14_dinov2.lvd142m`) downloaded successfully and training
reached 40/265 batches with loss decreasing from 0.7445 at batch 20 to 0.6636
at batch 40. This proves package integrity, dependency compatibility and active
GPU training only; it is not a completed model or evaluation result.

## 2026-08-29 — pre-training modern-generator and crop-preserving gate

Before adding any DiTFake image to training, the source-repaired DINOv2-L
checkpoint was evaluated on 1,500 FLUX.1-schnell, 1,500 Stable Diffusion 3
Medium and 1,500 PixArt-Sigma images plus the unchanged 4,410 external real
images. The command used aspect-ratio-preserving short-side resize plus centre
crop, not the earlier rectangular-to-square stretch:

```bash
.venv/bin/python -m aigc_detector.evaluate \
  --manifest datasets/family_mixture_v3/eval_modern_pretrain_diagnostic.jsonl \
  --checkpoint outputs/family-mix-v2-source-repair-dinov2l-robust-e1/model.pt \
  --output outputs/family-mix-v2-source-repair-dinov2l-robust-e1/eval-modern-pretrain-clean-crop.json \
  --batch-size 64 --workers 2 --preprocess-mode short_side_crop --device mps
```

The command completed successfully on 8,910 images. Clean AUC was **0.8289**.
Per-generator AUC against all reals was FLUX **0.7862**, SD3 Medium **0.8113**
and PixArt-Sigma **0.8893**. The weakest real-source aggregate was SID_Set at
**0.7243**; the weakest generator/real-source pair was FLUX versus SID_Set at
**0.6639**. This is the uncontaminated pre-training reference for the v3
modern-generator experiment. It confirms that the source-paired repair
transfers materially above random to recent DiTs but is not strong enough to
select as final. Because the evaluator did not yet record elapsed/resource
fields when this command started, no exact duration or MPS allocation is
claimed; that instrumentation has now been added for subsequent runs.

## 2026-08-29 — v3 transfer-package duplicate-member rejection and repair

The first v3 package contained 23,092 source paths and passed the local
same-path reuse test, but Kaggle's archive validator rejected it because
distinct source paths with identical bytes produced duplicate content-addressed
ZIP member names. No dataset was created from that upload. This exposed a real
packager defect rather than a training failure.

`package_kaggle_subset.py` now deduplicates by SHA-256 content rather than only
by resolved source path, records both source-path and unique-content counts,
and writes package format version 2. The regression test now uses two distinct
files with identical bytes and proves that the archive contains one image
member. Nine focused tests passed after the repair. The rejected staged upload
has not been deleted without action-time confirmation; the corrected package
is created under a different ignored filename.

## 2026-08-29 — deterministic WildFake real-content expansion

The full WildFake object inventory was re-read from ModelScope and confirmed a
reported 1.29 TB collection rather than a practical all-or-nothing download.
Three immutable real-image archives were selected to broaden subject matter
without using the prohibited COCO branch. The range-acquisition command
completed successfully for each archive with seed `20260829`, 16 selection
windows and 16 reserve candidates:

| Source | Archive revision | Whole archive bytes / SHA-256 | Retained | Remote bytes fetched | Selection inventory SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| CelebA-HQ portraits | `0f5dee265fd112acf1d661f14b1fbb644f185035` | `350991722` / `bfc71b04c16786267781110c52b36c515558c8889ed657cc97cf8d89b629b531` | 1,024 | 15,179,908 | `f2a52a050d260bee26fc67cb9e42d133fa9dc2b57dd521c5b9f6d67df2c720c6` |
| LSUN Church scenes | `3ed3ae4eebef779b872035a1474426616ce4773f` | `1162401362` / `e31134744e3b0b3e50032e39a701269660cac625f98460f8374c9e32e1fca855` | 1,024 | 23,211,274 | `a45963b6932937a11a7e24fe2e3978d35708bb86aaf74aa5b0f50870629cdc82` |
| LAION-5B web mix | `3a581485c428f5c62b6f340eef1b166abea779c5` | `24795313028` / `cf36b3595a4928d6915da667c87363c302ba29646de30d5ea19e8e98e41bb42b` | 1,024 | 124,645,968 | `7711555afb68d165251d4477133ca710f8650920bacab7c9b19ea39354d5c39f` |

All 3,072 selected files decoded successfully; no reserve candidate was needed.
Their manifest SHA-256 values are respectively
`d4acf854cfff13a2bd0e7fa7e4cfbb87685e12d5368408aa45449010002d2a19`,
`64a0f00557e5489112a20a2acc95d27968a2ac27bbc7ca3129eb5c3e396f3777`
and `f02cf5f9ff394cb1b835041f84bf7f0b812531d369ef8923645778241cd9675b`.
The retained directories occupy approximately 14 MB, 16 MB and 96 MB. They
remain ignored local data and are not redistributed.

Before any of these sources entered training, the content-holdout builder
combined them with 8,868 disjoint fake images spanning nine named groups. It
checked the active source-repair training manifest's 10,094 rows and found zero
resolved-path overlap and zero SHA-256 content overlap. This gate diagnoses the
remaining portrait/scene/web-content shortcut risk; it does not declare that
risk solved.

The clean evaluation of that first 11,940-image content gate completed on MPS
in **957.73 seconds**. The checkpoint contained 303,228,929 parameters and MPS
reported 1,212,915,968 current plus 3,272,278,016 driver-allocated bytes at the
end. Overall clean AUC collapsed to **0.6147**. The real-source AUCs were
CelebA-HQ **0.6489**, LAION-5B **0.5939** and LSUN Church **0.6012**; their
uncalibrated true-negative rates at 0.5 were only 0.2588, 0.2441 and 0.1904.
FLUX was the weakest fake generator at **0.4760** AUC against all three real
sources, and the worst pair was FLUX versus LSUN Church at **0.4560**.

This result rejects the source-repair checkpoint as a final candidate and
directly answers the content-shortcut question: earlier FFHQ and paired-source
repairs fixed two observed failures but did not create general content
invariance. The next mixture must broaden both labels, then retain at least one
complete content source and generator family as a veto gate.

Three additional WildFake generator archives were then sampled by the identical
range protocol, again with seed `20260829` and no decode rejection:

| Generator | Family | Archive bytes / SHA-256 | Retained bytes | Selection inventory SHA-256 | Manifest SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| ADM | pixel diffusion | `18548017747` / `f3221217f7c4e4a220bc51efc154fe79ac3efbd37b56409ec4f465b9cc41a09a` | 121,314,865 | `8445a56ea439b47882522f080c86da90d7631728ee5034c87d3d7b1bab05151d` | `f253d6e5a4c66ba9ebe5fbc56cb6159e29c7d06922461a35d1dd545c7d2ca329` |
| VQDM | vector-quantized diffusion | `17377092724` / `5538509c15db4ab69c0d7d45995799ac5975b6bab18975832d4c0e40a0306feb` | 115,338,564 | `17ee3f925a0431fd6187d2d2c00c7d04a020dca1d6aea3c2ed212738ea680512` | `270f1366782c6d62a2b275827d77da114c517012eb496d881cd77bbe8a671dc1` |
| Imagen | cascaded diffusion | `17069983804` / `a102dceb9e1031aee02d04422f9b89b854ecda3dcf9bc14fe6590d388130cdde` | 366,219,311 | `43c115005b8a6c5ed689f2894ed1c29943e6ccda43afc401af1f83336963b1e6` | `42f6cc0185f8051b95dd92bd02125defa35c04a16631d3342db47541d9cc3864` |

The balanced breadth gate contains these 3,072 fakes and the 3,072 unseen real
portrait/scene/web images. It also has zero path and SHA-256 content overlap
with the active training manifest. These new families remain diagnostic until
their results justify using any of them in a later mixture; adding data volume
without a held-out measurement is explicitly rejected.

The balanced 3,072-image breadth evaluation completed on MPS in **289.94
seconds**. Overall clean AUC was only **0.5236**. Per-generator AUC against all
three real sources was ADM **0.4656**, Imagen **0.4955** and VQDM **0.6096**;
the worst pair was ADM versus LSUN Church at **0.4521**. This is near-chance
transfer and independently confirms that the source-repair checkpoint learned
too narrow a boundary. ADM and VQDM are admitted to the v4 training experiment,
while Imagen remains a complete fake-generator holdout. CelebA-HQ and LSUN
Church enter v4 training, while the broad LAION-5B source remains completely
unseen.

The resulting v4 manifests contain 16,910 balanced training rows (8,455 per
label), 12 fake generator groups and seven real sources. The 12,326-row
selection manifest keeps Imagen and PixArt completely unseen as fake generators
and LAION-5B completely unseen as a real source. Train and selection have zero
resolved-path and zero SHA-256 content overlap. The container audit still finds
a real risk—6,642/8,455 reals are JPEG while 6,166/8,455 fakes are PNG, and the
source-resolution distributions differ—so v4 is an experiment, not a presumed
fix. Short-side crop removes direct aspect-ratio dimensions at model input, but
it cannot erase acquisition or compression history.

The v4 transfer package completed successfully with format version 2:

```text
source paths       29236
unique images      29233
source bytes       8085269095
package bytes      8105007227
package SHA-256    90421d04d8f70d391b28e703d0f7aca94393471336696e4138138107379f017e
inventory SHA-256  4239ddf2846a3b2d05ac03fbec872d7b0f8cbd5d606c59c52bc7c198df288a69
```

The self-contained P100 runner verifies those values, trains the identical
DINOv2-L and PE-Core-L frozen-head candidates, and evaluates both the full
selection manifest and the dedicated content holdout. It uses source-balanced
sampling—equal probability mass for real/fake, then equal mass for each named
source within its label—to prevent one large branch dominating the epoch. The
ordinary random-row policy remains a comparison, because balancing itself can
over-repeat small sources and is not assumed to be universally superior.

## 2026-08-29 — v4 source-balanced DINOv2-L clean selection result

The first v4 candidate completed locally with the fixed manifests and no
post-result split changes:

```bash
.venv/bin/python -m aigc_detector.train \
  --train-manifest datasets/family_mixture_v4/train.jsonl \
  --eval-manifest datasets/family_mixture_v4/eval_selection.jsonl \
  --output-dir outputs/family-mix-v4-dinov2l-source-balanced-robust-e1 \
  --model vit_large_patch14_dinov2.lvd142m --freeze-backbone \
  --image-size 224 --epochs 1 --batch-size 64 --workers 2 \
  --learning-rate 0.001 --weight-decay 0.0001 \
  --augmentation robust --sampling source-balanced \
  --eval-preprocess short_side_crop --device mps --seed 20260829
```

The command trained on 16,910 images and completed in **1,096.28 seconds**.
The model contains **303,228,929 total parameters** and 1,025 trainable
parameters. End-of-run MPS observations were 1,251,468,800 current allocated
bytes and 3,280,666,624 driver allocated bytes. Clean selection AUC on 12,326
disjoint images was **0.8900**.

The complete Imagen and PixArt fake holdouts reached **0.8693** and **0.9424**
AUC against all reals. The other fake groups ranged from 0.8048 for CIFAKE
Stable Diffusion to 0.9607 for DF-GAN. The real-source results expose a serious
remaining shortcut: CIFAKE real was 0.9238, FFHQ validation 0.9811, SID_Set
0.8580, but completely unseen LAION-5B real was only **0.6502**. At the
uncalibrated 0.5 threshold the LAION true-negative rate was only **0.2158**.
The weakest generator/source pair was CIFAKE Stable Diffusion versus LAION at
**0.4471**, below chance; Imagen versus LAION was 0.5956.

This is a large improvement over the old checkpoint's 0.6147 broad-content
result and demonstrates useful cross-generator transfer, but it does **not**
pass the source-collapse veto. The candidate is therefore diagnostic until its
independent content-holdout result is known and a training/control change
addresses the LAION false-positive problem. The 0.8900 pooled number must not
be presented alone as evidence of general real-vs-AI detection.

## 2026-08-29 — disjoint LAION control and v5 manifest

The v4 failure was concentrated on broad LAION real photographs, so a second
official WildFake LAION range subset was acquired from the same immutable
24,795,313,028-byte archive with seed `20260830`. The command fetched
127,196,121 remote bytes, retained 1,050 decoded images (101,983,729 bytes), and
completed without a rejected member. The new shard has zero archive-member and
zero byte-SHA overlap with the original 1,024-image shard. Its selection and
manifest SHA-256 values are respectively
`78d4d20075b21012f1a1426833b38df0099d0a2eb9b5419d5604c7191f0cda66`
and `8eb0ef4ba41140b02bc07cb7a56aa960d17c68375fd9cf56cd7609ef5b40d24c`.

The deterministic v5 manifest keeps the same balanced 16,910-row and
12-generator training size as v4, but replaces LSUN Church training reals with
LAION shard A. Evaluation contains the complete untouched LSUN Church source,
the first 1,024 images of byte-disjoint LAION shard B, and complete Imagen and
PixArt fake holdouts. Train versus the 13,350-row selection gate has zero path
and zero SHA-256 overlap. Train, selection and content-gate manifest SHA-256
values are respectively
`83c551f243f4327d9b4d649478c5dc5cc0919c3f35498f5929c8e1d485b1545a`,
`0b42c3d1db77ad9cad1d2a4e9f30b7398cf0b7a3245dfe3581d2dd4100d92c7d`
and `89a7585eb6351577226fe582da9efce06b5d16e9118ac403c090f370624116ac`.
This is a controlled response to one observed failure, not a post-hoc claim
that web-photo generalization is solved.

The independent v4 content-holdout command then completed successfully on MPS
in **580.50 seconds** over 7,916 images. As expected from the full selection
breakdown, its clean AUC was **0.6502**. CIFAKE Stable Diffusion versus LAION
was **0.4471**, Imagen versus LAION 0.5956 and PixArt versus LAION 0.7719.
End-of-run MPS observations were 1,212,915,968 current and 3,272,278,016
driver-allocated bytes. This independently confirms the v4 veto; v4 is not the
selected candidate.

## 2026-08-29 — anti-shortcut controls prepared before v5 selection

Two evidence-backed controls were implemented while the fixed v5 run remained
in progress. Neither is promoted without the identical v5 holdout result.

1. **Label-independent JPEG normalization.** The official UnbiasedGenImage
   implementation says its JPEG alignment is applied before the ordinary model
   transformations and reports that removing compression and size biases
   improved cross-generator performance. The harness now supports
   `--codec-normalization jpeg_q96`, applied to every real and fake image. For
   a workshop transformation, the one named condition is applied first and
   then the same deterministic model preprocessing is applied to both labels.
   This is explicitly an ablation because it can also erase useful generation
   evidence.
2. **Stay-Positive head.** The ICML 2025 paper specifies ReLU features, a
   zero-initialized last layer, frozen feature extractor, binary cross-entropy,
   and projection of negative final-layer weights back to zero after each
   optimizer update. The harness now exposes this independently implemented
   structure as `--head-mode stay_positive`. It is not mixed silently into the
   ordinary linear-head candidate.

Primary sources:

- <https://www.unbiased-genimage.org/>
- <https://github.com/gendetection/UnbiasedGenImage>
- <https://arxiv.org/abs/2502.07778>
- <https://github.com/AniSundar18/AlignedForensics>

The combined code path was smoke-tested successfully on CPU:

```bash
.venv/bin/python -m aigc_detector.train \
  --dataset-root datasets/cifake \
  --output-dir outputs/smoke-stay-positive-codec \
  --model resnet18.a1_in1k --no-pretrained --freeze-backbone \
  --head-mode stay_positive --codec-normalization jpeg_q96 \
  --image-size 32 --epochs 1 --batch-size 8 --workers 0 \
  --max-train-per-class 4 --max-eval-per-class 4 \
  --augmentation robust --eval-preprocess short_side_crop \
  --device cpu --seed 20260829
```

Observed: 8 training and 8 evaluation images, 11,177,025 total parameters,
513 trainable parameters, 0.854 seconds, clean AUC 0.625 on the deliberately
tiny non-performance smoke set. Checkpoint reload evaluation also succeeded
in 0.341 seconds and preserved both `stay_positive` and `jpeg_q96` metadata.
The current test suite reports 19 passing tests.

## 2026-08-29 — v5 source-swap result

The fixed v5 DINOv2-L command completed successfully on MPS. It trained for
one epoch on 16,910 rows and evaluated the unchanged 13,350-row selection
manifest. Runtime was **1,856.20 seconds**, final training loss **0.48309**,
and the model contained 303,228,929 total / 1,025 trainable parameters. MPS
reported 1,251,468,800 current and 3,280,666,624 driver-allocated bytes.

Clean AUC was **0.87224**. Training on LAION shard A repaired the held-out,
byte-disjoint LAION shard B to **0.86730** AUC and a 0.7236 true-negative rate
at threshold 0.5. But removing Church from training caused held-out LSUN Church
to fall to **0.65655** AUC and only 0.2197 true-negative rate. Imagen was the
weakest generator at **0.70809** AUC; the worst pair was Imagen versus Church
at **0.34840**, well below chance. PixArt remained strong at 0.94421.

The source swap therefore moved rather than solved the failure: v4 failed
unseen LAION while v5 fails unseen Church. v5 is rejected as a final candidate.
The next comparison must keep both fixed domains visible, test an anti-shortcut
head or codec alignment independently, and never use the pooled 0.87224 as the
selection claim.

## 2026-08-29 — byte-disjoint Church control acquired

To prevent the next mixture from treating Church as either fully seen or fully
unseen, the range-acquisition tool now accepts `--exclude-manifest` and removes
all recorded archive members before deterministic selection. Using seed
`20260830`, it fetched 23,146,124 bytes from the immutable 1,162,401,362-byte
Church archive and retained 1,024 decodable images in
`datasets/wildfake_real_church_subset_b`. The command completed successfully.
The original and new shards have zero archive-member overlap and zero SHA-256
content overlap. Shard-B's selection-inventory SHA-256 is
`bbe498f14e24dfcd50ab8aee26a96a9a2f1a2b4aba622e25d08d59ae6fd1c5c7`;
its manifest SHA-256 is
`b046bae166686436f9dd296cb9261be46c35f9ad7284469554718373d672dc33`.
This is a prepared control, not evidence of model improvement.

## 2026-08-29 — v6 paired-source candidate manifest prepared

`prepare_family_mixture_v6.py` completed successfully. It adds Church shard A
and 512 previously unused images from each of FLUX.1-schnell and Stable
Diffusion 3 Medium to v5, preserving exact class balance at 9,479 real and
9,479 fake images. LAION shard B and Church shard B remain evaluation-only;
Imagen and PixArt remain completely absent from training. The 18,958 training
paths and 13,350 selection paths have zero resolved-path overlap and zero
SHA-256 content overlap.

Manifest SHA-256 values are:

- train: `2393c4a648b9cd85be30ad664982e6c433cbf2c0df6c91398b99a9c9e4af319d`
- selection: `bb7a971ad2db592fabab5a52d2ef6127e7aa45d2054c59700696ffdf93d96974`
- content holdout: `34c9fdce95b5084a8a4330bbe05430be0f9b798d062659fdaa1f847dfd6e7a05`
- selection report: `b85c3bfd6082cd9c45289cf766450b81b47f7906bb764d548d90d33f2c897740`

This manifest is deliberately queued behind the active v5 head/backbone
comparisons. Its existence is not a score or a selected solution.

## 2026-08-29 — Kaggle evaluation reuse repair

The active v5 Kaggle run revealed that the self-contained runner evaluated the
13,350-row selection manifest and then redundantly forwarded the 8,940-row
content gate through the frozen backbone even though every content-gate image
is already in the selection manifest. The next runner now prints evaluation
progress and derives the content-gate metrics from the already-computed
predictions by immutable `image_sha256`. It rejects missing hashes, duplicate
selection hashes and label mismatches. Two regression tests cover successful
subset derivation and a missing-image failure. The complete suite now reports
**21 passing tests**. This repair does not alter the active run or any score;
it removes redundant inference from future candidates.

## 2026-08-29 — offline directory-to-JSON contract reverified

The submission entry point was rechecked independently of training:

```bash
PYTHON_EXECUTABLE=.venv/bin/python AIGC_DEVICE=cpu ./run.sh \
  /tmp/track5-run-smoke.UXoSGf \
  /tmp/track5-run-smoke-output.json \
  outputs/smoke-resnet18/model.pt
```

The command completed successfully on two symlinked CIFAKE real images. It
reported two images, 11,177,025 total parameters and CPU execution. The JSON
decoded successfully and contained one `image_path` plus continuous numeric
`pred` value per image (0.5199949 and 0.5103862). This verifies the interface,
not model quality. The eventual selected checkpoint must pass the same smoke
before release.

## 2026-08-29 — Kaggle P100 v5 DINOv2-L reproduction

The checksum-verified Kaggle v5 DINOv2-L candidate completed successfully on a
Tesla P100-PCIE-16GB using PyTorch 2.8.0+cu126. It trained on 16,910 rows and
evaluated 13,350 disjoint selection rows plus the 8,940-row content subset. The
run took **968.69 seconds**, reached final loss **0.483076**, and reported
2,397,286,912 peak CUDA-allocated bytes. The model contained 303,228,929 total
parameters and 1,025 trainable parameters.

Clean selection AUC was **0.872276**, reproducing the local MPS value 0.872238.
The content-holdout AUC was **0.762017**. Complete holdout generator AUCs were
Imagen **0.707934** and PixArt **0.944274**. Byte-disjoint LAION shard B scored
**0.867330**, while the completely unseen Church source scored only
**0.656705**. The weakest generator/source pair was Imagen versus Church at
**0.348354**. This repeatability validates the runner and confirms the veto;
it does not rescue v5. The identical PE-Core-L comparison then began and
reached 40/265 batches.

## 2026-08-29 — v6 checksum-pinned transfer package

The v6 packager completed successfully with ZIP_STORED format. It scanned
32,308 manifest references, stored 32,305 unique image byte streams and
deduplicated three repeated contents. The source images total 9,619,933,981
bytes. The private ignored package is 9,641,928,657 bytes with SHA-256
`1655bf6350fc60e18e43ad74fafe69df7954fab2229eaf0962033b72cf8547f3`;
its embedded inventory SHA-256 is
`30d8f4791b125c208a7dcd1d2b3915098b932dd0e0127363149c64c3ce41428d`.
Packaged train, selection and content manifest SHA-256 values are respectively
`c79a2c5844a10f4e22a241dac2e8e9e113759c5e4cd2a57d8fa2376804350d5c`,
`cf869ce92e19a6f4881ab1b5a522f42a1dcb29019f420b947d23e85aa2943a0b`
and `d7602925fc3433c1d0e0435afd1acefa6aa03d528fd3d5115c1bb8422f3346de`.
`kaggle_train_v6.py` pins these values and exact row counts. The package is
prepared, not yet a trained or evaluated result.

## 2026-08-29 — v5 Stay-Positive control rejected

The fixed v5 DINOv2-L Stay-Positive control completed successfully on Apple
MPS. It used the identical 16,910-row training manifest, 13,350-row selection
manifest, source-balanced sampler, one-epoch robust augmentation policy and
seed as the ordinary linear-head v5 candidate. Runtime was **2,678.06
seconds**; the model contained 303,228,929 total parameters and 1,025 trainable
parameters. The final training loss was 0.677108. MPS reported 1,251,726,848
current and 3,280,666,624 driver-allocated bytes.

Clean AUC was only **0.751409**. The weakest fake generator was Imagen at
**0.638180**, the weakest real source was CIFAKE at **0.589238**, and the worst
pair was Imagen versus CIFAKE at **0.428643**. This is worse than the ordinary
linear v5 head on every primary gate and is rejected. The implementation is
retained as a documented negative control; the paper result was not assumed to
transfer to this data and backbone.

## 2026-08-29 — v6 perceptual-overlap screen and exclusion

`audit_perceptual_overlap.py` computed 64-bit difference and average hashes for
all **18,958** v6 training images and **13,350** evaluation images, then used
Hamming radii four/four to produce manual-review candidates. It found three
candidate pairs and zero cross-label pairs. Visual review confirmed that one
pair was the same photographed black object with different size text:

- training SHA-256 `e8c7de2e490b3830403178cbdb26fcd06a6ed58e84f39cc60b507b1b213a1c0c`
- evaluation SHA-256 `ff3e8968b04eaa4d95b3d9b8bd88e61a673bf0589a7be761cd71470559cd3ab1`

The other two candidates were unrelated product photographs and remain in the
gate. The confirmed near-duplicate is excluded from scoring by immutable hash
in the Kaggle runner and by archive member in the local filtered manifest. The
effective fixed selection/content gates therefore contain **13,349** and
**8,939** rows. The uploaded package itself remains unchanged and checksum
verifiable; every report records the explicit exclusion. The test suite now
reports **24 passing tests**.

## 2026-08-29 — v6 parallel DINO/PE launch

The private v6 Kaggle dataset finished processing with 32,305 unique image
files and the expected embedded inventory SHA-256
`30d8f4791b125c208a7dcd1d2b3915098b932dd0e0127363149c64c3ce41428d`.
The notebook attached that private input successfully. A restored kernel had
reverted to incompatible PyTorch 2.10.0+cu128, so the first v6 start was
interrupted during package discovery before training. PyTorch 2.8.0+cu126 and
torchvision 0.23.0 were reinstalled; after a kernel restart, the runtime again
reported CUDA 12.6, an architecture list containing `sm_60`, and Tesla
P100-PCIE-16GB. The checksum-pinned **PE-Core-L-only** v6 run is now active.

The earlier v5 PE-Core process completed all 265 training batches at loss
0.301945 but the Kaggle kernel reconnected before evaluation, and the runner
saves checkpoints only after evaluation. Therefore no v5 PE score or reusable
checkpoint exists; its loss is not treated as a result. PE-Core advances to v6
because the already completed source-aligned cross-generator gate scored
0.924915 versus DINOv2-L's 0.716320, while DINOv2-L remains the parallel local
control. The local v6 DINOv2-L run is active on Apple MPS with the same cleaned
selection gate. Neither v6 candidate has a score yet.

## 2026-08-29 — fixed group-balanced robustness gate

Before any v6 score was available, `prepare_balanced_robustness_gate.py` built
a deterministic 3,071-image subset from the cleaned v6 selection manifest. It
selected the lowest SHA-256 ranks of seed plus immutable row identity: 192
images from each of eight fake-generator groups and 307 images from each of
five real-source groups. The input cleaned-manifest SHA-256 is
`8cdb27a7eb54e141eaa5118a0bdc7d4fbf1f7761be78c80ba5d3a787df343fce`;
the fixed robustness-manifest SHA-256 is
`d031309497971525825a2e62bed3508c09f094e0f8500ea44ed2cf4f8e3a6121`.
This subset is reserved for clean plus all 19 individual workshop conditions.
It was fixed before model selection so one large source cannot dominate the
robustness score and the subset cannot be changed to rescue a candidate.

## 2026-08-29 — checkpoint-before-evaluation reliability repair

The completed v5 PE-Core training state was lost when the Kaggle notebook
reconnected before evaluation because the self-contained runner wrote
`model.pt` only after both evaluation passes. This was an execution-design
failure, not a model-quality result. The runner now writes and announces the
trained checkpoint immediately after the final optimizer step, before any
selection inference. Report and prediction files remain post-evaluation
artifacts. Python compilation, the complete **24-test** suite and
`git diff --check` all passed after the repair.

The corrected v6 PE-Core cell was then launched in the live P100 notebook using
PyTorch 2.8.0+cu126. The runner subsequently verified all **32,305** expanded,
content-addressed images and every pinned packaged-manifest checksum, then
advanced to public-backbone loading. This establishes input integrity for the
relaunch, but is not yet a training or accuracy result. The local v6 DINOv2-L
control remained active in parallel. No v6 score existed when this entry was
written.

## 2026-08-29 — resumable individual-condition evaluator

The final robustness matrix requires 20 complete inference passes: clean plus
the 19 individual workshop transformations. To avoid repeating every completed
condition after a machine or notebook interruption, the evaluator now writes
an atomic progress file after each condition containing labels, continuous
scores and paths. The progress signature pins the checkpoint SHA-256, evaluation
manifest SHA-256, model/input configuration, seed and exact condition names;
an incompatible resume file is rejected. Each condition receives a distinct
deterministic random stream so resumed Gaussian-noise conditions reproduce the
same result as an uninterrupted run. The final JSON and progress-complete marker
are also atomic writes.

A focused regression test proves that saved condition predictions are reused
without invoking model inference. Python compilation, `git diff --check` and
the complete test suite succeeded with **25 passing tests**. This is evaluator
reliability evidence; no robustness score is claimed by this change.

## 2026-08-29 — fixed subgroup-collapse candidate comparator

`compare_selection_reports.py` now normalizes both local and Kaggle candidate
report schemas and exposes clean AUC, weakest fake generator, weakest real
source, weakest generator/source pair and content-holdout floors. Its declared
veto rejects any observed held-out subgroup below chance (AUC 0.5); eligible
candidates are ranked by the harmonic mean of clean AUC and their weakest
recorded held-out/content AUC. It deliberately returns no provisional leader
when every candidate is vetoed and warns that this comparison neither predicts
the hidden score nor replaces the 19-transform matrix.

The pre-v6 comparison completed and returned no leader: v4, v5 linear and v5
Stay-Positive all contain a below-chance subgroup. Two focused tests cover a
high-clean but collapsed candidate and the all-vetoed case. The complete suite
now reports **27 passing tests**.

## 2026-08-29 — v6 PE reconnect loss and clean restart

The first checkpoint-protected v6 PE-Core attempt reached step 60/297, but the
Kaggle editor later reconnected with no active execution control and the cell
was no longer running. Because the early checkpoint is intentionally written
only after the final optimizer step, no partial model was claimed or reused.
The unchanged cell was restarted only after the UI authoritatively showed zero
live executions. The restarted cell is active, uses the same pinned package,
seed and configuration, and has reverified all 32,305 cached content-addressed
images. It has not yet reached a logged training step, so this restart still has
no model-quality result.

## 2026-08-29 — error analysis reuses signed clean predictions

The error-analysis CLI can now consume the clean predictions in the resumable
robustness progress file instead of forwarding the selected model through the
same images again. It verifies both the evaluation-manifest path and SHA-256
before ranking highest-scoring real images, lowest-scoring AI images, and
threshold-0.5 false positives/negatives. The output retains the checkpoint
SHA-256 and preprocessing identity from the progress signature. A regression
test covers the signed reuse path; the complete suite reports **28 passing
tests**. No error examples are claimed until the selected checkpoint completes
the fixed robustness gate.

## 2026-08-29 — v6 DINOv2-L passes the clean source-collapse gate

The fixed local v6 DINOv2-L command completed successfully on Apple MPS. It
trained one source-balanced epoch on 18,958 rows and evaluated the unchanged,
perceptual-clean 13,349-row selection manifest. Runtime was **2,636.64
seconds**, final training loss **0.480728**, and the checkpoint contains
303,228,929 total / 1,025 trainable parameters. MPS reported 1,251,468,800
current and 3,280,666,624 driver-allocated bytes. Checkpoint SHA-256 is
`a26acbafa19a677f17ac053908e670dbf55e94b9a84354b9f1470132dc4cc7e7`;
report SHA-256 is
`41452190ecb5073ee1c40a0694dd2b25a69ec4d52d0232068bd81353b9559d23`.

Clean AUC was **0.920357**. The weakest complete fake holdout was Imagen at
**0.842141**; PixArt reached 0.970300. The weakest real source was SID_Set at
**0.842570**; byte-disjoint LAION and Church reached 0.864281 and 0.953451.
The weakest generator/source pair was Imagen versus SID_Set at **0.710490**.
This is the first broad candidate with no below-chance held-out subgroup. The
fixed comparator gives it hidden-set floor 0.710490 and clean/floor harmonic
mean 0.801920, while every v4/v5 control remains vetoed.

This clean result promotes v6 DINOv2-L only to provisional leader. Its
checkpoint immediately began the predeclared 3,071-image balanced clean plus
19-condition evaluation in resumable session 22823. PE-Core-L remains an
active independent comparison, so no final-model claim is made yet.

## 2026-08-29 — v6 PE checkpoint-loss evidence and second relaunch gate

The evaluation-only recovery cell revalidated all **32,305** packaged images
and their pinned manifests, then failed before model loading with the explicit
error that
`/kaggle/working/track5-v6-candidates/vit_pe_core_large_patch14_336/model.pt`
did not exist. A follow-up directory check printed `REPORT_EXISTS False` and
`FILES []`. The earlier completed PE training loss therefore has no surviving
checkpoint or clean score and is not a result.

The unchanged checksum-pinned PE cell was relaunched, reached 22,000/32,305
package-verification records, and then stopped when the notebook tab was
cleaned up; Kaggle subsequently showed no live execution control. It had not
reached training and created no checkpoint. A fresh Kaggle GPU session is now
starting. The runner-recreation plus training cell has been prepared but will
not be launched or claimed until Kaggle exposes a connected runtime. The local
DINO robustness evaluation is independent of this blocker and continues to
save one signed condition at a time.

## 2026-08-29 — provisional DINO error rates and source-aware threshold

The signed clean predictions from the fixed 3,071-image robustness manifest
were reused without another model forward pass. At the uncalibrated threshold
0.5, balanced accuracy was **0.833582**, with 361/1,535 real images false
positive (**23.5179%**) and 150/1,536 fake images false negative (**9.7656%**).
SID_Set, CIFAKE and LAION real false-positive rates were respectively 39.09%,
31.92% and 31.27%; the highest fake miss rates were Imagen at 31.77% and
CIFAKE Stable Diffusion at 24.48%. These results show why clean AUC alone is
not an operating-point claim.

`select_operating_threshold.py` evaluates every observed score against every
real-source specificity and fake-generator sensitivity, selecting the
threshold that maximizes the weakest known group recall and breaking ties by
global balanced accuracy. The command completed in **6.24 seconds** over 3,074
candidate thresholds. Its provisional threshold is **0.5260697603**: balanced
accuracy **0.840101**, false-positive rate **20.9772%**, false-negative rate
**11.0026%**, and weakest group recall **0.651466**, up from 0.609121 at 0.5.
The output explicitly states that this is internal permitted development data,
does not convert sigmoid scores into calibrated empirical probabilities, and
must never be fitted on the demo-only COCO/DALL-E resources. Two focused tests
raise the full suite to **33 passing tests**.

The recovered Kaggle P100 session reverified the full package and the
checksum-pinned PE-Core v6 experiment reached step 100/297. Logged mean loss
fell from 0.576519 at step 20 to 0.397884 at step 100. The execution control is
live, but neither training completion nor model quality is claimed yet.

The resumable DINO matrix has saved nine of twenty conditions. AUC values are
clean 0.921423; JPEG q90/q70/q50/q30 0.911860/0.906580/0.897729/0.889649;
blur sigma 0.5/1/2 0.920279/0.916872/0.903098; and 0.5x downscale-upscale
0.914271. These are per-condition results, not the final pooled transformed
AUC. The local evaluation process remains live.

The actual v6 DINO checkpoint also passed the organizer-style entry point on
CPU. A fresh temporary directory contained one known hard CIFAKE real and one
known hard CIFAKE fake image. `run.sh` loaded all 303,228,929 parameters and
emitted exactly two JSON rows containing only `image_path` and finite `pred`
values in [0, 1]. End-to-end tool wall time was **7.34 seconds**. The fake
scored 0.026062 and the real scored 0.946977, intentionally reproducing the
known false-negative and false-positive rather than claiming accuracy from a
contract smoke test. The temporary directory was removed after validation.

## 2026-08-29 — PE-Core cached-runner diagnosis and fresh-process result

The second apparent PE checkpoint loss was traced to notebook module caching,
not to the rewritten source file. The live Python kernel retained the old
`kaggle_train_v3` function objects, which still saved the checkpoint after
evaluation and still rejected same-label, same-score duplicate content. A
traceback displayed current source lines but executed the cached function
bytecode. The corrected command therefore launched a genuinely fresh process:

```text
subprocess.run([sys.executable, '/kaggle/working/kaggle_train_v6.py'], check=True)
```

That process verified all 32,305 package images, completed 297/297 PE-Core-L
linear-head steps, saved `model.pt` before evaluation, and completed both clean
selection passes on a Tesla P100. Runtime was **856.17 seconds** and peak CUDA
allocation was **2,311,815,680 bytes** under PyTorch 2.8.0+cu126. The model has
315,776,001 total / 1,025 trainable parameters. On the 13,349-row cleaned
selection gate, clean AUC was **0.981830**, weakest fake-generator AUC
**0.944051**, weakest real-source AUC **0.958568**, and weakest generator/real
pair AUC **0.872510**. On the content-focused subset, clean AUC was **0.994353**
and the weakest pair was **0.969759**. These are strong permitted-development
results, not hidden-set evidence; the full transform and codec-shortcut stress
run started immediately from the saved checkpoint.

## 2026-08-29 — local PE-Core device probe

The same public `vit_pe_core_large_patch14_336` backbone was probed locally
instead of assuming that Apple MPS could not run it. Batch size 1 succeeded
end to end with 316,103,681 total / 1,025 trainable parameters in **3.679
seconds** for the two-image smoke configuration; MPS reported 1,266,090,752
current and 2,234,187,776 driver-allocated bytes. Batch sizes 2, 4 and 8 each
failed in the current torch/MPS stack with an MPSNDArray buffer-size assertion
(`Must be 8192`, `16384` and `32768` bytes respectively). The two-image AUC of
zero is intentionally treated as meaningless. Local batch-1 is therefore a
slow emergency execution path, not the preferred training route and not model
quality evidence.

## 2026-08-29 — explicit container/shape shortcut alarm

`audit_manifest_shortcuts.py` inspected all 18,958 v6 training files and all
13,349 cleaned selection files. In training, real images are 7,666 JPEG plus
1,813 PNG and have aspect ratios from 0.222 to 2.946; fake images are 2,289
JPEG plus 7,190 PNG and every fake is square. In selection, every one of 6,457
real images is JPEG and real aspect ratios span 0.334 to 2.667, while 4,844 of
6,892 fakes are PNG and every fake is square. The model receives a square
short-side crop and robust training includes label-independent JPEG and resize
operations, so these raw-file differences do not prove causation. They remain
a serious shortcut hypothesis until a matched-codec diagnostic and new-source
gate succeed. All current clean scores are provisional under this alarm.

The official Community Forensics source was consequently investigated as a
breadth addition. Its pinned full release contains 2.7 million synthetic
images from 4,803 model variants, but a five-row official streaming probe took
over ten minutes, retained about 8 GB RAM, and returned five images from only
one generator. The process was stopped rather than misrepresenting that probe
as broad coverage. The pinned CommunityForensics-Small shard 0 is instead a
1,231,926,042-byte, SHA-256-locked source containing 2,993 fakes from 78 latent
diffusion model variants. A deterministic synthetic-only acquisition utility
now retains only non-NSFW rows whose prompt source is LAION; it rejects all
real rows and does not admit any organizer demo pixels. This source is not yet
part of a trained candidate or a reported score.

## 2026-08-29 — metadata-only shortcut probe

The shortcut alarm was tested with a deliberately non-visual logistic
regression probe. It received no image pixels and used only log width, log
height, log area, log file bytes, aspect ratio, square/not-square, container
format, colour mode and filename suffix. The exact command was:

```text
.venv/bin/python scripts/probe_manifest_metadata_shortcuts.py \
  --train datasets/family_mixture_v6/train.jsonl \
  --eval datasets/family_mixture_v6/eval_selection_perceptual_clean.jsonl \
  --output outputs/family-mixture-v6-metadata-shortcut-probe.json
```

It completed successfully over 18,958 training and 13,349 evaluation rows.
The pixel-free probe reached **0.838076 train AUC** and **0.950806 evaluation
AUC**. Restricted to square evaluation images it fell to **0.574106 AUC**;
restricted to PNG evaluation images it still reached **0.851422 AUC**. This
does not prove that either image model uses the leak, but it proves that the
current evaluation population makes shape/container shortcuts highly
predictive. Consequently, PE-Core's 0.981830 and DINOv2-L's 0.921423 clean
AUC remain untrusted for selection until both models pass label-independent
matched-codec and matched-geometry gates.

The saved DINOv2-L clean predictions were also stratified without changing the
model. Overall AUC is 0.921423 across 3,071 rows. Among the 2,583 square images
it is 0.943182, so a simple square/not-square rule cannot by itself explain the
model's score. Among the 1,727 JPEG images it falls to **0.828848**, and among
the 1,239 square JPEG images it is 0.869364. These subgroup populations still
mix sources and content, so the result is diagnostic rather than causal. The
large JPEG-only drop strengthens the requirement for a full matched-q96
re-encode gate.

## 2026-08-29 — completed DINOv2-L individual-transform matrix

The resumable local MPS command completed all clean plus 19 workshop-listed
conditions successfully in **5,535.58 seconds**. It used 1,212,915,968 current
and 3,272,278,016 driver-allocated MPS bytes at reporting time. Clean AUC was
**0.921423**, pooled transformed AUC was **0.903075**, and the 50/50
organizer-style score was **0.912249** over 3,071 clean and 58,349 transformed
predictions. The lowest individual AUC was Gaussian noise sigma 0.10 at
0.838005. The weakest pooled generator/real-source pair was **0.679782**.
These are completed internal-development measurements, but they remain under
the shortcut alarm; a separate identical JPEG-q96 evaluation was started from
the same checkpoint immediately afterward.

The DINOv2-L identical-codec gate completed in **109.76 seconds**. Re-encoding
every real and fake image through JPEG q96 before the unchanged short-side crop
produced **0.915078 AUC**, only 0.006345 below the 0.921423 unnormalized score.
Its weakest generator/real-source pair was 0.690876 versus 0.697679 on clean.
Thus container normalization does not collapse this model; the earlier
JPEG-only subgroup drop was not causal proof of codec reliance. Geometry is
still uncontrolled, so a JPEG-q96 full-frame stretch gate was started next.

That geometry gate completed in **108.39 seconds** at **0.912743 AUC**. It is
only 0.002335 below identical-JPEG short-side crop and 0.008680 below the
original clean score. The worst pair was Imagen versus SID_Set at 0.671095.
This rules out a large immediate collapse from replacing aspect-preserving
crop with full-frame stretch, but it does not solve weak modern-generator /
real-source transfer. More aggressive label-independent rectangular and square
patch gates remain required for PE-Core, the stronger candidate.

## 2026-08-29 — Community Forensics synthetic-only breadth acquisition

`scripts/acquire_community_forensics_shard.py` completed successfully against
the pinned `OwensLab/CommunityForensics-Small` revision
`6c539a534c07917307c381f5af4053c6091b5278`. It verified the
1,231,926,042-byte shard SHA-256
`0ce98c9b4f66eca160939982fe7aac84253af7d135485e5bc83ca8425cfe220c`,
then retained exactly four non-NSFW, LAION-prompt-source synthetic images from
each of 78 named model variants: **312 fake images total**. It rejected 54 NSFW
rows, 791 rows with another prompt source and 1,836 rows beyond the per-model
cap. The selected model inventory SHA-256 is
`f4a0f040c98122d6d576e0fc0c2812dc99dba8d256ec56fcecc3e3838da13dd8`.
No real Community Forensics rows were admitted. This remains an ignored,
external audit source; it has not trained or selected any reported candidate.
Its 78 models are latent-diffusion variants, not 78 independent architectures.

`scripts/prepare_community_forensics_gate.py` then built a no-copy external
manifest with 312 of those fakes and 312 untouched v6 evaluation reals. The
real side contains 63 CIFAKE, 63 FFHQ, 62 SID_Set, 62 LAION-5B and 62 LSUN
Church images. The manifest contains 624 unique content hashes, no forbidden
path term, and SHA-256
`4cba125a987bc656193d87ade84f0b805740c597910fd3164da90e2d0830f829`.
It is explicitly marked `training_allowed: false`. DINOv2-L evaluation on this
gate started only after the manifest was frozen.

The DINOv2-L gate evaluation completed in **30.51 seconds** at **0.878924
AUC**. Its fake sensitivity at threshold 0.5 was 0.846154. Pair AUCs against
CIFAKE, FFHQ, SID_Set, LAION-5B and LSUN Church were respectively 0.844780,
0.960419, 0.813431, 0.835401 and 0.939826. The gate was also packaged for the
Kaggle PE comparison as 624 unique images / 154,850,434 source bytes. The
155,242,056-byte ZIP SHA-256 is
`123a0e4bb8ae484a804a6a39a9f20063e62116e260fef813efe44824cc11a084`
and its content inventory SHA-256 is
`5b345de1d57badd4a9bbc6b33876a29f2660c9cd173cff54aa9a379038578943`.
A DINO identical-JPEG repeat was started before interpreting the transfer as
generation evidence.

That repeat completed in **31.23 seconds** at **0.874127 AUC**, only 0.004797
below the unnormalized gate. The weakest pair remained SID_Set at 0.809191.
This is evidence that the 78-model transfer is not primarily a PNG/JPEG
container effect. It is not evidence that all 78 four-image subgroups are
individually solved, nor does this latent-diffusion-heavy gate cover every
generator architecture.

The per-model diagnostic used all 312 reals as a common reference and only
four fakes per named model, so it is explicitly high variance. On clean, the
minimum/10th-percentile/median/90th-percentile model AUCs were
0.518429/0.753926/0.902644/0.971875. No model reversed below chance, two were
below 0.7, and none missed all four fakes at threshold 0.5. Identical JPEG gave
0.500000/0.753686/0.893830/0.969151 with the same two-below-0.7 count. The
weakest named model was `arpita22/my-pet-dog-rog`; visual inspection of its
four retained rows showed a car, politicians, a sofa and a ring, not dogs.
Therefore the model name is not used as evidence of an animal-content failure.

The same pixel-free metadata probe was fitted on v6 training metadata and
evaluated on the 624-row Community Forensics gate. It reached **1.000000 AUC**;
the gate is therefore perfectly separable from raw dimensions/container/file
size alone. Square-only AUC was 0.653846. This makes the raw 0.878924 image
score insufficient evidence by itself. The 0.874127 identical-JPEG result is
the first meaningful safeguard, and a geometry-normalized repeat remains
required.

The Community Forensics geometry-normalized DINO repeat used identical JPEG
q96 plus full-frame stretch. It completed in **75.77 seconds** at **0.870788
AUC**, only 0.003339 below q96 crop mode and 0.008136 below the raw gate. Its
weakest pair was SID_Set at 0.794872. Thus neither container normalization nor
this independent geometry rule causes a large collapse. The resumable full
19-transform gate was then continued from its saved clean prediction.

## 2026-08-29 — completed DINOv2-L Community Forensics transform matrix

The following audit-only command completed successfully on Apple MPS:

```bash
.venv/bin/python -m aigc_detector.evaluate \
  --manifest datasets/community_forensics_external_gate/manifest.jsonl \
  --checkpoint outputs/family-mix-v6-dinov2l-linear-source-balanced-robust-e1/model.pt \
  --output outputs/family-mix-v6-dinov2l-linear-source-balanced-robust-e1/eval-community-forensics-78-models-clean-plus-19.json \
  --batch-size 64 --workers 2 --robust \
  --preprocess-mode short_side_crop --device mps --seed 20260829
```

It evaluated 624 clean and 11,856 transformed predictions in **917.36
seconds**, reporting 1,212,915,968 current and 3,272,278,016 driver-allocated
MPS bytes. Clean AUC was **0.878924**, pooled transformed AUC was **0.862907**,
and the 50/50 organizer-style score was **0.870916**. The weakest individual
condition was Gaussian noise sigma 0.10 at **0.776699 AUC**; the strongest was
resize 0.5 at 0.895207. The weakest pooled real-source comparison was 0.798816.
Because this gate contains only four fakes per named model and is dominated by
latent-diffusion variants, these numbers are a transfer stress test rather
than a hidden-score estimate. The result confirms a real noise weakness while
also showing that DINOv2-L's 78-model transfer survives all 19 workshop-listed
transform types without falling to chance.

## 2026-08-29 — completed PE-Core v6 individual-transform matrix

The checksum-pinned Kaggle P100 command completed successfully with return
code 0 after resuming its per-condition checkpoints:

```python
subprocess.run(
    [sys.executable, "/kaggle/working/kaggle_stress_eval_v6.py"],
    check=True,
)
```

Across the fixed 3,071-row source-balanced v6 gate, PE-Core-L reached
**0.991079 clean AUC**, **0.978019 pooled transformed AUC**, and **0.984549**
on the 50/50 organizer-style score. The weakest individual condition was
Gaussian noise sigma 0.10 at **0.867729 AUC**. The weakest pooled fake
generator, real source and generator/real-source pair AUCs were respectively
0.911136, 0.932402 and **0.742043**. The run took **1,732.12 seconds**, peaked
at 2,894,289,920 allocated CUDA bytes, and reported PyTorch 2.8.0+cu126 on a
Tesla P100-PCIE-16GB.

Its label-independent identical-JPEG-q96 diagnostic scored **0.990578 AUC**,
only 0.000501 below raw clean. The weakest q96 generator/real-source pair was
0.850808. This is strong evidence against a simple PNG-versus-JPEG container
shortcut on this gate, but it does not yet control aggressive geometry or
prove transfer to the separate 78-model external audit. Both follow-up tests
remain hard promotion gates.

The separate PE-Core geometry command then completed successfully with return
code 0:

```python
subprocess.run(
    [sys.executable, "/kaggle/working/kaggle_shape_stress_v6.py"],
    check=True,
)
```

Every condition first applied identical JPEG q96. Full-frame stretch scored
**0.989831 AUC**, an identity-derived 75% square patch scored **0.989997**, and
an identity-derived forced aspect ratio selected from 4:3, 3:4, 16:9 and 9:16
scored **0.985943**. The three-condition run completed in 245.65 seconds and
peaked at 2,894,289,920 allocated CUDA bytes. None is more than 0.00514 below
raw clean. This substantially weakens the hypothesis that PE-Core's high score
comes from original container or square/non-square geometry. It does not prove
universal content or generator transfer, so the separately frozen 78-model
external audit remains mandatory.

## 2026-08-29 — completed PE-Core Community Forensics transform matrix

The private Kaggle upload exposed the ZIP members rather than the original ZIP
transport file. The first evaluator correctly stopped rather than claiming a
checksum it could not observe. The corrected script recorded
`zip_transport_verified: false`, then accepted the extracted package only
after verifying the pinned package inventory, manifest SHA-256, all **624**
individual image SHA-256 values, 312/312 class balance and all 78 named model
identities. Its own SHA-256 was
`e5c2eed1e5f4830a9ad177d24068e59b8e706d07cee7429dd151f90391b80d02`.

The corrected audit command completed successfully with return code 0:

```python
subprocess.run(
    [sys.executable, "/kaggle/working/kaggle_evaluate_community_forensics.py"],
    check=True,
)
```

PE-Core-L reached **0.992634 clean AUC**, **0.972330 pooled transformed AUC**
and **0.982482** on the 50/50 organizer-style score over the audit-only
78-model gate. Heavy Gaussian noise sigma 0.10 was again the weakest condition
at **0.797466 AUC**. The clean square-only subgroup retained 0.992551 AUC. All
78 four-image model alarms exceeded 0.90 AUC against the 312 common reals; the
minimum was 0.909054, but four samples per model remain too small for a precise
model-level estimate. The run took 358.03 seconds and peaked at 2,894,289,920
allocated CUDA bytes on the P100. These results are strong independent
transfer evidence and also reproduce the noise weakness. They still require
the queued external codec/geometry controls because the raw gate's labels are
perfectly separable from metadata alone.

The external PE codec/geometry command subsequently completed successfully
with return code 0. Every image hash and package checksum was reverified before
inference. Identical JPEG q96 plus full-frame stretch scored **0.988268 AUC**;
identical JPEG q96 plus an identity-derived 75% square patch scored **0.995007**;
and identical JPEG q96 plus an identity-derived forced aspect ratio scored
**0.993379**. Their weakest generator/real-source pair AUCs were respectively
0.972578, 0.990537 and 0.986197. The run took 52.43 seconds and peaked at
2,894,289,920 allocated CUDA bytes. The worst four-image named-model alarms
were 0.879808, 0.911859 and 0.919071; their sample size remains too small for
precise model-level estimates. Because none of the three controlled AUCs
collapses relative to 0.992634 raw clean, PE-Core v6 now passes the current
codec/shape shortcut veto on both the internal and external gates. It is the
leading candidate, subject to comparison with the controlled noise-weighted
v7 experiment and final offline packaging tests.

## 2026-08-29 — started controlled PE-Core v7 noise ablation

Both completed v6 matrices identify Gaussian noise as PE-Core's weakest
condition: sigma 0.10 scores 0.867729 internally and 0.797466 externally.
`scripts/kaggle_train_v7.py` therefore changes exactly one experiment axis:
conditional on applying one workshop-listed transform, 40% of draws are
Gaussian noise; the transform application probability is 0.90. JPEG, blur,
resize, colour and crop remain represented, and no draw chains two listed
transformations. Data, split, source-balanced sampler, frozen PE-Core-L
backbone, classifier-head optimizer and one-epoch budget remain the v6 values.

The uploaded `kaggle_train_v3.py` and `kaggle_train_v7.py` SHA-256 values are
respectively
`3942c95127652fa7b8b6c98745dd7078953090a24e72e2e6d93b8c2d9cca220a`
and `33f8fe9a4161f53954cd40fce2a175c3269da07b2eb39015ea62140e7d3fb2a4`.
The run verified those hashes and all 32,305 extracted package images, then
started training on the P100. It is not yet a completed or selected model. A
queued fixed-gate rule advances v7 to external audit only if sigma-0.10 AUC
exceeds v6 while clean AUC remains at least 0.986079 and the 50/50 internal
score remains at least 0.979549.

The original long-lived P100 draft stopped after clean-evaluation batch 75 of
105. Although its full checkpoint had been written, `/kaggle/working` is
ephemeral and the fresh session explicitly reported that the file no longer
existed. No clean score was produced, so this interrupted attempt remains
unscored. Kaggle's P100 draft then remained stuck at `Starting`; switching the
idle replacement session to the offered T4 x2 route succeeded. The interface
reported 20 of 30 weekly GPU hours remaining, PyTorch 2.10.0+cu128 detected two
Tesla T4 devices, and the unchanged training code uses device 0 only.

The recovery run reverified the v3, v6 and v7 source SHA-256 values and is
rehashing all 32,305 packaged images before training. It also adds a
persistence-only callback immediately after the full checkpoint write: the
1,025-parameter classifier head, full-checkpoint SHA-256 and training metadata
are written separately and emitted as base64 into durable notebook output
before the longer clean evaluation begins. This callback does not modify the
model, data, split, augmentation, optimizer or predictions; it only prevents a
second session interruption from destroying the trained head. The restarted
T4 run was unscored at this checkpoint; its completed result is reconciled
below.

The first T4 pass subsequently completed clean evaluation. At the time it was
provisionally rejected because the fresh image reported `timm 1.0.26` and
315,776,001 parameters, while a separate local smoke build had reported
316,103,681. Later inspection of the original P100 v6 report and checkpoint
corrected that inference: the actual v6 training run also has **315,776,001**
parameters and the same 257-token positional embedding. The T4 v7 result is
therefore architecture-compatible with the selected v6 checkpoint. It scored
0.978963 clean AUC, a 0.853132 weakest generator/real-source pair, and 0.993689
on the content holdout. Its 6,181-byte head was preserved with SHA-256
`33e3e0286e22b63ac8c1e8f758c3e7ace8228464b85e74858ea35aa99dadb20d`.

The follow-up rerun isolated `timm==1.0.19` under
`/kaggle/working/timm-1.0.19` while retaining the T4 session's PyTorch
2.10.0+cu128. Its subprocess printed the pinned timm, Torch, CUDA and Tesla T4
identity before revalidating the package. The experiment was retained as an
environment diagnostic; it did not change the architecture or predictions.

Direct tensor inspection showed that both T4 checkpoints were in fact
byte-identical despite the isolated subprocess printing `timm 1.0.19`: each
has SHA-256
`3ad24d97d0270b77f14b59be12c6054e6c6abc4949ce6d23b2e45eef7c835f16`,
315,776,001 state parameters and a `[1, 257, 1024]` positional embedding. The
subprocess import pin therefore had no effect. Both T4 reports are identical at
0.978963 clean AUC and 0.853132 worst generator/real-source pair. The original
P100 v6 checkpoint later proved to have the same state size and positional
embedding, so the earlier architecture-mismatch claim is withdrawn.

The 1,263,202,267-byte T4 checkpoint was downloaded and its SHA-256 reverified
locally. Because the local default `timm 1.0.19` constructs an incompatible
577-token positional embedding, an isolated `timm 1.0.26` package path was
used for the artifact smoke without changing the repository environment. A
four-image `run.sh` invocation completed on Apple MPS and emitted continuous
AI-positive predictions with the reported 315,776,001 parameters. The full
checkpoint was then exported to a 631,651,403-byte inference-only FP16 file
with SHA-256
`4f81310eb12aab62931fbe891caa3f50e9d3cee32b155fd86ca65ab164b9bcb7`.
A second Apple-MPS `run.sh` invocation completed; all four predictions were
finite and the maximum full-versus-FP16 absolute probability difference was
0.00003833 (mean 0.00001273). These are packaging and numerical-equivalence
results, not evidence that v7 should beat v6. V7 selection still depends on
the fixed clean-plus-19 promotion gate, whose completed result is recorded
below.

## 2026-08-29 — verified local FP16 PE-Core export and Apple-MPS inference

Before exporting the selected competition checkpoint, the inference-only
export path was tested end to end on the existing untrained PE-Core-L MPS
smoke checkpoint. This is a packaging equivalence test, not a detector-quality
result. The exact command was:

```bash
.venv/bin/python scripts/export_inference_checkpoint.py \
  outputs/smoke-pe-core-mps-batch1/model.pt \
  outputs/smoke-pe-core-mps-batch1/model-fp16.pt \
  --report outputs/smoke-pe-core-mps-batch1/export-fp16.json

SAMPLE_DIR=$(find datasets/community_forensics_small_shard0_fakes/images \
  -mindepth 1 -maxdepth 1 -type d | head -1)
PYTHON_EXECUTABLE=.venv/bin/python ./run.sh "$SAMPLE_DIR" \
  outputs/smoke-pe-core-mps-batch1/pred-full.json \
  outputs/smoke-pe-core-mps-batch1/model.pt
PYTHON_EXECUTABLE=.venv/bin/python ./run.sh "$SAMPLE_DIR" \
  outputs/smoke-pe-core-mps-batch1/pred-fp16.json \
  outputs/smoke-pe-core-mps-batch1/model-fp16.pt
```

The source artifact was 1,264,508,763 bytes with SHA-256
`cf88e81d43b6d1cb821109c82f0c018fdbc7fb87a88860af9abd832c5505d478`.
The inference-only FP16 artifact was 632,302,971 bytes with SHA-256
`e86886af1e4fee99381786d39d19558cf45460623ce53bf22d1024af91a95426`.
Both `run.sh` invocations completed on the 20-core Apple M5 Pro Metal device,
each emitted four continuous predictions, and the maximum absolute prediction
difference was **0.00001517** (mean **0.00000998**). All FP16 predictions were
finite and the parameter count remained 316,103,681. This verifies that the
offline PE-Core checkpoint can be halved in storage and executed locally at
batch size one without a material numerical change. The same equivalence gate
was still required on the selected trained checkpoint; the completed selected
artifact test is recorded below.

A separate subprocess then attempted one forward pass with batch size 2 on the
same Apple MPS device. It failed in 4.23 seconds with the Metal assertion
`Error: buffer is not large enough. Must be 8192 bytes`, after reaching a
5,725,260,368-byte peak memory footprint. Because batch size 1 had just
completed twice while batch size 2 failed, local PE-Core execution is retained
as a correct low-throughput inference fallback only. It is not treated as a
practical local training route; DINOv2-L remains the batched local control.

## 2026-08-30 — v7 rejected and exact v6 artifact selected

The fixed 3,071-image clean-plus-19 gate completed for v7 on one Tesla T4 in
**1,113.70 seconds**, peaking at **2,903,202,816 allocated CUDA bytes** under
PyTorch 2.10.0+cu128. V7 scored **0.989816 clean AUC**, **0.976795 pooled
transformed AUC** and **0.983306** on the 50/50 organizer-style score. Its
weakest condition was Gaussian noise sigma 0.10 at **0.863060 AUC**. These are
all below the corresponding v6 results: 0.991079 clean, 0.978019 pooled,
0.984549 official-style and 0.867729 heavy-noise AUC. The predeclared promotion
decision returned `advance: false`; v7 was not run on the external audit and is
rejected.

The original full v6 checkpoint was recovered locally at 1,263,202,267 bytes.
Its SHA-256 is
`4b8f3ac4776b0fddc689252de760d661916d9377374484703487538e8268766a`,
which exactly matches the checkpoint hash recorded by the successful v6 P100
stress run. Tensor inspection and the original training report both record
**315,776,001 total / 1,025 trainable parameters**, below the two-billion limit.
This exact hash match connects the preserved artifact to the completed
internal, external, codec and geometry evidence; the later T4 recovery retrain
is not substituted for it.

The selected artifact was exported with:

```bash
.venv/bin/python scripts/export_inference_checkpoint.py \
  "$HOME/Downloads/model.pt" models/model.pt \
  --report outputs/v6-selected-export.json \
  --expected-source-sha256 \
  4b8f3ac4776b0fddc689252de760d661916d9377374484703487538e8268766a
```

The inference-only FP16 file is 631,645,967 bytes with SHA-256
`48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644`.
Both the full and FP16 checkpoints completed the same four-image `run.sh` smoke
on Apple MPS. Every prediction was finite and AI-positive; the maximum
full-versus-FP16 absolute probability delta was **0.00038344** (mean
**0.00017435**). The FP16 checkpoint also completed the four-image run on CPU.
The complete repository test suite then reported **35 passed** after adding an
explicit source-checkpoint hash-mismatch rejection test.

For concrete error analysis, the exact FP16 artifact evaluated a deterministic
128-image sample (64 real, 64 fake) from the frozen v6 robustness manifest on
Apple MPS. The command completed in **8.18 seconds**, reported 1,263,235,328
current and 2,209,021,952 driver-allocated MPS bytes, and scored **0.995605
clean AUC**. The weakest observed generator/real-source pair was CIFAKE Stable
Diffusion versus CIFAKE real at **0.904762 AUC**. At the illustrative 0.5
threshold it made five false positives and no false negatives: three of the
five were 32-pixel CIFAKE/CIFAR-10 reals, one was a LAION web image and one an
FFHQ portrait. This small deterministic audit is error-analysis evidence only;
the 0.5 threshold is not claimed to be calibrated to the hidden distribution.

## 2026-08-30 — T4 recovery retrain retained only as reproducibility evidence

A fresh subprocess installed PyTorch 2.8.0+cu126, torchvision 0.23.0 and
`timm==1.0.19`, then repeated the v6 source, package, split, seed and 297-step
training command on a Tesla T4. It completed with final loss **0.2887365**, very
close to the original P100 loss 0.2887063, and clean selection AUC **0.981825**
versus the original 0.981830. Its weakest generator/real-source pair was
0.872491. However, this fresh combination instantiated a 316,103,681-parameter,
577-token build and produced a different checkpoint hash
`3213bafb13728874c709960d41d5628cb06641801db977f2859537b7b9d2d1a4`.
It is therefore recorded as close reproducibility evidence, not the selected
artifact and not a replacement for the hash-matched original v6 checkpoint.

## 2026-08-30 — deadline adversarial controls and independent NTIRE audit

The workshop transcript was re-read before adding any new test. It confirms
that the hidden evaluator uses continuous AIGC-positive scores, clean AUC plus
pooled singly transformed AUC, older Stable Diffusion material and recent
diffusion-transformer generators. The transformations are individual rather
than chained. It also explicitly warns that 99% clean performance may collapse
after aggressive processing and that JPEG/PNG source bias is a known shortcut.
Those statements, not speculative compound attacks, define this phase.

The exact selected v6 classifier head was extracted from the full checkpoint
into an ignored 6,469-byte artifact with SHA-256
`152e8555754613c230a42e150809e1879d707624d8709b067bb0bc8b4d11a56d`.
Reconstructing the public PE-Core-L backbone under pinned `timm==1.0.19`
matched every non-head tensor exactly except `pos_embed`, whose maximum absolute
delta was `5.960464477539063e-08`. This is numerical reconstruction evidence;
it is not a byte-identical reconstruction of the selected full checkpoint.

A predeclared 128-image inference-view screen compared reference, horizontal
flip mean, square stretch mean and a mild-blur blend across clean plus eight
workshop-aligned stress conditions. Only reference-plus-flip passed the small
screen: organizer-style screen AUC increased from `0.9860992432` to
`0.9871978760`, while the largest observed condition decrease was
`0.00170898` on resize. Stretch and blur-blend policies were rejected. The
runner remains unchanged while a paired full internal and external gate runs
on a Tesla T4. The Kaggle subprocess verified PyTorch `2.8.0+cu126`, CUDA 12.6,
`timm 1.0.19` and Tesla T4 before evaluating. No promotion decision has yet
been made.

The existing aggregate metadata-only audit produced 0.9508 AUC, proving that
the original mixed corpus contains easy source metadata. An exact CIFAKE
JPEG/RGB/32-by-32 matched gate reduced that confound; selected v6 still scored
0.8559299 reference AUC and 0.8575575 with flip mean. This weakens one simple
metadata explanation but does not eliminate semantic or dataset-source
shortcuts.

The official NTIRE 2026 training record was then added as an independent
source. The exact acquisition command was:

```bash
.venv/bin/python scripts/acquire_ntire2026_shard.py --shard 5
```

The command completed. The 11,370,161,676-byte archive matched SHA-256
`6d6628c983c43f1de44589151e2b3b9d33726691efbd9b0208e9f015ded9af8f`;
ZIP traversal passed with 27,646 members. A deterministic audit-only sample was
created with:

```bash
.venv/bin/python scripts/prepare_ntire2026_audit_sample.py \
  --archive datasets/ntire2026/shard_5.zip \
  --output-root datasets/ntire2026/shard_5_audit_sample \
  --per-class 256 --seed 20260830
```

The 512-image sample manifest has SHA-256
`6a8741c06b586e499c4db96c2dddf0a8e3571516f4951f859244b882fdce78c7`.
All files are JPEG/RGB, but geometry distributions still differed by label. An
exact format/mode/width/height match retained 43 real and 43 fake images across
31 strata. This tiny set is not used for training, threshold tuning or model
selection.

On all 512 audit images, selected v6 scored 0.9834137 clean AUC with the
reference view and 0.9835815 with flip mean. At the illustrative 0.5 threshold,
reference true-positive rate was 0.9726563 while true-negative rate was only
0.8085938, reinforcing that the hidden threshold must not be inferred from our
sources. On the exact 86-image metadata match, reference clean AUC was
0.9810708. The complete individual-transform command was:

```bash
PYTHONPATH=src:. .venv/bin/python -m aigc_detector.evaluate \
  --manifest datasets/ntire2026/shard_5_audit_sample/metadata_matched.jsonl \
  --checkpoint models/model.pt \
  --output outputs/deadline-adversarial/ntire-shard5-metadata-matched-robust-reference.json \
  --batch-size 1 --workers 0 --inference-policy reference --robust \
  --device mps --seed 20260830
```

It completed in 100.59 seconds on Apple MPS, with 1,263,235,328 current and
2,209,021,952 driver-allocated bytes. Clean AUC was 0.9810708, pooled robust
AUC 0.9580907 and the 50/50 score 0.9695808. Every non-noise condition scored
at least 0.9583559. Gaussian noise remained the clear failure: 0.9226609 at
sigma 0.02, 0.8134127 at sigma 0.05 and **0.7166036 at sigma 0.10**. This
independent result falsifies any claim that the current detector has solved
heavy noise, despite its high aggregate score.

The initial test invocation `.venv/bin/pytest -q` failed during collection
because the repository root was not on the import path. The corrected command
`PYTHONPATH=src:. .venv/bin/pytest -q` completed with **36 passed**. This was an
invocation error, not a code-test failure.

The paired flip evaluation on the same 86 images completed in 192.23 seconds
on Apple MPS. It held clean AUC at 0.9810708, improved pooled robust AUC from
0.9580907 to 0.9598660 and improved heavy-noise sigma 0.10 from 0.7166036 to
0.7290427. However, medium-noise sigma 0.05 regressed from 0.8134127 to
0.7961060, a **0.0173067 decrease** that breaches the predeclared 0.01
per-condition limit. The independent gate therefore rejects flip promotion,
despite its 0.0008877 aggregate-score gain. The full Kaggle run remains useful
as a larger paired measurement, but it cannot override this disclosed failure
without a new, separately justified policy and independent evidence.

## 2026-08-31 — prompt-matched 2026 frontier-family falsification and v8 design

The selected v6 checkpoint was tested against the pinned Apache-2.0 Qwen Image
Bench revision `d2493deb153b020cf169c7e3f57d15e4dd697038`. This is not the
organizer's prohibited DALL-E Advanced demonstration resource. The source
contains the same 1,000 prompts rendered by 18 current generators, which makes
it useful for separating generator weakness from prompt selection.

A diagnosis-only partition selected the same 16 prompt IDs for every generator.
The command completed after downloading and decode-verifying 288 fake images
(307,295,348 bytes). The source inventory SHA-256 is
`cc553cdf6d3be2ceb2143b7ef9dbb326cc9dc7530d55479d43ad35e981c7e5e0` and
the enriched manifest SHA-256 is
`9c8e8f9e95f426761df05b3dd9e2f8c6c99ada392de96f0db0803f737b0c0d98`.
It was combined with 288 frozen external-gate real images across CIFAKE, FFHQ,
SID_Set, LAION and LSUN Church. The combined gate had no training-pixel
overlap, used identical square stretching and JPEG quality 96 for both labels,
and has manifest SHA-256
`0bbfe48776b1274e6198be2cc96273a0be8797fb98f63110360a73399b9fe5f7`.

The exact v6 artifact completed this 576-image gate on Apple MPS in 29.22
seconds. Overall AUC was **0.9537760**, materially below the flattering internal
0.99 clean score. The weakest generator was Nano Banana 2 at **0.8951823 AUC**
against all reals; the weakest real source was CIFAKE at **0.8706897 AUC**
against all fakes; and the weakest pair was Nano Banana 2 versus CIFAKE at
**0.6993534 AUC**. This is direct evidence that modern generator breadth and
real-source composition remain unresolved. It is not evidence that the model
will fail on the hidden set, and it is not a promotion gate because this result
was used to decide what data to add.

A disjoint candidate-training partition then selected another 32 prompt IDs per
generator: 576 images and 612,250,940 bytes. It excludes every diagnosis prompt
and pixel. The acquisition manifest SHA-256 is
`d22a5af898dad311e69798d0ca5dca3f7ef511e7ed9c9284f4e27720bf0b7020`.
The transport package is 612,717,221 bytes with SHA-256
`650f309ddb4fd8d0b7ac05d101981460319b9de2f39fd7795b1413982145fa93`;
its content inventory SHA-256 is
`f7f2036aec27d64d855253fded6de4b3996838f085b18c84912c0d808b5725a0`.

The v8 experiment changes one factor from v6: it adds these 18 current named
generator groups to the existing 18,958-row training mixture while keeping the
same PE-Core-L backbone, frozen-head-only training, seed, optimizer,
source-balanced sampling, workshop-aligned augmentation and frozen v6 gates.
The combined 19,534-row train manifest has SHA-256
`63ad4137c27bdc3f1a93690915836330ae660bc2dd6b2ddb3347e8c1703b6332` and
zero SHA-256 overlap with either frozen gate. A third prompt-disjoint partition
completed with 288 images and 311,365,922 bytes. Its source inventory SHA-256
is `c6e3d40e202aa5743707a1d8c581698745c690a33d70d0c60ee04a7bf8103b61`
and manifest SHA-256 is
`5663056fe2b55f5a67a1395432e9fc417d58a6d10d368774e019b43e5a5ffb59`.
It has zero prompt and pixel-hash overlap with both earlier partitions and
remains unscored until the v8 candidate run is frozen.
After training, that set may compare v6 and v8 once; because its generators are
the same named families, it is prompt- and pixel-held-out but not
generator-unseen. The v6 checkpoint remains the protected fallback unless v8
passes the unchanged internal, external and independent condition guards.

The larger paired Kaggle flip measurement did not complete. It finished every
planned internal condition and ten external conditions through saturation 1.2,
then Kaggle restarted the idle interactive session before the remaining
external conditions and summary could be written. The partial external
evidence already included regressions at JPEG 50
(`0.9802915` reference versus `0.9793002` flip), noise sigma 0.05
(`0.8132653` versus `0.8127568`) and noise sigma 0.10 (`0.7973116` versus
`0.7938137`). No complete artifact survived the restart. This run is recorded
as interrupted partial evidence, never as a reproduced full gate; rerunning it
was deprioritized because the independent NTIRE guard had already rejected the
same policy.

## 2026-08-30 — v8 frontier breadth completed and rejected

The v8 single-factor run completed on a Kaggle Tesla T4 under PyTorch
`2.8.0+cu126`, CUDA 12.6 and `timm==1.0.19`. Before training it checksum-verified
all 32,305 existing package images, all 576 frontier images, the 18-by-32
frontier generator composition, and zero SHA-256 overlap with either frozen v6
gate. The exact wrapper script SHA-256 was
`c88a3e93bb92832088958b6009cf0dad7c087a9d89c1a2ae0498b2db66cbe12a`.
The command `/usr/bin/python3 /kaggle/working/kaggle_train_v8_frontier.py`
returned code 0 after 306 optimizer steps in 576.66 seconds. Peak allocated CUDA
memory was 2,311,815,680 bytes.

The resulting 315,776,001-total/1,025-trainable-parameter checkpoint has
SHA-256 `0136eab1532e8d7d4fd3471878e6928f423940e273c30b2be23b4c774d9f5c93`.
It scored **0.972346 clean AUC** on the unchanged internal selection gate,
with 0.911782 worst fake-generator AUC, 0.949879 worst real-source AUC and
0.839454 worst generator/real-source-pair AUC. The frozen content holdout was
0.980777 clean AUC with a 0.886614 worst pair. The like-for-like v6 training
report on this full clean selection view recorded 0.981830 clean AUC and a
0.872510 worst pair. V8 therefore decreased clean AUC by 0.009484 and worst-pair
AUC by 0.033056, breaching both predeclared guards. It is **rejected** without
consulting the sealed frontier holdout. The separately recorded 0.991079 v6
number comes from the smaller source-balanced robustness subset and is not used
for this comparison. No strong result on a familiar or pooled subset may
override the like-for-like regression.

The failure exposed a sampler error in experimental design rather than a data
licence or execution problem. Equal weight per named fake generator meant the
18 new Qwen generator names collectively received 60% of fake-label draws,
about 30% of all draws, although their 576 unique images were only 5.7% of the
fake training rows. This repeatedly oversampled a small, uniformly sourced
2026 set and displaced the much broader legacy evidence. The completed v8
directory was archived inside Kaggle as a 1,176,863,213-byte ZIP with SHA-256
`d7a226c67a341a0baac52cf36080aaf99bb86954644993decc93bcc0a334ff95`.
The browser file link did not produce a verifiable local download, so that hash
is recorded as Kaggle-side evidence only; the protected local v6 files remain
unchanged.

V9 is predeclared as a sampling-only ablation. It uses exactly the v8 images,
model, seed, optimizer, one-transform augmentation and frozen internal gates,
but reserves 50% of draws for real images, 42.5% for legacy fakes and 7.5% for
frontier fakes. Equivalently, frontier images receive 15% of the fake-label
budget rather than 60%. The wrapper SHA-256 is
`51697ba026ba0d0aa31ba69229350002b472fbae594925da3ed8d99c4c326ed8`.
The evaluator and sampler test compile, and `.venv/bin/python -m pytest -q`
completed with **37 passed**. V9 is not promoted merely for improving over v8:
it must first remain within 0.002 of v6 clean AUC and 0.01 of v6 worst-pair AUC,
then improve a once-opened prompt/pixel-held-out frontier gate without a
material full-NTIRE or individual-transform regression.

## 2026-08-30 — v9 sampling cap completed and rejected before sealed gates

V9 was launched only after its numerical decision floors were committed. The
Kaggle subprocess checksum-verified the 5,271-byte wrapper at SHA-256
`51697ba026ba0d0aa31ba69229350002b472fbae594925da3ed8d99c4c326ed8`,
all 32,305 v6 package images and all 576 frontier images. It completed 306/306
steps with final loss 0.2858571, saved the checkpoint before evaluation, and
returned code 0. The checkpoint SHA-256 is
`dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075`.
The complete train-plus-clean-evaluation process took 501.25 seconds and peaked
at 2,311,815,680 allocated CUDA bytes on a Tesla T4 under PyTorch
`2.8.0+cu126`, CUDA 12.6.

The sampling-only change materially recovered v8 but did not clear the
precommitted v6 guards:

| Frozen view | V6 reference | V9 | Required floor | Decision |
|---|---:|---:|---:|---|
| Full selection clean AUC | 0.981830 | 0.978066 | 0.979830 | fail |
| Full selection worst pair | 0.872510 | 0.861093 | 0.862510 | fail |
| Content clean AUC | 0.994353 | 0.988601 | 0.992353 | fail |
| Content worst pair | 0.969759 | 0.932552 | 0.959759 | fail |

V9's full-selection weakest fake-generator and real-source AUCs were 0.929905
and 0.956570. Its content-view weakest fake-generator and real-source AUCs were
0.964332 and 0.978486. Because all four guard rows failed, v9 is **rejected**
and the prompt/pixel-sealed Qwen holdout remains unopened. The same is true for
the sealed full promotion matrix: a failed internal candidate is not allowed to
consume independent evidence merely to search for a flattering result.

One already-open diagnosis remains legitimate: compare exact selected v6 and
v9 once on the earlier 576-image frontier audit that motivated the experiment.
That can establish whether the frontier data added any transferable signal at
all, but cannot promote v9. A checksum-addressed diagnosis package was produced
with 576 images, 330,663,624 source bytes, inventory SHA-256
`ae99a1e98c01f0a91a26aa7bc9728bbd7c8cbba0c708b55647e8e5375ecc8a48`
and ZIP SHA-256
`fad723b52a849550e01e4ff3cc7b08bd4aeec1bb35ec515ce9cec0c64b508cc4`.
The exact selected v6 FP16 checkpoint is separately attached to the private
GPU notebook at its known SHA-256; the sealed holdout is not part of this
diagnosis command.

## 2026-08-30 — protected frontier ensemble screen selects one bounded blend

The exact selected FP16 v6 checkpoint was re-evaluated on the complete
13,349-row selection view before combining scores. This avoided comparing the
smaller 3,071-image robustness subset with the full v9 selection report. The
v6 reference was 0.981814 clean AUC and 0.872413 worst named pair; its frozen
content view was 0.994350 clean and 0.969728 worst pair. The checksum-verified
screen script had SHA-256
`af9227b9f19634534816dc7f1253244a4fbefe865c780b0b2e786a4b9ca4035f`.
It completed with return code zero in 300.44 seconds on a Tesla T4 and reached
2,894,289,920 peak allocated CUDA bytes. The sealed holdout remained unopened.

The predeclared 90/10 v6/v9 probability blend passed all four floors: selection
clean 0.981701, selection worst pair 0.872127, content clean 0.994031 and
content worst pair 0.967536. Its already-open frontier diagnosis was 0.959901
clean and 0.711207 worst pair. The 75/25 blend also passed, with 0.981434,
0.871323, 0.993450 and 0.963631 on the four floor rows; its frontier diagnosis
was stronger at 0.966954 clean and 0.745151 worst pair. The 50/50 blend failed
the content floors at 0.992216 clean and 0.955548 worst pair, despite reaching
0.977063 on the frontier diagnosis. It is rejected.

The selection rule therefore freezes exactly the 75/25 blend for one remaining
promotion matrix. No weight search, calibration, threshold or per-image router
is permitted. Before opening that matrix, the numerical pass/fail rules were
written to `DEADLINE_ADVERSARIAL_PLAN.md`, and a paired evaluator was committed
that sends identical transformed tensors through both encoders. Local syntax
validation and the complete test suite succeeded with **41 passed**. Passing
the screen is not promotion and does not establish hidden-set generalization.

## 2026-08-30 — 75/25 blend passes statistics; feasibility still pending

The paired evaluator checksum-verified 32,305 internal package images, 624
external images, 1,088 promotion images and both checkpoint hashes before
scoring exact v6 and the preselected 75/25 v6/v9 probability blend on identical
transformed tensors. All 20 individual conditions completed on each of the
576-image sealed Qwen holdout, 512-image full NTIRE audit, 3,071-image internal
gate and 624-image Community Forensics gate: 80 paired condition files total.

The first final-summary command returned nonzero after every condition file was
saved. The cause was a reporting-only `KeyError`: the assessment asked for a
nonexistent generic `worst_pair_auc` key even though the summary and committed
rules use separate `clean_worst_pair_auc` and `pooled_worst_pair_auc` checks.
The fix removed only the duplicate nonexistent lookup and added a regression
test. It changed no model, input, transform, weight, prediction or numerical
floor. The corrected evaluator SHA-256 is
`10c258e251b3bb2f866a722700772502982cd17ba041a9e6d8ca0bfaf0d564c4`;
the corrected command returned code 0 and all four assessments passed.

The frontier gate improved materially: clean AUC 0.933274 to 0.957158, pooled
robust AUC 0.890295 to 0.920319, official-style score 0.911785 to 0.938739,
clean worst pair 0.643319 to 0.727371, pooled worst pair 0.556773 to 0.614070,
and heavy-noise AUC 0.716303 to 0.755160. On the other three gates, official
score changed by -0.000131 (NTIRE), -0.000236 (internal) and +0.000512
(Community Forensics), all within the fixed -0.002 floor. Heavy noise improved
on all three. Internal clean and pooled worst-pair deltas were -0.003681 and
-0.003156, inside the fixed -0.01 guards.

This is a statistical promotion pass, not yet a runnable-artifact promotion.
The final zero-return run resumed saved predictions, so its 4.51-second elapsed
time and 1,263,968,768-byte per-GPU peak are summary-only values and are not
claimed as full inference resource use. The earlier full inference process
ended at the reporting bug and did not persist trustworthy whole-run timing or
peak memory. Batch-size numerical consistency, exact v9 checkpoint packaging,
two-checkpoint inference, latency and actual peak memory remain required. Exact
metrics and this limitation are frozen in
`FRONTIER_ENSEMBLE_PROMOTION_RESULT.json`. The local v6 checkpoint and `run.sh`
remain unchanged.

## 2026-08-30 — batch-size numerical stability audit fails

The post-promotion audit script had SHA-256
`0317965adaeafb373c70ff161649f8291fc427526eed9dd8e7264fc79d135c82`.
It reverified the 1,088-image promotion package and both checkpoints, then
scored the already-open 576-image clean Qwen holdout at batches 64 and 128.
The command returned code 0 and the transformed CPU tensor stream had identical
SHA-256 `beac1b5cf272ced96d3c0c607721bf211939e036d8db98bae5108f0090caad77`
at both sizes.

The predeclared stability gate nevertheless failed. Batch 64 versus 128 changed
scores by at most 0.000488 and AUC by 0.0000121; v6 moved by up to two rank
positions and the blend by one. Batch 128 versus the saved clean promotion
predictions reached 0.000732 maximum blend-score drift and five rank positions.
These changes are small compared with the statistical promotion floors, but
they are larger than the independent consistency limits and cannot be waived
after inspection.

The audit also measured paired forward throughput, including both GPUs but not
model loading: 71.35 images/s at batch 64 with 2,380,880,896 peak allocated
bytes per GPU, and 81.60 images/s at batch 128 with 2,893,503,488 bytes per GPU.
Input decoding and tensor hashing made total wall times 20.27 and 19.46 seconds,
so those wall figures are audit overhead rather than deployment latency.
`FRONTIER_ENSEMBLE_BATCH_STABILITY_RESULT.json` preserves the exact report.
The candidate remains statistically promoted but operationally unselected.
The next frozen test uses a physical batch of 64 with deterministic padding;
v6 and `run.sh` remain unchanged.

## 2026-08-30 — fixed physical batch isolates an arithmetic-contract mismatch

The fixed-batch verifier had SHA-256
`835d760451b9843243568419e50a799b7204f0a342499b2558732100dfe23818`.
It reused only 64 balanced rows from the already-open clean Qwen gate, whose
selection digest was
`8e05c80d9e69e8bf289008bdd7f8e5a46897eafd733a7a2bbd35f38a8f194545`.
Both checkpoint hashes and the promotion-package inventory were reverified.
The command returned code 0 on two Tesla T4 GPUs under PyTorch `2.8.0+cu126`
and CUDA 12.6, but the frozen gate returned `passes: false`.

Keeping the physical CUDA batch at 64 with deterministic last-image padding
worked as intended: logical request batches 1, 17 and 64 produced identical
input-tensor SHA-256
`b1ef68c7ddba3bc0be9aabb5e3033ccceb91ec6cc1faac7a351376e4f24f686d`
and zero score, AUC or rank drift for both v6 and the blend. The one-image
logical path took 52.17 paired-forward seconds for 64 images because it ran 64
full physical batches; logical 17 took 3.34 seconds and logical 64 took 0.823
seconds. Peak allocation was about 2.38 GB per GPU. These are subset audit
measurements, not full submission-runtime claims.

The native-64 v6 scores exactly reproduced the saved promotion subset. The
blend did not: maximum absolute drift was 0.000366 with mean drift 0.0000877,
although AUC and ranks were identical. The predeclared maximum score drift was
0.000001, so it was not relaxed. Inspection of code history identified the
causal difference: the first promotion matrix formed the weighted probability
sum in FP16 on one GPU at physical batch 64; later paired-GPU feasibility code
converted each model score to FP32 on CPU before blending. The promotion
resume signature encoded checkpoints, weights, transforms and seed but not
batch size or arithmetic location/dtype, allowing newer reporting code to
reuse older predictions. The statistical metrics remain observations from the
older exact policy, but the newer paired-GPU implementation did not reproduce
them and must not be represented as doing so.

`FRONTIER_ENSEMBLE_FIXED_BATCH_RESULT.json` preserves the exact failed report.
The next predeclared audit changes no data, model, weight or score: it recreates
the original physical-64 FP16 blend on the two GPUs and requires zero drift
against the saved predictions and across logical batches 1, 17 and 64. Its
script SHA-256 is
`e5e125f12e5272f0bea6f40136c078a66e98be75e13fa423857070e59c9b7c6d`.
At that decision point, v6 remained selected and the ensemble remained a
research candidate pending the exact-contract command below.

## 2026-08-30 — exact FP16 arithmetic contract reproduces promotion scores

The checksum-verified exact-contract script had SHA-256
`e5e125f12e5272f0bea6f40136c078a66e98be75e13fa423857070e59c9b7c6d`
and returned code 0. It reused the same 64-row already-open Qwen subset, both
frozen checkpoint hashes, physical batch 64 and the original arithmetic:
sigmoid outputs and the 75/25 probability blend remained FP16, with the blend
formed on `cuda:0` before conversion to FP32 for storage. No data, model,
weight, threshold or transformation changed.

The test passed its exact zero-drift gate. Logical batches 1, 17 and 64 had the
same transformed-tensor SHA-256 and produced zero maximum or mean score drift,
zero AUC drift and zero rank displacement for both v6 and the blend. The native
physical-64 output also reproduced every saved v6 and blend score exactly.
The subset AUCs were 0.7705078125 for v6 and 0.826171875 for the blend; these
small-subset values are reproducibility controls, not new performance gates.
Model loading took 16.60 seconds. One physical batch forwarded 64 images
through both encoders in 0.813 seconds, with approximately 2.38 GB peak
allocation per GPU. Logical batch 1 is deliberately inefficient because each
image is padded to a full physical batch: 51.98 forward seconds for 64 images.

`FRONTIER_ENSEMBLE_FP16_CONTRACT_RESULT.json` preserves the exact report. The
promotion evaluator's resume signature now includes physical batch size,
model-device placement, autocast dtype, score-conversion policy and blend
location/dtype, preventing incompatible future code from reusing prediction
files. Numerical reproducibility is now established only for this exact
two-T4 FP16 contract. This does not establish hidden-set generalization or a
portable submission runner: the 1,263,202,267-byte v9 training checkpoint is
not yet packaged, and single-machine latency/memory plus fallback behavior
remain unresolved. The selected runnable artifact therefore remains v6.

## 2026-08-30 — full single-T4 inference contract passes exactly

The single-GPU verifier had SHA-256
`c40cf7dae1c15c16776665d8fd4a3a8f0a1787e71601cff3a638a0d52faed25a`.
It reverified the promotion package and both checkpoints, loaded both encoders
onto `cuda:0`, and evaluated all 576 already-open clean Qwen images in nine
physical batches of 64. The command returned code 0 on a Tesla T4 with PyTorch
`2.8.0+cu126` and CUDA 12.6.

Every saved v6 and 75/25 blend score reproduced exactly: zero maximum and mean
score drift, zero AUC drift and zero rank displacement. The input tensor stream
had SHA-256
`beac1b5cf272ced96d3c0c607721bf211939e036d8db98bae5108f0090caad77`.
The reproduced clean AUCs were 0.9332742573 for v6 and 0.9571578414 for the
blend. This is the same promotion evidence, not a newly tuned result.

Both checkpoints loaded in 16.14 seconds. The nine paired sequential forwards
took 13.80 seconds, or 41.72 images per forward second; decoding and input
hashing increased wall time to 26.02 seconds. Peak allocated CUDA memory was
4,275,770,368 bytes. This is direct evidence that the exact promoted arithmetic
does not require two GPUs and fits a 15 GB T4. It is not a CPU, Apple MPS or
unknown-judge-hardware guarantee. The raw result is preserved in
`FRONTIER_ENSEMBLE_SINGLE_GPU_RESULT.json`.

The remaining deployment blockers are checkpoint transport and runner
integration. The exact v6 and v9 files total 1,894,848,234 bytes, and the v9
checkpoint remains Kaggle-side only. No large weight is committed to Git and
no cloud model host is created without explicit publication authority. A
candidate ensemble runner may now be implemented and verified on Kaggle, but
`run.sh` and the selected local v6 checkpoint remain unchanged until that
package passes its own end-to-end gates.

## 2026-08-31 — candidate ensemble command passes end to end

The candidate directory-to-JSON implementation validates the exact v6 and v9
checkpoint SHA-256 values, rejects incompatible configurations and non-CUDA
devices, enforces the below-2B parameter limit, holds physical CUDA batches at
64 with deterministic final-batch padding, and forms the frozen 75/25
probability blend in CUDA FP16 before converting results to FP32. It is exposed
separately as `run_ensemble.sh`; the selected `run.sh` fallback was not changed.

The Kaggle notebook's interactive Python environment first failed while
importing Pillow because its installed packages were internally inconsistent.
No inference claim is attached to that attempt. The already-verified isolated
`/usr/bin/python3` environment was therefore used without mutating the notebook
environment. Its first runner invocation reached the code but failed before
inference because both checkpoints store null codec-normalization metadata and
the candidate passed null through instead of the effective `none` default.
Only that packaging default was corrected; no image, model, checkpoint,
transform, weight or output arithmetic changed.

The corrected module had SHA-256
`a1341afc3c62afa07f6d887c982394605c1fae5da427d486364863fab1b0d33b`.
Its checksum-addressed source archive had SHA-256
`9dec87af70455a797c76181a8faf0846baf932f0c2dcd4ab6cdb3bb57faca301`.
The full isolated command returned code 0 over all 576 preserved Qwen audit
images in 47.51 seconds, including hashing both 1.895 GB checkpoints, loading,
decoding and inference. It emitted 576 finite unit-interval probabilities,
reported 631,552,002 parameters and had zero maximum or mean score drift from
the saved promotion output. The output SHA-256 was
`ec170d9619b276240aef84db03812aa05623656314eac042723a7f2587a13fe4`.
Exact evidence is in `FRONTIER_ENSEMBLE_E2E_RESULT.json`.

The exact shell wrapper, SHA-256
`bf5baf73849e03fa596554ca96a2a11d575a328f3215a419340263afad758e54`,
was then tested rather than inferred from the module result. With
`PYTHON_EXECUTABLE=/usr/bin/python3` and `AIGC_DEVICE=cuda:0`, it returned code
0 on the same complete 576-image directory in 48.67 seconds, emitted no stderr,
and reproduced the identical output SHA-256. The raw wrapper evidence is in
`FRONTIER_ENSEMBLE_WRAPPER_RESULT.json`.

This closes single-T4 numerical and end-to-end runner feasibility for the exact
candidate contract. It does not establish hidden-set generalization, CPU/MPS
equivalence, unknown judge-hardware compatibility or checkpoint distribution.
The exact v9 checkpoint still exists only in the Kaggle working session, both
checkpoint distribution URLs are null, and final code/weight licensing must
preserve all upstream data restrictions. The candidate therefore remains
unpublished and separate from the locally complete v6 fallback.

## 2026-08-31 — v9 size is model substance, not removable training state

A read-only `/usr/bin/python3` inspection loaded the exact v9 checkpoint with
`weights_only=True` and returned code 0. The 1,263,202,267-byte file contains
310 state-dictionary tensors occupying 1,263,104,004 bytes. Its remaining keys
are only inference metadata such as model name, image size, normalization,
preprocessing, parameter count and seed. No optimizer state or training history
is present.

Therefore more than 99.99% of the file is model tensors. Removing metadata
cannot materially improve checkpoint transport. Converting FP32 tensors to
FP16 or another representation would create a different model artifact and
cannot inherit the existing exact-score evidence; it would require a new
full promotion and numerical-contract validation. No such conversion is
silently substituted. Exact evidence is preserved in
`FRONTIER_V9_CHECKPOINT_INSPECTION_RESULT.json`.

## 2026-08-31 — exact-metadata NTIRE ensemble diagnosis

Before reading any result, one diagnosis-only analysis was frozen in
`DEADLINE_ADVERSARIAL_PLAN.md`: reconstruct the prior NTIRE sample using exact
image format, color mode, width and height matches inside each retained label
stratum; require the known 86 rows balanced 43/43; join those images by SHA-256
to all 20 already-saved condition files; and compare the frozen v6 score with
the already-selected 75/25 blend. No new inference, model, data, transform,
weight, threshold or selection was permitted. The analysis script SHA-256 was
`3729857230938b997785f6e880539a9d3c0dad53623db7d8f4f61e1564e46f66`.

The isolated Kaggle `/usr/bin/python3` command returned code 0. The source
manifest SHA-256 was
`8327cb6ed314e9693c107719a875e4c6caf9825734206e3419d6c8c1e573a444`;
the matched-subset inventory SHA-256 was
`a09401eccd4455c73e845c84a5bb337b7e0abfa4cabc998eb93be72e1bb7467c`;
and the 20-file prediction inventory SHA-256 was
`193747446404aeb525e4a2579cd06d54dff57ec0ff2f795da155e81a14f438a0`.
The subset contained 86 images, 43 per class, across 31 exact metadata strata.

V6 scored 0.983775 clean AUC, 0.963344 pooled-robust AUC and 0.973559 on the
50/50 official-style score. The 75/25 blend scored 0.985398, 0.965475 and
0.975436 respectively, changes of +0.001622, +0.002131 and +0.001877. Heavy
Gaussian noise sigma 0.10 remained the weakest condition, improving from
0.727961 to 0.732829 (+0.004867). Every reported aggregate moved in the same
direction.

This is evidence against one narrow shortcut explanation: the small NTIRE gain
does not disappear when exact container and geometry opportunities are matched
inside retained strata. It is not hidden-set evidence. The sample is small,
and semantic content, acquisition source, generator identity and other
unmeasured properties remain confounded. The result therefore preserves the
ensemble candidate but does not increase its blend weight or replace any
frozen promotion decision. Exact output is in
`FRONTIER_ENSEMBLE_NTIRE_METADATA_MATCHED_RESULT.json`.

## 2026-08-31 — exact v9 privately preserved, release gates remain open

The Kaggle notebook was explicitly quick-saved with output preservation under
the version name `Preserve exact v9 checkpoint dd6b26c`. Version 1 completed
successfully with script version ID `346097785`. Its private output viewer
shows the exact-name v9 checkpoint at approximately 1.26 GB. Immediately before
the save, the source file was 1,263,202,267 bytes with SHA-256
`dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075`.

This is stronger preservation than a live Kaggle working directory, but it is
not overstated: the saved output has not been downloaded and re-hashed, v9 is
still absent locally, and the private version is not a public distribution URL.
The exact state and blockers are recorded in
`ENSEMBLE_CHECKPOINT_MANIFEST.json` and `MODEL_RELEASE_READINESS.md`.

## 2026-08-31 — compact v9 export fails exact-equivalence gate

The predeclared packaging-only ablation ran after repairing the restarted
Kaggle image's inconsistent Pillow installation. The first invocation failed
before loading the checkpoint with `ImportError: cannot import name '_Ink' from
PIL._typing`; `pip --force-reinstall Pillow==11.3.0` returned zero, and the
unchanged experiment then ran in a fresh `/usr/bin/python3` subprocess. The
executed 2,009-byte inline source had SHA-256
`5500e08b897bee2978acc5df02000a6b0b9abe67ea499cd06da36148acaae6f4`.

It verified the exact v9 source hash, converted floating state tensors to FP16
storage and produced a 631,625,115-byte checkpoint with SHA-256
`85094e995c17cca25c1e5367a580d88f5ceb927045fc978e52d4ba1b1c845c45`,
a reduction of 631,577,152 bytes. It then re-verified the sealed promotion
package and scored all 576 already-open Qwen images under the unchanged
one-T4, physical-batch-64, CUDA-FP16 blend contract.

The v6 branch remained bit-exact, but the compact-v9 blend did not: maximum
score drift was 0.00390625, mean drift 0.00020618, AUC drift 0.00015673 and
maximum rank displacement four. The command completed, but the strict
equivalence decision is **fail**. No tolerance was relaxed. The compact file is
rejected from the exact ensemble contract and the original FP32 v9 remains
required. Exact evidence is in `FRONTIER_V9_FP16_EXPORT_AUDIT_RESULT.json`.

## 2026-08-31 — condition-audit identity correction

A generic frozen-report auditor was added to rank aggregate conditions and
generator/real-source pairs without training, threshold selection or model
promotion. Its first inputs were the two locally preserved detailed DINOv2-L
control matrices. Those files expose `model: vit_large_patch14_dinov2.lvd142m`
and a checkpoint below the `family-mix-v6-dinov2l...` output directory.

The initial human description incorrectly called one DINO subgroup result a
selected-model result. Inspection of the embedded model identity caught the
mistake before any intervention. The DINO control's subgroup values are not
PE-Core evidence and are not used to alter the selected candidate. The tool now
emits the embedded model and checkpoint beside its rankings. Its two focused
tests and the full suite completed with **75 passed**. The retained lesson is
procedural: a filename containing `v6` identifies a dataset/experiment stage,
not necessarily the selected architecture; every score must be joined to the
embedded model and checkpoint before interpretation.

The same inspection found that `ENSEMBLE_CHECKPOINT_MANIFEST.json` had placed
the rejected 631,625,115-byte compact-v9 ablation beneath the v6 object. The
hash and rejection measurements matched the compact-v9 report, but the JSON
lineage was incorrect. The block was moved beneath v9; a regression assertion
now requires it to be absent from v6 and attached to the exact rejected-v9
hash. This was a documentation/provenance repair only and changed no artifact,
score, promotion decision or runner.

## 2026-08-31 — release lineage audit passes; distribution still blocked

`scripts/verify_release_artifact_lineage.py` cross-checked the ensemble
manifest against the promotion, fixed-batch, FP16 arithmetic, single-GPU,
end-to-end and compact-v9 audit records. It also streamed and re-hashed the
local ignored v6 file. The command returned zero: all recorded exact v6/v9
hashes agreed, the compact artifact was attached only to v9, the compact source
was exact v9, its rejection flag agreed across records, and the combined byte
count was internally consistent.

The local v6 file was present at 631,645,967 bytes with SHA-256
`48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644`.
Exact v9 remained absent locally. The report therefore truthfully set
`distribution_ready: false` with three blockers: missing local v9, no public
immutable v6 URL and no public immutable v9 URL. This is a consistency pass,
not a release or hidden-score claim.

## 2026-08-31 — selected-model prompt/content dependence diagnosis

The exact selected-v6 clean prediction file for the codec- and
geometry-controlled Qwen gate was joined to its 288-fake balanced grid: 16
prompt IDs repeated once across each of 18 current generators. The audit used
the embedded model identity `vit_pe_core_large_patch14_336`, checkpoint hash
`48ea5077...b644`, JPEG-q96 normalization, stretch preprocessing and reference
inference policy. It changed no prediction or checkpoint.

Prompt identity explained **52.7494%** of fake-score variance, generator
identity **14.7481%**, and the residual prompt-generator interaction 32.5025%.
Prompt 606 averaged 0.330218 AI score and 0.790702 AUC against all 288 frozen
reals; prompt 990 averaged 0.403275 and 0.835841 AUC. Three-generator visual
checks showed a photorealistic snowy palace for 606 and a beachwear
advertisement with a woman and Chinese promotional text for 990. The pinned
official metadata provided evaluation dimensions, not the prompt wording; its
SHA-256 was
`8ba913d292edd791a3abd19ce9d60fc7322a2d2a22b1c4b8b763f62f2d64c618`.

This is a serious content-sensitivity warning but not proof that semantics are
the learned shortcut: these prompts may simply produce unusually realistic
images across generators. The inspected audit rows remain excluded from
training, and the gate is now consumed for any content-directed selection. A
future intervention requires different lawful content-matched data and a new
sealed unseen-prompt gate. Full interpretation is in
`PROMPT_CONTENT_DEPENDENCE_AUDIT.md`.

## 2026-08-31 — second unseen-prompt Qwen gate confirms blend, not safety

A second frozen Qwen Image Bench package used prompt IDs 22, 183, 309, 327,
338, 420, 488, 536, 568, 631, 721, 772, 838, 859, 896 and 900 across all 18
generators, paired with the same 288 real rows. Its prompt IDs and fake hashes
were disjoint from both the original diagnosis gate and v9 training partition.
Both labels were converted to JPEG quality 96 and stretched to the model input.

The first Kaggle invocation failed before dataset download or scoring because
`kaggle_shape_stress_v6` was absent. After uploading the existing checksum-
verified helper, the unchanged evaluation command returned zero on one Tesla
T4. Exact physical batch 64 and GPU-FP16 sigmoid/blend arithmetic were retained.
V6 scored 0.932412 clean AUC, 0.850260 worst-generator AUC and 0.641164 worst
generator/real-source pair. The already-selected 75/25 blend scored 0.954638,
0.895399 and 0.722522 respectively. Its weakest prompt AUC improved from
0.704186 to 0.807581.

The stronger result did not eliminate the principal failure mode. Prompt
identity explained 61.4365% of v6 fake-score variance and 58.5940% for the
blend, while generator identity explained only 10.8625% and 11.6882%.
Inspection of the two weakest prompt groups showed realistic text-heavy/social
design: a rustic shop/counter scene with Chinese or Taiwanese signs and a
sunset/woman motivational post with Chinese text. These observations are
diagnostic only; the new gate is now consumed and excluded from training.

The completed run took 88.5824 seconds including download, model loading,
decoding and inference, with 4,275,770,368 bytes peak CUDA allocation. The
7,401-byte private full report has SHA-256
`9946c2f320df3d972c3e584d3e30fde22f39e7f2d78077cd023ce1f54600acc1`;
its compact committed record is `QWEN_UNSEEN_PROMPT_ENSEMBLE_RESULT.json`.
The earlier local MPS v6 score was 0.932653; the 0.000241 difference is retained
as device/arithmetic drift rather than mixed into the exact CUDA comparison.

## 2026-08-31 — normalized authentic scores remain source-dependent

The selected-v6 frozen predictions were joined to the 288 authentic rows shared
by both Qwen gates. This diagnosis occurred after every real and fake had been
converted to JPEG quality 96 and stretched to the same tensor geometry. Named
real source still explained 33.9694% of authentic score variance; a fixed-seed
10,000-permutation test returned p=0.000100 with plus-one correction.

CIFAKE-CIFAR10 reals averaged 0.356201 AI score and 0.825072 AUC against the
second unseen-prompt fake set. SID averaged 0.160842, LAION5B 0.098153, FFHQ
0.075182 and LSUN-Church 0.033626. At the illustrative, uncalibrated 0.5
threshold, 25.86% of CIFAKE reals but none of the LSUN-Church reals were false
positives.

Bounded visual inspection showed that the hardest CIFAKE examples were tiny
32-by-32 animal photographs enlarged for inference. The two highest-scoring
LAION examples were a watermarked anatomical stock render and framed abstract
artwork. A high-scoring SID example was a low-contrast parking-lot scene with
glare and clouds. This exposes low-resolution and non-photographic authentic
weaknesses, plus ambiguity in whether human-made render/art should count as
authentic. It does not prove the model's causal feature. Because both Qwen
gates reuse the same real rows, the identical real statistics are one audit,
not two independent confirmations. Compact evidence is in
`REAL_SOURCE_SCORE_DEPENDENCE_RESULT.json`.

## 2026-08-31 — manifest audit separates prompt holdout from generator holdout

`scripts/audit_generator_coverage.py` inventoried exact generator names and raw
family labels in v6, v9 and five frozen gates. V6 training has 12 exact names
across nine raw family labels. V9 has 30 across ten because the 18 Qwen names
were added through its disjoint training partition.

All 18 Qwen names are absent from v6 training but present in v9 training. Thus
v6's 0.933274 and 0.932412 clean Qwen AUCs test named-generator transfer, while
the blend's 0.957158 and 0.954638 test new prompts and pixels from known names.
The latter is still valuable, but it is not an unseen-generator result.

Six of eight internal gate names are absent from both manifests. Community
Forensics has 78 exact names absent from both, but labels them `LatDiff`, which
is semantically related to latent diffusion despite a raw spelling mismatch.
NTIRE records only `NTIRE-2026-shard-5-undisclosed`; treating that placeholder
as a novel generator would be false precision. The ignored 21,002-byte full
audit has SHA-256
`7424d54dfbebe6b1c4aef47aafa19d012a8266c994d780370c6f1d86a6341ed4`;
compact evidence is in `GENERATOR_COVERAGE_AUDIT_RESULT.json`.

## 2026-08-31 — exact v9 recovered and re-hashed from private saved output

The authenticated private Kaggle version metadata exposed a file-specific,
ephemeral download route for `000_model_v9_exact_dd6b26c.pt`. The signed route
was kept only in a permission-restricted temporary file, never printed into a
repository artifact, and deleted immediately after transfer. `curl` returned
HTTP 200 and downloaded exactly 1,263,202,267 bytes to the gitignored path
`models/model_v9.pt`.

Independent local hashing returned SHA-256
`dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075`,
exactly matching both the pre-save hash and ensemble manifest. The release-
lineage verifier then returned zero with both local checkpoints present and
matching. Its ignored 5,042-byte report has SHA-256
`9941d0b620be4a9e95c03660195ccce12b93ca1ed11340300e7c3e86b017e81b`.
`distribution_ready` remains false for two reasons: neither exact checkpoint
has a public immutable distribution URL. Local recovery is not publication,
submission, portability proof or a trained-weight licensing decision.

## 2026-08-31 — heavy-noise aggregate hides low-resolution-real inversion

Six exact row-level prediction files were recovered from the same private saved
promotion run: clean and Gaussian-noise-sigma-0.10 records for the 3,071-row
internal, 624-row Community Forensics and 576-row Qwen gates. Each recovered
file matched its saved byte count. The ephemeral signed routes were deleted
after transfer and no route or credential was written to the repository.

`scripts/audit_frozen_group_failures.py` verified identical clean/noise row
ordering, then recomputed v6 and blend AUC by fake generator, authentic source
and every generator/source pair. The headline blend noise AUCs were 0.870375,
0.796608 and 0.755160. Those aggregates conceal a repeatable inversion against
low-resolution CIFAKE-CIFAR10 reals: restricting the real side to that source
reduced noise AUC against all fakes to 0.529291 internally, 0.417989 on
Community Forensics and 0.386434 on Qwen. The blend's mean AI score for those
noisy authentic images was approximately 0.83 in all three gates, exceeding
the corresponding fake-pool means of 0.8031, 0.7185 and 0.6750.

The strongest adequately sized pair failures were Imagen versus CIFAKE reals
at 0.130344 AUC (192 fakes, 307 reals) and Seedream-4.5 versus CIFAKE reals at
0.273707 (16 fakes, 58 reals). Community named-model pairs contain only four
fakes each and are retained as example localization, not stable estimates.
This diagnosis is consumed and changes no model or weight. The prior v7 global
noise-augmentation ablation already regressed; any targeted repair now requires
new disjoint lawful low-resolution training data and a separately frozen gate.
Compact evidence is in `FROZEN_GROUP_FAILURE_AUDIT_RESULT.json`; the ignored
642,843-byte full report has SHA-256
`91f3bec083d17b19f34449a41c91ca5868f7a1426842bd075b24ff9ec7f0e46e`.

## 2026-08-31 — freeze independent CIFAR-100 low-resolution authentic gate

Before any new score or repair was observed, an independent authentic-side
gate was frozen from the CIFAR-100 official test split. The immutable public
mirror is pinned at revision
`aadb3af77e9048adbea6b47c21a81e47dd092ae5`; its 23,772,751-byte Parquet file
has SHA-256
`98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68`.
The selector takes the ten lowest image SHA-256 values within each of all 100
fine classes, yielding 1,000 unique authentic 32-by-32 rows. Its local manifest
SHA-256 is
`7e54acc25fd136bc8aa12532f8172f9c12735ccc9ff82a992cd3a5da01072fb8`.

The privacy-safe Kaggle package is 2,772,682 bytes with SHA-256
`f6520c1b36e81d04ef60ece5386927a5acc4dba0d5bdd577d43ab24b4dfde67b`.
Its embedded provenance names the source manifest without exposing an absolute
workstation path. The gate is evaluation-only because the official source page
does not state a reusable SPDX or Creative Commons licence. It must never enter
training, tuning, thresholding, calibration, blend selection or redistribution.

`CIFAR100_LOWRES_GATE_PLAN.json` freezes the interpretation before scoring.
The primary comparison will pair these new authentic scores with the unchanged
288 fake rows from the already-frozen first Qwen gate under clean and Gaussian
noise sigma 0.10. This creates new real-source evidence only; it does not create
a new fake-generator gate. A blend noise AUC below 0.50 means a new-source
ranking inversion, 0.50–0.70 means severe general low-resolution/noise failure,
0.70–0.80 remains materially inconclusive, and at least 0.80 makes a
CIFAKE-specific interaction more plausible without proving broad robustness.
No score has been observed yet. Six focused tests and the full **86-test** suite
passed; `git diff --check` passed. A first system-Python compile attempt was
blocked only by macOS denying its external bytecode-cache path; the unchanged
files compiled successfully with the repository virtual environment and a
temporary cache under `/tmp`.

## 2026-08-31 — independent low-resolution gate confirms ranking inversion

The checksum-frozen CIFAR-100 gate completed on one Kaggle Tesla T4 under
PyTorch `2.10.0+cu128` and CUDA 12.8. The evaluator reverified all 1,000 image
hashes plus both exact checkpoint hashes before scoring. The clean and
sigma-0.10 tensors have SHA-256 values
`4f33afcb40720e1a124893e31af142a8a14015f80bf2344ecc0ce85ba9484a99`
and `36b8be69c2d15b29ace88f9171399fdeffbd567d3980175f4abb28125065da8d`.
Each condition used sixteen physical batches of 64, padded only after hashing
the final logical batch. Peak allocated CUDA memory was 4,276,818,944 bytes.

Two invocations failed before scoring and are not results. The first correctly
refused to start because the exact v9 checkpoint was absent. After a private
checksum-matched v9 input was attached, the second failed while loading v6:
the evaluator had mistaken the upstream model-name suffix `_336` for the
checkpoint input size. Both exact checkpoints record `image_size=224` and a
257-token positional grid. The reconstruction was corrected to 224, an
explicit checkpoint-metadata guard and regression test were added, all 89
tests passed, and commit `c625b2e` preserved the correction before any score
was seen. No gate image, transform, seed, blend weight or interpretation band
changed.

The corrected command returned zero. It produced clean AUC 0.850858 for v6 and
0.891642 for the 75/25 blend against the unchanged 288 Qwen fake rows. Under
Gaussian noise sigma 0.10, v6 fell to 0.449342 and the blend to 0.473915. The
blend's 2,000-replicate 95-percentile interval was 0.426659 to 0.519826. Mean
AI scores inverted as well: 0.758414 for the new noisy authentic rows versus
0.674959 for the unchanged noisy fake rows. This is not merely a threshold
shift; the ordering is approximately reversed.

The predeclared interpretation is therefore consumed as **general
low-resolution authentic ranking inversion confirmed on the new source**.
The exact compact result is `CIFAR100_LOWRES_GATE_RESULT.json` with SHA-256
`1fac3bede6bc64d2eb2f86e92ff503186fa4e3638806ca09d89a1b47cf0911fc`.
The recovered clean/noise row files match the evaluator report at SHA-256
`18056c4124d41c8e5a90b5b37f485c31dcae97bb6a22a35ad9454aa416b56fdb`
and `9c5b5c761b8f2f2dc41b01c76d4c4ab2fca9216cfc24bffedbaa4f12514a4e9c`.
They remain ignored and consumed. They cannot be used to train, tune,
calibrate, choose a threshold, change the 75/25 blend or select a repair.

## 2026-08-31 — matched low-resolution v10 repair is rejected

The v10 run was fail-closed before learning. Three early invocations were not
treated as experiments: one exposed a filename-only exclusion mistake, one
found two exact training duplicates, and one found five frozen-gate overlaps.
The acquisition/assembly path was corrected before model updates. The final
29,534-row manifest has SHA-256
`02a09d816ce386a25ea94fdfafd87901dce5ee79ed01523f8d71c7945f1d8d69`,
zero frozen-gate content overlap and zero organizer demo rows.

One Kaggle Tesla T4 then completed the unchanged training command. PE-Core-L's
315,776,001-parameter backbone stayed frozen and only the 1,025-parameter
linear head trained for one epoch, 462 steps. Final loss was 0.2897299224,
elapsed time 648.5385 seconds and peak allocated CUDA memory 2,322,498,048
bytes. The ignored 1,263,202,267-byte checkpoint has SHA-256
`633d6e0dada31dc1bf3e97e1cf1b534b7a54a21d3c0f7bd3aee9609e8c5f71f9`.

Before opening the fresh gate, the predeclared 75/25 v6/v10 blend passed all
four existing internal clean checks. Selection clean AUC improved from
0.981815 to 0.987316 and worst-pair AUC from 0.872417 to 0.912475; content clean
AUC improved from 0.994350 to 0.995625 and worst-pair AUC from 0.969730 to
0.976922. The fresh gate was not read during this screen.

The one-shot promotion gate contained 1,000 new CIFAR-100 authentic rows and
144 new Qwen fake rows from 18 generators and eight new prompts. Clean AUC
improved from 0.925257 for v6 to 0.937642 for the blend. Under the exact
workshop Gaussian-noise sigma-0.10 transform, v6 scored 0.391219 and the blend
0.462979. Although the +0.071760 delta passed the improvement rule, it failed
the predeclared 0.60 absolute floor. Noisy authentic mean AI score 0.725172
also exceeded fake mean 0.683090, failing the no-inversion rule. The candidate
therefore failed two frozen checks and is rejected. Exact compact evidence is
`V10_LOWRES_REPAIR_RESULT.json`; the ignored promotion report has SHA-256
`75e04e559a9e85388bc499873ef0517acf43354ba6326f2065c00c8c131850ac`.

Only after rejection, the saved predictions were decomposed without changing
any model. V10 alone had clean AUC 0.958340 and noise AUC 0.687316; noisy real
mean 0.662199 was below fake mean 0.774356, so its head did move in the intended
direction. The fixed blend diluted that gain because the failed v6 ranking had
75% weight. This is mechanism diagnosis, not permission to choose a new blend.
The consumed gate cannot train, tune, calibrate, threshold or rescue v10. Any
new candidate requires a genuinely different frozen mechanism and another
disjoint gate.

## 2026-08-31 — prove one-backbone, three-head inference identity

Before constructing another candidate, the exact v6 inference export and exact
v9/v10 Kaggle checkpoints were compared tensor by tensor. V9 and v10 have 308
exactly equal full-precision backbone tensors with raw SHA-256
`9dbe9740d698ff3e32d4026f74ea0ed758c0cf9afe12d7011d7943e4da1f8615`.
Every v6 FP16 backbone tensor equals the corresponding v9 tensor after the
same cast; unequal tensor count is zero. The largest full-precision-to-FP16
difference before that cast is 0.0070381165, so the claim is explicitly limited
to the verified FP16 inference representation rather than an unavailable
full-precision v6 source checkpoint.

An independent Kaggle CUDA check then ran one FP16 image through separately
loaded v6, v9 and v10 models and through a single shared backbone with three
heads. Maximum absolute logit difference was exactly zero for all three heads.
One backbone plus three heads has 315,778,051 parameters and remains below the
2-billion limit. The ignored 6,181-byte v10 head export has SHA-256
`822aa20dad25199744a99c737c337bb15f5108b3b5c84f0154dbf6719f05ae2e`;
its source checkpoint remains the rejected v10 artifact. Compact evidence is
in `BACKBONE_MULTIHEAD_IDENTITY_RESULT.json`. This proves an implementation
identity, not that any routing policy improves generalization.

## 2026-08-31 — freeze label-blind quality router from training-only pixels

The post-rejection v10 diagnosis suggested a new mechanism: use v10 only for
severe-noise inputs and preserve the verified 75/25 v6/v9 path otherwise. The
router does not inspect labels, model scores, filenames, dimensions or codecs.
It estimates high-frequency noise from the median absolute diagonal 2-by-2
Haar coefficient after aspect-preserving resize and center crop, and routes to
v10 only when that estimate is at least 0.055.

The threshold was frozen using 1,760 training-only images across 22 source
groups before any router screen. The ignored audit report has SHA-256
`01a5ca930cb1544bf88d28049e66ee6b3aa2e1dedb9d409fa1e75e6d5cd1eb36`.
Clean route rate was 0, sigma-0.02 was 0, sigma-0.05 was 0.044318 and the exact
workshop sigma-0.10 route rate was 1.0. Clean-versus-sigma-0.10 transform AUC
was 1.0. These figures show only that the statistic distinguishes the synthetic
transform on this training sample; they do not prove that naturally grainy or
unseen images will route correctly. The rule and one-shot promotion criteria
were frozen in `QUALITY_ROUTED_MULTIHEAD_PLAN.json` at
`2026-08-31T13:43:28+08:00`.

## 2026-08-31 — preserve a fresh unopened NTIRE quality-route gate

A new evaluation-only gate was selected from pinned NTIRE 2026 shard 5 after
excluding all 512 earlier NTIRE source filenames. It contains 512 authentic and
512 fake rows with zero earlier filename overlap and zero earlier image-byte
overlap. The local manifest SHA-256 is
`dfd3f196106544d586a3eb32c22f94d213f0ddd0f642f07d5cfc9e1fb08e2bb6`.
The ignored private Kaggle package is 608,677,127 bytes with SHA-256
`5c68565fbf6a02242af5481e4720dd435b0145ae09306ff4fa80e00c30eeb8c9`.
It contains zero organizer demo-only rows. No explicit reusable licence was
found in the pinned source card, so the images remain evaluation-only and are
not redistribution artifacts.

The gate remains local, unattached to Kaggle and unopened. It may be opened
once only if every already-consumed internal screen passes. The frozen gate
plan is `NTIRE_V11_QUALITY_ROUTE_GATE_PLAN.json`. Its limitations are explicit:
the rows are byte-new but come from the same NTIRE shard as an earlier audit,
the generator identities remain undisclosed, and this gate cannot estimate the
organizer's hidden distribution.

## 2026-08-31 — start the consumed-gate quality-route screen

The first Kaggle invocation failed before model loading or scoring because the
uploaded evaluator omitted the exact existing
`kaggle_evaluate_community_forensics` helper module. This is a packaging failure,
not an experiment result. The missing checksum-pinned dependency and its five
transitive local helpers were uploaded, and the unchanged router and screen
scripts were rerun. The active script SHA-256 values are
`70bf8a7f3730acf7208b086fd3f16805685766a06195b09bf7a97dfc18fca15c`
and `9eba973e4119740d90a72e2a9a7274c68c4280ad871fc17068b9f496218b54d3`.
It successfully verified 32,305 family-package images, 1,088 earlier promotion
images and all 624 Community Forensics images before inference. The fresh
1,024-row NTIRE gate was not attached or opened during this screen.

The unchanged command returned zero. On the 13,349-row internal gate, clean AUC
was unchanged at 0.981415 while sigma-0.10 AUC improved from 0.842631 to
0.884594. On the 512-row NTIRE audit, clean AUC was unchanged at 0.982979 while
noise improved from 0.761833 to 0.792213. On the 624-row Community Forensics
audit, clean AUC was unchanged at 0.993040 while noise improved from 0.793012
to 0.819614. The router sent one of 13,349 internal clean rows and zero rows
from the other two clean gates to v10; every sigma-0.10 row routed to v10. No
gate had a mean-score inversion. All 12 frozen screen checks passed.

The exact ignored 6,771-byte report has SHA-256
`afc5b67c2e9d467c9dea49f12f00e8e6c330fbc6f6a6abf42c0d0d2902394813`;
compact evidence is `V11_QUALITY_ROUTE_SCREEN_RESULT.json`. This authorizes one
fresh-gate evaluation but does not promote v11. Kaggle has begun creating a
private dataset from the checksum-frozen 608,677,127-byte gate package; no
fresh image has yet been scored, and the final result must not be tuned after
it is observed.

## 2026-08-31 — v11 quality route fails its untouched promotion requirement

Kaggle attached the frozen private package as a content-equivalent extraction
rather than the original ZIP. The evaluator was adapted before execution to
accept only the unique extraction whose package inventory, manifest checksum,
row count and every image checksum matched the frozen values; no data,
candidate, threshold, condition or promotion rule changed. The exact evaluator
was 8,532 bytes with SHA-256
`61235bd2b90913c998bbe2a65f003bfb773f641b8852ec521c7c22e3461c2c0c`.
It verified all 1,024 image hashes and returned code zero.

On clean images, the normal and routed paths both scored 0.987041 AUC and the
route rate was zero. Under the exact workshop Gaussian-noise sigma-0.10
condition, every row routed to v10; AUC changed from 0.789886 for the normal
v6/v9 path to 0.805782 for the routed path. No mean-score inversion occurred.
This passes five frozen checks, including the 0.70 absolute noise floor, but
the improvement is only 0.015896 and therefore fails the predeclared 0.05
minimum. The all-checks-required decision rejects v11. The threshold, route,
head and decision rule will not be retuned using this consumed gate.

The exact ignored 3,098-byte report has SHA-256
`724916c915903746e2e4fdd071e013b26b6b06d2eaa1167d98ef8f3a08c1e1f8`;
compact evidence is `V11_QUALITY_ROUTE_PROMOTION_RESULT.json`. Organizer
demo-only rows used: zero. V11's shared-backbone engineering remains valid
implementation evidence, but the candidate is not selected. The retained
paths are the verified 75/25 v6/v9 CUDA ensemble and the simpler v6 fallback.

## 2026-08-31 — current release-lineage audit verifies both local checkpoints

After v11 rejection, the release audit was rerun against the exact local v6
and recovered v9 files plus the hash-bearing promotion, fixed-batch, FP16,
single-GPU and end-to-end records. A first invocation incorrectly included the
wrapper-only result, which contains an output hash but no checkpoint-hash
fields; the audit returned nonzero on those two missing fields and did not
change any artifact. The corrected applicability set returned zero.

V6 remained 631,645,967 bytes with SHA-256
`48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644`.
V9 remained 1,263,202,267 bytes with SHA-256
`dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075`.
Every lineage check passed. Distribution readiness remains false for exactly
two mechanical reasons: neither exact checkpoint has a public immutable URL.
The private Kaggle version and local files are preservation evidence, not
publication. `SUBMISSION_READINESS.md` now separates verified implementation,
external release gates and presentation/submission gates.

## 2026-08-31 — release-facing source tree fails closed only on weight URLs

`scripts/audit_submission_tree.py` was added to inspect the Git-tracked tree
without publishing it. The first run identified six environment-specific
personal/private Kaggle and local-home locators plus the missing checkpoint
URLs. The current release-facing files redact those locators while preserving
all checkpoint, prediction and report hashes. Earlier private Git history is
not rewritten and must not be exposed without a clean export or reviewed
squash.

After the redaction, the final audit inspected 189 tracked files. All required
judge-facing files were present, `run.sh` and `run_ensemble.sh` were executable,
no tracked checkpoint/archive or file over 10 MiB was found, no current-tree
private locator or private-key material was flagged, and the 631,552,002-
parameter candidate remained below the exclusive 2-billion limit. The command
returned nonzero for exactly one deliberate blocker: the exact v6 and v9
distribution URLs are still null. Compact evidence is
`SUBMISSION_TREE_AUDIT_RESULT.json`. The full suite then reported 112 passed.

## 2026-08-31 — history-free source bundle passes isolated verification

Commit `f97ab7f` was exported with `git archive` to an ignored ZIP, so no Git
history, ignored checkpoint, dataset, cache or output could enter by accident.
The 504,585-byte archive has SHA-256
`997459a29cd0098ee72ec0da57581b44559b8e4001c1f76e1e041bd7ec0554ba`.
It was extracted under a new `/tmp` directory and tested using the pinned local
environment with `PYTHONPATH` directed only at the extracted source. All 113
tests passed.

The same extracted tree then ran the release audit in history-free filesystem
mode. All 189 files, required deliverables, run-script modes and the parameter
limit passed; no private locator or forbidden artifact blocker appeared. The
audit returned nonzero only because the exact checkpoint distribution URLs
remain absent. Compact evidence is `PUBLIC_SOURCE_BUNDLE_RESULT.json`. The ZIP
is verification evidence, not a publication or submission, and must be rebuilt
from the final reviewed commit.

## 2026-08-31 — judge-facing narrative and timed demo are drafted

`DEMO_SCRIPT.md` now defines a 2-minute-45-second evidence-led recording: live
directory input, exact ensemble command, continuous JSON output, the strongest
Qwen and Community results, the 0.6141 weakest-pair warning, one real failure
example and the v6 fallback. `DEVPOST_DRAFT.md` maps the verified work to the
track criteria without claiming a hidden score, production readiness or a
solved detector. Neither file is published or submitted.

The source-tree audit now requires both drafts and rejects unresolved release
placeholders. With 192 tracked files, all mechanical checks still pass and no
current-tree private locator or forbidden artifact is flagged. The current
nonzero result is intentional: four repository/checkpoint/video placeholders
and the missing checkpoint distribution URLs require external publication and
logged-out verification before submission. These are action gates, not code or
model failures.

## 2026-08-31 — deterministic portable demo rehearsal passes on CPU

Before inference, `scripts/prepare_demo_rehearsal.py` selected four
already-consumed public-data rows using only a fixed seed, role and image
SHA-256: two CIFAKE-CIFAR10 authentic images, one CIFAKE Stable Diffusion image
and one Qwen Image Bench FLUX.2-pro image. The ignored manifest SHA-256 is
`b5efec0c4f2cff6472ce920eab07dce33aab53253a93025634cbab3dd04421a7`.
No organizer demo row was eligible, and the four outputs are rehearsal-only,
not model selection, calibration or evaluation evidence.

The first unchanged `run.sh` attempt requested MPS. The managed shell reported
that it did not expose a supported macOS MPS backend and returned nonzero while
moving the model, before inference or any score. The same frozen directory and
checkpoint then ran on CPU without modification. The command returned zero,
reported 315,776,001 parameters and produced four finite probabilities in
4.0 seconds wall time observed by the command runner. The output SHA-256 is
`8e12b9a5e1f15e8e80e50f9df9334b686cf6b3d733089e2bf8810bb3eff45b4f`.
Scores in filename order were 0.067177, 0.163845, 0.545811 and 0.986118. The
sample was selected before these scores and is too small for an accuracy claim.
Compact evidence is `DEMO_REHEARSAL_RESULT.json`; images and output remain
ignored.

The full repository suite then reported 115 passed.

## 2026-08-31 — standard checkpoint sidecar matches the release manifest

`CHECKPOINTS.sha256` now records the exact v6 and v9 paths and digests in a
format accepted by macOS `shasum -a 256 -c` and Linux `sha256sum -c`. A test
loads `ENSEMBLE_CHECKPOINT_MANIFEST.json` and requires both sidecar entries to
match the manifest exactly. This does not publish either file; it reduces the
chance that a judge or release step silently uses the wrong bytes after the
public URLs are supplied. The full suite reported 116 passed.

## 2026-08-31 — isolated wheel installation and CLI import pass

The project metadata now declares the exact runtime dependencies previously
present only in the requirements files, with pytest isolated in a `dev` extra.
An ignored source wheel was built successfully at 24,463 bytes with SHA-256
`80669d3b3493ce655129405a5256355c0607e3667944bd2201bd71b2125deebd`.
It installed first without dependencies in a new `/tmp` virtual environment,
where the package imported from site-packages and Torch was intentionally
absent. Installing the same wheel normally then installed every declared
dependency; imports reported Torch 2.8.0, timm 1.0.19, torchvision 0.23.0,
scikit-learn 1.6.1, PyArrow 21.0.0 and Pillow 11.3.0. From `/tmp`, the installed
`aigc_detector.predict --help` command returned zero. This verifies packaging
and CLI import only: no checkpoint was published or used and no inference was
claimed. Exact commands and limitations are in `PACKAGE_INSTALL_RESULT.json`.
The full repository suite then reported 118 passed after the release-audit
reporting regression was covered. The refreshed audit inspected 199 tracked
files and again found zero forbidden artifacts, oversized files, current-tree
private locators or private-key material. It remains intentionally nonzero only
for four public-link placeholders and missing immutable checkpoint URLs.

## 2026-08-31 — refreshed history-free source export passes in isolation

The first `git archive` invocation returned 128 before creating output because
the ignored `artifacts/source/` directory did not exist. After creating that
directory, the unchanged commit `9c4d45d` exported successfully to an ignored
520,096-byte ZIP with SHA-256
`1faab63ae60b5801671649afbd4d8b11bd4bc1f09d355eda369a43a28bc15845`.
The archive has 204 entries including directories and no Git history.

It was extracted to a new `/tmp` directory. The history-free 199-file audit
found zero forbidden artifacts, oversized files, private locators or private
keys and retained only the expected four release-link placeholders and absent
checkpoint URLs. With `PYTHONPATH` restricted to the extracted `src`, all 118
tests passed. The bundle is ignored, unpublished and not a submission. Because
this evidence is recorded in a later commit, the final reviewed commit must be
exported and reverified once more before publication.

## 2026-08-31 — installed wheel reproduces frozen v6 CPU rehearsal

The first installed-wheel inference was wrapped in macOS `/usr/bin/time -lp`.
The detector itself wrote all four outputs, but the wrapper then failed on the
managed shell's blocked `sysctl kern.clockrate` call and made the combined
command return one. It is therefore recorded as a timing-wrapper failure, not
as a successful inference command.

The unchanged installed-wheel prediction was rerun from `/tmp` without that
wrapper. It returned zero, reported 315,776,001 parameters on CPU and produced
the four scores 0.067177, 0.163845, 0.545811 and 0.986118 in the frozen manifest
order. They exactly match the earlier source-checkout rehearsal; the two output
files are byte-identical with SHA-256
`2350ba493c30b1d15ab54298e145578aa0824dba822bd9a8c944992d81ad938f`.
This closes v6 wheel-install inference portability on this local CPU only. The
four examples are too small for an accuracy claim, and the CUDA-only exact
ensemble contract remains separate.

## 2026-08-31 — trained-weight hosting is feasible; licence conflict remains

Official GitHub documentation currently permits each release asset below
2 GiB, up to 1,000 assets with no total release-size limit. The exact
631,645,967-byte v6 and 1,263,202,267-byte v9 files therefore fit as separate
assets. This confirms host capacity, not a publication.

The licence audit found a conflict that must not be hidden. Creative Commons'
2025 conservative AI-training guidance says NC training material implies
noncommercial model use/distribution and that publicly shared models should
follow ShareAlike terms where applicable. The OSI definition, by contrast,
requires commercial-use freedom and no restriction by field of endeavour.
Because the active lineage includes AFHQ-v2, FFHQ and other restricted/source-
specific real imagery, labelling the weights MIT, Apache-2.0 or OSI open source
would overclaim rights. `WEIGHT_RELEASE_DECISION.md` records the provisional
CC BY-NC-SA route for team-held rights, the mandatory third-party carveout, and
the alternative of retraining from a genuinely permissive lineage if literal
OSI compliance is required. No licence was applied and no release was made.

## 2026-08-31 — workshop licence rule invalidates the historical v6/v9 release lineage

The complete organizer workshop transcript was reread after the weight-licence
audit. In the Q&A the organizer states that a dataset marked non-commercial
cannot be used. The user has directed that workshop statements control where
they clarify the written page. V6 and v9 contain direct AFHQ-v2, FFHQ and
CelebA-HQ training lineage, all carrying non-commercial restrictions. They are
therefore historical experimental controls only and must not be published,
submitted or described as eligible candidates. A different output licence
cannot cure a prohibited training input. No organizer demo-only row was
involved in this discovery.

## 2026-08-31 — permissive train2017 acquisition passes and v12 rejects a forbidden path

The official 252,907,541-byte COCO annotation archive was verified at SHA-256
`113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`.
The first streaming parser was stopped with exit 130 after its quadratic buffer
handling made no useful download progress. A linear brace-aware parser then
scanned all 118,287 train2017 rows in about 2.16 seconds. A first acquisition
attempt returned nonzero because forcing HTTPS exposed a certificate-hostname
mismatch; the exact official HTTP image endpoint was restored without reducing
decode, geometry or checksum verification.

The successful command acquired exactly 6,000 deterministic train2017 images.
It rejects all 5,000 val2017 identities and licence IDs 1, 2, 3 and 6. The
retained counts are 3,757 licence-4 CC BY, 2,141 licence-5 CC BY-SA and 102
licence-7 no-known-restrictions images. Manifest SHA-256 is
`43fd192e9b784edaab6c19169dcba37896db673c4e673696b2d5d6f3ee620537`;
content-inventory SHA-256 is
`31dc92ce45d36b6a718899fe9e0d5ba797330ecb9a712b89c6d9f48038769b03`.
An independent audit confirmed 6,000 unique IDs, paths and image hashes, zero
demo rows, zero val2017 URLs and zero forbidden licence IDs.

The first v12 mixture build returned nonzero on a WildFake StyleGAN path whose
name explicitly contained `afhqv2`. The safety rule was kept; deterministic
selection now filters forbidden source terms before taking a group sample.
The predeclared StyleGAN count was reduced from 400 to the 250 path-clean rows
available, with ADM increased from 500 to 650 to preserve the frozen balance.
The rebuilt manifests contain 13,574 balanced training rows and 2,000 balanced
evaluation rows, 15,574 unique image hashes, zero train/evaluation overlap,
zero organizer demo rows and zero recorded non-commercial rows. Train and
evaluation manifest SHA-256 values are respectively
`b200306ae50e23ab31b41434410d54e2ae0672c863902a38c3aa9b598abb9ceb`
and `6c1c8f7004b3fae1129c3fdd8d1645dff6c8d6a2da5eef4f43aef15166013a85`.

## 2026-08-31 — raw v12 shortcut score is falsified by label-blind canonicalization

A metadata-only logistic probe scored 0.968933 train AUC and 0.998360 frozen-
evaluation AUC on the raw v12 files; square-only AUC was 0.862 and PNG-only AUC
was 0.966. This is evidence of a dataset shortcut, not a detector result, and
made the raw package ineligible as the trusted training input.

`scripts/materialize_canonical_v12.py` applies the same EXIF transpose, centre
square crop, 336-by-336 resize and JPEG quality-96/subsampling-0 encoding to
both labels. It completed all 15,574 rows, reducing 5,240,132,069 counted source
bytes to 950,456,665 derivative bytes with no canonical train/evaluation
overlap. Train/evaluation manifest SHA-256 values are
`8eaecdeb6b27220e4a1bff519a1a898321fb4f2577947b36af5440399e677611`
and `d61a8575f5330bd01a7351e52c0db2d8886731dee703d49943d381843ee50bd1`;
the derivative inventory is
`d407768b3daca652d546024a601c354bd1b7d37214fbf7c1996d0cfb5b8e6846`.
On the canonical manifests the same probe falls to 0.513093 evaluation AUC,
with square-only and PNG-only both exactly 0.500. File-byte complexity still
permits 0.657554 train AUC, but file size is never presented to the image model.
This removes the tested container/geometry shortcut; it does not prove semantic
or hidden-set transfer.

The canonical Kaggle archive is 970,978,505 bytes with SHA-256
`6bfcb918676cef772b7a71e2fad8ad2fd0789efab9803fb028fb1302cd801447`.
It contains 15,574 content-addressed images and inventory SHA-256
`ec78d74e62d8e1b1f75e661f2ea3338fa95be11e96694a3ed168b463fe314fa6`.
The v12 runner freezes those exact values, requires the canonicalization fields,
uses label-independent JPEG-q96 normalization at training and inference, and
passes the complete 134-test suite. The archive was uploaded only as a private
Kaggle dataset named `track5-permissive-canonical-v12`; Kaggle reported that it
was processing the dataset. No v12 training or score has yet been observed.

Before any v12 model score was observed, the two available T4 devices were
assigned one frozen representation control each: PE-Core-L on device 0 and
DINOv2-L on device 1. They use the identical package, seed, dataset-block
sampler, one-epoch frozen-backbone/linear-head recipe, label-independent input
alignment and predeclared clean group floors. Outputs are isolated. This is a
representation comparison, not permission to weaken a floor or tune either
candidate after opening the 2,000-row gate.

## 2026-08-31 — v12 Kaggle startup was falsified twice, then both clean screens completed

The first parallel launch did not reach package verification. Both child
processes were interrupted inside `Path("/kaggle/input").rglob(PACKAGE_NAME)`;
the notebook had many historical inputs attached, so the recursive discovery
walked their complete image trees. A reviewer then briefly suspected a doubled
transform call after concatenating overlapping `sed` ranges. Exact numbered
source inspection proved there is one call; the cancellation lost no checkpoint
and the new regression test invokes the transform exactly once. This was a
reviewer error, not a detector bug.

The first bounded lookup failed fast with `expected one extracted dataset ...
found []` because current Kaggle uses a nested owner/slug input layout rather
than only the older one-level mount layout.
The mount-aware lookup supports that exact layout and the legacy one-level
layout without descending into image trees. Runner SHA-256
`c7b63ccd30565c90b27c159bf086cbe16bee1b74cb6306e5ed23e99e84e24589`
then verified all 15,574 images independently in both processes. The full local
suite reports 140 passed tests.

The corrected launch returned zero for both candidates. DINOv2-L reached clean
AUC 0.882450, worst-generator AUC 0.681688, worst-real-source AUC 0.822223 and
worst generator/real-source pair AUC 0.605000. Its 1,213,023,315-byte checkpoint
has SHA-256 `db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b`.
PE-Core-L reached clean AUC 0.998540, worst-generator AUC 0.938063,
worst-real-source AUC 0.995837 and worst pair AUC 0.890500. Its
1,263,202,331-byte checkpoint has SHA-256
`f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2`.
Both clear the floors frozen before training. The PE score is treated as a
shortcut/leakage alarm, not a hidden-set claim.

Before reading any transformed v12 score, the exact evaluator was frozen at
SHA-256 `57d4b2e77b8c73ccc5a06862f3abd7fecea9abdec801347c8bb946f2a61180da`.
It applies the 19 workshop conditions individually, then the same label-blind
JPEG-q96 inference normalization, and persists each condition for resumability.
Both matrices are now running on separate T4 devices. No transformed score has
yet been used to choose, tune, blend or calibrate either candidate.

## 2026-08-31 — clean gate source confounding found; matched-source gate frozen

The v12 manifest audit found that the 2,000-row clean gate is byte-disjoint but
not source-matched. Every real row comes from COCO, CIFAKE or SID_Set, while
every fake row comes from WildFake, DiTFake or Qwen-Image-Bench. No evaluation
dataset contributes both labels. The train/evaluation dataset names are also
identical within their respective label roles. Therefore the clean screen can
reward collection recognition; PE-Core-L's 0.998540 is not trusted as causal AI
detection evidence. Four exact generator names and one family are held out, but
all three real-source names are seen in training.

A new matched-source falsification gate was frozen before its first score from
the permissively licensed CIFAKE test split. The first build was rejected
because 2,000 rows represented only 1,994 unique source images. Source-hash
deduplication was added and the rejected build moved to a temporary diagnostic
directory. The accepted build contains 1,000 real and 1,000 fake rows, 2,000
unique source hashes and 2,000 unique canonical images. It excludes 21,504
hashes recorded across 41 previous CIFAKE JSONL manifests and uses the same
label-blind v12 square JPEG canonicalization. Manifest SHA-256 is
`079f72e5f7839c7ea0bf88d9dfa68acd2a845da1229bfaf739132859dd6e9d8f`.

The ignored 50,966,812-byte package has SHA-256
`b91363fef08bceb3c72f86ca4e5d4fce8b0c0a530d79e56b431dfa8a0087d383`,
inventory SHA-256
`1b62ff5df23538879a4e922a68648fcee73b73d39fc359b6db71804e69c14f5c`
and 2,000 packaged unique images. It was uploaded as the private Kaggle dataset
`track5-cifake-matched-source-v12-gate`; the page read back Private, Version 1,
50.5 MB and the expected inventory. The clean interpretation bands and 0.60
transformed floor were committed in `CIFAKE_MATCHED_SOURCE_V12_GATE_PLAN.json`
before scoring. The current suite reports 141 passed tests.

## 2026-08-31 — completed v12 workshop matrices; matched-source evaluator frozen

Both predeclared workshop matrices completed successfully with return code zero
before the matched-source gate was attached. PE-Core-L reached **0.998540 clean
AUC**, **0.992608 pooled transformed AUC**, a **0.995574 official-style score**
and **0.935380 worst individual-condition AUC**. Its pooled worst fake-generator,
real-source and pair AUC values were respectively **0.936163**, **0.986104** and
**0.892196**. DINOv2-L reached **0.882450 clean AUC**, **0.861357 pooled
transformed AUC**, a **0.871904 official-style score** and **0.792617 worst
individual-condition AUC**. Its pooled worst fake-generator, real-source and
pair AUC values were **0.654952**, **0.802228** and **0.584120**. Gaussian noise
at sigma 0.10 was the weakest individual condition for both candidates.

These results establish that PE-Core-L is stronger on the source-confounded v12
gate and that the observed separation survives the workshop transformations.
They do **not** establish causal AIGC detection or hidden-set transfer because a
stable collection fingerprint can survive those same transformations.

Before the first matched-source score, the exact evaluator was frozen at
SHA-256 `ea1d65120274927d20678cf25583a735efb21c0eae031b68231b26695d7df70b`
(12,768 bytes). It verifies all 2,000 gate images, the immutable manifest and
package identities, zero raw/canonical identity overlap against both mounted
v12 manifests, the exact checkpoint hashes, and clean plus all 19 workshop
conditions. Its decision uses the already committed clean interpretation bands
and 0.60 worst-condition floor; it cannot train, tune, calibrate or change a
threshold. The complete local suite reports 148 passed tests. No matched-source
score had been observed when this entry and evaluator were frozen.

## 2026-08-31 — matched-source gate passes, earlier PE dominance is falsified

Both unchanged candidate processes returned zero after verifying all 2,000
matched-source images and proving zero source or canonical identity overlap
against all 15,574 v12 train/evaluation rows. PE-Core-L reached **0.922760 clean
AUC**, **0.892005 pooled transformed AUC**, a **0.907382 official-style score**
and **0.794470 worst-condition AUC**. DINOv2-L reached **0.913844 clean AUC**,
**0.899374 pooled transformed AUC**, a **0.906609 official-style score** and
**0.820027 worst-condition AUC**. Both pass the predeclared 0.80 clean and 0.60
transformed floors. Each ran on one Tesla T4; PE used 674.55 seconds and
2,895,338,496 peak allocated CUDA bytes, while DINO used 517.48 seconds and
2,979,893,760 bytes.

The official-style delta is only **0.000773** in PE's favour. PE has the higher
clean AUC, but DINO has the higher pooled robustness and noise floor. This
falsifies the earlier interpretation that PE overwhelmingly dominates DINO:
the source-confounded v12 gate exaggerated that difference. The new result is
useful within-source low-resolution CIFAKE evidence only; it cannot select a
broad hidden-set winner by itself.

The exact two checkpoints, four reports per candidate, evaluator and integrity
manifest were uploaded by the authenticated Kaggle CLI to the private dataset
`track5-v12-exact-checkpoints-and-gates`. The page read back **PRIVATE**,
Version 1, **2.48 GB** and **10 files**. Integrity-manifest SHA-256 is
`6bde71260d7afed2aa3d2fdb810442730f1770f74200ebf67c2c2d93a4401506`.
This is private preservation, not submission, release or publication. Exact
condition results and the claim boundary are in
`V12_MATCHED_SOURCE_GATE_RESULT.json`.

## 2026-08-31 — fresh high-resolution SID_Set gate frozen before scoring

The source-confounded v12 gate and the low-resolution CIFAKE matched-source
result leave a major unresolved question: whether either representation can
separate fresh higher-resolution real and synthetic images from one source.
The immutable SID_Set validation shard `validation-00001-of-00034.parquet` was
therefore acquired from revision
`dc03ead57929879319ce30a82bfcfb8d317b10bd`. The exact pinned-revision file is
505,844,042 bytes with SHA-256
`1447bbd98adf7eda68fca5615560c6b1de34c8e30157ff6b34ebd1e015a18042`.
An older first-upload size/hash did not match the pinned revision and was
rejected rather than treated as this artifact.

```bash
.venv/bin/python scripts/extract_sid_binary.py \
  datasets/sid_set/shards/validation-00001-of-00034.parquet \
  --output-root datasets/sid_binary_validation_00001_fresh \
  --output-split test
.venv/bin/python scripts/prepare_sid_fresh_matched_v12_gate.py \
  --source-manifest datasets/sid_binary_validation_00001_fresh/manifest-test-validation-00001-of-00034.jsonl \
  --source-shard datasets/sid_set/shards/validation-00001-of-00034.parquet \
  --dataset-root datasets \
  --output datasets/sid_fresh_matched_v12_gate
.venv/bin/python scripts/package_kaggle_subset.py \
  --manifest datasets/sid_fresh_matched_v12_gate/eval_matched.jsonl \
  --output datasets/packages/sid-fresh-matched-v12-gate.zip
```

The extraction command returned zero and produced 284 real, 290 fully
synthetic and 309 excluded tampered rows. The fresh-gate command then returned
zero, excluded 1,883 recorded historical SID identities across 34 JSONL
manifests, selected the lowest 284 source hashes per label and identically
canonicalized every row to 336-pixel JPEG q96. The result has 568 unique source
images, 568 unique derivatives, 284 rows per label, zero organizer-demo rows
and zero training-allowed rows. Its local manifest SHA-256 is
`d5480315eecea0123fb872a487420ea46fadffb4b8cc03bf51c574cae3577af3`.

A direct identity audit found zero source-hash and zero canonical-image overlap
against all 15,574 v12 rows plus the 2,000-row CIFAKE gate. The sealed
37,168,107-byte ZIP has SHA-256
`439434c4e59b3dbbd4cbe98b9b94464f9e201a3e60cb4560dd87e11ff31f74b0`,
inventory SHA-256
`19ca0c433aa4e5cb04f8e36262ff9ea430c382987bc0c5d535d3de42e8f71ca3`
and packaged-manifest SHA-256
`092f981ce515ee2061b9f406dd34e358cf0dc4aac5076f65675095135dcb7a27`.

Before any model score was observed, the exact candidate hashes, clean 0.80
floor, worst-condition 0.60 floor, all 20 workshop conditions and claim
boundary were frozen in `SID_FRESH_MATCHED_V12_GATE_PLAN.json`. The evaluator
is 13,175 bytes with SHA-256
`557714af13d4cef74d07a1f30679c9f8c4968cad46f2dc55b8ff67cc72b7d837`.
The targeted suite returned **9 passed**, and the full suite immediately before
the evaluator addition returned **150 passed**. The complete suite after the
evaluator addition returned **157 passed**. One syntax-check attempt was
blocked because system Python tried to write bytecode under the restricted
macOS user cache. The unchanged command with
`PYTHONPYCACHEPREFIX=/tmp/track5_sid_pycache` returned zero. No score exists at
this checkpoint. SID_Set does not expose exact generator identity, so even a
pass will be only fresh high-resolution same-source evidence.

## 2026-08-31 — fresh SID gate passes numerically but exposes a source confound

Both unchanged candidate processes returned zero on separate Tesla T4 GPUs
after verifying all 568 images, the frozen package and manifest identities,
the exact checkpoint hashes, zero source/canonical overlap against all 15,574
v12 rows, and zero demo or training-allowed rows. PE-Core-L scored **0.999733
clean AUC**, **0.997705 pooled transformed AUC**, **0.998719 official-style**
and **0.987186 worst-condition AUC**. DINOv2-L scored **0.860122**, **0.839642**,
**0.849882** and **0.788987** respectively. PE completed in 215.29 seconds
with 2,895,338,496 peak allocated CUDA bytes; DINO completed in 175.75 seconds
with 2,979,893,760 bytes. Both pass the frozen numerical floors.

The near-perfect PE number triggered, rather than ended, a post-score source
audit. Every one of the 284 selected fakes originated as a 1024 by 1024 PNG.
Of 284 selected reals, 283 originated as JPEG, one used another container, and
only 11 were square. On the original source metadata, a square-only rule has
0.980633802816901 AUC and a PNG-only rule has 1.0 AUC. The canonical evaluator
feeds only decoded 336-pixel JPEGs to the model, so this does not prove that the
model reads metadata. It does prove that crop composition and codec lineage
are label-confounded and may survive identical output conversion. The PE score
is therefore classified as a source-specific shortcut alarm, not evidence for
hidden-set promotion. DINO's lower result and the candidates' near tie on
CIFAKE strengthen that caution.

The exact remote progress JSON files, evaluator and a machine-readable source-
confound record were staged beside the privately preserved checkpoints. The
updated integrity manifest is 2,604 bytes with SHA-256
`75602edac15dff4ccd5e430766e4ad80a3a32c943e67f48232f368ebf8dc9141`.
The first private-version command used `/usr/bin/python3 -m kaggle` and returned
nonzero because this installed Kaggle package has no `__main__`. The retry used
the authenticated `/usr/local/bin/kaggle` executable and returned zero. The
private page read back **Private**, **Version 2**, **2.48 GB** and **14 files**.
No credential was read or printed. Exact scores and the claim boundary are in
`SID_FRESH_MATCHED_V12_GATE_RESULT.json`.

## 2026-08-31 — fixed representation blend and v6 same-gate comparison

Before any blend score was calculated, commit `b7ea6e7` froze an equal
probability rule: `0.5 * PE-Core-L probability + 0.5 * DINOv2-L probability`.
No blend weight was searched. The post-processor read the already-saved
predictions and completed successfully. On matched-source CIFAKE it reached
**0.945613 clean AUC**, **0.930195 pooled transformed AUC**, **0.937904
official-style** and **0.857868 worst-condition AUC**. The compact result file
was 8,086 bytes with SHA-256
`e4e260148cf00b437d54a9f73879b2cdd3289553f9cbc04d02cf1c95016667f4`.
On the source-confounded SID diagnostic it reached 0.993286 clean, 0.983885
pooled transformed, 0.988586 official-style and 0.948194 worst condition. The
SID output was 8,118 bytes with SHA-256
`5fdc1a4807d868deba4ff5e2c43582ad67d1fc29f528b06ba1da9f47a47cf811`.
SID cannot support promotion.

Commit `c3528a2` then froze a comparison against the protected v6 fallback before
v6 saw this gate. The first background launch command returned a PID but created
no process log and no evaluator process; it is recorded as a launch-control
failure. The explicit `subprocess.Popen` retry produced PID 1898, a live process
and a growing log. It returned zero after 633.76 seconds on one Tesla T4, with
2,895,338,496 peak allocated CUDA bytes. V6 scored **0.829156 clean**,
**0.716622 pooled transformed**, **0.772889 official-style** and **0.655350
worst-condition AUC**. Its 13,893-byte progress report has SHA-256
`cd2a675808e413677558ae2bae899a3a770bbcd0ebc6d60aa66c73eecbbde439`.

The v12 fixed blend therefore passed both predeclared comparison requirements:
official-style was 0.165015 higher and worst condition was 0.202519 higher. The
decision is to advance it as the leading representation-hedge candidate while
retaining v6 as an operational fallback. This is still an already-open,
low-resolution CIFAKE comparison and does not estimate the hidden score. Exact
machine-readable evidence is in `V12_FIXED_EQUAL_BLEND_RESULT.json` and
`V6_V12_MATCHED_SOURCE_COMPARISON_RESULT.json`.

## 2026-08-31 — unused modern-generator and fresh-real semantic audit pools

To attack the remaining content/source shortcut risk, the following commands
completed successfully without reading any organizer demo-only resource:

```bash
.venv/bin/python scripts/acquire_qwen_image_bench_audit.py \
  --output-root datasets/qwen_image_bench_semantic_gate \
  --per-model 8 --seed 20260901 --workers 8 \
  --exclude-manifest datasets/qwen_image_bench_audit/manifest.jsonl \
  --exclude-manifest datasets/qwen_image_bench_train_candidate/manifest.jsonl \
  --exclude-manifest datasets/qwen_image_bench_holdout/manifest.jsonl \
  --exclude-manifest datasets/qwen_image_bench_holdout_v2/manifest.jsonl \
  --purpose audit

.venv/bin/python scripts/acquire_coco_train2017_permissive_audit.py \
  --annotations datasets/source_archives/annotations_trainval2017.zip \
  --output datasets/coco_train2017_semantic_audit_pool \
  --count 1000 --seed 20260901 --workers 16 \
  --exclude-manifest datasets/coco_train2017_permissive_v12/manifest.jsonl
```

The Qwen command took 36.61 seconds and downloaded or verified 144 images: all
18 pinned benchmark generators at prompt IDs 23, 500, 549, 584, 739, 794, 882
and 919. Those IDs are disjoint from the 72 unique prompt IDs recorded across
all previous Qwen train/audit/holdout manifests. The source inventory SHA-256
is `3c84b4fc29f99838da156f8aa5932b5323bbecc01e3eccebe44d50d9a7535498`;
the manifest SHA-256 is
`6d666f65054429ecaf7705cddc2d6395d41a64a124444fd2833e0b968163fc65`.

The COCO command acquired 1,000 audit-only train2017 photographs after excluding
all 6,000 previously selected v12 COCO image IDs. It admitted only licence IDs
4, 5, 7 and 8, and reports zero val2017, NonCommercial or NoDerivatives rows.
Manifest SHA-256 is
`3841b934a925e94a5d37d86b4a0ad88d160bc24b8ba756723beb88303f7798e5`;
content-inventory SHA-256 is
`3bec4c153b4533f42bef149d878f9de019416b8ad83cf0797b6c4cc2ab4f5f78`.
The wrapper's seven focused tests passed. Both directories are ignored and
explicitly audit-only. No semantic matching, gate freeze or model score exists
yet.

## 2026-08-31 — reproducible semantic pairing and one-shot modern transfer audit

The first three semantic-pairing attempts were rejected before detector
scoring because a stripped timm CLIP loader was not reproducible: an unchanged
repeat changed 65 of 144 selected real assignments. The replacement pinned
`open_clip_torch==2.32.0`, `RN50-quickgelu`, Hugging Face repository
`timm/resnet50_clip.openai`, revision
`ec3d92cf63a5f9d591f0d611b736895966c73076` and checkpoint SHA-256
`da0baa37fb2211eee5729ab69ed587eb10d48f5b7076ceeed01552fc3d4cb4ee`.
Two independent builds then produced identical fake-feature, real-feature and
path-independent inventory hashes.

Before any detector score was observed, the 288-row, 144-pair gate, exact
checkpoint identities, eight prompts, 18 modern generator labels, 20
individual workshop conditions, fixed equal blend and five numerical floors
were committed. The sealed ZIP is 21,063,114 bytes with SHA-256
`52f6749bed16015a1511e6cc6e9e7072d50350b3755148e1c3bea6d645288d69`.
It contains zero training-allowed or organizer-demo rows and has zero source or
canonical identity overlap with all 15,574 v12 train/evaluation rows.

The Kaggle interactive session restarted before scoring and erased both
working checkpoint directories. The private preservation dataset was
downloaded again; recovered PE-Core-L and DINOv2-L files matched the frozen
checkpoint SHA-256 values exactly. The corrected evaluator recovery path then
passed the complete local suite (**178 passed, one non-failing core-count
warning**) and the Kaggle preflight verified the gate, v12 manifests and two
Tesla T4 devices.

Both unchanged candidate commands completed. PE-Core-L reached **0.998240
clean AUC**, **1.000000 paired accuracy**, **0.993827 worst-prompt AUC**,
**0.994108 pooled transformed AUC**, **0.996174 official-style** and
**0.984809 worst-condition AUC**. DINOv2-L reached **0.842761**, **0.881944**,
**0.702160**, **0.811537**, **0.827149** and **0.765529** respectively. The
predeclared 50/50 blend reached **0.987510**, **1.000000**, **0.956790**,
**0.974554**, **0.981032** and **0.949508**. Every candidate passed the loose
frozen audit floors; no weight was searched.

The near-perfect PE result is not treated as hidden proof. All fake rows still
come from Qwen Image Bench and all matched reals from fresh COCO train2017;
semantic pairing and identical JPEG canonicalization cannot rule out a
collection-level residual. Each generator has only eight pairs. The consumed
gate is forbidden for training, calibration, preprocessing, weight or model
selection, so the weaker blend result cannot be used to opportunistically
switch or tune the predeclared candidate. Exact compact evidence is in
`SEMANTIC_MATCHED_MODERN_V12_GATE_RESULT.json`. Full predictions, logs and
progress files were uploaded to the private Kaggle dataset
`track5-semantic-modern-v12-one-shot-audit`; the page read back **Private**.
This is preservation, not publication or submission.

## 2026-08-31 — compliant v12 directory-to-JSON contract verified on one T4

`run_v12.sh` and `aigc_detector.predict_v12` were added without replacing the
historical `run.sh`. The new runner verifies the exact v12 checkpoint hashes,
checks model/preprocessing/codec metadata, enumerates image paths
deterministically, outputs continuous AI-positive probabilities and fails
closed on missing artifacts. It loads PE-Core-L and DINOv2-L sequentially,
uses each checkpoint's own normalization and applies the frozen
short-side-crop plus label-independent JPEG q96 contract. Default mode is the
predeclared 50/50 float32 probability blend; `pe_core` is the single-model
fallback. The combined parameter count is checked against the exclusive 2B
limit.

The first timing-wrapper attempt failed before inference because Kaggle did
not contain `/usr/bin/time`. The unchanged runner was then instrumented
internally for elapsed time and CUDA peak allocation. The exact command on four
previously frozen, hash-verified rehearsal images returned zero on one Tesla
T4 in 19.4235 seconds, allocated at most 1,322,263,040 CUDA bytes and reported
619,004,930 parameters. Output contained four ordered unique paths and finite
probabilities in `[0, 1]`: 0.119282 and 0.225314 for the two real examples,
0.748089 for the Stable Diffusion example and 0.892030 for the FLUX.2-pro
example. Output SHA-256 is
`d37f8745b2191fe8ef788a11a22024bb068738d4ba0a1faaa8614cb1c4df7e7d`.

An unchanged second blend process returned zero in 20.2112 seconds with the
same peak allocation and produced the same 464 bytes exactly. The PE-only
fallback returned zero in 11.2456 seconds, reported 315,776,001 parameters and
produced output SHA-256
`280264334718a4d82e3df6bdabf46f6ebdb7d69e1e6faa265cdbec859a02ac13`.
These four rows prove the runnable contract and deterministic arithmetic on the
recorded environment; they are not accuracy, large-batch throughput, CPU, MPS
or hidden-set evidence. Exact evidence is in
`V12_RUNNABLE_CONTRACT_RESULT.json` and the checkpoint/release state is in
`V12_CHECKPOINT_MANIFEST.json`.

## 2026-08-31 — Open Images source rotation falsifies DINO as a universal hedge

Before scoring, an audit-only Open Images V7 validation pool was acquired from
the official CVDF public S3 listing. The deterministic 1,000-image sample used
seed 20260901, occupied 315,762,550 bytes and had key-inventory SHA-256
`1c602591ae6cb394e271fc32357b2b67e87613482acd94966ae322f36903ada5` and
content-inventory SHA-256
`9f9663d1ca43ceb9abb8fb5d7c7fcb21bcfc5ae72050d85766a0dfcd48f0e56a`.
Open Images labels the source CC BY 2.0, but this audit did not independently
verify each image's licence. Every row is therefore audit-only.

The unchanged 144 Qwen modern fakes were paired to 144 Open Images reals using
the already pinned CLIP pairing contract. The sealed gate ZIP SHA-256 was
`c9862550a1476e60b13e9c262e751d3d75e8c582a71d60496c064185b63e4906`;
manifest SHA-256 was
`a696d1a781dacaa66183fb96e6a4078ebdf3dd429dedb657524ddf846b3667d6`.
It contains zero training or organizer-demo rows and was attached as a private
Kaggle input before inference.

PE-Core-L completed the 20-condition audit in 118.056 seconds with peak CUDA
allocation 2,895,338,496 bytes. It scored **0.981891 clean AUC**, **0.969951
pooled robust AUC**, **0.975921 official-style**, **0.939742 worst condition**,
**0.919753 worst prompt** and **0.925817 pooled worst generator**. DINOv2-L
completed in 102.43 seconds with peak allocation 2,979,893,760 bytes and scored
**0.653815**, **0.629064**, **0.641440**, **0.577474**, **0.317901** and
**0.534854** respectively. The fixed equal blend scored **0.914280 clean**,
**0.890289 pooled**, **0.902284 official-style** and **0.834539 worst
condition**. No weights were searched.

Changing only the real collection from COCO train2017 to Open Images lowered
PE official-style AUC by 0.020253 but lowered DINO by 0.185710. PE therefore
survives this source rotation; DINO has a severe real-source dependence alarm,
and equal blending materially harms PE. This remains diagnostic only because
the same Qwen fake collection is reused and the gate is consumed. Compact exact
evidence is in `OPENIMAGES_SOURCE_ROTATION_V12_RESULT.json`.

## 2026-08-31 — identity-corrected Community Forensics v12 breadth audit

The initial 624-row Community Forensics package covered 312 fake images from 78
named latent-diffusion variants plus 312 real images across five sources. The
first launch failed before inference because a legacy audit-only manifest lacked
row-level `training_allowed` fields. The corrected validator accepts a missing
legacy field but still rejects any explicit true value. A stronger identity
preflight then stopped the second launch before inference: a canonical-only
check had missed 31 raw source identities overlapping the v12 evaluation set
(28 SID_Set and three CIFAKE real rows).

Those 31 rows were excluded before any v12 prediction. The corrected frozen
gate contains 593 rows: 281 real, 312 fake, all 78 named fake variants with four
examples each, and zero source or canonical identity overlap with 13,574 v12
training plus 2,000 v12 evaluation rows. Its derived manifest SHA-256 is
`5496fd110f73873c1d01f520ae3cf44893d3b036c3ab66f65ce4ef3ef04881d5`.
Because the post-filter classes are unequal, the plan uses AUC and deliberately
reports no threshold metric.

PE-Core-L completed clean plus all 19 individual workshop conditions in
328.725 seconds with peak CUDA allocation 2,895,338,496 bytes. It scored
**0.950201 clean AUC**, **0.947296 pooled robust**, **0.948748 official-style**,
**0.823421 worst condition**, **0.762456 clean worst fake model** and **0.753052
pooled worst fake model**, passing every frozen floor. DINOv2-L completed in
318.182 seconds with peak allocation 2,979,893,760 bytes and scored **0.811764**,
**0.807565**, **0.809665**, **0.796993**, **0.548043** and **0.550696**; it
failed the frozen 0.85 clean floor. The fixed 50/50 blend scored **0.918332
clean**, **0.911335 pooled**, **0.914833 official-style** and **0.847608 worst
condition**. The blend improves the single weakest condition by 0.024187 versus
PE but loses 0.033915 official-style AUC. No additional blend weight was tried.

This gate reinforces PE's aggregate breadth and falsifies the idea that equal
blending is universally safer. It is not a hidden estimate or permission to
switch candidates after observation: the source had been opened in prior v6
work, each named fake model has only four examples, and some real sources are
not eligible for submission training. Exact compact evidence is in
`COMMUNITY_FORENSICS_V12_AUDIT_RESULT.json`.

## 2026-09-01 — final source-coherent v12 arbitration selects PE-Core-L

`NTIRE_V12_FINAL_ARBITRATION_PLAN.json` froze a one-shot default decision before
either exact v12 checkpoint was scored on the 1,024-row NTIRE shard-5 gate. The
gate is balanced, all-JPEG, source-coherent, identity-disjoint from all v12
training/evaluation rows and has zero organizer-demo rows. It is evaluation
only: it was previously opened by the rejected v11 lineage, has no reusable
dataset licence and is not a hidden-set estimate.

The first Kaggle attempt produced no recoverable score. The interactive session
ended before uncommitted outputs were preserved and the weekly GPU quota was
then exhausted. `NTIRE_V12_LOCAL_RECOVERY_PLAN.json` froze the unchanged local
recovery before any local score. Its manifest SHA-256 is
`dfd3f196106544d586a3eb32c22f94d213f0ddd0f642f07d5cfc9e1fb08e2bb6`;
the retrieved private checkpoint archive passed its integrity test and both
checkpoint hashes matched the frozen plan.

The first Apple MPS PE process used batch eight and four loader workers. It
aborted before producing any condition output or score with return code 134:
`MPSNDArray buffer is not large enough`, followed by 20 leaked semaphores.
`NTIRE_V12_MPS_BUFFER_RECOVERY.json` froze batch one and zero workers before any
condition score. Gate rows, checkpoints, arithmetic, transformations, codec,
seeds, blend and decision thresholds were unchanged. Both exact candidates
then completed all 20 conditions with return code zero.

PE-Core-L scored **0.9906692505 clean AUC**, **0.9836782471 pooled transformed
AUC**, **0.9871737488 workshop 50/50 score** and **0.9308052063 worst condition**.
DINOv2-L scored 0.7303009033, 0.7202312121, 0.7252660577 and 0.6664962769. The
fixed, no-search equal-probability blend scored 0.9429893494, 0.9294399885,
0.9362146689 and 0.8483848572.

All five precommitted checks passed. PE improved over the blend by 0.0509590799
on the workshop score, 0.0476799011 on clean AUC and 0.0824203491 on the weakest
condition. The selected runtime default is therefore **PE-Core-L**. No weight,
threshold, preprocessing or calibration search is permitted on the consumed
gate. Exact plan hashes, prediction hashes, metrics and decision arithmetic are
in `NTIRE_V12_FINAL_ARBITRATION_RESULT.json`.

## 2026-09-01 — final selected runtime and error contract

The runtime default was changed from the rejected equal blend to `pe_core`.
Default batch size is one for Apple MPS safety, the DINO checkpoint is no longer
required for ordinary execution, and input decoding now applies EXIF orientation
before the frozen JPEG/crop preprocessing. The complete suite then returned
zero with **205 passed** and one non-failing physical-core discovery warning.

The exact public-facing command was executed without an `AIGC_V12_MODE`
override:

```bash
PYTHON_EXECUTABLE=.venv/bin/python AIGC_DEVICE=mps AIGC_BATCH_SIZE=1 \
  ./run_v12.sh artifacts/demo-rehearsal-input \
  /private/tmp/track5-v12-selected-default-mps.json
```

It returned zero, reported `mode=pe_core`, 315,776,001 parameters and 2.533843167
seconds of detector elapsed time. The 464-byte output has SHA-256
`767096b0c1ffb963fe12947e1038f1f5b1416521aaa42d5c637532ae09419157`
and exactly reproduces the earlier four frozen scores. This is portability and
default-selection evidence, not accuracy, throughput or hidden-set evidence.

Post-selection error analysis verified prediction indices, labels and image
hashes before summarizing the clean final gate. At an illustrative, uncalibrated
0.5 threshold, 161/512 authentic rows are false positives and 4/512 generated
rows are false negatives despite 0.990669 AUC. The strongest authentic false
positive is a surreal human-made animal collage; the weakest generated examples
look like ordinary travel/event photos. Exact rows and claim boundaries are in
`V12_ERROR_ANALYSIS_RESULT.json`.

## 2026-09-01 — final history-free source export passes isolated verification

The final source export was refreshed after adding the verified judge demo.
Commit `8d60257c3ede7d3ca3604e237ccbc8492cb8c18d` was exported with `git
archive` to the ignored 717,380-byte file
`artifacts/source/track5-source-8d60257.zip`. Its SHA-256 is
`fce1be36b11ecbea4a4dcf706244be0f55ede2fd8abbc73955ea706384aace9f`.
The archive contains 298 files and no Git history.

A fresh extraction outside the repository returned zero for all **210 tests**
with one non-failing physical-core warning. A separate pristine extraction was
audited before running Python. It found all required files, executable runners,
the selected 315,776,001-parameter PE mode, zero forbidden/oversized artifacts,
zero private locators and zero private-key material. The auditor returned one
by design because the three public-link placeholders and public checkpoint URL
remain unresolved. `PUBLIC_SOURCE_BUNDLE_RESULT.json` preserves the exact
commands, digest, counts, results and publication boundary.

The extracted source was then executed, rather than merely unit-tested, using
the exact local PE checkpoint and four frozen rehearsal images. The MPS command
returned zero, loaded `pe_core`, reported 315,776,001 parameters and completed
model work in 2.425809917 seconds. Its 812-byte JSON differs from the checkout
output only because the `image_path` strings are absolute external paths; all
four probabilities and filename order match exactly. The tracked evidence
intentionally redacts the private absolute locators and records the portable
command template in `CLEAN_EXPORT_RUNTIME_RESULT.json`. This closes local
export-to-local-model execution, not public or logged-out installation.

A brand-new temporary Python 3.9.6 environment then installed only
`requirements-runtime.txt`. The install returned zero in 18.0449 seconds and
resolved Pillow 11.3.0, timm 1.0.19, torch 2.8.0 and torchvision 0.23.0. The
same extracted runner and selected checkpoint returned zero on MPS in 2.9349
seconds and exactly reproduced all four probabilities. This closes fresh local
runtime installation; it still does not test an unauthenticated public model
download.

## 2026-09-01 — final live checkpoint and MPS rehearsal

The exact selected checkpoint passed `shasum -a 256 -c
SELECTED_CHECKPOINT.sha256`. A first MPS command in the managed shell returned
one at `model.to(device)` because that shell denied the Metal backend; inference
did not start. The unchanged approved device run then returned zero on Apple
MPS in 3.02065625 seconds, loaded only `pe_core`, reported 315,776,001
parameters and reproduced the frozen 464-byte JSON byte-for-byte with SHA-256
`767096b0c1ffb963fe12947e1038f1f5b1416521aaa42d5c637532ae09419157`.
All four probabilities exactly match the earlier default run. The failure and
success are both preserved in `FINAL_LIVE_REHEARSAL_RESULT.json`; this is
run-contract evidence, not a new evaluation result.
