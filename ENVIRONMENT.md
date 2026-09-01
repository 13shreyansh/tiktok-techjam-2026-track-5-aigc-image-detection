# Preparation environment evidence

Observed 26 August 2026 at 19:51 SGT.

## Host

```text
macOS 26.6.2 (Build 25G83)
Darwin 25.6.0 arm64
Apple M5 Pro
68,719,476,736 bytes physical memory (64 GiB)
20-core integrated Apple GPU, Metal 4
```

## Neutral tooling

```text
Python 3.9.6
/Library/Developer/CommandLineTools/usr/bin/python3
git 2.50.1 (Apple Git-155)
curl 8.7.1
Info-ZIP unzip 6.00
```

Preparation began with standard-library-only utilities. After the challenge
opened, the ignored `.venv` was populated from the pinned requirement files and
the experiment ledger became the authority for installed ML packages, models,
hardware observations and results.

## Input/output boundary

The official input boundary is a directory of images. The read-only utility
`scripts/verify_input_directory.py` verifies that a path is a readable directory
and inventories common image suffixes. It performs no decoding, transforms or
inference.

The official output requirement is JSON with `image_path` and continuous
AI-positive `pred` confidence for every image. `run.sh` now accepts an image
directory, output JSON path and optional checkpoint, recursively processes
supported image files in sorted order, and writes a JSON array of those keys.
The organizer still has not fixed the outer JSON container or error semantics,
so the chosen array representation is documented rather than described as an
organizer mandate.

## Commands executed and observed results

```bash
curl -fL -o datasets/source_archives/val2017.zip \
  http://images.cocodataset.org/zips/val2017.zip
# completed: 815585330 bytes

shasum -a 256 datasets/source_archives/val2017.zip
# 4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05

unzip -tq datasets/source_archives/val2017.zip
# No errors detected in compressed data

curl -fL -o datasets/source_archives/cifake-2023-03-28.zip \
  https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images
# completed: 109625224 bytes

unzip -tq datasets/source_archives/cifake-2023-03-28.zip
# No errors detected in compressed data
```

Peak preparation disk use, including the extracted COCO source directory, is
approximately 1.9 GiB. No GPU work, training or baseline inference was performed. There is no
organizer baseline to reproduce.

## Closing verification on 27 August 2026

```bash
python3 scripts/verify_inventory.py --help
# The script does not implement --help; it ran its full read-only verification.
# Result: all three archives and dalle3.csv verified; CIFAKE=120000 images,
# COCO=5000 source images with licence records, DALL-E Advanced=8843 rows.

python3 scripts/verify_remote_manifests.py --help
# The script does not implement --help; it ran its full read-only verification.
# Result: SID_Set=286 files/140056468470 bytes;
# WildFake=88 files/1287902462734 bytes.

python3 scripts/verify_input_directory.py \
  datasets/demo_only_DO_NOT_TRAIN/coco_val2017_source/val2017 \
  --require-images
# Result: supported_image_files=5000;
# first_relative_path=000000000139.jpg;
# last_relative_path=000000581781.jpg.

python3 scripts/acquire_resources.py list
# Result: seven resources listed with downloaded/verified or manifest-only state.

python3 scripts/acquire_resources.py verify cifake_archive
python3 scripts/acquire_resources.py verify coco_val2017_archive
python3 scripts/acquire_resources.py verify coco_trainval2017_annotations
python3 scripts/acquire_resources.py verify wildfake_dalle3_index
# Result: all four local resources verified.
```

One initial input-boundary command used the nonexistent path
`datasets/coco/val2017` and correctly failed with “not a readable directory.”
Three initial acquisition checks used resource names after the action in the
wrong positional order and correctly printed usage errors; the corrected
commands above succeeded. These failures did not change files.

`git check-ignore -v` confirmed the three source archives are covered by
`.gitignore` rule `datasets/`. The ignored dataset tree remains approximately
1.9 GiB. No secret, model, weight, cache or generated output was added.

## Active challenge environment on 29 August 2026

After the official 12:00 SGT start, an ignored `.venv/` was created with the
exact direct dependencies in `requirements.txt` and `requirements-dev.txt`.
Observed core versions were Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0,
timm 1.0.19, NumPy 1.26.4 and scikit-learn 1.6.1. PyTorch reported both
`torch.backends.mps.is_built()` and `is_available()` as true.

The first completed MPS checks exercised training, checkpoint loading, all
individual organizer transformations, ROC-AUC scoring and directory-to-JSON
inference. Exact commands, scores, elapsed time and MPS memory observations are
in `EXPERIMENT_LEDGER.md`; ignored checkpoints and outputs are not evidence by
themselves and are not committed.
