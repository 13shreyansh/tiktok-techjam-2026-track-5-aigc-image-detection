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

The preparation utilities use only the Python standard library. No ML
framework, model, weight, detector or robustness package is installed or chosen.

## Input/output boundary

The official input boundary is a directory of images. The read-only utility
`scripts/verify_input_directory.py` verifies that a path is a readable directory
and inventories common image suffixes. It performs no decoding, transforms or
inference.

The official output requirement is JSON with `image_path` and `pred` confidence
for every image. No output writer is implemented because the organizer has not
specified the JSON container, score direction/range or error semantics, and the
judged detector is deferred until the challenge window.

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
