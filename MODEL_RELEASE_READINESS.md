# Model release readiness

Observed 1 September 2026 SGT. This is an evidence inventory, not legal advice
or a publication action.

## Selected model

| Artifact | Bytes | Parameters | SHA-256 | Public URL |
|---|---:|---:|---|---|
| PE-Core-L v12 checkpoint | 1,263,202,331 | 315,776,001 | `f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2` | none |

The checkpoint is present in ignored local storage and a private preservation
location. Both are recovery evidence, not public distribution. The selected
sidecar is `SELECTED_CHECKPOINT.sha256`; machine-readable lineage and runtime
state are in `V12_CHECKPOINT_MANIFEST.json`.

## Licence boundary

- Upstream PE-Core-L/timm code and pretrained backbone: Apache-2.0, preserved
  in `THIRD_PARTY_NOTICES.md`.
- Repository code: MIT, subject to third-party notices.
- Selected trained checkpoint: released under Apache-2.0 only to the extent the
  team owns rights in its trained head and compilation; see
  `MODEL_WEIGHTS_LICENSE.md`.
- That weight licence does not relicense any training/evaluation image or grant
  rights the team does not hold.
- Dataset licences, access terms, competition use and trained-weight
  redistribution are distinct questions. Exact sources and boundaries remain
  in `THIRD_PARTY_NOTICES.md`.

## Training-lineage release checks

| Check | Result |
|---|---|
| organizer demo rows used in training/tuning/selection/calibration | 0 |
| recorded non-commercial training rows | 0 |
| train/evaluation byte overlap | 0 |
| balanced train rows | 13,574: 6,787 real, 6,787 fake |
| parameter limit | 315,776,001 < 2,000,000,000 |
| public immutable checkpoint URL | missing |
| unauthenticated redownload and hash | not performed |

Historical v6/v9 artifacts are not release candidates because their training
lineage includes non-commercial data. DINOv2-L and the equal blend are eligible
experimental controls but were rejected by the frozen final decision. Only PE
is required for the public selected runtime.

## Mechanical release procedure

1. Verify the ignored local model against `SELECTED_CHECKPOINT.sha256`.
2. Publish that exact file under an immutable/versioned URL only as part of
   the authorized release action.
3. Download it into a fresh temporary directory without authenticated state.
4. Recompute SHA-256 and require the exact selected digest.
5. Build the history-free source archive from the final reviewed commit.
6. Scan the archive for datasets, weights, caches, outputs, secrets and private
   locators; run the complete tests after extraction.
7. Install runtime requirements and execute `run_v12.sh` outside the working
   tree using only public inputs.
8. Record URL, retrieved bytes, hash, command, environment, elapsed time and
   result in the release audit.

## Current release blockers

The final history-free source bundle is complete: source commit `8d60257`,
717,380 bytes, SHA-256
`fce1be36b11ecbea4a4dcf706244be0f55ede2fd8abbc73955ea706384aace9f`;
its pristine 298-file tree passed the safety audit and all 210 tests after a
separate extraction.

1. No public immutable checkpoint URL.
2. No logged-out public installation/run has been verified.
3. Public repository and demo-video URLs remain unset.

Until all three close, the model is selected and reproducible locally but not
publicly distribution-ready.
