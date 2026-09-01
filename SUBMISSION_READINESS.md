# Submission readiness

Observed 1 September 2026 SGT. A local file, private Kaggle artifact or passing
command is not a public release or submission.

## Selected artifact — verified locally

- Candidate: PE-Core-L v12, selected by the precommitted one-shot rule in
  `NTIRE_V12_FINAL_ARBITRATION_RESULT.json`.
- Parameters: 315,776,001, below the organizer's exclusive 2B limit.
- Checkpoint: 1,263,202,331 bytes; SHA-256
  `f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2`.
- Default interface: `run_v12.sh INPUT_DIRECTORY OUTPUT.json`; ordered paths and
  continuous AI-positive probabilities; no calibrated threshold.
- Default device: automatic CUDA/MPS/CPU; batch one for MPS safety.
- Apple M5 Pro MPS proof: four frozen rehearsal images, return code zero,
  2.6812 seconds, output SHA-256
  `767096b0c1ffb963fe12947e1038f1f5b1416521aaa42d5c637532ae09419157`.
- Historical one-T4 PE-only proof: four images, return code zero, 11.2456
  seconds. These are run-contract checks, not throughput or accuracy claims.
- Final frozen arbitration: 0.990669 clean AUC, 0.983678 pooled transformed,
  0.987174 workshop score and 0.930805 weakest condition.
- Organizer-demo use for training, tuning, selection, calibration or
  thresholding: zero. Recorded non-commercial training rows: zero.

## Repository deliverables

- [x] Directory-to-JSON runner with checkpoint validation and atomic output.
- [x] Selected-checkpoint checksum and machine-readable manifest.
- [x] Requirements split for runtime and development.
- [x] Model card, robustness table and concrete error analysis.
- [x] Dataset and upstream provenance notices.
- [x] Weight-licence statement that does not license third-party images.
- [x] Exact frozen decision evidence and failed-attempt ledger.
- [x] Full suite rerun after final document/runtime reconciliation: 210 passed,
  one non-failing physical-core discovery warning.
- [x] Final tracked-tree safety audit: 298 files, zero forbidden or
  oversized artifacts, zero private locators and zero key material. It remains
  distribution-blocked only by three public-link placeholders and the missing
  public checkpoint URL.
- [x] Self-contained six-beat judge demo added and checked against the frozen
  result values, accessibility controls and recording command.
- [x] Clean history-free source archive rebuilt from source commit `8d60257`,
  SHA-256 `fce1be36b11ecbea4a4dcf706244be0f55ede2fd8abbc73955ea706384aace9f`.
  Its fresh extraction contains 298 files, passes all 210 tests and has zero
  forbidden artifacts, private locators or private-key findings.
- [x] The extracted source itself ran the exact selected local checkpoint on
  four frozen images through Apple MPS in 2.4258 seconds; all probabilities
  exactly matched the checkout run. This is not a public-download proof.
- [x] A fresh temporary Python environment installed only
  `requirements-runtime.txt` and reran the extracted source plus selected model
  successfully; all four probabilities again matched exactly.

## External release gates

- [ ] Publish the selected checkpoint at one immutable public URL.
- [ ] Redownload it without authentication and verify the exact SHA-256.
- [ ] Publish only a final audited source export or deliberately reviewed
  history; private development history is not automatically release-safe.
- [ ] Install and run from the public source and checkpoint while logged out.
- [ ] Verify the final repository visibility and every link. No visibility
  change has been performed yet.
- [ ] Record a public English two-to-four-minute YouTube demonstration.
- [ ] Replace all three placeholders in `DEVPOST_DRAFT.md`.
- [ ] Conduct an explicit action-time submission review, then submit. No
  Devpost or public-release action has been performed by this task.

Terminal access to the configured GitHub HTTPS remote was checked again on
1 September SGT and failed before remote inspection with `could not read
Username ... Device not configured`. No credential was created, exposed or
embedded. The connected GitHub application then synchronized all 297 audited
source blobs to remote commit `9457cc3`, and readback verified that `main`
points there, all three runners remain executable and repository visibility is
still **private**. `PRIVATE_REMOTE_SYNC_RESULT.json` records the exact tree and
boundary. This is preservation, not the required public release.

## Candidate decision

Submit PE-Core-L v12—not the historical v6/v9 lineages, DINO control or equal
blend. The older models include non-commercial training sources and remain
diagnostic only. DINO and the blend were submission-eligible experiments, but
the final untouched arbitration gate rejected both under the frozen rule. That
gate is now consumed: do not tune weights, preprocessing, thresholds or
calibration from it.
