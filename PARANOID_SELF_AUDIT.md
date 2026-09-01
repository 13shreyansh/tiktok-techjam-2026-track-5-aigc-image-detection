# Paranoid self-audit

Snapshot: 1 September 2026 SGT. This is a failure audit, not a hidden-score
prediction. The organizer's demo-only resources remain completely unused for
training, tuning, selection, calibration and thresholding.

## Bottom line

The final frozen evidence supports PE-Core-L v12 as the strongest eligible
candidate we have, but it does not make hidden performance certain. Confidence
in the experimental process is **91/100**; confidence that the model will be
strong on the unknown hidden mixture is **65/100**. The latter is deliberately
limited because generator families, real-image collections, content mix and
class balance remain unknown. The 0.9872 final development score must not be
read as a hidden-score estimate.

## Current scorecard

| Evidence | PE | DINO | fixed 50/50 blend | What it tells us |
|---|---:|---:|---:|---|
| CIFAKE workshop score | 0.9074 | 0.9066 | **0.9379** | blend helps one low-resolution domain |
| Modern semantic-paired | **0.9962** | 0.8271 | 0.9810 | strong PE result, but collection confounded |
| Open Images source rotation | **0.9759** | 0.6414 | 0.9023 | DINO's hedge collapses after changing real source |
| Community Forensics / 78 names | **0.9487** | 0.8097 | 0.9148 | PE has broader aggregate transfer |
| Final NTIRE clean | **0.9907** | 0.7303 | 0.9430 | untouched one-shot arbitration evidence |
| Final NTIRE transformed | **0.9837** | 0.7202 | 0.9294 | PE survives all individual workshop transforms |
| Final NTIRE weakest condition | **0.9308** | 0.6665 | 0.8484 | severe noise remains weakest but not collapsed |

All five frozen final-decision checks pass for PE. No blend weight, threshold or
preprocessing choice was searched after opening the final gate.

## What is working

1. The selected training lineage records zero demo rows and zero
   non-commercial rows; historical ineligible models are excluded.
2. Training is balanced across labels and source-sampled across three real and
   five fake collections rather than being one animals-versus-fakes split.
3. Label-blind canonicalization reduced metadata-only AUC from 0.9984 to 0.5131
   and literal PNG/square rules to chance.
4. Exact and canonical hashes, raw source identities and prior manifests are
   checked before frozen evaluations.
5. Evidence spans low-resolution matched source, modern generated content,
   independent authentic-source rotation, 78 named model variants and a final
   source-coherent gate.
6. The exact selected model runs through the real directory-to-JSON interface
   on Apple MPS and NVIDIA T4, with checksum validation and atomic output.
7. Simpler PE-only inference removes the slower second model and the severe
   DINO source-dependence observed after real-source rotation.

## What can still completely fail

1. **Unknown generator family.** The hidden set can contain a generator whose
   signal is absent from every public source. Mitigation is broad pretrained
   representation plus many families; no test proves universal novelty.
2. **Unknown authentic source.** Scanners, screenshots, illustrations,
   microscopy, rendered but non-AI graphics or unusual phones may shift real
   scores. Open Images rotation helps, but only one extra source is tested.
3. **Collection fingerprint.** Uniform JPEG removes obvious container clues,
   not camera pipelines, resampling kernels, web scraping or dataset-specific
   texture. Source-coherent and rotated-source gates reduce—not eliminate—this.
4. **Content shortcut.** A generated-content collection may differ in subject,
   composition or aesthetic. Semantic pairing addresses a slice but retains a
   Qwen-versus-COCO collection confound.
5. **Severe transformation.** Gaussian noise sigma 0.10 is repeatedly weakest
   because it destroys fine forensic detail. The final AUC is 0.9308, but a
   new source under noise can be worse.
6. **Threshold disaster.** At an illustrative 0.5 cutoff, 31.4% of final-gate
   reals are false positives. We must present a ranking score, not a calibrated
   binary accusation.
7. **Semantic edge cases.** Human-made surreal art can look synthetic and
   generated documentary-style photos can look authentic. These are observed,
   not hypothetical, final-gate errors.
8. **Small subgroup samples.** Community Forensics has four fakes per named
   model; its per-model minima are alarms, not precise population estimates.
9. **Runtime mismatch.** The final machine may lack compatible PyTorch/GPU
   support or memory. Batch-one MPS and CUDA/CPU fallbacks reduce this, but only
   clean-install rehearsal can close packaging risk.
10. **Release failure.** A private checkpoint, local commit or passing audit is
    not downloadable by judges. Public URL, redownloaded hash, source bundle,
    video and logged-out run remain mandatory external gates.

## What improved from the earlier approach

- We stopped treating very high source-confounded scores as proof.
- We replaced ineligible historical training lineages with a licence-filtered
  balanced mixture.
- We neutralized known shape/format leakage before training.
- We allowed a seemingly helpful ensemble to be falsified rather than keeping
  it for complexity's sake.
- We added one final untouched rule-bound arbitration instead of choosing from
  already consumed gates.
- We changed the runtime default to the selected model, batch one and EXIF-safe
  input handling, so the demonstration path matches the trained contract.

## Next gates before submission

1. Re-run every unit/integrity test and the exact default MPS command.
2. Stage the intended files, then run the tracked-tree safety audit.
3. Build a history-free archive from the final commit and test it outside the
   working tree.
4. Publish only after explicit action-time review; redownload and hash the model
   without credentials.
5. Record the live demo with a real error, not just favorable examples.
