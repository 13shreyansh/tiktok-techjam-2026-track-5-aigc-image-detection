# Robustness and error analysis

## Final selected result

The selected PE-Core-L v12 checkpoint scored 0.990669 clean AUC and 0.983678
pooled transformed AUC on the final 1,024-image source-coherent gate. The
organizer-style score is 0.987174. Gaussian noise sigma 0.10 is weakest at
0.930805 AUC.

The equal PE/DINO blend was not retained. Its organizer-style score was
0.936215 and its worst condition 0.848385. DINO alone scored 0.725266 and
0.666496 respectively. The candidate switch follows five thresholds frozen
before either v12 score was opened; see
`NTIRE_V12_FINAL_ARBITRATION_RESULT.json`.

## Individual final-gate conditions

| Condition | AUC | Condition | AUC |
|---|---:|---|---:|
| clean | 0.9907 | JPEG q30 | 0.9841 |
| JPEG q90 | 0.9919 | blur sigma 2 | 0.9833 |
| JPEG q70 | 0.9869 | resize 0.25 | 0.9837 |
| JPEG q50 | 0.9817 | noise sigma 0.02 | 0.9842 |
| blur sigma 0.5 | 0.9924 | noise sigma 0.05 | 0.9616 |
| blur sigma 1 | 0.9935 | **noise sigma 0.10** | **0.9308** |
| resize 0.5 | 0.9922 | centre crop 80% | 0.9891 |

Brightness, contrast and saturation at both workshop severities range from
0.9815 to 0.9901. Transformations were applied individually, never chained.

## Cross-source evidence

The final result is not supported by a single split. PE also records:

- 0.9074 organizer-style on 2,000 matched-source CIFAKE rows;
- 0.9759 after replacing COCO reals with Open Images while keeping the same
  modern fake pool;
- 0.9487 on 593 identity-disjoint Community Forensics rows spanning 78 named
  model variants;
- 0.9962 on semantic-paired modern images, with an explicit Qwen-versus-COCO
  collection-confound warning.

These gates attack different shortcuts, but none reproduces the hidden source
mixture. The SID result is omitted from promotion evidence because its original
format and geometry were label-confounded.

## Concrete false positives and false negatives

`V12_ERROR_ANALYSIS_RESULT.json` verifies prediction indices, labels and image
SHA-256 values before producing this summary. At the illustrative threshold
0.5:

- 161 of 512 authentic images are above threshold;
- 4 of 512 generated images are below threshold;
- clean ROC AUC remains 0.990669.

This is not a contradiction. AUC measures ranking across all thresholds; 0.5
was never calibrated. The score distribution is shifted toward AI, so a fixed
0.5 policy would falsely accuse too many authentic images.

Manual visual review of immutable top errors found:

- highest authentic score 0.9752: a deliberately surreal, human-made animal
  collage whose polished impossible composition resembles generation;
- authentic score 0.9505: a close-up phone photograph with sparse texture and
  unusual framing;
- lowest AI scores 0.3511 and 0.3672: convincing travel/event-style scenes that
  look like ordinary camera photographs.

The practical lesson is that the remaining errors are semantic and
photographic, not simply “AI has bad hands.” A human-made artwork can look
synthetic, while a generated bus or stadium snapshot can look mundane.

## Shortcut controls

The raw v12 data was trivially separable by file metadata: 0.998360 evaluation
AUC without pixels. Label-blind EXIF, crop, resize and JPEG canonicalization
reduced it to 0.513093; square-only and PNG-only rules became 0.5. The final
runtime repeats the same JPEG/crop contract and now honors EXIF orientation.

Other controls include raw and canonical SHA-256 deduplication, identity
comparison against training and frozen evaluation rows, generator and real-
source grouping, content/prompt pairing, source rotation and a veto on
source-confounded gates. These controls make simple leakage less likely; they
cannot prove the absence of all dataset fingerprints.

## Open limitations

1. The hidden generator, content and real-source proportions are unknown.
2. No threshold is calibrated; the prohibited demo resources remain untouched.
3. Severe noise still removes useful fine detail.
4. Human-made art and sparse or heavily processed real images may be false
   positives.
5. Very realistic generated photographs may be false negatives.
6. Community Forensics has only four fake images per named variant.
7. NTIRE generator identities are undisclosed, so exact-family novelty cannot
   be claimed.
8. A public immutable checkpoint URL and logged-out installation proof remain
   external release gates.
