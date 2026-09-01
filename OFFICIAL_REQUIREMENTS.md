# Official Track 5 requirements inventory

Observed on 26 August 2026, reconciled against the public statement on
27 August, and updated from the organizer workshop and Q&A supplied by the user
on 28 August 2026 (SGT). The user confirmed that the workshop is controlling
for these track details. No password, credential or private email content is
stored here.

## Sources and authority

- Public Track 5 statement: <https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc>
- Early Bird problem statements: <https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf>
- Public event information: <https://bit.ly/TikTokTechJam2026Info>
- Official Devpost page: <https://tiktoktechjam2026.devpost.com/>
- Official rules: <https://tiktoktechjam2026.devpost.com/rules>

The public copy is now the primary track-specific statement used by this
inventory. No material change to the recorded Track 5 technical constraints was
identified during the 27 August reconciliation.

## Problem boundary and constraints

The task is image-only detection of authentic versus AI-generated images after
realistic redistribution transformations. Production moderation systems and
video/audio are out of scope.

- The submitted model must contain **fewer than 2,000,000,000 parameters**.
- Public or properly licensed datasets are allowed.
- The workshop Q&A further stated that a dataset marked non-commercial cannot
  be used. Because the user designated workshop statements as controlling, the
  stricter rule governs candidate eligibility even though the earlier phrase
  “public or licensed” is broader.
- Transformed test samples may be generated.
- The organizer does not name an official baseline. No baseline is selected or
  claimed in this repository.

The organizer lists these tested transformation families:

- JPEG quality: 90, 70, 50 and 30.
- Gaussian blur sigma: 0.5, 1.0 and 2.0.
- Downscale to 0.5x or 0.25x, then upscale.
- Gaussian noise sigma: 0.02, 0.05 and 0.10.
- Brightness, contrast and saturation jitter: plus or minus 20%.
- 80% centre crop.

The workshop clarified that evaluation transformations are applied
**individually, not chained**, and that the evaluator may use a subset of the
listed settings. The hidden test includes older Stable Diffusion material and
recent diffusion-transformer generators, including unseen generator families.

These values are preserved as requirements only. No transformation,
augmentation or detector implementation is included before the challenge.

## Data restrictions

Suggested datasets are SID_Set, CIFAKE and WildFake. Their immutable references,
licences, byte counts and checksums are recorded in
`resources/resource_manifest.json`.

Any training source carrying a non-commercial restriction is excluded from the
submission candidate. It may remain in historical experimental evidence only
if clearly marked ineligible. Public availability alone is insufficient.

The demonstration-only validation subset is:

- 4,998 authentic COCO `val2017` images.
- 8,843 WildFake DALL-E Advanced images (the pinned `dalle3.csv` contains exactly
  8,843 data rows with `IsAdvanced=1`).

**This demonstration subset must not be used for training, tuning, model
selection or calibration and does not count toward the final score.** The
public COCO archive has 5,000 images; the organizer has not supplied the exact
two-image exclusion manifest, so this repository does not invent one.

## Track-specific deliverables

- A script accepting an image directory.
- JSON output containing `image_path` and `pred` confidence for every image.
- A clean-versus-transformed robustness table or visual.
- False-positive and false-negative error analysis.
- A public repository and a public **two-to-four-minute** YouTube demonstration.

General submission artifacts are a written Devpost solution/stack description,
a structured public repository with a comprehensive README, and a public
two-to-four-minute YouTube demonstration of the working solution end to end.
Materials must be in English and the build or test path must be reproducible.

The workshop clarified that the output is a continuous confidence with
AI-generated as the positive class; a fixed classification threshold is not
part of technical scoring. The technical metric is:

```text
0.50 * ROC_AUC(clean images) + 0.50 * ROC_AUC(pooled transformed images)
```

The evaluated inputs are pure authentic images and fully AI-generated images;
AI-edited/mixed images and localization are outside the stated test scope.

Only the input directory and per-image JSON keys are specified. The statement
does not define JSON container shape, traversal/order, supported extensions,
score range/direction, duplicate-path handling or failure semantics. Those items
remain unresolved rather than being treated as official requirements.

## Judging criteria

The Track 5 statement gives these weights:

- Technical execution: 35%.
- Innovation and problem insight: 20%.
- Impact and relevance: 20%.
- Feasibility and practicality: 15%.
- Final-event presentation and communication: 10%.

The current Devpost general rules instead say Stage Two uses four equally
weighted criteria. The user confirmed that the workshop's track-specific
weights above are controlling, so they are used here rather than averaged with
the general Devpost wording.
