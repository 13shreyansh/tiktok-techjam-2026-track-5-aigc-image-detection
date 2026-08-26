# Preparation status

## Allowed preparation

- [x] Index authorized public dataset sources and licences
- [x] Record dataset sizes and available upstream checksums before large downloads
- [x] Download and verify the manageable CIFAKE and COCO source archives locally
- [x] Validate the neutral environment and image-directory input boundary
- [x] Preserve the official transform, output and demonstration requirements
- [x] Record that no organizer baseline is currently specified

## Ready

- Immutable SID_Set Hugging Face revision and 140,056,468,470-byte snapshot metadata
- CIFAKE archive verified at 120,000 images with MD5 and SHA-256 recorded
- WildFake repository inventory plus per-file checksum acquisition route
- COCO `val2017` source archive verified at 5,000 images, with the official
  per-image licence metadata archive also verified
- WildFake DALL-E Advanced index identified as exactly 8,843 rows
- Guarded, checksum-verifying acquisition utility with no secrets
- Exact image-directory input and per-image JSON key contract recorded

## Blockers and intentionally incomplete items

- The organizer provides no official baseline; none was reproduced or selected.
- The public COCO archive has 5,000 images, but the organizer's two exclusions
  for the 4,998-image demo subset are not published.
- The 25,587,709,291-byte WildFake DALL-E archive, approximately 140 GB SID_Set
  snapshot and approximately 1.29 TB WildFake repository were not fully
  downloaded due to their resource size; checksummed acquisition records are ready.
- The public problem-statement copy is scheduled for 27 August at 12:00 SGT and
  must be rechecked against the Early Bird version.
- JSON container shape, ordering, confidence direction/range and failure
  semantics are not specified by the organizer.

## Deferred until the challenge window

- Detector selection or training
- Robustness augmentation strategy
- Competition inference and error-analysis implementation
