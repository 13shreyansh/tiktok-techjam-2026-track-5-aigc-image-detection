# Track 5 — Robust AIGC Image Detection

Private preparation repository for the TikTok TechJam 2026 robust
AI-generated-image detection track.

## Preparation scope

- Preserve official dataset references and evaluation constraints.
- Prepare an isolated Python environment and verify dataset access paths.
- Inventory any organizer-provided reference or demonstration procedure.
- Record transforms, expected output schema, resource sizes, licences, and
  blockers.

The organizer does not currently specify an official baseline model. No judged
detector, training pipeline, robustness method, or model selection is
implemented in this preparation snapshot. Substantive competition
implementation begins no earlier than **2026-08-29 12:00 SGT**.

## Preparation inventory

- [Official requirements and deliverables](OFFICIAL_REQUIREMENTS.md)
- [Dataset acquisition and provenance](DATASETS.md)
- [Environment and observed commands](ENVIRONMENT.md)
- [Third-party data notices](THIRD_PARTY_NOTICES.md)
- [Machine-readable resource lock](resources/resource_manifest.json)

The organizer suggests SID_Set, CIFAKE and WildFake, and does not specify an
official baseline. CIFAKE and the official COCO `val2017` source archive are
downloaded under the ignored `datasets/` tree and verified. SID_Set (about
140 GB), WildFake (about 1.29 TB) and the 25.6 GB WildFake DALL-E archive are
represented by pinned acquisition records instead of committed assets.

The organizer's 4,998 COCO plus 8,843 DALL-E Advanced validation subset is
**demo-only, forbidden for training, and excluded from final scoring**. The
exact two-image COCO exclusion list has not been published and remains a
blocker.
