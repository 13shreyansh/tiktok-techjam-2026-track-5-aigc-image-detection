# Trained-weight release decision

Observed on 31 August 2026 SGT. This is an engineering/provenance decision
record, not legal advice. It does not publish a model, change repository
visibility or make a submission.

## Conclusion

Do **not submit or publish the current v6/v9 checkpoints.** The exact workshop
Q&A states that datasets marked non-commercial cannot be used, and the user
designated workshop statements as controlling. V6/v9 include three such real-
image sources. The provisional CC BY-NC-SA release route below is therefore
superseded for this hackathon; it is preserved only to document why a licence
change cannot repair an ineligible training lineage.

Do **not** label the current v6/v9 checkpoints MIT, Apache-2.0 or OSI-open-source.
Their training lineage includes non-commercial and share-alike image sources,
and the team cannot grant downstream rights it may not hold. If the current
stronger checkpoints are released, the conservative provisional route is:

- make the exact files publicly downloadable with their recorded SHA-256
  values;
- apply CC BY-NC-SA 4.0 only to rights the team actually holds in the trained
  weights;
- state that underlying data, pretrained components, names and third-party
  rights remain governed by `THIRD_PARTY_NOTICES.md`;
- describe them as **publicly released research weights**, not OSI-open-source
  weights; and
- retain the non-commercial, attribution and share-alike conditions.

This is the least-permissive practical release path for the current lineage.
It does not resolve whether the workshop's phrase “open-source model weights”
requires an OSI-compatible licence. If literal OSI compliance is mandatory,
the current weights should remain blocked and a new candidate must be trained
only from sources whose terms permit commercial redistribution.

## Confirmed evidence

1. The exact v6 checkpoint is 631,645,967 bytes and the exact v9 checkpoint is
   1,263,202,267 bytes. GitHub documents a limit of **under 2 GiB per release
   asset**, up to 1,000 assets, with no total release-size limit. Both files fit
   individually: <https://docs.github.com/en/enterprise-cloud@latest/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas>.
2. The training lineage includes AFHQ-v2 under CC BY-NC 4.0 and FFHQ under
   CC BY-NC-SA 4.0, plus sources with non-commercial-research or asset-specific
   terms. Exact provenance is in `THIRD_PARTY_NOTICES.md` and
   `MODEL_RELEASE_READINESS.md`.
3. Creative Commons' 2025 conservative AI-training guidance says that when
   training material is NC, model use and distribution should remain
   noncommercial; for ShareAlike material, a publicly shared model should use
   the same CC licence under that conservative approach:
   <https://creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf>.
4. CC BY-NC-SA 4.0 requires attribution, noncommercial use and share-alike, and
   grants only rights the licensor has authority to grant:
   <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>.
5. The Open Source Initiative definition requires free redistribution and no
   restriction on fields of endeavour, including business use. A
   non-commercial licence is therefore not OSI open source:
   <https://opensource.org/osd>.

## Derived decision matrix

| Path | Technical score | Licence defensibility | Workshop wording risk | Deadline feasibility |
|---|---|---|---|---|
| Current v6/v9, public under conservative CC BY-NC-SA terms | strongest verified candidate | best available for current mixed lineage, but still subject to third-party rights | medium: public weights, but not OSI open source | high |
| Current v6/v9 under MIT/Apache | strongest verified candidate | unacceptable without additional rights analysis | low wording risk, high legal/provenance risk | high but rejected |
| New permissive-lineage candidate | unknown and likely lower without new validation | potentially supports an OSI licence | lowest if all sources are truly permissive | low within remaining challenge time |
| Code public, weights private | code remains reproducible only in part | avoids weight redistribution | high: conflicts with workshop request for released weights | high but not recommended |

## Exact release mechanics if the provisional path is approved

1. Export the final reviewed source commit without Git history and rerun all
   tests and the source-tree audit.
2. Create a GitHub release and upload `model.pt`, `model_v9.pt`,
   `CHECKPOINTS.sha256`, the ensemble manifest, this decision record and the
   complete third-party notices as separate assets.
3. Download both public assets in a fresh environment and require their hashes
   to equal:
   - v6: `48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644`
   - v9: `dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075`
4. Insert the immutable public URLs into `ENSEMBLE_CHECKPOINT_MANIFEST.json`,
   rerun the lineage audit and test both downloads from a logged-out session.
5. Use wording that separates MIT-licensed software from conservatively
   licensed trained weights and third-party materials.

No step above is authorized or completed by this document.

## Remaining unknowns

- Whether the organizer uses “open source” colloquially to mean public code,
  pipeline and weights, or requires an OSI-compatible weight licence.
- Whether any jurisdiction treats these discriminative weights as derivative
  of particular training images; Creative Commons expressly describes the
  issue as fact- and jurisdiction-dependent.
- Whether every upstream asset in aggregated datasets permits the intended
  public trained-weight distribution.

These unknowns are why the provisional path remains an action-time decision,
not a completed release gate.
