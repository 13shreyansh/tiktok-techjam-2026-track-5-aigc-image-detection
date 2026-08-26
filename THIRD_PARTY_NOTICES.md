# Third-party data notices

This preparation repository does not redistribute dataset contents. Local
copies remain ignored. Before any later use or public release, review the source
terms again and preserve required attribution.

## SID_Set

- Publisher: SIDA authors (`saberzl/SID_Set`)
- Dataset card licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Evidence: <https://huggingface.co/datasets/saberzl/SID_Set>
- Note: the card states that portions incorporate COCO, OpenImages V7 and
  Flickr30k material. Underlying asset terms and attributions still require
  review; the repository does not assume one blanket asset licence.

## CIFAKE

- Publisher: Jordan J. Bird and Ahmad Lotfi
- Stated licence: [MIT](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images/blob/e112a942abaecd02b6b1f6f646c807d56be8fb62/README.md#license)
- Required citations in the pinned README: CIFAR-10 and the CIFAKE paper.

## WildFake

- Publisher record: `hy2628982280/WildFake`
- ModelScope metadata licence: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Evidence: <https://modelscope.cn/datasets/hy2628982280/WildFake/summary>
- Note: WildFake aggregates material from multiple real-image and generator
  sources. Component provenance and restrictions must be audited before
  redistribution even though ModelScope labels the dataset Apache 2.0.

## COCO val2017

- Publisher: COCO Consortium
- Source archive: <http://images.cocodataset.org/zips/val2017.zip>
- COCO terms: <https://cocodataset.org/#termsofuse>
- Image licences: the acquired official `instances_val2017.json` assigns one of
  seven used licence IDs to every validation image (Creative Commons variants
  or no-known-restrictions). Licence ID 8 exists in the table but is not used by
  the 5,000 validation records.

The organizer's 4,998-image COCO subset and 8,843-image DALL-E Advanced subset
are demonstration-only and prohibited from training, independently of their
upstream copyright licences.
