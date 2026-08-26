# Dataset acquisition and provenance

Machine-readable lock data is in `resources/resource_manifest.json`. Dataset
archives, extracted images and metadata copies live under ignored `datasets/`.

## Acquired and verified locally

### CIFAKE

The public Kaggle archive was downloaded without credentials and verified:

```text
bytes   109625224
MD5     dedf8700a2a671c84c60c1986143548a
SHA256  1839206c1462371f5c885eda1660b448ea7abf52ee81fcc71e5e4651a4a11389
images  120000 (100000 train, 20000 test; 60000 REAL, 60000 FAKE)
```

The author states an MIT licence and requires citation of CIFAR-10 and the
CIFAKE paper. Repository evidence is pinned at commit
`e112a942abaecd02b6b1f6f646c807d56be8fb62`.

### COCO val2017 source archive

The official archive was downloaded and passed `unzip -tq`:

```text
bytes   815585330
MD5     442b8da7639aecaf257c1dceb8ba8c80
SHA256  4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05
images  5000 source JPG files
```

COCO images retain per-image Flickr/COCO licences. This source is demo-only for
Track 5 and must not enter training. The official annotations archive is also
acquired and verified (`252907541` bytes; MD5
`f4bbac642086de4f52a3fdda2de5fa2c`; SHA-256
`113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`).
All 5,000 validation image records carry one of the COCO licence IDs. The exact
organizer subset is blocked until the 4,998-image selection/exclusion manifest
is supplied.

## Pinned manifests, not fully downloaded

### SID_Set

- Hugging Face revision: `dc03ead57929879319ce30a82bfcfb8d317b10bd`
- Public snapshot: 286 files, 140,056,468,470 bytes.
- Canonical path/size/blob inventory SHA-256:
  `bca82b8b1811e30b157082a881327ba5476449987f0e61d175e74f6eadb0257e`.
- Available rows: 210,000 train and 30,000 validation.
- The stated 60,000-image test split is obtained separately through the authors'
  repository and is not part of the 240,000-row Hugging Face snapshot.
- Licence: CC BY 4.0, including attribution obligations described in the card.

The metadata API exposes every shard's blob/LFS identity. A full download was
not started because it is approximately 140 GB.

### WildFake

- Official ModelScope record reports 1,291,478,056,101 stored bytes.
- Recursive public tree on 26 August enumerated 88 files totalling
  1,287,902,462,734 bytes; the discrepancy is preserved, not reconciled away.
- Canonical path/size/SHA-256/revision inventory SHA-256:
  `4d19f1021f0d74f4062ced323665de9bf215d38da76eebcf5494afb3e0be2e82`.
- Licence reported by ModelScope: Apache 2.0.
- Official pointer repository commit:
  `27c7314bfb8016ab233472bb691029aa51e046ec`.

The full dataset was not downloaded because it is approximately 1.29 TB.
ModelScope supplies per-file sizes, revisions and SHA-256 values.

### DALL-E Advanced demo material within WildFake

The pinned `dalle3.csv` has 8,843 rows, all marked `IsAdvanced=1` and
`IsFake=1`. It indexes the organizer's stated 8,843-image DALL-E Advanced demo
subset. The containing `DALLE.zip` is 25,587,709,291 bytes and also contains
non-demo content; its SHA-256 is locked in the manifest. It was not downloaded
during this pass due to its size.

## Reproducible commands

```bash
python3 scripts/acquire_resources.py list
python3 scripts/acquire_resources.py verify cifake_archive
python3 scripts/acquire_resources.py verify coco_val2017_archive
python3 scripts/acquire_resources.py download wildfake_dalle3_index
python3 scripts/verify_inventory.py
python3 scripts/verify_remote_manifests.py
```

Large downloads additionally require `--allow-large-download`. That guard is
deliberate: it makes a multi-gigabyte acquisition explicit without embedding
credentials or placing assets under version control.
