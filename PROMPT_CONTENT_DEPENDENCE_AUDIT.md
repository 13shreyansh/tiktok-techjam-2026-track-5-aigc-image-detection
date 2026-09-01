# Prompt/content dependence audit

Observed 2026-08-31. This is a frozen-score diagnosis, not a training or model
selection result.

## Result

The selected PE-Core v6 checkpoint
`48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644`
was previously scored on a complete balanced grid of 288 Qwen Image Bench
fakes: 16 repeated prompt IDs across 18 current generators. The same frozen
file also contains 288 reals. Both labels were converted to JPEG quality 96 and
stretched to the same input geometry before inference.

An additive variance diagnosis of the 288 fake probabilities found:

- prompt identity: **52.7494%** of score variance;
- generator identity: **14.7481%**;
- remaining prompt-generator interaction: **32.5025%**.

This means image content changes the selected model's confidence more than the
choice among these 18 generators. It does not prove that the detector uses a
semantic shortcut: some prompts may simply cause every generator to make more
photorealistic images. It does disprove the comforting assumption that
generator breadth alone solves transfer.

The weakest repeated prompt groups were:

| Prompt ID | Fake images | Mean AI score | AUC against all 288 reals | Illustrative fraction above 0.5 |
|---:|---:|---:|---:|---:|
| 606 | 18 | 0.330218 | **0.790702** | 0.2222 |
| 990 | 18 | 0.403275 | **0.835841** | 0.3333 |
| 803 | 18 | 0.557357 | 0.917631 | 0.5556 |
| 107 | 18 | 0.624388 | 0.929591 | 0.7778 |
| 554 | 18 | 0.706781 | 0.951775 | 0.7222 |

The 0.5 fractions are not calibrated and are included only to make the score
shift intuitive. Prompt-specific AUC is the relevant threshold-free result.

## What the weak groups contain

The pinned benchmark metadata file at
`https://huggingface.co/datasets/Qwen/Qwen-Image-Bench/blob/d2493deb153b020cf169c7e3f57d15e4dd697038/metadata/bench_metadata.json`
has SHA-256
`8ba913d292edd791a3abd19ce9d60fc7322a2d2a22b1c4b8b763f62f2d64c618`.
It exposes evaluation dimensions, not the original prompt text. Its dimensions
describe prompt 606 as a real-world/cultural scene with lighting, composition,
3-D layout and edge detail; prompt 990 as graphic design, text layout,
composition, 2-D layout and resolution.

Direct visual inspection of three generators per weak prompt provides a
bounded observation, not recovered prompt wording:

- ID 606 consistently depicts a highly photorealistic ornate green-and-white
  palace facade in snow.
- ID 990 consistently depicts a polished beachwear advertisement with a woman
  in a light-blue jacket and Chinese promotional text.

These are exactly the kinds of synthetic images that can resemble ordinary
architecture photography, fashion advertising or edited social content.

## Independent unseen-prompt confirmation

A second checksum-frozen Qwen partition used 16 different prompt IDs across
the same 18 generators, for 288 fake and 288 real images. It had zero prompt-ID
or pixel-hash overlap with the earlier audit or v9 training partition. It was
sealed before the selected-v6 baseline was scored. Both labels again received
JPEG quality 96 and full-frame stretch preprocessing.

On Apple MPS, v6 scored 0.932653 clean AUC. The exact one-T4 FP16 arithmetic
contract subsequently scored v6 at 0.932412; the small difference is an
explicit device/arithmetic difference, not a hidden-set improvement claim.
Under that exact CUDA contract, the preselected 75/25 v6/v9 blend scored
**0.954638**. It improved worst-generator AUC from 0.850260 to 0.895399,
worst generator-by-real-source AUC from 0.641164 to 0.722522, and weakest
prompt AUC from 0.704186 to 0.807581.

The same warning survived the improvement. For v6, prompt identity explained
61.4365% of fake-score variance, compared with 10.8625% for generator identity.
For the blend those shares were 58.5940% and 11.6882%. The two weakest prompt
groups were IDs 900 and 721. Bounded visual inspection showed a realistic
rustic shop/counter scene with Chinese or Taiwanese signs and a Wi-Fi notice,
and a realistic sunset/woman social-media motivational post with Chinese text.
The blend is therefore better on a genuinely separate prompt set, but it has
not removed content sensitivity. The full private Kaggle report is preserved
by a 7,401-byte SHA-256 record in
`QWEN_UNSEEN_PROMPT_ENSEMBLE_RESULT.json`.

## Decision boundary

These 288 fake images remain excluded from training. Now that their prompt
groups have been inspected, this gate is no longer an untouched source for
choosing a content-specific training intervention. A legitimate follow-up must:

1. use different public/licensed images with no pixel or prompt-ID overlap;
2. balance the same subjects across real and fake labels so subject alone
   cannot solve the task;
3. freeze a new set of unseen Qwen prompt IDs before training (the second set
   above is now consumed and cannot serve this role again);
4. require improvement on the new prompt holdout without regression on NTIRE,
   Community Forensics, source-pair and transformation gates.

The correct conclusion is not "train on snowy palaces." It is that a robust
detector needs content-matched real/fake evidence, especially for realistic
architecture, people, advertisements and text-heavy compositions.
