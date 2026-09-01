# Two-to-four-minute demonstration script

Target duration: 2 minutes 45 seconds. Record in English. Use a fresh successful
run and replace bracketed links only after logged-out verification.

## 0:00–0:25 — the result we refused to trust

**Screen:** `demo/index.html#problem`.

**Narration:** “Our first great result was actually an alarm. A classifier that
never opened an image reached 0.9984 AUC using file type and dimensions. If we
had trusted that score, we would have built a dataset detector—not an AI-image
detector.”

## 0:25–0:50 — remove the shortcuts

**Screen:** `demo/index.html#method`.

**Narration:** “So we processed both labels identically: camera orientation,
crop, resize and JPEG encoding. Metadata-only AUC fell to 0.5131; PNG-only and
square-only rules fell to chance; train and evaluation byte overlap was zero.
We deliberately destroyed the easy path before improving the model.”

## 0:50–1:20 — broaden the world

**Screen:** `demo/index.html#broaden`.

**Narration:** “Then we broadened what the model had to survive: multiple real
sources and subjects; GANs, Stable Diffusion, SD3, FLUX and PixArt; and every
workshop transform applied one at a time. We held out complete sources,
generators, prompts and identities instead of trusting one random split.”

## 1:20–1:50 — show what failed

**Screen:** `demo/index.html#failures`.

**Narration:** “Most attractive improvements failed our own checks. A
low-resolution repair hurt clean behaviour. A noise router improved too little.
DINO weakened under a real-source rotation, and the two-model blend eventually
lost to one PE model. We kept the strongest survivor, not the fanciest idea.”

## 1:50–2:15 — lock the final decision

**Screen:** `demo/index.html#decision`.

**Narration:** “Before opening the final 1,024-image gate, we froze candidate
hashes and five pass/fail rules. After opening it, we searched zero weights and
zero thresholds, and organizer demo data was used zero times. That makes the
last result a decision—not another tuning loop.”

## 2:15–2:45 — what survived and the real run

**Screen:** `demo/index.html#run`, then a fresh terminal command and its JSON.

```bash
shasum -a 256 -c SELECTED_CHECKPOINT.sha256
PYTHON_EXECUTABLE=.venv/bin/python AIGC_DEVICE=auto \
  ./run_v12.sh demo-input demo-output.json
```

**Narration:** “What survived is one 315.8-million-parameter PE model. On our
untouched development gate it reached 0.9872 on the workshop score, with 0.9308
in the weakest heavy-noise condition. The command verifies the checkpoint and
returns a continuous AI-evidence score. These are development results, not a
hidden-score prediction. At an uncalibrated 0.5 threshold it still falsely
flags too many real images, so it supports human review rather than proving
deception.”

## Recording gate

- Keep the public video between 2:00 and 4:00.
- Show a fresh successful command and the generated JSON—not a mock.
- Make the rejected approaches part of the story; do not present a benchmark
  slideshow.
- State that 0.5 is not calibrated and show the final limitation.
- Do not expose private paths, accounts, notebooks, datasets or tokens.
- Do not say “hidden score”, “production-ready”, “solved” or “guaranteed”.
- Verify repository, checkpoint and video links from a logged-out browser.
