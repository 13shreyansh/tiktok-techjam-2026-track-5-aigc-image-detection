# Compute options checked at challenge start

Checked on 29 August 2026 after the official start. Account creation, phone
verification, payment details and acceptance of provider terms remain user
actions; no credential is stored in this repository.

## Recommended order

1. **Kaggle Notebooks:** best first free NVIDIA route. Kaggle's current official
   documentation states that free notebooks can use NVIDIA Tesla P100 GPUs and
   that the weekly quota is normally 30 hours, sometimes higher with demand and
   availability. Select `Accelerator > GPU` in notebook settings.
   <https://www.kaggle.com/docs/efficient-gpu-usage>
2. **Lightning AI:** useful second route if account verification is immediate.
   Official documentation advertises 15 monthly credits usable for GPUs;
   phone verification and plan-specific conditions apply. Lightning explicitly
   says only the first account is eligible for those free credits, and personal
   email verification can take two to three business days. The current pricing
   examples are not guaranteed immediate capacity.
   <https://lightning.ai/docs/platform/overview/faq/create-account>
3. **Google Colab free:** fallback. Google's official FAQ confirms free GPU/TPU
   access but explicitly says resources, GPU types and limits fluctuate and are
   not guaranteed. Free notebooks can run for at most 12 hours depending on
   availability and usage.
   <https://research.google.com/colaboratory/faq.html>

No current authoritative IBM page was found that establishes an immediately
available, general-purpose free GPU notebook allocation comparable to the three
routes above. IBM is therefore not on the critical path.

## Account and provider boundary

Creating or controlling extra accounts to multiply subsidized compute is not a
valid route. Kaggle's active terms say one person may not have, control or
operate more than one active Kaggle account. Colab's official FAQ likewise
prohibits multiple accounts used to work around resource limits, and Lightning
limits free credits to the first account. Genuine registered teammates may run
separate, checksum-identical experiment jobs through their own accounts, but
accounts and credentials are never shared. The practical parallel route is
therefore one Kaggle session plus a separately verified Lightning or Colab
session, local Apple MPS, and any legitimate university/team cloud allocation.

- Kaggle terms: <https://www.kaggle.com/terms>
- Colab FAQ: <https://research.google.com/colaboratory/faq.html>

## Kaggle access check

The signed-in Kaggle notebook exposed both `GPU T4 x2` and `GPU P100` and
reported a weekly allowance near 30 hours. Initial in-app-browser launches
failed before cell execution with Kaggle's generic worker-start error and
`Firebase: Error (auth/internal-error)`. The user's retry in a normal browser
then succeeded and `nvidia-smi` observed one Tesla P100-PCIE-16GB, driver
580.159.04, 16,384 MiB VRAM and CUDA 13.0.

Kaggle's default PyTorch 2.10.0+cu128 reported CUDA available but warned that
its compiled architectures start at `sm_70`; the P100 is `sm_60`. The official
PyTorch CUDA 12.6 build was then installed in the live session with:

```text
%pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

After a kernel restart, PyTorch reported `2.8.0+cu126`, CUDA runtime `12.6`,
an architecture list containing `sm_60`, and the Tesla P100. A seeded
4096-by-4096 CUDA matrix multiplication completed in 0.0412 seconds and
reported 142,737,408 peak allocated bytes. Hardware access and an actual CUDA
operation are therefore both verified. The forced reinstall also produced
dependency warnings for unrelated preinstalled Kaggle packages, so competition
code must use a minimal pinned dependency set rather than assuming the entire
notebook image is internally consistent.

## Local path already verified

The Apple M5 Pro host has 64 GiB unified memory and working PyTorch MPS support.
It is fast enough for data preparation, evaluation, frozen-feature experiments
and smaller fine-tuning runs. NVIDIA remains preferable for mixed precision,
larger batches and faster controlled comparisons.

PE-Core-L was tested more narrowly after candidate evaluation. Two separate
batch-one `run.sh` invocations completed on the 20-core Apple M5 Pro GPU, but a
batch-two forward pass terminated its isolated subprocess with the Metal
assertion `Error: buffer is not large enough. Must be 8192 bytes`. The failed
process reached a 5,725,260,368-byte peak memory footprint. PE-Core is therefore
a verified local batch-one inference fallback, not a practical local training
route on this host; DINOv2-L remains the batched Apple-MPS control.

After the long-lived P100 draft stopped during a v7 clean evaluation, Kaggle
started a fresh `GPU T4 x2` session and reported **20 of 30 weekly GPU hours
remaining**. PyTorch 2.10.0+cu128 detected both devices and identified device 0
as a Tesla T4. The restarted training code intentionally uses only one T4 and
keeps the identical v6/v7 data, split and optimizer settings; the second listed
GPU is not claimed as utilized unless a measured run explicitly does so.

The exact selected v6 checkpoint subsequently completed four-image inference
on both Apple MPS and CPU. On MPS, a separate deterministic 128-image clean
audit completed in 8.18 seconds with 1,263,235,328 current and 2,209,021,952
driver-allocated bytes. These results verify local packaging and low-throughput
inference; they do not change the earlier batch-two Metal failure or make local
PE-Core training the preferred route.
