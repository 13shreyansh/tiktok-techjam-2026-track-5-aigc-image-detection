# Track 5 research journey: only the findings that can help us

This is the beginner-readable version of the Track 5 research. It keeps only
information that does at least one of these things:

- clarifies what TikTok requires;
- explains how AI-image detection works;
- supplies a useful result or comparison;
- exposes a failure that could otherwise mislead us;
- identifies a legal, resource or evaluation constraint; or
- records an unknown that must not be replaced with a guess.

The full audit trail, including every source version, command, checksum and
unsuccessful check, remains in `RESEARCH_LEDGER.md`. No detector was selected,
trained or implemented. The official build period begins on **29 August 2026
at 12:00 SGT**.

## 1. What the challenge actually asks us to produce

TikTok expects a working **software program**, not an essay and not merely a
collection of images. The program must:

- accept a directory containing images;
- decide how likely every image is to be AI-generated;
- output JSON containing an `image_path` and a `pred` confidence for each
  image; and
- use a submitted model containing fewer than **2,000,000,000 parameters**.

The organizer also asks for a clean-versus-transformed robustness table or
visual, false-positive and false-negative analysis, a reproducible public
repository and a public three-minute YouTube demonstration of the working
system. The written submission and demonstration must be in English.

The useful core will probably be a trained or fine-tuned neural network. The
organizer does not require a network to be invented or trained entirely from
scratch, but every model, dataset and software dependency must be legally
usable. The organizer has supplied **no official detector or baseline**.
Therefore, there is no organizer model that teams are simply expected to run.

The task is about still images. Video, audio and a complete production
moderation platform are outside the stated technical boundary.

### The exact transformations named by TikTok

The hidden evaluation may use an organizer-selected subset of:

- JPEG quality 90, 70, 50 or 30;
- Gaussian blur with sigma 0.5, 1.0 or 2.0;
- downscaling to 0.5× or 0.25× followed by upscaling;
- Gaussian noise with sigma 0.02, 0.05 or 0.10;
- brightness, contrast or saturation changes of up to 20% in either direction;
  and
- an 80% centre crop.

These changes matter because social-media images are compressed, resized,
cropped and edited after creation. A detector that works only on pristine
files has not solved the stated problem.

### The named datasets are inputs, not solutions

| Collection | What it is | Why it matters | Important limitation |
|---|---|---|---|
| CIFAKE | 120,000 tiny real and generated images | Manageable source for basic experiments | Its 32×32 style is far from modern social-media images |
| SID_Set | A synthetic-image detection collection, about 140 GB in the pinned snapshot | Offers more generator and image variety | Expensive to acquire and process; licence and source balance still matter |
| WildFake | A very large multi-generator collection, about 1.29 TB in the pinned inventory | Broad source coverage | Too large to treat casually; provenance, duplication and class/source balance require auditing |

TikTok suggests these datasets. It does not say that all three are mandatory,
balanced, sufficient or representative of the hidden test.

The separate demonstration collection contains:

- 4,998 real COCO `val2017` images; and
- 8,843 DALL-E Advanced images indexed through WildFake.

This collection is **demo-only**. It must not be used for training and does not
contribute to the final score. The public COCO archive contains 5,000 images,
but the organizer has not identified the two excluded files. We must not invent
that list.

The most reasonable explanation for the prohibition is that learning those
exact files, their compression or the DALL-E-specific pattern could make a
detector look excellent in the demonstration without learning general AI-image
detection. The prohibition strongly indicates a separate scored evaluation,
but the final dataset is still unknown.

### Important official unknowns

The organizer has not published:

- the final test images or generator families;
- the final technical metric;
- whether several transformations will be chained;
- the exact transformation software and randomisation;
- the complete JSON container, ordering, confidence direction or error rules;
- whether partially AI-edited real images will appear; or
- exactly how every component of an ensemble is counted under the 2B limit.

There is also a judging conflict. The Track 5 statement gives 35% technical
execution, 20% innovation/problem insight, 20% impact/relevance, 15%
feasibility/practicality and 10% presentation/communication. The general
Devpost rules instead list four equally weighted categories and omit
presentation. Neither source says which formula controls Track 5.

Sources: [public Track 5 statement](https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc)
and [Devpost rules](https://tiktoktechjam2026.devpost.com/rules).

## 2. The minimum vocabulary needed to understand the evidence

An **AI-image detector** is a program that estimates whether an image was made
by AI. Most current detectors are themselves neural networks taught using
labelled real and generated examples.

An **unseen generator** is an image generator that the detector did not
encounter during training. Performance on unseen generators is important
because the final test is hidden and image generators change quickly.

A **transformation** is a post-creation change such as compression, blur,
resizing or cropping. Surviving transformations is called **robustness**.

A **parameter** is one learned number in a neural network. Parameter count is a
rough measure of model size. A frozen model still contains parameters and must
not be ignored merely because those parameters were not updated during
training.

**Inference** means running the finished model to make predictions. The size
rule applies to everything the program needs during inference, including frozen
components.

A **backbone** or **encoder** is the main image-understanding part of a model.
An **ensemble** combines several models. Ensembles can be stronger but consume
more parameters, memory and time.

**Accuracy** is the fraction of correct decisions at one chosen cutoff.
**AUC** measures how well generated images tend to receive higher suspicion
scores than real images across every possible cutoff. **F1** balances missed
fakes against real images wrongly accused. Results using different metrics,
datasets or rules are not directly comparable.

A **false positive** is a real image wrongly called generated. A **false
negative** is a generated image wrongly called real.

**Calibration** means choosing and maintaining a sensible cutoff for the
confidence score. A detector can rank images well, and therefore have high AUC,
while still making poor yes/no decisions because its cutoff is wrong.

## 3. How AI-image detectors work

There is no single defect present in every generated image. Current detectors
combine imperfect clues.

### Broad visual features

Large pretrained vision encoders such as CLIP, DINO, SigLIP and Perception
Encoder turn an image into a useful numerical representation of its objects,
composition and style. A classifier can learn where real and generated images
usually sit in that representation.

Why it helps: broad internet pretraining often transfers better than learning
one generator fingerprint from scratch.

Why it fails: the model can still learn subject matter, image-source style or
the visual habits of generators represented in its training data.

### Local texture and noise

Patch and convolutional methods examine fine texture, neighbouring pixels,
camera-like noise, low-order pixel bits and small residual patterns. Older GANs
often left repeated upsampling or checkerboard patterns. Diffusion systems can
leave decoder or denoising residuals.

Why it helps: these clues can expose generation even when the whole picture
looks visually convincing.

Why it fails: JPEG, blur, resizing, a new decoder, or photographing a screen
can erase or replace the clue.

### Frequency, wavelet and phase clues

These methods transform the image so repeated patterns and high-frequency
irregularities become easier to see. Wavelets examine detail at several
scales. Phase-based work tries to retain information that survives some JPEG
compression.

Why it helps: generated and camera images can have different spectral traces.

Why it fails: compression itself changes frequency information. The detector
may also learn “JPEG means real” instead of learning AI generation.

### Reconstruction clues

Some detectors pass an image through a diffusion decoder or reconstruction
model and measure how the image changes. An image made by a related generator
may reconstruct differently from a camera photograph.

Why it helps: reconstruction can expose a relationship to a known generation
process.

Why it fails: a new or pixel-space generator may not share that relationship.

### Hybrid systems

Hybrid detectors combine broad visual features with local, frequency, noise or
reconstruction evidence. Combining clues can help when their failures differ.

Why it fails: complexity, licences, memory use and total parameter count grow.
A large competition-winning ensemble may violate TikTok's size limit.

### Watermarks and signed provenance

A deliberate watermark such as SynthID hides a known signal during generation.
C2PA Content Credentials store signed information about an image's origin and
editing history. These can be stronger evidence than pixel guessing when they
are present and intact.

They cannot solve the complete challenge. An arbitrary image may never have
received a watermark or credential, and metadata can be removed. The absence
of provenance does not prove that an image is real.

Sources: [UnivFD](https://arxiv.org/html/2302.10174),
[ImageDetectBench](https://arxiv.org/html/2411.13553),
[SynthID](https://deepmind.google/blog/identifying-ai-generated-images-with-synthid/)
and [C2PA](https://spec.c2pa.org/specifications/specifications/2.2/explainer/Explainer.html).

## 4. The findings that most change how this challenge should be understood

### Finding 1: training-data coverage can matter more than the model name

Community Forensics used ordinary CLIP or ConvNeXt image models but generated
2.7 million fake images from 4,763 public latent-diffusion models plus 30 other
generators. Performance improved as generator diversity increased and began to
level off only after roughly 1,000 generators. Adding genuinely different
generator families mattered more than adding near-duplicate checkpoints.

SSAFE reached a related conclusion from the opposite direction. It used a
vision model to group similar sources, then selected representative
generator-and-content combinations. A curated 10,000-image set averaged 96.4%
RealWorldBench accuracy, compared with 94.9% from random selection under the
same budget. More images were not automatically better than more representative
images.

RINE supplied the clearest single number. Changing its training family from
ProGAN to latent diffusion moved Midjourney accuracy from **34.2% to 92.4%**.
The architecture name stayed essentially the same; the training source changed.

Useful conclusion: a high-quality architecture trained on the wrong generator
families can be worse than a simpler detector exposed to representative data.
Dataset composition is part of the model, not a secondary detail.

Sources: [Community Forensics](https://arxiv.org/html/2411.04125),
[SSAFE](https://arxiv.org/html/2606.08634) and
[RINE](https://arxiv.org/html/2402.19091).

### Finding 2: no released detector is universal

A 2026 study tested 16 methods, represented by 23 released detector versions,
on 12 datasets without retraining. Dataset-to-dataset ranking agreement ranged
from almost none, 0.01, to 0.87. The strongest average released system in that
study, Community Forensics, reached only 0.780 mean accuracy. SAFE ranged from
0.032 on one dataset to 0.998 on another.

Mean accuracy across the tested detectors was only 18% on Firefly v4, 19% on
Imagen 4, 21% on Flux Dev, 24% on Midjourney v7 and 31% on DALL-E 3. The exact
values depend on the released checkpoints and the study's fixed cutoff, but
the conclusion is firm: one strong paper score does not prove coverage of a
new generator.

AI-GenBench organised 36 generators chronologically from 2017 to 2024. Models
trained on the past could suffer a sudden drop when a new generator family
appeared, especially at the arrival of Stable Diffusion and DeepFloyd IF. This
is closer to real deployment than a random split mixing old and new generators.

Useful conclusion: “works on unseen generators” must be demonstrated on whole
generator families held out by time or source. A random image split can leak
almost identical generator patterns into both training and testing.

Sources: [2026 zero-shot benchmark](https://arxiv.org/html/2602.07814) and
[AI-GenBench](https://arxiv.org/html/2504.20865).

### Finding 3: robustness and unseen-generator generalisation are different

A detector can fail for two distinct reasons:

- **Robustness failure:** it knows the generator, but compression, blur, crop,
  noise or resizing destroys its clue.
- **Generalisation failure:** the image comes from a generator family whose
  clue the detector never learned.

NTIRE 2026 is the closest prior competition. It used 42 generators and 36
transformations, with one to five distortions chained in the robust track. The
winner reached 0.9974 clean AUC and 0.9723 robust AUC. Another entry reached
0.9953 clean AUC but only 0.8336 robust AUC. Near-perfect clean performance did
not guarantee transformed performance.

NTIRE's strongest systems were also extremely large. The runner-up combined
two 7-billion-parameter models and required about 78 GB of inference memory.
That system alone is seven times TikTok's total parameter limit. The winner
used six models and reported training with 32 A100 GPUs for about eight hours.
Prior rank does not override our resource rules.

RRBench studied repeated social-media transmission and physical recapture.
DRCT-ConvB's generated-image accuracy fell from 93.52% on originals to 64.34%
after printing or display recapture. DIRE fell from 89.72% to 1.42%. TikTok has
not promised such recapture tests, but this proves that “robust to JPEG” is not
the same as robust to every redistribution path.

Useful conclusion: clean accuracy, listed TikTok transformations, unseen
generators and more extreme real-world recapture are separate evaluation axes.
Success on one must not be reported as success on all.

Sources: [NTIRE 2026 report](https://arxiv.org/html/2604.11487) and
[RRBench](https://arxiv.org/html/2509.09172).

### Finding 4: a detector can cheat without anyone noticing

Many datasets store real images as variable-size JPEG files and generated
images as fixed-size PNG files. A model can then learn file history, image
dimensions or source collection instead of learning AI generation.

“Fake or JPEG?” demonstrated this causally with the **same genuine FFHQ
images**. A detector trained on a Midjourney subset recognised 80.45% of the
uncompressed PNG files as real. Saving those same real images as JPEG quality
95 raised real accuracy to 94.84%; JPEG quality 60 raised it to 100%. Nothing
about the picture's truth changed. The detector had learned “JPEG means real.”

Matching both compression and image size removed more than 75% of the available
training examples but improved ResNet mean accuracy from 71.68% to 82.74% and
Swin from 74.09% to 85.83%. Less data with fewer shortcuts was better than more
biased data.

B-Free catalogued related shortcuts: real and fake classes can differ in
format, fixed size, resizing history, subject matter or source collection. Even
if file format and size are matched, using ImageNet for real images and LAION
for fake-image source material can teach the detector “ImageNet versus LAION.”

Useful conclusion: JPEG augmentation alone does not remove dataset bias. Real
and generated classes must be compared with matched format, resolution,
content and processing history before a score can be trusted.

Sources: [Fake or JPEG?](https://arxiv.org/html/2403.17608) and
[B-Free](https://arxiv.org/html/2412.17671).

### Finding 5: the confidence cutoff can fail even when the model learned something

A detector produces a score, and a cutoff turns that score into “real” or
“generated.” When a new generator shifts the score distribution, the old
cutoff may become wrong.

In one historical-update study, a detector reached 0.92 AUC on Firefly before
seeing Firefly training examples, yet called only 35% of Firefly images
generated. It ranked them reasonably but used an unsuitable old boundary.
After direct Firefly training, AUC approached 1.00 and generated accuracy rose
to 99%.

A separate calibration study improved some detectors using very small labelled
target samples or estimated unlabelled score distributions, but some cases
worsened. Calibration changes the boundary; it does not create better image
features. Using TikTok's forbidden demonstration images to set that boundary
would still be using them to adapt the detector and must not be assumed legal.

Useful conclusion: AUC, accuracy and F1 answer different questions. Every
reported result needs its metric, cutoff, real-image performance and
generated-image performance. A single percentage can hide the failure that
matters.

Sources: [online detector update study](https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Epstein_Online_Detection_of_AI-Generated_Images__ICCVW_2023_paper.pdf)
and [calibration study](https://arxiv.org/html/2602.01973).

### Finding 6: hidden tests can reverse the apparent winner

In Meta's Deepfake Detection Challenge, the eventual second-place private
system had ranked **37th** on the public data. The winner had ranked fourth.
The public leaderboard did not predict the final order.

In Inclusion 2024, an EfficientNet baseline had 0.9939 validation AUC but only
0.92998 on the public unseen-type test. The winning system's public ensemble
reached 0.98051 AUC, but that number was not its hidden score or final weighted
score. Validation AUC, public AUC, hidden AUC and judged rank were different
facts.

MediaEval 2025 supplied another warning. Its official training data differed
substantially from its test domain, and one participant obtained much stronger
results using labelled validation data that resembled the test. TikTok has
explicitly forbidden the analogous shortcut with its demo collection.

Useful conclusion: the TikTok demonstration is not evidence about the final
ranking distribution. Public and demo performance can reward specialisation
that a separate hidden test punishes.

Sources: [DFDC report](https://arxiv.org/html/2006.07397),
[Inclusion 2024 report](https://arxiv.org/html/2412.20833) and
[MediaEval 2025](https://2025.multimediaeval.com/paper47.pdf).

### Finding 7: whole generated images and small AI edits may be different tasks

SAFE included fully generated images, traditional edits, splices and local AI
edits. A preliminary combined detector reached F1 0.82 on fully generated
images but only 0.25 on AI-edited images. Localising AI-edited pixels was
harder still, at F1 0.14.

The historical-update study found the same separation. For Stable Diffusion 1
inpainting, training only on whole generated images produced localisation F1
0.1807. Training on genuine inpainting examples reached 0.9795. Pasting
synthetic regions helped but was far weaker than using examples of the actual
editing process.

Useful conclusion: a detector trained only on wholly generated images cannot
be assumed to detect a mostly real image containing one generated region.
TikTok has not said whether partial edits appear, so this remains an important
unknown rather than a confirmed test requirement.

Sources: [SAFE challenge report](https://openaccess.thecvf.com/content/WACV2026W/SynRDinBAS/papers/Nguyen_The_SAFE_Image_Authenticity_Challenge_Detecting_and_Localizing_Partial_and_WACVW_2026_paper.pdf)
and [online detector update study](https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Epstein_Online_Detection_of_AI-Generated_Images__ICCVW_2023_paper.pdf).

### Finding 8: deliberate attacks are a different level of difficulty

AADD-2025 asked teams to make small changes to generated images so four
detectors would call them real. The best attack balanced image similarity
0.742 with attack success 0.672. Strong attacks combined gradients from several
known and substitute models so that the modification could transfer to hidden
detectors.

Useful conclusion: ordinary compression robustness does not imply security
against someone deliberately attacking the detector. TikTok has not said that
adversarial attacks are part of its test, so this is a boundary of current
technology, not a prediction about the hidden data.

Source: [AADD-2025 report](https://openreview.net/pdf/5a1add4a4f4e8cc99a1c5f2efe56492afcd3963c.pdf).

## 5. Prior detector ideas that contribute a distinct useful lesson

The systems below are not selected Track 5 solutions. Each is retained because
it established one useful idea or exposed one important limitation.

### UnivFD: broad pretraining can transfer beyond one fingerprint

UnivFD froze a CLIP ViT-L/14 image encoder and trained a simple real-versus-fake
boundary using only ProGAN examples. Compared with the paper's conventional
classifier, its linear version improved unseen diffusion/autoregressive
accuracy by 23.39 points. This demonstrated the value of broad internet
pretraining.

Later benchmarks showed that UnivFD still changes rank on newer datasets.
“Universal” describes the ambition, not a guarantee. Its code is MIT-licensed,
and the full encoder—not merely the small trained layer—counts at inference.

Source: [UnivFD](https://arxiv.org/html/2302.10174).

### Community Forensics: generator diversity can outweigh architecture changes

Changing between its tested ViT and ConvNeXt models mattered less than adding
diverse generator families. Its own high-resolution detector reported 0.923
mean accuracy across five in-paper benchmarks, while the later zero-shot study
measured 0.780 under another protocol. Both can be correct because the saved
model version, data and testing rules changed.

Useful caution: even the public reconstruction substitutes some real-image
sources that cannot be redistributed. Reproducing a score requires the exact
dataset edition, not only the model name.

Source: [Community Forensics](https://arxiv.org/html/2411.04125).

### CO-SPY: complementary clues can cover different failures

CO-SPY combined a CLIP semantic branch with a reconstruction-artifact branch.
Semantic clues survived JPEG better; the artifact branch transferred across
some content but lost roughly 17 percentage points under JPEG. The model
learned how much to trust each branch.

On one benchmark AIDE was stronger clean, 92.77% versus 87.75%, but CO-SPY was
stronger after JPEG, 79.76% versus 73.08%. This is direct evidence that the
best clean detector may not be the best robust detector. CO-SPY's artifact
branch was weaker on pixel-space generators, and its complete parameter count
was not disclosed.

Source: [CO-SPY](https://arxiv.org/html/2503.18286).

### AIDE: combining global and local evidence still does not remove hard cases

AIDE combined whole-image semantic features with high- and low-frequency
patches. It reported 92.77% average clean accuracy across 16 generator sets,
but fell to 69.60% at its strongest JPEG condition. On Chameleon—a collection
of highly convincing fakes and real images—nearly all tested detectors
approached chance.

Its code is MIT-licensed, but the Chameleon data is academic-only and
non-commercial. The complete size of AIDE's two ResNets plus very large
ConvNeXt must be counted before assuming Track 5 eligibility.

Source: [AIDE](https://arxiv.org/html/2406.19435).

### DRCT: making real and fake examples harder to separate can help

DRCT reconstructed both real and generated images through Stable Diffusion,
making superficial differences smaller. Under its aligned protocol, this
raised a ConvNeXt detector from 79.11% to as high as 96.55%.

The idea is useful because easy class differences encourage shortcuts. The
limitations are equally important: preparing two million reconstructed images
is expensive, only resize and JPEG were tested, and the audited repository had
no declared software licence.

Source: [DRCT](https://proceedings.mlr.press/v235/chen24ay.html).

### PatchCraft and LOTA: compact local clues can be useful but fragile

PatchCraft compared texture-rich and texture-poor patches using high-pass and
neighbouring-pixel relationships. LOTA examined low-order pixel bits and used
only a selected active patch. LOTA's reported versions had 23.6 or 28.4
million parameters, showing that low-level approaches can be compact.

Both produced strong results in their own protocols, but rankings shifted on
later datasets. PatchCraft failed badly on carefully selected Chameleon fakes,
and its repository has no declared licence. These methods show that local clues
can complement broad features; they do not establish a universal detector.

Sources: [PatchCraft](https://arxiv.org/html/2311.12397) and
[LOTA](https://arxiv.org/html/2510.14230).

### SPAI and CPTFormer: frequency information can help, but compression cuts both ways

SPAI learned real-image spectral patterns and combined frequency
reconstruction, context and attention. In an experiment that removed one part
at a time, removing spectral pretraining reduced AUC from 91.0 to 52.5;
removing distortion augmentation reduced it to 84.2. Its robustness tests used
individual transformations, not long chains.

CPTFormer focused specifically on phase information that can survive some JPEG
operations. It reported mean compressed accuracy of 76.3% on GANs and 63.5%
on diffusion models without knowing JPEG quality. Phase is useful evidence for
compression, not a solution to crop, blur, noise or unseen generators. Its
complete parameter count was not disclosed.

Sources: [SPAI](https://arxiv.org/html/2411.19417) and
[CPTFormer](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Detecting_Compressed_AI-Generated_Images_via_Phase_Spectrum_Robustness_CVPR_2026_paper.html).

### RINE and SAFE: advertised trainable size is not the same as total size

RINE trained only 6.32 million parameters, but still required a frozen CLIP
encoder of roughly 400 million-plus parameters at inference. Track 5 must count
the complete submitted system, not only the part updated during training.

SAFE reported only 1.44 million parameters and used crops, patch masking and
wavelet clues. It reached 96.7 mean accuracy in its own protocol, yet the later
zero-shot benchmark measured values ranging from 0.032 to 0.998 across
datasets. Its high-frequency clue also suffered under JPEG.

Useful conclusion: parameter efficiency is valuable, but neither a small head
nor a small weight file proves that the complete system is small or robust.

Sources: [RINE](https://arxiv.org/html/2402.19091) and
[SAFE](https://arxiv.org/html/2408.06741).

### SSAFE and SafeIMG: recent generators must actually appear in evaluation

SSAFE tested generators including Flux, Imagen 3/4, DALL-E 3, GPT-Image-1,
Nano Banana, Seedream and Recraft. Its curated system reported 99.0 AUC on
RealWorldBench, but individual fake accuracy still ranged from 72.4% on
Ideogram to 100% on several generators. Its 1.88-billion-parameter image tower
is nominally below TikTok's limit but leaves little margin, and code and weights
were not available when inspected.

SafeIMG evaluated GPT Image 2. Its strongest tested specialised detector found
only 33.1% of generated images, while humans averaged 81.7%. The detector set
was narrow and omitted many stronger recent systems, so this proves failure of
those tested detectors, not every detector. It still demonstrates that polished
new generators can invalidate confidence built on older benchmarks.

Sources: [SSAFE](https://arxiv.org/html/2606.08634) and
[SafeIMG](https://arxiv.org/html/2607.22745).

## 6. Additional competition lessons worth keeping

### IEEE VIP Cup 2022 shows that judging is broader than detector accuracy

VIP Cup used known and unseen generators, crops, resizing and JPEG, with a
one-hour runtime limit on a 16 GB GPU. The team with the strongest reported
technical accuracies finished second overall because final awards also
considered innovation, report and presentation. Unknown-generator accuracy was
roughly ten points below known-generator performance even for leading teams.

This directly helps interpret TikTok's mixed technical and judged criteria: a
technical leaderboard result and the final winner may not be the same thing.

Source: [VIP Cup report](https://arxiv.org/html/2309.12428).

### Defactify shows the difference between known-family detection and open generalisation

Defactify used 96,000 paired real/generated images from COCO and five named
generator families. The binary winner reached 0.8334 F1 versus a 0.80144
baseline, while five-way generator attribution reached only 0.4986 F1. The top
seven binary teams were separated by only 0.0042 F1.

The test remained tied to known supplied families. Good results there do not
prove performance on an entirely new generator.

Source: [Defactify report](https://arxiv.org/html/2605.20787).

### NIST supplies a better model of honest hidden evaluation

NIST used fresh images from participating generator teams in several rounds.
Detectors could not inspect or tune on the hidden test. It measured AUC,
detection at a fixed false-positive rate, equal-error rate and the quality of
the reported probabilities.

The useful idea is a moving evaluation with fresh generator outputs and
meaningful confidence scores, rather than repeatedly optimising against one
static public collection.

Source: [NIST evaluation plan](https://ai-challenges.nist.gov/pub/GenAI_Image_Discriminators_Evalplan.pdf).

## 7. What commercial detector apps really tell us

Hive, AI or Not, Illuminarty and Sightengine are black boxes. We can measure
their outputs but cannot honestly claim to know their private architecture or
training data.

Independent studies found real strengths and serious limits:

- one small 2024 study measured Hive at 98.03% clean accuracy, but its
  generated-image detection fell to 67.56% after strong Glaze processing;
- AI or Not reached 90.67% in that study but wrongly accused 24.47% of the real
  artworks;
- a study of 144,175 paired images found Sightengine much weaker when a real
  image was used as the starting point, including 25.12% in one Midjourney
  image-plus-text condition; and
- results changed when the same service was retested months later because the
  commercial detector itself had been updated.

The likely advantage of good commercial services is not a secret perfect
fingerprint. It is large current data, frequent updates and possibly several
specialised models. Their success supports the importance of coverage and
maintenance; their failures warn against treating one online result as truth.

Sources: [Organic or Diffused](https://arxiv.org/html/2402.03214),
[commercial-detector study](https://arxiv.org/pdf/2404.14581) and
[ImageDetectBench](https://arxiv.org/html/2411.13553).

## 8. How to read any detector claim without being misled

Before treating a result as relevant to Track 5, the evidence must answer:

- Which real sources and generator versions were used?
- Were whole generator families absent from training?
- Were real and generated classes matched for format, resolution, subject and
  processing history?
- Was the test clean, singly transformed or repeatedly transformed?
- Was the method retrained, and on what data?
- Which saved model version, or checkpoint, and confidence cutoff produced the
  number?
- What were real-image and generated-image results separately?
- Was the reported number accuracy, AUC, F1 or a judged final rank?
- What is the complete inference parameter count, memory use and speed?
- Are the code, weights and every data source legally redistributable?

If these answers are missing, a headline percentage is not enough evidence to
compare the method with another paper or predict TikTok performance.

## 9. The conclusions that survive every source

### Data quality and diversity are central

The detector must learn AI generation rather than file format, image source or
one generator family. Representative diversity can matter more than raw image
count or an exotic architecture.

### Generalisation and robustness must be separated

An unseen generator and a transformed known-generator image create different
failures. Both must be reported independently before any claim of robustness
is meaningful.

### Hidden evaluation is essential

Public demonstrations and validation sets can reward accidental specialisation.
Prior competitions repeatedly showed rank reversals on genuinely hidden data.
TikTok's demo collection cannot be treated as a sample to learn from.

### Confidence scores need their own scrutiny

A high AUC can coexist with poor yes/no accuracy. Thresholds can become stale
as generator families change. False positives on real images are as important
as missed generated images.

### Resource and licence checks can invalidate an otherwise strong method

NTIRE-scale ensembles can exceed the 2B limit. “Trainable parameters” may hide
a much larger frozen encoder. Missing, academic-only or non-commercial licences
can prevent code, weights or data from being redistributed in a public entry.

### No evidence justifies naming one guaranteed winner today

Every strong method has been measured on a particular mixture of generators,
real images, transformations, metrics and thresholds. The TikTok final mixture
is unpublished. Copying the largest published score would confuse a paper's
test distribution with this competition.

## 10. What is prepared and what remains unknown

Ready preparation evidence:

- the public and Early Bird Track 5 requirements have been reconciled;
- CIFAKE is verified locally at exactly 120,000 images;
- the official COCO source has 5,000 images and licence metadata;
- the DALL-E Advanced index has exactly 8,843 matching rows;
- SID_Set and WildFake have pinned remote inventories without committing their
  enormous contents;
- safe acquisition records and a neutral input-directory verifier exist;
- no organizer baseline exists, so none is falsely claimed as reproduced; and
- no detector, training pipeline, model weight, generated output or secret was
  added during preparation.

Still unknown:

- the final scored dataset and generator families;
- the final technical metric;
- whether transformations are chained;
- exact transformation software and randomisation;
- complete JSON formatting and failure behaviour;
- how the sub-2B rule counts ensembles and dependencies;
- whether partially edited images or provenance signals appear;
- which official judging formula controls Track 5; and
- the two COCO files excluded from the demo.

These are organizer omissions, not research gaps. They must remain labelled
unknown until authoritative clarification appears.

## Final conclusion

AI-image detection is a learned comparison between real and generated image
distributions, not a universal visual test. It can work very well when its data
cover the relevant generators and processing conditions. It can fail suddenly
on a new generator, after a transformation, because of a stale confidence
cutoff, or because the dataset taught it a shortcut such as “JPEG means real.”

The most useful lesson from prior methods and competitions is therefore not a
particular model name. It is the standard by which evidence must later be
judged: genuinely held-out generator families, matched real/fake data,
separate clean and transformed results, honest confidence metrics, complete
resource accounting and usable licences. Until the hidden evaluation and
remaining rules are published, that is the strongest defensible understanding
of Track 5.
