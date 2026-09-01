# Track 5 workshop: high-information questions

These are not questions to ask merely because information is missing. Each one
is designed to reveal something that materially changes how the challenge must
be understood, evaluated or reproduced.

During the session, listen first and skip anything the engineers answer in the
presentation. Ask the questions in order. If time is short, ask Questions 1–7.

## 1. How is the final scored data related to the data we can see?

**Ask exactly:**

> At the highest level you can disclose, how will the final scored image
> distribution relate to SID_Set, CIFAKE, WildFake and the demo collection?
> Should we expect the same generator families, unseen versions within known
> families, completely unseen generator families, or a mixture? Will authentic
> images also come from sources not represented in the suggested datasets?

**Why this is smart:** “Hidden test” can mean very different things. A random
split from familiar generators tests interpolation. Holding out generator
versions is harder. Holding out entire families and authentic-image sources
tests genuine generalisation. We do not need secret generator names; we need to
know what kind of generalisation the challenge intends to measure.

**Useful follow-up:**

> If some families overlap, is separation performed by exact model version or
> checkpoint, rather than randomly splitting images from the same generator?

## 2. What exactly counts as an AI-generated image?

**Ask exactly:**

> Is the final task strictly “fully authentic versus fully AI-generated,” or
> can it include real photographs altered with inpainting, generative fill,
> image-to-image tools or small AI-generated regions? If mixed images are in
> scope, how are they labelled?

**Why this is smart:** Detecting a wholly generated image and detecting a small
generated region inside a real photograph are materially different problems.
Prior evidence shows systems can perform well on the first and fail badly on
the second. The public statement does not resolve this boundary.

**Useful follow-up:**

> If partial edits are included, is there a minimum edited-area proportion, and
> is only an image-level confidence required or also localisation?

## 3. How are the transformed test images actually created?

**Ask exactly:**

> For each clean source image, will the organizer create one transformed copy,
> several separate transformed copies, or a chain of multiple transformations?
> What is the maximum chain length, and are the order and strengths random?

**Why this is smart:** One JPEG operation is different from crop followed by
resize, noise and JPEG. The number and order of transformations determine what
“robust” means and how results should be interpreted.

**Useful follow-ups:**

> Are transformations applied with the same probabilities and strengths to
> authentic and generated images?

> Are only the exact published strengths used, or can intermediate values
> appear?

> Is scoring performed per transformed file, or are several versions of one
> source image combined before scoring?

## 4. What is the real purpose of the forbidden demo collection?

**Ask exactly:**

> What is the demo collection intended to demonstrate? Is it merely a common
> showcase dataset, or is it a small representative sample of the final data,
> transformations or generator difficulty? Why did the organizer explicitly
> prohibit using it for training?

**Why this is smart:** If the demo is representative, its purpose may be to
compare frozen systems consistently. If it is deliberately unlike the final
test, strong demo performance says little about scoring. The reason for the
restriction tells us what conclusions the organizer considers legitimate.

**Useful follow-up:**

> Is every team required to evaluate on the 4,998 COCO plus 8,843 DALL-E
> Advanced images, and will the exact two excluded COCO files be published?

## 5. What uses of the demo data are prohibited besides gradient training?

**Ask exactly:**

> Does “must not be used for training” also prohibit selecting a model,
> choosing a confidence threshold, calibrating probabilities, weighting an
> ensemble, choosing preprocessing, or repeatedly inspecting errors on the demo
> collection? Must the complete system be frozen before it is run there?

**Why this is smart:** A system can learn from a dataset without updating neural
network weights. Threshold selection, model selection and repeated error-driven
changes can all overfit the same files. The phrase “not for training” does not
define these grey areas.

**Useful follow-up:**

> What is the permitted validation role of this collection, if any, before the
> final demonstration?

## 6. What does the submitted `pred` number mean to the scorer?

**Ask exactly:**

> What exact technical metric evaluates `pred`? Is it a ranking score measured
> by AUC, a probability measured by log loss, or a label produced using a
> threshold? Which direction and range mean “AI-generated”?

**Why this is smart:** A detector can rank images correctly but have a bad
threshold. Conversely, a useful binary classifier may output poorly calibrated
probabilities. We cannot interpret or validate `pred` until its mathematical
meaning is known.

**Useful follow-ups:**

> If a threshold is used, does the organizer choose it or does each team submit
> one?

> If teams choose it, which data are legally permitted for calibration?

> Are authentic and generated errors weighted equally if the test is
> imbalanced?

## 7. How is the final technical score assembled?

**Ask exactly:**

> How are clean images, transformed images, authentic images, generated images,
> generator families and transformation types combined into the final technical
> score? Is it one global average, an average of subgroup scores, or a worst-case
> robustness measure?

**Why this is smart:** A global average can hide complete failure on one
generator or transformation. Averaging groups equally rewards broad coverage.
A worst-case score rewards consistency. The aggregation rule defines what the
leaderboard actually values.

**Useful follow-up:**

> Will teams receive any per-generator, per-transformation or per-class
> breakdown, or only one number?

## 8. Which real-world failure matters most to TikTok?

**Ask exactly:**

> What real failure motivated this track most strongly: missing images from new
> generators, false accusations against authentic creators, failure after
> social-media processing, poor confidence calibration, or another problem?
> Is there a particular operating point—such as very low false-positive
> rate—that best represents the intended use?

**Why this is smart:** Two detectors with the same average score can have very
different practical value. At moderation scale, falsely accusing authentic
content may be especially costly. This question invites the engineers to
explain the real product problem without asking for hidden test contents.

**Useful follow-up:**

> Beyond the headline metric, what evidence would convince the engineers that a
> solution addresses that failure responsibly?

## 9. What kinds of content shift should the detector be expected to survive?

**Ask exactly:**

> Without revealing test files, will evaluation span substantially different
> content domains—for example photographs, artwork, text-heavy images, faces,
> products, screenshots or low-quality web images—or is content distribution
> controlled? Are authentic and generated classes matched by subject matter?

**Why this is smart:** If generated images and real images depict different
subjects, a detector can learn content instead of generation. Knowing whether
the test controls subject matter tells us whether shortcut resistance is an
explicit part of the challenge.

**Useful follow-up:**

> Does the organizer check for duplicated or near-duplicated source images
> across training resources, the demo and final evaluation?

## 10. What exactly is included in the two-billion-parameter limit?

**Ask exactly:**

> Does the sub-2B limit count every inference component: frozen encoders, all
> members of an ensemble, reconstruction networks, preprocessing models,
> watermark or provenance detectors and remotely called models? How should
> teams document the total?

**Why this is smart:** Papers often report only trainable parameters while
hiding a much larger frozen encoder. Without a complete counting rule, teams
can interpret the same limit differently.

**Useful follow-up:**

> Are there separate runtime, GPU-memory, model-file-size or per-image latency
> limits, and what evaluation hardware will be used?

## 11. Will the organizer execute the submitted software?

**Ask exactly:**

> Will judges run the repository on hidden images, or are technical results
> assessed from our own report and demonstration? If it will be run, what
> operating system, hardware, internet access, setup-time limit and dependency
> format should it support?

**Why this is smart:** A video-only demonstration and an organizer-executed
hidden test require different levels of reproducibility. External APIs and
unpinned downloads may also be impossible in an offline environment.

**Useful follow-up:**

> Will the organizer provide a schema, sample runner, container image or local
> validator before submission?

## 12. Which judging formula is authoritative, and what earns the non-metric points?

**Ask exactly:**

> The Track 5 statement gives five weighted categories, while the Devpost rules
> give four equally weighted categories. Which formula is authoritative? Within
> technical execution, how much depends on measured detector performance versus
> robustness analysis, code quality and reproducibility?

**Why this is smart:** “Best solution” can mean the highest hidden metric or the
highest overall judged score. Prior competitions have produced different
winners under those two definitions.

**Useful follow-up:**

> For innovation and problem insight, does the organizer value a novel model
> specifically, or can rigorous data design, failure analysis, calibration and
> efficient engineering also demonstrate innovation?

## 13. What exact evidence should the final submission show?

**Ask exactly:**

> For the required robustness comparison and error analysis, which breakdowns
> would judges consider most informative? Should results be separated by clean
> versus transformed, each transformation and strength, authentic versus
> generated, confidence range and generator family?

**Why this is smart:** This clarifies what the organizers mean by a convincing
analysis rather than guessing from a single aggregate score.

**Useful follow-up:**

> Should false-positive and false-negative examples be grouped by likely cause,
> and are confidence distributions or calibration plots expected?

## 14. Where will answers become binding written rules?

**Ask exactly:**

> If the webinar clarifies or changes anything in the current statement, where
> will the authoritative written clarification be posted, and which source
> controls if the webinar, Track 5 statement and Devpost rules differ?

**Why this is smart:** A spoken answer can be misunderstood. A written source
ensures every team receives the same rule and prevents later disputes.

## Avoid these weaker formulations

- Do not ask, “Which exact generators are in the hidden test?” Ask Question 1,
  which reveals the intended type of generalisation without seeking secret
  files.
- Do not ask, “Why can't we train on the demo?” alone. Ask Questions 4 and 5,
  which separate the demo's purpose from every prohibited form of adaptation.
- Do not ask, “How many transformations are there?” The published list already
  answers that. Ask Question 3 about how many are applied to one image, whether
  they are chained and how scoring aggregates them.
- Do not ask, “How is it judged?” alone. Ask Questions 6, 7 and 12, separating
  the detector metric, technical-score aggregation and overall judged rank.
- Do not ask, “Which model should we use?” That requests a solution choice
  instead of clarifying the problem.

## Live answer ledger

Record whether an answer is a confirmed rule, informal guidance or unresolved.

| Topic | Exact answer | Status | Written source or promised follow-up |
|---|---|---|---|
| Relationship between public, demo and final data |  |  |  |
| Whole generation versus partial edits |  |  |  |
| Transformation count, chaining and aggregation |  |  |  |
| Purpose and permitted uses of demo data |  |  |  |
| Meaning and metric for `pred` |  |  |  |
| Final technical-score aggregation |  |  |  |
| Intended real-world failure and operating point |  |  |  |
| Complete sub-2B counting and runtime limits |  |  |  |
| Organizer execution environment |  |  |  |
| Authoritative judging formula |  |  |  |
