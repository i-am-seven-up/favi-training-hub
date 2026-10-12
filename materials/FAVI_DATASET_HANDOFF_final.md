# FAVI Dataset Handoff

## Purpose

This file preserves the working context, decisions, dataset findings, and next actions for the FAVI graduation thesis so the work can continue in Codex without needing the original chat history.

---

# 1. Research Direction

## Thesis scope

FAVI is a social application with AI-assisted sensitive-content moderation.

Current research scope:

- **Image-level sensitive-content classification**
- Detect:
  - **Sexual / NSFW**
  - **Violence**
- Exclude:
  - Politics
  - Text-only moderation
  - Video moderation
  - Multimodal post context as a research variable
  - Retrieval quality as the main research variable
  - Production optimization as the main thesis contribution

## Model direction

Primary model family:

- CLIP / SigLIP-like pretrained vision-language model
- Parameter-efficient fine-tuning with **LoRA**

Current experimental comparison:

1. Frozen pretrained encoder + linear classification head
2. LoRA-adapted same encoder + same classification head
3. Optional later: full fine-tuning as an upper-bound / cost comparison

## Current output formulation

To keep the thesis simple, use **single-label 3-class classification**:

- `Safe`
- `Sexual`
- `Violence`

The team explicitly decided **not** to use multi-label Sexual+Violence detection for now because dataset annotation/evaluation would become more complex.

## Research question

> Can parameter-efficient LoRA fine-tuning of a pretrained vision-language model improve image-level sensitive-content classification for Safe / Sexual / Violence compared with a frozen pretrained baseline, under the same evaluation protocol?

## Hypothesis

> LoRA adaptation will improve sensitive-content classification performance over a frozen pretrained baseline while updating substantially fewer parameters than full fine-tuning.

---

# 2. Evaluation Protocol

Primary metric:

- **Macro F1**

Secondary metrics:

- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion matrix
- Accuracy as a descriptive metric only

Efficiency metrics:

- Trainable parameter count
- Peak VRAM
- Training time
- Optional later:
  - inference latency
  - model size

Comparison rule:

All compared methods should use the same:

- dataset version
- canonical label mapping
- train/validation/test split
- random seed
- preprocessing
- input resolution
- held-out test set
- metric implementation
- classification head where applicable

Main result:

`Δ Macro F1 = Macro F1(LoRA) - Macro F1(Frozen)`

Interpret the performance gain together with efficiency cost.

---

# 3. Dataset Working Procedure

For every candidate dataset, use the same procedure:

```text
fetch
→ inspect dataset structure / labels
→ count samples
→ extract or display 20–50 random representative samples
→ manually inspect semantics and domain quality
→ PASS / PASS WITH LIMITATION / REJECT
→ only then integrate into the canonical dataset
```

Do **not** merge a dataset into the final training set before visual inspection.

Important principles:

- Quality > raw dataset size
- Avoid source/domain shortcuts
- Avoid leakage
- Keep provenance
- Do not treat object presence as equivalent to violent action
- Do not assume `non-NSFW == Safe`
- Do not assume `non-violence == Safe`
- `Safe` is constructed only at the final canonical-label stage

Recommended eventual manifest:

```text
image_path
canonical_label
original_label
source
group_id
```

`group_id` is required when data comes from video frames or another grouped source.

---

# 4. Finalized Dataset Tree

```text
FAVI Sensitive Content Dataset
│
├── Sexual / NSFW
│   │
│   ├── Positive NSFW
│   │   └── deepghs/nsfw_detect
│   │       ├── hentai        ✅ OK
│   │       ├── porn          ✅ OK
│   │       └── sexy          ✅ OK
│   │
│   ├── Non-NSFW
│   │   └── deepghs/nsfw_detect
│   │       ├── neutral       ✅ OK
│   │       └── drawings      ✅ OK
│   │
│   └── Hard negatives
│       🟡 NEED ENRICHMENT
│       → swimwear / beach
│       → gym / exposed skin
│       → medical / skin-heavy images
│       → hugging / kissing
│       → safe anime/art
│       → other visually suggestive but non-sexual content
│
├── Violence
│   │
│   ├── Positive Violence
│   │   │
│   │   ├── Physical violence / fighting
│   │   │   └── VIMD
│   │   │       🟡 CANDIDATE
│   │   │       → fetch + inspect
│   │   │
│   │   ├── Weapon-related aggression
│   │   │   └── Subh775/WeaponDetection
│   │   │       🟡 CANDIDATE
│   │   │       → filter aggressor/stabbing/victim/violence/blood
│   │   │       → exclude weapon-only
│   │   │
│   │   ├── Blood / gore / visible injury
│   │   │   ├── NeuralShell/Gore-Blood-Dataset-v1.0
│   │   │   │   🟢 PASS WITH LIMITATION
│   │   │   │   → good stylized gore
│   │   │   │
│   │   │   └── Real-world blood/injury
│   │   │       🔴 MISSING
│   │   │
│   │   └── Crowd / riot violence
│   │       🔴 MISSING
│   │
│   ├── Non-Violence
│   │   🔴 NEED SOURCE / DEFINITION
│   │   → normal human interaction
│   │   → sports without actual aggression
│   │   → crowds without conflict
│   │   → people holding harmless objects
│   │   → ordinary street / indoor scenes
│   │
│   └── Hard negatives
│       🔴 NEED ENRICHMENT
│       → boxing/wrestling without harmful context
│       → action movie poses
│       → toy guns / prop weapons
│       → gun/knife product shots
│       → red paint / ketchup / fake blood
│       → medical procedures
│       → cosplay / horror makeup
│       → protest/crowd without violence
│
└── Final 3-class training taxonomy
    │
    ├── Safe
    │   → Non-NSFW
    │   + Non-Violence
    │   + selected hard negatives
    │
    ├── Sexual
    │   → Positive NSFW
    │
    └── Violence
        → Positive Violence
```

---

# 5. Dataset Findings So Far

## 5.1 deepghs/nsfw_detect

Status:

**PASS**

Downloaded and inspected successfully.

Observed folder names and counts:

```text
drawings   -> 5,600
hentai     -> 5,600
neutral    -> 5,600
porn       -> 5,600
sexy       -> 5,600
```

Total:

`28,000 images`

Important correction:

The dataset description says `drawing`, but the extracted folder is actually:

`drawings`

Visual inspection confirmed the class semantics.

Current mapping at concept level:

```text
Positive NSFW:
hentai
porn
sexy

Non-NSFW:
neutral
drawings
```

Do **not** directly call `neutral + drawings` the full Safe class.

They are only the negative side of the NSFW concept.

The final Safe class must also be verified as non-violent and should contain hard negatives.

Initial practical subset idea:

```text
Positive Sexual ~5,000
- porn    ~1,667
- hentai  ~1,667
- sexy    ~1,666

Non-NSFW ~5,000
- neutral   2,500
- drawings  2,500
```

Do not freeze this exact number until the final Violence and Safe composition is known.

---

## 5.2 onullusoy/harmful-contents

Status:

**REJECT AS PRIMARY VIOLENCE SOURCE**

Can potentially be used as supplementary manually curated data.

Dataset layout:

- train: 3,609
- validation: 771
- test: 773

Total image count:

`5,153`

Metadata fields:

```text
name
alcohol
drugs
weapons
gambling
nudity
sexy
smoking
violence
mask_alcohol
mask_drugs
mask_weapons
mask_gambling
mask_nudity
mask_sexy
mask_smoking
mask_violence
```

Violence-positive count:

```text
train       481
validation  102
test        103
----------------
TOTAL       686
```

Overlap among `violence=1`:

```text
weapons=1, nudity=0, sexy=0 -> 584
weapons=0, nudity=0, sexy=0 -> 99
weapons=0, nudity=0, sexy=1 -> 3
```

So:

- `584 / 686 ≈ 85.1%` are also weapon-positive
- `102 / 686` have violence without weapons

Visual inspection findings:

The violence subset is very mixed and contains:

- weapon-only / weapon-heavy imagery
- suicide / self-harm imagery
- memes
- symbolic/staged threat imagery
- some real fighting
- some blood / wounds
- some actual aggression
- ambiguous samples

Main concern:

The classifier could learn:

`gun / knife presence => Violence`

rather than learning actual violent content.

Decision:

```text
harmful-contents
→ REJECT as primary Violence source
→ MAYBE supplementary after manual filtering
```

---

## 5.3 NeuralShell/Gore-Blood-Dataset-v1.0

Status:

**PASS WITH LIMITATION**

Direct `load_dataset()` failed because of a schema/ClassLabel error:

```text
ValueError:
Invalid string class label
Gore-Blood-Dataset-v1.0@...
```

The correct workaround was:

- use `snapshot_download()`
- inspect repository files
- extract ZIP files manually

Repository contains:

```text
new-dataset.zip
old-dataset.zip
mixed-dataset.zip
README.md
...
```

Raw images are inside these ZIPs.

Approximate dataset size:

`~398 images`

Visual inspection:

- gore/blood signal is strong
- samples clearly represent graphic bloody content
- however, much of the sampled data appears:
  - stylized
  - illustration-like
  - AI-generated / synthetic-looking

Decision:

```text
Good for:
stylized gore / graphic bloody art

Not sufficient for:
real-world blood / injury / gore
```

Therefore:

```text
NeuralShell/Gore-Blood-Dataset-v1.0
→ PASS WITH LIMITATION
→ keep as stylized-gore source
→ still need real-world injury/blood dataset
```

---

## 5.4 VIMD

Intended role:

**Physical violence / fighting**

Current status:

`CANDIDATE — NOT YET VALIDATED`

Expected content:

- fighting
- assault
- physical aggression

Why it is interesting:

- designed specifically for violence image detection
- more semantically focused than a generic harmful-content dataset
- potentially useful as the core Physical Violence source

Required next action:

```text
fetch
→ inspect structure
→ count images/classes
→ display ~50 fighting samples
→ PASS / REJECT
```

Do not integrate before inspection.

---

## 5.5 Subh775/WeaponDetection

Intended role:

**Weapon-related aggression**

Current status:

`CANDIDATE — NOT YET VALIDATED`

Observed annotation schema includes categories such as:

```text
weapons
Aggressor
Blood
Guns
Guns perspective
Hand
Heavy Gun
Knife
Knife_Deploy
Knife_Weapon
Long guns
Person
Pistol
Rifle
Shotgun
Stabbing
Victim
...
violence
weapon
```

Important rule:

Do **not** classify every image containing a gun/knife as Violence.

Candidate violent-context annotations:

```text
Aggressor
Stabbing
Victim
violence
Blood
```

Potential filtering strategy:

Keep images where one or more violent-context labels are present.

Reject or separately treat:

- isolated firearm
- knife product shot
- weapon laying on a table
- generic object detection image without aggression

Required next action:

```text
fetch
→ identify correct schema
→ filter violent-context candidates
→ inspect ~50 images
→ PASS / PASS WITH FILTER / REJECT
```

---

# 6. Missing Dataset Areas

## 6.1 Sexual / NSFW hard negatives

Need more visually confusing but safe/non-sexual examples.

Priority examples:

- swimwear / beach
- gym / exposed skin
- medical / skin-heavy
- hugging
- kissing
- safe anime
- safe art
- suggestive fashion that is not explicit

Goal:

Prevent model from learning:

`skin exposure => Sexual`

---

## 6.2 Real-world blood / visible injury

Current gore dataset is too stylized.

Need real-world images covering:

- blood
- wounds
- visible injury
- post-assault injury
- graphic injury

Avoid making the class dominated by:

- AI art
- horror illustrations
- movie posters

---

## 6.3 Crowd / riot violence

Need a real-world source covering:

- riots
- mob violence
- protest clashes
- police/crowd confrontation
- crowd aggression

Important:

Do not automatically label all protests or crowds as Violence.

Need non-violent protest/crowd hard negatives as well.

---

## 6.4 Non-Violence

Need concept-specific negative data for Violence.

Examples:

- normal human interaction
- normal street scenes
- indoor social interaction
- crowds without conflict
- sports without harmful aggression
- ordinary people holding harmless objects

---

## 6.5 Violence hard negatives

These are especially important.

Suggested examples:

- boxing / wrestling in legitimate sports context
- action movie poses
- toy guns
- prop weapons
- firearm product images
- knife product images
- red paint
- ketchup
- fake blood
- medical procedures
- horror makeup
- cosplay
- peaceful protests
- crowd scenes without violence

Goal:

Prevent shortcuts such as:

```text
weapon => Violence
red color => Blood
crowd => Riot
physical contact => Fight
```

---

# 7. Target Violence Coverage

Keep the final model output as a single `Violence` class.

Internally, build diversity using subtypes:

```text
Positive Violence
├── Physical fight / assault
├── Weapon-related aggression
├── Blood / gore / visible injury
└── Crowd / riot violence
```

These are **dataset composition subtypes**, not separate output labels.

Optional manifestations that are currently out of scope unless later needed:

- accident/disaster injury
- self-harm
- suicide
- death/corpse-only imagery

Do not silently mix these into Violence without explicitly deciding to expand scope.

---

# 8. Final Safe Construction

Important semantic rule:

```text
Safe ≠ Non-NSFW only
Safe ≠ Non-Violence only
```

The final Safe class should be built from samples that are defensibly:

```text
non-sexual
AND
non-violent
```

plus selected hard negatives.

Conceptually:

```text
Safe
=
Non-NSFW
+ Non-Violence
+ selected Sexual hard negatives
+ selected Violence hard negatives
```

But verify labels before merging.

---

# 9. Dataset Quality Rules

For every final source, document:

- dataset name
- source URL
- license
- original task
- original labels
- canonical FAVI mapping
- sample count
- source/version/download date
- known limitations
- known domain bias
- whether samples originate from video frames
- whether group-level provenance exists

Avoid severe class imbalance.

For the first usable baseline, a practical target remains approximately:

```text
Safe      ~4k–5k
Sexual    ~4k–5k
Violence  ~4k–5k
```

Total:

`~12k–15k images`

This is not a theoretical minimum. It is a practical thesis-sized dataset target.

Quality and source diversity are more important than reaching an exact number.

---

# 10. Leakage Rules

Critical:

If any image source originates from videos:

- preserve `video_id` / source group
- all frames from one video must stay in the same split

Do **not** randomly split frames from the same video across train/test.

Also check for:

- exact duplicate images
- resized duplicates
- cropped duplicates
- reuploaded duplicates
- source duplicates across aggregated datasets

Freeze the test set once the canonical dataset is finalized.

---

# 11. Current M1 Status

Completed:

```text
✅ Research scope frozen
✅ Evaluation protocol frozen
✅ Sexual/NSFW source selected
✅ deepghs class semantics manually inspected
✅ harmful-contents investigated and rejected as primary Violence
✅ stylized Gore source investigated
```

In progress:

```text
🟡 Build Violence source composition
```

Still required:

```text
□ Validate VIMD
□ Validate Subh775/WeaponDetection
□ Find real-world blood/injury source
□ Find crowd/riot violence source
□ Define/find Non-Violence data
□ Enrich Sexual hard negatives
□ Enrich Violence hard negatives
□ Create canonical unified manifest
□ Deduplicate / leakage audit
□ Freeze train/val/test split
□ Train frozen CLIP/SigLIP baseline
□ Train LoRA version
□ Compare Macro F1 + efficiency
```

---

# 12. Immediate Codex Mission

Continue from here without redesigning the taxonomy unless new evidence requires it.

## Task 1 — Validate VIMD

Goal:

Determine whether VIMD is suitable as the main Physical Violence source.

Procedure:

1. Fetch VIMD
2. Inspect repository/dataset structure
3. Count fighting/non-fighting RGB images
4. Randomly display ~50 fighting images
5. Evaluate:
   - actual physical violence?
   - source diversity?
   - frame/video duplication?
   - domain bias?
   - label quality?
6. Mark:
   - PASS
   - PASS WITH LIMITATION
   - REJECT

Acceptance criteria:

- Most sampled images clearly show physical aggression/fighting
- Not dominated by one near-duplicate scene
- Provenance/splitting can be handled defensibly

---

## Task 2 — Validate Subh775/WeaponDetection

Goal:

Find a usable Weapon-related Aggression subset.

Procedure:

1. Load dataset
2. Resolve exact annotation structure
3. Filter candidates containing:
   - Aggressor
   - Stabbing
   - Victim
   - violence
   - Blood
4. Do not automatically include generic gun/knife labels
5. Randomly display ~50 filtered candidates
6. Evaluate whether scenes represent:
   - aggression
   - attack
   - threat against another person
   - stabbing
   - actual violent context

Acceptance criteria:

- Majority of retained images show violent context, not object-only presence
- A clear filtering rule can be reproduced

---

## Task 3 — Find Real-world Blood/Injury Dataset

Requirements:

- real-world images preferred
- raw image pixels
- labels or provenance clear enough to defend
- ideally hundreds to thousands of usable samples
- avoid AI-art-only / illustration-only datasets
- inspect 20–50 before approval

Output:

```text
candidate
source
license
size
labels
sample-quality assessment
PASS/REJECT
```

---

## Task 4 — Find Crowd/Riot Violence Dataset

Requirements:

- real-world crowd violence
- riots / clashes / mob aggression / protest confrontation
- avoid equating normal protests with violence
- source should permit a reproducible split
- inspect 20–50 before approval

Need complementary hard negatives:

- peaceful protest
- non-violent crowd
- police presence without aggression

---

## Task 5 — Hard Negative Enrichment

Sexual:

```text
swimwear
beach
gym
medical
hugging
kissing
safe anime/art
skin-heavy but non-sexual
```

Violence:

```text
combat sports
wrestling
toy gun
prop weapon
weapon product photo
red paint/ketchup
medical procedures
horror makeup
cosplay
peaceful protest
non-violent crowd
```

Find small, reproducible sources rather than trying to build a massive dataset.

---

# 13. Do Not Do Yet

Until dataset composition is frozen, do not spend significant time on:

- LoRA rank/alpha tuning
- extensive hyperparameter search
- full fine-tuning
- retrieval subsystem
- production deployment optimization
- model serving
- threshold tuning
- elaborate architecture changes

First milestone:

```text
canonical dataset
→ clean manifest
→ frozen split
→ frozen pretrained baseline
```

Then move to LoRA experiments.

---

# 14. Working Principle

The current thesis should remain simple:

> One pretrained VLM backbone, one 3-class classification task, one frozen baseline, one LoRA adaptation, and a carefully constructed dataset whose Safe/Sexual/Violence semantics can be defended.

The research value should come from:

- parameter-efficient adaptation
- careful dataset composition
- controlled comparison
- clear evaluation

not from unnecessary architecture complexity.
