# How We Taught a Computer Program About Health

*A plain-language walkthrough of the HealthLM project, written for a first-year IT student.*

You don't need to know any machine learning to read this. You do need to be curious. Every step
below says **what** we did and, more importantly, **why** — because in this kind of work the
"why" is the whole skill. Anyone can run a training script. Knowing why you're running it, and
knowing when its output is lying to you, is the actual job.

---

## Part 1 · What we were trying to do

We took a computer program that was already good at English in general, and made it better at
**health** — specifically, the kind of conversation you'd have with a GP (a general doctor).
You describe your symptoms; it suggests what might be going on, what to check, and what warning
signs mean "go to hospital now."

Then — and this is the part that actually matters — we **measured** whether it really improved,
instead of just assuming it did.

> ⚠️ Before anything else: this is a school project, not a doctor. It can be confidently wrong,
> which is more dangerous than being obviously wrong. Nobody should use it for real medical
> decisions.

---

## Part 2 · Four ideas you need first

### A language model is very fancy autocomplete

When your phone suggests the next word as you type, it's guessing based on what you've typed so
far. A **language model** does the same thing, but trained on an enormous amount of text.

That's genuinely all it does: *given the words so far, what word probably comes next?*

The surprising part is that being really good at this requires knowing grammar, facts, and how
the world works. You can't reliably predict the end of "the patient has a high fever and we
suspect…" without knowing some medicine. Knowledge comes along for the ride.

### Text gets chopped into tokens

Computers don't read words, they read numbers. So text is first split into **tokens** — pieces
of words. "diabetes" might be one token; an unusual drug name might be four or five. The model
sees a list of numbers, and predicts the next number.

When we say "we trained on 16 million words," the model actually saw roughly 20 million tokens.

### Training means adjusting millions of dials

Inside the model are billions of numbers called **parameters** (or weights). You can think of
them as dials. Training works like this:

1. Show the model some text with the next word hidden.
2. Let it guess.
3. Measure how wrong it was.
4. Nudge the dials slightly so it would have been less wrong.
5. Repeat millions of times.

### Loss is the "how wrong were you" score

Step 3 above produces a number called the **loss**. Think of it like golf: **lower is better**.
If loss goes down over time, the model is learning. If it doesn't, something is broken.

That's the single most important number you'll watch, and later we'll see why watching it alone
can fool you.

---

## Part 3 · What we actually did, step by step

### Step 1 · Decide exactly what job the model should do

**What:** We chose one narrow task — GP-style health consultation, with Ghana as the setting.

**Why:** "Make an AI for health" is not a goal, it's a wish. You can't test it. "Given a
patient's symptoms, suggest likely causes, questions to ask, next steps, and red-flag warning
signs" is a goal, because you can check whether it did that.

A vague goal is the most common way a project fails — not because the code breaks, but because
nobody can tell whether it worked.

### Step 2 · Collect the text to learn from

**What:** We gathered about 16 million words of health writing from three places:

| Source | What it is | How much |
|---|---|---|
| PubMed Central | Real medical research papers | 2,850 articles (15.4M words) |
| MedlinePlus | Plain-English health guides for patients | 913 topics (315k words) |
| Ghana Standard Treatment Guidelines | Ghana's official medical guidance | 167k words |

**Why these three:** They teach different things. Research papers teach technical medical
language. MedlinePlus teaches how to explain things to ordinary people. The Ghana guidelines
teach *local* practice — which malaria drugs are actually used here, what a Ghanaian clinic
actually does. A model trained only on American research papers would sound clever and be
useless in Accra.

### Step 3 · Check you're allowed to use it

**What:** For every single document, we recorded its licence — the legal terms for reuse. We
deliberately threw away some data we could easily have taken: scraped patient–doctor chat
websites (unclear rights, and real people's private medical questions), and a medical
encyclopaedia licensed "no derivatives," which forbids exactly what we wanted to do.

**Why:** "It was on the internet" is not permission. This is the step students skip and
professionals never do. Using data you have no right to can invalidate the entire project, and
in the case of scraped patient conversations, it's a privacy violation against people who never
agreed to be part of your experiment.

It cost us some data. It saved us a problem we couldn't have defended.

### Step 4 · Clean the text

**What:** Real documents are messy. The Ghana guidelines came as PDFs, and converting a PDF to
text goes wrong in odd ways. Our converter turned every bullet point (•) into the letters
**"yy"** — 9,412 times. That made "yy" the single most common word in the document, more common
than "the".

We also removed 23,536 broken characters from the table of contents, and cut ~9,000 words of
front matter (the printer's address, phone numbers, acknowledgements).

**Why:** The model learns whatever patterns you show it. Feed it "yy" 9,412 times and it will
faithfully learn that "yy" is the most important word in Ghanaian medicine. Garbage in, garbage
out — literally.

**A trap we nearly fell into:** We almost deleted every line that repeated many times, assuming
repeated lines were page headers. Then we checked, and found the most repeated lines were
`Adults` (682 times), `Children` (673) and `Treatment objectives` (233) — real, meaningful
medical headings. Deleting them would have destroyed the document's structure.

*Lesson: measure before you "clean". A rule that sounds obviously right can be obviously wrong.*

### Step 5 · Hide the test data BEFORE training (the most important step)

**What:** Before any training started, we locked away a portion of the text: 150 research
articles, 102 health guides, 42 Ghana passages. The model never saw them.

**Why:** Imagine a teacher who gives students the exam paper to study, then gives them the same
exam. Everyone scores 100%. Have they learned anything? You cannot possibly tell.

If test data leaks into training, the model can score brilliantly by simple memorisation, and
your results are meaningless. Splitting *first* is what makes the before/after comparison honest.

This is the step that separates a real experiment from a demo.

### Step 6 · Choose which model to start from

**What:** We started from an existing model, `Qwen3-4B-Base`, made by Alibaba. "4B" means 4
billion parameters (dials). We did **not** build one from scratch.

**Why not from scratch:** Building a language model from nothing needs *trillions* of words and
a warehouse of GPUs costing millions. We had 16 million words and a free cloud GPU. Not close.

**Why the "Base" version:** Models come in two flavours. A **Base** model has only learned to
predict text. An **Instruct** model has additionally been taught to follow instructions and
answer questions politely.

We deliberately chose Base — the *less* capable one. Why? Because if we'd started with a model
that already answers questions nicely, and ours answered questions nicely at the end, we could
never prove *we* caused it. Starting from Base means every improvement is provably ours.

**Why a text-only model:** Our first choice, Qwen3.5, turned out to also handle images. That
caused two separate crashes — including one where the text-reader tried to *download a web
address it found inside a research paper and open it as a photo*. Switching to a text-only
model deleted a whole category of bugs from a task that has no pictures in it.

### Step 7 · Decide how to train: LoRA

**What:** Instead of adjusting all 4 billion dials, we froze them and added a small set of new
ones — about 66 million, or **1.6%** of the model. Only those get adjusted. This trick is
called **LoRA**.

**Why:** Three reasons.

1. **It fits.** Adjusting all 4 billion dials needs far more memory than our GPU had.
2. **It's reversible.** The original model is untouched, so our "before" comparison is
   perfectly recoverable — just don't load the add-on.
3. **It's small.** The result is a 264 MB file instead of an 8 GB one.

**The analogy:** You want to add a chapter to an encyclopaedia. You could rewrite all 40 volumes
(full training), or write notes on sticky tabs and attach them to the relevant pages (LoRA). The
sticky notes are smaller, faster, and you can peel them off.

### Step 8 · Stage 1 — let it read

**What:** We showed the model our 16 million words of health text and let it do what it already
does: predict the next word. Same task as its original training, new material. This is called
**continued pre-training**.

**Why:** This teaches vocabulary, phrasing and facts of a field. The model learns that
"artemisinin-based combination therapy" is a normal phrase, that certain symptoms cluster
together, that Ghanaian guidelines are written a particular way.

**The analogy:** A medical student spending a year reading textbooks. Not yet seeing patients —
just absorbing the language and knowledge of the field.

### Step 9 · Stage 2 — let it practise answering

**What:** We showed it 3,000 examples of *question → good answer* pairs, in a fixed format:
a system message setting the role, a patient's question, and a good reply.

**Why:** Reading textbooks doesn't teach you to *hold a consultation*. After Stage 1 the model
knows a lot of health language but still just continues text — ask it a question and it might
write another question. Stage 2 teaches the *shape* of being helpful: read the question, answer
it directly, in a useful structure.

**The analogy:** After a year of reading, the student starts practising consultations with a
supervisor watching.

**One clever detail worth understanding.** When scoring the model in Stage 2, we told it to
ignore its mistakes on the *question* and only count mistakes on the *answer*.

Why? Because we don't want it to get good at predicting questions — patients supply those. We
want it good at producing answers. If you score everything, the model spends effort learning to
write realistic patient questions, which is completely useless to us.

### Step 10 · Test it three different ways

We ended up with three versions to compare:

- **A** — the original model, untouched
- **B** — after Stage 1 (reading)
- **C** — after Stage 1 and Stage 2 (reading + practice)

**Test 1: How surprised is it by health text it's never seen?**

This is called **perplexity**. Lower means less surprised, which means a better feel for that
kind of writing. Roughly, it's "how many words is the model torn between?" A perplexity of 5
means it's about as unsure as if it were choosing between 5 equally likely words.

*Why:* It directly measures the thing Stage 1 was supposed to improve, and it needs no human
marking — just text the model has never seen.

**Test 2: Medical exam questions**

We gave it 900 real multiple-choice medical exam questions and checked how many it got right.

*Why:* Perplexity says the model finds health text familiar. It doesn't say the model knows any
medicine. Exam questions test knowledge you can actually score. Random guessing scores 25%, so
anything near 25% means it knows nothing.

**Test 3: Fake patients, read by a human**

We wrote 12 realistic patient scenarios and read all three models' answers ourselves.

*Why:* An exam score can't tell you whether an answer is *helpful*, whether it mentions the
danger signs, or whether it invented a drug dose. Some things need a human to read them. This is
the test that catches "technically correct but useless or dangerous."

**And one control test — the trick that catches cheating**

We also tested all three models on ordinary Wikipedia text about non-medical topics.

*Why this is clever:* A model can get better at medicine by **forgetting** general English —
like a student who crams for biology and fails their English exam. That's called **catastrophic
forgetting**, and a health-only test cannot detect it. The Wikipedia test is the control: if
health scores improve and Wikipedia stays flat, we gained without losing.

Without a control you can't tell improvement from a trade-off. Always have one.

### Step 11 · Read the results honestly

Stage 1 worked:

| Text type | Before (A) | After Stage 1 (B) | Change |
|---|---|---|---|
| Research papers | 5.15 | 4.54 | **11.9% better** |
| Ghana guidelines | 6.59 | 5.99 | **9.1% better** |
| Patient guides | 4.12 | 4.06 | 1.6% better |
| Wikipedia *(control)* | 9.66 | 9.50 | unchanged |

All health text improved; general English didn't get worse. That's exactly the pattern you want.

The exam scores are where being honest matters. They moved by 1–2 points in places — but with
300 questions, the **margin of error is ±5.7 points**. So a 2-point change is meaningless noise,
not an improvement.

*Why this matters:* It is very tempting to report "accuracy improved!" when a number went up.
But if your measurement error is bigger than your change, you have measured nothing. Reporting
that honestly is better science than claiming a win — and any examiner who knows statistics will
check.

The one real gain: PubMedQA rose 6.0 points, which does clear the margin. And the base model
already scored 59% on medical exams *before* we did anything — it had clearly read a lot of
medicine already. Our contribution is a modest improvement on an already-capable model, and
saying so plainly is more credible than overclaiming.

---

## Part 4 · Things that broke (and why that's normal)

Real projects are mostly debugging. These are the failures worth learning from.

**The data that silently wasn't there.** The Ghana website blocked our downloader and sent back
an error page instead of the PDF. Our code saved it without checking, extracted no text, and
carried on. An entire training run — hours of GPU time — completed on a corpus containing *zero*
Ghanaian content. Nothing crashed. No error appeared.

*Lesson:* The worst bugs don't crash. They quietly produce nothing and let everything downstream
look fine. We now make the program **stop with an error** if the download isn't a real PDF or
the text is suspiciously short. Fail loudly, not silently.

**The hardware that lied.** We asked PyTorch "does this GPU support the fast bfloat16 number
format?" It said yes. It was wrong — or rather, technically right in a useless way: the GPU can
*simulate* that format slowly, and the function counted simulation as support. We were running
without the hardware acceleration we thought we had.

*Lesson:* Check what a function actually promises, not what its name suggests.

**The confident guess that ran out of memory.** We saw the model using 3.7 GB of 15.6 GB and
turned off a memory-saving feature to go faster. It crashed instantly. The 3.7 GB was memory at
*loading* time; training needs about 7 GB more for its working scratch space.

*Lesson:* One measurement taken at the wrong moment is worse than no measurement.

**The backwards numbers.** Our training score was consistently *worse* than our test score,
which should be impossible — like scoring higher on the exam than on the practice questions. The
cause was ours: we duplicated the Ghana text three times (to make the model notice it) *before*
splitting off the validation set, so some validation passages had near-identical twins in
training. We reported this in the write-up rather than hiding it.

*Lesson:* When a number looks impossible, it usually is. Chase it.

**The model that wouldn't shut up.** Our best model answered the pre-eclampsia question
correctly and then invented an entirely new, fictional patient and started answering that too.
Two small bugs combined: we never told it which symbol means "stop", and our cleanup code
deleted that symbol before the code that looked for it could find it.

*Lesson:* This one is worth sitting with. A medical AI that invents a patient is a small, funny
bug here. It is the same failure that, in a real clinic, invents a drug dose. Fluent text is not
correct text.

**The cloud that disconnected.** Our free GPU session dropped 3.5 hours into training, at step
85 of 89. We lost nothing, because we'd set it to save progress every 10 steps.

*Lesson:* Save your work automatically. Not as an optimisation — as the difference between
having a result and starting over.

---

## Part 5 · Glossary

| Term | Plain meaning |
|---|---|
| **Token** | A piece of a word; the unit a model actually reads |
| **Parameter / weight** | One of the billions of adjustable numbers inside the model |
| **Loss** | How wrong the model was. Lower is better |
| **Base model** | Trained only to predict text; doesn't reliably follow instructions |
| **Instruct model** | Also trained to follow instructions and answer questions |
| **Fine-tuning** | Further training of an existing model on new, focused data |
| **LoRA** | Training a small add-on instead of the whole model |
| **Continued pre-training** | Stage 1: more reading, same predict-the-next-word task |
| **Instruction tuning** | Stage 2: practice at answering questions properly |
| **Perplexity** | How surprised the model is by text it hasn't seen. Lower is better |
| **Held-out data** | Data hidden away before training, used only for testing |
| **Control set** | Data from a *different* topic, used to check nothing was lost |
| **Catastrophic forgetting** | Losing old skills while learning new ones |
| **Margin of error** | How big a change must be before it counts as real |
| **GPU** | The specialised chip that does the heavy maths |

---

## The short version

1. Define a job specific enough to test.
2. Collect data, and check you're allowed to use it.
3. Clean it, but measure before you delete anything.
4. **Hide the test data before you train.** Non-negotiable.
5. Start from an existing model; start from the *plain* version so improvements are yours.
6. Train a small add-on rather than the whole model.
7. Let it read (Stage 1), then let it practise answering (Stage 2).
8. Test in several different ways, and always include a control.
9. Don't claim an improvement smaller than your margin of error.
10. Save your progress, because the machine will disconnect.

The code, the data pipeline and the full technical report are in this repository. The most
valuable thing in it isn't the model — it's the record of what went wrong and how we found out.
