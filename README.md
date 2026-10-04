# nlp-group-3-prosit-1

Code for **Prosit 1, Building and Adapting Language Models**, ICS554 Natural Language
Processing, MSc Intelligent Computing Systems, Ashesi University.

The prosit has two practical parts, and this repository holds the notebooks for both.

| Folder | Section | What it is |
|---|---|---|
| [`ewe-ngram-language-model/`](ewe-ngram-language-model/) | B | A word-level n-gram language model for **Ewe**, a low-resource Ghanaian language, built from counts with interpolated smoothing |
| [`health-llm/`](health-llm/) | C | Adapting an open LLM to the **health domain** for Ghanaian general practice, by continued pre-training then instruction tuning, both via LoRA |

The written report (Sections A, B and C) is submitted separately and is not in this repository.

---

## Section B: Ewe n-gram language model

A from-scratch n-gram model over 47,131 Ewe sentences (1.53M tokens) drawn from a Kaggle
bilingual corpus and Waxal image-caption transcriptions, trained at orders 1 to 5 with add-k
smoothing at the unigram level and interpolation above it.

**Headline result:** test perplexity falls from **595.9** (unigram) to **167.3** (bigram) to
**139.2** (trigram), a 4.3x improvement, with dev and test agreeing closely at every order.
Longer contexts initially looked worse, but that turned out to be a tuning artefact: the 4- and
5-gram weights had settled on the edge of the search grid, and widening it brought them to
132.0 and 131.9.

```bash
cd ewe-ngram-language-model
pip install -r requirements.txt
jupyter notebook ewe_ngram_lm.ipynb     # runs offline in ~30 seconds, fixed seed
python wide_grid_check.py               # reproduces the tuning-grid finding independently
```

Contents: `ewe_ngram_lm.ipynb` is the full pipeline; `wide_grid_check.py` is a standalone check
of the grid result; `data/` holds the corpus.

---

## Section C: Health-domain LLM

`Qwen3-4B-Base` (text-only, base rather than instruct, so that any instruction-following is
attributable to our own training) adapted in two stages with LoRA rank 32, α = 64:

1. **Continued pre-training** on 16.1M words of open health text: PubMed Central open-access
   articles, MedlinePlus patient-information topics, and Ghana's Standard Treatment Guidelines
   and Essential Medicines List (upweighted x3 as the local-context source).
2. **Instruction tuning** on 3,000 medical question-answer pairs in a fixed ChatML format, with
   the loss computed on the answer only.

Three models are compared on data split off **before** any training: **A** the untouched base,
**B** base plus Stage 1, **C** base plus both stages.

**Held-out perplexity (lower is better)**

| Model | MedlinePlus | Ghana STG/EML | PubMed Central | WikiText-2 (control) |
|---|---|---|---|---|
| A: base | 4.12 | 6.59 | 5.15 | 9.66 |
| B: +stage 1 | **4.06** | **5.99** | **4.54** | 9.50 |
| C: +stage 1+2 | 4.07 | 6.20 | 4.58 | 9.51 |

Stage 1 improved every health set (PubMed by 11.9%, the Ghana guidelines by 9.1%) while the
general-English control did not degrade, so the domain gain did not come at the cost of
catastrophic forgetting. On zero-shot medical exam questions the base model already scored
59.3% against 25% chance, and most differences fall inside the ±5.7-point margin of error at
N = 300, so we report them as null. The clearest Stage 2 effect is the closing of the
chat-format penalty, from 4.3 points for the base model to 1.1 for the fully tuned one.

### Notebooks

| Notebook | Hardware | Use |
|---|---|---|
| [`health_llm_cuda.ipynb`](health-llm/notebooks/health_llm_cuda.ipynb) | CUDA GPU (Colab T4) | **The run behind the reported results.** Start here. |
| [`health_llm_mlx.ipynb`](health-llm/notebooks/health_llm_mlx.ipynb) | Apple Silicon (MLX) | A second run on `Qwen3.5-2B-Base`, kept as a cross-platform comparison |
| [`health_llm.ipynb`](health-llm/notebooks/health_llm.ipynb) | CUDA GPU (Unsloth) | The earlier Unsloth-based attempt, superseded |

Each notebook carries its outputs, so the result tables and loss curves can be read without
re-running anything.

### Reproducing

The dataset is not committed; it is rebuilt from source with the licence of every document
recorded per item.

```bash
cd health-llm
python collect.py medlineplus
python collect.py moh                 # needs pdftotext (apt install poppler-utils)
python collect.py hf
python collect.py pmc 3000            # the slow step
python prep.py                        # writes data/prep/ and colab_health_bundle.zip
```

Then upload `colab_health_bundle.zip` to Google Drive and run `health_llm_cuda.ipynb`, which
mounts Drive, unpacks it, and checkpoints the adapter every 10 steps so a dropped session does
not cost the run.

Self-tests: `python hl_lib.py`, `python prep.py --selftest`, `python collect.py --selftest`.

---

## Data and licences

Every source was checked before use and its licence stored with each document. PubMed Central
articles keep their per-article licence (CC BY, CC0, CC BY-NC, CC BY-NC-SA, with
no-derivatives excluded, since ND conflicts with fine-tuning); MedlinePlus is public domain
(courtesy of MedlinePlus, US National Library of Medicine); the Ghana guidelines are Ministry
of Health publications whose reuse terms should be checked before redistribution. We
deliberately excluded scraped patient-doctor chat datasets (unclear rights and possible
personal data), StatPearls (CC BY-NC-ND), and copyrighted exam textbooks. Neither Ewe source's
licence has been independently verified.

## Scope and limits

Both models are coursework artefacts.

The health model is **not a clinical tool**. No clinician has reviewed any of its output, and it
can state a wrong drug dose in fluent, confident prose, which is more dangerous than being
obviously wrong. Its outputs must not inform care. The Ghana guidelines used are the 2017
edition and a newer edition supersedes them; everything is English-only; and PubMed
over-represents research-country populations, so conditions and drug availability common in
rural Ghana are likely under-represented.

The Ewe model is trained on a corpus with a real religious skew concentrated in roughly half of
it, so it does not represent everyday Ewe, and no Ewe speaker has rated its output.

## Group

Group 3, ICS554 Prosit 1.

Eugene Emmanuel Sewor, Henry Kofi Brandoh, Henry Nii-Amarh Olletey, Liza Ekua Ndede Aikins.
