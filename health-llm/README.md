# Health-domain LLM (GP-style general consultation): Prosit 1, Section C

Adapt an open English LLM, **Qwen3-4B-Base**, to health text for Ghanaian general practice in two LoRA stages (continued pre-training, then instruction tuning), and show it is effective **as a language model**.

Everything is in one notebook, **`health_llm.ipynb`**, saved with the outputs of the run behind the reported results (Colab T4). It:

1. downloads and prepares the data from its original sources, keeping each document's licence (Step 0.4);
2. trains Stage 1 (3M tokens of health text) and Stage 2 (3,000 Q&A examples), checkpointing the adapter to Google Drive every 10 steps;
3. evaluates the base model (A), +Stage 1 (B) and +Stages 1 and 2 (C) on held-out perplexity, zero-shot medical exam questions and GP vignettes.

## Run it

Open `health_llm.ipynb` in Google Colab with a **T4 GPU** runtime and run all cells. Results and adapters are written to `MyDrive/NLP_LABS/health_llm/` (set `DRIVE_DIR` to change it). The first run builds the dataset, which takes a while (about 3,000 PubMed Central articles are downloaded); later runs reuse the prepared copy saved on Drive.

## Data and licences (checked 2026-09-21)

| Source | Use | Licence |
|---|---|---|
| PubMed Central open access (AWS open-data bucket), 3,000 clinical articles | continued pre-training and held-out perplexity | per article: CC BY, CC0, CC BY-NC, CC BY-NC-SA (no-derivatives excluded) |
| MedlinePlus health topics | continued pre-training and held-out perplexity | public domain (NLM), credit required |
| Ghana Standard Treatment Guidelines 2017, Essential Medicines List 2017 | continued pre-training (×3) and held-out perplexity | Ghana MoH publications; check reuse terms before redistributing |
| medical-o1-reasoning-SFT, MedText, MedQuAD, MedMCQA train, MedQA train | instruction tuning | Apache-2.0, CC BY 4.0, CC BY 4.0 (verify), Apache-2.0, CC BY 4.0 |
| MedQA test, MedMCQA validation, PubMedQA labelled | evaluation only (never trained on) | CC BY 4.0, Apache-2.0, MIT |
| WikiText-2 test | general-English perplexity control | CC BY-SA 3.0 |
| Deliberately not used | scraped patient-doctor chats, StatPearls (CC BY-NC-ND), USMLE textbooks | unclear or restrictive |

## Caveats

- Educational use only: not a clinical tool, and no clinician has reviewed its output.
- MCQ benchmarks measure exam knowledge, not consultation quality; the vignettes are read by hand.
- MedQA/MedMCQA *train* splits are in the instruction data; evaluation uses their *test/validation* splits.
