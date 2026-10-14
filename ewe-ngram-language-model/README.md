# Ewe N-Gram Language Model

A word-level n-gram language model for **Ewe**, a low-resource Gbe language of Ghana and Togo, built from scratch with only `pandas`, `matplotlib` and the Python standard library. It learns from about 47,000 Ewe sentences, scores text by perplexity, and generates Ewe-looking text, at orders 1 to 5.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook ewe_ngram_lm.ipynb     # Restart & Run All: about 40 seconds, offline, fixed seed
```

The notebook is saved with its outputs, so every number below can be read without re-running.

## Results

Test perplexity (lower is better; the test set is scored once, after tuning on the dev set):

| Model | Context | Dev | Test |
|---|---|---|---|
| Unigram | none | 588.9 | 595.9 |
| Bigram | 1 word | 167.6 | 167.3 |
| Trigram | 2 words | 140.9 | **139.2** |
| 4-gram | 3 words | 188.4 | 184.6 |
| 5-gram | 4 words | 307.2 | 299.6 |

The tuned weights for orders 3 to 5 sat at 0.5, the edge of the search grid. The notebook re-tunes on a wider grid (**dev set only**): trigram 134.5, 4-gram 132.0, 5-gram 131.9. So longer context plateaus rather than hurting; re-scoring the test set with the wider grid is the next step.

## Approach

1. **Clean:** lowercase, keep Ewe letters (`ɔ ɛ ŋ ɖ ƒ`) and `. , ! ?` as tokens, drop digits and other symbols, add `<s>` `</s>`.
2. **Split** 80 / 10 / 10 (seed 42): 37,704 / 4,713 / 4,714 sentences.
3. **Count** all 1- to 5-grams in the training set.
4. **Smooth:** add-k at the unigram level, recursive linear interpolation above it.
5. **Tune** `k` and one λ per order on the dev set by staged grid search.
6. **Evaluate** test perplexity once; **generate** text by sampling.

## Data

| File | Lines | Source |
|---|---|---|
| `data/EWE_ENGLISH.csv` | 28,614 | Ewe side of *EWE-English Bilingual Pairs*, Tchaye-Kondi & Gbedevi, Kaggle, DOI 10.34740/KAGGLE/DSV/1462736 |
| `data/waxal_transcriptions.xlsx` | 19,151 | Waxal Ewe image-caption transcriptions (University of Ghana, 2023) |

Neither licence has been independently verified. The Kaggle half is partly religious in register (35.8% of its lines carry a verse-style number); the Waxal half is everyday descriptions.

## Limitations

- No neural baseline, so no claim that n-grams beat neural models in general.
- The tokeniser splits Unicode combining marks (`melɔ̃a` → `melɔ` + `a`) and treats `Ð` and `ɖ` as different letters; 39 lines are corrupted junk.
- Perplexity only; generated text has not been rated by a native speaker.
- One split for the reported numbers; five seeds gave a trigram perplexity of 139.0 ± 1.6, so gaps of a couple of percent are noise.
