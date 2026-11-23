# Sentiment Classification of Nigerian Banking App Reviews

CS-4714 capstone, Mouhamadou Mbengue. Compares TF-IDF + logistic regression with
trainable embeddings + one GRU layer on Google Play reviews of 16 Nigerian banks' mobile apps.
Extension: both models are also tested on a bank never seen in training (Zenith Bank).

## Setup

    pip install -r requirements.txt

Run every command from this folder. Python 3.13 was used. No GPU is needed.

## Run order

| Step | Command | What it does | Time (2-core CPU) |
|---|---|---|---|
| 0 | `python3 scripts/00_download.py` | Downloads the 16 CSVs from Hugging Face (pinned revision) | < 1 min |
| 1 | `python3 scripts/01_audit.py` | Checks the files, labels ratings, removes empty/duplicate/conflicting texts | ~1 min |
| 2 | `python3 scripts/02_language_id.py` | Runs langid on every review (kept for reference only) | ~2.5 min |
| 3 | `python3 scripts/03_filter_sample_split.py` | English filter, near-duplicates, holds out Zenith Bank, 100k sample from 15 banks, 80/10/10 split | ~1 min |
| 4 | `python3 scripts/04_baselines.py` | Majority class + TF-IDF/logistic regression, tuned on validation | ~30 s |
| 5a | `python3 scripts/05_gru.py --pilot` | GRU: 1 epoch on 20k reviews (feasibility check) | ~20 s |
| 5b | `python3 scripts/05_gru.py --seed 42` (then 1, 2) | GRU: full training with early stopping | ~4-6 min each |
| 6 | `python3 scripts/06_evaluate.py` | ONE final evaluation of the frozen models on test + unseen Zenith set | ~30 s |
| 7 | `python3 scripts/07_error_analysis.py` | Error rates by review type, McNemar test, sample of errors to read | ~10 s |

`scripts/gru_model.py` holds the GRU tokenizer, batching and model; it is imported by steps 5 and 6.
The fixed data sets are included in `data/final/`, so steps 4-7 can run without steps 0-3.

## Folders

- `scripts/` - all code, one file per stage
- `data/final/` - train.csv (80,000), val.csv (10,000), test.csv (10,000) from 15 banks, and
  unseen_zenithBank.csv (12,041). Columns: text_hash, content, label (0 = negative, 1 = positive),
  banks, strat_bank, years, scores, n_rows
- `data/raw_README.md` - the dataset card (license Apache-2.0)
- `outputs/` - reports, removal logs, model results, `final_results.md`, `error_analysis.md`,
  per-review predictions, `figures/` (confusion matrices)
- `outputs/preliminary_16banks/` - validation results from the first run, before Zenith was held out
  (these are the numbers in the progress report)
- `reports/` - progress report PDF
- `ASSISTANCE_LOG.md` - record of AI assistance

Not included (recreated by the scripts): `data/raw/`, `data/interim/`, `models/`.
