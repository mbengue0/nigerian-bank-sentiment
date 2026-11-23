# AI assistance log (CS-4714 capstone)

Earlier sessions (per the handoff and slide 9): ChatGPT assisted with project scoping, proposal drafting and presentation preparation.

## 22 November 2026 — Claude (Anthropic), implementation session 1
- Checked environment: 2 CPU cores, 7.8 GB RAM, no GPU; Python 3.13, scikit-learn 1.9.1, pandas 3.0.5; installed PyTorch 2.15 (CPU use).
- Claude wrote `scripts/01_audit.py` (dataset download check, field validation, audit, removal log).
- Claude downloaded the 16 CSVs and dataset card (README.md) from the pinned revision 27d2b6e7d541a0b713dc8eed56b0daf27476d2a7 and ran the audit. All recorded counts reproduced exactly.
- Claude ran langid on all candidates, inspected samples, proposed and implemented the English
  word-share filter, near-duplicate removal, seeded 100k sample and 80/10/10 split
  (`02_language_id.py`, `03_filter_sample_split.py`). Manual sample check done by Claude
  (`outputs/language_check_notes.md`); not yet reviewed by the student.
- Claude wrote and ran `04_baselines.py` (majority class; TF-IDF + logistic regression, 6-setting
  validation grid) and `05_gru.py` (GRU pilot, then full run with early stopping).
- The test split has not been used.

## 23 November 2026 — Claude, session 2
- Claude ran two more full GRU seeds (1, 2); validation macro-F1 0.9046 / 0.9061 / 0.9089 for seeds 42 / 1 / 2.
- Claude drafted the progress report (Claude Docs) from the recorded outputs. Test split still unused.
