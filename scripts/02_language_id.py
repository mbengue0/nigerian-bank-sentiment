"""
Stage 2a: run a language detector on every candidate review (no filtering yet).

Input : data/interim/candidates.csv.gz   (from 01_audit.py)
Output: data/interim/candidates_lang.csv.gz  - candidates + detected language + confidence

The detector is langid.py (Lui & Baldwin, 2012). It was trained mostly on formal text,
so short reviews, emoji-only reviews and Nigerian Pidgin may be misjudged. That is why
the decision rule is chosen afterwards from a manual sample, not taken on trust.
"""
import time
import pandas as pd
from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

cand = pd.read_csv("data/interim/candidates.csv.gz", keep_default_na=False)
start = time.time()
results = [identifier.classify(text) for text in cand["content"]]
cand["lang"] = [lang for lang, _ in results]
cand["lang_prob"] = [round(prob, 4) for _, prob in results]
print(f"classified {len(cand):,} reviews in {time.time() - start:.0f} s")

cand.to_csv("data/interim/candidates_lang.csv.gz", index=False)
print(cand["lang"].value_counts().head(15).to_string())
