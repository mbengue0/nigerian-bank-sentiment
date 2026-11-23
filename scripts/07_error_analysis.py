"""
Stage 6: error analysis on the main test set (and the unseen bank), using saved predictions.

Input : outputs/predictions_test.csv, outputs/predictions_unseen_zenithBank.csv (from 06_evaluate.py)
Output: outputs/error_analysis.md            - error rate per review type, for LR and GRU (seed 42)
        outputs/error_sample_for_reading.csv - 150 random test errors to read and label by hand

Review types are found with simple, transparent word rules. They are rough signals, not
perfect detectors: e.g. "no" also appears in "no problem", and the Pidgin list is short.
A review can belong to several types. Text-rating disagreement cannot be detected by a rule,
so it is judged by reading the exported sample.
"""
import re
from pathlib import Path

import pandas as pd

from gru_model import tokenize

SEED = 42
NEGATION = re.compile(r"\b(not|no|never|nothing|none|nobody|cannot|can't|cant|don't|dont|doesn't|doesnt|"
                      r"didn't|didnt|isn't|isnt|won't|wont|wasn't|wasnt|n't)\b")
CONTRAST = re.compile(r"\b(but|however|although|though|except|yet)\b")
# Common Nigerian Pidgin / Nigerian English markers (short, hand-written list).
PIDGIN = re.compile(r"\b(dey|abeg|wetin|una|wahala|oga|sef|comot|abi|na wa|e no|no dey|dem|shey|sha|o|ooo|jare|mtchew)\b")

def tag(df):
    text = df["content"].str.lower()
    n_tokens = df["content"].map(lambda t: len(tokenize(t)))
    n_words = df["content"].str.split().str.len()
    return pd.DataFrame({
        "negation": text.str.contains(NEGATION),
        "contrast (but/however...)": text.str.contains(CONTRAST),
        "very short (1-3 words)": n_words <= 3,
        "long, truncated (>100 tokens)": n_tokens > 100,
        "Pidgin markers": text.str.contains(PIDGIN),
    }, index=df.index)

lines = ["# Error analysis\n",
         "Error rate = share of reviews in that group that the model got wrong. "
         "Types come from simple word rules (see scripts/07_error_analysis.py).\n"]

for set_name in ["test", "unseen_zenithBank"]:
    df = pd.read_csv(f"outputs/predictions_{set_name}.csv", keep_default_na=False)
    tags = tag(df)
    lr_wrong = df["lr_pred"] != df["label"]
    gru_wrong = df[f"gru{SEED}_pred"] != df["label"]
    rows = [("All reviews", pd.Series(True, index=df.index))] + [(c, tags[c]) for c in tags.columns]
    lines += [f"## {set_name} ({len(df):,} reviews)\n",
              "| Review type | Reviews | LR error rate | GRU error rate |", "|---|---:|---:|---:|"]
    for name, mask in rows:
        n = int(mask.sum())
        lines.append(f"| {name} | {n:,} | {lr_wrong[mask].mean():.1%} | {gru_wrong[mask].mean():.1%} |")
    both = (lr_wrong & gru_wrong).sum()
    only_lr = (lr_wrong & ~gru_wrong).sum()
    only_gru = (~lr_wrong & gru_wrong).sum()
    lines += ["",
              f"- Both wrong: {both:,}; only LR wrong: {only_lr:,}; only GRU wrong: {only_gru:,}; "
              f"models disagree on {int(df['models_disagree'].sum()):,} reviews.",
              f"- Errors by true label: LR misses {int((lr_wrong & (df.label == 0)).sum()):,} negatives and "
              f"{int((lr_wrong & (df.label == 1)).sum()):,} positives; GRU misses "
              f"{int((gru_wrong & (df.label == 0)).sum()):,} negatives and {int((gru_wrong & (df.label == 1)).sum()):,} positives.\n"]

    if set_name == "test":
        # examples where exactly one model is right, for the presentation
        ex = df.assign(n_words=df["content"].str.split().str.len())
        ex = ex[ex["n_words"].between(3, 30)]
        for title, m in [("GRU right, LR wrong", ex["lr_pred"].ne(ex.label) & ex[f"gru{SEED}_pred"].eq(ex.label)),
                         ("LR right, GRU wrong", ex["lr_pred"].eq(ex.label) & ex[f"gru{SEED}_pred"].ne(ex.label))]:
            lines.append(f"**{title}** (random examples, 3-30 words; label 1 = positive):\n")
            for _, r in ex[m].sample(8, random_state=SEED).iterrows():
                lines.append(f"- label {r.label}, LR p={r.lr_prob:.2f}, GRU p={r[f'gru{SEED}_prob']:.2f}: "
                             f"\"{r.content.strip()[:160]}\"")
            lines.append("")

        errors = df[lr_wrong | gru_wrong].join(tags)
        sample = errors.sample(150, random_state=SEED)
        sample["wrong_model"] = ["both" if a and b else ("LR only" if a else "GRU only")
                                 for a, b in zip(lr_wrong[sample.index], gru_wrong[sample.index])]
        sample["manual_category"] = ""   # filled by reading each review
        cols = ["text_hash", "content", "label", "scores", "lr_prob", f"gru{SEED}_prob", "wrong_model",
                *tags.columns, "manual_category"]
        sample[cols].to_csv("outputs/error_sample_for_reading.csv", index=False)

Path("outputs/error_analysis.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
