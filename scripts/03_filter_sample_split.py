"""
Stage 2b: English filter, near-duplicate removal, 100,000-review sample, 80/10/10 split.

Input : data/interim/candidates_lang.csv.gz  (196,317 candidates + langid output)
Output: data/final/train.csv, val.csv, test.csv   - the fixed partitions used by BOTH models
        data/interim/filter_flags.csv.gz          - every candidate with its keep/drop reason
        outputs/prep_report.txt, outputs/removal_log_stage2.csv

English rule (chosen after reading samples):
  Split the review into words made of Latin letters (after Unicode NFKC + casefold,
  which also turns styled letters like "𝚃𝚑𝚒𝚜" into "this"). Keep the review if it has at
  least one such word AND at least 50% of its words are in the 50,000 most frequent
  English words of the wordfreq package. Emoji-only reviews have no words and are dropped.
  langid output is kept for reference but NOT used: it misjudged many short, emoji-rich
  or capitalised English reviews and has no Yoruba, Hausa, Igbo or Pidgin model.

Near-duplicates: reviews that become identical after also removing punctuation, emoji and
digits (e.g. "Great app!!" vs "great app") are treated as one text. Groups whose labels
conflict are removed; otherwise one review per group is kept. This prevents the same
text appearing in both training and test data.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from wordfreq import top_n_list

SEED = 42
TARGET = 100_000
MIN_EN_SHARE = 0.5
EN_WORDS = set(top_n_list("en", 50_000))

Path("data/final").mkdir(parents=True, exist_ok=True)
report, log = [], []
def say(line=""):
    print(line)
    report.append(line)

cand = pd.read_csv("data/interim/candidates_lang.csv.gz", keep_default_na=False)
say(f"candidates in: {len(cand):,}")

# ---------------------------------------------------------------- English filter
def latin_words(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)

words = cand["content"].map(latin_words)
cand["n_words"] = words.str.len()
cand["en_share"] = [sum(w in EN_WORDS for w in ws) / len(ws) if ws else 0.0 for ws in words]
no_words = cand["n_words"] == 0
low_share = ~no_words & (cand["en_share"] < MIN_EN_SHARE)
cand["drop_reason"] = ""
cand.loc[no_words, "drop_reason"] = "no Latin-letter words (e.g. emoji only)"
cand.loc[low_share, "drop_reason"] = "under 50% common English words"
log += [("no Latin-letter words", int(no_words.sum())), ("under 50% common English words", int(low_share.sum()))]
english = cand[cand["drop_reason"] == ""].copy()
say(f"after English filter: {len(english):,}  (dropped {int(no_words.sum()):,} no-word, {int(low_share.sum()):,} low-share)")

# ---------------------------------------------------------------- near-duplicates
english["loose_key"] = words[english.index].str.join(" ")
g = english.groupby("loose_key")["label"].transform("nunique")
conflict = g > 1
cand.loc[english.index[conflict], "drop_reason"] = "near-duplicate group with conflicting labels"
english = english[~conflict]
dup = english.duplicated("loose_key", keep="first")
cand.loc[english.index[dup], "drop_reason"] = "near-duplicate of a kept review"
english = english[~dup]
log += [("near-duplicate group with conflicting labels", int(conflict.sum())),
        ("near-duplicate of a kept review", int(dup.sum()))]
say(f"after near-duplicate removal: {len(english):,}")
say(f"  negative / positive: {(english.label == 0).sum():,} / {(english.label == 1).sum():,}")
cand[["text_hash", "label", "banks", "lang", "lang_prob", "n_words", "en_share", "drop_reason"]] \
    .to_csv("data/interim/filter_flags.csv.gz", index=False)

# ---------------------------------------------------------------- stratification key
# Texts posted under several banks get the group "multi" so their bank is not chosen arbitrarily.
english["strat_bank"] = english["banks"].where(~english["banks"].str.contains("|", regex=False), "multi")
english["strata"] = english["strat_bank"] + "_" + english["label"].astype(str)

# ---------------------------------------------------------------- 100k sample
if len(english) > TARGET:
    sample, _ = train_test_split(english, train_size=TARGET, stratify=english["strata"], random_state=SEED)
else:
    sample = english
    say(f"pool smaller than target; using all {len(sample):,}")
log.append(("not selected in 100k sample", len(english) - len(sample)))

# ---------------------------------------------------------------- 80 / 10 / 10
train, rest = train_test_split(sample, test_size=0.2, stratify=sample["strata"], random_state=SEED)
val, test = train_test_split(rest, test_size=0.5, stratify=rest["strata"], random_state=SEED)

cols = ["text_hash", "content", "label", "banks", "strat_bank", "years", "scores", "n_rows"]
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[cols].sort_values("text_hash").to_csv(f"data/final/{name}.csv", index=False)

# ---------------------------------------------------------------- checks + report
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    A, B = {"train": train, "val": val, "test": test}[a], {"train": train, "val": val, "test": test}[b]
    assert not set(A["loose_key"]) & set(B["loose_key"]), f"overlap {a}/{b}"
say("no exact or near-duplicate overlap between partitions: OK")

say(f"\nseed={SEED}  target={TARGET:,}  min English share={MIN_EN_SHARE}")
say(f"{'split':<7}{'rows':>8}{'neg':>8}{'pos':>8}{'pos %':>7}")
for name, part in [("train", train), ("val", val), ("test", test)]:
    say(f"{name:<7}{len(part):>8,}{(part.label == 0).sum():>8,}{(part.label == 1).sum():>8,}{part.label.mean() * 100:>7.1f}")
say("\nrows per bank group in the 100k sample (neg / pos):")
by_bank = sample.groupby(["strat_bank", "label"]).size().unstack(fill_value=0)
for bank, r in by_bank.iterrows():
    say(f"  {bank:<14}{r[0]:>7,} /{r[1]:>7,}")
say("\nword count quantiles (train): " + ", ".join(
    f"p{int(q * 100)}={int(train.n_words.quantile(q))}" for q in [0.5, 0.9, 0.95, 0.99]))

pd.DataFrame(log, columns=["step", "removed"]).to_csv("outputs/removal_log_stage2.csv", index=False)
Path("outputs/prep_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
