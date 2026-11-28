"""
CS-4714 Final Project - Mouhamadou Mbengue
Sentiment Classification of Nigerian Banking App Reviews:
Comparing Logistic Regression and GRU Models

HOW TO RUN
    pip install pandas numpy scikit-learn torch scipy matplotlib
    python code.py

    Reads dataset.csv from the same folder as this file, trains both models, evaluates them once
    on the test set and on the unseen-bank set, runs the error analysis, and writes everything
    to a "results" folder next to this file. About 15-20 minutes on a 2-core CPU (no GPU needed);
    almost all of it is the three GRU training runs.

    Options:
      python code.py --quick          one GRU run instead of three (about 5-7 minutes)
      python code.py --rebuild-data   rebuild dataset.csv from the raw Hugging Face files first
                                      (needs internet and: pip install wordfreq)

WHAT dataset.csv CONTAINS
    112,041 Google Play reviews of Nigerian banking apps, already cleaned and split:
      split = train (80,000) / val (10,000) / test (10,000) from 15 banks,
              unseen_zenith (12,041) from Zenith Bank, never used for training (extension).
      label = 0 negative (1-2 stars) / 1 positive (4-5 stars).
      manual_error_category = hand-read category for the 150 sampled test errors (empty otherwise).
    Source: Federal University Lokoja, "Nigerian Banks: Bank Reviews Dataset Collection",
    Hugging Face, revision 27d2b6e7d541a0b713dc8eed56b0daf27476d2a7 (Apache-2.0).

SECTIONS
    1. Data preparation (only with --rebuild-data)    5. Final evaluation (test + unseen bank)
    2. Load the fixed data sets                       6. Error analysis
    3. Majority class + TF-IDF / logistic regression  7. Summary
    4. Embeddings + GRU
"""
import argparse
import hashlib
import json
import platform
import random
import re
import resource
import statistics
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------- dependency check
missing = []
for module in ["pandas", "numpy", "sklearn", "torch", "scipy", "matplotlib"]:
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    sys.exit("Missing packages: " + ", ".join(missing) +
             "\nInstall them with:  pip install pandas numpy scikit-learn torch scipy matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from scipy.stats import chi2
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent          # works wherever the folder is unzipped
DATASET = HERE / "dataset.csv"
RESULTS = HERE / "results"
SEED = 42

parser = argparse.ArgumentParser(description="Logistic regression vs GRU on Nigerian banking app reviews")
parser.add_argument("--quick", action="store_true", help="train one GRU (seed 42) instead of three")
parser.add_argument("--rebuild-data", action="store_true", help="rebuild dataset.csv from the raw Hugging Face files")
args = parser.parse_args()
RESULTS.mkdir(exist_ok=True)

report_lines = []
def say(line=""):
    print(line, flush=True)
    report_lines.append(line)


# =============================================================================
# 1. DATA PREPARATION  (only with --rebuild-data; dataset.csv is already provided)
# =============================================================================
def rebuild_dataset():
    """Raw Hugging Face files -> cleaned, de-duplicated, filtered, sampled and split dataset.csv."""
    import urllib.request
    try:
        from wordfreq import top_n_list
    except ImportError:
        sys.exit("--rebuild-data needs the wordfreq package:  pip install wordfreq")

    repo = "https://huggingface.co/datasets/Federal-University-Lokoja/Bank-Review"
    revision = "27d2b6e7d541a0b713dc8eed56b0daf27476d2a7"     # pinned: the data can never silently change
    files = ["access_reviews.csv", "ecoBank_reviews.csv", "fbn_reviews.csv", "fcmb_reviews.csv",
             "fidelityBank_reviews.csv", "gtb_reviews.csv", "jaizBank_reviews.csv", "keyStoneBank_reviews.csv",
             "polarisBank_reviews.csv", "stanbicIbtc_reviews.csv", "sterlingBank_reviews.csv", "uba_reviews.csv",
             "unionBank_reviews.csv", "unityBank_reviews.csv", "wemaBank_reviews.csv", "zenithBank_reviews.csv"]
    raw_dir = HERE / "raw_data"
    raw_dir.mkdir(exist_ok=True)

    # --- 1a. download and merge the 16 files; bank comes from the file name
    frames = []
    for name in files:
        target = raw_dir / name
        if not target.exists():
            urllib.request.urlretrieve(f"{repo}/resolve/{revision}/{name}", target)
        # keep_default_na=False: a review that literally says "NA" or "None" stays text
        df = pd.read_csv(target, dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace")
        df["bank"] = name.replace("_reviews.csv", "")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    say(f"raw reviews: {len(data):,}")

    # --- 1b. labels: 1-2 stars -> 0, 4-5 stars -> 1, 3 stars excluded; empty texts removed
    data["score_num"] = pd.to_numeric(data["score"], errors="coerce")
    data["norm"] = data["content"].map(lambda t: re.sub(r"\s+", " ", t.strip()).casefold())
    binary = data[(data["norm"] != "") & data["score_num"].isin([1, 2, 4, 5])].copy()
    binary["label"] = (binary["score_num"] >= 4).astype(int)
    say(f"non-empty, ratings 1/2/4/5: {len(binary):,}")

    # --- 1c. one row per normalized text; remove texts that received both labels
    groups = binary.groupby("norm", sort=False).agg(
        n_rows=("label", "size"), n_labels=("label", "nunique"), label=("label", "first"),
        content=("content", "first"), banks=("bank", lambda s: "|".join(sorted(set(s)))),
        years=("year", lambda s: "|".join(sorted(set(s)))), scores=("score", lambda s: "|".join(sorted(set(s)))),
    ).reset_index()
    say(f"distinct texts: {len(groups):,}")
    cand = groups[groups["n_labels"] == 1].drop(columns="n_labels").reset_index(drop=True)
    cand["text_hash"] = cand["norm"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    say(f"after removing conflicting labels: {len(cand):,}")

    # --- 1d. English filter: >= 50% of Latin-letter words among the 50,000 most frequent English words
    en_words = set(top_n_list("en", 50_000))
    def latin_words(text):
        return re.findall(r"[a-z]+(?:'[a-z]+)?", unicodedata.normalize("NFKC", text).casefold())
    words = cand["content"].map(latin_words)
    share = pd.Series([sum(w in en_words for w in ws) / len(ws) if ws else 0.0 for ws in words], index=cand.index)
    english = cand[(words.str.len() > 0) & (share >= 0.5)].copy()
    say(f"after English filter: {len(english):,}")

    # --- 1e. near-duplicates: same text once punctuation, digits and emoji are ignored
    english["loose_key"] = words[english.index].str.join(" ")
    english["banks"] = english.groupby("loose_key")["banks"].transform(       # banks of the whole group
        lambda s: "|".join(sorted(set("|".join(s).split("|")))))
    english = english[english.groupby("loose_key")["label"].transform("nunique") == 1]
    english = english[~english.duplicated("loose_key", keep="first")]
    say(f"after near-duplicate removal: {len(english):,}")

    # --- 1f. extension: hold out Zenith Bank; drop texts it shares with other banks
    holdout = "zenithBank"
    has_holdout = english["banks"].str.split("|").apply(lambda b: holdout in b)
    only_holdout = english["banks"] == holdout
    unseen = english[only_holdout].copy()
    say(f"Zenith Bank held out: {len(unseen):,}; shared texts dropped: {int((has_holdout & ~only_holdout).sum()):,}")
    english = english[~has_holdout].copy()

    # --- 1g. seeded 100,000 sample and 80/10/10 split, stratified by bank and label
    english["strat_bank"] = english["banks"].where(~english["banks"].str.contains("|", regex=False), "multi")
    english["strata"] = english["strat_bank"] + "_" + english["label"].astype(str)
    sample, _ = train_test_split(english, train_size=100_000, stratify=english["strata"], random_state=SEED)
    train, rest = train_test_split(sample, test_size=0.2, stratify=sample["strata"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["strata"], random_state=SEED)
    unseen["strat_bank"] = holdout
    for a, b in [(train, val), (train, test), (val, test), (train, unseen), (val, unseen), (test, unseen)]:
        assert not set(a["loose_key"]) & set(b["loose_key"]), "overlap between sets"

    parts = []
    for split, part in [("train", train), ("val", val), ("test", test), ("unseen_zenith", unseen)]:
        part = part.sort_values("text_hash").rename(columns={"content": "text", "strat_bank": "bank_group"})
        part.insert(0, "split", split)
        parts.append(part[["split", "text", "label", "banks", "bank_group", "years", "scores", "text_hash"]])
    new = pd.concat(parts, ignore_index=True)
    if DATASET.exists():   # keep the hand-read error categories, which cannot be recomputed
        old = pd.read_csv(DATASET, keep_default_na=False)
        new["manual_error_category"] = new["text_hash"].map(dict(zip(old.text_hash, old.manual_error_category))).fillna("")
    else:
        new["manual_error_category"] = ""
    new.to_csv(DATASET, index=False)
    say(f"wrote {DATASET.name}: {len(new):,} reviews")


if args.rebuild_data or not DATASET.exists():
    say("== 1. Data preparation ==")
    rebuild_dataset()


# =============================================================================
# 2. LOAD THE FIXED DATA SETS
# =============================================================================
data = pd.read_csv(DATASET, keep_default_na=False)
train = data[data.split == "train"].reset_index(drop=True)
val = data[data.split == "val"].reset_index(drop=True)
test = data[data.split == "test"].reset_index(drop=True)
unseen = data[data.split == "unseen_zenith"].reset_index(drop=True)
EVAL_SETS = {"test": test, "unseen_zenith": unseen}
say("== 2. Data ==")
for name, df in [("train", train), ("val", val), ("test", test), ("unseen_zenith", unseen)]:
    say(f"{name:<14}{len(df):>8,} reviews   negative {(df.label == 0).sum():>6,}   positive {(df.label == 1).sum():>6,}")

def metrics(y, pred):
    rep = classification_report(y, pred, target_names=["negative", "positive"], output_dict=True, zero_division=0)
    return {"macro_f1": round(f1_score(y, pred, average="macro"), 4), "accuracy": round(accuracy_score(y, pred), 4),
            "negative": {k: round(rep["negative"][k], 4) for k in ["precision", "recall", "f1-score"]},
            "positive": {k: round(rep["positive"][k], 4) for k in ["precision", "recall", "f1-score"]},
            "confusion_rows_true_cols_pred": confusion_matrix(y, pred).tolist()}


# =============================================================================
# 3. MAJORITY CLASS + TF-IDF / LOGISTIC REGRESSION  (fit on train, choose on validation)
# =============================================================================
say("\n== 3. Majority class and TF-IDF + logistic regression ==")
majority = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
say(f"majority class predicts {majority.predict(['x'])[0]}; validation macro-F1 "
    f"{f1_score(val.label, majority.predict(val.text), average='macro'):.4f}")

def make_lr(ngram, C):
    # TF-IDF features come from training text only. No stop-word removal, so "not", "no", "never" stay.
    return make_pipeline(TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True),
                         LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=SEED))

grid = []   # small, fixed tuning budget: 2 n-gram ranges x 3 values of C
for ngram in [(1, 1), (1, 2)]:
    for C in [0.25, 1.0, 4.0]:
        model = make_lr(ngram, C).fit(train["text"], train["label"])
        f1 = f1_score(val["label"], model.predict(val["text"]), average="macro")
        grid.append({"ngram_range": ngram, "C": C, "val_macro_f1": round(f1, 4), "n_features": len(model[0].vocabulary_)})
        say(f"  ngrams {ngram}, C={C}: validation macro-F1 {f1:.4f} ({len(model[0].vocabulary_):,} features)")
best = max(grid, key=lambda g: g["val_macro_f1"])
t0 = time.perf_counter()
lr = make_lr(best["ngram_range"], best["C"]).fit(train["text"], train["label"])
lr_train_s = time.perf_counter() - t0
lr_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if sys.platform != "darwin" else \
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
say(f"selected ngrams {best['ngram_range']}, C={best['C']}; trained in {lr_train_s:.2f} s")
names = np.array(lr[0].get_feature_names_out())
coef = lr[1].coef_[0]
say("strongest positive features: " + ", ".join(names[np.argsort(coef)[-15:][::-1]]))
say("strongest negative features: " + ", ".join(names[np.argsort(coef)[:15]]))


# =============================================================================
# 4. EMBEDDINGS + GRU  (vocabulary from train, early stopping on validation)
# =============================================================================
PAD, UNK = 0, 1
GRU_CFG = {"max_vocab": 30_000, "min_freq": 2, "max_len": 100, "embed_dim": 100, "hidden_dim": 128,
           "dropout": 0.3, "batch_size": 64, "lr": 1e-3, "max_epochs": 10, "patience": 2, "grad_clip": 1.0}
torch.set_num_threads(2)   # fixed thread count, as in the reported runs

def tokenize(text):
    """Lowercase; words keep apostrophes ("don't" is one token); punctuation and emoji are their own tokens."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]", text.lower())

def build_vocab(token_lists):
    counts = Counter(tok for toks in token_lists for tok in toks)
    words = [w for w, c in counts.most_common(GRU_CFG["max_vocab"] - 2) if c >= GRU_CFG["min_freq"]]
    return {w: i + 2 for i, w in enumerate(words)}          # ids 0 and 1 are <pad> and <unk>

def batches(token_lists, labels, vocab, shuffle):
    order = list(range(len(token_lists)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), GRU_CFG["batch_size"]):
        idx = order[start:start + GRU_CFG["batch_size"]]
        seqs = [[vocab.get(t, UNK) for t in token_lists[i][:GRU_CFG["max_len"]]] or [UNK] for i in idx]
        lengths = torch.tensor([len(s) for s in seqs])
        x = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
        for row, s in enumerate(seqs):
            x[row, :len(s)] = torch.tensor(s)
        yield x, lengths, torch.tensor(np.asarray(labels)[idx], dtype=torch.float32)

class GRUClassifier(nn.Module):
    """Embedding -> one GRU layer -> hidden state at the last real token -> dropout -> one logit."""
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, GRU_CFG["embed_dim"], padding_idx=PAD)
        self.gru = nn.GRU(GRU_CFG["embed_dim"], GRU_CFG["hidden_dim"], num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(GRU_CFG["dropout"])
        self.head = nn.Linear(GRU_CFG["hidden_dim"], 1)

    def forward(self, x, lengths):
        # packing gives the GRU each review's true length, so padding is never read
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(x), lengths, batch_first=True, enforce_sorted=False)
        _, h_last = self.gru(packed)
        return self.head(self.dropout(h_last[-1])).squeeze(1)

def gru_proba(model, token_lists, vocab):
    model.eval()
    out = []
    with torch.no_grad():
        for x, lengths, _ in batches(token_lists, np.zeros(len(token_lists)), vocab, shuffle=False):
            out.append(torch.sigmoid(model(x, lengths)))
    return torch.cat(out).numpy()

say("\n== 4. Embeddings + GRU ==")
train_tokens = [tokenize(t) for t in train["text"]]
val_tokens = [tokenize(t) for t in val["text"]]
vocab = build_vocab(train_tokens)
y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
say(f"vocabulary {len(vocab) + 2:,} entries; {np.mean([len(t) > GRU_CFG['max_len'] for t in train_tokens]):.1%} "
    f"of training reviews truncated at {GRU_CFG['max_len']} tokens")

seeds = [42] if args.quick else [42, 1, 2]
gru_models, gru_runs = {}, {}
for seed in seeds:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = GRUClassifier(len(vocab) + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_CFG["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []
    t_start = time.perf_counter()
    for epoch in range(1, GRU_CFG["max_epochs"] + 1):
        model.train()
        for x, lengths, y in batches(train_tokens, y_train, vocab, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRU_CFG["grad_clip"])
            optimizer.step()
        val_f1 = f1_score(y_val, (gru_proba(model, val_tokens, vocab) >= 0.5).astype(int), average="macro")
        history.append(round(val_f1, 4))
        say(f"  seed {seed}, epoch {epoch}: validation macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:                         # keep the best epoch
            best_f1, bad_epochs = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= GRU_CFG["patience"]:
                break
    model.load_state_dict(best_state)
    gru_models[seed] = model
    gru_runs[seed] = {"train_seconds": round(time.perf_counter() - t_start, 1), "best_epoch": history.index(max(history)) + 1,
                      "val_macro_f1": round(best_f1, 4)}
    say(f"  seed {seed}: best epoch {gru_runs[seed]['best_epoch']}, validation macro-F1 {best_f1:.4f}, "
        f"{gru_runs[seed]['train_seconds']} s")
n_params = sum(p.numel() for p in gru_models[42].parameters())
gru_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 ** 2)


# =============================================================================
# 5. FINAL EVALUATION: frozen models, test set and unseen Zenith Bank, used once
# =============================================================================
say("\n== 5. Final evaluation ==")
final = {}
for set_name, df in EVAL_SETS.items():
    y = df["label"].to_numpy()
    res = {"majority_class": metrics(y, majority.predict(df["text"]))}
    t0 = time.perf_counter()
    lr_prob = lr.predict_proba(df["text"])[:, 1]             # includes the TF-IDF transform
    res["tfidf_lr"] = metrics(y, (lr_prob >= 0.5).astype(int))
    res["tfidf_lr"]["inference_seconds"] = round(time.perf_counter() - t0, 3)
    df["lr_prob"] = lr_prob
    for seed, model in gru_models.items():
        t0 = time.perf_counter()
        prob = gru_proba(model, [tokenize(t) for t in df["text"]], vocab)   # includes tokenization
        res[f"gru_seed{seed}"] = metrics(y, (prob >= 0.5).astype(int))
        res[f"gru_seed{seed}"]["inference_seconds"] = round(time.perf_counter() - t0, 3)
        df[f"gru{seed}_prob"] = prob
    final[set_name] = res

    say(f"\n{set_name}: {len(df):,} reviews ({(y == 0).sum():,} negative / {(y == 1).sum():,} positive)")
    say(f"{'model':<22}{'macro-F1':>9}{'accuracy':>10}{'neg F1':>8}{'pos F1':>8}")
    for key, label in [("majority_class", "majority class"), ("tfidf_lr", "TF-IDF + LR")] + \
                      [(f"gru_seed{s}", f"GRU seed {s}") for s in gru_models]:
        r = res[key]
        say(f"{label:<22}{r['macro_f1']:>9.3f}{r['accuracy']:>10.3f}{r['negative']['f1-score']:>8.3f}{r['positive']['f1-score']:>8.3f}")
    if len(gru_models) > 1:
        f1s = [res[f"gru_seed{s}"]["macro_f1"] for s in gru_models]
        say(f"{'GRU mean +/- sd':<22}{statistics.mean(f1s):>9.3f} +/- {statistics.stdev(f1s):.3f}")

    # McNemar's test (continuity correction) on reviews where exactly one model is wrong
    lr_wrong = (lr_prob >= 0.5).astype(int) != y
    gru_wrong = (df["gru42_prob"].to_numpy() >= 0.5).astype(int) != y
    only_lr, only_gru = int((lr_wrong & ~gru_wrong).sum()), int((~lr_wrong & gru_wrong).sum())
    stat = (abs(only_lr - only_gru) - 1) ** 2 / (only_lr + only_gru)
    res["mcnemar"] = {"only_lr_wrong": only_lr, "only_gru_wrong": only_gru, "chi2": round(stat, 2), "p": round(chi2.sf(stat, 1), 3)}
    say(f"McNemar LR vs GRU (seed 42): only LR wrong {only_lr}, only GRU wrong {only_gru}, "
        f"chi-square {stat:.2f}, p = {chi2.sf(stat, 1):.3f}")

    pred_cols = ["split", "text", "label", "banks", "scores", "text_hash", "lr_prob"] + [f"gru{s}_prob" for s in gru_models]
    df[pred_cols].round(4).to_csv(RESULTS / f"predictions_{set_name}.csv", index=False)

say("\ncost on this machine:")
say(f"  TF-IDF + LR: training {lr_train_s:.2f} s, predicting 10,000 test reviews "
    f"{final['test']['tfidf_lr']['inference_seconds']:.2f} s")
say(f"  GRU: training {[gru_runs[s]['train_seconds'] for s in gru_models]} s per run, predicting 10,000 test reviews "
    f"{statistics.mean(final['test'][f'gru_seed{s}']['inference_seconds'] for s in gru_models):.2f} s, "
    f"{n_params:,} parameters")
say(f"  peak process memory: {lr_peak_mb:.0f} MB after logistic regression, {gru_peak_mb:.0f} MB after the GRU")

# confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
for ax, (set_name, key, title) in zip(axes.flat, [
        ("test", "tfidf_lr", "TF-IDF + LR, main test"), ("test", "gru_seed42", "GRU (seed 42), main test"),
        ("unseen_zenith", "tfidf_lr", "TF-IDF + LR, unseen Zenith Bank"),
        ("unseen_zenith", "gru_seed42", "GRU (seed 42), unseen Zenith Bank")]):
    cm = final[set_name][key]["confusion_rows_true_cols_pred"]
    top = max(map(max, cm))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=top * 1.1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=12, color="white" if cm[i][j] > top / 2 else "black")
    ax.set_xticks([0, 1], ["negative", "positive"]); ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label (star rating)"); ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(RESULTS / "confusion_matrices.png", dpi=200)
plt.close(fig)


# =============================================================================
# 6. ERROR ANALYSIS (main test set; GRU = seed 42)
# =============================================================================
say("\n== 6. Error analysis ==")
NEGATION = re.compile(r"\b(?:not|no|never|nothing|none|nobody|cannot|can't|cant|don't|dont|doesn't|doesnt|"
                      r"didn't|didnt|isn't|isnt|won't|wont|wasn't|wasnt|n't)\b")
CONTRAST = re.compile(r"\b(?:but|however|although|though|except|yet)\b")
PIDGIN = re.compile(r"\b(?:dey|abeg|wetin|una|wahala|oga|sef|comot|abi|na wa|e no|no dey|dem|shey|sha|o|ooo|jare|mtchew)\b")
for set_name, df in EVAL_SETS.items():
    text = df["text"].str.lower()
    types = {
        "all reviews": pd.Series(True, index=df.index),
        "negation": text.str.contains(NEGATION),
        "contrast (but, however...)": text.str.contains(CONTRAST),
        "very short (1-3 words)": df["text"].str.split().str.len() <= 3,
        "truncated (>100 tokens)": df["text"].map(lambda t: len(tokenize(t))) > GRU_CFG["max_len"],
        "Pidgin markers": text.str.contains(PIDGIN),
    }
    lr_wrong = (df["lr_prob"] >= 0.5).astype(int) != df["label"]
    gru_wrong = (df["gru42_prob"] >= 0.5).astype(int) != df["label"]
    say(f"{set_name}: error rate by review type")
    say(f"  {'type':<28}{'reviews':>8}{'LR':>8}{'GRU':>8}")
    for name, mask in types.items():
        say(f"  {name:<28}{int(mask.sum()):>8,}{lr_wrong[mask].mean():>8.1%}{gru_wrong[mask].mean():>8.1%}")

# 150 random test errors were read by hand; their categories are stored in dataset.csv
manual = data[(data.split == "test") & (data.manual_error_category != "")]
if len(manual):
    say(f"\nhand-read categories of {len(manual)} random test errors:")
    for cat, n in manual["manual_error_category"].value_counts().items():
        say(f"  {cat:<55}{n:>4}  ({n / len(manual):.0%})")


# =============================================================================
# 7. SUMMARY
# =============================================================================
summary = {"environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "torch": torch.__version__,
                           "torch_threads": torch.get_num_threads()},
           "tfidf_lr_grid": [{**g, "ngram_range": list(g["ngram_range"])} for g in grid],
           "tfidf_lr_selected": {"ngram_range": list(best["ngram_range"]), "C": best["C"], "train_seconds": round(lr_train_s, 2)},
           "gru_config": GRU_CFG, "gru_parameters": n_params, "gru_runs": {str(k): v for k, v in gru_runs.items()},
           "final_evaluation": final}
(RESULTS / "results.json").write_text(json.dumps(summary, indent=2))
(RESULTS / "results.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
say(f"\nDone. Results saved in {RESULTS}")
