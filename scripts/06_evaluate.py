"""
Stage 5: final evaluation. Run ONCE, after both pipelines are frozen.

Frozen models (trained on data/final/train.csv, settings chosen on val.csv only):
  - majority class (always predicts the most frequent training label)
  - TF-IDF + logistic regression: models/tfidf_lr.joblib   (from 04_baselines.py)
  - GRU, seeds 42, 1, 2: models/gru_full_seed*.pt        (from 05_gru.py)
Seed 42 is the GRU used for per-review error analysis (fixed in advance, not picked by score).

Evaluated on two sets that no model has seen:
  - test.csv               : 10,000 new reviews from the 15 training banks (main experiment)
  - unseen_zenithBank.csv  : 12,041 reviews of Zenith Bank, never used in training (extension)

Output: outputs/final_results.json, outputs/final_results.md,
        outputs/predictions_test.csv, outputs/predictions_unseen_zenithBank.csv,
        outputs/figures/confusion_*.png
"""
import json
import platform
import statistics
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from gru_model import load_model, predict_proba, tokenize

torch.set_num_threads(2)
SEEDS = [42, 1, 2]
ANALYSIS_SEED = 42
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/final/train.csv", keep_default_na=False)
sets = {
    "test": pd.read_csv("data/final/test.csv", keep_default_na=False),
    "unseen_zenithBank": pd.read_csv("data/final/unseen_zenithBank.csv", keep_default_na=False),
}

def metrics(y, pred):
    rep = classification_report(y, pred, target_names=["negative", "positive"], output_dict=True, zero_division=0)
    return {
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "negative": {k: round(rep["negative"][k], 4) for k in ["precision", "recall", "f1-score"]},
        "positive": {k: round(rep["positive"][k], 4) for k in ["precision", "recall", "f1-score"]},
        "confusion_rows_true_neg_pos_cols_pred_neg_pos": confusion_matrix(y, pred).tolist(),
    }

def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label (from star rating)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

lr = joblib.load("models/tfidf_lr.joblib")
gru = {s: load_model(f"models/gru_full_seed{s}.pt") for s in SEEDS}
majority = int(train["label"].mode()[0])

results = {"environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "torch": torch.__version__, "device": "CPU", "torch_threads": torch.get_num_threads(),
                           "timing": "time.perf_counter wall-clock; includes TF-IDF transform / tokenization of raw text"},
           "majority_label": majority}

for set_name, df in sets.items():
    y = df["label"].to_numpy()
    out = {"n_reviews": len(df), "n_negative": int((y == 0).sum()), "n_positive": int((y == 1).sum())}
    preds = df[["text_hash", "content", "label", "banks", "scores"]].copy()

    out["majority_class"] = metrics(y, np.full(len(y), majority))

    t0 = time.perf_counter()
    lr_prob = lr.predict_proba(df["content"])[:, 1]
    lr_s = time.perf_counter() - t0
    out["tfidf_lr"] = metrics(y, (lr_prob >= 0.5).astype(int))
    out["tfidf_lr"]["inference_seconds"] = round(lr_s, 3)
    preds["lr_prob"] = lr_prob.round(4)
    preds["lr_pred"] = (lr_prob >= 0.5).astype(int)

    per_seed = {}
    for s in SEEDS:
        model, vocab, cfg = gru[s]
        t0 = time.perf_counter()
        prob = predict_proba(model, [tokenize(t) for t in df["content"]], vocab, cfg)
        secs = time.perf_counter() - t0
        per_seed[s] = metrics(y, (prob >= 0.5).astype(int))
        per_seed[s]["inference_seconds"] = round(secs, 3)
        preds[f"gru{s}_prob"] = prob.round(4)
        preds[f"gru{s}_pred"] = (prob >= 0.5).astype(int)
    f1s = [per_seed[s]["macro_f1"] for s in SEEDS]
    accs = [per_seed[s]["accuracy"] for s in SEEDS]
    out["gru"] = {"per_seed": per_seed,
                  "macro_f1_mean": round(statistics.mean(f1s), 4), "macro_f1_sd": round(statistics.stdev(f1s), 4),
                  "accuracy_mean": round(statistics.mean(accs), 4),
                  "inference_seconds_mean": round(statistics.mean(per_seed[s]["inference_seconds"] for s in SEEDS), 3)}

    preds["models_disagree"] = (preds["lr_pred"] != preds[f"gru{ANALYSIS_SEED}_pred"]).astype(int)
    preds.to_csv(f"outputs/predictions_{set_name}.csv", index=False)
    plot_cm(out["tfidf_lr"]["confusion_rows_true_neg_pos_cols_pred_neg_pos"],
            f"TF-IDF + LR, {set_name}", FIG / f"confusion_lr_{set_name}.png")
    plot_cm(per_seed[ANALYSIS_SEED]["confusion_rows_true_neg_pos_cols_pred_neg_pos"],
            f"GRU (seed {ANALYSIS_SEED}), {set_name}", FIG / f"confusion_gru_{set_name}.png")
    results[set_name] = out

# training cost, recorded by the training scripts on the same machine
base = json.loads(Path("outputs/baseline_results.json").read_text())
gru_runs = [json.loads(Path(f"outputs/gru_full_seed{s}_results.json").read_text()) for s in SEEDS]
results["training_cost"] = {
    "tfidf_lr_train_seconds": base["tfidf_lr_selected"]["train_seconds"],
    "tfidf_lr_peak_memory_mb": base["tfidf_lr_selected"]["process_peak_memory_mb (whole script)"],
    "gru_train_seconds_per_seed": [r["train_seconds_total"] for r in gru_runs],
    "gru_best_epoch_per_seed": [r["best_epoch"] for r in gru_runs],
    "gru_val_macro_f1_per_seed": [r["validation"]["macro_f1"] for r in gru_runs],
    "gru_peak_memory_mb_per_seed": [r["process_peak_memory_mb"] for r in gru_runs],
    "gru_trainable_parameters": gru_runs[0]["trainable_parameters"],
}
Path("outputs/final_results.json").write_text(json.dumps(results, indent=2))

# short markdown summary
lines = ["# Final results (single evaluation of frozen models)\n"]
for set_name in sets:
    r = results[set_name]
    lines += [f"## {set_name}: {r['n_reviews']:,} reviews ({r['n_negative']:,} neg / {r['n_positive']:,} pos)\n",
              "| Model | Macro-F1 | Accuracy | Neg F1 | Pos F1 | Inference (s) |", "|---|---:|---:|---:|---:|---:|",
              f"| Majority class | {r['majority_class']['macro_f1']:.3f} | {r['majority_class']['accuracy']:.3f} | "
              f"{r['majority_class']['negative']['f1-score']:.3f} | {r['majority_class']['positive']['f1-score']:.3f} | – |",
              f"| TF-IDF + LR | {r['tfidf_lr']['macro_f1']:.3f} | {r['tfidf_lr']['accuracy']:.3f} | "
              f"{r['tfidf_lr']['negative']['f1-score']:.3f} | {r['tfidf_lr']['positive']['f1-score']:.3f} | "
              f"{r['tfidf_lr']['inference_seconds']:.2f} |"]
    for s in SEEDS:
        g = r["gru"]["per_seed"][s]
        lines.append(f"| GRU seed {s} | {g['macro_f1']:.3f} | {g['accuracy']:.3f} | {g['negative']['f1-score']:.3f} | "
                     f"{g['positive']['f1-score']:.3f} | {g['inference_seconds']:.2f} |")
    lines.append(f"| GRU mean ± sd | {r['gru']['macro_f1_mean']:.3f} ± {r['gru']['macro_f1_sd']:.3f} | "
                 f"{r['gru']['accuracy_mean']:.3f} | | | {r['gru']['inference_seconds_mean']:.2f} |\n")
tc = results["training_cost"]
lines += ["## Training cost (same 2-core CPU)\n",
          f"- TF-IDF + LR: {tc['tfidf_lr_train_seconds']} s, peak process memory {tc['tfidf_lr_peak_memory_mb']} MB",
          f"- GRU: {tc['gru_train_seconds_per_seed']} s per seed, best epochs {tc['gru_best_epoch_per_seed']}, "
          f"peak memory {tc['gru_peak_memory_mb_per_seed']} MB, {tc['gru_trainable_parameters']:,} parameters"]
Path("outputs/final_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
