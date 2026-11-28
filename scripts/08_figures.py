"""
Stage 7: report figure. Reads outputs/final_results.json (from 06_evaluate.py); trains nothing.

Output: outputs/figures/confusion_all.png - confusion matrices for both models on both test sets
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = json.load(open("outputs/final_results.json"))
panels = [("test", "tfidf_lr", "TF-IDF + LR, main test"),
          ("test", "gru", "GRU (seed 42), main test"),
          ("unseen_zenithBank", "tfidf_lr", "TF-IDF + LR, unseen Zenith Bank"),
          ("unseen_zenithBank", "gru", "GRU (seed 42), unseen Zenith Bank")]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
for ax, (set_name, model, title) in zip(axes.flat, panels):
    r = f[set_name][model] if model == "tfidf_lr" else f[set_name]["gru"]["per_seed"]["42"]
    cm = r["confusion_rows_true_neg_pos_cols_pred_neg_pos"]   # rows = true, columns = predicted
    top = max(map(max, cm))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=top * 1.1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=12, family="serif",
                    color="white" if cm[i][j] > top / 2 else "black")
    ax.set_xticks([0, 1], ["negative", "positive"], family="serif")
    ax.set_yticks([0, 1], ["negative", "positive"], family="serif")
    ax.set_xlabel("Predicted label", family="serif")
    ax.set_ylabel("True label (star rating)", family="serif")
    ax.set_title(title, fontsize=10, family="serif", color="#1F3864")
fig.tight_layout()
fig.savefig("outputs/figures/confusion_all.png", dpi=200)
print("saved outputs/figures/confusion_all.png")
