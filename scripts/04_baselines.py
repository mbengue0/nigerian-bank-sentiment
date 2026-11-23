"""
Stage 3: majority-class reference and TF-IDF + logistic regression.

Uses ONLY train (fit) and validation (choose settings). The test split is not read here;
it is used once, at the end, after both pipelines are frozen.

Tuning budget (small, fixed in advance): 2 n-gram ranges x 3 values of C = 6 fits.
TF-IDF turns text into feature values (computed from training text only, no labels);
logistic regression then learns one weight per feature from the labels.

Output: outputs/baseline_results.json, models/tfidf_lr.joblib
"""
import json
import platform
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

SEED = 42
Path("models").mkdir(exist_ok=True)
train = pd.read_csv("data/final/train.csv", keep_default_na=False)
val = pd.read_csv("data/final/val.csv", keep_default_na=False)

def evaluate(y_true, y_pred):
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "per_class": classification_report(y_true, y_pred, target_names=["negative", "positive"],
                                           output_dict=True, zero_division=0),
        # rows = true label (neg, pos), columns = predicted label (neg, pos)
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y_true, y_pred).tolist(),
    }

results = {"environment": {
    "python": platform.python_version(), "sklearn": sklearn.__version__,
    "cpu": platform.processor() or platform.machine(), "logical_cpus": __import__("os").cpu_count(),
    "timing": "time.perf_counter wall-clock; inference time includes TF-IDF transform of raw text",
}}

# ---------------------------------------------------------------- majority class
dummy = DummyClassifier(strategy="most_frequent").fit(train["content"], train["label"])
results["majority_class"] = {"predicts": int(dummy.predict(val["content"][:1])[0]),
                             "validation": evaluate(val["label"], dummy.predict(val["content"]))}
print("majority class  val macro-F1:", results["majority_class"]["validation"]["macro_f1"])

# ---------------------------------------------------------------- small grid on validation
grid = []
for ngram in [(1, 1), (1, 2)]:
    for C in [0.25, 1.0, 4.0]:
        model = make_pipeline(
            # lowercase=True; token pattern keeps words of 2+ letters/digits. No stop-word removal,
            # so negation words such as "not", "no", "never" stay in.
            TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True),
            LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=SEED),
        )
        t0 = time.perf_counter()
        model.fit(train["content"], train["label"])
        fit_s = time.perf_counter() - t0
        f1 = f1_score(val["label"], model.predict(val["content"]), average="macro")
        grid.append({"ngram_range": list(ngram), "C": C, "val_macro_f1": round(f1, 4),
                     "fit_seconds": round(fit_s, 2),
                     "n_features": len(model[0].vocabulary_)})
        print(grid[-1])
results["tfidf_lr_grid"] = grid

# ---------------------------------------------------------------- refit chosen setting, time it
best = max(grid, key=lambda g: g["val_macro_f1"])
model = make_pipeline(
    TfidfVectorizer(ngram_range=tuple(best["ngram_range"]), min_df=2, sublinear_tf=True),
    LogisticRegression(C=best["C"], max_iter=2000, solver="liblinear", random_state=SEED),
)
t0 = time.perf_counter()
model.fit(train["content"], train["label"])
fit_s = time.perf_counter() - t0
t0 = time.perf_counter()
pred = model.predict(val["content"])
infer_s = time.perf_counter() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

results["tfidf_lr_selected"] = {
    "settings": {"ngram_range": best["ngram_range"], "C": best["C"], "min_df": 2,
                 "sublinear_tf": True, "solver": "liblinear (single thread)"},
    "n_features": len(model[0].vocabulary_),
    "train_seconds": round(fit_s, 2),
    "val_inference_seconds_10k_reviews": round(infer_s, 3),
    "process_peak_memory_mb (whole script)": round(peak_mb),
    "validation": evaluate(val["label"], pred),
}
joblib.dump(model, "models/tfidf_lr.joblib")

# top-weighted features, to explain the model
names = np.array(model[0].get_feature_names_out())
coef = model[1].coef_[0]
results["tfidf_lr_selected"]["top_positive_features"] = names[np.argsort(coef)[-15:][::-1]].tolist()
results["tfidf_lr_selected"]["top_negative_features"] = names[np.argsort(coef)[:15]].tolist()

Path("outputs/baseline_results.json").write_text(json.dumps(results, indent=2))
sel = results["tfidf_lr_selected"]
print(f"\nselected {best}\nval macro-F1 {sel['validation']['macro_f1']}  acc {sel['validation']['accuracy']}")
print("confusion (rows true neg/pos, cols pred neg/pos):", sel["validation"]["confusion_matrix_rows_true_cols_pred"])
print("train s", sel["train_seconds"], " infer s", sel["val_inference_seconds_10k_reviews"], " peak MB", sel["process_peak_memory_mb (whole script)"])
print("top +", sel["top_positive_features"]); print("top -", sel["top_negative_features"])
