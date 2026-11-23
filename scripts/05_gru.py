"""
Stage 4: trainable embeddings + one GRU layer + binary classification head.

  python3 scripts/05_gru.py --pilot   # 1 epoch on a 20,000-review training subset (feasibility only)
  python3 scripts/05_gru.py           # full training set, early stopping on validation macro-F1

Uses train (fit, vocabulary) and validation (early stopping) only. Test is not read here.

Model:  token ids -> Embedding(vocab, 100, padding_idx=0) -> GRU(100 -> 128, 1 layer)
        -> final hidden state of the real (unpadded) sequence -> Dropout(0.3) -> Linear(128 -> 1)
Padding: sequences are packed with pack_padded_sequence, so the GRU never reads pad tokens,
         and the pad embedding is fixed at zero (padding_idx).
"""
import argparse
import json
import platform
import random
import re
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

CFG = {
    "seed": args.seed, "max_vocab": 30_000, "min_freq": 2, "max_len": 100,
    "embed_dim": 100, "hidden_dim": 128, "dropout": 0.3,
    "batch_size": 64, "lr": 1e-3, "optimizer": "Adam", "loss": "BCEWithLogitsLoss",
    "max_epochs": 1 if args.pilot else 10, "patience": 2, "grad_clip": 1.0,
    "pilot_train_size": 20_000 if args.pilot else None, "torch_threads": 2,
}
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
torch.set_num_threads(CFG["torch_threads"])
PAD, UNK = 0, 1

train = pd.read_csv("data/final/train.csv", keep_default_na=False)
val = pd.read_csv("data/final/val.csv", keep_default_na=False)
if args.pilot:
    train = train.sample(CFG["pilot_train_size"], random_state=CFG["seed"])

# ---------------------------------------------------------------- tokenize + vocabulary
def tokenize(text):
    # words (with apostrophes, so "don't" stays one token); any other non-space char
    # (punctuation, emoji) becomes its own token
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]", text.lower())

train_tokens = [tokenize(t) for t in train["content"]]
counts = Counter(tok for toks in train_tokens for tok in toks)
vocab_words = [w for w, c in counts.most_common(CFG["max_vocab"] - 2) if c >= CFG["min_freq"]]
vocab = {w: i + 2 for i, w in enumerate(vocab_words)}          # 0 = <pad>, 1 = <unk>

def encode(tokens):
    ids = [vocab.get(t, UNK) for t in tokens[: CFG["max_len"]]]  # keep the first max_len tokens
    return ids or [UNK]

def to_batches(token_lists, labels, shuffle):
    order = list(range(len(token_lists)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), CFG["batch_size"]):
        idx = order[start:start + CFG["batch_size"]]
        seqs = [encode(token_lists[i]) for i in idx]
        lengths = torch.tensor([len(s) for s in seqs])
        x = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
        for row, s in enumerate(seqs):
            x[row, : len(s)] = torch.tensor(s)
        yield x, lengths, torch.tensor(labels[idx], dtype=torch.float32)

# ---------------------------------------------------------------- model
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CFG["embed_dim"], padding_idx=PAD)
        self.gru = nn.GRU(CFG["embed_dim"], CFG["hidden_dim"], num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(CFG["dropout"])
        self.head = nn.Linear(CFG["hidden_dim"], 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(x), lengths, batch_first=True, enforce_sorted=False)
        _, h_last = self.gru(packed)                 # h_last: (1, batch, hidden) at each true end
        return self.head(self.dropout(h_last[-1])).squeeze(1)   # one logit per review

model = GRUClassifier(len(vocab) + 2)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
loss_fn = nn.BCEWithLogitsLoss()

def predict(token_lists):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, lengths, _ in to_batches(token_lists, np.zeros(len(token_lists)), shuffle=False):
            preds.append((model(x, lengths) > 0).long())
    return torch.cat(preds).numpy()

# ---------------------------------------------------------------- train
val_tokens = [tokenize(t) for t in val["content"]]
y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
t_start = time.perf_counter()
for epoch in range(1, CFG["max_epochs"] + 1):
    model.train()
    t0, total = time.perf_counter(), 0.0
    for x, lengths, y in to_batches(train_tokens, y_train, shuffle=True):
        optimizer.zero_grad()
        loss = loss_fn(model(x, lengths), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        total += loss.item() * len(y)
    epoch_s = time.perf_counter() - t0
    val_f1 = f1_score(y_val, predict(val_tokens), average="macro")
    history.append({"epoch": epoch, "train_loss": round(total / len(y_train), 4),
                    "val_macro_f1": round(val_f1, 4), "epoch_seconds": round(epoch_s, 1)})
    print(history[-1], flush=True)
    if val_f1 > best_f1:
        best_f1, bad_epochs = val_f1, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= CFG["patience"]:
            print("early stopping"); break
train_s = time.perf_counter() - t_start

# ---------------------------------------------------------------- evaluate best epoch on validation
model.load_state_dict(best_state)
t0 = time.perf_counter()
pred = predict([tokenize(t) for t in val["content"]])    # timing includes tokenization
infer_s = time.perf_counter() - t0
truncated = float(np.mean([len(t) > CFG["max_len"] for t in train_tokens]))

result = {
    "run": "pilot (1 epoch, 20k-train subset; feasibility only, not a final result)" if args.pilot else "full",
    "config": CFG, "vocab_size_incl_pad_unk": len(vocab) + 2, "trainable_parameters": n_params,
    "train_reviews": len(train), "share_train_reviews_truncated": round(truncated, 4),
    "history": history, "best_epoch": max(history, key=lambda h: h["val_macro_f1"])["epoch"],
    "train_seconds_total": round(train_s, 1),
    "val_inference_seconds_10k_reviews": round(infer_s, 2),
    "process_peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "CPU", "threads": torch.get_num_threads()},
    "validation": {
        "macro_f1": round(f1_score(y_val, pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_val, pred), 4),
        "per_class": classification_report(y_val, pred, target_names=["negative", "positive"], output_dict=True),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y_val, pred).tolist(),
    },
}
name = "gru_pilot" if args.pilot else f"gru_full_seed{args.seed}"
Path("models").mkdir(exist_ok=True)
torch.save({"state_dict": best_state, "vocab": vocab, "config": CFG}, f"models/{name}.pt")
Path(f"outputs/{name}_results.json").write_text(json.dumps(result, indent=2))
v = result["validation"]
print(f"\n{name}: val macro-F1 {v['macro_f1']} acc {v['accuracy']}  params {n_params:,}")
print(f"train {train_s:.1f}s  val inference {infer_s:.2f}s  peak memory {result['process_peak_memory_mb']} MB")
print("confusion (rows true neg/pos, cols pred neg/pos):", v["confusion_matrix_rows_true_cols_pred"])
