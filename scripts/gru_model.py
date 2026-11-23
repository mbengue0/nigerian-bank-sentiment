"""
Shared GRU pieces: tokenizer, batching, model and prediction.
Imported by 05_gru.py (training) and 06_evaluate.py (final evaluation), so both use identical code.
"""
import random
import re

import numpy as np
import torch
import torch.nn as nn

PAD, UNK = 0, 1


def tokenize(text):
    """Lowercase; words keep apostrophes ("don't" is one token); every other
    non-space character (punctuation, emoji) becomes its own token."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]", text.lower())


def build_vocab(token_lists, max_vocab, min_freq):
    """Training data only. Ids 0 and 1 are reserved for <pad> and <unk>."""
    from collections import Counter
    counts = Counter(tok for toks in token_lists for tok in toks)
    words = [w for w, c in counts.most_common(max_vocab - 2) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(tokens, vocab, max_len):
    """Map tokens to ids, keeping the first max_len tokens; unknown words -> <unk>."""
    ids = [vocab.get(t, UNK) for t in tokens[:max_len]]
    return ids or [UNK]


def batches(token_lists, labels, vocab, max_len, batch_size, shuffle):
    """Yield (padded ids, true lengths, labels) one batch at a time."""
    order = list(range(len(token_lists)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        seqs = [encode(token_lists[i], vocab, max_len) for i in idx]
        lengths = torch.tensor([len(s) for s in seqs])
        x = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
        for row, s in enumerate(seqs):
            x[row, : len(s)] = torch.tensor(s)
        yield x, lengths, torch.tensor(np.asarray(labels)[idx], dtype=torch.float32)


class GRUClassifier(nn.Module):
    """Embedding -> one GRU layer -> last real hidden state -> dropout -> one logit."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # Packing tells the GRU each review's true length, so padding is never read.
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(x), lengths, batch_first=True, enforce_sorted=False)
        _, h_last = self.gru(packed)
        return self.head(self.dropout(h_last[-1])).squeeze(1)


def predict_proba(model, token_lists, vocab, cfg):
    """Probability of the positive class for each review (model in eval mode, no dropout)."""
    model.eval()
    out = []
    with torch.no_grad():
        for x, lengths, _ in batches(token_lists, np.zeros(len(token_lists)), vocab,
                                     cfg["max_len"], cfg["batch_size"], shuffle=False):
            out.append(torch.sigmoid(model(x, lengths)))
    return torch.cat(out).numpy()


def load_model(path):
    """Load a saved GRU checkpoint written by 05_gru.py."""
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    model = GRUClassifier(len(ckpt["vocab"]) + 2, cfg["embed_dim"], cfg["hidden_dim"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["vocab"], cfg
