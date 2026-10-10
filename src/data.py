"""IMDB loading, word-level tokenizer, vocab and DataLoaders."""
import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.,]")


def tokenize(text: str):
    text = re.sub(r"<br\s*/?>", " ", text.lower())
    return _TOKEN_RE.findall(text)


def build_vocab(texts, max_vocab=20000, min_freq=2):
    counts = Counter(tok for t in texts for tok in tokenize(t))
    itos = [PAD, UNK] + [w for w, c in counts.most_common(max_vocab - 2) if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode(text, vocab, max_len):
    ids = [vocab.get(t, 1) for t in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))


def to_tensors(texts, labels, vocab, max_len):
    x = torch.tensor([encode(t, vocab, max_len) for t in texts], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def load_texts(source="hf", csv_path=None, subset=None, seed=42):
    """Return train/test texts+labels from Hugging Face IMDB or the Kaggle 50k CSV."""
    g = torch.Generator().manual_seed(seed)
    if source == "kaggle":
        import pandas as pd

        df = pd.read_csv(csv_path)  # columns: review, sentiment (positive/negative)
        texts = df["review"].tolist()
        labels = (df["sentiment"] == "positive").astype(int).tolist()
        idx = torch.randperm(len(texts), generator=g).tolist()
        cut = int(0.5 * len(idx))  # 50/50 split, same size as HF IMDB
        tr, te = idx[:cut], idx[cut:]
        tr_texts, tr_labels = [texts[i] for i in tr], [labels[i] for i in tr]
        te_texts, te_labels = [texts[i] for i in te], [labels[i] for i in te]
    else:
        from datasets import load_dataset

        ds = load_dataset("imdb")
        tr_texts, tr_labels = ds["train"]["text"], ds["train"]["label"]
        te_texts, te_labels = ds["test"]["text"], ds["test"]["label"]
    if subset:  # quick debug runs
        idx = torch.randperm(len(tr_texts), generator=g)[:subset].tolist()
        tr_texts, tr_labels = [tr_texts[i] for i in idx], [tr_labels[i] for i in idx]
        idx = torch.randperm(len(te_texts), generator=g)[:subset].tolist()
        te_texts, te_labels = [te_texts[i] for i in idx], [te_labels[i] for i in idx]
    return tr_texts, tr_labels, te_texts, te_labels


def get_loaders(max_len=256, max_vocab=20000, batch_size=64, val_frac=0.1, subset=None,
                seed=42, source="hf", csv_path=None):
    tr_texts, tr_labels, te_texts, te_labels = load_texts(source, csv_path, subset, seed)
    g = torch.Generator().manual_seed(seed)

    vocab = build_vocab(tr_texts, max_vocab)
    full = to_tensors(tr_texts, tr_labels, vocab, max_len)
    n_val = int(len(full) * val_frac)
    train_ds, val_ds = torch.utils.data.random_split(full, [len(full) - n_val, n_val], generator=g)
    test_ds = to_tensors(te_texts, te_labels, vocab, max_len)

    mk = lambda d, sh: DataLoader(d, batch_size=batch_size, shuffle=sh, num_workers=0)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), vocab
