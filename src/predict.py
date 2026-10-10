"""Inference.  Usage: python -m src.predict "This movie was fantastic!" """
import sys
import torch
from .data import encode
from .model import TransformerClassifier


def load(path="checkpoints/best.pt"):
    ckpt = torch.load(path, map_location="cpu")
    model = TransformerClassifier(**ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["vocab"], ckpt["cfg"]["max_len"]


def predict(text, model, vocab, max_len):
    ids = torch.tensor([encode(text, vocab, max_len)])
    with torch.no_grad():
        probs = torch.softmax(model(ids), dim=-1)[0]
    label = "positive" if probs[1] > probs[0] else "negative"
    return label, probs.max().item()


if __name__ == "__main__":
    model, vocab, max_len = load()
    samples = sys.argv[1:] or ["An absolute masterpiece, loved every minute.",
                               "Boring, predictable and a waste of time."]
    for t in samples:
        label, conf = predict(t, model, vocab, max_len)
        print(f"{label} ({conf:.1%}) <- {t}")
