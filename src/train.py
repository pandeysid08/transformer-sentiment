"""Train + evaluate.  Usage: python -m src.train --epochs 6"""
import argparse, json, os, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from .data import get_loaders
from .model import TransformerClassifier


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, sched=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                sched.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--max_vocab", type=int, default=20000)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--subset", type=int, default=None, help="use N samples for a quick test")
    p.add_argument("--source", choices=["hf", "kaggle"], default="hf", help="dataset source")
    p.add_argument("--csv", default=None, help="path to Kaggle IMDB CSV (with --source kaggle)")
    p.add_argument("--out", default="checkpoints")
    a = p.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    train_dl, val_dl, test_dl, vocab = get_loaders(a.max_len, a.max_vocab, a.batch_size, subset=a.subset,
                                                    source=a.source, csv_path=a.csv)
    cfg = dict(vocab_size=len(vocab), d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers,
               d_ff=a.d_ff, max_len=a.max_len, n_classes=2, dropout=a.dropout)
    model = TransformerClassifier(**cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=a.lr,
                                                total_steps=a.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = 0.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, device, criterion, optimizer, scaler, sched)
        vl, va = run_epoch(model, val_dl, device, criterion)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"Epoch {ep}/{a.epochs} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {time.time()-t0:.0f}s")
        if va > best:
            best = va
            torch.save({"model": model.state_dict(), "cfg": cfg, "vocab": vocab},
                       os.path.join(a.out, "best.pt"))

    # ---- Test evaluation with best checkpoint ----
    ckpt = torch.load(os.path.join(a.out, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    names = ["negative", "positive"]
    report = classification_report(ys, preds, target_names=names, digits=4)
    print("\nTEST RESULTS\n", report)
    open(os.path.join(a.out, "test_report.txt"), "w").write(report)
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))

    # ---- Plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig("assets/training_curves.png", dpi=150); plt.close()

    ConfusionMatrixDisplay(confusion_matrix(ys, preds), display_labels=names).plot(cmap="Blues")
    plt.title("Test confusion matrix"); plt.savefig("assets/confusion_matrix.png", dpi=150)
    print("Saved checkpoints/ and assets/ plots.")


if __name__ == "__main__":
    main()
