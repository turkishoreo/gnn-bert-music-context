"""
B2 baseline from spec Section 8: "CNN on mel-spectrogram (no graph, no text)",
trained on real GTZAN mel-spectrograms cached by prepare_gtzan.py, for direct
comparison against train_gtzan.py's GNN result (spec Task 2 deliverable:
"Comparison vs. CNN baseline on mel-spectrogram").

Run prepare_gtzan.py first (it caches both graphs AND mel-spectrograms). Then:
    python train_gtzan_cnn.py --config ../config.yaml --epochs 15
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, accuracy_score

from gnn_model import CNNMelBaseline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}


def load_split(splits_dir):
    with open(os.path.join(splits_dir, "gtzan_split.json")) as f:
        return json.load(f)


def load_mels(records):
    mels, labels = [], []
    for rec in records:
        m = torch.load(rec["mel_path"], weights_only=False)  # (n_mels, frames)
        mels.append(m)
        labels.append(GENRE_TO_IDX[rec["genre"]])
    x = torch.stack(mels).unsqueeze(1)  # (N, 1, n_mels, frames)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def batches(x, y, batch_size, shuffle=False):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i + batch_size]
        yield x[sel], y[sel]


@torch.no_grad()
def evaluate(model, x, y, batch_size=16):
    model.eval()
    preds = []
    for xb, yb in batches(x, y, batch_size):
        logits = model(xb)
        preds.append(logits.argmax(dim=-1))
    preds = torch.cat(preds).numpy()
    y_true = y.numpy()
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    return acc, macro_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                cfg["data"]["splits_dir"]))

    torch.manual_seed(cfg.get("train", {}).get("seed", 42))

    split = load_split(splits_dir)
    print(f"loading mel-spectrograms: {len(split['train'])} train / "
          f"{len(split['val'])} val / {len(split['test'])} test")
    x_train, y_train = load_mels(split["train"])
    x_val, y_val = load_mels(split["val"])
    x_test, y_test = load_mels(split["test"])

    model = CNNMelBaseline(num_tags=len(GENRES), n_mels=cfg["data"]["n_mels"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\n== Training CNN mel-spectrogram baseline (B2) on real GTZAN ==")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in batches(x_train, y_train, args.batch_size, shuffle=True):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_acc, val_f1 = evaluate(model, x_val, y_val, args.batch_size)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  "
              f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

    print("\n== Test-set results ==")
    test_acc, test_f1 = evaluate(model, x_test, y_test, args.batch_size)
    print(f"CNN mel-spec baseline (real GTZAN): accuracy={test_acc:.4f}   macro_f1={test_f1:.4f}")

    # merge into the same results file train_gtzan.py writes, so you have one
    # place with GNN vs CNN vs majority for your report table
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "gtzan_real_results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing["cnn_melspec_baseline"] = {"test_accuracy": test_acc, "test_macro_f1": test_f1}
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
