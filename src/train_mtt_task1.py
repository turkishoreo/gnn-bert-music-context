"""
Task 1 on REAL data: BERT multi-label tag classifier trained on real MagnaTagATune
tags (prepared by prepare_magnatagatune.py). Uses a REAL pretrained BERT
(bert-base-uncased via HuggingFace) -- needs internet access the first time it
runs, to download the weights.

The text input is each clip's own metadata (artist name and track title, parsed
from its filename) -- NOT its own tags. An earlier version of this script fed
each clip's own tags in as the input text while asking the model to predict
those same tags as the label, which is circular (the model can only learn to
echo the label back) and visibly collapsed during training. See tags_to_text()
below for the corrected, non-circular design.

Run prepare_magnatagatune.py first. Then:
    python train_mtt_task1.py --config ../config.yaml --epochs 5
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, precision_recall_curve, auc

from bert_encoder import BertTagClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_split(splits_dir):
    with open(os.path.join(splits_dir, "mtt_split.json")) as f:
        return json.load(f)


def tags_to_text(rec):
    """Song metadata (parsed from the mp3 filename) -> BERT input text.

    IMPORTANT: this does NOT use the track's own tags as input. Using tags to
    predict tags would be circular (the model would just learn to echo the
    label back to itself, which is what an earlier version of this script did
    and produced degenerate/collapsing results). Instead we parse the artist +
    track-name portion of the MTT filename convention
    (artist-album-track-startsec-endsec.mp3) as a metadata-based text proxy,
    per spec Section 4.1's "textual music context" -- a real, if weaker,
    signal than a proper caption dataset (MusicCaps) would give."""
    fname = rec["mp3_path"].split("/")[-1]
    fname = fname.rsplit(".", 1)[0]  # drop .mp3
    parts = fname.split("-")
    # drop the trailing "-start-end" numeric time-range segment if present
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        parts = parts[:-2]
    text = " ".join(parts).replace("_", " ")
    return text if text.strip() else "unknown track"


def encode_batch(model, records, max_len):
    texts = [tags_to_text(r) for r in records]
    ids, mask = model.encoder.tokenize(texts, max_len=max_len)
    return ids, mask


def label_tensor(records):
    return torch.tensor([r["tag_vector"] for r in records], dtype=torch.float32)


def batches(records, batch_size, shuffle=False):
    idx = np.arange(len(records))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i + batch_size]
        yield [records[j] for j in sel]


def macro_micro_f1(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return macro, micro


def mean_auc_pr(y_true, y_score):
    aucs = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, k], y_score[:, k])
        aucs.append(auc(recall, precision))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def evaluate(model, records, batch_size, max_len, threshold=0.5):
    model.eval()
    all_logits, all_y = [], []
    for batch in batches(records, batch_size):
        ids, mask = encode_batch(model, batch, max_len)
        y = label_tensor(batch)
        logits = model(ids, mask)
        all_logits.append(logits)
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > threshold).astype(np.float32)
    macro, micro = macro_micro_f1(y, preds)
    aucpr = mean_auc_pr(y, probs)
    return macro, micro, aucpr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 if --freeze_encoder (training a fresh linear head "
                        "needs a much higher LR than fine-tuning), else 2e-5")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="freeze BERT weights, only train the classifier head (much faster on CPU)")
    p.add_argument("--n_examples", type=int, default=5)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                cfg["data"]["splits_dir"]))
    max_len = cfg["data"]["max_text_len"]

    if args.lr is None:
        args.lr = 1e-3 if args.freeze_encoder else 2e-5
    print(f"using lr={args.lr} (freeze_encoder={args.freeze_encoder})")

    torch.manual_seed(cfg.get("train", {}).get("seed", 42))

    data = load_split(splits_dir)
    tag_vocab = data["tag_vocab"]
    split = data["splits"]
    print(f"real MTT data: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test, {len(tag_vocab)} tags")

    print("loading pretrained bert-base-uncased (downloads on first run, needs internet)...")
    model = BertTagClassifier(
        num_tags=len(tag_vocab), pretrained_name=cfg["bert"]["pretrained_name"],
        synthetic=False, freeze_encoder=args.freeze_encoder,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\n== Training BERT tag classifier on real MagnaTagATune tags ==")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batches(split["train"], args.batch_size, shuffle=True):
            ids, mask = encode_batch(model, batch, max_len)
            y = label_tensor(batch)
            logits = model(ids, mask)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_macro, val_micro, val_aucpr = evaluate(model, split["val"], args.batch_size, max_len)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  "
              f"val_macro_f1={val_macro:.4f}  val_micro_f1={val_micro:.4f}  val_aucpr={val_aucpr:.4f}")

    print("\n== Test-set results ==")
    test_macro, test_micro, test_aucpr = evaluate(model, split["test"], args.batch_size, max_len)
    print(f"BERT tagger (real MTT): macro_f1={test_macro:.4f}  micro_f1={test_micro:.4f}  "
          f"mean_aucpr={test_aucpr:.4f}")

    # qualitative examples
    model.eval()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(split["test"]), size=min(args.n_examples, len(split["test"])), replace=False)
    examples = []
    with torch.no_grad():
        for i in idx:
            rec = split["test"][i]
            ids, mask = encode_batch(model, [rec], max_len)
            probs = torch.sigmoid(model(ids, mask)).squeeze(0)
            top_idx = torch.topk(probs, k=min(5, len(tag_vocab))).indices.tolist()
            examples.append({
                "clip_id": rec["clip_id"],
                "true_tags": rec["tags"],
                "predicted_top5_tags": [tag_vocab[j] for j in top_idx],
                "predicted_top5_confidence": [round(probs[j].item(), 3) for j in top_idx],
            })
    print(f"\n== {len(examples)} example predictions ==")
    for ex in examples:
        print(f"  clip {ex['clip_id']}: true={ex['true_tags']}  pred_top5={ex['predicted_top5_tags']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "mtt_real_results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.update({
        "dataset": "MagnaTagATune (real)",
        "task1_bert": {"test_macro_f1": test_macro, "test_micro_f1": test_micro,
                        "test_mean_aucpr": test_aucpr},
        "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"]),
        "num_tags": len(tag_vocab),
    })
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "mtt_task1_examples.json"), "w") as f:
        json.dump(examples, f, indent=2)
    print(f"\nwrote {out_path} and results/mtt_task1_examples.json")


if __name__ == "__main__":
    main()
