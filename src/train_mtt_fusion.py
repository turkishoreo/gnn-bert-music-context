"""
Task 3 on REAL data: GNN-BERT fusion for multi-label tag prediction on real
MagnaTagATune graphs + metadata-text (prepared by prepare_magnatagatune.py).

Implements the spec's required ablation (Section 4.3 deliverables:
"Ablation: BERT-only, GNN-only, early concat, cross-attention") via a single
--ablation flag, so all four variants are trained on the *exact same* data
split for a fair comparison -- run this script 4 times, once per value.

No DEAM valence/arousal data is available for MagnaTagATune, so the emotion
regression head is disabled for this real run (predict_emotion=False); that
auxiliary loss only applies if you later bring in DEAM.

Run prepare_magnatagatune.py first. Then, e.g.:
    python train_mtt_fusion.py --config ../config.yaml --ablation gnn_only
    python train_mtt_fusion.py --config ../config.yaml --ablation bert_only --freeze_encoder
    python train_mtt_fusion.py --config ../config.yaml --ablation concat --freeze_encoder
    python train_mtt_fusion.py --config ../config.yaml --ablation cross_attention --freeze_encoder
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch

from bert_encoder import BertTagClassifier
from gnn_model import GNNTagClassifier
from fusion_model import GNNBERTFusion
from train_mtt_task1 import (load_split, tags_to_text, macro_micro_f1, mean_auc_pr)

GRAPH_IN_DIM = 32  # matches graph_builder.segment_graph's default proj_dim


def load_graph(rec):
    return torch.load(rec["graph_path"], weights_only=False)


def label_tensor(records):
    return torch.tensor([r["tag_vector"] for r in records], dtype=torch.float32)


def batches(records, batch_size, shuffle=False):
    idx = np.arange(len(records))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i + batch_size]
        yield [records[j] for j in sel]


def build_model(ablation, num_tags, cfg, freeze_encoder):
    common_bert_kwargs = dict(
        pretrained_name=cfg["bert"]["pretrained_name"], synthetic=False,
    )
    if ablation == "gnn_only":
        return GNNTagClassifier(
            in_dim=GRAPH_IN_DIM, num_tags=num_tags,
            hidden_dim=cfg["gnn"]["hidden_dim"], num_layers=cfg["gnn"]["num_layers"],
            encoder=cfg["gnn"]["encoder"], dropout=cfg["gnn"]["dropout"],
        )
    if ablation == "bert_only":
        return BertTagClassifier(num_tags=num_tags, freeze_encoder=freeze_encoder,
                                  **common_bert_kwargs)
    if ablation in ("concat", "cross_attention"):
        model = GNNBERTFusion(
            graph_in_dim=GRAPH_IN_DIM, num_tags=num_tags,
            bert_pretrained=cfg["bert"]["pretrained_name"], bert_synthetic=False,
            gnn_hidden=cfg["gnn"]["hidden_dim"], gnn_layers=cfg["gnn"]["num_layers"],
            gnn_encoder=cfg["gnn"]["encoder"], fusion_type=ablation,
            attn_heads=cfg["fusion"]["attn_heads"], predict_emotion=False,
        )
        if freeze_encoder:
            for p in model.bert.bert.parameters():
                p.requires_grad = False
        return model
    raise ValueError(f"unknown ablation {ablation}")


def forward_pass(model, ablation, records, max_len, bert_for_tokenize=None):
    """Returns tag_logits for whichever ablation variant is active."""
    if ablation == "gnn_only":
        g = Batch.from_data_list([load_graph(r) for r in records])
        return model(g.x, g.edge_index, g.batch)
    if ablation == "bert_only":
        texts = [tags_to_text(r) for r in records]
        ids, mask = model.encoder.tokenize(texts, max_len=max_len)
        return model(ids, mask)
    # fusion variants
    g = Batch.from_data_list([load_graph(r) for r in records])
    texts = [tags_to_text(r) for r in records]
    ids, mask = bert_for_tokenize.tokenize(texts, max_len=max_len)
    out = model(g.x, g.edge_index, g.batch, ids, mask)
    return out["tag_logits"]


@torch.no_grad()
def evaluate(model, ablation, records, batch_size, max_len, bert_for_tokenize=None, threshold=0.5):
    model.eval()
    all_logits, all_y = [], []
    for batch in batches(records, batch_size):
        logits = forward_pass(model, ablation, batch, max_len, bert_for_tokenize)
        all_logits.append(logits)
        all_y.append(label_tensor(batch))
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
    p.add_argument("--ablation", required=True,
                   choices=["gnn_only", "bert_only", "concat", "cross_attention"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 for gnn_only, or when --freeze_encoder; else 2e-5")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="for bert_only/concat/cross_attention: freeze BERT weights (much faster on CPU)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                cfg["data"]["splits_dir"]))
    max_len = cfg["data"]["max_text_len"]
    torch.manual_seed(cfg.get("train", {}).get("seed", 42))

    if args.lr is None:
        args.lr = 1e-3 if (args.ablation == "gnn_only" or args.freeze_encoder) else 2e-5
    print(f"ablation={args.ablation}  lr={args.lr}  freeze_encoder={args.freeze_encoder}")

    data = load_split(splits_dir)
    tag_vocab = data["tag_vocab"]
    split = data["splits"]
    print(f"real MTT data: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test, {len(tag_vocab)} tags")

    if args.ablation != "gnn_only":
        print("loading pretrained bert-base-uncased (downloads on first run, needs internet)...")
    model = build_model(args.ablation, len(tag_vocab), cfg, args.freeze_encoder)
    bert_for_tokenize = model.encoder if args.ablation == "bert_only" else \
                        (model.bert if args.ablation in ("concat", "cross_attention") else None)

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    print(f"\n== Training ablation='{args.ablation}' on real MagnaTagATune ==")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batches(split["train"], args.batch_size, shuffle=True):
            logits = forward_pass(model, args.ablation, batch, max_len, bert_for_tokenize)
            y = label_tensor(batch)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_macro, val_micro, val_aucpr = evaluate(
            model, args.ablation, split["val"], args.batch_size, max_len, bert_for_tokenize)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  "
              f"val_macro_f1={val_macro:.4f}  val_micro_f1={val_micro:.4f}  val_aucpr={val_aucpr:.4f}")

    print("\n== Test-set results ==")
    test_macro, test_micro, test_aucpr = evaluate(
        model, args.ablation, split["test"], args.batch_size, max_len, bert_for_tokenize)
    print(f"[{args.ablation}] macro_f1={test_macro:.4f}  micro_f1={test_micro:.4f}  "
          f"mean_aucpr={test_aucpr:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/mtt_real_results.json"
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.setdefault("task3_ablations", {})
    existing["task3_ablations"][args.ablation] = {
        "test_macro_f1": test_macro, "test_micro_f1": test_micro, "test_mean_aucpr": test_aucpr,
    }
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
