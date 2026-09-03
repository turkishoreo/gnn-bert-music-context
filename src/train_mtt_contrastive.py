"""
Task 4 on REAL data (bonus, spec Section 9: 18 marks): contrastive GNN-BERT
dual-encoder trained on real MagnaTagATune graphs + tag-based pseudo-captions.

Uses MagnaTagATune's own tags joined into a short caption-like string (e.g.
"guitar classical slow") as the text side of each (audio_graph, caption) pair --
a stand-in for real MusicCaps captions, which would require downloading audio
from YouTube video IDs (notably more involved than GTZAN/MTT's direct downloads).

IMPORTANT DISTINCTION from Task 1's earlier bug: using tags as text here is NOT
circular the way it was for classification. The model is never asked to predict
these same tags back out -- it's asked to match a caption to the ONE correct
audio clip out of many candidates in the batch (InfoNCE). Many other clips in
the batch will share similar/overlapping tags, so the model must learn genuine
graph<->text content alignment, not just memorize a word.

Run prepare_magnatagatune.py first. Then:
    python train_mtt_contrastive.py --config ../config.yaml --epochs 10 --freeze_encoder
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from contrastive import DualEncoder, info_nce_loss, retrieval_recall_at_k, top_k_matches
from train_mtt_task1 import load_split


def caption_text(rec):
    """Tags joined into a short pseudo-caption -- the text side of each
    (graph, caption) contrastive pair. See module docstring for why this is
    a legitimate (non-circular) choice for a retrieval task."""
    return " ".join(rec["tags"]) if rec["tags"] else "no tags"


def load_graph(rec):
    return torch.load(rec["graph_path"], weights_only=False)


def batches(records, batch_size, shuffle=False):
    idx = np.arange(len(records))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i + batch_size]
        if len(sel) < 2:
            continue  # InfoNCE needs >=2 in-batch negatives
        yield [records[j] for j in sel]


def encode_split(model, records, batch_size, max_len):
    """Encode an entire split (in mini-batches for compute, but returns the
    full concatenated embedding set so retrieval is evaluated against the
    WHOLE split as the candidate pool, not just one mini-batch)."""
    model.eval()
    g_embs, t_embs = [], []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            g_data = Batch.from_data_list([load_graph(r) for r in batch])
            texts = [caption_text(r) for r in batch]
            ids, mask = model.bert.tokenize(texts, max_len=max_len)
            g_emb, t_emb = model(g_data.x, g_data.edge_index, g_data.batch, ids, mask)
            g_embs.append(g_emb)
            t_embs.append(t_emb)
    return torch.cat(g_embs), torch.cat(t_embs)


def save_qualitative_examples(model, records, batch_size, max_len, out_path, n_examples=10, seed=42):
    g_emb, t_emb = encode_split(model, records, batch_size, max_len)
    sim = g_emb @ t_emb.t()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(records), size=min(n_examples, len(records)), replace=False)
    examples = []
    for i in idx:
        top3 = top_k_matches(sim[i], k=3)
        examples.append({
            "clip_id": records[i]["clip_id"],
            "query_caption": caption_text(records[i]),
            "top3_matched_clip_ids": [records[j]["clip_id"] for j in top3],
            "correct_in_top3": records[i]["clip_id"] in [records[j]["clip_id"] for j in top3],
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32,
                   help="larger batch = more in-batch negatives for InfoNCE, helps retrieval")
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 if --freeze_encoder, else 2e-5")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--n_examples", type=int, default=10)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                cfg["data"]["splits_dir"]))
    max_len = cfg["data"]["max_text_len"]
    torch.manual_seed(cfg.get("train", {}).get("seed", 42))

    if args.lr is None:
        args.lr = 1e-3 if args.freeze_encoder else 2e-5
    print(f"lr={args.lr}  freeze_encoder={args.freeze_encoder}  batch_size={args.batch_size}")

    data = load_split(splits_dir)
    split = data["splits"]
    print(f"real MTT data: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test")

    print("loading pretrained bert-base-uncased (downloads on first run, needs internet)...")
    model = DualEncoder(
        graph_in_dim=32, embedding_dim=cfg["contrastive"]["embedding_dim"],
        bert_pretrained=cfg["bert"]["pretrained_name"], bert_synthetic=False,
        gnn_hidden=cfg["gnn"]["hidden_dim"], gnn_layers=cfg["gnn"]["num_layers"],
        gnn_encoder=cfg["gnn"]["encoder"],
    )
    if args.freeze_encoder:
        for p_ in model.bert.bert.parameters():
            p_.requires_grad = False

    opt = torch.optim.Adam((p_ for p_ in model.parameters() if p_.requires_grad), lr=args.lr)
    temp = cfg["contrastive"]["temperature"]

    print("\n== Training contrastive GNN-BERT dual encoder on real MagnaTagATune ==")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batches(split["train"], args.batch_size, shuffle=True):
            g_data = Batch.from_data_list([load_graph(r) for r in batch])
            texts = [caption_text(r) for r in batch]
            ids, mask = model.bert.tokenize(texts, max_len=max_len)
            g_emb, t_emb = model(g_data.x, g_data.edge_index, g_data.batch, ids, mask)
            loss = info_nce_loss(g_emb, t_emb, temp)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        g_val, t_val = encode_split(model, split["val"], args.batch_size, max_len)
        val_metrics = retrieval_recall_at_k(g_val, t_val)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  "
              f"val_a2c_R@5={val_metrics['audio2caption_R@5']:.3f}  "
              f"val_c2a_R@5={val_metrics['caption2audio_R@5']:.3f}")

    print("\n== Test-set retrieval results ==")
    g_test, t_test = encode_split(model, split["test"], args.batch_size, max_len)
    test_metrics = retrieval_recall_at_k(g_test, t_test)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    examples = save_qualitative_examples(
        model, split["test"], args.batch_size, max_len,
        "results/mtt_task4_qualitative_examples.json", n_examples=args.n_examples,
    )
    n_correct = sum(e["correct_in_top3"] for e in examples)
    print(f"\n{n_correct}/{len(examples)} qualitative examples had correct audio in top-3")

    os.makedirs("results", exist_ok=True)
    out_path = "results/mtt_real_results.json"
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing["task4_contrastive"] = test_metrics
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nwrote {out_path} and results/mtt_task4_qualitative_examples.json")


if __name__ == "__main__":
    main()
