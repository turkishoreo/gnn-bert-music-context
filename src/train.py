"""
Unified training entrypoint for the four-task roadmap.

Examples:
    python train.py --task 1 --synthetic --epochs 3
    python train.py --task 2 --synthetic --epochs 3 --graph segment
    python train.py --task 3 --synthetic --epochs 3
    python train.py --task 4 --synthetic --epochs 3
    python train.py --task 1 --baseline majority --synthetic
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch

import synthetic_data as sd
from bert_encoder import BertTagClassifier
from gnn_model import GNNTagClassifier
from fusion_model import GNNBERTFusion, multi_task_loss
from contrastive import DualEncoder, info_nce_loss, retrieval_recall_at_k, top_k_matches

HIDDEN_DIM = 32  # synthetic graph node-feature width, see synthetic_data.py

# Always resolve results/ and config.yaml relative to the project root (the
# parent of this src/ folder), not the current working directory -- so it
# doesn't matter whether you run `python train.py` from the project root or
# from inside src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path=None):
    """Defaults to <project_root>/config.yaml regardless of current working directory."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def text_tensor(batch, key="text_tokens"):
    ids = torch.tensor(np.stack([t[key] for t in batch]), dtype=torch.long)
    mask = torch.ones_like(ids)
    return ids, mask


def tag_tensor(batch):
    return torch.tensor(np.stack([t["tag_labels"] for t in batch]), dtype=torch.float32)


def emotion_tensors(batch):
    v = torch.tensor([t["valence"] for t in batch], dtype=torch.float32)
    a = torch.tensor([t["arousal"] for t in batch], dtype=torch.float32)
    return v, a


def graph_batch(batch, graph_type="segment"):
    fn = sd.segment_graph if graph_type == "segment" else sd.chord_transition_graph
    graphs = [fn(t, hidden_dim=HIDDEN_DIM) for t in batch]
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------- Task 1 ----
def train_task1(cfg, args, split):
    print("== Task 1: BERT tag classifier ==")
    model = BertTagClassifier(
        num_tags=cfg["data"]["num_tags"],
        synthetic=True,
        synthetic_hidden_size=cfg["bert"]["synthetic_hidden_size"],
        synthetic_num_layers=cfg["bert"]["synthetic_num_layers"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["bert"]["learning_rate"]))
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in make_batches(split["train"], args.batch_size):
            ids, mask = text_tensor(batch)
            y = tag_tensor(batch)
            logits = model(ids, mask)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_f1 = eval_task1(model, split["val"])
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  val_macro_f1={val_f1:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_f1": val_f1})
    return model, history


@torch.no_grad()
def eval_task1(model, items, batch_size=32, threshold=0.5):
    from evaluate import macro_micro_f1
    model.eval()
    all_logits, all_y = [], []
    for batch in make_batches(items, batch_size):
        ids, mask = text_tensor(batch)
        y = tag_tensor(batch)
        logits = model(ids, mask)
        all_logits.append(logits)
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    preds = (torch.sigmoid(logits) > threshold).float()
    macro, _ = macro_micro_f1(y.numpy(), preds.numpy())
    return macro


# ---------------------------------------------------------------- Task 2 ----
def train_task2(cfg, args, split):
    print(f"== Task 2: GNN ({args.graph} graph, {cfg['gnn']['encoder']}) ==")
    model = GNNTagClassifier(
        in_dim=HIDDEN_DIM, num_tags=cfg["data"]["num_tags"],
        hidden_dim=cfg["gnn"]["hidden_dim"], num_layers=cfg["gnn"]["num_layers"],
        encoder=cfg["gnn"]["encoder"], dropout=cfg["gnn"]["dropout"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["gnn"]["learning_rate"]))
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in make_batches(split["train"], args.batch_size):
            g = graph_batch(batch, args.graph)
            y = tag_tensor(batch)
            logits = model(g.x, g.edge_index, g.batch)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_f1 = eval_task2(model, split["val"], args.graph)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  val_macro_f1={val_f1:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_f1": val_f1})
    return model, history


@torch.no_grad()
def eval_task2(model, items, graph_type, batch_size=32, threshold=0.5):
    from evaluate import macro_micro_f1
    model.eval()
    all_logits, all_y = [], []
    for batch in make_batches(items, batch_size):
        g = graph_batch(batch, graph_type)
        y = tag_tensor(batch)
        logits = model(g.x, g.edge_index, g.batch)
        all_logits.append(logits)
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    preds = (torch.sigmoid(logits) > threshold).float()
    macro, _ = macro_micro_f1(y.numpy(), preds.numpy())
    return macro


# ---------------------------------------------------------------- Task 3 ----
def train_task3(cfg, args, split):
    print("== Task 3: GNN-BERT cross-attention fusion ==")
    model = GNNBERTFusion(
        graph_in_dim=HIDDEN_DIM, num_tags=cfg["data"]["num_tags"],
        bert_synthetic=True, bert_synth_hidden=cfg["bert"]["synthetic_hidden_size"],
        bert_synth_layers=cfg["bert"]["synthetic_num_layers"],
        gnn_hidden=cfg["gnn"]["hidden_dim"], gnn_layers=cfg["gnn"]["num_layers"],
        gnn_encoder=cfg["gnn"]["encoder"], fusion_type=cfg["fusion"]["type"],
        attn_heads=cfg["fusion"]["attn_heads"], predict_emotion=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    alpha = beta = cfg["fusion"]["emotion_loss_weight"]
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in make_batches(split["train"], args.batch_size):
            g = graph_batch(batch, args.graph)
            ids, mask = text_tensor(batch)
            y = tag_tensor(batch)
            v, a = emotion_tensors(batch)
            out = model(g.x, g.edge_index, g.batch, ids, mask)
            loss, log = multi_task_loss(out, y, v, a, alpha, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(log["total_loss"])
        val_f1 = eval_task3(model, split["val"], args.graph)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  val_macro_f1={val_f1:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_f1": val_f1})
    return model, history


@torch.no_grad()
def eval_task3(model, items, graph_type, batch_size=32, threshold=0.5):
    from evaluate import macro_micro_f1
    model.eval()
    all_logits, all_y = [], []
    for batch in make_batches(items, batch_size):
        g = graph_batch(batch, graph_type)
        ids, mask = text_tensor(batch)
        y = tag_tensor(batch)
        out = model(g.x, g.edge_index, g.batch, ids, mask)
        all_logits.append(out["tag_logits"])
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    preds = (torch.sigmoid(logits) > threshold).float()
    macro, _ = macro_micro_f1(y.numpy(), preds.numpy())
    return macro


# ---------------------------------------------------------------- Task 4 ----
def train_task4(cfg, args, split):
    print("== Task 4: Contrastive GNN-BERT dual encoder (MusicCaps-style) ==")
    model = DualEncoder(
        graph_in_dim=HIDDEN_DIM, embedding_dim=cfg["contrastive"]["embedding_dim"],
        bert_synthetic=True, bert_synth_hidden=cfg["bert"]["synthetic_hidden_size"],
        bert_synth_layers=cfg["bert"]["synthetic_num_layers"],
        gnn_hidden=cfg["gnn"]["hidden_dim"], gnn_layers=cfg["gnn"]["num_layers"],
        gnn_encoder=cfg["gnn"]["encoder"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    temp = cfg["contrastive"]["temperature"]
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in make_batches(split["train"], args.batch_size):
            if len(batch) < 2:
                continue  # InfoNCE needs >=2 in-batch negatives
            g_data = graph_batch(batch, args.graph)
            cap_ids, cap_mask = text_tensor(batch, key="caption_tokens")
            g_emb, t_emb = model(g_data.x, g_data.edge_index, g_data.batch, cap_ids, cap_mask)
            loss = info_nce_loss(g_emb, t_emb, temp)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        metrics = eval_task4(model, split["val"], args.graph)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  "
              f"val_R@5(a2c)={metrics.get('audio2caption_R@5', float('nan')):.3f}")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
    return model, history


@torch.no_grad()
def eval_task4(model, items, graph_type, batch_size=64):
    model.eval()
    g_embs, t_embs = [], []
    for batch in make_batches(items, batch_size):
        g_data = graph_batch(batch, graph_type)
        cap_ids, cap_mask = text_tensor(batch, key="caption_tokens")
        g_emb, t_emb = model(g_data.x, g_data.edge_index, g_data.batch, cap_ids, cap_mask)
        g_embs.append(g_emb)
        t_embs.append(t_emb)
    g_embs = torch.cat(g_embs)
    t_embs = torch.cat(t_embs)
    return retrieval_recall_at_k(g_embs, t_embs)


def save_qualitative_retrieval(model, items, graph_type, out_dir, n_examples=10):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    g_data = graph_batch(items, graph_type)
    cap_ids, cap_mask = text_tensor(items, key="caption_tokens")
    with torch.no_grad():
        g_emb, t_emb = model(g_data.x, g_data.edge_index, g_data.batch, cap_ids, cap_mask)
    sim = g_emb @ t_emb.t()
    examples = []
    for i in range(min(n_examples, len(items))):
        top3 = top_k_matches(sim[i], k=3)
        examples.append({
            "query_caption_for_track": items[i]["track_id"],
            "top3_matched_track_ids": [items[j]["track_id"] for j in top3],
            "correct_in_top3": items[i]["track_id"] in [items[j]["track_id"] for j in top3],
        })
    with open(os.path.join(out_dir, "task4_qualitative_examples.json"), "w") as f:
        json.dump(examples, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--synthetic", action="store_true", help="required for now; real-data loaders live in audio_features.py / graph_builder.py")
    p.add_argument("--graph", choices=["segment", "chord"], default="segment")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_tracks", type=int, default=300)
    p.add_argument("--config", default=None, help="defaults to <project_root>/config.yaml")
    p.add_argument("--baseline", choices=["majority", "cnn_melspec", "bert_only", "pca_mlp"], default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    args.epochs = args.epochs or cfg.get("train", {}).get("epochs", 3)
    args.batch_size = args.batch_size or cfg.get("train", {}).get("batch_size", 16)
    set_seed(cfg.get("train", {}).get("seed", 42))

    if not args.synthetic:
        raise SystemExit("Real-data path not wired into train.py's CLI yet -- "
                          "use audio_features.py + graph_builder.py to build data/processed/*.pt, "
                          "then adapt make_batches()/graph_batch() to read from there. "
                          "Pass --synthetic to run the offline smoke test.")

    dataset = sd.make_dataset(n=args.n_tracks, num_tags=cfg["data"]["num_tags"])
    split = sd.make_split(dataset)
    print(f"synthetic dataset: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test")

    trainers = {1: train_task1, 2: train_task2, 3: train_task3, 4: train_task4}
    model, history = trainers[args.task](cfg, args, split)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    all_metrics[f"task{args.task}"] = history
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"wrote {metrics_path}")

    if args.task == 4:
        save_qualitative_retrieval(model, split["test"], args.graph, os.path.join(RESULTS_DIR, "retrieval_examples"))
        print("wrote results/retrieval_examples/task4_qualitative_examples.json")

    return model, split, cfg


if __name__ == "__main__":
    main()
