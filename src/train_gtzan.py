"""
Task 2 on REAL data: GraphSAGE/GAT genre classifier trained on GTZAN segment
graphs built by prepare_gtzan.py. Also reports the majority-class baseline (B1
in the spec's Section 8) for comparison, since the rubric requires >=2 baselines.

Run prepare_gtzan.py first. Then:
    python train_gtzan.py --config ../config.yaml --epochs 15
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch
from sklearn.metrics import f1_score, accuracy_score

from gnn_model import GNNTagClassifier

# Always resolve results/ relative to the project root (the parent of this
# src/ folder), not the current working directory -- so it doesn't matter
# whether you run this script from the project root or from inside src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}


def load_split(splits_dir):
    with open(os.path.join(splits_dir, "gtzan_split.json")) as f:
        return json.load(f)


def load_graphs(records):
    graphs, labels = [], []
    for rec in records:
        g = torch.load(rec["graph_path"], weights_only=False)
        graphs.append(g)
        labels.append(GENRE_TO_IDX[rec["genre"]])
    return graphs, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def save_example_predictions(model, test_records, test_graphs, test_labels,
                              out_path, n_examples=5, seed=42):
    """Qualitative deliverable (spec Section 4.1 style: 'N example predictions'):
    picks n_examples random test tracks, runs the model, and records the true
    genre vs. predicted genre + confidence for each."""
    model.eval()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(test_records), size=min(n_examples, len(test_records)), replace=False)

    examples = []
    for i in idx:
        g = Batch.from_data_list([test_graphs[i]])
        logits = model(g.x, g.edge_index, g.batch)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_idx = probs.argmax().item()
        examples.append({
            "track_id": test_records[i]["track_id"],
            "true_genre": GENRES[test_labels[i].item()],
            "predicted_genre": GENRES[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "correct": GENRES[pred_idx] == test_records[i]["genre"],
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    return examples


def batches(graphs, labels, batch_size):
    idx = np.arange(len(graphs))
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i + batch_size]
        batch_graphs = [graphs[j] for j in sel]
        yield Batch.from_data_list(batch_graphs), labels[sel]


@torch.no_grad()
def evaluate(model, graphs, labels, batch_size=16):
    model.eval()
    all_preds = []
    for g, y in batches(graphs, labels, batch_size):
        logits = model(g.x, g.edge_index, g.batch)
        all_preds.append(logits.argmax(dim=-1))
    preds = torch.cat(all_preds).numpy()
    y_true = labels.numpy()
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    return acc, macro_f1


def majority_baseline(train_labels, test_labels):
    counts = Counter(train_labels.tolist())
    majority_class = counts.most_common(1)[0][0]
    preds = np.full(len(test_labels), majority_class)
    y_true = test_labels.numpy()
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    return acc, macro_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_examples", type=int, default=5)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_dir = os.path.normpath(os.path.join(os.path.dirname(args.config),
                                                cfg["data"]["splits_dir"]))

    torch.manual_seed(cfg.get("train", {}).get("seed", 42))

    split = load_split(splits_dir)
    print(f"loading graphs: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test")
    train_graphs, train_labels = load_graphs(split["train"])
    val_graphs, val_labels = load_graphs(split["val"])
    test_graphs, test_labels = load_graphs(split["test"])

    in_dim = train_graphs[0].x.shape[1]
    model = GNNTagClassifier(
        in_dim=in_dim, num_tags=len(GENRES),
        hidden_dim=cfg["gnn"]["hidden_dim"], num_layers=cfg["gnn"]["num_layers"],
        encoder=cfg["gnn"]["encoder"], dropout=cfg["gnn"]["dropout"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\n== Training GNN genre classifier on real GTZAN graphs ==")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for g, y in batches(train_graphs, train_labels, args.batch_size):
            logits = model(g.x, g.edge_index, g.batch)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_acc, val_f1 = evaluate(model, val_graphs, val_labels, args.batch_size)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={np.mean(losses):.4f}  "
              f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

    print("\n== Test-set results ==")
    test_acc, test_f1 = evaluate(model, test_graphs, test_labels, args.batch_size)
    print(f"GNN (real GTZAN):     accuracy={test_acc:.4f}   macro_f1={test_f1:.4f}")

    maj_acc, maj_f1 = majority_baseline(train_labels, test_labels)
    print(f"Majority baseline:    accuracy={maj_acc:.4f}   macro_f1={maj_f1:.4f}")

    examples = save_example_predictions(
        model, split["test"], test_graphs, test_labels,
        out_path=os.path.join(RESULTS_DIR, "gtzan_example_predictions.json"), n_examples=args.n_examples,
    )
    print(f"\n== {len(examples)} example predictions ==")
    for ex in examples:
        mark = "correct" if ex["correct"] else "wrong"
        print(f"  {ex['track_id']}: true={ex['true_genre']}  "
              f"pred={ex['predicted_genre']} (conf={ex['confidence']})  [{mark}]")
    print("wrote results/gtzan_example_predictions.json")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "gtzan_real_results.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.update({
        "dataset": "GTZAN (real)",
        "gnn": {"test_accuracy": test_acc, "test_macro_f1": test_f1},
        "majority_baseline": {"test_accuracy": maj_acc, "test_macro_f1": maj_f1},
        "n_train": len(train_graphs), "n_val": len(val_graphs), "n_test": len(test_graphs),
    })
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
