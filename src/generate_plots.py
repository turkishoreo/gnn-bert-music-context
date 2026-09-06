"""
Generates the plots your spec's Section 10 asks for:
  "Evaluation tables + plots (F1, AUC-PR, t-SNE, retrieval examples)"

1. Bar charts from your already-saved results/*.json (no retraining needed):
   - GTZAN: Majority vs GNN vs CNN (Task 2)
   - MagnaTagATune: GNN-only vs BERT-only vs Concat vs Cross-attention (Task 3 ablation)
   - MagnaTagATune: Task 4 retrieval R@K vs chance baseline
2. A REAL t-SNE plot (spec Section 4.3: "t-SNE of z coloured by genre and mood") --
   this one retrains the cross_attention fusion model briefly (a few epochs) purely to
   extract its fused z embeddings on the test set, since none of the earlier training
   runs saved a model checkpoint to reload from.

Run from src/, after you've already produced results/mtt_real_results.json and
results/gtzan_real_results.json via the earlier training scripts:

    python generate_plots.py --config ../config.yaml
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch

from fusion_model import GNNBERTFusion
from train_mtt_task1 import load_split, tags_to_text
from evaluate import tsne_plot

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def bar_chart(labels, series_dict, title, ylabel, out_path):
    """series_dict: {series_name: [values matching labels]}"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_series = len(series_dict)
    x = np.arange(len(labels))
    width = 0.8 / n_series

    plt.figure(figsize=(7, 5))
    for i, (name, values) in enumerate(series_dict.items()):
        plt.bar(x + i * width, values, width, label=name)
    plt.xticks(x + width * (n_series - 1) / 2, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"wrote {out_path}")


def plot_gtzan_comparison():
    path = os.path.join(RESULTS_DIR, "gtzan_real_results.json")
    if not os.path.exists(path):
        print(f"skip: {path} not found")
        return
    with open(path) as f:
        d = json.load(f)
    models = ["majority_baseline", "gnn", "cnn_melspec_baseline"]
    labels = ["Majority", "GNN", "CNN mel-spec"]
    acc = [d[m]["test_accuracy"] for m in models]
    f1 = [d[m]["test_macro_f1"] for m in models]
    bar_chart(labels, {"Accuracy": acc, "Macro-F1": f1},
              "Task 2: GTZAN genre classification (real data)", "Score",
              f"{PLOTS_DIR}/gtzan_comparison.png")


def plot_mtt_ablation():
    path = os.path.join(RESULTS_DIR, "mtt_real_results.json")
    if not os.path.exists(path):
        print(f"skip: {path} not found")
        return
    with open(path) as f:
        d = json.load(f)
    if "task3_ablations" not in d:
        print("skip: no task3_ablations in mtt_real_results.json yet")
        return
    order = ["gnn_only", "bert_only", "concat", "cross_attention"]
    labels = ["GNN-only", "BERT-only", "Concat", "Cross-attn"]
    present = [k for k in order if k in d["task3_ablations"]]
    if len(present) < len(order):
        print(f"warning: only found ablations {present}, expected all of {order}")
    macro = [d["task3_ablations"][k]["test_macro_f1"] for k in present]
    aucpr = [d["task3_ablations"][k]["test_mean_aucpr"] for k in present]
    present_labels = [labels[order.index(k)] for k in present]
    bar_chart(present_labels, {"Macro-F1": macro, "AUC-PR": aucpr},
              "Task 3: GNN-BERT fusion ablation (real MagnaTagATune)", "Score",
              f"{PLOTS_DIR}/mtt_ablation_comparison.png")


def plot_task4_retrieval():
    path = os.path.join(RESULTS_DIR, "mtt_real_results.json")
    if not os.path.exists(path):
        print(f"skip: {path} not found")
        return
    with open(path) as f:
        d = json.load(f)
    if "task4_contrastive" not in d:
        print("skip: no task4_contrastive in mtt_real_results.json yet")
        return
    m = d["task4_contrastive"]
    n_test = d.get("n_test", 100)
    ks = [1, 5, 10]
    a2c = [m[f"audio2caption_R@{k}"] for k in ks]
    c2a = [m[f"caption2audio_R@{k}"] for k in ks]
    chance = [k / n_test for k in ks]
    labels = [f"R@{k}" for k in ks]
    bar_chart(labels, {"Audio->Caption": a2c, "Caption->Audio": c2a, "Chance baseline": chance},
              "Task 4: Retrieval vs. chance baseline (real MagnaTagATune)", "Recall",
              f"{PLOTS_DIR}/mtt_retrieval_vs_chance.png")


def plot_real_tsne(cfg, epochs=6):
    """Briefly trains a cross_attention fusion model (no checkpoint was saved from
    the earlier real run) purely to extract z embeddings on the test set for a
    real t-SNE plot, colored by each clip's most-active tag."""
    splits_dir = os.path.normpath(os.path.join("..", cfg["data"]["splits_dir"]))
    data = load_split(splits_dir)
    tag_vocab = data["tag_vocab"]
    split = data["splits"]

    print("briefly training cross_attention fusion model for t-SNE embeddings "
          f"({epochs} epochs)...")
    model = GNNBERTFusion(
        graph_in_dim=32, num_tags=len(tag_vocab),
        bert_pretrained=cfg["bert"]["pretrained_name"], bert_synthetic=False,
        gnn_hidden=cfg["gnn"]["hidden_dim"], gnn_layers=cfg["gnn"]["num_layers"],
        gnn_encoder=cfg["gnn"]["encoder"], fusion_type="cross_attention",
        attn_heads=cfg["fusion"]["attn_heads"], predict_emotion=False,
    )
    for p in model.bert.bert.parameters():
        p.requires_grad = False  # frozen, fast

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    max_len = cfg["data"]["max_text_len"]
    batch_size = 16

    def label_tensor(records):
        return torch.tensor([r["tag_vector"] for r in records], dtype=torch.float32)

    def run_batch(records):
        g = Batch.from_data_list([torch.load(r["graph_path"], weights_only=False) for r in records])
        texts = [tags_to_text(r) for r in records]
        ids, mask = model.bert.tokenize(texts, max_len=max_len)
        return model(g.x, g.edge_index, g.batch, ids, mask, return_z=True)

    for epoch in range(epochs):
        model.train()
        idx = np.arange(len(split["train"]))
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch = [split["train"][j] for j in idx[i:i + batch_size]]
            out = run_batch(batch)
            loss = F.binary_cross_entropy_with_logits(out["tag_logits"], label_tensor(batch))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  tsne-model epoch {epoch+1}/{epochs} done")

    model.eval()
    with torch.no_grad():
        out = run_batch(split["test"])
    z = out["z"]
    # color by each clip's single most-common tag (argmax of its multi-hot vector)
    tag_ids = [int(np.argmax(r["tag_vector"])) if sum(r["tag_vector"]) > 0 else -1
               for r in split["test"]]
    out_path = tsne_plot(z, tag_ids, out_path=f"{PLOTS_DIR}/tsne_task3_fusion.png",
                          title="Task 3: t-SNE of fused embeddings (colored by top tag)")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../config.yaml")
    p.add_argument("--skip_tsne", action="store_true",
                   help="skip the t-SNE plot (it retrains a small model, takes a few minutes)")
    p.add_argument("--tsne_epochs", type=int, default=6)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    plot_gtzan_comparison()
    plot_mtt_ablation()
    plot_task4_retrieval()
    if not args.skip_tsne:
        plot_real_tsne(cfg, epochs=args.tsne_epochs)

    print(f"\nall available plots written to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
