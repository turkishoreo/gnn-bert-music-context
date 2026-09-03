"""
Metrics from spec Section 6, plus a CLI that trains (briefly) and reports
held-out test performance for any of the 4 tasks on the synthetic smoke-test data.

    python evaluate.py --task 3 --synthetic
"""
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, auc, r2_score


def macro_micro_f1(y_true, y_pred):
    """y_true, y_pred: (N, K) binary arrays. Returns (macro_f1, micro_f1)."""
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return macro, micro


def mean_auc_pr(y_true, y_score):
    """Per spec: 'Area under precision-recall curve per tag; report mean AUC-PR over tags.'
    y_true: (N, K) binary, y_score: (N, K) probabilities."""
    aucs = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue  # tag never occurs in this split -> undefined, skip
        precision, recall, _ = precision_recall_curve(y_true[:, k], y_score[:, k])
        aucs.append(auc(recall, precision))
    return float(np.mean(aucs)) if aucs else float("nan")


def emotion_regression_metrics(y_true, y_pred):
    """MAE and R^2 for DEAM-style valence/arousal regression (spec Section 6)."""
    mae = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "R2": r2}


def graph_coherence_score(node_embeddings, edge_index, threshold=0.7):
    """Optional analysis metric (spec Section 6): fraction of edges whose endpoint
    embeddings are cosine-similar above `threshold`, i.e. whether high-attention
    edges align with repeated structure."""
    src, dst = edge_index
    h = torch.nn.functional.normalize(node_embeddings, dim=-1)
    sims = (h[src] * h[dst]).sum(dim=-1)
    return (sims > threshold).float().mean().item()


def tsne_plot(embeddings, labels, out_path="results/plots/tsne.png", title="t-SNE"):
    """t-SNE of fused embeddings z, coloured by a label array (genre or mood id).
    Deliverable for Task 3. Requires matplotlib + sklearn (both in requirements.txt)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    emb = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else embeddings
    n = emb.shape[0]
    perplexity = max(2, min(30, n // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42).fit_transform(emb)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=12)
    plt.title(title)
    plt.colorbar(sc)
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import train as T

    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--graph", choices=["segment", "chord"], default="segment")
    p.add_argument("--n_tracks", type=int, default=300)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    if not args.synthetic:
        raise SystemExit("Only the --synthetic smoke test is wired up; see train.py's real-data note.")

    cfg = T.load_config(args.config)
    T.set_seed(cfg.get("train", {}).get("seed", 42))
    dataset = T.sd.make_dataset(n=args.n_tracks, num_tags=cfg["data"]["num_tags"])
    split = T.sd.make_split(dataset)

    class A:
        epochs = args.epochs
        batch_size = args.batch_size
        graph = args.graph

    trainers = {1: T.train_task1, 2: T.train_task2, 3: T.train_task3, 4: T.train_task4}
    model, _ = trainers[args.task](cfg, A(), split)

    print("\n== Test-set evaluation ==")
    if args.task == 1:
        macro = T.eval_task1(model, split["test"])
        print(f"Macro-F1: {macro:.4f}")
    elif args.task == 2:
        macro = T.eval_task2(model, split["test"], args.graph)
        print(f"Macro-F1: {macro:.4f}")
    elif args.task == 3:
        macro = T.eval_task3(model, split["test"], args.graph)
        print(f"Macro-F1: {macro:.4f}")
    elif args.task == 4:
        metrics = T.eval_task4(model, split["test"], args.graph)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
