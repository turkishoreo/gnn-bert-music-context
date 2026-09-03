# GNN–BERT for Understanding Context from Music

Implementation of the CSE425/EEE474/CSE715 project spec: a hybrid **BERT + Graph Neural
Network** system for music context understanding (multi-label tagging, emotion regression,
cross-modal retrieval), built across the four-task roadmap (Easy → Advanced).

## Status of this scaffold

This repo is a **complete, runnable implementation of the architecture** for all four tasks.
It ships with a **synthetic data generator** (`src/synthetic_data.py`) so every script can be
run and sanity-checked end-to-end *right now, offline* — useful for debugging your pipeline
before you touch real data.

**You still need to plug in real datasets yourself** (FMA / MagnaTagATune / GTZAN / DEAM /
MusicCaps — see Table 1 of the spec). Those dataset hosts (`freemusicarchive.org`,
`zenodo.org`, Google's MusicCaps CSV on GitHub via research repos, etc.) and the HuggingFace
Hub (for downloading real `bert-base-uncased` weights) are **not reachable from the sandbox
this was built in**, so nothing here was trained or evaluated on real audio — treat all
metrics you see from a `--synthetic` run as pipeline smoke-tests, not results.

### What "swap in real data" means concretely
1. Download datasets per Table 1 into `data/raw/`.
2. Run `src/audio_features.py` to extract mel/chroma features and segment tracks.
3. Run `src/graph_builder.py` to build chord-transition / segment graphs → `data/processed/*.pt`.
4. Point `config.yaml` at the real paths and set `data.synthetic: false`.
5. `bert-base-uncased` will then download automatically via `transformers` the first time
   `src/bert_encoder.py` runs (needs internet access to huggingface.co).

## Directory structure

```
gnn-bert-music-context/
  README.md
  requirements.txt
  config.yaml
  data/
    raw/            # FMA, MagnaTagATune, MusicCaps, DEAM downloads go here
    processed/       # cached graphs (.pt), mel-spec/chroma arrays, BERT tokenizer caches
    splits/          # train/val/test JSON (no artist leakage)
  notebooks/
    eda.ipynb
    demo_context.ipynb   # end-to-end inference demo (synthetic data)
  src/
    synthetic_data.py    # NOT in the original spec — offline stand-in for real datasets
    audio_features.py    # mel/chroma extraction, segmentation (librosa)
    graph_builder.py      # chord-transition + segment graphs -> PyG Data objects
    bert_encoder.py        # Task 1: BERT tag/caption encoder + classifier head
    gnn_model.py            # Task 2: GraphSAGE / GAT encoder
    fusion_model.py          # Task 3: cross-attention GNN-BERT fusion, multi-task loss
    contrastive.py            # Task 4: InfoNCE dual-encoder + retrieval eval
    train.py                   # CLI: train any of the 4 tasks
    evaluate.py                  # CLI: Macro/Micro-F1, AUC-PR, MAE/R2, R@K
  results/
    metrics.json
    plots/
    retrieval_examples/
  report/
    final_report.pdf   # write this with the Overleaf templates linked in the spec
```

## Quick start (synthetic smoke-test, no internet needed)

```bash
pip install -r requirements.txt

# Task 1: BERT tag classifier
python src/train.py --task 1 --synthetic --epochs 3

# Task 2: GNN on segment/chord graphs
python src/train.py --task 2 --synthetic --epochs 3

# Task 3: GNN-BERT cross-attention fusion (+ DEAM emotion auxiliary loss)
python src/train.py --task 3 --synthetic --epochs 3

# Task 4: Contrastive dual-encoder + retrieval
python src/train.py --task 4 --synthetic --epochs 3

# Evaluate any trained task's held-out synthetic split
python src/evaluate.py --task 3 --synthetic
```

Each `train.py` run writes `results/metrics.json` (merged across tasks) and, for Task 4,
qualitative retrieval examples to `results/retrieval_examples/`.

## Real-data note on Task 1's BERT weights

`src/bert_encoder.py` uses `transformers.AutoModel.from_pretrained("bert-base-uncased")`
when `--synthetic` is **not** passed. In `--synthetic` mode it instead builds a randomly
initialized `BertModel` from a small `BertConfig` (2 layers, hidden=128) purely so the whole
pipeline can be exercised without a network call — swap this back to the pretrained checkpoint
for anything you intend to report results on.

## Mapping to the grading rubric

| Rubric item | Where it's addressed |
|---|---|
| Dataset & preprocessing | `audio_features.py`, `graph_builder.py`, `data/splits/` (artist-aware split helper in `synthetic_data.py::make_split` — replace with real artist IDs) |
| Model implementation | `bert_encoder.py`, `gnn_model.py`, `fusion_model.py`, `contrastive.py` |
| Context understanding quality | Run `evaluate.py` per task on real splits |
| Baseline comparison | `train.py --task 1 --baseline majority`, `--baseline cnn` (see `--help`) |
| Metrics & analysis | `evaluate.py` computes Macro/Micro-F1, AUC-PR, MAE/R², R@K; t-SNE helper in `evaluate.py::tsne_plot` |
| Report & presentation | `report/` — use one of the three Overleaf templates linked in the spec |
