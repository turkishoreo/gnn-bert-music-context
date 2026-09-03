"""
Offline synthetic data generator.

Not part of the original project spec — this stands in for FMA / MagnaTagATune /
MusicCaps / DEAM so every script in this repo can be run and debugged without
network access to the real dataset hosts. Replace calls to this module with real
loaders (see audio_features.py + graph_builder.py) once you have the datasets in
data/raw/.

Everything here is deterministic given a seed, and shaped to match what the real
pipeline expects downstream:
  - waveform          : (T,) float32, fake "audio"
  - chroma             : (12, frames) float32
  - mel                 : (n_mels, frames) float32
  - text_tokens          : (max_len,) int64  (fake vocab of size VOCAB_SIZE)
  - tag_labels            : (num_tags,) float32 multi-hot
  - valence, arousal       : scalars in [1, 9] (DEAM-style)
  - caption_tokens          : (max_len,) int64 (paired MusicCaps-style caption)
"""
import numpy as np
import torch
from torch_geometric.data import Data

VOCAB_SIZE = 3000
CHORD_VOCAB = 24  # 12 major + 12 minor triads, toy chord space


def _rng(seed):
    return np.random.RandomState(seed)


def make_track(idx, num_tags=50, max_len=128, n_mels=128, n_chroma=12,
               frames=64, seed_offset=0):
    r = _rng(idx + seed_offset)
    mel = r.normal(size=(n_mels, frames)).astype(np.float32)
    chroma = r.dirichlet(np.ones(n_chroma), size=frames).T.astype(np.float32)  # (12, frames)

    # a handful of "true" tags/genre correlated with a latent factor so the model
    # has *something* learnable, rather than pure noise
    latent = r.normal()
    tag_logits = r.normal(size=num_tags) + latent
    tag_labels = (tag_logits > np.percentile(tag_logits, 90)).astype(np.float32)
    if tag_labels.sum() == 0:
        tag_labels[np.argmax(tag_logits)] = 1.0

    text_tokens = r.randint(1, VOCAB_SIZE, size=max_len).astype(np.int64)
    caption_tokens = r.randint(1, VOCAB_SIZE, size=max_len).astype(np.int64)

    valence = float(np.clip(5 + latent * 1.5 + r.normal() * 0.5, 1, 9))
    arousal = float(np.clip(5 - latent * 1.2 + r.normal() * 0.5, 1, 9))

    # toy chord sequence derived from chroma argmax, for graph_builder to consume
    chord_seq = chroma.argmax(axis=0) % CHORD_VOCAB

    return {
        "track_id": f"synth_{idx:05d}",
        "artist_id": f"artist_{idx % 37}",   # 37 fake artists -> enables artist-disjoint splits
        "mel": mel,
        "chroma": chroma,
        "chord_seq": chord_seq,
        "text_tokens": text_tokens,
        "caption_tokens": caption_tokens,
        "tag_labels": tag_labels,
        "valence": valence,
        "arousal": arousal,
    }


def make_dataset(n=400, num_tags=50, seed=42):
    return [make_track(i, num_tags=num_tags, seed_offset=seed) for i in range(n)]


def make_split(dataset, val_frac=0.15, test_frac=0.15, seed=42):
    """Artist-disjoint split, mirroring the spec's 'no artist leakage' requirement."""
    artists = sorted({t["artist_id"] for t in dataset})
    r = _rng(seed)
    r.shuffle(artists)
    n_val = max(1, int(len(artists) * val_frac))
    n_test = max(1, int(len(artists) * test_frac))
    val_artists = set(artists[:n_val])
    test_artists = set(artists[n_val:n_val + n_test])
    train_artists = set(artists[n_val + n_test:])

    split = {"train": [], "val": [], "test": []}
    for t in dataset:
        if t["artist_id"] in train_artists:
            split["train"].append(t)
        elif t["artist_id"] in val_artists:
            split["val"].append(t)
        else:
            split["test"].append(t)
    return split


def chord_transition_graph(track, hidden_dim=32, seed=0):
    """Nodes = unique chords observed in the track; edges = observed transitions,
    weighted by count. Node features are a small learned-looking embedding table
    keyed by chord id (stand-in for real chroma-derived chord features)."""
    seq = track["chord_seq"]
    unique_chords = sorted(set(seq.tolist()))
    chord_to_node = {c: i for i, c in enumerate(unique_chords)}

    r = _rng(hash(track["track_id"]) % (2**31))
    node_feats = r.normal(size=(len(unique_chords), hidden_dim)).astype(np.float32)

    edge_counts = {}
    for a, b in zip(seq[:-1], seq[1:]):
        key = (chord_to_node[a], chord_to_node[b])
        edge_counts[key] = edge_counts.get(key, 0) + 1

    if edge_counts:
        edge_index = torch.tensor(list(edge_counts.keys()), dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(list(edge_counts.values()), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)

    x = torch.tensor(node_feats, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


def segment_graph(track, window=8, hidden_dim=32, sim_threshold=0.6):
    """Nodes = fixed-length time segments of the chroma matrix; edges = temporal
    adjacency + cosine-similarity above sim_threshold, as in the spec's Step 3."""
    chroma = track["chroma"]  # (12, frames)
    n_frames = chroma.shape[1]
    n_segments = max(1, n_frames // window)
    seg_feats = np.stack([
        chroma[:, i * window:(i + 1) * window].mean(axis=1) for i in range(n_segments)
    ])  # (n_segments, 12)

    # project toy 12-d chroma means into hidden_dim via a fixed random projection
    r = _rng(hash(track["track_id"] + "_seg") % (2**31))
    proj = r.normal(size=(12, hidden_dim)).astype(np.float32)
    x = seg_feats @ proj

    edges = []
    weights = []
    for i in range(n_segments - 1):
        edges.append((i, i + 1))
        weights.append(1.0)
        edges.append((i + 1, i))
        weights.append(1.0)

    norm = seg_feats / (np.linalg.norm(seg_feats, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T
    for i in range(n_segments):
        for j in range(i + 2, n_segments):  # skip already-connected neighbours
            if sims[i, j] > sim_threshold:
                edges.append((i, j))
                weights.append(float(sims[i, j]))
                edges.append((j, i))
                weights.append(float(sims[i, j]))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)

    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index, edge_weight=edge_weight)
