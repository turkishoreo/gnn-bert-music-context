"""
Music structure graph construction, per spec Section 3 step 3:
  - Chord-transition graph: nodes = unique chords; edges = observed transitions,
    weighted by count.
  - Segment graph: nodes = time segments; edges = temporal adjacency + cosine
    similarity of MFCC/chroma > tau.

These operate on real chroma / chord-sequence arrays (e.g. from audio_features.py).
For quick offline testing, src/synthetic_data.py provides equivalents that also
fabricate plausible node features (real usage should feed genuine chroma-derived
features into `node_feature_fn` below).
"""
import numpy as np
import torch
from torch_geometric.data import Data, Batch


def chord_transition_graph(chord_seq, node_feature_fn=None, hidden_dim=32):
    """
    chord_seq: 1D int array of chord ids per frame (see audio_features.estimate_chord_sequence)
    node_feature_fn: optional callable(chord_id) -> np.ndarray[hidden_dim]; if None,
                     falls back to a fixed random embedding table (toy features only).
    """
    unique_chords = sorted(set(int(c) for c in chord_seq))
    chord_to_node = {c: i for i, c in enumerate(unique_chords)}

    if node_feature_fn is None:
        rng = np.random.RandomState(0)
        table = rng.normal(size=(max(unique_chords) + 1, hidden_dim)).astype(np.float32)
        node_feature_fn = lambda c: table[c]

    node_feats = np.stack([node_feature_fn(c) for c in unique_chords])

    edge_counts = {}
    for a, b in zip(chord_seq[:-1], chord_seq[1:]):
        key = (chord_to_node[int(a)], chord_to_node[int(b)])
        edge_counts[key] = edge_counts.get(key, 0) + 1

    edge_index, edge_weight = _dict_to_edges(edge_counts)
    return Data(x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=edge_index, edge_weight=edge_weight)


def segment_graph(chroma, window=8, proj_dim=32, sim_threshold=0.6):
    """
    chroma: (n_chroma, frames) array, e.g. from audio_features.chroma_features.
    Segments the track into fixed windows, connects temporally adjacent segments
    and any pair whose cosine similarity exceeds sim_threshold.
    """
    n_chroma, n_frames = chroma.shape
    n_segments = max(1, n_frames // window)
    seg_feats = np.stack([
        chroma[:, i * window:(i + 1) * window].mean(axis=1) for i in range(n_segments)
    ])  # (n_segments, n_chroma)

    rng = np.random.RandomState(0)
    proj = rng.normal(size=(n_chroma, proj_dim)).astype(np.float32)
    x = seg_feats @ proj

    edges, weights = [], []
    for i in range(n_segments - 1):
        edges += [(i, i + 1), (i + 1, i)]
        weights += [1.0, 1.0]

    norm = seg_feats / (np.linalg.norm(seg_feats, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T
    for i in range(n_segments):
        for j in range(i + 2, n_segments):
            if sims[i, j] > sim_threshold:
                edges += [(i, j), (j, i)]
                weights += [float(sims[i, j]), float(sims[i, j])]

    edge_index, edge_weight = _list_to_edges(edges, weights)
    return Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index, edge_weight=edge_weight)


def collate_graphs(graph_list):
    """Batch a list of PyG Data objects for GNN forward passes (Task 2/3/4)."""
    return Batch.from_data_list(graph_list)


def _dict_to_edges(edge_counts):
    if not edge_counts:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
    edge_index = torch.tensor(list(edge_counts.keys()), dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(list(edge_counts.values()), dtype=torch.float32)
    return edge_index, edge_weight


def _list_to_edges(edges, weights):
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight
