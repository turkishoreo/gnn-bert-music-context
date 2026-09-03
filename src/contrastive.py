"""
Task 4 (Advanced): Cross-modal MusicCaps alignment via contrastive learning.

InfoNCE loss for paired (graph, caption) (g_i, t_i):
  L_NCE = -log( exp(sim(g_i,t_i)/tau) / sum_j exp(sim(g_i,t_j)/tau) )
  sim(u, v) = u^T v / (||u|| ||v||)

Retrieval metrics: Caption->Audio R@1/R@5/R@10, Audio->Caption R@K.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import GNNEncoder
from bert_encoder import BertTextEncoder


class DualEncoder(nn.Module):
    """Separate GNN (audio-graph) and BERT (caption) towers projected into a
    shared embedding space, per spec Algorithm 4."""

    def __init__(self, graph_in_dim, embedding_dim=128,
                 bert_pretrained="bert-base-uncased", bert_synthetic=False,
                 bert_synth_hidden=128, bert_synth_layers=2,
                 gnn_hidden=128, gnn_layers=3, gnn_encoder="graphsage"):
        super().__init__()
        self.gnn = GNNEncoder(graph_in_dim, gnn_hidden, gnn_layers, gnn_encoder)
        self.bert = BertTextEncoder(bert_pretrained, bert_synthetic, bert_synth_hidden,
                                     bert_synth_layers)
        self.graph_proj = nn.Linear(self.gnn.out_dim, embedding_dim)
        self.text_proj = nn.Linear(self.bert.hidden_size, embedding_dim)

    def encode_graph(self, x, edge_index, batch):
        _, g = self.gnn(x, edge_index, batch)
        g = self.graph_proj(g)
        return F.normalize(g, dim=-1)

    def encode_text(self, input_ids, attention_mask=None):
        _, t = self.bert(input_ids, attention_mask)
        t = self.text_proj(t)
        return F.normalize(t, dim=-1)

    def forward(self, x, edge_index, batch, input_ids, attention_mask=None):
        g = self.encode_graph(x, edge_index, batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(g, t, temperature=0.07):
    """Symmetric InfoNCE over a batch of N paired (graph, caption) embeddings.
    g, t are already L2-normalized, shape (N, d)."""
    N = g.shape[0]
    sim = g @ t.t() / temperature  # (N, N), sim[i, j] = sim(g_i, t_j)
    labels = torch.arange(N, device=g.device)
    loss_g2t = F.cross_entropy(sim, labels)        # audio -> caption
    loss_t2g = F.cross_entropy(sim.t(), labels)    # caption -> audio
    return (loss_g2t + loss_t2g) / 2


@torch.no_grad()
def retrieval_recall_at_k(g, t, ks=(1, 5, 10)):
    """g, t: (N, d) normalized embeddings of paired (graph_i, caption_i).
    Returns dict with audio->caption and caption->audio R@K."""
    N = g.shape[0]
    sim = g @ t.t()  # (N, N)
    results = {}

    # audio -> caption: for each row i, is the correct column i in top-K?
    ranks_a2c = (-sim).argsort(dim=1)
    for k in ks:
        hit = (ranks_a2c[:, :k] == torch.arange(N, device=g.device).unsqueeze(1)).any(dim=1)
        results[f"audio2caption_R@{k}"] = hit.float().mean().item()

    ranks_c2a = (-sim.t()).argsort(dim=1)
    for k in ks:
        hit = (ranks_c2a[:, :k] == torch.arange(N, device=g.device).unsqueeze(1)).any(dim=1)
        results[f"caption2audio_R@{k}"] = hit.float().mean().item()

    return results


@torch.no_grad()
def top_k_matches(sim_row, k=3):
    """Given one row of the similarity matrix (query vs all candidates), return
    top-k candidate indices for a qualitative retrieval example."""
    return torch.topk(sim_row, k=min(k, sim_row.shape[0])).indices.tolist()
