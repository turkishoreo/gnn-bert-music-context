"""
Task 3 (Hard): GNN-BERT fusion for multi-label context + emotion regression.

Cross-attention fusion:
  A = softmax(Q K^T / sqrt(d)),   Q = g W_Q,   K = H_text W_K
  z = CONCAT(g, A H_text)
  y_hat = sigmoid(W z)

Multi-task loss:
  L = L_tags + alpha ||v - v_hat||^2 + beta ||a - a_hat||^2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import GNNEncoder
from bert_encoder import BertTextEncoder


class CrossAttentionFusion(nn.Module):
    """Single-head-equivalent scaled dot-product attention of the graph embedding
    over the full BERT token sequence, as specified: Q from g, K from H_text."""

    def __init__(self, graph_dim, text_dim, attn_dim=128, heads=4):
        super().__init__()
        self.attn_dim = attn_dim
        self.heads = heads
        self.q_proj = nn.Linear(graph_dim, attn_dim)
        self.k_proj = nn.Linear(text_dim, attn_dim)
        self.v_proj = nn.Linear(text_dim, attn_dim)

    def forward(self, g, H_text, attention_mask=None):
        """g: (B, graph_dim), H_text: (B, L, text_dim) -> returns (B, attn_dim)"""
        B, L, _ = H_text.shape
        Q = self.q_proj(g).unsqueeze(1)                    # (B, 1, attn_dim)
        K = self.k_proj(H_text)                              # (B, L, attn_dim)
        V = self.v_proj(H_text)                                # (B, L, attn_dim)

        scores = (Q @ K.transpose(1, 2)) / (self.attn_dim ** 0.5)  # (B, 1, L)
        if attention_mask is not None:
            mask = (1 - attention_mask).unsqueeze(1).bool()          # (B, 1, L)
            scores = scores.masked_fill(mask, float("-inf"))
        A = F.softmax(scores, dim=-1)                                 # (B, 1, L)
        pooled = (A @ V).squeeze(1)                                    # (B, attn_dim)
        return pooled


class GNNBERTFusion(nn.Module):
    """
    End-to-end Task 3 model:
      - GNN encoder over the music structure graph -> graph embedding g
      - BERT encoder over lyrics/tags/captions -> H_text, CLS t
      - Cross-attention fusion (or plain concat, spec's ablation) -> z
      - Multi-label tag head (+ optional valence/arousal regression heads for DEAM)
    """

    def __init__(self, graph_in_dim, num_tags, bert_pretrained="bert-base-uncased",
                 bert_synthetic=False, bert_synth_hidden=128, bert_synth_layers=2,
                 gnn_hidden=128, gnn_layers=3, gnn_encoder="graphsage",
                 fusion_type="cross_attention", attn_heads=4, predict_emotion=True):
        super().__init__()
        self.fusion_type = fusion_type
        self.predict_emotion = predict_emotion

        self.gnn = GNNEncoder(graph_in_dim, gnn_hidden, gnn_layers, gnn_encoder)
        self.bert = BertTextEncoder(bert_pretrained, bert_synthetic, bert_synth_hidden,
                                     bert_synth_layers)

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(self.gnn.out_dim, self.bert.hidden_size,
                                               attn_dim=gnn_hidden, heads=attn_heads)
            z_dim = self.gnn.out_dim + gnn_hidden  # CONCAT(g, attended text)
        elif fusion_type == "concat":
            self.fusion = None
            z_dim = self.gnn.out_dim + self.bert.hidden_size
        else:
            raise ValueError(f"unknown fusion_type {fusion_type}")

        self.tag_head = nn.Linear(z_dim, num_tags)
        if predict_emotion:
            self.valence_head = nn.Linear(z_dim, 1)
            self.arousal_head = nn.Linear(z_dim, 1)

    def forward(self, x, edge_index, batch, input_ids, attention_mask=None):
        _, g = self.gnn(x, edge_index, batch)
        H_text, t = self.bert(input_ids, attention_mask)

        if self.fusion_type == "cross_attention":
            attended = self.fusion(g, H_text, attention_mask)
            z = torch.cat([g, attended], dim=-1)
        else:
            z = torch.cat([g, t], dim=-1)

        tag_logits = self.tag_head(z)
        out = {"tag_logits": tag_logits}
        if self.predict_emotion:
            out["valence"] = self.valence_head(z).squeeze(-1)
            out["arousal"] = self.arousal_head(z).squeeze(-1)
        return out


def multi_task_loss(outputs, tag_labels, valence=None, arousal=None, alpha=1.0, beta=1.0):
    """L = L_tags + alpha * MSE(valence) + beta * MSE(arousal), per spec Section 4.3."""
    loss = F.binary_cross_entropy_with_logits(outputs["tag_logits"], tag_labels)
    log = {"tag_loss": loss.item()}
    if valence is not None and "valence" in outputs:
        v_loss = F.mse_loss(outputs["valence"], valence)
        loss = loss + alpha * v_loss
        log["valence_mse"] = v_loss.item()
    if arousal is not None and "arousal" in outputs:
        a_loss = F.mse_loss(outputs["arousal"], arousal)
        loss = loss + beta * a_loss
        log["arousal_mse"] = a_loss.item()
    log["total_loss"] = loss.item()
    return loss, log
