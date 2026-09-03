"""
Task 1 (Easy): BERT baseline for music tag/caption understanding.

  t = BERT_CLS(X_text)
  y_hat_k = sigmoid(w_k^T t + b_k)
  L_BERT = -(1/K) sum_k [y_k log y_hat_k + (1 - y_k) log(1 - y_hat_k)]

Also exposes `encode()` standalone so gnn_model / fusion_model / contrastive can
reuse the same text tower (BERT is shared infrastructure across all 4 tasks).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel


class BertTextEncoder(nn.Module):
    """Wraps either a real pretrained BERT or a small randomly-initialized BertModel
    (synthetic mode: no network access to the HF Hub needed)."""

    def __init__(self, pretrained_name="bert-base-uncased", synthetic=False,
                 synthetic_hidden_size=128, synthetic_num_layers=2, freeze=False):
        super().__init__()
        self.synthetic = synthetic
        if synthetic:
            cfg = BertConfig(
                vocab_size=3000,
                hidden_size=synthetic_hidden_size,
                num_hidden_layers=synthetic_num_layers,
                num_attention_heads=4,
                intermediate_size=synthetic_hidden_size * 4,
                max_position_embeddings=256,
            )
            self.bert = BertModel(cfg)
            self.hidden_size = synthetic_hidden_size
            self.tokenizer = None  # synthetic tokens are pre-tokenized ints, see synthetic_data.py
        else:
            self.bert = AutoModel.from_pretrained(pretrained_name)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
            self.hidden_size = self.bert.config.hidden_size

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, texts, max_len=128):
        if self.synthetic:
            raise RuntimeError("Synthetic mode uses pre-tokenized int tensors; "
                               "see synthetic_data.py (no real tokenizer available).")
        enc = self.tokenizer(texts, padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def forward(self, input_ids, attention_mask=None):
        """Returns (H, pooled): H = full contextual embeddings (L x d).
        pooled = masked mean over tokens (NOT the raw CLS token). Raw CLS
        embeddings from a BERT that hasn't been fine-tuned for this task are a
        known-weak sentence representation; masked mean-pooling is the more
        robust default, especially important when using --freeze_encoder,
        where the encoder never adapts to the task at all."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state
        if attention_mask is None:
            pooled = H.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(H.dtype)  # (B, L, 1)
            summed = (H * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        return H, pooled


class BertTagClassifier(nn.Module):
    """Task 1 end-to-end: BERT encoder + multi-label tag classification head."""

    def __init__(self, num_tags, pretrained_name="bert-base-uncased", synthetic=False,
                 synthetic_hidden_size=128, synthetic_num_layers=2, freeze_encoder=False):
        super().__init__()
        self.encoder = BertTextEncoder(pretrained_name, synthetic, synthetic_hidden_size,
                                        synthetic_num_layers, freeze_encoder)
        self.head = nn.Linear(self.encoder.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask=None):
        _, cls = self.encoder(input_ids, attention_mask)
        logits = self.head(cls)
        return logits  # BCEWithLogitsLoss expected downstream (numerically stable sigmoid+BCE)
