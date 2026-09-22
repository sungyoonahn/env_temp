"""Embedding-space classifier used after frozen ESM-2 extraction."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EmbeddingClassifier(nn.Module):
    """Linear or one-hidden-layer MLP classifier that returns raw logits."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        classifier: str = "mlp",
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        if classifier == "linear":
            self.network = nn.Linear(embedding_dim, num_classes)
        elif classifier == "mlp":
            self.network = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError("classifier must be linear or mlp.")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)

    @torch.inference_mode()
    def predict_proba(self, embeddings: torch.Tensor, device: torch.device, batch_size: int = 4096) -> torch.Tensor:
        self.eval()
        outputs = []
        for start in range(0, len(embeddings), batch_size):
            logits = self(embeddings[start : start + batch_size].to(device)).float()
            outputs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(outputs, dim=0)
