"""Losses used by protein-sequence Balanced Contrastive Learning (BCL)."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LogitAdjustedCrossEntropy(nn.Module):
    """Cross entropy with BCL's class-prior adjustment."""

    def __init__(self, class_counts: list[int], tau: float = 1.0) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float)
        if counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("class_counts must contain one positive count per class.")
        self.register_buffer("log_prior", torch.log(counts / counts.sum()))
        self.tau = tau

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.tau * self.log_prior, targets)


class BalancedSupConLoss(nn.Module):
    """BCL loss for pairs of distinct, same-class protein sequences.

    ``features`` has shape ``[batch, 2, feature_dim]``. The two views are real
    protein sequences, not two augmented copies of the same sequence.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, centers: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 2:
            raise ValueError("features must have shape [batch_size, 2, feature_dim].")
        batch_size, _, feature_dim = features.shape
        if targets.shape != (batch_size,):
            raise ValueError("targets must have shape [batch_size].")
        if centers.ndim != 2 or centers.shape[1] != feature_dim:
            raise ValueError("centers must have shape [num_classes, feature_dim].")

        device, num_classes = features.device, centers.shape[0]
        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError("targets must index the supplied class centres.")

        # View-major flattening matches labels.repeat(2): anchors then positives.
        instance_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        all_features = torch.cat((instance_features, centers), dim=0)
        instance_labels = targets.repeat(2)
        all_labels = torch.cat((instance_labels, torch.arange(num_classes, device=device)))
        positive_mask = instance_labels[:, None].eq(all_labels[None, :]).float()
        self_mask = torch.ones_like(positive_mask)
        self_mask.scatter_(1, torch.arange(2 * batch_size, device=device)[:, None], 0)
        positive_mask *= self_mask  # centres balance the denominator, never positives

        logits = instance_features @ all_features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        class_counts = torch.bincount(all_labels, minlength=num_classes).to(features.dtype)
        denominator_weight = class_counts[all_labels].unsqueeze(0) - positive_mask
        exp_logits = torch.exp(logits) * self_mask
        denominator = (exp_logits / denominator_weight).sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - torch.log(denominator)
        return -((positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)).mean()
