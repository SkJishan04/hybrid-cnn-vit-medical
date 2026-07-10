"""
Focal loss — down-weights easy/majority-class examples so the model
doesn't just learn to predict "nv" (melanocytic nevi, ~67% of HAM10000)
for everything and call it a day.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()


def compute_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights, normalized so they average to 1."""
    weights = 1.0 / class_counts.clamp(min=1)
    weights = weights / weights.sum() * len(class_counts)
    return weights