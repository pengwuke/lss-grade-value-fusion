#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10-C3-R1: source-derived ORNet-style progression auxiliary module.

Old-system replacement scope
----------------------------
OLD:
    encoder -> CORAL ordinal logits
    encoder -> binary risk head (training auxiliary)

CANDIDATE:
    encoder -> CORAL ordinal logits -> differentiable 4-class probabilities
                                  -> Linear(4,1) progression score

Main CORAL inference remains unchanged. The progression branch is training-only.

Source mechanism
----------------
Official ORNet:
- produces classification output xc, then scalar regression xr = Linear(num_classes,1)(xc);
- maps ordered classes to equally-spaced continuous progression scores;
- uses SmoothL1 by default;
- combines classification and regression losses as
      (1-alpha) * L_cls + alpha * L_reg.
For the first attributable LSS candidate alpha is frozen to 0.125,
the best reported single-task IDRiD point in the official ORNet README.
No alpha scan is permitted in this branch.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_ALPHA = 0.125
NUM_CLASSES = 4


def coral_class_probabilities(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """Differentiable CORAL logits -> four class probabilities."""
    if ordinal_logits.ndim != 2 or ordinal_logits.shape[1] != 3:
        raise ValueError(f"Expected [B,3] CORAL logits, got {tuple(ordinal_logits.shape)}")
    cumulative = torch.sigmoid(ordinal_logits)
    cumulative = torch.cummin(cumulative, dim=1).values
    probabilities = torch.stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2],
        ],
        dim=1,
    )
    probabilities = torch.clamp(probabilities, min=0.0)
    return probabilities / torch.clamp(probabilities.sum(dim=1, keepdim=True), min=1e-12)


def progression_targets(labels: torch.Tensor) -> torch.Tensor:
    """
    ORNet's default equally-spaced cut construction gives class midpoints
    0,1,2,3 for a four-grade problem. Therefore target = grade as float.
    """
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if torch.any((labels < 0) | (labels > 3)):
        raise ValueError("labels must be in {0,1,2,3}")
    return labels.float()


class ORNetProgressionAux(nn.Module):
    """Source-derived scalar progression head operating on main 4-class state."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        if num_classes != 4:
            raise ValueError("This LSS candidate is frozen to four grades.")
        self.reg = nn.Linear(num_classes, 1)

    def forward(self, ordinal_logits: torch.Tensor) -> torch.Tensor:
        probs = coral_class_probabilities(ordinal_logits)
        return self.reg(probs).squeeze(1)


class ORNetStyleFusionLoss(nn.Module):
    """Official ORNet loss-combination form, adapted to CORAL main loss."""

    def __init__(self, alpha: float = SOURCE_ALPHA):
        super().__init__()
        if abs(alpha - SOURCE_ALPHA) > 1e-12:
            raise ValueError(
                f"C3-R1 alpha is source-frozen to {SOURCE_ALPHA}; no scan allowed."
            )
        self.alpha = float(alpha)
        self.regloss = nn.SmoothL1Loss()

    def forward(
        self,
        ordinal_data_loss: torch.Tensor,
        progression_score: torch.Tensor,
        labels: torch.Tensor,
    ):
        target = progression_targets(labels).type_as(progression_score)
        reg_loss = self.regloss(progression_score, target)
        joint = (1.0 - self.alpha) * ordinal_data_loss + self.alpha * reg_loss
        return joint, reg_loss


class RiskAwareOrdinalProgressionCNN(nn.Module):
    """
    Drop-in old-system model wrapper.

    Public tuple arity is preserved:
        (ordinal_logits, auxiliary_scalar)

    but the second scalar is now ORNet-style progression, not binary risk.
    """

    def __init__(self, source54_module, method: str = "coral", dropout: float = 0.5):
        super().__init__()
        if method != "coral":
            raise ValueError("C3-R1 first candidate is CORAL-only.")
        self.method = method
        self.encoder = source54_module.CustomCNNEncoder(dropout=dropout)
        self.ordinal_head = source54_module.RankConsistentCoralHead(32, 4)
        self.progression_aux = ORNetProgressionAux(4)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        ordinal_logits = self.ordinal_head(h)
        progression_score = self.progression_aux(ordinal_logits)
        return ordinal_logits, progression_score

    def forward_with_state(self, x: torch.Tensor):
        h = self.encoder(x)
        ordinal_logits = self.ordinal_head(h)
        probs = coral_class_probabilities(ordinal_logits)
        progression_score = self.progression_aux.reg(probs).squeeze(1)
        return ordinal_logits, progression_score, h, probs
