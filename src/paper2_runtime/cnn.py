"""Inference definitions extracted unchanged from the author frozen source.
Source file: 54_train_coral_risk_protocol_v3_full468_oof.py
Source SHA256: b8e5360657fa467e74db846c4d03e7f5f80901aa2108e66587b04a58e89cb259
No training controller is imported or executed.
"""

import math
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

class CustomCNNEncoder(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(dropout)
        self.fc_shared = nn.Linear(4 * 4 * 256, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        return self.dropout(F.relu(self.fc_shared(x)))


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class RankConsistentCoralHead(nn.Module):
    def __init__(self, in_features: int = 32, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.score = nn.Linear(in_features, 1, bias=False)
        self.threshold_base = nn.Parameter(torch.tensor(-1.0))
        initial_step = inverse_softplus(1.0)
        self.threshold_delta_raw = nn.Parameter(
            torch.full((num_classes - 2,), initial_step)
        )

    def ordered_thresholds(self) -> torch.Tensor:
        increments = F.softplus(self.threshold_delta_raw) + 1e-4
        return torch.cat(
            [
                self.threshold_base.reshape(1),
                self.threshold_base + torch.cumsum(increments, dim=0),
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        score = self.score(features)
        thresholds = self.ordered_thresholds().reshape(1, -1)
        return score - thresholds


class RiskAwareOrdinalCNN(nn.Module):
    def __init__(self, method: str, dropout: float = 0.5):
        super().__init__()
        self.method = method
        self.encoder = CustomCNNEncoder(dropout=dropout)
        if method == "coral":
            self.ordinal_head = RankConsistentCoralHead(32, 4)
        elif method == "corn":
            self.ordinal_head = nn.Linear(32, 3)
        else:
            raise ValueError(f"Unsupported ordinal method: {method}")
        self.risk_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        ordinal_logits = self.ordinal_head(features)
        risk_logits = self.risk_head(features).squeeze(1)
        return ordinal_logits, risk_logits


def cumulative_and_class_probabilities(
    logits: torch.Tensor,
    method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "coral":
        cumulative = torch.sigmoid(logits)
        cumulative = torch.cummin(cumulative, dim=1).values
    elif method == "corn":
        conditional = torch.sigmoid(logits)
        cumulative = torch.cumprod(conditional, dim=1)
    else:
        raise ValueError(method)

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
    probabilities = probabilities / torch.clamp(
        probabilities.sum(dim=1, keepdim=True),
        min=1e-12,
    )
    return cumulative, probabilities
