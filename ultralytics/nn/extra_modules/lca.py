# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class LCA(nn.Module):
    """Lightweight Context Attention with a main conv path and a light context branch."""

    def __init__(self, c1: int, c2: int, r: int = 16):
        super().__init__()
        self.main = Conv(c1, c2, 3, 1)
        c_ = max(c2 // r, 8)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c2, c_, 1, 1),
            Conv(c_, c2, 1, 1, act=nn.Sigmoid()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        return x * self.context(x)
