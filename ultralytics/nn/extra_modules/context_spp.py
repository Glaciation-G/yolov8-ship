# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class ContextSPP(nn.Module):
    """Lightweight context SPP using multi-scale pooling + pointwise convs."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (3, 5, 7), c_: int | None = None):
        super().__init__()
        c_ = c_ or max(c1 // 4, 16)  # smaller than SPPF hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in k])
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y = [x]
        y.extend(m(x) for m in self.m)
        return self.cv2(torch.cat(y, 1))
