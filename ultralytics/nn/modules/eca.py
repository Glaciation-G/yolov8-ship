import math

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention.

    If ``k_size`` is None, the 1D kernel size is selected adaptively from channel count.
    """

    def __init__(self, channels: int, k_size: int | None = None, gamma: int = 2, b: int = 1):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(max(channels, 1)) + b) / gamma))
            k_size = t if t % 2 else t + 1
            k_size = max(k_size, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)
        y = self.sigmoid(self.conv(y)).view(b, c, 1, 1)
        return x * y
