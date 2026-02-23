# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Efficient Channel Attention (ECA) module."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("ECA",)


class ECA(nn.Module):
    """Efficient Channel Attention module.

    Applies channel attention using global average pooling and a lightweight 1D convolution.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        conv (nn.Conv1d): 1D convolution for local cross-channel interaction.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, k_size: int = 3) -> None:
        """Initialize ECA module.

        Args:
            k_size (int): Kernel size for the 1D convolution.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ECA attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C, H, W).
        """
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, 1, C)
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        # (B, 1, C) -> (B, C, 1, 1)
        y = self.act(y).transpose(-1, -2).unsqueeze(-1)
        return x * y
