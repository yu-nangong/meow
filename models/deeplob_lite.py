"""
DeepLOB-inspired encoder (Zhang et al., 2019).
Simplified: Conv1d on feature axis + LSTM + scalar head.
Agents should extend toward full DeepLOB (Inception modules, proper LOB tensor).
"""
from __future__ import annotations

import torch
from torch import nn


class DeepLOBLite(nn.Module):
  def __init__(self, n_features: int, hidden: int = 64, lstm_layers: int = 1):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv1d(1, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv1d(32, 32, kernel_size=3, padding=1),
      nn.ReLU(),
    )
    self.lstm = nn.LSTM(
      input_size=32 * n_features,
      hidden_size=hidden,
      num_layers=lstm_layers,
      batch_first=True,
    )
    self.head = nn.Linear(hidden, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, F) row-wise snapshot -> treat as length-F sequence with 1 channel
    x = x.unsqueeze(1)  # (B, 1, F)
    h = self.conv(x)  # (B, 32, F)
    h = h.flatten(1).unsqueeze(1)  # (B, 1, 32*F)
    out, _ = self.lstm(h)
    return self.head(out[:, -1, :]).squeeze(-1)
