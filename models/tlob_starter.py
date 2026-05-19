"""
TLOB-style starter (Berti & Kasneci, 2024): Transformer over LOB feature tokens.
Agents should add proper positional encoding, causal mask, and LOB patch embedding.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TLOBStarter(nn.Module):
  def __init__(
    self,
    n_features: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
  ):
    super().__init__()
    self.token_embed = nn.Linear(1, d_model)
    self.pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
    enc_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=d_model * 4,
      dropout=dropout,
      batch_first=True,
      norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    self.head = nn.Linear(d_model, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, F) — each feature is a token (scalar -> d_model)
    b, f = x.shape
    tokens = self.token_embed(x.unsqueeze(-1)) + self.pos[:, :f, :]
    h = self.encoder(tokens)
    return self.head(h.mean(dim=1)).squeeze(-1)
