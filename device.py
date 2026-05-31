"""Torch device selection for MEOW deep models (DeepLOB, TLOB, …)."""
from __future__ import annotations

import os


def get_torch_device():
    """MEOW_DEVICE=cuda|cpu (default cuda if available)."""
    import torch

    pref = os.environ.get("MEOW_DEVICE", "cuda").strip().lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref if ":" in pref else "cuda")
    return torch.device("cpu")
