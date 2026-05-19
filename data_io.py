"""Stable data I/O for MEOW — agents should use this, not rewrite loaders blindly."""
from __future__ import annotations

import bisect
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Single physical copy: /data/moew/data/*.h5
CANONICAL_DATA_DIR = Path("/data/moew/data")


def default_h5dir() -> str:
    """Resolve the MEOW dataset location robustly across worktrees."""
    if os.environ.get("MEOW_DATA_DIR"):
        return os.environ["MEOW_DATA_DIR"]
    local_candidates = [
        Path(__file__).resolve().parent / "data",
        Path("data"),
    ]
    for candidate in local_candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return str(CANONICAL_DATA_DIR)


def verify_data_dir(h5dir: str | None = None) -> str:
    path = Path(h5dir or default_h5dir())
    if not path.is_dir():
        raise FileNotFoundError(
            f"MEOW data not found: {path}. Expected 144 *.h5 under /data/moew/data/"
        )
    n = len(list(path.glob("*.h5")))
    if n == 0:
        raise FileNotFoundError(f"No .h5 files in {path}")
    return str(path)

# Train / test split (fixed by grader)
TRAIN_START, TRAIN_END = 20230601, 20231130
TEST_START, TEST_END = 20231201, 20231229

INDEX_COLS = ["symbol", "date", "interval"]
LABEL_COL = "fret12"

# Raw LOB / trade / price fields useful for deep models (see DATA.md for full schema)
LOB_PRICE_SIZE_COLS = [
    "bid0", "ask0", "bid4", "ask4", "bid9", "ask9", "bid19", "ask19",
    "bsize0", "asize0", "bsize0_4", "asize0_4", "bsize5_9", "asize5_9",
    "bsize10_19", "asize10_19",
    "btr0_4", "atr0_4", "btr5_9", "atr5_9", "btr10_19", "atr10_19",
    "midpx", "lastpx",
    "tradeBuyQty", "tradeSellQty", "tradeBuyTurnover", "tradeSellTurnover",
]

DEFAULT_FEATURE_COLS = LOB_PRICE_SIZE_COLS  # agents may extend


class Calendar:
  def __init__(self, calendar_path: Optional[str] = None):
    if calendar_path is None:
      calendar_path = os.path.join(os.path.dirname(__file__), "resources", "calendar")
    with open(calendar_path) as f:
      self.trading_days = sorted(int(x) for x in f.read().splitlines())
    self.trading_day_set = set(self.trading_days)

  def range(self, start: int, end: int) -> List[int]:
    i = bisect.bisect_left(self.trading_days, start)
    j = bisect.bisect_right(self.trading_days, end)
    return self.trading_days[i:j]


def load_day(h5dir: str, date: int, calendar: Optional[Calendar] = None) -> pd.DataFrame:
  calendar = calendar or Calendar()
  if date not in calendar.trading_day_set:
    raise ValueError(f"Not a trading day: {date}")
  path = os.path.join(h5dir, f"{date}.h5")
  df = pd.read_hdf(path)
  df = df.copy()
  df["date"] = date
  front = [c for c in INDEX_COLS if c in df.columns]
  rest = [c for c in df.columns if c not in front]
  return df[front + rest]


def iter_days(h5dir: str, dates: Iterable[int]) -> Iterable[pd.DataFrame]:
  cal = Calendar()
  for d in dates:
    yield load_day(h5dir, d, cal)


def train_test_dates() -> tuple[list[int], list[int]]:
  cal = Calendar()
  return cal.range(TRAIN_START, TRAIN_END), cal.range(TEST_START, TEST_END)


def frame_to_xy(
  df: pd.DataFrame,
  feature_cols: Optional[List[str]] = None,
  label_col: str = LABEL_COL,
  max_rows: Optional[int] = None,
  seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
  feature_cols = feature_cols or [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
  sub = df[feature_cols + [label_col]].replace([np.inf, -np.inf], np.nan).dropna()
  if max_rows is not None and len(sub) > max_rows:
    sub = sub.sample(n=max_rows, random_state=seed)
  x = sub[feature_cols].to_numpy(dtype=np.float32)
  y = sub[label_col].to_numpy(dtype=np.float32)
  return x, y
