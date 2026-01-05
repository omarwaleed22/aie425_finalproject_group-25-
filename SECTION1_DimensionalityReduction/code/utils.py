"""
Utility helpers shared by the collaborative filtering notebooks.

These functions are intentionally small and dependency-light so you can
reuse them across notebooks without copy/paste.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# I/O helpers
# -----------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it doesn't exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a dataframe as CSV, ensuring the parent folder exists."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)
    return p


def load_ratings_csv(
    path: str | Path,
    nrows: Optional[int] = None,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Load a ratings CSV and standardize column existence.
    Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(path, nrows=nrows)
    missing = [c for c in (user_col, item_col, rating_col) if c not in df.columns]
    if missing:
        raise ValueError(f"ratings csv is missing columns: {missing}. Found: {list(df.columns)}")
    return df[[user_col, item_col, rating_col]].copy()


# -----------------------------
# Matrix utilities
# -----------------------------

def make_ratings_matrix(
    df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Convert long ratings (userId, movieId, rating) into a user–item matrix.
    By default fills missing entries with 0.0 (common for SVD baselines).
    """
    mat = df.pivot_table(index=user_col, columns=item_col, values=rating_col)
    return mat.fillna(fill_value)


def observed_mask(mat: pd.DataFrame, missing_value: float = 0.0) -> np.ndarray:
    """Return a boolean mask for observed entries (non-missing)."""
    return (mat.values != missing_value)


def rmse(
    true: np.ndarray,
    pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute RMSE. If `mask` is provided, computes RMSE only over masked entries.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        diff = (true[mask] - pred[mask])
    else:
        diff = (true - pred).ravel()
    return float(np.sqrt(np.mean(diff ** 2)))


# -----------------------------
# Simple analytics helpers
# -----------------------------

@dataclass(frozen=True)
class ActivityThresholds:
    """
    Thresholds for classifying users by number of ratings.
    Example:
      cold:   n < cold_max
      medium: cold_max <= n < rich_min
      rich:   n >= rich_min
    """
    cold_max: int = 20
    rich_min: int = 100


def user_item_counts(
    df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> Tuple[pd.Series, pd.Series]:
    """Return (#ratings per user, #ratings per item)."""
    n_u = df.groupby(user_col).size().sort_values(ascending=False)
    n_i = df.groupby(item_col).size().sort_values(ascending=False)
    return n_u, n_i


def user_item_means(
    df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> Tuple[pd.Series, pd.Series]:
    """Return (mean rating per user, mean rating per item)."""
    rbar_u = df.groupby(user_col)[rating_col].mean()
    rbar_i = df.groupby(item_col)[rating_col].mean()
    return rbar_u, rbar_i


def label_users_by_activity(
    n_u: pd.Series,
    thresholds: ActivityThresholds = ActivityThresholds(),
) -> pd.Series:
    """
    Label users as: cold / medium / rich based on rating counts.
    Input: n_u is a Series indexed by userId with counts.
    """
    def _label(n: int) -> str:
        if n < thresholds.cold_max:
            return "cold"
        if n >= thresholds.rich_min:
            return "rich"
        return "medium"

    return n_u.astype(int).map(_label)


def top_popular_items(
    n_i: pd.Series,
    top_k: int = 100,
) -> pd.Series:
    """Return the top-k most-rated items (by count)."""
    return n_i.head(top_k)
