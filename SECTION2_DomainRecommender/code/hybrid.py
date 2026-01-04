"""
Hybrid Recommendation (Weighted) for Remote Job Matching

HybridScore = alpha * CB_norm + (1 - alpha) * CF_norm

We combine:
- Content-Based scores from content_based.py
- Item-based Collaborative Filtering scores from collaborative.py

Important:
- Scores are normalized (min-max) before mixing
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# Import our own modules
from content_based import load_processed_data as load_cb_data, train_tfidf, recommend_top_n
from collaborative import load_processed_data as load_cf_data, build_sparse_matrices, train_item_cf, recommend_item_cf


# -----------------------------
# Utilities
# -----------------------------
def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Safe for constant arrays."""
    scores = scores.astype(float)
    smin, smax = scores.min(), scores.max()
    if np.isclose(smin, smax):
        return np.zeros_like(scores)
    return (scores - smin) / (smax - smin)


def normalize_df(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Return a copy with an added column: score_norm."""
    out = df.copy()
    if out.empty:
        out["score_norm"] = []
        return out
    out["score_norm"] = minmax_normalize(out[score_col].to_numpy())
    return out


def combine_weighted(
    cb_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    alpha: float,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Combine two recommendation lists using weighted sum on normalized scores.

    - Items missing from one list get 0 for that component.
    """
    cb_df = normalize_df(cb_df, "score")[["item_id", "score_norm"]].rename(columns={"score_norm": "cb_norm"})
    cf_df = normalize_df(cf_df, "score")[["item_id", "score_norm"]].rename(columns={"score_norm": "cf_norm"})

    merged = pd.merge(cb_df, cf_df, on="item_id", how="outer").fillna(0.0)
    merged["hybrid_score"] = alpha * merged["cb_norm"] + (1.0 - alpha) * merged["cf_norm"]

    merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)
    return merged[["item_id", "hybrid_score", "cb_norm", "cf_norm"]]


# -----------------------------
# Main Hybrid Runner
# -----------------------------
def hybrid_recommendations(
    user_id: str,
    alpha: float = 0.5,
    top_n: int = 10,
    cb_pool: int = 200,
    cf_pool: int = 200
) -> pd.DataFrame:
    """
    Generate hybrid recommendations for a user.

    Args:
        user_id: user identifier (string!)
        alpha: weight for content-based part
        top_n: number of final recommendations
        cb_pool: how many candidates to pull from CB before mixing
        cf_pool: how many candidates to pull from CF before mixing

    Returns:
        DataFrame [item_id, hybrid_score, cb_norm, cf_norm]
    """
    user_id = str(user_id)

    # --- Content-Based ---
    items_cb, interactions_cb = load_cb_data()
    cb_model = train_tfidf(items_cb)
    cb_recs = recommend_top_n(user_id, interactions_cb, cb_model, n=cb_pool)

    # --- Item-CF ---
    items_cf, interactions_cf = load_cf_data()
    md = build_sparse_matrices(interactions_cf)
    cf_model = train_item_cf(md, k=20)
    cf_recs = recommend_item_cf(user_id, cf_model, interactions_cf, n=cf_pool, k_neighbors=20)

    # Combine
    hybrid = combine_weighted(cb_recs, cf_recs, alpha=alpha, top_n=top_n)
    return hybrid


if __name__ == "__main__":
    # Demo run with alphas required by the assignment
    # Using a real user from processed interactions
    _, inter = load_cb_data()
    sample_user = str(inter["user_id"].iloc[0])

    print("Sample user_id =", sample_user)

    for a in [0.3, 0.5, 0.7]:
        print(f"\nHybrid Top-10 (alpha={a}):")
        hyb = hybrid_recommendations(sample_user, alpha=a, top_n=10)
        print(hyb)
