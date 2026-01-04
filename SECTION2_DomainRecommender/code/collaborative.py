"""
Collaborative Filtering + SVD for Remote Job Matching Recommender

Implements:
1) Item-based Collaborative Filtering (cosine) using k-NN on sparse item-user matrix
2) Matrix Factorization using TruncatedSVD (k=10 or 20)

Notes:
- Uses sparse matrices (does NOT build a dense 42k x 21k matrix)
- Keeps user_id as STRING (critical for this dataset)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


# -----------------------------
# Paths + Data Loading
# -----------------------------
def _project_paths() -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.dirname(__file__))  # SECTION2_DomainRecommender/
    processed_dir = os.path.join(base_dir, "data", "processed")
    return {
        "base_dir": base_dir,
        "items_csv": os.path.join(processed_dir, "items.csv"),
        "interactions_csv": os.path.join(processed_dir, "interactions.csv"),
    }


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load items + interactions with correct dtypes."""
    paths = _project_paths()
    items = pd.read_csv(paths["items_csv"])
    interactions = pd.read_csv(paths["interactions_csv"])

    # IMPORTANT: user_id must be string (your IDs include decimals)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(int)
    interactions["rating"] = interactions["rating"].astype(float)

    items["item_id"] = items["item_id"].astype(int)
    items["text"] = items["text"].fillna("").astype(str)
    return items, interactions


# -----------------------------
# Sparse Matrix Builder
# -----------------------------
@dataclass
class MatrixData:
    """Holds sparse matrices and index mappings."""
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_to_idx: Dict[str, int]
    item_to_idx: Dict[int, int]
    X_ui: csr_matrix  # shape (n_users, n_items) user-item
    X_iu: csr_matrix  # shape (n_items, n_users) item-user


def build_sparse_matrices(interactions: pd.DataFrame) -> MatrixData:
    """
    Build sparse user-item and item-user matrices.

    Returns:
        MatrixData with mappings and CSR matrices.
    """
    user_ids = interactions["user_id"].unique()
    item_ids = interactions["item_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {it: j for j, it in enumerate(item_ids)}

    row = interactions["user_id"].map(user_to_idx).to_numpy()
    col = interactions["item_id"].map(item_to_idx).to_numpy()
    data = interactions["rating"].to_numpy()

    X_ui = csr_matrix((data, (row, col)), shape=(len(user_ids), len(item_ids)))
    X_iu = X_ui.T.tocsr()

    return MatrixData(
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        X_ui=X_ui,
        X_iu=X_iu,
    )


# -----------------------------
# Item-based CF (k-NN, cosine)
# -----------------------------
@dataclass
class ItemCFModel:
    nn: NearestNeighbors
    matrix_data: MatrixData


def train_item_cf(matrix_data: MatrixData, k: int = 20) -> ItemCFModel:
    """
    Train item-based CF using k-NN over item-user vectors.

    We fit NearestNeighbors on X_iu (items x users).
    """
    nn = NearestNeighbors(
        n_neighbors=k + 1,      # +1 because the nearest neighbor is the item itself
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(matrix_data.X_iu)
    return ItemCFModel(nn=nn, matrix_data=matrix_data)


def recommend_item_cf(
    user_id: str,
    model: ItemCFModel,
    interactions: pd.DataFrame,
    n: int = 10,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """
    Recommend top-N items for a user using item-based CF.

    Method:
    - For each item the user interacted with, find k similar items
    - Aggregate candidate scores = sum(similarity * user_rating)
    - Remove already seen items
    """
    user_id = str(user_id)
    md = model.matrix_data

    if user_id not in md.user_to_idx:
        return pd.DataFrame(columns=["item_id", "score"])

    # Get this user's history
    user_hist = interactions[interactions["user_id"] == user_id][["item_id", "rating"]]
    if user_hist.empty:
        return pd.DataFrame(columns=["item_id", "score"])

    seen = set(user_hist["item_id"].tolist())
    scores: Dict[int, float] = {}

    for iid, rating in user_hist.itertuples(index=False):
        if iid not in md.item_to_idx:
            continue
        item_idx = md.item_to_idx[iid]

        # Neighbors for this item
        distances, indices = model.nn.kneighbors(md.X_iu[item_idx], n_neighbors=k_neighbors + 1)
        distances = distances.ravel()
        indices = indices.ravel()

        for dist, neigh_idx in zip(distances, indices):
            neigh_item_id = int(md.item_ids[neigh_idx])

            if neigh_item_id == iid:
                continue
            if neigh_item_id in seen:
                continue

            sim = 1.0 - float(dist)  # cosine distance -> similarity
            if sim <= 0:
                continue

            scores[neigh_item_id] = scores.get(neigh_item_id, 0.0) + sim * float(rating)

    if not scores:
        return pd.DataFrame(columns=["item_id", "score"])

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(top, columns=["item_id", "score"])


# -----------------------------
# SVD (Matrix Factorization)
# -----------------------------
@dataclass
class SVDModel:
    svd: TruncatedSVD
    matrix_data: MatrixData
    user_factors: np.ndarray  # shape (n_users, k)
    item_factors: np.ndarray  # shape (n_items, k)


def train_svd(matrix_data: MatrixData, k: int = 20, random_state: int = 42) -> SVDModel:
    """
    Train TruncatedSVD on the user-item matrix.

    TruncatedSVD works on sparse matrices and is suitable for large sparse data.
    """
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    user_factors = svd.fit_transform(matrix_data.X_ui)     # (n_users, k)
    item_factors = svd.components_.T                       # (n_items, k)

    return SVDModel(
        svd=svd,
        matrix_data=matrix_data,
        user_factors=user_factors,
        item_factors=item_factors,
    )


def recommend_svd(
    user_id: str,
    model: SVDModel,
    interactions: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Recommend top-N items for a user using SVD reconstructed scores.

    score(u, i) = dot(user_factors[u], item_factors[i])
    """
    user_id = str(user_id)
    md = model.matrix_data
    if user_id not in md.user_to_idx:
        return pd.DataFrame(columns=["item_id", "score"])

    uidx = md.user_to_idx[user_id]
    user_vec = model.user_factors[uidx]  # (k,)

    scores = model.item_factors @ user_vec  # (n_items,)
    scores = scores.astype(float)

    # remove already seen
    seen = set(interactions[interactions["user_id"] == user_id]["item_id"].tolist())

    candidates = []
    for j, iid in enumerate(md.item_ids):
        iid = int(iid)
        if iid in seen:
            continue
        candidates.append((iid, float(scores[j])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:n]
    return pd.DataFrame(top, columns=["item_id", "score"])


# -----------------------------
# Demo Run (for sanity)
# -----------------------------
if __name__ == "__main__":
    items_df, interactions_df = load_processed_data()
    md = build_sparse_matrices(interactions_df)

    # pick a real user (string)
    sample_user = str(interactions_df["user_id"].iloc[0])

    print("Sample user_id =", sample_user)
    print("Users:", len(md.user_ids), "Items:", len(md.item_ids), "Interactions:", md.X_ui.nnz)

    # Item-based CF
    print("\nTraining Item-CF (kNN cosine)...")
    item_cf = train_item_cf(md, k=20)
    recs_cf = recommend_item_cf(sample_user, item_cf, interactions_df, n=10, k_neighbors=20)
    print("\nTop-10 recommendations (Item-CF):")
    print(recs_cf)

    # SVD
    print("\nTraining SVD (k=20)...")
    svd_model = train_svd(md, k=20)
    recs_svd = recommend_svd(sample_user, svd_model, interactions_df, n=10)
    print("\nTop-10 recommendations (SVD):")
    print(recs_svd)



