"""
Content-Based Recommendation for Remote Job Matching

This module implements:
1) TF-IDF item representation (from job text)
2) User profile construction (weighted average of item vectors)
3) Cosine similarity scoring and top-N recommendation
4) Item-based kNN variant (k=10/20) for predicting user preference
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentBasedModel:
    """Holds all objects needed for content-based recommendation."""
    vectorizer: TfidfVectorizer
    item_ids: np.ndarray                 # shape (n_items,)
    item_matrix: "np.ndarray"            # sparse matrix (n_items, n_features)


def _project_paths() -> Dict[str, str]:
    """Return project paths using RELATIVE references only."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # SECTION2_DomainRecommender/
    processed_dir = os.path.join(base_dir, "data", "processed")
    return {
        "base_dir": base_dir,
        "items_csv": os.path.join(processed_dir, "items.csv"),
        "interactions_csv": os.path.join(processed_dir, "interactions.csv"),
    }


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed items and interactions with safe dtypes."""
    paths = _project_paths()
    items = pd.read_csv(paths["items_csv"])
    interactions = pd.read_csv(paths["interactions_csv"])

    # IMPORTANT: keep user_id as string (your IDs contain decimals and can be corrupted as float)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(int)
    interactions["rating"] = interactions["rating"].astype(float)

    items["item_id"] = items["item_id"].astype(int)
    items["text"] = items["text"].fillna("").astype(str)

    return items, interactions


def train_tfidf(items: pd.DataFrame,
               max_features: int = 20000,
               min_df: int = 2,
               stop_words: Optional[str] = None) -> ContentBasedModel:
    """
    Train TF-IDF vectors for all job items.

    Args:
        items: DataFrame with columns [item_id, text]
        max_features: limit vocabulary size (keeps model lightweight)
        min_df: ignore rare terms that appear in fewer than min_df documents
        stop_words: e.g., "english" (BUT your text looks Turkish; keep None)

    Returns:
        ContentBasedModel with item vectors and vectorizer.
    """
    # Ensure text column exists and is string
    texts = items["text"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words=stop_words,
        ngram_range=(1, 2),  # unigrams + bigrams often improves relevance
    )
    item_matrix = vectorizer.fit_transform(texts)

    return ContentBasedModel(
        vectorizer=vectorizer,
        item_ids=items["item_id"].to_numpy(),
        item_matrix=item_matrix,
    )


def build_user_profile(user_interactions: pd.DataFrame,
                       model: ContentBasedModel) -> np.ndarray:
    """
    Build a user profile vector using weighted average of item vectors.

    Weight rule: rating value (3 or 5) is used as weight.

    Args:
        user_interactions: rows for a single user with columns [item_id, rating]
        model: trained content model

    Returns:
        1 x n_features dense numpy vector (user profile)
    """
    # Map item_id -> row index in item_matrix
    item_id_to_index = {iid: idx for idx, iid in enumerate(model.item_ids)}

    indices = []
    weights = []
    for _, row in user_interactions.iterrows():
        iid = row["item_id"]
        if iid in item_id_to_index:
            indices.append(item_id_to_index[iid])
            weights.append(float(row["rating"]))

    if not indices:
        # No known items for user in item matrix
        return np.zeros((model.item_matrix.shape[1],), dtype=np.float32)

    item_vecs = model.item_matrix[indices]  # sparse (m, n_features)
    weights = np.array(weights, dtype=np.float32)

    # Weighted average: sum(w_i * v_i) / sum(w_i)
    user_vec = item_vecs.multiply(weights[:, None]).sum(axis=0) / (weights.sum() + 1e-9)
    return np.asarray(user_vec).ravel().astype(np.float32)


def recommend_top_n(user_id,
                    interactions: pd.DataFrame,
                    model: ContentBasedModel,
                    n: int = 10) -> pd.DataFrame:
    """
    Recommend top-N items for a given user.

    Returns:
        DataFrame with columns [item_id, score]
    """
    # Ensure consistent type for filtering
    user_id = str(user_id)

    # Filter this user's history
    user_hist = interactions[interactions["user_id"] == user_id][["item_id", "rating"]]
    user_vec = build_user_profile(user_hist, model)

    # If user profile is all zeros (no history), return empty → cold-start handled elsewhere
    if np.allclose(user_vec, 0):
        return pd.DataFrame(columns=["item_id", "score"])

    # Cosine similarity between user vector and all items
    scores = cosine_similarity(user_vec.reshape(1, -1), model.item_matrix).ravel()

    # Remove already seen items
    seen = set(user_hist["item_id"].tolist())
    candidates = [(iid, float(scores[idx])) for idx, iid in enumerate(model.item_ids) if iid not in seen]

    # Sort by score desc and return top-N
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:n]
    return pd.DataFrame(top, columns=["item_id", "score"])


def item_knn_predict(user_id: int,
                     target_item_id: int,
                     interactions: pd.DataFrame,
                     model: ContentBasedModel,
                     k: int = 10) -> float:
    """
    Item-based kNN prediction for a user and a target item.

    Idea:
    - Find the k most similar items to target_item (using TF-IDF cosine similarity)
    - Look at which of those neighbor items the user has rated
    - Predict as weighted average of user's ratings on neighbors

    Returns:
        Predicted score (not clipped). You can later clip to [1,5] if desired.
    """
    item_id_to_index = {iid: idx for idx, iid in enumerate(model.item_ids)}
    if target_item_id not in item_id_to_index:
        return 0.0

    target_idx = item_id_to_index[target_item_id]

    # Similarity of target item to all items
    target_vec = model.item_matrix[target_idx]
    sims = cosine_similarity(target_vec, model.item_matrix).ravel()

    # User history
    user_hist = interactions[interactions["user_id"] == user_id][["item_id", "rating"]]
    user_ratings = {int(r.item_id): float(r.rating) for r in user_hist.itertuples(index=False)}

    # Get top-k neighbor item_ids (excluding the item itself)
    neighbors = []
    for idx, iid in enumerate(model.item_ids):
        if iid == target_item_id:
            continue
        neighbors.append((int(iid), float(sims[idx])))
    neighbors.sort(key=lambda x: x[1], reverse=True)
    neighbors = neighbors[:k]

    # Weighted average using similarities, but only for neighbors user has rated
    num = 0.0
    den = 0.0
    for iid, sim in neighbors:
        if iid in user_ratings and sim > 0:
            num += sim * user_ratings[iid]
            den += sim

    return num / (den + 1e-9)


if __name__ == "__main__":
    # Quick sanity run: train TF-IDF and show recommendations for 1 sample user
    items_df, interactions_df = load_processed_data()
    model = train_tfidf(items_df)

    sample_user = str(interactions_df["user_id"].iloc[0])
    recs10 = recommend_top_n(sample_user, interactions_df, model, n=10)
    recs20 = recommend_top_n(sample_user, interactions_df, model, n=20)

    print(f"Sample user_id = {sample_user}")
    print("\nTop-10 recommendations (content-based):")
    print(recs10.head(10))

    print("\nTop-20 recommendations (content-based):")
    print(recs20.head(20))

    # Example: item-kNN prediction for one recommended item
    if not recs10.empty:
        target_item = int(recs10.iloc[0]["item_id"])
        pred10 = item_knn_predict(sample_user, target_item, interactions_df, model, k=10)
        pred20 = item_knn_predict(sample_user, target_item, interactions_df, model, k=20)
        print(f"\nItem-kNN predicted preference for item {target_item}: k=10 -> {pred10:.4f}, k=20 -> {pred20:.4f}")
