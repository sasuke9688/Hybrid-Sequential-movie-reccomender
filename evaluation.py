"""
Evaluation metrics for the recommendation system.
Implements Precision@K, Recall@K, and NDCG@K.
"""

import numpy as np


def precision_at_k(recommended, relevant, k=10):
    """
    Precision@K: fraction of recommended items that are relevant.

    Args:
        recommended: list of recommended item indices (ordered)
        relevant: set of relevant item indices
        k: number of recommendations to consider

    Returns: precision score
    """
    if k == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """
    Recall@K: fraction of relevant items that are recommended.

    Args:
        recommended: list of recommended item indices (ordered)
        relevant: set of relevant item indices
        k: number of recommendations to consider

    Returns: recall score
    """
    if len(relevant) == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def dcg_at_k(relevance_scores, k=10):
    """
    Discounted Cumulative Gain at K.

    Args:
        relevance_scores: list of relevance scores in ranking order
        k: number of positions to consider

    Returns: DCG score
    """
    relevance_scores = np.array(relevance_scores[:k], dtype=np.float64)
    positions = np.arange(1, len(relevance_scores) + 1)
    discounts = np.log2(positions + 1)
    return np.sum(relevance_scores / discounts)


def ndcg_at_k(recommended, relevant, k=10):
    """
    Normalized Discounted Cumulative Gain at K.

    NDCG = DCG / IDCG

    Args:
        recommended: list of recommended item indices (ordered)
        relevant: set of relevant item indices
        k: number of recommendations to consider

    Returns: NDCG score
    """
    # Build binary relevance vector for the recommended list
    relevance = [1.0 if item in relevant else 0.0 for item in recommended[:k]]

    dcg = dcg_at_k(relevance, k)

    # Ideal DCG: all relevant items at the top
    ideal_relevance = sorted(relevance, reverse=True)
    ideal_length = min(len(relevant), k)
    ideal_scores = [1.0] * ideal_length + [0.0] * (k - ideal_length)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_recommendations(engine, test_users, ratings_df, tmdb_df, k=10):
    """
    Evaluate the recommendation system on test data.

    Uses a leave-one-out style evaluation:
    For each test user, hide their highest-rated movies and see if
    the system can recommend them.

    Args:
        engine: RecommendationEngine instance
        test_users: list of user data (each with selected movies and ground truth)
        ratings_df: original ratings DataFrame
        tmdb_df: TMDB DataFrame
        k: evaluation cutoff

    Returns: dict of average metrics
    """
    precisions = []
    recalls = []
    ndcgs = []

    for user_data in test_users:
        selected = user_data["selected"]
        relevant = set(user_data["relevant_indices"])

        if not selected or not relevant:
            continue

        # Get recommendations
        recs, _ = engine.recommend(selected, top_k=k)
        if recs.empty:
            continue

        # Map recommendation titles back to indices
        rec_indices = []
        for _, row in recs.iterrows():
            mask = tmdb_df["title"] == row["title"]
            matched = tmdb_df[mask].index.tolist()
            if matched:
                rec_indices.append(matched[0])

        # Compute metrics
        precisions.append(precision_at_k(rec_indices, relevant, k))
        recalls.append(recall_at_k(rec_indices, relevant, k))
        ndcgs.append(ndcg_at_k(rec_indices, relevant, k))

    results = {
        "precision@k": np.mean(precisions) if precisions else 0.0,
        "recall@k": np.mean(recalls) if recalls else 0.0,
        "ndcg@k": np.mean(ndcgs) if ndcgs else 0.0,
        "num_evaluated": len(precisions),
    }

    return results


def print_evaluation_results(results, k=10):
    """Print evaluation results in a formatted manner."""
    print(f"\n{'='*40}")
    print(f"  Evaluation Results (K={k})")
    print(f"{'='*40}")
    print(f"  Precision@{k}: {results['precision@k']:.4f}")
    print(f"  Recall@{k}:    {results['recall@k']:.4f}")
    print(f"  NDCG@{k}:      {results['ndcg@k']:.4f}")
    print(f"  Users evaluated: {results['num_evaluated']}")
    print(f"{'='*40}")
