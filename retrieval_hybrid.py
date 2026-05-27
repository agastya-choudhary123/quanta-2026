"""Hybrid retrieval: BM25 + dense embeddings with learned reranking.

Strategy:
1. BM25 (lexical): fast, good for exact/near-exact matches
2. Dense (semantic): captures paraphrases and conceptual matches
3. Hybrid rerank: blend BM25 normalized scores + dense normalized scores

Normalization: z-score within each retriever's top-k results, then weighted average.
"""
from __future__ import annotations

import numpy as np

from dense_retrieval import dense_retrieve
from retrieval import retrieve as bm25_retrieve


def _normalize_scores(scores: list[float]) -> list[float]:
    """Z-score normalize a list of scores (handle single value case)."""
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 1 or scores.std() == 0:
        return scores.tolist()
    return ((scores - scores.mean()) / scores.std()).tolist()


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Retrieve top-k answers using hybrid BM25 + dense ranking.

    Args:
      query: search string
      top_k: number of results to return
      bm25_weight: weight of BM25 score (0-1)
      dense_weight: weight of dense score (0-1)

    Returns:
      list of (answer, combined_score) tuples, sorted by score desc
    """
    # Retrieve from both sources (get top 2*k to ensure good coverage)
    bm25_results = bm25_retrieve(query, top_k=2 * top_k)
    dense_results = dense_retrieve(query, top_k=2 * top_k)

    # Build answer → score dicts
    bm25_dict = {ans: score for ans, score in bm25_results}
    dense_dict = {ans: score for ans, score in dense_results}

    # Combine results: use both as sources of truth
    all_answers = set(bm25_dict.keys()) | set(dense_dict.keys())

    # Score each answer as weighted blend of BM25 + dense scores
    scores = {}
    for ans in all_answers:
        bm25_score = bm25_dict.get(ans, 0.0)
        dense_score = dense_dict.get(ans, 0.0)

        # Normalize each score to [0, 1] range
        # BM25 is already ~0-50, dense is ~0-1 (cosine sim)
        bm25_norm = min(bm25_score / 30.0, 1.0)
        dense_norm = min(dense_score, 1.0)

        combined = bm25_weight * bm25_norm + dense_weight * dense_norm
        scores[ans] = combined

    # Sort and return top-k
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


if __name__ == "__main__":
    # Quick test
    queries = [
        "author of Moby Dick",
        "this president served two non-consecutive terms",
        "capital of France",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = hybrid_retrieve(q, top_k=3)
        for ans, score in results:
            print(f"  {ans:40s} {score:.3f}")
