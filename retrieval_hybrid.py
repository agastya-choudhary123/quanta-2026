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
    bm25_weight: float = 0.6,
    dense_weight: float = 0.4,
) -> list[tuple[str, float]]:
    """
    Retrieve top-k answers using hybrid BM25 + dense ranking.

    Strategy: retrieve from both sources, rerank by weighted combination.
    BM25 is lexical (better for exact matches), dense is semantic.
    Weights: 60% BM25, 40% dense (BM25 is stronger signal for quiz bowl).

    Args:
      query: search string
      top_k: number of results to return
      bm25_weight: weight of BM25 score
      dense_weight: weight of dense score

    Returns:
      list of (answer, combined_score) tuples, sorted by score desc
    """
    # Retrieve from both sources
    bm25_results = bm25_retrieve(query, top_k=10)
    dense_results = dense_retrieve(query, top_k=10)

    # Collect all unique answers with their scores
    all_scores = {}

    for ans, score in bm25_results:
        all_scores[ans] = {"bm25": score, "dense": 0.0}

    for ans, score in dense_results:
        if ans not in all_scores:
            all_scores[ans] = {"bm25": 0.0, "dense": 0.0}
        all_scores[ans]["dense"] = score

    # Normalize scores to [0, 1] using min-max within retrieved sets
    bm25_scores = [s["bm25"] for s in all_scores.values()]
    dense_scores = [s["dense"] for s in all_scores.values()]

    bm25_max = max(bm25_scores) if bm25_scores else 1.0
    dense_max = max(dense_scores) if dense_scores else 1.0

    # Combine with normalization
    combined = {}
    for ans, scores_dict in all_scores.items():
        bm25_norm = scores_dict["bm25"] / bm25_max if bm25_max > 0 else 0.0
        dense_norm = scores_dict["dense"] / dense_max if dense_max > 0 else 0.0

        combined[ans] = bm25_weight * bm25_norm + dense_weight * dense_norm

    # Sort and return top-k
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
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
