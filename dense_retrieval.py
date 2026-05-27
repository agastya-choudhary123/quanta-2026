"""Dense embedding retrieval using all-MiniLM-L6-v2 (22.7M params, ~45MB fp16).

Encodes all answer candidates once, then uses cosine similarity for fast retrieval.
Hybrid reranking blends BM25 scores + semantic similarity for better recall.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize as l2_normalize

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "answer_embeddings.npy"
EMBEDDINGS_META_PATH = DATA_DIR / "answer_embeddings_meta.pkl"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load precomputed answer embeddings and answer list."""
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    with open(EMBEDDINGS_META_PATH, "rb") as f:
        answers = pickle.load(f)
    return embeddings, answers


def encode_query(query: str) -> np.ndarray:
    """Encode query to 384-dim embedding."""
    model = _load_model()
    emb = model.encode(query, convert_to_numpy=True)
    # Normalize to unit length for cosine similarity
    return emb / (np.linalg.norm(emb) + 1e-8)


def dense_retrieve(query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Retrieve top-k answers by cosine similarity to query embedding."""
    query_emb = encode_query(query)
    embeddings, answers = _load_embeddings()

    # Cosine similarity: (query @ embeddings.T)
    # Both are L2-normalized, so dot product = cosine similarity
    scores = np.dot(query_emb, embeddings.T)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [(answers[i], float(scores[i])) for i in top_indices]


def build_embeddings():
    """Precompute and save embeddings for all answers in the vocabulary."""
    from retrieval import _load_index

    print("Loading BM25 index for answer vocabulary...")
    bm25, answers = _load_index()

    print(f"Encoding {len(answers)} answer candidates...")
    model = _load_model()
    embeddings = model.encode(answers, convert_to_numpy=True, show_progress_bar=True)
    embeddings = l2_normalize(embeddings, norm="l2")

    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=False)

    print(f"Saving metadata to {EMBEDDINGS_META_PATH}...")
    with open(EMBEDDINGS_META_PATH, "wb") as f:
        pickle.dump(answers, f)

    print(f"Done. Embeddings: {embeddings.shape}, dtype: {embeddings.dtype}")


if __name__ == "__main__":
    build_embeddings()
