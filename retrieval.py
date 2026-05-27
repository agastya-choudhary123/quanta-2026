"""BM25 retrieval over QANTA answer corpus."""
import pickle
import re
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def _load_index():
    with open(DATA_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(DATA_DIR / "bm25_meta.pkl", "rb") as f:
        answers = pickle.load(f)
    return bm25, answers


def retrieve(query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Return [(answer, score), ...] for top_k BM25 results."""
    bm25, answers = _load_index()
    tokens = re.findall(r"[a-z']+", query.lower())
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(answers)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(answers[i], float(scores[i])) for i in top_idx]
