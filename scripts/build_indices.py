"""Build BM25 retrieval index from QANTA/Protobowl training data.

Produces in data/:
  bm25.pkl + bm25_meta.pkl   BM25 index over question texts
  answer_vocab.json          all unique answers
"""
import json
import pickle
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
WORKERS = cpu_count()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _tokenize_doc(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def build_answer_vocab(questions: list[dict]) -> None:
    out = DATA_DIR / "answer_vocab.json"
    if out.exists():
        print("Answer vocab already built, skipping.")
        return
    answers = sorted({
        q.get("page") or q.get("answer", "")
        for q in questions
        if q.get("page") or q.get("answer")
    })
    with open(out, "w") as f:
        json.dump(answers, f)
    print(f"Answer vocab: {len(answers)} answers")


def build_bm25(questions: list[dict]) -> None:
    from rank_bm25 import BM25Okapi

    out = DATA_DIR / "bm25.pkl"
    out_meta = DATA_DIR / "bm25_meta.pkl"
    if out.exists():
        print("BM25 index already built, skipping.")
        return

    # One representative text per answer
    answer_to_text: dict[str, str] = {}
    for q in questions:
        answer = q.get("page") or q.get("answer", "")
        text = q.get("full_question") or q.get("text", "")
        if answer and answer not in answer_to_text:
            answer_to_text[answer] = text

    answers = list(answer_to_text.keys())
    corpus = [answer_to_text[a] for a in answers]
    print(f"  Tokenizing {len(answers)} docs across {WORKERS} workers...")

    with Pool(WORKERS) as pool:
        tokenized = pool.map(_tokenize_doc, corpus, chunksize=200)

    print("  Fitting BM25...")
    bm25 = BM25Okapi(tokenized)

    with open(out, "wb") as f:
        pickle.dump(bm25, f)
    with open(out_meta, "wb") as f:
        pickle.dump(answers, f)

    print(f"  BM25: {out.stat().st_size / 1e6:.1f} MB ({len(answers)} docs)")


if __name__ == "__main__":
    t0 = time.time()
    print(f"Loading training data...")
    questions = load_jsonl(DATA_DIR / "guesstrain.jsonl")
    print(f"Loaded {len(questions)} questions")

    build_answer_vocab(questions)
    build_bm25(questions)

    print(f"\nDone in {time.time() - t0:.1f}s")
