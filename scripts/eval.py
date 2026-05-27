"""Evaluate tossup and retrieval accuracy on the dev set.

Metrics:
  retrieval_acc@1   BM25 top-1 answer matches gold (fast, no VLM)
  retrieval_acc@5   BM25 top-5 contains gold
  tossup_acc        VLM pipeline answer matches gold (slow, sampled)

We evaluate retrieval on the full dev set (~1055 questions), and tossup
pipeline on a sample (default 50) since each call takes ~1s.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
TOSSUP_SAMPLE = 50  # how many questions to run through full VLM pipeline


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def normalize(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")


def eval_retrieval(questions: list[dict]) -> dict:
    from retrieval import retrieve
    hit1 = hit5 = 0
    for q in questions:
        gold = normalize(q.get("page") or q.get("answer", ""))
        text = q.get("text") or q.get("full_question", "")
        hits = retrieve(text, top_k=5)
        top5 = [normalize(a) for a, _ in hits]
        if top5 and top5[0] == gold:
            hit1 += 1
        if gold in top5:
            hit5 += 1
    n = len(questions)
    return {"acc@1": hit1 / n, "acc@5": hit5 / n, "n": n}


def eval_tossup_pipeline(questions: list[dict]) -> dict:
    from tossup import answer_tossup, TossupState
    correct = 0
    buzzed_early = 0  # buzzed before last word
    total_words = []
    t0 = time.time()
    for i, q in enumerate(questions):
        gold = normalize(q.get("page") or q.get("answer", ""))
        text = q.get("text") or q.get("full_question", "")
        state = TossupState()
        result = answer_tossup(text, state=state)
        pred = normalize(result["answer"])
        if pred == gold:
            correct += 1
        word_count = len(text.split())
        total_words.append(word_count)
        if result["buzz"] and word_count < len((q.get("full_question") or text).split()):
            buzzed_early += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(questions)}] acc so far: {correct/(i+1):.3f}  "
                  f"({elapsed:.1f}s, {elapsed/(i+1):.2f}s/q)")
    n = len(questions)
    return {
        "acc": correct / n,
        "buzzed_early_pct": buzzed_early / n,
        "avg_words_seen": sum(total_words) / n,
        "n": n,
        "total_time": time.time() - t0,
    }


if __name__ == "__main__":
    import random
    random.seed(42)

    print("Loading dev set...")
    questions = load_jsonl(DATA_DIR / "guessdev.jsonl")
    print(f"Dev set: {len(questions)} questions\n")

    # --- Retrieval eval (full dev set, fast) ---
    print("=== Retrieval Accuracy (BM25) ===")
    t0 = time.time()
    ret_results = eval_retrieval(questions)
    print(f"  acc@1 : {ret_results['acc@1']:.3f}")
    print(f"  acc@5 : {ret_results['acc@5']:.3f}")
    print(f"  n     : {ret_results['n']}")
    print(f"  time  : {time.time()-t0:.1f}s\n")

    # --- Full pipeline eval (sampled) ---
    print(f"=== Tossup Pipeline Accuracy (sample n={TOSSUP_SAMPLE}) ===")
    sample = random.sample(questions, TOSSUP_SAMPLE)
    pipe_results = eval_tossup_pipeline(sample)
    print(f"\n  acc              : {pipe_results['acc']:.3f}")
    print(f"  buzzed early %   : {pipe_results['buzzed_early_pct']:.3f}")
    print(f"  avg words seen   : {pipe_results['avg_words_seen']:.1f}")
    print(f"  total time       : {pipe_results['total_time']:.1f}s")
    print(f"  time/question    : {pipe_results['total_time']/pipe_results['n']:.2f}s")
