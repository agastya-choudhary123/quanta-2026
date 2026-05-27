"""Simplified tossup: BM25 only, no VLM.

Pure retrieval-based buzzer:
- Retrieve top answer with BM25
- Buzz when: retrieval_score > threshold AND words_seen > min_words
- No VLM overhead, simple and fast
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent

# Tuning parameters
BUZZ_THRESHOLD_SCORE = 10.0  # BM25 score above this
MIN_WORDS_TO_BUZZ = 20  # Must see at least this many words


@dataclass
class TossupState:
    """Tracks per-question state for incremental buzzing."""
    score_history: list[float] = field(default_factory=list)
    last_answer: str = ""
    buzzed: bool = False


def answer_tossup(
    question_text: str,
    images: Optional[list] = None,
    state: Optional[TossupState] = None,
) -> dict:
    """
    Process one tossup: retrieve answer and decide whether to buzz.

    Simple strategy: high BM25 score + enough words = buzz.
    """
    from retrieval import retrieve

    word_count = len(question_text.split())

    # Retrieve top answer
    hits = retrieve(question_text, top_k=1)
    answer = hits[0][0] if hits else ""
    score = hits[0][1] if hits else 0.0

    # Buzz decision: simple threshold
    buzz = (
        not (state and state.buzzed)
        and word_count >= MIN_WORDS_TO_BUZZ
        and score >= BUZZ_THRESHOLD_SCORE
    )

    # Confidence: normalized score
    confidence = min(score / 20.0, 1.0)

    if state is not None:
        state.score_history.append(score)
        state.last_answer = answer
        if buzz:
            state.buzzed = True

    return {
        "answer": answer,
        "buzz": buzz,
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    # Quick test
    sample = "This author wrote a novel about a white whale and a captain named Ahab."
    print("Running tossup...")
    state = TossupState()

    words = sample.split()
    for i in range(10, len(words) + 1, 5):
        partial = " ".join(words[:i])
        result = answer_tossup(partial, state=state)
        print(f"[{i:3d} words] answer={result['answer']!r:40s} score={result['confidence']:.3f} buzz={result['buzz']}")
        if result["buzz"]:
            print("  → BUZZED!")
            break
