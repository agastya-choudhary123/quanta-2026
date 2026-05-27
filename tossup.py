"""Tossup pipeline: incremental clue text → answer + buzz decision.

Input (called word-by-word as clues are revealed):
  question_text: str         text revealed so far
  images: list | None        images if present

Output:
  answer: str
  buzz: bool                 True = commit now
  confidence: float          0.0-1.0
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "smolvlm-256m-mlx"

# Buzz threshold: commit when confidence exceeds this
BUZZ_THRESHOLD = 0.72

# How many words minimum before we consider buzzing
MIN_WORDS_TO_BUZZ = 15


@lru_cache(maxsize=1)
def _load_model():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    model, processor = load(str(MODEL_PATH))
    config = load_config(str(MODEL_PATH))
    return model, processor, config


def _build_scoring_prompt(question_text: str, candidate: str) -> str:
    """Build a yes/no prompt: is `candidate` the answer to this clue?"""
    from mlx_vlm.prompt_utils import apply_chat_template
    _, processor, config = _load_model()
    user_msg = (
        f"Quiz bowl question clue: {question_text}\n\n"
        f"Is the answer \"{candidate}\"? Reply with only Yes or No."
    )
    return apply_chat_template(processor, config, user_msg, num_images=0)


def _score_candidate(question_text: str, candidate: str) -> float:
    """Return VLM confidence that `candidate` is correct (0-1)."""
    from mlx_vlm import generate
    model, processor, _ = _load_model()
    prompt = _build_scoring_prompt(question_text, candidate)
    result = generate(model, processor, prompt, max_tokens=4, verbose=False)
    text = result.text.strip().lower()
    # Use logprob of first token as confidence signal
    lp = np.array(result.logprobs.tolist(), dtype=np.float32)
    first_gen = lp[lp > -50]
    mean_lp = float(np.mean(first_gen)) if len(first_gen) > 0 else -5.0
    base_conf = float(1.0 / (1.0 + math.exp(mean_lp + 2.0)))
    # Boost if model explicitly said yes
    if text.startswith("yes"):
        return min(base_conf + 0.3, 1.0)
    elif text.startswith("no"):
        return max(base_conf - 0.2, 0.0)
    return base_conf


def _compute_confidence(vlm_conf: float, retrieval_score: float) -> float:
    """Blend VLM yes/no confidence with BM25 retrieval score."""
    ret_conf = min(retrieval_score / 20.0, 1.0)
    return 0.6 * vlm_conf + 0.4 * ret_conf


@dataclass
class TossupState:
    """Tracks per-question state for incremental buzzing."""
    confidence_history: list[float] = field(default_factory=list)
    last_answer: str = ""
    buzzed: bool = False


def answer_tossup(
    question_text: str,
    images: Optional[list] = None,
    state: Optional[TossupState] = None,
) -> dict:
    """
    Process one tossup update. Call repeatedly as more text is revealed.
    Returns: {answer, buzz, confidence}
    """
    from retrieval import retrieve

    word_count = len(question_text.split())

    # Retrieve candidates
    hits = retrieve(question_text, top_k=5)
    retrieval_answer = hits[0][0] if hits else ""
    retrieval_score = hits[0][1] if hits else 0.0

    # Use retrieval top answer; VLM scores it for confidence
    answer = retrieval_answer
    vlm_conf = _score_candidate(question_text, retrieval_answer) if retrieval_answer else 0.0
    confidence = _compute_confidence(vlm_conf, retrieval_score)

    # Buzz logic: commit if confident enough and seen enough words
    buzz = (
        not (state and state.buzzed)
        and word_count >= MIN_WORDS_TO_BUZZ
        and confidence >= BUZZ_THRESHOLD
    )

    if state is not None:
        state.confidence_history.append(confidence)
        state.last_answer = answer
        if buzz:
            state.buzzed = True

    return {
        "answer": answer,
        "buzz": buzz,
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    # Quick test on a sample question
    sample = (
        "This author wrote a novel about a white whale and a captain named Ahab. "
        "His most famous work features the narrator Ishmael and takes place on a whaling ship."
    )
    print("Loading model...")
    _load_model()
    print("Running tossup...")
    state = TossupState()

    # Simulate word-by-word reveal
    words = sample.split()
    for i in range(10, len(words) + 1, 5):
        partial = " ".join(words[:i])
        result = answer_tossup(partial, state=state)
        print(f"[{i:3d} words] answer={result['answer']!r:40s} conf={result['confidence']:.3f} buzz={result['buzz']}")
        if result["buzz"]:
            print("  → BUZZED!")
            break
