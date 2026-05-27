"""Bonus pipeline: leadin + part → answer + confidence + explanation.

Bonus questions have 3 parts (A/B/C). Each part is answered independently
but the leadin gives thematic context shared across all parts.

Input:
  leadin: str          theme paragraph
  part: str            specific part question
  images: list | None

Output:
  answer: str
  confidence: float
  explanation: str     one-sentence rationale
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "smolvlm-256m-mlx"


@lru_cache(maxsize=1)
def _load_model():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    model, processor = load(str(MODEL_PATH))
    config = load_config(str(MODEL_PATH))
    return model, processor, config


def _generate(prompt: str, max_tokens: int = 60) -> tuple[str, float]:
    """Generate text, return (text, mean_logprob)."""
    from mlx_vlm import generate
    model, processor, _ = _load_model()
    result = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    lp = np.array(result.logprobs.tolist(), dtype=np.float32)
    gen_lp = lp[lp > -50]
    mean_lp = float(np.mean(gen_lp)) if len(gen_lp) > 0 else -5.0
    # Confidence: higher mean logprob = more confident
    import math
    conf = float(1.0 / (1.0 + math.exp(mean_lp + 2.5)))
    return result.text.strip(), conf


def answer_bonus(
    leadin: str,
    part: str,
    images: Optional[list] = None,
) -> dict:
    """
    Answer one bonus part. Returns {answer, confidence, explanation}.
    """
    from mlx_vlm.prompt_utils import apply_chat_template
    from retrieval import retrieve

    combined_query = f"{leadin} {part}"
    hits = retrieve(combined_query, top_k=5)
    candidates = ", ".join(ans for ans, _ in hits[:3])
    retrieval_answer = hits[0][0] if hits else ""
    retrieval_score = hits[0][1] if hits else 0.0

    _, processor, config = _load_model()
    user_msg = (
        f"You are a quiz bowl expert. Answer the following bonus question part.\n\n"
        f"Theme: {leadin}\n\n"
        f"Question: {part}\n\n"
        f"Top candidates from database: {candidates}\n\n"
        f"Give your answer as:\nAnswer: <entity name>\nReason: <one sentence why>"
    )
    prompt = apply_chat_template(processor, config, user_msg, num_images=0)
    text, vlm_conf = _generate(prompt, max_tokens=60)

    # Parse answer and explanation from generated text
    answer = retrieval_answer
    explanation = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("answer:"):
            raw = line[7:].strip()
            # Only use VLM answer if it looks like an entity (short, no sentences)
            if raw and len(raw.split()) <= 5 and not raw.endswith("?"):
                answer = raw
        elif line.lower().startswith("reason:"):
            explanation = line[7:].strip()

    if not explanation:
        explanation = f"Retrieved from quiz bowl database with score {retrieval_score:.1f}."

    ret_conf = min(retrieval_score / 20.0, 1.0)
    confidence = round(0.6 * vlm_conf + 0.4 * ret_conf, 4)

    return {
        "answer": answer,
        "confidence": confidence,
        "explanation": explanation,
    }


if __name__ == "__main__":
    leadin = "This author wrote novels set in 19th century America, including one about life on the Mississippi River."
    parts = [
        "Name this author of Adventures of Huckleberry Finn.",
        "Name his novel about a boy who fakes his own death to escape his abusive father.",
        "Name the pen name this author used, derived from a riverboat term meaning two fathoms deep.",
    ]

    print("Loading model...")
    _load_model()
    print("Running bonus...\n")

    for i, part in enumerate(parts):
        result = answer_bonus(leadin, part)
        print(f"Part {chr(65+i)}: {result['answer']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Explanation: {result['explanation']}\n")
