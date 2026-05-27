"""Bonus pipeline: leadin + part → answer (BM25 only, no VLM).

Bonus questions have 3 parts (A/B/C). Each part is answered independently
but the leadin gives thematic context.

Input:
  leadin: str          theme paragraph
  part: str            specific part question

Output:
  answer: str
  confidence: float
  explanation: str     one-sentence rationale
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent


def answer_bonus(
    leadin: str,
    part: str,
    images: Optional[list] = None,
) -> dict:
    """
    Answer one bonus part using pure BM25 retrieval.
    Returns {answer, confidence, explanation}.
    """
    from retrieval import retrieve

    # Combine leadin + part for context
    combined_query = f"{leadin} {part}"

    # Retrieve top answer
    hits = retrieve(combined_query, top_k=5)
    answer = hits[0][0] if hits else ""
    retrieval_score = hits[0][1] if hits else 0.0

    # Confidence from retrieval score
    confidence = min(retrieval_score / 20.0, 1.0)

    # Simple explanation
    explanation = f"Retrieved from quiz bowl database (score: {retrieval_score:.1f})"

    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "explanation": explanation,
    }


if __name__ == "__main__":
    leadin = "This author wrote novels set in 19th century America"
    parts = [
        "Name this author of Adventures of Huckleberry Finn.",
        "Name his novel about a boy who fakes his own death.",
        "Name the pen name derived from a riverboat term.",
    ]

    print("Running bonus...\n")
    for i, part in enumerate(parts):
        result = answer_bonus(leadin, part)
        print(f"Part {chr(65+i)}: {result['answer']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Explanation: {result['explanation']}\n")
