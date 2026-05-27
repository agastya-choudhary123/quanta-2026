"""Offline RL buzzer: train on Protobowl buzz logs to optimize buzz timing.

Input: 12 features derived from question state
Output: buzz decision (binary)

Features:
  - current confidence
  - confidence trend (increasing/decreasing)
  - words seen so far
  - words remaining (estimated)
  - VLM agreement (does VLM say yes?)
  - retrieval score
  - number of times answer changed
  - is top-1 answer common/rare
  + more...

Train on Protobowl buzzdev data where we know the correct answer,
then use to make buzz decisions in tossup pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class BuzzerFeatures:
    """Features for the buzzer model."""
    confidence: float  # current confidence [0, 1]
    conf_trend: float  # (curr_conf - prev_conf) over last 3 steps
    words_seen: int  # number of words revealed so far
    words_remaining: int  # estimated words left (assume avg 150 total)
    retrieval_score: float  # BM25 score
    answer_changed: int  # how many times did top answer change
    is_high_confidence: bool  # conf > threshold
    is_increasing: bool  # confidence increasing
    has_entity: bool  # top answer contains capitalized word
    previous_buzzed: bool  # did we buzz on this answer before
    time_pressure: float  # fraction of words revealed
    min_confidence_allowed: float  # regulatory minimum (0.3)

    def to_array(self) -> np.ndarray:
        """Convert to feature vector for model."""
        return np.array([
            self.confidence,
            self.conf_trend,
            self.words_seen,
            self.words_remaining,
            self.retrieval_score,
            self.answer_changed,
            float(self.is_high_confidence),
            float(self.is_increasing),
            float(self.has_entity),
            float(self.previous_buzzed),
            self.time_pressure,
            self.min_confidence_allowed,
        ], dtype=np.float32)


def build_features(
    question_text: str,
    confidence: float,
    confidence_history: list[float],
    retrieval_score: float,
    answer: str,
    answer_history: list[str],
) -> BuzzerFeatures:
    """Build feature vector from question state."""
    words_seen = len(question_text.split())
    # Estimate words remaining (assume ~150 words per tossup)
    total_words = max(words_seen + 1, 50)  # Conservative estimate
    words_remaining = max(0, total_words - words_seen)

    # Confidence trend: change over last 3 updates
    conf_trend = 0.0
    if len(confidence_history) >= 2:
        conf_trend = confidence_history[-1] - confidence_history[-2]

    # How many times did answer change
    answer_changed = 0
    for i in range(1, len(answer_history)):
        if answer_history[i] != answer_history[i-1]:
            answer_changed += 1

    # Time pressure: what fraction of question revealed
    time_pressure = min(words_seen / max(total_words, 1), 1.0)

    # Parse answer for entities
    has_entity = any(c.isupper() for c in answer)

    # Previous buzzes on this answer
    previous_buzzed = answer_history.count(answer) > 1

    return BuzzerFeatures(
        confidence=confidence,
        conf_trend=conf_trend,
        words_seen=words_seen,
        words_remaining=words_remaining,
        retrieval_score=min(retrieval_score / 20.0, 1.0),  # normalize
        answer_changed=answer_changed,
        is_high_confidence=confidence > 0.7,
        is_increasing=conf_trend > 0.05,
        has_entity=has_entity,
        previous_buzzed=previous_buzzed,
        time_pressure=time_pressure,
        min_confidence_allowed=0.3,
    )


def should_buzz_default(
    confidence: float,
    words_seen: int,
    min_words: int = 15,
    min_conf: float = 0.72,
) -> bool:
    """Default buzzer: simple threshold-based."""
    return words_seen >= min_words and confidence >= min_conf


if __name__ == "__main__":
    # Test feature building
    q = "This is a test question with some words"
    features = build_features(
        question_text=q,
        confidence=0.8,
        confidence_history=[0.5, 0.7, 0.8],
        retrieval_score=18.0,
        answer="TestAnswer",
        answer_history=["A", "B", "TestAnswer"],
    )
    print(f"Features: {features.to_array()}")
