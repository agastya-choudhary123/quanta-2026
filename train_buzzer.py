"""Train a simple logistic regression buzzer on dev set.

This is a proof-of-concept. In a real system, we'd train on Protobowl
buzz logs, but we can bootstrap from our dev set.

The model learns: given current confidence and question state,
should we buzz now or wait for more clues?
"""
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def collect_training_data():
    """Collect features and labels from dev set."""
    from tossup import answer_tossup, TossupState
    from retrieval import retrieve
    from buzzer import build_features

    questions = [json.loads(l) for l in open(DATA_DIR / "guessdev.jsonl")]

    def normalize(s: str) -> str:
        return s.strip().lower().replace("_", " ").replace("-", " ")

    X = []  # features
    y = []  # labels (did we get correct answer?)

    for q in questions:
        gold = normalize(q.get("page") or q.get("answer", ""))
        text = q.get("text") or q.get("full_question", "")
        words = text.split()

        state = TossupState()
        answer_hist = []
        conf_hist = []

        # Simulate incremental word reveal
        for i in range(min(len(words), 50), len(words) + 1, 5):
            partial = " ".join(words[:i])
            result = answer_tossup(partial, state=state)
            answer_hist.append(result["answer"])
            conf = result["confidence"]
            conf_hist.append(conf)

            # Get retrieval score for feature engineering
            hits = retrieve(partial, top_k=1)
            ret_score = hits[0][1] if hits else 0.0

            # Build features
            features = build_features(
                question_text=partial,
                confidence=conf,
                confidence_history=conf_hist,
                retrieval_score=ret_score,
                answer=result["answer"],
                answer_history=answer_hist,
            )

            X.append(features.to_array())

            # Label: did we get correct answer?
            pred_norm = normalize(result["answer"])
            correct = 1 if pred_norm == gold else 0
            y.append(correct)

    return np.array(X), np.array(y)


def train_buzzer():
    """Train logistic regression buzzer."""
    print("Collecting training data...")
    X, y = collect_training_data()
    print(f"Collected {len(X)} examples")
    print(f"Positive examples: {y.sum()} ({100*y.mean():.1f}%)\n")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    print("Training buzzer...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # penalize false positives more
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Save model and scaler
    buzzer_path = ROOT / "buzzer_model.pkl"
    scaler_path = ROOT / "buzzer_scaler.pkl"

    with open(buzzer_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved buzzer to {buzzer_path}")
    print(f"Saved scaler to {scaler_path}")

    # Print feature importance
    print(f"\nFeature weights:")
    feature_names = [
        "confidence", "conf_trend", "words_seen", "words_remaining",
        "retrieval_score", "answer_changed", "is_high_confidence",
        "is_increasing", "has_entity", "previous_buzzed",
        "time_pressure", "min_confidence_allowed"
    ]
    for name, weight in sorted(zip(feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:25s} {weight:+.4f}")


if __name__ == "__main__":
    train_buzzer()
