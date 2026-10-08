"""Generate the synthetic example predictions shipped in this directory.

    python examples/make_sample.py

The files imitate what two graders produce on a 0/1/2 scale: a stronger "model" and a
weaker "baseline" scoring the same 400 answers to 40 questions. They contain no real
student answers. Questions differ in difficulty, so answers to one question succeed or
fail together, which is exactly the dependence the question-cluster bootstrap is for.

The committed JSONL files are the reference; rerun this script only to change them.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QUESTIONS, ANSWERS_PER_QUESTION = 40, 10
SCORE_PRIOR = [0.36, 0.21, 0.43]  # roughly the SciEntsBank 3-way label balance
SEED = 2026


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = QUESTIONS * ANSWERS_PER_QUESTION
    question = np.repeat(np.arange(QUESTIONS), ANSWERS_PER_QUESTION)
    true = rng.choice(3, size=n, p=SCORE_PRIOR)
    signal = np.eye(3)[true]
    # A hard question weakens the evidence for the right score in all of its answers.
    ease = rng.normal(1.0, 0.45, size=QUESTIONS)[question, None]
    shared_noise = rng.normal(size=(n, 3))

    graders = {
        # (evidence strength, sharpening): the sharper model is also overconfident.
        "sample_predictions.jsonl": (2.3, 1.7, rng.normal(size=(n, 3))),
        "sample_baseline.jsonl": (1.3, 1.0, rng.normal(size=(n, 3))),
    }
    unparsed = {"sample_predictions.jsonl": set(), "sample_baseline.jsonl": {57, 191, 304}}

    for name, (strength, sharpening, own_noise) in graders.items():
        logits = strength * ease * signal + 0.7 * shared_noise + 0.7 * own_noise
        probs = softmax(sharpening * logits).round(4)
        lines = []
        for i in range(n):
            scored = i not in unparsed[name]
            record = {
                "id": f"q{question[i] + 1:02d}-a{i % ANSWERS_PER_QUESTION + 1:02d}",
                "question_id": f"q{question[i] + 1:02d}",
                "true_score": int(true[i]),
                "predicted_score": int(probs[i].argmax()) if scored else None,
            }
            if scored:
                record["score_probabilities"] = {str(k): float(p) for k, p in enumerate(probs[i])}
            lines.append(json.dumps(record))
        (HERE / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
