"""Calibration and selective prediction: can the model's confidence decide who grades?

A grader that routes uncertain answers to a teacher needs two things from its
probabilities. They must be calibrated (80% confidence means 80% correct), and they must
rank answers so that the errors sit at the low-confidence end.

Answers with equal confidence cannot be told apart by a threshold. Every function here
treats such a tie as one group, so no result depends on the order of rows in the file.
"""

import math

import numpy as np

from asag_eval.metrics import agreement_metrics, confusion_matrix

# Keeps the log finite when the model gave the gold score a probability of exactly zero.
MIN_PROBABILITY = 1e-12


def confidence_of(probs: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """The probability the model assigned to the score it emitted."""
    return probs[np.arange(len(y_pred)), y_pred]


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Non-empty equal-width confidence bins with their mean confidence and accuracy."""
    index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return [
        {
            "lower": b / n_bins,
            "upper": (b + 1) / n_bins,
            "count": int((index == b).sum()),
            "confidence": float(confidence[index == b].mean()),
            "accuracy": float(correct[index == b].mean()),
        }
        for b in np.unique(index).tolist()
    ]


def expected_calibration_error(bins: list[dict[str, float | int]]) -> float:
    """Mean gap between confidence and accuracy, weighted by the answers in each bin."""
    total = sum(b["count"] for b in bins)
    return sum(b["count"] / total * abs(b["accuracy"] - b["confidence"]) for b in bins)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class Brier score: 0 is perfect, 2 is full confidence in a wrong score."""
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean negative log-probability of the gold score."""
    gold = confidence_of(probs, y_true)
    return float(-np.log(np.maximum(gold, MIN_PROBABILITY)).mean())


def risk_coverage(confidence: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Error rate among the accepted answers as the acceptance threshold is lowered.

    Point ``i`` accepts the ``i + 1`` most confident answers: ``coverage`` is their share of
    all answers and ``risk`` their error rate. Inside a group of equal confidence each
    answer carries the group's mean error, which is the expected curve over all orders of
    the tied answers.
    """
    order = np.argsort(-confidence, kind="stable")
    _, group, size = np.unique(-confidence[order], return_inverse=True, return_counts=True)
    tied_error = (np.bincount(group, weights=errors[order].astype(float)) / size)[group]
    accepted = np.arange(1, len(order) + 1)
    return accepted / len(order), np.cumsum(tied_error) / accepted


def aurc(risk: np.ndarray) -> float:
    """Area under the risk-coverage curve: the mean risk over all coverage levels."""
    return float(risk.mean())


def selective_agreement(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, coverage: float, n_classes: int
) -> dict[str, float | int]:
    """Agreement metrics on the answers that a confidence threshold accepts.

    ``threshold`` is the confidence of the ``ceil(coverage * n)``-th most confident answer.
    Every answer at or above it is accepted and the rest would go to a human reviewer.
    Answers tied at the threshold are accepted together, so the realised ``coverage`` can
    be larger than ``target_coverage``.
    """
    rank = max(1, math.ceil(coverage * len(confidence)))
    threshold = np.sort(confidence)[-rank]
    accepted = confidence >= threshold
    return {
        "target_coverage": coverage,
        "coverage": float(accepted.mean()),
        "n": int(accepted.sum()),
        "threshold": float(threshold),
        **agreement_metrics(confusion_matrix(y_true[accepted], y_pred[accepted], n_classes)),
    }
