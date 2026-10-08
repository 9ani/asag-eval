"""Agreement between gold and predicted scores on an ordinal scale.

Every metric here is a function of the confusion matrix. A bootstrap replicate then costs
one ``bincount``, and each metric can be checked against scikit-learn in the tests.
"""

import numpy as np

METRICS = ("qwk", "kappa", "macro_f1", "accuracy", "mae", "extreme_error_rate")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts with gold scores in rows and predicted scores in columns."""
    flat = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2)
    return flat.reshape(n_classes, n_classes)


def _f1(cm: np.ndarray) -> np.ndarray:
    # A class absent from both gold and predictions gets 0, as in scikit-learn.
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    return np.divide(2 * np.diag(cm), denominator, out=np.zeros(len(cm)), where=denominator > 0)


def agreement_metrics(cm: np.ndarray) -> dict[str, float]:
    """Headline metrics; ``qwk`` and ``kappa`` are NaN when chance disagreement is zero."""
    n = cm.sum()
    scores = np.arange(len(cm))
    distance = np.abs(scores[:, None] - scores[None, :])
    chance = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n

    def kappa(weights: np.ndarray) -> float:
        expected = (weights * chance).sum()
        return float(1 - (weights * cm).sum() / expected) if expected else float("nan")

    return {
        "qwk": kappa(distance**2),
        "kappa": kappa(distance > 0),
        "macro_f1": float(_f1(cm).mean()),
        "accuracy": float(np.trace(cm) / n),
        "mae": float((distance * cm).sum() / n),
        # Errors across the whole scale, e.g. a correct answer graded as incorrect.
        "extreme_error_rate": float(cm[distance == len(cm) - 1].sum() / n),
    }


def per_class_metrics(cm: np.ndarray) -> list[dict[str, float | int]]:
    """Precision, recall and F1 for each score, with its gold and predicted counts."""
    hits, predicted, support = np.diag(cm), cm.sum(axis=0), cm.sum(axis=1)
    zeros = np.zeros(len(cm))
    precision = np.divide(hits, predicted, out=zeros.copy(), where=predicted > 0)
    recall = np.divide(hits, support, out=zeros.copy(), where=support > 0)
    return [
        {
            "label": label,
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1),
            "support": int(support[label]),
            "predicted": int(predicted[label]),
        }
        for label, f1 in enumerate(_f1(cm))
    ]
