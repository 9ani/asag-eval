import math

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from asag_eval.metrics import agreement_metrics, confusion_matrix, per_class_metrics

# Worked example: 6 answers on the 0/1/2 scale, checked by hand in the comments below.
Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])


def test_confusion_matrix_has_gold_in_rows_and_predictions_in_columns():
    cm = confusion_matrix(Y_TRUE, Y_PRED, n_classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_agreement_metrics_match_the_hand_calculation():
    metrics = agreement_metrics(confusion_matrix(Y_TRUE, Y_PRED, n_classes=3))
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    # Squared-distance disagreement: observed 1 + 4 = 5, expected by chance 7.
    assert metrics["qwk"] == pytest.approx(1 - 5 / 7)
    # Unweighted: 2 observed disagreements, 4 expected by chance.
    assert metrics["kappa"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert metrics["mae"] == pytest.approx(3 / 6)
    # One answer with gold 2 was graded 0: the only error across the whole scale.
    assert metrics["extreme_error_rate"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("seed", range(10))
def test_agreement_metrics_match_scikit_learn(n_classes, seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(n_classes, size=200)
    # Mostly right, sometimes random: realistic agreement instead of pure noise.
    y_pred = np.where(rng.random(200) < 0.6, y_true, rng.integers(n_classes, size=200))
    labels = list(range(n_classes))

    metrics = agreement_metrics(confusion_matrix(y_true, y_pred, n_classes))

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics["qwk"] == pytest.approx(
        cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
    )
    assert metrics["kappa"] == pytest.approx(cohen_kappa_score(y_true, y_pred, labels=labels))
    assert metrics["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    assert metrics["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred))


def test_perfect_agreement_scores_one():
    metrics = agreement_metrics(confusion_matrix(Y_TRUE, Y_TRUE, n_classes=3))
    assert metrics["qwk"] == metrics["kappa"] == metrics["accuracy"] == metrics["macro_f1"] == 1.0
    assert metrics["mae"] == metrics["extreme_error_rate"] == 0.0


def test_kappa_is_undefined_when_only_one_class_occurs():
    # Chance disagreement is zero, so kappa is 0/0: report NaN instead of a made-up number.
    metrics = agreement_metrics(confusion_matrix(np.zeros(4, int), np.zeros(4, int), n_classes=3))
    assert math.isnan(metrics["qwk"]) and math.isnan(metrics["kappa"])
    assert metrics["accuracy"] == 1.0


def test_class_that_never_occurs_counts_as_zero_f1():
    # No class 2 in gold or predictions: macro-F1 still averages over all three classes.
    metrics = agreement_metrics(confusion_matrix(np.array([0, 1]), np.array([0, 1]), n_classes=3))
    assert metrics["macro_f1"] == pytest.approx(2 / 3)


def test_per_class_metrics_match_scikit_learn():
    precision, recall, f1, support = precision_recall_fscore_support(
        Y_TRUE, Y_PRED, labels=[0, 1, 2], zero_division=0
    )
    rows = per_class_metrics(confusion_matrix(Y_TRUE, Y_PRED, n_classes=3))
    assert [row["label"] for row in rows] == [0, 1, 2]
    assert [row["precision"] for row in rows] == pytest.approx(precision)
    assert [row["recall"] for row in rows] == pytest.approx(recall)
    assert [row["f1"] for row in rows] == pytest.approx(f1)
    assert [row["support"] for row in rows] == support.tolist()
