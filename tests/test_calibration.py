import numpy as np
import pytest
from sklearn.metrics import log_loss

from asag_eval.calibration import (
    aurc,
    brier_score,
    confidence_of,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    risk_coverage,
    selective_agreement,
)


def test_confidence_is_the_probability_of_the_emitted_score():
    probs = np.array([[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]])
    assert confidence_of(probs, np.array([1, 0])).tolist() == [0.2, 0.5]


def test_ece_is_zero_when_confidence_equals_accuracy():
    # Four answers at 75% confidence, three of them right.
    confidence = np.full(4, 0.75)
    correct = np.array([True, True, True, False])
    assert expected_calibration_error(reliability_bins(confidence, correct)) == pytest.approx(0)


def test_ece_weights_each_bin_by_its_share_of_answers():
    confidence = np.array([0.95, 0.95, 0.65, 0.65])
    correct = np.array([True, True, False, False])

    bins = reliability_bins(confidence, correct, n_bins=10)

    assert [(b["lower"], b["count"]) for b in bins] == [(0.6, 2), (0.9, 2)]
    assert [b["accuracy"] for b in bins] == [0.0, 1.0]
    assert [b["confidence"] for b in bins] == pytest.approx([0.65, 0.95])
    # 0.5 * |0 - 0.65| + 0.5 * |1 - 0.95|
    assert expected_calibration_error(bins) == pytest.approx(0.35)


def test_full_confidence_falls_into_the_last_bin():
    bins = reliability_bins(np.array([1.0, 1.0]), np.array([True, False]), n_bins=10)
    assert [(b["lower"], b["upper"], b["count"]) for b in bins] == [(0.9, 1.0, 2)]
    assert expected_calibration_error(bins) == pytest.approx(0.5)


def test_brier_score_ranges_from_zero_to_two():
    y_true = np.array([0])
    assert brier_score(np.array([[1.0, 0.0, 0.0]]), y_true) == 0.0
    assert brier_score(np.array([[0.0, 1.0, 0.0]]), y_true) == 2.0
    assert brier_score(np.array([[0.5, 0.5, 0.0]]), y_true) == pytest.approx(0.5)


def test_negative_log_likelihood_matches_scikit_learn():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=50)
    y_true = rng.integers(3, size=50)
    assert negative_log_likelihood(probs, y_true) == pytest.approx(
        log_loss(y_true, probs, labels=[0, 1, 2])
    )


def test_negative_log_likelihood_stays_finite_for_a_zero_probability():
    assert np.isfinite(negative_log_likelihood(np.array([[0.0, 1.0, 0.0]]), np.array([0])))


def test_risk_coverage_accepts_the_most_confident_answers_first():
    confidence = np.array([0.6, 0.9, 0.7, 0.8])
    errors = np.array([True, False, True, False])

    coverage, risk = risk_coverage(confidence, errors)

    assert coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk == pytest.approx([0, 0, 1 / 3, 0.5])
    assert risk[-1] == errors.mean()
    assert aurc(risk) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_aurc_is_worse_when_confidence_ranks_errors_first():
    errors = np.array([True, True, False, False])
    _, good = risk_coverage(np.array([0.1, 0.2, 0.8, 0.9]), errors)
    _, bad = risk_coverage(np.array([0.9, 0.8, 0.2, 0.1]), errors)
    assert aurc(good) < errors.mean() < aurc(bad)


def test_risk_coverage_does_not_depend_on_the_order_of_tied_answers():
    # A threshold cannot separate answers with equal confidence, so where the one error
    # sits in the file must not matter.
    confidence = np.full(4, 0.9)
    _, error_first = risk_coverage(confidence, np.array([True, False, False, False]))
    _, error_last = risk_coverage(confidence, np.array([False, False, False, True]))

    assert error_first == pytest.approx([0.25] * 4)
    assert error_last == pytest.approx(error_first)


def test_risk_coverage_averages_errors_inside_a_tie_only():
    confidence = np.array([0.9, 0.7, 0.7, 0.5])
    errors = np.array([False, True, False, True])
    _, risk = risk_coverage(confidence, errors)
    # The two answers at 0.7 share one error: half an error each.
    assert risk == pytest.approx([0, 0.25, 1 / 3, 0.5])


def test_selective_agreement_keeps_the_most_confident_share():
    y_true = np.array([2, 1, 0, 2])
    y_pred = np.array([2, 1, 2, 0])
    confidence = np.array([0.9, 0.8, 0.7, 0.6])

    half = selective_agreement(y_true, y_pred, confidence, coverage=0.5, n_classes=3)
    everything = selective_agreement(y_true, y_pred, confidence, coverage=1.0, n_classes=3)

    assert (half["n"], half["threshold"], half["accuracy"]) == (2, 0.8, 1.0)
    assert (half["target_coverage"], half["coverage"]) == (0.5, 0.5)
    assert (everything["n"], everything["threshold"], everything["accuracy"]) == (4, 0.6, 0.5)
    assert everything["extreme_error_rate"] == 0.5


def test_selective_agreement_accepts_every_answer_at_the_threshold():
    # Three answers tie at the cut. A threshold accepts all of them or none, so the
    # realised coverage is 100% although 50% was asked for, whatever the row order.
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 2])
    for confidence in (np.array([0.9, 0.8, 0.8, 0.8]), np.array([0.8, 0.8, 0.8, 0.9])):
        result = selective_agreement(y_true, y_pred, confidence, 0.5, n_classes=3)
        assert (result["n"], result["coverage"], result["threshold"]) == (4, 1.0, 0.8)


def test_selective_agreement_rounds_the_kept_share_up():
    # 30% of 4 answers is 1.2: keep 2, never fewer than asked for.
    result = selective_agreement(
        np.array([0, 1, 2, 0]), np.array([0, 1, 2, 0]), np.arange(4.0) / 4, 0.3, n_classes=3
    )
    assert result["n"] == 2
