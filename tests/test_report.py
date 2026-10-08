import json
from dataclasses import replace

import numpy as np
import pytest

from asag_eval.data import load_predictions
from asag_eval.report import compare, evaluate, render_comparison, render_evaluation, write_json


def test_evaluation_reproduces_the_committed_results(sample, expected, assert_matches):
    # The scientific regression test: same file, seed and resamples, same published numbers.
    result = evaluate(load_predictions(sample), n_boot=2000, seed=0)
    reference = json.loads((expected / "metrics.json").read_text(encoding="utf-8"))
    assert_matches(result, reference)


def test_comparison_reproduces_the_committed_results(sample, baseline, expected, assert_matches):
    result = compare(load_predictions(sample), load_predictions(baseline), n_boot=2000, seed=0)
    reference = json.loads((expected / "comparison.json").read_text(encoding="utf-8"))
    assert_matches(result, reference)


def test_every_interval_contains_its_point_estimate_on_the_sample(sample):
    result = evaluate(load_predictions(sample), n_boot=300)
    estimates = {**result["metrics"], **result["confidence"]["metrics"]}
    # ECE is biased upwards in resamples, so its interval may sit above the estimate.
    for name in estimates.keys() - {"ece"}:
        row = estimates[name]
        assert row["ci_low"] <= row["value"] <= row["ci_high"], name


def test_the_seed_changes_intervals_but_not_estimates(sample):
    preds = load_predictions(sample)
    first, second = evaluate(preds, n_boot=200, seed=1), evaluate(preds, n_boot=200, seed=2)
    assert first["metrics"]["qwk"]["value"] == second["metrics"]["qwk"]["value"]
    assert first["metrics"]["qwk"]["ci_low"] != second["metrics"]["qwk"]["ci_low"]
    assert evaluate(preds, n_boot=200, seed=1) == first


def test_result_records_where_the_numbers_came_from(sample):
    result = evaluate(load_predictions(sample), n_boot=50, seed=3)
    assert result["input"]["file"] == "sample_predictions.jsonl"
    assert len(result["input"]["sha256"]) == 64
    assert (result["input"]["answers"], result["input"]["questions"]) == (400, 40)
    assert result["settings"] == {
        "n_boot": 50,
        "seed": 3,
        "confidence_level": 0.95,
        "resampling_unit": "question",
        "n_bins": 10,
    }


def test_selective_table_trades_coverage_for_accuracy(sample):
    rows = evaluate(load_predictions(sample), n_boot=50)["confidence"]["selective"]
    assert [row["target_coverage"] for row in rows] == [1.0, 0.9, 0.8, 0.7, 0.5]
    assert rows[0]["n"] == 400
    assert rows[-1]["accuracy"] > rows[0]["accuracy"]


def test_without_probabilities_only_agreement_is_reported(sample):
    preds = replace(load_predictions(sample), probs=None)
    result = evaluate(preds, n_boot=50)
    assert result["confidence"] is None
    assert "calibration was not evaluated" in render_evaluation(result)


def test_undefined_metrics_become_null_in_strict_json(tmp_path):
    path = tmp_path / "one_class.jsonl"
    rows = [{"id": f"a{i}", "true_score": 0, "predicted_score": 0} for i in range(4)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = evaluate(load_predictions(path), n_boot=20)
    write_json(result, tmp_path / "metrics.json")

    assert result["metrics"]["qwk"] == {"value": None, "ci_low": None, "ci_high": None}
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == result
    assert "| Quadratic weighted kappa (QWK) | n/a | [n/a, n/a] |" in render_evaluation(result)


def test_report_warns_when_answers_cannot_be_grouped_by_question(tmp_path):
    path = tmp_path / "flat.jsonl"
    rows = [{"id": f"a{i}", "true_score": i % 3, "predicted_score": i % 2} for i in range(12)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = evaluate(load_predictions(path), n_boot=20)

    assert result["settings"]["resampling_unit"] == "answer"
    assert "No question ids were found" in render_evaluation(result)


def test_report_shows_the_headline_numbers(sample):
    result = evaluate(load_predictions(sample), n_boot=50)
    report = render_evaluation(result, figures=("confusion_matrix.png",))
    qwk = result["metrics"]["qwk"]
    assert f"| Quadratic weighted kappa (QWK) | {qwk['value']:.3f} | [" in report
    assert "400 (400 scored, 0 without a parsed score) from 40 questions" in report
    assert "![confusion matrix](confusion_matrix.png)" in report


def test_a_model_does_not_differ_from_itself(sample):
    preds = load_predictions(sample)
    result = compare(preds, preds, n_boot=100)
    for row in result["metrics"].values():
        assert (row["difference"], row["ci_low"], row["ci_high"], row["p_value"]) == (0, 0, 0, 1)


def test_comparison_uses_only_answers_scored_by_both_models(sample, baseline):
    result = compare(load_predictions(sample), load_predictions(baseline), n_boot=200)
    # The baseline failed to produce a score for three answers.
    assert result["paired_answers"] == 397
    assert result["unpaired"] == {"a": 3, "b": 0}
    assert "left out: 3 scored only by A, 0 only by B" in render_comparison(result)
    qwk = result["metrics"]["qwk"]
    assert qwk["difference"] == pytest.approx(qwk["a"] - qwk["b"], abs=2e-6)
    assert qwk["ci_low"] > 0 and qwk["p_value"] < 0.05
    assert "| Quadratic weighted kappa (QWK) |" in render_comparison(result)


def test_comparison_rejects_files_from_different_test_sets(sample):
    preds = load_predictions(sample)
    other_ids = replace(preds, ids=tuple(f"x{i}" for i in range(len(preds.ids))))
    with pytest.raises(ValueError, match="share no answer ids"):
        compare(preds, other_ids)
    other_gold = replace(preds, y_true=np.roll(preds.y_true, 1))
    with pytest.raises(ValueError, match="different gold scores"):
        compare(preds, other_gold)
