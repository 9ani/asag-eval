import json

import numpy as np
import pytest

from asag_eval.data import load_predictions


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def record(answer_id="EM.45b.299.1", true=2, predicted=2, probabilities=(0.1, 0.2, 0.7), **extra):
    row = {"id": answer_id, "true_score": true, "predicted_score": predicted, **extra}
    if probabilities is not None:
        row["score_probabilities"] = {str(k): p for k, p in enumerate(probabilities)}
    return row


def test_loads_scores_probabilities_and_question_ids(tmp_path):
    path = write_jsonl(
        tmp_path / "preds.jsonl",
        [record("EM.45b.299.1", 2, 2), record("EM.45b.387.1", 1, 0, (0.5, 0.4, 0.1))],
    )
    preds = load_predictions(path)

    assert preds.ids == ("EM.45b.299.1", "EM.45b.387.1")
    assert preds.y_true.tolist() == [2, 1]
    assert preds.y_pred.tolist() == [2, 0]
    assert preds.probs.tolist() == [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]]
    assert preds.n_classes == 3 and preds.n_unscored == 0
    assert preds.source == "preds.jsonl" and len(preds.sha256) == 64


def test_question_is_taken_from_a_scientsbank_style_id(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [record("EM.45b.299.1"), record("MX.36a.12.1")])
    assert load_predictions(path).question_ids.tolist() == ["EM.45b", "MX.36a"]


def test_explicit_question_id_wins(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [record("a1", question_id="q7")])
    assert load_predictions(path).question_ids.tolist() == ["q7"]


def test_answer_without_a_question_is_its_own_cluster(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [record("a1"), record("a2")])
    assert load_predictions(path).question_ids.tolist() == ["a1", "a2"]


def test_unscored_answers_are_counted_not_dropped_silently(tmp_path):
    # A model output that could not be parsed has no score; the failure rate is a result.
    path = write_jsonl(
        tmp_path / "p.jsonl", [record("a1"), record("a2", predicted=None, probabilities=None)]
    )
    preds = load_predictions(path)
    assert preds.ids == ("a1",)
    assert preds.n_unscored == 1


def test_probabilities_are_optional_but_all_or_nothing(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [record("a1"), record("a2", probabilities=None)])
    assert load_predictions(path).probs is None


def test_integral_floats_written_by_pandas_are_accepted(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [record("a1", true=2.0, predicted=1.0)])
    preds = load_predictions(path)
    assert (preds.y_true.tolist(), preds.y_pred.tolist()) == ([2], [1])
    assert preds.y_true.dtype == np.int64


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "no records"),
        ([record("a1"), record("a1")], "line 2: duplicate id 'a1'"),
        ([{"true_score": 1, "predicted_score": 1}], "line 1: missing 'id'"),
        ([{"id": "a1", "predicted_score": 1}], "line 1: missing 'true_score'"),
        ([record("a1", true=3)], "line 1: true_score must be an integer from 0 to 2, got 3"),
        ([record("a1", predicted=1.5)], "line 1: predicted_score must be an integer"),
        ([record("a1", predicted=True)], "line 1: predicted_score must be an integer"),
        ([record("a1", probabilities=(0.5, 0.5))], "line 1: score_probabilities needs keys"),
        ([record("a1", probabilities=(0.5, 0.4, 0.4))], "line 1: score_probabilities sum to 1.3"),
        ([record("a1", probabilities=(1.2, -0.2, 0.0))], "line 1: score_probabilities must be"),
    ],
)
def test_invalid_files_are_rejected_with_the_line_number(tmp_path, records, message):
    path = write_jsonl(tmp_path / "bad.jsonl", records)
    with pytest.raises(ValueError, match=message):
        load_predictions(path)


def test_broken_json_is_reported_with_the_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record("a1")) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl line 2: not valid JSON"):
        load_predictions(path)


def test_blank_lines_are_skipped_and_non_objects_rejected(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(record("a1")) + "\n\n", encoding="utf-8")
    assert load_predictions(path).ids == ("a1",)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: expected a JSON object"):
        load_predictions(path)


def test_file_where_no_answer_was_scored_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [record("a1", predicted=None, probabilities=None)])
    with pytest.raises(ValueError, match="no scored answers"):
        load_predictions(path)
