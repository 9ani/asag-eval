import json
import runpy
import sys

import pytest

import asag_eval
from asag_eval.cli import main

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_evaluate_writes_tables_and_figures(sample, tmp_path, capsys):
    main(["evaluate", str(sample), "--out", str(tmp_path / "out"), "--n-boot", "100"])

    out = tmp_path / "out"
    assert sorted(path.name for path in out.iterdir()) == [
        "confusion_matrix.png",
        "metrics.json",
        "reliability.png",
        "report.md",
        "risk_coverage.png",
    ]
    for figure in out.glob("*.png"):
        assert figure.read_bytes().startswith(PNG_SIGNATURE)
    assert (
        json.loads((out / "metrics.json").read_text(encoding="utf-8"))["settings"]["n_boot"] == 100
    )
    assert "![risk coverage](risk_coverage.png)" in (out / "report.md").read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "Quadratic weighted kappa (QWK)" in printed and "![" not in printed


def test_evaluate_without_plots_writes_tables_only(sample, tmp_path):
    main(["evaluate", str(sample), "--out", str(tmp_path), "--n-boot", "50", "--no-plots"])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.json", "report.md"]


def test_evaluate_without_probabilities_draws_the_confusion_matrix_only(tmp_path):
    path = tmp_path / "preds.jsonl"
    rows = [{"id": f"a{i}", "true_score": i % 3, "predicted_score": i % 2} for i in range(12)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    main(["evaluate", str(path), "--out", str(tmp_path / "out"), "--n-boot", "20"])

    assert [figure.name for figure in (tmp_path / "out").glob("*.png")] == ["confusion_matrix.png"]


def test_compare_writes_the_paired_comparison(sample, baseline, tmp_path, capsys):
    main(["compare", str(sample), str(baseline), "--out", str(tmp_path), "--n-boot", "100"])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["comparison.json", "comparison.md"]
    assert "A - B" in capsys.readouterr().out


def test_invalid_input_exits_with_one_clear_line(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a1", "true_score": 7, "predicted_score": 1}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate", str(path), "--out", str(tmp_path / "out")])

    assert exit_info.value.code == 2
    assert capsys.readouterr().err == (
        "asag-eval: error: bad.jsonl line 1: true_score must be an integer from 0 to 2, got 7\n"
    )
    assert not (tmp_path / "out").exists()


def test_missing_file_exits_with_code_two(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate", str(tmp_path / "absent.jsonl")])
    assert exit_info.value.code == 2
    assert "asag-eval: error:" in capsys.readouterr().err


def test_resample_count_must_be_positive(sample, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate", str(sample), "--n-boot", "0"])
    assert exit_info.value.code == 2
    assert "must be at least 1, got 0" in capsys.readouterr().err


def test_a_scale_needs_at_least_two_scores(sample, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate", str(sample), "--n-classes", "1"])
    assert exit_info.value.code == 2
    assert "must be at least 2, got 1" in capsys.readouterr().err


def test_module_can_be_run_with_python_m(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["asag-eval", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("asag_eval", run_name="__main__")
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"asag-eval {asag_eval.__version__}\n"
