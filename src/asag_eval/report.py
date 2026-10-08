"""Turn predictions into the fixed set of result tables.

``evaluate`` and ``compare`` are pure: the same file, seed and number of resamples give
the same numbers. Nothing here reads the clock, so a result can be committed next to a
paper and regenerated bit for bit.
"""

import json
import math
from pathlib import Path

import numpy as np

from asag_eval import __version__
from asag_eval.bootstrap import cluster_bootstrap, percentile_interval, two_sided_p_value
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
from asag_eval.data import Predictions
from asag_eval.metrics import METRICS, agreement_metrics, confusion_matrix, per_class_metrics

CONFIDENCE_LEVEL = 0.95
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
CONFIDENCE_METRICS = ("ece", "brier", "nll", "aurc")
DECIMALS = 6

LABELS = {
    "qwk": "Quadratic weighted kappa (QWK)",
    "kappa": "Cohen's kappa",
    "macro_f1": "Macro-F1",
    "accuracy": "Accuracy",
    "mae": "Mean absolute error",
    "extreme_error_rate": "Extreme error rate",
    "ece": "Expected calibration error (ECE)",
    "brier": "Brier score",
    "nll": "Negative log-likelihood",
    "aurc": "Area under the risk-coverage curve (AURC)",
}


def _agreement(preds: Predictions, index: np.ndarray) -> list[float]:
    cm = confusion_matrix(preds.y_true[index], preds.y_pred[index], preds.n_classes)
    metrics = agreement_metrics(cm)
    return [metrics[name] for name in METRICS]


def _describe(preds: Predictions) -> dict:
    return {
        "file": preds.source,
        "sha256": preds.sha256,
        "answers": len(preds.ids) + preds.n_unscored,
        "scored": len(preds.ids),
        "unscored": preds.n_unscored,
    }


def evaluate(preds: Predictions, n_boot: int = 2000, seed: int = 0, n_bins: int = 10) -> dict:
    """All result tables for one model, with question-cluster bootstrap intervals."""
    has_probs = preds.probs is not None
    correct = preds.y_true == preds.y_pred
    confidence = confidence_of(preds.probs, preds.y_pred) if has_probs else None

    def statistic(index: np.ndarray) -> list[float]:
        values = _agreement(preds, index)
        if has_probs:
            bins = reliability_bins(confidence[index], correct[index], n_bins)
            _, risk = risk_coverage(confidence[index], ~correct[index])
            values += [
                expected_calibration_error(bins),
                brier_score(preds.probs[index], preds.y_true[index]),
                negative_log_likelihood(preds.probs[index], preds.y_true[index]),
                aurc(risk),
            ]
        return values

    names = METRICS + CONFIDENCE_METRICS if has_probs else METRICS
    point = statistic(np.arange(len(preds.ids)))
    replicates = cluster_bootstrap(statistic, preds.question_ids, n_boot, seed)
    low, high = percentile_interval(replicates, CONFIDENCE_LEVEL)
    estimates = {
        name: {"value": value, "ci_low": lo, "ci_high": hi}
        for name, value, lo, hi in zip(names, point, low, high, strict=True)
    }
    cm = confusion_matrix(preds.y_true, preds.y_pred, preds.n_classes)
    result = {
        "tool": {"name": "asag-eval", "version": __version__},
        "input": {
            **_describe(preds),
            "questions": len(np.unique(preds.question_ids)),
            "n_classes": preds.n_classes,
        },
        "settings": {
            "n_boot": n_boot,
            "seed": seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "resampling_unit": "question",
            "n_bins": n_bins,
        },
        "metrics": {name: estimates[name] for name in METRICS},
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics(cm),
        "confidence": None,
    }
    if has_probs:
        result["confidence"] = {
            "metrics": {name: estimates[name] for name in CONFIDENCE_METRICS},
            # Greedy decoding emits the most likely score; a mismatch means the
            # probabilities and the scores in the file do not belong together.
            "argmax_mismatches": int((preds.probs.argmax(axis=1) != preds.y_pred).sum()),
            "reliability_bins": reliability_bins(confidence, correct, n_bins),
            "selective": [
                selective_agreement(
                    preds.y_true, preds.y_pred, confidence, coverage, preds.n_classes
                )
                for coverage in COVERAGES
            ],
        }
    return _clean(result)


def compare(a: Predictions, b: Predictions, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired difference A - B on the answers both models scored.

    Both models are evaluated on the same resampled questions in every replicate, so the
    interval and p-value describe the difference itself. Comparing two separate intervals
    for overlap would ignore that the models were tested on the same answers.
    """
    position = {answer_id: i for i, answer_id in enumerate(b.ids)}
    pairs = [(i, position[answer_id]) for i, answer_id in enumerate(a.ids) if answer_id in position]
    if not pairs:
        raise ValueError(f"{a.source} and {b.source} share no answer ids")
    in_a, in_b = (np.array(side) for side in zip(*pairs, strict=True))
    if not np.array_equal(a.y_true[in_a], b.y_true[in_b]):
        raise ValueError(f"{a.source} and {b.source} have different gold scores for the same ids")

    def difference(index: np.ndarray) -> list[float]:
        return [
            x - y
            for x, y in zip(_agreement(a, in_a[index]), _agreement(b, in_b[index]), strict=True)
        ]

    questions = a.question_ids[in_a]
    replicates = cluster_bootstrap(difference, questions, n_boot, seed)
    low, high = percentile_interval(replicates, CONFIDENCE_LEVEL)
    rows = zip(
        METRICS,
        _agreement(a, in_a),
        _agreement(b, in_b),
        low,
        high,
        two_sided_p_value(replicates),
        strict=True,
    )
    return _clean(
        {
            "tool": {"name": "asag-eval", "version": __version__},
            "inputs": {"a": _describe(a), "b": _describe(b)},
            "paired_answers": len(pairs),
            "questions": len(np.unique(questions)),
            "settings": {
                "n_boot": n_boot,
                "seed": seed,
                "confidence_level": CONFIDENCE_LEVEL,
                "resampling_unit": "question",
            },
            "metrics": {
                name: {
                    "a": value_a,
                    "b": value_b,
                    "difference": value_a - value_b,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p_value,
                }
                for name, value_a, value_b, lo, hi, p_value in rows
            },
        }
    )


def _clean(value):
    """Plain JSON types: floats rounded so files are stable, NaN written as null."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else round(float(value), DECIMALS)
    return value


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _interval(row: dict) -> str:
    return f"[{_number(row['ci_low'])}, {_number(row['ci_high'])}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return [*lines, *("| " + " | ".join(row) + " |" for row in rows), ""]


def _estimates_table(estimates: dict, level: float) -> list[str]:
    return _table(
        ["Metric", "Value", f"{level:.0%} CI"],
        [[LABELS[name], _number(row["value"]), _interval(row)] for name, row in estimates.items()],
    )


def render_evaluation(result: dict, figures: tuple[str, ...] = ()) -> str:
    """The evaluation as a Markdown report."""
    source, settings = result["input"], result["settings"]
    labels = range(source["n_classes"])
    lines = [
        f"# Evaluation of `{source['file']}`",
        "",
        f"- Answers: {source['answers']} ({source['scored']} scored, {source['unscored']} "
        f"without a parsed score) from {source['questions']} questions",
        f"- Input SHA-256: `{source['sha256']}`",
        f"- Intervals: {settings['confidence_level']:.0%} percentile bootstrap over questions, "
        f"{settings['n_boot']} resamples, seed {settings['seed']}",
        f"- Tool: asag-eval {result['tool']['version']}",
        "",
        "## Agreement with the gold scores",
        "",
        *_estimates_table(result["metrics"], settings["confidence_level"]),
        "## Confusion matrix",
        "",
        "Rows are gold scores, columns are predicted scores.",
        "",
        *_table(
            ["Gold \\ Predicted", *(str(label) for label in labels)],
            [
                [f"**{label}**", *(str(count) for count in row)]
                for label, row in zip(labels, result["confusion_matrix"], strict=True)
            ],
        ),
        "## Per-score results",
        "",
        *_table(
            ["Score", "Precision", "Recall", "F1", "Gold answers", "Predicted"],
            [
                [
                    str(row["label"]),
                    _number(row["precision"]),
                    _number(row["recall"]),
                    _number(row["f1"]),
                    str(row["support"]),
                    str(row["predicted"]),
                ]
                for row in result["per_class"]
            ],
        ),
    ]
    confidence = result["confidence"]
    if confidence is None:
        lines += ["No score probabilities in the input: calibration was not evaluated.", ""]
    else:
        lines += [
            "## Calibration and selective prediction",
            "",
            *_estimates_table(confidence["metrics"], settings["confidence_level"]),
            f"Answers whose emitted score is not the most probable one: "
            f"{confidence['argmax_mismatches']}.",
            "",
            "Accepting only the most confident answers and sending the rest to a teacher:",
            "",
            *_table(
                ["Coverage", "Answers", "Lowest confidence", "Accuracy", "QWK", "Extreme errors"],
                [
                    [
                        f"{row['coverage']:.0%}",
                        str(row["n"]),
                        _number(row["threshold"]),
                        _number(row["accuracy"]),
                        _number(row["qwk"]),
                        _number(row["extreme_error_rate"]),
                    ]
                    for row in confidence["selective"]
                ],
            ),
        ]
    for figure in figures:
        lines += [f"![{Path(figure).stem.replace('_', ' ')}]({figure})", ""]
    return "\n".join(lines)


def render_comparison(result: dict) -> str:
    """The paired comparison as a Markdown report."""
    a, b, settings = result["inputs"]["a"], result["inputs"]["b"], result["settings"]
    return "\n".join(
        [
            f"# Paired comparison: `{a['file']}` (A) vs `{b['file']}` (B)",
            "",
            f"- Answers scored by both models: {result['paired_answers']} "
            f"from {result['questions']} questions",
            f"- A SHA-256: `{a['sha256']}`",
            f"- B SHA-256: `{b['sha256']}`",
            f"- Intervals and p-values: paired bootstrap over questions, "
            f"{settings['n_boot']} resamples, seed {settings['seed']}",
            f"- Tool: asag-eval {result['tool']['version']}",
            "",
            *_table(
                ["Metric", "A", "B", "A - B", f"{settings['confidence_level']:.0%} CI", "p"],
                [
                    [
                        LABELS[name],
                        _number(row["a"]),
                        _number(row["b"]),
                        _number(row["difference"]),
                        _interval(row),
                        _number(row["p_value"], digits=4),
                    ]
                    for name, row in result["metrics"].items()
                ],
            ),
        ]
    )


def write_json(result: dict, path: Path) -> None:
    """Write a result as strict JSON with a stable layout."""
    text = json.dumps(result, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
