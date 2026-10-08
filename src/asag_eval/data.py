"""The one predictions schema every model writes, and its loader.

One JSON object per line, one line per graded answer::

    {"id": "EM.45b.299.1", "true_score": 2, "predicted_score": 2,
     "score_probabilities": {"0": 0.01, "1": 0.04, "2": 0.95}, "question_id": "EM.45b"}

``predicted_score`` is null when the model output could not be parsed.
``score_probabilities`` and ``question_id`` are optional; other fields are ignored.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Predictions:
    """The scored answers of one model on one test set, in file order."""

    ids: tuple[str, ...]
    question_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray | None
    n_classes: int
    n_unscored: int
    source: str
    sha256: str


def _answer_id(row: dict, line: str) -> str:
    raw = row["id"]
    # 1 and 1.0 would become different ids in two files, so only text and integers pass.
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise ValueError(f"{line}: id must be a string or an integer, got {raw!r}")
    return str(raw)


def _score(row: dict, field: str, n_classes: int, line: str) -> int:
    value = row[field]
    is_integer = isinstance(value, int) and not isinstance(value, bool)
    # pandas writes integer columns with missing values as 2.0, so integral floats pass.
    is_integer = is_integer or (isinstance(value, float) and value.is_integer())
    if not is_integer or not 0 <= value < n_classes:
        raise ValueError(
            f"{line}: {field} must be an integer from 0 to {n_classes - 1}, got {value!r}"
        )
    return int(value)


def _probabilities(row: dict, n_classes: int, line: str) -> list[float] | None:
    raw = row.get("score_probabilities")
    if raw is None:
        return None
    keys = [str(label) for label in range(n_classes)]
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ValueError(f"{line}: score_probabilities needs keys {', '.join(keys)}")
    values = [raw[key] for key in keys]
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1
        for value in values
    ):
        raise ValueError(f"{line}: score_probabilities must be numbers between 0 and 1")
    if abs(sum(values) - 1) > SUM_TOLERANCE:
        raise ValueError(f"{line}: score_probabilities sum to {sum(values):.4g}, not 1")
    return values


def _question_id(row: dict, answer_id: str) -> str:
    """``question_id`` if given, else a SciEntsBank id without its student and answer parts.

    ``EM.45b.299.1`` belongs to question ``EM.45b``. An id of another shape has no known
    question, so the answer becomes its own cluster.
    """
    if row.get("question_id") is not None:
        return str(row["question_id"])
    return ".".join(answer_id.split(".")[:-2]) or answer_id


def _parse(text: str, line: str) -> dict:
    try:
        row = json.loads(text)
    except (ValueError, RecursionError) as exc:
        # Besides syntax errors: nesting too deep, or an integer too long to convert.
        reason = exc.msg if isinstance(exc, json.JSONDecodeError) else exc
        raise ValueError(f"{line}: not valid JSON ({reason})") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{line}: expected a JSON object")
    return row


def load_predictions(path: str | Path, n_classes: int = 3) -> Predictions:
    """Read and validate a predictions file; raise ``ValueError`` naming the bad line."""
    path = Path(path)
    content = path.read_bytes()
    try:
        text = content.decode("utf-8-sig")  # tolerate the byte order mark editors add
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8 ({exc.reason})") from exc
    ids: list[str] = []
    questions: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[list[float] | None] = []
    seen: set[str] = set()

    # Split on "\n" only: str.splitlines would also break inside a JSON string that
    # contains a Unicode line separator.
    for number, record in enumerate(text.split("\n"), start=1):
        if not record.strip():
            continue
        line = f"{path.name} line {number}"
        row = _parse(record, line)
        for field in ("id", "true_score"):
            if row.get(field) is None:
                raise ValueError(f"{line}: missing '{field}'")
        answer_id = _answer_id(row, line)
        if answer_id in seen:
            raise ValueError(f"{line}: duplicate id '{answer_id}'")
        seen.add(answer_id)
        true_score = _score(row, "true_score", n_classes, line)
        if row.get("predicted_score") is None:
            continue
        ids.append(answer_id)
        questions.append(_question_id(row, answer_id))
        y_true.append(true_score)
        y_pred.append(_score(row, "predicted_score", n_classes, line))
        probs.append(_probabilities(row, n_classes, line))

    if not seen:
        raise ValueError(f"{path.name}: no records")
    if not ids:
        raise ValueError(f"{path.name}: no scored answers")
    without_probs = sum(row is None for row in probs)
    if 0 < without_probs < len(probs):
        # A partial file would silently lose the whole calibration analysis.
        raise ValueError(
            f"{path.name}: {without_probs} of {len(probs)} scored answers have no "
            "score_probabilities; give them for every answer or for none"
        )
    return Predictions(
        ids=tuple(ids),
        question_ids=np.array(questions),
        y_true=np.array(y_true, dtype=np.int64),
        y_pred=np.array(y_pred, dtype=np.int64),
        probs=None if without_probs else np.array(probs, dtype=float),
        n_classes=n_classes,
        n_unscored=len(seen) - len(ids),
        source=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
    )
