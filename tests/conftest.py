from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def sample():
    return EXAMPLES / "sample_predictions.jsonl"


@pytest.fixture
def baseline():
    return EXAMPLES / "sample_baseline.jsonl"


@pytest.fixture
def expected():
    return EXAMPLES / "expected"


def _assert_matches(actual, reference, where="result"):
    """Equal structure and values; floats agree to the precision the files are written with."""
    if isinstance(reference, dict):
        assert actual.keys() == reference.keys(), where
        for key, value in reference.items():
            _assert_matches(actual[key], value, f"{where}.{key}")
    elif isinstance(reference, list):
        assert len(actual) == len(reference), where
        for position, value in enumerate(reference):
            _assert_matches(actual[position], value, f"{where}[{position}]")
    elif isinstance(reference, float):
        assert actual == pytest.approx(reference, abs=2e-6), where
    else:
        assert actual == reference, where


@pytest.fixture
def assert_matches():
    return _assert_matches
