"""Bootstrap over questions, not over answers.

Answers to one question share its wording, rubric and difficulty, so they are not
independent observations. Resampling whole questions keeps that dependence in every
replicate; resampling single answers would report intervals that are too narrow.
"""

import warnings
from collections.abc import Callable, Sequence

import numpy as np

Statistic = Callable[[np.ndarray], Sequence[float]]


def cluster_bootstrap(
    statistic: Statistic, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> np.ndarray:
    """Evaluate ``statistic`` on ``n_boot`` resamples of whole clusters.

    ``clusters`` holds one label per item. ``statistic`` receives the item indices of a
    resample and returns one or more numbers. The result has shape ``(n_boot, n_values)``.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    _, codes = np.unique(clusters, return_inverse=True)
    members = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    replicates = [
        statistic(
            np.concatenate([members[c] for c in rng.integers(len(members), size=len(members))])
        )
        for _ in range(n_boot)
    ]
    return np.asarray(replicates, dtype=float).reshape(n_boot, -1)


def percentile_interval(
    replicates: np.ndarray, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper percentile bounds per column; undefined (NaN) replicates are skipped."""
    tail = (1 - level) / 2 * 100
    with warnings.catch_warnings():
        # A statistic undefined in every replicate has no interval: NaN, not a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(replicates, [tail, 100 - tail], axis=0)
    return low, high


def two_sided_p_value(replicates: np.ndarray) -> np.ndarray:
    """Bootstrap p-value per column for "the true value is zero".

    Twice the share of replicates on the less likely side of zero, with the usual +1
    correction so that a finite number of replicates never yields p = 0. A statistic that
    is undefined in every replicate has no p-value (NaN).
    """
    valid = (~np.isnan(replicates)).sum(axis=0)
    not_above = (replicates <= 0).sum(axis=0)
    not_below = (replicates >= 0).sum(axis=0)
    p_value = np.minimum(1.0, 2 * (np.minimum(not_above, not_below) + 1) / (valid + 1))
    return np.where(valid == 0, np.nan, p_value)
