import numpy as np
import pytest

from asag_eval.bootstrap import cluster_bootstrap, percentile_interval, two_sided_p_value


def mean_of(values):
    return lambda index: [values[index].mean()]


def test_replicates_are_reproducible_for_a_seed():
    values = np.random.default_rng(0).normal(size=60)
    clusters = np.repeat(np.arange(12), 5)

    first = cluster_bootstrap(mean_of(values), clusters, n_boot=50, seed=7)
    again = cluster_bootstrap(mean_of(values), clusters, n_boot=50, seed=7)
    other = cluster_bootstrap(mean_of(values), clusters, n_boot=50, seed=8)

    assert first.shape == (50, 1)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_whole_clusters_are_resampled():
    # Clusters of size 1, 2 and 3: a replicate can only contain sums of those sizes, and
    # an answer always travels together with the rest of its question.
    clusters = np.array(["a", "b", "b", "c", "c", "c"])
    seen = []

    def record(index):
        seen.append(sorted(index.tolist()))
        return [len(index)]

    sizes = cluster_bootstrap(record, clusters, n_boot=200, seed=0)

    assert set(sizes.ravel()) <= {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
    assert len(set(sizes.ravel())) > 1
    for index in seen:
        assert index.count(1) == index.count(2)
        assert index.count(3) == index.count(4) == index.count(5)


def test_statistic_may_return_several_values():
    values = np.arange(20.0)
    replicates = cluster_bootstrap(
        lambda index: [values[index].min(), values[index].max()], np.arange(20), n_boot=30
    )
    assert replicates.shape == (30, 2)
    assert (replicates[:, 0] <= replicates[:, 1]).all()


def test_cluster_bootstrap_is_wider_than_item_bootstrap_for_correlated_answers():
    # 20 questions, 10 answers each, and every answer of a question has the same value:
    # there are 20 independent observations, not 200.
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(20), 10)
    values = rng.normal(size=20)[clusters]

    by_question = cluster_bootstrap(mean_of(values), clusters, n_boot=500, seed=0).std()
    by_answer = cluster_bootstrap(mean_of(values), np.arange(200), n_boot=500, seed=0).std()

    assert by_question > 2 * by_answer
    # sqrt(10) = 3.16 in theory; allow for bootstrap noise.
    assert by_question / by_answer == pytest.approx(10**0.5, rel=0.25)


def test_replicates_centre_on_the_sample_statistic():
    values = np.random.default_rng(2).normal(loc=3.0, size=300)
    replicates = cluster_bootstrap(mean_of(values), np.repeat(np.arange(30), 10), n_boot=1000)
    assert replicates.mean() == pytest.approx(values.mean(), abs=0.02)


def test_n_boot_must_be_positive():
    with pytest.raises(ValueError, match="n_boot must be at least 1"):
        cluster_bootstrap(mean_of(np.ones(3)), np.arange(3), n_boot=0)


def test_percentile_interval_ignores_undefined_replicates():
    replicates = np.column_stack([np.arange(101.0), np.arange(101.0)])
    replicates[0, 1] = np.nan  # e.g. kappa of a resample with a single class

    low, high = percentile_interval(replicates, level=0.9)

    assert (low[0], high[0]) == pytest.approx((5.0, 95.0))
    assert (low[1], high[1]) == pytest.approx((5.95, 95.05))


def test_p_value_is_small_when_every_replicate_has_the_same_sign():
    assert two_sided_p_value(np.full((99, 1), 0.2))[0] == pytest.approx(0.02)
    assert two_sided_p_value(np.full((99, 1), -0.2))[0] == pytest.approx(0.02)


def test_p_value_is_one_when_there_is_no_difference():
    assert two_sided_p_value(np.zeros((99, 1)))[0] == 1.0
    symmetric = np.linspace(-1, 1, 99).reshape(-1, 1)
    assert two_sided_p_value(symmetric)[0] == 1.0


def test_p_value_counts_the_replicates_on_the_far_side_of_zero():
    replicates = np.array([-0.1, -0.2] + [0.3] * 97).reshape(-1, 1)
    # (2 + 1) of (99 + 1) on the smaller side, doubled for a two-sided test.
    assert two_sided_p_value(replicates)[0] == pytest.approx(0.06)
