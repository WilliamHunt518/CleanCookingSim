import numpy as np

from sim import run, score


def test_clean_cooking_share_in_unit_range_and_scoreboard_sorted_descending():
    results, population = run.run_sweep(["flat", "evening_peak"], seed=7, R=10, n_agents=30)
    board = score.scoreboard(results, population)
    assert set(board["tariff"]) == {"flat", "evening_peak"}
    for _, row in board.iterrows():
        assert 0.0 <= row["clean_cooking_share"] <= 1.0
        assert row["clean_cooking_share"] == score.clean_cooking_share(results[row["tariff"]])
        assert row["peak_kw"] >= 0.0
        assert 0.0 <= row["load_factor"] <= 1.0
    sorted_desc = sorted(board["clean_cooking_share"].values, reverse=True)
    assert (board["clean_cooking_share"].values == sorted_desc).all()


def test_clean_cooking_share_is_complement_of_wood_share():
    results, population = run.run_sweep(["flat"], seed=3, R=10, n_agents=30)
    result = results["flat"]
    assert score.clean_cooking_share(result) == 1.0 - score.wood_share(result)


def test_zero_events_scores_zero_clean_share_not_full_marks():
    """A tariff extreme enough to suppress all cooking (see extreme_test) shouldn't be scored as
    100% clean -- it achieved no cooking, not clean cooking."""
    results, population = run.run_sweep(["extreme_test"], seed=0, R=10, n_agents=30)
    result = results["extreme_test"]
    if not result.events_all_runs:
        assert score.clean_cooking_share(result) == 0.0
        assert score.load_factor(result) == 0.0


def test_load_factor_matches_manual_mean_over_peak_definition():
    """Sanity check load_factor is really the standard grid-engineering mean/peak ratio, computed
    per simulated day and averaged across runs -- not, say, computed from an averaged curve."""
    results, _ = run.run_sweep(["flat"], seed=11, R=15, n_agents=40)
    result = results["flat"]
    expected = np.mean([c.mean() / c.max() for c in result.demand_curves if c.max() > 0])
    assert score.load_factor(result) == expected
