from sim import run, score


def test_wood_share_in_unit_range_and_scoreboard_sorted_ascending():
    results, population = run.run_sweep(["flat", "evening_peak"], seed=7, R=10, n_agents=30)
    board = score.scoreboard(results, population)
    assert set(board["tariff"]) == {"flat", "evening_peak"}
    for _, row in board.iterrows():
        assert 0.0 <= row["wood_share"] <= 1.0
    assert (board["wood_share"].values == sorted(board["wood_share"].values)).all()
