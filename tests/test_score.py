import numpy as np

from sim import config, run, score


def test_wood_share_and_p_exceed_in_unit_range_and_score_formula():
    results, population = run.run_sweep(["flat", "evening_peak"], seed=7, R=10, n_agents=30)
    board = score.scoreboard(results, population)
    assert set(board["tariff"]) == {"flat", "evening_peak"}
    for _, row in board.iterrows():
        assert 0.0 <= row["wood_share"] <= 1.0
        assert 0.0 <= row["P_exceed"] <= 1.0
        assert np.isclose(row["score"], row["wood_share"] + config.SCORING.PI * row["P_exceed"])
    assert (board["score"].values == sorted(board["score"].values)).all()
