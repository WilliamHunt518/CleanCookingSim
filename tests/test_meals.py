import numpy as np

from sim import meals, config


def test_boxcar_profile_conserves_energy():
    for idx0 in range(meals.K):
        duration = 10
        profile = meals.power_profile(idx0, duration)
        energy = profile.sum() * meals.BLOCK_HOURS
        assert np.isclose(energy, meals.E_KWH[idx0])


def test_preheat_simmer_profile_conserves_energy(monkeypatch):
    monkeypatch.setattr(config, "MEAL_PROFILE_SHAPE", "preheat_simmer")
    idx0 = meals.IDX_BY_NAME["big_e_cook"]
    duration = 12
    profile = meals.preheat_simmer_shape(duration, meals.E_KWH[idx0])
    energy = profile.sum() * meals.BLOCK_HOURS
    assert np.isclose(energy, meals.E_KWH[idx0])


def test_sampled_durations_are_clipped_and_positive():
    rng = np.random.default_rng(0)
    idx = np.zeros(1000, dtype=int)  # all big_e_cook
    durations = meals.sample_durations_blocks(idx, rng)
    assert np.all(durations >= 1)
    assert np.all(durations <= meals.DMAX_BLOCKS)
