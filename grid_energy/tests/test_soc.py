from __future__ import annotations

import numpy as np

from grid_energy import config, soc as soc_mod


def test_zero_usage_constant_pv_charges_linearly():
    T = 9  # stays under 100 kWh cap for every block, so nothing clips yet
    pv_kw = np.full(T, 10.0)
    usage_kw = np.zeros(T)
    result = soc_mod.compute_soc(pv_kw, usage_kw, block_minutes=60.0, capacity_kwh=100.0, soc_init_pct=0.0)
    # +10 kWh/hour into a 100 kWh battery -> +10 percentage points per block
    expected = np.arange(1, T + 1) * 10.0
    assert np.allclose(result.socs_pct, expected)
    assert np.allclose(result.actual_soc_pct, expected)  # no clipping needed yet


def test_socs_exceeds_100_when_actual_soc_is_clipped():
    T = 5
    pv_kw = np.full(T, 20.0)
    usage_kw = np.zeros(T)
    result = soc_mod.compute_soc(pv_kw, usage_kw, block_minutes=60.0, capacity_kwh=50.0, soc_init_pct=80.0)
    # starts at 40 kWh, +20 kWh/block -> potential energy keeps climbing past 50 kWh cap
    assert result.socs_pct[-1] > 100.0
    assert np.all(result.actual_soc_pct <= 100.0 + 1e-9)
    assert result.surplus_kwh.sum() > 0.0


def test_socs_goes_negative_on_sustained_deficit():
    T = 5
    pv_kw = np.zeros(T)
    usage_kw = np.full(T, 10.0)
    result = soc_mod.compute_soc(pv_kw, usage_kw, block_minutes=60.0, capacity_kwh=30.0, soc_init_pct=10.0)
    assert result.socs_pct[-1] < 0.0
    assert np.all(result.actual_soc_pct >= 0.0 - 1e-9)
    assert result.deficit_kwh.sum() > 0.0


def test_actual_soc_never_leaves_0_100_band():
    rng = np.random.default_rng(0)
    T = config.TIME.T
    pv_kw = rng.uniform(0, 20, size=T)
    usage_kw = rng.uniform(0, 15, size=T)
    result = soc_mod.compute_soc(pv_kw, usage_kw)
    assert np.all(result.actual_soc_pct >= -1e-9)
    assert np.all(result.actual_soc_pct <= 100.0 + 1e-9)


def test_reset_daily_resets_only_the_potential_tracker_not_the_battery():
    # 2 days at 1h blocks (24 blocks/day), constant surplus of +5 kWh/block into a 100 kWh
    # battery -- the real (clipped) battery fills up and caps at 100% after 20 blocks (block
    # index 19), well before day 1 ends.
    T = 48
    pv_kw = np.full(T, 5.0)
    usage_kw = np.zeros(T)
    result = soc_mod.compute_soc(pv_kw, usage_kw, block_minutes=60.0, capacity_kwh=100.0, soc_init_pct=0.0)

    # the real battery: fills to 100% partway through day 1, then carries that charge straight
    # into day 2 with no reset -- it's the same physical battery, still full.
    assert result.actual_soc_pct[19] == 100.0   # first block the battery is full, mid-day-1
    assert result.actual_soc_pct[23] == 100.0   # end of day 1: still full
    assert result.actual_soc_pct[24] == 100.0   # start of day 2: still full, NOT reset
    assert result.actual_soc_pct[47] == 100.0   # end of day 2: still full (still in surplus)

    # the "potential" tracker: never clips, and each day restarts from the real battery's
    # *current* charge rather than a fixed baseline.
    assert result.socs_pct[23] == 120.0         # day 1: 0 + 5kWh x 24 blocks, unclipped
    assert result.socs_pct[24] == 105.0         # day 2 restarts from the real battery (100%), then +5
    assert np.isclose(result.socs_pct[47], 220.0)  # day 2: 100 + 5kWh x 24 blocks, unclipped


def test_reset_daily_false_keeps_one_continuous_integration():
    T = 48
    pv_kw = np.full(T, 5.0)
    usage_kw = np.zeros(T)
    result = soc_mod.compute_soc(pv_kw, usage_kw, block_minutes=60.0, capacity_kwh=100.0, soc_init_pct=0.0,
                                  reset_daily=False)
    assert result.socs_pct[23] == 120.0
    assert result.socs_pct[24] == 125.0        # keeps climbing, no reset
    assert result.socs_pct[47] == 240.0


def test_mismatched_lengths_raise():
    try:
        soc_mod.compute_soc(np.zeros(5), np.zeros(4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched pv/usage lengths")
