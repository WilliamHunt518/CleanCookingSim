import numpy as np

from sim import config, tariffs


def test_all_candidates_normalise_to_common_mean():
    """extreme_test is deliberately excluded: it's a sanity-check stress tariff at an elevated
    price *level* (see test_extreme_test_tariff_is_uniformly_elevated), not a realistic candidate
    reshaped around the same average like the others. The forecast-driven candidates (green_light
    etc.) are excluded too -- unnormalised by design (TARIFF_STRATEGIES.md section 0.5) and, more
    to the point, need a live PV forecast to build at all (see test_tariffs_forecast_driven.py,
    which mocks that out)."""
    skip = {"extreme_test"} | tariffs.FORECAST_DRIVEN_NAMES
    for name, fn in tariffs.CANDIDATES.items():
        if name in skip:
            continue
        price = fn()
        assert np.isclose(price.mean(), config.TARIFF.p_bar), name


def test_extreme_test_tariff_is_uniformly_elevated():
    price = tariffs.tariff_extreme_test()
    expected = config.TARIFF.p_bar * config.TARIFF.extreme_test_multiplier
    assert np.all(np.isclose(price, expected))
    assert expected > config.TARIFF.p_hi


def test_evening_peak_is_higher_in_peak_window_than_flat():
    flat = tariffs.tariff_flat()
    peak = tariffs.tariff_evening_peak()
    t = tariffs._t_hours()
    lo, hi = config.TARIFF.w_peak_hr
    in_peak = (t >= lo) & (t < hi)
    assert np.mean(peak[in_peak]) > np.mean(flat[in_peak])
    assert np.mean(peak[~in_peak]) < np.mean(flat[~in_peak])


def test_pv_profile_zero_outside_daylight():
    pv = tariffs.pv_profile()
    t = tariffs._t_hours()
    outside = (t < config.TARIFF.pv_t_rise_hr) | (t > config.TARIFF.pv_t_set_hr)
    assert np.all(pv[outside] == 0)
    assert np.max(pv) > 0
