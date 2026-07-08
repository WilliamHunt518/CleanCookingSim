import numpy as np

from sim import config, tariffs


def test_all_candidates_normalise_to_common_mean():
    for name, price in tariffs.all_tariffs().items():
        assert np.isclose(price.mean(), config.TARIFF.p_bar), name


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
