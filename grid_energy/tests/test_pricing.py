from __future__ import annotations

import numpy as np
import pandas as pd

from grid_energy import config, pricing
from grid_energy.quartz_forecast import ForecastResult
from grid_energy.soc import SOCResult


def _make_soc(n: int = 8, pv_kw=None, usage_kw=None, actual_soc_pct=None,
              surplus_kwh=None, deficit_kwh=None) -> SOCResult:
    zeros = np.zeros(n)
    return SOCResult(
        t_hr=np.arange(n) * 0.25,
        pv_kw=zeros if pv_kw is None else np.asarray(pv_kw, dtype=float),
        usage_kw=zeros if usage_kw is None else np.asarray(usage_kw, dtype=float),
        net_kw=zeros,
        energy_potential_kwh=zeros,
        socs_pct=zeros,
        energy_actual_kwh=zeros,
        actual_soc_pct=zeros if actual_soc_pct is None else np.asarray(actual_soc_pct, dtype=float),
        surplus_kwh=zeros if surplus_kwh is None else np.asarray(surplus_kwh, dtype=float),
        deficit_kwh=zeros if deficit_kwh is None else np.asarray(deficit_kwh, dtype=float),
    )


def _make_forecast(power_kw_values) -> ForecastResult:
    power_kw_values = np.asarray(power_kw_values, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(power_kw_values), freq="15min")
    power_kw = pd.Series(power_kw_values, index=idx)
    return ForecastResult(power_kw=power_kw, energy_wh=power_kw * 1000.0 * 0.25)


# --------------------------------------------------------------------------- green_light (A)

def test_green_light_discounts_only_surplus_blocks():
    soc = _make_soc(n=6, surplus_kwh=[0, 0, 1.5, 2.0, 0, 0])
    price = pricing.green_light(soc)
    np.testing.assert_array_equal(price, [40, 40, 30, 30, 40, 40])


def test_green_light_empty_surplus_window_is_flat_all_day():
    """No surplus anywhere (a cloudy day) -> the tariff self-adapts to flat, no discount offered."""
    soc = _make_soc(n=6, surplus_kwh=np.zeros(6))
    price = pricing.green_light(soc)
    assert np.all(price == config.PRICING.P_FLAT)


def test_green_light_pad_blocks_dilates_the_discount_window():
    soc = _make_soc(n=8, surplus_kwh=[0, 0, 0, 5.0, 0, 0, 0, 0])
    price = pricing.green_light(soc, pad_blocks=1)
    np.testing.assert_array_equal(price, [40, 40, 30, 30, 30, 40, 40, 40])


def test_green_light_custom_params_and_clamp():
    soc = _make_soc(n=3, surplus_kwh=[1, 0, 0])
    price = pricing.green_light(soc, p_disc=-10, p_flat=200)
    # both custom values still clamp to [P_MIN, P_MAX]
    assert price[0] == config.PRICING.P_MIN
    assert price[1] == config.PRICING.P_MAX


# --------------------------------------------------------------------------- pv_following_real (B)

def test_pv_following_real_zero_pv_is_expensive_max_pv_is_cheap():
    forecast = _make_forecast([0.0, 5.0, 10.0])
    price = pricing.pv_following_real(forecast, p_lo=10.0, p_hi=80.0)
    assert price[0] == 80.0  # pv=0 -> p_hi
    assert price[2] == 10.0  # pv=p_ref (max) -> p_lo
    assert 10.0 < price[1] < 80.0


def test_pv_following_real_dark_day_is_flat_p_hi():
    forecast = _make_forecast([0.0, 0.0, 0.0])
    price = pricing.pv_following_real(forecast, p_lo=10.0, p_hi=80.0)
    assert np.all(price == 80.0)


def test_pv_following_real_gamma_shapes_the_curve():
    forecast = _make_forecast([0.0, 2.5, 5.0, 7.5, 10.0])
    front_loaded = pricing.pv_following_real(forecast, p_lo=0.0, p_hi=100.0, gamma=0.5)
    peak_reserved = pricing.pv_following_real(forecast, p_lo=0.0, p_hi=100.0, gamma=2.0)
    # at the same modest-sun block (index 1, pv=2.5 of 10 max), gamma<1 should have already
    # discounted more than gamma>1 (front-loads vs. reserves-for-peak)
    assert front_loaded[1] < peak_reserved[1]


def test_pv_following_real_explicit_p_ref_guards_against_a_single_block_spike():
    forecast = _make_forecast([0.0, 1.0, 100.0])  # index 2 is an outlier spike
    default_ref = pricing.pv_following_real(forecast, p_lo=0.0, p_hi=100.0)
    explicit_ref = pricing.pv_following_real(forecast, p_lo=0.0, p_hi=100.0, p_ref=2.0)
    # with the spike as p_ref, block 1's discount is tiny; with a sane explicit p_ref it's much bigger
    assert explicit_ref[1] < default_ref[1]


# --------------------------------------------------------------------------- soc_banded (C)

def test_soc_banded_three_bands():
    soc = _make_soc(n=5, actual_soc_pct=[95, 50, 10, 90, 20])
    price = pricing.soc_banded(soc, theta_hi=90, theta_lo=20, p_cheap=20, p_mid=40, p_dear=70)
    np.testing.assert_array_equal(price, [20, 40, 70, 20, 70])


def test_soc_banded_never_crossing_thresholds_is_flat_mid():
    soc = _make_soc(n=4, actual_soc_pct=[50, 55, 60, 45])
    price = pricing.soc_banded(soc, theta_hi=90, theta_lo=20, p_mid=40)
    assert np.all(price == 40)


# --------------------------------------------------------------------------- residual_load (D)

def test_residual_load_surplus_is_cheap_deficit_is_expensive():
    # block 0: pv >> usage (big surplus); block 1: usage >> pv (big deficit); block 2: balanced
    soc = _make_soc(n=3, pv_kw=[10.0, 0.0, 5.0], usage_kw=[0.0, 10.0, 5.0])
    price = pricing.residual_load(soc, p_lo=0.0, p_hi=80.0)
    assert price[0] == 0.0    # max surplus -> p_lo
    assert price[1] == 80.0   # max deficit -> p_hi
    assert price[2] == 40.0   # balanced -> midpoint


def test_residual_load_identically_balanced_day_is_flat_midpoint():
    soc = _make_soc(n=4, pv_kw=[3, 3, 3, 3], usage_kw=[3, 3, 3, 3])
    price = pricing.residual_load(soc, p_lo=0.0, p_hi=80.0)
    assert np.all(price == 40.0)


# --------------------------------------------------------------------------- deficit_guard (E)

def test_deficit_guard_surcharges_deficit_and_lead_in_window():
    soc = _make_soc(n=10, deficit_kwh=[0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0])
    price = pricing.deficit_guard(soc, p_flat=40, p_surge=80, lead_blocks=2)
    # deficit at index 5 -> surcharge indices [3, 4, 5], flat everywhere else
    np.testing.assert_array_equal(price, [40, 40, 40, 80, 80, 80, 40, 40, 40, 40])


def test_deficit_guard_no_deficit_is_flat_all_day():
    soc = _make_soc(n=6, deficit_kwh=np.zeros(6))
    price = pricing.deficit_guard(soc, p_flat=40)
    assert np.all(price == 40)


def test_deficit_guard_lead_window_clips_at_day_start():
    soc = _make_soc(n=5, deficit_kwh=[2.0, 0, 0, 0, 0])
    price = pricing.deficit_guard(soc, p_flat=40, p_surge=80, lead_blocks=3)
    assert price[0] == 80  # doesn't crash indexing before block 0
