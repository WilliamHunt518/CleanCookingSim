"""Tests for sim.tariffs's five forecast-driven adapters (green_light, pv_following_real,
soc_banded, residual_load, deficit_guard). All monkeypatch tariffs._cached_forecast /
_cached_reference_soc directly -- these are the *only* two entry points to a live network call in
sim.tariffs, so mocking them is sufficient to guarantee no test here ever touches the network,
without needing to mock quartz_solar_forecast/HTTP itself."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grid_energy.quartz_forecast import BLOCK_MINUTES, ForecastResult
from grid_energy.soc import SOCResult
from sim import config, tariffs

N_WEEK = 672  # 7 days x 96 x 15-min blocks


def _fake_forecast(power_kw_values) -> ForecastResult:
    power_kw_values = np.asarray(power_kw_values, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(power_kw_values), freq="15min")
    power_kw = pd.Series(power_kw_values, index=idx)
    return ForecastResult(power_kw=power_kw, energy_wh=power_kw * 1000.0 * (BLOCK_MINUTES / 60.0))


def _fake_soc(n=N_WEEK, pv_kw=None, usage_kw=None, actual_soc_pct=None,
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


@pytest.fixture(autouse=True)
def _clear_cache_and_stub_network(monkeypatch):
    """Every test gets a fresh cache (so one test's monkeypatched values can't leak into the
    next) and a hard failure if anything tries to actually hit the network."""
    tariffs.reset_forecast_driven_cache()

    def _boom():
        raise AssertionError("a forecast-driven tariff test tried to hit the live network")

    monkeypatch.setattr(tariffs, "_grid_component", _boom)
    yield
    tariffs.reset_forecast_driven_cache()


def test_forecast_driven_names_all_registered_in_candidates():
    assert tariffs.FORECAST_DRIVEN_NAMES == {
        "green_light", "pv_following_real", "soc_banded", "residual_load", "deficit_guard", "ga_optimal"}
    for name in tariffs.FORECAST_DRIVEN_NAMES:
        assert name in tariffs.CANDIDATES


def test_pv_following_real_only_needs_forecast_not_reference_soc(monkeypatch):
    """The one strategy that needs no usage forecast -- must not touch _cached_reference_soc at
    all (which would otherwise trigger a reference sim.run.simulate_day)."""
    pv = np.tile(np.concatenate([np.zeros(48), np.linspace(0, 10, 48)]), 7)  # night/day cycle
    monkeypatch.setattr(tariffs, "_cached_forecast", lambda: _fake_forecast(pv))
    monkeypatch.setattr(tariffs, "_cached_reference_soc",
                         lambda: (_ for _ in ()).throw(AssertionError("should not be called")))

    price = tariffs.tariff_pv_following_real()
    assert price.shape == (config.STATE.T,)
    assert np.all(np.isfinite(price))


def test_green_light_output_shape_and_unit_conversion(monkeypatch):
    surplus = np.zeros(N_WEEK)
    surplus[10:20] = 1.0  # some surplus early in day 0 (REFERENCE_DAY)
    monkeypatch.setattr(tariffs, "_cached_reference_soc", lambda: _fake_soc(surplus_kwh=surplus))

    price = tariffs.tariff_green_light()
    assert price.shape == (config.STATE.T,)

    # KES 40 (P_FLAT) -> sim units via KES_PER_SIM_UNIT=160 -> 0.25 == p_bar, the calibration anchor
    from grid_energy import config as grid_config
    expected_flat_sim_price = 40.0 / grid_config.PRICING.KES_PER_SIM_UNIT
    assert expected_flat_sim_price == pytest.approx(config.TARIFF.p_bar)
    # blocks well outside the (upsampled) surplus window should be at the flat KES 40 rate
    assert price[-1] == pytest.approx(expected_flat_sim_price)


def test_soc_banded_uses_reference_soc(monkeypatch):
    soc_pct = np.full(N_WEEK, 50.0)
    soc_pct[5] = 95.0  # cheap band somewhere in day 0
    monkeypatch.setattr(tariffs, "_cached_reference_soc", lambda: _fake_soc(actual_soc_pct=soc_pct))

    price = tariffs.tariff_soc_banded()
    assert price.shape == (config.STATE.T,)
    assert np.all(np.isfinite(price))
    assert price.min() < price.max()  # the cheap-band block should be visible after upsampling


def test_residual_load_uses_reference_soc(monkeypatch):
    pv = np.zeros(N_WEEK)
    usage = np.zeros(N_WEEK)
    pv[0] = 10.0     # surplus block
    usage[1] = 10.0  # deficit block, same day
    monkeypatch.setattr(tariffs, "_cached_reference_soc", lambda: _fake_soc(pv_kw=pv, usage_kw=usage))

    price = tariffs.tariff_residual_load()
    assert price.shape == (config.STATE.T,)
    assert price[0] < price[3]  # surplus block cheaper than the deficit block just after it


def test_deficit_guard_uses_reference_soc(monkeypatch):
    deficit = np.zeros(N_WEEK)
    deficit[50] = 1.0
    monkeypatch.setattr(tariffs, "_cached_reference_soc", lambda: _fake_soc(deficit_kwh=deficit))

    price = tariffs.tariff_deficit_guard()
    assert price.shape == (config.STATE.T,)
    assert price.max() == pytest.approx(80.0 / 160.0)  # P_MAX in sim units, surged somewhere


def test_forecast_is_cached_across_multiple_calls(monkeypatch):
    calls = []

    def fake_forecast_week():
        calls.append(1)
        return _fake_forecast(np.ones(N_WEEK) * 5.0)

    class FakeComponent:
        def forecast_pv_week(self):
            return fake_forecast_week()

    monkeypatch.setattr(tariffs, "_grid_component", lambda: FakeComponent())

    tariffs.tariff_pv_following_real()
    tariffs.tariff_pv_following_real()
    tariffs._cached_forecast()

    assert len(calls) == 1  # fetched exactly once despite three calls


def test_reset_forecast_driven_cache_forces_a_refetch(monkeypatch):
    calls = []

    class FakeComponent:
        def forecast_pv_week(self):
            calls.append(1)
            return _fake_forecast(np.ones(N_WEEK) * 5.0)

    monkeypatch.setattr(tariffs, "_grid_component", lambda: FakeComponent())

    tariffs.tariff_pv_following_real()
    tariffs.reset_forecast_driven_cache()
    tariffs.tariff_pv_following_real()

    assert len(calls) == 2


# --------------------------------------------------------------------------- ga_optimal

def test_ga_optimal_raises_a_clear_error_when_nothing_saved_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(tariffs, "GA_OPTIMAL_PATH", str(tmp_path / "does_not_exist.json"))
    with pytest.raises(FileNotFoundError, match="No GA-optimised tariff saved yet"):
        tariffs.tariff_ga_optimal()


def test_save_and_load_ga_optimal_roundtrips(tmp_path, monkeypatch):
    path = str(tmp_path / "ga_winner.json")
    monkeypatch.setattr(tariffs, "GA_OPTIMAL_PATH", path)

    theta = np.array([0.2, 0.8, 0.0, 0.0, 0.0, 1.3, 88.0])
    price = np.linspace(0.05, 0.5, config.STATE.T)

    saved_path = tariffs.save_ga_optimal(theta, price)
    assert saved_path == path

    loaded = tariffs.tariff_ga_optimal()
    np.testing.assert_allclose(loaded, price)


def test_ga_optimal_is_registered_in_candidates_and_forecast_driven():
    assert "ga_optimal" in tariffs.CANDIDATES
    assert "ga_optimal" in tariffs.FORECAST_DRIVEN_NAMES
