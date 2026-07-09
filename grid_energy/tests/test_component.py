from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from grid_energy.component import GridEnergyComponent
from grid_energy.quartz_forecast import BLOCK_MINUTES, ForecastResult


def _fake_week_forecast(power_kw_value: float, n_blocks: int) -> ForecastResult:
    idx = pd.date_range("2026-01-01", periods=n_blocks, freq="15min")
    power_kw = pd.Series(np.full(n_blocks, power_kw_value), index=idx)
    energy_wh = power_kw * 1000.0 * (BLOCK_MINUTES / 60.0)
    return ForecastResult(power_kw=power_kw, energy_wh=energy_wh)


def test_compute_soc_for_usage_resamples_and_aligns_to_forecast_length():
    # 2 days (192 x 15-min blocks) of constant 10 kW PV.
    forecast = _fake_week_forecast(power_kw_value=10.0, n_blocks=192)
    # One simulated "day" at sim's 5-min resolution (288 blocks), constant 4 kW usage.
    usage_5min = np.full(288, 4.0)

    component = GridEnergyComponent(capacity_kwh=1000.0, soc_init_pct=0.0)
    with patch.object(component, "forecast_pv_week", return_value=forecast):
        result = component.compute_soc_for_usage(usage_5min, usage_block_minutes=5.0)

    # aligned to the forecast's length, not the usage series' original length
    assert len(result.usage_kw) == 192
    assert len(result.pv_kw) == 192
    # resampling 5-min -> 15-min preserves the constant value; tiling one day (96 x 15-min
    # blocks) to 192 blocks just repeats it once more, so usage should stay a flat 4.0 throughout
    assert np.allclose(result.usage_kw, 4.0)
    assert np.allclose(result.pv_kw, 10.0)


def test_compute_soc_for_usage_uses_a_prefetched_forecast_without_refetching():
    forecast = _fake_week_forecast(power_kw_value=5.0, n_blocks=96)
    usage_5min = np.full(288, 1.0)

    component = GridEnergyComponent()
    with patch.object(component, "forecast_pv_week") as mock_fetch:
        result = component.compute_soc_for_usage(usage_5min, usage_block_minutes=5.0, forecast=forecast)

    mock_fetch.assert_not_called()
    assert len(result.pv_kw) == 96


def test_forecast_pv_week_passes_through_site_overrides():
    component = GridEnergyComponent(latitude=1.0, longitude=2.0, capacity_kwp=99.0, nwp_source="gfs")
    with patch("grid_energy.component.quartz_forecast.forecast_week_kw") as mock_forecast:
        component.forecast_pv_week(start="2026-01-01")

    mock_forecast.assert_called_once_with(
        latitude=1.0, longitude=2.0, capacity_kwp=99.0, start="2026-01-01", nwp_source="gfs")


def test_compute_soc_for_usage_passes_through_battery_overrides():
    """capacity_kwh / soc_init_pct set on the component must reach soc.compute_soc unchanged --
    this is what lets a caller size the battery per-scenario instead of only via config.py."""
    forecast = _fake_week_forecast(power_kw_value=1.0, n_blocks=96)
    usage_5min = np.full(288, 1.0)

    component = GridEnergyComponent(capacity_kwh=42.0, soc_init_pct=77.0)
    with patch("grid_energy.component.soc_mod.compute_soc") as mock_compute_soc:
        component.compute_soc_for_usage(usage_5min, usage_block_minutes=5.0, forecast=forecast)

    _, kwargs = mock_compute_soc.call_args
    assert kwargs["capacity_kwh"] == 42.0
    assert kwargs["soc_init_pct"] == 77.0


def test_unset_capacity_and_battery_fields_pass_through_as_none_to_use_config_defaults():
    """No capacity_kwp/capacity_kwh/soc_init_pct given -> the component must forward None,
    not read config.py itself -- config.PV/config.BATTERY defaults are applied one layer down,
    inside quartz_forecast.forecast_week_kw / soc.compute_soc, not duplicated here."""
    component = GridEnergyComponent()

    with patch("grid_energy.component.quartz_forecast.forecast_week_kw") as mock_forecast:
        component.forecast_pv_week()
    assert mock_forecast.call_args.kwargs["capacity_kwp"] is None

    forecast = _fake_week_forecast(power_kw_value=1.0, n_blocks=96)
    usage_5min = np.full(288, 1.0)
    with patch("grid_energy.component.soc_mod.compute_soc") as mock_compute_soc:
        component.compute_soc_for_usage(usage_5min, usage_block_minutes=5.0, forecast=forecast)
    assert mock_compute_soc.call_args.kwargs["capacity_kwh"] is None
    assert mock_compute_soc.call_args.kwargs["soc_init_pct"] is None


def test_capacity_kwh_override_changes_soc_math_end_to_end():
    """Not just a pass-through check -- confirms a smaller/larger capacity_kwh on the component
    actually changes the resulting SOC numbers, using the real (unmocked) soc.compute_soc."""
    # constant PV surplus of +2 kW every 15-min block, zero usage
    forecast = _fake_week_forecast(power_kw_value=2.0, n_blocks=8)
    usage_5min = np.zeros(8 * 3)  # 5-min-resolution zeros, resamples 1:1 in energy terms

    small_battery = GridEnergyComponent(capacity_kwh=1.0, soc_init_pct=0.0)
    big_battery = GridEnergyComponent(capacity_kwh=100.0, soc_init_pct=0.0)

    small_result = small_battery.compute_soc_for_usage(usage_5min, usage_block_minutes=5.0, forecast=forecast)
    big_result = big_battery.compute_soc_for_usage(usage_5min, usage_block_minutes=5.0, forecast=forecast)

    # same PV, same usage: the small battery fills up and starts overflowing almost immediately,
    # the big one never does over just 8 blocks
    assert small_result.surplus_kwh.sum() > 0.0
    assert big_result.surplus_kwh.sum() == 0.0
    assert small_result.actual_soc_pct.max() == 100.0
    assert big_result.actual_soc_pct.max() < 100.0
