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
