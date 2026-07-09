from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

from grid_energy import quartz_forecast


def _fake_day(ts: pd.Timestamp, power_kw: float) -> pd.DataFrame:
    idx = pd.date_range(start=ts, periods=quartz_forecast.BLOCKS_PER_DAY, freq="15min")
    return pd.DataFrame({"power_kw": np.full(len(idx), power_kw)}, index=idx)


def test_forecast_week_concatenates_seven_daily_calls():
    calls = []

    def fake_forecast_day_kw(latitude, longitude, capacity_kwp, ts, nwp_source="icon"):
        calls.append(ts)
        return _fake_day(ts, power_kw=10.0)

    with patch.object(quartz_forecast, "forecast_day_kw", side_effect=fake_forecast_day_kw):
        result = quartz_forecast.forecast_week_kw(start="2026-01-01")

    assert len(calls) == 7  # one daily-anchored call per day of the week, see module docstring
    assert len(result.power_kw) == 7 * quartz_forecast.BLOCKS_PER_DAY
    assert result.peak_power_kw == 10.0


def test_energy_wh_matches_power_times_quarter_hour():
    fake = lambda latitude, longitude, capacity_kwp, ts, nwp_source="icon": _fake_day(ts, power_kw=4.0)
    with patch.object(quartz_forecast, "forecast_day_kw", side_effect=fake):
        result = quartz_forecast.forecast_week_kw(start="2026-01-01")

    # 4 kW held for a 15-min block -> 4 * 1000 * 0.25 = 1000 Wh per block
    assert np.allclose(result.energy_wh.to_numpy(), 1000.0)
    assert result.total_energy_wh == 7 * quartz_forecast.BLOCKS_PER_DAY * 1000.0


def test_forecast_one_day_kw_makes_a_single_call():
    calls = []

    def fake_forecast_day_kw(latitude, longitude, capacity_kwp, ts, nwp_source="icon"):
        calls.append(ts)
        return _fake_day(ts, power_kw=8.0)

    with patch.object(quartz_forecast, "forecast_day_kw", side_effect=fake_forecast_day_kw):
        result = quartz_forecast.forecast_one_day_kw(start="2026-01-01")

    assert len(calls) == 1  # the whole point -- 1 HTTP request, not forecast_week_kw's 7
    assert len(result.power_kw) == quartz_forecast.BLOCKS_PER_DAY
    assert result.peak_power_kw == 8.0


def test_forecast_day_builds_a_fresh_site_every_call():
    """Regression test: quartz_solar_forecast.forecast.predict_ocf mutates a PVSite's
    capacity_kwp to 4 in place for capacity_kwp > 4 (see forecast_day_kw's docstring).
    Reusing one PVSite object across the week's 7 calls used to mean every call after the
    first silently stopped rescaling to the real capacity. forecast_day_kw must build a new
    PVSite (via _build_site) on every call, not accept/reuse one from the caller."""
    build_calls = []

    def fake_build_site(latitude, longitude, capacity_kwp):
        site = object()
        build_calls.append(site)
        return site

    def fake_run_forecast(site, ts, nwp_source, model):
        assert site is build_calls[-1]  # forecast_day_kw must use the site it just built
        return _fake_day(ts, power_kw=10.0)

    fake_pkg = types.ModuleType("quartz_solar_forecast")
    fake_forecast_mod = types.ModuleType("quartz_solar_forecast.forecast")
    fake_forecast_mod.run_forecast = fake_run_forecast
    fake_pkg.forecast = fake_forecast_mod

    with patch.object(quartz_forecast, "_build_site", side_effect=fake_build_site), \
         patch.object(quartz_forecast, "_patch_requests_cache", lambda: None), \
         patch.dict(sys.modules, {"quartz_solar_forecast": fake_pkg,
                                   "quartz_solar_forecast.forecast": fake_forecast_mod}):
        quartz_forecast.forecast_week_kw(start="2026-01-01")

    assert len(build_calls) == 7
    assert len(set(build_calls)) == 7  # every call got a distinct, freshly-built site object


def test_forecast_day_clips_negative_power_to_zero():
    """Regression test: quartz-solar-forecast's 'gb' model is a gradient-boosted regressor with
    no physical non-negativity constraint on its own output -- it can and does predict slightly
    negative power_kw during low-irradiance transitions (observed live: ~4% of blocks, down to
    about -7 kW at 120 kWp scale, clustered around dawn ramp-up and brief midday cloud dips).
    forecast_day_kw must clip that to 0 before it reaches soc.compute_soc, which would otherwise
    book negative PV as extra usage."""

    def fake_run_forecast(site, ts, nwp_source, model):
        idx = pd.date_range(start=ts, periods=quartz_forecast.BLOCKS_PER_DAY, freq="15min")
        values = np.full(len(idx), 5.0)
        values[10] = -3.5   # simulate the model's dawn/midday negative-regression artifact
        values[11] = -0.1
        return pd.DataFrame({"power_kw": values}, index=idx)

    fake_pkg = types.ModuleType("quartz_solar_forecast")
    fake_forecast_mod = types.ModuleType("quartz_solar_forecast.forecast")
    fake_forecast_mod.run_forecast = fake_run_forecast
    fake_pkg.forecast = fake_forecast_mod

    with patch.object(quartz_forecast, "_patch_requests_cache", lambda: None), \
         patch.object(quartz_forecast, "_build_site", return_value=object()), \
         patch.dict(sys.modules, {"quartz_solar_forecast": fake_pkg,
                                   "quartz_solar_forecast.forecast": fake_forecast_mod}):
        day_df = quartz_forecast.forecast_day_kw(None, None, None, pd.Timestamp("2026-01-01"))

    assert np.all(day_df["power_kw"] >= 0.0)
    assert day_df["power_kw"].iloc[10] == 0.0
    assert day_df["power_kw"].iloc[11] == 0.0
    assert day_df["power_kw"].iloc[0] == 5.0  # untouched elsewhere
