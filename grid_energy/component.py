"""
GridEnergyComponent: the single entry point this folder is meant to be
driven through once something outside grid_energy (namely sim, later) wants
real PV + battery SOC without caring about this folder's internals.

grid_energy still imports nothing from sim (only demo_forecast_week.py does,
as a worked example of a *consumer*, not a dependency of the component
itself) -- the intended direction is sim -> grid_energy, never the reverse.
A future integration would look like:

    from grid_energy import GridEnergyComponent

    component = GridEnergyComponent()                       # Oloika defaults, see config.py
    day = sim.run.simulate_day(population, price, scenario, rng)
    result = component.compute_soc_for_usage(day.demand_kw, usage_block_minutes=sim.config.STATE.block_minutes)
    # result.socs_pct / result.actual_soc_pct / result.surplus_kwh / result.deficit_kwh
    # are now real-PV-forecast-driven, at grid_energy's native 15-min resolution.

That is deliberately the whole surface: one object, one config (all fields
optional, defaulting to config.SITE/config.PV/config.BATTERY), one method
that takes an arbitrary usage_kw series at an arbitrary fixed block size and
returns a full SOCResult. Everything else in this folder (quartz_forecast.py,
soc.py, resample.py) is an implementation detail behind that method.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, quartz_forecast, soc as soc_mod
from .resample import resample_kw, tile_to_length


@dataclass
class GridEnergyComponent:
    latitude: float | None = None
    longitude: float | None = None
    capacity_kwp: float | None = None       # PV array, defaults to config.PV.rated_kwp
    capacity_kwh: float | None = None       # battery, defaults to config.BATTERY.capacity_kwh
    soc_init_pct: float | None = None       # defaults to config.BATTERY.soc_init_pct
    nwp_source: str = "icon"

    def forecast_pv_day(self, start=None) -> quartz_forecast.ForecastResult:
        """Real PV(t) for 1 day at quartz_forecast's native 15-minute resolution -- 1 HTTP request
        instead of forecast_pv_week's 7, a fast default for a UI that fetches automatically."""
        return quartz_forecast.forecast_one_day_kw(
            latitude=self.latitude, longitude=self.longitude, capacity_kwp=self.capacity_kwp,
            start=start, nwp_source=self.nwp_source)

    def forecast_pv_week(self, start=None) -> quartz_forecast.ForecastResult:
        """Real PV(t) for 7 days at quartz_forecast's native 15-minute resolution."""
        return quartz_forecast.forecast_week_kw(
            latitude=self.latitude, longitude=self.longitude, capacity_kwp=self.capacity_kwp,
            start=start, nwp_source=self.nwp_source)

    def compute_soc_for_usage(self, usage_kw, usage_block_minutes: float,
                               forecast: quartz_forecast.ForecastResult | None = None,
                               reset_daily: bool = True) -> soc_mod.SOCResult:
        """Real PV(t) for one week vs an arbitrary usage_kw series at usage_block_minutes
        resolution (e.g. sim's 5-min demand_kw) -> socs(t).

        usage_kw is resampled to quartz_forecast.BLOCK_MINUTES (resample_kw) and then
        tiled/truncated to the forecast's length (tile_to_length) -- e.g. one simulated day
        (288 x 5-min blocks) becomes a full week (672 x 15-min blocks) by resampling to
        96 blocks/day and repeating 7x. Pass a pre-fetched `forecast` (from forecast_pv_week)
        to avoid re-fetching the same week's PV forecast across multiple usage scenarios.
        """
        forecast = self.forecast_pv_week() if forecast is None else forecast
        usage_native = resample_kw(usage_kw, usage_block_minutes, quartz_forecast.BLOCK_MINUTES)
        usage_aligned = tile_to_length(usage_native, len(forecast.power_kw))

        return soc_mod.compute_soc(
            forecast.power_kw.to_numpy(), usage_aligned, block_minutes=quartz_forecast.BLOCK_MINUTES,
            capacity_kwh=self.capacity_kwh, soc_init_pct=self.soc_init_pct, reset_daily=reset_daily)
