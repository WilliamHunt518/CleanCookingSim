"""Grid energy-balance model: real PV forecast, usage, battery SOC.

Additive to the rest of the repo -- nothing outside this folder imports it,
and it only reaches into `sim` optionally (grid_energy/demo_forecast_week.py,
a worked example, not a dependency of the component itself). See README.md
for the equation, every assumption, and how a future caller (sim, later)
would use GridEnergyComponent to drive this with its own usage timeseries.
"""
from __future__ import annotations

from .component import GridEnergyComponent
from .quartz_forecast import ForecastResult, forecast_day_kw, forecast_week_kw
from .resample import resample_kw, tile_to_length
from .soc import SOCResult, compute_soc

__all__ = [
    "GridEnergyComponent",
    "ForecastResult", "forecast_day_kw", "forecast_week_kw",
    "resample_kw", "tile_to_length",
    "compute_soc", "SOCResult",
]
