"""Grid energy-balance model: real PV forecast, usage, battery SOC.

Additive to the rest of the repo -- nothing outside this folder imports it,
and it only reaches into `sim` optionally (grid_energy/demo_forecast_week.py,
to drive Usage(t) from the real cooking simulation). See README.md for the
equation and every assumption behind it.
"""
from __future__ import annotations

from .quartz_forecast import ForecastResult, forecast_day_kw, forecast_week_kw
from .soc import SOCResult, compute_soc

__all__ = [
    "ForecastResult", "forecast_day_kw", "forecast_week_kw",
    "compute_soc", "SOCResult",
]
