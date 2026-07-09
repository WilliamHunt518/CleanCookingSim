"""
PV(t) from a real forecast, via the `quartz-solar-forecast` library
(https://github.com/openclimatefix/open-source-quartz-solar-forecast, docs
at https://open.quartz.solar/docs). This is the only PV source in this
folder -- an earlier assumption-based clear-sky-shape model (weather.py/
pv.py) was retired once this real forecast was wired in.

    pip install quartz-solar-forecast

`quartz_solar_forecast.forecast.run_forecast(site, ts, ...)` predicts
`power_kw` at 15-minute resolution for a fixed 48h horizon starting at `ts`
(gradient-boosted model trained on real PV site history, driven by Open-Meteo
NWP weather data -- no API key needed). It does not expose a "give me a week"
option directly, so `forecast_week_kw` below calls it once per day (7 daily
anchors at 00:00, keeping each call's first 24h/96 blocks) and concatenates
-- the standard way to build a longer rolling forecast out of a fixed-horizon
model: each day is predicted from the freshest NWP run available for it,
rather than trusting one 7-day-old 48h forecast to stay accurate a week out.

Known install-time gotcha (worked around here, not upstream): as of this
writing, a plain `pip install quartz-solar-forecast` pulls in `attrs`/
`cattrs` releases new enough that `requests_cache`'s response serializer
raises `NameError: name 'RequestsCookieJar' is not defined` while trying to
disk-cache the Open-Meteo HTTP response (a forward-reference resolution bug
in that dependency combination, not in quartz-solar-forecast's own code).
Since on-disk response caching isn't needed for a handful of calls per week,
`_patch_requests_cache()` swaps `requests_cache.CachedSession` for a plain
`requests.Session` before the library's HTTP calls happen, sidestepping the
broken serializer entirely. Safe / a no-op if a future release fixes this
upstream (the patch is applied unconditionally, but `run_forecast` never
touches the cache serialization path we removed).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from . import config

BLOCK_MINUTES = 15.0
BLOCKS_PER_DAY = 96  # 24h / 15min


def _patch_requests_cache() -> None:
    import requests
    import requests_cache

    if getattr(requests_cache.CachedSession, "_grid_energy_patched", False):
        return
    uncached_session = lambda *a, **k: requests.Session()  # noqa: E731
    uncached_session._grid_energy_patched = True
    requests_cache.CachedSession = uncached_session


@dataclass
class ForecastResult:
    power_kw: pd.Series      # index: 15-min DatetimeIndex, values: forecast PV power [kW]
    energy_wh: pd.Series     # same index, per-block energy [Wh] = power_kw * 1000 * (15/60)

    @property
    def total_energy_wh(self) -> float:
        return float(self.energy_wh.sum())

    @property
    def peak_power_kw(self) -> float:
        return float(self.power_kw.max())


def _build_site(latitude: float | None, longitude: float | None, capacity_kwp: float | None):
    from quartz_solar_forecast.pydantic_models import PVSite

    return PVSite(
        latitude=config.SITE.latitude if latitude is None else latitude,
        longitude=config.SITE.longitude if longitude is None else longitude,
        capacity_kwp=config.PV.rated_kwp if capacity_kwp is None else capacity_kwp,
        tilt=config.SITE.tilt_deg,
        orientation=config.SITE.orientation_deg,
    )


def forecast_day_kw(latitude: float | None, longitude: float | None, capacity_kwp: float | None,
                     ts: pd.Timestamp, nwp_source: str = "icon") -> pd.DataFrame:
    """One run_forecast call (48h horizon, 15-min steps), trimmed to the first 24h/96 blocks
    starting at `ts` -- the freshest-NWP slice of that call.

    Builds a fresh PVSite every call rather than accepting/reusing one: for capacity_kwp > 4,
    quartz_solar_forecast.forecast.predict_ocf mutates site.capacity_kwp to 4 in place (the "gb"
    model is only trained up to 4 kWp, so it runs at 4 kWp and rescales the output afterwards).
    Reusing one PVSite object across multiple calls means every call after the first sees an
    already-mutated capacity_kwp=4, so predict_ocf's `if site.capacity_kwp > 4` check silently
    fails and it stops rescaling -- every call after the first would return a forecast ~scaled
    for a 4 kWp site while claiming to be for the real (larger) capacity.

    Also clips power_kw to >= 0: the "gb" model is a gradient-boosted regression tree predicting
    power_kw as a raw numeric target, with no physical non-negativity constraint enforced by the
    library. It regularly regresses slightly below zero during low-irradiance transitions (dawn
    ramp-up, brief heavy-cloud dips) -- observed ~4% of blocks in testing, up to about -7 kW at
    120 kWp scale. That's a fitting artifact of the underlying model, not a real "the panel drew
    power from the grid" event, so it's clipped here rather than passed through to soc.compute_soc
    (which would otherwise book it as extra, physically-meaningless usage).
    """
    _patch_requests_cache()
    from quartz_solar_forecast.forecast import run_forecast

    site = _build_site(latitude, longitude, capacity_kwp)
    pred_df = run_forecast(site=site, ts=ts, nwp_source=nwp_source, model="gb")
    pred_df["power_kw"] = pred_df["power_kw"].clip(lower=0.0)
    return pred_df.iloc[:BLOCKS_PER_DAY]


def _default_start(start: datetime | pd.Timestamp | str | None) -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize() if start is None else pd.Timestamp(start)


def forecast_one_day_kw(latitude: float | None = None, longitude: float | None = None,
                         capacity_kwp: float | None = None, start: datetime | pd.Timestamp | None = None,
                         nwp_source: str = "icon") -> ForecastResult:
    """PV(t) forecast for exactly 1 day at 15-minute resolution (96 blocks) -- a single
    forecast_day_kw call, 1 HTTP request instead of forecast_week_kw's 7. The fast default: a UI
    that auto-fetches on every load should reach for this first, and only pay for a full week
    (forecast_week_kw) when a caller explicitly asks to see more than one day."""
    start = _default_start(start)
    day_df = forecast_day_kw(latitude, longitude, capacity_kwp, start, nwp_source=nwp_source)
    power_kw = day_df["power_kw"]
    energy_wh = power_kw * 1000.0 * (BLOCK_MINUTES / 60.0)
    return ForecastResult(power_kw=power_kw, energy_wh=energy_wh)


def forecast_week_kw(latitude: float | None = None, longitude: float | None = None,
                      capacity_kwp: float | None = None, start: datetime | pd.Timestamp | None = None,
                      nwp_source: str = "icon") -> ForecastResult:
    """PV(t) forecast for 7 days at 15-minute resolution (672 blocks), built from
    7 daily-anchored calls to quartz-solar-forecast's run_forecast (see module docstring
    for why one call can't cover a week directly)."""
    start = _default_start(start)

    days = [forecast_day_kw(latitude, longitude, capacity_kwp, start + pd.Timedelta(days=d),
                             nwp_source=nwp_source) for d in range(7)]
    week_df = pd.concat(days)

    power_kw = week_df["power_kw"]
    energy_wh = power_kw * 1000.0 * (BLOCK_MINUTES / 60.0)
    return ForecastResult(power_kw=power_kw, energy_wh=energy_wh)


def _plot(result: ForecastResult, out_path: str) -> None:
    import matplotlib.pyplot as plt

    daily_kwh = result.energy_wh.resample("1D").sum() / 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(result.power_kw.index, result.power_kw, color="tab:orange")
    axes[0].set_ylabel("PV power [kW]")
    axes[0].set_title(f"PV(t) forecast -- lat={config.SITE.latitude}, lon={config.SITE.longitude}, "
                       f"{config.PV.rated_kwp} kWp (quartz-solar-forecast, 15-min steps)")

    axes[1].bar(daily_kwh.index.date.astype(str), daily_kwh, color="tab:orange")
    axes[1].set_ylabel("daily PV energy [kWh]")
    axes[1].set_title("Daily totals")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Real weekly PV forecast summary (and optional plot), no sim dependency.")
    ap.add_argument("--plot", action="store_true", help="also write a PNG of PV(t) and daily totals")
    ap.add_argument("--out", default="grid_energy/out_quartz_forecast.png")
    args = ap.parse_args()

    result = forecast_week_kw()
    print(f"site: lat={config.SITE.latitude}, lon={config.SITE.longitude}, "
          f"capacity={config.PV.rated_kwp} kWp")
    print(f"forecast window: {result.power_kw.index[0]} .. {result.power_kw.index[-1]} "
          f"({len(result.power_kw)} x 15-min blocks)")
    print(f"peak power           : {result.peak_power_kw:.2f} kW")
    print(f"total weekly energy  : {result.total_energy_wh:,.0f} Wh "
          f"({result.total_energy_wh / 1000:,.2f} kWh)")
    print(f"mean daily energy    : {result.total_energy_wh / 7:,.0f} Wh/day")

    if args.plot:
        _plot(result, args.out)
