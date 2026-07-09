"""
Parameters for the grid energy-balance model (PV forecast, usage, battery).

Defaults are anchored to the Oloika mini-grid (Kajiado West, Kenya) as
reported in "Mini-grid resilience through integration of e-cooking loads".
These are sensible starting points, not calibrated values -- override them
per-scenario. See README.md for the full assumption list.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteConfig:
    latitude: float = -0.80895       # Oloika, Kajiado West, Kenya
    longitude: float = 36.24232
    tilt_deg: float = 15.0           # near-equator site -- a low tilt suits a near-overhead sun
    orientation_deg: float = 180.0   # panel azimuth, 180 = south-facing (quartz-solar-forecast default)


@dataclass(frozen=True)
class PVConfig:
    rated_kwp: float = 120.0          # nameplate PV array capacity [kWp], passed to quartz-solar-forecast


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 30.0       # BC [kWh] (Oloika upgraded lithium-ion bank)
    soc_init_pct: float = 0.0       # starting state of charge, percent of BC


@dataclass(frozen=True)
class TimeConfig:
    block_minutes: float = 5.0
    T: int = 288                     # 24h / 5min blocks


SITE = SiteConfig()
PV = PVConfig()
BATTERY = BatteryConfig()
TIME = TimeConfig()
