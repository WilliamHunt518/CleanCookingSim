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

PV_max_kwp = 60.0          # nameplate PV array capacity [kWp], passed to quartz-solar-forecast

@dataclass(frozen=True)
class PVConfig:
    rated_kwp: float = PV_max_kwp          # nameplate PV array capacity [kWp], passed to quartz-solar-forecast


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = PV_max_kwp * (25/54)  # BC [kWh] (Oloika upgraded lithium-ion bank)
    soc_init_pct: float = 0.0       # starting state of charge, percent of BC


@dataclass(frozen=True)
class TimeConfig:
    block_minutes: float = 5.0
    T: int = 288                     # 24h / 5min blocks


@dataclass(frozen=True)
class PricingConfig:
    """Shared parameters for the forecast-driven tariff strategies in pricing.py (see
    TARIFF_STRATEGIES.md sections 0.4/0.6). Prices here are real-world KES/kWh -- sim.tariffs'
    adapters convert to sim's internal currency units before registering a CANDIDATES entry."""
    P_MIN: float = 0.0                # KES/kWh, hard price floor every strategy clamps to
    P_MAX: float = 80.0               # KES/kWh, hard price ceiling every strategy clamps to
    P_FLAT: float = 40.0              # KES/kWh, current real flat rate (Oloika prepay), reference level
    P_DISC: float = 30.0              # KES/kWh, discounted tier (paper's actual Green Light Hours rate)
    # KES per sim-unit: sim's internal price scale is calibrated against p_bar=0.25 representing
    # the real KES 40 flat rate, so sim_price = kes_price / KES_PER_SIM_UNIT. KES [0, 80] then
    # maps to sim [0, 0.50] -- inside the calibrated [p_lo=0.05, p_hi=0.60] envelope. Load-bearing,
    # not cosmetic: feeding raw KES into sim would scale base_gamma_cost/kappa_price_time's
    # calibrated price sensitivity ~160x and destroy it (see TARIFF_STRATEGIES.md section 0.4).
    KES_PER_SIM_UNIT: float = 160.0


SITE = SiteConfig()
PV = PVConfig()
BATTERY = BatteryConfig()
TIME = TimeConfig()
PRICING = PricingConfig()
