"""Forecast-driven tariff price-curve builders (KES/kWh) -- pure functions over grid_energy's own
ForecastResult / SOCResult, no sim imports (sim -> grid_energy only, never the reverse).

Implements strategies A-E from TARIFF_STRATEGIES.md (see that file for the full design rationale,
formulas, and edge-case reasoning behind every strategy -- this module is its "section 0.2
grid_energy half": pure price-curve math only). The resolution bridge (day slicing, 15-min ->
5-min upsampling) and unit bridge (KES/kWh -> sim's internal currency units) both live in
sim.tariffs's adapters, not here -- every function below returns a price array the same length as
whatever pv_kw/usage_kw slice it's given (typically one day's 96 x 15-minute blocks, already
sliced out of a week-length SOCResult/ForecastResult by the caller).

Every builder's final step is a hard clamp to [P_MIN, P_MAX] regardless of strategy -- see
config.PricingConfig.
"""
from __future__ import annotations

import numpy as np

from . import config
from .quartz_forecast import ForecastResult
from .soc import SOCResult


def _clamp(price: np.ndarray, p_min: float | None, p_max: float | None) -> np.ndarray:
    p_min = config.PRICING.P_MIN if p_min is None else p_min
    p_max = config.PRICING.P_MAX if p_max is None else p_max
    return np.clip(price, p_min, p_max)


def _as_array(power_kw) -> np.ndarray:
    return power_kw.to_numpy() if hasattr(power_kw, "to_numpy") else np.asarray(power_kw, dtype=float)


def green_light(soc: SOCResult, p_disc: float | None = None, p_flat: float | None = None,
                 pad_blocks: int = 0) -> np.ndarray:
    """Strategy A: two-tier day-ahead surplus window. Discount (p_disc) whenever forecast PV
    would overflow a full battery (surplus_kwh > 0), optionally dilated by pad_blocks on each
    side; flat p_flat everywhere else. Self-adapts to an all-flat day when there's no surplus at
    all (e.g. a cloudy day) -- no discount is offered on a day with nothing to give away.
    Multiple disjoint surplus intervals in one day are all discounted, no contiguity required."""
    p_disc = config.PRICING.P_DISC if p_disc is None else p_disc
    p_flat = config.PRICING.P_FLAT if p_flat is None else p_flat

    window = soc.surplus_kwh > 0.0
    if pad_blocks > 0 and np.any(window):
        dilated = window.copy()
        for i in np.nonzero(window)[0]:
            lo, hi = max(0, i - pad_blocks), min(len(window), i + pad_blocks + 1)
            dilated[lo:hi] = True
        window = dilated
    return _clamp(np.where(window, p_disc, p_flat), None, None)


def pv_following_real(forecast: ForecastResult, p_lo: float | None = None, p_hi: float | None = None,
                       gamma: float = 1.0, p_ref: float | None = None) -> np.ndarray:
    """Strategy B: continuous, generation-indexed -- price(t) = p_hi - (p_hi-p_lo) *
    (pv(t)/p_ref)^gamma. p_ref defaults to this slice's own max forecast PV (guarantees it
    actually touches p_lo somewhere); pass an explicit p_ref (e.g. a week's 95th percentile
    computed by the caller) to guard against a single-block forecast spike inflating the
    reference and muting every other block's discount. gamma < 1 front-loads the discount
    (modest sun already cheap); gamma > 1 reserves it for true peak sun. A fully dark slice
    (p_ref <= 0) prices flat at p_hi throughout, including every night block regardless of
    p_ref's source."""
    p_lo = config.PRICING.P_MIN if p_lo is None else p_lo
    p_hi = config.PRICING.P_MAX if p_hi is None else p_hi

    pv = _as_array(forecast.power_kw)
    ref = float(np.max(pv)) if p_ref is None else float(p_ref)
    if ref <= 0.0:
        return _clamp(np.full(len(pv), p_hi), p_lo, p_hi)
    frac = np.clip(pv / ref, 0.0, 1.0) ** gamma
    price = p_hi - (p_hi - p_lo) * frac
    return _clamp(price, p_lo, p_hi)


def soc_banded(soc: SOCResult, theta_hi: float = 90.0, theta_lo: float = 20.0,
                p_cheap: float = 20.0, p_mid: float | None = None, p_dear: float = 70.0) -> np.ndarray:
    """Strategy C: three-band battery-state pricing, keyed off the physically-real, clipped
    actual_soc_pct (not the unbounded socs_pct -- a battery that's actually empty/full, not
    merely "would be" under an unclipped running balance). A battery that never crosses either
    threshold in this slice prices flat p_mid throughout."""
    p_mid = config.PRICING.P_FLAT if p_mid is None else p_mid

    s = soc.actual_soc_pct
    price = np.where(s >= theta_hi, p_cheap, np.where(s <= theta_lo, p_dear, p_mid))
    return _clamp(price, None, None)


def residual_load(soc: SOCResult, p_lo: float | None = None, p_hi: float | None = None) -> np.ndarray:
    """Strategy D: net-load marginal-cost proxy -- separates PV-covered usage (nearly free to
    serve) from usage that must be drawn from the battery (every kWh costs throughput).
    Residual r(t) = usage_kw(t) - pv_kw(t) (= -net_kw(t)), normalised by this slice's own max
    |r| to [-1, 1], mapped affinely: max surplus -> p_lo, max unmet residual -> p_hi, balanced
    -> the midpoint (lands exactly on P_FLAT with the [0, 80] defaults). A day with r(t)
    identically 0 throughout (theoretical, perfectly balanced every block) prices flat at the
    midpoint."""
    p_lo = config.PRICING.P_MIN if p_lo is None else p_lo
    p_hi = config.PRICING.P_MAX if p_hi is None else p_hi

    r = soc.usage_kw - soc.pv_kw
    max_abs_r = float(np.max(np.abs(r)))
    mid = (p_hi + p_lo) / 2.0
    if max_abs_r <= 0.0:
        return _clamp(np.full(len(r), mid), p_lo, p_hi)
    r_hat = r / max_abs_r
    price = mid + r_hat * (p_hi - p_lo) / 2.0
    return _clamp(price, p_lo, p_hi)


def deficit_guard(soc: SOCResult, p_flat: float | None = None, p_surge: float | None = None,
                   lead_blocks: int = 8) -> np.ndarray:
    """Strategy E: flat + targeted scarcity surcharge, the conservative baseline A-D should
    beat. Surcharges (p_surge) not just a forecast deficit block itself but the lead_blocks
    immediately before it too -- causality: a deficit at block t is caused by cumulative drain
    *before* t, so surcharging only the deficit block acts after the battery is already empty.
    lead_blocks is this strategy's real design variable. A slice with no forecast deficit at all
    prices flat p_flat throughout, indistinguishable from the flat tariff -- that's intended."""
    p_flat = config.PRICING.P_FLAT if p_flat is None else p_flat
    p_surge = config.PRICING.P_MAX if p_surge is None else p_surge

    deficit = soc.deficit_kwh > 0.0
    if not np.any(deficit):
        return _clamp(np.full(len(deficit), p_flat), None, None)
    window = deficit.copy()
    for i in np.nonzero(deficit)[0]:
        lo = max(0, i - lead_blocks)
        window[lo:i + 1] = True
    return _clamp(np.where(window, p_surge, p_flat), None, None)
