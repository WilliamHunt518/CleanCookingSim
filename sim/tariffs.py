"""Day-ahead tariff candidates, PV stub, and common-mean normalisation.

The PV stub feeds tariff design ONLY -- it must never be referenced from
agent utility code (sim.agent / sim.population).
"""
from __future__ import annotations

import numpy as np

from sim import config

T = config.STATE.T
BLOCK_HOURS = config.STATE.block_minutes / 60.0


def _t_hours() -> np.ndarray:
    return np.arange(T) * BLOCK_HOURS


def pv_profile() -> np.ndarray:
    """PV(t) stub in kW, length T. Zero outside [t_rise, t_set]."""
    t = _t_hours()
    tr, ts = config.TARIFF.pv_t_rise_hr, config.TARIFF.pv_t_set_hr
    raw = np.sin(np.pi * (t - tr) / (ts - tr))
    raw = np.clip(raw, 0.0, None)
    daylight = (t >= tr) & (t <= ts)
    return config.TARIFF.pv_clearness * config.TARIFF.pv_p_max_kw * raw * daylight


def _normalise(price: np.ndarray) -> np.ndarray:
    p_bar = config.TARIFF.p_bar
    return price * (p_bar * T / np.sum(price))


def tariff_flat() -> np.ndarray:
    price = np.full(T, config.TARIFF.p_bar)
    return _normalise(price)


def tariff_evening_peak() -> np.ndarray:
    t = _t_hours()
    lo, hi = config.TARIFF.w_peak_hr
    in_peak = (t >= lo) & (t < hi)
    price = np.where(in_peak, config.TARIFF.p_hi, config.TARIFF.p_lo)
    return _normalise(price)


def tariff_solar_following() -> np.ndarray:
    pv = pv_profile()
    pv_max = np.max(pv) if np.max(pv) > 0 else 1.0
    price = config.TARIFF.p_hi - (config.TARIFF.p_hi - config.TARIFF.p_lo) * (pv / pv_max)
    return _normalise(price)


def tariff_extreme_test() -> np.ndarray:
    """Deliberately absurd flat price at extreme_test_multiplier x p_bar (default 5x), held all
    day -- a sanity-check tariff, not a realistic candidate. Every other tariff is normalised
    back to the same time-average p_bar (see _normalise) so only *shape* differs between them;
    this one is intentionally NOT renormalised, since the point is to test an elevated price
    *level*. If the model's price response is working, firing should be heavily suppressed at
    every hour (Stage 1), not just shifted -- meals/day should collapse toward 0, not settle at
    a normal-looking number. Because Stage 1 doesn't know which fuel an agent would pick until
    *after* it fires, an extreme enough price suppresses cooking altogether rather than diverting
    it to wood -- wood_share isn't a reliable saturation signal here the way meals/day is."""
    return np.full(T, config.TARIFF.p_bar * config.TARIFF.extreme_test_multiplier)


CANDIDATES = {
    "flat": tariff_flat,
    "evening_peak": tariff_evening_peak,
    "solar_following": tariff_solar_following,
    "extreme_test": tariff_extreme_test,
}


def build_tariff(name: str) -> np.ndarray:
    return CANDIDATES[name]()


def all_tariffs() -> dict[str, np.ndarray]:
    return {name: fn() for name, fn in CANDIDATES.items()}
