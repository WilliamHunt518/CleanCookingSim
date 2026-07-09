"""Day-ahead tariff candidates, PV stub, and common-mean normalisation.

The PV stub feeds tariff design ONLY -- it must never be referenced from
agent utility code (sim.agent / sim.population).

The five forecast-driven candidates below (green_light, pv_following_real, soc_banded,
residual_load, deficit_guard) are the sim.tariffs "thin adapter" half of TARIFF_STRATEGIES.md's
design -- grid_energy.pricing has the pure price-curve math (KES/kWh, no sim imports); this module
bridges resolution (15-min forecast week -> 5-min sim day, section 0.3) and units (KES/kWh -> sim's
internal currency scale, section 0.4), then registers each in CANDIDATES like any other tariff.

Unlike flat/evening_peak/solar_following, these five need a real PV forecast (a live weather API
call, grid_energy.quartz_forecast) and, for all but pv_following_real, a day-ahead usage estimate
(section 0.7's POC: one reference sim.run.simulate_day under the flat tariff, decoupled from
whatever population an actual sweep uses). Both are fetched/computed at most once per process
(module-level cache below) no matter how many of the five are swept together or how many times
build_tariff is called -- calling any of them requires quartz-solar-forecast installed and internet
access; a failure surfaces as a normal Python exception, left for the caller (CLI/app.py) to
report, not swallowed here.
"""
from __future__ import annotations

import os

import numpy as np

from grid_energy import GridEnergyComponent
from grid_energy import config as grid_config
from grid_energy import pricing
from grid_energy.quartz_forecast import ForecastResult
from grid_energy.resample import resample_kw
from grid_energy.soc import SOCResult
from sim import config

T = config.STATE.T
BLOCK_HOURS = config.STATE.block_minutes / 60.0
FORECAST_BLOCK_MINUTES = 15.0
BLOCKS_PER_FORECAST_DAY = 96                # 24h / 15min
REFERENCE_DAY = 0                            # which day (0-6) of the fetched week each
                                              # forecast-driven tariff prices against


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
    every hour (Stage 1) -- meals/day should collapse to a small fraction of normal, not settle at
    a normal-looking number. It doesn't collapse all the way to 0: config.TIMING.DELTA_WOOD_FLOOR
    gives every agent a small, price-immune 'free firewood fallback' pathway, so an extreme price
    mostly redirects cooking to wood rather than suppressing it altogether (empirically: ~15% of
    flat's total events remain, ~98% of those wood -- see DELTA_WOOD_FLOOR's docstring)."""
    return np.full(T, config.TARIFF.p_bar * config.TARIFF.extreme_test_multiplier)


# ---------------------------------------------------------------------------
# Forecast-driven candidates (TARIFF_STRATEGIES.md strategies A-E)
# ---------------------------------------------------------------------------
_FORECAST_CACHE: ForecastResult | None = None
_REFERENCE_SOC_CACHE: SOCResult | None = None


def _grid_component() -> GridEnergyComponent:
    return GridEnergyComponent()  # Oloika defaults (grid_energy/config.py)


def _cached_forecast() -> ForecastResult:
    """This process's one live PV forecast fetch (a week, 7 HTTP calls) -- reused by every one of
    the five forecast-driven tariffs below and never re-fetched, no matter how many are called or
    how many times. Reset with reset_forecast_driven_cache() if a fresh forecast is wanted."""
    global _FORECAST_CACHE
    if _FORECAST_CACHE is None:
        _FORECAST_CACHE = _grid_component().forecast_pv_week()
    return _FORECAST_CACHE


def _cached_reference_soc() -> SOCResult:
    """Section 0.7's POC day-ahead usage assumption: one reference sim.run.simulate_day under the
    flat tariff (a fixed reference population/seed, decoupled from whatever population an actual
    sweep uses -- CANDIDATES entries are zero-argument, so there's no live population to thread
    through here), pushed through the same cached forecast week to get a week-length SOCResult.
    Computed at most once per process; every one of green_light/soc_banded/residual_load/
    deficit_guard slices REFERENCE_DAY out of this same SOCResult rather than each re-simulating."""
    global _REFERENCE_SOC_CACHE
    if _REFERENCE_SOC_CACHE is None:
        from sim import population as population_mod  # lazy: sim.run imports sim.tariffs
        from sim import run as run_mod                # (this module) -- avoid a circular import

        population = population_mod.build_population(
            np.random.default_rng(config.DEFAULT_SEED), n_agents=config.N_AGENTS)
        rng = np.random.default_rng(config.DEFAULT_SEED)
        day = run_mod.simulate_day(population, tariff_flat(), config.SCENARIOS["reference"], rng)

        component = _grid_component()
        _REFERENCE_SOC_CACHE = component.compute_soc_for_usage(
            day.demand_kw, usage_block_minutes=config.STATE.block_minutes, forecast=_cached_forecast())
    return _REFERENCE_SOC_CACHE


def reset_forecast_driven_cache() -> None:
    """Clear the cached forecast/reference-SOC so the next call to any of the five forecast-driven
    tariffs re-fetches/re-simulates -- mainly for tests, or a UI that wants a fresh forecast."""
    global _FORECAST_CACHE, _REFERENCE_SOC_CACHE
    _FORECAST_CACHE = None
    _REFERENCE_SOC_CACHE = None


def _slice_soc_day(soc: SOCResult, day: int, blocks_per_day: int = BLOCKS_PER_FORECAST_DAY) -> SOCResult:
    lo, hi = day * blocks_per_day, (day + 1) * blocks_per_day
    return SOCResult(
        t_hr=soc.t_hr[lo:hi], pv_kw=soc.pv_kw[lo:hi], usage_kw=soc.usage_kw[lo:hi],
        net_kw=soc.net_kw[lo:hi], energy_potential_kwh=soc.energy_potential_kwh[lo:hi],
        socs_pct=soc.socs_pct[lo:hi], energy_actual_kwh=soc.energy_actual_kwh[lo:hi],
        actual_soc_pct=soc.actual_soc_pct[lo:hi], surplus_kwh=soc.surplus_kwh[lo:hi],
        deficit_kwh=soc.deficit_kwh[lo:hi],
    )


def _slice_forecast_day(forecast: ForecastResult, day: int,
                         blocks_per_day: int = BLOCKS_PER_FORECAST_DAY) -> ForecastResult:
    lo, hi = day * blocks_per_day, (day + 1) * blocks_per_day
    return ForecastResult(power_kw=forecast.power_kw.iloc[lo:hi], energy_wh=forecast.energy_wh.iloc[lo:hi])


def _kes_day_to_sim_5min(price_kes_15min_day: np.ndarray) -> np.ndarray:
    """One day's 96 x 15-min KES/kWh prices -> T x 5-min sim-currency prices -- the resolution
    bridge (upsample x3, repeating each value) and unit bridge (divide by KES_PER_SIM_UNIT) from
    TARIFF_STRATEGIES.md sections 0.3/0.4, in that order."""
    price_kes_5min = resample_kw(price_kes_15min_day, from_block_minutes=FORECAST_BLOCK_MINUTES,
                                  to_block_minutes=config.STATE.block_minutes)
    return price_kes_5min / grid_config.PRICING.KES_PER_SIM_UNIT


def tariff_green_light() -> np.ndarray:
    """Strategy A -- see grid_energy.pricing.green_light and TARIFF_STRATEGIES.md."""
    soc_day = _slice_soc_day(_cached_reference_soc(), REFERENCE_DAY)
    return _kes_day_to_sim_5min(pricing.green_light(soc_day))


def tariff_pv_following_real() -> np.ndarray:
    """Strategy B -- see grid_energy.pricing.pv_following_real and TARIFF_STRATEGIES.md. The
    only one of the five that needs no usage forecast, just the cached PV forecast."""
    forecast_day = _slice_forecast_day(_cached_forecast(), REFERENCE_DAY)
    return _kes_day_to_sim_5min(pricing.pv_following_real(forecast_day))


def tariff_soc_banded() -> np.ndarray:
    """Strategy C -- see grid_energy.pricing.soc_banded and TARIFF_STRATEGIES.md."""
    soc_day = _slice_soc_day(_cached_reference_soc(), REFERENCE_DAY)
    return _kes_day_to_sim_5min(pricing.soc_banded(soc_day))


def tariff_residual_load() -> np.ndarray:
    """Strategy D -- see grid_energy.pricing.residual_load and TARIFF_STRATEGIES.md."""
    soc_day = _slice_soc_day(_cached_reference_soc(), REFERENCE_DAY)
    return _kes_day_to_sim_5min(pricing.residual_load(soc_day))


def tariff_deficit_guard() -> np.ndarray:
    """Strategy E -- see grid_energy.pricing.deficit_guard and TARIFF_STRATEGIES.md."""
    soc_day = _slice_soc_day(_cached_reference_soc(), REFERENCE_DAY)
    return _kes_day_to_sim_5min(pricing.deficit_guard(soc_day))


# ---------------------------------------------------------------------------
# ga_optimal (TARIFF_STRATEGIES.md section 7.7): the frozen output of sim.ga's search, produced
# offline and loaded here -- not recomputed at simulation time.
# ---------------------------------------------------------------------------
GA_OPTIMAL_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "ga_optimal_tariff.json")


def save_ga_optimal(theta: np.ndarray, price_sim_5min: np.ndarray, path: str | None = None) -> str:
    """Freeze a GA winner (sim.ga.GAResult.best_theta + its already-built sim-currency price
    array) to a small JSON file so tariff_ga_optimal can load it without ever re-fetching a
    forecast or re-running the search -- exactly the "produced offline, not recomputed at sim
    time" artifact section 7.7 describes."""
    import json

    path = GA_OPTIMAL_PATH if path is None else path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"theta": np.asarray(theta, dtype=float).tolist(),
                    "price_sim_5min": np.asarray(price_sim_5min, dtype=float).tolist()}, f)
    return path


def tariff_ga_optimal() -> np.ndarray:
    """Loads the JSON written by save_ga_optimal. Raises FileNotFoundError with an actionable
    message if the GA hasn't been run/saved yet -- there is no sensible silent fallback (falling
    back to flat would make ga_optimal look like it always ties flat, hiding that the search
    never ran) rather than mysteriously erroring deep inside a Monte Carlo sweep."""
    import json

    if not os.path.exists(GA_OPTIMAL_PATH):
        raise FileNotFoundError(
            "No GA-optimised tariff saved yet -- run the GA search (app.py's 'Tariff optimizer' "
            "section, or sim.ga.run_ga(...) followed by sim.tariffs.save_ga_optimal(...)) before "
            "selecting ga_optimal.")
    with open(GA_OPTIMAL_PATH) as f:
        data = json.load(f)
    return np.asarray(data["price_sim_5min"], dtype=float)


# Forecast-driven candidates (and ga_optimal, which needs a saved GA result) need external state
# before they'll work -- kept out of CANDIDATES' non-extreme_test "default sweep" selection in
# app.py the same way extreme_test is, so a normal Run simulation click never silently depends on
# network access or an as-yet-unrun search.
FORECAST_DRIVEN_NAMES = {
    "green_light", "pv_following_real", "soc_banded", "residual_load", "deficit_guard", "ga_optimal",
}

CANDIDATES = {
    "flat": tariff_flat,
    "evening_peak": tariff_evening_peak,
    "solar_following": tariff_solar_following,
    "extreme_test": tariff_extreme_test,
    "green_light": tariff_green_light,
    "pv_following_real": tariff_pv_following_real,
    "soc_banded": tariff_soc_banded,
    "residual_load": tariff_residual_load,
    "deficit_guard": tariff_deficit_guard,
    "ga_optimal": tariff_ga_optimal,
}


def build_tariff(name: str) -> np.ndarray:
    return CANDIDATES[name]()


def all_tariffs() -> dict[str, np.ndarray]:
    return {name: fn() for name, fn in CANDIDATES.items()}
