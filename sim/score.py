"""Scoring: clean-cooking share, peakiness, scoreboard assembly.

    clean_cooking_share = fraction of all eaten meals (population, all runs) that were electric

Higher clean_cooking_share wins (more meals cooked clean, not over fire). Peakiness
(peak_kw / load_factor) is scored separately: these tariffs are explicitly meant to flatten the
village's demand curve, not just shift fuel, so it's tracked as its own metric rather than folded
into clean_cooking_share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sim import meals
from sim.run import TariffRunResult
from sim.population import Population

WOOD_IDX0 = {i for i, name in enumerate(meals.MEAL_NAMES) if meals.WOOD_MASK[i]}


def wood_share(result: TariffRunResult) -> float:
    """Fraction of all eaten meals that were fire-only. Kept as the underlying count -- see
    clean_cooking_share for the headline, positively-framed scoreboard metric."""
    events = result.events_all_runs
    if not events:
        return 0.0
    n_wood = sum(1 for e in events if e.meal_idx0 in WOOD_IDX0)
    return n_wood / len(events)


def clean_cooking_share(result: TariffRunResult) -> float:
    """Fraction of all eaten meals that were electric (clean), pooled across all runs -- the
    positive-framed complement of wood_share. Zero events (e.g. the extreme_test tariff, which
    suppresses cooking altogether) is treated as 0% clean, not 100%: a tariff that stops people
    cooking at all hasn't achieved clean cooking, it's achieved no cooking."""
    events = result.events_all_runs
    if not events:
        return 0.0
    n_wood = sum(1 for e in events if e.meal_idx0 in WOOD_IDX0)
    return 1.0 - (n_wood / len(events))


def peak_kw(result: TariffRunResult) -> float:
    """Mean, across runs, of each simulated day's peak aggregate demand (kW) -- how tall the
    village's demand spike gets under this tariff."""
    return float(np.mean([curve.max() for curve in result.demand_curves]))


def load_factor(result: TariffRunResult) -> float:
    """Mean, across runs, of each day's (average demand / peak demand) -- the standard
    grid-engineering measure of how flat a load curve is: 1.0 = perfectly flat all day, closer to
    0 = one sharp spike dominating an otherwise quiet day. Evening-peak-style tariffs are
    explicitly meant to flatten the village's demand curve (not just relocate which fuel meets
    it), so this is the metric that actually tests whether a tariff is doing that job. Days with
    zero demand throughout (e.g. under extreme_test) are excluded rather than counted as "flat"."""
    factors = [float(c.mean() / c.max()) for c in result.demand_curves if c.max() > 0]
    return float(np.mean(factors)) if factors else 0.0


def household_kwh_stats(result: TariffRunResult, population: Population) -> tuple[float, float]:
    """Mean/median daily kWh among household-persona agents, pooled across all runs."""
    household_mask = population.persona_idx == 0
    values = np.concatenate([kwh[household_mask] for kwh in result.daily_kwh_per_run])
    return float(np.mean(values)), float(np.median(values))


def scoreboard(results: dict[str, TariffRunResult], population: Population) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        mean_kwh, median_kwh = household_kwh_stats(result, population)
        rows.append({
            "tariff": name,
            "clean_cooking_share": clean_cooking_share(result),
            "peak_kw": peak_kw(result),
            "load_factor": load_factor(result),
            "mean_daily_kwh_household": mean_kwh,
            "median_daily_kwh_household": median_kwh,
        })
    df = pd.DataFrame(rows).sort_values("clean_cooking_share", ascending=False).reset_index(drop=True)
    return df
