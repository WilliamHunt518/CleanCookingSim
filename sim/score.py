"""Scoring: wood share, exceedance probability, scoreboard assembly.

    score(tariff) = wood_share + PI * P_exceed
    wood_share    = fraction of all eaten meals (population, all runs) that were wood
    P_exceed      = fraction of Monte Carlo runs where any block's aggregate demand
                    exceeded the grid cap

Lower score wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sim import config, meals
from sim.run import TariffRunResult
from sim.population import Population

WOOD_IDX0 = {i for i, name in enumerate(meals.MEAL_NAMES) if meals.WOOD_MASK[i]}


def wood_share(result: TariffRunResult) -> float:
    events = result.events_all_runs
    if not events:
        return 0.0
    n_wood = sum(1 for e in events if e.meal_idx0 in WOOD_IDX0)
    return n_wood / len(events)


def p_exceed(result: TariffRunResult) -> float:
    if not result.exceed_flags:
        return 0.0
    return float(np.mean(result.exceed_flags))


def score_of(result: TariffRunResult) -> float:
    return wood_share(result) + config.SCORING.PI * p_exceed(result)


def household_kwh_stats(result: TariffRunResult, population: Population) -> tuple[float, float]:
    """Mean/median daily kWh among household-persona agents, pooled across all runs."""
    household_mask = population.persona_idx == 0
    values = np.concatenate([kwh[household_mask] for kwh in result.daily_kwh_per_run])
    return float(np.mean(values)), float(np.median(values))


def scoreboard(results: dict[str, TariffRunResult], population: Population) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        ws = wood_share(result)
        pe = p_exceed(result)
        mean_kwh, median_kwh = household_kwh_stats(result, population)
        rows.append({
            "tariff": name,
            "wood_share": ws,
            "P_exceed": pe,
            "score": ws + config.SCORING.PI * pe,
            "mean_daily_kwh_household": mean_kwh,
            "median_daily_kwh_household": median_kwh,
        })
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    return df
