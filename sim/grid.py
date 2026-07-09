"""sim -> grid_energy: push a tariff's simulated cooking demand through the real PV-forecast +
battery model, and score the result.

grid_energy never imports sim (see its README) -- this module is the other half of that
boundary, the one place sim reaches into grid_energy. A future ML search over tariff parameters
has exactly one thing to call: evaluate_tariff(tariff_name, ...) -> GridFitnessResult, whose
.fitness is a single scalar to optimise (higher is better), with every component metric it's
built from also exposed for interpretability/debugging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_energy import GridEnergyComponent
from grid_energy.quartz_forecast import ForecastResult
from grid_energy.soc import SOCResult
from sim import config
from sim.run import TariffRunResult, run_sweep


def demand_kw_for_tariff(result: TariffRunResult) -> np.ndarray:
    """One representative day's demand_kw for this tariff: the mean, block by block, across all
    Monte Carlo runs. Averaging (not picking one run) is deliberate -- a single run carries all of
    sim.agent's per-block noise (sigma_logit_noise etc.), so feeding the grid model one noisy
    sample would make battery/deficit numbers depend on which run happened to get sampled rather
    than on the tariff itself. The mean is exactly what score.py's peak_kw/load_factor already
    average over, so this stays consistent with the rest of the scoreboard."""
    if not result.demand_curves:
        return np.zeros(config.STATE.T)
    return np.mean(result.demand_curves, axis=0)


@dataclass
class GridFitnessResult:
    tariff_name: str
    soc: SOCResult                  # full week-long PV/usage/battery trace, native 15-min blocks
    demand_kw_typical_day: np.ndarray  # the sim demand_kw this was run with, before tiling to a week

    min_actual_soc_pct: float       # how close to empty the real, physical battery got, over the week
    end_actual_soc_pct: float       # real battery charge at the end of the week
    total_deficit_kwh: float        # total unmet demand over the week (battery empty, PV insufficient)
    total_surplus_kwh: float        # total PV beyond what the full battery could store (not penalised
                                     # -- see grid_energy's README, this could power e-cooking directly)
    total_usage_kwh: float
    total_pv_kwh: float

    battery_preserved: float        # min_actual_soc_pct / 100, in [0, 1] -- 1 = never dipped near empty
    demand_met: float                # 1 - total_deficit_kwh / total_usage_kwh, in [0, 1] -- 1 = no deficit
    fitness: float                  # battery_weight*battery_preserved + demand_weight*demand_met


def score_grid_fitness(tariff_name: str, soc: SOCResult, demand_kw_typical_day: np.ndarray,
                        battery_weight: float = 0.5, demand_weight: float = 0.5) -> GridFitnessResult:
    """Turn one SOCResult into the two things the spec asked for -- "is the battery preserved" and
    "do we generally fit within our usage" -- plus a single weighted-average fitness scalar for a
    later ML search to optimise. Both weights are exposed (rather than hard-coded 0.5/0.5) so that
    search can also explore trading one off against the other, not just which tariff scores best
    under one fixed weighting.

    battery_preserved uses min(actual_soc_pct) over the whole week, not just whether a deficit
    happened at all: a battery that grazes 1% charge and recovers is a much closer call than one
    that stays at 40% throughout, even if neither ever actually ran out. demand_met is linear in
    the *amount* of unmet energy (kWh), not the block-count of deficits, for the same reason --
    one large shortfall and many tiny ones should score differently.
    """
    total_deficit_kwh = float(soc.deficit_kwh.sum())
    total_surplus_kwh = float(soc.surplus_kwh.sum())
    total_usage_kwh = float(soc.usage_kw.sum() * (soc.t_hr[1] - soc.t_hr[0]))
    total_pv_kwh = float(soc.pv_kw.sum() * (soc.t_hr[1] - soc.t_hr[0]))

    battery_preserved = float(np.clip(soc.actual_soc_pct.min() / 100.0, 0.0, 1.0))
    demand_met = float(np.clip(1.0 - total_deficit_kwh / total_usage_kwh, 0.0, 1.0)) if total_usage_kwh > 0 else 1.0
    fitness = battery_weight * battery_preserved + demand_weight * demand_met

    return GridFitnessResult(
        tariff_name=tariff_name, soc=soc, demand_kw_typical_day=demand_kw_typical_day,
        min_actual_soc_pct=float(soc.actual_soc_pct.min()), end_actual_soc_pct=float(soc.actual_soc_pct[-1]),
        total_deficit_kwh=total_deficit_kwh, total_surplus_kwh=total_surplus_kwh,
        total_usage_kwh=total_usage_kwh, total_pv_kwh=total_pv_kwh,
        battery_preserved=battery_preserved, demand_met=demand_met, fitness=fitness,
    )


def run_grid_for_tariff_result(result: TariffRunResult, *, component: GridEnergyComponent | None = None,
                                forecast: ForecastResult | None = None, reset_daily: bool = True,
                                battery_weight: float = 0.5, demand_weight: float = 0.5) -> GridFitnessResult:
    """Push an already-simulated TariffRunResult's typical-day demand through the grid model.

    Pass a pre-fetched `forecast` (GridEnergyComponent.forecast_pv_week()) to score several
    tariffs against the same week's PV without re-fetching it each time -- PV doesn't depend on
    the tariff, only usage_kw does, so this is the efficient path for comparing tariffs (see
    evaluate_all_tariffs below).
    """
    component = GridEnergyComponent() if component is None else component
    demand_kw = demand_kw_for_tariff(result)
    soc = component.compute_soc_for_usage(demand_kw, usage_block_minutes=config.STATE.block_minutes,
                                            forecast=forecast, reset_daily=reset_daily)
    return score_grid_fitness(result.tariff_name, soc, demand_kw,
                               battery_weight=battery_weight, demand_weight=demand_weight)


def evaluate_tariff(tariff_name: str, *, scenario_name: str = "reference", seed: int | None = None,
                     R: int | None = None, n_agents: int | None = None,
                     component: GridEnergyComponent | None = None,
                     forecast: ForecastResult | None = None, reset_daily: bool = True,
                     battery_weight: float = 0.5, demand_weight: float = 0.5,
                     ) -> tuple[GridFitnessResult, TariffRunResult]:
    """The single entry point: set a tariff name, get a fitness back. Runs its own Monte Carlo
    sweep (sim.run.run_sweep) for just this one tariff, pushes the result through the grid model,
    and scores it -- everything a later ML search over tariff *parameters* needs to turn "try this
    tariff" into "here's how good it was," without the caller having to know sim.run or
    grid_energy exist. Returns the TariffRunResult too, since clean_cooking_share/peak_kw/
    load_factor (sim.score) are a separate, complementary set of scores over the same run.
    """
    results, _population = run_sweep([tariff_name], scenario_name=scenario_name, seed=seed, R=R,
                                      n_agents=n_agents)
    result = results[tariff_name]
    fitness = run_grid_for_tariff_result(result, component=component, forecast=forecast,
                                          reset_daily=reset_daily, battery_weight=battery_weight,
                                          demand_weight=demand_weight)
    return fitness, result
