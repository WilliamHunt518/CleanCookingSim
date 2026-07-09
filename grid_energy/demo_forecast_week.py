"""
Runnable demo: python -m grid_energy.demo_forecast_week [--seed N]

Pulls one real week of PV(t) at 15-minute resolution from quartz-solar-forecast
(see quartz_forecast.py) for the Oloika site (grid_energy.config.SITE and
.PV/.BATTERY), builds a matching week of Usage(t) by repeating one simulated
cooking day (sim.run.simulate_day, resampled from 5-min to 15-min blocks)
seven times, and runs soc.compute_soc over the full week with its default
reset_daily=True -- the real battery (actual_soc_pct) carries its charge
continuously across the whole week, it is never reset; only socs_pct (the
unbounded "how much surplus/deficit did today produce") restarts each day
from wherever the real battery actually is that morning, so the per-day
figures in the printed table read as that day's own balance, not a running
week-to-date total (see soc.py's docstring for why).

Repeating a single simulated day for all 7 days is a deliberate simplification
-- sim only simulates one Monte Carlo day at a time and has no day-to-day
variation (weekday/weekend, festivals, etc.) built in. The point here is to
demonstrate the real PV forecast driving the socs(t) equation end-to-end, not
to claim a calibrated weekly usage profile.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, quartz_forecast, soc as soc_mod


def weekly_usage_kw_from_sim(seed: int, n_agents: int) -> np.ndarray:
    """One simulated cooking day (5-min blocks) resampled to 15-min blocks by
    averaging each consecutive triple, then tiled 7x to fill a week."""
    from sim import config as sim_config, population as population_mod, run as run_mod, tariffs as tariffs_mod

    rng = np.random.default_rng(seed)
    population = population_mod.build_population(rng, n_agents=n_agents)
    price = tariffs_mod.build_tariff("flat")
    day = run_mod.simulate_day(population, price, sim_config.REFERENCE_SCENARIO, rng)

    demand_5min = day.demand_kw  # 288 blocks
    demand_15min = demand_5min.reshape(-1, 3).mean(axis=1)  # 96 blocks
    return np.tile(demand_15min, 7)  # 672 blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    # sim.config.N_AGENTS=100 defaults to ~287 kWh/day of cooking demand -- roughly 5-6x the
    # Oloika site's actual ~42-57 kWh/day total load. 18 agents brings sim's demand down to
    # roughly that real scale (~2.9 kWh/day/agent x 18 ~= 52 kWh/day), so PV vs usage in this
    # demo are at least comparable orders of magnitude for the same physical minigrid.
    ap.add_argument("--n-agents", type=int, default=18)
    ap.add_argument("--out", default="grid_energy/out_socs_week.png")
    args = ap.parse_args()

    forecast = quartz_forecast.forecast_week_kw()
    pv_kw = forecast.power_kw.to_numpy()
    usage_kw = weekly_usage_kw_from_sim(args.seed, args.n_agents)

    # reset_daily=True (compute_soc's default): socs_pct/surplus_kwh/deficit_kwh each snap back
    # to soc_init_pct at every day boundary, so they read as "this day's own balance", not a
    # running week-to-date total -- see soc.py's docstring.
    result = soc_mod.compute_soc(pv_kw, usage_kw, block_minutes=quartz_forecast.BLOCK_MINUTES)
    t_days = np.arange(len(pv_kw)) * (quartz_forecast.BLOCK_MINUTES / 60.0) / 24.0
    blocks_per_day = quartz_forecast.BLOCKS_PER_DAY
    dates = forecast.power_kw.index[::blocks_per_day].date

    print(f"forecast window       : {forecast.power_kw.index[0]} .. {forecast.power_kw.index[-1]}")
    print(f"{'day':<12}{'PV kWh':>10}{'usage kWh':>12}{'peak socs%':>12}{'min SOC%':>11}"
          f"{'surplus kWh':>13}{'deficit kWh':>13}")
    for d, date in enumerate(dates):
        sl = slice(d * blocks_per_day, (d + 1) * blocks_per_day)
        pv_day_kwh = pv_kw[sl].sum() * quartz_forecast.BLOCK_MINUTES / 60
        usage_day_kwh = usage_kw[sl].sum() * quartz_forecast.BLOCK_MINUTES / 60
        print(f"{str(date):<12}{pv_day_kwh:>10.2f}{usage_day_kwh:>12.2f}"
              f"{result.socs_pct[sl].max():>12.1f}{result.actual_soc_pct[sl].min():>11.1f}"
              f"{result.surplus_kwh[sl].sum():>13.2f}{result.deficit_kwh[sl].sum():>13.2f}")
    print(f"{'week total':<12}{pv_kw.sum() * quartz_forecast.BLOCK_MINUTES / 60:>10.2f}"
          f"{usage_kw.sum() * quartz_forecast.BLOCK_MINUTES / 60:>12.2f}{'':>12}{'':>11}"
          f"{result.surplus_kwh.sum():>13.2f}{result.deficit_kwh.sum():>13.2f}"
          f"   <- sum of each day's own surplus/deficit, not a running week-to-date balance")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t_days, result.pv_kw, label="PV(t) [kW] -- quartz-solar-forecast", color="tab:orange")
    axes[0].plot(t_days, result.usage_kw, label="Usage(t) [kW] -- sim day, tiled x7", color="tab:blue")
    axes[0].set_ylabel("kW")
    axes[0].legend()
    axes[0].set_title(f"PV forecast vs usage -- lat={config.SITE.latitude}, lon={config.SITE.longitude}, "
                       f"{config.PV.rated_kwp} kWp")

    axes[1].axhline(100, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for d in range(1, 7):
        axes[1].axvline(d, color="lightgray", linestyle=":", linewidth=0.8)
    axes[1].plot(t_days, result.socs_pct,
                 label="socs(t) -- today's surplus/deficit, resets daily to the real battery's level",
                 color="tab:green")
    axes[1].plot(t_days, result.actual_soc_pct,
                 label="actual SOC (clipped 0-100%) [%] -- the real battery, carries over day to day",
                 color="tab:red")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("% of battery capacity")
    axes[1].legend()
    axes[1].set_title("Grid state of charge -- battery carries over, only the surplus/deficit view resets daily")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
