"""All matplotlib output, saved to ./out/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import agent, config, meals, score
from sim.population import Population, persona_gamma_vector, persona_gamma_cost, PERSONA_NAMES
from sim.run import TariffRunResult

BLOCK_HOURS = config.STATE.block_minutes / 60.0
T = config.STATE.T
T_HR = np.arange(T) * BLOCK_HOURS


def _ensure_out(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def plot_load_curves(results: dict[str, TariffRunResult], out_dir: str = "out") -> str:
    _ensure_out(out_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    for name, result in results.items():
        mean_curve = np.mean(np.stack(result.demand_curves), axis=0)
        ax.plot(T_HR, mean_curve, label=name)
        ax2.plot(T_HR, result.price, alpha=0.25, linestyle="--")

    ax.axhline(config.TARIFF.cap_kw, color="black", linestyle=":", label=f"cap ({config.TARIFF.cap_kw:g} kW)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("aggregate demand (kW)")
    ax2.set_ylabel("price (currency/kWh, faint dashed)")
    ax.set_title("Aggregate load curve by tariff (mean over runs)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "load_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_wood_share(results: dict[str, TariffRunResult], out_dir: str = "out") -> str:
    _ensure_out(out_dir)
    names = list(results.keys())
    shares = [score.wood_share(results[n]) for n in names]
    exceeds = [score.p_exceed(results[n]) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, shares, color="tab:brown")
    for bar, pe in zip(bars, exceeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"P_exceed={pe:.2f}", ha="center", va="bottom")
    ax.set_ylabel("wood share (fraction of all meals cooked)")
    ax.set_title("Wood share by tariff (annotated with cap-exceedance probability)")
    fig.tight_layout()
    path = os.path.join(out_dir, "wood_share.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_meal_timing(results: dict[str, TariffRunResult], population: Population,
                      out_dir: str = "out") -> str:
    _ensure_out(out_dir)
    all_events = [e for result in results.values() for e in result.events_all_runs]
    stages = ["breakfast", "lunch", "dinner"]

    fig, axes = plt.subplots(len(PERSONA_NAMES), len(stages), figsize=(12, 6), sharex=True)
    for row, persona in enumerate(PERSONA_NAMES):
        persona_idx = PERSONA_NAMES.index(persona)
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            starts_hr = [e.start_block * BLOCK_HOURS for e in all_events
                         if e.stage_idx == col and e.persona_idx == persona_idx]
            lo, hi = config.TIMING.stage_windows_hr[stage]
            ax.hist(starts_hr, bins=int((hi - lo) * 4), range=(lo, hi), color="tab:blue")
            if row == 0:
                ax.set_title(stage)
            if col == 0:
                ax.set_ylabel(persona)
            if row == len(PERSONA_NAMES) - 1:
                ax.set_xlabel("hour of day")
    fig.suptitle("Meal start-time distribution by stage and persona")
    fig.tight_layout()
    path = os.path.join(out_dir, "meal_timing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_utility_waterfall(t_hr: float = 19.0, price_t: float | None = None, persona: str = "household",
                            out_dir: str = "out") -> str:
    """Bar chart per meal, stacking the four utility terms, for one representative
    (mean, noise-free) agent at a chosen time and price -- the 'why did they choose
    wood at 7pm' demo slide."""
    _ensure_out(out_dir)
    if price_t is None:
        price_t = config.TARIFF.p_hi

    gamma = persona_gamma_vector(persona)
    gamma_cost = persona_gamma_cost(persona)
    scenario = config.SCENARIOS["reference"]

    appeal = gamma @ meals.Z.T                                   # (K,)
    eta_k = agent.eta_k_vector(scenario)                         # (K,)
    cost_term = -gamma_cost * price_t * meals.E_KWH               # (K,)
    hunger_val = 1.0  # representative moderate hunger for the demo slide
    hunger_term = meals.ALPHA_K * hunger_val                      # (K,)

    fig, ax = plt.subplots(figsize=(max(10, meals.K * 0.85), 7))
    x = np.arange(meals.K)
    bottom_pos = np.zeros(meals.K)
    bottom_neg = np.zeros(meals.K)
    colors = ["tab:blue", "tab:green", "tab:purple", "tab:pink"]
    terms = [("appeal (gamma . z)", appeal), ("scenario (eta_k)", eta_k),
             ("cost (-gamma_cost * price * e_k)", cost_term), ("hunger (alpha_k * hunger)", hunger_term)]
    for (label, values), color in zip(terms, colors):
        pos = np.clip(values, 0, None)
        neg = np.clip(values, None, 0)
        ax.bar(x, pos, bottom=bottom_pos, label=label, color=color)
        ax.bar(x, neg, bottom=bottom_neg, color=color)
        bottom_pos += pos
        bottom_neg += neg

    totals = appeal + eta_k + cost_term + hunger_term
    ax.scatter(x, totals, color="black", zorder=5, marker="D", label="total utility")
    for xi, tot in zip(x, totals):
        offset = 0.06 if tot >= 0 else -0.06
        va = "bottom" if tot >= 0 else "top"
        ax.text(xi, tot + offset, f"{tot:.2f}", ha="center", va=va, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(meals.MEAL_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("utility contribution")
    ax.set_title(f"Utility decomposition per meal -- t={t_hr:g}h, price={price_t:.2f}, persona={persona}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "utility_waterfall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all_plots(results: dict[str, TariffRunResult], population: Population, out_dir: str = "out") -> list[str]:
    paths = [
        plot_load_curves(results, out_dir),
        plot_wood_share(results, out_dir),
        plot_meal_timing(results, population, out_dir),
        plot_utility_waterfall(out_dir=out_dir),
    ]
    return paths
