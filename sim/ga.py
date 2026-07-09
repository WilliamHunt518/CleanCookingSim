"""Genetic-algorithm search over forecast-driven tariff shapes -- TARIFF_STRATEGIES.md section 7.

Chromosome encoding (parametric, option (a) of section 7.2): 7 genes -- a non-negative blend
weight for each of the five grid_energy.pricing strategies, plus two of their own shape
parameters (pv_following_real's gamma, soc_banded's theta_hi). The five strategies' own KES/kWh
day price curves are combined as a weighted average; all-zero weights falls back to a flat
P_FLAT curve. This makes every pure strategy (weight 1 on one, 0 on the rest) and flat (all
weights 0) a literal corner of the search space -- exactly the "seeded with A-E's curves +
flat-40" population section 7.5 asks for, which is what guarantees the GA's result is never worse
than the best single heuristic by construction.

Fitness (section 7.3, minimised) reuses sim.grid/sim.score exactly as a normal tariff evaluation
would -- this module's only real job is turning one chromosome into a price array and scoring it,
then evolving the population. Common random numbers (section 7.4): every candidate across the
*entire* run (not just one generation) is evaluated against the same fixed set of R day-seeds and
the same population, so fitness is a deterministic function of theta -- differences reflect price
differences, not Monte Carlo luck, and there's no need to estimate a noise floor from replicate
evaluations (there isn't any noise to estimate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from grid_energy import GridEnergyComponent, config as grid_config, pricing
from grid_energy.quartz_forecast import ForecastResult
from grid_energy.soc import SOCResult
from sim import config, grid as grid_mod, population as population_mod, run as run_mod, score, tariffs

GENE_NAMES = ["w_green_light", "w_pv_following_real", "w_soc_banded", "w_residual_load",
              "w_deficit_guard", "gamma", "theta_hi"]
N_GENES = len(GENE_NAMES)

# (lo, hi) bounds per gene, same order as GENE_NAMES.
GENE_BOUNDS = np.array([
    [0.0, 1.0],    # w_green_light
    [0.0, 1.0],    # w_pv_following_real
    [0.0, 1.0],    # w_soc_banded
    [0.0, 1.0],    # w_residual_load
    [0.0, 1.0],    # w_deficit_guard
    [0.3, 3.0],    # gamma (pv_following_real's response exponent)
    [70.0, 99.0],  # theta_hi (soc_banded's cheap-band SOC threshold, %)
])

# Population seeds guaranteeing GA >= best heuristic by construction (section 7.5): one pure
# corner per strategy, plus flat (all weights 0). gamma/theta_hi default to each strategy's own
# TARIFF_STRATEGIES.md default so a pure-strategy seed reproduces that strategy exactly.
SEED_CHROMOSOMES: dict[str, np.ndarray] = {
    "flat":               np.array([0., 0., 0., 0., 0., 1.0, 90.0]),
    "green_light":        np.array([1., 0., 0., 0., 0., 1.0, 90.0]),
    "pv_following_real":  np.array([0., 1., 0., 0., 0., 1.0, 90.0]),
    "soc_banded":         np.array([0., 0., 1., 0., 0., 1.0, 90.0]),
    "residual_load":      np.array([0., 0., 0., 1., 0., 1.0, 90.0]),
    "deficit_guard":      np.array([0., 0., 0., 0., 1., 1.0, 90.0]),
}


def clamp(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1])


def price_curve_kes_day(theta: np.ndarray, forecast_day: ForecastResult, soc_day: SOCResult) -> np.ndarray:
    """One day's 96 x 15-min KES/kWh price curve for chromosome theta: a weighted blend of the
    five pricing.py strategies (all-zero weights -> flat P_FLAT, see module docstring)."""
    weights = np.asarray(theta[:5], dtype=float)
    gamma, theta_hi = float(theta[5]), float(theta[6])
    total_w = float(weights.sum())
    if total_w <= 0.0:
        return np.full(len(soc_day.pv_kw), grid_config.PRICING.P_FLAT)

    curves = np.stack([
        pricing.green_light(soc_day),
        pricing.pv_following_real(forecast_day, gamma=gamma),
        pricing.soc_banded(soc_day, theta_hi=theta_hi),
        pricing.residual_load(soc_day),
        pricing.deficit_guard(soc_day),
    ])
    return (weights[:, None] * curves).sum(axis=0) / total_w


def price_curve_sim_5min(theta: np.ndarray, forecast_day: ForecastResult, soc_day: SOCResult) -> np.ndarray:
    """price_curve_kes_day, resolution- and unit-bridged to sim's native 5-min/T=288 currency
    array -- reuses sim.tariffs' own bridge so the GA's price arrays are built identically to
    every other forecast-driven tariff."""
    return tariffs._kes_day_to_sim_5min(price_curve_kes_day(theta, forecast_day, soc_day))


def describe_chromosome(theta: np.ndarray) -> str:
    weights = np.asarray(theta[:5], dtype=float)
    total_w = float(weights.sum())
    names = ["green_light", "pv_following_real", "soc_banded", "residual_load", "deficit_guard"]
    if total_w <= 0.0:
        return "flat (all blend weights ~0)"
    parts = [f"{100 * w / total_w:.0f}% {name}" for w, name in zip(weights, names) if w / total_w > 0.03]
    parts.sort(key=lambda s: -float(s.split("%")[0]))
    return ", ".join(parts) + f" (gamma={theta[5]:.2f}, theta_hi={theta[6]:.0f})"


@dataclass
class FitnessBreakdown:
    fitness: float
    total_deficit_kwh: float
    total_surplus_kwh: float
    wood_share: float
    mean_paid_price_kes: float


def evaluate(theta: np.ndarray, *, population, seeds, scenario, forecast_day: ForecastResult,
             soc_day: SOCResult, grid_component: GridEnergyComponent, forecast_full: ForecastResult,
             w_def: float = 1.0, w_sur: float = 0.1, w_wood: float = 50.0, w_rev: float = 0.0,
             ) -> FitnessBreakdown:
    """One chromosome -> one fitness (section 7.3), built entirely from sim.grid/sim.score's
    normal tariff-evaluation machinery -- price the chromosome, run R Monte Carlo sim days under
    it (the same `seeds` for every candidate, see module docstring on CRN), push the resulting
    typical-day demand through the grid model against the one pre-fetched `forecast_full`, and
    combine reliability (deficit), waste (surplus), and adoption (wood_share) into a single
    minimise-me score. w_wood defaults far larger than w_def/w_sur because wood_share is a
    fraction in [0, 1] while deficit/surplus are kWh magnitudes typically in the tens -- the
    weights, not the raw terms, are what make the three objectives comparable."""
    price_sim = price_curve_sim_5min(theta, forecast_day, soc_day)

    demand_curves, all_events = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        day = run_mod.simulate_day(population, price_sim, scenario, rng)
        demand_curves.append(day.demand_kw)
        all_events.extend(day.events)
    result = run_mod.TariffRunResult(tariff_name="ga_candidate", price=price_sim,
                                      events_all_runs=all_events, demand_curves=demand_curves, trace_rows=[])

    grid_fit = grid_mod.run_grid_for_tariff_result(result, component=grid_component, forecast=forecast_full)
    wood = score.wood_share(result)

    demand_typical = grid_mod.demand_kw_for_tariff(result)
    total_usage_kwh = float(demand_typical.sum() * tariffs.BLOCK_HOURS)
    if total_usage_kwh > 0.0:
        mean_paid_sim = float((price_sim * demand_typical).sum() * tariffs.BLOCK_HOURS / total_usage_kwh)
    else:
        mean_paid_sim = config.TARIFF.p_bar
    mean_paid_price_kes = mean_paid_sim * grid_config.PRICING.KES_PER_SIM_UNIT

    fitness = (w_def * grid_fit.total_deficit_kwh + w_sur * grid_fit.total_surplus_kwh
               + w_wood * wood + w_rev * abs(mean_paid_price_kes - grid_config.PRICING.P_FLAT))

    return FitnessBreakdown(fitness=fitness, total_deficit_kwh=grid_fit.total_deficit_kwh,
                             total_surplus_kwh=grid_fit.total_surplus_kwh, wood_share=wood,
                             mean_paid_price_kes=mean_paid_price_kes)


@dataclass
class Individual:
    theta: np.ndarray
    fitness: float | None = None
    breakdown: FitnessBreakdown | None = None


@dataclass
class GAResult:
    best_theta: np.ndarray
    best_fitness: float
    best_breakdown: FitnessBreakdown
    history: list[dict] = field(default_factory=list)   # per-generation best/mean/worst, for a live chart
    final_population: list[Individual] = field(default_factory=list)
    n_generations_run: int = 0


def _random_individual(rng: np.random.Generator) -> np.ndarray:
    lo, hi = GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]
    return lo + rng.random(N_GENES) * (hi - lo)


def _seed_population(pop_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    individuals = [s.copy() for s in SEED_CHROMOSOMES.values()]
    while len(individuals) < pop_size:
        individuals.append(_random_individual(rng))
    return individuals[:pop_size]


def _tournament_select(pop: list[Individual], rng: np.random.Generator, k: int = 3) -> Individual:
    idx = rng.integers(0, len(pop), size=k)
    return min((pop[i] for i in idx), key=lambda ind: ind.fitness)


def _blend_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, alpha: float = 0.5) -> np.ndarray:
    lo = np.minimum(a, b) - alpha * np.abs(a - b)
    hi = np.maximum(a, b) + alpha * np.abs(a - b)
    return clamp(lo + rng.random(N_GENES) * (hi - lo))


def _mutate(theta: np.ndarray, rng: np.random.Generator, sigma_frac: float, gene_prob: float) -> np.ndarray:
    theta = theta.copy()
    ranges = GENE_BOUNDS[:, 1] - GENE_BOUNDS[:, 0]
    hits = rng.random(N_GENES) < gene_prob
    theta[hits] += rng.normal(0.0, sigma_frac * ranges[hits])
    return clamp(theta)


def run_ga(*, pop_size: int = 24, n_generations: int = 20, R: int = 10, n_agents: int | None = None,
           scenario_name: str = "reference", seed: int = 0, reference_day: int = 0,
           w_def: float = 1.0, w_sur: float = 0.1, w_wood: float = 50.0, w_rev: float = 0.0,
           n_elite: int = 2, tournament_k: int = 3, crossover_prob: float = 0.8,
           mutation_gene_prob: float | None = None, mutation_sigma_frac: float = 0.10,
           patience: int = 5, convergence_epsilon: float = 1e-3,
           forecast: ForecastResult | None = None, reference_soc: SOCResult | None = None,
           grid_component: GridEnergyComponent | None = None,
           generation_callback: Callable[[dict], None] | None = None) -> GAResult:
    """The GA loop (section 7.5/7.6). `forecast`/`reference_soc` default to sim.tariffs' own
    process-wide cache (tariffs._cached_forecast / _cached_reference_soc) -- the one live PV
    forecast fetch this whole search needs, matching every other forecast-driven tariff (pass
    them explicitly, e.g. a synthetic ForecastResult/SOCResult, to run entirely offline in tests).

    Stops at n_generations (the hard cap, section 7.6's ~40) or once the best fitness hasn't
    improved by more than convergence_epsilon for `patience` consecutive generations, whichever
    comes first -- no noise-floor estimation needed since CRN (module docstring) makes fitness a
    deterministic function of theta.
    """
    n_agents = config.N_AGENTS if n_agents is None else n_agents
    mutation_gene_prob = 1.0 / N_GENES if mutation_gene_prob is None else mutation_gene_prob
    scenario = config.SCENARIOS[scenario_name]
    grid_component = GridEnergyComponent() if grid_component is None else grid_component

    forecast = tariffs._cached_forecast() if forecast is None else forecast
    reference_soc = tariffs._cached_reference_soc() if reference_soc is None else reference_soc
    forecast_day = tariffs._slice_forecast_day(forecast, reference_day)
    soc_day = tariffs._slice_soc_day(reference_soc, reference_day)

    ss = np.random.SeedSequence(seed)
    pop_seed, day_seeds_seed, ga_op_seed = ss.spawn(3)
    population = population_mod.build_population(np.random.default_rng(pop_seed), n_agents=n_agents)
    day_seeds = day_seeds_seed.spawn(R)  # fixed for the whole run -- CRN, see module docstring
    rng_ga = np.random.default_rng(ga_op_seed)  # GA operators only, never touches sim RNG

    def _evaluate(theta: np.ndarray) -> FitnessBreakdown:
        return evaluate(theta, population=population, seeds=day_seeds, scenario=scenario,
                         forecast_day=forecast_day, soc_day=soc_day, grid_component=grid_component,
                         forecast_full=forecast, w_def=w_def, w_sur=w_sur, w_wood=w_wood, w_rev=w_rev)

    genomes = _seed_population(pop_size, rng_ga)
    pop_inds = [Individual(theta=g) for g in genomes]

    history: list[dict] = []
    best_ever: Individual | None = None
    stale_generations = 0
    n_generations_run = 0

    for gen in range(n_generations):
        for ind in pop_inds:
            if ind.fitness is None:
                ind.breakdown = _evaluate(ind.theta)
                ind.fitness = ind.breakdown.fitness

        pop_inds.sort(key=lambda ind: ind.fitness)
        gen_best = pop_inds[0]
        history.append({
            "generation": gen, "best": gen_best.fitness,
            "mean": float(np.mean([i.fitness for i in pop_inds])), "worst": pop_inds[-1].fitness,
        })
        n_generations_run = gen + 1

        if generation_callback is not None:
            generation_callback({"generation": gen, "n_generations": n_generations,
                                  "best_fitness": gen_best.fitness, "best_theta": gen_best.theta.copy(),
                                  "mean_fitness": history[-1]["mean"]})

        if best_ever is None or gen_best.fitness < best_ever.fitness - convergence_epsilon:
            best_ever = Individual(theta=gen_best.theta.copy(), fitness=gen_best.fitness,
                                    breakdown=gen_best.breakdown)
            stale_generations = 0
        else:
            stale_generations += 1
        if stale_generations >= patience:
            break
        if gen == n_generations - 1:
            break  # last generation evaluated -- no need to breed a generation that never runs

        next_genomes = [ind.theta.copy() for ind in pop_inds[:n_elite]]
        sigma_frac = mutation_sigma_frac * max(1.0 - gen / max(n_generations, 1), 0.2)  # anneal, floor at 20%
        while len(next_genomes) < pop_size:
            parent_a = _tournament_select(pop_inds, rng_ga, k=tournament_k)
            parent_b = _tournament_select(pop_inds, rng_ga, k=tournament_k)
            child = (_blend_crossover(parent_a.theta, parent_b.theta, rng_ga)
                     if rng_ga.random() < crossover_prob else parent_a.theta.copy())
            child = _mutate(child, rng_ga, sigma_frac=sigma_frac, gene_prob=mutation_gene_prob)
            next_genomes.append(child)

        new_inds = []
        for i, g in enumerate(next_genomes):
            if i < n_elite:  # elites keep their already-computed fitness, unchanged genome
                new_inds.append(Individual(theta=g, fitness=pop_inds[i].fitness, breakdown=pop_inds[i].breakdown))
            else:
                new_inds.append(Individual(theta=g))
        pop_inds = new_inds

    return GAResult(best_theta=best_ever.theta, best_fitness=best_ever.fitness,
                     best_breakdown=best_ever.breakdown, history=history,
                     final_population=pop_inds, n_generations_run=n_generations_run)
