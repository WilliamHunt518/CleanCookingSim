"""Day loop, Monte Carlo sweep, and demand assembly.

One simulated day = 288 blocks stepped through sim.agent's pure fire/which/update
functions. Demand assembly adds each cook's power profile (sim.meals) into an
aggregate kW timeline. Monte Carlo runs R independent days per tariff with a
population sampled once and shared across tariffs/runs (fresh dice, same agents).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim import agent, config, meals, tariffs as tariffs_mod
from sim import population as population_mod

BLOCK_HOURS = config.STATE.block_minutes / 60.0
T = config.STATE.T


@dataclass
class CookEvent:
    agent_idx: int
    persona_idx: int
    stage_idx: int
    meal_idx0: int
    start_block: int
    duration_blocks: int
    e_kwh: float


@dataclass
class DayResult:
    events: list[CookEvent]
    demand_kw: np.ndarray
    trace_rows: list[dict] = field(default_factory=list)
    agent_power: np.ndarray | None = None  # (n_agents, T) kW, only populated if requested


def _trace_row(idx: int, t_block: int, t_hr: float, state: agent.AgentState,
                fr: agent.FireResult, wh: agent.WhichResult, dur_by_agent: dict) -> dict:
    row = {
        "block": t_block, "t_hr": round(t_hr, 4),
        "h_B": int(state.h[idx, 0]), "h_L": int(state.h[idx, 1]), "h_D": int(state.h[idx, 2]),
        "tau": float(state.tau[idx]), "hunger": float(fr.hunger[idx]),
        "stage_idx": fr.stage_idx, "eligible": bool(fr.eligible[idx]),
        "w": float(fr.w), "eta_t": float(fr.eta_t), "lam": float(fr.lam[idx]),
        "alpha0_hunger": float(fr.alpha0_hunger[idx]), "q": float(fr.q[idx]),
        "fired": bool(fr.fired[idx]),
    }
    if fr.fired[idx]:
        choice = int(wh.choice[idx])
        row["choice_idx0"] = choice
        row["choice_name"] = meals.MEAL_NAMES[choice]
        row["duration_blocks"] = dur_by_agent.get(idx)
        for k, name in enumerate(meals.MEAL_NAMES):
            row[f"appeal_{name}"] = float(wh.appeal[idx, k])
            row[f"eta_k_{name}"] = float(wh.eta_k[k])
            row[f"cost_{name}"] = float(wh.cost_term[idx, k])
            row[f"hunger_term_{name}"] = float(wh.hunger_term[idx, k])
            row[f"prob_{name}"] = float(wh.probs[idx, k])
    return row


def simulate_day(population: population_mod.Population, price: np.ndarray, scenario,
                  rng: np.random.Generator, no_hunger: bool = False,
                  trace_agent: int | None = None, track_agent_power: bool = False) -> DayResult:
    state = agent.init_state(population.n_agents)
    demand = np.zeros(T)
    events: list[CookEvent] = []
    trace_rows: list[dict] = []
    agent_power = np.zeros((population.n_agents, T)) if track_agent_power else None

    for t_block in range(T):
        t_hr = t_block * BLOCK_HOURS
        fr = agent.fire(state, t_hr, population, scenario, rng, no_hunger=no_hunger)
        wh = agent.which(state, population, scenario, price[t_block], fr.fired, fr.hunger,
                          rng, no_hunger=no_hunger)

        fired_idx = np.nonzero(fr.fired)[0]
        dur_by_agent: dict = {}
        if fired_idx.size > 0:
            durations = meals.sample_durations_blocks(wh.choice[fired_idx], rng)
            for k, a_idx in enumerate(fired_idx):
                meal0 = int(wh.choice[a_idx])
                dur = int(durations[k])
                dur_by_agent[int(a_idx)] = dur
                profile = meals.power_profile(meal0, dur)
                end = min(T, t_block + dur)
                demand[t_block:end] += profile[: end - t_block]
                if agent_power is not None:
                    agent_power[a_idx, t_block:end] += profile[: end - t_block]
                events.append(CookEvent(
                    agent_idx=int(a_idx), persona_idx=int(population.persona_idx[a_idx]),
                    stage_idx=fr.stage_idx, meal_idx0=meal0, start_block=t_block,
                    duration_blocks=dur, e_kwh=float(meals.E_KWH[meal0]),
                ))

        if trace_agent is not None:
            trace_rows.append(_trace_row(trace_agent, t_block, t_hr, state, fr, wh, dur_by_agent))

        state = agent.update(state, fr, wh)

    return DayResult(events=events, demand_kw=demand, trace_rows=trace_rows, agent_power=agent_power)


@dataclass
class TariffRunResult:
    tariff_name: str
    price: np.ndarray
    events_all_runs: list[CookEvent]
    demand_curves: list[np.ndarray]
    exceed_flags: list[bool]
    trace_rows: list[dict]
    daily_kwh_per_run: list[np.ndarray] = field(default_factory=list)  # each (n_agents,), per-agent kWh that day


def run_sweep(tariff_names: list[str], scenario_name: str = "reference", seed: int | None = None,
              R: int | None = None, n_agents: int | None = None, no_hunger: bool = False,
              no_cost: bool = False, no_personas: bool = False,
              trace_agent: int | None = None) -> tuple[dict[str, TariffRunResult], population_mod.Population]:
    seed = config.DEFAULT_SEED if seed is None else seed
    R = config.SCORING.R if R is None else R
    n_agents = config.N_AGENTS if n_agents is None else n_agents
    scenario = config.SCENARIOS[scenario_name]

    ss = np.random.SeedSequence(seed)
    pop_seed, *tariff_seeds = ss.spawn(1 + len(tariff_names))
    pop_rng = np.random.default_rng(pop_seed)
    population = population_mod.build_population(pop_rng, n_agents=n_agents,
                                                    no_personas=no_personas, no_cost=no_cost)

    results: dict[str, TariffRunResult] = {}
    for tname, tseed in zip(tariff_names, tariff_seeds):
        price = tariffs_mod.build_tariff(tname)
        run_seeds = tseed.spawn(R)
        demand_curves, exceed_flags, all_events, trace_rows, daily_kwh_per_run = [], [], [], [], []
        for r, rseed in enumerate(run_seeds):
            rng = np.random.default_rng(rseed)
            day = simulate_day(population, price, scenario, rng, no_hunger=no_hunger,
                                trace_agent=trace_agent if r == 0 else None)
            demand_curves.append(day.demand_kw)
            exceed_flags.append(bool(np.any(day.demand_kw > config.TARIFF.cap_kw)))
            all_events.extend(day.events)
            kwh = np.zeros(population.n_agents)
            for e in day.events:
                kwh[e.agent_idx] += e.e_kwh
            daily_kwh_per_run.append(kwh)
            if r == 0:
                trace_rows = day.trace_rows
        results[tname] = TariffRunResult(tariff_name=tname, price=price, events_all_runs=all_events,
                                          demand_curves=demand_curves, exceed_flags=exceed_flags,
                                          trace_rows=trace_rows, daily_kwh_per_run=daily_kwh_per_run)
    return results, population
