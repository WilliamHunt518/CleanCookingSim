"""Per-block agent dynamics: state, fire (hazard), which (softmax choice), update.

Pure functions, vectorised across the whole population (arrays of shape (N,...)).
No side effects, no I/O -- sim.run drives these block-by-block and handles
demand assembly / event logging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim import config, meals
from sim.utils import sigmoid, softmax

STAGE_ORDER = ["breakfast", "lunch", "dinner"]
BLOCK_HOURS = config.STATE.block_minutes / 60.0


@dataclass
class AgentState:
    h: np.ndarray    # (N, 3) int, 0 = not yet eaten, else 1-based meal index
    tau: np.ndarray  # (N,) float, hours since last meal


def init_state(n_agents: int) -> AgentState:
    return AgentState(h=np.zeros((n_agents, 3), dtype=int), tau=np.zeros(n_agents, dtype=float))


def n_eaten(h: np.ndarray) -> np.ndarray:
    return np.sum(h > 0, axis=1)


def hunger_of(h: np.ndarray, tau: np.ndarray, t_hr: float) -> np.ndarray:
    nb = config.nbar(t_hr)
    n = n_eaten(h)
    return np.maximum(0, nb - n) + config.HUNGER.kappa * tau


def active_stage(t_hr: float) -> int:
    """Index into STAGE_ORDER of the stage whose window contains t_hr, or -1 if none (overnight gap)."""
    for i, stage in enumerate(STAGE_ORDER):
        lo, hi = config.TIMING.stage_windows_hr[stage]
        if lo <= t_hr < hi:
            return i
    return -1


def w_of_t(t_hr: float) -> float:
    """Base hazard logit: relaxes to overnight_base_logit away from every meal-time bump,
    and reaches bump_heights[stage] exactly at that stage's centre."""
    base = config.TIMING.overnight_base_logit
    total = base
    for stage in STAGE_ORDER:
        centre = config.TIMING.bump_centers_hr[stage]
        height = config.TIMING.bump_heights[stage]
        width = config.TIMING.bump_width_hr
        z = (t_hr - centre) / width
        total += (height - base) * np.exp(-0.5 * z * z)
    return total


def eta_t_of_t(t_hr: float, scenario) -> float:
    """Scenario timing offset, additive on top of w(t). Zero for the reference scenario."""
    eta = 0.0
    for stage, params in scenario.eta_t_hr_offsets.items():
        centre = config.TIMING.bump_centers_hr[stage] + params["centre_shift_hr"]
        z = (t_hr - centre) / params["width_hr"]
        eta += params["height"] * np.exp(-0.5 * z * z)
    return eta


def eta_k_vector(scenario) -> np.ndarray:
    """Scenario meal-appeal offset as a (K,) array in meals.MEAL_NAMES order. Zero by default."""
    eta_k = np.zeros(meals.K)
    for name, val in scenario.eta_k.items():
        eta_k[meals.IDX_BY_NAME[name]] = val
    return eta_k


@dataclass
class FireResult:
    fired: np.ndarray       # (N,) bool
    eligible: np.ndarray    # (N,) bool -- stage open AND that slot still empty
    stage_idx: int          # active stage this block, or -1
    q: np.ndarray           # (N,) per-block firing probability
    w: float
    eta_t: float
    lam: np.ndarray         # (N,) persona lam for the active stage (0 if no stage active)
    hunger: np.ndarray      # (N,)
    alpha0_hunger: np.ndarray  # (N,) = alpha0_eff * hunger (the logit contribution)


def fire(state: AgentState, t_hr: float, population, scenario, rng: np.random.Generator,
         no_hunger: bool = False) -> FireResult:
    n_agents = state.h.shape[0]
    hunger = hunger_of(state.h, state.tau, t_hr)
    alpha0_eff = 0.0 if no_hunger else config.HUNGER.alpha0

    stage_idx = active_stage(t_hr)
    w = w_of_t(t_hr)
    eta_t = eta_t_of_t(t_hr, scenario)

    if stage_idx == -1:
        lam = np.zeros(n_agents)
        eligible = np.zeros(n_agents, dtype=bool)
    else:
        lam = population.lam[:, stage_idx]
        eligible = state.h[:, stage_idx] == 0

    alpha0_hunger = alpha0_eff * hunger
    logit = w + eta_t + lam + alpha0_hunger
    q = sigmoid(logit) * config.TIMING.DELTA
    fired = (rng.random(n_agents) < q) & eligible

    return FireResult(fired=fired, eligible=eligible, stage_idx=stage_idx, q=q, w=w, eta_t=eta_t,
                       lam=lam, hunger=hunger, alpha0_hunger=alpha0_hunger)


@dataclass
class WhichResult:
    choice: np.ndarray    # (N,) int, 0-based meal index; -1 where not fired
    probs: np.ndarray     # (N, K) softmax probabilities (only meaningful where fired)
    appeal: np.ndarray    # (N, K) gamma . z_k term
    eta_k: np.ndarray     # (K,) scenario offset (broadcast)
    cost_term: np.ndarray  # (N, K) = -gamma_cost * price(t) * e_k
    hunger_term: np.ndarray  # (N, K) = alpha_k * hunger


def which(state: AgentState, population, scenario, price_t: float, fired: np.ndarray,
          hunger: np.ndarray, rng: np.random.Generator, no_hunger: bool = False) -> WhichResult:
    n_agents = state.h.shape[0]
    appeal = population.gamma @ meals.Z.T                      # (N, K)
    eta_k = eta_k_vector(scenario)                             # (K,)
    cost_term = -population.gamma_cost[:, None] * price_t * meals.E_KWH[None, :]  # (N, K)
    alpha_k_eff = np.zeros_like(meals.ALPHA_K) if no_hunger else meals.ALPHA_K
    hunger_term = alpha_k_eff[None, :] * hunger[:, None]        # (N, K)

    u = appeal + eta_k[None, :] + cost_term + hunger_term
    probs = softmax(u, axis=1)

    gumbel = -np.log(-np.log(rng.random((n_agents, meals.K))))
    sampled = np.argmax(np.log(probs) + gumbel, axis=1)
    choice = np.where(fired, sampled, -1)

    return WhichResult(choice=choice, probs=probs, appeal=appeal, eta_k=eta_k,
                        cost_term=cost_term, hunger_term=hunger_term)


def update(state: AgentState, fire_result: FireResult, which_result: WhichResult) -> AgentState:
    new_h = state.h.copy()
    new_tau = state.tau.copy()
    fired = fire_result.fired
    if fire_result.stage_idx != -1 and np.any(fired):
        new_h[fired, fire_result.stage_idx] = which_result.choice[fired] + 1  # store 1-based meal idx
    new_tau[fired] = 0.0
    new_tau[~fired] = new_tau[~fired] + BLOCK_HOURS
    new_tau = np.clip(new_tau, 0.0, config.STATE.tau_cap_hr)
    return AgentState(h=new_h, tau=new_tau)
