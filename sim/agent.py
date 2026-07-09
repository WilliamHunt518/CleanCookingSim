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
    """Index into STAGE_ORDER of the nominal stage window containing t_hr, or -1 if none
    (overnight gap). Not used by fire() (no hard clock gate there -- see stage_bump_per_agent's
    docstring) nor by the Explainability worked example or PRISM export (both use stage_bump's
    argmax instead, for consistency with fire()) -- kept as a simple illustrative "which meal-time
    window is this, roughly" classifier for stage_windows_hr's own plotting/reference use and for
    tests that want a ground-truth nominal window to check against."""
    for i, stage in enumerate(STAGE_ORDER):
        lo, hi = config.TIMING.stage_windows_hr[stage]
        if lo <= t_hr < hi:
            return i
    return -1


def stage_bump(t_hr: float) -> np.ndarray:
    """(3,) -- each stage's own hazard-bump value at t_hr at the population-mean bump centre (no
    individual jitter), in STAGE_ORDER. Scalar/no-jitter counterpart of stage_bump_per_agent's
    per-agent version, used wherever a single representative agent is enough: the Explainability
    worked example and the PRISM export. argmax(stage_bump(t_hr)) is the "no jitter" version of
    the same which-stage-is-this-decision-about test fire() applies per-agent."""
    base = config.TIMING.overnight_base_logit
    out = np.empty(len(STAGE_ORDER))
    for i, stage in enumerate(STAGE_ORDER):
        centre = config.TIMING.bump_centers_hr[stage]
        height = config.TIMING.bump_heights[stage]
        width = config.TIMING.bump_width_hr
        z = (t_hr - centre) / width
        out[i] = base + (height - base) * np.exp(-0.5 * z * z)
    return out


def stage_bump_per_agent(t_hr: float, bump_center_jitter: np.ndarray) -> np.ndarray:
    """(N,3) -- per-agent counterpart of stage_bump: each stage's own hazard-bump value at t_hr,
    using each agent's personal jittered centre instead of the population mean.

    This is what lets fire() decide *which* stage an agent is deciding about right now without a
    hard stage_windows_hr clock gate: whichever not-yet-eaten stage's bump is highest at this
    moment is "the" stage for this agent this block. Because each stage's bump already rises and
    falls smoothly (bump_width_hr controls how fast, overnight_base_logit is the floor away from
    every bump), the moment one stage's bump overtakes another's -- e.g. lunch's rising bump
    exceeding breakfast's decaying one somewhere around mid-morning -- happens gradually and
    per-agent (via bump_center_jitter), not as a population-wide cliff at a fixed clock boundary.
    A hard window gate on top of this smooth math was redundant *and* was exactly what produced
    that cliff: it doesn't matter how gently a bump decays if eligibility is switched off/on at a
    fixed clock instant regardless of the bump's actual shape."""
    base = config.TIMING.overnight_base_logit
    n_agents = bump_center_jitter.shape[0]
    out = np.empty((n_agents, len(STAGE_ORDER)))
    for i, stage in enumerate(STAGE_ORDER):
        centre = config.TIMING.bump_centers_hr[stage] + bump_center_jitter[:, i]
        height = config.TIMING.bump_heights[stage]
        width = config.TIMING.bump_width_hr
        z = (t_hr - centre) / width
        out[:, i] = base + (height - base) * np.exp(-0.5 * z * z)
    return out


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
    eligible: np.ndarray    # (N,) bool -- best-stage-now slot still empty OR a repeat-meal roll hit
    stage_idx: np.ndarray   # (N,) int -- each agent's own "most-relevant-right-now" stage (argmax bump)
    q: np.ndarray           # (N,) per-block firing probability
    w: np.ndarray           # (N,) per-agent, that agent's chosen stage's own bump value
    eta_t: float
    lam: np.ndarray         # (N,) persona lam for each agent's chosen stage
    hunger: np.ndarray      # (N,)
    alpha0_hunger: np.ndarray  # (N,) = alpha0_eff * hunger (the logit contribution)
    price_term: np.ndarray  # (N,) = -kappa_price_time * gamma_cost * price(t) (the logit contribution)
    noise: np.ndarray       # (N,) ~ Normal(0, sigma_logit_noise), redrawn every block (the logit contribution)
    q_price_sensitive: np.ndarray  # (N,) firing probability from the normal, price_term-included pathway alone
    q_wood_floor: np.ndarray       # (N,) firing probability from the price-immune "free wood fallback" pathway alone


def fire(state: AgentState, t_hr: float, population, scenario, price_t: float, rng: np.random.Generator,
         no_hunger: bool = False) -> FireResult:
    n_agents = state.h.shape[0]
    hunger = hunger_of(state.h, state.tau, t_hr)
    alpha0_eff = 0.0 if no_hunger else config.HUNGER.alpha0
    eta_t = eta_t_of_t(t_hr, scenario)
    rows = np.arange(n_agents)

    # Which stage is this agent's decision "about" right now? Whichever stage's own bump is
    # highest at this moment -- see stage_bump_per_agent's docstring. No hard stage_windows_hr
    # clock gate: a hungry agent who hasn't fired by, say, 10:45 naturally finds lunch's rising
    # bump has already overtaken breakfast's decaying one and is considered for lunch instead,
    # smoothly and per-agent (via bump_center_jitter) -- not a population-wide cliff at 11:00:00.
    bump = stage_bump_per_agent(t_hr, population.bump_center_jitter)  # (N,3)
    stage_idx = np.argmax(bump, axis=1)  # (N,)
    w = bump[rows, stage_idx]
    lam = population.lam[rows, stage_idx]

    already_eaten = state.h[rows, stage_idx] != 0
    # Normally today's most-relevant stage can fire at most once (already_eaten locks it out). A
    # tiny, independent per-block chance re-opens it anyway -- a second helping, a snack, someone
    # eating again sooner than the model's clean 1-meal-per-stage structure would otherwise
    # allow. It's not hunger-driven (opportunistic, not need-driven), so it doesn't touch the
    # hunger accounting: h[stage] just gets overwritten with whatever was eaten this time, which
    # doesn't change n_eaten (still counted as one nonzero slot either way).
    repeat_roll = already_eaten & (rng.random(n_agents) < config.TIMING.repeat_meal_prob)
    eligible = (~already_eaten) | repeat_roll

    alpha0_hunger = alpha0_eff * hunger
    # Re-uses each agent's own gamma_cost (already zeroed by the --no-cost ablation) so a
    # price-sensitive persona is price-sensitive about *when* to cook, not just *what* --
    # a high price right now makes cooking at all less attractive, delaying the decision
    # until price drops or hunger overrides it. kappa_price_time sets the relative strength
    # of this effect vs. the existing Stage 2 meal-choice cost term.
    #
    # Penalise price(t) *relative to the tariff's own time-average* (p_bar), not its raw
    # level: every candidate tariff is normalised to the same p_bar (see sim.tariffs), so a
    # raw-price penalty would uniformly suppress firing under every tariff by roughly the same
    # amount -- including "flat", which should be the price-shape-neutral reference and see no
    # timing effect at all. Centering on p_bar makes only each tariff's time-varying *shape*
    # (cheap vs. expensive relative to its own average) push cooking earlier/later.
    #
    # Clipped to only ever penalise (never reward): price is a deterrent that can delay/suppress
    # a meal, not an inducement that invents a new eating occasion. Without the clip, a cheap
    # off-peak hour gives *every* agent a positive hazard boost -- including a school in its
    # breakfast/dinner stage, whose lam=-6 is a hard institutional-schedule constraint ("school
    # only serves lunch"), not an economic one. That boost was eroding lam=-6 enough that cheap
    # tariffs (evening_peak's off-peak hours, solar_following's midday trough) made schools
    # noticeably more likely to fire breakfast/dinner than under flat -- a school shouldn't
    # start serving breakfast at 10am just because power is cheap then.
    price_term = -config.TIMING.kappa_price_time * population.gamma_cost * np.maximum(
        price_t - config.TARIFF.p_bar, 0.0)
    # Idiosyncratic per-agent, per-block noise -- see TIMING.sigma_logit_noise's docstring for why
    # this exists: without it every agent's hazard at a given hour is identical up to
    # lam/hunger/price, so a large population synchronises into an almost perfectly clean,
    # razor-sharp peak with dead silence between meal times. This is what breaks that up.
    noise = rng.normal(0.0, config.TIMING.sigma_logit_noise, size=n_agents)
    logit_base = w + eta_t + lam + alpha0_hunger + noise
    q_price_sensitive = sigmoid(logit_base + price_term) * config.TIMING.DELTA
    # Price-immune "free firewood fallback" pathway: the same propensity to cook (hunger, timing,
    # persona habit) but with price_term forced to 0 -- an agent priced out of electric cooking can
    # still cook on wood, which grid tariffs don't touch. Capped by a much smaller DELTA_WOOD_FLOOR
    # (see its docstring) so this pathway is negligible next to a normal, un-suppressed
    # q_price_sensitive and only matters once price crushes that pathway toward 0 -- e.g.
    # extreme_test's 5x p_bar, which used to suppress cooking altogether (wood included) rather
    # than just electric cooking, since Stage 1 has no idea what fuel Stage 2 will pick.
    q_wood_floor = sigmoid(logit_base) * config.TIMING.DELTA_WOOD_FLOOR
    # Independent union, not max/sum: an agent fires if EITHER pathway's own (unobserved) coin
    # flip would fire -- P(A or B) = 1 - P(not A)*P(not B) for independent A, B.
    q = 1.0 - (1.0 - q_price_sensitive) * (1.0 - q_wood_floor)
    fired = (rng.random(n_agents) < q) & eligible

    return FireResult(fired=fired, eligible=eligible, stage_idx=stage_idx, q=q, w=w, eta_t=eta_t,
                       lam=lam, hunger=hunger, alpha0_hunger=alpha0_hunger, price_term=price_term,
                       noise=noise, q_price_sensitive=q_price_sensitive, q_wood_floor=q_wood_floor)


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
    if np.any(fired):
        # Each fired agent may have a different "stage_idx" now (see fire()'s docstring), so this
        # is fancy-indexed per-agent rather than a single shared column like the old hard-window
        # version.
        new_h[fired, fire_result.stage_idx[fired]] = which_result.choice[fired] + 1  # 1-based meal idx
    new_tau[fired] = 0.0
    new_tau[~fired] = new_tau[~fired] + BLOCK_HOURS
    new_tau = np.clip(new_tau, 0.0, config.STATE.tau_cap_hr)
    return AgentState(h=new_h, tau=new_tau)
