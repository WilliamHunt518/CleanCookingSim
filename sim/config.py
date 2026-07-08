"""
Single source of truth for every numeric parameter in the simulator.

Every value is registered through `p(...)` so that:
  - `python -m sim explain` can print a full glossary (value, units, meaning,
    what turning it up/down does), grouped by topic.
  - `python -m sim audit` can list every parameter still flagged `tbd=True`
    (a placeholder guess, not a sourced number).

Do not hardcode numbers anywhere else in the codebase -- import from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Param:
    group: str
    name: str
    value: Any
    units: str
    meaning: str
    effect: str
    tbd: bool


_REGISTRY: list[Param] = []


def p(group: str, name: str, value: Any, units: str, meaning: str, effect: str, tbd: bool = True) -> Any:
    """Register a parameter for the explain/audit glossary and return its value."""
    _REGISTRY.append(Param(group=group, name=name, value=value, units=units,
                            meaning=meaning, effect=effect, tbd=tbd))
    return value


def all_params() -> list[Param]:
    return list(_REGISTRY)


def tbd_params() -> list[Param]:
    return [x for x in _REGISTRY if x.tbd]


GROUP_ORDER = ["state", "timing", "hunger", "meals", "personas", "scenario", "tariff", "scoring"]


# ---------------------------------------------------------------------------
# state -- discrete-time chain shape
# ---------------------------------------------------------------------------
@dataclass
class StateConfig:
    block_minutes: float
    T: int
    K: int
    tau_cap_hr: float
    n_stages: int


STATE = StateConfig(
    block_minutes=p("state", "block_minutes", 5.0, "minutes",
                     "Length of one discrete simulation block.",
                     "Smaller = finer time resolution, more blocks, slower run.",
                     tbd=False),
    T=p("state", "T", 288, "blocks/day",
        "Number of 5-minute blocks in a 24h day (24*60/5).",
        "Derived from block_minutes; not an independent knob.",
        tbd=False),
    K=p("state", "K", 6, "meals",
        "Size of the fixed meal menu.",
        "Structural -- add/remove rows in the meal table to change this.",
        tbd=False),
    tau_cap_hr=p("state", "tau_cap_hr", 24.0, "hours",
                  "Cap on 'hours since last meal' (tau) so hunger doesn't grow unboundedly.",
                  "Higher cap lets hunger (and firing pressure) keep climbing longer after a skipped meal.",
                  tbd=True),
    n_stages=p("state", "n_stages", 3, "stages",
                "Number of meal stages per day: breakfast, lunch, dinner.",
                "Structural.", tbd=False),
)


# ---------------------------------------------------------------------------
# timing -- hazard w(t), stage windows, DELTA
# ---------------------------------------------------------------------------
@dataclass
class TimingConfig:
    overnight_base_logit: float
    bump_centers_hr: dict
    bump_heights: dict
    bump_width_hr: float
    DELTA: float
    stage_windows_hr: dict


TIMING = TimingConfig(
    overnight_base_logit=p("timing", "overnight_base_logit", -6.0, "logit",
                             "Firing-hazard logit far from any meal bump (i.e. overnight).",
                             "More negative = agents essentially never start cooking outside meal windows.",
                             tbd=False),
    bump_centers_hr=p("timing", "bump_centers_hr",
                       {"breakfast": 7.5, "lunch": 12.5, "dinner": 18.5}, "hour (24h clock)",
                       "Clock-time centre of each stage's cooking-hazard bump.",
                       "Shifting a centre later moves that meal's typical start time later.",
                       tbd=True),
    bump_heights=p("timing", "bump_heights",
                   {"breakfast": 1.2, "lunch": 1.5, "dinner": 1.8}, "logit",
                   "Peak hazard logit at the centre of each stage's bump (w(t) relaxes to "
                   "overnight_base_logit away from the bump).",
                   "Higher = sharper, more certain firing right at that meal's peak time.",
                   tbd=True),
    bump_width_hr=p("timing", "bump_width_hr", 1.5, "hours",
                     "Gaussian width (std dev) of each meal-time hazard bump.",
                     "Wider = firing probability spread over a longer window around the peak.",
                     tbd=True),
    DELTA=p("timing", "DELTA", 0.15, "probability/block (max)",
            "Scale factor converting the hazard logit's sigmoid into a per-block firing "
            "probability; also the hard cap on that probability (sigmoid in [0,1]). "
            "Calibrated empirically (spec suggested 0.05 as a starting guess; that gave only "
            "~37% of agents eating 3 meals/day under the reference tariff, so it was raised to "
            "0.15, which gives ~90%, matching the spec's calibration target).",
            "The single most important tuning knob for how often anyone cooks at all -- "
            "raise it so more agents complete 3 meals/day, lower it for a sparser day.",
            tbd=True),
    stage_windows_hr=p("timing", "stage_windows_hr",
                        {"breakfast": (5.0, 11.0), "lunch": (11.0, 16.0), "dinner": (16.0, 23.0)},
                        "hour (24h clock)",
                        "Clock-time window during which each stage's slot may fire at all; "
                        "a stage cannot fire outside its window, and cannot fire again once its "
                        "slot in h is nonzero.",
                        "Narrower windows force meals into a tighter part of the day.",
                        tbd=True),
)


# ---------------------------------------------------------------------------
# hunger
# ---------------------------------------------------------------------------
@dataclass
class HungerConfig:
    nbar_step_times_hr: dict
    kappa: float
    alpha0: float


HUNGER = HungerConfig(
    nbar_step_times_hr=p("hunger", "nbar_step_times_hr",
                          {"0": 0.0, "1": 9.0, "2": 14.0, "3": 21.0}, "hour (24h clock)",
                          "Step curve nbar(t): expected cumulative meals eaten by this time of day "
                          "(0 meals before 09:00, 1 after 09:00, 2 after 14:00, 3 after 21:00).",
                          "Moving a step earlier makes agents 'behind schedule' (and hence hungrier) sooner.",
                          tbd=True),
    kappa=p("hunger", "kappa", 0.05, "hunger-units/hour",
            "Weight converting hours-since-last-meal (tau) into hunger.",
            "Higher = time-since-eating dominates hunger; lower = only being behind on meal count matters.",
            tbd=True),
    alpha0=p("hunger", "alpha0", 0.4, "logit/hunger-unit",
             "Weight of hunger in the firing-hazard logit (Stage 1).",
             "Higher = hungrier agents become much more likely to start cooking at all.",
             tbd=True),
)


def nbar(t_hr: float) -> int:
    """Step curve: expected cumulative meals eaten by clock time t_hr."""
    steps = HUNGER.nbar_step_times_hr
    if t_hr >= steps["3"]:
        return 3
    if t_hr >= steps["2"]:
        return 2
    if t_hr >= steps["1"]:
        return 1
    return 0


# ---------------------------------------------------------------------------
# meals -- fixed menu (K=6). idx 1..6, matching the slot values written into h.
# ---------------------------------------------------------------------------
@dataclass
class MealRow:
    idx: int
    name: str
    e_kwh: float
    dbar_min: float
    taste: float
    trad: float
    effort: float
    fuelcost: float
    alpha_k: float
    stage: str  # which of h's 3 slots this meal can fill: breakfast/lunch/dinner (all, in practice)


def _meal(idx, name, e_kwh, dbar_min, taste, trad, effort, fuelcost, alpha_k):
    return MealRow(
        idx=idx, name=name,
        e_kwh=p("meals", f"{name}.e_kwh", e_kwh, "kWh",
                 f"Grid electrical energy drawn per cook of '{name}'.",
                 "0 for wood/cold meals -> immune to the electricity tariff.", tbd=True),
        dbar_min=p("meals", f"{name}.dbar_min", dbar_min, "minutes",
                    f"Mean cook duration for '{name}'.",
                    "Longer duration spreads the same energy over more blocks (lower average kW).", tbd=True),
        taste=p("meals", f"{name}.taste", taste, "utils",
                 f"Taste attribute of '{name}' (feeds gamma.z in the choice utility).",
                 "Higher = more appealing regardless of cost/effort.", tbd=True),
        trad=p("meals", f"{name}.trad", trad, "utils",
                f"Traditionality attribute of '{name}'.",
                "Higher = favoured by agents/scenarios with strong trad weight (e.g. festival_day).", tbd=True),
        effort=p("meals", f"{name}.effort", effort, "utils",
                  f"Effort attribute of '{name}' (should carry negative gamma weight).",
                  "Higher effort = more deterred, all else equal.", tbd=True),
        fuelcost=p("meals", f"{name}.fuelcost", fuelcost, "utils",
                    f"Non-grid fuel cost attribute of '{name}' (e.g. charcoal/firewood spend; "
                    "distinct from the grid tariff, which only applies via e_kwh).",
                    "Higher = more deterred; this is how wood meals still cost the household something.",
                    tbd=True),
        alpha_k=p("meals", f"{name}.alpha_k", alpha_k, "logit/hunger-unit",
                   f"Extra utility '{name}' gets per unit of hunger (Stage 2 boost).",
                   "Positive only for big/filling meals -- makes them relatively more attractive when very hungry.",
                   tbd=True),
        stage="any",
    )


MEALS: list[MealRow] = [
    _meal(1, "big_e_cook", 1.5, 45, 0.8, 0.0, 0.5, 0.0, 0.8),
    _meal(2, "small_e_cook", 0.6, 20, 0.5, 0.0, 0.3, 0.0, 0.0),
    _meal(3, "quick_e_snack", 0.2, 10, 0.3, 0.0, 0.1, 0.0, 0.0),
    _meal(4, "big_wood_cook", 0.0, 60, 0.7, 1.0, 0.9, 0.6, 0.8),
    _meal(5, "small_wood", 0.0, 30, 0.4, 1.0, 0.7, 0.4, 0.0),
    _meal(6, "cold_no_cook", 0.0, 5, 0.2, 0.0, 0.05, 0.1, 0.0),
]
WOOD_MEAL_INDICES = (4, 5)

MEAL_DURATION_SIGMA_MIN = p("meals", "duration_sigma_min", 8.0, "minutes",
                              "Std dev of per-cook duration draw D ~ Normal(Dbar_k, sigma).",
                              "Higher = more variability in how long a cook (and its demand pulse) lasts.",
                              tbd=True)
MEAL_DURATION_MAX_MIN = p("meals", "duration_max_min", 150.0, "minutes",
                            "Hard clip on sampled cook duration.",
                            "Prevents rare huge draws from producing absurdly long demand pulses.",
                            tbd=True)

MEAL_PROFILE_SHAPE = p("meals", "profile_shape", "boxcar", "n/a",
                        "Power-profile shape phi_k[j] used for every meal: 'boxcar' (flat kW for the "
                        "whole cook, default) or 'preheat_simmer' (short high-power spike then a lower "
                        "simmer level, same total energy). Plumbing for the latter exists; boxcar ships.",
                        "Switching to preheat_simmer makes short-lived demand spikes sharper without "
                        "changing daily household kWh.",
                        tbd=True)
PREHEAT_SPIKE_FRAC = p("meals", "preheat_spike_frac", 0.2, "fraction of duration",
                        "Fraction of a cook's duration spent in the preheat spike (preheat_simmer shape only).",
                        "Larger = spike lasts longer (still same total energy, so simmer level drops).",
                        tbd=True)
PREHEAT_POWER_MULT = p("meals", "preheat_power_mult", 2.0, "multiple of boxcar power",
                        "Preheat spike power as a multiple of the equivalent boxcar power (preheat_simmer only).",
                        "Larger = sharper, taller spike at the start of the cook.",
                        tbd=True)


# ---------------------------------------------------------------------------
# personas -- household / school, as sparse overrides on a base
# ---------------------------------------------------------------------------
@dataclass
class PersonaConfig:
    base_gamma: dict
    base_gamma_cost: float
    sigma_ind: float
    base_lam: dict
    school_overrides: dict
    mix: dict


PERSONAS = PersonaConfig(
    base_gamma=p("personas", "base_gamma",
                 {"taste": 1.0, "trad": 0.3, "effort": -0.8, "fuelcost": -1.0}, "utils/attribute-unit",
                 "Household taste-weight vector applied to meal attributes z_k in the choice utility.",
                 "effort and fuelcost should stay negative (they are costs); raising |effort|/|fuelcost| "
                 "makes agents avoid demanding/costly meals more strongly.",
                 tbd=True),
    base_gamma_cost=p("personas", "base_gamma_cost", 1.2, "utils per (currency/kWh * kWh)",
                       "Household price sensitivity: weight on -price(t)*e_k in the choice utility. "
                       "THE only channel through which the tariff affects behaviour.",
                       "The single most important knob for tariff response -- raise it and high prices "
                       "push agents toward wood/cold meals much more sharply.",
                       tbd=True),
    sigma_ind=p("personas", "sigma_ind", 0.15, "utils (std dev)",
                 "Std dev of per-agent individual variation around persona-mean gamma and gamma_cost, "
                 "sampled once at init: gamma_i ~ Normal(gamma_persona, sigma_ind^2 * I).",
                 "Higher = more heterogeneous population, less binary 'everyone defects at once' behaviour.",
                 tbd=True),
    base_lam=p("personas", "base_lam",
               {"breakfast": 0.0, "lunch": 0.0, "dinner": 0.0}, "logit",
               "Household per-stage firing-hazard bias lam[persona][stage], added into Stage 1's logit.",
               "Zero for the reference household persona (no stage is suppressed or boosted).",
               tbd=True),
    school_overrides=p("personas", "school_overrides",
                        {"gamma": {"trad": 0.0}, "lam": {"breakfast": -6.0, "lunch": 2.0, "dinner": -6.0}},
                        "n/a",
                        "Sparse overrides applied on top of the household base for the 'school' persona: "
                        "zero trad weight (no attachment to traditional wood cooking), breakfast and dinner "
                        "hazard effectively switched off (-6, same magnitude as the overnight base logit), and "
                        "a strongly boosted midday hazard (+2.0 at lunch) -- schools cook an essentially "
                        "unimodal, lunch-only day (one big canteen meal), not three meals like a household.",
                        "Encodes 'schools cook one big lunch and nothing else' without a separate model.",
                        tbd=True),
    mix=p("personas", "mix", {"household": 95, "school": 5}, "agent count",
          "Population composition.",
          "More schools shifts the aggregate load curve toward a midday peak.",
          tbd=True),
)

N_AGENTS = p("personas", "N_AGENTS", 100, "agents",
             "Total population size simulated per run.",
             "More agents = smoother aggregate demand curve, slower run.",
             tbd=True)


# ---------------------------------------------------------------------------
# scenario -- eta offsets, default zero (reference scenario); festival_day example
# ---------------------------------------------------------------------------
@dataclass
class ScenarioConfig:
    name: str
    eta_t_hr_offsets: dict  # sparse: {stage: extra bump (centre_shift_hr, height, width_hr)} or {}
    eta_k: dict  # sparse: {meal_name: extra utility offset}


REFERENCE_SCENARIO = ScenarioConfig(
    name=p("scenario", "reference.name", "reference", "n/a",
           "Default scenario: no timing or appeal offsets applied to anyone.",
           "This is the baseline every tariff sweep runs under unless --scenario overrides it.",
           tbd=False),
    eta_t_hr_offsets=p("scenario", "reference.eta_t_hr_offsets", {}, "n/a",
                        "Scenario timing offset eta_t(t), reference case: identically zero everywhere.",
                        "n/a", tbd=False),
    eta_k=p("scenario", "reference.eta_k", {}, "n/a",
            "Scenario meal-appeal offset eta_k, reference case: identically zero for every meal.",
            "n/a", tbd=False),
)

FESTIVAL_DAY_SCENARIO = ScenarioConfig(
    name=p("scenario", "festival_day.name", "festival_day", "n/a",
           "Example alternative scenario demonstrating eta portability: a festival raises the appeal "
           "of big/traditional meals and pushes dinner later.",
           "Included as a template for adding new scenarios; not used unless --scenario festival_day.",
           tbd=True),
    eta_t_hr_offsets=p("scenario", "festival_day.eta_t_hr_offsets",
                        {"dinner": {"centre_shift_hr": 1.5, "height": 1.0, "width_hr": 1.0}},
                        "hour / logit / hour",
                        "Adds an extra hazard bump 1.5h after the normal dinner centre, height 1.0, "
                        "width 1h -- on top of the base dinner bump, biasing firing later in the evening.",
                        "Larger centre_shift_hr / height = dinner drifts later / more sharply.",
                        tbd=True),
    eta_k=p("scenario", "festival_day.eta_k",
            {"big_e_cook": 0.5, "big_wood_cook": 1.0}, "utils",
            "Flat utility bonus added to the two 'big meal' options, biggest for the traditional wood one, "
            "representing festival appeal.",
            "Raises the odds those meals are chosen conditional on firing.",
            tbd=True),
)

SCENARIOS = {"reference": REFERENCE_SCENARIO, "festival_day": FESTIVAL_DAY_SCENARIO}


# ---------------------------------------------------------------------------
# tariff -- candidates, PV stub, cap
# ---------------------------------------------------------------------------
@dataclass
class TariffConfig:
    p_bar: float
    p_lo: float
    p_hi: float
    w_peak_hr: tuple
    pv_p_max_kw: float
    pv_clearness: float
    pv_t_rise_hr: float
    pv_t_set_hr: float
    cap_kw: float


TARIFF = TariffConfig(
    p_bar=p("tariff", "p_bar", 0.25, "currency/kWh",
            "Common time-average price every candidate tariff is normalised to.",
            "Scales all tariffs up/down together; relative shape (peak/flat/solar) is what's compared.",
            tbd=True),
    p_lo=p("tariff", "p_lo", 0.10, "currency/kWh",
           "Off-peak / low price level used by evening_peak and solar_following.",
           "Lower = cheaper off-peak incentive to shift load away from the peak/evening.",
           tbd=True),
    p_hi=p("tariff", "p_hi", 0.45, "currency/kWh",
           "Peak price level used by evening_peak and solar_following.",
           "Higher = stronger price signal to avoid cooking electrically at peak times -- more wood defection.",
           tbd=True),
    w_peak_hr=p("tariff", "w_peak_hr", (17.0, 21.0), "hour (24h clock)",
                 "Evening peak window for the evening_peak tariff.",
                 "Wider window = longer stretch of expensive electricity.",
                 tbd=True),
    pv_p_max_kw=p("tariff", "pv_p_max_kw", 40.0, "kW",
                   "Peak PV output used only to shape the solar_following tariff (never feeds agent utility).",
                   "Larger = solar_following's price trough is based on a bigger notional solar fleet.",
                   tbd=True),
    pv_clearness=p("tariff", "pv_clearness", 0.8, "fraction",
                    "Sky clearness factor scaling the PV stub's output.",
                    "Lower = cloudier reference day, less pronounced solar-following price dip.",
                    tbd=True),
    pv_t_rise_hr=p("tariff", "pv_t_rise_hr", 6.5, "hour (24h clock)",
                    "Sunrise time used by the PV stub.",
                    "Later sunrise shrinks the window where solar_following prices are cheap.",
                    tbd=True),
    pv_t_set_hr=p("tariff", "pv_t_set_hr", 18.5, "hour (24h clock)",
                   "Sunset time used by the PV stub.",
                   "Earlier sunset shrinks the window where solar_following prices are cheap.",
                   tbd=True),
    cap_kw=p("tariff", "cap_kw", 30.0, "kW",
              "Grid/mini-grid aggregate demand cap used for the exceedance penalty.",
              "Lower cap = exceedance probability rises for the same population -> tariffs scored harsher.",
              tbd=True),
)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoringConfig:
    PI: float
    R: int


SCORING = ScoringConfig(
    PI=p("scoring", "PI", 5.0, "score/probability",
         "Penalty weight on P_exceed in score = wood_share + PI * P_exceed.",
         "Higher = the scoreboard cares much more about avoiding cap breaches than about wood share.",
         tbd=True),
    R=p("scoring", "R", 200, "runs",
        "Number of independent Monte Carlo days simulated per tariff.",
        "More runs = more stable P_exceed estimate (spec target: stable to +-0.02 across seeds), slower sweep.",
        tbd=True),
)


DEFAULT_SEED = p("state", "default_seed", 0, "n/a",
                  "Default RNG seed for reproducibility when --seed is not given.",
                  "Same seed + same config = identical output.",
                  tbd=False)
