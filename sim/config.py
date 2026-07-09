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
    K=p("state", "K", 18, "meals",
        "Size of the fixed meal menu (must match len(MEALS) below).",
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
    kappa_price_time: float
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
    kappa_price_time=p("timing", "kappa_price_time", 4.0, "logit per (currency/kWh, scaled by gamma_cost)",
                        "Stage 1 (does it fire at all) price penalty: price_term = -kappa_price_time * "
                        "gamma_cost * max(price(t) - p_bar, 0), added into the firing-hazard logit "
                        "alongside w(t)/eta_t/lam/hunger. Centered on p_bar (the time-average every "
                        "candidate tariff is normalised to) so a flat tariff -- always at p_bar -- "
                        "contributes exactly 0 and only a tariff's cheap/expensive *shape* moves timing; "
                        "a raw-price penalty would otherwise suppress firing under every tariff by "
                        "roughly the same amount, including the reference case. Clipped at 0 (never "
                        "negative, i.e. never a reward) so a below-average price cannot *boost* firing "
                        "hazard -- price is a deterrent that can delay/suppress a meal, not an "
                        "inducement that invents a new eating occasion. Without the clip, a cheap "
                        "off-peak hour gave every agent a positive hazard boost including a school in "
                        "its breakfast/dinner stage, where lam=-6 is a hard institutional-schedule "
                        "constraint, not an economic one -- that boost was eroding lam=-6 enough that "
                        "schools fired breakfast noticeably more often under evening_peak/solar_following "
                        "than under flat. Re-uses each agent's own gamma_cost (so --no-cost zeroes this "
                        "too) rather than a separate parameter -- a price-sensitive persona is "
                        "price-sensitive about *when* to cook, not just *what*. Without this term at all, "
                        "price only ever changed Stage 2 (meal choice at a fixed time); cooking timing "
                        "was identical across every tariff. Empirically tuned alongside base_gamma_cost "
                        "and the widened p_hi/p_lo gap: 4.0 cuts the share of evening_peak dinners still "
                        "starting inside its 17-21h peak window from ~85% to ~0% and evening_peak's "
                        "meals/day from ~2.8 to ~2.0 -- displaced agents mostly reschedule into cheap "
                        "hours and cook electrically there, so the tariff's harm shows up as "
                        "skipped/delayed meals as much as fuel-switching.",
                        "Higher = tariffs visibly reshape *when* the population cooks, not just what they "
                        "cook; too high and expensive-hour agents mostly stop firing rather than shifting.",
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
# meals -- 18-meal menu transcribed from master_table_Z.md (Magadi/Kajiado West
# meal feature matrix). idx 1..18, matching the slot values written into h.
#
# Decision features (gamma-weighted, see ATTR_ORDER in population.py): taste,
# tradition, kid-acceptance, batch/leftover potential, ingredient cost,
# active prep labour, and kcal (as a proxy for "how filling") are all given
# directly by the source table; the eighth attribute, fuelcost, is *not* in
# the source table as a gamma-weighted column -- it is reconstructed here
# from the table's charcoal kg/KES column so that fire-only meals still carry
# a real (if fixed, non-tariffed) cost, preserving the "wood is immune to the
# grid tariff but not free" property. ing_cost/prep_min/kcal/charcoal_kes are
# normalised by the *_MAX constants below so every gamma-weighted feature
# sits on a roughly comparable ~0-1 scale, matching taste/tradition/kid/batch
# (which the source table already gives on 0-1). This normalisation is a
# modelling choice, not specified by master_table_Z.md, and is flagged tbd.
# ---------------------------------------------------------------------------
@dataclass
class MealRow:
    idx: int
    name: str
    meal_type: str          # "ELEC" or "FIRE" (informational; fire_only is authoritative)
    dbar_min: float          # cook duration, minutes
    e_kwh: float             # grid energy per cook; 0 for fire-only meals
    charcoal_kg: float
    charcoal_kes: float
    kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    ing_cost_kes: float
    prep_min: float
    taste: float             # 0-1
    tradition: float         # 0-1
    kid: float                # 0-1, kid-acceptance
    batch: float              # 0-1, batch/leftover potential
    fire_only: bool           # 1 = zero clean-cooking proposition (wood/charcoal only)


ING_COST_MAX_KES = p("meals", "ing_cost_max_kes", 150.0, "KES/serving",
                      "Reference max used to normalise ing_cost_kes into the ~0-1 gamma-weighted "
                      "ing_cost feature (150 = the priciest meal in master_table_Z.md, nyama choma).",
                      "Lower max makes ing_cost saturate (hit 1.0) for cheaper meals too, flattening "
                      "the cost signal between mid-priced dishes.", tbd=True)
PREP_MIN_MAX = p("meals", "prep_min_max", 45.0, "minutes",
                  "Reference max used to normalise prep_min into the ~0-1 gamma-weighted prep_min "
                  "feature (45 = chapati, the most labour-intensive meal in the table).",
                  "Lower max makes the labour penalty saturate for less demanding meals too.", tbd=True)
KCAL_MAX = p("meals", "kcal_max", 800.0, "kcal/serving",
             "Reference max used to normalise kcal into the ~0-1 gamma-weighted kcal feature, and "
             "to scale ALPHA_SCALE-based hunger boosts (800 = ugali_sukuma_beef_stew, the richest meal).",
             "Lower max makes both the kcal appeal term and the hunger boost saturate sooner.", tbd=True)
CHARCOAL_KES_MAX = p("meals", "charcoal_kes_max", 108.0, "KES/serving",
                      "Reference max used to normalise charcoal_kes into the fuelcost feature for "
                      "fire-only meals (108 = the fire-only bone soup, the priciest charcoal cost).",
                      "Lower max makes the fuelcost disincentive saturate for cheaper fire meals too.",
                      tbd=True)
ALPHA_SCALE = p("meals", "alpha_scale", 1.0, "logit/hunger-unit (at kcal_norm=1)",
                 "Scales kcal_norm into each meal's alpha_k hunger boost (alpha_k = ALPHA_SCALE * "
                 "kcal/KCAL_MAX) -- richer meals satisfy hunger more, instead of hand-flagging "
                 "individual meals as 'big'.",
                 "Higher = kcal-rich meals become dramatically more attractive the hungrier an agent is.",
                 tbd=True)


def _meal(idx, name, meal_type, dbar_min, e_kwh, charcoal_kg, charcoal_kes, kcal, protein_g, carb_g,
          fat_g, ing_cost_kes, prep_min, taste, tradition, kid, batch, fire_only) -> MealRow:
    return MealRow(idx=idx, name=name, meal_type=meal_type, dbar_min=dbar_min, e_kwh=e_kwh,
                   charcoal_kg=charcoal_kg, charcoal_kes=charcoal_kes, kcal=kcal, protein_g=protein_g,
                   carb_g=carb_g, fat_g=fat_g, ing_cost_kes=ing_cost_kes, prep_min=prep_min,
                   taste=taste, tradition=tradition, kid=kid, batch=batch, fire_only=bool(fire_only))


# name, type, duration_min, e_kwh, charcoal_kg, charcoal_kes, kcal, P, C, F, ing_cost, prep_min,
# taste, tradition, kid, batch, fire_only -- transcribed verbatim from master_table_Z.md.
MEALS: list[MealRow] = [
    _meal(1, "uji_honey_sweetpotato", "ELEC", 45, 0.75, 0.4, 24, 280, 7, 60, 2, 25, 10,
          0.50, 0.80, 1.00, 0.40, 0),
    _meal(2, "mahamri_mbaazi_za_nazi", "ELEC", 60, 1.20, 0.6, 36, 780, 21, 112, 27, 60, 40,
          0.80, 0.40, 0.90, 0.80, 0),
    _meal(3, "chai_roasted_sweetpotato_cassava", "FIRE", 45, 0.0, 0.6, 36, 360, 6, 70, 6, 30, 10,
          0.55, 0.85, 0.80, 0.30, 1),
    _meal(4, "githeri_avocado_epc", "ELEC", 60, 1.00, 0.8, 48, 640, 22, 92, 21, 45, 20,
          0.55, 0.50, 0.70, 1.00, 0),
    _meal(5, "ugali_ndengu_stew", "ELEC", 60, 1.20, 0.6, 36, 620, 26, 108, 9, 40, 15,
          0.60, 0.55, 0.70, 0.80, 0),
    _meal(6, "ugali_sukuma_beef_stew", "ELEC", 90, 2.25, 0.8, 48, 800, 37, 98, 27, 90, 30,
          0.80, 0.60, 0.80, 0.60, 0),
    _meal(7, "ugali_kuku_kienyeji_managu", "ELEC", 90, 2.10, 0.9, 54, 770, 42, 88, 25, 110, 35,
          0.90, 0.70, 0.90, 0.50, 0),
    _meal(8, "chapati_maharagwe_ya_nazi", "ELEC", 90, 2.40, 0.9, 54, 680, 21, 92, 25, 55, 45,
          0.85, 0.50, 0.95, 0.70, 0),
    _meal(9, "ugali_fried_tilapia_kachumbari", "ELEC", 45, 1.13, 0.5, 30, 730, 46, 74, 28, 100, 20,
          0.85, 0.50, 0.60, 0.20, 0),
    _meal(10, "mukimo_beef_stew", "ELEC", 75, 1.88, 0.8, 48, 790, 36, 95, 27, 85, 30,
          0.70, 0.60, 0.80, 0.50, 0),
    _meal(11, "matoke_beef", "ELEC", 45, 0.90, 0.5, 30, 420, 19, 58, 13, 60, 20,
          0.60, 0.50, 0.70, 0.50, 0),
    _meal(12, "motori_bone_soup_cassava", "ELEC", 150, 2.00, 1.4, 84, 550, 28, 72, 16, 70, 15,
          0.70, 1.00, 0.50, 0.70, 0),
    _meal(13, "nyama_choma_oven_kachumbari", "ELEC", 60, 2.00, 1.5, 90, 385, 45, 4, 21, 150, 15,
          0.90, 0.70, 0.60, 0.50, 0),
    _meal(14, "nyama_choma_open_fire", "FIRE", 60, 0.0, 1.5, 90, 385, 45, 4, 21, 150, 15,
          1.00, 1.00, 0.60, 0.50, 1),
    _meal(15, "tilapia_catfish_grilled_fire", "FIRE", 30, 0.0, 0.7, 42, 280, 40, 2, 12, 90, 15,
          0.80, 0.50, 0.50, 0.20, 1),
    _meal(16, "motori_bone_soup_cassava_fire", "FIRE", 150, 0.0, 1.8, 108, 550, 28, 72, 16, 70, 15,
          0.70, 1.00, 0.50, 0.70, 1),
    _meal(17, "ugali_sukuma_side", "ELEC", 45, 1.13, 0.5, 30, 410, 11, 77, 7, 30, 15,
          0.60, 0.60, 0.80, 0.30, 0),
    _meal(18, "ugali_kachumbari_side", "ELEC", 30, 0.75, 0.4, 24, 320, 8, 68, 3, 20, 10,
          0.55, 0.60, 0.80, 0.30, 0),
]

_MEAL_TABLE_NOTE = p(
    "meals", "meal_table", f"{len(MEALS)} meals (see sim/config.py MEALS list)", "n/a",
    "The full 18-meal feature table transcribed from master_table_Z.md (Magadi/Kajiado West meal "
    "feature matrix): physical/cost columns (duration, kWh, charcoal kg/KES), nutrition (kcal, "
    "protein/carb/fat), and decision features (taste, tradition, kid-acceptance, batch potential, "
    "ingredient cost, prep labour). Individual meal values aren't tracked as separate glossary rows "
    "(300+ would swamp `explain`/`audit`) but the whole table is still a calibration placeholder.",
    "Per master_table_Z.md: 'All decision-column values are calibration placeholders -- priced at "
    "rough 2026 local market rates and scored by judgement. Replace with survey data (KAOP or "
    "household interviews) when available; the Z structure and lambda.z + gamma machinery don't change.'",
    tbd=True)

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
# personas -- household / school / kiosk, as sparse additive offsets on a base
#
# household is the base persona (lambda in master_table_Z.md's "house (base
# lambda)" column). school and kiosk are "group offsets" (gamma) ADDED on top
# of that base per feature -- a persona with no entry for a feature simply
# inherits the household value unchanged. This matches master_table_Z.md's
# own framing ("gamma upweights/downweights ...") and its dash ("--") entries
# for unspecified feature/persona combinations. lam (Stage 1 hazard bias,
# timing) is a separate, orthogonal REPLACE-semantics override -- unrelated
# to gamma/Z and not covered by master_table_Z.md at all.
# ---------------------------------------------------------------------------
@dataclass
class PersonaConfig:
    base_gamma: dict
    base_gamma_cost: float
    sigma_ind: float
    base_lam: dict
    persona_gamma_offsets: dict  # {persona_name: {feature: additive delta}}
    persona_lam: dict            # {persona_name: {stage: replacement value}}
    mix: dict


PERSONAS = PersonaConfig(
    base_gamma=p("personas", "base_gamma",
                 {"taste": 1.0, "tradition": 0.6, "kid": 0.4, "batch": 0.3, "ing_cost": -0.8,
                  "prep_min": -0.4, "kcal": 0.2, "fuelcost": -0.8},
                 "utils/attribute-unit",
                 "Household base taste-weight vector (lambda in master_table_Z.md) applied to meal "
                 "attributes z_k. First 7 features + values are transcribed directly from "
                 "master_table_Z.md's 'house (base lambda)' column; fuelcost has no source-table row "
                 "(the table doesn't list a gamma-weighted fuel-cost feature) so it reuses this "
                 "project's earlier placeholder to keep fire meals non-free.",
                 "ing_cost/prep_min should stay negative (they are costs); raising |ing_cost|/|prep_min| "
                 "makes agents avoid expensive/demanding meals more strongly.",
                 tbd=True),
    base_gamma_cost=p("personas", "base_gamma_cost", 4.5, "utils per (currency/kWh * kWh)",
                       "Household price sensitivity: weight on -price(t)*e_k in the choice utility. "
                       "THE only channel through which the grid tariff affects meal choice (Stage 2; "
                       "see kappa_price_time for the analogous Stage 1 timing channel). Not part of "
                       "master_table_Z.md (that table has no tariff/price-sensitivity concept); shared "
                       "across all personas since the source table gives no override. Empirically "
                       "re-tuned twice (see README): 1.2 gave only a ~17% relative wood_share gap "
                       "between tariffs -- too subtle to read off the scoreboard. 2.5 fixed that but "
                       "still felt muted once kappa_price_time/p_hi/p_lo let agents dodge expensive "
                       "hours by rescheduling rather than switching fuel. 4.5 (alongside "
                       "kappa_price_time=4.0) makes the scoreboard spread unmistakable -- wood_share "
                       "0.22 (evening_peak) to 0.42 (flat), a ~2x range -- while mean household kWh "
                       "still stays in a plausible 2.0-2.3 range. Note the ordering: evening_peak now "
                       "scores *best*, not worst -- its cheap off-peak hours (p_lo=0.05) dominate its "
                       "day, and a price-responsive population mostly reschedules into them rather than "
                       "eating electric at peak, so it out-performs flat's constant moderate price. Still "
                       "flagged tbd=True since 4.5 itself is a guess, not a sourced number.",
                       "The single most important knob for tariff response -- raise it and high prices "
                       "push agents toward fire-only meals much more sharply.",
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
    persona_gamma_offsets=p(
        "personas", "persona_gamma_offsets",
        {"school": {"taste": 0.3, "kid": 1.2, "batch": 1.5, "ing_cost": -1.5},
         "kiosk": {"taste": 1.4, "batch": 1.0, "prep_min": -0.8}},
        "utils/attribute-unit (additive delta on the household base)",
        "Group offsets transcribed from master_table_Z.md's 'school gamma' / 'kiosk gamma' columns: "
        "school upweights kid-acceptance and batch potential and is far more ingredient-cost averse "
        "(institutional budget); kiosk (mama ntilie / small food vendor) upweights taste even further "
        "and is much more prep-labour averse (cooks at commercial scale/pace, batches instead). "
        "Note: master_table_Z.md's prose says school 'downweights taste' but its own table lists "
        "school's taste offset as +0.3 (not negative) -- implemented per the table's number, flagging "
        "the discrepancy for review. Unspecified feature/persona pairs (tradition, kcal, fuelcost for "
        "both; ing_cost/kid for kiosk) get a 0 offset, i.e. inherit the household base unchanged.",
        "Larger |offset| = that persona diverges further from household on that one feature only.",
        tbd=True),
    persona_lam=p(
        "personas", "persona_lam",
        {"school": {"breakfast": -6.0, "lunch": 2.0, "dinner": -6.0}, "kiosk": {}},
        "logit (replacement value, not additive)",
        "Per-stage hazard-bias overrides: school's breakfast/dinner are switched off (-6, matching the "
        "overnight base logit) with a strongly boosted lunch (+2.0), making it essentially unimodal at "
        "lunch (one canteen meal/day). kiosk has no override at all (empty dict) -- master_table_Z.md "
        "gives no vendor operating-hours data, so it currently inherits household's flat, no-bias "
        "timing across all 3 stages. Revisit once real kiosk trading-hours data is available.",
        "An empty dict for a persona means 'same timing as household' -- not necessarily realistic for "
        "a vendor that might only trade at lunch/dinner.",
        tbd=True),
    mix=p("personas", "mix", {"household": 80, "school": 10, "kiosk": 10}, "agent count",
          "Population composition across the three personas.",
          "More schools shifts the aggregate load curve toward a midday peak; more kiosks shift it "
          "toward whatever timing kiosk ends up with (currently household-like, see persona_lam).",
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
            {"nyama_choma_oven_kachumbari": 0.5, "nyama_choma_open_fire": 1.0}, "utils",
            "Flat utility bonus added to the two nyama choma (grilled meat) options -- the classic "
            "Kenyan celebration dish -- biggest for the traditional open-fire one, representing "
            "festival appeal.",
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
    extreme_test_multiplier: float


TARIFF = TariffConfig(
    p_bar=p("tariff", "p_bar", 0.25, "currency/kWh",
            "Common time-average price every candidate tariff is normalised to.",
            "Scales all tariffs up/down together; relative shape (peak/flat/solar) is what's compared.",
            tbd=True),
    p_lo=p("tariff", "p_lo", 0.05, "currency/kWh",
           "Off-peak / low price level used by evening_peak and solar_following.",
           "Lower = cheaper off-peak incentive to shift load away from the peak/evening. Widened from "
           "0.10 alongside p_hi so the peak/off-peak gap is large enough to visibly move both meal "
           "choice and cooking timing (see kappa_price_time).",
           tbd=True),
    p_hi=p("tariff", "p_hi", 0.60, "currency/kWh",
           "Peak price level used by evening_peak and solar_following.",
           "Higher = stronger price signal to avoid cooking electrically at peak times -- more wood "
           "defection and more timing displacement. Raised from 0.45: at 0.45 the peak/off-peak gap "
           "(0.35) gave a real but modest response; 0.60 (gap 0.55 around p_bar=0.25) makes the "
           "evening_peak tariff's effect unmistakable on both the wood_share scoreboard and the "
           "cooking-events-over-time chart.",
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
    extreme_test_multiplier=p("tariff", "extreme_test_multiplier", 5.0, "multiple of p_bar",
                               "Flat price used by the extreme_test tariff candidate: p_bar * this "
                               "multiplier, held constant all day. Deliberately NOT normalised back "
                               "down to p_bar like every other candidate (see sim.tariffs.tariff_"
                               "extreme_test) -- the whole point is an absurdly elevated price *level*, "
                               "not just a reshaped-but-equal-average price. A sanity-check tariff: if "
                               "the model is responding to price correctly, this should crush cooking "
                               "toward near-zero rather than silently no-op at extreme inputs. At the "
                               "current base_gamma_cost/kappa_price_time it does exactly that -- zero "
                               "cook-start events across 100 independent simulated days, not just a low "
                               "wood_share, since Stage 1 (does it fire at all) doesn't know which fuel "
                               "an agent would pick until *after* it fires, so a price this extreme "
                               "suppresses firing altogether rather than diverting it to wood.",
                               "Higher = a harsher stress test; should push meals/day toward 0.",
                               tbd=True),
)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoringConfig:
    R: int


SCORING = ScoringConfig(
    R=p("scoring", "R", 200, "runs",
        "Number of independent Monte Carlo days simulated per tariff.",
        "More runs = a more stable wood_share estimate, at the cost of a slower sweep.",
        tbd=True),
)


DEFAULT_SEED = p("state", "default_seed", 0, "n/a",
                  "Default RNG seed for reproducibility when --seed is not given.",
                  "Same seed + same config = identical output.",
                  tbd=False)
