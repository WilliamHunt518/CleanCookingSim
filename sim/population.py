"""Personas (household/school/kiosk) and per-agent individual sampling.

Three layers, all offsets on the same equations (see sim.agent):
  - base household gamma/gamma_cost/lam
  - sparse persona overrides: gamma is an ADDITIVE offset on the household
    base (missing feature = 0 offset = inherits household); lam is a
    per-stage REPLACEMENT value (missing stage = inherits household's, i.e.
    unbiased timing) -- these are two different combination rules because
    they come from two different sources (master_table_Z.md's group-offset
    gamma columns vs. this project's own hand-tuned firing-hazard biases.
  - per-agent individual draws around the persona mean

Adding a new persona (a fourth agent type) only requires adding an entry to
config.PERSONAS.persona_gamma_offsets / persona_lam / mix and to
PERSONA_NAMES below -- no other code changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim import config
from sim.agent import STAGE_ORDER

ATTR_ORDER = ["taste", "tradition", "kid", "batch", "ing_cost", "prep_min", "kcal", "fuelcost"]
PERSONA_NAMES = ["household", "school", "kiosk"]


def _base_gamma_vector() -> np.ndarray:
    g = config.PERSONAS.base_gamma
    return np.array([g[a] for a in ATTR_ORDER], dtype=float)


def _base_lam_vector() -> np.ndarray:
    lam = config.PERSONAS.base_lam
    return np.array([lam[s] for s in STAGE_ORDER], dtype=float)


def persona_gamma_vector(persona: str) -> np.ndarray:
    """household base + this persona's additive offsets (0 for any unspecified feature)."""
    gamma = _base_gamma_vector().copy()
    offsets = config.PERSONAS.persona_gamma_offsets.get(persona, {})
    for attr, delta in offsets.items():
        gamma[ATTR_ORDER.index(attr)] += delta
    return gamma


def persona_gamma_cost(persona: str) -> float:
    return config.PERSONAS.base_gamma_cost  # no persona override defined for gamma_cost


def persona_lam_vector(persona: str) -> np.ndarray:
    """household base lam, with this persona's per-stage REPLACEMENT values applied."""
    lam = _base_lam_vector().copy()
    overrides = config.PERSONAS.persona_lam.get(persona, {})
    for stage, val in overrides.items():
        lam[STAGE_ORDER.index(stage)] = val
    return lam


@dataclass
class Population:
    persona_idx: np.ndarray  # (N,) index into PERSONA_NAMES
    gamma: np.ndarray        # (N, len(ATTR_ORDER)) individual taste-weight vectors
    gamma_cost: np.ndarray   # (N,) individual price sensitivity
    lam: np.ndarray          # (N, 3) persona-level per-stage hazard bias (not individually sampled)

    @property
    def n_agents(self) -> int:
        return self.persona_idx.shape[0]


def _persona_counts(n_agents: int) -> dict:
    mix = config.PERSONAS.mix
    total = sum(mix.values())
    counts = {name: int(round(n_agents * cnt / total)) for name, cnt in mix.items()}
    diff = n_agents - sum(counts.values())
    largest = max(counts, key=counts.get)
    counts[largest] += diff
    return counts


def build_population(rng: np.random.Generator, n_agents: int | None = None,
                      no_personas: bool = False, no_cost: bool = False) -> Population:
    """Sample a population. Ablation flags: no_personas forces everyone to base
    household; no_cost zeroes every agent's price sensitivity (gamma_cost)."""
    n_agents = config.N_AGENTS if n_agents is None else n_agents

    if no_personas:
        counts = {name: 0 for name in PERSONA_NAMES}
        counts["household"] = n_agents
    else:
        counts = _persona_counts(n_agents)

    persona_idx = np.concatenate([
        np.full(counts[name], PERSONA_NAMES.index(name), dtype=int) for name in PERSONA_NAMES
    ])
    rng.shuffle(persona_idx)

    base_gamma = np.stack([persona_gamma_vector(p) for p in PERSONA_NAMES])   # (n_personas, n_attrs)
    base_gcost = np.array([persona_gamma_cost(p) for p in PERSONA_NAMES])    # (n_personas,)
    base_lam = np.stack([persona_lam_vector(p) for p in PERSONA_NAMES])      # (n_personas, 3)

    sigma = config.PERSONAS.sigma_ind
    mean_gamma = base_gamma[persona_idx]          # (N, n_attrs)
    gamma = rng.normal(loc=mean_gamma, scale=sigma)
    gamma_cost = rng.normal(loc=base_gcost[persona_idx], scale=sigma)
    if no_cost:
        gamma_cost = np.zeros_like(gamma_cost)
    gamma_cost = np.clip(gamma_cost, 0.0, None)

    lam = base_lam[persona_idx]  # persona-level, not individually sampled

    return Population(persona_idx=persona_idx, gamma=gamma, gamma_cost=gamma_cost, lam=lam)
