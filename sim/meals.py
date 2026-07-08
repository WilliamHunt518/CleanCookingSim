"""Meal table (K=18, from master_table_Z.md) and power profiles phi_k.

All numeric values come from sim.config -- this module only assembles them
into arrays convenient for vectorised agent code, and provides the duration
sampler + power-profile shapes used by demand assembly.
"""
from __future__ import annotations

import numpy as np

from sim import config

MEAL_NAMES = [m.name for m in config.MEALS]
IDX_BY_NAME = {name: i for i, name in enumerate(MEAL_NAMES)}  # 0-based array index
K = config.STATE.K
assert K == len(config.MEALS), "config.STATE.K must match len(config.MEALS)"

# z_k attribute matrix, columns = ATTR_ORDER (see sim.population), rows = meals (0-based).
# taste/tradition/kid/batch come straight from master_table_Z.md (already ~0-1). ing_cost,
# prep_min and kcal are normalised by the *_MAX reference constants so they sit on a
# comparable scale. fuelcost is reconstructed from charcoal_kes for fire-only meals only
# (0 for electric meals -- they don't buy charcoal).
_ing_cost_norm = np.array([m.ing_cost_kes for m in config.MEALS], dtype=float) / config.ING_COST_MAX_KES
_prep_min_norm = np.array([m.prep_min for m in config.MEALS], dtype=float) / config.PREP_MIN_MAX
_kcal_norm = np.array([m.kcal for m in config.MEALS], dtype=float) / config.KCAL_MAX
_fuelcost_norm = np.array([
    (m.charcoal_kes / config.CHARCOAL_KES_MAX) if m.fire_only else 0.0 for m in config.MEALS
], dtype=float)

Z = np.column_stack([
    np.array([m.taste for m in config.MEALS], dtype=float),
    np.array([m.tradition for m in config.MEALS], dtype=float),
    np.array([m.kid for m in config.MEALS], dtype=float),
    np.array([m.batch for m in config.MEALS], dtype=float),
    _ing_cost_norm,
    _prep_min_norm,
    _kcal_norm,
    _fuelcost_norm,
])

E_KWH = np.array([m.e_kwh for m in config.MEALS], dtype=float)
ALPHA_K = config.ALPHA_SCALE * _kcal_norm
WOOD_MASK = np.array([m.fire_only for m in config.MEALS], dtype=bool)

# Reference (unnormalised) metadata, for display/explainability only -- not fed to gamma.z.
KCAL = np.array([m.kcal for m in config.MEALS], dtype=float)
PROTEIN_G = np.array([m.protein_g for m in config.MEALS], dtype=float)
CARB_G = np.array([m.carb_g for m in config.MEALS], dtype=float)
FAT_G = np.array([m.fat_g for m in config.MEALS], dtype=float)
ING_COST_KES = np.array([m.ing_cost_kes for m in config.MEALS], dtype=float)
CHARCOAL_KES = np.array([m.charcoal_kes for m in config.MEALS], dtype=float)
MEAL_TYPE = [m.meal_type for m in config.MEALS]

BLOCK_HOURS = config.STATE.block_minutes / 60.0
DBAR_BLOCKS = np.array([m.dbar_min for m in config.MEALS], dtype=float) / config.STATE.block_minutes
SIGMA_BLOCKS = config.MEAL_DURATION_SIGMA_MIN / config.STATE.block_minutes
DMAX_BLOCKS = int(round(config.MEAL_DURATION_MAX_MIN / config.STATE.block_minutes))


def sample_durations_blocks(meal_indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """meal_indices: 0-based array index per cook event. Returns integer block counts, clipped [1, Dmax]."""
    dbar = DBAR_BLOCKS[meal_indices]
    draws = rng.normal(loc=dbar, scale=SIGMA_BLOCKS)
    blocks = np.round(draws).astype(int)
    return np.clip(blocks, 1, DMAX_BLOCKS)


def boxcar_shape(duration_blocks: int, e_kwh: float) -> np.ndarray:
    if duration_blocks <= 0:
        return np.zeros(0)
    power_kw = e_kwh / (duration_blocks * BLOCK_HOURS)
    return np.full(duration_blocks, power_kw)


def preheat_simmer_shape(duration_blocks: int, e_kwh: float) -> np.ndarray:
    """Short high-power spike then a lower simmer level; conserves total energy."""
    if duration_blocks <= 0:
        return np.zeros(0)
    boxcar_power = e_kwh / (duration_blocks * BLOCK_HOURS)
    n_spike = max(1, int(round(config.PREHEAT_SPIKE_FRAC * duration_blocks)))
    n_spike = min(n_spike, duration_blocks)
    spike_power = boxcar_power * config.PREHEAT_POWER_MULT
    n_simmer = duration_blocks - n_spike
    total_energy_blocks = e_kwh / BLOCK_HOURS  # kW*block units
    spike_energy_blocks = spike_power * n_spike
    remaining = total_energy_blocks - spike_energy_blocks
    simmer_power = max(remaining, 0.0) / n_simmer if n_simmer > 0 else 0.0
    return np.concatenate([np.full(n_spike, spike_power), np.full(n_simmer, simmer_power)])


PROFILE_SHAPES = {"boxcar": boxcar_shape, "preheat_simmer": preheat_simmer_shape}


def power_profile(meal_idx0: int, duration_blocks: int) -> np.ndarray:
    """kW profile for one cook of meal (0-based index) lasting duration_blocks."""
    shape_fn = PROFILE_SHAPES[config.MEAL_PROFILE_SHAPE]
    return shape_fn(duration_blocks, E_KWH[meal_idx0])
