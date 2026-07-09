"""
Resolution-matching helpers: the piece a future caller needs to line up its
own usage_kw timeseries (e.g. sim.run.simulate_day's 5-minute demand_kw)
against quartz_forecast's fixed 15-minute PV resolution before compute_soc
can run. Deliberately generic -- no dependency on `sim` or any other block
size, so any fixed-block-size producer can be wired in later.
"""
from __future__ import annotations

import numpy as np


def resample_kw(series_kw: np.ndarray, from_block_minutes: float, to_block_minutes: float) -> np.ndarray:
    """Resample a power [kW] series from one fixed block size to another.

    Downsampling (from_block_minutes < to_block_minutes) averages each
    consecutive group of blocks -- e.g. 5-min blocks -> 15-min blocks
    averages every 3. Upsampling repeats each block -- e.g. 15-min -> 5-min
    repeats each value 3x. Averaging (not summing) on downsampling and
    repeating (not spreading) on upsampling both preserve mean power, which
    is what compute_soc needs (it multiplies by its own block_minutes to get
    per-block energy) -- only integer ratios are supported, since anything
    else would need interpolation this module deliberately doesn't do.
    """
    series_kw = np.asarray(series_kw, dtype=float)
    if from_block_minutes == to_block_minutes:
        return series_kw

    ratio = to_block_minutes / from_block_minutes
    if ratio > 1:
        factor = round(ratio)
        if not np.isclose(ratio, factor):
            raise ValueError(f"to_block_minutes/from_block_minutes ({ratio}) must be an integer "
                              f"for downsampling, got from={from_block_minutes}, to={to_block_minutes}")
        if len(series_kw) % factor != 0:
            raise ValueError(f"series length {len(series_kw)} is not divisible by the downsampling "
                              f"factor {factor} (from {from_block_minutes}min to {to_block_minutes}min blocks)")
        return series_kw.reshape(-1, factor).mean(axis=1)
    else:
        factor = round(1.0 / ratio)
        if not np.isclose(1.0 / ratio, factor):
            raise ValueError(f"from_block_minutes/to_block_minutes ({1 / ratio}) must be an integer "
                              f"for upsampling, got from={from_block_minutes}, to={to_block_minutes}")
        return np.repeat(series_kw, factor)


def tile_to_length(series_kw: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat (tile) a shorter series to cover target_len blocks, truncating the final
    repeat -- used to stretch e.g. one simulated day's usage across a full week of PV
    forecast blocks. No-op (truncates) if series_kw is already >= target_len."""
    series_kw = np.asarray(series_kw, dtype=float)
    if len(series_kw) == 0:
        raise ValueError("series_kw must be non-empty")
    if len(series_kw) >= target_len:
        return series_kw[:target_len]
    reps = -(-target_len // len(series_kw))  # ceil division
    return np.tile(series_kw, reps)[:target_len]
