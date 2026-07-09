"""
Grid state-of-charge model.

    socs(t) = E_potential(t) / BC * 100                      [%, UNBOUNDED]

where E_potential is a running, UNCLIPPED energy balance:

    E_potential(0) = soc_init_pct/100 * BC
    E_potential(t) = E_potential(t-1) + (PV(t) - Usage(t)) * dt

This is the corrected form of the requested `socs = (PV - usage) / BC`:
that expression alone is an instantaneous power *rate* normalised by
capacity (units 1/hour, not a percentage) and has no memory -- computed
block by block it would just reset every block, so it can never represent
charge actually sitting in the battery. A state of charge is a *stock*, not
a flow, so it has to integrate the net power balance (PV - usage) over
time. socs(t) above is exactly that integral, expressed as a % of BC, and
it is deliberately left unclipped so it can go:

  - above 100%: cumulative net energy since t=0 exceeds what BC could ever
    hold -- the battery would already be saturated, and the excess,
    ((socs(t) - 100) / 100) * BC kWh, is surplus PV generation that's either
    curtailed or available to power extra load (e.g. e-cooking) directly,
    without ever touching the battery. This is the "surplus" case the spec
    asked for.
  - below 0%: cumulative usage has outrun cumulative generation plus the
    initial charge -- an energy deficit that a real system could only cover
    with a backup source or by shedding load.

`actual_soc_pct` is the physically realisable counterpart: the same
integration, but clipped to [0, BC] at every block (a real battery can
neither overcharge past 100% nor discharge past empty), so any surplus
above 100% or shortfall below 0% is *lost* at that block rather than carried
forward. Comparing the unbounded `socs_pct` ("potential") against the
clipped `actual_soc_pct` ("realised") is the diagnostic requested: it shows
how much power was demanded or available beyond what the battery could
actually deliver or absorb, block by block (`surplus_kwh` / `deficit_kwh`).

By default (`reset_daily=True`) only `E_potential` (the unbounded tracker
behind `socs_pct`) restarts at the beginning of every calendar day. `E_actual`
(the physically realisable, clipped tracker behind `actual_soc_pct`) is
**never** reset except once at t=0 -- a real battery's charge carries over
from one day to the next, it doesn't get wiped at midnight. So each day,
`E_potential` restarts from wherever `E_actual` actually is *right then*
(not from a fixed `soc_init_pct` baseline), and grows unclipped for that day
alone. That makes `socs_pct` read as "starting from where the real battery
truly is this morning, how far past 100% (surplus) or below 0% (deficit)
would today's own PV-usage balance push it" -- a per-day figure -- while
`actual_soc_pct` keeps tracking the real, continuously-carried battery charge
across the whole run. `surplus_kwh`/`deficit_kwh` (computed from the clipped
tracker, which is never reset) are therefore already per-block/per-day
figures regardless of `reset_daily`; what `reset_daily` changes is only how
far back `socs_pct` looks when judging "surplus". Pass `reset_daily=False`
to make `E_potential` also run as one continuous integration from t=0
(the original, pre-daily-reset behaviour) instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class SOCResult:
    t_hr: np.ndarray
    pv_kw: np.ndarray
    usage_kw: np.ndarray
    net_kw: np.ndarray                  # PV(t) - Usage(t)
    energy_potential_kwh: np.ndarray    # unclipped running energy balance
    socs_pct: np.ndarray                # energy_potential / BC * 100, UNBOUNDED
    energy_actual_kwh: np.ndarray       # same integration, clipped to [0, BC]
    actual_soc_pct: np.ndarray          # energy_actual / BC * 100, in [0, 100]
    surplus_kwh: np.ndarray             # per-block energy that overflowed a full battery
    deficit_kwh: np.ndarray             # per-block unmet demand once the battery was empty


def compute_soc(pv_kw: np.ndarray, usage_kw: np.ndarray, block_minutes: float | None = None,
                 capacity_kwh: float | None = None, soc_init_pct: float | None = None,
                 reset_daily: bool = True) -> SOCResult:
    """Run the socs model over PV(t)/Usage(t) timeseries of equal length T.

    reset_daily: if True (default), the unbounded E_potential tracker (socs_pct)
    restarts every 24h/block_minutes blocks from wherever the real, clipped
    battery (E_actual / actual_soc_pct) currently is -- so socs_pct reads as
    "this day's own surplus/deficit relative to today's actual starting
    charge". E_actual itself is never reset (the battery's charge always
    carries over day to day) -- see module docstring. If False, E_potential
    also runs as one continuous integration from t=0 instead.
    """
    pv_kw = np.asarray(pv_kw, dtype=float)
    usage_kw = np.asarray(usage_kw, dtype=float)
    if pv_kw.shape != usage_kw.shape:
        raise ValueError(f"pv_kw and usage_kw must be the same shape, got {pv_kw.shape} vs {usage_kw.shape}")

    block_minutes = config.TIME.block_minutes if block_minutes is None else block_minutes
    capacity_kwh = config.BATTERY.capacity_kwh if capacity_kwh is None else capacity_kwh
    soc_init_pct = config.BATTERY.soc_init_pct if soc_init_pct is None else soc_init_pct

    dt_hr = block_minutes / 60.0
    T = pv_kw.shape[0]
    t_hr = np.arange(T) * dt_hr
    blocks_per_day = round(24.0 / dt_hr) if reset_daily else None

    net_kw = pv_kw - usage_kw
    d_energy_kwh = net_kw * dt_hr
    e_init = (soc_init_pct / 100.0) * capacity_kwh

    energy_potential = np.empty(T)
    energy_actual = np.empty(T)
    surplus = np.zeros(T)
    deficit = np.zeros(T)
    e_prev_potential = e_init
    e_prev_actual = e_init  # the real, physically-clipped battery charge -- carries over day to day, never reset
    for i in range(T):
        if blocks_per_day and i % blocks_per_day == 0:
            # start each day's "surplus" tracker from wherever the real battery actually is right now
            e_prev_potential = e_prev_actual

        e_prev_potential = e_prev_potential + d_energy_kwh[i]
        energy_potential[i] = e_prev_potential

        e_new = e_prev_actual + d_energy_kwh[i]
        if e_new > capacity_kwh:
            surplus[i] = e_new - capacity_kwh
            e_new = capacity_kwh
        elif e_new < 0.0:
            deficit[i] = -e_new
            e_new = 0.0
        energy_actual[i] = e_new
        e_prev_actual = e_new

    socs_pct = energy_potential / capacity_kwh * 100.0
    actual_soc_pct = energy_actual / capacity_kwh * 100.0

    return SOCResult(t_hr=t_hr, pv_kw=pv_kw, usage_kw=usage_kw, net_kw=net_kw,
                      energy_potential_kwh=energy_potential, socs_pct=socs_pct,
                      energy_actual_kwh=energy_actual, actual_soc_pct=actual_soc_pct,
                      surplus_kwh=surplus, deficit_kwh=deficit)
