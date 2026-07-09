# Grid energy-balance model

Module estimating how much energy the mini-grid has available, combining a
real PV forecast, agent usage, and battery capacity.

This file explains the *reasoning* (why the equation is unbounded, why only
one tracker resets daily, why PV comes from a real forecast). For a complete
parameter-by-parameter reference of the callable API (`GridEnergyComponent`
and everything it returns), see **[`COMPONENT_API.md`](COMPONENT_API.md)**.

## The equation, reformulated

You proposed:

```
socs = (PV - usage) / BC
```

`PV` and `usage` are **power** (kW) at a single instant/block, and `BC` is
an **energy** capacity (kWh), so `(PV - usage) / BC` has units of 1/hour --
it's the *rate* at which the battery would fill or drain right now, not a
state of charge. Computed fresh each block, it also has no memory: it can't
represent energy actually sitting in the battery from previous blocks.

A state of charge is a **stock**, not a flow, so it has to *integrate* the
net power balance over time:

```
E(0) = (soc_init_pct / 100) * BC
E(t) = E(t-1) + (PV(t) - usage(t)) * dt          dt = block length in hours

socs(t) = E(t) / BC * 100          <-- your equation, integrated, as a %
```

This `E(t)` is left **unclipped** on purpose, exactly as you asked: it can
run past 100% (battery already "full" and then some) or below 0%. That is
`socs_pct` in `soc.py`:

- **`socs(t) > 100%`** -- more energy has been generated than the battery
  could ever hold. The excess, `((socs(t) - 100) / 100) * BC` kWh, is PV
  that would be curtailed by a physically capped battery -- or that could
  power extra load (e-cooking, in this project's context) directly instead
  of ever touching storage.
- **`socs(t) < 0%`** -- cumulative usage has outrun cumulative generation
  plus the starting charge -- an energy deficit only a backup source or
  load-shedding could cover in reality.

Because a real battery *can't* actually exceed its capacity or go negative,
`soc.py` also computes the physically realisable counterpart,
`actual_soc_pct`: the same integration, clipped to `[0, BC]` every block, so
any overflow/shortfall is dropped at that block instead of carried forward
(tracked separately as `surplus_kwh` / `deficit_kwh`). Comparing the two is
the diagnostic you described -- "how much potential energy do we have when
people use energy while the battery is already full" is exactly
`socs_pct - actual_soc_pct` at that block.

`actual_soc_pct` is the real, physical battery -- it carries its charge
continuously from one block to the next, day after day, and is **never**
reset. `socs_pct` (`reset_daily=True`, the default in `compute_soc`)
restarts at the beginning of every calendar day, but *from wherever the real
battery actually is that morning*, not from a fixed baseline -- so it reads
as "given today's real starting charge, how far past 100%/below 0% did
today's own PV-usage balance push things," i.e. a per-day surplus/deficit
view layered on top of the one real, continuously-carried battery. See
`soc.py`'s docstring for the full reasoning.

## PV(t): a real forecast, not an assumption

`quartz_forecast.py` wraps the
[`quartz-solar-forecast`](https://github.com/openclimatefix/open-source-quartz-solar-forecast)
library (docs: https://open.quartz.solar/docs) -- a gradient-boosted model
trained on real PV site history, driven by Open-Meteo NWP weather data (no
API key needed). An earlier assumption-based clear-sky-shape + synthetic-
weather-index model (`weather.py`/`pv.py`) was retired once this was wired
in; there is now exactly one PV source in this folder.

```
pip install quartz-solar-forecast
python -m grid_energy.quartz_forecast --plot
```

`quartz_solar_forecast.forecast.run_forecast(site, ts, ...)` returns
`power_kw` at **15-minute** resolution for a fixed **48-hour** horizon
starting at `ts`. It has no "give me a week" option, so `forecast_week_kw()`
builds a full week by calling it once per day (7 daily anchors at 00:00 UTC,
keeping each call's first 24h/96 blocks) and concatenating -- each day is
predicted from the freshest NWP run available for it, rather than trusting
one 7-day-old 48h forecast.

For the requested site (`lat=-0.80895, lon=36.24232` -- see `config.SITE`,
`config.PV.rated_kwp`), one run on 2026-07-09 returned:

```
forecast window      : 2026-07-09 00:00 .. 2026-07-15 23:45  (672 x 15-min blocks)
peak power            : 12.49 kW
total weekly energy   : 130,169 Wh  (130.17 kWh)
mean daily energy     : 18,596 Wh/day
```

This is a live weather-dependent forecast, so re-running it on a different
day will return a different number -- that's the point.

### Two things worth knowing before running it

**Install-time gotcha, worked around in `quartz_forecast.py`, not upstream:**
a plain `pip install quartz-solar-forecast` currently pulls in `attrs`/
`cattrs` releases new enough that `requests_cache`'s response serializer
raises `NameError: name 'RequestsCookieJar' is not defined` while trying to
disk-cache the Open-Meteo HTTP response -- a forward-reference resolution
bug in that specific dependency combination, not in quartz-solar-forecast's
own code (downgrading `attrs`/`cattrs` alone did not fix it in testing).
Since on-disk response caching isn't needed for a handful of calls a week,
`quartz_forecast._patch_requests_cache()` swaps `requests_cache.CachedSession`
for a plain `requests.Session` before the library's HTTP calls happen,
sidestepping the broken serializer entirely. This patch is applied
automatically the first time `forecast_day_kw`/`forecast_week_kw` is called
-- no manual pinning needed. Also note: this library needs **Python <=3.11**
(some of its pinned ML dependencies don't yet publish wheels for newer
interpreters) -- use a 3.11 virtualenv if your default `python3` is newer.

**Mutable-`PVSite` gotcha, fixed here:** `quartz_solar_forecast.forecast.
predict_ocf` mutates its `PVSite.capacity_kwp` to `4` in place whenever the
real capacity is > 4 kWp (the model is only trained up to 4 kWp, so it runs
at 4 kWp and rescales the output afterwards). `forecast_day_kw` therefore
builds a **fresh** `PVSite` on every one of the week's 7 calls -- reusing one
`PVSite` object across calls used to mean every call after the first saw the
already-mutated `capacity_kwp=4`, silently skipped rescaling, and returned a
forecast for a 4 kWp system while claiming to be for the real (much larger)
one. Covered by `test_forecast_day_builds_a_fresh_site_every_call`.

## Using this as a component (e.g. from `sim`, later)

`GridEnergyComponent` (`component.py`) is the single entry point this folder
is meant to be driven through by anything outside it. `grid_energy` still
imports nothing from `sim` -- the intended direction is `sim -> grid_energy`,
never the reverse, so wiring it in later is additive, not a refactor of
either side:

```python
from grid_energy import GridEnergyComponent

component = GridEnergyComponent()                            # Oloika defaults, see config.py
day = sim.run.simulate_day(population, price, scenario, rng)  # sim's own 5-min-block demand_kw
result = component.compute_soc_for_usage(day.demand_kw, usage_block_minutes=5.0)

result.socs_pct        # today's surplus/deficit, % of BC, unbounded, resets daily
result.actual_soc_pct  # the real battery, % of BC, [0, 100], carries over continuously
result.surplus_kwh     # per-block energy beyond what the battery could store
result.deficit_kwh     # per-block unmet demand once the battery was empty
```

`usage_kw` can be at *any* fixed block size -- `compute_soc_for_usage`
resamples it to `quartz_forecast`'s native 15-minute resolution
(`resample.py`: averages on downsampling, repeats on upsampling, so mean
power is preserved either way) and tiles/truncates it to the forecast's
length (e.g. one simulated day repeated across a week). This is exactly the
`weekly_usage_kw_from_sim`/manual-reshape logic `demo_forecast_week.py` used
to do inline, now behind one reusable method so a future `sim` caller
doesn't have to reimplement it.

`GridEnergyComponent()` with no arguments uses `config.SITE`/`config.PV`/
`config.BATTERY`'s Oloika defaults; every field (`latitude`, `longitude`,
`capacity_kwp`, `capacity_kwh`, `soc_init_pct`, `nwp_source`) can be
overridden per instance for a different scenario without touching `config.py`.

## Files

```
grid_energy/config.py             site / PV / battery / timing parameters + Oloika defaults
grid_energy/quartz_forecast.py    PV(t) from quartz-solar-forecast, real weekly 15-min forecast
grid_energy/soc.py                socs(t) -- the core equation, unbounded + clipped variants, daily reset
grid_energy/resample.py           block-size resampling + tiling (5-min sim <-> 15-min forecast)
grid_energy/component.py          GridEnergyComponent -- the single entry point, see above
grid_energy/demo_forecast_week.py runnable example + plot, one real week: python -m grid_energy.demo_forecast_week
grid_energy/tests/                pytest -- conservation, >100%/<0% behaviour, clipping bounds, daily reset, forecast wiring, resampling, component
grid_energy/COMPONENT_API.md      complete parameter-by-parameter API reference (types, units, errors, worked examples)
```

## Run it

```
pip install quartz-solar-forecast              # needs Python <=3.11, see above

python -m grid_energy.quartz_forecast --plot    # real weekly PV forecast only, prints a summary + PNG
python -m grid_energy.demo_forecast_week        # GridEnergyComponent worked example: PV forecast + sim usage -> socs(t)
python -m pytest grid_energy/tests -v
```

## What this deliberately does not model

Battery efficiency losses (charge/discharge round-trip efficiency < 100%),
battery degradation/aging, temperature effects on PV or battery
performance, and curtailment logic (what actually happens to `surplus_kwh`
-- export, dump load, or waste). These were out of scope for this
first-pass model; `surplus_kwh` / `deficit_kwh` are exposed specifically so
a later pass can plug in what should happen to them.
