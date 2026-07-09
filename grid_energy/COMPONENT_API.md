# `grid_energy` component API reference

This document is a complete, self-contained reference for calling
`grid_energy` as a component from other code (e.g. `sim`). It is written so
that a reader with no other context on this repository can call the API
correctly on the first try. For the *reasoning* behind the model (why the
equation is unbounded, why only one tracker resets daily, why PV comes from
a real forecast instead of an assumption), see `README.md` in this same
directory — this file only covers the callable surface: functions, classes,
parameters, return types, units, and error conditions.

If you only read one section, read "Quick reference" and "Worked example."

---

## Quick reference

```python
from grid_energy import GridEnergyComponent

component = GridEnergyComponent()  # all Oloika, Kenya defaults; every field is overridable
result = component.compute_soc_for_usage(
    usage_kw=my_usage_array,        # numpy array, kW, any fixed block size
    usage_block_minutes=5.0,        # the block size of my_usage_array
)

result.socs_pct        # np.ndarray, %, UNBOUNDED (can exceed 100 or go below 0), resets daily
result.actual_soc_pct  # np.ndarray, %, bounded [0, 100], NEVER resets (real battery, carries over)
result.surplus_kwh     # np.ndarray, kWh, per 15-min block, energy that overflowed a full battery
result.deficit_kwh     # np.ndarray, kWh, per 15-min block, unmet demand once battery was empty
result.pv_kw           # np.ndarray, kW, the real PV forecast, resampled to match result's length
result.usage_kw         # np.ndarray, kW, your usage_kw, resampled/tiled to 15-min/672-block week
```

This one call makes **7 live HTTP requests** to Open-Meteo (no API key
needed) and takes a few seconds. Everything below explains every parameter,
every returned field, and every failure mode of this call.

---

## Requirements before calling anything

- **Python <= 3.11.** The `quartz-solar-forecast` package's pinned ML
  dependencies do not publish wheels for newer interpreters (as of this
  writing). Calling any function that fetches a forecast (`GridEnergyComponent.
  forecast_pv_week`, `GridEnergyComponent.compute_soc_for_usage` without a
  pre-fetched `forecast=`, `quartz_forecast.forecast_week_kw`,
  `quartz_forecast.forecast_day_kw`) under a newer interpreter will fail at
  `pip install` time or at the `import quartz_solar_forecast...` line inside
  these functions.
- **`pip install quartz-solar-forecast`** must be run first. This is a heavy
  dependency (pulls in `xarray`, gradient-boosted-tree libraries, etc.) and
  is **not required just to import `grid_energy`** -- see "Import-time vs
  call-time dependencies" below.
- **Internet access** is required at call time (Open-Meteo weather API, no
  key). There is no offline/cached mode.
- `numpy` and `pandas` are required unconditionally (used even without
  `quartz-solar-forecast` installed, e.g. for `soc.compute_soc` and
  `resample.py`).

### Import-time vs call-time dependencies

`import grid_energy` (or any individual module in this folder) **never**
requires `quartz-solar-forecast` to be installed. The heavy import
(`from quartz_solar_forecast.forecast import run_forecast`) is deferred to
inside the function bodies that actually need it. This means:

- You can `from grid_energy import GridEnergyComponent, compute_soc` and
  construct objects freely without `quartz-solar-forecast` installed.
- Calling `component.forecast_pv_week()` or `component.compute_soc_for_usage(...)`
  **without** a pre-fetched `forecast=` argument will raise
  `ModuleNotFoundError: No module named 'quartz_solar_forecast'` if the
  package isn't installed.
- Calling `component.compute_soc_for_usage(usage_kw, usage_block_minutes, forecast=some_forecast_result)`
  with a manually-constructed or previously-fetched `ForecastResult` never
  touches `quartz_solar_forecast` at all, so it works with only
  `numpy`/`pandas` installed. Useful for tests or for callers that fetch the
  forecast once and reuse it (see `test_component.py` for the pattern).

---

## `GridEnergyComponent` (`grid_energy/component.py`)

The single entry point. A `dataclass` -- construct it with keyword arguments
(or none, for all defaults), then call its two methods.

```python
from grid_energy import GridEnergyComponent
component = GridEnergyComponent(capacity_kwp=50.0, capacity_kwh=100.0)
```

### Constructor parameters

| field | type | default | meaning |
|---|---|---|---|
| `latitude` | `float \| None` | `None` -> `config.SITE.latitude` (`-0.80895`) | Site latitude, decimal degrees. Passed straight to `quartz-solar-forecast`'s `PVSite.latitude` (valid range `[-90, 90]`, enforced by that library, not by `grid_energy`). |
| `longitude` | `float \| None` | `None` -> `config.SITE.longitude` (`36.24232`) | Site longitude, decimal degrees. Valid range `[-180, 180]`. |
| `capacity_kwp` | `float \| None` | `None` -> `config.PV.rated_kwp` (`60.0`) | Nameplate PV array capacity, kilowatts-peak. The underlying forecast model is only trained up to 4 kWp; `quartz-solar-forecast` internally runs at 4 kWp and rescales the output linearly by `capacity_kwp / 4` when `capacity_kwp > 4` (this is handled correctly per-call by `grid_energy`, see "Known upstream gotchas" below -- you do not need to do anything about it). |
| `capacity_kwh` | `float \| None` | `None` -> `config.BATTERY.capacity_kwh` (`~27.8`, `= config.PV.PV_max_kwp * 25/54`) | Battery capacity `BC`, kilowatt-hours. This is the `BC` in the `socs` equation, and is only used by `compute_soc_for_usage` (has no effect on `forecast_pv_week`). Deliberately *smaller* than a naive 54/25 (Oloika's real kWh/kWp) scaling would give -- Oloika's actual battery bank is oversized relative to what this site needs, so scaling proportionally to the *inverse* ratio keeps a bigger PV array from getting an equally oversized battery by default. |
| `soc_init_pct` | `float \| None` | `None` -> `config.BATTERY.soc_init_pct` (`0.0`) | Starting battery charge, percent of `capacity_kwh`, at `t=0` of the run (i.e. the first block of the returned series only -- see `soc_init_pct` semantics under `compute_soc` below for how this interacts with daily resets). |
| `nwp_source` | `str` | `"icon"` | Which numerical-weather-prediction dataset `quartz-solar-forecast` pulls from Open-Meteo. One of `"icon"`, `"gfs"`, `"ukmo"`, `"ecmwf"`. `"icon"` is the library's own default and is what `grid_energy` also defaults to. Only affects `forecast_pv_week`/PV data, never `usage_kw`. |

Every field with a `None` default falls back to a module-level default in
`grid_energy/config.py` (`SITE`, `PV`, `BATTERY` dataclass instances) *at
call time inside the methods*, not at construction time -- so mutating
`grid_energy.config.PV.rated_kwp` after constructing a `GridEnergyComponent()`
with no `capacity_kwp` override will NOT be picked up (dataclasses are
frozen; `config.PV` itself would need to be replaced, which is unusual and
not a supported workflow -- prefer passing the value explicitly to the
constructor).

### Method: `forecast_pv_week(start=None) -> ForecastResult`

Fetches one real week of PV(t) forecast at 15-minute resolution.

**Parameters:**

| name | type | default | meaning |
|---|---|---|---|
| `start` | `datetime \| pandas.Timestamp \| str \| None` | `None` -> "now, in UTC, floored to midnight" | The first day of the 7-day forecast window. If a `str`, must be parseable by `pandas.Timestamp(...)` (e.g. `"2026-07-09"`). The forecast covers `[start, start + 7 days)`. |

**Returns:** `ForecastResult` (see below).

**Side effects:** makes 7 live HTTP requests to Open-Meteo (one per day of
the week). Takes roughly 2-5 seconds total under normal network conditions.

**Raises:**
- `ModuleNotFoundError` if `quartz-solar-forecast` is not installed.
- Whatever `quartz_solar_forecast.forecast.run_forecast` itself raises on
  malformed inputs (e.g. `latitude`/`longitude` out of range raises a
  `pydantic.ValidationError` from that library, not from `grid_energy`).
- Network errors (`requests.exceptions.*`) propagate uncaught if Open-Meteo
  is unreachable.

### Method: `compute_soc_for_usage(usage_kw, usage_block_minutes, forecast=None, reset_daily=True) -> SOCResult`

The main method: aligns an arbitrary usage timeseries against a real PV
forecast and runs the `socs` state-of-charge model.

**Parameters:**

| name | type | required? | meaning |
|---|---|---|---|
| `usage_kw` | `array-like of float` (list, `np.ndarray`, or anything `np.asarray`-convertible) | yes | Power drawn by the load, kilowatts, at a **fixed** block size given by `usage_block_minutes`. Any positive length is accepted -- it does not need to already be a week or already be at 15-minute resolution; see "Resampling and alignment" below. |
| `usage_block_minutes` | `float` | yes | The fixed time step, in minutes, between consecutive `usage_kw` samples. E.g. `5.0` for `sim`'s native block size (`sim.config.STATE.block_minutes`), `15.0` if already matching the PV forecast's resolution, `60.0` for hourly data. **Must produce an integer ratio** against `15.0` (`quartz_forecast.BLOCK_MINUTES`) -- see `resample_kw` below for the exact rule and error case. |
| `forecast` | `ForecastResult \| None` | no, default `None` | A pre-fetched forecast (from an earlier call to `forecast_pv_week()`, or a hand-built `ForecastResult` for testing). If `None`, this method calls `self.forecast_pv_week()` internally (making the 7 live HTTP requests described above). **Pass this explicitly when calling `compute_soc_for_usage` multiple times against the same week** (e.g. sweeping many usage scenarios / candidate tariffs against one PV forecast) to avoid re-fetching identical data every time. |
| `reset_daily` | `bool` | no, default `True` | Forwarded directly to `soc.compute_soc`'s `reset_daily` parameter -- see the "`reset_daily` semantics" section below. Almost always leave this at the default. |

**Returns:** `SOCResult` (see below), always at `quartz_forecast.BLOCK_MINUTES`
(15-minute) resolution and `len(forecast.power_kw)` blocks long (672 for a
full untruncated week), **regardless of the length or resolution of the
`usage_kw` you passed in**.

**Raises:**
- Everything `forecast_pv_week` can raise, if `forecast=None`.
- `ValueError` from `resample_kw` if `usage_block_minutes` and `15.0` do not
  form an integer ratio, or (for downsampling, i.e.
  `usage_block_minutes < 15.0`) if `len(usage_kw)` is not evenly divisible
  by that ratio. See `resample_kw`'s own section for exact wording and
  worked numeric examples.
- `ValueError` from `tile_to_length` if `usage_kw` is empty after resampling
  (i.e. the input `usage_kw` was empty).

#### Resampling and alignment (what happens to `usage_kw` internally)

`compute_soc_for_usage` performs exactly two transformations on `usage_kw`
before running the SOC model, in this order:

1. **`resample_kw(usage_kw, usage_block_minutes, 15.0)`** — converts your
   block size to 15 minutes. If `usage_block_minutes < 15.0` (finer than the
   forecast, e.g. `sim`'s 5-minute blocks), consecutive blocks are
   **averaged** in groups (5-min -> 15-min averages every 3). If
   `usage_block_minutes > 15.0` (coarser, e.g. hourly), each value is
   **repeated** to fill the finer grid (60-min -> 15-min repeats each value
   4x). Averaging/repeating (not summing/spreading) is chosen specifically
   because it preserves mean *power*, which is what `compute_soc` needs (it
   multiplies power by its own block duration to get energy internally).
2. **`tile_to_length(<step 1 result>, len(forecast.power_kw))`** — if the
   resampled series is shorter than the forecast (typically: you passed one
   day, 96 blocks, against a full week, 672 blocks), it is **tiled
   (repeated) and the final repetition truncated** to exactly match the
   forecast's length. If it is already >= that length, it is simply
   truncated to that length (no repetition). This is how "one simulated
   day" becomes "a full week of usage" without you having to do that
   tiling yourself.

**Concrete example:** `sim.run.simulate_day(...)` returns `day.demand_kw`,
a 288-element array (24h at 5-minute blocks, `sim.config.STATE.block_minutes == 5.0`).
Calling `compute_soc_for_usage(day.demand_kw, usage_block_minutes=5.0)`
against a full 672-block (7-day) forecast will: average every 3 of the 288
values down to 96 values (one day at 15-minute resolution), then tile those
96 values 7 times (672 values) to cover the week. The returned
`result.usage_kw` is that final 672-length array -- read it directly if you
want to see exactly what usage series was actually used in the SOC
calculation.

---

## `ForecastResult` (`grid_energy/quartz_forecast.py`)

Returned by `forecast_pv_week` / `forecast_day_kw`. A `dataclass`, not
frozen (but callers should treat it as read-only).

| field / property | type | shape / index | units | meaning |
|---|---|---|---|---|
| `power_kw` | `pandas.Series` | `DatetimeIndex`, 15-min frequency, 672 rows for a full week | kW | PV power output forecast, one value per 15-minute block. |
| `energy_wh` | `pandas.Series` | same index as `power_kw` | Wh | Per-block energy: `power_kw * 1000 * 0.25`. |
| `total_energy_wh` | `float` (property, computed on access) | scalar | Wh | `energy_wh.sum()`. |
| `peak_power_kw` | `float` (property, computed on access) | scalar | kW | `power_kw.max()`. |

`power_kw.index` is a real `pandas.DatetimeIndex` (naive, no timezone),
starting at the `start` you passed to `forecast_pv_week` (or "now, UTC,
floored to midnight" if you passed `None`). You can slice by date directly,
e.g. `result.power_kw["2026-07-10"]`.

---

## `SOCResult` (`grid_energy/soc.py`)

Returned by `compute_soc_for_usage` (and by `soc.compute_soc` directly, if
calling it yourself). A `dataclass` of plain `numpy.ndarray`s, all the same
length `T` (no `pandas` index -- unlike `ForecastResult`).

| field | type | units | range | meaning |
|---|---|---|---|---|
| `t_hr` | `np.ndarray[float]` | hours | `0` to `T * block_minutes / 60` | Elapsed time since the start of the series, per block. `t_hr[0] == 0.0`. |
| `pv_kw` | `np.ndarray[float]` | kW | `>= 0` | The PV(t) input as passed to `compute_soc` (echoed back, not recomputed). |
| `usage_kw` | `np.ndarray[float]` | kW | `>= 0` typically | The usage(t) input as passed to `compute_soc` (echoed back — this is the *already resampled/tiled* array when called via `compute_soc_for_usage`). |
| `net_kw` | `np.ndarray[float]` | kW | any sign | `pv_kw - usage_kw`, per block. |
| `energy_potential_kwh` | `np.ndarray[float]` | kWh | any sign, **unbounded** | The unclipped running energy integral behind `socs_pct`. Resets (see `reset_daily`) to `actual`'s current value at each day boundary. |
| `socs_pct` | `np.ndarray[float]` | % | any value, **unbounded** (can exceed 100 or go below 0) | `energy_potential_kwh / capacity_kwh * 100`. **This is the corrected/integrated form of the originally-requested `socs = (PV - usage) / BC` equation.** Read values `> 100` as "surplus beyond what the battery could store today"; values `< 0` as "deficit beyond what the battery could supply today." |
| `energy_actual_kwh` | `np.ndarray[float]` | kWh | `[0, capacity_kwh]` | The physically realisable, clipped energy integral behind `actual_soc_pct`. **Never reset** except at the very first block (`t=0`) of the whole series -- represents a real battery's charge, carried continuously across day boundaries. |
| `actual_soc_pct` | `np.ndarray[float]` | % | `[0, 100]` | `energy_actual_kwh / capacity_kwh * 100`. This is "what the real battery's charge indicator would show." |
| `surplus_kwh` | `np.ndarray[float]` | kWh | `>= 0` | Per-block energy that would have overflowed a full battery (i.e. `pv - usage` exceeded remaining headroom that block) — this is real, physically-lost/curtailed energy, computed from the clipped tracker. |
| `deficit_kwh` | `np.ndarray[float]` | kWh | `>= 0` | Per-block unmet demand once the battery was empty — real, physically-unserved energy, computed from the clipped tracker. |

All arrays have the same length `T` and the same block spacing (whatever
`block_minutes` was passed to `compute_soc`; always `15.0` when reached via
`GridEnergyComponent.compute_soc_for_usage`). Index `i` across every field
refers to the same time block.

### `reset_daily` semantics (applies to both `soc.compute_soc` directly and `GridEnergyComponent.compute_soc_for_usage`)

- **`reset_daily=True` (default).** At the start of every calendar day
  (every `round(24*60 / block_minutes)` blocks — `96` blocks at 15-minute
  resolution), `energy_potential_kwh` (and therefore `socs_pct`) is reset to
  **whatever `energy_actual_kwh` currently equals at that instant** — i.e.
  "restart the surplus/deficit view from today's real starting battery
  charge." `energy_actual_kwh`/`actual_soc_pct` themselves are **never**
  touched by this reset — the real battery's charge is continuous across the
  whole series, exactly as a physical battery would behave.
- **`reset_daily=False`.** `energy_potential_kwh` runs as a single
  continuous integral from `t=0` to the end of the series with no resets at
  all (the pre-daily-reset behaviour). `energy_actual_kwh` is unaffected
  either way — it is never reset regardless of this flag.
- `soc_init_pct` (from the constructor / `config.BATTERY.soc_init_pct`) only
  sets the value at `t=0` of the **entire series** — it is not re-applied on
  subsequent days under either setting. Day 2 onward always continues from
  wherever the real battery (`energy_actual_kwh`) actually ended day 1.

---

## `resample_kw` and `tile_to_length` (`grid_energy/resample.py`)

Lower-level utilities used internally by `compute_soc_for_usage`. Call these
directly only if you need resampling/tiling behavior *without* also running
the SOC model (e.g. you're building your own pipeline on top of `grid_energy`).

### `resample_kw(series_kw, from_block_minutes, to_block_minutes) -> np.ndarray`

| name | type | meaning |
|---|---|---|
| `series_kw` | array-like of float | Power series, kW, at `from_block_minutes` resolution. |
| `from_block_minutes` | `float` | The block size `series_kw` is currently at. |
| `to_block_minutes` | `float` | The desired output block size. |

**Returns:** `np.ndarray[float]`, power in kW, at `to_block_minutes`
resolution.

**Behavior:**
- If `from_block_minutes == to_block_minutes`: returns `series_kw` unchanged
  (as a float `np.ndarray`).
- If `to_block_minutes > from_block_minutes` (downsampling, coarser output):
  requires `to_block_minutes / from_block_minutes` to be (within floating
  point tolerance of) an integer `factor`; averages every consecutive group
  of `factor` input blocks into one output block. **Raises `ValueError`** if
  the ratio isn't an integer, or if `len(series_kw)` isn't evenly divisible
  by `factor`.
- If `to_block_minutes < from_block_minutes` (upsampling, finer output):
  requires `from_block_minutes / to_block_minutes` to be an integer
  `factor`; repeats each input block `factor` times. **Raises `ValueError`**
  if that ratio isn't an integer. (No length-divisibility constraint in this
  direction — any input length works.)

**Worked examples:**
```python
resample_kw([1, 2, 3, 4, 5, 6], from_block_minutes=5, to_block_minutes=15)
# -> array([2.0, 5.0])   (mean of [1,2,3], mean of [4,5,6])

resample_kw([10, 20], from_block_minutes=15, to_block_minutes=5)
# -> array([10., 10., 10., 20., 20., 20.])

resample_kw(np.zeros(10), from_block_minutes=7, to_block_minutes=15)
# -> raises ValueError (15/7 is not an integer)

resample_kw(np.zeros(5), from_block_minutes=5, to_block_minutes=15)
# -> raises ValueError (5 elements not divisible by factor 3)
```

### `tile_to_length(series_kw, target_len) -> np.ndarray`

| name | type | meaning |
|---|---|---|
| `series_kw` | array-like of float, non-empty | Series to stretch/truncate. |
| `target_len` | `int` | Desired output length. |

**Returns:** `np.ndarray[float]` of exactly length `target_len`. If
`len(series_kw) >= target_len`, simply truncates (`series_kw[:target_len]`)
— no repetition happens in this case. If shorter, tiles (repeats) the whole
series enough times to cover `target_len`, then truncates the final
repetition to land exactly on `target_len`.

**Raises:** `ValueError` if `series_kw` is empty.

**Worked example:**
```python
tile_to_length([1, 2, 3], target_len=7)   # -> array([1,2,3,1,2,3,1])
tile_to_length([1,2,3,4,5], target_len=3) # -> array([1,2,3])  (truncated, not tiled)
```

---

## `config.py` defaults (all overridable via `GridEnergyComponent` constructor arguments)

```python
SITE = SiteConfig(
    latitude=-0.80895, longitude=36.24232,   # Oloika, Kajiado West, Kenya
    tilt_deg=15.0, orientation_deg=180.0,     # panel tilt/azimuth, passed to quartz-solar-forecast
)
PV = PVConfig(
    rated_kwp=60.0,                           # nameplate PV array capacity, kWp (= PV_max_kwp)
)
BATTERY = BatteryConfig(
    capacity_kwh=27.8,                        # BC, kWh (= PV_max_kwp * 25/54, deliberately smaller
                                                # than Oloika's real, oversized 54/25 kWh/kWp ratio)
    soc_init_pct=0.0,                         # starting charge, % of BC, at t=0 only
)
TIME = TimeConfig(
    block_minutes=5.0, T=288,                 # NOTE: this is soc.compute_soc's OWN default block
                                                # size when called directly with no block_minutes
                                                # argument -- it has NO effect on
                                                # GridEnergyComponent/quartz_forecast, which are
                                                # always 15-minute resolution.
)
```

`tilt_deg`/`orientation_deg` are **not** exposed as `GridEnergyComponent`
constructor fields (unlike `latitude`/`longitude`/`capacity_kwp`) — they are
always read from `config.SITE` inside `quartz_forecast._build_site`. To
change them, either edit `config.py`'s `SiteConfig` defaults, or call
`quartz_forecast.forecast_week_kw`/`forecast_day_kw` directly with a
hand-built `PVSite` (bypassing `GridEnergyComponent` entirely) if per-call
tilt/orientation overrides are needed.

---

## Known upstream gotchas (already handled inside `grid_energy` — informational only)

You do not need to work around either of these; they are documented here so
a caller understands why the code contains what might otherwise look like
unnecessary complexity.

1. **`requests_cache` serialization bug.** A plain `pip install
   quartz-solar-forecast` pulls in `attrs`/`cattrs` versions that make
   `requests_cache` raise `NameError: name 'RequestsCookieJar' is not
   defined` when it tries to disk-cache the Open-Meteo HTTP response.
   `quartz_forecast._patch_requests_cache()` (called automatically inside
   `forecast_day_kw`) replaces `requests_cache.CachedSession` with a plain
   `requests.Session` before any HTTP call happens, sidestepping the broken
   code path entirely. This means **no on-disk response caching happens at
   all** — every call to `forecast_pv_week`/`forecast_day_kw` makes fresh
   live HTTP requests, even for a `ts`/date combination requested before.
2. **Mutable `PVSite` bug.** `quartz_solar_forecast.forecast.predict_ocf`
   mutates its `PVSite.capacity_kwp` argument to `4` in place whenever the
   real capacity is `> 4 kWp` (it runs the model at 4 kWp, then rescales
   the output by `real_capacity_kwp / 4`). `forecast_day_kw` builds a
   **brand new `PVSite`** on every one of the week's 7 calls specifically to
   avoid a shared, already-mutated `PVSite` silently disabling that
   rescaling on calls after the first. If you ever call
   `quartz_solar_forecast.forecast.run_forecast` yourself directly
   (bypassing `grid_energy` entirely) with `capacity_kwp > 4`, construct a
   fresh `PVSite` for every call, never reuse one across multiple calls.

---

## Worked example: driving this from a `sim`-shaped caller

This is the reference pattern demonstrated in `demo_forecast_week.py`,
written out standalone. Nothing in `grid_energy` imports `sim` — this
example shows the intended calling direction, `sim -> grid_energy`.

```python
import numpy as np
from grid_energy import GridEnergyComponent

# --- caller-side: produce a usage_kw series at whatever resolution it has ---
# e.g. sim.run.simulate_day(...) returns a DayResult with `.demand_kw`,
# a 288-element array at 5-minute blocks (sim.config.STATE.block_minutes == 5.0)
usage_kw = my_simulated_day.demand_kw          # shape (288,), kW
usage_block_minutes = 5.0

# --- grid_energy side: one call gets real PV + full SOC dynamics ---
component = GridEnergyComponent(capacity_kwp=120.0, capacity_kwh=30.0, soc_init_pct=0.0)
result = component.compute_soc_for_usage(usage_kw, usage_block_minutes)

# --- read results back out ---
print(f"peak socs this run: {result.socs_pct.max():.1f}%")
print(f"battery ever ran dry for: {np.sum(result.deficit_kwh > 0)} of {len(result.deficit_kwh)} blocks")
print(f"total curtailed/surplus energy this week: {result.surplus_kwh.sum():.2f} kWh")

# --- sweeping many usage scenarios against the SAME week's PV forecast (avoid re-fetching) ---
forecast = component.forecast_pv_week()
for scenario_usage_kw in many_candidate_usage_scenarios:
    r = component.compute_soc_for_usage(scenario_usage_kw, usage_block_minutes=5.0, forecast=forecast)
    # score r somehow...
```
