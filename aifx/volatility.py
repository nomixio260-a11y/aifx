"""Forecast uncertainty: how far the price can plausibly move by each horizon.

Daily: RiskMetrics EWMA variance that decays toward the long-run variance.

Hourly: FX volatility has a strong time-of-day pattern (quiet late New York
and early Tokyo, busy London/New York overlap). Returns are de-seasonalised
by an hour-of-day profile, an EWMA runs on the de-seasonalised series, and
the profile is put back for each future hour. Weekend gaps and scheduled
high-impact releases (from the economic calendar) add extra variance.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .timeutil import HOUR, trading_hours_between

EVENT_KAPPA_1H = {"High": 3.0, "Medium": 1.0}      # extra variance, in "normal hours" at that time of day
EVENT_KAPPA_1D = {"High": 0.25, "Medium": 0.08}    # extra variance, as a share of a normal day
WEEKEND_GAP_HOURS = 2.0


def daily_sigma_path(y: np.ndarray, horizon: int, lam: float = 0.94,
                     long_window: int = 500, reversion: float = 0.97) -> np.ndarray:
    """Cumulative log-return standard deviation for steps 1..horizon (daily bars)."""
    return np.sqrt(np.cumsum(daily_step_variance(y, horizon, lam, long_window, reversion)))


def daily_step_variance(y: np.ndarray, horizon: int, lam: float = 0.94,
                        long_window: int = 500, reversion: float = 0.97) -> np.ndarray:
    r = np.diff(y)
    ewma = r[:20].var()
    for x in r[20:]:
        ewma = lam * ewma + (1 - lam) * x * x
    long_var = r[-long_window:].var()
    k = np.arange(1, horizon + 1)
    return long_var + (ewma - long_var) * reversion ** k


def hour_profile(times, r: np.ndarray, window: int = 1500) -> np.ndarray:
    """Relative variance by UTC hour of day (24 values, mean 1 over the sample)."""
    hours = np.asarray([t.hour for t in times[-window:]])
    rr = r[-window:]
    prof = np.ones(24)
    for h in range(24):
        sel = rr[hours == h]
        if len(sel) >= 5:
            prof[h] = np.mean(sel * sel)
    # Smooth over neighbouring hours (circular) to damp sampling noise.
    prof = 0.25 * np.roll(prof, 1) + 0.5 * prof + 0.25 * np.roll(prof, -1)
    counts = np.bincount(hours, minlength=24).astype(float)
    mean = float(np.sum(prof * counts) / max(counts.sum(), 1))
    return prof / mean if mean > 0 else np.ones(24)


def hourly_variance_path(times, y: np.ndarray, origin: datetime, steps: int,
                         events: list[dict] | None = None, lam: float = 0.97,
                         reversion: float = 0.985) -> tuple[np.ndarray, list[datetime]]:
    """Per-step variance for the next ``steps`` open-market hours after ``origin``.

    ``times`` are bar open times aligned with ``y`` (log closes), all ending at or
    before ``origin``. Returns (variance per step, end time of each step).
    """
    r = np.diff(y)
    t_r = list(times[1:])
    prof = hour_profile(t_r, r)
    u2 = (r * r) / prof[[t.hour for t in t_r]]
    long_var = float(np.mean(u2[-1500:]))
    ewma = float(np.mean(u2[:20]))
    for x in u2[20:]:
        ewma = lam * ewma + (1 - lam) * x
    # Future open-market slots (start times) until we have enough steps.
    slots: list[datetime] = []
    horizon_end = origin
    while len(slots) < steps:
        horizon_end = horizon_end + 24 * HOUR
        slots = trading_hours_between(origin, horizon_end)
    slots = slots[:steps]
    var = np.empty(steps)
    prev_end = origin
    for k, start in enumerate(slots, start=1):
        v = (long_var + (ewma - long_var) * reversion ** k) * prof[start.hour]
        if start > prev_end:  # the market was shut in between (weekend)
            v += WEEKEND_GAP_HOURS * long_var
        var[k - 1] = v
        prev_end = start + HOUR
    ends = [s + HOUR for s in slots]
    for ev in events or []:
        t = ev["time"]
        kappa = EVENT_KAPPA_1H.get(ev.get("impact", ""), 0.0)
        if not kappa:
            continue
        for k, (start, end) in enumerate(zip(slots, ends)):
            if start <= t < end:
                var[k] += kappa * long_var * prof[start.hour]
                break
    return var, ends


def add_daily_events(var: np.ndarray, day_ends: list[datetime], origin: datetime,
                     events: list[dict] | None) -> np.ndarray:
    """Add event variance to daily steps. ``day_ends[k]`` closes step k+1."""
    out = var.copy()
    starts = [origin] + day_ends[:-1]
    for ev in events or []:
        kappa = EVENT_KAPPA_1D.get(ev.get("impact", ""), 0.0)
        if not kappa:
            continue
        for k, (a, b) in enumerate(zip(starts, day_ends)):
            if a <= ev["time"] < b:
                out[k] += kappa * var[k]
                break
    return out
