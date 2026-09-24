"""Forecast uncertainty: how far the price can plausibly move by each horizon.

Daily: EWMA variance that decays toward the long-run variance. When hourly
bars are available, each day's variance is measured from them (realized
variance: the sum of squared hourly returns), which is much less noisy than
one squared daily return, so the EWMA can react faster.

Hourly: FX volatility has a strong time-of-day pattern (quiet late New York
and early Tokyo, busy London/New York overlap). Each bar's variance is
measured from its high-low range (Parkinson), de-seasonalised by an
hour-of-day profile, smoothed by an EWMA, and the profile is put back for
each future hour. Weekend gaps and scheduled high-impact releases (from the
economic calendar) add extra variance.

The settings were chosen on 2002-2016 (daily) / the first 60 % of two years
of hourly bars and confirmed on the later period (research/report.md).
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from .timeutil import HOUR, LONDON, trading_hours_between

EVENT_KAPPA_1H = {"High": 3.0, "Medium": 1.0}      # extra variance, in "normal hours" at that time of day
EVENT_KAPPA_1D = {"High": 0.25, "Medium": 0.08}    # extra variance, as a share of a normal day
WEEKEND_GAP_HOURS = 2.0

DAILY_LAM = 0.94          # EWMA decay on squared daily returns
DAILY_RV_LAM = 0.80       # EWMA decay on realized variance (less noisy, so it can adapt faster)
DAILY_REVERSION = 0.90    # how fast today's variance level fades into the long-run level, per day
RV_WINDOW = 250           # days used to put realized variance on the scale of squared daily returns
RV_MIN_DAYS = 50
RANGE_WINDOW = 1500       # hourly bars used to put the high-low range on the scale of squared returns


def daily_sigma_path(y: np.ndarray, horizon: int, lam: float = DAILY_LAM,
                     long_window: int = 500, reversion: float = DAILY_REVERSION) -> np.ndarray:
    """Cumulative log-return standard deviation for steps 1..horizon (daily bars)."""
    return np.sqrt(np.cumsum(daily_step_variance(y, horizon, lam, long_window, reversion)))


def daily_step_variance(y: np.ndarray, horizon: int, lam: float = DAILY_LAM, long_window: int = 500,
                        reversion: float = DAILY_REVERSION, sq: np.ndarray | None = None) -> np.ndarray:
    """Per-step variance for the next ``horizon`` days.

    ``sq`` optionally replaces the squared returns as each day's variance
    measurement (aligned with ``diff(y)``).
    """
    r = np.diff(y)
    if sq is None:
        ewma = r[:20].var()
        sq = r * r
    else:
        ewma = float(np.mean(sq[:20]))
    for x in sq[20:]:
        ewma = lam * ewma + (1 - lam) * x
    long_var = r[-long_window:].var()
    k = np.arange(1, horizon + 1)
    return long_var + (ewma - long_var) * reversion ** k


def scale_proxy(alt: np.ndarray, r: np.ndarray, window: int) -> np.ndarray | None:
    """Put a variance measurement on the scale of squared returns (same mean over the
    trailing ``window``); where it is missing, the squared return is used.
    None when there are too few measurements."""
    r2 = r * r
    ok = np.isfinite(alt[-window:])
    if ok.sum() < RV_MIN_DAYS:
        return None
    c = float(np.mean(r2[-window:][ok]) / np.mean(alt[-window:][ok]))
    return np.where(np.isfinite(alt), c * alt, r2)


def range_variance(bars: pd.DataFrame) -> np.ndarray:
    """Parkinson high-low variance of each bar after the first (aligned with diff(log close)); NaN if unusable."""
    hl = np.log(bars["high"].to_numpy(dtype=float) / bars["low"].to_numpy(dtype=float))[1:]
    return np.where(hl > 0, hl * hl / (4 * math.log(2)), np.nan)


def realized_daily_variance(hourly: pd.DataFrame, until: datetime) -> pd.Series:
    """Sum of squared hourly log returns per London business day, from bars ended by ``until``.

    Index: London date (midnight timestamps). Sunday-evening trading counts
    towards Monday, matching the daily bars.
    """
    ends = hourly.index + pd.Timedelta(hours=1)
    h = hourly[ends <= pd.Timestamp(until)]
    if len(h) < 2:
        return pd.Series(dtype=float)
    r = np.diff(np.log(h["close"].to_numpy(dtype=float)))
    local = (h.index[1:] + pd.Timedelta(minutes=59)).tz_convert(LONDON).tz_localize(None).normalize()
    wd = local.dayofweek
    local = local + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")
    rv = pd.Series(r * r).groupby(np.asarray(local)).sum()
    return rv.iloc[1:]   # the first day is usually incomplete


def daily_variance_inputs(daily: pd.DataFrame, hourly: pd.DataFrame | None,
                          until: datetime) -> tuple[np.ndarray | None, float]:
    """Per-day variance measurements for ``daily`` (aligned with its returns) and the EWMA decay to use."""
    if hourly is None or not len(hourly):
        return None, DAILY_LAM
    rv = realized_daily_variance(hourly, until)
    if not len(rv):
        return None, DAILY_LAM
    alt = rv.reindex(pd.DatetimeIndex(daily.index[1:]).normalize()).to_numpy(dtype=float)
    r = np.diff(np.log(daily["close"].to_numpy(dtype=float)))
    sq = scale_proxy(alt, r, RV_WINDOW)
    return (None, DAILY_LAM) if sq is None else (sq, DAILY_RV_LAM)


def hour_profile(times, sq: np.ndarray, window: int = 1500, smooth: float = 0.25) -> np.ndarray:
    """Relative variance by UTC hour of day (24 values, mean 1 over the sample).

    ``sq`` is the per-bar variance proxy (squared returns), aligned with ``times``.
    ``smooth`` is the weight given to each neighbouring hour.
    """
    hours = np.asarray([t.hour for t in times[-window:]])
    ss = sq[-window:]
    prof = np.ones(24)
    for h in range(24):
        sel = ss[hours == h]
        if len(sel) >= 5:
            prof[h] = np.mean(sel)
    # Smooth over neighbouring hours (circular) to damp sampling noise.
    prof = smooth * np.roll(prof, 1) + (1 - 2 * smooth) * prof + smooth * np.roll(prof, -1)
    counts = np.bincount(hours, minlength=24).astype(float)
    mean = float(np.sum(prof * counts) / max(counts.sum(), 1))
    return prof / mean if mean > 0 else np.ones(24)


def hourly_variance_path(times, y: np.ndarray, origin: datetime, steps: int,
                         events: list[dict] | None = None, lam: float = 0.97,
                         reversion: float = 0.985, seasonal: bool = True,
                         sq: np.ndarray | None = None, profile_window: int = 1500,
                         smooth: float = 0.25) -> tuple[np.ndarray, list[datetime]]:
    """Per-step variance for the next ``steps`` open-market hours after ``origin``.

    ``times`` are bar open times aligned with ``y`` (log closes), all ending at or
    before ``origin``. ``sq`` optionally replaces the squared returns as the
    per-bar variance proxy. Returns (variance per step, end time of each step).
    """
    r = np.diff(y)
    sq = r * r if sq is None else sq
    t_r = list(times[1:])
    prof = hour_profile(t_r, sq, profile_window, smooth) if seasonal else np.ones(24)
    u2 = sq / prof[[t.hour for t in t_r]]
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
