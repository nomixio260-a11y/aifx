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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .timeutil import HOUR, LONDON, trading_slots_between

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


def _slot_keys(times, slot_minutes: int, tz=None, weekday: bool = False) -> np.ndarray:
    """Time-of-day slot (or weekday-and-time slot) of each bar start, in ``tz`` (default UTC)."""
    idx = pd.DatetimeIndex(times)
    if tz is not None:
        idx = idx.tz_convert(tz)
    tod = (idx.hour.to_numpy() * 60 + idx.minute.to_numpy()) // slot_minutes
    return idx.dayofweek.to_numpy() * (1440 // slot_minutes) + tod if weekday else tod


def slot_profile(times, sq: np.ndarray, window: int = 1500, smooth: float = 0.25,
                 slot_minutes: int = 60, tz=None, weekday: bool = False, shrink: float = 20.0) -> np.ndarray:
    """Relative variance by time of day in slots of ``slot_minutes`` (mean 1 over the sample).

    ``sq`` is the per-bar variance proxy (squared returns), aligned with ``times``.
    ``smooth`` is the weight given to each neighbouring slot. With ``weekday``
    the slots are weekday-and-time (7 x slots per day), each pulled toward its
    time of day with the weight of ``shrink`` bars; ``tz`` sets the clock the
    slots follow (e.g. New York time, whose daylight saving the market follows).
    """
    n = 1440 // slot_minutes
    tod = _slot_keys(times[-window:], slot_minutes, tz)
    ss = np.asarray(sq[-window:], dtype=float)
    prof = np.ones(n)
    for h in range(n):
        sel = ss[tod == h]
        if len(sel) >= 5:
            prof[h] = np.mean(sel)
    if weekday:
        ok = np.isfinite(ss)
        mean_all = float(np.mean(ss[ok])) if ok.any() else 1.0
        base = prof / mean_all
        keys = _slot_keys(times[-window:], slot_minutes, tz, weekday=True)
        cnt = np.bincount(keys[ok], minlength=7 * n).astype(float)
        tot = np.bincount(keys[ok], weights=ss[ok], minlength=7 * n)
        out = (tot / mean_all + shrink * np.tile(base, 7)) / (cnt + shrink)
        mean = float(np.sum(out * cnt) / max(cnt.sum(), 1))
        return out / mean if mean > 0 else np.ones(7 * n)
    # Smooth over neighbouring slots (circular) to damp sampling noise.
    prof = smooth * np.roll(prof, 1) + (1 - 2 * smooth) * prof + smooth * np.roll(prof, -1)
    counts = np.bincount(tod, minlength=n).astype(float)
    mean = float(np.sum(prof * counts) / max(counts.sum(), 1))
    return prof / mean if mean > 0 else np.ones(n)


def hour_profile(times, sq: np.ndarray, window: int = 1500, smooth: float = 0.25) -> np.ndarray:
    """Relative variance by UTC hour of day (24 values, mean 1 over the sample)."""
    return slot_profile(times, sq, window, smooth, 60)


def _ewma(x: np.ndarray, lam: float) -> float:
    """EWMA of ``x`` started at the mean of its first 20 values (same as a loop, vectorised)."""
    init = float(np.mean(x[:20]))
    if len(x) <= 20:
        return init
    return float(pd.Series(np.concatenate([[init], x[20:]])).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1])


def intraday_variance_path(times, y: np.ndarray, origin: datetime, steps: int, minutes: int = 60,
                           events: list[dict] | None = None, lam: float = 0.97, reversion: float = 0.985,
                           seasonal: bool = True, sq: np.ndarray | None = None, profile_window: int = 1500,
                           smooth: float = 0.25, profile_minutes: int | None = None,
                           long_window: int = 1500, profile_tz=None,
                           profile_weekday: bool = False) -> tuple[np.ndarray, list[datetime]]:
    """Per-step variance for the next ``steps`` open-market bars of ``minutes`` after ``origin``.

    ``times`` are bar open times aligned with ``y`` (log closes), all ending at or
    before ``origin``. ``sq`` optionally replaces the squared returns as the
    per-bar variance proxy. The time-of-day profile uses slots of
    ``profile_minutes`` (default: the bar length), on the clock of ``profile_tz``
    and by weekday with ``profile_weekday``. Weekend gaps and scheduled events
    add variance measured in hours of normal trading, whatever the bar length.
    Returns (variance per step, end time of each step).
    """
    pm = profile_minutes or minutes
    per_hour = 60 / minutes
    r = np.diff(y)
    sq = r * r if sq is None else sq
    t_r = list(times[1:])
    if seasonal:
        prof = slot_profile(t_r, sq, profile_window, smooth, pm, profile_tz, profile_weekday)
    else:
        prof = np.ones(7 * (1440 // pm) if profile_weekday else 1440 // pm)

    def slots_of(ts) -> np.ndarray:
        return _slot_keys(ts, pm, profile_tz, profile_weekday)

    u2 = sq / prof[slots_of(t_r)]
    long_var = float(np.mean(u2[-long_window:]))
    ewma = _ewma(u2, lam)
    # Future open-market slots (start times) until we have enough steps.
    step = timedelta(minutes=minutes)
    slots: list[datetime] = []
    horizon_end = origin
    while len(slots) < steps:
        horizon_end = horizon_end + max(24 * HOUR, steps * step)
        slots = trading_slots_between(origin, horizon_end, minutes)
    slots = slots[:steps]
    fut = prof[slots_of(slots)]
    var = np.empty(steps)
    prev_end = origin
    for k, start in enumerate(slots, start=1):
        v = (long_var + (ewma - long_var) * reversion ** k) * fut[k - 1]
        if start > prev_end:  # the market was shut in between (weekend)
            v += WEEKEND_GAP_HOURS * per_hour * long_var
        var[k - 1] = v
        prev_end = start + step
    ends = [s + step for s in slots]
    for ev in events or []:
        t = ev["time"]
        kappa = EVENT_KAPPA_1H.get(ev.get("impact", ""), 0.0)
        if not kappa:
            continue
        for k, (start, end) in enumerate(zip(slots, ends)):
            if start <= t < end:
                var[k] += kappa * per_hour * long_var * fut[k]
                break
    return var, ends


def hourly_variance_path(times, y: np.ndarray, origin: datetime, steps: int,
                         events: list[dict] | None = None, lam: float = 0.97,
                         reversion: float = 0.985, seasonal: bool = True,
                         sq: np.ndarray | None = None, profile_window: int = 1500,
                         smooth: float = 0.25, profile_tz=None,
                         profile_weekday: bool = False) -> tuple[np.ndarray, list[datetime]]:
    """Per-step variance for the next ``steps`` open-market hours after ``origin``."""
    return intraday_variance_path(times, y, origin, steps, 60, events, lam, reversion, seasonal, sq,
                                  profile_window, smooth, profile_tz=profile_tz, profile_weekday=profile_weekday)


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
