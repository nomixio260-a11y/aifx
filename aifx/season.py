"""Time-of-day drift: the one direction effect that held out of sample (research/direction.md).

Quoted prices move in a consistent direction at some hours, above all around
the daily rollover (about 20:00-23:00 UTC, when the value date changes, swap
points are applied and liquidity is thin): pairs whose base currency pays the
higher rate tend to dip then, three times as much on Wednesdays (the weekend's
swap). Measured on the pair's last ``WINDOW`` hourly bars, each weekday-and-hour
slot (or, failing that, each hour) whose average move is clearly not zero
(|t| >= ``T_MIN``) gives the expected move of a bar in that slot; other slots
give none. On the test period this called the next hour's direction right
57 % of the time in the slots where it made a call (about one hour in six,
t = 4.5; 55 % on the tune period). The forecast centre uses it for the first
hour only; each forecast candle's colour uses its own bar's drift.

It is a regularity of quoted prices, not a trading edge: at rollover the
spread widens and the swap offsets the move.

The statistics use only hourly bars that started before the origin's UTC day,
so everything can be rebuilt from the committed prices and is the same for
all origins of a day.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .timeutil import add_trading_minutes

WINDOW = 6000        # hourly bars, about one year
CENTRE_MINUTES = 60  # the forecast centre uses the drift of the bars in the first hour only (see centre_drift)
MIN_N = 15           # bars a slot needs before its average counts
T_MIN = 2.0
HOUR_NS = 3_600_000_000_000


def slot_stats(hourly: pd.DataFrame, origin: datetime) -> dict:
    """Average move (bp) and t statistic of each hour (24) and weekday-and-hour slot (7 x 24)."""
    day0 = pd.Timestamp(origin).tz_convert("UTC").normalize()
    idx = hourly.index
    cut = int(idx.searchsorted(day0, side="left"))
    lo = max(0, cut - WINDOW - 1)
    c = hourly["close"].to_numpy(float)[lo:cut]
    t = idx[lo:cut]
    r = np.diff(np.log(c)) * 1e4 if len(c) > 1 else np.array([])
    if len(r):                                             # a bar after a pause (weekend, missing hours) carries its gap
        r[np.diff(t.as_unit("ns").asi8) > HOUR_NS] = np.nan
    hour = t.hour.to_numpy()[1:]
    slot = t.dayofweek.to_numpy()[1:] * 24 + hour
    return {"hour": _stats(r, hour, 24), "slot": _stats(r, slot, 168)}


def _stats(r: np.ndarray, key: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(r)
    r, key = r[ok], key[ok]
    cnt = np.bincount(key, minlength=n).astype(float)
    s1 = np.bincount(key, weights=r, minlength=n)
    s2 = np.bincount(key, weights=r * r, minlength=n)
    use = cnt >= MIN_N
    safe = np.where(use, cnt, 1.0)
    mu = np.where(use, s1 / safe, 0.0)
    sd = np.sqrt(np.maximum(s2 / safe - mu * mu, 0.0))
    t = np.where(use & (sd > 0), mu / np.where(sd > 0, sd, 1.0) * np.sqrt(safe), 0.0)
    return mu, t


def bar_drift(stats: dict, starts, minutes: int) -> np.ndarray:
    """Expected move (bp) of bars starting at ``starts`` (UTC) of ``minutes``; 0 where no slot is clear."""
    starts = pd.DatetimeIndex(starts)
    hour = starts.hour.to_numpy()
    slot = starts.dayofweek.to_numpy() * 24 + hour
    mu_h, t_h = stats["hour"]
    mu_s, t_s = stats["slot"]
    d = np.where(np.abs(t_s[slot]) >= T_MIN, mu_s[slot], np.where(np.abs(t_h[hour]) >= T_MIN, mu_h[hour], 0.0))
    return d * (minutes / 60.0)


def centre_drift(bar_drift: np.ndarray, minutes: int) -> np.ndarray:
    """The drift in the forecast centre after each step: the expected moves of the bars in the first
    hour, and nothing from then on. The first hour's move tends to be given back over the next hours
    (after the rollover dip), so carrying it, or adding later bars' drifts, did not help 4 or 24 hours
    ahead (research/direction.md)."""
    cum = np.cumsum(np.asarray(bar_drift, dtype=float))
    cum[CENTRE_MINUTES // minutes if minutes else 0:] = 0.0
    return cum


def step_drift(minutes: int, hourly: pd.DataFrame | None, origin: datetime, steps: int) -> np.ndarray:
    """Expected move (bp) of each of the next ``steps`` bars after ``origin``; zeros for daily bars."""
    if not minutes or hourly is None or len(hourly) < 200:
        return np.zeros(steps)
    starts = [add_trading_minutes(origin, k, minutes) - timedelta(minutes=minutes) for k in range(1, steps + 1)]
    return bar_drift(slot_stats(hourly, origin), starts, minutes)
