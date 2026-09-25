"""Time-of-day drift: the direction effect that held out of sample (research/direction.md).

Quoted prices move in a consistent direction at some times of the week, above
all around the daily rollover at 17:00 New York time, when the value date
changes, swap points are applied and liquidity is thin: pairs whose base
currency pays the higher rate tend to dip in the hour into the roll, three
times as much on Wednesdays (the weekend's swap), and to recover after it.
Because the roll follows New York time, the slots are New York weekday-and-hour
(hourly bars) or weekday-and-quarter-hour (15-minute bars), measured on the
timeframe's own last ``WINDOW`` bars; failing that, the hour (quarter-hour)
of the day. A slot whose average move is clearly not zero (|t| >= ``T_MIN``)
gives the expected move of a bar in it; the larger |t|, the more reliable
(``T_HIGH`` marks the high-confidence calls). Other slots give none.

On the test period, next-hour calls were right 61 % of the time overall and
80 % for |t| >= 4 (about 2 % of hours); 15-minute calls 57 % and 86 % (last
~55 days). It is a regularity of quoted prices, not a trading edge: at the
roll the spread widens and the swap offsets the move.

The statistics use only bars that started before the origin's UTC day, so
everything can be rebuilt from the committed prices and is the same for all
origins of a day.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .timeutil import add_trading_minutes

NEW_YORK = ZoneInfo("America/New_York")
WINDOW = 6000        # bars: about a year of hourly bars, the ~60 days kept of 15-minute bars
CENTRE_MINUTES = 60  # the forecast centre uses the drift of the bars in the first hour only (see centre_drift)
MIN_N = 15           # bars a slot needs before its average counts
T_MIN = 2.0          # a call
T_HIGH = 4.0         # a high-confidence call
_CACHE: dict = {}


def _slots(times: pd.DatetimeIndex, minutes: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(weekday-and-time slot, time-of-day slot, slots per day) in New York time."""
    local = times.tz_convert(NEW_YORK)
    per_day = 24 * 60 // minutes
    tod = (local.hour.to_numpy() * 60 + local.minute.to_numpy()) // minutes
    return local.dayofweek.to_numpy() * per_day + tod, tod, per_day


def slot_stats(bars: pd.DataFrame, origin: datetime, minutes: int) -> dict:
    """Average move (bp) and t statistic of each New York weekday-and-time slot and time-of-day slot,
    from the last ``WINDOW`` bars that started before the origin's UTC day."""
    day0 = pd.Timestamp(origin).tz_convert("UTC").normalize()
    idx = bars.index
    cut = int(idx.searchsorted(day0, side="left"))
    lo = max(0, cut - WINDOW - 1)
    c = bars["close"].to_numpy(float)[lo:cut]
    key = (minutes, WINDOW, MIN_N, day0, cut - lo, float(c[0]) if len(c) else None, float(c[-1]) if len(c) else None,
           idx[lo] if cut > lo else None, idx[cut - 1] if cut > lo else None)
    if key in _CACHE:
        return _CACHE[key]
    t = idx[lo:cut]
    r = np.diff(np.log(c)) * 1e4 if len(c) > 1 else np.array([])
    if len(r):                                             # a bar after a pause (weekend, missing bars) carries its gap
        r[np.diff(t.as_unit("ns").asi8) > minutes * 60_000_000_000] = np.nan
    week, tod, per_day = _slots(t[1:], minutes) if len(r) else (np.array([], int), np.array([], int), 24 * 60 // minutes)
    out = {"slot": _stats(r, week, 7 * per_day), "tod": _stats(r, tod, per_day)}
    if len(_CACHE) > 4096:
        _CACHE.clear()
    _CACHE[key] = out
    return out


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


def bar_drift(stats: dict, starts, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected move (bp) of bars starting at ``starts`` and its t statistic; 0 where no slot is clear."""
    week, tod, _ = _slots(pd.DatetimeIndex(starts), minutes)
    mu_w, t_w = stats["slot"]
    mu_d, t_d = stats["tod"]
    use_w = np.abs(t_w[week]) >= T_MIN
    use_d = ~use_w & (np.abs(t_d[tod]) >= T_MIN)
    d = np.where(use_w, mu_w[week], np.where(use_d, mu_d[tod], 0.0))
    t = np.where(use_w, t_w[week], np.where(use_d, t_d[tod], 0.0))
    return d, t


def combined_t(d: np.ndarray, t: np.ndarray) -> float:
    """t statistic of a sum of bar drifts (independent slot means)."""
    m = t != 0
    if not m.any():
        return 0.0
    se = np.abs(d[m] / t[m])
    return float(d[m].sum() / np.sqrt(np.sum(se * se)))


def centre_drift(bar_drift: np.ndarray, minutes: int) -> np.ndarray:
    """The drift in the forecast centre after each step: the expected moves of the bars in the first
    hour, and nothing from then on. The first hour's move tends to be given back over the next hours
    (after the rollover dip), so carrying it, or adding later bars' drifts, did not help 4 or 24 hours
    ahead (research/direction.md)."""
    cum = np.cumsum(np.asarray(bar_drift, dtype=float))
    cum[CENTRE_MINUTES // minutes if minutes else 0:] = 0.0
    return cum


def centre_t(bar_drift: np.ndarray, bar_t: np.ndarray, minutes: int) -> np.ndarray:
    """The t statistic of the centre drift after each step (0 where the centre has none)."""
    k = CENTRE_MINUTES // minutes if minutes else 0
    return np.array([combined_t(bar_drift[: j + 1], bar_t[: j + 1]) if j < k else 0.0 for j in range(len(bar_drift))])


def step_drift(minutes: int, bars: pd.DataFrame | None, origin: datetime, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected move (bp) and its t statistic for each of the next ``steps`` bars of ``minutes`` after
    ``origin``, from the timeframe's own ``bars``; zeros for daily bars."""
    if not minutes or bars is None or len(bars) < 200:
        return np.zeros(steps), np.zeros(steps)
    starts = [add_trading_minutes(origin, k, minutes) - timedelta(minutes=minutes) for k in range(1, steps + 1)]
    return bar_drift(slot_stats(bars, origin, minutes), starts, minutes)


def tier(t: float) -> str | None:
    """"high" (|t| >= T_HIGH), "mid" (|t| >= T_MIN) or None (no call)."""
    a = abs(t)
    return "high" if a >= T_HIGH else "mid" if a >= T_MIN else None
