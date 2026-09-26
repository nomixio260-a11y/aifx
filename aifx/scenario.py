"""Forecast candles: a whole future candle chart (open, high, low, close per bar).

Each future candle's size (high minus low) is the part that can be predicted:
the current level of the bars' range (an exponentially weighted average)
times the usual ratio of the range for that New York time of day and weekday
(intraday bars; the time of day alone when there are too few such bars) or
weekday (daily bars) to the level before it, measured on the last
``PROFILE_N`` such bars. In the research this was more accurate than the plain
14-bar average range (ATR) on every timeframe.

The shape of each candle (where it closes, how long its wicks are, as parts
of its range) is taken from the most typical of the past situations that
looked most like now: the last ``L`` bars' moves in units of their volatility,
at the same time of day for intraday bars. Of the ``K`` nearest situations,
the one whose continuation is closest to all the others (the medoid) is used.
The one direction effect that held out of sample is the time-of-day drift
(season.py): where it makes a call for a bar, that candle is drawn in the
called direction (the analogue's candle, mirrored if needed). The path is then
tilted, through the other candles, so that it ends at the forecast centre (the
most accurate end point in the research).

Everything is computed from the bars up to the origin, so the same candles
can be rebuilt by anyone from the stored prices.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .season import NEW_YORK
from .timeutil import add_business_days, add_trading_minutes

PATTERN = {"15m": 32, "1h": 24, "1d": 20}
K = 30
LAM = {"15m": 0.97, "1h": 0.97, "1d": 0.94}
LEVEL_LAM = {"15m": 0.95, "1h": 0.95, "1d": 0.95}   # weight of the previous level in the range level
PROFILE_N = 500
MIN_CASES = 20


def _sigma(r: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty(len(r))
    v = np.nanmean(r[: min(50, len(r))] ** 2) if len(r) else 0.0
    for i, x in enumerate(r):
        v = lam * v + (1 - lam) * x * x
        out[i] = np.sqrt(v)
    return out


def _slots(times, minutes: int) -> list[np.ndarray]:
    """Keys that group bars with the same usual size, most specific first: time of day and weekday,
    then time of day (intraday bars, on New York time, whose daylight saving the market sessions
    follow); weekday (daily bars)."""
    if minutes:
        times = pd.DatetimeIndex(times).tz_convert(NEW_YORK)
        tod = np.asarray(times.hour) * 60 + np.asarray(times.minute)
        return [tod * 10 + np.asarray(times.dayofweek), tod]
    return [np.asarray(times.dayofweek)]


def future_times(index: pd.DatetimeIndex, steps: int, minutes: int) -> pd.DatetimeIndex:
    """Start times (intraday) or days (daily) of the next ``steps`` bars on the trading calendar."""
    last = index[-1]
    if minutes:
        origin = last.to_pydatetime() + timedelta(minutes=minutes)
        return pd.DatetimeIndex([add_trading_minutes(origin, k, minutes) - timedelta(minutes=minutes)
                                 for k in range(1, steps + 1)])
    return pd.DatetimeIndex([pd.Timestamp(add_business_days(last.date(), k)) for k in range(1, steps + 1)])


def sizes(tf_key: str, bars: pd.DataFrame, steps: int, minutes: int = 0) -> np.ndarray:
    """Expected range of each of the next ``steps`` bars, in log price (log high - log low)."""
    lr = np.log(bars["high"].to_numpy(float) / bars["low"].to_numpy(float))
    lr = np.where(np.isfinite(lr) & (lr >= 0), lr, 0.0)
    lam = LEVEL_LAM[tf_key]
    level = pd.Series(lr).ewm(alpha=1 - lam).mean().to_numpy()
    keys = _slots(bars.index, minutes)
    fut = _slots(future_times(bars.index, steps, minutes), minutes)
    now = len(lr) - 1
    out = np.empty(steps)
    for j in range(steps):
        past = np.arange(min(200, now // 2), now - j)          # bars whose range j+1 bars later is known now
        past = past[level[past] > 0]
        use = past[-PROFILE_N:]
        for key, f in zip(keys, fut):                          # the most specific group with enough cases
            same = past[key[past + 1 + j] == f[j]][-PROFILE_N:]
            if len(same) >= MIN_CASES:
                use = same
                break
        ratio = float(np.median(lr[use + 1 + j] / level[use])) if len(use) else 1.0
        out[j] = ratio * level[now]
    return out


def candles(tf_key: str, bars: pd.DataFrame, steps: int, end_target: float | None = None,
            minutes: int = 0, drift: np.ndarray | None = None) -> tuple[list[list[float]], dict]:
    """Forecast candles [open, high, low, close] for ``steps`` bars after the last one, and how they
    were made. ``end_target``: the price the path should end at (the forecast centre). ``drift``:
    the expected move of each bar from the time-of-day drift (bp, 0 where it makes no call)."""
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    L = PATTERN[tf_key]
    if n < L + steps + 200:
        return [], {}
    y = np.log(c)
    r = np.diff(y, prepend=y[0])
    sig = _sigma(r, LAM[tf_key])
    sig = np.where(sig > 0, sig, np.nan)
    z = r / np.concatenate([[np.nan], sig[:-1]])            # move of each bar in units of the volatility before it
    now = n - 1
    pat_now = z[now - L + 1: now + 1]
    if not np.all(np.isfinite(pat_now)):
        return [], {}
    w = np.linspace(0.5, 1.5, L)                              # recent bars matter more
    ends = np.arange(L + 60, now - steps)                     # the whole future of a candidate is known
    if minutes:
        tod = _slots(bars.index, minutes)[1]
        ends = ends[tod[ends] == tod[now]]
    if len(ends) < K:
        return [], {}
    idx = ends[:, None] - np.arange(L - 1, -1, -1)[None, :]
    P = z[idx]
    ok = np.all(np.isfinite(P), axis=1)
    ends, P = ends[ok], P[ok]
    d = ((P - pat_now) ** 2 * w).sum(axis=1)
    order = np.argsort(d, kind="stable")
    chosen: list[int] = []
    for i in order:                                          # distinct situations, not neighbours of one another
        e = ends[i]
        if all(abs(e - x) > steps for x in chosen):
            chosen.append(int(e))
        if len(chosen) == K:
            break
    ch = np.array(chosen)
    fut = np.arange(1, steps + 1)
    # the medoid: the continuation (closes in volatility units) closest to all the others
    cl = (y[ch[:, None] + fut[None, :]] - y[ch][:, None]) / sig[ch][:, None]
    m = int(np.argmin(((cl[:, None, :] - cl[None, :, :]) ** 2).sum(axis=2).sum(axis=1)))
    # its candles as parts of their range, opening at the previous close
    k = ch[m] + fut
    pc, hh, ll, cc = np.log(c[k - 1]), np.log(h[k]), np.log(lo[k]), np.log(c[k])
    top, bot = np.maximum(hh, np.maximum(pc, cc)), np.minimum(ll, np.minimum(pc, cc))
    rng = top - bot
    safe = np.where(rng > 0, rng, 1.0)
    fc = np.where(rng > 0, (cc - pc) / safe, 0.0)
    fh = np.where(rng > 0, (top - pc) / safe, 0.5)
    fl = np.where(rng > 0, (bot - pc) / safe, -0.5)
    size = sizes(tf_key, bars, steps, minutes)
    call = np.zeros(steps) if drift is None else np.sign(np.asarray(drift, dtype=float)[:steps])
    flip = call * np.sign(fc) < 0                           # a called bar's candle points the called way
    fc, fh, fl = np.where(flip, -fc, fc), np.where(flip, -fl, fh), np.where(flip, -fh, fl)
    doji = (call != 0) & (fc == 0)
    fc = np.where(doji, call * 0.25, fc)
    fh, fl = np.maximum(fh, fc), np.minimum(fl, fc)
    body = fc * size
    tilt = np.zeros(steps)
    free = size * (call == 0) if np.any(call == 0) else size
    if end_target is not None and free.sum() > 0:           # shift to the forecast centre through the uncalled candles
        tilt = (np.log(end_target) - y[now] - body.sum()) * free / free.sum()
    out = []
    op = y[now]
    for j in range(steps):
        cp = op + body[j] + tilt[j]
        hp = max(op + fh[j] * size[j], op, cp)
        lp = min(op + fl[j] * size[j], op, cp)
        out.append([float(np.exp(op)), float(np.exp(hp)), float(np.exp(lp)), float(np.exp(cp))])
        op = cp
    info = {"analog_end": bars.index[ch[m]], "k": len(chosen), "pattern": L, "size": [float(x) for x in size],
            "call": [int(x) for x in call]}
    return out, info
