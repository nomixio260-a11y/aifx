"""Market analysis from prices (descriptive, not forecasts): currency strength,
how turbulent each pair is compared with its own past year, the trend state,
and what is coming up on the economic calendar.

Currency strength: each pair's log change is the base currency's move minus
the quote currency's move. With 7 pairs and 5 currencies the moves are solved
by least squares, with the constraint that they sum to zero (only relative
strength is observable).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import indicators
from .data import CURRENCIES, PAIRS
from .timeutil import iso

WINDOWS = {"24h": ("1h", 24), "5d": ("1d", 5), "20d": ("1d", 20)}


def _design(codes: list[str]) -> np.ndarray:
    A = np.zeros((len(codes) + 1, len(CURRENCIES)))
    for i, code in enumerate(codes):
        A[i, CURRENCIES.index(PAIRS[code].base)] = 1.0
        A[i, CURRENCIES.index(PAIRS[code].quote)] = -1.0
    A[-1, :] = 1.0                      # sum of moves = 0
    return A


def strength(changes: dict[str, float]) -> dict[str, float]:
    """Per-currency move (%, relative to the basket) from pair log changes (%)."""
    codes = [c for c, v in changes.items() if v is not None and math.isfinite(v)]
    if len(codes) < 3:
        return {}
    b = np.array([changes[c] for c in codes] + [0.0])
    s, *_ = np.linalg.lstsq(_design(codes), b, rcond=None)
    return {cur: round(float(v), 4) for cur, v in zip(CURRENCIES, s)}


def _closes(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.DataFrame({code: bars["close"] for code, bars in series.items() if len(bars)})
    return df.dropna()


def strength_history(hourly: dict[str, pd.DataFrame], bars: int = 120) -> dict:
    """Cumulative currency strength over the last ``bars`` hours (common hours of all pairs)."""
    closes = _closes(hourly).iloc[-(bars + 1):]
    if len(closes) < 3:
        return {"t": [], "s": {}}
    rel = np.log(closes / closes.iloc[0]) * 100
    codes = list(rel.columns)
    pinv = np.linalg.pinv(_design(codes))
    mat = np.hstack([rel.to_numpy(), np.zeros((len(rel), 1))]) @ pinv.T
    t = [iso(x.to_pydatetime() + timedelta(hours=1)) for x in rel.index]
    return {"t": t, "s": {cur: [round(float(v), 4) for v in mat[:, i]] for i, cur in enumerate(CURRENCIES)}}


def volatility(hourly: pd.DataFrame, pip: float) -> dict | None:
    """Realized move of the last 24 hours against every 24-hour window of the past year."""
    y = np.log(hourly["close"].to_numpy(dtype=float))
    if len(y) < 24 * 40:
        return None
    r2 = np.diff(y) ** 2
    rv = np.sqrt(np.convolve(r2, np.ones(24), "valid"))          # 24-bar realized volatility
    hist = rv[-24 * 260:]
    now = float(rv[-1])
    pctile = float(np.mean(hist <= now))
    label = "穏やか" if pctile < 0.3 else "普通" if pctile < 0.7 else "やや荒い" if pctile < 0.9 else "荒い"
    price = float(hourly["close"].iloc[-1])
    return {"rv24_pips": round(now * price / pip, 1), "median_pips": round(float(np.median(hist)) * price / pip, 1),
            "percentile": round(pctile, 3), "label": label}


def trend(daily: pd.DataFrame) -> dict | None:
    """Where the price sits against its moving averages, and RSI. Descriptive only."""
    if len(daily) < 80:
        return None
    ind = indicators.compute_all(daily)
    last = ind.iloc[-1]
    close = float(daily["close"].iloc[-1])
    s20, s75 = float(last["sma20"]), float(last["sma75"])
    slope = float(ind["sma20"].iloc[-1] / ind["sma20"].iloc[-6] - 1) * 100
    rsi = float(last["rsi14"])
    if close > s20 > s75 and slope > 0:
        state, bias = "上昇基調", 1
    elif close < s20 < s75 and slope < 0:
        state, bias = "下落基調", -1
    else:
        state, bias = "方向感なし", 0
    return {"state": state, "bias": bias, "vs_sma20_pct": round((close / s20 - 1) * 100, 3),
            "vs_sma75_pct": round((close / s75 - 1) * 100, 3), "sma20_slope_pct": round(slope, 3),
            "rsi14": round(rsi, 1), "rsi_state": "買われすぎ" if rsi >= 70 else "売られすぎ" if rsi <= 30 else "中立"}


def build(hourly: dict[str, pd.DataFrame], daily: dict[str, pd.DataFrame], events: list[dict],
          press: dict[str, dict], ranges: dict[str, dict], now: datetime) -> dict:
    """The market analysis document (api/market.json)."""
    changes: dict[str, dict] = {w: {} for w in WINDOWS}
    for w, (tf, n) in WINDOWS.items():
        src = hourly if tf == "1h" else daily
        for code, bars in src.items():
            if len(bars) > n:
                changes[w][code] = float(np.log(bars["close"].iloc[-1] / bars["close"].iloc[-1 - n]) * 100)
    pairs = []
    for code, pair in PAIRS.items():
        h, d = hourly.get(code), daily.get(code)
        if h is None or not len(h):
            continue
        pairs.append({
            "pair": code, "label": pair.label,
            "change": {w: None if code not in changes[w] else round(changes[w][code], 3) for w in WINDOWS},
            "volatility": volatility(h, pair.pip),
            "trend": trend(d) if d is not None else None,
            "range24": ranges.get(code),
        })
    upcoming = sorted([e for e in events if iso(now) <= e["time"] <= iso(now + timedelta(hours=48))],
                      key=lambda e: e["time"])
    return {
        "at": iso(now),
        "strength": {w: strength(changes[w]) for w in WINDOWS},
        "strength_history": strength_history(hourly),
        "news": {c: press[c]["p"] for c in CURRENCIES},
        "pairs": pairs,
        "events": upcoming[:30],
    }
