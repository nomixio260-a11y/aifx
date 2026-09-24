"""Timeframe-generic forecasting: model paths, uncertainty, and walk-forward priors.

Forecasts are expressed as log returns from the origin price in basis points
(1 bp = 0.01 %), which makes errors comparable across pairs and timeframes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .models import Model, PatternMatch, default_models
from .timeutil import HOUR, add_business_days, add_trading_hours, london_date, london_day_end
from .volatility import add_daily_events, daily_step_variance, hourly_variance_path

BP = 1e4
BAND_Z = {"50": 0.6745, "80": 1.2816, "95": 1.9600}
MODEL_KEYS = [m.key for m in default_models()]


@dataclass(frozen=True)
class Timeframe:
    key: str
    label: str
    unit: str
    horizons: tuple[int, ...]
    steps: int                 # bars forecast for the chart
    backtest_origins: int      # walk-forward origins per pair for the prior
    backtest_step: int
    fit_bars: int              # history handed to the models
    half_life: float           # learning discount, in scored forecasts per horizon

    def horizon_label(self, h: int) -> str:
        if self.key == "1h":
            return f"{h}時間後"
        return "翌営業日" if h == 1 else f"{h}営業日後"


TIMEFRAMES = {
    # Walk-forward priors span ~3 months of hourly bars and ~1 year of daily bars.
    "1h": Timeframe("1h", "1時間足", "時間", (1, 4, 24), 24, 120, 12, 3000, 400.0),
    "1d": Timeframe("1d", "日足", "営業日", (1, 5, 10, 20), 20, 60, 5, 1000, 120.0),
}


def origin_of_daily(d) -> datetime:
    return london_day_end(d)


def day_of_origin(origin: datetime):
    return london_date(origin - timedelta(minutes=1))


def target_times(tf: Timeframe, origin: datetime, horizons=None) -> list[datetime]:
    hs = horizons or tf.horizons
    if tf.key == "1h":
        return [add_trading_hours(origin, h) for h in hs]
    d = day_of_origin(origin)
    return [london_day_end(add_business_days(d, h)) for h in hs]


def step_ends(tf: Timeframe, origin: datetime, steps: int) -> list[datetime]:
    if tf.key == "1h":
        return [add_trading_hours(origin, k) for k in range(1, steps + 1)]
    d = day_of_origin(origin)
    return [london_day_end(add_business_days(d, k)) for k in range(1, steps + 1)]


# ------------------------------------------------------------- base models

def model_paths(y: np.ndarray, steps: int, models: list[Model] | None = None) -> tuple[dict[str, np.ndarray], list[int]]:
    """Per-model forecast paths in bp from the last value of ``y``.

    Also returns the positions of the pattern-matching model's closest
    historical analogues (indices into ``y``).
    """
    models = models or default_models()
    out = {}
    analogs: list[int] = []
    for m in models:
        path = m.forecast(y, steps)
        out[m.key] = (path - y[-1]) * BP
        if isinstance(m, PatternMatch):
            analogs = list(m.last_matches)
    return out, analogs


def sigma_steps(tf: Timeframe, bars: pd.DataFrame, origin: datetime, steps: int,
                events: list[dict] | None = None) -> np.ndarray:
    """Raw (uncalibrated) per-step variance in bp^2 for the next ``steps`` bars."""
    y = np.log(bars["close"].to_numpy())
    if tf.key == "1h":
        tail = bars.iloc[-3000:]
        var, _ = hourly_variance_path(list(tail.index.to_pydatetime()), np.log(tail["close"].to_numpy()),
                                      origin, steps, events)
    else:
        var = daily_step_variance(y, steps)
        var = add_daily_events(var, step_ends(tf, origin, steps), origin, events)
    return var * BP * BP


def horizon_sigma(var_steps: np.ndarray, horizons) -> list[float]:
    cum = np.cumsum(var_steps)
    return [float(math.sqrt(cum[h - 1])) for h in horizons]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def combine(model_bp: dict[str, float], weights: list[float], sigma_raw: float, k: float,
            news: float, beta: float, gain: float) -> dict:
    """Blend models, scale the blend by its learned gain, calibrate the spread and
    apply the learned news tilt. A gain near 0 means the models have shown no
    directional skill, so the forecast stays close to "no change"."""
    c0 = float(sum(w * model_bp[key] for w, key in zip(weights, MODEL_KEYS)))
    sigma = sigma_raw * k
    c = gain * c0 + beta * news * sigma
    return {"c0": c0, "c": c, "sigma": sigma, "p_up": norm_cdf(c / sigma) if sigma > 0 else 0.5}


# ----------------------------------------------------- walk-forward priors

def bars_until(tf: Timeframe, bars: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Bars that had completed by ``cutoff``."""
    if tf.key == "1h":
        return bars[bars.index + pd.Timedelta(hours=1) <= pd.Timestamp(cutoff)]
    ends = np.array([london_day_end(d.date()) <= cutoff for d in bars.index], dtype=bool)
    return bars[ends]


def backtest_pair(tf: Timeframe, bars: pd.DataFrame) -> dict:
    """Walk-forward test on one pair: every origin sees only bars up to itself."""
    H = max(tf.horizons)
    bars = bars.iloc[-(tf.fit_bars + tf.backtest_origins * tf.backtest_step + H):]
    y = np.log(bars["close"].to_numpy())
    n = len(y)
    last = n - 1 - H
    origins = [last - i * tf.backtest_step for i in range(tf.backtest_origins)][::-1]
    origins = [o for o in origins if o >= 400]
    rows = []
    for o in origins:
        hist = bars.iloc[: o + 1]
        yy = y[: o + 1]
        paths, _ = model_paths(yy[-tf.fit_bars:], H)
        if tf.key == "1h":
            origin_t = hist.index[-1].to_pydatetime() + HOUR
        else:
            origin_t = london_day_end(hist.index[-1].date())
        var = sigma_steps(tf, hist.iloc[-tf.fit_bars:], origin_t, H)
        sig = horizon_sigma(var, tf.horizons)
        for j, h in enumerate(tf.horizons):
            rows.append({
                "h": h,
                "m": [float(paths[k][h - 1]) for k in MODEL_KEYS],
                "a": float((y[o + h] - y[o]) * BP),
                "s": sig[j],
            })
    return {"rows": rows, "origins": len(origins)}


def summarise_rows(rows: list[dict], horizons) -> dict:
    """Equal-weight ensemble statistics and per-model squared standardized errors."""
    out = {}
    for h in horizons:
        sel = [r for r in rows if r["h"] == h]
        if not sel:
            continue
        m = np.array([r["m"] for r in sel])
        a = np.array([r["a"] for r in sel])
        s = np.array([r["s"] for r in sel])
        z2 = ((m - a[:, None]) / s[:, None]) ** 2
        ens = m.mean(axis=1)
        ze = np.abs(ens - a) / s
        u, v = ens / s, a / s
        rmse = np.sqrt(np.mean((m - a[:, None]) ** 2, axis=0))
        moving = (np.abs(ens) > 1e-9) & (np.abs(a) > 1e-9)
        out[str(h)] = {
            "n": int(len(sel)),
            "mse_z": [float(v) for v in z2.mean(axis=0)],
            "k": float(np.quantile(ze, 0.8) / BAND_Z["80"]),
            "rmse_bp": [float(v) for v in rmse],
            "rmse_rw_bp": float(np.sqrt(np.mean(a ** 2))),
            "ens_rmse_bp": float(np.sqrt(np.mean((ens - a) ** 2))),
            "ens_hit": float(np.mean(np.sign(ens[moving]) == np.sign(a[moving]))) if moving.sum() else None,
            "hit": [
                (float(np.mean(np.sign(m[:, i][(np.abs(m[:, i]) > 1e-9) & (np.abs(a) > 1e-9)])
                               == np.sign(a[(np.abs(m[:, i]) > 1e-9) & (np.abs(a) > 1e-9)])))
                 if ((np.abs(m[:, i]) > 1e-9) & (np.abs(a) > 1e-9)).sum() >= 5 else None)
                for i in range(m.shape[1])
            ],
            "cover80": float(np.mean(ze <= BAND_Z["80"])),
            "suu": float(np.sum(u * u)),
            "suv": float(np.sum(u * v)),
        }
    return out


def backtest_prior(tf: Timeframe, series: dict[str, pd.DataFrame], cutoff: datetime) -> tuple[dict, dict]:
    """Pooled walk-forward prior for learning (goes in the ledger) and per-pair detail (display)."""
    pooled: list[dict] = []
    per_pair = {}
    for code, bars in series.items():
        b = bars_until(tf, bars, cutoff)
        if len(b) < 600:
            continue
        res = backtest_pair(tf, b)
        pooled.extend(res["rows"])
        per_pair[code] = {"origins": res["origins"], "stats": summarise_rows(res["rows"], tf.horizons)}
    stats = summarise_rows(pooled, tf.horizons)
    prior = {
        "tf": tf.key,
        "cutoff": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": MODEL_KEYS,
        "h": {h: {"n": s["n"], "mse_z": [round(v, 6) for v in s["mse_z"]], "k": round(s["k"], 6),
                  "suu": round(s["suu"], 6), "suv": round(s["suv"], 6),
                  # equal-weight ensemble results, for display
                  "hit": None if s["ens_hit"] is None else round(s["ens_hit"], 4),
                  "skill": round(1 - s["ens_rmse_bp"] / s["rmse_rw_bp"], 5) if s["rmse_rw_bp"] > 0 else None,
                  "cover80": round(s["cover80"], 4)}
              for h, s in stats.items()},
    }
    return prior, {"pooled": stats, "pairs": per_pair}
