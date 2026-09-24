"""Walk-forward backtest, ensemble weighting, prediction intervals and the
per-pair report that the dashboard renders."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import indicators
from .data import Pair
from .models import Model, PatternMatch, default_models

REPORT_HORIZONS = (1, 5, 10, 20)
BAND_Z = {"50": 0.6745, "80": 1.2816, "95": 1.9600}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def volatility_path(y: np.ndarray, horizon: int, lam: float = 0.94,
                    long_window: int = 500, reversion: float = 0.97) -> np.ndarray:
    """Cumulative log-return standard deviation for steps 1..horizon.

    Short-run variance comes from a RiskMetrics EWMA and decays towards the
    long-run variance, so near-term bands react to the current regime while
    one-month bands stay anchored to typical volatility.
    """
    r = np.diff(y)
    ewma = r[:20].var()
    for x in r[20:]:
        ewma = lam * ewma + (1 - lam) * x * x
    long_var = r[-long_window:].var()
    k = np.arange(1, horizon + 1)
    daily = long_var + (ewma - long_var) * reversion ** k
    return np.sqrt(np.cumsum(daily))


def inverse_mse_weights(sq_err: np.ndarray, min_obs: int = 5) -> np.ndarray:
    """sq_err: (models, samples, horizon) with NaN where unavailable -> weights (models, horizon)."""
    counts = np.sum(~np.isnan(sq_err), axis=1)
    total = np.nansum(sq_err, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where((counts > 0) & (total > 0), counts / total, 0.0)  # 1 / MSE
    enough = counts.min(axis=0) >= min_obs
    w[:, ~enough] = 1.0
    return w / w.sum(axis=0, keepdims=True)


@dataclass
class Backtest:
    origins: np.ndarray                    # indices into y used as forecast origins
    preds: dict[str, np.ndarray]           # model key -> (origins, horizon) predicted log price
    ensemble: np.ndarray                   # (origins, horizon), online-weighted
    sigma: np.ndarray                      # (origins, horizon) cumulative std at each origin
    actual: np.ndarray                     # (origins, horizon)
    base: np.ndarray                       # y at each origin
    keys: list[str] = field(default_factory=list)

    def sq_errors(self) -> np.ndarray:
        return np.stack([(self.preds[k] - self.actual) ** 2 for k in self.keys])


def run_backtest(y: np.ndarray, models: list[Model], horizon: int = 20,
                 test_days: int = 250, step: int = 5) -> Backtest:
    """Rolling-origin evaluation. Each origin only sees data up to itself, and the
    ensemble weights at an origin only use errors that were already realised."""
    n = len(y)
    last_origin = n - 1 - horizon
    first_origin = max(last_origin - test_days, max(m.min_history for m in models))
    origins = np.arange(last_origin, first_origin - 1, -step)[::-1]
    keys = [m.key for m in models]
    H = horizon
    steps = np.arange(1, H + 1)

    preds = {k: np.empty((len(origins), H)) for k in keys}
    actual = y[origins[:, None] + steps[None, :]]
    base = y[origins]
    sigma = np.empty((len(origins), H))
    for i, t in enumerate(origins):
        hist = y[: t + 1]
        for m in models:
            preds[m.key][i] = m.forecast(hist, H)
        sigma[i] = volatility_path(hist, H)

    ensemble = np.empty((len(origins), H))
    stacked = np.stack([preds[k] for k in keys])            # (M, O, H)
    sq = (stacked - actual[None]) ** 2
    realised_at = origins[:, None] + steps[None, :]           # when each error becomes known
    for i, t in enumerate(origins):
        known = np.where(realised_at <= t, sq, np.nan)        # (M, O, H)
        w = inverse_mse_weights(known)
        ensemble[i] = (w * stacked[:, i, :]).sum(axis=0)
    return Backtest(origins, preds, ensemble, sigma, actual, base, keys)


def _metrics(pred: np.ndarray, actual: np.ndarray, base: np.ndarray, rw_rmse: float | None) -> dict:
    err = pred - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pch = pred - base
    ach = actual - base
    moving = (np.abs(pch) > 1e-12) & (np.abs(ach) > 1e-12)
    hit = float(np.mean(np.sign(pch[moving]) == np.sign(ach[moving]))) if moving.sum() >= 5 else None
    return {
        "rmse_pct": round(rmse * 100, 4),
        "mae_pct": round(float(np.mean(np.abs(err))) * 100, 4),
        "hit": None if hit is None else round(hit, 4),
        "skill": None if not rw_rmse else round(1 - rmse / rw_rmse, 4),
        "n": int(len(err)),
    }


def summarise_backtest(bt: Backtest, horizons=REPORT_HORIZONS) -> dict:
    horizons = [h for h in horizons if h <= bt.actual.shape[1]]
    table: dict[str, dict[str, dict]] = {}
    for key in bt.keys + ["ensemble"]:
        p = bt.ensemble if key == "ensemble" else bt.preds[key]
        table[key] = {}
        for h in horizons:
            j = h - 1
            rw = float(np.sqrt(np.mean((bt.preds["rw"][:, j] - bt.actual[:, j]) ** 2))) if "rw" in bt.preds else None
            table[key][str(h)] = _metrics(p[:, j], bt.actual[:, j], bt.base, None if key == "rw" else rw)
    coverage = {}
    for band, z in BAND_Z.items():
        inside = np.abs(bt.actual - bt.ensemble) <= z * bt.sigma
        coverage[band] = {str(h): round(float(inside[:, h - 1].mean()), 4) for h in horizons}
    return {"horizons": horizons, "metrics": table, "coverage": coverage, "origins": int(len(bt.origins))}


def _outlook_label(p_up: float) -> str:
    if p_up >= 0.55:
        return "上昇"
    if p_up <= 0.45:
        return "下落"
    return "横ばい"


def _round(a, d):
    return [None if (x is None or not np.isfinite(x)) else round(float(x), d) for x in a]


def build_report(pair: Pair, df: pd.DataFrame, source: str, horizon: int = 20,
                 history_days: int = 520, test_days: int = 250, step: int = 5,
                 models: list[Model] | None = None) -> dict:
    models = models or default_models()
    df = df.dropna()
    y = np.log(df["close"].to_numpy())
    dec = pair.decimals + 1
    H = horizon

    bt = run_backtest(y, models, horizon=H, test_days=test_days, step=step)
    weights = inverse_mse_weights(bt.sq_errors())               # (M, H)

    paths = {}
    analogs: list[str] = []
    for m in models:
        paths[m.key] = m.forecast(y, H)
        if isinstance(m, PatternMatch):
            analogs = [df.index[i].strftime("%Y-%m-%d") for i in m.last_matches]
    stacked = np.stack([paths[m.key] for m in models])
    center = (weights * stacked).sum(axis=0)
    sig = volatility_path(y, H)
    last_y = y[-1]
    last_close = float(df["close"].iloc[-1])

    future = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=H)
    bands = {
        name: {"lower": _round(np.exp(center - z * sig), dec), "upper": _round(np.exp(center + z * sig), dec)}
        for name, z in BAND_Z.items()
    }

    outlook = []
    for h in (1, 5, 10, 20):
        if h > H:
            continue
        j = h - 1
        p_up = _norm_cdf((center[j] - last_y) / sig[j])
        ups = int(sum(paths[m.key][j] > last_y + 1e-12 for m in models if m.key != "rw"))
        outlook.append({
            "h": h,
            "date": future[j].strftime("%Y-%m-%d"),
            "price": round(float(np.exp(center[j])), dec),
            "change_pct": round(float(np.expm1(center[j] - last_y) * 100), 3),
            "change_pips": round(float((np.exp(center[j]) - last_close) / pair.pip), 1),
            "p_up": round(p_up, 4),
            "lo80": bands["80"]["lower"][j],
            "hi80": bands["80"]["upper"][j],
            "label": _outlook_label(p_up),
            "models_up": ups,
            "models_total": len(models) - 1,
        })

    ind = indicators.compute_all(df)
    hist = df.iloc[-history_days:]
    hind = ind.loc[hist.index]
    d = pair.decimals

    # A few past ensemble forecasts next to what actually happened.
    hindcasts = []
    for i in range(len(bt.origins) - 1, -1, -max(1, 20 // step)):
        t = int(bt.origins[i])
        dates = df.index[t + 1: t + 1 + H]
        hindcasts.append({
            "origin": df.index[t].strftime("%Y-%m-%d"),
            "dates": [x.strftime("%Y-%m-%d") for x in dates],
            "pred": _round(np.exp(bt.ensemble[i]), dec),
            "actual": _round(np.exp(bt.actual[i]), dec),
        })
        if len(hindcasts) == 6:
            break

    atr = float(ind["atr14"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    return {
        "pair": pair.code,
        "label": pair.label,
        "name": pair.name,
        "decimals": d,
        "pip": pair.pip,
        "source": source,
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "last_close": round(last_close, dec),
        "prev_close": round(prev_close, dec),
        "change_1d_pct": round((last_close / prev_close - 1) * 100, 3),
        "atr14": round(atr, dec),
        "atr14_pips": round(atr / pair.pip, 1),
        "history": {
            "dates": [x.strftime("%Y-%m-%d") for x in hist.index],
            "ohlc": [[round(float(v), dec) for v in row] for row in hist[["open", "close", "low", "high"]].to_numpy()],
            "sma20": _round(hind["sma20"], dec),
            "sma75": _round(hind["sma75"], dec),
            "bb_upper": _round(hind["bb_upper"], dec),
            "bb_lower": _round(hind["bb_lower"], dec),
            "rsi14": _round(hind["rsi14"], 2),
        },
        "forecast": {
            "dates": [x.strftime("%Y-%m-%d") for x in future],
            "ensemble": _round(np.exp(center), dec),
            "models": {m.key: _round(np.exp(paths[m.key]), dec) for m in models},
            "bands": bands,
            "sigma_pct": _round(sig * 100, 4),
        },
        "weights": {m.key: _round(weights[i], 4) for i, m in enumerate(models)},
        "outlook": outlook,
        "backtest": summarise_backtest(bt),
        "hindcasts": hindcasts,
        "technical": indicators.technical_summary(df, ind),
        "analogs": analogs,
    }


def build_bundle(reports: list[dict], models: list[Model] | None = None, horizon: int = 20) -> dict:
    models = models or default_models()
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon": horizon,
        "models": [{"key": m.key, "name": m.name, "description": m.description} for m in models]
        + [{"key": "ensemble", "name": "アンサンブル", "description": "各モデルを過去の予測誤差の逆数で重み付けした平均。"}],
        "pairs": reports,
    }
