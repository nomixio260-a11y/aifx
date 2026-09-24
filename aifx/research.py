"""Walk-forward research on long history: which changes improve accuracy out of sample.

Stage 1 (slow, cached per pair in data/research/): at every origin, each model
forecasts from data up to that origin only; the outcome, volatility-model
variants and candidate signals (interest-rate carry, momentum, time of day)
are recorded.

Stage 2 (fast): candidate methods are scored. Anything with a parameter (a
gain, a regression coefficient, an interval scale, a distribution shape, a
volatility setting) is chosen on the older "tune" period only and then scored
on the later "test" period, which is never used for choosing. Only changes
that also hold on the test period are adopted in the live forecaster.

    aifx research --download   # fetch price history (Yahoo Finance) and short rates (FRED)
    aifx research              # run both stages, write research/report.md and research/results.json
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS
from .engine import BAND_NU, BAND_Z, BP, MODEL_KEYS, model_paths
from .learning import GAIN_ESS, GAIN_LAMBDA
from .stats import diebold_mariano, wilson
from .volatility import (RANGE_WINDOW, RV_WINDOW, daily_step_variance, hourly_variance_path, range_variance,
                         realized_daily_variance, scale_proxy)

OUT_DIR = Path("data/research")
REPORT_DIR = Path("research")
DAILY_H = (1, 5, 10, 20)
HOURLY_H = (1, 4, 24)
DAILY_SPLIT = "2017-01-01"
# volatility variants: name -> (EWMA decay, reversion[, seasonal])
DAILY_VOL = {"live": (0.94, 0.97), "slow": (0.97, 0.99), "fast_rev": (0.94, 0.90), "long_only": (0.94, 0.0)}
HOURLY_VOL = {"live": (0.97, 0.985, True), "no_season": (0.97, 0.985, False),
              "fast": (0.94, 0.985, True), "slow": (0.99, 0.995, True)}
MOM_K = {"daily": (20, 60, 120, 250), "hourly": (4, 24, 120)}
OLD_GAIN_LAMBDA = 11.0
VOL_NAMES = {
    "live": "従来 (終値の変化, EWMA)", "slow": "ゆっくり反応", "fast_rev": "平常水準へ早く戻る",
    "long_only": "長期平均のみ (EWMAなし)", "no_season": "時間帯の補正なし", "fast": "速く反応",
    "range": "高値・安値の幅 (Parkinson)", "blend": "終値の変化と高値・安値の平均",
}


# ------------------------------------------------------------- stage 1

def _daily_pair(code: str) -> dict:
    pair = PAIRS[code]
    df = history.load_daily(code)
    y = np.log(df["close"].to_numpy())
    rates = history.rates_panel(df.index)
    carry = (rates[pair.base] - rates[pair.quote]).to_numpy()
    H = max(DAILY_H)
    rows = []
    for o in range(750, len(y) - 1 - H, 5):
        hist = y[max(0, o + 1 - 1000): o + 1]
        paths, _ = model_paths(hist, H)
        var = {k: np.cumsum(daily_step_variance(y[: o + 1], H, lam, 500, rev)) * BP * BP
               for k, (lam, rev) in DAILY_VOL.items()}
        r = np.diff(y[o - 20: o + 1])
        rows.append({
            "t": df.index[o].strftime("%Y-%m-%d"),
            "m": [[float(paths[k][h - 1]) for k in MODEL_KEYS] for h in DAILY_H],
            "a": [float((y[o + h] - y[o]) * BP) for h in DAILY_H],
            "s": {k: [float(math.sqrt(v[h - 1])) for h in DAILY_H] for k, v in var.items()},
            "carry": None if np.isnan(carry[o]) else float(carry[o]),
            "mom": [float((y[o] - y[o - k]) * BP) for k in MOM_K["daily"]],
            "rv": float(r.std() * BP),
        })
    return {"pair": code, "tf": "1d", "rows": rows}


def _hourly_pair(code: str) -> dict:
    df = history.load_hourly(code)
    y = np.log(df["close"].to_numpy())
    times = list(df.index.to_pydatetime())
    H = max(HOURLY_H)
    rows = []
    for o in range(3000, len(y) - 1 - H, 6):
        lo = o + 1 - 3000
        hist = y[lo: o + 1]
        paths, _ = model_paths(hist, H)
        origin = times[o] + timedelta(hours=1)
        sig = {}
        for k, (lam, rev, seas) in HOURLY_VOL.items():
            v, _ = hourly_variance_path(times[lo: o + 1], hist, origin, H, None, lam, rev, seas)
            cum = np.cumsum(v) * BP * BP
            sig[k] = [float(math.sqrt(cum[h - 1])) for h in HOURLY_H]
        rows.append({
            "t": times[o].strftime("%Y-%m-%dT%H:%M"),
            "m": [[float(paths[k][h - 1]) for k in MODEL_KEYS] for h in HOURLY_H],
            "a": [float((y[o + h] - y[o]) * BP) for h in HOURLY_H],
            "s": sig,
            "mom": [float((y[o] - y[o - k]) * BP) for k in MOM_K["hourly"]],
            "hour": times[o].hour,
        })
    return {"pair": code, "tf": "1h", "rows": rows}


def _hourly_extra(code: str) -> dict:
    """Range-based volatility (the live functions) and the last hour's move, at the stage-1 origins."""
    df = history.load_hourly(code)
    y = np.log(df["close"].to_numpy())
    times = list(df.index.to_pydatetime())
    H = max(HOURLY_H)
    rows = []
    for o in range(3000, len(y) - 1 - H, 6):
        lo = o + 1 - 3000
        hist = y[lo: o + 1]
        r = np.diff(hist)
        rng = scale_proxy(range_variance(df.iloc[lo: o + 1]), r, RANGE_WINDOW)
        rng = r * r if rng is None else rng
        origin = times[o] + timedelta(hours=1)
        sig = {}
        for name, sq in (("range", rng), ("blend", 0.5 * (r * r + rng))):
            v, _ = hourly_variance_path(times[lo: o + 1], hist, origin, H, None, 0.97, 0.985, True, sq=sq)
            cum = np.cumsum(v) * BP * BP
            sig[name] = [float(math.sqrt(cum[h - 1])) for h in HOURLY_H]
        rows.append({"t": times[o].strftime("%Y-%m-%dT%H:%M"), "s": sig, "r1": float(r[-1] * BP)})
    return {"pair": code, "tf": "1h", "rows": rows}


PROFILE_VARIANTS = {"p1500_s25": (1500, 0.25), "p1500_s10": (1500, 0.10), "p1500_s0": (1500, 0.0),
                    "p3000_s25": (3000, 0.25), "p3000_s10": (3000, 0.10), "p3000_s0": (3000, 0.0)}


def _hourly_profile(code: str) -> dict:
    """Range-based hourly volatility with different hour-of-day profiles (window, neighbour smoothing)."""
    df = history.load_hourly(code)
    y = np.log(df["close"].to_numpy())
    times = list(df.index.to_pydatetime())
    H = max(HOURLY_H)
    rows = []
    for o in range(3000, len(y) - 1 - H, 6):
        lo = o + 1 - 3000
        hist = y[lo: o + 1]
        r = np.diff(hist)
        rng = scale_proxy(range_variance(df.iloc[lo: o + 1]), r, RANGE_WINDOW)
        rng = r * r if rng is None else rng
        origin = times[o] + timedelta(hours=1)
        sig = {}
        for name, (win, sm) in PROFILE_VARIANTS.items():
            v, _ = hourly_variance_path(times[lo: o + 1], hist, origin, H, None, 0.97, 0.985, True, sq=rng,
                                        profile_window=win, smooth=sm)
            cum = np.cumsum(v) * BP * BP
            sig[name] = [float(math.sqrt(cum[h - 1])) for h in HOURLY_H]
        rows.append({"t": times[o].strftime("%Y-%m-%dT%H:%M"), "s": sig})
    return {"pair": code, "tf": "1h", "rows": rows}


def _daily_rv(code: str) -> dict:
    """Daily volatility measured from hourly bars (realized variance), for the ~2 years with hourly data.

    An origin every business day; each origin only uses days up to itself.
    """
    d = history.load_daily(code)
    h = history.load_hourly(code)
    rv = realized_daily_variance(h, h.index[-1].to_pydatetime() + timedelta(hours=1))
    y = np.log(d["close"].to_numpy())
    r = np.diff(y)
    rv_d = rv.reindex(d.index[1:]).to_numpy(dtype=float)       # aligned with r
    H = max(DAILY_H)
    rows = []
    start = int(np.searchsorted(d.index, rv.index[0] + pd.Timedelta(days=90)))
    for o in range(max(start, 1000), len(y) - 1 - H):
        yy = y[o + 1 - 1000: o + 1]
        rr = r[o - 999: o]
        alt = scale_proxy(rv_d[o - 999: o], rr, RV_WINDOW)
        alt = rr * rr if alt is None else alt
        sig = {"live": np.cumsum(daily_step_variance(yy, H, 0.94, 500, 0.97)) * BP * BP,
               "live_rev90": np.cumsum(daily_step_variance(yy, H, 0.94, 500, 0.90)) * BP * BP}
        for proxy, sq in (("rv", alt), ("blend", 0.5 * (rr * rr + alt))):
            for lam in (0.8, 0.9, 0.94):
                for rev in (0.90, 0.97):
                    sig[f"{proxy}_{lam:g}_{rev:g}"] = np.cumsum(daily_step_variance(yy, H, lam, 500, rev, sq=sq)) * BP * BP
        rows.append({
            "t": d.index[o].strftime("%Y-%m-%d"),
            "a": [float((y[o + k] - y[o]) * BP) for k in DAILY_H],
            "s": {k: [float(math.sqrt(v[k2 - 1])) for k2 in DAILY_H] for k, v in sig.items()},
        })
    return {"pair": code, "tf": "1d_rv", "rows": rows}


_JOBS = {"1d": _daily_pair, "1h": _hourly_pair, "1h_extra": _hourly_extra, "1d_rv": _daily_rv,
         "1h_profile": _hourly_profile}


def _run(job):
    code, tf = job
    path = OUT_DIR / f"{code}_{tf}.json"
    if path.exists():
        return str(path)
    res = _JOBS[tf](code)
    path.write_text(json.dumps(res), encoding="utf-8")
    return str(path)


def compute(workers: int = 4, timeframes=tuple(_JOBS), log=print) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(c, tf) for tf in timeframes for c in PAIRS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path in ex.map(_run, jobs):
            log(f"done {path}")


# ------------------------------------------------------------- stage 2

class Panel:
    """All pairs' walk-forward rows for one timeframe, as arrays."""

    def __init__(self, tf: str):
        self.tf = tf
        self.h = DAILY_H if tf == "1d" else HOURLY_H
        rows, pairs = [], []
        for code in PAIRS:
            data = json.loads((OUT_DIR / f"{code}_{tf}.json").read_text())
            rows.extend(data["rows"])
            pairs.extend([code] * len(data["rows"]))
        self.pair = np.array(pairs)
        self.t = np.array([r["t"] for r in rows])
        self.time = self.t.astype("datetime64[m]")
        self.m = np.array([r["m"] for r in rows])              # (N, H, models)
        self.a = np.array([r["a"] for r in rows])              # (N, H)
        self.s = {k: np.array([r["s"][k] for r in rows]) for k in rows[0]["s"]}
        self.mom = np.array([r["mom"] for r in rows])
        self.carry = np.array([np.nan if r.get("carry") is None else r["carry"] for r in rows])
        self.hour = np.array([r.get("hour", 0) for r in rows])
        self.r1 = np.full(len(rows), np.nan)
        if tf == "1h":
            extra = []
            for code in PAIRS:
                extra.extend(json.loads((OUT_DIR / f"{code}_1h_extra.json").read_text())["rows"])
            assert [e["t"] for e in extra] == list(self.t)
            for k in extra[0]["s"]:
                self.s[k] = np.array([e["s"][k] for e in extra])
            self.r1 = np.array([e["r1"] for e in extra])
            cut = np.quantile(self.time.astype("int64"), 0.6)
            self.tune = self.time.astype("int64") < cut
        else:
            self.tune = self.t < DAILY_SPLIT
        self.test = ~self.tune
        self.step = 5 if tf == "1d" else 6


class RvPanel:
    """Daily origins of the last ~2 years with realized-variance volatility variants."""

    def __init__(self):
        rows = []
        for code in PAIRS:
            rows.extend(json.loads((OUT_DIR / f"{code}_1d_rv.json").read_text())["rows"])
        self.tf = "1d"
        self.h = DAILY_H
        self.t = np.array([r["t"] for r in rows])
        self.a = np.array([r["a"] for r in rows])
        self.s = {k: np.array([r["s"][k] for r in rows]) for k in rows[0]["s"]}
        days = sorted(set(self.t))
        self.tune = self.t < days[int(len(days) * 0.6)]
        self.test = ~self.tune
        self.step = 1


def _span(P, mask) -> list[str]:
    t = sorted(P.t[mask])
    return [t[0][:10], t[-1][:10]]


def _dm_by_date(t, loss_model, loss_base, lag):
    """DM test on the cross-pair average loss per date (pairs move together)."""
    df = pd.DataFrame({"t": t, "d": loss_model - loss_base}).groupby("t")["d"].mean().sort_index()
    return diebold_mariano(df.to_numpy(), np.zeros(len(df)), lag)


def point_metrics(P, c: np.ndarray, mask: np.ndarray) -> dict:
    """c: (N, H) centre forecasts in bp. Skill = RMSE improvement over "no change"."""
    out = {}
    for j, h in enumerate(P.h):
        sel = mask & np.isfinite(c[:, j])
        a, f = P.a[sel, j], c[sel, j]
        rmse = float(np.sqrt(np.mean((f - a) ** 2)))
        rmse_rw = float(np.sqrt(np.mean(a ** 2)))
        moving = (np.abs(f) > 1e-9) & (np.abs(a) > 1e-9)
        hits, n_dir = int(np.sum(np.sign(f[moving]) == np.sign(a[moving]))), int(moving.sum())
        stat, p = _dm_by_date(P.t[sel], (f - a) ** 2, a ** 2, max(0, math.ceil(h / P.step) - 1))
        lo, hi = wilson(hits, n_dir)
        out[str(h)] = {"n": int(sel.sum()), "skill": 1 - rmse / rmse_rw if rmse_rw else None,
                       "hit": hits / n_dir if n_dir else None, "hit_lo": lo, "hit_hi": hi, "dm": stat, "dm_p": p}
    return out


def _fit_wls(X, a, s, lam):
    """Weighted ridge (weights 1/sigma^2) of outcome on features, no intercept."""
    w = 1.0 / (s * s)
    A = (X * w[:, None]).T @ X + lam * np.eye(X.shape[1]) * np.mean(w * np.sum(X * X, axis=1)) / X.shape[1]
    return np.linalg.solve(A, (X * w[:, None]).T @ a)


def fitted_centre(P, features: np.ndarray, lam: float = 0.0, sig: str = "live") -> tuple[np.ndarray, list]:
    """Fit on tune (per horizon) and predict everywhere. features: (N, H, F)."""
    c = np.full(P.a.shape, np.nan)
    coefs = []
    for j in range(len(P.h)):
        X = features[:, j, :]
        ok = np.all(np.isfinite(X), axis=1)
        fit = P.tune & ok
        b = _fit_wls(X[fit], P.a[fit, j], P.s[sig][fit, j], lam)
        coefs.append([round(float(v), 4) for v in b])
        c[ok, j] = X[ok] @ b
    return c, coefs


def hour_drift_centre(P, lam0: float = 200.0, sig: str = "live") -> np.ndarray:
    """Per pair and hour of day: shrunk mean standardized move over the tune period."""
    c = np.full(P.a.shape, np.nan)
    for j in range(len(P.h)):
        s = P.s[sig][:, j]
        v = P.a[:, j] / s
        for code in np.unique(P.pair):
            for hr in range(24):
                cell = (P.pair == code) & (P.hour == hr)
                fit = cell & P.tune
                c[cell, j] = float(np.sum(v[fit]) / (fit.sum() + lam0)) * s[cell]
    return c


def learner_sim(P, window_days: float | None, lam: float, ess: float = GAIN_ESS, sig: str = "live") -> np.ndarray:
    """Replays the live learning rule through time.

    At each origin the gain and model weights come only from earlier origins
    whose outcome was already known (with a conservative embargo), inside a
    trailing window, as the live walk-forward prior sets them.
    """
    c = np.full(P.a.shape, np.nan)
    T = P.time
    ens_eq = P.m.mean(axis=2)
    for j, h in enumerate(P.h):
        emb = np.timedelta64(int((h * 1.5 + 4) * 24 * 60), "m") if P.tf == "1d" else np.timedelta64((h + 72) * 60, "m")
        s = P.s[sig][:, j]
        u, v = ens_eq[:, j] / s, P.a[:, j] / s
        z2 = ((P.m[:, j, :] - P.a[:, j:j + 1]) / s[:, None]) ** 2
        for d in np.unique(T):
            train = T < d - emb
            if window_days is not None:
                train &= T >= d - np.timedelta64(int(window_days * 24 * 60), "m")
            if train.sum() < 50:
                continue
            g = ess * float(np.sum(u[train] * v[train])) / (lam + ess * float(np.sum(u[train] ** 2)))
            w = 1 / np.maximum(z2[train].mean(axis=0), 1e-9)
            at = T == d
            c[at, j] = float(np.clip(g, -0.5, 1.0)) * (P.m[at, j, :] @ (w / w.sum()))
    return c


# ---------------------------------------------------------- range forecasts

def _npdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _ncdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


def _k80(P, key: str, j: int) -> float:
    """Interval scale fitted on the tune period so that 80 % of outcomes fall inside."""
    return float(np.quantile(np.abs(P.a[P.tune, j]) / P.s[key][P.tune, j], 0.8) / BAND_Z["80"])


def _crps(P, key: str, j: int, mask) -> np.ndarray:
    """Gaussian CRPS (bp) of a "no change" centre with the tune-calibrated spread."""
    sd = P.s[key][mask, j] * _k80(P, key, j)
    z = P.a[mask, j] / sd
    return sd * (z * (2 * _ncdf(z) - 1) + 2 * _npdf(z) - 1 / math.sqrt(math.pi))


def vol_metrics(P) -> dict:
    """Per volatility variant: tune/test CRPS relative to the live one, test 80 % coverage, DM test on test."""
    out = {}
    base = {j: (_crps(P, "live", j, P.tune), _crps(P, "live", j, P.test)) for j in range(len(P.h))}
    for key in P.s:
        res = {}
        for j, h in enumerate(P.h):
            tune, test = _crps(P, key, j, P.tune), _crps(P, key, j, P.test)
            _, p = _dm_by_date(P.t[P.test], test, base[j][1], max(0, math.ceil(h / P.step) - 1))
            z = np.abs(P.a[P.test, j]) / (P.s[key][P.test, j] * _k80(P, key, j))
            res[str(h)] = {"tune_crps": float(tune.mean()), "tune_rel": float(tune.mean() / base[j][0].mean() - 1),
                           "test_crps": float(test.mean()), "test_rel": float(test.mean() / base[j][1].mean() - 1),
                           "p": p if key != "live" else None, "cover80": float(np.mean(z <= BAND_Z["80"]))}
        out[key] = res
    return out


_TAU = np.linspace(0.005, 0.995, 199)
_LEVELS = {"50": 0.75, "80": 0.90, "95": 0.975}
SHAPES_NU = (3, 4, 5, 6, 8, 10, 15, 30)


def _quantiles(pdf, lo, hi) -> np.ndarray:
    x = np.linspace(lo, hi, 400001)
    cdf = np.cumsum(pdf(x))
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp(_TAU, cdf, x)


def _crps_q(a: np.ndarray, scale: np.ndarray, q: np.ndarray) -> np.ndarray:
    """CRPS of centred predictive distributions given by standardized quantiles ``q`` at _TAU."""
    e = a[:, None] - scale[:, None] * q[None, :]
    return 2 * np.mean(np.maximum(_TAU * e, (_TAU - 1) * e), axis=1)


def shape_metrics(P, key: str, fit=None) -> dict:
    """Band shape: normal vs Student-t, both scaled so the 80 % band matches the fit period
    (as the live learner does); the t degrees of freedom are chosen by CRPS on the fit period."""
    fit = P.tune if fit is None else fit
    gq = _quantiles(lambda x: np.exp(-0.5 * x * x), -12, 12)
    tq = {nu: _quantiles(lambda x, nu=nu: (1 + x * x / nu) ** (-(nu + 1) / 2), -400, 400) for nu in SHAPES_NU}
    out = {}
    for j, h in enumerate(P.h):
        z = P.a[:, j] / P.s[key][:, j]
        q80 = float(np.quantile(np.abs(z[fit]), 0.8))

        def norm(q):
            return q / np.interp(0.9, _TAU, q) * q80

        nu = min(SHAPES_NU, key=lambda n: float(np.mean(_crps_q(z[fit], np.ones(fit.sum()), norm(tq[n])))))
        res = {"nu": nu}
        for name, q in (("normal", norm(gq)), ("t", norm(tq[nu]))):
            zz = z[P.test]
            res[name] = {
                "cover": {lv: float(np.mean(np.abs(zz) <= np.interp(tau, _TAU, q))) for lv, tau in _LEVELS.items()},
                "crps": float(np.mean(_crps_q(P.a[P.test, j], P.s[key][P.test, j], q))),
            }
        out[str(h)] = res
    return out


# ------------------------------------------------------------- evaluation

def evaluate(tf: str) -> dict:
    P = Panel(tf)
    H = len(P.h)
    res = {"tf": tf, "n": int(len(P.t)), "tune": _span(P, P.tune), "test": _span(P, P.test),
           "point": {}, "coefs": {}}
    zero = np.zeros(P.a.shape)
    ens = P.m.mean(axis=2)
    cands = {"変化なし (ランダムウォーク)": zero, "6モデルの均等平均": ens}
    for i, k in enumerate(MODEL_KEYS[1:], start=1):
        cands[f"単独モデル: {k}"] = P.m[:, :, i]
    c, co = fitted_centre(P, ens[:, :, None])
    cands["均等平均 × ゲイン (調整期間で推定)"] = c
    res["coefs"]["gain"] = co
    for i, k in enumerate(MOM_K["daily" if tf == "1d" else "hourly"]):
        c, co = fitted_centre(P, np.repeat(P.mom[:, i][:, None], H, axis=1)[:, :, None])
        cands[f"モメンタム (過去{k}本の値動き)"] = c
        res["coefs"][f"mom{k}"] = co
    if tf == "1d":
        hs = np.array(P.h, dtype=float)
        carry_bp = P.carry[:, None] * hs[None, :] * 100 / 252          # rate differential over h days, in bp
        c, co = fitted_centre(P, carry_bp[:, :, None])
        cands["金利差 (キャリー)"] = c
        res["coefs"]["carry"] = co
        combo = np.concatenate([ens[:, :, None], carry_bp[:, :, None], np.repeat(P.mom[:, None, :], H, axis=1)], axis=2)
        c, co = fitted_centre(P, combo, 10.0)
        cands["組合せ (均等平均+金利差+モメンタム)"] = c
        res["coefs"]["combo"] = co
    else:
        cands["時間帯ごとの平均的な値動き"] = hour_drift_centre(P)
        f = np.repeat((P.r1 / P.s["live"][:, 0])[:, None], H, axis=1)[:, :, None] * P.s["live"][:, :, None]
        c, co = fitted_centre(P, f)
        cands["直前1時間の値動き (反転/継続)"] = c
        res["coefs"]["r1"] = co
        combo = np.concatenate([ens[:, :, None], np.repeat(P.mom[:, None, :], H, axis=1)], axis=2)
        c, co = fitted_centre(P, combo, 10.0)
        cands["組合せ (均等平均+モメンタム)"] = c
        res["coefs"]["combo"] = co
    for name, c in cands.items():
        res["point"][name] = {"tune": point_metrics(P, c, P.tune), "test": point_metrics(P, c, P.test)}

    # the live learning rule replayed through time
    live_window = 420 if tf == "1d" else 84
    sims = {
        f"旧設定 (λ={OLD_GAIN_LAMBDA:g})": (live_window, OLD_GAIN_LAMBDA, GAIN_ESS),
        f"新設定 (λ={GAIN_LAMBDA:g})": (live_window, GAIN_LAMBDA, GAIN_ESS),
        "縮小なし": (live_window, 0.0, 1.0),
        f"学習期間を全期間に (λ={OLD_GAIN_LAMBDA:g})": (None, OLD_GAIN_LAMBDA, GAIN_ESS),
    }
    res["sim"] = {}
    for name, (win, lam, ess) in sims.items():
        c = learner_sim(P, win, lam, ess)
        res["sim"][name] = {"tune": point_metrics(P, c, P.tune), "test": point_metrics(P, c, P.test)}

    res["vol"] = vol_metrics(P)
    new_vol = "fast_rev" if tf == "1d" else "range"
    res["shape"] = {"before": shape_metrics(P, "live"), "after": shape_metrics(P, new_vol), "after_vol": new_vol}
    return res


def evaluate_rv() -> dict:
    P = RvPanel()
    res = {"n": int(len(P.t)), "tune": _span(P, P.tune), "test": _span(P, P.test), "vol": vol_metrics(P)}
    ranked = sorted((k for k in P.s if k != "live"),
                    key=lambda k: np.mean([res["vol"][k][str(h)]["tune_rel"] for h in P.h]))
    res["chosen"] = ranked[0]
    res["ranked"] = ranked
    res["shape"] = {"before": shape_metrics(P, "live"), "after": shape_metrics(P, ranked[0])}
    return res


# ------------------------------------------------ second round of ideas

def _replay_k(P, key: str, j: int, window_days, emb_hours: float) -> np.ndarray:
    """Interval scale at each origin from a trailing window of already-scored outcomes."""
    T = P.t.astype("datetime64[m]")
    z = np.abs(P.a[:, j]) / P.s[key][:, j]
    order = np.argsort(T, kind="stable")
    Ts, zs = T[order], z[order]
    emb = np.timedelta64(int(emb_hours * 60), "m")
    k = np.full(len(z), np.nan)
    for d in np.unique(T):
        hi = np.searchsorted(Ts, d - emb)
        lo = 0 if window_days is None else np.searchsorted(Ts, d - emb - np.timedelta64(int(window_days * 1440), "m"))
        if hi - lo >= 20:
            k[T == d] = np.quantile(zs[lo:hi], 0.8) / BAND_Z["80"]
    return k


def _gauss_crps(a, sd):
    z = a / sd
    return sd * (z * (2 * _ncdf(z) - 1) + 2 * _npdf(z) - 1 / math.sqrt(math.pi))


def study_k_window(P, key: str, windows: list, live, emb) -> dict:
    """How far back the interval scale should look (live: ~420 days daily, ~84 days hourly)."""
    out = {}
    for j, h in enumerate(P.h):
        ks = {w: _replay_k(P, key, j, w, emb(h)) for w in windows}
        ok = np.all([np.isfinite(k) for k in ks.values()], axis=0)
        res = {}
        for w, k in ks.items():
            res[str(w)] = {nm: float(_gauss_crps(P.a[m & ok, j], P.s[key][m & ok, j] * k[m & ok]).mean())
                           for nm, m in (("tune", P.tune), ("test", P.test))}
            res[str(w)]["cover_test"] = float(np.mean(np.abs(P.a[P.test & ok, j]) <= BAND_Z["80"] * P.s[key][P.test & ok, j] * k[P.test & ok]))
        base = res[str(live)]
        best = min(res, key=lambda w: res[w]["tune"])
        out[str(h)] = {"best": best, "tune_rel": res[best]["tune"] / base["tune"] - 1,
                       "test_rel": res[best]["test"] / base["test"] - 1, "all": res}
    return out


def study_asymmetry(P, key: str) -> dict:
    """Separate lower and upper band widths (fitted on tune) vs symmetric bands."""
    out = {}
    for j, h in enumerate(P.h):
        z = P.a[:, j] / P.s[key][:, j]
        zt, zx = z[P.tune], z[P.test]
        res = {}
        for tau in (0.05, 0.10, 0.90, 0.95):
            qs = np.quantile(np.abs(zt), abs(2 * tau - 1)) * np.sign(tau - 0.5)
            qa = np.quantile(zt, tau)
            loss = [float(np.mean(np.maximum(tau * (zx - q), (tau - 1) * (zx - q)))) for q in (qs, qa)]
            res[str(tau)] = loss[1] / loss[0] - 1
        out[str(h)] = {"pinball_rel": res, "below10": float(np.mean(zx < np.quantile(zt, 0.1))),
                       "above90": float(np.mean(zx > np.quantile(zt, 0.9)))}
    return out


def study_vix(P, key: str) -> dict:
    """Daily spread scaled by the VIX relative to its one-year average; exponent chosen on tune."""
    lv = np.log(history.load_vix())
    x_all = lv - lv.rolling(250, min_periods=100).mean()
    t = pd.to_datetime(P.t)
    u = pd.DatetimeIndex(sorted(set(t)))
    x = x_all.reindex(u.union(x_all.index)).ffill().reindex(u).reindex(t).to_numpy()
    x = np.where(np.isfinite(x), x, 0.0)
    out = {}
    for j, h in enumerate(P.h):
        res = {}
        for beta in np.round(np.arange(-0.4, 1.01, 0.1), 2):
            s = P.s[key][:, j] * np.exp(beta * x)
            k = np.quantile(np.abs(P.a[P.tune, j]) / s[P.tune], 0.8) / BAND_Z["80"]
            res[float(beta)] = [float(_gauss_crps(P.a[m, j], s[m] * k).mean()) for m in (P.tune, P.test)]
        best = min(res, key=lambda b: res[b][0])
        out[str(h)] = {"beta": best, "tune_rel": res[best][0] / res[0.0][0] - 1, "test_rel": res[best][1] / res[0.0][1] - 1}
    return out


def study_profile(P) -> dict:
    """Hour-of-day profile: sample length and smoothing over neighbouring hours (live: 1500 bars, 0.25)."""
    extra = []
    for code in PAIRS:
        extra.extend(json.loads((OUT_DIR / f"{code}_1h_profile.json").read_text())["rows"])
    s_all = {k: np.array([e["s"][k] for e in extra]) for k in extra[0]["s"]}
    out = {}
    for j, h in enumerate(P.h):
        res = {}
        for key, s in s_all.items():
            k = np.quantile(np.abs(P.a[P.tune, j]) / s[P.tune, j], 0.8) / BAND_Z["80"]
            res[key] = [float(_gauss_crps(P.a[m, j], s[m, j] * k).mean()) for m in (P.tune, P.test)]
        best = min(res, key=lambda v: res[v][0])
        out[str(h)] = {"best": best, "tune_rel": res[best][0] / res["p1500_s25"][0] - 1,
                       "test_rel": res[best][1] / res["p1500_s25"][1] - 1}
    return out


def round2() -> dict:
    D, H = Panel("1d"), Panel("1h")
    return {
        "k_window": {"1d": study_k_window(D, "fast_rev", [60, 120, 250, 420, 1000, None], 420, lambda h: h * 1.5 + 4),
                     "1h": study_k_window(H, "range", [14, 28, 56, 84, 180, None], 84, lambda h: h + 72)},
        "asym": {"1d": study_asymmetry(D, "fast_rev"), "1h": study_asymmetry(H, "range")},
        "vix": study_vix(D, "fast_rev"),
        "profile": study_profile(H),
    }


def deployed_shape(rv_choice: str) -> dict:
    """Degrees of freedom refit on all data (tune + test) for the live volatility models."""
    P = Panel("1h")
    R = RvPanel()
    return {"1h": {h: x["nu"] for h, x in shape_metrics(P, "range", np.ones(len(P.t), bool)).items()},
            "1d": {h: x["nu"] for h, x in shape_metrics(R, rv_choice, np.ones(len(R.t), bool)).items()}}


# ------------------------------------------------------------- report

def _pct(x, digits=2):
    return "—" if x is None else f"{x * 100:+.{digits}f}%"


def _p(x):
    return "—" if x is None else ("<0.001" if x < 0.001 else f"{x:.3f}")


def _point_table(rows: dict, hs, unit, period="test") -> list[str]:
    out = ["| 方法 | " + " | ".join(f"{h}{unit}先" for h in hs) + " |", "|---|" + "---|" * len(hs)]
    for name, m in rows.items():
        cells = []
        for h in hs:
            x = m[period][str(h)]
            hit = "—" if x["hit"] is None else f"{x['hit'] * 100:.1f}%"
            cells.append(f"{_pct(x['skill'])} / {hit} / p={_p(x['dm_p'])}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


def _nu_text(nu: dict) -> str:
    return "、".join(f"{'1時間足' if tf == '1h' else '日足'} " + " / ".join(f"{h}{'時間' if tf == '1h' else '営業日'}先 ν={v}"
                                                                         for h, v in hs.items())
                    for tf, hs in nu.items())


def _nu_matches(nu: dict) -> bool:
    return all(BAND_NU[tf].get(int(h)) == v for tf, hs in nu.items() for h, v in hs.items())


def _round2_report(r2: dict) -> list[str]:
    def row(name, hs, unit, cells):
        return f"| {name} | " + " | ".join(cells(h) for h in hs) + " |"

    L = ["## 4. 追加で試したこと (採用なし)", "",
         "予測レンジについて、さらに次の5つを同じ手順 (調整期間で選び検証期間で確認) で試しました。"
         "各欄: 調整期間で選んだ設定 / 検証期間の CRPS 差 (負が改善)。改善はどれも 0.2% 程度以下 "
         "(上下で幅の違うレンジはむしろ悪化) で、現在の設定を変えていません。", "",
         "| 試したこと | 時間軸 | 予測先 1 | 予測先 2 | 予測先 3 | 予測先 4 |", "|---|---|---|---|---|---|"]
    kw = r2["k_window"]
    for tf, unit, win in (("1d", "営業日", "日"), ("1h", "時間", "日")):
        hs = list(kw[tf])
        cells = [f"{h}{unit}先: {kw[tf][h]['best'].replace('None', '全期間')}{'' if kw[tf][h]['best'] == 'None' else win} / {_pct(kw[tf][h]['test_rel'])}" for h in hs]
        cells += [""] * (4 - len(cells))
        L.append(f"| レンジ補正に使う実績の期間 (本番: 日足420日・1時間足84日) | {'日足' if tf == '1d' else '1時間足'} | " + " | ".join(cells) + " |")
    for tf, unit in (("1d", "営業日"), ("1h", "時間")):
        a = r2["asym"][tf]
        cells = [f"{h}{unit}先: 5%点 {_pct(a[h]['pinball_rel']['0.05'])}, 95%点 {_pct(a[h]['pinball_rel']['0.95'])}" for h in a]
        cells += [""] * (4 - len(cells))
        L.append(f"| 上下で幅の違うレンジ (分位点損失の差) | {'日足' if tf == '1d' else '1時間足'} | " + " | ".join(cells) + " |")
    v = r2["vix"]
    L.append("| VIX (米国株の予想変動率) で幅を調整 | 日足 | " + " | ".join(
        f"{h}営業日先: 指数 {v[h]['beta']:+.1f} / {_pct(v[h]['test_rel'])}" for h in v) + " |")
    pr = r2["profile"]
    cells = [f"{h}時間先: {pr[h]['best']} / {_pct(pr[h]['test_rel'])}" for h in pr] + [""]
    L.append("| 時間帯ごとの変動の推定 (期間・ならし方) | 1時間足 | " + " | ".join(cells) + " |")
    L += ["", "週末をはさむ予測や曜日ごとの的中率も確認しましたが、調整期間と検証期間で一貫した偏りはありませんでした。"
          "1時間足で目立つ値動き (2024年5月の円買い介入、2025年8月の米雇用統計など) は実際の出来事で、データの誤りではありません。", ""]
    return L


def report(res: dict) -> str:
    d, h, rv = res["1d"], res["1h"], res["rv"]
    hd = h["point"]["時間帯ごとの平均的な値動き"]
    L = ["# 過去データによる検証", "",
         "2002年からの日足 (7ペア) と直近約2年分の1時間足を使い、"
         "**各時点ではその時点までのデータだけで予測する** (ウォークフォワード) 方法で検証しました。", "",
         "- 係数・設定値は古い期間 (**調整期間**) だけで決め、新しい期間 (**検証期間**) で成績を確認しています。"
         "検証期間のデータは、設定を選ぶのに一切使っていません。",
         f"- 日足: 調整期間 {d['tune'][0]}〜{d['tune'][1]}, 検証期間 {d['test'][0]}〜{d['test'][1]} (予測 {d['n']:,}件、5営業日ごと)",
         f"- 1時間足: 調整期間 {h['tune'][0]}〜{h['tune'][1]}, 検証期間 {h['test'][0]}〜{h['test'][1]} (予測 {h['n']:,}件、6時間ごと)",
         "- 誤差改善率 = 「変化なし」(ランダムウォーク) と比べた二乗誤差平方根の改善率。p値は日付ごとに7ペアを平均した損失差の"
         " Diebold-Mariano 検定 (両側)。p が小さくても改善率が負なら「有意に悪い」という意味です。",
         "- 金利差には FRED の短期金利を使い、公表の遅れを考慮して当時知り得た値だけを使っています。", "",
         "再実行: `pip install -e . && aifx research --download && aifx research`", "",
         "## 1. 方向 (中心値) の予測", "",
         "各欄: 誤差改善率 / 方向的中率 / p値 (検証期間)", "", "### 日足", ""]
    L += _point_table(d["point"], DAILY_H, "営業日")
    L += ["", "### 1時間足", ""]
    L += _point_table(h["point"], HOURLY_H, "時間")
    L += ["", "**どの方法も、検証期間で「変化なし」を有意に上回れませんでした。** "
          "6つのモデルをそのまま平均すると、むしろ有意に悪化します。", "",
          "過去データに合わせすぎると成績がよく見える例: 「時間帯ごとの平均的な値動き」は、"
          f"調整期間では1時間先の的中率 {hd['tune']['1']['hit'] * 100:.1f}%・誤差改善 {_pct(hd['tune']['1']['skill'])} でしたが、"
          f"検証期間では的中率 {hd['test']['1']['hit'] * 100:.1f}%・誤差改善 {_pct(hd['test']['1']['skill'])} に落ちました "
          f"(24時間先は {_pct(hd['test']['24']['skill'])} で有意に悪化)。調整に使ったデータで成績を測ると、このように実力以上に見えます。", "",
          "### 学習ルールの再現", "",
          "本番の学習ルール (直近の成績からモデルの重みと、方向予測にどこまで従うか (ゲイン) を決める) を、"
          "過去の各時点で再現した成績です。λ はゲインを0 (=変化なし) に引き寄せる強さです。", "",
          "#### 日足 — 調整期間", ""]
    L += _point_table(d["sim"], DAILY_H, "営業日", "tune")
    L += ["", "#### 日足 — 検証期間", ""] + _point_table(d["sim"], DAILY_H, "営業日")
    L += ["", "#### 1時間足 — 調整期間", ""] + _point_table(h["sim"], HOURLY_H, "時間", "tune")
    L += ["", "#### 1時間足 — 検証期間", ""] + _point_table(h["sim"], HOURLY_H, "時間")
    L += ["", f"旧設定 (λ={OLD_GAIN_LAMBDA:g}) は日足の10〜20営業日先で「変化なし」よりわずかに (有意に) 悪く、縮小なしではさらに悪化しました。"
          f"調整期間で旧設定を上回った λ={GAIN_LAMBDA:g} を採用し、検証期間でも悪化幅が縮むことを確認しました。", "",
          "## 2. 予測レンジ", "", "### 値動きの大きさの推定", "",
          "各欄: 調整期間の CRPS 差 / 検証期間の CRPS 差 (p値)。CRPS は予測分布の誤差 (小さいほど良い)、差は従来方式との比較 (負が改善)。", ""]
    for name, r, hs, unit in (("日足 (2002〜2026)", d, DAILY_H, "営業日"), ("1時間足", h, HOURLY_H, "時間")):
        L += [f"#### {name}", "", "| 推定方法 | " + " | ".join(f"{x}{unit}先" for x in hs) + " |", "|---|" + "---|" * len(hs)]
        for key, m in r["vol"].items():
            if key == "live":
                continue
            L.append(f"| {VOL_NAMES.get(key, key)} | " + " | ".join(
                f"{_pct(m[str(x)]['tune_rel'])} / {_pct(m[str(x)]['test_rel'])} (p={_p(m[str(x)]['p'])})" for x in hs) + " |")
        L.append("")
    L += [f"#### 日足: 1時間足から測る日々の変動 (実現ボラティリティ)", "",
          f"1時間足がある期間だけの検証です (調整期間 {rv['tune'][0]}〜{rv['tune'][1]}, 検証期間 {rv['test'][0]}〜{rv['test'][1]}, 毎営業日, {rv['n']:,}件)。"
          "調整期間の成績順に上位6件。rv = 1時間足の変化の2乗和、blend = それと日足の変化の2乗の平均。"
          "数字は EWMA の減衰率と平常水準へ戻る速さ。", "",
          "| 推定方法 | " + " | ".join(f"{x}営業日先" for x in DAILY_H) + " |", "|---|" + "---|" * 4]
    for key in rv["ranked"][:6]:
        m = rv["vol"][key]
        L.append(f"| {key} | " + " | ".join(
            f"{_pct(m[str(x)]['tune_rel'])} / {_pct(m[str(x)]['test_rel'])} (p={_p(m[str(x)]['p'])})" for x in DAILY_H) + " |")
    L += ["", "### レンジの形", "",
          "正規分布のレンジは、実際の値動きに比べて中心付近が広すぎ、外側が狭すぎました (値動きの分布は裾が太い)。"
          "裾の太い t 分布の形にし、80%レンジの幅は従来どおり実績で補正します。t 分布の自由度は調整期間で選びました。", "",
          "| 時間軸 | 予測先 | 50%レンジの的中 (前→後) | 80% (前→後) | 95% (前→後) |", "|---|---|---|---|---|"]
    for name, r, hs, unit in (("1時間足", h, HOURLY_H, "時間"), ("日足 (2017〜)", d, DAILY_H, "営業日"),
                              ("日足 (実現ボラ, 直近1年)", rv, DAILY_H, "営業日")):
        for x in hs:
            b, a = r["shape"]["before"][str(x)]["normal"]["cover"], r["shape"]["after"][str(x)]["t"]["cover"]
            L.append(f"| {name} | {x}{unit}先 | {b['50'] * 100:.1f}% → {a['50'] * 100:.1f}% | "
                     f"{b['80'] * 100:.1f}% → {a['80'] * 100:.1f}% | {b['95'] * 100:.1f}% → {a['95'] * 100:.1f}% |")
    L += ["", "前 = 従来の変動推定 + 正規分布、後 = 新しい変動推定 + t 分布 (検証期間)。"
          "80%レンジの幅は調整期間の実績で合わせています (本番では実績から継続的に補正)。"
          "直近1年は値動きが調整期間より穏やかだったため、どちらの方法でも80%レンジが広めになっています。", "",
          "## 3. 本番に採用した変更", "",
          f"1. **ゲインの縮小を強化** (λ {OLD_GAIN_LAMBDA:g} → {GAIN_LAMBDA:g}): 有効性が示されていない方向予測で「変化なし」より悪くなるのを防ぐ。",
          "2. **1時間足の変動を高値・安値の幅から推定**: 1・4時間先で CRPS が有意に改善、24時間先は同等。",
          f"3. **日足の変動を1時間足から測る** (実現ボラティリティ, 調整期間で1位の設定 `{rv['chosen']}`): 1・5営業日先で有意に改善、"
          "10・20営業日先は有意差なし。1時間足が足りないときは日足の変化を使い、平常水準へ戻る速さは2002〜2016年で選んだ 0.90。",
          "4. **レンジを t 分布の形に**: 50%/95%レンジの的中率が名目値に近づく。本番の自由度は全期間で推定し直した値: "
          + _nu_text(res["deployed_nu"]) + (" (本番の設定と一致)" if _nu_matches(res["deployed_nu"]) else
                                            " (**本番の設定 `BAND_NU` と異なります。更新してください**)") + "。", "",
          "採用しなかったもの: 金利差 (キャリー)・モメンタム・時間帯の癖・直前の値動き・モデルの組合せ。"
          "調整期間で推定しても検証期間で改善せず、多くは悪化しました。", ""]
    L += _round2_report(res["round2"]) + [
          "## 5. 精度についての結論", "",
          "- 1時間〜20営業日先の**方向**は、20年以上の過去データで試したどの方法でも「変化なし」を安定して上回れませんでした。"
          "このため本番の予測は、実績で有効性が示されるまで中心値を「変化なし」付近に保ちます。成績をよく見せるためではなく、"
          "過去データの検証で最も誤差が小さかったためです。",
          "- **値動きの幅** (予測レンジ) は改善できました。レンジの的中率は Web ページで実績として公開され、"
          "この検証とは別に、これからの予測で確かめられます。", ""]
    return "\n".join(L)


def run(workers: int = 4, log=print) -> dict:
    compute(workers, log=log)
    res = {"1d": evaluate("1d"), "1h": evaluate("1h"), "rv": evaluate_rv()}
    res["deployed_nu"] = deployed_shape(res["rv"]["chosen"])
    res["round2"] = round2()
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (REPORT_DIR / "report.md").write_text(report(res), encoding="utf-8")
    log(f"wrote {REPORT_DIR}/report.md")
    return res
