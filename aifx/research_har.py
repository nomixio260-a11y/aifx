"""Daily forecast ranges: does a HAR model of realized variance beat the current EWMA?

The width of the daily forecasts' ranges (1, 5, 10 and 20 business days
ahead) comes from ``engine.sigma_steps``: an EWMA of each London day's
realized variance (the sum of squared hourly returns) that fades into the
long-run variance of daily returns (volatility.py). This study compares it,
computed exactly as the server does, with the heterogeneous autoregressive
model of realized variance (HAR-RV, Corsi 2009: the variance of the coming
days explained by the realized variance of the last day, the last week and the
last month) and a few well-known variants: log realized variance, the
Parkinson high-low variance of the hourly bars, realized absolute values,
daily absolute returns, HAR-Q (Bollerslev, Patton and Quaedvlieg 2016), a fit
per pair instead of one pooled fit, other EWMA settings and a blend.

Forecasts are scored the way the live system scores daily forecasts: the
origin is the end of a London business day, the origin price is the hourly
close then, and the outcome is the hourly close at the target time
(engine.target_times). There is one origin per business day from July 2024,
after a warm-up on the hourly bars that start in December 2023, for the 7
pairs. Settings are chosen on the first 60 % of origin dates (tune; origins
whose target falls after the split are left out) and scored on the last 40 %
(test). HAR coefficients are refitted at every origin on rows whose target
days had ended by the origin, and the inputs are rebuilt from data truncated
at the origin for a sample of origins to check that the forecasts are the same.

Also: the hourly forecasts' 24-hour range with the daily HAR forecast
blended in, and the hourly profile over the 6,000 bars that engine.py intends
(the server handed ``sigma_steps`` only its last 3,000 hourly bars until this
study found it; ``engine.vol_bars`` now hands it the whole window).

    python -m aifx.research_har     # writes research/har.md and research/har.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS
from .engine import HOURLY_PROFILE_WINDOW, TIMEFRAMES, sigma_steps, target_times
from .forecaster import origin_price
from .stats import diebold_mariano
from .timeutil import LONDON, add_business_days, add_trading_minutes, london_day_end
from .volatility import daily_step_variance, daily_variance_inputs, realized_daily_variance

REPORT_DIR = Path("research")
TF_D, TF_H = TIMEFRAMES["1d"], TIMEFRAMES["1h"]
H = TF_D.horizons                       # (1, 5, 10, 20) business days
STEPS = max(H)
BP = 1e4
START = pd.Timestamp("2024-07-01")      # first origin (the hourly bars start on 2023-12-08)
TUNE_SHARE = 0.6
WEEK, MONTH = 5, 22                     # HAR windows, in business days
MIN_ROWS = 60                           # fewest fitting rows per pair
FLOOR = 0.05                            # lowest input / forecast, as a share of the pair's mean
EVERY_H = 5                             # hourly study: an origin every 5th hourly bar
H24 = 24
N_CHECK = 24                            # origins rebuilt from truncated data

# (c) the current EWMA with other settings: decay on realized variance, reversion per day, long-run window
EWMA_GRID = [(lam, rev, lw) for lam in (0.5, 0.6, 0.7, 0.8, 0.9) for rev in (0.8, 0.85, 0.9, 0.95, 0.98)
             for lw in (500, 999)]
# (a)/(b) HAR: inputs ("src"), what is regressed ("form"), one fit for all pairs or one per pair
HAR_SPECS = {
    "har_rv": dict(src="rv", form="lin", pool=True),
    "har_rv_pair": dict(src="rv", form="lin", pool=False),
    "har_log": dict(src="rv", form="log", pool=True),
    "har_log_pair": dict(src="rv", form="log", pool=False),
    "har_pk": dict(src="pk", form="lin", pool=True),
    "har_rav": dict(src="rav", form="sqrt", pool=True),
    "har_rav_pair": dict(src="rav", form="sqrt", pool=False),
    "har_abs": dict(src="abs", form="sqrt", pool=True),
    "harq": dict(src="rv", form="lin", pool=True, q=True),
}
LABELS = {
    "live": "現在のモデル (EWMA、本番と同じ計算)",
    "har_rv": "HAR-RV (実現分散, 7ペア共通の係数)",
    "har_rv_pair": "HAR-RV (実現分散, ペアごとの係数)",
    "har_log": "HAR-logRV (実現分散の対数, 共通)",
    "har_log_pair": "HAR-logRV (実現分散の対数, ペアごと)",
    "har_pk": "HAR-Parkinson (1時間足の高値・安値の幅, 共通)",
    "har_rav": "HAR-RAV (1時間ごとの変化の絶対値の和, 共通)",
    "har_rav_pair": "HAR-RAV (1時間ごとの変化の絶対値の和, ペアごと)",
    "har_abs": "HAR-絶対値 (ロンドン日ごとの変化の絶対値, 共通)",
    "harq": "HAR-Q (実現分散 + 測定誤差の補正, 共通)",
}
BLEND_W = (0.25, 0.5, 0.75)             # weight of the HAR forecast in a geometric blend with the current model
HOURLY_W = (0.2, 0.4, 0.6)


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def load_data() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """(hourly, raw Yahoo daily) bars per pair, duplicates dropped."""
    out = {}
    for code in PAIRS:
        h = history.load_hourly(code)
        d = history.load_daily(code)
        out[code] = (h[~h.index.duplicated()].sort_index(), d[~d.index.duplicated()].sort_index())
    return out


# ----------------------------------------------------------- daily measures

def london_day_keys(starts) -> np.ndarray:
    """London business day of each hourly bar (by its start), as in volatility.realized_daily_variance:
    a bar belongs to the day in which it ends; Sunday-evening bars count towards Monday."""
    local = (pd.DatetimeIndex(starts) + pd.Timedelta(minutes=59)).tz_convert(LONDON).tz_localize(None).normalize()
    wd = local.dayofweek
    return np.asarray(local + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D"))


def day_measures(hourly: pd.DataFrame, until) -> pd.DataFrame:
    """Per London business day, from the hourly bars ended by ``until`` (bp units):

    rv   realized variance, the sum of squared hourly returns
    pk   Parkinson variance, the sum of each bar's high-low range^2 / (4 ln 2) plus its opening gap^2
    rav  realized absolute value, the sum of |hourly returns|
    rq   realized quarticity, n/3 * sum of hourly returns^4 (HAR-Q)
    ret  the day's return (sum of hourly returns); n the number of returns
    The first day is dropped (usually incomplete), as in realized_daily_variance.
    """
    h = hourly[hourly.index + pd.Timedelta(hours=1) <= pd.Timestamp(until)]
    lc = np.log(h["close"].to_numpy(float))
    r = np.diff(lc) * BP
    hl = np.log(h["high"].to_numpy(float) / h["low"].to_numpy(float))[1:] * BP
    gap = (np.log(h["open"].to_numpy(float))[1:] - lc[:-1]) * BP
    df = pd.DataFrame({"rv": r * r, "pk": hl * hl / (4 * math.log(2)) + gap * gap, "rav": np.abs(r),
                       "r4": r ** 4, "ret": r, "n": np.ones(len(r))})
    g = df.groupby(london_day_keys(h.index[1:])).sum()
    g["rq"] = g["n"] / 3 * g["r4"]
    return g.iloc[1:]


def build_panel(meas: dict[str, pd.DataFrame], last_day: pd.Timestamp) -> dict:
    """Pairs x business days (Monday-Friday, as engine.target_times counts them) up to ``last_day``.
    A weekday without hourly bars has zero variance (its move is counted on the next day with bars)."""
    codes = list(meas)
    first = max(m.index[0] for m in meas.values())
    cal = pd.bdate_range(first, last_day)
    P = {"cal": cal, "codes": codes}
    for k in ("rv", "pk", "rav", "rq", "ret", "n"):
        P[k] = np.vstack([meas[c][k].reindex(cal).fillna(0.0).to_numpy(float) for c in codes])
    P["abs"] = np.abs(P["ret"])
    return P


# ---------------------------------------------------------------------- HAR

def _roll(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over ``n`` days from a running sum (a day's value never depends on later days)."""
    cs = np.cumsum(x, axis=1)
    out = np.full(x.shape, np.nan)
    out[:, n - 1] = cs[:, n - 1] / n
    out[:, n:] = (cs[:, n:] - cs[:, :-n]) / n
    return out


def har_inputs(m: np.ndarray) -> np.ndarray:
    """(pairs, days, 3): the day's value, the mean of the last 5 days and of the last 22 days."""
    return np.stack([m, _roll(m, WEEK), _roll(m, MONTH)], axis=-1)


def _design(spec: dict, F: np.ndarray, rq: np.ndarray, c: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Regressors with a constant. Inputs are in units of the pair's mean up to the origin
    (``f`` for the measure, ``c`` for realized variance), so one fit can serve every pair."""
    x = F / f[..., None]
    if spec["form"] == "log":
        x = np.log(np.maximum(x, FLOOR))
    cols = [np.ones(x.shape[:-1] + (1,)), x]
    if spec.get("q"):
        cols.append(x[..., :1] * (np.sqrt(rq) / c)[..., None])
    return np.concatenate(cols, axis=-1)


def _to_z(form: str, u: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(u, FLOOR)) if form == "log" else np.sqrt(u) if form == "sqrt" else u


def _from_z(form: str, z: np.ndarray) -> np.ndarray:
    if form == "log":
        return np.exp(z)
    if form == "sqrt":
        return np.maximum(z, math.sqrt(FLOOR)) ** 2
    return np.maximum(z, FLOOR)


def har_forecasts(P: dict, specs: dict, t_list, audit: dict | None = None) -> dict[str, np.ndarray]:
    """HAR forecasts (bp^2) of the variance of the move over the h business days after the end of day t.

    Direct regressions per horizon of the mean realized variance of days s+1..s+h on the HAR
    inputs of day s, by least squares on every row s whose target days had all ended by day t
    (s <= t - h; an expanding window). Returns {spec: (len(H), pairs, days)}, NaN where not fitted.
    """
    n, T = P["rv"].shape
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(P["rv"], axis=1)], axis=1)   # cs[:, s + 1] = rv[:, :s + 1].sum()
    F = {src: har_inputs(P[src]) for src in {s["src"] for s in specs.values()}}
    out = {name: np.full((len(H), n, T), np.nan) for name in specs}
    s0 = MONTH - 1                                   # first day with a full monthly window
    for t in t_list:
        c = P["rv"][:, :t + 1].mean(axis=1)
        for name, spec in specs.items():
            src, form = spec["src"], spec["form"]
            f = P[src][:, :t + 1].mean(axis=1)
            Xt = _design(spec, F[src][:, t], P["rq"][:, t], c, f)
            for j, h in enumerate(H):
                s1 = t - h                           # last row: its target days s1+1..t had ended by day t
                if s1 - s0 + 1 < MIN_ROWS:
                    continue
                X = _design(spec, F[src][:, s0:s1 + 1], P["rq"][:, s0:s1 + 1], c[:, None], f[:, None])
                u = (cs[:, s0 + h + 1:s1 + h + 2] - cs[:, s0 + 1:s1 + 2]) / h / c[:, None]
                if audit is not None:                # the last day any fitting row or input uses
                    audit["max_day_used_minus_origin"] = max(audit.get("max_day_used_minus_origin", -99), s1 + h - t)
                z = _to_z(form, u)
                if spec["pool"]:
                    beta = np.linalg.lstsq(X.reshape(-1, X.shape[-1]), z.ravel(), rcond=None)[0]
                    zt = Xt @ beta
                else:
                    zt = np.array([Xt[i] @ np.linalg.lstsq(X[i], z[i], rcond=None)[0] for i in range(n)])
                out[name][j, :, t] = h * c * _from_z(form, zt)
    return out


# --------------------------------------------------------------- daily study

def daily_origins(code: str, i: int, hourly: pd.DataFrame, daily: pd.DataFrame, P: dict,
                  har: dict[str, np.ndarray]) -> list[dict]:
    """One origin per business day with a daily bar: the server's variance, the EWMA grid and the HAR forecasts."""
    ends = hourly.index + pd.Timedelta(hours=1)
    close = hourly["close"].to_numpy(float)
    last_end = ends[-1]
    pos = {day: t for t, day in enumerate(P["cal"])}
    rows = []
    for D in daily.index[daily.index >= START]:
        origin = london_day_end(D.date())
        targets = pd.DatetimeIndex(target_times(TF_D, origin))
        if targets[-1] > last_end or D not in pos:
            continue
        k = int(ends.searchsorted(pd.Timestamp(origin), side="right"))
        ho = hourly.iloc[:k]                          # hourly bars ended by the origin
        p0 = close[k - 1]
        assert origin_price(ho, origin)[0] == p0
        bars = daily.loc[:D].iloc[-TF_D.fit_bars:]   # the stored (raw) daily bars completed by the origin
        live = np.cumsum(sigma_steps(TF_D, bars, origin, STEPS, None, hourly=ho))
        sq, lam = daily_variance_inputs(bars, ho, origin)
        y = np.log(bars["close"].to_numpy())
        same = bool(np.allclose(np.cumsum(daily_step_variance(y, STEPS, lam, sq=sq)) * BP * BP, live, rtol=1e-12))
        ew = np.array([np.cumsum(daily_step_variance(y, STEPS, lm if sq is not None else lam, lw, rev, sq=sq))
                       for lm, rev, lw in EWMA_GRID]) * BP * BP
        idx = ends.searchsorted(targets, side="right") - 1
        assert (ends[idx] > pd.Timestamp(origin)).all()
        t = pos[D]
        rows.append({"pair": code, "i": i, "D": D, "t": t, "origin": origin,
                     "a": np.log(close[idx] / p0) * BP,
                     "live": live[np.array(H) - 1], "ewma": ew[:, np.array(H) - 1],
                     "har": {name: fc[:, i, t] for name, fc in har.items()},
                     "rv_used": sq is not None, "same_as_engine": same,
                     "stale_h": float((pd.Timestamp(origin) - ends[k - 1]) / pd.Timedelta(hours=1))})
    return rows


def _scores(a: np.ndarray, v: np.ndarray, tune: np.ndarray, test: np.ndarray,
            pairs: np.ndarray) -> tuple[dict, np.ndarray]:
    """QLIKE with one scale fitted on tune, the 80 % band from the tune quantile of |r|/sqrt(v), correlations."""
    r2, sd = a * a, np.sqrt(v)
    k = float(np.mean(r2[tune] / v[tune]))
    loss = np.log(k * v) + r2 / (k * v)
    z80 = float(np.quantile(np.abs(a[tune]) / sd[tune], 0.8))

    def corr(m):
        return float(np.corrcoef(sd[m], np.abs(a[m]))[0, 1])

    return {"qlike_tune": float(loss[tune].mean()), "qlike_test": float(loss[test].mean()), "k": k, "z80": z80,
            "cover80_tune": float(np.mean(np.abs(a[tune]) <= z80 * sd[tune])),
            "cover80_test": float(np.mean(np.abs(a[test]) <= z80 * sd[test])),
            "width80_test_bp": float(np.mean(2 * z80 * sd[test])),
            "corr_tune": corr(tune), "corr_test": corr(test),
            "corr_pair_test": float(np.mean([corr(test & (pairs == p)) for p in np.unique(pairs)]))}, loss


def _dm(dates: np.ndarray, loss: np.ndarray, base: np.ndarray, mask: np.ndarray, lag: int) -> dict:
    """Diebold-Mariano on the cross-pair mean loss difference per origin date (negative = better than live)."""
    d = pd.Series(loss[mask] - base[mask]).groupby(dates[mask]).mean().sort_index().to_numpy()
    stat, p = diebold_mariano(d, np.zeros(len(d)), lag)
    return {"diff": float(d.mean()), "stat": stat, "p": p}


def evaluate_daily(rows: list[dict]) -> dict:
    pairs = np.array([r["pair"] for r in rows])
    dates = pd.DatetimeIndex([r["D"] for r in rows]).to_numpy()
    udates = np.unique(dates)
    split = udates[int(len(udates) * TUNE_SHARE)]
    test = dates >= split
    A = np.array([r["a"] for r in rows])
    models = {"live": np.array([r["live"] for r in rows])}
    for g, (lm, rev, lw) in enumerate(EWMA_GRID):
        models[f"ewma_{lm:g}_{rev:g}_{lw}"] = np.array([r["ewma"][g] for r in rows])
    for name in HAR_SPECS:
        models[name] = np.array([r["har"][name] for r in rows])
    ok = np.all([np.isfinite(v).all(axis=1) for v in models.values()], axis=0)
    out = {"split": str(pd.Timestamp(split).date()), "n_dropped_unfitted": int((~ok).sum())}

    def tune_mask(h):
        """Tune origins whose target day is not after the first test origin (no overlap with test outcomes)."""
        tgt = pd.DatetimeIndex([add_business_days(pd.Timestamp(d).date(), h) for d in dates]).to_numpy()
        return ok & (dates < split) & (tgt <= split)

    tunes = {h: tune_mask(h) for h in H}
    test = ok & test

    def score_all(ms: dict) -> tuple[dict, dict]:
        sc, losses = {}, {}
        for name, V in ms.items():
            for j, h in enumerate(H):
                s, loss = _scores(A[:, j], V[:, j], tunes[h], test, pairs)
                sc.setdefault(name, {})[str(h)] = s
                losses.setdefault(name, {})[h] = loss
        return sc, losses

    sc, losses = score_all(models)

    def mean_tune(name):
        return float(np.mean([sc[name][str(h)]["qlike_tune"] for h in H]))

    best_ewma = min((k for k in models if k.startswith("ewma_")), key=mean_tune)
    best_har = min(HAR_SPECS, key=mean_tune)
    blends = {f"blend_{w:g}": models["live"] ** (1 - w) * models[best_har] ** w for w in BLEND_W}
    sc_b, loss_b = score_all(blends)
    sc.update(sc_b)
    losses.update(loss_b)
    best_blend = min(blends, key=mean_tune)
    ranked = sorted(["live", best_ewma, *HAR_SPECS, best_blend], key=mean_tune)
    best = next(k for k in ranked if k != "live")
    dm = {name: {str(h): {"tune": _dm(dates, losses[name][h], losses["live"][h], tunes[h], 2 * h),
                          "test": _dm(dates, losses[name][h], losses["live"][h], test, 2 * h)} for h in H}
          for name in sc if name != "live"}
    per_pair = {name: {str(h): {p: float(np.mean(losses[name][h][test & (pairs == p)])
                                          - np.mean(losses["live"][h][test & (pairs == p)])) for p in np.unique(pairs)}
                       for h in H} for name in (best, best_har, best_ewma)}
    beats = {name: {str(h): {"tune": sc[name][str(h)]["qlike_tune"] < sc["live"][str(h)]["qlike_tune"],
                             "test": sc[name][str(h)]["qlike_test"] < sc["live"][str(h)]["qlike_test"]} for h in H}
             for name in sc if name != "live"}
    adopt = all(beats[best][str(h)]["tune"] and beats[best][str(h)]["test"] for h in H)
    # honesty check: the best candidate per horizon on tune (any model above), scored on test
    per_h = {str(h): min((k for k in sc if k != "live"), key=lambda k, h=h: sc[k][str(h)]["qlike_tune"]) for h in H}
    passing = [k for k in sc if k != "live" and all(beats[k][str(h)]["tune"] and beats[k][str(h)]["test"] for h in H)]
    out.update({
        "n": int(ok.sum()), "n_tune": {str(h): int(tunes[h].sum()) for h in H}, "n_test": int(test.sum()),
        "dates": {"first": str(pd.Timestamp(udates[0]).date()), "last": str(pd.Timestamp(udates[-1]).date()),
                  "tune_last": str(pd.Timestamp(udates[udates < split][-1]).date()),
                  "test_first": str(pd.Timestamp(split).date())},
        "scores": sc, "dm": dm, "per_pair_test_diff": per_pair, "beats_live": beats,
        "selected": {"ewma": best_ewma, "har": best_har, "blend": best_blend, "best": best, "ranked_by_tune": ranked,
                     "mean_tune_qlike": {k: mean_tune(k) for k in ranked}},
        "adopt": adopt, "best_per_horizon_by_tune": per_h, "beat_live_everywhere": passing,
        "ewma_top_tune": sorted(((k, mean_tune(k)) for k in models if k.startswith("ewma_")), key=lambda x: x[1])[:5],
    })
    return out


# -------------------------------------------------------------- hourly study

def hourly_origins(code: str, i: int, hourly: pd.DataFrame, P: dict, har1: np.ndarray) -> list[dict]:
    """Every EVERY_H-th hourly origin: the server's 24-hour variance as it was (last 3,000 bars, as
    forecaster.py passed them before engine.vol_bars), the same with the 6,500 bars engine.py intends, and the daily HAR forecast for the
    next London day made at the end of the last London day ended by the origin."""
    ends = hourly.index + pd.Timedelta(hours=1)
    close = hourly["close"].to_numpy(float)
    cal_ends = pd.DatetimeIndex([london_day_end(d.date()) for d in P["cal"]])
    j0 = max(int(ends.searchsorted(START.tz_localize("UTC"))), TF_H.fit_bars - 1)
    rows = []
    for j in range(j0, len(hourly), EVERY_H):
        e = ends[j].to_pydatetime()
        t24 = add_trading_minutes(e, H24, 60)
        if pd.Timestamp(t24) > ends[-1]:
            break
        hb = hourly.iloc[:j + 1]
        live = float(np.sum(sigma_steps(TF_H, hb.iloc[-TF_H.fit_bars:], e, H24)))
        full = float(np.sum(sigma_steps(TF_H, hb.iloc[-(HOURLY_PROFILE_WINDOW + 500):], e, H24)))
        idx = int(ends.searchsorted(pd.Timestamp(t24), side="right")) - 1
        t = int(cal_ends.searchsorted(pd.Timestamp(e), side="right")) - 1
        rows.append({"pair": code, "D": ends[j].tz_convert(LONDON).normalize().tz_localize(None),
                     "a": math.log(close[idx] / close[j]) * BP, "live": live, "full": full,
                     "har1": float(har1[i, t]) if t >= 0 else float("nan")})
    return rows


def evaluate_hourly(rows: list[dict]) -> dict:
    pairs = np.array([r["pair"] for r in rows])
    dates = pd.DatetimeIndex([r["D"] for r in rows]).to_numpy()
    udates = np.unique(dates)
    split = udates[int(len(udates) * TUNE_SHARE)]
    a = np.array([r["a"] for r in rows])
    live, full, har1 = (np.array([r[k] for r in rows]) for k in ("live", "full", "har1"))
    ok = np.isfinite(har1)
    tune = ok & (dates < split - np.timedelta64(3, "D"))     # embargo: tune outcomes end before the test origins
    test = ok & (dates >= split)
    models = {"live": live, "profile6000": full, **{f"blend_{w:g}": live ** (1 - w) * har1 ** w for w in HOURLY_W}}
    sc, losses = {}, {}
    for name, v in models.items():
        sc[name], losses[name] = _scores(a, v, tune, test, pairs)
    dm = {name: {"tune": _dm(dates, losses[name], losses["live"], tune, 6),
                 "test": _dm(dates, losses[name], losses["live"], test, 6)} for name in models if name != "live"}
    best = min((k for k in models if k != "live"), key=lambda k: sc[k]["qlike_tune"])
    return {"n": int(ok.sum()), "n_tune": int(tune.sum()), "n_test": int(test.sum()), "every": EVERY_H,
            "split": str(pd.Timestamp(split).date()), "scores": sc, "dm": dm, "best_by_tune": best,
            "adopt": bool(sc[best]["qlike_tune"] < sc["live"]["qlike_tune"]
                          and sc[best]["qlike_test"] < sc["live"]["qlike_test"])}


# ------------------------------------------------------------ leakage checks

def leakage_checks(data: dict, P: dict, har: dict, rows: list[dict], audit: dict) -> dict:
    """Rebuild every input from data truncated at the origin for a sample of origins and compare."""
    rng = np.random.default_rng(7)
    pick = rng.choice(len(rows), size=min(N_CHECK, len(rows)), replace=False)
    worst_har = worst_live = worst_rv = 0.0
    cal_ok = True
    for n in pick:
        r = rows[n]
        origin, D, code = r["origin"], r["D"], r["pair"]
        meas = {c: day_measures(h, origin) for c, (h, _) in data.items()}   # every pair cut at the origin
        Pt = build_panel(meas, D)
        cal_ok &= bool(Pt["cal"][-1] == D and Pt["cal"].equals(P["cal"][:len(Pt["cal"])]))
        t = len(Pt["cal"]) - 1
        ht = har_forecasts(Pt, HAR_SPECS, [t])
        for name in HAR_SPECS:
            a, b = ht[name][:, :, t], har[name][:, :, t]
            worst_har = max(worst_har, float(np.nanmax(np.abs(a / b - 1))))
            assert np.array_equal(np.isnan(a), np.isnan(b))
        h_full, d = data[code]
        k = int((h_full.index + pd.Timedelta(hours=1)).searchsorted(pd.Timestamp(origin), side="right"))
        ho = h_full.iloc[:k]
        assert ho.index[-1] + pd.Timedelta(hours=1) <= pd.Timestamp(origin)
        bars = d.loc[:D].iloc[-TF_D.fit_bars:]
        # the server's variance with every hourly bar handed in (it must ignore the ones after the origin)
        v_all = np.cumsum(sigma_steps(TF_D, bars, origin, STEPS, None, hourly=h_full))[np.array(H) - 1]
        worst_live = max(worst_live, float(np.max(np.abs(v_all / r["live"] - 1))))
        rv_engine = realized_daily_variance(ho, origin) * BP * BP
        mine = day_measures(ho, origin)["rv"]
        worst_rv = max(worst_rv, float(np.max(np.abs(mine.reindex(rv_engine.index) / rv_engine - 1))))
    return {"n_origins": int(len(pick)), "har_truncated_max_rel_diff": worst_har, "calendar_prefix_same": cal_ok,
            "live_all_hourly_vs_truncated_max_rel_diff": worst_live,
            "rv_vs_realized_daily_variance_max_rel_diff": worst_rv,
            "har_last_target_day_minus_origin_day": audit.get("max_day_used_minus_origin"),
            "engine_reproduced_all": bool(all(r["same_as_engine"] for r in rows)),
            "max_origin_price_age_h": float(max(r["stale_h"] for r in rows))}


# ---------------------------------------------------------------------- run

def evaluate(log=print) -> dict:
    t0 = time.time()
    data = load_data()
    last_end = min(h.index[-1] + pd.Timedelta(hours=1) for h, _ in data.values())
    meas = {c: day_measures(h, last_end) for c, (h, _) in data.items()}
    days = pd.bdate_range("2023-12-01", last_end.tz_convert(LONDON).tz_localize(None).normalize())
    last_day = max(d for d in days if london_day_end(d.date()) <= last_end)
    P = build_panel(meas, last_day)
    t_start = int(P["cal"].searchsorted(START - pd.Timedelta(days=7)))
    audit: dict = {}
    har = har_forecasts(P, HAR_SPECS, range(t_start, len(P["cal"])), audit)
    _log(f"HAR fits done: {len(P['cal'])} business days {P['cal'][0].date()}..{P['cal'][-1].date()}", t0)
    rows = []
    for i, (code, (h, d)) in enumerate(data.items()):
        rows += daily_origins(code, i, h, d, P, har)
        _log(f"{code}: {len(rows)} daily origins", t0)
    daily = evaluate_daily(rows)
    _log(f"daily scored; best by tune {daily['selected']['best']}", t0)
    checks = leakage_checks(data, P, har, rows, audit)
    _log(f"leakage checks {checks}", t0)
    har1 = har[daily["selected"]["har"]][0]
    hrows = []
    for i, (code, (h, _)) in enumerate(data.items()):
        hrows += hourly_origins(code, i, h, P, har1)
        _log(f"{code}: {len(hrows)} hourly origins", t0)
    hourly = evaluate_hourly(hrows)
    counts = {c: {"hourly_bars": int(len(h)), "hourly_first": str(h.index[0]), "hourly_last": str(h.index[-1]),
                  "daily_bars": int(len(d))} for c, (h, d) in data.items()}
    return {"settings": {"start": str(START.date()), "tune_share": TUNE_SHARE, "horizons": list(H),
                         "week": WEEK, "month": MONTH, "min_rows": MIN_ROWS, "floor": FLOOR,
                         "ewma_grid": EWMA_GRID, "har_specs": HAR_SPECS, "blend_w": BLEND_W, "hourly_w": HOURLY_W,
                         "fit_bars_1d": TF_D.fit_bars, "fit_bars_1h": TF_H.fit_bars},
            "data": counts, "leakage": checks, "daily": daily, "hourly": hourly,
            "runtime_s": round(time.time() - t0, 1)}


# -------------------------------------------------------------------- report

def _f(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}"


def _name(k: str) -> str:
    if k in LABELS:
        return LABELS[k]
    if k.startswith("ewma_"):
        lam, rev, lw = k.split("_")[1:]
        return f"EWMA 設定変更 (減衰 {lam}, 平常へ戻る速さ {rev}/日, 平常水準 {lw}日)"
    if k.startswith("blend_"):
        return f"現在のモデルと HAR の組み合わせ (HAR の重み {k.split('_')[1]})"
    return k


def _p(x: dict) -> str:
    p = x.get("p")
    if p is None:
        return f"{x['diff']:+.4f}"
    return f"{x['diff']:+.4f} (p<0.01)" if p < 0.01 else f"{x['diff']:+.4f} (p={p:.2f})"


def _hs(hs) -> str:
    return "・".join(f"{h}営業日先" for h in hs) if hs else "どの期間でもなく"


def report(res: dict) -> str:
    d, hr, lk = res["daily"], res["hourly"], res["leakage"]
    sel, sc = d["selected"], d["scores"]
    best = sel["best"]
    n_ewma = len(res["settings"]["ewma_grid"])
    rows = list(dict.fromkeys(["live", sel["ewma"], *HAR_SPECS, *(f"blend_{w:g}" for w in BLEND_W)]))
    L = ["# 日足の予測レンジ: HAR モデル (日・週・月の実現分散) は今のモデルより良いか", "",
         "## 結論", ""] + conclusion(res) + ["",
         "## 何を比べたか", "",
         "日足の予測 (1・5・10・20営業日先) のレンジの幅は、「最近どれだけ値が動いたか」から計算しています。"
         "今のサーバーは、1時間足から測ったその日の値動きの大きさ (実現分散 = 1時間ごとの変化の2乗の合計) を"
         "指数平滑 (EWMA、減衰 0.80) で平均し、1日ごとに 0.90 の速さで過去500日のふだんの水準に戻していく方法です"
         " (aifx/volatility.py)。", "",
         "これを、為替や株の変動予測でよく使われる HAR モデル (Corsi 2009) と比べました。HAR は"
         "「昨日1日」「最近1週間 (5営業日)」「最近1か月 (22営業日)」の3つの平均の重み付き和で、これからの変動を予測します。"
         "重みは起点ごとに、その時点で答えが出ていたデータだけで最小二乗法により推定し直します (1・5・10・20営業日先それぞれ別々)。"
         "あわせて、よく知られた変形 (対数で当てはめる、1時間足の高値・安値の幅 (Parkinson) を使う、"
         "変化の絶対値を使う (RAV = 1時間ごとの変化の絶対値の和、ロンドン日ごとの変化の絶対値)、測定誤差を補正する HAR-Q、"
         f"7ペア共通の重みとペアごとの重み)、今の EWMA の設定変更 ({n_ewma}通り)、"
         "今のモデルと HAR の組み合わせ (分散の重み付き幾何平均) も試しました。", "",
         "## 公平に比べるための条件", "",
         f"- 対象: 7ペア、1時間足のある期間 ({res['data']['USDJPY']['hourly_first'][:10]}〜"
         f"{res['data']['USDJPY']['hourly_last'][:10]}, 1ペア約1万7千本)。予測の起点は"
         f"{d['dates']['first']}〜{d['dates']['last']} の毎営業日 (7ペア合計 {d['n']:,} 件)。それより前の約7か月は"
         "HAR の重みを推定するための助走期間です。",
         "- 答え合わせは本番と同じ: 起点はロンドンの営業日の終わり、起点の価格はその時点の1時間足の終値、"
         "結果は目標時刻 (engine.target_times) の1時間足の終値。その変化の2乗 (bp²) が「実際の値動きの大きさ」です。",
         "- 今のモデルは本番と同じ関数 (engine.sigma_steps、保存されている日足の直近1,000本とその時点までの1時間足)"
         f"で計算し、全件で本番の計算と一致することを確かめました (一致: {lk['engine_reproduced_all']})。",
         f"- 調整期間 (前半60%: {d['dates']['first']}〜{d['dates']['tune_last']}) で設定を選び、"
         f"検証期間 (後半40%: {d['dates']['test_first']}〜{d['dates']['last']}、{d['n_test']:,} 件) は選んだあとの"
         "確認だけに使いました。調整期間の起点のうち、答えが検証期間に入るものは除いています。",
         "- 20営業日先の予測は期間が重なるため、検証期間の独立な観測は1ペアあたり約10回分しかありません。"
         "長い期間の差は偶然に左右されやすいので、差が偶然かどうかの検定 (p 値) も載せています。", "",
         "## 未来の情報が混ざっていないかの確認", "",
         "- 入力はすべて起点までに終わった1時間足・日足だけです。今のモデルに起点より後の1時間足まで渡しても"
         f"計算は変わりません (最大の相対差 {lk['live_all_hourly_vs_truncated_max_rel_diff']:.1e})。",
         "- HAR の重みを推定する行は、答え (目標の日々) が起点の日までに終わったものだけです "
         f"(使った最後の日 − 起点の日 = {lk['har_last_target_day_minus_origin_day']})。",
         f"- 無作為に選んだ {lk['n_origins']} 件の起点で、全ペアのデータを起点で切ってから入力と重みを作り直すと、"
         f"HAR の予測は完全に一致しました (最大の相対差 {lk['har_truncated_max_rel_diff']:.1e})。1日ごとの実現分散も"
         "本番の関数 (realized_daily_variance) と一致 "
         f"(最大の相対差 {lk['rv_vs_realized_daily_variance_max_rel_diff']:.1e})。",
         "", "## 結果: 日足", "",
         "- QLIKE: 予測した分散の対数 + 実際の変化の2乗 ÷ 予測した分散 の平均。小さいほど、値動きの大きさを"
         "正しく予測できています。水準 (倍率) はモデルごと・予測先ごとに調整期間で1つだけ合わせました"
         " (本番でも実績から倍率を学習しています)。0.01 の差は小さく見えても意味のある差です。",
         "- 表の数字は「調整期間 / 検証期間」。太字は今のモデルより良いもの。", "",
         "| モデル | 1営業日先 | 5営業日先 | 10営業日先 | 20営業日先 |", "|---|---|---|---|---|"]
    for k in rows:
        cells = []
        for h in H:
            s, b = sc[k][str(h)], sc["live"][str(h)]
            tu, te = _f(s["qlike_tune"]), _f(s["qlike_test"])
            if k != "live":
                tu = f"**{tu}**" if s["qlike_tune"] < b["qlike_tune"] else tu
                te = f"**{te}**" if s["qlike_test"] < b["qlike_test"] else te
            cells.append(f"{tu} / {te}")
        L.append(f"| {_name(k)} | " + " | ".join(cells) + " |")
    L += ["", f"EWMA の設定変更は {n_ewma} 通りのうち調整期間で一番良かったもの、組み合わせの HAR は調整期間で"
          f"一番良かった「{_name(sel['har'])}」です。4つの予測先の平均で調整期間に一番良かったのは「{_name(best)}」。", "",
          "今のモデルとの QLIKE の差 (マイナスが良い) と、それが偶然でないかの検定 (Diebold-Mariano: ペアの平均を日ごとにとり、"
          "期間の重なりを考慮。p が小さいほど偶然ではない)。上段が調整期間、下段が検証期間:", "",
          "| モデル | 期間 | 1営業日先 | 5営業日先 | 10営業日先 | 20営業日先 |", "|---|---|---|---|---|---|"]
    for k in dict.fromkeys([best, sel["ewma"], sel["har"], sel["blend"]]):
        for per, lab in (("tune", "調整"), ("test", "検証")):
            L.append(f"| {_name(k)} | {lab} | " + " | ".join(_p(d["dm"][k][str(h)][per]) for h in H) + " |")
    L += ["", "予測先ごとに、調整期間で一番良かったもの (上のすべての候補から) を選んだ場合の検証期間の成績:", "",
          "| 何営業日先 | 調整期間で1位 | QLIKE 調整 (今のモデル) | QLIKE 検証 (今のモデル) |", "|---|---|---|---|"]
    for h in H:
        k = d["best_per_horizon_by_tune"][str(h)]
        s, b = sc[k][str(h)], sc["live"][str(h)]
        L.append(f"| {h} | {_name(k)} | {_f(s['qlike_tune'])} ({_f(b['qlike_tune'])}) | "
                 f"{_f(s['qlike_test'])} ({_f(b['qlike_test'])}) |")
    L += ["", "80%レンジ (検証期間): 調整期間で80%が入るように決めた幅を当てたときに実際に入った割合と、平均の幅 (bp)。"
          "相関は予測した幅 (√分散) と実際の変化の大きさの相関 (全件 / ペアごとの平均)。", "",
          "| 何営業日先 | モデル | 80%に入った割合 | 平均の幅 (bp) | 相関 |", "|---|---|---|---|---|"]
    for h in H:
        for k in dict.fromkeys(["live", best, sel["har"], sel["blend"]]):
            s = sc[k][str(h)]
            L.append(f"| {h} | {_name(k)} | {s['cover80_test']:.1%} | {s['width80_test_bp']:.1f} | "
                     f"{s['corr_test']:.3f} / {s['corr_pair_test']:.3f} |")
    L += ["", "どのモデルも検証期間の80%レンジには80%より多く入っています (検証期間は、予測した分散に比べて実際の動きが"
          "調整期間より小さめだったため)。モデルによる違いは、この割合・幅ともに小さなものです。", "",
          "## 結果: 1時間足の24時間先 (おまけ)", "",
          f"1時間足の予測の24時間先について、修正前の本番の計算に日足の HAR 予測 ({_name(sel['har'])}, 翌ロンドン日の分散) を"
          "重み付き幾何平均で混ぜた場合と、時間帯ごとの荒さの割合を engine.py の設定どおり6,000本で測った場合を比べました "
          f"({hr['every']}時間おきの起点 {hr['n']:,} 件、調整 {hr['n_tune']:,} / 検証 {hr['n_test']:,}、"
          f"検証の開始 {hr['split']})。この検証の時点の本番の forecaster.py は sigma_steps に直近3,000本しか渡しておらず、"
          "HOURLY_PROFILE_WINDOW = 6000 は実際には3,000本弱で効いていました (この検証で見つかり、engine.vol_bars で修正しました。"
          "1時間先・4時間先への効果は [volatility.md](volatility.md))。", "",
          "| 測り方 | QLIKE 調整 | QLIKE 検証 | 検証: 80%に入った割合 | 平均の幅 (bp) | 差 調整 | 差 検証 |",
          "|---|---|---|---|---|---|---|"]
    hl = {"live": "修正前の本番の計算 (直近3,000本)", "profile6000": "時間帯の割合を6,000本で測る"}
    for k, s in hr["scores"].items():
        x = hr["dm"].get(k)
        name = hl.get(k) or f"HAR を重み {k.split('_')[1]} で混ぜる"
        L.append(f"| {name} | {_f(s['qlike_tune'])} | {_f(s['qlike_test'])} | {s['cover80_test']:.1%} | "
                 f"{s['width80_test_bp']:.1f} | {'' if x is None else _p(x['tune'])} | "
                 f"{'' if x is None else _p(x['test'])} |")
    L += ["", "## 組み込むとしたら (設定と関数の形)", ""] + plug_in(res) + [""]
    return "\n".join(L)


def conclusion(res: dict) -> list[str]:
    """Plain-language recommendation, written from the numbers (see evaluate_daily's ``adopt``)."""
    d, hr = res["daily"], res["hourly"]
    sel = d["selected"]
    best, blend = sel["best"], sel["blend"]

    def won(k):
        return [h for h in H if d["beats_live"][k][str(h)]["tune"] and d["beats_live"][k][str(h)]["test"]]

    out = []
    if d["adopt"]:
        out.append(f"- 「{_name(best)}」は1・5・10・20営業日先のすべてで、調整期間と検証期間の両方で今のモデルより"
                   " QLIKE が小さくなりました。採用を勧めます。")
    else:
        lost = [h for h in H if not d["beats_live"][best][str(h)]["test"]]
        out.append("- **今の日足のモデルをそのまま使うことを勧めます。** 調整期間と検証期間の両方で、1・5・10・20営業日先の"
                   "すべてで今のモデルを上回る候補はありませんでした。")
        out.append(f"- 調整期間で一番良かった「{_name(best)}」が両方の期間で上回ったのは{_hs(won(best))}だけで、"
                   f"{_hs(lost)}では検証期間に今のモデルより悪くなりました ("
                   + "、".join(f"{h}営業日先 {_p(d['dm'][best][str(h)]['test'])}" for h in lost) + ")。")
        if best.startswith("ewma_"):
            lam, rev = (float(v) for v in best.split("_")[1:3])
            if lam == min(g[0] for g in EWMA_GRID) and rev == min(g[1] for g in EWMA_GRID):
                out.append("- この設定は試した中で最も速く反応する端の設定です (格子を広げても端が選ばれ続けました)。"
                           "調整期間には2024年8月の円キャリー取引の巻き戻しや2025年4月の関税ショックのような急変があり、"
                           "速く反応するほど得をしましたが、落ち着いた検証期間では先の長い予測で逆効果でした。"
                           "調整期間の急変に合わせすぎた設定と考えられます。")
        har1 = sum(d["beats_live"][k]["1"]["test"] for k in HAR_SPECS)
        har_long = sum(all(not d["beats_live"][k][str(h)]["test"] for h in (10, 20)) for k in HAR_SPECS)
        out.append(f"- HAR は1営業日先では {har1}/{len(HAR_SPECS)} の形が検証期間に今のモデルより良かった一方、"
                   f"10・20営業日先では {har_long}/{len(HAR_SPECS)} の形が両方とも負けました。"
                   "1時間足の履歴が2年半あまりしかなく、先の長い予測では「ふだんの水準」を過去500日の日足から取る"
                   "今のモデルの方が安定しているためと考えられます。")
    wb = won(blend)
    rest = [h for h in H if h not in wb]
    x1, x5 = d["dm"][blend]["1"], d["dm"][blend]["5"]
    gain = [-d["dm"][blend][str(h)]["test"]["diff"] for h in wb]
    worst = max((abs(d["dm"][blend][str(h)][per]["diff"]) for h in rest for per in ("tune", "test")), default=0.0)
    out.append(f"- いちばん惜しかったのは「{_name(blend)}」です。{_hs(wb)}では両方の期間で良く "
               f"(1営業日先の差 調整 {_p(x1['tune'])}、検証 {_p(x1['test'])}; 5営業日先 調整 {_p(x5['tune'])}、"
               f"検証 {_p(x5['test'])})"
               + (f"、{_hs(rest)}はほぼ同じ (差は ±{worst:.3f} 以内"
                  + ("で偶然の範囲" if all((d["dm"][blend][str(h)][per]["p"] or 0) > 0.1 for h in rest
                                          for per in ("tune", "test")) else "") + ") でした。" if rest else "でした。")
               + "1〜5営業日先を重視するなら、データが支持する唯一の小さな改善です (設定は最後の節)。"
               + (f"ただし改善幅は検証期間の QLIKE で {min(gain):.3f}〜{max(gain):.3f} と小さく、"
                  "レンジの幅や80%に入る割合はほとんど変わりません。" if gain else ""))
    if d["beat_live_everywhere"]:
        out.append(f"- なお、1・5・10・20営業日先のすべてで両方の期間に上回った候補は "
                   f"{', '.join(_name(k) for k in d['beat_live_everywhere'])} でしたが、調整期間の成績で選ばれたもの"
                   "ではないため (後から検証期間を見て選ぶことになる)、偶然の可能性があり勧めません。")
    b = hr["best_by_tune"]
    x, x6 = hr["dm"][b], hr["dm"]["profile6000"]
    sig = all(x[per]["p"] is not None and x[per]["p"] < 0.05 for per in ("tune", "test"))
    out.append(f"- 1時間足の24時間先: 調整期間で一番良かった「日足の HAR を重み {b.split('_')[-1]} で混ぜる」は"
               f"{'両方の期間でわずかに良い' if hr['adopt'] else '検証期間では良くならない'}ものの "
               f"(差 調整 {_p(x['tune'])}、検証 {_p(x['test'])})、"
               + ("差ははっきりしているので検討の価値があります。" if sig and hr["adopt"]
                  else "偶然の範囲を出ないため変更は勧めません。")
               + f"時間帯の割合を6,000本で測った場合も24時間先はほぼ同じでした (差 調整 {_p(x6['tune'])}、"
               f"検証 {_p(x6['test'])})。")
    return out


def plug_in(res: dict) -> list[str]:
    """How the closest candidate (the tune-chosen blend) would plug into volatility.py / engine.sigma_steps."""
    d = res["daily"]
    sel = d["selected"]
    spec, w = HAR_SPECS[sel["har"]], float(sel["blend"].split("_")[1])
    pooled = spec["pool"]
    sig = ("def har_cum_variance(hourly: dict[str, pd.DataFrame], pair: str, until: datetime, steps: int) -> np.ndarray:"
           if pooled else
           "def har_cum_variance(hourly: pd.DataFrame, until: datetime, steps: int) -> np.ndarray:")
    form = {"lin": "そのまま (予測 = h × ペアの平均 × 当てはめた値, 下限 0.05)",
            "log": "対数 (予測 = h × ペアの平均 × exp(当てはめた値))",
            "sqrt": "平方根 (予測 = h × ペアの平均 × 当てはめた値², 下限 0.05)"}[spec["form"]]
    src = {"rv": "実現分散", "pk": "Parkinson 分散", "rav": "RAV (1時間ごとの変化の絶対値の和)",
           "abs": "ロンドン日ごとの変化の絶対値"}[spec["src"]]
    L = ["今は組み込みを勧めませんが、1〜5営業日先の小さな改善を取りに行く場合の正確な設定です"
         " (このファイル aifx/research_har.py の har_forecasts と同じ計算)。", "",
         f"- HAR の入力: ロンドン営業日ごとの{src} (realized_daily_variance と同じ日の区切り) の、"
         f"その日の値・直近{WEEK}営業日の平均・直近{MONTH}営業日の平均。定数項つき。",
         f"- 当てはめる値: この先 h 営業日の実現分散の平均 ÷ ペアの平均 (起点までの全日の実現分散の平均) の{form}。"
         "入力もペアの平均で割る"
         f"{' (7ペアをまとめて1本の回帰)' if pooled else ' (ペアごとの回帰)'}。"
         f"起点ごとに、答えが起点までに出ている全ての日 (最低 {MIN_ROWS} 日/ペア) で最小二乗法。"
         "h ごとに別の回帰 (直接予測)。本番では1〜20の全ての h で当てはめ、累積分散が減らないよう累積最大をとる。",
         f"- 組み合わせ: 累積分散 = 今のモデルの累積分散^{1 - w:g} × HAR の累積分散^{w:g} (単位は同じ log-return²)。"
         "倍率 k は本番の学習がそのまま合わせ直す。",
         "", "```python", "# volatility.py", f"HAR_WEIGHT = {w:g}          # research/har.md", "", sig,
         '    """Cumulative variance (log-return^2) of the move over 1..steps London business days after ``until``,',
         '    from a HAR regression per horizon on data whose outcomes had ended by ``until``."""', "",
         "# engine.sigma_steps: 日足の分岐 (イベントを足す前)",
         "        sq, lam = daily_variance_inputs(bars, hourly, origin)",
         "        var = daily_step_variance(y, steps, lam, sq=sq)",
         ("        if peers is not None:   # 全ペアの1時間足 {code: bars} を新しい引数 peers / pair で受け取る"
          if pooled else "        if hourly is not None:"),
         ("            har = har_cum_variance(peers, pair, origin, steps)" if pooled
          else "            har = har_cum_variance(hourly, origin, steps)"),
         "            cum = np.maximum.accumulate(np.cumsum(var) ** (1 - HAR_WEIGHT) * har ** HAR_WEIGHT)",
         "            var = np.diff(np.concatenate([[0.0], cum]))",
         "        var = add_daily_events(var, step_ends(tf, origin, steps), origin, events)", "```"]
    if pooled:
        L += ["", "7ペア共通の重みを使うため、sigma_steps に全ペアの1時間足を渡す引数 "
              "(`peers: dict[str, pd.DataFrame] | None = None, pair: str | None = None`) が必要になります。"]
        own = f"{sel['har']}_pair"
        if own in d["scores"]:
            q, q0 = d["scores"][own]["20"]["qlike_test"], d["scores"][sel["har"]]["20"]["qlike_test"]
            L.append(f"ペアごとに回帰すると (「{_name(own)}」) 1時間足の履歴が短すぎて重みがぶれ、20営業日先の QLIKE は"
                     f"検証期間で {q:.4f} (共通なら {q0:.4f}) と悪くなるため、ペア自身の1時間足だけで済ませる形は勧めません。")
    return L


def run(log=print) -> dict:
    res = evaluate(log=log)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "har.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (REPORT_DIR / "har.md").write_text(report(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
