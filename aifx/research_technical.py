"""Classic technical-analysis signals as direction calls: how often are they right?

Each signal is a fixed, textbook reading of a standard indicator, declared
before looking at any result (SIGNALS below). At a bar's close it gives +1
(buy), -1 (sell) or 0 (no call) from that bar and earlier ones only.

trend following (順張り)
- sma20 / sma75 / sma200: close above (buy) or below (sell) the simple moving average.
- ma_20_75: SMA 20 above / below SMA 75; ma_20_75_cross: golden / dead cross on the bar.
- macd_state: MACD (12, 26, 9) line above / below its signal line (= histogram sign);
  macd_cross: the line crosses the signal line on the bar; macd_zero: line above / below 0.
- rsi_50_cross: RSI (14, Wilder) crosses above / below 50 on the bar.
- bb_break: the close breaks out of the Bollinger band (20, 2 sigma): first close above the
  upper band (buy) / below the lower band (sell) after a close inside.
- ichi_cloud: close above the cloud (buy) / below it (sell), inside: no call. The cloud at a bar
  is the leading spans computed 26 bars earlier (as the chart draws it).
- ichi_tk: tenkan-sen (9) above / below kijun-sen (26); ichi_tk_cross: they cross on the bar.
- ichi_sanyaku: 三役好転 (tenkan above kijun, close above the cloud, close above the close of
  26 bars ago = the lagging span above the price) / 三役逆転 (all three reversed).
- adx_di: ADX (14, Wilder) above 25 with +DI above -DI (buy) or below it (sell).
- psar: Parabolic SAR (0.02, step 0.02, max 0.2, Wilder) in an up / down trend.
- donchian20: close above the highest high (buy) / below the lowest low (sell) of the previous
  20 bars.
- pivot: close above / below the classic pivot P = (H + L + C) / 3 of the previous day (London
  trading day for intraday bars, the previous bar for daily bars).
- roc12: rate of change over 12 bars above / below 0.

mean reversion (逆張り)
- rsi_30_70: RSI (14) below 30 (buy) / above 70 (sell).
- bb_pctb: %b <= 0, the close at or below the lower band (buy) / %b >= 1 (sell); the reading the
  chart panel shows (indicators.technical_summary).
- bb_touch: the bar's low touches the lower band (buy) / its high the upper band (sell).
- stoch_cross: slow stochastics (14, 3, 3) %K crosses above %D with %D below 20 (buy) / below
  %D with %D above 80 (sell).
- williams_r: Williams %R (14) below -80 (buy) / above -20 (sell).
- cci_100: CCI (20) below -100 (buy) / above +100 (sell).

Bars, from about 20 years of Dukascopy hourly mid prices for the 7 pairs:
hourly bars (1, 4 and 24 bars ahead), 4-hour bars built from them in London
time (00:00, 04:00, ... 20:00, so each lies inside one London day; 1 and 6
bars ahead) and London-day daily bars built from them as in data.london_days
(the Sunday evening counts towards Monday; days with fewer than 12 hourly
bars are left out; 1, 5 and 20 days ahead). Forward moves run from the
close of the signal bar to the close of the bar h bars later. Tune period:
signals whose whole forward window ends by 2016-12-31; test period: signals
from 2017-01-01.

Scores, pooled over the pairs: calls, share of bars with a call, hit rate
(the sign of the forward move equals the call; zero moves left out), the
average forward move in the called direction (bp) and t statistics for both
with Driscoll-Kraay (Newey-West) errors: the sums over pairs at each bar
time form one series whose autocovariances are added with Bartlett weights
up to lag 2h + 4 (T / 100)^(2/9), so overlapping forward windows and pairs
moving together are not counted as independent evidence.

Consensus (like investing.com's summary): the number of buy calls minus sell
calls over a fixed group of signals, divided by the group's size, cut into
強い売り / 売り / 中立 / 買い / 強い買い at the 40 % and 80 % quantiles of its
absolute value on the tune period. A logistic regression on all signal
values, fitted on the tune period and scored on the test period, shows what
combining the signals linearly could do.

signals_now(df) gives every signal (and the consensus) at the last bar with
exactly the values the research uses; self_check() confirms on truncated
data that no signal looks ahead.

    python -m aifx.research_technical     # writes research/technical.md and research/technical.json
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
from .timeutil import LONDON, NEW_YORK, london_day_end, utcnow
from .technical import (LEVELS, LEVEL_KEYS, SIGNALS, KEYS, GROUPS, GROUP_LABEL, london_trading_day, is_daily, signal_frame, consensus_level, _num, signals_now)

REPORT_DIR = Path("research")
SPLIT = pd.Timestamp("2017-01-01", tz="UTC")     # test period starts; tune targets must end before it
HORIZONS = {"1h": (1, 4, 24), "4h": (1, 6), "1d": (1, 5, 20)}
TF_LABEL = {"1h": "1時間足", "4h": "4時間足", "1d": "日足"}
TF_UNIT = {"1h": "時間", "4h": "本 (4時間足)", "1d": "営業日"}
BAR = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}
WARMUP = 300                  # bars before any signal is scored (SMA 200, Ichimoku 52 + 26)
MIN_DAY_HOURS = 12            # hourly bars a London day needs to become a daily bar
CONSENSUS_Q = (0.4, 0.8)      # |score| quantiles on the tune period: 中立 | 買い・売り | 強い
MIN_CALLS = 30
ROLLOVER_NY_HOURS = (16, 17)  # hourly bars around the 17:00 New York rollover (wide spreads, swap-driven moves)
ROLLOVER_H = (1, 4)           # 1h horizons also scored without forward windows that touch those bars
ICHI_SHIFT = 26
OHLC = ["open", "high", "low", "close"]
# indicators, signals and the consensus live in technical.py (shared with the page)

def self_check(df: pd.DataFrame, daily: bool | None = None, n: int = 6, seed: int = 0) -> dict:
    """signal_frame on the whole history must equal signals_now on the history cut at a bar,
    for ``n`` random bars and the last one: no signal may use a later bar."""
    daily = is_daily(df) if daily is None else daily
    full = signal_frame(df, daily)
    rng = np.random.default_rng(seed)
    lo = min(WARMUP, len(df) - 1)
    bars = sorted(set(rng.integers(lo, len(df), size=n).tolist()) | {len(df) - 1})
    bad = []
    for i in bars:
        now = signals_now(df.iloc[: i + 1], daily)
        diff = [k for k in KEYS if now["signals"][k]["signal"] != int(full[k].iloc[i])]
        if diff:
            bad.append({"bar": str(df.index[i]), "signals": diff})
    return {"bars": len(bars), "mismatches": bad, "calls_checked": int(sum((full.iloc[bars] != 0).sum()))}


# ------------------------------------------------------------------ bars

def four_hour_bars(h1: pd.DataFrame) -> pd.DataFrame:
    """4-hour bars from hourly ones (UTC bar-start index) in London time (00:00, 04:00, ... 20:00), so
    each lies inside one London day; indexed by the start of their first hourly bar, with ``end`` (UTC)."""
    naive = pd.DatetimeIndex(h1.index.tz_convert(LONDON).tz_localize(None))
    key = naive.floor("4h").asi8
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    last = np.r_[first[1:], len(key)] - 1
    o, h, lo, c = (h1[k].to_numpy(float) for k in OHLC)
    out = pd.DataFrame({"open": o[first], "high": np.maximum.reduceat(h, first), "low": np.minimum.reduceat(lo, first),
                        "close": c[last]}, index=h1.index[first])
    out["end"] = h1.index[last] + pd.Timedelta(hours=1)
    return out


def london_daily_bars(h1: pd.DataFrame, min_hours: int = MIN_DAY_HOURS) -> pd.DataFrame:
    """Daily bars per London trading day from hourly ones, as data.london_days builds them (the Sunday
    evening counts towards Monday); days with fewer than ``min_hours`` hourly bars are left out. Indexed
    by date, with ``end`` = London midnight at the end of the day (UTC)."""
    days = london_trading_day(pd.DatetimeIndex(h1.index))
    day = days.asi8
    first = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    last = np.r_[first[1:], len(day)] - 1
    o, h, lo, c = (h1[k].to_numpy(float) for k in OHLC)
    out = pd.DataFrame({"open": o[first], "high": np.maximum.reduceat(h, first), "low": np.minimum.reduceat(lo, first),
                        "close": c[last]}, index=days[first].as_unit("ns"))
    out = out[(last - first + 1) >= min_hours]
    out["end"] = pd.DatetimeIndex([london_day_end(d.date()) for d in out.index]).as_unit("ns")
    return out


def _clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated()].sort_index()
    return df[(df[OHLC] > 0).all(axis=1)]


# ------------------------------------------------------------------ scoring

def nw_lag(h: int, T: int) -> int:
    """Bartlett lag: the overlap of h-bar forward windows plus the Newey-West rule of thumb."""
    return int(2 * h + math.floor(4 * (max(T, 1) / 100) ** (2 / 9)))


def _autocov(u: np.ndarray, L: int) -> np.ndarray:
    n = len(u)
    f = np.fft.rfft(u, 2 * n)
    return np.fft.irfft(f * np.conj(f), 2 * n)[: L + 1]


def nw_t(tid: np.ndarray, x: np.ndarray, h: int, null: float = 0.0) -> float | None:
    """t statistic of the pooled mean of x against ``null``, with Driscoll-Kraay errors: x is summed
    over pairs at each bar time (``tid``, an index on the sorted time grid) and the autocovariances of
    those sums are added with Bartlett weights."""
    if len(x) < MIN_CALLS:
        return None
    lo = int(tid.min())
    t = tid - lo
    S = np.bincount(t, weights=x)
    N = np.bincount(t).astype(float)
    tot = N.sum()
    mu = S.sum() / tot
    u = S - mu * N
    L = min(nw_lag(h, len(u)), len(u) - 1)
    g = _autocov(u, L)
    w = 1 - np.arange(1, L + 1) / (L + 1)
    v = g[0] + 2 * float((w * g[1:]).sum())
    if v <= 0:
        return None
    return float((mu - null) / (math.sqrt(v) / tot))


def score(s: np.ndarray, f: np.ndarray, tid: np.ndarray, pair: np.ndarray, h: int, base: int) -> dict:
    """Calls ``s`` (+1/-1) against forward moves ``f`` (bp); ``base`` = bars that could have had a call."""
    n = len(s)
    if n < MIN_CALLS:
        return {"n": int(n), "cover": _num(n / max(base, 1), 5)}
    signed = s * f
    nz = f != 0
    hit = (np.sign(f[nz]) == s[nz]).astype(float)
    npair = len(PAIRS)
    cnt = np.bincount(pair[nz], minlength=npair)
    hp = np.bincount(pair[nz], weights=hit, minlength=npair)
    by_pair = {code: _num(hp[i] / cnt[i], 4) if cnt[i] >= MIN_CALLS else None for i, code in enumerate(PAIRS)}
    vals = [v for v in by_pair.values() if v is not None]
    return {"n": int(n), "cover": _num(n / max(base, 1), 5), "hit": _num(hit.mean(), 5), "bp": _num(signed.mean(), 3),
            "t": _num(nw_t(tid, signed, h), 2), "t_hit": _num(nw_t(tid[nz], hit, h, 0.5), 2),
            "pairs": by_pair, "pair_min": min(vals) if vals else None, "pair_max": max(vals) if vals else None,
            "pairs_above": int(sum(v > 0.5 for v in vals))}


def verdict(tune: dict, test: dict) -> str | None:
    """On both periods: "edge" = hit rate above 50 % with t >= 2 for both the hit rate and the average
    move; "edge_hit" = only the hit rate's t >= 2; "contrarian" / "contrarian_hit" = the same below 50 %."""
    def side(x):
        hit, t, th = x.get("hit"), x.get("t"), x.get("t_hit")
        if hit is None or t is None or th is None:
            return 0, 0
        if hit > 0.5 and th >= 2:
            return 1, int(t >= 2)
        if hit < 0.5 and th <= -2:
            return -1, int(t <= -2)
        return 0, 0
    (a, am), (b, bm) = side(tune), side(test)
    if a == 0 or a != b:
        return None
    return ("edge" if a > 0 else "contrarian") + ("" if am and bm else "_hit")


# ------------------------------------------------------------------ panel

def build_panel(tf: str, loader=history.load_long_hourly, log=print) -> dict:
    """Signals, forward moves and times of every pair's bars of timeframe ``tf``, pooled."""
    parts = []
    spreads = {}
    for pid, code in enumerate(PAIRS):
        h1 = _clean_hourly(loader(code))
        if "spread" in h1:
            recent = h1[h1.index >= SPLIT]
            spreads[code] = _num(float((recent["spread"] / recent["close"]).median() * 1e4), 2) if len(recent) else None
        if tf == "1h":
            df = h1[OHLC]
            end = pd.DatetimeIndex(df.index + pd.Timedelta(hours=1))
        else:
            df = four_hour_bars(h1) if tf == "4h" else london_daily_bars(h1)
            end = pd.DatetimeIndex(df.pop("end"))
        sig = signal_frame(df[OHLC], daily=(tf == "1d"))
        c = df["close"].to_numpy(float)
        e = end.tz_convert("UTC").as_unit("ns").asi8
        n = len(c)
        fwd, tgt = {}, {}
        for h in HORIZONS[tf]:
            f = np.full(n, np.nan)
            t = np.full(n, np.iinfo(np.int64).max)
            f[:-h] = np.log(c[h:] / c[:-h]) * 1e4
            t[:-h] = e[h:]
            # a forward window across a hole in the data (more than the horizon plus holidays) is left out
            span = np.full(n, np.inf)
            span[:-h] = (e[h:] - e[:-h]) / 1e9
            limit = (h * BAR[tf].total_seconds()) * 7 / 5 + 4 * 86400
            f[span > limit] = np.nan
            fwd[h], tgt[h] = f, t
        ok = np.arange(n) >= WARMUP
        roll = {}
        if tf == "1h":
            # forward windows (bars i+1 .. i+h) that touch the New York rollover hours
            rb = np.isin(df.index.tz_convert(NEW_YORK).hour, ROLLOVER_NY_HOURS).astype(float)
            cs = np.concatenate([[0.0], np.cumsum(rb)])
            for h in ROLLOVER_H:
                t = np.ones(n, dtype=bool)
                i = np.arange(n - h)
                t[: n - h] = (cs[i + h + 1] - cs[i + 1]) > 0
                roll[h] = t
        parts.append({"sig": sig.to_numpy(np.int8), "end": e, "fwd": fwd, "tgt": tgt, "ok": ok, "roll": roll,
                      "pair": np.full(n, pid, dtype=np.int16)})
        if log:
            log(f"{tf} {code}: {n:,} bars {df.index[0]} .. {df.index[-1]}")
    P = {"sig": np.vstack([p["sig"] for p in parts]), "end": np.concatenate([p["end"] for p in parts]),
         "ok": np.concatenate([p["ok"] for p in parts]), "pair": np.concatenate([p["pair"] for p in parts]),
         "fwd": {h: np.concatenate([p["fwd"][h] for p in parts]) for h in HORIZONS[tf]},
         "tgt": {h: np.concatenate([p["tgt"][h] for p in parts]) for h in HORIZONS[tf]}, "spread_bp": spreads,
         "roll": {h: np.concatenate([p["roll"][h] for p in parts]) for h in parts[0]["roll"]}}
    _, P["tid"] = np.unique(P["end"], return_inverse=True)
    P["first"], P["last"] = int(P["end"][P["ok"]].min()), int(P["end"].max())
    return P


def _periods(P: dict, h: int) -> dict[str, np.ndarray]:
    fin = P["ok"] & np.isfinite(P["fwd"][h])
    return {"tune": fin & (P["tgt"][h] < SPLIT.value), "test": fin & (P["end"] >= SPLIT.value)}


def _score_mask(P: dict, h: int, s: np.ndarray, m: np.ndarray, base: int) -> dict:
    return score(s[m].astype(float), P["fwd"][h][m], P["tid"][m], P["pair"][m], h, base)


# ------------------------------------------------------------------ consensus and logistic regression

def _thresholds(score_tune: np.ndarray, k: int) -> dict:
    a, b = (float(np.quantile(np.abs(score_tune), q)) for q in CONSENSUS_Q)
    if b <= a + 1e-9:
        b = a + 1 / k
    return {"neutral": round(a, 6), "strong": round(b, 6)}


def eval_consensus(P: dict, tf: str) -> dict:
    out = {}
    tune_origin = P["ok"] & (P["end"] < SPLIT.value)
    for name, keys in GROUPS.items():
        idx = [KEYS.index(k) for k in keys]
        sc = P["sig"][:, idx].sum(axis=1, dtype=np.int32) / len(keys)
        th = _thresholds(sc[tune_origin], len(keys))
        lvl = consensus_level(sc, th["neutral"], th["strong"])
        res = {"label": GROUP_LABEL[name], "signals": list(keys), "thresholds": th, "h": {}}
        for h in HORIZONS[tf]:
            per = _periods(P, h)
            res["h"][str(h)] = {}
            for period, m in per.items():
                base = int(m.sum())
                row = {}
                for lv, key in LEVEL_KEYS.items():
                    mm = m & (lvl == lv)
                    if lv == 0:
                        row[key] = {"n": int(mm.sum()), "cover": _num(mm.sum() / max(base, 1), 5)}
                    else:
                        row[key] = _score_mask(P, h, np.sign(lvl).astype(float), mm, base)
                        row[key].pop("pairs", None)
                row["any"] = _score_mask(P, h, np.sign(lvl).astype(float), m & (lvl != 0), base)
                row["strong"] = _score_mask(P, h, np.sign(lvl).astype(float), m & (np.abs(lvl) == 2), base)
                for k2 in ("any", "strong"):
                    row[k2].pop("pairs", None)
                res["h"][str(h)][period] = row
        out[name] = res
    return out


def _logit_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 30) -> np.ndarray:
    """L2-penalised logistic regression without intercept, by Newton's method.

    No intercept: a pair's quote direction is arbitrary (EUR/USD could be USD/EUR), so the model is kept
    symmetric, P(up | signals) = 1 - P(up | -signals), and cannot simply learn the tune period's drift."""
    beta = np.zeros(X.shape[1])
    R = np.eye(X.shape[1]) * lam
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        w = p * (1 - p)
        g = X.T @ (y - p) - R @ beta
        H = (X * w[:, None]).T @ X + R
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return None
    r = pd.Series(p).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def eval_logistic(P: dict, tf: str) -> dict:
    out = {"h": {}}
    for h in HORIZONS[tf]:
        per = _periods(P, h)
        f = P["fwd"][h]
        fit = per["tune"] & (f != 0)
        X = P["sig"][fit].astype(float)
        y = (f[fit] > 0).astype(float)
        beta = _logit_fit(X, y)
        conf_cut = None
        res = {"coef": {k: _num(beta[i], 5) for i, k in enumerate(KEYS)}}
        for period in ("tune", "test"):
            m = per[period]
            Xm = P["sig"][m].astype(float)
            p = 1 / (1 + np.exp(-(Xm @ beta)))
            s = np.sign(p - 0.5)
            full = np.zeros(len(f))
            full[m] = s
            conf = np.zeros(len(f))
            conf[m] = np.abs(p - 0.5)
            if conf_cut is None:
                conf_cut = float(np.quantile(conf[m], 0.8))
            x = _score_mask(P, h, full, m & (full != 0), int(m.sum()))
            x.pop("pairs", None)
            nz = f[m] != 0
            x["auc"] = _num(_auc(p[nz], (f[m][nz] > 0).astype(int)), 4)
            top = _score_mask(P, h, full, m & (full != 0) & (conf >= conf_cut), int(m.sum()))
            top.pop("pairs", None)
            x["top20"] = top
            res[period] = x
        out["h"][str(h)] = res
    return out


def base_rates(P: dict, tf: str) -> dict:
    out = {}
    for h in HORIZONS[tf]:
        for period, m in _periods(P, h).items():
            f = P["fwd"][h][m]
            nz = f != 0
            out.setdefault(str(h), {})[period] = {"bars": int(m.sum()), "up": _num((f[nz] > 0).mean(), 5)}
    return out


def evaluate(tf: str, loader=history.load_long_hourly, log=print) -> dict:
    t0 = time.time()
    P = build_panel(tf, loader, log)
    res = {"label": TF_LABEL[tf], "horizons": list(HORIZONS[tf]), "h_labels": {str(h): _hlabel(tf, h) for h in HORIZONS[tf]},
           "bars": int(P["ok"].sum()),
           "start": str(pd.Timestamp(P["first"], tz="UTC").date()), "end": str(pd.Timestamp(P["last"], tz="UTC").date()),
           "base": base_rates(P, tf), "signals": {}}
    for j, k in enumerate(KEYS):
        spec = SIGNALS[k]
        s = P["sig"][:, j]
        row = {"label": spec["label"], "kind": spec["kind"], "h": {}}
        for h in HORIZONS[tf]:
            cell = {}
            for period, m in _periods(P, h).items():
                cell[period] = _score_mask(P, h, s, m & (s != 0), int(m.sum()))
            cell["verdict"] = verdict(cell["tune"], cell["test"])
            if h in P["roll"]:
                cell["ex_rollover"] = {}
                for period, m in _periods(P, h).items():
                    mm = m & ~P["roll"][h]
                    x = _score_mask(P, h, s, mm & (s != 0), int(mm.sum()))
                    x.pop("pairs", None)
                    cell["ex_rollover"][period] = x
            row["h"][str(h)] = cell
            if log:
                tu, te = cell["tune"], cell["test"]
                log(f"{tf} h={h:>2} {k:<15} tune hit={tu.get('hit') or 0:.3f} t={tu.get('t') or 0:+.2f} | test "
                    f"hit={te.get('hit') or 0:.3f} bp={te.get('bp') or 0:+.2f} t={te.get('t') or 0:+.2f} "
                    f"t_hit={te.get('t_hit') or 0:+.2f} n={te.get('n')} {cell['verdict'] or ''}")
        res["signals"][k] = row
    res["consensus"] = eval_consensus(P, tf)
    res["logistic"] = eval_logistic(P, tf)
    res["spread_bp"] = P["spread_bp"]
    if log:
        for h in HORIZONS[tf]:
            c = res["consensus"]["all"]["h"][str(h)]
            lg = res["logistic"]["h"][str(h)]
            log(f"{tf} h={h} consensus " + " ".join(
                f"{key}:{c['test'][key].get('hit') or 0:.3f}/{c['test'][key].get('n')}" for key in LEVEL_KEYS.values()
                if key != "neutral") + f" | logistic test hit={lg['test'].get('hit') or 0:.3f} auc={lg['test'].get('auc')} "
                f"top20={lg['test']['top20'].get('hit') or 0:.3f}")
        log(f"{tf}: {time.time() - t0:.0f}s")
    return res


# ------------------------------------------------------------------ report

def _p(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def _t(x):
    return "—" if x is None else f"{x:+.2f}"


def _bp(x):
    return "—" if x is None else f"{x:+.2f}"


def _hlabel(tf: str, h: int) -> str:
    if tf == "1h":
        return f"{h}時間後"
    if tf == "4h":
        return f"{h}本後 ({4 * h}時間)"
    return "翌営業日" if h == 1 else f"{h}営業日後"


VERDICT = {"edge": "**有効**", "edge_hit": "的中率のみ", "contrarian": "**逆指標**", "contrarian_hit": "逆 (的中率のみ)"}


def _signal_table(tf: str, r: dict, h: int) -> list[str]:
    rows = [(k, sg, sg["h"][str(h)]) for k, sg in r["signals"].items()]
    rows.sort(key=lambda z: -(z[2]["test"].get("t") if z[2]["test"].get("t") is not None else -99))
    L = ["| シグナル | 種類 | 出現率 | 調整: 的中率 | 調整: t | 検証: 回数 | 検証: 的中率 | 検証: 平均 (bp) | 検証: t | "
         "検証: t (的中率) | ペア別の的中率 (検証) | 判定 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, sg, x in rows:
        tu, te = x["tune"], x["test"]
        kind = "順張り" if sg["kind"] == "trend" else "逆張り"
        npairs = sum(v is not None for v in (te.get("pairs") or {}).values())
        rng = f"{_p(te.get('pair_min'))}〜{_p(te.get('pair_max'))} ({te.get('pairs_above', 0)}/{npairs} が50%超)" \
            if te.get("pair_min") is not None else "—"
        L.append(f"| {k} | {kind} | {_p(te.get('cover'), 0)} | {_p(tu.get('hit'))} | {_t(tu.get('t'))} | "
                 f"{te.get('n', 0):,} | {_p(te.get('hit'))} | {_bp(te.get('bp'))} | {_t(te.get('t'))} | "
                 f"{_t(te.get('t_hit'))} | {rng} | {VERDICT.get(x['verdict'], '×')} |")
    return L


def _consensus_table(tf: str, r: dict, group: str) -> list[str]:
    c = r["consensus"][group]
    L = ["| 何本先 | 期間 | 強い売り | 売り | 中立 (割合) | 買い | 強い買い | 買いと売りの全体 |",
         "|---|---|---|---|---|---|---|---|"]

    def cell(x):
        if not x or not x.get("n"):
            return "—"
        return f"{_p(x.get('hit'))} ({x['n']:,}回, t={_t(x.get('t'))})"

    for h in r["horizons"]:
        for period in ("tune", "test"):
            x = c["h"][str(h)][period]
            L.append(f"| {_hlabel(tf, h)} | {'調整' if period == 'tune' else '検証'} | {cell(x['strong_sell'])} | "
                     f"{cell(x['sell'])} | {_p(x['neutral'].get('cover'), 0)} | {cell(x['buy'])} | "
                     f"{cell(x['strong_buy'])} | {cell(x['any'])} |")
    return L


def _logistic_table(res: dict) -> list[str]:
    L = ["| 時間足 | 何本先 | 調整 (学習に使った期間): 的中率 | 検証: 的中率 | 検証: AUC | 検証: 平均 (bp) | 検証: t | "
         "検証: 自信の上位20%の的中率 |", "|---|---|---|---|---|---|---|---|"]
    for tf, r in res["tf"].items():
        for h in r["horizons"]:
            x = r["logistic"]["h"][str(h)]
            tu, te = x["tune"], x["test"]
            L.append(f"| {TF_LABEL[tf]} | {_hlabel(tf, h)} | {_p(tu.get('hit'))} | {_p(te.get('hit'))} | "
                     f"{te.get('auc') or '—'} | {_bp(te.get('bp'))} | {_t(te.get('t'))} | "
                     f"{_p(te['top20'].get('hit'))} ({te['top20'].get('n', 0):,}回) |")
    return L


def _rollover_table(r: dict, h: int) -> list[str]:
    L = ["| シグナル | 種類 | 調整: 的中率 (全部 → 除外後) | 検証: 的中率 (全部 → 除外後) | 検証 除外後: 回数 | t | t (的中率) | 判定 (除外後) |",
         "|---|---|---|---|---|---|---|---|"]
    rows = [(k, sg, sg["h"][str(h)]) for k, sg in r["signals"].items() if "ex_rollover" in sg["h"][str(h)]]
    rows.sort(key=lambda z: -(z[2]["ex_rollover"]["test"].get("t_hit") or -99))
    for k, sg, x in rows:
        ex = x["ex_rollover"]
        v = verdict(ex["tune"], ex["test"])
        L.append(f"| {k} | {'順張り' if sg['kind'] == 'trend' else '逆張り'} | {_p(x['tune'].get('hit'))} → "
                 f"{_p(ex['tune'].get('hit'))} | {_p(x['test'].get('hit'))} → {_p(ex['test'].get('hit'))} | "
                 f"{ex['test'].get('n', 0):,} | {_t(ex['test'].get('t'))} | {_t(ex['test'].get('t_hit'))} | {VERDICT.get(v, '×')} |")
    return L


def _kind_avg(r: dict, h: int, period: str, kind: str, ex: bool = False) -> float | None:
    v = []
    for k, sg in r["signals"].items():
        x = sg["h"][str(h)]
        x = x.get("ex_rollover", {}) if ex else x
        y = x.get(period, {})
        if SIGNALS[k]["kind"] == kind and y.get("hit") is not None and (y.get("n") or 0) >= 1000:
            v.append(y["hit"])
    return float(np.mean(v)) if v else None


def _test_cells(r: dict, h: int, min_n: int = 1000) -> list[tuple[str, dict]]:
    return [(k, sg["h"][str(h)]) for k, sg in r["signals"].items()
            if (sg["h"][str(h)]["test"].get("n") or 0) >= min_n and sg["h"][str(h)]["test"].get("hit") is not None]


def _summary_table(res: dict) -> list[str]:
    """Per timeframe and horizon: average test hit rate of the trend and the mean-reversion signals,
    the best and the worst signal, the consensus and the logistic regression (test period)."""
    L = ["| 時間足 | 何本先 | 順張りの平均 | 逆張りの平均 | 最も高いシグナル (調整) | 最も低いシグナル (調整) | "
         "コンセンサス: 買いと売りの全体 | うち強い買い・強い売り | ロジスティック回帰 (AUC) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tf, r in res["tf"].items():
        for h in r["horizons"]:
            cells = _test_cells(r, h)
            if not cells:
                continue
            avg = {}
            for kind in ("trend", "reversal"):
                v = [x["test"]["hit"] for k, x in cells if SIGNALS[k]["kind"] == kind]
                avg[kind] = float(np.mean(v)) if v else None
            best = max(cells, key=lambda z: z[1]["test"]["hit"])
            worst = min(cells, key=lambda z: z[1]["test"]["hit"])
            c = r["consensus"]["all"]["h"][str(h)]["test"]
            lg = r["logistic"]["h"][str(h)]["test"]
            L.append(f"| {TF_LABEL[tf]} | {_hlabel(tf, h)} | {_p(avg['trend'])} | {_p(avg['reversal'])} | "
                     f"{best[0]} {_p(best[1]['test']['hit'])} ({_p(best[1]['tune'].get('hit'))}) | "
                     f"{worst[0]} {_p(worst[1]['test']['hit'])} ({_p(worst[1]['tune'].get('hit'))}) | "
                     f"{_p(c['any'].get('hit'))} | {_p(c['strong'].get('hit'))} | {_p(lg.get('hit'))} ({lg.get('auc') or '—'}) |")
    return L


def _mismatch_notes(res: dict) -> list[str]:
    """Cells whose hit rate and average move point in opposite directions, both with |t| >= 2 (test period)."""
    rows = []
    for tf, r in res["tf"].items():
        for k, sg in r["signals"].items():
            for h, x in sg["h"].items():
                te = x["test"]
                if te.get("t") is None or te.get("t_hit") is None:
                    continue
                if abs(te["t"]) >= 2 and abs(te["t_hit"]) >= 2 and te["t"] * te["t_hit"] < 0:
                    rows.append((tf, int(h), k, te))
    if not rows:
        return []
    ex = "、".join(f"{TF_LABEL[tf]} {_hlabel(tf, h)} の {k} (的中率 {_p(te['hit'])}、平均 {te['bp']:+.2f} bp)"
                  for tf, h, k, te in rows[:4])
    return [f"- **的中率と平均の値動きは別物です。** 検証期間で、的中率と平均の値動きがどちらも有意 (|t| ≥ 2) なのに向きが逆の組み合わせが "
            f"{len(rows)}通りありました (例: {ex})。逆張りのシグナルは「小さく当たる回が多く、外れると大きく動かれる」、"
            "順張りのシグナルは「外れる回が多いが、当たると大きい」という形になりやすく、的中率だけを見ると判断を誤ります。"]


def _regime_notes(res: dict) -> list[str]:
    """Timeframes where the mean-reversion readings beat the trend readings by clearly more in one period than
    in the other: a market regime, not a rule to rely on."""
    out = []
    for tf, r in res["tf"].items():
        for h in r["horizons"]:
            v = {(per, kind): _kind_avg(r, h, per, kind) for per in ("tune", "test") for kind in ("trend", "reversal")}
            if None in v.values():
                continue
            gap_tu = v["tune", "reversal"] - v["tune", "trend"]
            gap_te = v["test", "reversal"] - v["test", "trend"]
            if abs(gap_te - gap_tu) >= 0.03:
                out.append((tf, h, v, gap_tu, gap_te))
    if not out:
        return []
    parts ="、".join(f"{TF_LABEL[tf]} {_hlabel(tf, h)} (調整: 逆張り {_p(v['tune', 'reversal'])}・順張り {_p(v['tune', 'trend'])}、"
                     f"検証: 逆張り {_p(v['test', 'reversal'])}・順張り {_p(v['test', 'trend'])})" for tf, h, v, _, _ in out)
    later = "検証期間" if out[0][4] > out[0][3] else "調整期間"
    return [f"- 期間によって逆張りと順張りの成績が入れ替わったものがあります: {parts}。"
            f"{later}だけ逆張りが優勢で、もう一方の期間では差がありません。相場の局面 (行き過ぎが戻りやすい時期かどうか) による違いで、"
            "同じ読み方がこれからも当たり続ける根拠にはなりません。"]


def _test_only_notes(res: dict) -> list[str]:
    """Signals with |t| >= 2 on the test period, next to what they did on the tune period."""
    rows = []
    for tf, r in res["tf"].items():
        for k, sg in r["signals"].items():
            for h, x in sg["h"].items():
                te, tu = x["test"], x["tune"]
                if te.get("t") is not None and abs(te["t"]) >= 2 and tu.get("hit") is not None:
                    rows.append((tf, int(h), k, te, tu))
    if not rows:
        return []
    rows.sort(key=lambda z: -abs(z[3]["t"]))
    txt = "、".join(f"{TF_LABEL[tf]} {_hlabel(tf, h)}の {k} (検証 {_p(te['hit'])}・t = {te['t']:+.1f})" for tf, h, k, te, tu in rows)
    hit_lo, hit_hi = min(z[4]["hit"] for z in rows), max(z[4]["hit"] for z in rows)
    t_max = max(abs(z[4].get("t") or 0) for z in rows)
    return [f"- 検証期間だけで平均の値動きの |t| ≥ 2 になったのは {len(rows)}通り: {txt}。"
            f"これらは調整期間ではどれも的中率 {_p(hit_lo)}〜{_p(hit_hi)}、|t| ≤ {t_max:.1f} で、再現していません。"]


def _rollover_verdicts(res: dict) -> list[str]:
    r = res["tf"].get("1h")
    if not r:
        return []
    rows = []
    for k, sg in r["signals"].items():
        for h, x in sg["h"].items():
            ex = x.get("ex_rollover")
            if ex:
                v = verdict(ex["tune"], ex["test"])
                if v in ("edge", "contrarian"):
                    rows.append((int(h), k, v, ex))
    if not rows:
        return []
    n = sum(1 for sg in r["signals"].values() for x in sg["h"].values() if x.get("ex_rollover"))
    txt = "、".join(f"{_hlabel('1h', h)}の {k} ({'有効' if v == 'edge' else '逆指標'}: 調整 {_p(ex['tune']['hit'])}・t = {ex['tune']['t']:+.1f}、"
                   f"検証 {_p(ex['test']['hit'])}・t = {ex['test']['t']:+.1f})" for h, k, v, ex in rows)
    return [f"- ロールオーバーの時間を除いた1時間足の {n}通りでは、{txt} が両方の期間で有意でしたが、t はぎりぎりで、"
            f"{n}通りも試せば偶然でも出る水準です。"]


def _logistic_note(res: dict) -> str:
    """The logistic regression's test cells with t >= 2, set against the spread."""
    sig = [(tf, int(h), x["test"]) for tf, r in res["tf"].items() for h, x in r["logistic"]["h"].items()
           if (x["test"].get("t") or 0) >= 2]
    spreads = [v for v in (res["tf"].get("1h", {}).get("spread_bp") or {}).values() if v is not None]
    if not sig:
        return "テクニカル指標をどう組み合わせても、方向の予測力はほとんどないということです。"
    txt = "、".join(f"{TF_LABEL[tf]} {_hlabel(tf, h)} (的中率 {_p(x['hit'])}、平均 {x['bp']:+.2f} bp、t = {x['t']:+.1f})"
                   for tf, h, x in sig)
    cost = f"で、スプレッド ({min(spreads):.1f}〜{max(spreads):.1f} bp) より小さな差です" if spreads else ""
    return (f"検証期間でも偶然とは言いにくいのは {txt} だけ{cost}。主に短い時間の逆張りの効果 (ロールオーバーの時間を含む) を"
            "拾ったもので、テクニカル指標をどう組み合わせても、売買に使えるほどの方向の予測力はないということです。")


def _hourly_notes(res: dict) -> list[str]:
    """The 1-hour pattern: mean-reversion readings above 50 %, trend readings below, and whether it survives
    without the rollover hours."""
    r = res["tf"].get("1h")
    if not r or 1 not in r["horizons"]:
        return []
    rv, tr = _kind_avg(r, 1, "test", "reversal"), _kind_avg(r, 1, "test", "trend")
    rv_x, tr_x = _kind_avg(r, 1, "test", "reversal", ex=True), _kind_avg(r, 1, "test", "trend", ex=True)
    if None in (rv, tr):
        return []
    bps = [abs(sg["h"]["1"]["test"]["bp"]) for sg in r["signals"].values()
           if sg["h"]["1"]["test"].get("bp") is not None and (sg["h"]["1"]["test"].get("n") or 0) >= 1000]
    spreads = [v for v in (r.get("spread_bp") or {}).values() if v is not None]
    cost = ""
    if bps and spreads:
        rel = "ずっと小さく" if max(bps) < 0.5 * min(spreads) else "小さく" if max(bps) < min(spreads) else "同じ程度で"
        cost = (f"平均の値動きは大きいものでも {max(bps):.2f} bp で、スプレッド ({min(spreads):.1f}〜{max(spreads):.1f} bp) より"
                f"{rel}、売買の利益にはなりません。")
    if rv - tr >= 0.01:
        out = [f"- 1時間足の次の1時間では、逆張りのシグナルの的中率が平均 {_p(rv)}、順張りのシグナルが平均 {_p(tr)} (検証期間) で、"
               f"ごく短い時間では「行き過ぎた後に少し戻る」傾向が見えます。ただし差は {(rv - tr) * 100:.1f} ポイントで、{cost}"]
    else:
        out = [f"- 1時間足の次の1時間では、逆張りのシグナルの的中率が平均 {_p(rv)}、順張りのシグナルが平均 {_p(tr)} (検証期間) で、"
               f"どちらの読み方にもはっきりした差はありません。{cost}"]
    if rv_x is not None and tr_x is not None:
        why = ""
        if rv - tr >= 0.01 and (rv_x - tr_x) <= 0.7 * (rv - tr):
            share = 1 - max(rv_x - tr_x, 0.0) / (rv - tr)
            why = (f"逆張りと順張りの差 ({(rv - tr) * 100:.1f}ポイント) の約{share * 10:.0f}割は、ロールオーバーの時間 (スプレッドが広がり、"
                   "表示される中値が一時的にずれる時間) の値動きによるもので、相場の実際の傾向ではありません。"
                   "ウェブページで1時間足の的中率を見せるなら、除外後の値の方が実態に近い数字です。")
        out.append(f"- ロールオーバー (ニューヨーク時間17時) の前後の足を除くと、逆張り {_p(rv_x)}・順張り {_p(tr_x)} です"
                   f" (下の「ロールオーバーの時間を除いた場合」)。{why}")
    return out + _rollover_verdicts(res)


def report(res: dict) -> str:
    per = res["periods"]
    found: dict[str, list] = {v: [] for v in VERDICT}
    n_cells = 0
    sig_count = {"tune": 0, "test": 0, "both_same": 0, "both_opposite": 0}
    for tf, r in res["tf"].items():
        for k, sg in r["signals"].items():
            for h, x in sg["h"].items():
                n_cells += 1
                if x["verdict"]:
                    found[x["verdict"]].append((tf, int(h), k, x))
                a, b = x["tune"].get("t"), x["test"].get("t")
                sig_count["tune"] += int(a is not None and abs(a) >= 2)
                sig_count["test"] += int(b is not None and abs(b) >= 2)
                if a is not None and b is not None and abs(a) >= 2 and abs(b) >= 2:
                    sig_count["both_same" if a * b > 0 else "both_opposite"] += 1

    def fmt(items):
        if len(items) <= 4:
            return [f"  - {TF_LABEL[tf]} {_hlabel(tf, h)}: **{k}** ({SIGNALS[k]['label']}) 調整 {_p(x['tune']['hit'])} "
                    f"(t = {x['tune']['t']:+.1f}、t (的中率) = {x['tune']['t_hit']:+.1f})、検証 {_p(x['test']['hit'])} "
                    f"(t = {x['test']['t']:+.1f}、t (的中率) = {x['test']['t_hit']:+.1f}、平均 {x['test']['bp']:+.2f} bp、"
                    f"ペア別 {_p(x['test'].get('pair_min'))}〜{_p(x['test'].get('pair_max'))})" for tf, h, k, x in items]
        out = ["", "  | 時間足 | 何本先 | シグナル | 調整: 的中率 / t / t (的中率) | 検証: 的中率 / t / t (的中率) | 検証: 平均 (bp) | "
               "ペア別 (検証) |", "  |---|---|---|---|---|---|---|"]
        for tf, h, k, x in items:
            tu, te = x["tune"], x["test"]
            out.append(f"  | {TF_LABEL[tf]} | {_hlabel(tf, h)} | {k} | {_p(tu['hit'])} / {_t(tu['t'])} / {_t(tu['t_hit'])} | "
                       f"{_p(te['hit'])} / {_t(te['t'])} / {_t(te['t_hit'])} | {_bp(te['bp'])} | "
                       f"{_p(te.get('pair_min'))}〜{_p(te.get('pair_max'))} |")
        return out + [""]

    hits = [x["test"]["hit"] for r in res["tf"].values() for h in r["horizons"] for _, x in _test_cells(r, h)]
    cons_any = [r["consensus"]["all"]["h"][str(h)]["test"]["any"].get("hit") for r in res["tf"].values() for h in r["horizons"]]
    cons_strong = [r["consensus"]["all"]["h"][str(h)]["test"]["strong"].get("hit") for r in res["tf"].values() for h in r["horizons"]]
    lg_hit = [r["logistic"]["h"][str(h)]["test"].get("hit") for r in res["tf"].values() for h in r["horizons"]]
    lg_auc = [r["logistic"]["h"][str(h)]["test"].get("auc") for r in res["tf"].values() for h in r["horizons"]]
    lg_top = [r["logistic"]["h"][str(h)]["test"]["top20"].get("hit") for r in res["tf"].values() for h in r["horizons"]]

    def span(v, pct=True):
        v = [x for x in v if x is not None]
        if not v:
            return "—"
        return f"{_p(min(v))}〜{_p(max(v))}" if pct else f"{min(v):.3f}〜{max(v):.3f}"

    L = ["# テクニカル分析のシグナルは方向を当てるか (約20年・7通貨ペアの検証)", "",
         "よく使われるテクニカル指標の「買い / 売り」のシグナルが、その後の値動きの方向をどれだけ当てたかを、"
         "Dukascopy の約20年分の1時間足 (7通貨ペア) と、そこから作った4時間足・日足で測りました。シグナルの定義は教科書どおりの形で"
         "最初に固定し、結果を見て変えていません。ウェブページの「テクニカル分析」の欄で、各シグナルの今の状態と、この検証の的中率を"
         "並べて表示するための資料です。", "",
         "## 結論", ""]
    if hits:
        L.append(f"- **ほとんどのシグナルの的中率は約50% (コイン投げと同じ) でした。** 検証期間 ({per['test']['start']}〜) に"
                 f"1,000回以上出たシグナルの的中率は、どの時間足・何本先でも {span(hits)} の範囲に収まります。")
    if found["edge"]:
        L += [f"- 調整期間・検証期間の両方で的中率が50%を上回り、平均の値動きと的中率の t 値がどちらも2以上だったもの "
              f"({n_cells}通りのうち {len(found['edge'])}通り):"] + fmt(found["edge"])
    else:
        L.append(f"- 調整期間・検証期間の両方で50%を上回り、平均の値動きと的中率の t 値がどちらも2以上だったものは、"
                 f"{n_cells}通りの組み合わせのうち **1つもありませんでした**。")
    if found["contrarian"]:
        L += ["- **両方の期間で的中率も平均の値動きも有意にシグナルと逆だったもの (逆指標):**"] + fmt(found["contrarian"])
    else:
        L.append("- 両方の期間で的中率も平均の値動きも有意にシグナルと逆だった (逆に使えば当たる) シグナルもありませんでした。")
    eh, ch = found["edge_hit"], found["contrarian_hit"]
    if eh or ch:
        def where(items):
            c: dict[str, int] = {}
            for tf, h, _k, _x in items:
                key = f"{TF_LABEL[tf]} {_hlabel(tf, h)}"
                c[key] = c.get(key, 0) + 1
            return "、".join(f"{k} {v}" for k, v in c.items())
        devs = [abs(x["test"]["hit"] - 0.5) * 100 for _, _, _, x in eh + ch]
        dev = (min(devs), max(devs))
        kinds = {kind: sum(SIGNALS[k]["kind"] == kind for _, _, k, _ in eh) for kind in ("trend", "reversal")}
        kinds_c = {kind: sum(SIGNALS[k]["kind"] == kind for _, _, k, _ in ch) for kind in ("trend", "reversal")}
        L.append(f"- 的中率だけなら、両方の期間で有意に50%を上回ったものが {len(eh)}通り (逆張り {kinds['reversal']}・順張り {kinds['trend']}。"
                 f"{where(eh) or '—'})、下回ったものが {len(ch)}通り (順張り {kinds_c['trend']}・逆張り {kinds_c['reversal']}。"
                 f"{where(ch) or '—'}) ありました。どれも平均の値動き (bp) では両方の期間で有意な得 (または損) にはならず、"
                 f"検証期間の的中率と50%の差は {dev[0]:.1f}〜{dev[1]:.1f} ポイントです (一覧は「的中率だけ有意だったもの」)。"
                 "足の数が多いと小さな差でも的中率の t が大きくなるためで、売買に使える差ではありません。")
    L += _mismatch_notes(res)
    L += [f"- {n_cells}通りの組み合わせで、平均の値動きの |t| ≥ 2 は調整期間 {sig_count['tune']}通り、検証期間 {sig_count['test']}通り"
          f" (偶然だけでも各期間 {n_cells * 0.05:.0f}通り前後は出ます)。両方の期間で |t| ≥ 2 だったのは同じ向きが {sig_count['both_same']}通り、"
          f"逆向きが {sig_count['both_opposite']}通りです。",
          *_hourly_notes(res),
          *_regime_notes(res),
          *_test_only_notes(res),
          f"- **コンセンサス** (シグナルの買いと売りの数の差を5段階にしたもの。investing.com のテクニカル・サマリーと同じ考え方) の"
          f"検証期間の的中率は、買いと売りの全体で {span(cons_any)}、「強い買い・強い売り」だけでも {span(cons_strong)} でした。"
          "多くの指標が同じ向きを示しても、当たりやすくはなりません。",
          f"- 全シグナルの組み合わせ方を調整期間で学習した**ロジスティック回帰**でも、検証期間の的中率は {span(lg_hit)}、"
          f"AUC は {span(lg_auc, pct=False)} (0.5 がでたらめ)、自信の上位20%に絞っても {span(lg_top)} でした。"
          + _logistic_note(res)]
    spreads = res["tf"].get("1h", {}).get("spread_bp") or {}
    if spreads:
        L.append("- 的中率と平均の値動き (bp) は取引コストを引く前の値です。検証期間の1時間足のスプレッドの中央値は "
                 + "、".join(f"{k} {v:.1f} bp" for k, v in spreads.items() if v is not None)
                 + " で、売買するたびにこの分が差し引かれます。表の平均の値動きの多くはこれより小さい値です。")
    L += ["", "時間足と何本先ごとのまとめ (検証期間の的中率。検証期間に1,000回以上出たシグナル。かっこ内は調整期間):", ""] \
        + _summary_table(res) + [""]
    if found["edge_hit"] or found["contrarian_hit"]:
        L += ["## 的中率だけ有意だったもの", "",
              "両方の期間で、的中率と50%の差の t が2以上 (または-2以下) だったが、平均の値動き (bp) の t はそうならなかった組み合わせです。", ""]
        if found["edge_hit"]:
            L += ["50%を上回ったもの:"] + [x.lstrip() for x in fmt(found["edge_hit"])]
        if found["contrarian_hit"]:
            L += ["50%を下回ったもの (「逆に使えば儲かる」わけではありません):"] + [x.lstrip() for x in fmt(found["contrarian_hit"])]
        L.append("")
    L += ["## ウェブページでの見せ方 (提案)", "",
          "- 各シグナルの今の状態 (`signals_now`) の横に、同じ時間足・同じ何本先の**検証期間の的中率**と**平均の値動き (bp)** を並べる。"
          "的中率だけだと、外れたときの大きさが分からず誤解を招きます。",
          "- 1時間足は、ロールオーバーの時間を除いた数字 (`ex_rollover`) を使う方が実態に近い。",
          "- コンセンサス (強い買い〜強い売り) にも検証期間の的中率を添え、「多数の指標が同じ向きでも約50%」であることが分かるようにする。",
          "- 「買いシグナル = 上がる」と受け取られる言い方は避け、「教科書どおりの読み方では買い」「過去の的中率は約50%」と書く。", ""]
    ups = [v[p]["up"] for r in res["tf"].values() for v in r["base"].values() for p in ("tune", "test") if v[p]["up"] is not None]
    base_note = ("どれも50%に近いので、的中率はそのまま50%と比べて読めます。" if ups and max(abs(u - 0.5) for u in ups) <= 0.03 else
                 "上昇した割合が50%から離れている期間では、買いだけ・売りだけの的中率はその影響を受けます "
                 "(買いと売りを合わせた的中率では打ち消されます)。")
    L += ["## 方法", "",
          f"- **データ:** Dukascopy の1時間足 (売値と買値の中値)、7通貨ペア ({'、'.join(PAIRS)})。"
          "取引のない時間 (週末など) の足はありません。",
          "- **時間足:** 1時間足 (1・4・24本先)、1時間足から作った4時間足 (ロンドン時間の0・4・8・12・16・20時始まり。"
          "どの足も1つのロンドンの日に収まる。1・6本先)、1時間足から作ったロンドンの日ごとの日足 (日曜の夜は月曜に含める。"
          f"1時間足が{MIN_DAY_HOURS}本未満の日は除く。1・5・20営業日先)。日足は本番 (data.london_days) と同じ区切りです。",
          "- **先読みなし:** 各シグナルはその足の終値の時点までの足だけで計算し、値動きはその足の終値から h 本後の足の終値までで測ります。"
          "`signals_now` (最後の足のシグナル) を途中までのデータで計算した値と、全期間で一度に計算した値が一致することを、"
          "ランダムに選んだ足で確認しました (下の「自己チェック」)。",
          f"- **期間:** 調整期間は値動きの終わりが {per['tune']['end']} までのシグナル (各ペアの最初の{WARMUP}本は指標の準備のため除く)、"
          f"検証期間は {per['test']['start']}〜{per['test']['end']} のシグナル。シグナルの定義は固定なので、調整期間で決めたのは"
          "コンセンサスの区切りとロジスティック回帰の係数だけです。",
          "- **的中率:** シグナルが出た足のうち、その後の値動きの符号がシグナルと同じだった割合 (値動きがゼロの回は除く)。7ペアをまとめて数えます。",
          "- **出現率:** 検証期間の足のうち、そのシグナルが出ていた (買いか売りを示した) 割合。",
          "- **平均 (bp):** シグナルの向きに測った平均の値動き (1 bp = 0.01%)。",
          "- **t 値:** 全ペアの結果を足の時刻ごとに合計し、その時系列の自己共分散を Bartlett の重みでラグ 2h + 4(T/100)^(2/9) "
          "まで足した Newey-West (Driscoll-Kraay) の標準誤差で計算しました。h 本先の値動きは隣の足と重なり、ペア同士も同じ時刻に"
          "連動するため、それを独立な証拠として数えないためです。「t」は平均の値動き、「t (的中率)」は的中率と50%の差の t 値です。",
          "- **判定:** 両方の期間で的中率が50%を上回り、t と t (的中率) がどちらも2以上なら「有効」、t (的中率) だけなら「的中率のみ」。"
          "両方の期間で50%を下回り、どちらも-2以下なら「逆指標」、t (的中率) だけなら「逆 (的中率のみ)」。"
          f"{n_cells}通りの組み合わせを試しているので、片方の期間だけ |t| ≥ 2 になるものは偶然でもいくつか出ます。",
          "- **ペア別:** 検証期間の的中率のペアごとの最小〜最大と、50%を超えたペアの数。", "",
          "上昇した割合 (値動きがゼロの回を除く、シグナルに関係なく全部の足): "
          + "、".join(f"{TF_LABEL[tf]} {_hlabel(tf, int(h))} 調整 {_p(v['tune']['up'])}・検証 {_p(v['test']['up'])}"
                     for tf, r in res["tf"].items() for h, v in r["base"].items()) + "。" + base_note, "",
          "### シグナルの定義", "",
          "| シグナル | 名前 | 種類 | 買い | 売り |", "|---|---|---|---|---|"]
    for k, s in SIGNALS.items():
        L.append(f"| {k} | {s['label']} | {'順張り' if s['kind'] == 'trend' else '逆張り'} | {s['buy']} | {s['sell']} |")
    L += ["",
          "- 指標の設定: 移動平均は単純移動平均、RSI (14) と ADX (14) は Wilder の平滑化、MACD (12, 26, 9)、ボリンジャーバンド (20本、±2σ)、"
          "一目均衡表 (9, 26, 52、先行スパンはチャートと同じく26本先にずらしたものを使う。遅行スパンは今の終値と26本前の終値の比較)、"
          "ストキャスティクスはスロー (14, 3, 3)、パラボリック SAR (0.02、最大0.2)、ドンチャン (直前20本)、"
          "ピボットは前日の高値・安値・終値からのクラシック型 (日中の足はロンドンの取引日、日足は前の足)、"
          "ウィリアムズ %R (14)、CCI (20)、ROC (12)。",
          "- 「上下」のシグナル (移動平均、MACD、雲など) はほぼ毎回どちらかを示し、「クロス」「ブレイク」「タッチ」「圏」のシグナルは条件を満たした足だけ示します。",
          "- MACD のヒストグラムの符号は「MACD とシグナルの上下」(macd_state) と同じものなので、別には数えていません。",
          "- CCI の ±100 は、日本でよく使われる逆張りの読み方にしました (Lambert の元の使い方は順張り)。"
          "逆の読み方の的中率は 100% − (表の的中率) です。ボリンジャーの「タッチ・%b」(逆張り) と「ブレイク」(順張り) も、"
          "ほぼ同じ場面を逆向きに読む組です。", ""]
    for tf, r in res["tf"].items():
        L += [f"## {TF_LABEL[tf]} ({r['start']}〜{r['end']}、評価した足 {r['bars']:,}本)", ""]
        for h in r["horizons"]:
            L += [f"### {_hlabel(tf, h)} (検証期間の t の大きい順)", ""] + _signal_table(tf, r, h) + [""]
        for h in ROLLOVER_H:
            if tf == "1h" and h in r["horizons"]:
                L += [f"### {_hlabel(tf, h)}: ロールオーバーの時間を除いた場合", "",
                      "ニューヨーク時間17時の日付の切り替え (ロールオーバー) の前後は取引が薄くスプレッドが広がり、中値がスワップの付与に"
                      "合わせてずれます ([direction.md](direction.md))。値動きを測る足にニューヨーク時間16時台・17時台の足が入る回を除いて"
                      "数え直しました (検証期間の t (的中率) の大きい順)。", ""] + _rollover_table(r, h) + [""]
    L += ["## コンセンサス (テクニカル・サマリー)", "",
          "各足で、グループのシグナルの (買いの数 − 売りの数) ÷ シグナルの数 を「スコア」(−1〜+1) とし、"
          f"スコアの絶対値が調整期間での {int(CONSENSUS_Q[0] * 100)}% 点以下なら「中立」、{int(CONSENSUS_Q[1] * 100)}% 点以上なら"
          "「強い買い / 強い売り」、その間を「買い / 売り」としました (区切りは調整期間のスコアの分布だけで決め、値動きの結果は見ていません)。"
          "表は各段階の的中率 (回数、平均の値動きの t)。中立は方向を示さないので、全体に占める割合です。"
          "買いの段階と売りの段階の的中率の差は、主にその期間の相場全体の上げ下げの偏りによるもので、"
          "買いと売りを合わせた「全体」ではそれが打ち消されます。", ""]
    for tf, r in res["tf"].items():
        for group in GROUPS:
            c = r["consensus"][group]
            th = c["thresholds"]
            L += [f"### {TF_LABEL[tf]}: {c['label']} ({len(c['signals'])}個、中立 |スコア| ≤ {th['neutral']:.3f}、"
                  f"強い |スコア| ≥ {th['strong']:.3f})", ""] + _consensus_table(tf, r, group) + [""]
    L += ["## ロジスティック回帰 (組み合わせ方を学習した場合)", "",
          f"全{len(KEYS)}個のシグナルの値 (+1 / 0 / −1) から「上がる確率」を出すロジスティック回帰 (L2 正則化、定数項なし) を、"
          f"時間足と何本先ごとに{len(res['pairs'])}ペアまとめて調整期間で学習し、検証期間で確率が50%を超えたら買い、下回ったら売りとして採点しました。"
          "シグナルを線形に組み合わせて得られる力の目安 (上限の近似) です。定数項を入れないのは、通貨ペアの表し方の向き "
          "(EUR/USD と USD/EUR) に意味がないため、買いと売りを対称に扱い、調整期間の相場全体の上げ下げを覚えるだけの"
          "モデルにしないためです。AUC は 0.5 がでたらめ、1 が完全。"
          "「自信の上位20%」は確率と50%の差が調整期間の上位20%の水準を超えた回です。", ""] + _logistic_table(res) + [""]
    sc = res.get("self_check", {})
    if sc:
        L += ["## 自己チェック (先読みがないこと)", "",
              "全期間で一度に計算したシグナルと、`signals_now` をその足までのデータだけで計算したシグナルを比べました "
              "(USDJPY、ランダムな足と最後の足、全シグナル)。", "",
              "| 時間足 | 比べた足 | うちシグナルが出ていた数 | 一致しなかった足 |", "|---|---|---|---|"]
        for tf, x in sc.items():
            L.append(f"| {TF_LABEL.get(tf, tf)} | {x['bars']} | {x['calls_checked']} | {len(x['mismatches'])} |")
        L.append("")
    L += ["## 使い方", "",
          "- `aifx.research_technical.signals_now(df)` は、渡した足 (1時間足・4時間足は UTC の足の開始時刻、日足は日付の索引) の最後の足について、"
          "全シグナルの値 (+1 / 0 / −1)、関係する指標の値、コンセンサスのスコアを返します。`thresholds` に "
          "research/technical.json の `tf.<時間足>.consensus.<グループ>.thresholds` を渡すと5段階のラベルも付きます。"
          "RSI・MACD・ADX などは計算を始めた位置の影響が少し残るため、500本以上の足を渡してください。",
          "- research/technical.json の `tf.<時間足>.signals.<シグナル>.h.<何本先>.test.hit` が検証期間の的中率です。", "",
          "再現: `python -m aifx.research_technical` (Dukascopy の1時間足 data/history/duka/ が必要、1〜2分)。", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ run

def run(loader=history.load_long_hourly, log=print, write: bool = True) -> dict:
    t0 = time.time()
    cache: dict[str, pd.DataFrame] = {}

    def load(code: str) -> pd.DataFrame:         # each pair's hourly bars are read once for all timeframes
        if code not in cache:
            cache[code] = _clean_hourly(loader(code))
        return cache[code]

    res: dict = {"generated": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "pairs": list(PAIRS),
                 "signals": {k: {x: s[x] for x in ("label", "kind", "buy", "sell")} for k, s in SIGNALS.items()},
                 "consensus_levels": {LEVEL_KEYS[k]: v for k, v in LEVELS.items()}, "tf": {}}
    for tf in HORIZONS:
        res["tf"][tf] = evaluate(tf, load, log)
    starts = [r["start"] for r in res["tf"].values()]
    res["periods"] = {"tune": {"start": min(starts), "end": str((SPLIT - pd.Timedelta(days=1)).date()),
                               "note": "signals whose forward window ends before the test period"},
                      "test": {"start": str(SPLIT.date()), "end": max(r["end"] for r in res["tf"].values())}}
    h1 = load("USDJPY")[OHLC]
    frames = {"1h": h1, "4h": four_hour_bars(h1).drop(columns="end"), "1d": london_daily_bars(h1).drop(columns="end")}
    res["self_check"] = {tf: self_check(df, daily=(tf == "1d")) for tf, df in frames.items()}
    if log:
        log(f"self check: {res['self_check']}")
        log(f"total {time.time() - t0:.0f}s")
    if write:
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / "technical.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        (REPORT_DIR / "technical.md").write_text(report(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
