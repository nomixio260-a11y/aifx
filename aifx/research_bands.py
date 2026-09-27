"""Forecast ranges on ~20 years of hourly bars: the band shape and how the live scale is learned.

A forecast's spread is sigma = raw sigma (the variance model, engine.sigma_steps)
x k, where k is learned per horizon from scored outcomes so that the 80 % band
holds about 80 % of them (learning.py); the 50 % and 95 % bands follow a
Student-t shape with ``engine.BAND_NU`` degrees of freedom. Those degrees of
freedom were fitted on ~2.8 years of hourly bars. This study refits them on
Dukascopy hourly bid-ask mid bars from 2003 (7 pairs) and checks them on
Yahoo's bars (the live source).

Hourly forecasts (1, 4, 24 hours): an origin at every hourly bar once 6,500
bars exist (mid-2004 on), with the variance path of ``engine.sigma_steps`` for
"1h" from the 6,500 bars ending at the origin (``engine.vol_bars``, no
events). The path is recomputed here in vectorised form (window sums per New
York weekday-and-hour slot, the EWMA per slot) and checked against
``sigma_steps`` itself on random origins. The outcome is the close of the last
bar that ended by the target (engine.target_times), as the server scores it.

Daily forecasts (1, 5, 10, 20 business days): an origin at the end of every
London business day from 2005, London-day bars built from the hourly bars as
``data.london_days`` builds them, and the variance of ``engine.sigma_steps``
for "1d" (realised variance from the hourly bars that had ended by the origin).
On Yahoo the stored daily bars are used, as the server does.

The live scale k is replayed through time. At each origin it comes from the
forecasts (7 pairs pooled) whose target had passed, by the rule of
``learning.learn_arrays``: the recency-weighted 80 % quantile of |move| / raw
sigma over 1.2816 (half-life ``Timeframe.half_life`` scored forecasts),
blended with a walk-forward prior of weight PRIOR_N_K and clipped to K_BOUNDS.
The weighted quantile is kept on a fine log grid (k within ~0.2 %, checked
against learn_arrays). The centre is "no change" (the live gain stays near 0;
the time-of-day drift is left out). The prior is refreshed every UTC day: the
unweighted 80 % quantile over the forecasts of the ~12 weeks (hourly) / ~320
business days (daily) before, like ``engine.backtest_prior``.

Scores of z = move / (raw sigma x k): coverage of the 50/80/95 % bands and the
average log score (log density) under Student-t shapes with nu in NUS, or
normal. The shape is chosen per horizon on the tune period (targets before
2017) by log score, confirmed on 2017- and on Yahoo's bars (first 60 % / last
40 % of their dates). A few pre-declared alternatives (the k half-life, the
variance model's EWMA and range scaling) are scored the same way.

A change is recommended only if, on Dukascopy, it raises the log score and
brings the 50/80/95 % coverage closer to nominal on both the tune and the test
period, and does not lower the log score on Yahoo's bars (``decide``). A
setting shared by all horizons must pass on every horizon. Choosing the k
half-life per horizon was added after seeing the results and is reported
separately.

    python -m aifx.research_bands     # writes research/bands.md and research/bands.json
"""

from __future__ import annotations

import json
import math
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .data import OHLC, PAIRS, london_days
from .engine import (BAND_Z, HOURLY_PROFILE_WINDOW, MODEL_KEYS, TIMEFRAMES, band_z, dist_pdf, sigma_steps,
                     target_times, vol_bars)
from .learning import K_BOUNDS, PRIOR_N_K, learn_arrays
from .stats import diebold_mariano
from .timeutil import LONDON, NEW_YORK, add_business_days, london_day_end, market_open_mask
from .volatility import (DAILY_LAM, DAILY_REVERSION, DAILY_RV_LAM, RANGE_WINDOW, RV_MIN_DAYS, WEEKEND_GAP_HOURS,
                         daily_step_variance, daily_variance_inputs, hourly_variance_path, range_variance,
                         scale_proxy)

# the live degrees of freedom when this study was made (engine.BAND_NU has since taken the daily 1 and
# 20 day choices)
BAND_NU = {"15m": {1: 5, 4: 5, 16: 4}, "1h": {1: 5, 4: 5, 24: 6}, "1d": {1: 10, 5: 10, 10: 15, 20: 30}}
REPORT_DIR = Path("research")
TF_H, TF_D = TIMEFRAMES["1h"], TIMEFRAMES["1d"]
H_H, H_D = TF_H.horizons, TF_D.horizons          # (1, 4, 24) hours, (1, 5, 10, 20) business days
BP = 1e4
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
NUS = (3, 4, 5, 6, 8, 10, 15, 30, None)            # None = normal
LEVELS = ("50", "80", "95")
NOMINAL = {"50": 0.50, "80": 0.80, "95": 0.95}
SPLIT = pd.Timestamp("2017-01-01", tz="UTC")       # tune: targets before it; test: origins from it
TAIL = vol_bars(TF_H)                              # 6,500 hourly bars handed to sigma_steps
LONG_WINDOW = 1500                                 # intraday_variance_path's long-run window (its default)
H_LAM, H_REV = 0.97, 0.985                         # hourly_variance_path's defaults, which sigma_steps uses
SHRINK = 20.0                                      # slot_profile's pull of each weekday slot toward its hour
H_START = pd.Timestamp("2004-01-01", tz="UTC")
D_START = pd.Timestamp("2005-01-01")
D_MIN_BARS = 500                                   # daily bars before the first origin (the long-run window)
D_BARS = vol_bars(TF_D)                            # 1,000 daily bars handed to sigma_steps
HOURLY_SLICE = pd.Timedelta(days=420)              # hourly bars handed to the daily variance (> RV_WINDOW days)
YAHOO_D_START = pd.Timestamp("2024-03-01")         # Yahoo's hourly bars start 2023-12-08
YAHOO_TUNE_SHARE = 0.6
WARMUP = {"1h": pd.Timedelta(days=30), "1d": pd.Timedelta(days=120)}   # k replay run-in, not scored
PRIOR_SPAN = {"1h": pd.Timedelta(days=85), "1d": pd.Timedelta(days=448)}
N_CHECK = 25                                       # origins per pair and source checked against the server
WORKERS = 2

# Pre-declared alternatives (nothing else was tried).
HL_MULT = (0.5, 2.0, 4.0)                          # k half-life x this (live: 400 hourly / 120 daily forecasts)
H_VARIANTS = {                                     # hourly: (EWMA decay, bars that scale the high-low range)
    "base": (H_LAM, RANGE_WINDOW),
    "lam95": (0.95, RANGE_WINDOW),
    "lam985": (0.985, RANGE_WINDOW),
    "rw6000": (H_LAM, 6000),
}
D_VARIANTS = {                                     # daily: (EWMA decay on realised variance, reversion per day)
    "base": (DAILY_RV_LAM, DAILY_REVERSION),
    "lam70": (0.70, DAILY_REVERSION),
    "lam90": (0.90, DAILY_REVERSION),
    "rev85": (DAILY_RV_LAM, 0.85),
    "rev95": (DAILY_RV_LAM, 0.95),
}
VARIANT_LABELS = {
    "base": "現在の設定",
    "lam95": "EWMA の減衰 0.97→0.95 (速く反応)",
    "lam985": "EWMA の減衰 0.97→0.985 (ゆっくり反応)",
    "rw6000": "高値・安値の幅を値動きの2乗に合わせる期間 1,500→6,000本",
    "lam70": "実現分散の EWMA の減衰 0.80→0.70 (速く反応)",
    "lam90": "実現分散の EWMA の減衰 0.80→0.90 (ゆっくり反応)",
    "rev85": "平常水準へ戻る速さ 0.90→0.85/日 (速く戻る)",
    "rev95": "平常水準へ戻る速さ 0.90→0.95/日 (ゆっくり戻る)",
    "hl0.5": "k の半減期 ×0.5 (速く追従)",
    "hl2.0": "k の半減期 ×2",
    "hl4.0": "k の半減期 ×4 (ゆっくり追従)",
}
# weighted quantile grid of |z| for the k replay
ZBINS = 6000
EPS = 1e-9                                         # differences smaller than this are ties
ZLOG = (math.log(1e-4), math.log(1e2))


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ data

def _clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicates, bad prices and bars that start while the market is shut dropped."""
    df = df[~df.index.duplicated()].sort_index()
    df = df[(df[OHLC] > 0).all(axis=1)]
    return df[market_open_mask(df.index)]


def load_hourly(code: str, source: str) -> pd.DataFrame:
    raw = history.load_long_hourly(code) if source == "duka" else history.load_hourly(code)
    return _clean_hourly(raw[OHLC].astype(float))


def london_day_bars(h: pd.DataFrame) -> pd.DataFrame:
    """Daily bars of the London business day built from hourly bars by ``data.london_days``: every
    weekday with hourly bars (the Sunday evening counts towards Monday), rebuilt from its hourly bars
    where they cover it and otherwise closed at the next day's opening price."""
    local = (h.index + pd.Timedelta(minutes=59)).tz_convert(LONDON).tz_localize(None).normalize()
    wd = local.dayofweek
    key = local + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")
    g = h.groupby(np.asarray(key)).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                       close=("close", "last"))
    g.index = pd.DatetimeIndex(g.index, name="date")
    return london_days(g[g.index.dayofweek < 5], h)


# ------------------------------------------------------------- hourly: variance

def _open_grid(first: pd.Timestamp, last: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Start times (ns) of all open-market hours from ``first`` to 40 days after ``last``, and their
    New York weekday-and-hour slots."""
    grid = pd.date_range(first.floor("h"), last + pd.Timedelta(days=40), freq="h", tz="UTC")
    grid = grid[market_open_mask(grid)]
    ny = grid.tz_convert(NEW_YORK)
    return grid.as_unit("ns").asi8, ny.dayofweek.to_numpy() * 24 + ny.hour.to_numpy()


class HourlyPaths:
    """``engine.sigma_steps`` for hourly bars at many origins of one pair, vectorised.

    For the bar ``o`` (origin = its end) sigma_steps takes the 6,500 bars up to it:
    each bar's variance is its Parkinson range scaled to squared returns over the
    last RANGE_WINDOW bars (``scale_proxy``, squared return where the range is
    unusable), divided by a New York weekday-and-hour profile measured over the
    last 6,000 bars (``slot_profile``); the EWMA of that and its mean over the
    last 1,500 bars give the level, which reverts per hour and is multiplied back
    by the profile of each coming open-market hour, plus the weekend gap.
    The profile and the EWMA depend on the origin only through window sums and
    decayed sums per slot of three per-bar series (range, squared return where the
    range is usable, squared return where it is not), which is what is kept here.
    (The EWMA here runs from the first bar; the server's starts 6,480 bars back, a
    difference of 0.985^6480 at most.)
    """

    def __init__(self, bars: pd.DataFrame):
        self.bars = bars
        self.start = bars.index.as_unit("ns").asi8
        self.end = self.start + HOUR_NS
        self.y = np.log(bars["close"].to_numpy(float))
        n = len(self.y)
        r2 = np.r_[0.0, np.diff(self.y) ** 2]
        alt = np.r_[np.nan, range_variance(bars)]
        ok = np.isfinite(alt)
        self.X = np.stack([np.where(ok, alt, 0.0), np.where(ok, r2, 0.0), np.where(ok, 0.0, r2)])  # A, D, B
        self.cum = np.concatenate([np.zeros((4, 1)), np.cumsum(np.vstack([self.X, ok[None, :]]), axis=1)], axis=1)
        ny = bars.index.tz_convert(NEW_YORK)
        self.key = ny.dayofweek.to_numpy() * 24 + ny.hour.to_numpy()
        self.pos, self.kcum = [], []
        for k in range(168):
            p = np.flatnonzero(self.key == k)
            p = p[p >= 1]
            self.pos.append(p)
            self.kcum.append(np.concatenate([np.zeros((3, 1)), np.cumsum(self.X[:, p], axis=1)], axis=1))
        self.grid, self.gkey = _open_grid(bars.index[0], bars.index[-1])
        self.lams = np.array(sorted({v[0] for v in H_VARIANTS.values()}))
        self._E = np.zeros((len(self.lams), 3, 168))
        self._next = 1
        self.n = n

    def _scale(self, org: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
        """scale_proxy's factor over the last ``W`` bars, and whether it applies (enough usable ranges)."""
        c = self.cum
        s = [c[q, org + 1] - c[q, org + 1 - W] for q in range(4)]
        return s[1] / np.where(s[0] > 0, s[0], np.nan), s[3] >= RV_MIN_DAYS

    def _key_sums(self, org: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
        """Sums of A, D, B (3, n, 168) and counts (n, 168) per slot over the last ``W`` bars of each origin."""
        S = np.zeros((3, len(org), 168))
        N = np.zeros((len(org), 168))
        for k in range(168):
            p = self.pos[k]
            hi = np.searchsorted(p, org, "right")
            lo = np.searchsorted(p, org - W, "right")
            N[:, k] = hi - lo
            S[:, :, k] = self.kcum[k][:, hi] - self.kcum[k][:, lo]
        return S, N

    @staticmethod
    def _profile(S: np.ndarray, N: np.ndarray) -> np.ndarray:
        """slot_profile(weekday=True) from per-slot sums of the variance proxy and counts."""
        s_tod = S.reshape(-1, 7, 24).sum(axis=1)
        n_tod = N.reshape(-1, 7, 24).sum(axis=1)
        prof = np.where(n_tod >= 5, s_tod / np.maximum(n_tod, 1), 1.0)
        mean_all = S.sum(axis=1) / N.sum(axis=1)
        base = prof / mean_all[:, None]
        out = (S / mean_all[:, None] + SHRINK * np.tile(base, (1, 7))) / (N + SHRINK)
        mean = (out * N).sum(axis=1) / np.maximum(N.sum(axis=1), 1)
        return out / mean[:, None]

    def _ewma_states(self, a: int, b: int) -> np.ndarray:
        """Decayed per-slot sums of A, D, B for every EWMA decay, at origins a..b-1 (origins in order)."""
        rec = np.empty((b - a, len(self.lams), 3, 168))
        E, lv, X, key = self._E, self.lams[:, None, None], self.X, self.key
        for i in range(self._next, b):
            E *= lv
            E[:, :, key[i]] += X[:, i]
            if i >= a:
                rec[i - a] = E
        self._next = b
        return rec

    def compute(self, a: int, b: int) -> dict:
        """Cumulative variance (bp^2) at the horizons for every variant, and the outcomes, for origins a..b-1."""
        org = np.arange(a, b)
        rec = self._ewma_states(a, b)
        S6, N6 = self._key_sums(org, HOURLY_PROFILE_WINDOW)
        S15, _ = self._key_sums(org, LONG_WINDOW)
        T0 = self.end[org]
        j0 = np.searchsorted(self.grid, T0, "left")
        J = j0[:, None] + np.arange(max(H_H))
        starts = self.grid[J]
        prev = np.concatenate([T0[:, None], starts[:, :-1] + HOUR_NS], axis=1)
        gap = (starts > prev).astype(float)
        steps = np.arange(1, max(H_H) + 1)
        cols = [h - 1 for h in H_H]
        out = {"o": org, "t0": T0}
        by_window = {}
        for name, (lam, rw) in H_VARIANTS.items():
            if rw not in by_window:
                c, proxy = self._scale(org, rw)
                cc, px = np.nan_to_num(c)[:, None], proxy[:, None]
                prof = self._profile(np.where(px, cc * S6[0] + S6[2], S6[1] + S6[2]), N6)
                long_var = (np.where(px, cc * S15[0] + S15[2], S15[1] + S15[2]) / prof).sum(axis=1) / LONG_WINDOW
                by_window[rw] = (cc, px, prof, long_var)
            cc, px, prof, long_var = by_window[rw]
            li = int(np.flatnonzero(self.lams == lam)[0])
            R = rec[:, li]
            ewma = (1 - lam) * (np.where(px, cc * R[:, 0] + R[:, 2], R[:, 1] + R[:, 2]) / prof).sum(axis=1)
            fut = np.take_along_axis(prof, self.gkey[J], axis=1)
            var = ((long_var[:, None] + (ewma - long_var)[:, None] * H_REV ** steps) * fut
                   + gap * WEEKEND_GAP_HOURS * long_var[:, None])
            out[name] = np.cumsum(var, axis=1)[:, cols] * BP * BP
        tt = self.grid[j0[:, None] + np.array(H_H) - 1] + HOUR_NS
        ai = np.searchsorted(self.end, tt, "right") - 1
        out["tt"] = tt
        out["valid"] = (ai > org[:, None]) & (tt <= self.end[-1])
        out["a"] = (self.y[ai] - self.y[org][:, None]) * BP
        return out

    def direct(self, o: int) -> dict:
        """The same numbers from the server's own functions (sigma_steps for the current settings)."""
        tail = self.bars.iloc[o + 1 - TAIL: o + 1]
        origin = (tail.index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
        cols = [h - 1 for h in H_H]
        res = {"base": np.cumsum(sigma_steps(TF_H, tail, origin, max(H_H)))[cols]}
        yt = np.log(tail["close"].to_numpy(float))
        for name, (lam, rw) in H_VARIANTS.items():
            if name == "base":
                continue
            sq = scale_proxy(range_variance(tail), np.diff(yt), rw)
            var, _ = hourly_variance_path(list(tail.index.to_pydatetime()), yt, origin, max(H_H), None, lam=lam,
                                          sq=sq, profile_window=HOURLY_PROFILE_WINDOW, profile_tz=NEW_YORK,
                                          profile_weekday=True)
            res[name] = np.cumsum(var)[cols] * BP * BP
        res["tt"] = np.array([pd.Timestamp(t).value for t in target_times(TF_H, origin)])
        return res


def hourly_pair(args: tuple[str, str]) -> dict:
    """Every hourly origin of one pair and source: raw variance per variant, outcomes, and the check
    against the server's functions on random origins."""
    code, source = args
    bars = load_hourly(code, source)
    hp = HourlyPaths(bars)
    first = TAIL - 1
    if source == "duka":
        first = max(first, int(np.searchsorted(hp.start, H_START.value)))
    parts = []
    for a in range(first, hp.n, 8000):
        parts.append(hp.compute(a, min(a + 8000, hp.n)))
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # check against the server's functions
    rng = np.random.default_rng(zlib.crc32(f"{code}{source}".encode()))
    idx = rng.choice(len(res["o"]), size=min(N_CHECK, len(res["o"])), replace=False)
    worst = {name: 0.0 for name in H_VARIANTS}
    t_bad = 0
    for i in idx:
        d = hp.direct(int(res["o"][i]))
        for name in H_VARIANTS:
            worst[name] = max(worst[name], float(np.max(np.abs(res[name][i] / d[name] - 1))))
        t_bad += int(np.any(d["tt"] != res["tt"][i]))
    keep = ["t0", "tt", "valid", "a", *H_VARIANTS]
    return {"pair": code, "n_bars": hp.n, **{k: res[k] for k in keep},
            "check": {"n": len(idx), "max_rel_diff": worst, "target_mismatch": t_bad}}


# -------------------------------------------------------------- daily: variance

def daily_pair(args: tuple[str, str]) -> dict:
    """Every daily origin (end of a London business day) of one pair: raw variance per variant from the
    server's functions, outcomes on the hourly bars, and a check against sigma_steps with all hourly bars."""
    code, source = args
    h = load_hourly(code, source)
    if source == "duka":
        D, start = london_day_bars(h), D_START
    else:
        raw = history.load_daily(code)
        D, start = raw[~raw.index.duplicated()].sort_index(), YAHOO_D_START
    h_start = h.index.as_unit("ns").asi8
    h_end = h_start + HOUR_NS
    hy = np.log(h["close"].to_numpy(float))
    close = D["close"].to_numpy(float)
    steps, cols = max(H_D), [x - 1 for x in H_D]
    rows = []
    checks = []
    rng = np.random.default_rng(zlib.crc32(f"{code}{source}d".encode()))
    for i, day in enumerate(D.index):
        if day < start or i < D_MIN_BARS:
            continue
        origin = london_day_end(day.date())
        o_ns = pd.Timestamp(origin).value
        if o_ns > h_end[-1]:
            break
        hi = int(np.searchsorted(h_end, o_ns, "right"))
        if hi == 0:
            continue
        lo = int(np.searchsorted(h_start, o_ns - HOURLY_SLICE.value, "left"))
        bars = D.iloc[max(0, i + 1 - D_BARS): i + 1]
        y = np.log(close[max(0, i + 1 - D_BARS): i + 1])
        sq, lam = daily_variance_inputs(bars, h.iloc[lo:hi], origin)
        row = {"t0": o_ns, "day": day}
        for name, (rv_lam, rev) in D_VARIANTS.items():
            var = daily_step_variance(y, steps, rv_lam if sq is not None else DAILY_LAM, sq=sq, reversion=rev)
            row[name] = np.cumsum(var)[cols] * BP * BP
        tt = np.array([pd.Timestamp(london_day_end(add_business_days(day.date(), x))).value for x in H_D])
        ai = np.searchsorted(h_end, tt, "right") - 1
        row["tt"] = tt
        row["valid"] = (ai >= hi) & (tt <= h_end[-1])
        row["a"] = (hy[ai] - hy[hi - 1]) * BP
        row["rv"] = sq is not None
        rows.append(row)
        if rng.random() < 0.006 and len(checks) < N_CHECK:
            full = sigma_steps(TF_D, bars, origin, steps, None, hourly=h.iloc[:hi])
            tts = np.array([pd.Timestamp(t).value for t in target_times(TF_D, origin)])
            checks.append((float(np.max(np.abs(np.cumsum(full)[cols] / row["base"] - 1))), bool(np.any(tts != tt))))
    out = {"pair": code, "t0": np.array([r["t0"] for r in rows]), "tt": np.stack([r["tt"] for r in rows]),
           "valid": np.stack([r["valid"] for r in rows]), "a": np.stack([r["a"] for r in rows]),
           "rv": np.array([r["rv"] for r in rows])}
    for name in D_VARIANTS:
        out[name] = np.stack([r[name] for r in rows])
    out["check"] = {"n": len(checks), "max_rel_diff": max((c[0] for c in checks), default=None),
                    "target_mismatch": sum(c[1] for c in checks)}
    return out


# ------------------------------------------------------------------ k replay

_ZC = np.exp(ZLOG[0] + (np.arange(ZBINS) + 0.5) * (ZLOG[1] - ZLOG[0]) / ZBINS)


def _zbin(ze: np.ndarray) -> np.ndarray:
    u = (np.log(np.maximum(ze, 1e-300)) - ZLOG[0]) / (ZLOG[1] - ZLOG[0]) * ZBINS
    return np.clip(u.astype(np.int64), 0, ZBINS - 1)


def prior_k(t0: np.ndarray, tt: np.ndarray, ze: np.ndarray, anchors: np.ndarray, span_ns: int) -> np.ndarray:
    """The walk-forward prior's k at each anchor (a UTC midnight): the 80 % quantile of |z| over the
    forecasts made in the ``span_ns`` before it whose outcome was known by then (1.0 with fewer than 50)."""
    order = np.argsort(t0, kind="stable")
    t0s, tts, zs = t0[order], tt[order], ze[order]
    lo = np.searchsorted(t0s, anchors - span_ns, "left")
    hi = np.searchsorted(t0s, anchors, "right")
    out = np.ones(len(anchors))
    for u, (a, b) in enumerate(zip(lo, hi)):
        sel = zs[a:b][tts[a:b] <= anchors[u]]
        if len(sel) >= 50:
            out[u] = float(np.quantile(sel, 0.8)) / BAND_Z["80"]
    return out


def replay_k(t0: np.ndarray, tt: np.ndarray, pair: np.ndarray, ze: np.ndarray, half_life: float,
             prior: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The k the live learner (learning.learn_arrays) would have given each forecast.

    At an origin the known forecasts are those whose target had passed, in the
    order the ledger gives them (target, origin, pair); their weights halve every
    ``half_life`` forecasts back from the newest. ``prior``: (UTC midnights, prior
    k from that midnight on)."""
    order = np.lexsort((pair, t0, tt))
    tts = tt[order]
    bins = _zbin(ze[order])
    T, inv = np.unique(t0, return_inverse=True)
    j_at = np.searchsorted(tts, T, "right")
    anchors, pk = prior
    p_at = pk[np.clip(np.searchsorted(anchors, T, "right") - 1, 0, len(pk) - 1)]
    d = 0.5 ** (1.0 / half_life)
    hist = np.zeros(ZBINS)
    w_sum, j, k_hat = 0.0, 0, None
    k_T = np.empty(len(T))
    for u in range(len(T)):
        j1 = int(j_at[u])
        if j1 > j:
            m = j1 - j
            w = d ** np.arange(m - 1, -1, -1.0)
            hist *= d ** m
            np.add.at(hist, bins[j:j1], w)
            w_sum = w_sum * d ** m + float(w.sum())
            j = j1
            cum = np.cumsum(hist)
            k_hat = _ZC[min(int(np.searchsorted(cum, 0.8 * cum[-1])), ZBINS - 1)] / BAND_Z["80"]
        k = p_at[u] if k_hat is None else (PRIOR_N_K * p_at[u] + w_sum * k_hat) / (PRIOR_N_K + w_sum)
        k_T[u] = round(min(max(k, K_BOUNDS[0]), K_BOUNDS[1]), 6)
    return k_T[inv]


def check_replay(t0, tt, pair, ze, half_life, prior, k, n: int = 12, seed: int = 0) -> float:
    """Largest |k - learn_arrays' k| over ``n`` random origins (learn_arrays on all forecasts known then)."""
    order = np.lexsort((pair, t0, tt))
    tts, zs = tt[order], ze[order]
    anchors, pk = prior
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in rng.choice(len(t0), size=n, replace=False):
        j = int(np.searchsorted(tts, t0[i], "right"))
        p = float(pk[max(int(np.searchsorted(anchors, t0[i], "right")) - 1, 0)])
        z = np.zeros(j)
        st = learn_arrays({"mse_z": [1.0] * len(MODEL_KEYS), "k": p, "suu": 0.0, "suv": 0.0},
                          np.zeros((j, len(MODEL_KEYS))), zs[:j],
                          np.ones(j), np.ones(j), z, z, z, z, half_life)
        worst = max(worst, abs(st.k - float(k[i])))
    return worst


# ------------------------------------------------------------------ scoring

NU_KEYS = tuple("normal" if nu is None else str(nu) for nu in NUS)


def _nu_key(nu: int | None) -> str:
    return "normal" if nu is None else str(nu)


def _nu_of(key: str) -> int | None:
    return None if key == "normal" else int(key)


def log_pdf(z: np.ndarray, nu: int | None) -> np.ndarray:
    """log of engine.dist_pdf (per unit of sigma), without underflow far in the tails."""
    z = np.asarray(z, dtype=float)
    if nu is None:
        return -0.5 * z * z - 0.5 * math.log(2 * math.pi)
    at0 = float(dist_pdf(np.zeros(1), nu)[0])                # const / scale
    const = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi)
    scale = const / at0
    return math.log(at0) - (nu + 1) / 2 * np.log1p((z / scale) ** 2 / nu)


def _cover_dist(cover: dict) -> float:
    """Mean distance of the 50/80/95 % coverage from nominal."""
    return float(np.mean([abs(cover[lv] - NOMINAL[lv]) for lv in LEVELS]))


def _scores(z: np.ndarray, log_sk: np.ndarray, masks: dict[str, np.ndarray], lp: dict) -> dict:
    """Coverage and mean log score per shape and period. ``ls`` is the log density of z (per sigma);
    ``jac`` the mean of -log(sigma in bp), so ls + jac is the log score of the move itself."""
    out = {}
    for p, m in masks.items():
        n = int(m.sum())
        if not n:
            out[p] = {"n": 0}
            continue
        az = np.abs(z[m])
        out[p] = {"n": n, "jac": float(-log_sk[m].mean()), "nu": {}}
        for nu, nk in zip(NUS, NU_KEYS):
            bz = band_z(nu)
            out[p]["nu"][nk] = {"ls": float(lp[nk][m].mean()),
                                "cover": {lv: float(np.mean(az <= bz[lv])) for lv in LEVELS}}
    return out


def _dm(loss_a: np.ndarray, loss_b: np.ndarray, day: np.ndarray, mask: np.ndarray, lag: int) -> dict:
    """Mean loss difference (a - b) and a Diebold-Mariano p value on daily means (pairs and origins of a
    day averaged; Newey-West with ``lag`` days for overlapping horizons)."""
    d = pd.Series((loss_a - loss_b)[mask]).groupby(day[mask]).mean().to_numpy()
    stat, p = diebold_mariano(d, np.zeros(len(d)), lag)
    return {"diff": float(np.mean((loss_a - loss_b)[mask])), "p": p, "days": len(d)}


def _periods(t0: np.ndarray, tt: np.ndarray, source: str, warm: pd.Timedelta) -> tuple[dict, dict]:
    """Masks of the scored periods, and their date ranges."""
    ok = t0 >= t0.min() + warm.value
    day = (t0 - HOUR_NS) // DAY_NS              # the origin's (UTC) day; daily origins are London midnights
    if source == "duka":
        cut = SPLIT.value
        masks = {"tune": ok & (tt < cut), "test": ok & (t0 >= cut)}
    else:
        days = np.unique(day[ok])
        cut_day = days[int(len(days) * YAHOO_TUNE_SHARE)]
        cut = int(cut_day * DAY_NS)
        masks = {"tune": ok & (day < cut_day) & (tt < cut), "test": ok & (day >= cut_day), "all": ok}
    rng = {p: [str(pd.Timestamp(int(t0[m].min()), tz="UTC").date()), str(pd.Timestamp(int(t0[m].max()), tz="UTC").date())]
           for p, m in masks.items() if m.any()}
    return masks, rng


def pooled(parts: list[dict], names) -> dict:
    """Concatenate the per-pair results (pair as an integer id)."""
    out = {"pair": np.concatenate([np.full(len(p["t0"]), i) for i, p in enumerate(parts)])}
    for key in ("t0", "tt", "valid", "a", *names):
        out[key] = np.concatenate([p[key] for p in parts])
    return out


def horizon_arrays(P: dict, tf_key: str, j: int) -> dict:
    """The scored forecasts of horizon index ``j``: origin, target, pair, move (bp) and raw variance per variant."""
    v = P["valid"][:, j]
    names = H_VARIANTS if tf_key == "1h" else D_VARIANTS
    return {"t0": P["t0"][v], "tt": P["tt"][v, j], "pair": P["pair"][v], "a": P["a"][v, j],
            **{name: P[name][v, j] for name in names}}


def analyse_h(args: tuple[str, str, int, dict]) -> tuple[str, str, int, dict, dict]:
    """Replay k and score every variant and shape for one timeframe, source and horizon.
    Returns the results for the JSON and the per-forecast z of the current setup (tables over time)."""
    tf_key, source, h, A = args
    tf = TIMEFRAMES[tf_key]
    var_names = list(H_VARIANTS if tf_key == "1h" else D_VARIANTS)
    variants = var_names + [f"hl{m}" for m in HL_MULT]
    lag = (1 if h < 24 else 2) if tf_key == "1h" else h        # days of overlap between the daily means
    t0, tt, pair, a = A["t0"], A["tt"], A["pair"], A["a"]
    masks, ranges = _periods(t0, tt, source, WARMUP[tf_key])
    day = (t0 - HOUR_NS) // DAY_NS
    anchors = np.unique(day) * DAY_NS
    cur = _nu_key(BAND_NU[tf_key][h])
    hres = {"n": int(len(t0)), "periods": ranges, "variants": {}}
    priors: dict = {}
    keep: dict = {}
    ref_loss = None
    for name in variants:
        vname, mult = (name, 1.0) if name in var_names else ("base", float(name[2:]))
        s = np.sqrt(A[vname])
        ze = np.abs(a) / s
        if vname not in priors:
            priors[vname] = (anchors, prior_k(t0, tt, ze, anchors, PRIOR_SPAN[tf_key].value))
        prior = priors[vname]
        k = replay_k(t0, tt, pair, ze, tf.half_life * mult, prior)
        z = a / (s * k)
        log_sk = np.log(s * k)
        lp = {nk: log_pdf(z, nu) for nu, nk in zip(NUS, NU_KEYS)}
        sc = _scores(z, log_sk, masks, lp)
        best = max(NU_KEYS, key=lambda nk: sc["tune"]["nu"][nk]["ls"])
        entry = {"scores": sc, "best_nu": best, "k_median": {p: float(np.median(k[m])) for p, m in masks.items()}}
        if name == "base":
            entry["replay_check"] = check_replay(t0, tt, pair, ze, tf.half_life, prior, k)
            entry["dm_nu"] = {p: _dm(-lp[best], -lp[cur], day, m, lag) for p, m in masks.items()}
            entry["dm_vs_current"] = {nk: {p: _dm(-lp[nk], -lp[cur], day, m, lag)["p"] for p, m in masks.items()}
                                      for nk in NU_KEYS if nk != cur}
            ref_loss = -(lp[best] - log_sk)
            keep = {"t0": t0, "z": z, "pair": pair, "tune": masks["tune"], "test": masks["test"]}
        else:
            entry["dm_vs_base"] = {p: _dm(-(lp[best] - log_sk), ref_loss, day, m, lag) for p, m in masks.items()}
        hres["variants"][name] = entry
    return tf_key, source, h, hres, keep


def collect(pairs=None, workers: int = WORKERS, t0: float | None = None) -> dict:
    """Per-pair raw variances and outcomes for (timeframe, source), computed in ``workers`` processes."""
    t0 = t0 or time.time()
    pairs = list(pairs or PAIRS)
    jobs = [(fn, (code, src)) for fn in (hourly_pair, daily_pair) for src in ("duka", "yahoo") for code in pairs]
    out: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [(fn, arg, ex.submit(fn, arg)) for fn, arg in jobs]
        for fn, (code, src), fut in futs:
            tf_key = "1h" if fn is hourly_pair else "1d"
            r = fut.result()
            out.setdefault((tf_key, src), []).append(r)
            _log(f"{tf_key} {src} {code}: {len(r['t0']):,} origins, check {r['check']}", t0)
    return out


# ------------------------------------------------------------- over time

def _cover_by(z: np.ndarray, groups: np.ndarray, nus: dict[str, str]) -> dict:
    """Coverage of the 50/80/95 % bands per group, for each named shape."""
    out = {}
    az = np.abs(z)
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = {"n": int(m.sum()), **{name: {lv: float(np.mean(az[m] <= band_z(_nu_of(nk))[lv])) for lv in LEVELS}
                                          for name, nk in nus.items()}}
    return out


def over_time(keep: dict, tf_key: str, source: str, choice: dict) -> dict:
    """Coverage per year (per month over the last 18 months) with the current and the recommended shape, and
    the spread of the 80 % coverage over rolling windows of about 60 days (all pairs pooled)."""
    out = {"year": {}, "month": {}, "rolling": {}}
    for h, kp in keep.items():
        nus = {"cur": _nu_key(BAND_NU[tf_key][h]), "new": choice[h]}
        ts = pd.to_datetime(kp["t0"] - HOUR_NS, utc=True)
        scored = kp["tune"] | kp["test"]
        z, ts = kp["z"][scored], ts[scored]
        out["year"][str(h)] = _cover_by(z, ts.year.to_numpy(), nus)
        month = ts.strftime("%Y-%m").to_numpy()
        last = sorted(set(month))[-18:]
        sel = np.isin(month, last)
        out["month"][str(h)] = _cover_by(z[sel], month[sel], nus)
        # windows of 60 calendar days of origins (the server's "recent" backtest window), every 5th day
        inside = pd.Series((np.abs(z) <= BAND_Z["80"]).astype(float), index=ts.floor("D"))
        by_day = inside.groupby(level=0).agg(["sum", "count"])
        roll = by_day.rolling("60D").sum()
        roll = roll[roll.index >= by_day.index[0] + pd.Timedelta(days=59)].iloc[::5]
        cov = roll["sum"] / roll["count"]
        per = {}
        for p, lo, hi in (("tune", None, SPLIT), ("test", SPLIT, None)) if source == "duka" else (("all", None, None),):
            c = cov
            if lo is not None:
                c = c[c.index >= lo]
            if hi is not None:
                c = c[c.index < hi]
            if not len(c):
                continue
            worst = c.nsmallest(40)
            picked = []
            for t, v in worst.items():                        # the lowest windows at least 120 days apart
                if all(abs((t - q).days) > 120 for q, _ in picked):
                    picked.append((t, v))
                if len(picked) == 5:
                    break
            per[p] = {"n": int(len(c)), "q05": float(c.quantile(0.05)), "q10": float(c.quantile(0.10)),
                      "median": float(c.median()), "le77": float(np.mean(c <= 0.77)),
                      "le69": float(np.mean(c <= 0.69)), "le61": float(np.mean(c <= 0.61)),
                      "worst": [[str(t.date()), float(v)] for t, v in picked]}
        out["rolling"][str(h)] = per
    return out


# ------------------------------------------------------------------ decisions

def _passes(c: dict) -> bool:
    """The adoption rule on the differences (new - reference); exact ties do not count as improvements."""
    return (c["ls_tune"] > EPS and c["ls_test"] > EPS and c["cov_tune"] < -EPS and c["cov_test"] < -EPS
            and c["ls_yahoo"] >= -EPS)


def _compare(dn: dict, dr: dict, yn: dict, yr: dict, nk_new: str, nk_ref: str, full: bool) -> dict:
    """Differences of the log score (with the -log sigma term when ``full``) and of the mean coverage
    distance from nominal: Dukascopy tune and test, Yahoo (all dates)."""
    def ls(sc, nk):
        return sc["nu"][nk]["ls"] + (sc["jac"] if full else 0.0)
    c = {f"ls_{p}": ls(dn[p], nk_new) - ls(dr[p], nk_ref) for p in ("tune", "test")}
    c.update({f"cov_{p}": _cover_dist(dn[p]["nu"][nk_new]["cover"]) - _cover_dist(dr[p]["nu"][nk_ref]["cover"])
              for p in ("tune", "test")})
    c["ls_yahoo"] = ls(yn["all"], nk_new) - ls(yr["all"], nk_ref)
    c["cov_yahoo"] = _cover_dist(yn["all"]["nu"][nk_new]["cover"]) - _cover_dist(yr["all"]["nu"][nk_ref]["cover"])
    return c


def decide(res: dict) -> dict:
    """The adoption rule: a change is recommended only if, on Dukascopy, it raises the log score and
    brings the 50/80/95 % coverage closer to nominal (mean distance of the three) on both the tune and
    the test period, and on Yahoo's bars (all dates) its log score is not lower.

    1. Shape: per horizon, the tune-best nu against the current one.
    2. Alternatives (each at its own tune-best nu) against the reference: the current settings with the
       shape step 1 recommends. A setting shared by all horizons must pass on every horizon.
    3. Secondary: the k half-life chosen per horizon (tune log score among x1, x0.5, x2, x4), against the
       same reference. Using it would need a half-life per horizon in the code."""
    out: dict = {"nu": {}, "variants": {}, "half_life": {}}
    for tf_key in ("1h", "1d"):
        R = res["tf"][tf_key]
        hs = [str(h) for h in TIMEFRAMES[tf_key].horizons]
        out["nu"][tf_key], out["variants"][tf_key], out["half_life"][tf_key] = {}, {}, {}
        for h in hs:
            dv, yv = R["duka"]["h"][h]["variants"]["base"], R["yahoo"]["h"][h]["variants"]["base"]
            cur, best = _nu_key(BAND_NU[tf_key][int(h)]), dv["best_nu"]
            c = _compare(dv["scores"], dv["scores"], yv["scores"], yv["scores"], best, cur, False)
            ok = best != cur and _passes(c)
            out["nu"][tf_key][h] = {"current": cur, "best": best, "adopt": bool(ok), "new": best if ok else cur, **c}
        names = [n for n in R["duka"]["h"][hs[0]]["variants"] if n != "base"]
        for name in names:
            per_h = {}
            for h in hs:
                dv, yv = R["duka"]["h"][h]["variants"], R["yahoo"]["h"][h]["variants"]
                ref, nk = out["nu"][tf_key][h]["new"], dv[name]["best_nu"]
                c = _compare(dv[name]["scores"], dv["base"]["scores"], yv[name]["scores"], yv["base"]["scores"],
                             nk, ref, True)
                per_h[h] = {**c, "ok": _passes(c), "nu": nk}
            out["variants"][tf_key][name] = {"h": per_h, "adopt": all(x["ok"] for x in per_h.values())}
        for h in hs:
            dv = R["duka"]["h"][h]["variants"]
            ref = out["nu"][tf_key][h]["new"]
            tune = {1.0: dv["base"]["scores"]["tune"]["nu"][ref]["ls"] + dv["base"]["scores"]["tune"]["jac"]}
            for m in HL_MULT:
                sc = dv[f"hl{m}"]
                tune[m] = sc["scores"]["tune"]["nu"][sc["best_nu"]]["ls"] + sc["scores"]["tune"]["jac"]
            m = max(tune, key=tune.get)
            entry = {"mult": m, "half_life": TIMEFRAMES[tf_key].half_life * m, "nu": ref, "adopt": False}
            if m != 1.0:
                x = out["variants"][tf_key][f"hl{m}"]["h"][h]
                entry.update({"nu": x["nu"], "adopt": x["ok"], **{k: v for k, v in x.items() if k not in ("ok", "nu")}})
            out["half_life"][tf_key][h] = entry
    return out


# ------------------------------------------------------------------ run

def evaluate(workers: int = WORKERS, pairs=None) -> dict:
    t0 = time.time()
    data = collect(pairs, workers=workers, t0=t0)
    res: dict = {"meta": {}, "checks": {}, "tf": {}, "time": {}}
    jobs = []
    for (tf_key, src), parts in data.items():
        names = list(H_VARIANTS if tf_key == "1h" else D_VARIANTS)
        P = pooled(parts, names)
        res["checks"].setdefault(tf_key, {})[src] = {p["pair"]: p["check"] for p in parts}
        res["tf"].setdefault(tf_key, {})[src] = {"n_origins": int(len(P["t0"])), "h": {},
                                                 "range": [str(pd.Timestamp(int(P["t0"].min()), tz="UTC").date()),
                                                           str(pd.Timestamp(int(P["t0"].max()), tz="UTC").date())]}
        for j, h in enumerate(TIMEFRAMES[tf_key].horizons):
            jobs.append((tf_key, src, h, horizon_arrays(P, tf_key, j)))
    del data
    jobs.sort(key=lambda job: -len(job[3]["t0"]))
    keeps: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for tf_key, src, h, hres, keep in ex.map(analyse_h, jobs):
            res["tf"][tf_key][src]["h"][str(h)] = hres
            keeps.setdefault((tf_key, src), {})[h] = keep
            b = hres["variants"]["base"]
            _log(f"{tf_key} {src} h={h}: {hres['n']:,} forecasts, best nu {b['best_nu']} "
                 f"(current {_nu_key(BAND_NU[tf_key][h])}), replay check {b['replay_check']:.4f}", t0)
    for tf_key in res["tf"]:
        for src in res["tf"][tf_key]:
            res["tf"][tf_key][src]["h"] = {str(h): res["tf"][tf_key][src]["h"][str(h)]
                                           for h in TIMEFRAMES[tf_key].horizons}
    res["decision"] = decide(res)
    for (tf_key, src), keep in keeps.items():
        choice = {h: res["decision"]["nu"][tf_key][str(h)]["new"] for h in keep}
        res["time"].setdefault(tf_key, {})[src] = over_time(dict(sorted(keep.items())), tf_key, src, choice)
    res["meta"] = {
        "pairs": list(pairs or PAIRS), "nus": list(NU_KEYS), "split": str(SPLIT.date()),
        "half_life": {"1h": TF_H.half_life, "1d": TF_D.half_life}, "hl_mult": list(HL_MULT),
        "h_variants": {k: list(v) for k, v in H_VARIANTS.items()}, "d_variants": {k: list(v) for k, v in D_VARIANTS.items()},
        "band_nu": {tf: {str(h): _nu_key(nu) for h, nu in BAND_NU[tf].items()} for tf in ("1h", "1d")},
        "prior_n_k": PRIOR_N_K, "k_bounds": list(K_BOUNDS), "seconds": round(time.time() - t0, 1),
    }
    _log("done", t0)
    return res


def run(workers: int = WORKERS) -> dict:
    res = evaluate(workers)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "bands.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORT_DIR / "bands.md").write_text(report(res), encoding="utf-8")
    return res


TF_LABEL = {"1h": "1時間足", "1d": "日足"}
PERIOD_LABEL = {("duka", "tune"): "Dukascopy 調整 (〜2016)", ("duka", "test"): "Dukascopy 検証 (2017〜)",
                ("yahoo", "tune"): "Yahoo 前半6割", ("yahoo", "test"): "Yahoo 後半4割", ("yahoo", "all"): "Yahoo 全体"}
ROWS = (("duka", "tune"), ("duka", "test"), ("yahoo", "tune"), ("yahoo", "test"), ("yahoo", "all"))


def _hl(tf_key: str, h) -> str:
    return f"{h}時間先" if tf_key == "1h" else f"{h}営業日先"


def _nl(nk: str) -> str:
    return "正規分布" if nk == "normal" else f"ν={nk}"


def _pc(x: float) -> str:
    return f"{x * 100:.1f}%"


def _pv(p) -> str:
    return "—" if p is None else "<0.001" if p < 0.001 else f"{p:.3f}"


def _mil(x: float) -> str:
    """A log score difference per forecast, in thousandths."""
    v = x * 1000
    return f"{v:+.2f}" if abs(v) < 1 else f"{v:+.1f}"


def _nu_dict(tf_key: str, nus: dict) -> str:
    return "{" + ", ".join(f"{h}: {'None' if nk == 'normal' else nk}" for h, nk in nus.items()) + "}"


def _base(res: dict, tf_key: str, src: str, h) -> dict:
    return res["tf"][tf_key][src]["h"][str(h)]["variants"]["base"]


def _new_nu(res: dict) -> dict:
    return {tf: {h: x["new"] for h, x in res["decision"]["nu"][tf].items()} for tf in ("1h", "1d")}


def _nu_rows(res: dict, tf_key: str, pick) -> list[str]:
    """Coverage and log score, current shape -> another (``pick(h)``), for every horizon and period."""
    L = ["| 予測先 | 自由度 | 期間 | 件数 | 50%レンジ | 80%レンジ | 95%レンジ | 対数スコアの差 (×1000) | p |",
         "|---|---|---|---|---|---|---|---|---|"]
    for h in TIMEFRAMES[tf_key].horizons:
        cur, new = _nu_key(BAND_NU[tf_key][h]), pick(h)
        for src, p in ROWS:
            b = _base(res, tf_key, src, h)
            sc = b["scores"][p]
            if not sc.get("n"):
                continue
            a, c = sc["nu"][cur], sc["nu"][new]
            cov = " | ".join(f"{_pc(a['cover'][lv])} → {_pc(c['cover'][lv])}" for lv in LEVELS)
            pval = None if new == cur else b.get("dm_vs_current", {}).get(new, {}).get(p)
            L.append(f"| {_hl(tf_key, h)} | {_nl(cur)} → {_nl(new)} | {PERIOD_LABEL[(src, p)]} | {sc['n']:,} | {cov} | "
                     f"{_mil(c['ls'] - a['ls'])} | {_pv(pval)} |")
    return L


def _ls_rows(res: dict, tf_key: str) -> list[str]:
    L = ["| 予測先 | 期間 | " + " | ".join(_nl(nk) for nk in NU_KEYS) + " |", "|---|---|" + "---|" * len(NU_KEYS)]
    for h in TIMEFRAMES[tf_key].horizons:
        cur = _nu_key(BAND_NU[tf_key][h])
        best = {src: _base(res, tf_key, src, h)["best_nu"] for src in ("duka", "yahoo")}
        for src, p in (("duka", "tune"), ("duka", "test"), ("yahoo", "tune"), ("yahoo", "test")):
            sc = _base(res, tf_key, src, h)["scores"][p]
            cells = []
            for nk in NU_KEYS:
                v = _mil(sc["nu"][nk]["ls"] - sc["nu"][cur]["ls"]) if nk != cur else "(今) 0"
                cells.append(f"**{v}**" if p == "tune" and nk == best[src] else v)
            L.append(f"| {_hl(tf_key, h)} | {PERIOD_LABEL[(src, p)]} | " + " | ".join(cells) + " |")
    return L


def _variant_rows(res: dict, tf_key: str) -> list[str]:
    L = ["| 案 | 予測先 | ν | 対数スコアの差 ×1000: 調整 | 検証 | Yahoo 全体 | p (検証) | 検証の 50/80/95% | Yahoo の 50/80/95% | 条件 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    dec = res["decision"]["variants"][tf_key]
    for name in ["base", *dec]:
        for h in TIMEFRAMES[tf_key].horizons:
            dv = res["tf"][tf_key]["duka"]["h"][str(h)]["variants"][name]
            yv = res["tf"][tf_key]["yahoo"]["h"][str(h)]["variants"][name]
            nk = res["decision"]["nu"][tf_key][str(h)]["new"] if name == "base" else dv["best_nu"]
            ct = " / ".join(_pc(dv["scores"]["test"]["nu"][nk]["cover"][lv]) for lv in LEVELS)
            cy = " / ".join(_pc(yv["scores"]["all"]["nu"][nk]["cover"][lv]) for lv in LEVELS)
            if name == "base":
                L.append(f"| {VARIANT_LABELS[name]} (案A) | {_hl(tf_key, h)} | {_nl(nk)} | 0 | 0 | 0 | — | {ct} | {cy} | — |")
                continue
            c = dec[name]["h"][str(h)]
            L.append(f"| {VARIANT_LABELS[name]} | {_hl(tf_key, h)} | {_nl(nk)} | {_mil(c['ls_tune'])} | {_mil(c['ls_test'])} | "
                     f"{_mil(c['ls_yahoo'])} | {_pv(dv['dm_vs_base']['test']['p'])} | {ct} | {cy} | {'満たす' if c['ok'] else '—'} |")
    return L


def _year_rows(res: dict, tf_key: str, level: str) -> list[str]:
    hs = TIMEFRAMES[tf_key].horizons
    Y = res["time"][tf_key]["duka"]["year"]
    years = sorted(Y[str(hs[0])])
    L = ["| 年 | " + " | ".join(_hl(tf_key, h) for h in hs) + " |", "|---|" + "---|" * len(hs)]
    for yr in years:
        cells = []
        for h in hs:
            x = Y[str(h)].get(yr)
            if x is None:
                cells.append("")
            elif level == "80":
                cells.append(_pc(x["cur"]["80"]))
            else:
                cells.append(f"{_pc(x['cur'][level])} → {_pc(x['new'][level])}" if x["cur"] != x["new"] else _pc(x["cur"][level]))
        L.append(f"| {yr} | " + " | ".join(cells) + " |")
    return L


def _month_rows(res: dict) -> list[str]:
    hs = TF_D.horizons
    M = {src: res["time"]["1d"][src]["month"] for src in ("duka", "yahoo")}
    months = sorted(set(M["duka"][str(hs[0])]) | set(M["yahoo"][str(hs[0])]))[-18:]
    L = ["| 起点の月 | " + " | ".join(f"Dukascopy {h}" for h in hs) + " | " + " | ".join(f"Yahoo {h}" for h in hs) + " |",
         "|---|" + "---|" * (2 * len(hs))]
    for mo in months:
        cells = []
        for src in ("duka", "yahoo"):
            for h in hs:
                x = M[src][str(h)].get(mo)
                cells.append(f"{_pc(x['cur']['80'])} ({x['n']})" if x else "")
        L.append(f"| {mo} | " + " | ".join(cells) + " |")
    return L


def _rolling_rows(res: dict, tf_key: str, hs) -> list[str]:
    L = ["| 予測先 | 期間 | 窓の数 | 5%点 | 10%点 | 中央値 | ≤77% の割合 | ≤69% の割合 | ≤61% の割合 | 低かった窓 (窓の終わり: 的中率) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for h in hs:
        for src, p in (("duka", "tune"), ("duka", "test"), ("yahoo", "all")):
            x = res["time"][tf_key][src]["rolling"][str(h)].get(p)
            if not x:
                continue
            worst = ", ".join(f"{d}: {_pc(v)}" for d, v in x["worst"][:4])
            L.append(f"| {_hl(tf_key, h)} | {PERIOD_LABEL[(src, p)]} | {x['n']} | {_pc(x['q05'])} | {_pc(x['q10'])} | "
                     f"{_pc(x['median'])} | {_pc(x['le77'])} | {_pc(x['le69'])} | {_pc(x['le61'])} | {worst} |")
    return L


def _checks(res: dict) -> list[str]:
    ch = res["checks"]
    hr = max(v for src in ch["1h"].values() for c in src.values() for v in c["max_rel_diff"].values())
    hn = sum(c["n"] for src in ch["1h"].values() for c in src.values())
    ht = sum(c["target_mismatch"] for src in ch["1h"].values() for c in src.values())
    dr = max((c["max_rel_diff"] or 0.0) for src in ch["1d"].values() for c in src.values())
    dn = sum(c["n"] for src in ch["1d"].values() for c in src.values())
    dt = sum(c["target_mismatch"] for src in ch["1d"].values() for c in src.values())
    rk = max(x["variants"]["base"]["replay_check"] for tf in res["tf"].values() for src in tf.values()
             for x in src["h"].values())
    return [f"- 1時間足の分散: 無作為に選んだ {hn} 件の起点で、engine.sigma_steps (と候補の設定では hourly_variance_path) を直接呼んだ結果と"
            f"この検証の高速版が一致 (最大の相対差 {hr:.1e})。目標時刻も engine.target_times と全件一致 (不一致 {ht} 件)。",
            f"- 日足の分散: 無作為に選んだ {dn} 件の起点で、engine.sigma_steps に起点までの全ての1時間足を渡した結果と一致 "
            f"(最大の相対差 {dr:.1e}。この検証では速さのため直近420日分の1時間足だけを渡しています)。目標時刻の不一致 {dt} 件。",
            f"- k の再現: 各予測先で無作為に選んだ12の起点で、learning.learn_arrays をその時点で判定済みの全予測に当てた k との差は最大 {rk:.4f} "
            f"(重み付き80%点を細かい対数の目盛りで求めているための差)。"]


def report(res: dict) -> str:
    meta = res["meta"]
    L = ["# 予測レンジの形と幅の補正: 20年分の1時間足での検証", ""]
    L += ["## 結論", ""] + _conclusions(res) + [""]
    L += ["## 何を調べたか", "",
          "予測レンジの幅は「値動きの大きさの予測 (engine.sigma_steps、以下 生の σ)」× 「実績から学習する倍率 k」で決まります。"
          "k は予測先ごとに、判定済みの予測 (7ペアまとめて) で80%レンジにちょうど80%が入るように合わせ続けています (learning.py)。"
          "50%・95%レンジは t 分布の形 (自由度 ν、engine.BAND_NU) に従うので、80%レンジは k が、50%・95%レンジは ν が決めます。"
          "今の ν は約2.8年分の1時間足で選んだもので、ローリングのバックテストでは1時間足の24時間先で80%レンジ 77%・95%レンジ 90%、"
          "日足の5〜20営業日先で直近数か月の80%レンジ 61〜69% と、レンジが狭い (裾が細い) 傾向が出ていました。",
          "",
          "そこで、Dukascopy の20年分の1時間足 (bid と ask の仲値、7ペア、2003〜2026年8月) で、本番とまったく同じ計算を毎時点くり返し、"
          "本番の k の学習も時系列どおりに再現したうえで、どの ν がよいかを選び直しました。",
          "",
          f"- 1時間足 (1・4・24時間先): 6,500本の1時間足がそろった時点 (2004年半ば) から**毎時**の起点 (7ペア合計 "
          f"{res['tf']['1h']['duka']['n_origins']:,} 件)。生の σ は本番と同じ計算 (直近6,500本、ニューヨーク時間の曜日×時間の割合、"
          "高値・安値の幅、週末の窓開け。経済指標は含めず)。結果は目標時刻までに確定した最後の1時間足の終値 (本番の判定と同じ)。",
          f"- 日足 (1・5・10・20営業日先): 2005年からの**毎営業日**の起点 (ロンドンの営業日の終わり、7ペア合計 "
          f"{res['tf']['1d']['duka']['n_origins']:,} 件)。日足は Dukascopy の1時間足から data.london_days と同じ区切りで作り、"
          "生の σ は本番と同じ計算 (起点までの1時間足の実現分散の EWMA が過去500日の平常水準へ戻る)。",
          f"- 確認用に Yahoo の1時間足 (本番のデータ源、1時間足は {res['tf']['1h']['yahoo']['range'][0]}〜"
          f"{res['tf']['1h']['yahoo']['range'][1]}、日足は保存されている Yahoo の日足と1時間足で "
          f"{res['tf']['1d']['yahoo']['range'][0]}〜{res['tf']['1d']['yahoo']['range'][1]}) でも同じ計算をしました。",
          "- k の再現: 各起点で、目標時刻を過ぎた予測だけ (台帳の順: 目標時刻・起点・ペア) を使い、learning.learn_arrays と同じ規則 "
          f"(|値動き| ÷ 生の σ の重み付き80%点 ÷ 1.2816、重みは新しい方から数えて半減期 {meta['half_life']['1h']:.0f} 件 (1時間足) / "
          f"{meta['half_life']['1d']:.0f} 件 (日足) で半分、事前情報 (重み {meta['prior_n_k']:.0f} 件分) と混ぜ、{meta['k_bounds'][0]}〜"
          f"{meta['k_bounds'][1]} に収める)。事前情報 (本番では毎日のウォークフォワード) は、毎日0時 (UTC) にその前の約12週 (1時間足) / "
          "約320営業日 (日足) の予測の |値動き| ÷ 生の σ の80%点としました。本番と違う点は、中心を「変化なし」としたこと "
          "(本番の中心は学習したゲイン × モデル平均 + 時間帯の偏りですが、ゲインはほぼ0) と、事前情報の中心をモデル平均でなく「変化なし」にしたことです。"
          "最初の30日 (1時間足) / 120日 (日足) は k の助走として採点から外しました。",
          "- 採点: 標準化した誤差 z = 値動き ÷ (生の σ × k) で、50/80/95%レンジに入った割合と、対数スコア (予測分布の密度の対数の平均。"
          "大きいほど良い。分布の形の良し悪しを測る適切なスコアで、1件あたり0.001の差でも件数が多いと意味があります)。"
          f"ν の候補は {', '.join(_nl(nk) for nk in meta['nus'])}。",
          "- 選び方: 調整期間 (目標時刻が2016年末まで) の対数スコアで予測先ごとに ν を1つ選び、検証期間 (2017年〜) と Yahoo "
          "(日付の前半6割 / 後半4割) で確かめました。p は今の ν との対数スコアの差の Diebold-Mariano 検定 (日ごとの平均、予測期間の重なりを考慮)。",
          "- 採用の条件 (事前に決めたもの): Dukascopy の調整期間と検証期間の**両方**で対数スコアが上がり、かつ 50/80/95% の的中率が"
          "名目値に近づく (3つのずれの平均が小さくなる)、さらに Yahoo 全体で対数スコアが下がらないこと。満たさなければ今の値のままにします。",
          ""]
    L += ["## 公平に比べるための条件", "",
          "- 生の σ は起点までに確定した足だけから計算し、k は起点の時点で判定済みの予測だけから学習しています。ν や設定の選択は調整期間だけで行いました。",
          "- Yahoo の1時間足のうち市場が閉まっている時間の足 (金曜の引け後の1本など) は除きました (timeutil.market_open_mask)。"] + _checks(res) + [""]
    for tf_key in ("1h", "1d"):
        L += [f"## 結果: {TF_LABEL[tf_key]}", "",
              "### 今の自由度と、調整期間の対数スコアで選んだ自由度", "",
              "各セルは「今の ν → 調整期間で選んだ ν」。対数スコアの差は選んだ ν − 今の ν (1件あたり、×1000、プラスが良い)。"
              "80%レンジの幅は k が決めるため ν によらず同じです。", ""]
        L += _nu_rows(res, tf_key, lambda h, tf_key=tf_key: _base(res, tf_key, "duka", h)["best_nu"]) + [""]
        L += ["### ν ごとの対数スコア (今の ν との差、×1000)", "",
              "太字はその期間の前半 (Dukascopy は調整期間、Yahoo は前半6割) で一番良かった ν。", ""]
        L += _ls_rows(res, tf_key) + [""]
    L += ["## k の追従の速さと値動きの大きさの測り方 (事前に決めた候補)", "",
          "ν と同じ手順で、k の半減期 (0.5倍・2倍・4倍) と、生の σ の計算の設定を少しだけ変えた案を比べました。候補はこの表のものだけです。"
          "各案は自分の ν (調整期間で選択) で採点し、今の設定に案A の ν を入れたもの (本番で実際に使う設定) との対数スコアの差を出しています "
          "(ここでは幅の違いも含めた、値動きそのものの対数スコア)。「条件」は上の採用条件をその予測先で満たすかどうかです。"
          "EWMA の減衰などは全ての予測先に共通なので、全ての予測先で条件を満たすときだけ採用します。k の半減期も今のコードでは全予測先に共通です。"
          "(最初の集計では、比べる相手の ν も調整期間で選び直していましたが、本番で使う設定と比べる方が正しいため直しました。"
          "どちらの比べ方でも、全ての予測先で条件を満たす案はありません。)", ""]
    for tf_key in ("1h", "1d"):
        L += [f"### {TF_LABEL[tf_key]}", ""] + _variant_rows(res, tf_key) + [""]
    L += ["## 年ごとの的中率 (Dukascopy、今の設定)", "",
          "80%レンジ (k が決めるので ν によらない) と、95%レンジ (今の ν → 案A の ν。変えない予測先は1つの値)。起点の年ごと、7ペア合計。", "",
          "### 日足: 80%レンジ", ""] + _year_rows(res, "1d", "80") + [""]
    L += ["### 日足: 95%レンジ", ""] + _year_rows(res, "1d", "95") + [""]
    L += ["### 1時間足: 80%レンジ", ""] + _year_rows(res, "1h", "80") + [""]
    L += ["### 1時間足: 95%レンジ", ""] + _year_rows(res, "1h", "95") + [""]
    L += ["## 直近の日足のレンジ不足は局面によるものか", "",
          "起点の月ごとの80%レンジの的中率 (7ペア合計、括弧内は件数)。Dukascopy は2026年8月まで、Yahoo は9月まで (先の長い予測ほど、"
          "最近の起点はまだ判定できていません)。", ""] + _month_rows(res) + [""]
    L += ["80%レンジの的中率を、起点の60日 (暦日。本番のバックテストの「直近60日」と同じ長さ) の窓で7ペアまとめて測り、5日ずつずらしたときの分布です。"
          "窓の的中率がどのくらいの頻度で本番のバックテストの値 (日足 61〜69%、1時間足の24時間先 77%) まで下がるかを示します。", ""]
    L += _rolling_rows(res, "1d", TF_D.horizons) + [""] + _rolling_rows(res, "1h", (24,)) + [""]
    L += ["## 15分足", "",
          "15分足は Yahoo に約60日分しかなく、20年分のデータ (Dukascopy の1時間足) では作れないため、この検証の対象外です。"
          "今の ν (15分・1時間・4時間先で 5・5・4) のままにします。1時間足の結果 (1・4時間先) は参考になりますが、15分足の分散の計算"
          "(15分ごとの EWMA、20営業日の時間帯の割合) は別物なので、そのまま当てはめてはいません。", ""]
    L += ["## 推奨する変更", ""] + _recommend(res) + [""]
    L += ["## 再実行", "", "```bash",
          f"python -m aifx.research_bands   # research/bands.md と research/bands.json を作成 (2並列で約{meta['seconds'] / 60:.0f}分)",
          "```", ""]
    return "\n".join(L)


def _hs_text(tf_key: str, hs) -> str:
    unit = "時間先" if tf_key == "1h" else "営業日先"
    return "・".join(str(h) for h in hs) + unit


def _span(vals: list[float]) -> str:
    lo, hi = f"{min(vals) * 100:.1f}", f"{max(vals) * 100:.1f}"
    return f"{lo}%" if lo == hi else f"{lo}〜{hi}%"


def _after_rows(res: dict, tf_key: str, cfg: dict) -> list[str]:
    """Before (current settings and shape) -> after (``cfg``: {h: (variant, nu)}) for the changed horizons."""
    L = ["| 予測先 | 変更 | 期間 | 件数 | 50%レンジ | 80%レンジ | 95%レンジ | 対数スコアの差 (×1000) |",
         "|---|---|---|---|---|---|---|---|"]
    for h, (name, nk) in cfg.items():
        cur = _nu_key(BAND_NU[tf_key][int(h)])
        what = f"{_nl(cur)} → {_nl(nk)}" + ("" if name == "base" else f"、k の半減期 {TIMEFRAMES[tf_key].half_life:.0f} → "
                                                               f"{TIMEFRAMES[tf_key].half_life * float(name[2:]):.0f} 件")
        for src, p in (("duka", "tune"), ("duka", "test"), ("yahoo", "all")):
            v = res["tf"][tf_key][src]["h"][str(h)]["variants"]
            a, b = v["base"]["scores"][p], v[name]["scores"][p]
            cov = " | ".join(f"{_pc(a['nu'][cur]['cover'][lv])} → {_pc(b['nu'][nk]['cover'][lv])}" for lv in LEVELS)
            d = (b["nu"][nk]["ls"] + b["jac"]) - (a["nu"][cur]["ls"] + a["jac"])
            L.append(f"| {_hl(tf_key, h)} | {what} | {PERIOD_LABEL[(src, p)]} | {a['n']:,} | {cov} | {_mil(d)} |")
    return L


def _conclusions(res: dict) -> list[str]:
    dec = res["decision"]
    L = []
    for tf_key in ("1h", "1d"):
        nd = dec["nu"][tf_key]
        hs = list(nd)
        cur_txt = "・".join(_nl(x["current"])[2:] if x["current"] != "normal" else "正規" for x in nd.values())
        covs = [_base(res, tf_key, "duka", h)["scores"][p]["nu"][nd[h]["current"]]["cover"] for h in hs for p in ("tune", "test")]
        adopted = [h for h in hs if nd[h]["adopt"]]
        head = (f"- **{TF_LABEL[tf_key]}の自由度は今のまま ({_hs_text(tf_key, hs)}で ν={cur_txt}) を勧めます。**" if not adopted else
                f"- **{TF_LABEL[tf_key]}の自由度は、" + "、".join(f"{_hl(tf_key, h)} {_nl(nd[h]['current'])} → {_nl(nd[h]['new'])}"
                                                           for h in adopted) + " に変えることを勧めます"
                + ("" if len(adopted) == len(hs) else " (" + "・".join(h for h in hs if h not in adopted)
                   + ("時間先" if tf_key == "1h" else "営業日先") + "は今のまま)") + "。**")
        txt = (head + f" 本番と同じ k の学習を再現すると、今の ν での 50/80/95%レンジの的中は20年分 (調整期間・検証期間) で "
               f"{_span([c['50'] for c in covs])} / {_span([c['80'] for c in covs])} / {_span([c['95'] for c in covs])} でした。")
        for h in adopted:
            x = nd[h]
            b = _base(res, tf_key, "duka", h)["scores"]
            y = _base(res, tf_key, "yahoo", h)["scores"]["all"]
            txt += (f" {_hl(tf_key, h)}は {_nl(x['new'])} で95%レンジの的中が調整 {_pc(b['tune']['nu'][x['current']]['cover']['95'])}→"
                    f"{_pc(b['tune']['nu'][x['new']]['cover']['95'])}・検証 {_pc(b['test']['nu'][x['current']]['cover']['95'])}→"
                    f"{_pc(b['test']['nu'][x['new']]['cover']['95'])} (Yahoo {_pc(y['nu'][x['current']]['cover']['95'])}→"
                    f"{_pc(y['nu'][x['new']]['cover']['95'])}) と名目に近づき、対数スコアも調整 {_mil(x['ls_tune'])}・検証 "
                    f"{_mil(x['ls_test'])}・Yahoo {_mil(x['ls_yahoo'])} (×1000) と上がります。")
            yh = {p: _base(res, tf_key, "yahoo", h)["scores"][p]["nu"] for p in ("tune", "test")}
            neg = [p for p in ("tune", "test") if yh[p][x["new"]]["ls"] < yh[p][x["current"]]["ls"]]
            for p in neg:
                txt += (f"ただし Yahoo の{'前半6割' if p == 'tune' else '後半4割'}だけは今の ν の方が良く "
                        f"({_mil(yh[p][x['new']]['ls'] - yh[p][x['current']]['ls'])})、"
                        + ("先の長い予測は期間が重なるため、この期間の独立な観測は1ペアあたり十数回分しかありません。" if int(h) >= 10 else
                           "期間が短く、偶然の範囲も大きい結果です。"))
        for h in hs:
            x = nd[h]
            if x["adopt"] or x["best"] == x["current"]:
                continue
            why = []
            if x["ls_test"] <= EPS:
                why.append(f"検証期間の対数スコアが上がらない ({_mil(x['ls_test'])})")
            for p, lab in (("tune", "調整期間"), ("test", "検証期間")):
                if x[f"cov_{p}"] > EPS:
                    why.append(f"{lab}の的中が名目から離れる")
                elif abs(x[f"cov_{p}"]) <= EPS:
                    why.append(f"{lab}の的中の名目からのずれが変わらない (50%レンジが離れる分と95%レンジが近づく分がちょうど同じ)")
            if x["ls_yahoo"] < -EPS:
                why.append("Yahoo で対数スコアが下がる")
            txt += (f" {_hl(tf_key, h)}は {_nl(x['best'])} が調整期間で一番良く、対数スコアは調整 {_mil(x['ls_tune'])}・検証 "
                    f"{_mil(x['ls_test'])}・Yahoo {_mil(x['ls_yahoo'])} ですが、" + "、".join(why) + "ため今のままです。")
        L.append(txt)
    # the 80 % band and the live rolling backtest
    r24 = res["time"]["1h"]["duka"]["rolling"]["24"]
    yw24 = res["time"]["1h"]["yahoo"]["rolling"]["24"]["all"]
    c24 = [_base(res, "1h", "duka", 24)["scores"][p]["nu"][_nu_key(BAND_NU["1h"][24])]["cover"]["80"] for p in ("tune", "test")]
    y24 = [x["cur"]["80"] for x in res["time"]["1h"]["duka"]["year"]["24"].values()]
    L.append(f"- **1時間足の24時間先のレンジ不足 (本番のバックテストで80%レンジ 77%・95%レンジ 90%) は、主に直近60日という短い期間のぶれです。** "
             f"20年分では24時間先の80%レンジは {_span(c24)} (年ごとに {_span(y24)}) で、60日の窓で 77% 以下になるのは Dukascopy で "
             f"{_pc(r24['tune']['le77'])} (調整)・{_pc(r24['test']['le77'])} (検証)、Yahoo で {_pc(yw24['le77'])} の窓です。"
             "ただし 80% をわずかに下回り続けるのは、k が約2.4日分 (半減期400件) の直近の結果で決まり、24時間先では重なり合った少数の結果に振り回されるためです (下の k の半減期の節)。")
    rd = res["time"]["1d"]["duka"]["rolling"]
    yr = res["time"]["1d"]["duka"]["year"]
    ends = sorted({d[:7] for h in ("10", "20") for p in ("tune", "test") for d, v in rd[h][p]["worst"][:3] if v <= 0.69})
    worst = [f"{int(m[:4])}年{int(m[5:])}月" for m in ends]
    mo = {src: res["time"]["1d"][src]["month"] for src in ("duka", "yahoo")}
    low_m = sorted({m for src in mo for h in ("10", "20") for m, x in mo[src][h].items() if x["n"] >= 70 and x["cur"]["80"] < 0.70})
    low = [f"{int(m[:4])}年{int(m[5:])}月" for m in low_m]
    last = sorted(yr["1"])[-2:]
    L.append("- **日足の直近のレンジ不足 (5〜20営業日先の80%レンジ 61〜69%) は、局面によるもので、仕組みの問題ではないと考えます。** "
             f"60日の窓で80%レンジが 69% 以下になるのは、20年分で 5営業日先 {_pc(rd['5']['tune']['le69'])}・{_pc(rd['5']['test']['le69'])}、"
             f"10営業日先 {_pc(rd['10']['tune']['le69'])}・{_pc(rd['10']['test']['le69'])}、20営業日先 {_pc(rd['20']['tune']['le69'])}・"
             f"{_pc(rd['20']['test']['le69'])} (調整・検証) の窓で、" + "、".join(worst) + " などの急変の時期 (窓の終わりの月) に集中しています "
             "(値動きが急に大きくなると、生の σ (過去の値動きの EWMA) と k が追いつくまでの数週間はレンジが狭すぎる)。年単位では "
             + "、".join(f"{y}年 {_span([yr[str(h)][y]['cur']['80'] for h in TF_D.horizons])}" for y in last)
             + " と平年並みで、月別では " + "、".join(low) + " の起点が低くなっています (下の表)。")
    va = [f"{TF_LABEL[tf]}の{VARIANT_LABELS[n]}" for tf in ("1h", "1d") for n, v in dec["variants"][tf].items() if v["adopt"]]
    L.append("- **値動きの大きさの測り方 (EWMA の減衰、高値・安値の幅の合わせ方、平常水準へ戻る速さ) と、全予測先共通の k の半減期は、"
             + ("今のままを勧めます。** 事前に決めた候補のどれも、全ての予測先で採用条件を満たしませんでした。" if not va else
                "次の案が採用条件を満たしました: " + "、".join(va) + "。**"))
    hl = dec["half_life"]
    ok = {tf: [h for h, x in hl[tf].items() if x["adopt"]] for tf in ("1h", "1d")}
    if ok["1h"] or ok["1d"]:
        parts = []
        for tf in ("1h", "1d"):
            for h in ok[tf]:
                x = hl[tf][h]
                parts.append(f"{TF_LABEL[tf]}の{_hl(tf, h)} (半減期 {TIMEFRAMES[tf].half_life:.0f}→{x['half_life']:.0f} 件、{_nl(x['nu'])}: "
                             f"対数スコア 調整 {_mil(x['ls_tune'])}・検証 {_mil(x['ls_test'])}・Yahoo {_mil(x['ls_yahoo'])})")
        L.append("- **追加の発見 (結果を見てから加えた分析): k の半減期を予測先ごとに選べるようにすると、先の長い予測で大きく良くなります。** "
                 "今の k は予測先によらず同じ半減期 (1時間足400件 ≈ 2.4日、日足120件 ≈ 17営業日) で学習していますが、先の長い予測ほど"
                 "判定済みの結果が重なり合って少なく、k がぶれます。調整期間で半減期を選び、上と同じ条件で確かめると、"
                 + "、".join(parts) + " で条件を満たしました。ν の変更よりも改善が大きい一方、コード (learning.learn と engine.Timeframe) "
                 "の変更が必要で、予測先ごとに選ぶ使い方は結果を見たあとで加えたものなので、別の案として示します (推奨する変更の案B)。")
    L.append("- **15分足**は20年分のデータがないため対象外です (今の ν のまま)。")
    return L


def _recommend(res: dict) -> list[str]:
    dec = res["decision"]
    new = _new_nu(res)
    fmt = {tf: _nu_dict(tf, new[tf]) for tf in new}
    L = ["### 案A: 採用条件を満たした変更 (定数の変更のみ)", ""]
    if any(x["adopt"] for tf in dec["nu"].values() for x in tf.values()):
        L += ["engine.py:", "", "```python",
              f'BAND_NU = {{"15m": {{1: 5, 4: 5, 16: 4}}, "1h": {fmt["1h"]}, "1d": {fmt["1d"]}}}', "```", "",
              "変更した予測先の前後 (前 = 今の設定、後 = 案A。対数スコアの差は後 − 前):", ""]
        for tf in ("1h", "1d"):
            cfg = {h: ("base", x["new"]) for h, x in dec["nu"][tf].items() if x["adopt"]}
            if cfg:
                L += _after_rows(res, tf, cfg) + [""]
    else:
        L += ["なし (engine.BAND_NU は今のまま)。", ""]
    L += ["値動きの大きさの測り方 (volatility.py) と k の学習 (learning.py、Timeframe.half_life) は変えません。", ""]
    hl = dec["half_life"]
    if any(x["adopt"] for tf in hl.values() for x in tf.values()):
        halves = {tf: {h: (x["half_life"] if x["adopt"] else TIMEFRAMES[tf].half_life) for h, x in hl[tf].items()} for tf in hl}
        nus_b = {tf: {h: (hl[tf][h]["nu"] if hl[tf][h]["adopt"] else new[tf][h]) for h in new[tf]} for tf in new}
        L += ["### 案B: 案A に加えて、k の半減期を予測先ごとに (コードの変更が必要)", "",
              "結果を見たあとで加えた分析なので、案A より証拠は弱いものです (候補の半減期 ×0.5・×2・×4 は事前に決めたもの、予測先ごとに選ぶことは事後)。"
              "採用するなら、learning.learn が予測先ごとの半減期を受け取れるようにし (例: Timeframe に `half_lives: dict[int, float]`)、"
              "それを呼ぶ pipeline.py・audit.py・backtest._replay も同じ値を使うようにします (engine.py と learning.py が変わるので model_version も変わり、"
              "抜き取りの再計算はこれまでどおり同じ版の予測だけが対象になります)。", "",
              "```python",
              f'BAND_NU = {{"15m": {{1: 5, 4: 5, 16: 4}}, "1h": {_nu_dict("1h", nus_b["1h"])}, "1d": {_nu_dict("1d", nus_b["1d"])}}}',
              "# k の半減期 (判定済みの予測の件数、7ペア合計)",
              "HALF_LIFE = {" + ", ".join(f'"{tf}": {{' + ", ".join(f"{h}: {v:.0f}" for h, v in halves[tf].items()) + "}"
                                          for tf in ("1h", "1d")) + "}",
              "```", "", "前後 (前 = 今の設定、後 = 案B):", ""]
        for tf in ("1h", "1d"):
            cfg = {}
            for h in new[tf]:
                x = hl[tf][h]
                if x["adopt"]:
                    cfg[h] = (f"hl{x['mult']}", x["nu"])
                elif dec["nu"][tf][h]["adopt"]:
                    cfg[h] = ("base", new[tf][h])
            if cfg:
                L += _after_rows(res, tf, cfg) + [""]
    return L


if __name__ == "__main__":
    run()
