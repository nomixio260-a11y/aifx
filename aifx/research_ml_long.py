"""Machine-learned direction on ~20 years of hourly bars: better than chance, better than the time of day?

research/ml.md and research/dl.md found direction hit rates of about 50 % on the ~2.8 years of Yahoo
hourly bars; the only direction signal that held out of sample was the time-of-day drift (season.py),
whose high-confidence calls (|t| >= 4) were right about 80 % of the time. Here the same question is
asked of about 20 years of Dukascopy hourly mid bars (history.load_long_hourly), pooled over the 7 pairs.

* Features, all known at the close of the origin bar: moves over 1-120 hours (in units of recent
  volatility), realised volatility and its ratios, bar range and ATR, distance from moving averages,
  RSI, MACD histogram, Bollinger position, stochastics, New York time of day and weekday (cyclical and
  one-hot) and the hours to the 17:00 rollover, the time-of-day drift statistics of the next bars
  exactly as season.py computes them from earlier days only (plus a ~3-year window), the same-hour
  moves of all 7 pairs and currency strengths (USD and JPY factors), the spread, the short-rate
  difference (point in time) and VIX (two days late, as published).
* Targets: the sign of the move from the origin close to the close 1, 4 and 24 bars later.
* Models: a standardised logistic regression and a small regularised LightGBM, each with and without
  the pair as an input, LightGBM without any time-of-day input (does anything remain?), and the
  time-of-day drift alone (the sign of the past slot drift, calls when |t| >= 2, high confidence >= 4).
* Walk-forward by calendar year: each year from 2010 is predicted by models trained on everything
  before it; training targets must end a day before the year starts (purge). Hyperparameters and the
  confidence thresholds (top 10/5/1/0.1 % of |p - 0.5|) are chosen on the 2010-2016 predictions
  ("tune") only and applied unchanged to 2017 onwards ("test").

    python -m aifx.research_ml_long   # writes research/ml_long.md and research/ml_long.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import history, season
from .data import PAIRS

REPORT_DIR = Path("research")
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
WEEK_NS = 7 * DAY_NS
CODES = tuple(PAIRS)
CURS = ("USD", "JPY", "EUR", "GBP", "AUD")
HORIZONS = (1, 4, 24)
LAGS = (1, 2, 4, 8, 24, 72, 120)
STR_LAGS = (1, 4, 24, 120)
VOL_WINDOWS = (24, 120, 480)
SMA = (24, 120, 480)
START = "2004-01-01"        # first origin used (the 6,000-bar slot statistics need about a year of bars)
FIRST_TEST = 2010           # first walk-forward year
LAST_TUNE = 2016            # hyperparameters and thresholds come from the years up to this one
TOPS = (0.10, 0.05, 0.01, 0.001)
SCAN_TOPS = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)
SEASON_TS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
ROLL_BP = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)   # levels of the expected swap-point shift at the roll (bp)
GOAL_HIT, GOAL_N = 0.9, 200  # "90 %": hit rate on both periods with at least this many test calls
BEAT_MARGIN = 0.02          # "beats the time-of-day drift": this much higher hit rate on both periods
TRAIN_KEEP = 0.2            # share of training rows kept (random, seeded): hourly neighbours are near-duplicates
EMBARGO_NS = DAY_NS         # training targets end this long before the test year starts
LONG_WINDOW = 18000         # a second, ~3-year window for the slot statistics (a feature only)
ROLL_HOURS = (16, 17)       # New York start hours of the bars into and out of the 17:00 rollover
                            # (a forecast "involves the roll" when one of its H bars starts at these hours)
SEED = 7

LGB_BASE = {"objective": "binary", "learning_rate": 0.05, "feature_fraction": 0.5, "bagging_fraction": 0.5,
            "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63, "verbose": -1, "seed": SEED,
            "deterministic": True, "num_threads": 1, "force_col_wise": True}
LGB_GRID = ({"num_leaves": 7, "min_data_in_leaf": 2000}, {"num_leaves": 31, "min_data_in_leaf": 2000})
LGB_ROUNDS = (25, 50, 100, 200, 400)
LOGIT_C = (0.001, 0.01, 0.1)

# name -> (kind, pair as an input, time-of-day inputs); the first of each kind is the one tuned
MODELS = {
    "logit": ("logit", False, True),
    "logit_pair": ("logit", True, True),
    "lgbm": ("lgbm", False, True),
    "lgbm_pair": ("lgbm", True, True),
    "lgbm_notime": ("lgbm", True, False),
}
MODEL_NAMES = {"logit": "ロジスティック回帰", "logit_pair": "ロジスティック回帰 (+ペア)", "lgbm": "LightGBM",
               "lgbm_pair": "LightGBM (+ペア)", "lgbm_notime": "LightGBM (+ペア、時間の入力なし)",
               "season": "時間帯の偏りだけ (season.py)", "season_long": "時間帯の偏りだけ (直近18,000本)",
               "roll_rule": "ロールオーバーのスワップ調整 (規則)"}


# ------------------------------------------------------------------ time-of-day drift, vectorised

def _slot_day_stats(r: np.ndarray, key: np.ndarray, n_keys: int, cut: np.ndarray,
                    window: int) -> tuple[np.ndarray, np.ndarray]:
    """season._stats of the returns ``r`` (bp, NaN = left out) grouped by ``key``, for every day at once:
    row d uses the returns of bars max(0, cut[d] - window - 1) + 1 .. cut[d] - 1, as season.slot_stats."""
    lo = np.maximum(cut - window - 1, 0) + 1
    hi = np.maximum(cut, lo)
    ok = np.flatnonzero(np.isfinite(r))
    order = np.argsort(key[ok], kind="stable")
    pos, k = ok[order], key[ok][order]
    v = r[pos]
    edges = np.searchsorted(k, np.arange(n_keys + 1))
    cnt = np.zeros((len(cut), n_keys))
    s1, s2 = np.zeros_like(cnt), np.zeros_like(cnt)
    for s in range(n_keys):
        p, x = pos[edges[s]:edges[s + 1]], v[edges[s]:edges[s + 1]]
        c1 = np.concatenate([[0.0], np.cumsum(x)])
        c2 = np.concatenate([[0.0], np.cumsum(x * x)])
        a, b = np.searchsorted(p, lo), np.searchsorted(p, hi)
        cnt[:, s], s1[:, s], s2[:, s] = b - a, c1[b] - c1[a], c2[b] - c2[a]
    use = cnt >= season.MIN_N
    safe = np.where(use, cnt, 1.0)
    mu = np.where(use, s1 / safe, 0.0)
    sd = np.sqrt(np.maximum(s2 / safe - mu * mu, 0.0))
    t = np.where(use & (sd > 0), mu / np.where(sd > 0, sd, 1.0) * np.sqrt(safe), 0.0)
    return mu, t


def season_drift(bars: pd.DataFrame, window: int = season.WINDOW,
                 horizons: tuple[int, ...] = HORIZONS) -> dict[str, np.ndarray]:
    """For every origin (the close of each bar), the time-of-day drift as the server computes it
    (season.slot_stats over the ``window`` bars before the origin's UTC day, then season.bar_drift and
    season.combined_t over the next bars): the next bar's weekday-hour and hour slot averages (bp) and
    t (``mu_w``, ``t_w``, ``mu_d``, ``t_d``) and, per horizon h, the summed drift ``d{h}`` of the next h
    bars and its t ``t{h}`` (0 where no slot is clear, NaN past the end)."""
    idx = bars.index
    ns = idx.as_unit("ns").asi8
    n = len(ns)
    r = np.full(n, np.nan)
    r[1:] = np.diff(np.log(bars["close"].to_numpy(float))) * 1e4
    r[1:][np.diff(ns) > HOUR_NS] = np.nan            # a bar after a pause carries its gap
    week, tod, per_day = season._slots(idx, 60)
    days, inv = np.unique((ns + HOUR_NS) // DAY_NS * DAY_NS, return_inverse=True)
    cut = np.searchsorted(ns, days, side="left")
    mu_w, t_w = _slot_day_stats(r, week, 7 * per_day, cut, window)
    mu_d, t_d = _slot_day_stats(r, tod, per_day, cut, window)
    out: dict[str, np.ndarray] = {}
    pos = np.arange(n)
    cum_d, se2 = np.zeros(n), np.zeros(n)
    for k in range(1, max(horizons) + 1):
        j = np.minimum(pos + k, n - 1)
        mw, tw, md, td = mu_w[inv, week[j]], t_w[inv, week[j]], mu_d[inv, tod[j]], t_d[inv, tod[j]]
        use_w = np.abs(tw) >= season.T_MIN
        use_d = ~use_w & (np.abs(td) >= season.T_MIN)
        d = np.where(use_w, mw, np.where(use_d, md, 0.0))
        t = np.where(use_w, tw, np.where(use_d, td, 0.0))
        if k == 1:
            out.update(mu_w=mw, t_w=tw, mu_d=md, t_d=td)
        cum_d += d
        se2 += np.where(t != 0, (d / np.where(t != 0, t, 1.0)) ** 2, 0.0)
        if k in horizons:
            ct = np.where(se2 > 0, cum_d / np.sqrt(np.where(se2 > 0, se2, 1.0)), 0.0)
            past = pos + k >= n
            out[f"d{k}"] = np.where(past, np.nan, cum_d)
            out[f"t{k}"] = np.where(past, np.nan, ct)
    return out


def check_season(bars: pd.DataFrame, samples: int = 40, seed: int = SEED) -> dict:
    """Compare season_drift with season.slot_stats / bar_drift / combined_t at random origins."""
    vec = season_drift(bars)
    rng = np.random.default_rng(seed)
    idx = bars.index
    n = len(idx)
    worst, calls, sign_diff = 0.0, 0, 0
    for i in rng.choice(np.arange(season.WINDOW, n - max(HORIZONS) - 1), size=min(samples, n // 2), replace=False):
        origin = (idx[i] + pd.Timedelta(hours=1)).to_pydatetime()
        stats = season.slot_stats(bars, origin, 60)
        d, t = season.bar_drift(stats, idx[i + 1: i + 1 + max(HORIZONS)], 60)
        for h in HORIZONS:
            ref_d, ref_t = float(d[:h].sum()), season.combined_t(d[:h], t[:h])
            worst = max(worst, abs(ref_d - vec[f"d{h}"][i]), abs(ref_t - vec[f"t{h}"][i]))
            calls += int(ref_d != 0)
            sign_diff += int(np.sign(ref_d) != np.sign(vec[f"d{h}"][i]))
    return {"origins": int(min(samples, n // 2)), "max_abs_diff": float(worst), "calls": calls,
            "sign_mismatches": sign_diff}


# ------------------------------------------------------------------ features

def _ewm_sd(r: pd.Series, lam: float) -> pd.Series:
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def _load(root: Path | None) -> dict[str, pd.DataFrame]:
    out = {}
    for code in CODES:
        df = history.load_long_hourly(code, root)
        df = df[~df.index.duplicated()].sort_index()
        out[code] = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return out


def _cross(frames: dict[str, pd.DataFrame]) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per lag: the moves of all pairs (in units of their volatility) and currency strengths, on the
    union of the pairs' hours (a missing hour takes the last price up to 3 hours back, never a later one)."""
    union = frames[CODES[0]].index
    for code in CODES[1:]:
        union = union.union(frames[code].index)
    closes = pd.DataFrame({c: frames[c]["close"].reindex(union).ffill(limit=3) for c in CODES})
    lc = np.log(closes)
    sig = lc.diff().apply(lambda s: _ewm_sd(s, 0.97))
    out = {}
    for k in STR_LAGS:
        z = (lc - lc.shift(k)) / (sig * math.sqrt(k))
        st = {}
        for cur in CURS:
            parts = [z[c] if p.base == cur else -z[c] for c, p in PAIRS.items() if cur in (p.base, p.quote)]
            st[cur] = pd.concat(parts, axis=1).mean(axis=1)
        out[k] = (z, pd.DataFrame(st))
    return out


def _pair_rows(code: str, df: pd.DataFrame, cross: dict, carry: pd.Series, vix: pd.DataFrame) -> dict[str, np.ndarray]:
    """Feature columns (``f_*``), targets and bookkeeping for every bar of one pair."""
    pair = PAIRS[code]
    idx = df.index
    ns = idx.as_unit("ns").asi8
    n = len(ns)
    o, h, lo, c = (df[k] for k in ("open", "high", "low", "close"))
    lc = np.log(c)
    r1 = lc.diff()
    sig = _ewm_sd(r1, 0.97)
    F: dict[str, pd.Series | np.ndarray] = {}
    for k in LAGS:
        F[f"z{k}"] = (lc - lc.shift(k)) / (sig * math.sqrt(k))
    vol = {w: r1.rolling(w).std() * 1e4 for w in VOL_WINDOWS}
    for w in VOL_WINDOWS:
        F[f"vol{w}"] = vol[w]
    F["vol_ewm"] = sig * 1e4
    F["volr_24_480"] = vol[24] / vol[480]
    F["volr_120_480"] = vol[120] / vol[480]
    prev = c.shift(1)
    tr = np.log(pd.concat([h, prev], axis=1).max(axis=1) / pd.concat([lo, prev], axis=1).min(axis=1))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rng_ = np.log(h / lo)
    F["range"] = rng_ / sig
    F["atr"] = atr / sig
    F["range_atr"] = rng_ / atr
    F["body"] = np.log(c / o) / sig
    F["clv"] = ((c - lo) / (h - lo)).where(h > lo) - 0.5
    for m in SMA:
        F[f"ma{m}"] = np.log(c / c.rolling(m).mean()) / sig
    for m in (24, 120):
        hi_m, lo_m = h.rolling(m).max(), lo.rolling(m).min()
        F[f"pos{m}"] = ((c - lo_m) / (hi_m - lo_m)).where(hi_m > lo_m) - 0.5
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    F["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50) / 50
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    F["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c * sig)
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    F["boll"] = ((c - sma20) / (2 * sd20)).where(sd20 > 0)
    hi14, lo14 = h.rolling(14).max(), lo.rolling(14).min()
    stoch_k = ((c - lo14) / (hi14 - lo14)).where(hi14 > lo14) - 0.5
    F["stoch_k"] = stoch_k
    F["stoch_d"] = stoch_k.rolling(3).mean()
    spread_bp = (df["spread"] / c * 1e4).clip(lower=0)
    F["spread"] = spread_bp
    F["spread_rel"] = spread_bp / spread_bp.rolling(120, min_periods=24).median().replace(0, np.nan)
    gap = np.full(n, np.nan)
    gap[1:] = np.diff(ns) / HOUR_NS
    F["gap_h"] = np.log(gap)
    # time: the next bar starts at the origin; New York time follows the rollover through daylight saving
    ny = (idx + pd.Timedelta(hours=1)).tz_convert(season.NEW_YORK)
    hr, wd = ny.hour.to_numpy(), ny.dayofweek.to_numpy()
    F["ny_hour"], F["ny_wd"] = hr.astype(float), wd.astype(float)
    F["hsin"], F["hcos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    wk = (wd * 24 + hr) / 168
    F["wsin"], F["wcos"] = np.sin(2 * np.pi * wk), np.cos(2 * np.pi * wk)
    F["to_roll"] = ((17 - hr) % 24).astype(float)
    for k in range(24):
        F[f"oh_h{k:02d}"] = (hr == k).astype(float)
    for k in range(7):
        F[f"oh_wd{k}"] = (wd == k).astype(float)
    sd6 = season_drift(df, season.WINDOW, HORIZONS)
    for key in ("mu_w", "t_w", "mu_d", "t_d"):
        F[f"s_{key}"] = sd6[key]
    for H in HORIZONS:
        F[f"s_d{H}"], F[f"s_t{H}"] = sd6[f"d{H}"], sd6[f"t{H}"]
    F["s_z1"] = sd6["d1"] / (sig.to_numpy() * 1e4)
    sdl = season_drift(df, LONG_WINDOW, (1,))
    for key in ("mu_w", "t_w", "mu_d", "t_d", "d1", "t1"):
        F[f"sl_{key}"] = sdl[key]
    for k in STR_LAGS:
        z, st = cross[k]
        F[f"str_b{k}"] = st[pair.base].reindex(idx).to_numpy()
        F[f"str_q{k}"] = st[pair.quote].reindex(idx).to_numpy()
        F[f"str_d{k}"] = F[f"str_b{k}"] - F[f"str_q{k}"]
        F[f"usd{k}"] = st["USD"].reindex(idx).to_numpy()
        F[f"jpy{k}"] = st["JPY"].reindex(idx).to_numpy()
        if k in (1, 4):
            for other in CODES:
                F[f"x{k}_{other}"] = z[other].reindex(idx).to_numpy()
    day = pd.to_datetime((ns + HOUR_NS) // DAY_NS * DAY_NS)          # the origin's UTC date
    F["carry"] = carry.reindex(day).to_numpy()
    # the value date moves at 17:00 New York: the quote of the higher-yielding currency then drops by the
    # forward points of the days rolled (3 on Wednesdays, over the weekend), which the swap pays back
    bar_ny = idx.tz_convert(season.NEW_YORK)
    cum_days = np.concatenate([[0.0], np.cumsum(np.where(bar_ny.hour == 17, np.where(bar_ny.dayofweek == 2, 3.0, 1.0), 0.0))])
    pos = np.arange(n)
    for H in HORIZONS:
        days = cum_days[np.minimum(pos + H + 1, n)] - cum_days[np.minimum(pos + 1, n)]
        F[f"roll_bp{H}"] = -F["carry"] * days / 360 * 100          # expected shift of the next H bars, bp
    F["vix"] = vix["vix"].reindex(day).to_numpy()
    F["vix_ch5"] = vix["ch5"].reindex(day).to_numpy()
    F["pair_id"] = np.full(n, float(CODES.index(code)))
    for other in CODES:
        F[f"oh_p_{other}"] = np.full(n, float(other == code))
    out = {f"f_{k}": np.asarray(v, dtype=np.float32) for k, v in F.items()}
    out["pair"] = np.full(n, CODES.index(code), dtype=np.int8)
    out["t"] = ns
    out["hour"] = hr.astype(np.int8)
    nxt = np.full(n, np.inf)
    nxt[:-1] = np.diff(ns)
    out["valid"] = (nxt == HOUR_NS) & (ns >= pd.Timestamp(START, tz="UTC").value)
    lcv, spv = lc.to_numpy(), spread_bp.to_numpy()
    at_roll = np.isin(idx.tz_convert(season.NEW_YORK).hour.to_numpy(), ROLL_HOURS)     # bars that start at the roll
    cum_roll = np.concatenate([[0], np.cumsum(at_roll)])
    for H in HORIZONS:
        out[f"roll{H}"] = cum_roll[np.minimum(pos + H + 1, n)] - cum_roll[np.minimum(pos + 1, n)] > 0
        y = np.full(n, np.nan)
        y[:-H] = (lcv[H:] - lcv[:-H]) * 1e4
        cost = np.full(n, np.nan)
        cost[:-H] = (spv[H:] + spv[:-H]) / 2
        tend = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        tend[:-H] = ns[H:] + HOUR_NS
        out[f"y{H}"], out[f"cost{H}"], out[f"tend{H}"] = y.astype(np.float32), cost.astype(np.float32), tend
        out[f"sd{H}"], out[f"st{H}"] = sd6[f"d{H}"].astype(np.float32), sd6[f"t{H}"].astype(np.float32)
    out["sld1"], out["slt1"] = sdl["d1"].astype(np.float32), sdl["t1"].astype(np.float32)
    for H in HORIZONS:
        out[f"rollbp{H}"] = np.asarray(F[f"roll_bp{H}"], dtype=np.float32)
    return out


def build(root: Path | None = None, log=print) -> dict:
    """All origins of all pairs from START: a float32 feature matrix ``X`` (columns ``cols``) and
    per-row arrays (pair, origin bar start ``t``, New York hour of the next bar, targets ``y{H}`` in bp,
    trading cost ``cost{H}`` = the average spread at entry and exit in bp, target end time ``tend{H}``,
    the time-of-day drift ``sd{H}``/``st{H}``)."""
    frames = _load(root)
    cross = _cross(frames)
    all_days = pd.DatetimeIndex(sorted({d for df in frames.values()
                                        for d in pd.to_datetime((df.index.as_unit("ns").asi8 + HOUR_NS)
                                                                // DAY_NS * DAY_NS)}))
    rates = history.rates_panel(all_days)
    v = history.load_vix()
    v = pd.DataFrame({"vix": np.log(v), "ch5": np.log(v).diff(5)})
    vix = v.reindex(all_days.union(v.index)).ffill().reindex(all_days)
    parts = []
    for code in CODES:
        pair = PAIRS[code]
        carry = rates[pair.base] - rates[pair.quote]
        rows = _pair_rows(code, frames[code], cross, carry, vix)
        keep = rows["t"] >= pd.Timestamp(START, tz="UTC").value
        parts.append({k: a[keep] for k, a in rows.items()})
        if log:
            log(f"features {code}: {int(keep.sum()):,} origins from {START}, "
                f"{frames[code].index[0]:%Y-%m-%d} .. {frames[code].index[-1]:%Y-%m-%d}")
    cols = [k[2:] for k in parts[0] if k.startswith("f_")]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if not k.startswith("f_")}
    D["X"] = np.empty((len(D["t"]), len(cols)), dtype=np.float32)
    at = 0
    for p in parts:
        m = len(p["t"])
        for j, c in enumerate(cols):
            D["X"][at:at + m, j] = p["f_" + c]
        at += m
    D["X"][~np.isfinite(D["X"])] = np.nan
    D["cols"] = cols
    D["year"] = pd.to_datetime(D["t"], utc=True).year.to_numpy()
    D["week"] = (D["t"] + 3 * DAY_NS) // WEEK_NS
    D["keep"] = np.random.default_rng(SEED).random(len(D["t"])) < TRAIN_KEEP
    D["spans"] = {c: [str(frames[c].index[0]), str(frames[c].index[-1])] for c in CODES}
    return D


TIME_PREFIXES = ("ny_", "hsin", "hcos", "wsin", "wcos", "to_roll", "oh_h", "oh_wd", "s_", "sl_", "roll_bp")


def model_columns(cols: list[str], kind: str, with_pair: bool, with_time: bool) -> list[int]:
    """Column positions a model uses: one-hot hour/weekday/pair columns for the logistic regression,
    the integer pair (categorical) for LightGBM, no time-of-day inputs when ``with_time`` is False."""
    out = []
    for j, c in enumerate(cols):
        if not with_time and c.startswith(TIME_PREFIXES):
            continue
        if c.startswith("oh_p_") or c == "pair_id":
            if with_pair and (kind == "logit") == c.startswith("oh_p_"):
                out.append(j)
            continue
        if c.startswith(("oh_h", "oh_wd")) and kind != "logit":
            continue
        out.append(j)
    return out


# ------------------------------------------------------------------ walk-forward

def _lgb_fit(X: np.ndarray, y: np.ndarray, params: dict, rounds: int, cat: list[int]):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y, categorical_feature=cat or "auto", free_raw_data=True)
    return lgb.train({**LGB_BASE, **params}, ds, num_boost_round=rounds)


def _logit_fit(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=500))
    with threadpool_limits(1):                       # one core: the machine is shared
        return model.fit(X, y)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def walk_forward(D: dict, first_test: int = FIRST_TEST, last_tune: int = LAST_TUNE, log=print) -> tuple[dict, dict, dict]:
    """Yearly walk-forward predictions P(up) of every model and horizon (NaN outside the predicted years),
    the hyperparameter search on the tune years, and feature weights (LightGBM gain, logistic
    regression standardised coefficients) summed over the yearly models."""
    X, cols = D["X"], D["cols"]
    n = len(D["t"])
    years = list(range(first_test, int(D["year"].max()) + 1))
    preds = {m: {H: np.full(n, np.nan, np.float32) for H in HORIZONS} for m in MODELS}
    weights = {m: {H: np.zeros(len(cols)) for H in HORIZONS} for m in MODELS}
    tuning: dict = {}
    for H in HORIZONS:
        y = D[f"y{H}"]
        lab = D["valid"] & np.isfinite(y) & (y != 0)
        rows = D["valid"] & np.isfinite(y)

        def split(Y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            start = pd.Timestamp(f"{Y}-01-01", tz="UTC").value
            tr = np.flatnonzero(lab & D["keep"] & (D[f"tend{H}"] < start - EMBARGO_NS))
            return tr, (y[tr] > 0).astype(np.int8), np.flatnonzero(rows & (D["year"] == Y))

        # 1. hyperparameters: walk-forward over the tune years only
        grid_p: dict = {}
        grid_w: dict = {}
        for Y in [Y for Y in years if Y <= last_tune]:
            tr, yt, te = split(Y)
            t0 = time.time()
            cj = model_columns(cols, "lgbm", False, True)
            Xtr, Xte = X[np.ix_(tr, cj)], X[np.ix_(te, cj)]
            for gi, g in enumerate(LGB_GRID):
                booster = _lgb_fit(Xtr, yt, g, max(LGB_ROUNDS), [])
                for r in LGB_ROUNDS:
                    grid_p.setdefault(("lgbm", gi, r), np.full(n, np.nan, np.float32))[te] = booster.predict(Xte, num_iteration=r)
                    w = grid_w.setdefault(("lgbm", gi, r), np.zeros(len(cols)))
                    w[cj] += booster.feature_importance("gain", iteration=r)
            cj = model_columns(cols, "logit", False, True)
            Xtr, Xte = X[np.ix_(tr, cj)], X[np.ix_(te, cj)]
            for C in LOGIT_C:
                model = _logit_fit(Xtr, yt, C)
                grid_p.setdefault(("logit", C), np.full(n, np.nan, np.float32))[te] = model.predict_proba(Xte)[:, 1]
                grid_w.setdefault(("logit", C), np.zeros(len(cols)))[cj] += model[-1].coef_[0]
            del Xtr, Xte
            if log:
                log(f"tune H={H} {Y}: {len(tr):,} training rows, {len(te):,} predicted, {time.time() - t0:.0f}s")
        tune_rows = lab & (D["year"] >= first_test) & (D["year"] <= last_tune)
        up = (y[tune_rows] > 0).astype(int)
        scores = [{"kind": key[0], "params": ({**LGB_GRID[key[1]], "rounds": key[2]} if key[0] == "lgbm" else {"C": key[1]}),
                   "auc": _auc(up, p[tune_rows]),
                   "logloss": float(-np.mean(up * np.log(np.clip(p[tune_rows], 1e-6, 1))
                                             + (1 - up) * np.log(np.clip(1 - p[tune_rows], 1e-6, 1))))}
                  for key, p in grid_p.items()]
        chosen = {}
        for kind in ("lgbm", "logit"):
            best = max((s for s in scores if s["kind"] == kind), key=lambda s: s["auc"])
            chosen[kind] = best["params"]
        tuning[str(H)] = {"grid": scores, "chosen": chosen}
        if log:
            log(f"H={H} chosen: {chosen}")
        # 2. every model with the chosen settings, every year
        for Y in years:
            tr, yt, te = split(Y)
            t0 = time.time()
            for m, (kind, with_pair, with_time) in MODELS.items():
                if Y <= last_tune and m in ("lgbm", "logit"):
                    key = ("lgbm", LGB_GRID.index({k: v for k, v in chosen["lgbm"].items() if k != "rounds"}),
                           chosen["lgbm"]["rounds"]) if kind == "lgbm" else ("logit", chosen["logit"]["C"])
                    preds[m][H][te] = grid_p[key][te]
                    continue
                cj = model_columns(cols, kind, with_pair, with_time)
                Xtr, Xte = X[np.ix_(tr, cj)], X[np.ix_(te, cj)]
                if kind == "lgbm":
                    params = {k: v for k, v in chosen["lgbm"].items() if k != "rounds"}
                    cat = [cj.index(cols.index("pair_id"))] if with_pair else []
                    booster = _lgb_fit(Xtr, yt, params, chosen["lgbm"]["rounds"], cat)
                    preds[m][H][te] = booster.predict(Xte)
                    weights[m][H][cj] += booster.feature_importance("gain")
                else:
                    model = _logit_fit(Xtr, yt, chosen["logit"]["C"])
                    preds[m][H][te] = model.predict_proba(Xte)[:, 1]
                    weights[m][H][cj] += model[-1].coef_[0]
                del Xtr, Xte
            if log:
                log(f"final H={H} {Y}: {len(tr):,} training rows, {len(te):,} predicted, {time.time() - t0:.0f}s")
        # the tune years of the tuned models come from the grid: add their weights
        for m in ("lgbm", "logit"):
            if m == "lgbm":
                key = ("lgbm", LGB_GRID.index({k: v for k, v in chosen["lgbm"].items() if k != "rounds"}),
                       chosen["lgbm"]["rounds"])
            else:
                key = ("logit", chosen["logit"]["C"])
            weights[m][H] += grid_w[key]
    return preds, tuning, weights


# ------------------------------------------------------------------ scoring

def _block_t(v: np.ndarray, block: np.ndarray) -> float | None:
    """t statistic of the mean with weekly blocks (neighbouring hours and pairs are not independent)."""
    if len(v) < 3:
        return None
    _, inv = np.unique(block, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    if len(sums) < 3:
        return None
    sd = sums.std(ddof=1)
    return float(sums.mean() / sd * math.sqrt(len(sums))) if sd > 0 else None


def _score(m: np.ndarray, s: np.ndarray, D: dict, H: int, n_all: int, sea: np.ndarray | None = None,
           full: bool = True) -> dict:
    """Calls ``s`` (+1 / -1) on the rows ``m`` (targets already non-zero) against the move H bars on."""
    y = D[f"y{H}"][m].astype(float)
    ss = s[m]
    n = int(m.sum())
    out: dict = {"n": n, "share": n / n_all if n_all else None}
    if not n:
        return out
    hit = np.sign(y) == ss
    signed = ss * y
    out.update(hit=float(hit.mean()), bp=float(signed.mean()))
    roll = D[f"roll{H}"][m]
    out["roll_share"] = float(roll.mean())
    out["n_ex_roll"] = int((~roll).sum())
    out["hit_ex_roll"] = float(hit[~roll].mean()) if (~roll).any() else None
    if not full:
        return out
    week = D["week"][m]
    out["hit_t"] = _block_t(hit - 0.5, week)
    out["bp_t"] = _block_t(signed, week)
    cost = D[f"cost{H}"][m].astype(float)
    out["cost_med"] = float(np.nanmedian(cost))
    out["net_bp"] = float(np.nanmean(signed - cost))
    out["survives"] = bool(out["bp"] > out["cost_med"])
    # a position held over the 17:00 roll is paid (or pays) the swap, which offsets the forward-point shift
    shift = np.nan_to_num(D[f"rollbp{H}"][m].astype(float))
    out["shift_bp"] = float(np.mean(ss * shift))
    out["net_swap_bp"] = float(np.nanmean(ss * (y - shift) - cost))
    if sea is not None:
        sv = sea[m]
        has = sv != 0
        out["season_share"] = float(has.mean())
        out["season_agree"] = float((sv[has] == ss[has]).mean()) if has.any() else None
        out["season_hit"] = float((np.sign(y[has]) == sv[has]).mean()) if has.any() else None
        out["n_no_season"] = int((~has).sum())
        out["hit_no_season"] = float(hit[~has].mean()) if (~has).any() else None
    return out


def _pct(x: float) -> str:
    return f"{x * 100:g}%"


def summarise(D: dict, preds: dict, first_test: int = FIRST_TEST, last_tune: int = LAST_TUNE) -> dict:
    """Scores of every model and of the time-of-day drift, per horizon and period (tune / test)."""
    out: dict = {"models": {}, "season": {}, "season_long": {}, "roll_rule": {}, "goal": [], "test90": []}
    years = D["year"]
    for H in HORIZONS:
        y = D[f"y{H}"]
        base = D["valid"] & np.isfinite(y) & (y != 0)
        per = {"tune": base & (years >= first_test) & (years <= last_tune), "test": base & (years > last_tune)}
        n_all = {k: int(v.sum()) for k, v in per.items()}
        # the time-of-day drift alone: the server's window and (next hour only) a ~3-year window
        S, sea = _baseline(D, H, D[f"sd{H}"], D[f"st{H}"], per, n_all, base & (years >= first_test))
        out["season"][str(H)] = S
        if H == 1:
            out["season_long"]["1"] = _baseline(D, H, D["sld1"], D["slt1"], per, n_all, base & (years >= first_test))[0]
            out["roll_rule"]["1"] = _baseline(D, H, D["rollbp1"], D["rollbp1"], per, n_all, base & (years >= first_test),
                                              1e-9, ROLL_BP, (1.0, 3.0), "シフト ≥ {:g} bp")[0]
        shift = np.nan_to_num(D["rollbp1"].astype(float))
        rule_call = np.where(np.abs(shift) >= 1.0, np.sign(shift), 0.0)
        for m in MODELS:
            p = preds[m][H].astype(float)
            ok = np.isfinite(p)
            s = np.where(p > 0.5, 1.0, -1.0)
            conf = np.where(ok, np.abs(p - 0.5), -1.0)
            R: dict = {"overall": {}, "tops": {}, "deciles": {}, "years": {}, "scan": [], "by_pair": {}, "by_hour": {}}
            for k, rows in per.items():
                rk = rows & ok
                R["overall"][k] = {**_score(rk, s, D, H, n_all[k], sea),
                                   "auc": _auc((y[rk] > 0).astype(int), p[rk])}
            tune_conf = conf[per["tune"] & ok]
            edges = np.quantile(tune_conf, np.arange(1, 10) / 10)
            dec = np.searchsorted(edges, conf, side="right")
            for k, rows in per.items():
                R["deciles"][k] = [_score(rows & ok & (dec == j), s, D, H, n_all[k], full=False) for j in range(10)]
            R["decile_edges"] = [float(e) for e in edges]
            for x in TOPS:
                thr = float(np.quantile(tune_conf, 1 - x))
                R["tops"][_pct(x)] = {"thr": thr, **{k: _score(rows & ok & (conf >= thr), s, D, H, n_all[k], sea)
                                                     for k, rows in per.items()}}
                if H == 1:                       # how many of the calls are bars with a swap-point shift of >= 1 bp
                    for k, rows in per.items():
                        sel = rows & ok & (conf >= thr)
                        rc, sc = rule_call[sel], s[sel]
                        R["tops"][_pct(x)][k]["rule_share"] = float(np.mean(rc != 0)) if sel.any() else None
                        R["tops"][_pct(x)][k]["rule_agree"] = float(np.mean(rc[rc != 0] == sc[rc != 0])) if (rc != 0).any() else None
                if x in (0.05, 0.01, 0.001):
                    R["years"][_pct(x)] = _by_year(D, H, base & ok & (years >= first_test) & (conf >= thr), s)
                if x == 0.01:
                    test_top = per["test"] & ok & (conf >= thr)
                    for j, code in enumerate(CODES):
                        R["by_pair"][code] = _score(test_top & (D["pair"] == j), s, D, H, n_all["test"], full=False)
                    R["by_hour"] = {int(h): _brief(_score(test_top & (D["hour"] == h), s, D, H, n_all["test"], full=False))
                                    for h in range(24) if (test_top & (D["hour"] == h)).any()}
            for x in SCAN_TOPS:
                thr = float(np.quantile(tune_conf, 1 - x))
                sel = {k: rows & ok & (conf >= thr) for k, rows in per.items()}
                agree = {k: v & (sea == s) for k, v in sel.items()}
                R["scan"].append({"rule": f"上位{_pct(x)}",
                                  **{k: _brief(_score(sel[k], s, D, H, n_all[k], full=False)) for k in per}})
                R["scan"].append({"rule": f"上位{_pct(x)}、時間帯の偏りと同じ向き",
                                  **{k: _brief(_score(agree[k], s, D, H, n_all[k], full=False)) for k in per}})
            out["models"].setdefault(m, {})[str(H)] = R
        baselines = [("season", S["scan"])] + ([(k, out[k]["1"]["scan"]) for k in ("season_long", "roll_rule")] if H == 1 else [])
        for name, blocks in [(m, out["models"][m][str(H)]["scan"]) for m in MODELS] + baselines:
            for row in blocks:
                for variant in ("all", "ex_roll"):
                    tu, te = row["tune"], row["test"]
                    key_hit, key_n = ("hit", "n") if variant == "all" else ("hit_ex_roll", "n_ex_roll")
                    if (tu.get(key_hit) or 0) >= GOAL_HIT and (te.get(key_hit) or 0) >= GOAL_HIT and te.get(key_n, 0) >= GOAL_N:
                        out["goal"].append({"model": name, "H": H, "rule": row["rule"], "hours": variant,
                                            "tune": tu, "test": te})
                    elif variant == "all" and (te.get("hit") or 0) >= GOAL_HIT and te.get("n", 0) >= GOAL_N:
                        out["test90"].append({"model": name, "H": H, "rule": row["rule"], "tune": tu, "test": te})
    out["best"] = _best(out)
    return out


def _baseline(D: dict, H: int, d: np.ndarray, t: np.ndarray, per: dict, n_all: dict, all_years: np.ndarray,
              min_conf: float = season.T_MIN, levels: tuple = SEASON_TS,
              year_levels: tuple = (season.T_MIN, season.T_HIGH), label: str = "|t| ≥ {:g}") -> tuple[dict, np.ndarray]:
    """A rule as a forecast: the sign of ``d`` where |t| >= ``min_conf``, by fixed |t| ``levels`` and by the
    same top shares as the models (thresholds from the tune years)."""
    t, d = np.abs(np.nan_to_num(t.astype(float))), np.nan_to_num(d.astype(float))
    sea = np.where((t >= min_conf) & (d != 0), np.sign(d), 0.0)
    conf = np.where(sea != 0, t, 0.0)
    S: dict = {"fixed": {}, "tops": {}, "years": {}, "scan": []}
    for thr in levels:
        for k, rows in per.items():
            S["fixed"].setdefault(f"{thr:g}", {})[k] = _score(rows & (conf >= thr), sea, D, H, n_all[k])
        r = S["fixed"][f"{thr:g}"]
        S["scan"].append({"rule": label.format(thr), "tune": _brief(r["tune"]), "test": _brief(r["test"])})
    for x in TOPS:
        thr = max(float(np.quantile(conf[per["tune"]], 1 - x)), min_conf)
        S["tops"][_pct(x)] = {"thr": thr, **{k: _score(rows & (conf >= thr) & (sea != 0), sea, D, H, n_all[k])
                                             for k, rows in per.items()}}
    for thr in year_levels:
        S["years"][f"{thr:g}"] = _by_year(D, H, all_years & (conf >= thr) & (sea != 0), sea)
    return S, sea


def _brief(r: dict) -> dict:
    return {k: r.get(k) for k in ("n", "share", "hit", "bp", "roll_share", "n_ex_roll", "hit_ex_roll")}


def _by_year(D: dict, H: int, m: np.ndarray, s: np.ndarray) -> dict:
    y = D[f"y{H}"]
    out = {}
    for Y in np.unique(D["year"][m]):
        k = m & (D["year"] == Y)
        out[str(int(Y))] = {"n": int(k.sum()), "hit": float(np.mean(np.sign(y[k]) == s[k]))}
    return out


def _best(res: dict) -> list[dict]:
    """The subsets with the highest hit rate on the weaker of the two periods, with at least GOAL_N test calls."""
    cands, seen = [], set()
    for name, per_h in list(res["models"].items()) + [(k, res[k]) for k in ("season", "season_long", "roll_rule")]:
        for H, R in per_h.items():
            for row in R["scan"]:
                tu, te = row["tune"], row["test"]
                key = (name, H, tu.get("n"), te.get("n"), tu.get("hit"), te.get("hit"))
                if key in seen:                  # "agrees with the drift" often keeps every call
                    continue
                seen.add(key)
                if te.get("n", 0) >= GOAL_N and tu.get("hit") is not None and te.get("hit") is not None:
                    cands.append({"model": name, "H": int(H), "rule": row["rule"], "tune": tu, "test": te,
                                  "worse": min(tu["hit"], te["hit"])})
    return sorted(cands, key=lambda c: -c["worse"])[:12]


def importance(D: dict, weights: dict, top: int = 20) -> dict:
    """Largest LightGBM gains (share of the total) and logistic regression coefficients per horizon."""
    cols = D["cols"]
    out: dict = {}
    for m, per_h in weights.items():
        for H, w in per_h.items():
            if MODELS[m][0] == "lgbm":
                share = w / w.sum() if w.sum() > 0 else w
                order = np.argsort(-share)[:top]
                out.setdefault(m, {})[str(H)] = [(cols[j], float(share[j])) for j in order if share[j] > 0]
            else:
                order = np.argsort(-np.abs(w))[:top]
                out.setdefault(m, {})[str(H)] = [(cols[j], float(w[j])) for j in order if w[j] != 0]
    return out


# ------------------------------------------------------------------ report

H_NAMES = {"1": "1時間先", "4": "4時間先", "24": "24時間先"}
FEATURE_NOTES = [
    ("z{k}", "過去 k 本の値動き (最近の値動きの荒さで割った値)"),
    ("vol{w} / vol_ewm / volr", "実現ボラティリティ (bp) とその比"),
    ("range / atr / range_atr / body / clv", "足の値幅・ATR・実体・終値の位置"),
    ("ma{n} / pos{n}", "移動平均からの乖離・直近 n 本の高値安値の中での位置"),
    ("rsi / macd_hist / boll / stoch", "RSI・MACD ヒストグラム・ボリンジャーバンド・ストキャスティクス"),
    ("spread / spread_rel / gap_h", "スプレッド (bp)・その直近の中央値との比・前の足からの時間"),
    ("ny_hour / ny_wd / hsin / wsin / to_roll", "次の足のニューヨーク時間の時刻・曜日・ロールオーバーまでの時間"),
    ("s_* / sl_*", "時間帯の偏り (season.py と同じ計算、直近 6,000 本 / 18,000 本): 枠の平均 mu と t、次の H 本の合計 d と t"),
    ("x1_* / x4_* / str_* / usd* / jpy*", "7ペアの直近の値動き・通貨の強さ (ドル・円の要因)"),
    ("roll_bp{H}", "次の H 本でロールオーバーの受け渡し日が進むときに予想される価格の調整 (金利差 × 日数 / 360、bp。時間の入力として扱う)"),
    ("carry / vix / vix_ch5", "金利差 (その時点で分かる値)・VIX (2日遅れ) とその5日変化"),
    ("pair_id / oh_p_*", "通貨ペア (+ペアのモデルだけ)"),
]


def _p(x, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{d}f}%"


def _t(x) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.2f}"


def _cell(r: dict, share: bool = True) -> str:
    if not r.get("n"):
        return "0回"
    return f"{_p(r['hit'])} ({r['n']:,}回" + (f", {_p(r['share'], 2)})" if share else ")")


def _best_model(res: dict, H: str) -> str:
    """The model with the highest AUC on the tune years (never chosen on the test years)."""
    return max(MODELS, key=lambda m: res["models"][m][H]["overall"]["tune"]["auc"])


def compare_season(res: dict) -> list[dict]:
    """Each model's top x % against the time-of-day drift's top x % (both thresholds from the tune years)."""
    rows = []
    for H in res["season"]:
        S = res["season"][H]["tops"]
        for m in list(MODELS) + (["roll_rule"] if H in res.get("roll_rule", {}) else []):
            T = res["roll_rule"][H]["tops"] if m == "roll_rule" else res["models"][m][H]["tops"]
            for x, r in T.items():
                s = S[x]
                beats = all(s[k].get("n", 0) >= 30 and (r[k].get("hit") or 0) >= s[k]["hit"] + BEAT_MARGIN for k in ("tune", "test"))
                rows.append({"model": m, "H": int(H), "top": x, "ml_tune": r["tune"].get("hit"), "ml_test": r["test"].get("hit"),
                             "ml_n_test": r["test"].get("n", 0), "season_tune": s["tune"].get("hit"),
                             "season_test": s["test"].get("hit"), "season_n_test": s["test"].get("n", 0),
                             "beats": bool(beats and r["test"].get("n", 0) >= GOAL_N)})
    return rows


def report(res: dict) -> str:
    meta = res["meta"]
    first_test, last_tune = meta["first_test"], meta["last_tune"]
    hs = [str(H) for H in HORIZONS]
    tops = [_pct(x) for x in TOPS]
    best = {H: _best_model(res, H) for H in hs}
    L = ["# 約20年の1時間足による機械学習の方向予測 (ウォークフォワード検証)", "",
         "これまでの研究 (Yahoo の約2.8年分の1時間足) では、機械学習・ディープラーニングの方向の的中率は約50%で、はっきり当たったのは"
         "時間帯の偏り (ニューヨーク時間のロールオーバー前後、aifx/season.py) だけでした ([ml.md](ml.md)、[dl.md](dl.md)、"
         "[direction.md](direction.md))。ここでは Dukascopy の約20年分の1時間足 (7ペア) を使い、機械学習で方向を偶然より、"
         "そして時間帯の偏りより当てられるか、特に「自信のある予測に絞れば90%当たるか」を、未来のデータが混ざらない形で確かめました。", ""]
    # ---- conclusions
    L += ["## 結論", ""]
    ov = {H: res["models"][best[H]][H]["overall"] for H in hs}
    hits = [ov[H]["test"]["hit"] for H in hs]
    level = ("ほぼ偶然 (50%) と同じ" if max(hits) < 0.52 else "偶然 (50%) をわずかに上回る程度" if max(hits) < 0.55
             else "偶然 (50%) を上回ります")
    L.append(f"- **全体の的中率は{level}です。** 検証期間 ({last_tune + 1}年〜) の的中率 / AUC (0.5 がでたらめ)、調整期間の AUC で選んだモデル: "
             + "、".join(f"{H_NAMES[H]} {MODEL_NAMES[best[H]]} {_p(ov[H]['test']['hit'])} / {ov[H]['test']['auc']:.3f}"
                        f" (調整期間 {_p(ov[H]['tune']['hit'])} / {ov[H]['tune']['auc']:.3f})" for H in hs) + "。")
    b1 = res["models"][best["1"]]["1"]
    s1 = res["season"]["1"]
    top1 = b1["tops"]["1%"]["test"]
    if (top1.get("season_share") or 0) >= 0.6 and (top1.get("season_agree") or 0) >= 0.9:
        head = "確信度で絞ると的中率は上がりますが、上がる分の大部分は時間帯の偏りです。"
    elif (top1.get("roll_share") or 0) >= 0.5:
        head = "確信度で絞ると的中率は上がりますが、上位はロールオーバー前後の足が中心です。"
    else:
        head = "確信度で絞ったときの的中率:"
    L.append(f"- **{head}** 1時間先 ({MODEL_NAMES[best['1']]}) の上位 "
             + " / ".join(tops) + " の検証期間の的中率: " + " / ".join(_p(b1["tops"][x]["test"].get("hit")) for x in tops)
             + " (調整期間 " + " / ".join(_p(b1["tops"][x]["tune"].get("hit")) for x in tops) + ")。"
             + "時間帯の偏りだけで同じ割合に絞ると、検証期間 " + " / ".join(_p(s1["tops"][x]["test"].get("hit")) for x in tops)
             + " (調整期間 " + " / ".join(_p(s1["tops"][x]["tune"].get("hit")) for x in tops) + ")。"
             + f"上位1%のうち {_p(b1['tops']['1%']['test'].get('roll_share'), 0)} はロールオーバーの直前・直後の足 "
             f"(ニューヨーク時間16時・17時に始まる足) で、{_p(b1['tops']['1%']['test'].get('season_share'), 0)} は時間帯の偏りも方向を示す足"
             f" (うち {_p(b1['tops']['1%']['test'].get('season_agree'), 0)} が同じ向き) でした。")
    nt = res["models"]["lgbm_notime"]["1"]
    wt = res["models"]["lgbm_pair"]["1"]
    keep_ = (nt["overall"]["test"]["auc"] - 0.5) / max(wt["overall"]["test"]["auc"] - 0.5, 1e-9)
    drop = (wt["tops"]["1%"]["test"].get("hit") or 0) - (nt["tops"]["1%"]["test"].get("hit") or 0)
    auc_word = "ほぼ消え" if keep_ < 0.4 else "弱まり" if keep_ < 0.8 else "少し下がるだけ"
    effect = (f"確信度の高い予測の的中率は大きく下がります (全体の AUC は{auc_word}{'です' if keep_ >= 0.8 else 'ます'})" if drop >= 0.05
              else f"1時間先の力は{auc_word}{'です' if keep_ >= 0.8 else 'ます'}")
    L.append(f"- **時間の入力 (時刻・曜日・時間帯の偏り・スワップ調整) を外すと、{effect}**: LightGBM (+ペア) の検証期間の AUC "
             f"{wt['overall']['test']['auc']:.3f} → {nt['overall']['test']['auc']:.3f}、上位1%の的中率 "
             f"{_p(wt['tops']['1%']['test'].get('hit'))} → {_p(nt['tops']['1%']['test'].get('hit'))}、上位0.1% "
             f"{_p(wt['tops']['0.1%']['test'].get('hit'))} → {_p(nt['tops']['0.1%']['test'].get('hit'))}。"
             f"時間帯の偏りが方向を示さない足に限ると、{MODEL_NAMES[best['1']]} の上位1%の検証期間の的中率は "
             f"{_p(b1['tops']['1%']['test'].get('hit_no_season'))} ({b1['tops']['1%']['test'].get('n_no_season', 0):,}回) です。"
             + ("時間の入力なしのモデルで効いた入力は " + "、".join(f"`{c}`" for c, _ in res["importance"]["lgbm_notime"]["1"][:3])
                + f" で、その上位1%の予測した向きへの平均の値動きは {nt['tops']['1%']['test'].get('bp', float('nan')):+.2f} bp、"
                f"スプレッドの中央値 {nt['tops']['1%']['test'].get('cost_med', float('nan')):.2f} bp です。"
                if res.get("importance", {}).get("lgbm_notime", {}).get("1") else ""))
    far, both_pos = [], []
    for H in ("4", "24"):
        r = res["models"][best[H]][H]
        tu, te = r["tops"]["1%"]["tune"], r["tops"]["1%"]["test"]
        far.append(f"{H_NAMES[H]} ({MODEL_NAMES[best[H]]}) 全体 {_p(r['overall']['test']['hit'])} (平均の値動きの t {_t(r['overall']['test'].get('bp_t'))}、"
                   f"調整期間 t {_t(r['overall']['tune'].get('bp_t'))})、上位1% 検証 {_cell(te, False)} / 調整 {_cell(tu, False)}、"
                   f"平均 {te.get('bp', float('nan')):+.2f} / {tu.get('bp', float('nan')):+.2f} bp、スプレッドとスワップを引いた平均 "
                   f"{te.get('net_swap_bp', float('nan')):+.2f} / {tu.get('net_swap_bp', float('nan')):+.2f} bp (検証 / 調整)、"
                   f"ロールオーバーを含む予測 {_p(te.get('roll_share'), 0)}")
        if (te.get("net_swap_bp") or 0) > 0 and (tu.get("net_swap_bp") or 0) > 0:
            both_pos.append(H)
    L.append("- **4時間先・24時間先**: " + "; ".join(far) + "。"
             + ("".join(f"{H_NAMES[H]}の上位は、スプレッドとスワップを引いても両方の期間でわずかにプラスでした。" for H in both_pos)
                + "ただしプラス幅は数 bp で、Dukascopy の気配のスプレッド (業者の実際のコストより狭い) で測った値です。" if both_pos else ""))
    goal = res["goal"]
    if goal:
        g_roll = all(g["model"] == "roll_rule" or (g["test"].get("roll_share") or 0) >= 0.5 for g in goal if g["hours"] == "all")
        g_ex = [g for g in goal if g["hours"] == "ex_roll" and g["model"] != "roll_rule"]
        L.append(f"- **「90%」の条件 (調整期間・検証期間とも的中率90%以上、検証期間で{GOAL_N}回以上) を満たす絞り方は {len(goal)} 通りありました**: "
                 + "; ".join(f"{MODEL_NAMES.get(g['model'], g['model'])} {H_NAMES[str(g['H'])]} {g['rule']}"
                             f"{' (ロールオーバー前後を除く)' if g['hours'] == 'ex_roll' else ''}: 調整 "
                             f"{_p(g['tune']['hit' if g['hours'] == 'all' else 'hit_ex_roll'])} / 検証 "
                             f"{_p(g['test']['hit' if g['hours'] == 'all' else 'hit_ex_roll'])} "
                             f"({g['test']['n' if g['hours'] == 'all' else 'n_ex_roll']:,}回、全体の {_p(g['test']['share'], 2)})"
                             for g in goal[:6]) + "。"
                 + (" **どれもロールオーバーの直後の足で、受け渡し日が進むときのスワップポイント分の価格の調整 (下の「ロールオーバーの価格調整」) を当てているだけです。**"
                    "ポジションを持てばその分はスワップで相殺され、この時間はスプレッドも広いため、取引の利益にはなりません。" if g_roll and not g_ex else ""))
    else:
        c = res["best"][0] if res["best"] else None
        L.append(f"- **「90%」の条件 (調整期間・検証期間とも的中率90%以上、検証期間で{GOAL_N}回以上) を満たす絞り方はありませんでした**"
                 f" (モデル {len(MODELS)} 種類 × 予測先 3 × 絞り方 {len(SCAN_TOPS) * 2} 通り、時間帯の偏り (窓2通り) の |t| の基準 {len(SEASON_TS)} 通り、"
                 f"スワップ調整の規則 {len(ROLL_BP)} 段階、ロールオーバー前後を除いた場合も含む)。"
                 + (f"最も近いのは {MODEL_NAMES.get(c['model'], c['model'])} {H_NAMES[str(c['H'])]} {c['rule']} で、調整期間 "
                    f"{_p(c['tune']['hit'])} ({c['tune']['n']:,}回) / 検証期間 {_p(c['test']['hit'])} ({c['test']['n']:,}回、"
                    f"全体の {_p(c['test']['share'], 2)}、うちロールオーバー前後 {_p(c['test'].get('roll_share'), 0)})。" if c else ""))
    t90 = sorted(res.get("test90", []), key=lambda g: -(g["tune"].get("hit") or 0))
    if t90:
        L.append(f"- **検証期間だけなら90%に届く絞り方はあります** ({len(t90)} 通り) が、調整期間では届きません: "
                 + "; ".join(f"{MODEL_NAMES.get(g['model'], g['model'])} {H_NAMES[str(g['H'])]} {g['rule']}: 検証 {_cell(g['test'], False)}、"
                             f"調整 {_cell(g['tune'], False)}、検証期間のうちロールオーバーを含む予測 {_p(g['test'].get('roll_share'), 0)}"
                             for g in t90[:4])
                 + ("。どれもロールオーバーを含む予測で、ロールオーバーの価格調整 (下の「ロールオーバーの価格調整」) を当てています。年ごとの表の"
                    "とおり、当たりは金利差が大きく、ロールオーバーの気配が落ち着いていた最近の年に集中しています。"
                    if all(g["model"] == "roll_rule" or (g["test"].get("roll_share") or 0) >= 0.5 for g in t90[:4]) else "。"))
    rr = res.get("roll_rule", {}).get("1")
    if rr:
        lv = [k for k in ("1", "3") if k in rr["fixed"]]
        L.append("- **ロールオーバーのスワップ調整 (規則)**: ニューヨーク時間17時に始まる足で「金利の高い通貨が、金利差 × 日数 / 360 だけ下がる」"
                 "(水曜日は3日分) と予想する規則は、" + "、".join(
                     f"予想する動き {k} bp 以上で 調整期間 {_cell(rr['fixed'][k]['tune'], False)} / 検証期間 {_cell(rr['fixed'][k]['test'], False)}"
                     for k in lv) + "。"
                 + f"{MODEL_NAMES[best['1']]} の1時間先の上位1% (検証期間) のうち {_p(top1.get('rule_share'), 0)} が予想する調整 1 bp 以上の足"
                 f" (うち {_p(top1.get('rule_agree'), 0)} が規則と同じ向き)、上位0.1% では {_p(b1['tops']['0.1%']['test'].get('rule_share'), 0)}"
                 f" (同じ向き {_p(b1['tops']['0.1%']['test'].get('rule_agree'), 0)}) です。")
    t1 = b1["tops"]["1%"]["test"]
    net_pos = (t1.get("net_bp") or 0) > 0
    head = ("スプレッドを引いても残りますが、取引の利益とは限りません。" if t1.get("survives") and net_pos else
            "スプレッドの中央値はわずかに上回りますが、平均のコストを引くとマイナスです。" if t1.get("survives") else
            "売買の利益にはなりません。")
    L.append(f"- **{head}** 1時間先の上位1% ({MODEL_NAMES[best['1']]}) の予測した向きへの平均の値動きは "
             f"{t1.get('bp', float('nan')):+.2f} bp、その足の売買にかかるスプレッド (入口と出口の平均) の中央値は "
             f"{t1.get('cost_med', float('nan')):.2f} bp"
             f" (1回ごとのコストを引いた平均 {t1.get('net_bp', float('nan')):+.2f} bp、さらにロールオーバーをまたぐときのスワップを引くと "
             f"{t1.get('net_swap_bp', float('nan')):+.2f} bp)。ロールオーバーの前後はスプレッドが広がり、価格のずれはスワップで相殺されるため、"
             "表示される価格としては当たっても、取引の利益にはなりにくい動きです。")
    beats = [r for r in compare_season(res) if r["beats"]]
    if beats:
        L.append("- **時間帯の偏りを、同じ割合に絞って調整期間・検証期間とも上回った組み合わせ**: "
                 + "; ".join(f"{MODEL_NAMES[r['model']]} {H_NAMES[str(r['H'])]} 上位{r['top']}: 検証 {_p(r['ml_test'])} ({r['ml_n_test']:,}回) 対 "
                             f"{_p(r['season_test'])} ({r['season_n_test']:,}回)、調整 {_p(r['ml_tune'])} 対 {_p(r['season_tune'])}"
                             for r in beats[:4]) + (f" ほか {len(beats) - 4} 通り" if len(beats) > 4 else "") + "。")
    else:
        L.append("- **時間帯の偏りを、同じ割合に絞って調整期間・検証期間の両方で上回った組み合わせはありません**"
                 f" (検証期間で{GOAL_N}回以上のもの)。")
    L.append(f"- **本番への組み込み**: {res['recommendation']}")
    L.append("")
    # ---- data and method
    spans = meta["spans"]
    L += ["## データと方法", "",
          "- **データ**: Dukascopy の1時間足 (買値と売値の平均の四本値と、終値の時点のスプレッド)。"
          + "、".join(f"{c} {spans[c][0][:10]}〜{spans[c][1][:10]}" for c in CODES)
          + f"。予測の起点は {START[:4]}年からの各足の終値 (計 {meta['rows']:,}行、7ペア合計)。次の足が1時間後に始まらない起点 "
          "(週末・休日の前) は除きました。動きがゼロの回は的中率から除きます。",
          "- **特徴** (すべて起点の足の終値までのデータから計算。重要度の表の名前):",
          *[f"  - `{k}`: {v}" for k, v in FEATURE_NOTES],
          f"- **時間帯の偏りの特徴**は aifx/season.py と同じ計算 (起点の日の0時 (UTC) より前の {season.WINDOW:,} 本、ニューヨーク時間の"
          "曜日×時間と時間の枠) をまとめて行う形に書き直し、無作為に選んだ起点で season.py の結果と一致することを確かめました "
          f"({meta['season_check']['origins']}起点 × 3つの予測先、差の最大 {meta['season_check']['max_abs_diff']:.1e}、向きの不一致 "
          f"{meta['season_check']['sign_mismatches']})。",
          "- **予測するもの**: 起点の終値から 1・4・24 本後の終値までが上か下か。",
          "- **モデル**: 標準化したロジスティック回帰 (L2 正則化) と、小さく正則化した LightGBM (学習率 0.05、1葉あたり最低2,000件、"
          "特徴・行の間引き、L2 = 10)。どちらも7ペアをまとめて1つのモデルで学習し、通貨ペアを入力に「入れない / 入れる」の2通りを試しました。"
          "さらに、時刻・曜日・時間帯の偏り・スワップ調整の入力をすべて外した LightGBM (+ペア) で、時間帯以外に何が残るかを確かめました。"
          "比較の基準は **時間帯の偏りだけ** (過去の枠の平均の向き。|t| ≥ 2 で方向を示し、|t| ≥ 4 を高確度とする。4本・24本先は次の H 本の"
          "偏りの合計とその t)。次の1時間については、枠の平均を直近約3年 (18,000本) で測った時間帯の偏りと、ロールオーバーのスワップ調整の"
          "規則 (下の「ロールオーバーの価格調整」) も比べました。",
          f"- **ウォークフォワード**: {first_test}年から毎年、その年より前のデータだけで学習し直してその年を予測しました。学習データの"
          f"予測先 (H 本後の終値) がその年の始まりの1日前までに収まる行だけを使っています (パージ)。隣り合う時間の行はほとんど同じ情報なので、"
          f"学習には行の {TRAIN_KEEP:.0%} を無作為に選んで使いました (予測・評価はすべての行)。",
          f"- **調整期間と検証期間**: {first_test}〜{last_tune}年の予測を「調整期間」、{last_tune + 1}年以降を「検証期間」とします。"
          "ハイパーパラメータ (LightGBM の葉の数と学習回数、ロジスティック回帰の正則化の強さ) は調整期間の AUC だけで選び、"
          "確信度 (|p − 0.5|) で絞る区切り (上位10% / 5% / 1% / 0.1%) も調整期間の予測から決めて、検証期間にはそのまま当てはめました。"
          "そのため検証期間で実際に絞られる割合は目標からずれます (表の「割合」)。検証期間の成績は、ロールオーバーの価格調整の規則と入力を"
          "加えたこと (下の「ロールオーバーの価格調整」) を除き、何の選択にも使っていません。",
          "- **成績の見方**: 的中率は方向が当たった割合、AUC は予測確率の順番の正しさ (0.5 = でたらめ)、平均 (bp) は予測した向きの平均の"
          "値動き (1 bp = 0.01%、コスト抜き)。t 値は週ごとに全ペアの結果を合計して計算しました (隣り合う足やペアは独立ではないため)。"
          "コストは、起点と H 本後の足の終値の時点のスプレッドの平均 (買値と売値の平均で測った値動きに対し、1往復で払う額) です。",
          f"- 計算時間 {meta['seconds'] / 60:.0f}分 (1スレッド)。", ""]
    # ---- overall
    L += ["## 全体の成績", "", "| モデル | 予測先 | 期間 | 件数 | 的中率 (t) | AUC | 平均 bp (t) |", "|---|---|---|---|---|---|---|"]
    for H in hs:
        for m in MODELS:
            for k, name in (("tune", "調整"), ("test", "検証")):
                r = res["models"][m][H]["overall"][k]
                L.append(f"| {MODEL_NAMES[m]} | {H_NAMES[H]} | {name} | {r['n']:,} | {_p(r['hit'])} ({_t(r.get('hit_t'))}) | "
                         f"{r['auc']:.3f} | {r['bp']:+.2f} ({_t(r.get('bp_t'))}) |")
        for base_name, label in (("season", "時間帯の偏り \\|t\\| ≥ {}"), ("season_long", "時間帯の偏り (18,000本) \\|t\\| ≥ {}"),
                                 ("roll_rule", "スワップ調整 ≥ {} bp")):
            if H not in res[base_name]:
                continue
            for thr in (("1", "3") if base_name == "roll_rule" else ("2", "4")):
                for k, name in (("tune", "調整"), ("test", "検証")):
                    r = res[base_name][H]["fixed"][thr][k]
                    if r.get("n"):
                        L.append(f"| {label.format(thr)} | {H_NAMES[H]} | {name} | {r['n']:,} ({_p(r['share'])}) | "
                                 f"{_p(r['hit'])} ({_t(r.get('hit_t'))}) | — | {r['bp']:+.2f} ({_t(r.get('bp_t'))}) |")
    L.append("")
    # ---- top confidence
    L += ["## 確信度で絞った的中率", "",
          "各マスは「的中率 (回数, 全体に占める割合)」。区切りは調整期間で決め、検証期間にそのまま当てはめています。時間帯の偏りは |t| の"
          "大きい順に同じ割合 (区切りは |t| ≥ 2 以上) に絞ったものです。「コスト」は検証期間の平均の値動き (bp) / スプレッドの中央値 (bp)、"
          "「ロール」は検証期間の予測のうち、予測する H 本の足にロールオーバーの直前・直後の足 (ニューヨーク時間16時・17時に始まる足) "
          "が含まれるものの割合 (24時間先は常に含まれます)、「ロール以外」はそれを除いた的中率です。「スワップ込み」は、予測した向きにポジションを持ったとして、スプレッドと、ロールオーバーを"
          "またぐときのスワップ (金利差 × 日数 / 360。業者の手数料分は含まない) を引いた1回あたりの平均 (bp、検証期間) です。", ""]
    for H in hs:
        L += [f"### {H_NAMES[H]}", "", "| モデル | 上位 | 調整期間 | 検証期間 | コスト (bp) | スワップ込み | ロール | ロール以外 (検証) |",
              "|---|---|---|---|---|---|---|---|"]
        for m in list(MODELS) + ["season"] + ([k for k in ("season_long", "roll_rule") if H in res[k]]):
            T = res["models"][m][H]["tops"] if m in MODELS else res[m][H]["tops"]
            for x in tops:
                r, te = T[x], T[x]["test"]
                cost = f"{te['bp']:+.2f} / {te['cost_med']:.2f}" if te.get("n") else "—"
                swap = f"{te['net_swap_bp']:+.2f}" if te.get("n") else "—"
                L.append(f"| {MODEL_NAMES[m]} | {x} | {_cell(r['tune'])} | {_cell(te)} | {cost} | {swap} | {_p(te.get('roll_share'), 0)} | "
                         f"{_p(te.get('hit_ex_roll'))} ({te.get('n_ex_roll', 0):,}回) |")
        L.append("")
    # ---- the roll
    rr = res.get("roll_rule", {}).get("1")
    if rr:
        L += ["## ロールオーバーの価格調整 (スワップポイント)", "",
              "ニューヨーク時間17時に、価格の受け渡し日が1日 (水曜日は週末をまたぐため3日) 先に進みます。受け渡し日が遅い価格ほど、"
              "金利の高い通貨が安くなる (先渡しのディスカウント、金利平価) ため、17時以降の気配は、金利の高い方の通貨がおよそ"
              "「金利差 × 日数 / 360」だけ下がった水準に移ります。ポジションを持ち越すと、この分はスワップの受け取り・支払いでちょうど相殺されます。"
              "つまり向きの予測としては当たっても、利益にはならない値動きです。",
              "",
              "「ロールオーバーのスワップ調整 (規則)」は、17時に始まる足 (次の1時間) で、金利の高い通貨が下がる向きを予想し、"
              "予想する動きの大きさ (bp、金利差はその時点で分かる値) を確信度とします。機械学習の入力 `roll_bp1` / `roll_bp4` / `roll_bp24` "
              "も同じ計算 (次の H 本の合計) です。**この規則と入力は、USDJPY・EURJPY の2ペアで予備的に計算した結果 (検証期間を含む) を見た後に"
              "加えました。** 向きは金利平価から決まり、基準は調整期間の成績だけで比べられるよう全段階を示していますが、事前に決めていた他のモデルより"
              "割り引いて見てください。", "",
              "| 予想する動き | 調整期間 | 検証期間 | 平均 bp / スプレッド (検証) | スワップ込み (検証) |", "|---|---|---|---|---|"]
        for k, v in rr["fixed"].items():
            te = v["test"]
            L.append(f"| {k} bp 以上 | {_cell(v['tune'])} | {_cell(te)} | "
                     + (f"{te['bp']:+.2f} / {te['cost_med']:.2f} | {te['net_swap_bp']:+.2f} |" if te.get("n") else "— | — |"))
        Y = rr["years"]
        years = sorted(set().union(*[set(v) for v in Y.values()]))
        L += ["", "年ごとの的中率 (回数):", "", "| 年 | " + " | ".join(f"{k} bp 以上" for k in Y) + " |", "|---" * (len(Y) + 1) + "|"]
        L += [f"| {y}{' (検証)' if int(y) > last_tune else ''} | "
              + " | ".join(f"{_p(Y[k][y]['hit'])} ({Y[k][y]['n']:,})" if y in Y[k] else "—" for k in Y) + " |" for y in years]
        L.append("")
        lf = res.get("live_feed_roll") or {}
        if lf.get("by_hour"):
            L += [f"本番と同じ Yahoo の1時間足 ({lf['span'][0]}〜{lf['span'][1]}、7ペア) で、同じ向き (金利の高い通貨が下がる) を"
                  "ニューヨーク時間16時・17時・18時に始まる足で確かめた的中率 (回数):", "",
                  "| 足の始まり | 1 bp 以上 | 2 bp 以上 | 3 bp 以上 |", "|---|---|---|---|"]
            for h, v in lf["by_hour"].items():
                L.append(f"| {h}時 | " + " | ".join(f"{_p(v[k]['hit'])} ({v[k]['n']:,})" if v[k]["n"] else "—" for k in ("1", "2", "3")) + " |")
            L.append("")
    # ---- same bars
    L += ["## 時間帯の偏りとの比較 (同じ足で)", "",
          "確信度の上位に入った足 (検証期間) について、時間帯の偏り (|t| ≥ 2) が方向を示していた割合、そのうち向きが同じだった割合、"
          "その足での時間帯の偏りの的中率、偏りが方向を示さなかった足だけでのモデルの的中率です。", "",
          "| モデル | 予測先 | 上位 | モデルの的中率 | 偏りも方向を示した | 同じ向き | 偏りの的中率 (同じ足) | 偏りなしの足での的中率 |",
          "|---|---|---|---|---|---|---|---|"]
    for H in hs:
        for m in MODELS:
            for x in tops:
                te = res["models"][m][H]["tops"][x]["test"]
                if te.get("n"):
                    L.append(f"| {MODEL_NAMES[m]} | {H_NAMES[H]} | {x} | {_cell(te, False)} | {_p(te.get('season_share'), 0)} | "
                             f"{_p(te.get('season_agree'), 0)} | {_p(te.get('season_hit'))} | {_p(te.get('hit_no_season'))} "
                             f"({te.get('n_no_season', 0):,}回) |")
    L.append("")
    L += ["上位1% (検証期間) の中身: ニューヨーク時間 (次の足の始まり) ごとの回数と的中率 (回数の多い順、上位6つ)、通貨ペアごとの的中率。", ""]
    for H in hs:
        m = best[H]
        R = res["models"][m][H]
        hours = sorted(R["by_hour"].items(), key=lambda kv: -kv[1]["n"])[:6]
        L.append(f"- {H_NAMES[H]} ({MODEL_NAMES[m]}): " + "、".join(f"{h}時 {v['n']:,}回 {_p(v['hit'])}" for h, v in hours)
                 + " / " + "、".join(f"{c} {_p(v.get('hit'))} ({v['n']:,})" for c, v in R["by_pair"].items() if v.get("n")))
    L.append("")
    # ---- deciles
    L += ["## 確信度の十分位", "", "調整期間の予測で区切った確信度の十分位 (10 が最も自信のある10%) ごとの的中率 (割合)。モデルは調整期間の AUC で選んだもの。", ""]
    for H in hs:
        m = best[H]
        D = res["models"][m][H]["deciles"]
        L += [f"**{H_NAMES[H]} ({MODEL_NAMES[m]})**", "", "| 十分位 | " + " | ".join(str(j + 1) for j in range(10)) + " |",
              "|---" * 11 + "|"]
        for k, name in (("tune", "調整"), ("test", "検証")):
            L.append(f"| {name} | " + " | ".join(f"{_p(d.get('hit'))} ({_p(d.get('share'), 0)})" if d.get("n") else "—"
                                                  for d in D[k]) + " |")
        L.append("")
    # ---- per year
    L += ["## 年ごとの安定性", "", "確信度の上位 (区切りは調整期間で決めた値) の年ごとの的中率 (回数)。", ""]
    for H in hs:
        m = best[H]
        Y = res["models"][m][H]["years"]
        Sy = res["season"][H]["years"]
        years = sorted(set(Y["1%"]) | set(Y["0.1%"]) | set(Y["5%"]) | set(Sy.get("4", {})))
        L += [f"**{H_NAMES[H]} ({MODEL_NAMES[m]})**", "",
              "| 年 | 上位5% | 上位1% | 上位0.1% | 時間帯の偏り \\|t\\| ≥ 4 | 時間帯の偏り \\|t\\| ≥ 2 |", "|---|---|---|---|---|---|"]
        for y in years:
            cells = [Y[x].get(y) for x in ("5%", "1%", "0.1%")] + [Sy.get("4", {}).get(y), Sy.get("2", {}).get(y)]
            L.append(f"| {y}{' (検証)' if int(y) > last_tune else ''} | "
                     + " | ".join(f"{_p(c['hit'])} ({c['n']:,})" if c else "—" for c in cells) + " |")
        L.append("")
    # ---- 90 % scan
    L += ["## 90%に届く絞り方はあるか", "",
          f"確信度の区切り {len(SCAN_TOPS)} 段階 (上位20%〜0.05%)、それぞれ「時間帯の偏りと同じ向きのときだけ」も加え、時間帯の偏り (直近6,000本と"
          f"18,000本) は |t| の基準 {len(SEASON_TS)} 段階、スワップ調整の規則は予想する動き {len(ROLL_BP)} 段階、ロールオーバー前後を除いた場合も"
          f"含めて、調整期間・検証期間の両方で的中率が90%以上、"
          f"検証期間で{GOAL_N}回以上になるものを探しました。", ""]
    if res["goal"]:
        L += ["| モデル | 予測先 | 絞り方 | 時間 | 調整期間 | 検証期間 | ロール (検証) | 平均 bp (検証) |", "|---|---|---|---|---|---|---|---|"]
        for g in res["goal"]:
            k_hit, k_n = ("hit", "n") if g["hours"] == "all" else ("hit_ex_roll", "n_ex_roll")
            L.append(f"| {MODEL_NAMES.get(g['model'], g['model'])} | {H_NAMES[str(g['H'])]} | {g['rule']} | "
                     f"{'すべて' if g['hours'] == 'all' else 'ロール以外'} | {_p(g['tune'][k_hit])} ({g['tune'][k_n]:,}) | "
                     f"{_p(g['test'][k_hit])} ({g['test'][k_n]:,}) | {_p(g['test'].get('roll_share'), 0)} | {g['test']['bp']:+.2f} |")
    else:
        L.append("**該当なし。** 両方の期間で的中率が高かった順 (低い方の期間の的中率の順):")
    L += ["", "| モデル | 予測先 | 絞り方 | 調整期間 | 検証期間 | ロール (検証) | ロール以外 (検証) |", "|---|---|---|---|---|---|---|"]
    for c in res["best"]:
        te = c["test"]
        L.append(f"| {MODEL_NAMES.get(c['model'], c['model'])} | {H_NAMES[str(c['H'])]} | {c['rule']} | {_cell(c['tune'])} | {_cell(te)} | "
                 f"{_p(te.get('roll_share'), 0)} | {_p(te.get('hit_ex_roll'))} ({te.get('n_ex_roll', 0):,}回) |")
    L.append("")
    # ---- importance
    L += ["## 特徴の重要度", "", "LightGBM (+ペア) の分岐による損失の減少 (gain) の割合 (全年のモデルの合計、上位15)。", ""]
    for H in hs:
        imp = res["importance"]["lgbm_pair"][H][:15]
        L.append(f"- {H_NAMES[H]}: " + "、".join(f"`{c}` {v:.1%}" for c, v in imp))
    L += ["", "時間の入力なしの LightGBM (+ペア): " + "; ".join(
        f"{H_NAMES[H]} " + "、".join(f"`{c}` {v:.1%}" for c, v in res["importance"]["lgbm_notime"][H][:8]) for H in hs), "",
          "ロジスティック回帰 (標準化した係数、全年の合計、上位10): " + "; ".join(
              f"{H_NAMES[H]} " + "、".join(f"`{c}` {v:+.2f}" for c, v in res["importance"]["logit"][H][:10]) for H in hs), ""]
    # ---- tuning
    L += ["## ハイパーパラメータの選択 (調整期間の AUC)", "", "| 予測先 | モデル | 設定 | AUC | 対数損失 |", "|---|---|---|---|---|"]
    for H in hs:
        T = res["tuning"][H]
        for g in sorted(T["grid"], key=lambda g: (g["kind"], -g["auc"])):
            chosen = g["params"] == T["chosen"][g["kind"]]
            L.append(f"| {H_NAMES[H]} | {'LightGBM' if g['kind'] == 'lgbm' else 'ロジスティック回帰'} | "
                     f"{', '.join(f'{k}={v}' for k, v in g['params'].items())}{' **(採用)**' if chosen else ''} | {g['auc']:.4f} | {g['logloss']:.4f} |")
    L += ["", "## 注意", "",
          "- 検証期間だけでも数百通りの数字を見ているため、t ≈ 2 の結果が少しあっても偶然で説明できます。判断は調整期間と検証期間の両方で"
          "同じ結果になるかで行いました。",
          "- 的中率は「価格の向き」を当てた割合で、利益ではありません。特にロールオーバーの前後は、スプレッドの拡大とスワップの付与により、"
          "表示される価格が一方向にずれやすい時間で、そのずれは取引の利益にはなりません。",
          "- Dukascopy の価格は1つの業者の気配 (買値・売値の平均) で、Yahoo や他の業者とは細かな動きが違います。", ""]
    return "\n".join(L)


def live_feed_roll_check(root: Path | None = None) -> dict:
    """The swap-point rule on the live feed (Yahoo hourly bars, the last ~2.8 years): hit rate of the
    call "the higher-yielding currency falls" for the bars that start at 16, 17 and 18 New York time
    (the rule uses 17), by the expected shift. Empty if the Yahoo files are missing."""
    rows = []
    for code, pair in PAIRS.items():
        try:
            df = history.load_hourly(code, root)
        except OSError:
            return {}
        df = df[~df.index.duplicated()].sort_index()
        ns = df.index.as_unit("ns").asi8
        r = np.full(len(df), np.nan)
        r[1:] = np.diff(np.log(df["close"].to_numpy(float))) * 1e4
        r[1:][np.diff(ns) != HOUR_NS] = np.nan
        days = pd.to_datetime(ns // DAY_NS * DAY_NS)
        rates = history.rates_panel(pd.DatetimeIndex(sorted(set(days))))
        carry = (rates[pair.base] - rates[pair.quote]).reindex(days).to_numpy()
        ny = df.index.tz_convert(season.NEW_YORK)
        shift = -carry * np.where(ny.dayofweek == 2, 3.0, 1.0) / 360 * 100
        rows.append(pd.DataFrame({"r": r, "shift": shift, "hour": ny.hour}))
    R = pd.concat(rows)
    R = R[np.isfinite(R.r) & (R.r != 0) & np.isfinite(R["shift"])]
    out = {"span": [str(df.index[0])[:10], str(df.index[-1])[:10]], "by_hour": {}}
    for h in (16, 17, 18):
        g = R[R.hour == h]
        for lv in (1.0, 2.0, 3.0):
            k = g[g["shift"].abs() >= lv]
            out["by_hour"].setdefault(str(h), {})[f"{lv:g}"] = {
                "n": int(len(k)), "hit": float(np.mean(np.sign(k.r) == np.sign(k["shift"]))) if len(k) else None}
    return out


def recommend(res: dict) -> str:
    """Integration advice. A candidate must beat the time-of-day drift at the same share on both periods (by
    BEAT_MARGIN, at least GOAL_N test calls); a model must also beat the swap-point rule where the rule reaches
    that share. Even then only a shadow trial on the live feed is advised: the models learned Dukascopy's
    quotes (with their spread), and the live Yahoo bars show the roll at different hours."""
    beats = [r for r in compare_season(res) if r["beats"]]
    rr = res.get("roll_rule", {}).get("1", {}).get("tops", {})

    def rule_reaches(x: str) -> bool:
        return x in rr and (rr[x]["tune"].get("share") or 0) >= 0.67 * float(x.rstrip("%")) / 100

    def over_rule(r: dict) -> bool:
        if r["H"] != 1 or not rule_reaches(r["top"]):
            return True
        return all((r["ml_" + k] or 0) >= (rr[r["top"]][k].get("hit") or 0) + BEAT_MARGIN for k in ("tune", "test"))

    ml = [r for r in beats if r["model"] in MODELS and over_rule(r)]
    ml_rule = [r for r in beats if r["model"] in MODELS and not over_rule(r)]
    rule = [r for r in beats if r["model"] == "roll_rule" and rule_reaches(r["top"])]
    lf = (res.get("live_feed_roll") or {}).get("by_hour", {})
    parts = []
    if not ml and not rule:
        parts.append("入れません。" + ("機械学習の上位の予測は時間帯の偏りを上回っても、同じ割合のスワップ調整の規則を上回らず、中身はロールオーバーの"
                                       "価格調整です。" if ml_rule else
                                       f"同じ割合に絞った時間帯の偏りを、調整期間・検証期間の両方で{BEAT_MARGIN * 100:.0f}ポイント以上上回るもの"
                                       f" (検証期間で{GOAL_N}回以上) がありません。")
                     + "本番は今の時間帯の偏り (season.py) のままにします。")
        return " ".join(parts)
    cands = sorted(ml + rule, key=lambda r: -min(r["ml_tune"], r["ml_test"]))
    items = []
    for r in cands[:4]:
        T = rr if r["model"] == "roll_rule" else res["models"][r["model"]][str(r["H"])]["tops"]
        te = T[r["top"]]["test"]
        how = (f"予想する動き {T[r['top']]['thr']:.2f} bp 以上" if r["model"] == "roll_rule"
               else f"|p − 0.5| ≥ {T[r['top']]['thr']:.4f}")
        items.append(f"{MODEL_NAMES[r['model']]} {H_NAMES[str(r['H'])]} 上位{r['top']} ({how}): 調整 {_p(r['ml_tune'])} / 検証 "
                     f"{_p(r['ml_test'])} ({r['ml_n_test']:,}回)、同じ割合の時間帯の偏り {_p(r['season_tune'])} / {_p(r['season_test'])}、"
                     f"検証期間のうちロールオーバーを含む予測 {_p(te.get('roll_share'), 0)}")
    parts.append("同じ割合に絞った時間帯の偏りを調整期間・検証期間とも上回ったのは " + "; ".join(items) + "。")
    paid = []
    for r in cands[:4]:
        T = rr if r["model"] == "roll_rule" else res["models"][r["model"]][str(r["H"])]["tops"]
        tu, te = T[r["top"]]["tune"], T[r["top"]]["test"]
        if (tu.get("net_swap_bp") or 0) > 0 and (te.get("net_swap_bp") or 0) > 0:
            paid.append(f"{MODEL_NAMES[r['model']]} {H_NAMES[str(r['H'])]} 上位{r['top']} (調整 {tu['net_swap_bp']:+.2f} / 検証 "
                        f"{te['net_swap_bp']:+.2f} bp)")
    if paid:
        parts.append("スプレッドとスワップを引いても両方の期間でプラスだったのは " + "、".join(paid)
                     + " で、ほかはコストを引くとなくなります。プラス幅は Dukascopy の気配のスプレッドで測った数 bp で、業者の実際のコスト (より広いスプレッド、"
                     "スワップの手数料) では残らない可能性があります。")
    roll_heavy = all((res["models"][r["model"]][str(r["H"])]["tops"][r["top"]]["test"].get("roll_share") or 0) >= 0.5
                     if r["model"] in MODELS else True for r in cands[:4])
    parts.append("ただし、本番へそのまま入れることは勧めません。"
                 + ("上回る分の多くはロールオーバーを含む予測 (表示される価格の調整が中心) で、" if roll_heavy else "")
                 + "的中率は時期で大きく変わり (調整期間と検証期間の差、年ごとの表)、学習に使った Dukascopy の気配と本番の Yahoo の足では"
                 "ロールオーバーの現れ方が違うためです"
                 + (f" (Yahoo では金利の高い通貨が下がる向きが16時に始まる足で {_p(lf['16']['2']['hit'])}、17時で {_p(lf['17']['2']['hit'])}、"
                    f"18時で {_p(lf['18']['2']['hit'])} (予想する動き 2 bp 以上)。Dukascopy では17時の足に集中)" if lf.get("17", {}).get("2", {}).get("n") else "")
                 + "。モデルの入力のスプレッドも Yahoo の足にはありません。試すなら、まず予測に影響させない形で本番の足での予測を記録し、"
                 "時間帯の偏りより当たることを確かめてから、予測の中心に反映するのが安全です。")
    return " ".join(parts)


def run(root: Path | None = None, report_dir: Path = REPORT_DIR, first_test: int = FIRST_TEST,
        last_tune: int = LAST_TUNE, log=print) -> dict:
    t0 = time.time()
    D = build(root, log=log)
    check = check_season(history.load_long_hourly(CODES[0], root), samples=200)
    if log:
        log(f"{len(D['t']):,} rows x {len(D['cols'])} features; season check {check}")
    preds, tuning, weights = walk_forward(D, first_test, last_tune, log=log)
    return finish(D, preds, tuning, weights, check, root, report_dir, first_test, last_tune, time.time() - t0, log)


def finish(D: dict, preds: dict, tuning: dict, weights: dict, check: dict, root: Path | None = None,
           report_dir: Path = REPORT_DIR, first_test: int = FIRST_TEST, last_tune: int = LAST_TUNE,
           seconds: float = 0.0, log=print) -> dict:
    """Score the walk-forward predictions and write ml_long.json and ml_long.md."""
    t0 = time.time()
    res = summarise(D, preds, first_test, last_tune)
    res["tuning"] = tuning
    res["importance"] = importance(D, weights)
    res["compare_season"] = compare_season(res)
    res["live_feed_roll"] = live_feed_roll_check(root)
    res["recommendation"] = recommend(res)
    res["meta"] = {"rows": int(len(D["t"])), "features": D["cols"], "spans": D["spans"], "start": START,
                   "first_test": first_test, "last_tune": last_tune, "train_keep": TRAIN_KEEP, "horizons": list(HORIZONS),
                   "tops": list(TOPS), "goal": {"hit": GOAL_HIT, "n": GOAL_N}, "roll_hours_ny": list(ROLL_HOURS),
                   "season_check": check, "lgb_base": LGB_BASE, "seconds": seconds + time.time() - t0}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "ml_long.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    (report_dir / "ml_long.md").write_text(report(res), encoding="utf-8")
    if log:
        log(f"done in {res['meta']['seconds'] / 60:.1f} min")
    return res


def _jsonable(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    return str(o)


if __name__ == "__main__":
    run()
