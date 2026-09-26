"""Is the time-of-day drift real? Yahoo against Dukascopy mid prices, 2004 onwards (research/season_long.md).

The live direction calls (season.py) come from the average move of each New
York weekday-and-hour slot over the last 6,000 hourly bars. On Yahoo's hourly
bars (the live source, about 2.8 years) they were right 61 % of the time and
80 % for |t| >= 4, mostly around the 17:00 New York rollover. Two questions:

1. Artefact or real? Yahoo's FX quotes may be one side of the book (bid) or
   stale, and at the rollover the spread widens: a bid quote then dips and
   recovers although the mid price does not move. Dukascopy's bid and ask
   candles give the mid price and the spread for the same hours, so for the
   overlap (Yahoo's period) this compares the price levels by hour, the slot
   averages, and the calls scored on Yahoo against the same calls scored on
   mid prices.
2. Does it last? The same rule, walked forward over every hour from 2004 on
   Dukascopy mid prices (tune 2004-2016, test 2017-), per year, and stricter
   tiers (|t| >= 5 ... 12, the top N slots of the week) with their share of
   hours and their average move after the spread and, over the roll, the swap.

Every call uses the rule of season.py (slot statistics from the 6,000 bars
before the origin's UTC day, weekend gaps left out, weekday slot first, then
hour of day); the vectorised version here is checked against season.py on a
sample of days. A new threshold is chosen on the tune period only.
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
from .research_direction import HOURLY_TUNE_SHARE, SESSION_MIN_BARS, score

REPORT_DIR = Path("research")
DUKA_SPLIT = pd.Timestamp("2017-01-01", tz="UTC")     # tune: origins before, test: from
DUKA_FIRST_YEAR = 2004                                  # first year with a full 6,000-bar window
T_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)
TOP_N = (1, 2, 3, 5, 10)
TARGET_HIT = 0.90                                       # the accuracy asked for a "very high" tier
MIN_TEST_CALLS = 200
NS_H = 3_600_000_000_000


# ------------------------------------------------------------------ the rule of season.py, vectorised

def _returns(idx: pd.DatetimeIndex, close: np.ndarray, minutes: int) -> np.ndarray:
    """Log return (bp) into each bar from the previous close; NaN after a pause (as season.slot_stats)."""
    r = np.full(len(close), np.nan)
    r[1:] = np.diff(np.log(close)) * 1e4
    gap = np.diff(idx.as_unit("ns").asi8) > minutes * 60_000_000_000
    r[1:][gap] = np.nan
    return r


def _slot_prefix(r: np.ndarray, key: np.ndarray, n_slots: int, points: np.ndarray) -> np.ndarray:
    """Per-slot (count, sum, sum of squares) of the valid returns r[j] with j < p, for each p in ``points``
    (sorted, unique): shape (len(points), n_slots, 3)."""
    ok = np.isfinite(r)
    j = np.flatnonzero(ok)
    b = np.searchsorted(points, j, side="right")         # r[j] counts for points[m] > j, i.e. m >= b
    H = np.zeros((len(points) + 1, n_slots, 3))
    np.add.at(H, (b, key[j], 0), 1.0)
    np.add.at(H, (b, key[j], 1), r[j])
    np.add.at(H, (b, key[j], 2), r[j] * r[j])
    return np.cumsum(H, axis=0)[: len(points)]


def _mu_t(cnt: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """season._stats from the sums (population sd, MIN_N bars at least)."""
    cnt = np.round(cnt)
    use = cnt >= season.MIN_N
    safe = np.where(use, cnt, 1.0)
    mu = np.where(use, s1 / safe, 0.0)
    sd = np.sqrt(np.maximum(s2 / safe - mu * mu, 0.0))
    t = np.where(use & (sd > 1e-12), mu / np.where(sd > 1e-12, sd, 1.0) * np.sqrt(safe), 0.0)
    return mu, t


def walk_forward(bars: pd.DataFrame, minutes: int = 60, window: int = season.WINDOW) -> pd.DataFrame:
    """The call season.py makes for the next bar at every origin (the end of each bar but the last).

    Row i: origin = end of bar i, the called bar is bar i + 1. Columns: d (expected move, bp; 0 = no
    call), t (its t statistic), f (the bar's actual log move, bp), rank (rank of the weekday slot's |t|
    among the week's slots that day, 1 = strongest; 0 if the call came from the hour-of-day slot or none),
    ny_hour / ny_dow (New York start of the called bar), gap (the called bar follows a pause)."""
    idx = pd.DatetimeIndex(bars.index).as_unit("ns")
    close = bars["close"].to_numpy(float)
    n = len(close)
    r = _returns(idx, close, minutes)
    week, tod, per_day = season._slots(idx, minutes)
    ns = idx.asi8
    day_ns = 86_400_000_000_000
    origin = ns[:-1] + minutes * 60_000_000_000
    cut = np.searchsorted(ns, origin // day_ns * day_ns, side="left")
    low = np.maximum(cut - window, 0)                      # returns lo+1 .. cut-1 with lo = cut - window - 1
    points = np.unique(np.concatenate([cut, low]))
    Pw = _slot_prefix(r, week, 7 * per_day, points)
    Pd = _slot_prefix(r, tod, per_day, points)
    hi, lo = np.searchsorted(points, cut), np.searchsorted(points, low)
    nxt = np.arange(1, n)
    sw = (Pw[hi, week[nxt]] - Pw[lo, week[nxt]])
    sd = (Pd[hi, tod[nxt]] - Pd[lo, tod[nxt]])
    mu_w, t_w = _mu_t(sw[:, 0], sw[:, 1], sw[:, 2])
    mu_d, t_d = _mu_t(sd[:, 0], sd[:, 1], sd[:, 2])
    use_w = np.abs(t_w) >= season.T_MIN
    use_d = ~use_w & (np.abs(t_d) >= season.T_MIN)
    d = np.where(use_w, mu_w, np.where(use_d, mu_d, 0.0))
    t = np.where(use_w, t_w, np.where(use_d, t_d, 0.0))
    # rank of the called weekday slot's |t| among all weekday slots of the day's statistics
    days, first = np.unique(cut, return_index=True)
    W = Pw[hi[first]] - Pw[lo[first]]
    _, tw_all = _mu_t(W[..., 0], W[..., 1], W[..., 2])
    order = np.argsort(-np.abs(tw_all), axis=1, kind="stable")
    rank_all = np.empty_like(order)
    np.put_along_axis(rank_all, order, np.arange(1, order.shape[1] + 1)[None, :].repeat(len(days), 0), axis=1)
    k = np.searchsorted(days, cut)
    rank = np.where(use_w, rank_all[k, week[nxt]], 0)
    local = idx[1:].tz_convert(season.NEW_YORK)
    return pd.DataFrame({"d": d, "t": t, "f": np.log(close[1:] / close[:-1]) * 1e4,
                         "rank": rank, "ny_hour": local.hour.to_numpy(), "ny_dow": local.dayofweek.to_numpy(),
                         "gap": np.diff(ns) > minutes * 60_000_000_000, "i": np.arange(n - 1)},
                        index=pd.DatetimeIndex(origin, tz="UTC"))


def check_against_season(bars: pd.DataFrame, calls: pd.DataFrame, minutes: int = 60, n_days: int = 25,
                         seed: int = 0) -> dict:
    """Recompute a sample of days with season.slot_stats / bar_drift and compare (max abs difference)."""
    rng = np.random.default_rng(seed)
    idx = bars.index
    days = calls.index.normalize().unique()
    days = days[len(days) // 20:]
    pick = np.sort(rng.choice(len(days), size=min(n_days, len(days)), replace=False))
    worst_d = worst_t = 0.0
    n_checked = mismatched = 0
    for p in pick:
        rows = calls[calls.index.normalize() == days[p]]
        stats = season.slot_stats(bars, rows.index[0].to_pydatetime(), minutes)
        d, t = season.bar_drift(stats, idx[rows["i"].to_numpy() + 1], minutes)
        worst_d = max(worst_d, float(np.max(np.abs(d - rows["d"].to_numpy()))))
        worst_t = max(worst_t, float(np.max(np.abs(t - rows["t"].to_numpy()))))
        mismatched += int(np.sum((d != 0) != (rows["d"].to_numpy() != 0)))
        n_checked += len(rows)
    return {"days": len(pick), "origins": n_checked, "max_abs_d": worst_d, "max_abs_t": worst_t,
            "call_mismatch": mismatched}


# ------------------------------------------------------------------ data

def _closed(dow: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """New York weekday/hour of a bar start when spot FX is shut (timeutil.is_market_open): the live
    server never forecasts these bars, but Yahoo sometimes prints one (a single tick after Friday's close)."""
    return ((dow == 4) & (hour >= 17)) | (dow == 5) | ((dow == 6) & (hour < 17))


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated()].sort_index()


def load_duka(code: str) -> pd.DataFrame:
    """Dukascopy hourly mid bars with the bid and ask closes and the half spread (bp) at each close."""
    D = _dedup(history.load_long_hourly(code))
    D["bid"] = D["close"] - D["spread"] / 2
    D["ask"] = D["close"] + D["spread"] / 2
    D["hs"] = D["spread"] / 2 / D["close"] * 1e4
    return D


def _frame(bars: pd.DataFrame, minutes: int = 60, code: str | None = None) -> pd.DataFrame:
    """walk_forward plus the columns used for scoring: closed market, year, weekly block, and for
    Dukascopy bars the round-trip cost (half spread at the origin and at the target close, bp) and the
    swap a long position earns over the roll (bars starting 17:00 New York, bp; negative = pays)."""
    C = walk_forward(bars, minutes)
    C["closed"] = _closed(C["ny_dow"].to_numpy(), C["ny_hour"].to_numpy())
    C["year"] = C.index.year
    C["block"] = C.index.tz_convert(None).to_period("W").astype(str)
    if "hs" in bars:
        hs = bars["hs"].to_numpy(float)
        C["cost"] = hs[:-1] + hs[1:]
    if code is not None and minutes == 60:
        C["swap"] = _swap_bp(code, C)
    return C


def _swap_bp(code: str, C: pd.DataFrame) -> np.ndarray:
    """Interest a long position earns by holding the called bar over the 17:00 New York roll (bp): the
    base-minus-quote short-rate difference for one day, three on Wednesday (value date over the weekend).
    The roll moves the spot quote by the same amount the other way, so price move + swap is what a trade
    keeps (interbank terms; retail swaps are worse)."""
    dow, hour = C["ny_dow"].to_numpy(), C["ny_hour"].to_numpy()
    spans = (hour == 17) & ~_closed(dow, hour)
    _, diff = _carry_sign(code, pd.DatetimeIndex(C.index))
    return np.where(spans, diff / 100 / 360 * np.where(dow == 2, 3.0, 1.0) * 1e4, 0.0)


# ------------------------------------------------------------------ scoring

def _wilson_low(k: float, n: int, z: float = 1.96) -> float | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    return float((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den)


def tier_score(g: pd.DataFrame, n_origins: int, f: str = "f") -> dict:
    """Calls in ``g`` scored on the move column ``f``: research_direction.score (hit, bp, weekly-block t)
    plus the share of all eligible origins, the 95 % lower bound of the hit rate, and, where a cost is
    known, the move after the spread (``net``: actual Dukascopy spread paid; ``net_med``: the median
    round-trip spread of the pair at that New York hour) and the share of calls that beat the spread."""
    g = g[np.isfinite(g[f].to_numpy(float))]
    s = np.sign(g["d"].to_numpy(float))
    x = g[f].to_numpy(float)
    out = score(s, x, g["block"].to_numpy()) if len(g) else {"n": 0}
    out["n"] = int(len(g))
    out["share"] = len(g) / n_origins if n_origins else None
    nz = x != 0
    if nz.any():
        k = float(np.sum(np.sign(x[nz]) == s[nz]))
        out["hit"] = k / int(nz.sum())
        out["hit_lo"] = _wilson_low(k, int(nz.sum()))
        out["bp"] = float(np.mean(s * x))
    for col, key in (("cost", "net"), ("cost_med", "net_med")):
        if col in g and len(g):
            c = g[col].to_numpy(float)
            ok = np.isfinite(c)
            if ok.any():
                out[col] = float(np.mean(c[ok]))
                out[key] = float(np.mean(s[ok] * x[ok] - c[ok]))
                out[key + "_win"] = float(np.mean(s[ok] * x[ok] > c[ok]))
    cc = "cost_med" if "cost_med" in g else "cost"                   # same basis as the spread-only figure shown
    if "swap" in g and cc in g and len(g):
        c, w = g[cc].to_numpy(float), g["swap"].to_numpy(float)
        ok = np.isfinite(c) & np.isfinite(w)
        if ok.any():
            kept = s[ok] * (x[ok] + w[ok]) - c[ok]
            out["swap"] = float(np.mean(s[ok] * w[ok]))
            out["net_swap"] = float(np.mean(kept))
            out["net_swap_win"] = float(np.mean(kept > 0))
    return out


def ladder(C: pd.DataFrame, n_origins: int, f: str = "f") -> dict:
    """Scores by threshold (|t| >= T for T in T_GRID) and by the top N weekday slots of the day."""
    a = C["t"].abs()
    out = {f"t{T:g}": tier_score(C[a >= T], n_origins, f) for T in T_GRID}
    out["mid"] = tier_score(C[(a >= season.T_MIN) & (a < season.T_HIGH)], n_origins, f)
    for N in TOP_N:
        out[f"top{N}"] = tier_score(C[(C["rank"] > 0) & (C["rank"] <= N)], n_origins, f)
    return out


def choose_threshold(tune: dict) -> float | None:
    """The smallest T of T_GRID whose tune-period hit rate reaches TARGET_HIT (at least 30 calls)."""
    for T in T_GRID:
        x = tune.get(f"t{T:g}", {})
        if x.get("n", 0) >= 30 and (x.get("hit") or 0) >= TARGET_HIT:
            return T
    return None


# ------------------------------------------------------------------ Yahoo against Dukascopy (overlap)

def _carry_sign(code: str, when: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(+1 where the base currency paid the higher short rate at the time, else -1; the base-minus-quote
    rate difference in % a year), as known then: the spot quote steps down in the direction of the
    higher-rate currency at the roll, when the value date moves one day later."""
    pair = PAIRS[code]
    days = pd.DatetimeIndex(when.tz_convert(None).normalize().unique())
    r = history.rates_panel(days)
    diff = (r[pair.base] - r[pair.quote]).reindex(when.tz_convert(None).normalize()).to_numpy(float)
    return np.where(diff >= 0, 1.0, -1.0), diff


def overlap_pair(code: str, Y: pd.DataFrame, D: pd.DataFrame) -> dict:
    """Yahoo's hourly closes against Dukascopy's mid / bid / ask closes for the same bars."""
    common = Y.index.intersection(D.index)
    y = np.log(Y.loc[common, "close"].to_numpy(float))
    m = np.log(D.loc[common, "close"].to_numpy(float))
    b = np.log(D.loc[common, "bid"].to_numpy(float))
    a = np.log(D.loc[common, "ask"].to_numpy(float))
    ny = common.tz_convert(season.NEW_YORK)
    hour, dow = ny.hour.to_numpy(), ny.dayofweek.to_numpy()
    lev = pd.DataFrame({"mid": (y - m) * 1e4, "bid": (y - b) * 1e4, "ask": (y - a) * 1e4,
                        "hs": D.loc[common, "hs"].to_numpy(float), "hour": hour})
    near = np.argmin(np.abs(np.stack([lev["bid"], lev["mid"], lev["ask"]])), axis=0)
    fy = (Y.loc[common, "high"] == Y.loc[common, "low"]).to_numpy()
    fd = (D.loc[common, "high"] == D.loc[common, "low"]).to_numpy()
    # hourly moves where both sources have this bar and the one an hour earlier
    ns = common.as_unit("ns").asi8
    step = np.concatenate([[False], np.diff(ns) == NS_H])
    R = pd.DataFrame({k: np.concatenate([[np.nan], np.diff(v) * 1e4]) for k, v in (("yahoo", y), ("mid", m),
                                                                                    ("bid", b), ("ask", a))},
                     index=common)[step]
    R["hour"], R["dow"] = hour[step], dow[step]
    sgn, _ = _carry_sign(code, R.index)
    lag = {str(k): float(np.corrcoef(R["yahoo"].to_numpy()[max(k, 0): len(R) + min(k, 0)],
                                     R["mid"].to_numpy()[max(-k, 0): len(R) - max(k, 0)])[0, 1]) for k in (-1, 0, 1)}
    # which quote Yahoo follows: Yahoo - mid against the half spread (slope -1: bid, 0: mid plus a constant,
    # +1: ask), and the tracking error of Yahoo's hourly moves around the roll (16:00-18:59 New York)
    hs_slope, hs_icpt = np.polyfit(lev["hs"], lev["mid"], 1)
    side = "bid" if hs_slope < -0.5 else "ask" if hs_slope > 0.5 else "mid"
    rim = R[R["hour"].isin((16, 17, 18))]
    track = {k: float(np.sqrt(np.mean((rim["yahoo"] - rim[k]) ** 2))) for k in ("bid", "mid", "ask")}
    slot = R.groupby(["dow", "hour"])[["yahoo", "mid", "bid", "ask"]].agg(["mean", "std", "count"])
    tt = {k: slot[(k, "mean")] / slot[(k, "std")] * np.sqrt(slot[(k, "count")]) for k in ("yahoo", "mid", "bid")}
    strong_y = tt["yahoo"].abs() >= 4
    return {
        "span": [str(common[0]), str(common[-1])], "bars": int(len(common)), "moves": int(len(R)),
        "side": side, "hs_slope": float(hs_slope), "offset_bp": float(hs_icpt), "track_rms_roll": track,
        "corr_lag": lag,
        "level_bp": {k: float(lev[k].median()) for k in ("mid", "bid", "ask")},
        "half_spread_bp": float(lev["hs"].median()),
        "nearest": {k: float(np.mean(near == i)) for i, k in enumerate(("bid", "mid", "ask"))},
        "abs_diff_bp": {k: float(lev[k].abs().median()) for k in ("mid", "bid", "ask")},
        "level_by_hour": {int(h): {"mid": float(g["mid"].median()), "bid": float(g["bid"].median()),
                                   "hs": float(g["hs"].median())} for h, g in lev.groupby("hour")},
        # stale quotes: hours whose close did not move, overall and in the roll hour; single-price bars
        "stale": {"zero_yahoo": float(np.mean(R["yahoo"] == 0)), "zero_duka": float(np.mean(R["mid"] == 0)),
                  "zero_yahoo_17": float(np.mean(R["yahoo"][R["hour"] == 17] == 0)),
                  "zero_duka_17": float(np.mean(R["mid"][R["hour"] == 17] == 0)),
                  "flat_yahoo": float(np.mean(fy)), "flat_duka": float(np.mean(fd))},
        "slot_corr": {k: float(np.corrcoef(slot[("yahoo", "mean")], slot[(k, "mean")])[0, 1]) for k in ("mid", "bid", "ask")},
        "slot_slope_mid": float(np.polyfit(slot[("yahoo", "mean")], slot[("mid", "mean")], 1)[0]),
        "strong_yahoo": int(strong_y.sum()),
        "strong_mid": int((tt["mid"].abs() >= 4).sum()),
        "strong_same_sign_mid": int(((np.sign(tt["mid"]) == np.sign(tt["yahoo"])) & strong_y).sum()),
        "strong_mid_t_ge2": int(((np.sign(tt["mid"]) == np.sign(tt["yahoo"])) & strong_y & (tt["mid"].abs() >= 2)).sum()),
        # moves signed so that + is towards the higher-rate currency, by New York hour (Wednesday apart)
        "carry_profile": {w: {int(h): {k: float(np.mean(g[k] * g["_s"])) for k in ("yahoo", "mid", "bid", "ask")} | {"n": int(len(g))}
                              for h, g in R.assign(_s=sgn)[sel].groupby("hour")}
                          for w, sel in (("wed", R["dow"].to_numpy() == 2), ("other", np.isin(R["dow"].to_numpy(), (0, 1, 3))),
                                         ("fri", R["dow"].to_numpy() == 4))},
    }


def overlap_calls(code: str, Y: pd.DataFrame, D: pd.DataFrame, CY: pd.DataFrame, CD: pd.DataFrame,
                  side: str = "bid") -> pd.DataFrame:
    """Yahoo's calls (as the server makes them) at the origins where Dukascopy has the same two bars,
    with the called bar's move on Yahoo, on the mid, on the quote Yahoo follows (``side``), the actual
    round-trip spread, and the call Dukascopy's own mid prices give at the same origin."""
    yi = Y.index
    t0, t1 = yi[CY["i"].to_numpy()], yi[CY["i"].to_numpy() + 1]
    ok = t0.isin(D.index) & t1.isin(D.index)
    X = CY[ok].copy()
    t0, t1 = t0[ok], t1[ok]
    X["f_mid"] = np.log(D.loc[t1, "close"].to_numpy() / D.loc[t0, "close"].to_numpy()) * 1e4
    col = {"bid": "bid", "ask": "ask", "mid": "close"}[side]
    X["f_side"] = np.log(D.loc[t1, col].to_numpy() / D.loc[t0, col].to_numpy()) * 1e4
    X["cost"] = D.loc[t0, "hs"].to_numpy() + D.loc[t1, "hs"].to_numpy()
    di = D.index
    pos = di.get_indexer(t0)
    own = CD.set_index("i").reindex(pos)
    X["d_duka"] = own["d"].to_numpy()
    X["t_duka"] = own["t"].to_numpy()
    X["same_bar"] = di[np.minimum(pos + 1, len(di) - 1)] == t1
    X["swap"] = _swap_bp(code, X)
    X["pair"] = code
    return X


# ------------------------------------------------------------------ the long run on Dukascopy mid prices

def roll_profile(code: str, D: pd.DataFrame) -> pd.DataFrame:
    """Hourly mid moves (bp, consecutive hours only) signed so that + is towards the higher-rate
    currency, with the New York weekday/hour and year, and the expected step at the roll: the one-day
    interest difference (three days on Wednesday, when the value date jumps over the weekend)."""
    ns = D.index.as_unit("ns").asi8
    ok = np.concatenate([[False], np.diff(ns) == NS_H])
    r = np.concatenate([[np.nan], np.diff(np.log(D["close"].to_numpy(float))) * 1e4])
    idx = D.index[ok]
    sgn, diff = _carry_sign(code, idx)
    ny = idx.tz_convert(season.NEW_YORK)
    dow = ny.dayofweek.to_numpy()
    days = np.where(dow == 2, 3.0, 1.0)
    return pd.DataFrame({"r": r[ok] * sgn, "raw": r[ok], "sgn": sgn, "step": -np.abs(diff) / 100 / 360 * days * 1e4,
                         "diff": diff, "hour": ny.hour.to_numpy(), "dow": dow, "year": idx.year, "pair": code,
                         "day": ny.tz_localize(None).normalize()}, index=idx)


ROLL_WINDOWS = ((17,), (16, 17, 18), (15, 16, 17, 18, 19))   # New York start hours of the bars summed


def _roll_fit(P: pd.DataFrame, hours: tuple) -> dict:
    """Pair-year-weekday averages of the signed move over ``hours`` (Monday to Thursday rolls) against
    the expected step; slope 1 = the quote moves exactly by the one-day interest difference."""
    roll = P[P["hour"].isin(hours) & P["dow"].isin((0, 1, 2, 3))]
    w = roll.groupby(["day", "pair"]).agg(r=("r", "sum"), step=("step", "first"), dow=("dow", "first"),
                                                    year=("year", "first"), n=("r", "size"))
    w = w[w["n"] == len(hours)]
    g = w.groupby(["pair", "year", w["dow"] == 2]).agg(r=("r", "mean"), step=("step", "mean"), n=("r", "size"))
    g = g[g["n"] >= 20]
    slope, icpt = np.polyfit(g["step"], g["r"], 1)
    by_wed = {("wed" if k else "mon_tue_thu"): {"move": float(x["r"].mean()), "step": float(x["step"].mean()),
                                                "days": int(len(x))}
              for k, x in w.groupby(w["dow"] == 2)}
    return {"hours": list(hours), "slope": float(slope), "intercept": float(icpt),
            "corr": float(np.corrcoef(g["step"], g["r"])[0, 1]), "groups": int(len(g)), "by_weekday": by_wed}


def roll_mechanism(P: pd.DataFrame) -> dict:
    """Does the move at the roll match the one-day interest difference (the spot value date moving a day
    later; three days on Wednesday)? Main window: the bar starting 17:00 New York, which holds the roll;
    wider windows (16:00-19:00, 15:00-20:00) are kept for comparison. Plus the signed hourly profile."""
    fits = [_roll_fit(P, h) for h in ROLL_WINDOWS]
    prof = {}
    for period, sel in (("tune", P.index < DUKA_SPLIT), ("test", P.index >= DUKA_SPLIT)):
        q = P[sel]
        prof[period] = {w_: {int(h): float(x.mean()) for h, x in q[q["dow"].isin(ds)].groupby("hour")["r"]}
                        for w_, ds in (("wed", (2,)), ("other", (0, 1, 3)), ("fri", (4,)))}
    return fits[0] | {"windows": fits, "profile": prof, "all_hours": float(P["r"].mean())}


def per_year(C: pd.DataFrame, n_by_year: pd.Series) -> dict:
    """Calls (|t| >= 2), mid tier (2 <= |t| < 4), high tier (|t| >= 4), |t| >= 6 and 8, per calendar year."""
    out = {}
    for y, g in C.groupby("year"):
        n0 = int(n_by_year.get(y, 0))
        a = g["t"].abs()
        out[int(y)] = {"all": tier_score(g, n0), "mid": tier_score(g[a < season.T_HIGH], n0),
                       "high": tier_score(g[a >= season.T_HIGH], n0), "t6": tier_score(g[a >= 6], n0),
                       "t8": tier_score(g[a >= 8], n0)}
    return out


def by_hour(C: pd.DataFrame, lo: float) -> dict:
    g = C[C["t"].abs() >= lo]
    return {int(h): {"n": int(len(x)), "hit": float(np.mean(np.sign(x["f"][x["f"] != 0]) == np.sign(x["d"][x["f"] != 0])))
                     if (x["f"] != 0).any() else None} for h, x in g.groupby("ny_hour")}


# ------------------------------------------------------------------ assemble

def _yahoo_split(Y: dict) -> pd.Timestamp:
    """The tune/test split of research_direction: 60 % of all Yahoo hourly bar times, pooled over pairs."""
    all_t = np.sort(np.concatenate([pd.DatetimeIndex(H.index).as_unit("ns").asi8 for H in Y.values()]))
    return pd.Timestamp(all_t[int(len(all_t) * HOURLY_TUNE_SHARE)], tz="UTC")


def _calls(C: pd.DataFrame) -> pd.DataFrame:
    return C[C["d"] != 0]


def _periods(C: pd.DataFrame, n0: pd.Series, f: str = "f") -> dict:
    return {p: ladder(_calls(C[C["period"] == p]), int(n0.get(p, 0)), f) for p in ("tune", "test")}


def evaluate_yahoo(Y: dict, split: pd.Timestamp, cost_med: dict) -> tuple[dict, dict]:
    """Yahoo hourly calls, as research_direction.session_eval (origins with 2,000 earlier bars), on every
    bar ("all", as published) and on the bars the server actually forecasts ("open": market open)."""
    frames = {}
    for code, H in Y.items():
        C = _frame(H)
        C = C[(C["i"] >= SESSION_MIN_BARS) & (C["i"] + 24 < len(H))].copy()
        C["period"] = np.where(C.index >= split, "test", "tune")
        C["pair"] = code
        cm = cost_med.get(code)
        C["cost_med"] = C["ny_hour"].map(cm).to_numpy(float) if cm is not None else np.nan
        C["swap"] = _swap_bp(code, C)
        frames[code] = C
    A = pd.concat(frames.values())
    out = {}
    for variant, M in (("all", np.ones(len(A), bool)), ("open", ~A["closed"].to_numpy())):
        X = A[M]
        out[variant] = _periods(X, X.groupby("period").size())
    out["closed_calls"] = {p: int(((A["period"] == p) & A["closed"] & (A["d"] != 0)).sum()) for p in ("tune", "test")}
    out["closed_high"] = {p: tier_score(_calls(A[(A["period"] == p) & A["closed"] & (A["t"].abs() >= season.T_HIGH)]), 0)
                          for p in ("tune", "test")}
    out["span"] = [str(A.index.min()), str(A.index.max())]
    out["split"] = str(split)
    return out, frames


def evaluate_yahoo_15m() -> dict:
    """15-minute calls on Yahoo's ~60 days (origins with 500 earlier bars), next bar."""
    parts = []
    for code in PAIRS:
        M = _dedup(history.load_intraday(code, "15m"))
        C = _frame(M, 15)
        C = C[(C["i"] >= 500) & (C["i"] + 4 < len(M))].copy()
        C["pair"] = code
        parts.append(C)
    A = pd.concat(parts)
    out = {"span": [str(A.index.min()), str(A.index.max())]}
    for variant, M in (("all", np.ones(len(A), bool)), ("open", ~A["closed"].to_numpy())):
        out[variant] = ladder(_calls(A[M]), int(M.sum()))
    return out


def evaluate_overlap(Y: dict, D: dict, CYall: dict, CD: dict) -> dict:
    """Per pair, Yahoo's prices against Dukascopy's; pooled, Yahoo's calls scored on Yahoo, mid and followed-quote moves
    by tier and by hour, and Dukascopy's own calls over the same span."""
    out = {"pairs": {}}
    parts = []
    for code in D:
        out["pairs"][code] = overlap_pair(code, Y[code], D[code])
        parts.append(overlap_calls(code, Y[code], D[code], _calls(CYall[code]), CD[code], out["pairs"][code]["side"]))
    X = pd.concat(parts)
    lo, hi = X.index.min(), X.index.max()
    out["calls"] = {}
    for T in (2.0, 4.0, 6.0, 8.0):
        g = X[X["t"].abs() >= T]
        same_dir = np.sign(g["d_duka"].fillna(0)) == np.sign(g["d"])
        same_tier = same_dir & (g["t_duka"].abs() >= T)
        out["calls"][f"t{T:g}"] = {"yahoo": tier_score(g, 0, "f"), "mid": tier_score(g, 0, "f_mid"),
                                   "side": tier_score(g, 0, "f_side"),
                                   "agree": float(same_dir.mean()) if len(g) else None,       # mid rule calls the same way
                                   "agree_tier": float(same_tier.mean()) if len(g) else None}  # ... with |t| >= T too
    # where the difference comes from: the bars around the roll (16:00 and 18:00: spread widening and
    # narrowing, 17:00: the roll itself) against the rest of the day
    side = X["pair"].map({c: x["side"] for c, x in out["pairs"].items()})
    out["by_hour"], out["by_hour_side"] = {}, {}
    for T in (2.0, 4.0):
        for sd in [None] + sorted(side.unique()):
            g = X[(X["t"].abs() >= T) & ((side == sd) if sd else True)]
            grp = np.where(g["ny_hour"].isin((16, 17, 18)), g["ny_hour"].astype(str), "other")
            res = {k: {"yahoo": tier_score(x, 0, "f"), "mid": tier_score(x, 0, "f_mid"),
                       "side": tier_score(x, 0, "f_side")} for k, x in g.groupby(grp)}
            if sd is None:
                out["by_hour"][f"t{T:g}"] = res
            else:
                out["by_hour_side"].setdefault(sd, {})[f"t{T:g}"] = res
    # the calls left when the bid-only bars (16:00 and 18:00 on pairs where Yahoo follows the bid) are dropped
    rim_bid = (side == "bid") & X["ny_hour"].isin((16, 18))
    out["core"] = {f"t{T:g}": {"yahoo": tier_score(X[~rim_bid & (X["t"].abs() >= T)], 0, "f"),
                               "mid": tier_score(X[~rim_bid & (X["t"].abs() >= T)], 0, "f_mid")} for T in (2.0, 4.0)}
    # Dukascopy's own calls over the same span, scored on the mid (the rule applied to mid prices)
    own = pd.concat([_calls(C[(C.index >= lo) & (C.index <= hi) & ~C["closed"]]) for C in CD.values()])
    out["duka_own"] = {f"t{T:g}": tier_score(own[own["t"].abs() >= T], 0) for T in (2.0, 4.0, 6.0, 8.0)}
    out["span"] = [str(lo), str(hi)]
    return out


def evaluate_long(CD: dict) -> dict:
    """Every hourly origin from 2004 with a full window, bars when the market is open, on the mid."""
    parts = []
    for code, C in CD.items():
        E = C[(C.index >= pd.Timestamp(f"{DUKA_FIRST_YEAR}-01-01", tz="UTC")) & (C["i"] >= season.WINDOW) & ~C["closed"]].copy()
        E["period"] = np.where(E.index >= DUKA_SPLIT, "test", "tune")
        E["pair"] = code
        med = E.groupby(["year", "ny_hour"])["cost"].median()
        E["cost_med"] = med.reindex(pd.MultiIndex.from_arrays([E["year"], E["ny_hour"]])).to_numpy()
        parts.append(E)
    A = pd.concat(parts)
    out = {"span": [str(A.index.min()), str(A.index.max())], "origins": int(len(A))}
    out["ladder"] = _periods(A, A.groupby("period").size())
    out["per_year"] = per_year(_calls(A), A.groupby("year").size())
    L = _calls(A)
    out["by_hour"] = {p: {"high": by_hour(L[L["period"] == p], season.T_HIGH), "t6": by_hour(L[L["period"] == p], 6.0)}
                      for p in ("tune", "test")}
    out["by_pair"] = {p: {code: {"all": tier_score(g, int(((A["period"] == p) & (A["pair"] == code)).sum())),
                                 "high": tier_score(g[g["t"].abs() >= season.T_HIGH], 0)}
                          for code, g in L[L["period"] == p].groupby("pair")} for p in ("tune", "test")}
    return out


def run(log=print) -> dict:
    t_start = time.time()
    Y = {code: _dedup(history.load_hourly(code)) for code in PAIRS}
    split = _yahoo_split(Y)
    D = {code: load_duka(code) for code in PAIRS if (history.HIST_DIR / history.DUKA_DIR / f"{code}_1h.csv").exists()}
    log(f"Dukascopy pairs: {', '.join(D)}")
    CD = {code: _frame(B, code=code) for code, B in D.items()}
    res: dict = {"params": {"window": season.WINDOW, "t_min": season.T_MIN, "t_high": season.T_HIGH, "min_n": season.MIN_N,
                            "t_grid": list(T_GRID), "top_n": list(TOP_N), "target": TARGET_HIT,
                            "min_test_calls": MIN_TEST_CALLS, "duka_split": str(DUKA_SPLIT.date()), "pairs": list(D)}}
    res["check"] = {code: check_against_season(D[code], CD[code], n_days=20, seed=k) for k, code in enumerate(list(D)[:2])}
    log(f"check against season.py: {res['check']}")
    # median round-trip spread by New York hour of the called bar, over Yahoo's period
    y0 = min(H.index[0] for H in Y.values())
    cost_med = {code: C[C.index >= y0].groupby("ny_hour")["cost"].median() for code, C in CD.items()}
    res["cost_by_hour"] = {code: {int(h): float(v) for h, v in s.items()} for code, s in cost_med.items()}
    res["yahoo"], CY = evaluate_yahoo(Y, split, cost_med)
    log(f"yahoo done ({time.time() - t_start:.0f}s)")
    res["yahoo_15m"] = evaluate_yahoo_15m()
    res["overlap"] = evaluate_overlap(Y, D, CY, CD)
    log(f"overlap done ({time.time() - t_start:.0f}s)")
    res["long"] = evaluate_long(CD)
    res["roll"] = roll_mechanism(pd.concat([roll_profile(code, B) for code, B in D.items()]))
    log(f"long run done ({time.time() - t_start:.0f}s)")
    res["choice"] = recommend(res)
    res["seconds"] = round(time.time() - t_start, 1)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "season_long.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (REPORT_DIR / "season_long.md").write_text(report(res), encoding="utf-8")
    return res


# ------------------------------------------------------------------ recommendation

def _at(lad: dict, T: float | None) -> dict:
    return lad.get(f"t{T:g}", {}) if T is not None else {}


def recommend(res: dict) -> dict:
    """A "very high" tier only if a threshold chosen on the tune period reaches TARGET_HIT on both periods
    of the live source (Yahoo, bars the server forecasts) with MIN_TEST_CALLS test calls, and the mid
    prices (Dukascopy, 2004-2016 and 2017-) do not contradict it."""
    yo, ya, lg = res["yahoo"]["open"], res["yahoo"]["all"], res["long"]["ladder"]
    out: dict = {}
    for name, lad in (("yahoo_open", yo), ("yahoo_all", ya), ("duka", lg)):
        T = choose_threshold(lad["tune"])
        out[name] = {"t": T, "tune": _at(lad["tune"], T), "test": _at(lad["test"], T)}
        # every threshold and top-N tier that reaches the target on both periods, whatever its size
        out[name]["both"] = [k for k in lad["tune"] if k != "mid" and (lad["tune"][k].get("n", 0) >= 30 and
                                                        (lad["tune"][k].get("hit") or 0) >= TARGET_HIT and
                                                        (lad["test"][k].get("hit") or 0) >= TARGET_HIT)]
    T = out["yahoo_open"]["t"]
    te, du = _at(yo["test"], T), (_at(lg["tune"], T), _at(lg["test"], T))
    checks = {"chosen_on_tune": T is not None,
              "test_hit": (te.get("hit") or 0) >= TARGET_HIT,
              "test_calls": te.get("n", 0) >= MIN_TEST_CALLS,
              "mid_tune": (du[0].get("hit") or 0) >= TARGET_HIT,
              "mid_test": (du[1].get("hit") or 0) >= TARGET_HIT}
    out["checks"] = checks
    out["add_top_tier"] = all(checks.values())
    hi = f"t{season.T_HIGH:g}"
    out["keep_high"] = {"yahoo_open": [yo["tune"][hi].get("hit"), yo["test"][hi].get("hit")],
                        "duka": [lg["tune"][hi].get("hit"), lg["test"][hi].get("hit")]}
    return out


# ------------------------------------------------------------------ report

def _p(x, d=1) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{d}f}%"


def _b(x, d=2) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{d}f}"


def _nh(x: dict) -> str:
    """"n / hit" of a score."""
    return f"{x.get('n', 0):,} / {_p(x.get('hit'))}" if x.get("n") else "0 / —"


def _hn(x: dict) -> str:
    """"hit (n回)" in prose; "0回" when there were no calls."""
    return f"{_p(x.get('hit'))} ({x['n']:,}回)" if x.get("n") else "0回"


def _tier_label(k: str) -> str:
    return f"週の上位{k[3:]}枠" if k.startswith("top") else f"\\|t\\| ≥ {k[1:]}"


def _ladder_table(lad: dict, keys: list[str], periods=(("tune", "調整"), ("test", "検証")), cost=True) -> list[str]:
    head = "| 基準 | " + " | ".join(f"{lab}: 回数 / 的中率 (95%下限)" for _, lab in periods) + " | 検証: 足に占める割合 | 検証: 平均 (bp)"
    head += " | 検証: スプレッド後 (bp) | 検証: スプレッド・スワップ後 (bp) |" if cost else " |"
    L = [head, "|" + "---|" * (head.count("|") - 1)]
    last = periods[-1][0]
    for k in keys:
        row = f"| {_tier_label(k)} | "
        row += " | ".join(f"{_nh(lad[p][k])} ({_p(lad[p][k].get('hit_lo'))})" for p, _ in periods)
        x = lad[last][k]
        row += f" | {_p(x.get('share'), 2)} | {_b(x.get('bp'))}"
        if cost:
            row += f" | {_b(x.get('net_med'))} | {_b(x.get('net_swap'))} |"
        else:
            row += " |"
        L.append(row)
    return L


def _mean(vals) -> float:
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


SIDE_NAME = {"bid": "売値", "mid": "仲値 + 一定の差", "ask": "買値"}


def _groups(ov: dict, pairs: list[str]) -> dict[str, list[str]]:
    """Pairs by the quote Yahoo follows (bid / mid plus a constant / ask)."""
    out: dict[str, list[str]] = {}
    for c in pairs:
        out.setdefault(ov["pairs"][c]["side"], []).append(c)
    return out


def report(res: dict) -> str:
    ov, lg, yh, q15, rl = res["overlap"], res["long"], res["yahoo"], res["yahoo_15m"], res["roll"]
    pairs = res["params"]["pairs"]
    keys = [f"t{T:g}" for T in T_GRID if T >= 4] + [f"top{N}" for N in TOP_N if N <= 3]
    L = ["# 時間帯の偏りは本物か: Yahoo と Dukascopy の仲値、2004年からの検証", ""]
    L += _conclusion(res) + [""]
    # ---------------- overlap
    s0 = min(ov["pairs"][c]["span"][0] for c in pairs)
    s1 = max(ov["pairs"][c]["span"][1] for c in pairs)
    L += ["## 1. Yahoo の価格と仲値の比較 (重なる期間)", "",
          f"Yahoo の1時間足 (本番のデータ) と、同じ時刻の Dukascopy の1時間足 (売値と買値の平均 = 仲値、スプレッドつき) を、"
          f"両方にある足で比べました ({s0[:10]}〜{s1[:10]}、{len(pairs)}ペア)。売値 (bid) は仲値 − スプレッド/2、買値 (ask) は仲値 + スプレッド/2 です。", "",
          "Yahoo が売値・仲値・買値のどれに従うかは、Yahoo − 仲値 の差を半スプレッドに回帰した傾きで判定しました (−1 なら売値、0 なら仲値に一定の差、+1 なら買値)。", "",
          "| ペア | 足の数 | 値動きの相関 (同時 / 1本ずれ) | Yahoo − 仲値 (bp、中央値) | 半スプレッド (bp) | 半スプレッドへの傾き / 一定の差 (bp) | Yahoo が従う気配 | 一番近い価格: 売値 / 仲値 / 買値 | 曜日×時間の平均の相関 (Yahoo と仲値) | \\|t\\| ≥ 4 の枠: Yahoo / 仲値 / 同じ向きで仲値も \\|t\\| ≥ 2 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for code in pairs:
        x = ov["pairs"][code]
        c = x["corr_lag"]
        L.append(f"| {code} | {x['bars']:,} | {c['0']:.3f} / {max(c['-1'], c['1']):.3f} | {_b(x['level_bp']['mid'])} | {x['half_spread_bp']:.2f} | "
                 f"{x['hs_slope']:+.2f} / {_b(x['offset_bp'])} | {SIDE_NAME[x['side']]} | "
                 f"{_p(x['nearest']['bid'], 0)} / {_p(x['nearest']['mid'], 0)} / {_p(x['nearest']['ask'], 0)} | "
                 f"{x['slot_corr']['mid']:.2f} | {x['strong_yahoo']} / {x['strong_mid']} / {x['strong_mid_t_ge2']} |")
    groups = _groups(ov, pairs)
    L += ["", "時間帯 (ニューヨーク時間、足の始まり) ごとの Yahoo − 仲値 の差と半スプレッド (bp、各ペアの中央値の平均):", "",
          "| ペア | 項目 | " + " | ".join(str(h) for h in range(12, 22)) + " |", "|---|---|" + "---|" * 10]
    for side, cs in groups.items():
        lv = {h: (_mean([ov["pairs"][c]["level_by_hour"].get(h, {}).get("mid") for c in cs]),
                  _mean([ov["pairs"][c]["level_by_hour"].get(h, {}).get("hs") for c in cs])) for h in range(12, 22)}
        lab = "、".join(cs)
        L += [f"| {lab} | Yahoo − 仲値 | " + " | ".join(_b(lv[h][0]) for h in range(12, 22)) + " |",
              f"| {lab} | 半スプレッド | " + " | ".join(f"{lv[h][1]:.2f}" if math.isfinite(lv[h][1]) else "—" for h in range(12, 22)) + " |"]
    L.append("")
    L += ["古い気配 (更新されない価格) の目安として、終値が前の足から動かなかった足と、値が1つしかない足 (高値 = 安値) の割合、"
          "およびロールオーバー前後 (16〜18時台) の1時間ごとの値動きが売値・仲値・買値のどれに一番近いか (差の二乗平均):", "",
          "| ペア | 動きなし: Yahoo 全体 / 17時台 | 動きなし: Dukascopy 全体 / 17時台 | 高値 = 安値: Yahoo / Dukascopy | 16〜18時台の値動きの差 (bp): 売値 / 仲値 / 買値 |",
          "|---|---|---|---|---|"]
    for code in pairs:
        st, tr = ov["pairs"][code]["stale"], ov["pairs"][code]["track_rms_roll"]
        L.append(f"| {code} | {_p(st['zero_yahoo'])} / {_p(st['zero_yahoo_17'])} | {_p(st['zero_duka'])} / {_p(st['zero_duka_17'])} | "
                 f"{_p(st['flat_yahoo'])} / {_p(st['flat_duka'])} | {tr['bid']:.2f} / {tr['mid']:.2f} / {tr['ask']:.2f} |")
    L.append("")
    L += ["ロールオーバー前後の1時間ごとの平均の動き (bp)。金利の高い通貨の向きを + にそろえ、全ペアで平均しました "
          "(日付の切り替えで受け渡し日が1日 (水曜日は3日) 延びると、仲値はその分の金利差だけ金利の高い通貨の安い方へずれるはずです):", ""]
    for side, cs in groups.items():
        for w, lab in (("other", "月・火・木曜日"), ("wed", "水曜日")):
            L += [f"{'、'.join(cs)} (Yahoo は{SIDE_NAME[side]})、{lab}:", "",
                  "| 時間 (NY) | " + " | ".join(str(h) for h in range(14, 21)) + " |", "|---|" + "---|" * 7]
            for k, name in (("yahoo", "Yahoo"), ("mid", "仲値"), ("bid", "売値"), ("ask", "買値")):
                L.append(f"| {name} | " + " | ".join(_b(_mean([ov['pairs'][c]['carry_profile'][w].get(h, {}).get(k) for c in cs]))
                                                     for h in range(14, 21)) + " |")
            L.append("")
    L += [f"同じ時点で Yahoo から計算した方向 (本番と同じ) を、Yahoo の値動き、仲値の値動き、Yahoo が従う気配 (上の判定: 売値または仲値) の値動きで採点した結果 "
          f"({ov['span'][0][:10]}〜{ov['span'][1][:10]}。Yahoo で2,000本の履歴がそろった時点のうち、両方に同じ2本の足がある時点のみ。スプレッド後は Dukascopy の実際のスプレッドで買値で買い売値で売った場合、"
          "スワップ後はさらにロールオーバーをまたぐ足のスワップ (短期金利差、水曜日は3日分) を加えたもの):", "",
          "| 基準 | 回数 | 的中率: Yahoo | 仲値 | Yahoo が従う気配 | 平均: Yahoo (bp) | 仲値 (bp) | スプレッド後 (bp) | スプレッド・スワップ後 (bp) | 仲値で計算した方向と一致 | 仲値でも同じ基準 | 仲値で計算した方向の的中率 (同じ期間): 回数 / 的中率 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in ov["calls"].items():
        own = ov["duka_own"][k]
        L.append(f"| {_tier_label(k)} | {x['yahoo']['n']:,} | {_p(x['yahoo'].get('hit'))} | {_p(x['mid'].get('hit'))} | {_p(x['side'].get('hit'))} | "
                 f"{_b(x['yahoo'].get('bp'))} | {_b(x['mid'].get('bp'))} | {_b(x['mid'].get('net'))} | {_b(x['mid'].get('net_swap'))} | "
                 f"{_p(x['agree'], 0)} | {_p(x['agree_tier'], 0)} | {_nh(own)} |")
    L += ["", "時間帯別 (ニューヨーク時間、足の始まり。回数 / 的中率: Yahoo・仲値・Yahoo が従う気配):", "",
          "| 基準 | 時間 | 回数 | Yahoo | 仲値 | Yahoo が従う気配 |", "|---|---|---|---|---|---|"]
    for k, bh in ov["by_hour"].items():
        for h in ("16", "17", "18", "other"):
            if h in bh:
                x = bh[h]
                L.append(f"| {_tier_label(k)} | {h if h != 'other' else 'その他'} | {x['yahoo']['n']:,} | {_p(x['yahoo'].get('hit'))} | "
                         f"{_p(x['mid'].get('hit'))} | {_p(x['side'].get('hit'))} |")
    L.append("")
    # ---------------- long run
    lt = lg["ladder"]
    L += ["## 2. 2004年からのウォークフォワード (Dukascopy の仲値)", "",
          f"本番と同じ規則 (直近6,000本、予測日の0時 UTC より前の足だけ、曜日×時間の枠を優先) を、{lg['span'][0][:10]}〜{lg['span'][1][:10]} "
          f"のすべての1時間足の時点 ({lg['origins']:,}時点、{len(pairs)}ペア、市場が開いている足) に当てはめました。"
          "調整期間は2016年まで、検証期間は2017年からです。", "",
          "| 期間 | 方向を示した回: 回数 / 的中率 | 平均 (bp) | 足に占める割合 | うち中 (2 ≤ \\|t\\| < 4) | うち高 (\\|t\\| ≥ 4) | 高の平均 (bp) | 高のスプレッド後 (bp) | 高のスプレッド・スワップ後 (bp) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p, lab in (("tune", "調整 (2004〜2016)"), ("test", "検証 (2017〜)")):
        a, h = lt[p]["t2"], lt[p]["t4"]
        L.append(f"| {lab} | {_nh(a)} | {_b(a.get('bp'))} | {_p(a.get('share'), 1)} | {_nh(lt[p]['mid'])} | {_nh(h)} | "
                 f"{_b(h.get('bp'))} | {_b(h.get('net_med'))} | {_b(h.get('net_swap'))} |")
    L += ["", "年ごと (回数 / 的中率):", "",
          "| 年 | 方向を示した回 | 中 (2 ≤ \\|t\\| < 4) | 高 (\\|t\\| ≥ 4) | \\|t\\| ≥ 6 | \\|t\\| ≥ 8 | 高の平均 (bp) | 高のスプレッド後 (bp) |",
          "|---|---|---|---|---|---|---|---|"]
    for y, x in lg["per_year"].items():
        L.append(f"| {y} | {_nh(x['all'])} | {_nh(x['mid'])} | {_nh(x['high'])} | {_nh(x['t6'])} | {_nh(x['t8'])} | "
                 f"{_b(x['high'].get('bp'))} | {_b(x['high'].get('net_med'))} |")
    L += ["", "高確度 (\\|t\\| ≥ 4) の時間帯別 (ニューヨーク時間、足の始まり、回数 / 的中率):", "",
          "| 期間 | " + " | ".join(str(h) for h in range(24)) + " |", "|---|" + "---|" * 24]
    for p, lab in (("tune", "調整"), ("test", "検証")):
        bh = lg["by_hour"][p]["high"]
        L.append(f"| {lab} | " + " | ".join(f"{bh[h]['n']} / {_p(bh[h]['hit'], 0)}" if h in bh else "—" for h in range(24)) + " |")
    L += ["", "ペア別 (検証期間、方向を示した回 / 高確度):", "", "| ペア | 方向を示した回 | 高確度 |", "|---|---|---|"]
    for code, x in lg["by_pair"]["test"].items():
        L.append(f"| {code} | {_nh(x['all'])} | {_nh(x['high'])} |")
    L += ["", "### ロールオーバーの仕組みの確認", "",
          "月〜木曜日のロールオーバーの時刻を含む足 (ニューヨーク時間17時に始まる足) の仲値の動きを、ペア×年×(水曜日か否か) で平均し、"
          f"受け渡し日が延びる分の金利差 (短期金利の差 ÷ 360、水曜日は3日分) と比べました ({rl['groups']}組)。"
          f"傾き {rl['slope']:.2f} (1 なら金利差どおり)、切片 {_b(rl['intercept'])} bp、相関 {rl['corr']:.2f}。"
          "前後の時間を含めると、16時台のスプレッドの広がり方の偏り (買値が大きく上がる) で仲値も動くため、関係はぼやけます:", "",
          "| 足 (NY、始まり) | 組 | 傾き | 相関 | 月・火・木: 平均の動き / 見込み (bp) | 水: 平均の動き / 見込み (bp) |", "|---|---|---|---|---|---|"]
    for x in rl["windows"]:
        o, w_ = x["by_weekday"].get("mon_tue_thu", {}), x["by_weekday"].get("wed", {})
        L.append(f"| {'・'.join(str(h) for h in x['hours'])}時 | {x['groups']} | {x['slope']:.2f} | {x['corr']:.2f} | "
                 f"{_b(o.get('move'))} / {_b(o.get('step'))} | {_b(w_.get('move'))} / {_b(w_.get('step'))} |")
    L += ["", "金利の高い通貨の向きを + にそろえた仲値の1時間ごとの平均の動き (bp、ニューヨーク時間):", "",
          "| 期間 | 曜日 | " + " | ".join(str(h) for h in range(14, 21)) + " |", "|---|---|" + "---|" * 7]
    for p, lab in (("tune", "調整"), ("test", "検証")):
        for w, wl in (("other", "月・火・木"), ("wed", "水"), ("fri", "金")):
            pr = rl["profile"][p][w]
            L.append(f"| {lab} | {wl} | " + " | ".join(_b(pr.get(h)) for h in range(14, 21)) + " |")
    L.append("")
    # ---------------- stricter tiers
    L += ["## 3. さらに厳しい基準", "",
          "的中率90%を目標に、|t| の基準を上げた場合と、その日の統計で |t| が大きい上位 N 枠 (週168枠のうち) だけにした場合を比べました。"
          "新しい基準は調整期間だけで選び (調整期間で90%に届く最小の |t|、30回以上)、検証期間はその確認にだけ使います。"
          "「足に占める割合」はすべての予測時点に対する方向を示した回の割合です。スプレッド後は、Dukascopy のそのペア・その時間 (ニューヨーク時間) の"
          "往復スプレッド (予測時点と次の足の終値の半スプレッドの和) の中央値を平均の動きから引いた値で、Yahoo には重なる期間の中央値、"
          "Dukascopy にはその年の中央値を使いました (実際のスプレッドで引いた値は season_long.json の net)。"
          "Yahoo の平均の動きには、売値に従うペアで売値だけに出る16時・18時の動きが含まれるため、Yahoo のスプレッド後の値は売買で得られる損益ではありません "
          "(売買で得られる損益は、1. の表の仲値・実際のスプレッドで採点した値です)。", "",
          f"### Yahoo 1時間足: 本番が予測する足 (市場が開いている足) — 調整 〜{yh['split'][:10]}、検証 {yh['split'][:10]}〜", ""]
    L += _ladder_table(yh["open"], keys) + [""]
    L += ["### Yahoo 1時間足: すべての足 (research/direction.md と同じ数え方)", "",
          f"Yahoo には金曜日のニューヨーク時間17時 (市場が閉まった後) に1回だけの気配の足が出ることがあり、research/direction.md の検証はこの足も数えていました "
          f"(方向を示した回: 調整 {yh['closed_calls']['tune']:,}回・検証 {yh['closed_calls']['test']:,}回、うち高確度の的中率 "
          f"調整 {_p(yh['closed_high']['tune'].get('hit'))} ({yh['closed_high']['tune'].get('n', 0)}回)・検証 {_p(yh['closed_high']['test'].get('hit'))} "
          f"({yh['closed_high']['test'].get('n', 0)}回))。本番のサーバーは市場が閉まっている足を予測しないため、上の表の方が画面の成績に近い数字です。", ""]
    L += _ladder_table(yh["all"], keys) + [""]
    L += [f"### Yahoo 15分足 (直近約60日: {q15['span'][0][:10]}〜{q15['span'][1][:10]}、調整期間なし)", "",
          "| 基準 | 回数 / 的中率 (95%下限) | 足に占める割合 | 平均 (bp) |", "|---|---|---|---|"]
    for k in [k for k in keys if not k.startswith("top")]:
        x = q15["open"][k]
        L.append(f"| {_tier_label(k)} | {_nh(x)} ({_p(x.get('hit_lo'))}) | {_p(x.get('share'), 2)} | {_b(x.get('bp'))} |")
    L += ["", "### Dukascopy 仲値 1時間足 (調整 2004〜2016、検証 2017〜)", ""]
    L += _ladder_table(lt, keys) + [""]
    L += _choice_text(res) + [""]
    L += _method_notes(res)
    return "\n".join(L) + "\n"


def _hour_groups(ov: dict, tier: str, side: str | None = None) -> dict:
    """Overlap calls of a tier pooled over the bars at 16:00 and 18:00 New York ("rim": the spread widens
    and narrows), at 17:00 ("roll") and the rest ("other"): n and hit on Yahoo, mid and the quote Yahoo
    follows; all pairs, or only those where Yahoo follows ``side``."""
    bh = ov["by_hour"][tier] if side is None else ov["by_hour_side"].get(side, {}).get(tier, {})
    out = {}
    for name, hours in (("rim", ("16", "18")), ("roll", ("17",)), ("other", ("other",))):
        xs = [bh[h] for h in hours if h in bh]
        n = sum(x["yahoo"]["n"] for x in xs)
        out[name] = {"n": n} | {k: (sum((x[k].get("hit") or 0) * x[k]["n"] for x in xs) / n if n else None)
                                 for k in ("yahoo", "mid", "side")}
    return out


def _summary(res: dict) -> dict:
    """The pooled figures the conclusion and the recommendation quote."""
    ov, lg, yh = res["overlap"], res["long"], res["yahoo"]
    pairs = res["params"]["pairs"]
    P = ov["pairs"]
    years = lg["per_year"]
    big = {y: x["high"] for y, x in years.items() if x["high"].get("n", 0) >= 50}
    groups = _groups(ov, pairs)
    bid = groups.get("bid", [])
    return {
        "groups": groups,
        "bid_name": "円のペア (" + "・".join(bid) + ")" if bid and all(PAIRS[c].quote == "JPY" for c in bid) else "・".join(bid),
        "slope": {k: _mean([P[c]["hs_slope"] for c in cs]) for k, cs in groups.items()},
        "offset": {k: _mean([P[c]["offset_bp"] for c in cs]) for k, cs in groups.items()},
        "near_bid": _mean([P[c]["nearest"]["bid"] for c in bid]),
        "lev_mid": _mean([P[c]["level_bp"]["mid"] for c in bid]),
        "hs": _mean([P[c]["half_spread_bp"] for c in bid]),
        "lev_roll": _mean([P[c]["level_by_hour"].get(h, {}).get("mid") for c in bid for h in (16, 17)]),
        "hs_roll": _mean([P[c]["level_by_hour"].get(h, {}).get("hs") for c in bid for h in (16, 17)]),
        "lev_roll_mid": {c: _mean([P[c]["level_by_hour"].get(h, {}).get("mid") for h in (16, 17)]) for c in groups.get("mid", [])},
        "stale": {c: P[c]["stale"] for c in pairs if P[c]["stale"]["zero_yahoo"] > max(0.02, 3 * P[c]["stale"]["zero_duka"])},
        "fresh": [c for c in pairs if not P[c]["stale"]["zero_yahoo"] > max(0.02, 3 * P[c]["stale"]["zero_duka"])],
        "g2": _hour_groups(ov, "t2"), "g4": _hour_groups(ov, "t4"),
        "b2": _hour_groups(ov, "t2", "bid"), "b4": _hour_groups(ov, "t4", "bid"),
        "big_min": min(big.items(), key=lambda kv: kv[1]["hit"]) if big else None,
        "big_max": max(big.items(), key=lambda kv: kv[1]["hit"]) if big else None,
        "thin_years": [y for y, x in years.items() if x["high"].get("n", 0) < 50],
        "yahoo_open_high": (yh["open"]["tune"]["t4"], yh["open"]["test"]["t4"]),
        "yahoo_all_high": (yh["all"]["tune"]["t4"], yh["all"]["test"]["t4"]),
    }


def _years_text(years: list[int]) -> str:
    """"2009〜2021, 2004" style list of runs of consecutive years."""
    out, run = [], []
    for y in sorted(years):
        if run and y == run[-1] + 1:
            run.append(y)
        else:
            if run:
                out.append(run)
            run = [y]
    if run:
        out.append(run)
    return "、".join(f"{r[0]}" if len(r) == 1 else f"{r[0]}〜{r[-1]}" for r in out)


def _conclusion(res: dict) -> list[str]:
    S = _summary(res)
    ov, lg, rl, ch = res["overlap"], res["long"], res["roll"], res["choice"]
    lt = lg["ladder"]
    c4, c2 = ov["calls"]["t4"], ov["calls"]["t2"]
    g2, g4, b2, b4 = S["g2"], S["g4"], S["b2"], S["b4"]
    yo_tu, yo_te = S["yahoo_open_high"]
    G = S["groups"]
    first = []
    if G.get("bid"):
        first.append(f"{S['bid_name']} では売値 (bid) です。Yahoo − 仲値 は半スプレッドとともに動き (傾き {S['slope']['bid']:+.2f}、−1 なら売値そのもの)、"
                     f"中央値 {_b(S['lev_mid'])} bp (半スプレッド {S['hs']:.2f} bp)、一番近いのは売値 ({_p(S['near_bid'], 0)} の足)。"
                     f"ロールオーバー前後 (ニューヨーク時間16〜17時台) はスプレッドが広がり、差も {_b(S['lev_roll'])} bp (半スプレッド {S['hs_roll']:.2f} bp) に広がります")
    if G.get("mid"):
        offs = "・".join(f"{c} {_b(res['overlap']['pairs'][c]['offset_bp'])}" for c in G["mid"])
        first.append("・".join(G["mid"]) + f" では仲値に一定の差 ({offs} bp) を足した価格で、スプレッドが広がっても差はほとんど変わらず "
                     f"(傾き {S['slope']['mid']:+.2f}、16〜17時台の差 " + "・".join(_b(v) for v in S["lev_roll_mid"].values()) + " bp)、"
                     "ロールオーバー前後の値動きも仲値に一番近くなります")
    if G.get("ask"):
        first.append("・".join(G["ask"]) + f" では買値です (傾き {S['slope']['ask']:+.2f})")
    head = "売値 (bid) です" if list(G) == ["bid"] else "ペアによって売値 (bid) か、仲値に一定の差を足した価格です"
    stale_text = ""
    if S["fresh"]:
        fr = [res["overlap"]["pairs"][c]["stale"] for c in S["fresh"]]
        stale_text += ("・".join(S["fresh"]) + f" は終値が前の足から動かない足が {_p(min(x['zero_yahoo'] for x in fr))}〜"
                       f"{_p(max(x['zero_yahoo'] for x in fr))} (Dukascopy {_p(min(x['zero_duka'] for x in fr))}〜{_p(max(x['zero_duka'] for x in fr))}) "
                       "と少なく、古い気配は目立ちません")
    for c, x in S["stale"].items():
        stale_text += (("。" if stale_text else "") + f"{c} の Yahoo は終値が前の足から動かない足が {_p(x['zero_yahoo'])} (17時台 {_p(x['zero_yahoo_17'])}) と、"
                       f"Dukascopy ({_p(x['zero_duka'])}) より多く、古い気配が残ることがあります")
    stale_text += "。金曜日の取引終了後には Yahoo だけに1本だけの気配の足が出ます (後述)。" if stale_text else ""
    if rl["corr"] >= 0.5 and 0.5 <= rl["slope"] <= 2.0:
        mech = (f"17時の足の動きは、受け渡し日が1日 (水曜日は3日) 延びる分の金利差 (スワップポイント) とともに大きくなります "
                f"(ペア×年の平均で傾き {rl['slope']:.2f}・相関 {rl['corr']:.2f}。傾き 1 なら短期金利差どおり)。")
    else:
        mech = (f"17時の足の動きと、受け渡し日が延びる分の金利差との関係ははっきりしません (傾き {rl['slope']:.2f}・相関 {rl['corr']:.2f})。")
    L = ["## 結論", "",
         f"- **Yahoo の為替の価格は{head}。** 重なる期間の同じ足で比べると、" + "。".join(first) + "。",
         "- **古い気配:** " + stale_text,
         f"- **偏りのうち、ロールオーバーの前後の時間 (16時・18時に始まる足) の分は、{S['bid_name']} の Yahoo (売値) だけに出る見かけです。** "
         f"Yahoo で計算した方向 (本番と同じ) を同じ足の仲値で採点すると、これらのペアの16時・18時の足 ({b2['rim']['n']:,}回) は Yahoo "
         f"{_p(b2['rim']['yahoo'])}・売値 {_p(b2['rim']['side'])} に対して仲値 {_p(b2['rim']['mid'])}、うち確度: 高 ({b4['rim']['n']:,}回) は Yahoo "
         f"{_p(b4['rim']['yahoo'])} に対して仲値 {_p(b4['rim']['mid'])} でした。16時台にスプレッドが広がって売値が下がり、18時台に戻って売値が上がる動きで、"
         "仲値には偏りがありません。",
         f"- **ロールオーバーの時刻を含む足 (17時に始まる足) の偏りは仲値にもあります。** 17時の足は Yahoo {_p(g2['roll']['yahoo'])}・仲値 "
         f"{_p(g2['roll']['mid'])} ({g2['roll']['n']:,}回)、うち確度: 高 は Yahoo {_p(g4['roll']['yahoo'])}・仲値 {_p(g4['roll']['mid'])} "
         f"({g4['roll']['n']:,}回)。ほかの時間は Yahoo {_p(g2['other']['yahoo'])}・仲値 {_p(g2['other']['mid'])} で差がありません。" + mech +
         "このずれはポジションを持ち越す人が受け取る (払う) スワップと同じ額で相殺されるため、**価格の表示としては本物でも、売買で得られるものではありません。**",
         f"- そのため確度: 高 (|t| ≥ 4) の的中率は、重なる期間で Yahoo {_p(c4['yahoo'].get('hit'))} に対して仲値 {_p(c4['mid'].get('hit'))} "
         f"({c4['yahoo']['n']:,}回)、方向を示した回全体では Yahoo {_p(c2['yahoo'].get('hit'))}・仲値 {_p(c2['mid'].get('hit'))} です。"
         f"確度: 高 の平均の動きから実際のスプレッドを引くと {_b(c4['mid'].get('net'))} bp、スワップも含めると {_b(c4['mid'].get('net_swap'))} bp です。",
         f"- **20年の仲値 (Dukascopy、2004年〜) でも効果はありますが、強さは金利差しだいです。** 方向を示した回の的中率は調整期間 (〜2016) "
         f"{_p(lt['tune']['t2'].get('hit'))}・検証期間 (2017〜) {_p(lt['test']['t2'].get('hit'))}、確度: 高 は {_p(lt['tune']['t4'].get('hit'))} "
         f"({lt['tune']['t4']['n']:,}回)・{_p(lt['test']['t4'].get('hit'))} ({lt['test']['t4']['n']:,}回)。"]
    if S["big_min"]:
        (y0, x0), (y1, x1) = S["big_min"], S["big_max"]
        L[-1] += (f"確度: 高 が年50回以上あった年の的中率は {_p(x0.get('hit'))} ({y0}年) 〜 {_p(x1.get('hit'))} ({y1}年) で、"
                  f"{_years_text(S['thin_years'])} 年は確度: 高 が年50回に届きません。ずれの大きさが金利差とともに変わるため、"
                  "金利差が大きい年ほど偏りがはっきりし、金利がゼロ近くに並んだ年はほとんど確度: 高 になりません。")
    yo = ch["yahoo_open"]
    T = yo["t"]
    q = res["yahoo_15m"]["open"]
    best = max((k for k in q if k.startswith("t") and q[k].get("n", 0) >= 30), key=lambda k: q[k].get("hit") or 0, default=None)
    q15_text = (f"15分足で一番高いのは {_tier_label(best).replace(chr(92), '')} の {_p(q[best].get('hit'))} ({q[best]['n']:,}回、95%下限 "
                f"{_p(q[best].get('hit_lo'))}) ですが、直近約60日しかなく調整期間で確かめられません。") if best else ""
    if ch["add_top_tier"]:
        L.append(f"- **的中率90%の基準 (|t| ≥ {T:g}) は、調整・検証の両方と仲値で90%を超えました。**")
    else:
        why = []
        if T is None:
            why.append("調整期間で90%に届く基準がありません")
        else:
            te = yo["test"]
            why.append(f"調整期間で選んだ |t| ≥ {T:g} (調整 {_p(yo['tune'].get('hit'))}、{yo['tune'].get('n', 0):,}回、95%下限 "
                       f"{_p(yo['tune'].get('hit_lo'))}) は検証期間で {_p(te.get('hit'))} ({te.get('n', 0):,}回)")
            du = (_at(lt["tune"], T), _at(lt["test"], T))
            why.append(f"20年の仲値では同じ基準で調整 {_hn(du[0])}・検証 {_hn(du[1])}")
        L.append("- **的中率90%を安定して出せる基準はありませんでした。** 本番が予測する足 (市場が開いている足) の Yahoo で、"
                 + "、".join(why) + "。|t| を上げるほど回数が減り (検証期間で数十回、全体の0.1%程度)、的中率は80%台で頭打ちです。"
                 + q15_text + "そのため「確度: 最高」は追加しません。")
    L.append(f"- 本番の区分 (確度: 中・高) はそのままでよいと考えます。ただし研究 (research/direction.md) の確度: 高 の成績 "
             f"(調整 {_p(S['yahoo_all_high'][0].get('hit'))}・検証 {_p(S['yahoo_all_high'][1].get('hit'))}) は、本番が予測しない金曜日の取引終了後の足を含んでおり、"
             f"本番が予測する足だけでは調整 {_p(yo_tu.get('hit'))} ({yo_tu['n']:,}回)・検証 {_p(yo_te.get('hit'))} ({yo_te['n']:,}回) です。"
             "画面の注意書きは「偏りはロールオーバー (日本時間の朝6〜7時) の気配の仕組みによるもので、スプレッドとスワップを差し引くと利益になりません」"
             "とするのがよいと考えます (詳しくは「4. 推奨」)。")
    return L


def _choice_text(res: dict) -> list[str]:
    ch = res["choice"]
    L = ["## 4. 推奨", "",
         "新しい基準は、調整期間で的中率90%に届く最小の |t| (30回以上) として選び、検証期間とDukascopyの仲値で確かめました。"
         f"採用の条件は、本番のデータ (Yahoo、本番が予測する足) の調整・検証の両方で90%以上、検証期間で{MIN_TEST_CALLS}回以上、"
         "仲値の2004〜2016年・2017年〜の両方で90%以上です。", "",
         "| データ | 調整期間で選んだ基準 | 調整: 回数 / 的中率 | 検証: 回数 / 的中率 | 両方の期間で90%以上になった基準 |", "|---|---|---|---|---|"]
    for name, lab in (("yahoo_open", "Yahoo 1時間足 (本番が予測する足)"), ("yahoo_all", "Yahoo 1時間足 (すべての足)"),
                      ("duka", "Dukascopy 仲値 (2004〜)")):
        x = ch[name]
        both = "、".join(_tier_label(k) for k in x["both"]) or "なし"
        chosen = "なし" if x["t"] is None else _tier_label(f"t{x['t']:g}")
        L.append(f"| {lab} | {chosen} | {_nh(x['tune'])} | {_nh(x['test'])} | {both} |")
    ck = ch["checks"]
    names = {"chosen_on_tune": "調整期間で基準が決まる", "test_hit": "検証期間で90%以上", "test_calls": f"検証期間で{MIN_TEST_CALLS}回以上",
             "mid_tune": "仲値 2004〜2016 で90%以上", "mid_test": "仲値 2017〜 で90%以上"}
    L += ["", "条件の確認 (Yahoo、本番が予測する足で選んだ基準): " + "、".join(f"{names[k]} {'○' if v else '×'}" for k, v in ck.items()), ""]
    kh = ch["keep_high"]
    S = _summary(res)
    b4, core = S["b4"], res["overlap"]["core"]["t4"]
    L += ["1. **本番の区分はそのまま (|t| ≥ 2 で方向、|t| ≥ 4 で確度: 高)。** 確度: 高 は本番が予測する足の Yahoo で "
          f"調整 {_p(kh['yahoo_open'][0])}・検証 {_p(kh['yahoo_open'][1])}、20年の仲値でも {_p(kh['duka'][0])}・{_p(kh['duka'][1])} と、"
          "偶然より明らかに当たります。表示する検証の的中率は、本番が予測する足 (市場が開いている足) で数えた値にするのが正確です。"
          f"なお、実際に売買できる価格 (仲値) の向きを示したい場合は、{S['bid_name']} の16時・18時に始まる足 (売値だけの動き) の方向を外す選択肢があります "
          f"(重なる期間の確度: 高 で、外す {b4['rim']['n']:,}回は仲値で {_p(b4['rim']['mid'])}、残る {core['mid']['n']:,}回は仲値で {_p(core['mid'].get('hit'))}・"
          f"Yahoo で {_p(core['yahoo'].get('hit'))})。"
          "ただし画面のチャートと答え合わせは Yahoo の価格 (円のペアでは売値) なので、外すと画面上の的中率は下がります。",
          "2. " + ("**確度: 最高 を追加する。**" if ch["add_top_tier"] else
                   "**確度: 最高 (的中率90%) は追加しない。** 条件を満たす基準がなく、|t| を上げても的中率は80%台で、回数は検証期間で数十回まで減ります。"),
          "3. **画面の注意書き (案):** 「方向の偏りの大部分は、ニューヨーク時間17時 (日本時間の朝6〜7時) のロールオーバー前後の価格の仕組みによるものです。"
          "円のペアの表示価格 (Yahoo) は売値で、この時間はスプレッドが広がって売値が下がり、その後に戻ります。ロールオーバーでは受け渡し日が延びる分だけ価格がずれ "
          "(水曜日は3日分)、その分はスワップで相殺されます。表示される価格の向きとしては当たりますが、スプレッドとスワップを差し引くと売買の利益にはなりません。」"]
    return L


def _method_notes(res: dict) -> list[str]:
    ck = res["check"]
    worst = max((x["max_abs_t"] for x in ck.values()), default=float("nan"))
    n = sum(x["origins"] for x in ck.values())
    return ["## 方法の注記", "",
            "- 方向の計算は aifx/season.py と同じ規則を、全時点をまとめて計算できる形に書き直したものです。"
            f"Dukascopy の {', '.join(ck)} で無作為に選んだ日の {n:,}時点を season.slot_stats / bar_drift で計算し直し、"
            f"t 値の差は最大 {worst:.1e}、方向を示すかどうかの食い違いは {sum(x['call_mismatch'] for x in ck.values())} 件でした。"
            "Yahoo では research/direction.md の数字 (1時間足 8,493回・61.1%、確度: 高 1,069回・79.8% など) をそのまま再現します。",
            "- 各時点の統計は、予測する日の0時 (UTC) より前の6,000本だけから作ります (先読みなし)。週明けの窓開けなど、1時間より空いた足の動きは統計から除きます。",
            "- Dukascopy は1時間ごとの売値と買値のローソク足 (取引のない週末・祝日の時間は欠ける) から、仲値 = (売値 + 買値) / 2、スプレッド = 買値 − 売値 (終値の時点) を作りました。"
            "2004年から、6,000本の窓がそろう時点を使います。",
            "- 「スプレッド後」は、予測の時点の買値で買い (売りなら売値で売り)、次の足の終値の売値で売る (買値で買い戻す) ときの損益です。"
            "「スワップ後」は、ロールオーバーをまたぐ足 (ニューヨーク時間17時に始まる足) に、短期金利の差 (FRED、公表の遅れを考慮した当時の値) ÷ 360 "
            "(水曜日は3日分) のスワップを加えたものです (銀行間の条件。個人向けのスワップはもっと不利です)。",
            "- 的中率は動きがゼロの回を除いて数え、95%下限は Wilson の区間です。t は週ごとに全ペアの結果を合計して計算しています。",
            "- 週の上位 N 枠: その日の統計で、曜日×時間の168枠のうち |t| が大きい N 枠に入る足だけ方向を示します。"
            "15分足は約60日分しかなく、曜日×15分の枠は15本に届かないため、15分足の方向はすべて「15分」の枠 (曜日なし) から出ています。",
            f"- 計算時間 {res.get('seconds', 0):.0f} 秒 (`python -m aifx.research_season_long`)。"]


if __name__ == "__main__":
    run()
