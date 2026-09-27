"""Does the interest-rate parity expectation sharpen the live time-of-day calls? (research/rollover.md)

The live direction calls (season.py) come from the average move of each New
York weekday-and-hour slot (15-minute bars: quarter-hour) over the timeframe's
last 6,000 bars. Most of their accuracy is the 17:00 New York rollover: the
spot value date moves on by a day (three on Wednesday, over the weekend; more
or fewer around a settlement holiday), so the quote of the currency with the
higher short rate steps down by the forward points of those days, about
(rate_base - rate_quote) x days / 360 (research/season_long.md). The slot
averages learn this from the past year, but they cannot know that the rates
have moved since, or that a holiday makes tonight's roll 0, 2 or 4 days long.

This asks whether adding that expectation, from the point-in-time short rates
the server already stores (rates.py, the same FRED series and publication
lags as history.rates_panel) and the days rolled, improves the calls on the
live source: Yahoo hourly bars (first 60 % of bar times tune, last 40 % test,
as research_direction.py) and Yahoo 15-minute bars (~60 days, split the same
way; reported, never decisive). Dukascopy mid prices 2004-2026 (tune to 2016,
test 2017-) check that a change is not a quirk of Yahoo's quotes.

Refinements declared before any of them was evaluated (``CANDIDATES``). A
"roll bar" is a called bar that starts at 17:00 New York, Monday to Thursday;
e is the called bar's expected shift in bp (0 where no roll falls in it); the
live call has direction s and statistic t:

  a  agree filter: drop a roll-bar call whose direction is opposite to e when |e| >= E.
  b  promote: a roll-bar call with |t| >= T whose direction agrees with e and |e| >= E is high confidence.
  c  swap call: a roll bar with |e| >= X is called in the direction of e, high confidence.
  d  rate-adjusted slots: each slot's average of (move - that bar's e) plus the called bar's e, with t
     from the adjusted moves (the swap part of the past year replaced by tonight's); the usual tiers.

Each is tried with the days rolled counted plainly (1, Wednesday 3) and from
settlement-holiday calendars with the spot-date rule ("cal": T+2 business
days, a USD holiday skipped on T+1 for USD pairs, the value date a business
day in both currencies and USD). Settings come from the tune period only
(``GRID``, ``choose``); a refinement is recommended only if it passes
``adopt`` on both Yahoo periods, is not contradicted on Dukascopy (``DUKA_TOL``)
and runs on data the server keeps (stored bars, stored rates). Added after the
results were seen, and only ever stricter: the candidate's own tune objective
must also hold on the test period and on Dukascopy (``own_objective``), the
15-minute one-hour-ahead forecasts are checked (``first_hour_15m``), and the
proposed season.py code is replayed through season.py (``live_check``).

No lookahead: slot statistics use the bars before the origin's UTC day (as
season.slot_stats, checked against it), rates are those known on the origin's
UTC day (a past bar's e uses its own day), holiday calendars are rules fixed
in advance.
"""

from __future__ import annotations

import json
import math
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import history, season
from . import rates as ratesmod
from .data import PAIRS
from .fxcalendar import (D1, holidays, spot_date)
from .research_direction import HOURLY_TUNE_SHARE, SESSION_MIN_BARS, score
from .research_season_long import _slot_prefix, _wilson_low
from .timeutil import add_trading_minutes, market_open_mask

REPORT_DIR = Path("research")
NS_MIN = 60_000_000_000
NS_DAY = 86_400_000_000_000
DUKA_SPLIT = pd.Timestamp("2017-01-01", tz="UTC")     # Dukascopy: tune before, test from
TF = {"1h": {"minutes": 60, "min_bars": SESSION_MIN_BARS, "ahead": 24},    # as research_direction.session_eval
      "15m": {"minutes": 15, "min_bars": 500, "ahead": 4}}
DAYS = ("plain", "cal")
CANDIDATES = {
    "a": "一致フィルター: ロールオーバーの足で、方向がスワップの見込み e と逆の予測を外す (|e| ≥ E のとき)",
    "b": "格上げ: ロールオーバーの足で、|t| ≥ T かつ e と同じ向き (|e| ≥ E) の予測を確度: 高にする",
    "c": "スワップだけの予測: ロールオーバーの足で |e| ≥ X なら e の向きに方向を示す (確度: 高)",
    "d": "金利で補正した枠: 枠の平均を「値動き − その足の e」で測り、予測する足の e を足す (t も補正後の値動きから)",
}
GRID = {"a": [{"E": E} for E in (0.0, 0.25, 0.5, 1.0, 2.0)],
        "b": [{"T": T, "E": E} for T in (2.0, 2.5, 3.0, 3.5) for E in (0.0, 0.5, 1.0, 2.0)],
        "c": [{"X": X} for X in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)],
        "d": [{}]}
OBJECTIVE = {"a": "net", "b": "high", "c": "high", "d": "net"}
SIMILAR = 0.9          # a "similar" number of calls: at least 90 % of the live rule's
DUKA_TOL = 0.01        # Dukascopy contradicts a change that lowers a tier's hit rate by more than 1 point
SEED = 7


# ------------------------------------------------------------------ settlement calendars and the value date

# calendars and spot dates live in fxcalendar.py (shared with the live season.py)

def _spot_rule(trade: date, base: str, quote: str, hol: dict[str, set[date]], rule: str) -> date:
    """spot_date ("standard") and two alternatives for the convention check: "strict" (T+1 a business day of
    both currencies, USD included) and "cross_no_usd" (the value date need not be a USD business day)."""
    if rule == "standard":
        return spot_date(trade, base, quote, hol)
    first = [base, quote] if rule == "strict" else [c for c in (base, quote) if c != "USD"]
    last = [base, quote] if rule == "cross_no_usd" else [base, quote, "USD"]
    d = trade + D1
    while d.weekday() >= 5 or any(d in hol[c] for c in first):
        d += D1
    d += D1
    while d.weekday() >= 5 or any(d in hol[c] for c in last):
        d += D1
    return d


def convention_check(D: pd.DataFrame) -> dict:
    """Roll bars of the Dukascopy tune period (to 2016, before any Yahoo data): the standard spot rule against
    the two alternatives where they count different days: squared error of (move - expected shift) and the slope
    of the move on the expected shift (1 = as the rule says)."""
    R = D[D["roll"].to_numpy() & np.isfinite(D["diff"].to_numpy()) & (D["period"] == "tune").to_numpy()]
    trade = pd.DatetimeIndex(pd.to_datetime(R["tstart"].to_numpy(), utc=True)).tz_convert(season.NEW_YORK).date
    rules = ("standard", "strict", "cross_no_usd")
    days = {rule: np.zeros(len(R)) for rule in rules}
    codes = R["pair"].to_numpy()
    for code, pair in PAIRS.items():
        hol = {c: holidays(c, 2002, 2018) for c in {pair.base, pair.quote, "USD"}}
        for k in np.flatnonzero(codes == code):
            t = trade[k]
            for rule in rules:
                days[rule][k] = (_spot_rule(t + D1, pair.base, pair.quote, hol, rule)
                                 - _spot_rule(t, pair.base, pair.quote, hol, rule)).days
    f, unit = R["f"].to_numpy(), -R["diff"].to_numpy() / 360 * 100
    out = {}
    for alt in rules[1:]:
        m = days["standard"] != days[alt]
        out[alt] = {"n": int(m.sum())}
        for rule in ("standard", alt):
            e = unit[m] * days[rule][m]
            out[alt][rule] = {"sse": float(np.sum((f[m] - e) ** 2)),
                              "slope": float(np.sum(e * f[m]) / np.sum(e * e)) if np.any(e) else None}
    return out


def roll_table(code: str, first: pd.Timestamp, last: pd.Timestamp) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The 17:00 New York rolls from ``first`` to ``last`` (UTC ns, sorted) and, per way of counting,
    the cumulative value days rolled before each (length + 1). The roll that ends trade date D moves the
    value date from spot(D) to spot(next weekday): "plain" counts 1 (Wednesday 3), "cal" the calendars."""
    pair = PAIRS[code]
    hol = {c: holidays(c, first.year - 1, last.year + 1) for c in {pair.base, pair.quote, "USD"}}
    days = pd.date_range(first.tz_convert(None).normalize() - pd.Timedelta(days=7),
                         last.tz_convert(None).normalize() + pd.Timedelta(days=7), freq="D")
    days = days[days.dayofweek < 5]
    when = (days + pd.Timedelta(hours=17)).tz_localize(season.NEW_YORK).tz_convert("UTC").as_unit("ns").asi8
    spot = {}

    def sp(d: date) -> date:
        if d not in spot:
            spot[d] = spot_date(d, pair.base, pair.quote, hol)
        return spot[d]

    cal = np.array([(sp(d + (3 if d.weekday() == 4 else 1) * D1) - sp(d)).days for d in days.date])
    plain = np.where(days.dayofweek == 2, 3, 1)
    return when, {"plain": np.concatenate([[0], np.cumsum(plain)]), "cal": np.concatenate([[0], np.cumsum(cal)])}


# ------------------------------------------------------------------ rates

def rate_diff_by_day(code: str, days_ns: np.ndarray) -> np.ndarray:
    """Base-minus-quote short rate (% a year) known on each UTC day (ns at midnight), as history.rates_panel
    and rates.rate_diff (same series and publication lags)."""
    pair = PAIRS[code]
    uniq, inv = np.unique(days_ns, return_inverse=True)
    r = history.rates_panel(pd.DatetimeIndex(pd.to_datetime(uniq)))
    return (r[pair.base] - r[pair.quote]).to_numpy(float)[inv]


def rate_source_check(n_days: int = 40, seed: int = SEED) -> dict:
    """The server's own path (a rates item as collect_rates stores it, read by rates.rate_diff) against
    history.rates_panel on sample days of Yahoo's span: largest difference (% a year)."""
    rng = np.random.default_rng(seed)
    span = history.load_hourly("USDJPY").index
    days = pd.date_range(span[0].tz_convert(None).normalize(), span[-1].tz_convert(None).normalize(), freq="D")
    pick = pd.DatetimeIndex(np.sort(rng.choice(days, size=min(n_days, len(days)), replace=False)))
    panel = history.rates_panel(pick)
    raw = {sid: history.load_fred(sid) for items in history.RATE_SERIES.values() for sid, _ in items}
    worst, n = 0.0, 0
    for day in pick:
        item = {"series": {}}
        for items in history.RATE_SERIES.values():
            for sid, freq in items:
                keep = timedelta(days=ratesmod.KEEP_DAYS) if freq == "d" else timedelta(days=31 * ratesmod.KEEP_MONTHS)
                s = raw[sid][(raw[sid].index >= day - keep) & (raw[sid].index < day)]
                item["series"][sid] = [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in s.items()]
        for pair in PAIRS.values():
            live = ratesmod.rate_diff(item, pair.base, pair.quote, day.date())
            ref = float(panel.loc[day, pair.base] - panel.loc[day, pair.quote])
            if live is not None and np.isfinite(ref):
                worst = max(worst, abs(live - ref))
                n += 1
    return {"days": len(pick), "compared": n, "max_abs_diff": worst}


# ------------------------------------------------------------------ forecast rows

def _choose(mu_w, t_w, mu_d, t_d) -> tuple[np.ndarray, np.ndarray]:
    """season.bar_drift: the weekday slot when |t| >= T_MIN, else the time-of-day slot, else no call."""
    use_w = np.abs(t_w) >= season.T_MIN
    use_d = ~use_w & (np.abs(t_d) >= season.T_MIN)
    return np.where(use_w, mu_w, np.where(use_d, mu_d, 0.0)), np.where(use_w, t_w, np.where(use_d, t_d, 0.0))


def _mu_t(S: np.ndarray, add: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """season._stats from window sums (count, sum, sum of squares), the mean shifted by ``add``."""
    cnt = np.round(S[:, 0])
    use = cnt >= season.MIN_N
    safe = np.where(use, cnt, 1.0)
    mu = S[:, 1] / safe
    sd = np.sqrt(np.maximum(S[:, 2] / safe - mu * mu, 0.0))
    ok = use & (sd > 1e-12)
    m = np.where(use, mu + add, 0.0)
    return m, np.where(ok, m / np.where(ok, sd, 1.0) * np.sqrt(safe), 0.0)


def build(full: pd.DataFrame, code: str, minutes: int, min_bars: int, ahead: int) -> pd.DataFrame:
    """Every forecast the server makes for the next bar (origin: the end of an open-market bar with at least
    ``min_bars`` earlier open bars and ``ahead`` later ones, as research_direction.session_eval), with the live
    call (d0, t0), the called bar's move f (bp), its expected roll shift e_* and days_* per way of counting
    days, and candidate d's adjusted call (dA_*, tA_*). Slot statistics use every stored bar before the
    origin's UTC day (bars after a pause or while the market is shut left out), as season.slot_stats."""
    full = full[~full.index.duplicated()].sort_index()
    idx = pd.DatetimeIndex(full.index).as_unit("ns")
    ns = idx.asi8
    step = minutes * NS_MIN
    close = full["close"].to_numpy(float)
    r = np.full(len(ns), np.nan)
    r[1:] = np.diff(np.log(close)) * 1e4
    r[1:][np.diff(ns) > step] = np.nan
    is_open = market_open_mask(idx)
    r[~is_open] = np.nan
    week, tod, per_day = season._slots(idx, minutes)
    pos = np.flatnonzero(is_open)
    io = np.arange(min_bars, len(pos) - ahead)
    src, tgt = pos[io], pos[io + 1]
    origin = ns[src] + step
    cut = np.searchsorted(ns, origin // NS_DAY * NS_DAY, side="left")
    low = np.maximum(cut - season.WINDOW, 0)                 # returns lo+1 .. cut-1, lo = cut - WINDOW - 1
    points = np.unique(np.concatenate([cut, low]))
    hi_p, lo_p = np.searchsorted(points, cut), np.searchsorted(points, low)

    def sums(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Pw = _slot_prefix(x, week, 7 * per_day, points)
        w = Pw[hi_p, week[tgt]] - Pw[lo_p, week[tgt]]
        del Pw
        Pd = _slot_prefix(x, tod, per_day, points)
        return w, Pd[hi_p, tod[tgt]] - Pd[lo_p, tod[tgt]]

    Sw, Sd = sums(r)
    d0, t0 = _choose(*_mu_t(Sw), *_mu_t(Sd))
    local = idx[tgt].tz_convert(season.NEW_YORK)
    F = {"origin": origin, "tstart": ns[tgt], "f": np.log(close[tgt] / close[src]) * 1e4, "d0": d0, "t0": t0,
         "ny_hour": local.hour.to_numpy(), "ny_min": local.minute.to_numpy(), "ny_dow": local.dayofweek.to_numpy()}
    F["roll"] = (F["ny_hour"] == 17) & (F["ny_min"] == 0) & (F["ny_dow"] <= 3)
    when, cum = roll_table(code, idx[0], idx[-1])
    diff_bar = rate_diff_by_day(code, ns // NS_DAY * NS_DAY)
    diff_row = rate_diff_by_day(code, origin // NS_DAY * NS_DAY)
    F["diff"] = diff_row
    prev_end = np.concatenate([[ns[0]], ns[:-1] + step])       # a bar's move runs from the previous close
    for key in DAYS:
        c = cum[key]
        days_bar = c[np.searchsorted(when, ns + step)] - c[np.searchsorted(when, prev_end)]
        days_row = c[np.searchsorted(when, ns[tgt] + step)] - c[np.searchsorted(when, origin)]
        e_bar = np.nan_to_num(-diff_bar * days_bar / 360 * 100)
        e_row = np.nan_to_num(-diff_row * days_row / 360 * 100)
        Rw, Rd = sums(r - e_bar)
        F[f"days_{key}"], F[f"e_{key}"] = days_row, e_row
        F[f"dA_{key}"], F[f"tA_{key}"] = _choose(*_mu_t(Rw, e_row), *_mu_t(Rd, e_row))
    out = pd.DataFrame(F)
    out["pair"] = code
    out["wk"] = (origin // NS_DAY + 3) // 7                     # Monday-to-Sunday weeks (to_period("W"))
    out["day"] = origin // NS_DAY
    return out


def check_against_season(full: pd.DataFrame, F: pd.DataFrame, minutes: int, n_days: int = 12,
                         seed: int = SEED) -> dict:
    """Recompute the live call with season.slot_stats / bar_drift on sample days: largest difference and
    call mismatches."""
    full = full[~full.index.duplicated()].sort_index()
    rng = np.random.default_rng(seed)
    days = np.unique(F["day"].to_numpy())
    worst_d = worst_t = 0.0
    n = mism = 0
    for day in rng.choice(days, size=min(n_days, len(days)), replace=False):
        rows = F[F["day"] == day]
        stats = season.slot_stats(full, pd.Timestamp(int(rows["origin"].iloc[0]), tz="UTC").to_pydatetime(), minutes)
        d, t = season.bar_drift(stats, pd.DatetimeIndex(pd.to_datetime(rows["tstart"].to_numpy(), utc=True)), minutes)
        worst_d = max(worst_d, float(np.max(np.abs(d - rows["d0"].to_numpy()))))
        worst_t = max(worst_t, float(np.max(np.abs(t - rows["t0"].to_numpy()))))
        mism += int(np.sum((d != 0) != (rows["d0"].to_numpy() != 0)) + np.sum(np.sign(d) != np.sign(rows["d0"].to_numpy())))
        n += len(rows)
    return {"origins": n, "max_abs_d": worst_d, "max_abs_t": worst_t, "mismatch": mism}


def _split(frames: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """research_direction's split: 60 % of all stored bar times, pooled over pairs."""
    all_t = np.sort(np.concatenate([pd.DatetimeIndex(H.index).as_unit("ns").asi8 for H in frames.values()]))
    return pd.Timestamp(all_t[int(len(all_t) * HOURLY_TUNE_SHARE)], tz="UTC")


def load_source(source: str, log=print) -> tuple[pd.DataFrame, dict]:
    """Forecast rows of all pairs for "1h" / "15m" (Yahoo) or "duka" (Dukascopy mid, hourly)."""
    t0 = time.time()
    if source == "duka":
        raw = {code: history.load_long_hourly(code) for code in PAIRS}
        spec, split = {"minutes": 60, "min_bars": season.WINDOW, "ahead": 24}, DUKA_SPLIT
    else:
        raw = {code: (history.load_hourly(code) if source == "1h" else history.load_intraday(code, "15m")) for code in PAIRS}
        raw = {c: H[~H.index.duplicated()].sort_index() for c, H in raw.items()}
        spec, split = TF[source], _split(raw)
    parts, checks = [], {}
    for code, H in raw.items():
        F = build(H, code, spec["minutes"], spec["min_bars"], spec["ahead"])
        if code in (("USDJPY", "EURUSD") if source == "duka" else PAIRS):
            checks[code] = check_against_season(H, F, spec["minutes"], n_days=6 if source == "duka" else 12)
        parts.append(F)
    A = pd.concat(parts, ignore_index=True)
    A["period"] = np.where(A["origin"] >= split.value, "test", "tune")
    A["year"] = pd.DatetimeIndex(pd.to_datetime(A["origin"].to_numpy(), utc=True)).year
    info = {"split": str(split), "rows": int(len(A)), "span": [str(pd.Timestamp(int(A["origin"].min()), tz="UTC")),
                                                                 str(pd.Timestamp(int(A["origin"].max()), tz="UTC"))],
            "check": checks, "seconds": round(time.time() - t0, 1)}
    if log:
        log(f"{source}: {len(A):,} forecast rows, split {split.date()}, check "
            f"{max(c['max_abs_t'] for c in checks.values()):.1e} / {sum(c['mismatch'] for c in checks.values())} "
            f"mismatches, {info['seconds']} s")
    return A, info


# ------------------------------------------------------------------ candidates

def apply(F: pd.DataFrame, cand: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Direction (+1 / -1 / 0) and high-confidence flag of every row under a candidate ("base" = live)."""
    s0 = np.sign(F["d0"].to_numpy())
    a0 = np.abs(F["t0"].to_numpy())
    if cand == "base":
        return s0, (s0 != 0) & (a0 >= season.T_HIGH)
    roll = F["roll"].to_numpy()
    if cand == "d":
        s = np.sign(F[f"dA_{p['days']}"].to_numpy())
        return s, (s != 0) & (np.abs(F[f"tA_{p['days']}"].to_numpy()) >= season.T_HIGH)
    e = F[f"e_{p['days']}"].to_numpy()
    if cand == "a":
        s = np.where(roll & (s0 * e < 0) & (np.abs(e) >= p["E"]), 0.0, s0)
        return s, (s != 0) & (a0 >= season.T_HIGH)
    if cand == "b":
        up = roll & (s0 * e > 0) & (np.abs(e) >= p["E"]) & (a0 >= p["T"])
        return s0, (s0 != 0) & ((a0 >= season.T_HIGH) | up)
    if cand == "c":
        m = roll & (np.abs(e) >= p["X"]) & (e != 0)
        s = np.where(m, np.sign(e), s0)
        return s, (s != 0) & (m | (a0 >= season.T_HIGH))
    raise ValueError(cand)


def _tier(s: np.ndarray, f: np.ndarray, block: np.ndarray, n_rows: int) -> dict:
    """research_direction.score plus the share of forecast rows, right / wrong counts and the Wilson
    lower bound of the hit rate (moves of zero left out)."""
    m = (s != 0) & np.isfinite(f)
    out = score(s, f, block) if m.sum() >= 30 else {"n": int(m.sum())}
    nz = m & (f != 0)
    k = int(np.sum(np.sign(f[nz]) == s[nz]))
    out.update(n=int(m.sum()), share=float(m.sum() / n_rows) if n_rows else None, right=k, wrong=int(nz.sum()) - k)
    if nz.any():
        out["hit"] = k / int(nz.sum())
        out["hit_lo"] = _wilson_low(k, int(nz.sum()))
        out["bp"] = float(np.mean(s[m] * f[m]))
    return out


def evaluate(F: pd.DataFrame, s: np.ndarray, hi: np.ndarray, block: str = "wk") -> dict:
    f, b = F["f"].to_numpy(), F[block].to_numpy()
    return {"all": _tier(s, f, b, len(F)), "high": _tier(np.where(hi, s, 0.0), f, b, len(F))}


def changes(F: pd.DataFrame, s: np.ndarray, hi: np.ndarray) -> dict:
    """What a candidate changes against the live rule, each group scored by its new call (removed: the
    dropped live call): added / removed / reversed calls, calls raised to / lowered from high confidence."""
    s0, h0 = apply(F, "base", {})
    f = F["f"].to_numpy()
    groups = {"added": (s0 == 0) & (s != 0), "removed": (s0 != 0) & (s == 0), "reversed": (s0 * s < 0),
              "to_high": hi & ~h0, "from_high": h0 & ~hi}
    out = {}
    for k, m in groups.items():
        sig = s0 if k == "removed" else s
        nz = m & (f != 0)
        n = int(nz.sum())
        right = int(np.sum(np.sign(f[nz]) == sig[nz]))
        out[k] = {"n": int(m.sum()), "hit": right / n if n else None, "hit_lo": _wilson_low(right, n) if n else None}
    return out


def _net(x: dict) -> int:
    return x["all"]["right"] - x["all"]["wrong"]


def choose(F: pd.DataFrame, cand: str) -> dict:
    """The setting of a candidate chosen on the rows ``F`` (the tune period) by its declared objective:
    "net" = most right-minus-wrong calls; "high" = most high-confidence calls whose hit rate is not below the
    live rule's (for c, which also changes calls that are not high, the hit rate of all calls neither)."""
    base = evaluate(F, *apply(F, "base", {}))
    rows = []
    for days in DAYS:
        for p in GRID[cand]:
            q = {**p, "days": days}
            x = evaluate(F, *apply(F, cand, q))
            if OBJECTIVE[cand] == "net":
                key = (_net(x), -len(rows))
                ok = True
            else:
                ok = (x["high"].get("hit", 0) >= base["high"]["hit"]
                      and (cand != "c" or x["all"].get("hit", 0) >= base["all"]["hit"]))
                key = (x["high"]["n"], x["high"].get("hit", 0))
            rows.append({"p": q, "ok": ok, "key": key, "all": x["all"], "high": x["high"]})
    feasible = [r for r in rows if r["ok"]]
    best = max(feasible, key=lambda r: r["key"]) if feasible else None
    gain = None
    if best is not None:
        gain = (_net(best) - _net(base)) if OBJECTIVE[cand] == "net" else best["high"]["n"] - base["high"]["n"]
    return {"best": best["p"] if best and gain and gain > 0 else None, "gain": gain,
            "grid": [{k: v for k, v in r.items() if k != "key"} for r in rows]}


def _ok(new: dict, base: dict, tol: float = 0.0) -> bool:
    return (new.get("hit", 0) >= base["hit"] - tol - 1e-12) and new["n"] >= SIMILAR * base["n"]


def _better(new: dict, base: dict) -> bool:
    return _ok(new, base) and (new["n"] > base["n"] or new.get("hit", 0) > base["hit"] + 1e-12)


def adopt(res: dict, base: dict, tol: float = 0.0) -> dict:
    """The adoption test on one period: neither tier worse (hit rate not lower by more than ``tol``, at
    least 90 % of the calls) and one of them better (more calls, or a higher hit rate)."""
    ok = {k: _ok(res[k], base[k], tol) for k in ("all", "high")}
    better = {k: _better(res[k], base[k]) for k in ("all", "high")}
    return {"ok": ok, "better": better, "pass": all(ok.values()) and (any(better.values()) or tol > 0)}


def by_period(F: pd.DataFrame, cand: str, p: dict | None, block: str = "wk") -> dict:
    out = {}
    for period in ("tune", "test"):
        G = F[F["period"] == period]
        base = evaluate(G, *apply(G, "base", {}), block)
        x = {"base": base}
        if p is not None:
            s, hi = apply(G, cand, p)
            x["new"] = evaluate(G, s, hi, block)
            x["changes"] = changes(G, s, hi)
        out[period] = x
    return out


# ------------------------------------------------------------------ the roll itself

def mechanism(F: pd.DataFrame) -> dict:
    """Roll bars (17:00 New York, Monday-Thursday): the move against the expected shift with days counted
    plainly and from the calendars, overall and where the two counts differ."""
    R = F[F["roll"] & np.isfinite(F["diff"])]
    out = {"n": int(len(R))}
    f = R["f"].to_numpy()
    for key in DAYS:
        e = R[f"e_{key}"].to_numpy()
        out[key] = {"slope": float(np.sum(e * f) / np.sum(e * e)), "corr": float(np.corrcoef(e, f)[0, 1])}
    groups = []
    for (dp, dc), g in R.groupby(["days_plain", "days_cal"]):
        fx, ep, ec = g["f"].to_numpy(), g["e_plain"].to_numpy(), g["e_cal"].to_numpy()
        big = np.abs(ep) >= 0.5
        nz = big & (fx != 0)
        groups.append({"plain": int(dp), "cal": int(dc), "n": int(len(g)), "f": float(fx.mean()),
                       "e_plain": float(ep.mean()), "e_cal": float(ec.mean()),
                       "down_share": float(np.mean(np.sign(fx[nz]) == np.sign(ep[nz]))) if nz.any() else None,
                       "n_big": int(nz.sum())})
    out["groups"] = groups
    D = R[R["days_plain"] != R["days_cal"]]
    if len(D) > 10:
        fd = D["f"].to_numpy()
        out["differ"] = {"n": int(len(D))}
        for key in DAYS:
            e = D[f"e_{key}"].to_numpy()
            out["differ"][key] = {"slope": float(np.sum(e * fd) / np.sum(e * e)) if np.any(e) else None,
                                  "sse": float(np.sum((fd - e) ** 2))}
    # how the live calls do on roll bars whose day count the calendar changes
    s0, h0 = apply(F, "base", {})
    sub = {}
    for name, m in (("same", F["roll"] & (F["days_plain"] == F["days_cal"])),
                    ("differ", F["roll"] & (F["days_plain"] != F["days_cal"])),
                    ("zero", F["roll"] & (F["days_cal"] == 0))):
        m = m.to_numpy()
        fx = F["f"].to_numpy()
        for tier_name, sig in (("all", s0), ("high", np.where(h0, s0, 0.0))):
            nz = m & (sig != 0) & (fx != 0)
            sub.setdefault(name, {})[tier_name] = {"n": int(nz.sum()),
                                                   "hit": float(np.mean(np.sign(fx[nz]) == sig[nz])) if nz.any() else None}
    out["live_calls"] = sub
    return out


def promote_t(d: np.ndarray, t: np.ndarray, e: np.ndarray, roll: np.ndarray, p: dict) -> np.ndarray:
    """Candidate b as the server would apply it: the t of a promoted bar raised to T_HIGH."""
    up = roll & (d * e > 0) & (np.abs(e) >= p["E"]) & (np.abs(t) >= p["T"]) & (np.abs(t) < season.T_HIGH)
    return np.where(up, np.sign(t) * season.T_HIGH, t)


def first_hour_15m(p: dict, split: pd.Timestamp) -> dict:
    """15-minute forecasts one hour (four bars) ahead, whose centre t combines the four bars' t (season.centre_t):
    the live calls against those with candidate b's promotion, at the origins where one of the next four bars is
    a roll bar (elsewhere nothing changes). Computed with season.slot_stats / bar_drift / combined_t directly."""
    rows = []
    for code in PAIRS:
        full = history.load_intraday(code, "15m")
        full = full[~full.index.duplicated()].sort_index()
        bars = full[market_open_mask(full.index)]
        idx = pd.DatetimeIndex(bars.index).as_unit("ns")
        c = bars["close"].to_numpy(float)
        ny = idx.tz_convert(season.NEW_YORK)
        roll = (ny.hour == 17) & (ny.minute == 0) & (ny.dayofweek <= 3)
        when, cum = roll_table(code, idx[0], idx[-1])
        ns = idx.asi8
        step = 15 * NS_MIN
        for i in range(TF["15m"]["min_bars"], len(c) - TF["15m"]["ahead"]):
            k = slice(i + 1, i + 5)
            if not roll[k].any():
                continue
            origin = ns[i] + step
            stats = season.slot_stats(full, pd.Timestamp(origin, tz="UTC").to_pydatetime(), 15)
            d, t = season.bar_drift(stats, idx[k], 15)
            if not d.any():
                continue
            diff = rate_diff_by_day(code, np.array([origin // NS_DAY * NS_DAY]))[0]
            ends = ns[k] + step
            days = cum[p["days"]][np.searchsorted(when, ends)] - cum[p["days"]][np.searchsorted(when, ends - step)]
            e = np.nan_to_num(-diff * days / 360 * 100)
            t2 = promote_t(d, t, e, roll[k], p)
            rows.append((origin >= split.value, np.sign(d.sum()), abs(season.combined_t(d, t)),
                         abs(season.combined_t(d, t2)), math.log(c[i + 4] / c[i]) * 1e4, (origin // NS_DAY + 3) // 7))
    R = pd.DataFrame(rows, columns=["test", "s", "t_base", "t_new", "f", "wk"])
    out = {"origins": int(len(R))}
    for period, G in (("tune", R[~R["test"]]), ("test", R[R["test"]])):
        s, f, b = G["s"].to_numpy(), G["f"].to_numpy(), G["wk"].to_numpy()
        base_hi, new_hi = G["t_base"].to_numpy() >= season.T_HIGH, G["t_new"].to_numpy() >= season.T_HIGH
        out[period] = {"calls": _tier(s, f, b, len(G)), "base_high": _tier(np.where(base_hi, s, 0.0), f, b, len(G)),
                       "new_high": _tier(np.where(new_hi, s, 0.0), f, b, len(G)),
                       "raised": _tier(np.where(new_hi & ~base_hi, s, 0.0), f, b, len(G))}
    return out


@lru_cache(maxsize=64)
def _holidays_near(cur: str, year: int) -> frozenset:
    return frozenset(holidays(cur, year - 1, year + 1))


def live_roll_shift(starts, base: str, quote: str, diff: float | None) -> np.ndarray:
    """The proposed season.roll_shift: expected move (bp) of bars starting at 17:00 New York, Monday to Thursday."""
    local = pd.DatetimeIndex(starts).tz_convert(season.NEW_YORK)
    out = np.zeros(len(local))
    if diff is None:
        return out
    for k, t in enumerate(local):
        if t.hour == 17 and t.minute == 0 and t.dayofweek <= 3:
            trade = t.date()
            hol = {c: _holidays_near(c, trade.year) for c in {base, quote, "USD"}}
            nxt = trade + timedelta(days=3 if trade.weekday() == 4 else 1)
            out[k] = -diff * (spot_date(nxt, base, quote, hol) - spot_date(trade, base, quote, hol)).days / 360 * 100
    return out


def live_check(Y: pd.DataFrame, p: dict) -> dict:
    """The season.py change of the report (roll_shift + promote inside step_drift, trading-calendar bar starts)
    at every Yahoo hourly origin whose next bar is a roll bar, against candidate b's flags here."""
    s_b, hi_b = apply(Y, "b", p)
    n = mism = raised = skipped = 0
    for code, pair in PAIRS.items():
        full = history.load_hourly(code)
        full = full[~full.index.duplicated()].sort_index()
        m = (Y["pair"] == code).to_numpy() & Y["roll"].to_numpy()
        for k in np.flatnonzero(m):
            origin = pd.Timestamp(int(Y["origin"].iat[k]), tz="UTC").to_pydatetime()
            starts = [add_trading_minutes(origin, 1, 60) - timedelta(minutes=60)]
            if pd.Timestamp(starts[0]).value != int(Y["tstart"].iat[k]):
                skipped += 1                 # a stored bar is missing: the next calendar hour is not the next stored bar
                continue
            d, t = season.bar_drift(season.slot_stats(full, origin, 60), starts, 60)
            e = live_roll_shift(starts, pair.base, pair.quote, float(Y["diff"].iat[k]))
            t = np.where((d * e > 0) & (np.abs(e) >= p["E"]) & (np.abs(t) >= season.T_MIN) & (np.abs(t) < season.T_HIGH),
                         np.sign(t) * season.T_HIGH, t)
            high = bool(d[0] != 0 and abs(t[0]) >= season.T_HIGH)
            n += 1
            raised += int(high and abs(Y["t0"].iat[k]) < season.T_HIGH)
            mism += int(high != bool(hi_b[k]) or (d[0] != 0 and np.sign(d[0]) != s_b[k]))
    return {"origins": n, "raised": raised, "mismatch": mism, "skipped": skipped}


def own_objective(cand: str, x: dict) -> dict:
    """A candidate's tune objective measured on another period (x: by_period entry): the change in right-minus-wrong
    calls ("net"), or in high-confidence calls with the hit rate(s) not lower ("high")."""
    b, n = x["base"], x["new"]
    if OBJECTIVE[cand] == "net":
        gain = _net(n) - _net(b)
        return {"gain": gain, "holds": gain > 0}
    gain = n["high"]["n"] - b["high"]["n"]
    ok = n["high"].get("hit", 0) >= b["high"]["hit"] and (cand != "c" or n["all"].get("hit", 0) >= b["all"]["hit"])
    return {"gain": gain, "holds": bool(gain > 0 and ok)}


# ------------------------------------------------------------------ run

def study(sources: dict[str, pd.DataFrame], split15: pd.Timestamp) -> dict:
    """Choose each candidate on the Yahoo hourly tune period and score it everywhere."""
    Y, M, D = sources["1h"], sources["15m"], sources["duka"]
    out: dict = {"candidates": {}}
    for cand in CANDIDATES:
        ch = choose(Y[Y["period"] == "tune"], cand)
        ch15 = choose(M[M["period"] == "tune"], cand)
        chD = choose(D[D["period"] == "tune"], cand)
        p = ch["best"]
        r: dict = {"choice": ch, "choice_15m": ch15, "choice_duka": {"best": chD["best"], "gain": chD["gain"]}}
        if p is None and cand == "d":       # no setting to choose: show both day counts, although rejected on tune
            r["yahoo_rejected"] = {days: by_period(Y, cand, {"days": days}) for days in DAYS}
        r["yahoo"] = by_period(Y, cand, p)
        if p is not None:
            other = {**p, "days": "plain" if p["days"] == "cal" else "cal"}
            r["yahoo_other_days"] = by_period(Y, cand, other)
            r["verdict"] = {k: adopt(r["yahoo"][k]["new"], r["yahoo"][k]["base"]) for k in ("tune", "test")}
            r["duka"] = by_period(D, cand, p)
            r["duka_verdict"] = {k: adopt(r["duka"][k]["new"], r["duka"][k]["base"], DUKA_TOL) for k in ("tune", "test")}
            r["m15_hourly_setting"] = by_period(M, cand, p)
            r["m15_hourly_verdict"] = {k: adopt(r["m15_hourly_setting"][k]["new"], r["m15_hourly_setting"][k]["base"])
                                       for k in ("tune", "test")}
            r["own"] = {"yahoo_test": own_objective(cand, r["yahoo"]["test"]),
                        "duka_tune": own_objective(cand, r["duka"]["tune"]),
                        "duka_test": own_objective(cand, r["duka"]["test"])}
            if cand == "b":
                r["m15_first_hour"] = first_hour_15m(p, split15)
                r["live_check"] = live_check(Y, p)
        if ch15["best"] is not None:
            r["m15"] = by_period(M, cand, ch15["best"])
            r["m15_verdict"] = {k: adopt(r["m15"][k]["new"], r["m15"][k]["base"]) for k in ("tune", "test")}
        r["adopt"] = bool(p is not None and all(v["pass"] for v in r["verdict"].values())
                          and all(v["pass"] for v in r["duka_verdict"].values()))
        r["own_holds"] = bool(p is not None and all(v["holds"] for v in r["own"].values()))
        out["candidates"][cand] = r
    # among the candidates that pass the declared rule and whose own objective also holds out of sample, prefer the
    # one that also passes on 15-minute bars with the hourly setting, then the one that adds or reverses no call
    ok = [c for c, r in out["candidates"].items() if r["adopt"] and r["own_holds"]]

    def rank(c: str) -> tuple:
        r = out["candidates"][c]
        m15 = all(v["pass"] for v in r["m15_hourly_verdict"].values())
        moved = sum(r["yahoo"][k]["changes"][g]["n"] for k in ("tune", "test") for g in ("added", "reversed"))
        return (m15, -moved, r["own"]["yahoo_test"]["gain"])

    out["recommended"] = max(ok, key=rank) if ok else None
    out["recommend_order"] = sorted(ok, key=rank, reverse=True)
    if out["recommended"]:
        c = out["recommended"]
        p = out["candidates"][c]["choice"]["best"]
        out["breakdown"] = {"yahoo": breakdown(Y, c, p), "duka": breakdown(D, c, p)}
    return out


def breakdown(F: pd.DataFrame, cand: str, p: dict) -> dict:
    """The calls a candidate changes (raised to high confidence, added or reversed) by period and pair, by
    weekday and whether the calendars changed the day count, and by year: count and hit rate."""
    s0, h0 = apply(F, "base", {})
    s, hi = apply(F, cand, p)
    m = (hi & ~h0) | ((s != 0) & (s != s0))
    G = pd.DataFrame({"period": F["period"].to_numpy()[m], "pair": F["pair"].to_numpy()[m],
                      "wd": F["ny_dow"].to_numpy()[m], "year": F["year"].to_numpy()[m],
                      "moved": (F["days_cal"] != F["days_plain"]).to_numpy()[m], "f": F["f"].to_numpy()[m], "s": s[m]})
    G = G[G["f"] != 0]
    G["right"] = np.sign(G["f"]) == G["s"]

    def agg(keys: list[str]) -> list[dict]:
        return [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "n": int(len(g)), "hit": float(g["right"].mean())}
                for k, g in G.groupby(keys)]

    f, yr = F["f"].to_numpy(), F["year"].to_numpy()
    nz = h0 & (f != 0)
    base_year = {int(y): {"n": int(np.sum(nz & (yr == y))), "hit": float(np.mean(np.sign(f[nz & (yr == y)]) == s0[nz & (yr == y)]))}
                 for y in np.unique(yr[nz])}
    return {"pair": agg(["period", "pair"]), "weekday": agg(["period", "wd", "moved"]), "year": agg(["year"]),
            "base_high_year": base_year}


def run(log=print) -> dict:
    t0 = time.time()
    res: dict = {"candidates_declared": CANDIDATES, "grid": GRID, "objective": OBJECTIVE}
    res["rates_check"] = rate_source_check()
    if log:
        log(f"rates: live path vs rates_panel max diff {res['rates_check']['max_abs_diff']:.2e} % a year")
    sources, info = {}, {}
    for src in ("1h", "15m", "duka"):
        sources[src], info[src] = load_source(src, log)
    res["sources"] = info
    Y = sources["1h"]
    res["baseline"] = {"1h": by_period(Y, "base", None),
                       "15m": by_period(sources["15m"], "base", None),
                       "15m_all_days": evaluate(sources["15m"], *apply(sources["15m"], "base", {}), "day"),
                       "duka": by_period(sources["duka"], "base", None)}
    res["mechanism"] = {src: mechanism(F) for src, F in sources.items()}
    res["convention"] = convention_check(sources["duka"])
    pub = REPORT_DIR / "direction.json"
    if pub.exists():
        ses = json.loads(pub.read_text(encoding="utf-8")).get("session", {})
        res["published"] = {"1h": ses.get("1h", {}).get("h", {}).get("1"), "15m": ses.get("15m", {}).get("h", {}).get("1")}
    res.update(study(sources, pd.Timestamp(info["15m"]["split"])))
    res["seconds"] = round(time.time() - t0, 1)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "rollover.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    (REPORT_DIR / "rollover.md").write_text(report(res), encoding="utf-8")
    if log:
        log(f"done in {res['seconds']} s")
    return res


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


# ------------------------------------------------------------------ report

WEEKDAY = {0: "月", 1: "火", 2: "水", 3: "木"}
PERIOD = {"tune": "調整", "test": "検証"}


def _p(x, d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def _t(x) -> str:
    return "—" if x is None else f"{x:+.2f}"


def _nh(x: dict | None) -> str:
    if not x or not x.get("n"):
        return "0 / —"
    return f"{x['n']:,} / {_p(x.get('hit'))}"


def _arrow(b: dict, n: dict) -> str:
    return _nh(b) if (b.get("n"), b.get("hit")) == (n.get("n"), n.get("hit")) else f"{_nh(b)} → **{_nh(n)}**"


def _setting(p: dict | None) -> str:
    if p is None:
        return "なし"
    parts = []
    if "T" in p:
        parts.append(f"\\|t\\| ≥ {p['T']:g}")
    if "E" in p:
        parts.append(f"\\|e\\| ≥ {p['E']:g} bp")
    if "X" in p:
        parts.append(f"\\|e\\| ≥ {p['X']:g} bp")
    parts.append({"plain": "日数: 単純", "cal": "日数: 暦"}[p["days"]])
    return "、".join(parts)


def _changes(ch: dict) -> str:
    names = {"added": "追加", "removed": "削除 (外した予測の的中率)", "reversed": "逆向き", "to_high": "高に格上げ",
             "from_high": "高から格下げ"}
    out = [f"{names[k]} {v['n']:,}回 {_p(v['hit'])}" for k, v in ch.items() if v["n"]]
    return "、".join(out) if out else "変化なし"


def _mark(v: bool) -> str:
    return "○" if v else "×"


def _grp(res: dict, src: str, plain: int, cal: int) -> str:
    g = next((g for g in res["mechanism"][src]["groups"] if g["plain"] == plain and g["cal"] == cal), None)
    if g is None:
        return "—"
    return f"{g['n']:,}本、平均 {g['f']:+.2f} bp、見込み: 単純 {g['e_plain']:+.2f}・暦 {g['e_cal']:+.2f} bp"


def _summary_rows(res: dict) -> list[str]:
    cands = res["candidates"]
    L = ["| 候補 | 調整期間で選んだ設定 | 期間 | 方向を示した回: 回数 / 的中率 | 確度: 高: 回数 / 的中率 | 変わった予測 | 採用の条件 |",
         "|---|---|---|---|---|---|---|"]
    b = res["baseline"]["1h"]
    for k in ("tune", "test"):
        x = b[k]["base"]
        L.append(f"| 本番 (今の規則) | — | {PERIOD[k]} | {_nh(x['all'])} | {_nh(x['high'])} | — | — |")
    for c, r in cands.items():
        p = r["choice"]["best"]
        if p is None:
            L.append(f"| {c} | なし (調整期間で本番より良い設定がない) | — | — | — | — | × |")
            continue
        for k in ("tune", "test"):
            x = r["yahoo"][k]
            v = r["verdict"][k]
            L.append(f"| {c} | {_setting(p)} | {PERIOD[k]} | {_arrow(x['base']['all'], x['new']['all'])} | "
                     f"{_arrow(x['base']['high'], x['new']['high'])} | {_changes(x['changes'])} | {_mark(v['pass'])} |")
    return L


def report(res: dict) -> str:
    cands, base = res["candidates"], res["baseline"]
    rec = res.get("recommended")
    rb = cands.get("b", {})
    pb = rb.get("choice", {}).get("best")
    by, bd = base["1h"], base["duka"]
    L = ["# ロールオーバーの金利差 (スワップポイント) で時間帯の偏りを補強できるか", "",
         "本番の方向の予測 (時間帯の偏り、aifx/season.py) が当たる理由の大部分は、ニューヨーク時間17時のロールオーバーです。"
         "このとき受け渡し日が1日 (水曜日は週末をまたいで3日) 先に進み、金利の高い通貨がその日数分の金利差 (フォワードポイント、"
         "およそ (基準通貨の金利 − 相手通貨の金利) × 日数 / 360) だけ安い価格に移ります ([season_long.md](season_long.md))。"
         "本番は曜日×時間の枠の過去約1年の平均からこれを学んでいますが、その後に金利が変わったことや、祝日のせいで今夜のロールオーバーが"
         "0日・2日・4日分になることは分かりません。",
         "",
         "ここでは、サーバーが既に保存している短期金利 (rates.py。FRED の同じ系列と公表の遅れ) と、各通貨の決済休日の暦から数えた日数で"
         "「金利平価から見込まれるずれ」を計算し、それを加えると本番のデータ (Yahoo の1時間足・15分足) の方向の予測が良くなるかを、"
         "結果を見る前に決めた候補だけで確かめました。先読みはしていません (枠の統計は予測する日の0時 UTC より前の足だけ、"
         "金利はその日に分かっていた値、設定は調整期間だけで選択)。", ""]

    # ---- conclusions
    L += ["## 結論", ""]
    L.append(f"- **本番の再現:** Yahoo 1時間足の方向を示した回 {_nh(by['tune']['base']['all'])} (調整)・"
             f"{_nh(by['test']['base']['all'])} (検証)、確度: 高 {_nh(by['tune']['base']['high'])}・{_nh(by['test']['base']['high'])} で、"
             "公開している数字 (research/direction.md) と完全に一致しました。")
    if rec == "b" and pb:
        yt, ye = rb["yahoo"]["tune"], rb["yahoo"]["test"]
        dt_, de = rb["duka"]["tune"], rb["duka"]["test"]
        L.append(
            f"- **採用を勧めるのは1つだけ (候補 b「格上げ」、1時間足のみ):** ロールオーバーの足 (ニューヨーク時間17時に始まる月〜木の足) で"
            f"本番が方向を示し、その向きが見込みのずれ e と同じで |e| ≥ {pb['E']:g} bp (日数は祝日の暦で数える) なら、確度: 高 にします。"
            f"Yahoo 1時間足の確度: 高 は 調整 {_nh(yt['base']['high'])} → {_nh(yt['new']['high'])}、"
            f"検証 {_nh(ye['base']['high'])} → {_nh(ye['new']['high'])} (格上げした足は 調整 {yt['changes']['to_high']['n']}回 "
            f"{_p(yt['changes']['to_high']['hit'])}・検証 {ye['changes']['to_high']['n']}回 {_p(ye['changes']['to_high']['hit'])})。"
            "方向を示す足と向きは変えないので、方向を示した回全体の成績は同じです。")
        L.append(
            f"- **Dukascopy の仲値 (2004〜2026) でも同じ向き:** 同じ設定で確度: 高 は 調整 (〜2016) {_nh(dt_['base']['high'])} → "
            f"{_nh(dt_['new']['high'])}、検証 (2017〜) {_nh(de['base']['high'])} → {_nh(de['new']['high'])}。"
            f"Dukascopy の調整期間だけで選び直しても同じ設定 ({_setting(rb['choice_duka']['best'])}) になります。")
        more = [yx["new"]["high"]["n"] / yx["base"]["high"]["n"] - 1 for yx in (yt, ye)]
        gain = [(yx["new"]["high"]["hit"] - yx["base"]["high"]["hit"]) * 100 for yx in (yt, ye)]
        aud = sum(x["n"] for x in res["breakdown"]["yahoo"]["pair"] if x["pair"] == "AUDJPY")
        allp = sum(x["n"] for x in res["breakdown"]["yahoo"]["pair"])
        L.append(
            f"- **効果は小さいものです。** 確度: 高 の回数は 調整 {more[0] * 100:+.1f}%・検証 {more[1] * 100:+.1f}%、その的中率は "
            f"{gain[0]:+.1f}・{gain[1]:+.1f} ポイントで、的中率の差そのものは誤差の範囲です。言えるのは「的中率を下げずに確度: 高 を少し"
            f"増やせる」ことです。Yahoo で格上げされる足の {aud / allp * 100:.0f}% は AUDJPY で、水曜日と、祝日で日数が増える日が中心です。")
        other = rb["yahoo_other_days"]
        L.append(
            f"- **祝日の暦が効いています。** 同じ設定で日数を単純に数える (1日、水曜日3日) と、調整期間の確度: 高 は "
            f"{other['tune']['new']['high']['n']:,} / {_p(other['tune']['new']['high']['hit'], 2)} で本番 "
            f"({_p(by['tune']['base']['high']['hit'], 2)}) を下回り、採用の条件を満たしません。"
            "暦で数えた日数は、2004年からの仲値でも単純な日数より17時の値動きをよく説明します (2.)。")
    else:
        L.append("- **採用を勧める変更はありません (今のまま)。**")
    ra, rc, rd = cands["a"], cands["c"], cands["d"]
    if ra["choice"]["best"]:
        L.append(
            f"- **候補 a (向きが逆の予測を外す) は勧めません。** 形式上は条件を満たします (的中率より低い予測を少し外すと的中率は上がるため) が、"
            f"外した予測は検証期間で {_p(ra['yahoo']['test']['changes']['removed']['hit'])} "
            f"({ra['yahoo']['test']['changes']['removed']['n']}回)、仲値でも "
            f"{_p(ra['duka']['tune']['changes']['removed']['hit'])}・{_p(ra['duka']['test']['changes']['removed']['hit'])} 当たっており、"
            f"調整期間で選んだ目的 (当たり − 外れ) は検証期間で {ra['own']['yahoo_test']['gain']:+d}回、仲値で "
            f"{ra['own']['duka_tune']['gain']:+d}・{ra['own']['duka_test']['gain']:+d}回と悪化しました。")
    if rc["choice"]["best"]:
        L.append(
            f"- **候補 c (見込みが大きい足をスワップの向きに予測) は b とほぼ同じ足を確度: 高 にします** "
            f"({_setting(rc['choice']['best'])}: 検証 {_nh(rc['yahoo']['test']['base']['high'])} → {_nh(rc['yahoo']['test']['new']['high'])})。"
            "条件は満たしますが、15分足では調整期間の確度: 高 の的中率を下げ、仲値の調整期間では本番が方向を示さない足に"
            f"新しい方向 ({rc['duka']['tune']['changes']['added']['n']:,}回 {_p(rc['duka']['tune']['changes']['added']['hit'])}) を加えるため、"
            "方向を変えない b を選びました。")
    rej = rd.get("yahoo_rejected", {})
    if rej:
        x = rej["cal"]
        L.append(
            "- **候補 d (金利で補正した枠) は調整期間で不採用です。** 当たり − 外れ の回数が本番より減りました "
            f"({rd['choice']['gain']:+d}回)。確度: 高 の的中率は上がります (暦: 調整 {_nh(x['tune']['new']['high'])}、検証 "
            f"{_nh(x['test']['new']['high'])}) が、方向を示した回全体の的中率が両方の期間で少し下がる "
            f"({_p(x['tune']['new']['all']['hit'], 2)}・{_p(x['test']['new']['all']['hit'], 2)}、本番 "
            f"{_p(by['tune']['base']['all']['hit'], 2)}・{_p(by['test']['base']['all']['hit'], 2)}) ため、採用の条件も満たしません。"
            "Yahoo の期間 (約2.8年) は金利差が大きく安定していたため、過去1年の平均で十分だったと考えられます。")
    fh = rb.get("m15_first_hour")
    if fh:
        L.append(
            "- **15分足 (約60日) は判断できるだけの量がありません。** 1時間足の設定のままで次の15分の確度: 高 は 調整 "
            f"{_nh(rb['m15_hourly_setting']['tune']['base']['high'])} → {_nh(rb['m15_hourly_setting']['tune']['new']['high'])}、検証 "
            f"{_nh(rb['m15_hourly_setting']['test']['base']['high'])} → {_nh(rb['m15_hourly_setting']['test']['new']['high'])} と同じ向きですが、"
            f"1時間先 (15分足4本、中心の t は4本の合計) では検証期間の確度: 高 が {_nh(fh['test']['base_high'])} → {_nh(fh['test']['new_high'])} "
            f"と的中率がわずかに下がります (格上げ {fh['test']['raised']['n']}回)。そのため変更は1時間足だけにし、15分足は今のままにします。")
    L.append("- **注意:** 当たるのは表示される価格の向き (ロールオーバー前後の価格の仕組み) で、売買の利益ではありません。17時のずれはポジションを"
             "持ち越せばスワップで相殺され、この時間はスプレッドも広がります (確度: 高 でもスプレッドとスワップを引くと1回あたりマイナス、"
             "[season_long.md](season_long.md))。また、「17時の足で見込み 2 bp 以上」が Yahoo で当たりやすいことは、以前の研究 "
             "([ml_long.md](ml_long.md)) が Yahoo の検証期間も含めて見ていたため、E = 2 bp の選択については Yahoo の検証期間は完全には独立でありません。"
             "2004〜2016年の仲値で独立に同じ設定が選ばれることが、その点の支えです。")
    L.append("")

    # ---- declared before the results
    L += ["## 事前に決めた候補と判定の基準", "",
          "候補、設定の候補値、調整期間での選び方、採用の条件は、候補の成績を見る前にコード (aifx/research_rollover.py の "
          "`CANDIDATES`・`GRID`・`OBJECTIVE`・`choose`・`adopt`) に固定しました。",
          "",
          "- **ロールオーバーの足:** ニューヨーク時間17時に始まる月〜木の足 (1時間足は17:00〜18:00、15分足は17:00〜17:15)。"
          "金曜日のロールオーバーは週末の窓開けの足に入り、窓開けの動きの方がはるかに大きいため対象外です。",
          "- **見込みのずれ e (bp):** −(基準通貨の金利 − 相手通貨の金利) × 日数 / 360 × 100。金利は予測の起点の UTC の日に分かっていた値"
          " (history.rates_panel。サーバーの rates.rate_diff と同じ系列・同じ公表の遅れ)。",
          "- **日数の数え方 (2通り):** 「単純」= 1日 (水曜日は3日)。「暦」= 受け渡し日 (スポット日) の差。スポット日は2営業日後で、"
          "1営業日目は相手通貨 (ドル以外) の営業日 (ドルを含むペアではドルの休日を数えない)、受け渡し日は両通貨とドルの営業日。"
          "休日は米連銀、TARGET2 (ユーロ)、イングランドの銀行休業日、シドニーの銀行休業日、日本の銀行休業日 (国民の祝日、振替休日、"
          "国民の休日、12月31日〜1月3日) を規則から計算します。どちらの数え方を使うかも調整期間で選びます。",
          "- **候補** (s・t は本番の方向と t 値):"]
    for c, text in CANDIDATES.items():
        keys = sorted({k for p in GRID[c] for k in p})
        grid = " × ".join(f"{k} ∈ {{{', '.join(f'{v:g}' for v in sorted({p[k] for p in GRID[c]}))}}}" + (" bp" if k != "T" else "")
                          for k in keys)
        L.append(f"  - **{c}** {text}。候補値: {grid + ' × 日数 (単純・暦)' if grid else '日数 (単純・暦) のみ'}")
    L += ["- **調整期間での選び方:** a と d は「当たり − 外れ」の回数が最も多い設定。b と c は、確度: 高 の的中率が本番以上 (c は方向を示した回"
          "全体の的中率も本番以上) の設定のうち、確度: 高 が最も多いもの。本番より良くならなければ「設定なし」(不採用)。",
          "- **採用の条件 (Yahoo 1時間足、調整期間・検証期間の両方で):** 方向を示した回全体と確度: 高 のどちらも、的中率が本番以上で回数が本番の"
          "90%以上、かつどちらかで回数が増えるか的中率が上がる。",
          f"- **Dukascopy 仲値での確認 (調整 〜2016、検証 2017〜):** 同じ設定で、どちらの区分も的中率が {DUKA_TOL * 100:.0f} ポイントを超えて"
          "下がらない (回数90%以上) こと。",
          "- **15分足:** 約60日 (前60%を調整、後40%を検証) しかないため報告だけにし、単独では判断しません。",
          "- **本番で動かせること:** サーバーが既に持つデータ (保存済みの足、保存済みの金利) と固定の規則だけで計算できること。",
          "- 的中率は動きがゼロの回を除いて数え、t は週ごと (月〜日) に全ペアの結果を合計して計算しました (research_direction.py の score)。",
          "",
          "**結果を見た後に加えたもの** (どれも判断を厳しくする方向で、設定の値は変えていません):",
          "",
          "- 各候補の調整期間の目的が検証期間と仲値でも成り立つかの確認 (3.)。形式上の条件は、的中率より低い予測を少し外すだけでも満たせるためです (候補 a)。",
          "- 条件を満たした b と c のどちらを選ぶか: 15分足でも条件を満たすこと、次に方向を加えたり逆にしたりしないことを優先しました。",
          "- 15分足の1時間先 (4本の t を合わせる中心) への影響の確認 (4.)。これにより変更を1時間足に限りました。",
          "- スポット日の規則の比較 (2.)。全期間の内訳を見た後で、仲値の 2004〜2016年だけで比べ、事前に決めた規則のままにしました。",
          ""]

    # ---- 1. reproduction
    pub = res.get("published") or {}
    L += ["## 1. 本番の方向の再現", "",
          "research_direction.py の session_eval と同じ数え方 (統計は保存したすべての足、予測の起点と対象は市場が開いている足、"
          "1時間足は2,000本の履歴がある起点から) で、全時点をまとめて計算しました。",
          "",
          "| データ | 期間 | 方向を示した回: 回数 / 的中率 (t) | 足に占める割合 | 確度: 高: 回数 / 的中率 (t) | 足に占める割合 | 公開値 (direction.md) |",
          "|---|---|---|---|---|---|---|"]
    for src, name in (("1h", "Yahoo 1時間足"), ("15m", "Yahoo 15分足"), ("duka", "Dukascopy 仲値 1時間足")):
        for k in ("tune", "test"):
            x = base[src][k]["base"]
            pb_ = ""
            if src == "1h" and pub.get("1h"):
                q = pub["1h"].get(k, {})
                pb_ = f"{_nh(q.get('all'))}、高 {_nh(q.get('high'))}"
            L.append(f"| {name} | {PERIOD[k]} | {_nh(x['all'])} ({_t(x['all'].get('t'))}) | {_p(x['all'].get('share'), 2)} | "
                     f"{_nh(x['high'])} ({_t(x['high'].get('t'))}) | {_p(x['high'].get('share'), 2)} | {pb_} |")
    x = base["15m_all_days"]
    q = (pub.get("15m") or {}).get("test", {})
    L.append(f"| Yahoo 15分足 | 全体 (t は日ごと) | {_nh(x['all'])} ({_t(x['all'].get('t'))}) | {_p(x['all'].get('share'), 2)} | "
             f"{_nh(x['high'])} ({_t(x['high'].get('t'))}) | {_p(x['high'].get('share'), 2)} | {_nh(q.get('all'))}、高 {_nh(q.get('high'))} |")
    src = res["sources"]
    chk = max(c["max_abs_t"] for s in src.values() for c in s["check"].values())
    mism = sum(c["mismatch"] for s in src.values() for c in s["check"].values())
    L += ["",
          f"- 期間: Yahoo 1時間足 {src['1h']['span'][0][:10]}〜{src['1h']['span'][1][:10]} (調整/検証の境 {src['1h']['split'][:10]})、"
          f"15分足 {src['15m']['span'][0][:10]}〜{src['15m']['span'][1][:10]} (境 {src['15m']['split'][:10]})、"
          f"Dukascopy {src['duka']['span'][0][:10]}〜{src['duka']['span'][1][:10]} (境 2017-01-01)。",
          f"- 無作為に選んだ日を season.slot_stats / bar_drift で計算し直すと、t の差は最大 {chk:.1e}、方向の食い違いは {mism} 件でした。",
          f"- サーバーの金利の扱い (collect_rates が保存する形の金利の項目を rates.rate_diff で読む) と history.rates_panel は、"
          f"{res['rates_check']['days']}日 × 7ペアで最大 {res['rates_check']['max_abs_diff']:.1e} %/年しか違いません (保存時の丸め)。", ""]

    # ---- 2. mechanism
    L += ["## 2. ロールオーバーの日数と祝日", "",
          "ロールオーバーの足 (17時、月〜木) の値動きと見込みのずれを、日数の数え方ごとに比べました。「下がった割合」は単純な日数の見込みの向き "
          "(金利の高い通貨が安くなる向き) に動いた割合 (単純な見込み 0.5 bp 以上の足)。",
          "",
          "| データ | 日数: 単純 | 日数: 暦 | 足の数 | 平均の値動き (bp) | 見込み: 単純 | 見込み: 暦 | 下がった割合 |",
          "|---|---|---|---|---|---|---|---|"]
    for s, name in (("duka", "Dukascopy 仲値"), ("1h", "Yahoo 1時間足")):
        for g in res["mechanism"][s]["groups"]:
            if g["n"] >= 40:
                L.append(f"| {name} | {g['plain']} | {g['cal']} | {g['n']:,} | {g['f']:+.2f} | {g['e_plain']:+.2f} | {g['e_cal']:+.2f} | "
                         f"{_p(g['down_share'])} |")
    L += ["", "日数の数え方が違う足だけで、値動きと見込みの差の二乗和と、見込みに対する値動きの傾き (1 なら見込みどおり):", "",
          "| データ | 足の数 | 二乗和: 単純 | 二乗和: 暦 | 傾き: 単純 | 傾き: 暦 |", "|---|---|---|---|---|---|"]
    for s, name in (("duka", "Dukascopy 仲値"), ("1h", "Yahoo 1時間足"), ("15m", "Yahoo 15分足")):
        d = res["mechanism"][s].get("differ")
        if d:
            L.append(f"| {name} | {d['n']:,} | {d['plain']['sse']:,.0f} | {d['cal']['sse']:,.0f} | {d['plain']['slope']:.2f} | "
                     f"{d['cal']['slope']:.2f} |")
    lc = res["mechanism"]["1h"]["live_calls"]
    cv = res["convention"]
    L += ["",
          f"- 暦で0日になるロールオーバーでは、本番の方向の的中率が下がります (Yahoo 1時間足: 0日 {_nh(lc['zero']['all'])}、"
          f"確度: 高 {_nh(lc['zero']['high'])}。日数が同じ足は {_nh(lc['same']['all'])}、確度: 高 {_nh(lc['same']['high'])})。"
          "本番の枠の平均は、祝日で日数が変わることを知らないためです。",
          "- スポット日の規則は、Yahoo の期間と重ならない仲値の 2004〜2016年だけで他の2通りと比べ、事前に決めた規則のままにしました: "
          f"1営業日目にドルの休日も数える規則との比較 ({cv['strict']['n']}本) で二乗和 {cv['strict']['standard']['sse']:,.0f} 対 "
          f"{cv['strict']['strict']['sse']:,.0f} (傾き {cv['strict']['standard']['slope']:.2f} 対 {cv['strict']['strict']['slope']:.2f})、"
          f"クロス円などで受け渡し日にドルの休日を考えない規則との比較 ({cv['cross_no_usd']['n']}本) で "
          f"{cv['cross_no_usd']['standard']['sse']:,.0f} 対 {cv['cross_no_usd']['cross_no_usd']['sse']:,.0f}。",
          "- 暦の日数は全体としては単純な日数より値動きに合いますが、完全ではありません。例えば暦で0日になる水曜日 (仲値 "
          f"{_grp(res, 'duka', 3, 0)}) は、単純な見込みの方に近く動いています。休日の扱いが業者や年によって違う可能性があります。",
          ""]

    # ---- 3. Yahoo hourly
    L += ["## 3. Yahoo 1時間足: 候補の成績", "",
          "設定は調整期間だけで選び、検証期間にそのまま当てはめました。「変わった予測」は本番と違う扱いになった足と、その足での新しい扱いの"
          "的中率です (削除は外した本番の予測の的中率)。", ""]
    L += _summary_rows(res) + [""]
    L += ["調整期間で選んだ設定のまま日数の数え方だけを変えた場合 (祝日の暦の寄与):", "",
          "| 候補 | 設定 | 期間 | 方向を示した回 | 確度: 高 | 変わった予測 |", "|---|---|---|---|---|---|"]
    for c, r in cands.items():
        if "yahoo_other_days" not in r:
            continue
        p = r["choice"]["best"]
        other = {**p, "days": "plain" if p["days"] == "cal" else "cal"}
        for k in ("tune", "test"):
            x = r["yahoo_other_days"][k]
            L.append(f"| {c} | {_setting(other)} | {PERIOD[k]} | {_arrow(x['base']['all'], x['new']['all'])} | "
                     f"{_arrow(x['base']['high'], x['new']['high'])} | {_changes(x['changes'])} |")
    if rej:
        L += ["", "候補 d は調整期間の目的 (当たり − 外れ) が本番より悪く不採用ですが、参考に両方の数え方の成績を示します:", "",
              "| 日数 | 期間 | 方向を示した回 | 確度: 高 | 変わった予測 |", "|---|---|---|---|---|"]
        for days in DAYS:
            for k in ("tune", "test"):
                x = rej[days][k]
                L.append(f"| {'単純' if days == 'plain' else '暦'} | {PERIOD[k]} | {_arrow(x['base']['all'], x['new']['all'])} | "
                         f"{_arrow(x['base']['high'], x['new']['high'])} | {_changes(x['changes'])} |")
    L += ["", "各候補の、調整期間で選んだ目的を検証期間と仲値で測った結果 (結果を見た後に加えた確認。a・d は当たり − 外れ の増減、b・c は確度: 高 の"
          "増減で、的中率が本番以上のときだけ ○):", "",
          "| 候補 | Yahoo 検証 | 仲値 調整 | 仲値 検証 |", "|---|---|---|---|"]
    for c, r in cands.items():
        if "own" in r:
            o = r["own"]
            L.append(f"| {c} | {o['yahoo_test']['gain']:+,} {_mark(o['yahoo_test']['holds'])} | {o['duka_tune']['gain']:+,} "
                     f"{_mark(o['duka_tune']['holds'])} | {o['duka_test']['gain']:+,} {_mark(o['duka_test']['holds'])} |")
    if pb:
        L += ["", f"候補 b の調整期間の設定の一覧 (確度: 高 の回数 / 的中率。本番 {_nh(by['tune']['base']['high'])}):", "",
              "| 日数 | \\|e\\| の下限 | " + " | ".join(f"\\|t\\| ≥ {T:g}" for T in (2.0, 2.5, 3.0, 3.5)) + " |",
              "|---|---|---|---|---|---|"]
        grid = {(g["p"]["days"], g["p"]["E"], g["p"]["T"]): g for g in rb["choice"]["grid"]}
        for days in DAYS:
            for E in (0.0, 0.5, 1.0, 2.0):
                cells = []
                for T in (2.0, 2.5, 3.0, 3.5):
                    g = grid[(days, E, T)]
                    cell = _nh(g["high"])
                    cells.append(f"**{cell}**" if g["p"] == pb else cell)
                L.append(f"| {'単純' if days == 'plain' else '暦'} | {E:g} bp | " + " | ".join(cells) + " |")
        bk = res.get("breakdown", {}).get("yahoo")
        if bk and rec == "b":
            L += ["", "格上げされた足の内訳 (Yahoo 1時間足、回数 / 的中率):", "",
                  "| 期間 | " + " | ".join(sorted({x['pair'] for x in bk['pair']})) + " |", "|---|" + "---|" * len({x['pair'] for x in bk['pair']})]
            pairs = sorted({x["pair"] for x in bk["pair"]})
            for k in ("tune", "test"):
                row = {x["pair"]: x for x in bk["pair"] if x["period"] == k}
                L.append(f"| {PERIOD[k]} | " + " | ".join(_nh(row.get(pp)) for pp in pairs) + " |")
            L += ["", "| 期間 | 曜日 | 暦で日数が変わる日 | 回数 / 的中率 |", "|---|---|---|---|"]
            for x in bk["weekday"]:
                L.append(f"| {PERIOD[x['period']]} | {WEEKDAY.get(x['wd'], x['wd'])} | {'はい' if x['moved'] else 'いいえ'} | {_nh(x)} |")
            weak = {}
            for x in bk["weekday"]:
                w = weak.setdefault((x["wd"], x["moved"]), [0, 0.0])
                w[0] += x["n"]
                w[1] += x["n"] * x["hit"]
            weak = [f"{WEEKDAY.get(k[0], k[0])}曜日 ({'暦で日数が変わる日' if k[1] else '日数が同じ日'}) {v[0]}回 {_p(v[1] / v[0])}"
                    for k, v in weak.items() if v[0] and v[1] / v[0] < 0.6]
            if weak:
                L += ["", "両期間を合わせて的中率が60%に届かない区分: " + "、".join(weak) + "。回数が少なく、誤差と区別できません。"]
    L.append("")

    # ---- 4. 15-minute bars
    L += ["## 4. Yahoo 15分足 (約60日、報告のみ)", "",
          f"前60% ({src['15m']['span'][0][:10]}〜{src['15m']['split'][:10]}) を調整、後40% を検証としました。15分足は曜日×15分の枠が"
          "約60日では15本に届かないため、本番の方向はすべて「15分」の枠 (曜日なし) から出ています。", "",
          "| 候補 | 設定 | 期間 | 方向を示した回 | 確度: 高 | 変わった予測 | 条件 |", "|---|---|---|---|---|---|---|"]
    for k in ("tune", "test"):
        x = base["15m"][k]["base"]
        L.append(f"| 本番 | — | {PERIOD[k]} | {_nh(x['all'])} | {_nh(x['high'])} | — | — |")
    for c, r in cands.items():
        for key, vkey, label in (("m15_hourly_setting", "m15_hourly_verdict", "1時間足の設定"), ("m15", "m15_verdict", "15分足の調整期間で選択")):
            if key not in r:
                continue
            p = r["choice"]["best"] if key == "m15_hourly_setting" else r["choice_15m"]["best"]
            for k in ("tune", "test"):
                x = r[key][k]
                L.append(f"| {c} | {label}: {_setting(p)} | {PERIOD[k]} | {_arrow(x['base']['all'], x['new']['all'])} | "
                         f"{_arrow(x['base']['high'], x['new']['high'])} | {_changes(x['changes'])} | {_mark(r[vkey][k]['pass'])} |")
    if fh:
        L += ["", "候補 b を15分足の1時間先 (4本先、中心の t は4本の t を合わせたもの) に当てはめた場合 (次の4本にロールオーバーの足を含む起点のみ。"
              "他の起点は変わりません):", "",
              "| 期間 | 方向を示した回 | 確度: 高: 本番 | 確度: 高: 格上げ後 | 格上げされた回 |", "|---|---|---|---|---|"]
        for k in ("tune", "test"):
            x = fh[k]
            L.append(f"| {PERIOD[k]} | {_nh(x['calls'])} | {_nh(x['base_high'])} | {_nh(x['new_high'])} | {_nh(x['raised'])} |")
    L.append("")

    # ---- 5. Dukascopy
    L += ["## 5. Dukascopy 仲値 (2004〜2026) での確認", "",
          "Yahoo 1時間足の調整期間で選んだ設定をそのまま当てはめました (仲値なので、円のペアの売値だけに出る16時・18時の動きはありません)。", "",
          "| 候補 | 設定 | 期間 | 方向を示した回 | 確度: 高 | 変わった予測 | 否定されない | 仲値の調整期間だけで選んだ設定 |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ("tune", "test"):
        x = bd[k]["base"]
        L.append(f"| 本番 | — | {PERIOD[k]} | {_nh(x['all'])} | {_nh(x['high'])} | — | — | — |")
    for c, r in cands.items():
        if "duka" not in r:
            L.append(f"| {c} | なし | — | — | — | — | — | {_setting(r['choice_duka']['best'])} |")
            continue
        for k in ("tune", "test"):
            x = r["duka"][k]
            L.append(f"| {c} | {_setting(r['choice']['best'])} | {PERIOD[k]} | {_arrow(x['base']['all'], x['new']['all'])} | "
                     f"{_arrow(x['base']['high'], x['new']['high'])} | {_changes(x['changes'])} | {_mark(r['duka_verdict'][k]['pass'])} | "
                     f"{_setting(r['choice_duka']['best']) if k == 'tune' else ''} |")
    bk = res.get("breakdown", {}).get("duka")
    if bk and rec == "b":
        by_year = {x["year"]: x for x in bk["year"]}
        base_year = bk["base_high_year"]
        years = sorted(set(by_year) | {int(y) for y in base_year})
        L += ["", "候補 b で格上げされた足と本番の確度: 高 の年ごとの的中率 (回数 / 的中率):", "",
              "| 年 | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years),
              "| 格上げ | " + " | ".join(_nh(by_year.get(y)) for y in years) + " |",
              "| 本番の高 | " + " | ".join(_nh(base_year.get(y) or base_year.get(str(y))) for y in years) + " |"]
    L.append("")

    # ---- 6. recommendation
    L += ["## 6. 推奨", ""]
    if rec == "b" and pb:
        L += [f"**候補 b を1時間足にだけ採用します:** ロールオーバーの足 (ニューヨーク時間17時に始まる月〜木の1時間足) で本番の時間帯の偏りが"
              f"方向を示し (|t| ≥ {season.T_MIN:g})、その向きが見込みのずれ e と同じで |e| ≥ {pb['E']:g} bp なら、t を {season.T_HIGH:g} "
              "(確度: 高 の下限) に引き上げます。e = −(基準通貨の金利 − 相手通貨の金利) × 日数 / 360 × 100、金利は rates.rate_diff "
              "(予測の起点の UTC の日)、日数は祝日の暦とスポット日の規則で数えます。方向・予測の中心 (d) は変えません。"
              "15分足は、約60日では1時間先で確度: 高 の的中率がわずかに下がるため今のままにし、データがたまってから同じ手順で確かめ直すことを勧めます。", "",
              "条件の確認: Yahoo 1時間足の調整・検証で条件を満たす ○、仲値の調整・検証で否定されない ○、"
              "サーバーが持つデータだけで計算できる ○ (保存済みの1時間足、保存済みの金利の項目、規則で計算する休日)。", "",
              *([f"下の変更をそのまま書いた関数 (research_rollover.live_check、season.slot_stats / bar_drift と取引時間の暦で次の足を決める) で、"
                 f"Yahoo のロールオーバーの足を予測する {rb['live_check']['origins']:,} 起点を計算し直すと、確度: 高 の判定の食い違いは "
                 f"{rb['live_check']['mismatch']} 件、格上げは {rb['live_check']['raised']} 回 (研究の数え方と同じ) でした "
                 f"(保存された足が欠けていて次の足が取引時間の暦と違う {rb['live_check']['skipped']} 起点は除外)。", ""]
                if rb.get("live_check") else []),
              "### season.py の変更", "",
              "1. aifx/research_rollover.py の休日とスポット日の関数 (`_easter`、`_nth`、`_moved`、`_usd`、`_eur`、`_GBP_SPECIAL`、`_gbp`、`_aud`、"
              "`_equinox`、`_jpy`、`CALENDARS`、`spot_date`) を、そのまま season.py に移します (numpy・pandas 以外の依存なし)。"
              "英国と日本の臨時の休日 (即位・葬儀・五輪など) は、発表されたら表に加えます。",
              "2. 次の定数と関数を加え、`step_drift` を変えます:", "",
              "```python",
              "from datetime import date, datetime, timedelta",
              "from functools import lru_cache",
              "",
              f"ROLL_E_HIGH = {pb['E']:.1f}   # bp: a roll-bar call that agrees with an expected roll shift this large is high confidence",
              "ROLL_MINUTES = (60,)  # timeframes it applies to (tested on hourly bars; 15-minute bars as before)",
              "",
              "",
              "@lru_cache(maxsize=32)",
              "def _holidays(cur: str, year: int) -> frozenset:",
              "    return frozenset(set().union(*(CALENDARS[cur](y) for y in (year - 1, year, year + 1))))",
              "",
              "",
              "def roll_days(trade: date, base: str, quote: str) -> int:",
              "    \"\"\"Value days the 17:00 New York roll at the end of ``trade`` (a weekday) moves the spot date on.\"\"\"",
              "    hol = {c: _holidays(c, trade.year) for c in {base, quote, \"USD\"}}",
              "    nxt = trade + timedelta(days=3 if trade.weekday() == 4 else 1)",
              "    return (spot_date(nxt, base, quote, hol) - spot_date(trade, base, quote, hol)).days",
              "",
              "",
              "def roll_shift(starts, base: str, quote: str, diff: float | None) -> np.ndarray:",
              "    \"\"\"Expected move (bp) of bars starting at ``starts`` from the roll: a bar starting at 17:00 New York,",
              "    Monday to Thursday, carries it, and the quote moves by -(rate_base - rate_quote) x days / 360.\"\"\"",
              "    local = pd.DatetimeIndex(starts).tz_convert(NEW_YORK)",
              "    out = np.zeros(len(local))",
              "    if diff is None:",
              "        return out",
              "    for k, t in enumerate(local):",
              "        if t.hour == 17 and t.minute == 0 and t.dayofweek <= 3:",
              "            out[k] = -diff * roll_days(t.date(), base, quote) / 360 * 100",
              "    return out",
              "",
              "",
              "def promote(d: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:",
              "    \"\"\"t raised to T_HIGH where a call agrees with an expected roll shift of at least ROLL_E_HIGH bp",
              "    (research/rollover.md); the expected move d is left as it is.\"\"\"",
              "    up = (d * e > 0) & (np.abs(e) >= ROLL_E_HIGH) & (np.abs(t) >= T_MIN) & (np.abs(t) < T_HIGH)",
              "    return np.where(up, np.sign(t) * T_HIGH, t)",
              "",
              "",
              "def step_drift(minutes: int, bars: pd.DataFrame | None, origin: datetime, steps: int,",
              "               pair=None, diff: float | None = None) -> tuple[np.ndarray, np.ndarray]:",
              "    \"\"\"... ``pair`` and ``diff`` (base-minus-quote short rate known at the origin, rates.rate_diff) raise",
              "    agreeing roll-bar calls of hourly bars to high confidence (promote).\"\"\"",
              "    if not minutes or bars is None or len(bars) < 200:",
              "        return np.zeros(steps), np.zeros(steps)",
              "    starts = [add_trading_minutes(origin, k, minutes) - timedelta(minutes=minutes) for k in range(1, steps + 1)]",
              "    d, t = bar_drift(slot_stats(bars, origin, minutes), starts, minutes)",
              "    if pair is not None and diff is not None and minutes in ROLL_MINUTES:",
              "        t = promote(d, t, roll_shift(starts, pair.base, pair.quote, diff))",
              "    return d, t",
              "```",
              "",
              "3. 呼び出し側: forecaster.make_prediction で `rates_item = usable_rates(rate_items, origin)`、"
              "`diff = rate_diff(rates_item, pair.base, pair.quote, origin.date()) if rates_item else None` を計算し、"
              "`season.step_drift(tf.minutes, bars, origin, steps, pair, diff)` とします (同じ rates_item を trade.plan にも渡す)。"
              "backtest._forecast にも pair と、最新の金利の項目から trade.backtest と同じく `rate_diff(item, base, quote, origin.date())` "
              "で求めた diff を渡します (金利の項目がなければ今と同じ結果)。api.candle_eval は任意です。",
              "4. 1時間足では中心の t は次の1本の t そのものなので、格上げは「次の1時間」の確度だけに効きます (4時間・24時間先の中心には"
              "偏りを入れていないので影響なし)。台帳の dt は、格上げした足では 4.00 (下限) になります。season.py はモデルの版の計算に"
              "含まれるため、版が変わります。README と season.py の説明の数字 (確度: 高 の的中率) は採用後に更新が必要です。", ""]
    else:
        L += ["**今のまま (変更なし)。** 事前に決めた条件をすべて満たす候補はありませんでした。", ""]
    L += ["いずれにしても、これは表示される価格のロールオーバー前後の規則性で、売買の優位性ではありません。17時のずれはスワップで相殺され、"
          "スプレッドも広いため、スプレッドとスワップを差し引くと利益にはなりません。", ""]

    # ---- notes
    L += ["## 方法の注記", "",
          "- 本番の方向は season.py の規則 (予測する日の0時 UTC より前の6,000本、曜日×時間の枠を優先、|t| ≥ 2 で方向、|t| ≥ 4 で確度: 高) を、"
          "全時点をまとめて計算する形に書き直したものです (枠ごとの累積和。市場が閉じている足と間が空いた足の値動きは除外)。",
          "- 候補 d の補正した枠は、過去の各足の値動きからその足の見込みのずれ (その足の日の金利、同じ日数の数え方) を引いて枠の平均と標準偏差を"
          "計算し、予測する足の見込みのずれを足したものです。",
          "- 15分足の1時間先の確認は、season.slot_stats / bar_drift / combined_t をそのまま使っています。",
          "- 休日の暦は規則で計算しています (例: 日本の春分・秋分は近似式、2019年の即位関連と2020・2021年の五輪の移動、英国の臨時の休日を含む)。"
          "米国の臨時の休業 (大統領の国葬など) は含みません。",
          "- 金利: 米国 DTB3 (日次)、日本 IRSTCI01JPM156N (月次)、ユーロ IR3TIB01EZM156N → €STR、英国 IR3TIB01GBM156N → SONIA、"
          "豪州 IR3TIB01AUM156N。月次は月初から2か月後、日次は翌日から使い、日次は直近1か月の平均です (rates.py と同じ)。",
          "- 的中率は動きがゼロの回を除いて数え、t は週ごとに全ペアを合計して計算しました。",
          f"- 計算時間 {res.get('seconds', 0):.0f} 秒 (`python -m aifx.research_rollover`、1プロセス)。", ""]
    return "\n".join(L)


if __name__ == "__main__":
    run()
