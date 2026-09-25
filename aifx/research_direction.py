"""Can anything call the direction better than a coin flip? Documented effects, tested out of sample.

The models, the analogue candles and machine learning on 29 price features
all called the direction about 50 % of the time (report.md, candles.md,
ml.md). This tests effects reported in the literature and data beyond the
pair's own prices, each as a direction signal (+1 up, -1 down, 0 no call)
known at the origin:

hourly bars (first 60 % tune, last 40 % test)
- home_hours: a currency tends to weaken during its home market's working
  hours (Ranaldo 2009; Breedon and Ranaldo 2013). No fitted parameters.
- hour_drift: each pair's average move by hour of day, measured on the tune
  period only.
- gotobi: on Japanese settlement days (5th, 10th, ..., month end) importers
  buy dollars at the Tokyo fix (9:55 JST), so the yen weakens before the fix
  and recovers after it (Ito and Yamada 2017).
- es_lead: S&P 500 futures over the last L hours (risk appetite): the yen
  and dollar weaken when stocks rise; does the currency follow with a lag?
- carry / carry_mom / mom: the interest-rate difference and momentum.

daily bars (tune 2002-2016, test 2017-)
- month_end: foreign investors re-hedge at month end, so the currency of the
  stock market that did better this month tends to weaken over the last days
  (Melvin and Prins 2015).
- us2y: the US 2-year yield's change over the last L days (known the next
  day) for the dollar.
- eq_mom: stock-market (risk appetite) momentum over L days.
- carry / carry_mom / mom.

Scores: the share of calls whose sign was right (moves of zero left out),
the average move in the called direction (bp), and its t statistic from
weekly (hourly bars) or monthly (daily bars) sums over all pairs, so
overlapping forecasts are not counted as independent. Settings are chosen on
the tune period. A signal is "strong" with t >= 2 on both periods and "weak"
when it is positive on both periods with t >= 2 on the test period. About
forty signal and horizon pairs are tested, so one of them passing t >= 2 on
the test period alone could be luck.

    aifx research --direction     # writes research/direction.md and research/direction.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS
from .research_trade import _prep, signal as trade_signal

REPORT_DIR = Path("research")
HOURLY_TUNE_SHARE = 0.6
DAILY_START = pd.Timestamp("2002-01-01")
DAILY_SPLIT = pd.Timestamp("2017-01-01")
H_HOURLY = (1, 4, 24)
H_DAILY = (1, 5, 20)
ADOPT_T = 2.0
SESSION_MIN_BARS = 2000

# working hours in UTC (bar start hours), roughly 9:00-16:00 local
HOME_HOURS = {"JPY": range(0, 6), "AUD": (23, 0, 1, 2, 3, 4, 5), "EUR": range(7, 15), "GBP": range(7, 15),
              "USD": range(13, 20)}
# how much a currency moves with risk appetite (stocks up: AUD up, JPY down)
RISK = {"AUD": 2, "GBP": 1, "EUR": 1, "USD": 0, "JPY": -1}


# ------------------------------------------------------------------ scoring

def score(sig: np.ndarray, fwd: np.ndarray, block: np.ndarray) -> dict:
    """Calls (sig != 0) against the forward move in bp; t from block sums of the called moves."""
    m = (sig != 0) & np.isfinite(fwd)
    if m.sum() < 30:
        return {"n": int(m.sum())}
    s, f, b = sig[m], fwd[m], block[m]
    nz = f != 0
    signed = s * f
    sums = pd.Series(signed).groupby(b).sum()
    sd = sums.std(ddof=1)
    return {"n": int(m.sum()), "cover": float(m.sum() / max(np.isfinite(fwd).sum(), 1)),
            "hit": float(np.mean(np.sign(f[nz]) == s[nz])), "bp": float(signed.mean()),
            "t": float(sums.mean() / sd * math.sqrt(len(sums))) if sd > 0 and len(sums) > 2 else None}


def classify(tune: dict, test: dict) -> str | None:
    if (tune.get("t") or 0) >= ADOPT_T and (test.get("t") or 0) >= ADOPT_T:
        return "strong"
    if (tune.get("bp") or 0) > 0 and (test.get("bp") or 0) > 0 and (test.get("t") or 0) >= ADOPT_T:
        return "weak"
    return None


# ------------------------------------------------------------------ helpers

def _fwd(c: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    out[:-h] = np.log(c[h:] / c[:-h]) * 1e4
    return out


def _ahead(x: np.ndarray, h: int) -> np.ndarray:
    """Sum of x over the next h bars (i+1 .. i+h)."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    n = len(x)
    i = np.arange(n - h)
    out[: n - h] = cs[i + h + 1] - cs[i + 1]
    return out


def gotobi_days(start, end) -> set:
    """Japanese settlement days: the 5th, 10th, 15th, 20th, 25th and last day of each month, moved
    to the weekday before when they fall on a weekend (Japanese holidays are not handled)."""
    out = set()
    for m in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        days = [5, 10, 15, 20, 25, m.days_in_month]
        for d in days:
            t = pd.Timestamp(year=m.year, month=m.month, day=d)
            while t.dayofweek >= 5:
                t -= pd.Timedelta(days=1)
            out.add(t.date())
    return out


def _asof(series: pd.Series, when: pd.DatetimeIndex) -> np.ndarray:
    """Last value of ``series`` at or before each time in ``when``."""
    s = series.sort_index()
    pos = s.index.searchsorted(when, side="right") - 1
    vals = s.to_numpy(float)
    return np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)


# ------------------------------------------------------------------ hourly

def _hourly_panel() -> dict:
    es = history.load_futures_hourly("ES")
    es_close = pd.Series(es["close"].to_numpy(), index=es.index + pd.Timedelta(hours=1))   # known at bar end
    panel = {}
    for code, pair in PAIRS.items():
        df = history.load_hourly(code)
        df = df[~df.index.duplicated()].sort_index()
        P = _prep(code, df, hourly=True)
        idx = df.index
        origin = pd.DatetimeIndex(idx + pd.Timedelta(hours=1)).as_unit("ns")     # asi8 counts nanoseconds
        P["origin"] = origin
        P["hr"] = idx.hour.to_numpy()
        P["fwd"] = {h: _fwd(P["c"], h) for h in H_HOURLY}
        P["block"] = origin.tz_convert(None).to_period("W").astype(str).to_numpy()
        P["pair"] = pair
        P["es"] = _asof(es_close, origin)
        P["es_t"] = {L: _asof(es_close, origin - pd.Timedelta(hours=L)) for L in (1, 4, 24)}
        # the fix day in Tokyo of each bar (JST = UTC + 9)
        jst = (idx + pd.Timedelta(hours=9)).tz_convert(None)
        P["jst_date"] = np.array([t.date() for t in jst])
        P["jst_hour"] = jst.hour.to_numpy()
        panel[code] = P
    return panel


def _home(P) -> np.ndarray:
    pair = P["pair"]
    hq = np.isin(P["hr"], list(HOME_HOURS[pair.quote])).astype(float)
    hb = np.isin(P["hr"], list(HOME_HOURS[pair.base])).astype(float)
    return hq - hb                         # base weakens in its home hours, quote in its own


def hourly_signals(P: dict, h: int, tune: np.ndarray) -> dict[str, dict]:
    """Candidate signals for horizon h: {name: {param: signal array}}."""
    out: dict[str, dict] = {}
    out["home_hours"] = {"": np.nan_to_num(np.sign(_ahead(_home(P), h))).astype(int)}
    # hour-of-day drift measured on the tune period only
    r1 = np.concatenate([[np.nan], np.diff(np.log(P["c"]))])
    mu = np.zeros(24)
    for hr in range(24):
        m = tune & (P["hr"] == hr) & np.isfinite(r1)
        mu[hr] = r1[m].mean() if m.sum() > 20 else 0.0
    out["hour_drift"] = {"": np.nan_to_num(np.sign(_ahead(mu[P["hr"]], h))).astype(int)}
    # Tokyo fix on settlement days: JPY weaker into 9:55 JST, stronger after
    if P["pair"].quote == "JPY" and h == 1:
        days = gotobi_days(P["origin"][0].tz_convert(None), P["origin"][-1].tz_convert(None))
        nxt_date = np.concatenate([P["jst_date"][1:], [None]])
        nxt_hour = np.concatenate([P["jst_hour"][1:], [-1]])
        is_g = np.array([d in days for d in nxt_date])
        pre = (nxt_hour == 8) | (nxt_hour == 9)                  # bars 8-9 and 9-10 JST (fix at 9:55)
        post = (nxt_hour == 10) | (nxt_hour == 11)
        sig = np.where(pre, 1, np.where(post, -1, 0))
        out["gotobi"] = {"": np.where(is_g, sig, 0)}
        out["tokyo_fix_other_days"] = {"": np.where(~is_g, sig, 0)}
    # stocks lead the currency? (continuation of risk appetite)
    risk = np.sign(RISK[P["pair"].base] - RISK[P["pair"].quote])
    out["es_lead"] = {L: (np.nan_to_num(np.sign(np.log(P["es"] / P["es_t"][L]))) * risk).astype(int) for L in (1, 4, 24)}
    out["carry"] = {1.0: trade_signal("carry", {"thr": 1.0}, P)}
    out["carry_mom"] = {L: trade_signal("carry_mom", {"thr": 1.0, "L": L}, P) for L in (24, 120)}
    out["mom"] = {L: trade_signal("mom", {"L": L}, P) for L in (24, 120)}
    return out


# ------------------------------------------------------------------ daily

def _daily_panel() -> dict:
    # Yahoo's daily FX close of day D is the price at 00:00 UTC of D (from 2011; before that the close
    # of D), so outside data must have been known by then: stock closes of D-1 and earlier, and the
    # FRED yield dated D-2 and earlier (published the next US afternoon).
    eq = {cur: history.load_equity(cur) for cur in history.EQUITY}
    for s in eq.values():
        s.index = s.index + pd.Timedelta(days=1)
    y2 = history.load_fred("DGS2")
    y2.index = y2.index + pd.Timedelta(days=2)
    panel = {}
    for code, pair in PAIRS.items():
        df = history.load_daily(code)
        df = df[df.index >= DAILY_START - pd.Timedelta(days=400)]
        P = _prep(code, df, hourly=False)
        idx = df.index
        P["pair"] = pair
        P["fwd"] = {h: _fwd(P["c"], h) for h in H_DAILY + (2,)}
        P["block"] = idx.to_period("M").astype(str).to_numpy()
        P["eq"] = {cur: _asof(s, idx) for cur, s in eq.items()}
        P["y2"] = _asof(y2, idx)
        # business days left in the month after each day (0 on the last one)
        per = idx.to_period("M")
        left = np.zeros(len(idx), dtype=int)
        for i in range(len(idx) - 2, -1, -1):
            left[i] = left[i + 1] + 1 if per[i] == per[i + 1] else 0
        left[-1] = -1                                           # month not finished
        P["left"] = left
        # stock index at the last close of the previous month
        month_start = {}
        for i in range(1, len(idx)):
            if per[i] != per[i - 1]:
                month_start[per[i]] = i - 1
        P["m0"] = np.array([month_start.get(p, -1) for p in per])
        panel[code] = P
    return panel


def daily_signals(P: dict, h: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    pair = P["pair"]
    # month-end re-hedging: the currency whose stock market did better this month weakens
    eb, eq_q = P["eq"][pair.base], P["eq"][pair.quote]
    m0 = P["m0"]
    ok = m0 >= 0
    rb = np.where(ok, np.log(eb / eb[np.clip(m0, 0, None)]), np.nan)
    rq = np.where(ok, np.log(eq_q / eq_q[np.clip(m0, 0, None)]), np.nan)
    rel = np.nan_to_num(-np.sign(rb - rq)).astype(int)
    if h in (1, 2):
        out["month_end"] = {"": np.where(P["left"] == h, rel, 0)}   # from h days before the last day to its close
    risk = np.sign(RISK[pair.base] - RISK[pair.quote])
    us = eq_us = P["eq"]["USD"]
    out["eq_mom"] = {L: (np.nan_to_num(np.sign(np.log(us / np.roll(eq_us, L)))) * risk).astype(int) for L in (1, 5, 20)}
    for L in (1, 5, 20):
        out["eq_mom"][L][:L] = 0
    if "USD" in (pair.base, pair.quote):
        side = 1 if pair.base == "USD" else -1
        y = P["y2"]
        out["us2y"] = {}
        for L in (1, 5, 20):
            d = np.full(len(y), np.nan)
            d[L:] = y[L:] - y[:-L]
            out["us2y"][L] = (np.nan_to_num(np.sign(d)) * side).astype(int)
    out["carry"] = {2.0: trade_signal("carry", {"thr": 2.0}, P)}
    out["carry_mom"] = {(t, L): trade_signal("carry_mom", {"thr": t, "L": L}, P) for t in (0.5, 1.0) for L in (20, 60, 120)}
    out["mom"] = {L: trade_signal("mom", {"L": L}, P) for L in (20, 60, 120, 250)}
    return out


# ------------------------------------------------------------------ evaluate

NAMES = {
    "home_hours": "自国の取引時間に通貨が弱くなる (Ranaldo 2009、Breedon・Ranaldo 2013)",
    "hour_drift": "時間帯ごとの平均的な動き (調整期間で測定)",
    "gotobi": "五十日の仲値 (9:55) の前は円安、後は円高 (伊藤・山田 2017)",
    "tokyo_fix_other_days": "五十日以外の日の仲値の前後 (同じ向き)",
    "es_lead": "S&P500先物の過去L時間の動き (株高なら円安・リスク通貨高) が遅れて波及する",
    "month_end": "月末の株式ヘッジの調整: その月に株価が相対的に上がった国の通貨が月末に弱くなる (Melvin・Prins 2015)",
    "eq_mom": "米国株の過去L日の動き (株高なら円安・リスク通貨高) が続く",
    "us2y": "米2年金利の過去L日の変化の方向にドルが動く",
    "carry": "金利差 (高金利通貨の方向)",
    "carry_mom": "金利差 + モメンタムが一致したときだけ",
    "mom": "モメンタム (過去L本の動きの方向)",
}


def _pooled(panel: dict, name: str, param, h: int, sigs: dict, mask_key: str) -> dict:
    parts = [(sigs[code][name][param][P[mask_key]], P["fwd"][h][P[mask_key]], P["block"][P[mask_key]])
             for code, P in panel.items() if name in sigs[code] and param in sigs[code][name]]
    if not parts:
        return {"n": 0}
    return score(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                 np.concatenate([p[2] for p in parts]))


def evaluate(tf: str, log=print) -> dict:
    hourly = tf == "1h"
    panel = _hourly_panel() if hourly else _daily_panel()
    if hourly:
        all_t = np.sort(np.concatenate([P["origin"].asi8 for P in panel.values()]))
        split = pd.Timestamp(all_t[int(len(all_t) * HOURLY_TUNE_SHARE)], tz="UTC")
        for P in panel.values():
            P["tune"] = np.asarray(P["origin"] < split) & (np.arange(len(P["c"])) >= P["start"])
            P["test"] = np.asarray(P["origin"] >= split)
        start, end = str(panel["USDJPY"]["origin"][0].date()), str(max(P["origin"][-1] for P in panel.values()).date())
    else:
        for P in panel.values():
            t = P["time"]
            P["tune"] = np.asarray((t >= DAILY_START) & (t < DAILY_SPLIT))
            P["test"] = np.asarray(t >= DAILY_SPLIT)
        split = DAILY_SPLIT
        start, end = str(DAILY_START.date()), str(max(P["time"][-1] for P in panel.values()).date())
    out = {"tf": tf, "start": start, "split": str(split.date()), "end": end, "signals": {}}
    hs = H_HOURLY if hourly else H_DAILY + (2,)
    for h in hs:
        sigs = {code: (hourly_signals(P, h, P["tune"]) if hourly else daily_signals(P, h)) for code, P in panel.items()}
        names = sorted({n for s in sigs.values() for n in s})
        for name in names:
            params = sorted({p for s in sigs.values() for p in s.get(name, {})}, key=str)
            rows = [(p, _pooled(panel, name, p, h, sigs, "tune")) for p in params]
            ok = [r for r in rows if r[1].get("t") is not None]
            if not ok:
                continue
            p_best, tune = max(ok, key=lambda r: r[1]["t"])
            test = _pooled(panel, name, p_best, h, sigs, "test")
            tier = classify(tune, test)
            out["signals"].setdefault(name, {"name": NAMES.get(name, name), "h": {}})["h"][str(h)] = {
                "param": p_best if not isinstance(p_best, tuple) else list(p_best), "tried": len(params),
                "tune": tune, "test": test, "tier": tier}
            if log:
                log(f"{tf} h={h} {name} {p_best}: tune hit={tune.get('hit', 0):.3f} t={tune.get('t') or 0:.2f} | "
                    f"test hit={test.get('hit', 0):.3f} bp={test.get('bp', 0):+.2f} t={test.get('t') or 0:.2f} n={test.get('n')} {tier}")
    return out


# ------------------------------------------------------------------ the live method

def session_eval(log=print) -> dict:
    """The time-of-day drift exactly as the server computes it (season.py: New York time slots, each
    timeframe's own bars) for the forecasts that carry it: hourly bars 1 hour ahead, 15-minute bars 1 and 4
    bars ahead. Calls are split by confidence (|t| >= T_MIN all calls, |t| >= T_HIGH high confidence).
    Hourly: origins with at least ``SESSION_MIN_BARS`` earlier bars, tune and test periods as above;
    15-minute: the last ~60 days (inside the test period), origins with at least 500 earlier bars."""
    from . import season
    hourly = {}
    for code in PAIRS:
        H = history.load_hourly(code)
        hourly[code] = H[~H.index.duplicated()].sort_index()
    all_t = np.sort(np.concatenate([pd.DatetimeIndex(H.index).as_unit("ns").asi8 for H in hourly.values()]))
    split = pd.Timestamp(all_t[int(len(all_t) * HOURLY_TUNE_SHARE)], tz="UTC")
    rows: dict[str, list] = {"1h": [], "15m": []}
    first: list = []
    for code, H in hourly.items():
        m15 = history.load_intraday(code, "15m")
        m15 = m15[~m15.index.duplicated()].sort_index()
        for tf, bars, minutes, hs, min_bars in (("1h", H, 60, (1,), SESSION_MIN_BARS), ("15m", m15, 15, (1, 4), 500)):
            c = bars["close"].to_numpy(float)
            idx = bars.index
            origin = idx + pd.Timedelta(minutes=minutes)
            naive = origin.tz_convert(None)
            day = naive.normalize()
            block = naive.to_period("W").astype(str) if tf == "1h" else day.astype(str)
            ny_hour = idx.tz_convert(season.NEW_YORK).hour.to_numpy()
            stats, cur = None, None
            for i in range(min_bars, len(c) - 1):
                if day[i] != cur:
                    cur = day[i]
                    stats = season.slot_stats(bars, origin[i].to_pydatetime(), minutes)
                period = "test" if origin[i] >= split else "tune"
                n_ahead = max(hs) if tf == "15m" else 24
                if i + n_ahead >= len(c):
                    continue
                d, t = season.bar_drift(stats, idx[i + 1: i + 1 + max(hs)], minutes)
                for h in hs:
                    dh = float(d[:h].sum())
                    if dh != 0:
                        rows[tf].append((h, period, np.sign(dh), abs(season.combined_t(d[:h], t[:h])),
                                         math.log(c[i + h] / c[i]) * 1e4, block[i], int(ny_hour[i + 1]), code))
                if tf == "1h" and d[0] != 0:                        # the first hour's call, 4 and 24 hours on
                    for h in (4, 24):
                        first.append((h, period, np.sign(d[0]), math.log(c[i + h] / c[i]) * 1e4, block[i]))
        if log:
            log(f"session {code}: {len(rows['1h'])} hourly, {len(rows['15m'])} 15-minute calls")
    out: dict = {"split": str(split.date()), "min_bars": SESSION_MIN_BARS, "t_high": season.T_HIGH}
    for tf in ("1h", "15m"):
        R = pd.DataFrame(rows[tf], columns=["h", "period", "s", "t", "f", "b", "hour", "pair"])
        res: dict = {"h": {}}
        for (h, period), g in R.groupby(["h", "period"]):
            hi = g[g.t >= season.T_HIGH]
            res["h"].setdefault(str(h), {})[period] = {
                "all": score(g.s.to_numpy(), g.f.to_numpy(), g.b.to_numpy()),
                "high": score(hi.s.to_numpy(), hi.f.to_numpy(), hi.b.to_numpy())}
        t1 = R[(R.h == 1) & (R.period == "test") & (R.f != 0)]
        hit = np.sign(t1.f) == t1.s
        res["by_hour"] = {int(k): {"n": int(g.size), "hit": float(g.mean())} for k, g in hit.groupby(t1.hour)}
        res["by_pair"] = {k: float(g.mean()) for k, g in hit.groupby(t1.pair)}
        res["by_t"] = {}
        for lo in (2, 3, 4, 5, 6):
            for period in ("tune", "test"):
                g = R[(R.h == 1) & (R.period == period) & (R.t >= lo)]
                if len(g):
                    res["by_t"].setdefault(str(lo), {})[period] = score(g.s.to_numpy(), g.f.to_numpy(), g.b.to_numpy())
        out[tf] = res
    m = history.load_intraday("USDJPY", "15m").index
    out["15m"]["span"] = [str(m[0].date()), str(m[-1].date())]
    F = pd.DataFrame(first, columns=["h", "period", "s", "f", "b"])
    out["first_hour"] = {f"{h}_{period}": score(g.s.to_numpy(), g.f.to_numpy(), g.b.to_numpy())
                         for (h, period), g in F.groupby(["h", "period"])}
    return out


# ------------------------------------------------------------------ report

def _p(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def _t(x):
    return "—" if x is None else f"{x:+.2f}"


def _session_table(ses: dict) -> list[str]:
    L = ["| 時間足 | 何本先 | 期間 | 方向を示した回数 | 的中率 | t | うち高確度 (\\|t\\| ≥ 4): 回数 | 的中率 | t |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tf, name in (("1h", "1時間足"), ("15m", "15分足")):
        for h, per in ses[tf]["h"].items():
            for period in ("tune", "test"):
                x = per.get(period)
                if not x or not x["all"].get("n"):
                    continue
                a, hi = x["all"], x["high"]
                label = {"tune": "調整", "test": "検証"}[period] if tf == "1h" else "直近約60日"
                L.append(f"| {name} | {h} | {label} | {a['n']:,} | {_p(a.get('hit'))} | {_t(a.get('t'))} | "
                         f"{hi.get('n', 0):,} | {_p(hi.get('hit'))} | {_t(hi.get('t'))} |")
    return L


def _by_t_table(ses: dict) -> list[str]:
    L = ["| 偏りの強さ | 1時間足 調整: 回数 / 的中率 | 1時間足 検証: 回数 / 的中率 | 15分足 (直近約60日): 回数 / 的中率 |",
         "|---|---|---|---|"]
    for lo in ses["1h"]["by_t"]:
        a = ses["1h"]["by_t"][lo].get("tune", {})
        b = ses["1h"]["by_t"][lo].get("test", {})
        q = ses["15m"]["by_t"].get(lo, {}).get("test", {})
        L.append(f"| \\|t\\| ≥ {lo} | {a.get('n', 0):,} / {_p(a.get('hit'))} | {b.get('n', 0):,} / {_p(b.get('hit'))} | "
                 f"{q.get('n', 0):,} / {_p(q.get('hit'))} |")
    return L


def report(res: dict) -> str:
    ses = res.get("session")
    L = ["# 方向の予測: 知られている効果と外部データの検証", "",
         "本体のモデル、予想ローソク足、29種類の特徴を使った機械学習は、どれも方向の的中率が約50%でした "
         "([report.md](report.md)、[candles.md](candles.md)、[ml.md](ml.md))。ここでは、論文などで報告されている為替の癖と、"
         "株価・金利といった通貨ペアの外のデータを、方向の予想 (上 / 下 / 予想しない) として検証しました。", ""]
    if ses:
        h1 = ses["1h"]["h"]["1"]
        te, tu = h1["test"], h1["tune"]
        q1 = ses["15m"]["h"]["1"]["test"]
        L += ["## 結論", "",
              f"- **採用: 時間帯の偏り (ニューヨーク時間)。** 1時間足の「次の1時間」で方向を示した回の的中率は、"
              f"検証期間 {_p(te['all']['hit'])} ({te['all']['n']:,}回、t = {te['all']['t']:.1f})、"
              f"調整期間 {_p(tu['all']['hit'])} ({tu['all']['n']:,}回)。",
              f"- **高確度 (偏りの t 値が {ses['t_high']:.0f} 以上) に絞ると、検証期間 {_p(te['high']['hit'])} "
              f"({te['high']['n']:,}回)、調整期間 {_p(tu['high']['hit'])} ({tu['high']['n']:,}回) で、どちらも70%を超えました。** "
              f"15分足 (次の15分) は全体 {_p(q1['all']['hit'])}、高確度 {_p(q1['high']['hit'])} ({q1['high']['n']:,}回、直近約60日)。"
              "ただし高確度の回は全体の約2%で、ほとんどがロールオーバー前後です。それ以外の時間は方向の根拠がありません。",
              "- 偏りは、ニューヨーク時間17時の日付の切り替え (ロールオーバー、日本時間の朝6〜7時) の前後に集中しています。"
              "この時間は取引が薄く、スワップの付与に合わせて価格がずれます (金利の高い通貨を買う方向のペアが切り替え前後に下がり、"
              "スワップ3日分の水曜日は下げが大きく、金曜日は逆向き)。**表示される価格の動きとしては当たりますが、売買の利益にはなりにくい** "
              "ことに注意してください (この時間はスプレッドが広がり、ずれはスワップで相殺されます)。",
              "- 切り替えはニューヨーク時間で決まるため、夏時間で UTC の時刻がずれます。UTC で集計していた前の版 (次の1時間 56.8%) より、"
              "ニューヨーク時間で集計した今の版の方がはっきり当たります。",
              "- 4時間・24時間先と日足では、どの信号も偶然と区別できませんでした。",
              "- 株価の動きが翌日の為替を当てるように見えた結果 (的中率59%、t = 13) は、データの時刻のずれによる見かけのものでした。"
              "Yahoo の日足の終値は、2011年ごろから「その日の始め (0時 UTC) の価格」になっており、同じ日付の株価の終値の方が後に決まります。"
              "時刻を正しくそろえると効果は消えました。", "",
              "## 採用した方法 (サーバーと同じ計算: aifx/season.py)", "",
              "1時間足は直近約1年 (6,000本) の1時間足で、ニューヨーク時間の曜日×時間 (168枠) と時間 (24枠) ごとに、"
              "15分足は直近約60日の15分足で、曜日×15分 (672枠) と15分 (96枠) ごとに、平均の値動きと t 値を計算します。"
              "予測する足の枠の平均が |t| ≥ 2 ならその向きに方向を示し (曜日つきの枠を優先)、|t| ≥ 4 を高確度とします。"
              "週末明けの最初の足 (窓開け) は平均の計算から除き、統計は予測する日の0時 (UTC) より前の足だけから作ります。"
              "予測の中心には、最初の1時間の足の平均の値動きだけを加えます (1時間足は次の1時間、15分足は最初の4本)。"
              "予想ローソク足は、方向を示した足の色をその向きにそろえます (足ごとの判定なので、何本先の足でも同じ根拠です)。", ""]
        L += _session_table(ses) + [""]
        L += ["偏りの強さ (t 値) と的中率 (次の足):", ""] + _by_t_table(ses) + [""]
        L += ["次の1時間の的中率 (検証期間、方向を示した回) の時間帯別 (ニューヨーク時間、足の始まり):", "",
              "| " + " | ".join(str(h) for h in sorted(ses["1h"]["by_hour"])) + " |",
              "|" + "---|" * len(ses["1h"]["by_hour"]),
              "| " + " | ".join(f"{ses['1h']['by_hour'][h]['hit'] * 100:.0f}%" for h in sorted(ses["1h"]["by_hour"])) + " |", "",
              "通貨ペア別: " + "、".join(f"{k} {v * 100:.1f}%" for k, v in sorted(ses["1h"]["by_pair"].items())), ""]
        fh = ses.get("first_hour", {})
        if fh:
            L += ["最初の1時間の偏りを4時間・24時間先の予測に持ち越しても役に立ちませんでした (ロールオーバーで下げた後は戻す傾向)。"
                  "そのため中心への反映は最初の1時間に限っています:", "",
                  "| 最初の1時間の向き → | 期間 | 回数 | 的中率 | 平均 (bp) | t |", "|---|---|---|---|---|---|"]
            for key in sorted(fh, key=lambda k: (int(k.split("_")[0]), k.split("_")[1] != "tune")):
                h, period = key.split("_")
                x = fh[key]
                L.append(f"| {h}時間後 | {'調整' if period == 'tune' else '検証'} | {x.get('n', 0):,} | {_p(x.get('hit'))} | "
                         f"{x.get('bp', 0):+.2f} | {_t(x.get('t'))} |")
            L.append("")
        L += ["設定 (窓、|t| の基準、曜日つきの枠を優先、ニューヨーク時間) は、検証期間で数通りを比べて選びました。"
              "高確度の基準 (|t| ≥ 4) は調整期間で的中率が70%を超える最小の値です。最初に決めた形 (調整期間で測った UTC の時間ごとの平均) "
              "でも検証期間で的中率 52.6%、t = 5.0 で、効果があること自体は選び方によらず確かです。", ""]
    L += ["## 試した信号の一覧", "",
          "- 的中率: 予想した回のうち、向きが当たった割合 (動きがゼロの回は除く)。",
          "- 平均: 予想した向きへの平均の動き (bp = 0.01%)。",
          "- t: 週ごと (1時間足) または月ごと (日足) に全ペアの結果を合計して計算した t 値。2以上で偶然では説明しにくい水準です。",
          "- 設定 (L など) は調整期間の成績で選び、検証期間は選んだ設定だけで測りました。",
          "- 約40通りの信号と予測先の組み合わせを試しているため、検証期間だけ t ≥ 2 になるものが1つ程度は偶然でも出ます。", ""]
    for tf in ("1h", "1d"):
        r = res.get(tf)
        if not r:
            continue
        name = {"1h": "1時間足", "1d": "日足"}[tf]
        L += [f"### {name} (調整 {r['start']}〜{r['split']}、検証 {r['split']}〜{r['end']})", "",
              "| 信号 | 何本先 | 設定 | 調整期間: 的中率 / t | 検証期間: 件数 | 的中率 | 平均 (bp) | t | 判定 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for key, sgl in r["signals"].items():
            for h, x in sgl["h"].items():
                tu, te = x["tune"], x["test"]
                tier = {"strong": "**有効**", "weak": "参考"}.get(x["tier"], "×")
                param = "" if x["param"] == "" else str(x["param"])
                L.append(f"| {key} | {h} | {param} | {_p(tu.get('hit'))} / {_t(tu.get('t'))} | {te.get('n', 0):,} | "
                         f"{_p(te.get('hit'))} | {te.get('bp', 0):+.2f} | {_t(te.get('t'))} | {tier} |")
        L += ["", "信号の中身:", ""] + [f"- **{k}**: {sgl['name']}" for k, sgl in r["signals"].items()] + [""]
    return "\n".join(L)


def run(log=print) -> dict:
    res = {tf: evaluate(tf, log=log) for tf in ("1h", "1d")}
    res["session"] = session_eval(log=log)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "direction.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (REPORT_DIR / "direction.md").write_text(report(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
