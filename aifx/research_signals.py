"""The live trade rules re-tested on about 20 years of history, with a small pre-declared search.

The reference rules of the trade plans (trade.RULES) were chosen on thin data: the hourly rule
on Yahoo's 2.8 years of hourly bars, the daily rule on Yahoo's daily bars (whose close is not
the London-midnight price the live plans settle at). This module replays the same rules, with
the same code (research_trade.signal / simulate, trade.COST_PIPS / SWAP_MARKUP), on:

- Dukascopy hourly mid bars since 2003 (the hourly rule);
- London-day bars built from those hourly bars, and Yahoo's daily bars since 2002 (the daily rule).

Trades are split by entry date: tune = before 2017-01-01, test = from 2017-01-01. Costs per trade
are the larger of the live round-trip cost and Dukascopy's recorded bid-ask spread (half at the
entry bar's close, half at the exit bar's close); the swap accrues from the point-in-time rates
panel minus the broker's markup, as in research_trade.

Improvements are a short list of single changes declared here before any result was seen
(HOURLY_CHANGES / DAILY_CHANGES), plus one combination built from the tune period only. A change
is recommended only if it beats the current rule on both periods (mean pips after costs and the
t statistic of monthly R sums, all pairs together), improves the mean in at least 4 of the 7
pairs on the test period, and still beats the current rule on the test period with any one pair
left out. For the daily rule it must also hold on both daily data sources.

    python -m aifx.research_signals     # writes research/signals.md and research/signals.json
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
from .research_trade import (DAILY_START, HOURLY_TUNE_SHARE, MIN_TUNE_TRADES, _f, _params_text, _prep, classify, metrics,
                             signal, simulate)
from .trade import COST_PIPS, RULES, SWAP_MARKUP, rule_signal, vol_allowed

REPORT_DIR = Path("research")
SPLIT = pd.Timestamp("2017-01-01")
MIN_PAIRS_BETTER = 4             # of 7: the test-period mean must improve in at least this many pairs
SPREAD_CAP = 20.0                # a recorded spread above 20x the pair's median is a data error, capped there
BARS_PER_DAY = 24                # hourly bars per trading day (weekend hours have no bars)
RV_BARS = {True: 120, False: 20}  # realised volatility: 5 days of hourly returns, 20 daily returns
RV_YEAR = 250                    # trading days of the trailing year each value is ranked in
VIX_LEVEL, VIX_JUMP = 25.0, 0.20

# Pre-declared single changes. Each is (id, group, Japanese label, rule overrides, filters).
HOURLY_CHANGES = [
    ("thr0.5", "thr", "金利差0.5%以上", {"thr": 0.5}, []),
    ("thr2", "thr", "金利差2%以上 (= 日足ルールとの一致)", {"thr": 2.0}, []),
    ("thr3", "thr", "金利差3%以上", {"thr": 3.0}, []),
    ("L24", "L", "流れ: 過去24時間", {"L": 24}, []),
    ("L72", "L", "流れ: 過去72時間", {"L": 72}, []),
    ("L240", "L", "流れ: 過去240時間", {"L": 240}, []),
    ("sl2", "sl", "損切り ATR×2", {"sl": 2.0}, []),
    ("sl4", "sl", "損切り ATR×4", {"sl": 4.0}, []),
    ("tp2", "tp", "利確 ATR×2", {"tp": 2.0}, []),
    ("tp4", "tp", "利確 ATR×4", {"tp": 4.0}, []),
    ("tpnone", "tp", "利確なし", {"tp": None}, []),
    ("hold12", "hold", "最長12本", {"hold": 12}, []),
    ("hold48", "hold", "最長48本", {"hold": 48}, []),
    ("hold120", "hold", "最長120本", {"hold": 120}, []),
    ("noroll", "time", "21〜23時 (UTC) に確定する足では入らない (ロールオーバー前後)", {}, ["noroll"]),
    ("ldnny", "time", "8〜17時 (UTC) に確定する足でだけ入る (ロンドン・NY時間)", {}, ["ldnny"]),
    ("vol80", "vol", "値動きが過去1年の80%点を超えたら見送り", {}, ["vol80"]),
    ("vol50", "vol", "値動きが過去1年の中央値以下のときだけ", {}, ["vol50"]),
    ("ma200", "trend", "200日移動平均と同じ向きのときだけ", {}, ["ma200"]),
    ("vix25", "risk", f"VIX {VIX_LEVEL:.0f}以上なら見送り", {}, ["vix25"]),
    ("vixjump", "risk", f"VIX が5営業日前より{VIX_JUMP:.0%}以上高ければ見送り", {}, ["vixjump"]),
]
DAILY_CHANGES = [
    ("thr1", "thr", "金利差1%以上", {"thr": 1.0}, []),
    ("thr3", "thr", "金利差3%以上", {"thr": 3.0}, []),
    ("sl2.5", "sl", "損切り ATR×2.5", {"sl": 2.5}, []),
    ("sl6", "sl", "損切り ATR×6", {"sl": 6.0}, []),
    ("tp4", "tp", "利確 ATR×4", {"tp": 4.0}, []),
    ("tpnone", "tp", "利確なし", {"tp": None}, []),
    ("hold10", "hold", "最長10本", {"hold": 10}, []),
    ("hold20", "hold", "最長20本", {"hold": 20}, []),
    ("mom5", "trend", "過去5日間の流れも同じ向きのときだけ (= 1時間足ルールとの一致)", {}, ["mom5"]),
    ("mom20", "trend", "過去20日間の流れも同じ向きのときだけ", {}, ["mom20"]),
    ("ma200", "trend", "200日移動平均と同じ向きのときだけ", {}, ["ma200"]),
    ("vol80", "vol", "値動きが過去1年の80%点を超えたら見送り", {}, ["vol80"]),
    ("vol50", "vol", "値動きが過去1年の中央値以下のときだけ", {}, ["vol50"]),
    ("vix25", "risk", f"VIX {VIX_LEVEL:.0f}以上なら見送り", {}, ["vix25"]),
    ("vixjump", "risk", f"VIX が5営業日前より{VIX_JUMP:.0%}以上高ければ見送り", {}, ["vixjump"]),
]
def _filter_impl(f: str, hourly: bool) -> str:
    """What adding filter ``f`` to the live rule (trade.rule_signal) would take, for the recommendation text."""
    from .engine import TIMEFRAMES
    if f in ("vol80", "vol50"):
        n, w = RV_BARS[hourly], RV_YEAR * (BARS_PER_DAY if hourly else 1)
        fit = TIMEFRAMES["1h" if hourly else "1d"].fit_bars
        return (f"加えて `rule_signal` に、直近{n}本の対数変化の標準偏差が、その前の{w:,}本の同じ値の{'80%点' if f == 'vol80' else '中央値'}"
                f"を超えたら見送る判定を足し、`forecaster.py` から `plan()` に渡す足を{n + w + 1:,}本以上にする必要があります"
                f" (今は `fit_bars` = {fit:,}本)。")
    return {"ma200": "加えて `rule_signal` に200日移動平均の向きの判定を足す必要があります。",
            "mom5": "加えて `rule_signal` に過去5本の値動きの向きの判定を足す必要があります。",
            "mom20": "加えて `rule_signal` に過去20本の値動きの向きの判定を足す必要があります。",
            "noroll": "加えて `rule_signal` に、21〜23時 (UTC) に確定する足では入らない判定を足す必要があります。",
            "ldnny": "加えて `rule_signal` に、8〜17時 (UTC) に確定する足でだけ入る判定を足す必要があります。",
            "vix25": "加えてサーバーで VIX を取得・保存し、`rule_signal` に VIX の判定を足す必要があります。",
            "vixjump": "加えてサーバーで VIX を取得・保存し、`rule_signal` に VIX の判定を足す必要があります。"}[f]


SOURCES = {
    "duka_1h": "Dukascopy 1時間足",
    "duka_1d": "Dukascopy 1時間足から作ったロンドン日足",
    "yahoo_1d": "Yahoo 日足",
    "yahoo_1h": "Yahoo 1時間足 (本番と同じ、直近約2.8年)",
}


# ------------------------------------------------------------------- data

def london_days(h: pd.DataFrame, min_hours: int = 3) -> pd.DataFrame:
    """Daily bars per London calendar day (complete at London midnight, like the live daily
    origins) from hourly bars; the Sunday-evening open is part of Monday. ``spread`` is the one
    recorded at the day's last hour."""
    local = h.index.tz_convert("Europe/London")
    day = pd.DatetimeIndex(local.date)
    wd = day.weekday.to_numpy()
    day = day + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")
    g = h.groupby(day)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "spread": g["spread"].last(), "hours": g.size()})
    out.index.name = "date"
    return out[out["hours"] >= min_hours].drop(columns="hours")


def _vix(root: Path | None) -> pd.DataFrame:
    """VIX as known on each date (FRED's publication delay, history.load_vix) and its 5-day change."""
    v = history.load_vix(root).dropna()
    return pd.DataFrame({"vix": v, "jump": v / v.shift(5) - 1})


def _asof(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    known = pd.DatetimeIndex(frame.index).as_unit("ns").asi8          # file dates may load as [us]
    pos = np.searchsorted(known, pd.DatetimeIndex(dates).as_unit("ns").asi8, side="right") - 1
    vals = frame.to_numpy()[np.clip(pos, 0, None)]
    vals[pos < 0] = np.nan
    return pd.DataFrame(vals, columns=frame.columns, index=dates)


def prepare(code: str, df: pd.DataFrame, hourly: bool, vix: pd.DataFrame, start: int | None = None) -> dict:
    """research_trade._prep plus what the filters need, all known at each bar's close."""
    P = _prep(code, df[["open", "high", "low", "close"]], hourly)
    if start is not None:
        P["start"] = start
    t = P["time"]
    per_day = BARS_PER_DAY if hourly else 1
    close_t = t + pd.Timedelta(hours=1) if hourly else t          # daily bars: labelled by their London day
    c = pd.Series(P["c"])
    P["ma200"] = c.rolling(200 * per_day, min_periods=200 * per_day).mean().to_numpy()
    rv = np.log(c).diff().rolling(RV_BARS[hourly], min_periods=RV_BARS[hourly]).std()
    win = RV_YEAR * per_day
    P["rv"] = rv.to_numpy()
    P["rv_q80"] = rv.rolling(win, min_periods=win // 4).quantile(0.8).shift(1).to_numpy()
    P["rv_q50"] = rv.rolling(win, min_periods=win // 4).quantile(0.5).shift(1).to_numpy()
    vx = _asof(vix, pd.DatetimeIndex(close_t.normalize()))
    P["vix"], P["vix_jump"] = vx["vix"].to_numpy(), vx["jump"].to_numpy()
    P["close_hour"] = close_t.hour.to_numpy() if hourly else None
    if "spread" in df.columns:
        sp = df["spread"].to_numpy(float) / P["pip"]
        med = float(np.nanmedian(sp[sp > 0])) if np.any(sp > 0) else 0.0
        P["spread_pips"] = np.clip(np.nan_to_num(sp, nan=med), 0.0, SPREAD_CAP * med)
        P["spread_median"] = med
    else:
        P["spread_pips"] = None
    return P


def load_panel(source: str, root: Path | None = None, vix: pd.DataFrame | None = None,
               hourly_cache: dict | None = None, codes: list[str] | None = None) -> dict:
    vix = _vix(root) if vix is None else vix
    panel = {}
    for code in codes or PAIRS:
        if source in ("duka_1h", "duka_1d"):
            if hourly_cache is not None and code in hourly_cache:
                h = hourly_cache[code]
            else:
                h = history.load_long_hourly(code, root)
                if hourly_cache is not None:
                    hourly_cache[code] = h
            if source == "duka_1h":
                panel[code] = prepare(code, h, True, vix)
            else:
                panel[code] = prepare(code, london_days(h), False, vix, start=20)
        elif source == "yahoo_1d":
            d = history.load_daily(code, root)
            panel[code] = prepare(code, d[d.index >= DAILY_START - pd.Timedelta(days=400)], False, vix)
        elif source == "yahoo_1h":
            panel[code] = prepare(code, history.load_hourly(code, root), True, vix)
        else:
            raise ValueError(source)
    return panel


# ---------------------------------------------------------------- variants

def _allowed(P: dict, f: str, sig: np.ndarray) -> np.ndarray:
    """Whether a signal may be taken at each bar under filter ``f``. A filter whose input is not
    known yet (the first year of a series) does not block."""
    if f == "noroll":
        return ~np.isin(P["close_hour"], (21, 22, 23))
    if f == "ldnny":
        return (P["close_hour"] >= 8) & (P["close_hour"] <= 17)
    if f in ("vol80", "vol50"):
        q = P["rv_q80" if f == "vol80" else "rv_q50"]
        return ~(P["rv"] > q)                                       # NaN compares False: allowed
    if f == "ma200":
        ma = P["ma200"]
        return ~np.isfinite(ma) | (np.sign(P["c"] - ma) == sig)
    if f in ("mom5", "mom20"):
        return signal("mom", {"L": 5 if f == "mom5" else 20}, P) == sig
    if f == "vix25":
        return ~(P["vix"] >= VIX_LEVEL)
    if f == "vixjump":
        return ~(P["vix_jump"] >= VIX_JUMP)
    raise ValueError(f)


def variant_signal(P: dict, key: str, rule: dict, filters: list[str]) -> np.ndarray:
    sig = signal(key, rule, P)
    for f in filters:
        sig = np.where(_allowed(P, f, sig), sig, 0)
    return sig


def run_variant(panel: dict, key: str, rule: dict, filters: list[str], since: pd.Timestamp | None = None) -> pd.DataFrame:
    """All trades of one variant: research_trade.simulate without costs, then each trade's cost is
    the larger of the live cost and the recorded spread (half at entry, half at exit)."""
    rows = []
    for code, P in panel.items():
        sig = variant_signal(P, key, rule, filters)
        trades = simulate(P, sig, rule["sl"], rule["tp"], rule["hold"], 0.0)
        if not trades:
            continue
        i, j, d, gross = (np.array([t[k] for t in trades]) for k in range(4))
        live = COST_PIPS[code]
        sp = P["spread_pips"]
        cost = np.maximum(live, (sp[i] + sp[j]) / 2) if sp is not None else np.full(len(i), live)
        risk = rule["sl"] * P["atr"][i] / P["pip"]
        cum_long, cum_short = np.cumsum(P["acc"][0]), np.cumsum(P["acc"][1])   # the swap simulate adds, bars i+1..j
        swap = np.where(d > 0, cum_long[j] - cum_long[i], cum_short[j] - cum_short[i])
        rows.append(pd.DataFrame({"code": code, "i": i, "j": j, "d": d, "t_in": P["time"][i], "t_out": P["time"][j],
                                  "gross": gross, "swap": swap, "cost": cost, "pips": gross - cost, "pips_live": gross - live,
                                  "R": (gross - cost) / risk, "R_live": (gross - live) / risk}))
    if not rows:
        return pd.DataFrame(columns=["code", "i", "j", "d", "t_in", "t_out", "gross", "swap", "cost", "pips", "pips_live", "R", "R_live"])
    tr = pd.concat(rows, ignore_index=True)
    return tr[tr["t_in"] >= since].reset_index(drop=True) if since is not None else tr


# ------------------------------------------------------------------ metrics

def _monthly_t(x: pd.Series, when: pd.Series) -> float | None:
    m = x.groupby(pd.DatetimeIndex(when).to_period("M")).sum()
    if len(m) < 3:
        return None
    m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"), fill_value=0.0)
    sd = m.std(ddof=1)
    return float(m.mean() / sd * math.sqrt(len(m))) if sd > 0 else None


def summarize(tr: pd.DataFrame, times: dict) -> dict:
    """research_trade.metrics (t from monthly sums of R over all pairs, drawdown in R) plus pip
    totals, the t statistic of monthly pip sums, the pip drawdown and the naive trade-level t."""
    if not len(tr):
        return {"n": 0}
    tuples = list(zip(tr["i"], tr["j"], tr["d"], tr["pips"], tr["R"], tr["code"]))
    out = metrics(tuples, times)
    order = np.argsort(tr["t_out"].to_numpy(), kind="stable")
    eq = np.cumsum(tr["pips"].to_numpy()[order])
    R = tr["R"].to_numpy()
    out.update({"total_pips": float(tr["pips"].sum()), "t_pips": _monthly_t(tr["pips"], tr["t_out"]),
                "maxdd_pips": float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq)),
                "t_naive": float(R.mean() / R.std(ddof=1) * math.sqrt(len(R))) if len(R) > 2 and R.std() > 0 else None,
                "pips_live": float(tr["pips_live"].mean()), "R_live": float(tr["R_live"].mean()),
                "t_live": _monthly_t(tr["R_live"], tr["t_out"]),
                "cost": float(tr["cost"].mean()), "swap": float(tr["swap"].mean())})
    return out


def _block(year: int) -> str:
    if year <= 2005:
        return "〜2005"
    for a, b in ((2006, 2008), (2009, 2011), (2012, 2014), (2015, 2016), (2017, 2019), (2020, 2022)):
        if a <= year <= b:
            return f"{a}〜{b}"
    return "2023〜"


def breakdown(tr: pd.DataFrame, times: dict) -> dict:
    """Per year and per ~3-year block (by entry date), and per pair on each period."""
    if not len(tr):
        return {"year": {}, "block": {}, "pair": {}}
    yr = tr["t_in"].dt.year
    years = {int(y): {"n": int(len(g)), "win": float((g["pips"] > 0).mean()), "pips": float(g["pips"].mean()),
                      "total_pips": float(g["pips"].sum()), "R_sum": float(g["R"].sum())} for y, g in tr.groupby(yr)}
    blocks = {b: summarize(g, times) for b, g in tr.groupby(yr.map(_block), sort=False)}
    pair = {code: {"tune": summarize(g[g["t_in"] < SPLIT], times), "test": summarize(g[g["t_in"] >= SPLIT], times)}
            for code, g in tr.groupby("code", sort=False)}
    return {"year": years, "block": blocks, "pair": pair}


def split(tr: pd.DataFrame, times: dict) -> dict:
    return {"tune": summarize(tr[tr["t_in"] < SPLIT], times), "test": summarize(tr[tr["t_in"] >= SPLIT], times)}


def beats(a: dict, b: dict) -> bool:
    """Higher mean pips after costs and a higher t (monthly R sums) than ``b``."""
    if not a.get("n") or not b.get("n") or a.get("t") is None:
        return False
    return a["pips"] > b["pips"] and a["t"] > (b.get("t") if b.get("t") is not None else -math.inf)


# ------------------------------------------------------------------ search

# the live rules when this study was made (trade.RULES has since adopted the 1h result: carry_mom_vol)
STUDY_RULES = {"1d": {"key": "carry", "thr": 2.0, "sl": 4.0, "tp": 2.0, "hold": 5},
               "1h": {"key": "carry_mom", "thr": 1.0, "L": 120, "sl": 3.0, "tp": 3.0, "hold": 24}}


def _rule_of(tf: str) -> tuple[str, dict]:
    r = STUDY_RULES[tf]
    return r["key"], {k: v for k, v in r.items() if k != "key"}


def search(tf: str, panels: dict[str, dict], primary: str, since: dict, log=print) -> dict:
    """Baseline, the pre-declared single changes and the tune-built combination, on every data
    source in ``panels``; the choice uses the tune period of ``primary`` only."""
    key, base = _rule_of(tf)
    changes = HOURLY_CHANGES if tf == "1h" else DAILY_CHANGES
    times = {src: {c: P["time"] for c, P in pan.items()} for src, pan in panels.items()}
    trades: dict[str, dict] = {src: {} for src in panels}
    res: dict[str, dict] = {}

    def evaluate(vid, group, label, over, filters):
        rule = {**base, **over}
        row = {"id": vid, "group": group, "label": label, "rule": rule, "filters": filters, "by_source": {}}
        for src, pan in panels.items():
            tr = run_variant(pan, key, rule, filters, since.get(src))
            trades[src][vid] = tr
            row["by_source"][src] = split(tr, times[src])
        res[vid] = row
        p = row["by_source"][primary]
        if log:
            log(f"{tf} {vid:8s} tune n={p['tune'].get('n')} pips={_f(p['tune'].get('pips'), 2, True)} "
                f"t={_f(p['tune'].get('t'))} | test n={p['test'].get('n')} pips={_f(p['test'].get('pips'), 2, True)} "
                f"t={_f(p['test'].get('t'))}")

    evaluate("base", "base", "現行ルール", {}, [])
    for vid, group, label, over, filters in changes:
        evaluate(vid, group, label, over, filters)
    base_tune = res["base"]["by_source"][primary]["tune"]

    def qualifies(vid):
        tu = res[vid]["by_source"][primary]["tune"]
        return tu.get("n", 0) >= MIN_TUNE_TRADES and beats(tu, base_tune)

    # combination: the best single change of each group among those beating the current rule on
    # the tune period, and of those the two groups with the highest tune t (no more, to limit fitting)
    picked: dict[str, str] = {}
    for vid, group, *_ in changes:
        if qualifies(vid) and (group not in picked or res[vid]["by_source"][primary]["tune"]["t"]
                               > res[picked[group]]["by_source"][primary]["tune"]["t"]):
            picked[group] = vid
    top = sorted(picked.values(), key=lambda v: -res[v]["by_source"][primary]["tune"]["t"])[:2]
    combo = None
    if len(top) == 2:
        over, filters = {}, []
        for vid in top:
            c = next(c for c in changes if c[0] == vid)
            over.update(c[3])
            filters += c[4]
        combo = "combo"
        evaluate(combo, "combo", "組み合わせ: " + " + ".join(res[v]["label"] for v in top), over, filters)
    # the best candidate: highest tune t among variants that beat the current rule on the tune period
    cands = [v for v in res if v != "base" and qualifies(v)]
    best = max(cands, key=lambda v: res[v]["by_source"][primary]["tune"]["t"]) if cands else None
    verdict = judge(best, res, trades, times, list(panels)) if best else None
    if verdict and best == "combo":
        # after the fact, for reading only: each part of the combination through the same test
        verdict["parts"] = {v: judge(v, res, trades, times, list(panels)) for v in top}
    tested = len(res) - 1
    beat_test = sum(1 for v in res if v != "base"
                    and beats(res[v]["by_source"][primary]["test"], res["base"]["by_source"][primary]["test"]))
    return {"tf": tf, "key": key, "base_rule": base, "primary": primary, "variants": res, "picked": picked,
            "combined": top, "combo": combo, "best": best, "verdict": verdict, "n_variants": tested, "n_beat_tune": len(cands),
            "n_beat_test": beat_test,
            "base_detail": {src: breakdown(trades[src]["base"], times[src]) for src in panels},
            "best_detail": {src: breakdown(trades[src][best], times[src]) for src in panels} if best else None,
            "base_monthly": {src: _monthly_curve(trades[src]["base"]) for src in panels}}


def judge(best: str, res: dict, trades: dict, times: dict, sources: list[str]) -> dict:
    """The recommendation test: both periods on every source, at least MIN_PAIRS_BETTER pairs
    better on the test period, and any one pair left out still better on the test period."""
    out = {"sources": {}}
    ok = True
    for src in sources:
        b, c = res["base"]["by_source"][src], res[best]["by_source"][src]
        tb, tc = trades[src]["base"], trades[src][best]
        tb_te, tc_te = tb[tb["t_in"] >= SPLIT], tc[tc["t_in"] >= SPLIT]
        pairs_better, lopo = [], {}
        traded = sorted(set(tb_te["code"]) | set(tc_te["code"]), key=list(PAIRS).index)
        for code in traded:
            pb, pc = tb_te[tb_te["code"] == code], tc_te[tc_te["code"] == code]
            if len(pc) and len(pb) and pc["pips"].mean() > pb["pips"].mean():
                pairs_better.append(code)
            lopo[code] = beats(summarize(tc_te[tc_te["code"] != code], times[src]),
                               summarize(tb_te[tb_te["code"] != code], times[src]))
        s = {"tune": beats(c["tune"], b["tune"]), "test": beats(c["test"], b["test"]),
             "pairs_better": pairs_better, "pairs_traded": traded, "lopo": lopo}
        s["pass"] = s["tune"] and s["test"] and len(pairs_better) >= MIN_PAIRS_BETTER and all(lopo.values())
        out["sources"][src] = s
        ok = ok and s["pass"]
    out["recommend"] = ok
    return out


def _monthly_curve(tr: pd.DataFrame) -> dict:
    if not len(tr):
        return {}
    m = tr.groupby(pd.DatetimeIndex(tr["t_out"]).to_period("M"))[["pips", "R"]].sum()
    return {"month": [str(p) for p in m.index], "cum_pips": np.round(m["pips"].cumsum(), 1).tolist(),
            "cum_R": np.round(m["R"].cumsum(), 2).tolist()}


# ------------------------------------------------------------------ checks

def check_signals(panel: dict, tf: str) -> dict:
    """research_trade.signal (used here) against trade.rule_signal (the live function), bar by bar."""
    key, base = _rule_of(tf)
    vol = RULES[tf].get("vol")       # the adopted 1h rule: this study's rule + the vol80 filter
    out = {}
    for code, P in panel.items():
        research = signal(key, base, P)
        c, diff = P["c"], P["diff"]
        if vol:
            research = np.where(_allowed(P, "vol80", research), research, 0)
            allow = vol_allowed(c, **vol)
            live = np.array([rule_signal(tf, c[: i + 1], float(diff[i]), bool(allow[i])) for i in range(len(c))])
        else:
            live = np.array([rule_signal(tf, c[: i + 1], float(diff[i])) for i in range(len(c))])
        out[code] = {"bars": int(len(c)), "mismatch": int(np.sum(live != research)), "active": int(np.sum(research != 0))}
    return out


def _rates_item(root: Path | None, since: str) -> dict:
    series = {}
    for items in history.RATE_SERIES.values():
        for sid, freq in items:
            s = history.load_fred(sid, root)
            s = s[s.index >= pd.Timestamp(since) - pd.DateOffset(months=4 if freq == "m" else 2)]
            series[sid] = [[d.strftime("%Y-%m-%d"), float(v)] for d, v in s.items()]
    return {"series": series}


def check_live_backtest(root: Path | None = None, codes: list[str] | None = None) -> dict:
    """trade.backtest (the live replay, its own loop) against research_trade.simulate on Yahoo's
    hourly bars: the same entries and exits, pips within the swap approximation."""
    from .engine import TIMEFRAMES
    from .trade import backtest
    tf = TIMEFRAMES["1h"]
    key, base = _rule_of("1h")
    vix = _vix(root)
    out = {}
    for code in codes or PAIRS:
        pair = PAIRS[code]
        bars = history.load_hourly(code, root)
        item = _rates_item(root, str(bars.index[0].date()))
        live = backtest(tf, pair, bars, item)
        P = prepare(code, bars, True, vix)
        res = simulate(P, signal(key, base, P), base["sl"], base["tp"], base["hold"], COST_PIPS[code])
        ends = P["time"] + pd.Timedelta(hours=1)
        mine = {(str(ends[i]), str(ends[j])): p for i, j, _d, p, _r in res}
        theirs = {(t["origin"][:19].replace("T", " "), t["t"][:19].replace("T", " ")): t["pips"] for t in live}
        both = set(mine) & set(theirs)
        diff = [abs(mine[k] - theirs[k]) for k in both]
        out[code] = {"live": len(theirs), "research": len(mine), "same": len(both),
                     "max_pip_diff": float(max(diff)) if diff else None,
                     "live_mean": float(np.mean(list(theirs.values()))) if theirs else None,
                     "research_mean": float(np.mean(list(mine.values()))) if mine else None}
    return out


def yahoo_close_check(hourly: dict, root: Path | None = None) -> dict:
    """Which London-midnight price Yahoo's daily close is: the one ending that day, or the one
    starting it (the previous day's close), before and from 2011."""
    out = {}
    for code, h in hourly.items():
        ld = london_days(h)
        ld.index = pd.DatetimeIndex(ld.index).as_unit("ns")
        y = history.load_daily(code, root)
        y.index = pd.DatetimeIndex(y.index).as_unit("ns")
        row = {}
        for name, (a, b) in (("to2010", (2000, 2010)), ("from2011", (2011, 2100))):
            yy = y[(y.index.year >= a) & (y.index.year <= b)]["close"]
            pip = PAIRS[code].pip
            row[name] = {"same_day": float((yy - ld["close"].reindex(yy.index)).abs().median() / pip),
                         "prev_day": float((yy - ld["close"].shift(1).reindex(yy.index)).abs().median() / pip)}
        out[code] = row
    return out


def reproduce(old: dict, res: dict, yahoo_1h: dict) -> dict:
    """The current rules' numbers in research/trade.json, recomputed here: the daily rule on Yahoo's
    daily bars (2017 split) and the hourly rule on Yahoo's hourly bars (split at 60 % of the bars)."""
    out = {}
    now = res["1d"]["variants"]["base"]["by_source"]["yahoo_1d"]
    was = old["1d"]["families"][RULES["1d"]["key"]]["chosen"]
    out["1d"] = {"was": {p: [was[p]["n"], was[p]["pips"]] for p in ("tune", "test")},
                 "now": {p: [now[p].get("n"), now[p].get("pips")] for p in ("tune", "test")}}
    key, base = _rule_of("1h")
    tr = run_variant(yahoo_1h, key, base, [])
    all_t = np.sort(np.concatenate([P["time"].asi8 for P in yahoo_1h.values()]))
    cut = pd.Timestamp(all_t[int(len(all_t) * HOURLY_TUNE_SHARE)])
    was = old["1h"]["families"][key]["chosen"]
    out["1h"] = {"was": {p: [was[p]["n"], was[p]["pips"]] for p in ("tune", "test")},
                 "now": {"tune": [int((tr["t_in"] < cut).sum()), float(tr.loc[tr["t_in"] < cut, "pips"].mean())],
                         "test": [int((tr["t_in"] >= cut).sum()), float(tr.loc[tr["t_in"] >= cut, "pips"].mean())]}}
    for tf in ("1d", "1h"):
        w, n = out[tf]["was"], out[tf]["now"]
        out[tf]["match"] = bool(all(w[p][0] == n[p][0] and abs(w[p][1] - n[p][1]) < 1e-6 for p in ("tune", "test")))
    return out


def overlap_check(duka: dict, yahoo: dict) -> dict:
    """The current hourly rule on Dukascopy and on Yahoo bars over the months both cover
    (live cost only, so only the price source differs)."""
    key, base = _rule_of("1h")
    t0 = max(P["time"][0] for P in yahoo.values()) + pd.Timedelta(days=30)   # after the momentum warm-up
    t1 = min(P["time"][-1] for P in duka.values()) - pd.Timedelta(days=3)
    out = {"from": str(t0.date()), "to": str(t1.date())}
    for name, pan in (("duka", duka), ("yahoo", yahoo)):
        tr = run_variant(pan, key, base, [])
        tr = tr[(tr["t_in"] >= t0) & (tr["t_out"] <= t1)]
        out[name] = {"n": int(len(tr)), "pips_live": float(tr["pips_live"].mean()) if len(tr) else None,
                     "win": float((tr["pips_live"] > 0).mean()) if len(tr) else None,
                     "entries": set(zip(tr["code"], tr["t_in"]))}
    both = out["duka"].pop("entries") & out["yahoo"].pop("entries")
    out["same_entries"] = len(both)
    closes = []
    for code in duka:
        a = pd.Series(duka[code]["c"], index=duka[code]["time"])
        b = pd.Series(yahoo[code]["c"], index=yahoo[code]["time"])
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        j = j[(j.index >= t0) & (j.index <= t1)]
        closes.append({"code": code, "hours": int(len(j)),
                       "median_abs_diff_pips": float((j[0] - j[1]).abs().median() / PAIRS[code].pip)})
    out["close_diff"] = closes
    return out


# ------------------------------------------------------------------- report

def _row(s: dict) -> str:
    if not s.get("n"):
        return "0 / — / — / — / —"
    return (f"{s['n']} / {s['win']:.0%} / {_f(s['pips'], 1, True)} / {_f(s['total_pips'], 0, True)} / "
            f"{_f(s.get('t'))}")


def _rule_text(tf: str, rule: dict, filters: list[str] | None = None) -> str:
    p = {k: rule[k] for k in ("thr", "L", "sl", "tp", "hold") if k in rule}
    t = _params_text(p)
    return t + (" + " + "・".join(filters) if filters else "")


def _stats_table(rows: list[tuple[str, dict]]) -> list[str]:
    L = ["| データ・期間 | 取引数 | 勝率 | 平均pips | 合計pips | 平均R | PF | t (月ごと) | t (取引ごと、参考) | 最大DD (R) | "
         "最大DD (pips) | 本番コストだけ: 平均pips / t | 1回の平均: コスト / スワップ (pips) |", "|---|" + "---|" * 12]
    for name, s in rows:
        if not s.get("n"):
            L.append(f"| {name} | 0 |" + " — |" * 11)
            continue
        L.append(f"| {name} | {s['n']} | {s['win']:.0%} | {_f(s['pips'], 1, True)} | {_f(s['total_pips'], 0, True)} | "
                 f"{_f(s['R'], 3, True)} | {_f(s.get('pf'))} | {_f(s.get('t'))} | {_f(s.get('t_naive'))} | "
                 f"{_f(s['maxdd_R'], 1)} | {_f(s['maxdd_pips'], 0)} | {_f(s['pips_live'], 1, True)} / {_f(s.get('t_live'))} | {_f(s['cost'], 2)} / {_f(s['swap'], 2, True)} |")
    return L


def _block_table(details: dict[str, dict]) -> list[str]:
    srcs = list(details)
    blocks = []
    for d in details.values():
        blocks += [b for b in d["block"] if b not in blocks]
    blocks.sort(key=lambda b: (b != "〜2005", b))
    L = ["| 期間 (エントリー) |" + "".join(f" {SOURCES[s]}: 取引数 / 勝率 / 平均pips / 合計R / t |" for s in srcs),
         "|---|" + "---|" * len(srcs)]
    for blk in blocks:
        cells = []
        for src in srcs:
            x = details[src]["block"].get(blk)
            cells.append(f"{x['n']} / {x['win']:.0%} / {_f(x['pips'], 1, True)} / {_f(x['R'] * x['n'], 1, True)} / {_f(x.get('t'))}"
                         if x and x.get("n") else "—")
        L.append(f"| {blk} | " + " | ".join(cells) + " |")
    return L


def _pair_table(d: dict) -> list[str]:
    L = ["| ペア | 調整: 取引数 / 勝率 / 平均pips / 合計pips / t | 検証: 取引数 / 勝率 / 平均pips / 合計pips / t |", "|---|---|---|"]
    for code in PAIRS:
        s = d["pair"].get(code, {"tune": {"n": 0}, "test": {"n": 0}})
        L.append(f"| {code} | {_row(s['tune'])} | {_row(s['test'])} |")
    return L


def _base_section(r: dict, sources: list[str], title: str) -> list[str]:
    rows = []
    for src in sources:
        b = r["variants"]["base"]["by_source"][src]
        rows += [(f"{SOURCES[src]}、調整期間", b["tune"]), (f"{SOURCES[src]}、検証期間", b["test"])]
    L = [f"### {title}: 現行ルール ({_rule_text(r['tf'], r['base_rule'])})", ""] + _stats_table(rows)
    L += ["", "t (月ごと) は月ごとの損益 (全ペア合計の R) から計算した t 値で、採否にはこれを使います。t (取引ごと) は"
          "同じ時期に複数のペアで持つ取引を独立とみなすため、過大になりやすい参考値です。", "",
          "#### 3年ごとの成績", ""] + _block_table(r["base_detail"])
    src = r["primary"]
    d = r["base_detail"][src]
    L += ["", f"#### 年ごとの成績 ({SOURCES[src]})", "", "| 年 | 取引数 | 勝率 | 平均pips | 合計pips | 合計R |", "|---|---|---|---|---|---|"]
    for y, s in d["year"].items():
        L.append(f"| {y} | {s['n']} | {s['win']:.0%} | {_f(s['pips'], 1, True)} | {_f(s['total_pips'], 0, True)} | "
                 f"{_f(s['R_sum'], 1, True)} |")
    for src in sources:
        L += ["", f"#### 通貨ペアごとの成績 ({SOURCES[src]})", ""] + _pair_table(r["base_detail"][src])
    L.append("")
    return L


def _best_section(r: dict, sources: list[str]) -> list[str]:
    if not r["best"]:
        return []
    v = r["variants"][r["best"]]
    rows = []
    for src in sources:
        rows += [(f"{SOURCES[src]}、調整期間", v["by_source"][src]["tune"]), (f"{SOURCES[src]}、検証期間", v["by_source"][src]["test"])]
    L = [f"### 候補「{v['label']}」の詳細", ""] + _stats_table(rows) + ["", "#### 3年ごとの成績", ""] + _block_table(r["best_detail"])
    for src in sources:
        L += ["", f"#### 通貨ペアごとの成績 ({SOURCES[src]})", ""] + _pair_table(r["best_detail"][src])
    return L + [""]


def _search_section(r: dict, sources: list[str]) -> list[str]:
    prim = r["primary"]
    others = [s for s in sources if s != prim]
    head = "| 変更 | 調整: 取引数 / 平均pips / t | 検証: 取引数 / 平均pips / t | 本番コストだけ: 調整 / 検証 (平均pips / t) |" + "".join(
        f" {SOURCES[s]}: 調整 / 検証 (平均pips / t) |" for s in others) + " 調整で上回る | 検証で上回る |"
    L = [head, "|" + "---|" * (6 + len(others))]
    bp = r["variants"]["base"]["by_source"]
    for vid, v in r["variants"].items():
        p = v["by_source"][prim]
        cells = [f"{v['label']}" + (" **(候補)**" if vid == r["best"] else ""),
                 f"{p['tune'].get('n', 0)} / {_f(p['tune'].get('pips'), 1, True)} / {_f(p['tune'].get('t'))}",
                 f"{p['test'].get('n', 0)} / {_f(p['test'].get('pips'), 1, True)} / {_f(p['test'].get('t'))}",
                 f"{_f(p['tune'].get('pips_live'), 1, True)} / {_f(p['tune'].get('t_live'))} ・ "
                 f"{_f(p['test'].get('pips_live'), 1, True)} / {_f(p['test'].get('t_live'))}"]
        for s in others:
            q = v["by_source"][s]
            cells.append(f"{_f(q['tune'].get('pips'), 1, True)} / {_f(q['tune'].get('t'))} ・ "
                         f"{_f(q['test'].get('pips'), 1, True)} / {_f(q['test'].get('t'))}")
        if vid == "base":
            cells += ["—", "—"]
        else:
            cells += ["○" if beats(p["tune"], bp[prim]["tune"]) else "×",
                      "○" if beats(p["test"], bp[prim]["test"]) else "×"]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    return L


def _verdict_lines(r: dict, tfn: str) -> list[str]:
    if not r["best"]:
        return [f"- {tfn}: 調整期間で現行ルールを平均pipsと t の両方で上回る変更はありませんでした。現行ルールのままとします。"]
    v = r["variants"][r["best"]]
    ver = r["verdict"]
    L = [f"- {tfn}: 調整期間で最もよかった変更は「{v['label']}」({_rule_text(r['tf'], v['rule'], v['filters'])})。"]
    for src, s in ver["sources"].items():
        b, c = r["variants"]["base"]["by_source"][src], v["by_source"][src]
        L.append(f"  - {SOURCES[src]}: 調整 {_f(b['tune'].get('pips'), 1, True)} → {_f(c['tune'].get('pips'), 1, True)} pips "
                 f"(t {_f(b['tune'].get('t'))} → {_f(c['tune'].get('t'))})、検証 {_f(b['test'].get('pips'), 1, True)} → "
                 f"{_f(c['test'].get('pips'), 1, True)} pips (t {_f(b['test'].get('t'))} → {_f(c['test'].get('t'))})。"
                 f"検証期間で平均が改善したペア {len(s['pairs_better'])}/{len(s['pairs_traded'])} ({'・'.join(s['pairs_better']) or 'なし'})、どの1ペアを除いても検証期間で上回るか: "
                 f"{'はい' if all(s['lopo'].values()) else 'いいえ (' + '、'.join(k for k, ok in s['lopo'].items() if not ok) + ' を除くと現行ルール以下)'}。"
                 f"判定: {'合格' if s['pass'] else '不合格'}")
    L.append(f"  - **{'変更を推奨' if ver['recommend'] else '推奨しない (現行ルールのまま)'}**")
    for part, pv in (ver.get("parts") or {}).items():
        cells = [f"{SOURCES[src]}: 調整 {'○' if s['tune'] else '×'}・検証 {'○' if s['test'] else '×'}・改善したペア "
                 f"{len(s['pairs_better'])}/{len(s['pairs_traded'])}・1ペア除外 {'○' if all(s['lopo'].values()) else '×'}"
                 for src, s in pv["sources"].items()]
        L.append(f"  - 補足 (事後の確認、採否には使っていません): 組み合わせの一部「{r['variants'][part]['label']}」だけの場合 — "
                 + "、".join(cells) + f" → 推奨の条件を{'満たす' if pv['recommend'] else '満たさない'}。")
    if ver["recommend"]:
        changed = {k: v["rule"][k] for k in v["rule"] if v["rule"][k] != r["base_rule"].get(k)}
        L.append(f"  - `aifx/trade.py` の `RULES[\"{r['tf']}\"]` を " + "、".join(f"`{k}`: {r['base_rule'].get(k)} → {x}" for k, x in changed.items())
                 + " に変更。" + "".join(_filter_impl(f, r["tf"] == "1h") for f in v["filters"]))
    return L


def report(res: dict) -> str:
    h, d = res["1h"], res["1d"]
    L = ["# 売買シグナルの再検証 (約20年分のデータ)", "",
         "本番の売買プラン ([research/trade.md](trade.md)) の2つのルールを、はるかに長い期間で同じコードのまま検証し直し、"
         "少数の改善案を事前に決めた手順で試しました。売買の判定・決済・コスト・スワップは `aifx/trade.py` と "
         "`aifx/research_trade.py` の関数をそのまま使っています。", "",
         "## 結論", ""]
    L += res["conclusion"]
    L += ["", "## データと方法", "",
          f"- 1時間足: Dukascopy の1時間足 (買値・売値の中間、各1時間の終わりの売値−買値の差を記録) {res['data']['duka_1h']}。"
          "本番が使う Yahoo の1時間足は直近約2.8年しかありません。",
          f"- 日足: Yahoo の日足 ({res['data']['yahoo_1d']}、2002年から) と、Dukascopy の1時間足からロンドン時間の1日ごとに作った日足 "
          f"({res['data']['duka_1d']})。本番の日足の売買プランはロンドン時間の0時に発行され、1時間足の価格で決済するので、"
          "ロンドン日足の方が本番の取引に近く、Yahoo の日足は2011年から終値が「その日の始めの価格」になっている問題があります (下の「確認」)。"
          "日足のルールは両方で確かめ、改善案の選択はロンドン日足の調整期間で行いました。",
          "- 期間: 調整期間 = 2016年末まで、検証期間 = 2017年1月から最後まで (エントリーの日付で分割)。",
          "- エントリーは足の終値、損切り・利確は ATR (14本) の倍数、最長保有本数で時間決済、1本の足の中で損切りと利確の両方に届いたら損切り、"
          "損切り・利確の先で始まった足はその始値で決済 (`research_trade.simulate`)。1ペアにつき同時に1つのポジションだけ。",
          "- コスト: 1回の取引ごとに、本番の往復コスト (" + "、".join(f"{c} {v}" for c, v in COST_PIPS.items()) +
          " pips) と、Dukascopy が記録した売値−買値の差 (エントリーの足の終わりで半分、決済の足の終わりで半分) の大きい方。"
          f"記録の誤りを除くため、差はペアの中央値の{SPREAD_CAP:.0f}倍で頭打ちにしました。Yahoo の日足は差の記録がないので本番のコストだけです。"
          "Dukascopy の1時間足の終わりの差の中央値 (pips、調整期間 / 検証期間): " +
          "、".join(f"{c} {_f(v['tune'], 2)} / {_f(v['test'], 2)}" for c, v in res["spread_median_pips"].items()) + "。",
          f"- スワップ: その時点で知り得た短期金利 (FRED、公表の遅れを考慮) の差から、業者の取り分 {SWAP_MARKUP}% を引いたものを保有時間に比例して加算。",
          "- 統計: 平均pips はコスト・スワップ込みの1回あたり。t は月ごとの損益 (全ペア合計の R、R は損切り幅を1とした損益) の t 値で、"
          "同じ時期に複数のペアで持つ取引 (円安・円高で一緒に動く) を独立に数えないためのものです。最大DD は全ペア合計の損益の最大の落ち込み。",
          "- VIX は FRED の終値を2日遅らせて使いました (公表の遅れ)。実現ボラティリティ・200日移動平均・VIX は各足の終わりまでのデータだけで計算。", "",
          "### 事前に決めた改善案と採否の基準", "",
          "結果を見る前に次の変更を1つずつ試すと決め、選択には調整期間だけを使いました。",
          "",
          "- 1時間足 (" + str(len(HOURLY_CHANGES)) + "通り): " + "、".join(c[2] for c in HOURLY_CHANGES),
          "- 日足 (" + str(len(DAILY_CHANGES)) + "通り): " + "、".join(c[2] for c in DAILY_CHANGES),
          "- 組み合わせ (1通りだけ): 種類 (金利差の基準・流れの期間・損切り・利確・保有期間・時間帯・値動きの大きさ・トレンド・リスクオフ) ごとに、"
          "調整期間で現行ルールを上回った変更のうち t が最も高いものを選び、その中で調整期間の t が高い2種類をまとめて1つの案にする "
          "(多く組み合わせるほど調整期間に合わせすぎるため2種類まで)。",
          f"- 候補: 調整期間の取引が {MIN_TUNE_TRADES} 回以上で、調整期間に現行ルールを平均pips と t の両方で上回った案のうち、調整期間の t が最も高いもの。",
          f"- 推奨の条件: 候補が検証期間でも平均pips と t の両方で現行ルールを上回り、検証期間で平均が改善したペアが {MIN_PAIRS_BETTER}/7 以上、"
          "どの1ペアを除いても検証期間で現行ルールを上回ること (1ペアだけの偶然でないこと)。日足は Yahoo 日足・ロンドン日足の両方で満たすこと。",
          "- 「1時間足と日足のシグナルの一致」は、1時間足では「金利差2%以上」(日足ルールの条件) と同じになり、"
          "日足では「過去5日間 (≒120時間) の流れも同じ向き」と同じになるので、それぞれその案として数えています。",
          f"- 試した案の数: 1時間足 {h['n_variants']} 通り、日足 {d['n_variants']} 通り (組み合わせを含む、現行ルールを除く)。", ""]
    L += ["## 1時間足: 金利差 + 5日間の流れ", ""]
    L += _base_section(h, list(h["variants"]["base"]["by_source"]), "1時間足")
    L += ["### 改善案 (Dukascopy 1時間足)", "", "各列: 取引数 / 1回平均pips (コスト・スワップ込み) / t。○ = 平均pips と t の両方で現行ルールを上回る。", ""]
    L += _search_section(h, list(h["variants"]["base"]["by_source"]))
    L += [f"調整期間で上回った案: {h['n_beat_tune']} / {h['n_variants']}、検証期間で上回った案: {h['n_beat_test']} / {h['n_variants']}。", ""]
    L += _best_section(h, list(h["variants"]["base"]["by_source"]))
    L += ["## 日足: 金利差", ""]
    L += _base_section(d, list(d["variants"]["base"]["by_source"]), "日足")
    L += ["### 改善案 (ロンドン日足で選択、Yahoo 日足も併記)", "",
          "各列: 取引数 / 1回平均pips (コスト・スワップ込み) / t。○ = 平均pips と t の両方で現行ルールを上回る (ロンドン日足)。", ""]
    L += _search_section(d, list(d["variants"]["base"]["by_source"]))
    L += [f"調整期間で上回った案: {d['n_beat_tune']} / {d['n_variants']}、検証期間で上回った案: {d['n_beat_test']} / {d['n_variants']}。", ""]
    L += _best_section(d, list(d["variants"]["base"]["by_source"]))
    L += ["## 推奨", ""]
    L += _verdict_lines(h, "1時間足") + _verdict_lines(d, "日足")
    L += ["", "## 確認", ""]
    L += res["check_lines"]
    L += ["", "## 注意", "",
          "- Dukascopy の価格は1社の気配値で、国内の業者の価格・約定とは少し違います。スプレッドは各1時間の終わりの値で、"
          "損切りが1時間の途中で約定するときの実際の差とは異なります。",
          "- スワップは金利差 (3か月物・翌日物の金利) から一定の取り分を引いた近似で、業者ごとの実際のスワップや水曜日の3日分の付与は再現していません。",
          "- 検証期間は1つの時期です (2017年以降)。調整期間で選んだ案が検証期間でも良いかは、偶然にも左右されます。",
          "- 現行の1時間足ルールは Yahoo の1時間足 (2023年12月〜2025年8月) で選ばれたので、この検証期間の一部はすでに選択に使われています。"
          "検証期間の現行ルールの成績はその分よく見えている可能性があり、改善案との比較は現行ルールに有利です。",
          "- 改善案の一覧、組み合わせの作り方、候補の選び方と推奨の条件は、Dukascopy のデータで結果を見る前に決めました "
          "(プログラムの動作確認は Yahoo のデータで行いました)。「補足 (事後の確認)」の行は結果を見た後に加えたもので、採否には使っていません。", ""]
    return "\n".join(L)


def _holds(tu: dict, te: dict, key: str = "") -> str:
    pk, tk = ("pips_live", "t_live") if key == "live" else ("pips", "t")
    pos = (tu.get(pk) or 0) > 0 and (te.get(pk) or 0) > 0
    if pos and (tu.get(tk) or 0) >= 2 and (te.get(tk) or 0) >= 2:
        return "両方の期間でプラス、t ≥ 2"
    if pos:
        return "両方の期間でプラスだが t < 2 の期間あり"
    if (te.get(pk) or 0) > 0:
        return "調整期間はマイナス、検証期間はプラス"
    if (tu.get(pk) or 0) > 0:
        return "調整期間はプラス、検証期間はマイナス"
    return "両方の期間でマイナス"


def _tier(t: str | None) -> str:
    return {"strong": "「採用」(両方の期間で t ≥ 2)", "weak": "「参考 (弱)」(両方でプラス、検証期間で t ≥ 2)"}.get(t, "「表示しない」")


def _summary(r: dict, name: str) -> str:
    prim = r["primary"]
    b = r["variants"]["base"]["by_source"][prim]
    txt = f"{name}: 現行ルールは20年のデータ ({SOURCES[prim]}、コスト込み) で{_holds(b['tune'], b['test'])} (t = {_f(b['tune'].get('t'))} / {_f(b['test'].get('t'))})、"
    txt += f"表示の基準では{_tier(classify(b))}。"
    if r["verdict"] and r["verdict"]["recommend"]:
        c = r["variants"][r["best"]]["by_source"][prim]
        txt += (f"事前に決めた手順で選んだ「{r['variants'][r['best']]['label']}」は両方の期間で現行ルールを上回り、推奨の条件を満たしました"
                f" (t = {_f(c['tune'].get('t'))} / {_f(c['test'].get('t'))})")
        txt += "が、検証期間の t は 2 に届いていません。" if (c["test"].get("t") or 0) < 2 else "。"
    else:
        txt += "改善案で推奨の条件を満たすものはなく、現行ルールのままとします。"
    return txt


def _conclusion(res: dict) -> list[str]:
    out = ["- 要約 (t は調整期間 / 検証期間):", "  - " + _summary(res["1h"], "1時間足"), "  - " + _summary(res["1d"], "日足")]
    for r, tfn in ((res["1h"], "1時間足ルール (金利差 + 5日間の流れ)"), (res["1d"], "日足ルール (金利差)")):
        out.append(f"- **{tfn}**")
        for src, b in r["variants"]["base"]["by_source"].items():
            tu, te = b["tune"], b["test"]
            blocks = r["base_detail"][src]["block"]
            pos = sum(1 for x in blocks.values() if x.get("n") and x["pips"] > 0)
            out.append(f"  - {SOURCES[src]}: 調整期間 (〜2016) {tu.get('n', 0)} 回・1回平均 {_f(tu.get('pips'), 1, True)} pips・t = {_f(tu.get('t'))}、"
                       f"検証期間 (2017〜) {te.get('n', 0)} 回・1回平均 {_f(te.get('pips'), 1, True)} pips・t = {_f(te.get('t'))} "
                       f"→ {_holds(tu, te)}。本番のコストだけなら 調整 {_f(tu.get('pips_live'), 1, True)} pips (t = {_f(tu.get('t_live'))})、"
                       f"検証 {_f(te.get('pips_live'), 1, True)} pips (t = {_f(te.get('t_live'))}) → {_holds(tu, te, 'live')}。"
                       f"3年ごとの区切りでは {len(blocks)} 期間中 {pos} 期間でプラス。")
        rec = r["verdict"]["recommend"] if r["verdict"] else False
        shown = [("現行ルール", "base")] + ([("候補", r["best"])] if r["best"] else [])
        cells = []
        for name, vid in shown:
            b = r["variants"][vid]["by_source"][r["primary"]]
            live = {p: {"t": b[p].get("t_live"), "R": b[p].get("R_live")} for p in ("tune", "test")}
            cells.append(f"{name}は{_tier(classify(b))} (本番のコストだけなら{_tier(classify(live))})")
        out.append(f"  - research/trade.md と同じ表示の基準 ({SOURCES[r['primary']]}): " + "、".join(cells) + "。")
        if r["best"]:
            v = r["variants"][r["best"]]
            out.append(f"  - 改善案: 調整期間で最もよかった「{v['label']}」は" +
                       ("検証期間と他の条件も満たしたので、変更を推奨します。" if rec else "推奨の条件を満たさなかったので、変更しません。"))
        else:
            out.append("  - 改善案: 調整期間で現行ルールを上回るものがなく、変更しません。")
    return out


def _check_lines(res: dict) -> list[str]:
    c = res["checks"]
    L = []
    sig = c["signals"]
    for tf, rows in sig.items():
        mm = sum(r["mismatch"] for r in rows.values())
        bars = sum(r["bars"] for r in rows.values())
        L.append(f"- 本番の判定関数 (`trade.rule_signal`) と、ここで使った `research_trade.signal` の判定 ({tf}、{SOURCES[c['signal_source'][tf]]}): "
                 f"{bars:,} 本中 {mm} 本が不一致。")
    lb = c.get("live_backtest")
    if lb:
        same = sum(r["same"] for r in lb.values())
        n_live = sum(r["live"] for r in lb.values())
        n_res = sum(r["research"] for r in lb.values())
        mx = max((r["max_pip_diff"] or 0) for r in lb.values())
        L.append(f"- 本番のバックテスト (`trade.backtest`、Yahoo 1時間足) と `research_trade.simulate`: 取引数 {n_live} と {n_res}、"
                 f"エントリーと決済の時刻が同じ取引 {same}、1回の pips の差は最大 {mx:.2f} (スワップの計算の違い)。")
    rp = c.get("reproduce")
    if rp:
        for tf, name in (("1d", "日足ルールを Yahoo 日足"), ("1h", "1時間足ルールを Yahoo 1時間足 (前半6割で分割)")):
            n = rp[tf]["now"]
            L.append(f"- 現行の{name}で計算し直すと、research/trade.md の数値と{'一致' if rp[tf]['match'] else '不一致'} "
                     f"(調整 {n['tune'][0]} 回・平均 {_f(n['tune'][1], 2, True)} pips、検証 {n['test'][0]} 回・平均 {_f(n['test'][1], 2, True)} pips)。")
    yc = c.get("yahoo_close")
    if yc:
        m = {k: float(np.median([r[k.split("|")[0]][k.split("|")[1]] for r in yc.values()]))
             for k in ("to2010|same_day", "to2010|prev_day", "from2011|same_day", "from2011|prev_day")}
        L.append("- Yahoo の日足の終値と、Dukascopy から作ったロンドン日足の終値の差 (中央値、pips、ペアの中央値): "
                 f"2010年まではその日の終値と {m['to2010|same_day']:.1f}、前日の終値と {m['to2010|prev_day']:.1f}。"
                 f"2011年からはその日の終値と {m['from2011|same_day']:.1f}、前日の終値と {m['from2011|prev_day']:.1f}。"
                 "2011年からの Yahoo の日足の終値は、その日の始め (前日のロンドン0時) の価格です。そのため Yahoo の日足での検証は、"
                 "エントリーした日の値動きを損切り・利確の判定に使えていません。")
    ov = c.get("overlap")
    if ov:
        med = ", ".join(f"{x['code']} {x['median_abs_diff_pips']:.1f}" for x in ov["close_diff"])
        L.append(f"- 同じ期間 ({ov['from']}〜{ov['to']}) の1時間足ルールを Dukascopy と Yahoo で比べると (本番のコストだけ): "
                 f"Dukascopy {ov['duka']['n']} 回・平均 {_f(ov['duka']['pips_live'], 2, True)} pips、Yahoo {ov['yahoo']['n']} 回・"
                 f"平均 {_f(ov['yahoo']['pips_live'], 2, True)} pips、同じ時刻のエントリー {ov['same_entries']} 回。"
                 f"終値の差の中央値 (pips): {med}。")
    return L


# --------------------------------------------------------------------- run

def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def run(log=print, root: Path | None = None, checks: bool = True, codes: list[str] | None = None,
        out_dir: Path | None = None) -> dict:
    """``codes`` and ``out_dir`` are for trying the module on part of the data."""
    t0 = time.time()
    out_dir = out_dir or REPORT_DIR
    vix = _vix(root)
    cache: dict = {}
    duka_1h = load_panel("duka_1h", root, vix, cache, codes)
    duka_1d = load_panel("duka_1d", root, vix, cache, codes)
    yahoo_1d = load_panel("yahoo_1d", root, vix, codes=codes)
    log(f"data loaded in {time.time() - t0:.0f}s")

    def span(pan):
        a = min(P["time"][0] for P in pan.values())
        b = max(P["time"][-1] for P in pan.values())
        n = sum(len(P["c"]) for P in pan.values())
        return f"{a:%Y-%m-%d}〜{b:%Y-%m-%d}、{len(pan)}ペア計 {n:,} 本"

    spreads = {c: {"tune": float(np.median(P["spread_pips"][P["time"] < SPLIT])) if np.any(P["time"] < SPLIT) else None,
                   "test": float(np.median(P["spread_pips"][P["time"] >= SPLIT])) if np.any(P["time"] >= SPLIT) else None}
               for c, P in duka_1h.items()}
    res = {"split": str(SPLIT.date()), "data": {"duka_1h": span(duka_1h), "duka_1d": span(duka_1d), "yahoo_1d": span(yahoo_1d)},
           "spread_median_pips": spreads}
    res["1h"] = search("1h", {"duka_1h": duka_1h}, "duka_1h", {}, log)
    res["1d"] = search("1d", {"duka_1d": duka_1d, "yahoo_1d": yahoo_1d}, "duka_1d", {"yahoo_1d": DAILY_START}, log)
    log(f"search done in {time.time() - t0:.0f}s")
    chk: dict = {"signals": {"1h": check_signals(duka_1h, "1h"), "1d": check_signals(duka_1d, "1d")},
                 "signal_source": {"1h": "duka_1h", "1d": "duka_1d"}}
    if checks:
        yahoo_1h = load_panel("yahoo_1h", root, vix, codes=codes)
        prev = REPORT_DIR / "trade.json"
        if prev.exists():
            chk["reproduce"] = reproduce(json.loads(prev.read_text(encoding="utf-8")), res, yahoo_1h)
        chk["overlap"] = overlap_check(duka_1h, yahoo_1h)
        chk["yahoo_close"] = yahoo_close_check(cache, root)
        chk["live_backtest"] = check_live_backtest(root, codes)
        log(f"checks done in {time.time() - t0:.0f}s")
    res["checks"] = chk
    res["conclusion"] = _conclusion(res)
    res["check_lines"] = _check_lines(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signals.json").write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "signals.md").write_text(report(res), encoding="utf-8")
    log(f"wrote {out_dir / 'signals.md'} in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    run()
