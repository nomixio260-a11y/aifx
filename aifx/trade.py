"""Trade plans: which way to trade now, where to take profit and where to stop.

The rules are the ones research_trade.py tested as trades (entry at a bar's
close, stop and target from ATR, a time limit, costs and swap). Only rules
that were profitable on both the tuning and the later test period, with
t >= 2 on the test period, give a signal; they are shown as reference
signals with their tested numbers, because none reached t >= 2 on both
periods. Every forecast record carries its plan, so a signal is written to
the ledger before its outcome and settled from the stored prices.

Both-side levels (where the stop and target would be for a buy or a sell)
are given for every timeframe, signal or not.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .data import Pair
from .engine import Timeframe, bar_end, target_times
from .rates import rate_diff
from .timeutil import iso, parse_iso

# spread + slippage per round trip, pips (a typical Japanese retail account, a little conservative)
COST_PIPS = {"USDJPY": 0.5, "EURJPY": 0.7, "GBPJPY": 1.3, "AUDJPY": 0.9, "EURUSD": 0.5, "GBPUSD": 0.9, "AUDUSD": 0.7}
SWAP_MARKUP = 0.5          # % a year the broker keeps from the rate difference, on either side


def atr(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's average true range, known at each bar's close."""
    prev = np.concatenate([[c[0]], c[:-1]]) if len(c) else c
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev), np.abs(lo - prev)))
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(c)):
        out[i] = out[i - 1] + (tr[i] - out[i - 1]) / n
    return out

# Reference rules per timeframe (research/trade.md): parameters chosen on the tuning period only.
RULES = {
    "1d": {"key": "carry", "name": "金利差", "desc": "金利差が2%以上ある通貨ペアを、スワップがつく方向に持つ",
           "thr": 2.0, "sl": 4.0, "tp": 2.0, "hold": 5},
    "1h": {"key": "carry_mom", "name": "金利差 + 5日間の流れ", "desc": "金利差が1%以上あり、過去120時間の値動きも同じ向きのときだけ、その方向に持つ",
           "thr": 1.0, "L": 120, "sl": 3.0, "tp": 3.0, "hold": 24},
}
# stop / target / time limit for the both-side levels where no rule is tested
DEFAULT_EXITS = {"15m": {"sl": 3.0, "tp": 3.0, "hold": 16}, "1h": {"sl": 3.0, "tp": 3.0, "hold": 24},
                 "1d": {"sl": 4.0, "tp": 2.0, "hold": 5}}


def exits(tf_key: str) -> dict:
    r = RULES.get(tf_key)
    return {k: r[k] for k in ("sl", "tp", "hold")} if r else DEFAULT_EXITS[tf_key]


def rule_signal(tf_key: str, closes: np.ndarray, diff: float | None) -> int:
    """+1 buy, -1 sell, 0 none, from closes up to the origin and the known rate difference."""
    r = RULES.get(tf_key)
    if not r or diff is None:
        return 0
    car = 1 if diff >= r["thr"] else -1 if diff <= -r["thr"] else 0
    if r["key"] == "carry":
        return car
    if r["key"] == "carry_mom":
        L = r["L"]
        if len(closes) <= L:
            return 0
        mom = np.sign(closes[-1] / closes[-1 - L] - 1)
        return car if car != 0 and mom == car else 0
    return 0


def plan(tf: Timeframe, pair: Pair, bars: pd.DataFrame, origin: datetime, p0: float, rates_item: dict | None) -> dict:
    """The plan recorded with a forecast: signal, stop, target and time limit (prices from p0)."""
    a = atr(bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float))
    a_last = float(a[-1]) if len(a) and np.isfinite(a[-1]) else None
    diff = rate_diff(rates_item, pair.base, pair.quote, origin.date()) if rates_item else None
    d = rule_signal(tf.key, bars["close"].to_numpy(float), diff) if a_last else 0
    ex = exits(tf.key)
    dec = pair.decimals + 1
    out = {"rule": RULES[tf.key]["key"] if tf.key in RULES else None, "dir": d,
           "atr": round(a_last, dec + 1) if a_last else None, "diff": diff,
           "until": iso(target_times(tf, origin, (ex["hold"],))[0])}
    if a_last and d:
        out["sl"] = round(p0 - d * ex["sl"] * a_last, dec)
        out["tp"] = round(p0 + d * ex["tp"] * a_last, dec)
    return out


def levels(tr: dict, p0: float, tf_key: str, decimals: int) -> dict:
    """Stop and target for a buy and for a sell from the recorded ATR."""
    ex = exits(tf_key)
    if not tr or not tr.get("atr"):
        return {}
    a = tr["atr"]
    dec = decimals + 1
    return {"buy": {"sl": round(p0 - ex["sl"] * a, dec), "tp": round(p0 + ex["tp"] * a, dec)},
            "sell": {"sl": round(p0 + ex["sl"] * a, dec), "tp": round(p0 - ex["tp"] * a, dec)},
            "sl_mult": ex["sl"], "tp_mult": ex["tp"], "hold": ex["hold"]}


def settle(d: int, entry: float, sl: float, tp: float, until: str, origin: str, ref: pd.DataFrame, ref_minutes: int,
           pip: float, cost: float, diff: float | None) -> dict | None:
    """Outcome of a trade from the reference bars after its origin: the stop is checked before
    the target inside a bar, and a bar that opens beyond a level fills at its open. None while open."""
    ends = ref.index + pd.Timedelta(minutes=ref_minutes)
    o_t, u_t = pd.Timestamp(origin), pd.Timestamp(until)
    after = ref[(ends > o_t)]
    exit_px, how, t_exit = None, None, None
    for ts, row in after.iterrows():
        end = ts + pd.Timedelta(minutes=ref_minutes)
        o, h, lo, c = row["open"], row["high"], row["low"], row["close"]
        if d > 0:
            if o <= sl or o >= tp:
                exit_px, how = o, "sl" if o <= sl else "tp"
            elif lo <= sl:
                exit_px, how = sl, "sl"
            elif h >= tp:
                exit_px, how = tp, "tp"
        else:
            if o >= sl or o <= tp:
                exit_px, how = o, "sl" if o >= sl else "tp"
            elif h >= sl:
                exit_px, how = sl, "sl"
            elif lo <= tp:
                exit_px, how = tp, "tp"
        if exit_px is None and end >= u_t:
            exit_px, how = c, "time"
        if exit_px is not None:
            t_exit = end
            break
    if exit_px is None:
        return None
    days = (t_exit - o_t).total_seconds() / 86400
    swap = 0.0
    if diff is not None:
        swap = ((d * diff) - SWAP_MARKUP) * entry / pip / 100 / 365 * days
    pips = d * (exit_px - entry) / pip - cost + swap
    risk = abs(entry - sl) / pip
    return {"exit": float(exit_px), "how": how, "t": iso(t_exit.to_pydatetime()), "pips": round(pips, 2),
            "R": round(pips / risk, 4) if risk > 0 else None}


def live_trades(preds: list[dict], ref: pd.DataFrame, ref_minutes: int, pair: Pair) -> list[dict]:
    """The recorded signals of one pair and timeframe, one position at a time (a signal while a
    position is open is not a new trade), each settled from the stored prices."""
    out, busy_until = [], None
    for p in sorted(preds, key=lambda q: q["origin"]):
        tr = p.get("trade") or {}
        if not tr.get("dir") or "sl" not in tr:
            continue
        if busy_until is not None and p["origin"] < busy_until:
            continue
        res = settle(tr["dir"], p["p0"], tr["sl"], tr["tp"], tr["until"], p["origin"], ref, ref_minutes, pair.pip,
                     COST_PIPS.get(pair.code, 1.0), tr.get("diff"))
        out.append({"seq": p["seq"], "origin": p["origin"], "dir": tr["dir"], "entry": p["p0"], "sl": tr["sl"],
                    "tp": tr["tp"], "until": tr["until"], "result": res})
        busy_until = res["t"] if res else "9999"
    return out


def backtest(tf: Timeframe, pair: Pair, bars: pd.DataFrame, rates_item: dict | None, since: datetime | None = None) -> list[dict]:
    """The reference rule replayed over stored bars (entry at each bar's close, same exits as the
    research). Rate differences use the stored rates as they were known at each bar."""
    r = RULES.get(tf.key)
    if not r or rates_item is None or len(bars) < 200:
        return []
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(h, lo, c)
    ends = [bar_end(tf, ts) for ts in bars.index]
    days = sorted({e.date() for e in ends})
    diff_by_day = {d: rate_diff(rates_item, pair.base, pair.quote, d) for d in days}
    cost = COST_PIPS.get(pair.code, 1.0)
    out, i, n = [], 150, len(c)
    while i < n - 1:
        if since is not None and ends[i] < since:
            i += 1
            continue
        diff = diff_by_day[ends[i].date()]
        d = rule_signal(tf.key, c[: i + 1], diff) if np.isfinite(a[i]) else 0
        if not d:
            i += 1
            continue
        entry, sl, tp = c[i], c[i] - d * r["sl"] * a[i], c[i] + d * r["tp"] * a[i]
        j, ex, how = i + 1, None, None
        while j < n:
            if d > 0:
                if o[j] <= sl or o[j] >= tp:
                    ex, how = o[j], "sl" if o[j] <= sl else "tp"
                elif lo[j] <= sl:
                    ex, how = sl, "sl"
                elif h[j] >= tp:
                    ex, how = tp, "tp"
            else:
                if o[j] >= sl or o[j] <= tp:
                    ex, how = o[j], "sl" if o[j] >= sl else "tp"
                elif h[j] >= sl:
                    ex, how = sl, "sl"
                elif lo[j] <= tp:
                    ex, how = tp, "tp"
            if ex is None and j - i >= r["hold"]:
                ex, how = c[j], "time"
            if ex is not None:
                break
            j += 1
        if ex is None:
            break
        dd = (ends[j] - ends[i]).total_seconds() / 86400
        swap = ((d * diff) - SWAP_MARKUP) * entry / pair.pip / 100 / 365 * dd
        pips = d * (ex - entry) / pair.pip - cost + swap
        out.append({"origin": iso(ends[i]), "dir": d, "entry": float(entry), "sl": float(sl), "tp": float(tp),
                    "exit": float(ex), "how": how, "t": iso(ends[j]), "pips": round(pips, 2),
                    "R": round(pips / (r["sl"] * a[i] / pair.pip), 4)})
        i = j
    return out


def stats(trades: list[dict]) -> dict:
    done = [t for t in trades if t.get("pips") is not None]
    if not done:
        return {"n": 0}
    p = np.array([t["pips"] for t in done])
    R = np.array([t["R"] for t in done if t.get("R") is not None])
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    return {"n": len(done), "win": float(np.mean(p > 0)), "avg_pips": float(p.mean()), "total_pips": float(p.sum()),
            "avg_R": float(R.mean()) if len(R) else None, "pf": float(wins / losses) if losses > 0 else None,
            "tp": sum(1 for t in done if t["how"] == "tp"), "sl": sum(1 for t in done if t["how"] == "sl"),
            "time": sum(1 for t in done if t["how"] == "time")}


__all__ = ["RULES", "plan", "levels", "settle", "live_trades", "backtest", "stats", "parse_iso"]
