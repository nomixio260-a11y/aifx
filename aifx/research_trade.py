"""Trading rules on long history, after costs: which, if any, make money out of sample.

The forecaster's direction has no demonstrated edge (research/report.md), so a
chart that says "buy here, take profit there, stop there" must rest on rules
that were tested as trades: entry at a bar's close, a stop and a target from
ATR, a time limit, the spread and slippage of a retail account, and the swap
(interest-rate difference minus the broker's markup) while the position is
held. Stops are checked before targets inside a bar (the conservative order),
and a bar that opens beyond a level fills at its open.

Each family of rules has a few settings. They are chosen on the older "tune"
period only (daily: 2002-2016; hourly: the first 60 %) and then scored on the
later "test" period. A family is adopted only if its chosen setting is still
profitable on the test period with a t statistic of at least 2 (monthly
returns, all pairs together, so correlated trades are not counted as
independent).

    aifx research --trade      # writes research/trade.md and research/trade.json
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS
from .trade import COST_PIPS, SWAP_MARKUP, atr

REPORT_DIR = Path("research")
DAILY_SPLIT = pd.Timestamp("2017-01-01")
DAILY_START = pd.Timestamp("2002-01-01")
HOURLY_TUNE_SHARE = 0.6
MIN_TUNE_TRADES = 150
ADOPT_T = 2.0


def classify(best: dict | None) -> str | None:
    """"strong": t >= 2 on both periods. "weak": profitable on both periods and t >= 2 on the
    test period (shown as a reference signal with its numbers). Otherwise not used."""
    if not best:
        return None
    tu, te = best["tune"], best["test"]
    if (tu.get("t") or 0) >= ADOPT_T and (te.get("t") or 0) >= ADOPT_T:
        return "strong"
    if (tu.get("R") or 0) > 0 and (te.get("R") or 0) > 0 and (te.get("t") or 0) >= ADOPT_T:
        return "weak"
    return None


# --------------------------------------------------------------- indicators

def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    au, ad = up[1:n + 1].mean(), dn[1:n + 1].mean()
    for i in range(n, len(c)):
        if i > n:
            au += (up[i] - au) / n
            ad += (dn[i] - ad) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).max().shift(1).to_numpy()     # the n bars before this one


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).min().shift(1).to_numpy()


# ------------------------------------------------------------------ signals

def signal(kind: str, p: dict, P: dict) -> np.ndarray:
    """Direction to open at each bar's close (+1 buy, -1 sell, 0 none), from data up to that close."""
    c = P["c"]
    if kind == "carry":
        diff = P["diff"]
        return np.where(diff >= p["thr"], 1, np.where(diff <= -p["thr"], -1, 0))
    if kind == "mom":
        L = p["L"]
        r = np.full(len(c), np.nan)
        r[L:] = c[L:] / c[:-L] - 1
        return np.nan_to_num(np.sign(r)).astype(int)
    if kind == "carry_mom":
        car = signal("carry", {"thr": p["thr"]}, P)
        mom = signal("mom", {"L": p["L"]}, P)
        return np.where(car == mom, car, 0)
    if kind == "donchian":
        hi, lo = rolling_max(P["h"], p["N"]), rolling_min(P["l"], p["N"])
        return np.where(c > hi, 1, np.where(c < lo, -1, 0))
    if kind == "rsi_rev":
        r = P["rsi"]
        return np.where(r < p["lo"], 1, np.where(r > 100 - p["lo"], -1, 0))
    if kind == "ma_trend":
        f = pd.Series(c).rolling(p["f"]).mean().to_numpy()
        s = pd.Series(c).rolling(p["s"]).mean().to_numpy()
        return np.nan_to_num(np.sign(f - s)).astype(int)
    if kind == "session_bo":
        # hourly: break of the 00-07 UTC (Tokyo) range during 07-12 UTC, first break of the day only
        hour, day = P["hour"], P["day"]
        out = np.zeros(len(c), dtype=int)
        cur_day, lo_r, hi_r, done = None, np.inf, -np.inf, False
        for i in range(len(c)):
            if day[i] != cur_day:
                cur_day, lo_r, hi_r, done = day[i], np.inf, -np.inf, False
            if hour[i] < 7:
                hi_r, lo_r = max(hi_r, P["h"][i]), min(lo_r, P["l"][i])
            elif hour[i] < 12 and not done and np.isfinite(hi_r):
                if c[i] > hi_r:
                    out[i], done = 1, True
                elif c[i] < lo_r:
                    out[i], done = -1, True
        return out
    raise ValueError(kind)


# ---------------------------------------------------------------- simulation

def simulate(P: dict, sig: np.ndarray, sl_atr: float, tp_atr: float | None, hold: int, cost: float) -> list[tuple]:
    """Trades as (entry index, exit index, direction, net pips, R). One position per pair."""
    o, h, lo, c, a, pip = P["o"], P["h"], P["l"], P["c"], P["atr"], P["pip"]
    acc = P["acc"]            # swap per bar in pips for (long, short)
    n = len(c)
    out = []
    i = P["start"]
    while i < n - 1:
        d = sig[i]
        if d == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        entry = c[i]
        risk = sl_atr * a[i]
        sl = entry - d * risk
        tp = entry + d * tp_atr * a[i] if tp_atr else None
        swap = 0.0
        ex = None
        j = i + 1
        while j < n:
            swap += acc[0 if d > 0 else 1][j]
            if d > 0:
                if o[j] <= sl or (tp is not None and o[j] >= tp):
                    ex = o[j]
                elif lo[j] <= sl:
                    ex = sl
                elif tp is not None and h[j] >= tp:
                    ex = tp
            else:
                if o[j] >= sl or (tp is not None and o[j] <= tp):
                    ex = o[j]
                elif h[j] >= sl:
                    ex = sl
                elif tp is not None and lo[j] <= tp:
                    ex = tp
            if ex is None and j - i >= hold:
                ex = c[j]
            if ex is not None:
                break
            j += 1
        if ex is None:
            break                      # still open at the end of the data
        pips = d * (ex - entry) / pip - cost + swap
        out.append((i, j, int(d), pips, pips / (risk / pip)))
        i = j
    return out


def _prep(code: str, df: pd.DataFrame, hourly: bool) -> dict:
    pair = PAIRS[code]
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
    idx = pd.DatetimeIndex(idx).as_unit("ns")          # asi8 below counts nanoseconds
    days = pd.DatetimeIndex(idx.normalize())
    rates = history.rates_panel(pd.DatetimeIndex(days.unique()))
    rb = rates[pair.base].reindex(days).to_numpy()
    rq = rates[pair.quote].reindex(days).to_numpy()
    diff = rb - rq
    gap_days = np.concatenate([[0.0], np.diff(idx.asi8) / 86_400e9])
    prev_c = np.concatenate([[c[0]], c[:-1]])
    per_day = prev_c / pair.pip / 100 / 365
    acc_long = np.nan_to_num((diff - SWAP_MARKUP) * per_day * gap_days)
    acc_short = np.nan_to_num((-diff - SWAP_MARKUP) * per_day * gap_days)
    P = {"code": code, "o": o, "h": h, "l": lo, "c": c, "pip": pair.pip, "atr": atr(h, lo, c),
         "rsi": rsi(c), "diff": np.nan_to_num(diff), "acc": (acc_long, acc_short), "time": idx,
         "start": 260 if not hourly else 150}
    if hourly:
        P["hour"] = idx.hour.to_numpy()
        P["day"] = days.asi8
    return P


# ------------------------------------------------------------------ metrics

def metrics(trades: list[tuple], times: dict) -> dict:
    """Pooled trade statistics; the t statistic uses monthly sums of R across all pairs."""
    if not trades:
        return {"n": 0}
    pips = np.array([t[3] for t in trades])
    R = np.array([t[4] for t in trades])
    exit_t = pd.DatetimeIndex([times[t[5]][t[1]] for t in trades])
    monthly = pd.Series(R, index=exit_t).groupby(exit_t.to_period("M")).sum()
    months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(months, fill_value=0.0)
    sd = monthly.std(ddof=1)
    order = np.argsort(exit_t.asi8, kind="stable")
    eq = np.cumsum(R[order])
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq)) if len(eq) else 0.0
    gains, losses = R[R > 0].sum(), -R[R < 0].sum()
    years = max((exit_t.max() - exit_t.min()).days / 365.25, 1 / 12)
    return {"n": int(len(R)), "win": float(np.mean(pips > 0)), "pips": float(pips.mean()), "R": float(R.mean()),
            "pf": float(gains / losses) if losses > 0 else None, "R_year": float(R.sum() / years),
            "sharpe": float(monthly.mean() / sd * math.sqrt(12)) if sd > 0 else None,
            "t": float(monthly.mean() / sd * math.sqrt(len(monthly))) if sd > 0 else None,
            "maxdd_R": dd, "per_year": float(len(R) / years)}


# ------------------------------------------------------------------ families

DAILY_FAMILIES = {
    "carry": {"name": "金利差 (スワップのつく方向に保有)", "grid": {"thr": [0.5, 1.0, 2.0]}},
    "mom": {"name": "モメンタム (過去L日の値動きの方向)", "grid": {"L": [20, 60, 120, 250]}},
    "carry_mom": {"name": "金利差 + モメンタムが一致したときだけ", "grid": {"thr": [0.5, 1.0], "L": [20, 60, 120]}},
    "donchian": {"name": "ブレイクアウト (過去N日の高値・安値を抜けた方向)", "grid": {"N": [20, 55]}},
    "ma_trend": {"name": "移動平均の向き (短期 > 長期で買い)", "grid": {"f": [20, 50], "s": [100, 200]}},
    "rsi_rev": {"name": "逆張り (RSIの売られすぎで買い・買われすぎで売り)", "grid": {"lo": [20, 30]}},
}
DAILY_EXITS = {"sl": [1.5, 2.5, 4.0], "tp": [None, 2.0, 4.0], "hold": [5, 20, 60]}
HOURLY_FAMILIES = {
    "mom": {"name": "モメンタム (過去L時間の値動きの方向)", "grid": {"L": [6, 24, 72]}},
    "donchian": {"name": "ブレイクアウト (過去N時間の高値・安値を抜けた方向)", "grid": {"N": [24, 72]}},
    "session_bo": {"name": "東京時間の高値・安値をロンドン時間に抜けた方向", "grid": {}},
    "rsi_rev": {"name": "逆張り (RSIの売られすぎで買い・買われすぎで売り)", "grid": {"lo": [20, 30]}},
    "carry_mom": {"name": "金利差 + モメンタムが一致したときだけ", "grid": {"thr": [1.0], "L": [24, 120]}},
}
HOURLY_EXITS = {"sl": [1.5, 3.0], "tp": [None, 1.5, 3.0], "hold": [6, 24]}


def _variants(fam: dict, exits: dict) -> list[dict]:
    keys = list(fam["grid"])
    sig_sets = [dict(zip(keys, v)) for v in itertools.product(*fam["grid"].values())] or [{}]
    out = []
    for sp in sig_sets:
        for sl, tp, hold in itertools.product(exits["sl"], exits["tp"], exits["hold"]):
            if tp is not None and tp <= 0:
                continue
            out.append({**sp, "sl": sl, "tp": tp, "hold": hold})
    return out


def evaluate(tf: str, log=print) -> dict:
    hourly = tf == "1h"
    fams, exits = (HOURLY_FAMILIES, HOURLY_EXITS) if hourly else (DAILY_FAMILIES, DAILY_EXITS)
    panel = {}
    for code in PAIRS:
        df = history.load_hourly(code) if hourly else history.load_daily(code)
        if not hourly:
            df = df[df.index >= DAILY_START - pd.Timedelta(days=400)]
        panel[code] = _prep(code, df, hourly)
    times = {code: P["time"] for code, P in panel.items()}
    if hourly:
        all_t = np.sort(np.concatenate([P["time"].asi8 for P in panel.values()]))
        split = pd.Timestamp(all_t[int(len(all_t) * HOURLY_TUNE_SHARE)])
        start = pd.Timestamp(all_t[0])
    else:
        split, start = DAILY_SPLIT, DAILY_START
    out = {"tf": tf, "split": str(split.date()), "start": str(start.date()),
           "end": str(max(P["time"][-1] for P in panel.values()).date()), "families": {}}
    for key, fam in fams.items():
        rows = []
        sig_cache: dict = {}
        for v in _variants(fam, exits):
            trades = []
            sk = tuple(sorted((k, v[k]) for k in fam["grid"]))
            for code, P in panel.items():
                if (code, sk) not in sig_cache:
                    sig_cache[(code, sk)] = signal(key, v, P)
                sig = sig_cache[(code, sk)]
                for tr in simulate(P, sig, v["sl"], v["tp"], v["hold"], COST_PIPS[code]):
                    if P["time"][tr[0]] >= start:
                        trades.append(tr + (code,))
            tune = [t for t in trades if times[t[5]][t[0]] < split]
            test = [t for t in trades if times[t[5]][t[0]] >= split]
            rows.append({"params": v, "tune": metrics(tune, times), "test": metrics(test, times)})
        ok = [r for r in rows if r["tune"].get("n", 0) >= MIN_TUNE_TRADES and r["tune"].get("t") is not None]
        best = max(ok, key=lambda r: r["tune"]["t"]) if ok else None
        pos_test = sum(1 for r in rows if (r["test"].get("R") or 0) > 0) / len(rows)
        tier = classify(best)
        adopted = tier == "strong"
        out["families"][key] = {"name": fam["name"], "chosen": best, "share_positive_test": pos_test,
                                "variants": len(rows), "adopted": adopted, "tier": tier}
        if log:
            b = best or {"tune": {}, "test": {}}
            log(f"{tf} {key}: tune t={b['tune'].get('t')} test R={b['test'].get('R')} t={b['test'].get('t')} "
                f"adopted={adopted} params={b.get('params')}")
    return out


# ------------------------------------------------------------------- report

def _f(x, d=2, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _params_text(p: dict) -> str:
    parts = []
    for k, v in p.items():
        if k == "sl":
            parts.append(f"損切り ATR×{v}")
        elif k == "tp":
            parts.append("利確なし" if v is None else f"利確 ATR×{v}")
        elif k == "hold":
            parts.append(f"最長{v}本")
        elif k == "thr":
            parts.append(f"金利差{v}%以上")
        else:
            parts.append(f"{k}={v}")
    return "、".join(parts)


def report(res: dict) -> str:
    L = ["# 売買ルールの検証 (コスト込み)", "",
         "予測の中心 (方向) には検証で優位性が見つかっていないため、「どこで買い、どこで利確・損切りするか」を示すには、"
         "売買ルールそのものを取引として検証する必要があります。各ルールは、足の終値で注文し、ATR (平均的な値幅) から決めた"
         "損切り・利確と最長保有期間で決済し、スプレッドとスリッページ、保有中のスワップ (金利差から業者の取り分 "
         f"{SWAP_MARKUP}% を引いたもの) を含めて損益を計算しました。1本の足の中で損切りと利確の両方に届いたときは損切りとしています。", "",
         "コスト (往復、pips): " + "、".join(f"{c} {v}" for c, v in COST_PIPS.items()), "",
         "設定は古い期間 (調整期間) だけで選び、新しい期間 (検証期間) の成績で採否を決めました。"
         f"採用の条件は、調整期間と検証期間の両方で、月ごとの損益 (全ペア合計) から計算した t 値が {ADOPT_T} 以上であることです。"
         f"これを満たさなくても、両方の期間で平均がプラスで、検証期間の t 値が {ADOPT_T} 以上のルールは「参考シグナル (信頼度: 弱)」として、成績を添えて表示します。"
         "R は損切り幅を1とした損益 (1R = 損切り1回分) です。", ""]
    for tf, r in res.items():
        L += [f"## {'日足' if tf == '1d' else '1時間足'} (調整 {r['start']}〜{r['split']}、検証 {r['split']}〜{r['end']})", "",
              "| ルール | 選んだ設定 | 調整: 取引数 / 勝率 / 平均 / t | 検証: 取引数 / 勝率 / 平均 (pips) / 平均 (R) / PF / t / 最大DD (R) | 採用 |",
              "|---|---|---|---|---|"]
        for key, f in r["families"].items():
            b = f["chosen"]
            if not b:
                L.append(f"| {f['name']} | — | 取引が少なすぎる | — | — |")
                continue
            tu, te = b["tune"], b["test"]
            L.append(f"| {f['name']} | {_params_text(b['params'])} | {tu['n']} / {tu['win']:.0%} / {_f(tu['R'], 3, True)}R / {_f(tu['t'])} | "
                     f"{te.get('n', 0)} / {te.get('win', 0):.0%} / {_f(te.get('pips'), 1, True)} / {_f(te.get('R'), 3, True)} / "
                     f"{_f(te.get('pf'))} / {_f(te.get('t'))} / {_f(te.get('maxdd_R'), 1)} | "
                     f"{ {'strong': '**採用**', 'weak': '参考 (弱)'}.get(f.get('tier'), '不採用') } |")
        L += ["", "検証期間で平均がプラスだった設定の割合 (選ぶ前の全設定): " +
              "、".join(f"{f['name'].split(' (')[0]} {f['share_positive_test']:.0%}" for f in r["families"].values()), ""]
    L += ["## 結論", ""]
    for tf, r in res.items():
        tfn = "日足" if tf == "1d" else "1時間足"
        used = [(k, f) for k, f in r["families"].items() if f.get("tier")]
        if not used:
            L.append(f"- {tfn}: 両方の期間で利益が出たルールはありません。売買シグナルは出しません。")
            continue
        k, f = max(used, key=lambda kf: (kf[1]["tier"] == "strong", kf[1]["chosen"]["test"]["t"]))
        te = f["chosen"]["test"]
        L.append(f"- {tfn}: 「{f['name']}」({_params_text(f['chosen']['params'])}) を"
                 f"{'売買シグナルに採用' if f['tier'] == 'strong' else '参考シグナル (信頼度: 弱) として表示'}。"
                 f"検証期間は {te['n']} 回、勝率 {te['win']:.0%}、1回平均 {te['pips']:+.1f} pips (コスト・スワップ込み)、t = {te['t']:.2f}。")
    L += ["", "どのルールも期待値は小さく、損切り幅に対して1回あたり数%の上乗せにとどまります。金利差を使うルールは、"
          "スワップ収入と金利差の上乗せ分が利益の源ですが、急な円高などで短期間に大きく負ける時期があります (最大DD を参照)。", ""]
    return "\n".join(L)


def run(log=print) -> dict:
    res = {tf: evaluate(tf, log=log) for tf in ("1d", "1h")}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "trade.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (REPORT_DIR / "trade.md").write_text(report(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
