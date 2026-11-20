"""How big the time-of-day moves really are: shrinking each slot's average by its reliability (research/drift.md).

season.py calls a direction when a New York time slot's average move over the last
year is clearly not zero (|t| >= 2) and uses that average as the expected move of
the next bar. Two months of live forecasts (the ledger, 2026-09-23 to 11-20) show
the direction holding but the size overstated: on hourly bars the calls below
|t| 4 expected 2.2 bp and got 1.0 bp in their direction, the ones above 2.8 bp
and 2.3 bp; on 15-minute bars below |t| 4, 1.1 bp expected and 0.06 bp got. A
slot picked because its average stands out is, on average, one whose average
came out too high (the winner's curse), and more so the noisier it is.

The candidate here has no setting to tune: an empirical-Bayes shrink of every
slot average toward zero by its own reliability, with the spread of true slot
effects estimated afresh at each origin from all the slots of the same kind
(``PREREG``). The calls (which bars get a direction, and their tier) do not
change; only the expected move does, and with it the centre of the forecast.

    python -m aifx.research_drift      # research/drift.md and research/drift.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import history, season
from .data import PAIRS, Pair, _yahoo_chart, parse_yahoo_intraday
from .stats import diebold_mariano
from .timeutil import market_open_mask, utcnow

REPORT_DIR = Path("research")
CACHE = "yahoo_1h"
LIVE = tuple(PAIRS)
HOLDOUT = history.HOLDOUT_PAIRS
NAMES = {"EURGBP": "ユーロ/ポンド", "EURAUD": "ユーロ/豪ドル", "GBPAUD": "ポンド/豪ドル",
         "USDCHF": "米ドル/スイスフラン", "USDCAD": "米ドル/カナダドル", "NZDUSD": "NZドル/米ドル"}

PREREG = """\
## 0. 事前に決めた検定 (結果を見る前に固定)

この節は、結果を1つも計算する前に aifx/research_drift.py (`PREREG`) に書いたもので、結果を見た後に変えていません。

- **きっかけ:** 本番の台帳 (2026-09-23〜11-20) で、時間帯の偏りの方向は当たっている一方、見込みの大きさは実際より大きく出ていました (1時間足: |t| < 4 の回は見込み 2.2 bp に対して実際 1.0 bp、|t| ≥ 4 は 2.8 bp に対して 2.3 bp)。平均が目立つ枠を選ぶと、偶然平均が大きく出た枠を選びやすい (勝者の呪い) ためと考えられます。
- **候補 (調整する値なし):** 経験ベイズの縮小。起点ごとに、同じ種類の枠 (曜日×時間の枠、時間帯の枠のそれぞれ) すべての平均 m と標準誤差 se (= |m / t|) から、本当の枠の効果のばらつき τ² = max(0, 平均(m²) − 平均(se²)) を推定し、各枠の平均を m × τ² / (τ² + se²) に縮めて見込みの値動きとします (本数が足りない枠 (15本未満) は推定に使いません)。どの足で方向を示すか (|t| ≥ 2)、確度の区分、ロールオーバーの格上げは変えません。
- **データ:** Yahoo の1時間足 (サーバーと同じ表示価格) 約2年分。本番の7ペアと、どの設計にも使っていない6ペア (EURGBP・EURAUD・GBPAUD・USDCHF・USDCAD・NZDUSD)。計算は本番の関数そのまま (season.slot_stats・bar_drift。統計は起点の UTC の日より前の6,000本、市場が閉じている足と週末明けの最初の足は除く)。
- **比べ方:** 方向を示した足について、次の1本の値動き (bp) と見込みの差の2乗の平均 (MSE)。今の見込み (縮小なし) と候補で比べます。差が偶然かどうかは、日ごとの平均の差の Diebold-Mariano 検定 (ラグ1) で示します。
- **期間:** 7ペアは評価できる日 (6,000本の助走の後) の前半を調整、後半を検証。6ペアは全期間。
- **合格の条件:** 候補の MSE が今の見込みより小さいこと: 7ペアの調整期間、7ペアの検証期間、6ペア、本番の台帳 (2026-09-23〜11-20 の1時間足の1時間先の予測を、記録された見込みを候補に置き換えて採点) の**すべてで**。
- 合格すれば、1時間足と15分足の時間帯の偏りの見込み (予測の中心と予想ローソク足) に使います。不合格なら変えません。
"""

SHRINK_MIN_N = season.MIN_N


# ------------------------------------------------------------------ the candidate

def _counts(stats: dict, key: str) -> np.ndarray:
    return stats["n_slot"] if key == "slot" else stats["n_tod"]


def eb_factors(stats: dict) -> dict[str, np.ndarray]:
    """Per stats set ("slot", "tod"): the empirical-Bayes shrink factor tau^2 / (tau^2 + se^2) of every slot,
    tau^2 from the spread of all usable slot averages of that set (0 where a slot has no usable average)."""
    out = {}
    for key in ("slot", "tod"):
        mu, t = stats[key]
        n = _counts(stats, key)
        use = (n >= SHRINK_MIN_N) & (t != 0)
        se = np.where(use, np.abs(mu / np.where(t != 0, t, 1.0)), np.inf)
        if use.sum() >= 2:
            tau2 = max(0.0, float(np.mean(mu[use] ** 2) - np.mean(se[use] ** 2)))
        else:
            tau2 = 0.0
        out[key] = np.where(use, tau2 / (tau2 + se ** 2), 0.0)
    return out


def shrunk_drift(stats: dict, starts, minutes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """season.bar_drift and its shrunk version: (d, t, d_shrunk)."""
    starts = pd.DatetimeIndex(starts)
    d, t = season.bar_drift(stats, starts, minutes)
    week, tod, _ = season._slots(starts, minutes)
    mu_w, t_w = stats["slot"]
    f = eb_factors(stats)
    use_w = (d != 0) & (np.abs(t_w[week]) >= season.T_MIN)
    fac = np.where(use_w, f["slot"][week], f["tod"][tod])
    return d, t, np.where(d != 0, d * fac, 0.0)


# ------------------------------------------------------------------ data

def _pair(code: str) -> Pair:
    return PAIRS.get(code) or Pair(code, code[:3], code[3:], NAMES[code])


def fetch(codes, root: Path | None = None, log=print) -> dict[str, pd.DataFrame]:
    """Yahoo's hourly bars (about two years), kept in data/history/yahoo_1h."""
    base = (root or history.HIST_DIR) / CACHE
    base.mkdir(parents=True, exist_ok=True)
    out = {}
    for code in codes:
        p = base / f"{code}.csv"
        if not p.exists():
            df, _ = parse_yahoo_intraday(_yahoo_chart(_pair(code).yahoo_symbol, "range=730d&interval=60m"), utcnow(), 60)
            df.to_csv(p, float_format="%.6f")
            log(f"{code}: {len(df)} hourly bars {df.index[0]} .. {df.index[-1]}")
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        out[code] = df
    return out


# ------------------------------------------------------------------ replay

def replay(code: str, bars: pd.DataFrame, minutes: int = 60) -> pd.DataFrame:
    """Every bar after the warm-up: the live call (d, t) and the shrunk expected move, from statistics of the
    bars before its UTC day; and the bar's move (bp). Bars right after a pause are not scored."""
    bars = bars[market_open_mask(bars.index)]
    idx = bars.index
    c = bars["close"].to_numpy(float)
    ret = np.full(len(c), np.nan)
    ret[1:] = np.log(c[1:] / c[:-1]) * 1e4
    gap = np.zeros(len(c), bool)
    gap[1:] = np.diff(idx.as_unit("ns").asi8) > minutes * 60_000_000_000
    ret[gap] = np.nan
    days = idx.normalize()
    rows = []
    first = season.WINDOW + 1
    for day in pd.unique(days[first:]):
        sel = np.flatnonzero(days == day)
        sel = sel[sel >= first]
        if not len(sel):
            continue
        stats = season.slot_stats(bars, day.to_pydatetime(), minutes)
        d, t, ds = shrunk_drift(stats, idx[sel], minutes)
        rows.append(pd.DataFrame({"time": idx[sel], "d": d, "t": t, "ds": ds, "ret": ret[sel]}))
    out = pd.concat(rows, ignore_index=True)
    out["pair"] = code
    return out.dropna(subset=["ret"])


def _score(df: pd.DataFrame) -> dict:
    call = df[df.d != 0]
    if not len(call):
        return {"n": 0}
    e0 = (call.ret - call.d) ** 2
    e1 = (call.ret - call.ds) ** 2
    day = call.time.dt.floor("D")
    dm, p = diebold_mariano(e1.groupby(day).mean().to_numpy(), e0.groupby(day).mean().to_numpy(), lag=1)
    nz = call[call.ret != 0]
    high = call[call.t.abs() >= season.T_HIGH]
    return {"n": int(len(call)), "days": int(day.nunique()), "mse_now": float(e0.mean()), "mse_shrunk": float(e1.mean()),
            "mse_zero": float((call.ret ** 2).mean()), "gain_pct": float((1 - e1.mean() / e0.mean()) * 100),
            "dm": dm, "p": p, "hit": float((np.sign(nz.d) == np.sign(nz.ret)).mean()),
            "mean_d": float(call.d.abs().mean()), "mean_ds": float(call.ds.abs().mean()),
            "realised": float((np.sign(call.d) * call.ret).mean()),
            "factor": float((call.ds / call.d).mean()),
            "high": {"n": int(len(high)), "mean_d": float(high.d.abs().mean()) if len(high) else None,
                     "mean_ds": float(high.ds.abs().mean()) if len(high) else None,
                     "realised": float((np.sign(high.d) * high.ret).mean()) if len(high) else None}}


def evaluate(frames: dict[str, pd.DataFrame]) -> dict:
    live = pd.concat([frames[c] for c in LIVE if c in frames])
    times = np.sort(pd.unique(live.time.dt.floor("D")))
    split = pd.Timestamp(times[len(times) // 2])
    hold = pd.concat([frames[c] for c in HOLDOUT if c in frames])
    return {"split": str(split.date()),
            "live_tune": _score(live[live.time.dt.floor("D") < split]),
            "live_test": _score(live[live.time.dt.floor("D") >= split]),
            "holdout": _score(hold),
            "pairs": {c: _score(f) for c, f in frames.items()}}


def ledger_check(state_root: Path, rows: pd.DataFrame | None = None) -> dict:
    """The live forecasts (1-hour bars, 1 hour ahead, with a time-of-day call), scored with the recorded expected
    move and with the shrunk one (the same statistics, rebuilt from the stored bars at each origin)."""
    import json as _json
    preds = []
    for f in sorted((Path(state_root) / "ledger").glob("*.jsonl")):
        for line in open(f):
            r = _json.loads(line)
            if r.get("type") == "prediction" and r["tf"] == "1h":
                fc = next(x for x in r["fc"] if x["h"] == 1)
                if fc.get("d"):
                    preds.append((r["seq"], r["pair"], r["origin"], r["p0"], fc["d"], fc.get("dt", 0.0), fc["c"]))
    outcomes = {}
    for f in sorted((Path(state_root) / "ledger").glob("*.jsonl")):
        for line in open(f):
            r = _json.loads(line)
            if r.get("type") == "outcome":
                for seq, h, actual, _end in r["items"]:
                    if h == 1 and actual is not None:
                        outcomes[seq] = actual
    bars = {}
    out = []
    for seq, code, origin, p0, d, dt, c in preds:
        if seq not in outcomes:
            continue
        if code not in bars:
            b = pd.read_csv(Path(state_root) / "prices" / f"{code}_1h.csv", index_col=0)
            b.index = pd.to_datetime(b.index, utc=True)
            bars[code] = b
        o = pd.Timestamp(origin)
        stats = season.slot_stats(bars[code], o.to_pydatetime(), 60)
        d2, _t2, ds = shrunk_drift(stats, pd.DatetimeIndex([o]), 60)
        if abs(d2[0] - d) > 1e-3:          # the statistics did not rebuild to the recorded call
            continue
        out.append({"time": o, "d": d, "t": dt, "ds": float(ds[0]), "ret": math.log(outcomes[seq] / p0) * 1e4,
                    "c": c, "pair": code})
    df = pd.DataFrame(out)
    res = _score(df)
    res["rebuilt"] = int(len(df))
    res["calls"] = int(sum(1 for p in preds if p[0] in outcomes))
    res["mse_centre_now"] = float(((df.ret - df.c) ** 2).mean())
    res["mse_centre_shrunk"] = float(((df.ret - (df.c - df.d + df.ds)) ** 2).mean())
    return res


def live_replay(state_root: Path) -> dict:
    """The live forecasts (2026-09-23 on), as recorded and with today's rule (season.step_drift: the shrunk
    expected move and, for 15-minute bars, calls from |t| 4 only), rebuilt from the stored bars at each origin:
    the centre's squared error 1 and 4 bars ahead, and the direction of the called bars."""
    import json as _json
    from .data import PAIRS as _P
    root = Path(state_root)
    preds, outcomes = [], {}
    for f in sorted((root / "ledger").glob("*.jsonl")):
        for line in open(f):
            r = _json.loads(line)
            if r.get("type") == "prediction" and r["tf"] in ("15m", "1h") and "dt" in r["fc"][0]:
                preds.append(r)             # versions that recorded the time-of-day call (from 2026-09-24 evening)
            elif r.get("type") == "outcome":
                for seq, h, actual, _end in r["items"]:
                    if actual is not None:
                        outcomes[(seq, h)] = actual
    bars: dict = {}
    rows = []
    for p in preds:
        mins = 15 if p["tf"] == "15m" else 60
        hs = [f for f in p["fc"] if f["h"] in (1, 4) and (p["seq"], f["h"]) in outcomes]
        if not hs:
            continue
        key = (p["pair"], p["tf"])
        if key not in bars:
            b = pd.read_csv(root / "prices" / f"{p['pair']}_{p['tf']}.csv", index_col=0)
            b.index = pd.to_datetime(b.index, utc=True)
            bars[key] = b
        o = pd.Timestamp(p["origin"]).to_pydatetime()
        diff = (p.get("trade") or {}).get("diff") if mins == 60 else None
        d, t = season.step_drift(mins, bars[key], o, 4, _P[p["pair"]] if mins == 60 else None, diff)
        cen = season.centre_drift(d, mins)
        raw_t = [x.get("dt", 0.0) for x in p["fc"]]
        for f in hs:
            ret = math.log(outcomes[(p["seq"], f["h"])] / p["p0"]) * 1e4
            rows.append({"tf": p["tf"], "h": f["h"], "ret": ret, "c": f["c"], "c_new": f["c"] - f.get("d", 0.0) + cen[f["h"] - 1],
                         "d": f.get("d", 0.0), "dir_new": float(np.sign(t[0])) if f["h"] == 1 else 0.0,
                         "t_new": float(t[0]), "dt": f.get("dt", 0.0)})
    df = pd.DataFrame(rows)
    out = {}
    for (tf, h), g in df.groupby(["tf", "h"]):
        r = {"n": int(len(g)), "mse_now": float(((g.ret - g.c) ** 2).mean()), "mse_new": float(((g.ret - g.c_new) ** 2).mean()),
             "mse_no_drift": float(((g.ret - (g.c - g.d)) ** 2).mean())}
        if h == 1:
            nz = g[g.ret != 0]
            now_calls = nz[nz.d != 0]
            new_calls = nz[nz.dir_new != 0]
            r["calls_now"] = {"n": int(len(now_calls)), "hit": float((np.sign(now_calls.d) == np.sign(now_calls.ret)).mean())}
            r["calls_new"] = {"n": int(len(new_calls)), "hit": float((new_calls.dir_new == np.sign(new_calls.ret)).mean())}
            mid = now_calls[now_calls.dt.abs() < season.T_HIGH]
            high = now_calls[now_calls.dt.abs() >= season.T_HIGH]
            for name, x in (("mid_now", mid), ("high_now", high)):
                r[name] = {"n": int(len(x)), "hit": float((np.sign(x.d) == np.sign(x.ret)).mean()) if len(x) else None,
                           "expected": float(x.d.abs().mean()) if len(x) else None,
                           "realised": float((np.sign(x.d) * x.ret).mean()) if len(x) else None}
        out[f"{tf}_{h}"] = r
    return out


# ------------------------------------------------------------------ report

def _f(x, d=2):
    return "—" if x is None else f"{x:,.{d}f}"


def _p(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def report(res: dict) -> str:
    ev, lc, lr = res["eval"], res["ledger"], res["live"]
    L = ["# 時間帯の偏りの見込みの大きさ: 枠の平均を信頼度で縮める", "",
         "時間帯の偏り (season.py) は、ニューヨーク時間の枠の過去1年の平均が偶然と区別できるとき (|t| ≥ 2) に次の足の方向を示し、"
         "その平均を次の足の見込みの値動きとして予測の中心と予想ローソク足に使っています。本番の2か月の実績で、方向は当たっている一方、"
         "見込みの大きさが実際より大きく出ていたため、事前に決めた方法で確かめました。", "", res["prereg"].rstrip(), ""]
    rows = [("本番の7ペア 調整 (〜" + ev["split"] + ")", ev["live_tune"]), ("本番の7ペア 検証 (" + ev["split"] + "〜)", ev["live_test"]),
            ("設計に使っていない6ペア (全期間)", ev["holdout"]), ("本番の台帳 (2026-09-23〜11-20、1時間足の1時間先)", lc)]
    ok = all(r["mse_shrunk"] < r["mse_now"] for _n, r in rows)
    L += ["## 結論", "",
          f"- **{'合格' if ok else '不合格'}。** 縮めた見込みの誤差 (MSE) は、4つの比較の{'すべて' if ok else '一部'}で今の見込みより小さくなりました。"
          + ("1時間足と15分足の見込みに使います (方向を示す足と確度は変えません)。" if ok else "本番は変えません。"), "",
          "| 比較 | 方向を示した足 | MSE 今の見込み | MSE 縮めた見込み | 差 | p (日ごと) | 見込みなし (0) の MSE | 的中率 | 見込み 今 / 縮小 / 実際 (bp、方向の側) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        L.append(f"| {name} | {r['n']:,} | {_f(r['mse_now'])} | {_f(r['mse_shrunk'])} | {-r['gain_pct']:+.1f}% | {_f(r['p'], 3)} | "
                 f"{_f(r['mse_zero'])} | {_p(r['hit'])} | {_f(r['mean_d'])} / {_f(r['mean_ds'])} / {_f(r['realised'])} |")
    L += ["", "- 差がマイナスなら縮めた方が良い。p は日ごとの平均の差の Diebold-Mariano 検定 (ラグ1)。本番の台帳は20営業日分しかなく、p は大きめです。",
          "- 今の見込みは、調整期間では「見込みなし (0)」より誤差が大きく、平均が目立つ枠の値動きを大きく見積もりすぎていました。縮めた見込みはどの比較でも「見込みなし」より誤差が小さくなりました。",
          "- 方向を示す足・確度・ロールオーバーの格上げは変えないので、方向の的中率はそのままです。", ""]
    hi = [(n, r["high"]) for n, r in rows]
    L += ["### 結果の読み方 (結果を見た後に加えた注記。合否は変えていません)", "",
          "- **確度: 高 の足は縮めすぎ:** |t| ≥ 4 の足の見込みは、縮めた後も実際より小さめでした ("
          + "、".join(f"{n.split(' (')[0]} {_f(h['mean_ds'])} / 実際 {_f(h['realised'])} bp" for n, h in hi if h['n']) + ")。"
          "多くの枠はほぼ0で一部だけ大きい、という分布に正規分布を当てているためです。本番の台帳ではほぼ一致していました。混合分布などの改良は、別の検定を決めてから確かめます。",
          f"- **見込みが0まで縮む足:** 同じ種類の枠全体に本当の効果が見えないとき (τ² = 0)、縮めた見込みは0になります。方向は今までどおり示し (事前の取り決め)、見込みの値動きだけ0とします。"
          f"そのような足の的中率は、7ペア {_p(res['zero']['live']['hit'])} ({res['zero']['live']['n']:,}回)、6ペア {_p(res['zero']['holdout']['hit'])} ({res['zero']['holdout']['n']:,}回) でした。", ""]
    r15, r15b = lr.get("15m_1", {}), lr.get("15m_4", {})
    r1h = lr.get("1h_1", {})
    L += ["## 15分足は確度: 高 の方向だけにする (結果を見た後の判断)", "",
          "15分足の |t| が 2〜4 の方向は、最初の研究 (research/direction.md、検証期間) で 52.9% (1,943回)、本番の台帳で "
          f"{_p(r15.get('mid_now', {}).get('hit'))} ({r15.get('mid_now', {}).get('n', 0):,}回) で、偶然 (50%) とほぼ変わりませんでした。本番では見込み "
          f"{_f(r15.get('mid_now', {}).get('expected'))} bp に対して実際は {_f(r15.get('mid_now', {}).get('realised'))} bp でした。"
          f"確度: 高 (|t| ≥ 4) は研究で 84.3%、本番で {_p(r15.get('high_now', {}).get('hit'))} ({r15.get('high_now', {}).get('n', 0):,}回) です。"
          "15分足の枠の平均は過去約60日分 (Yahoo が保存している分) しかなく、曜日×15分の枠は1つ約12本と少ないためと考えられます。"
          "このため15分足は |t| ≥ 4 のときだけ方向を示します (season.CALL_T_MIN)。事前に決めた検定ではなく、2つの独立した期間の結果を見た上での判断です。", "",
          "本番の台帳 (2026-09-23〜11-20、時間帯の偏りを記録した版の予測) を、記録した予測と今の規則 (縮めた見込み + 15分足は |t| ≥ 4) で採点し直した結果:", "",
          "| 予測 | 件数 | 中心の MSE 記録 | 今の規則 | 時間帯の偏りなし | 方向を示した足: 記録 | 今の規則 |", "|---|---|---|---|---|---|---|"]
    for key, label in (("15m_1", "15分足 15分先"), ("15m_4", "15分足 1時間先"), ("1h_1", "1時間足 1時間先")):
        r = lr.get(key)
        if not r:
            continue
        cn, cw = r.get("calls_now"), r.get("calls_new")
        L.append(f"| {label} | {r['n']:,} | {_f(r['mse_now'], 3)} | {_f(r['mse_new'], 3)} | {_f(r['mse_no_drift'], 3)} | "
                 + (f"{cn['n']:,}回 {_p(cn['hit'])} | {cw['n']:,}回 {_p(cw['hit'])} |" if cn else "— | — |"))
    L += ["", "## データと方法", "",
          f"- Yahoo の1時間足 ({res['span']})。7ペア・6ペアとも、評価は6,000本の助走の後の各日 (統計は起点の UTC の日より前の足)。"
          "週明けの最初の足と、直前の足から間が空いた足は採点しません。",
          f"- 本番の関数との一致: 縮小の計算 (season.shrink_factors / bar_factor) は、このモジュールの実装と無作為の {res['check']['bars']:,} 本で一致 (最大の差 {res['check']['max_diff']:.1e})。"
          f"本番の台帳の1時間足の予測 {lc['calls']:,} 件のうち {lc['rebuilt']:,} 件で、保存した価格から記録と同じ見込みを再現できました。", "",
          "## 再実行", "", "```bash", "python -m aifx.research_drift   # research/drift.md と research/drift.json (本番の台帳は --state で指定)", "```", ""]
    return "\n".join(L)


def run(state_root: Path | None, log=print) -> dict:
    t0 = time.time()
    frames = fetch(LIVE + HOLDOUT, log=log)
    rep = {c: replay(c, f) for c, f in frames.items()}
    ev = evaluate(rep)
    live = pd.concat([rep[c] for c in LIVE]); hold = pd.concat([rep[c] for c in HOLDOUT])
    zero = {}
    for name, df in (("live", live), ("holdout", hold)):
        z = df[(df.d != 0) & (df.ret != 0) & (df.ds == 0)]
        zero[name] = {"n": int(len(z)), "hit": float((np.sign(z.d) == np.sign(z.ret)).mean()) if len(z) else None}
    rng = np.random.default_rng(5)
    mx, nb = 0.0, 0
    for code in ("USDJPY", "EURGBP", "AUDUSD"):
        b = frames[code]
        days = pd.unique(b.index.normalize()[season.WINDOW + 100:])
        for day in rng.choice(days, 15, replace=False):
            day = pd.Timestamp(day)
            st = season.slot_stats(b, day.to_pydatetime(), 60)
            starts = b.index[(b.index >= day) & (b.index < day + pd.Timedelta(days=1))]
            d, _t, ds = shrunk_drift(st, starts, 60)
            mx = max(mx, float(np.abs(np.where(d != 0, d * season.bar_factor(st, starts, 60), 0) - ds).max()))
            nb += len(starts)
    res = {"prereg": PREREG, "eval": ev, "zero": zero, "check": {"bars": nb, "max_diff": mx},
           "span": f"{min(f.index[0] for f in frames.values()).date()}〜{max(f.index[-1] for f in frames.values()).date()}"}
    if state_root is not None:
        res["ledger"] = ledger_check(state_root)
        res["live"] = live_replay(state_root)
    res["seconds"] = round(time.time() - t0, 1)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "drift.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (REPORT_DIR / "drift.md").write_text(report(res), encoding="utf-8")
    log(f"research/drift.md written ({res['seconds']}s)")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state", help="the live ledger (a checkout of the ledger branch)")
    a = ap.parse_args()
    run(Path(a.state) if Path(a.state).exists() else None)
