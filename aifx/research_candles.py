"""How accurate are the forecast candles (scenario.py)? Tested on the later part of the history.

For each origin, the forecast candles are built from the bars up to the
origin only and compared with the real candles that followed: direction of
the close (against the origin price), colour of each candle, and the size
of each candle (high minus low) against a plain 14-bar average range (ATR).

    aifx research --candles    # writes research/candles.md and research/candles.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import history, scenario, season
from .data import PAIRS

REPORT_DIR = Path("research")
SETUP = {"15m": (16, 15, 5), "1h": (24, 60, 7), "1d": (20, 0, 3)}     # steps, bar minutes, origin every n bars


def _load(tf: str, code: str) -> pd.DataFrame:
    df = history.load_hourly(code) if tf == "1h" else history.load_intraday(code, "15m") if tf == "15m" else history.load_daily(code)
    df = df[~df.index.duplicated()].sort_index()
    return df[df.index >= "2010-01-01"] if tf == "1d" else df


def evaluate(tf: str, log=print) -> dict:
    steps, minutes, every = SETUP[tf]
    rows = []
    span = None
    for code in PAIRS:
        df = _load(tf, code)
        n = len(df)
        start = int(n * 0.6)
        o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        atr = pd.Series(h - lo).rolling(14).mean().to_numpy()
        span = (str(df.index[start])[:10], str(df.index[-1])[:10])
        for t in range(start, n - steps, every):
            origin = (df.index[t] + pd.Timedelta(minutes=minutes)).to_pydatetime() if minutes else None
            drift = season.step_drift(minutes, df, origin, steps)[0] if minutes else np.zeros(steps)
            end = season.centre_drift(drift, minutes)[-1] if minutes else 0.0
            cs, _ = scenario.candles(tf, df.iloc[: t + 1], steps, c[t] * np.exp(end / 1e4), minutes, drift)
            if not cs:
                continue
            for j, (po, ph, pl, pc) in enumerate(cs):
                k = t + 1 + j
                rows.append((j + 1, np.sign(pc - c[t]), np.sign(c[k] - c[t]), np.sign(pc - po), np.sign(c[k] - o[k]),
                             ph - pl, h[k] - lo[k], atr[t], np.sign(drift[j])))
        if log:
            log(f"{tf} {code}: {len(rows)} rows")
    R = pd.DataFrame(rows, columns=["j", "pd", "ad", "pb", "ab", "pr", "ar", "atr", "call"])
    out = {"tf": tf, "test": span, "steps": steps, "h": {}}
    for j in sorted({1, 4, steps}):
        s = R[R.j == j]
        m1 = (s.pd != 0) & (s.ad != 0)
        m2 = (s.pb != 0) & (s.ab != 0)
        m3 = (s.call != 0) & (s.ab != 0)
        out["h"][str(j)] = {"n": int(len(s)), "dir_hit": float((s.pd[m1] == s.ad[m1]).mean()),
                            "color_hit": float((s.pb[m2] == s.ab[m2]).mean()),
                            "call_share": float((s.call != 0).mean()),
                            "call_hit": float((s.call[m3] == s.ab[m3]).mean()) if m3.any() else None,
                            "other_color_hit": float((s.pb[m2 & (s.call == 0)] == s.ab[m2 & (s.call == 0)]).mean()),
                            "size_corr": float(np.corrcoef(s.pr, s.ar)[0, 1]),
                            "atr_corr": float(np.corrcoef(s.atr, s.ar)[0, 1]),
                            "size_mae_vs_atr": float(np.mean(np.abs(s.pr - s.ar)) / np.mean(np.abs(s.atr - s.ar)))}
    return out


def report(res: dict, ml: dict | None = None) -> str:
    L = ["# 予想ローソク足の精度", "",
         "予想ローソク足の各足の大きさ (高値−安値) は、今の値幅の水準 (値幅の指数加重平均) に、その時間帯 (日足は曜日) の"
         "ふだんの比率 (直近500回の中央値) を掛けたものです。各足の形 (どこで引けるか、ヒゲの長さ) は、直近の値動きの形 "
         "(値動きの大きさで割った各足の動き) と時間帯が似た過去の局面を30個選び、その続きのうち最も代表的なもの "
         "(他との差が最も小さいもの) から取り、全体の水準は予想の中心 (検証で最も誤差が小さかった値) で終わるように傾けています。"
         "15分足と1時間足では、時間帯の偏り ([direction.md](direction.md)) が方向を示す足だけ、陽線・陰線をその向きにそろえます。", "",
         "履歴の後半4割で、各時点までのデータだけから作った予想ローソク足を、その後の実際のローソク足と比べました。", "",
         "| 時間足 | 何本先 | 件数 | 終値の方向の的中率 | 陽線・陰線の的中率 | うち方向を示した足: 割合 / 的中率 | それ以外の足 | "
         "足の大きさの相関 (ATR) | 足の大きさの誤差 (ATR比) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tf, r in res.items():
        name = {"1h": "1時間足", "1d": "日足", "15m": "15分足"}[tf]
        for j, x in r["h"].items():
            call = (f"{x['call_share']:.0%} / {x['call_hit']:.1%}" if x.get("call_hit") is not None else "—")
            other = f"{x['other_color_hit']:.1%}" if x.get("other_color_hit") is not None else "—"
            L.append(f"| {name} | {j} | {x['n']:,} | {x['dir_hit']:.1%} | {x['color_hit']:.1%} | {call} | {other} | "
                     f"{x['size_corr']:.2f} ({x['atr_corr']:.2f}) | {x['size_mae_vs_atr'] - 1:+.0%} |")
    L += ["", "- 足の大きさは、単純な平均値幅 (直近14本の ATR) より正確に予想できています (誤差がマイナスなら ATR より小さい)。",
          "- 足の大きさの計算方法 (水準の重み 0.9 / 0.95、時間帯のみ / 時間帯×曜日、500回 / 250回 / 1000回) は数通りの候補を"
          "この期間で比べて選んだため、わずかに良く見えている可能性があります (候補の間の差は 1〜2% 程度)。",
          "- 陽線・陰線は、時間帯の偏りが方向を示した足 (1時間足の次の足ではおよそ6本に1本) だけ偶然より当たります。"
          "それ以外の足の向きや並びは、過去の似た局面の一例であって、当てにできるものではありません。", ""]
    if ml:
        L += ["## 機械学習による方向の予測", "",
              "予想ローソク足の方向を当てられないか、29種類の特徴 (複数の期間の値動き、RSI、移動平均からの乖離、値動きの荒さ、時間帯、"
              "7ペアから計算した通貨の強さ、金利差) を使った LightGBM とロジスティック回帰でも検証しました (詳細は [ml.md](ml.md))。"
              "検証期間の AUC はどれも 0.49〜0.51、確率予想の Brier スキルはすべて 0 以下で、方向を偶然より当てることはできませんでした。", ""]
    return "\n".join(L)


def run(log=print) -> dict:
    res = {tf: evaluate(tf, log=log) for tf in ("1h", "15m", "1d")}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "candles.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    ml = None
    try:
        ml = json.loads((REPORT_DIR / "ml.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    (REPORT_DIR / "candles.md").write_text(report(res, ml), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
