"""Hourly forecast ranges: which time-of-day profile of volatility forecasts best?

The spread of an hourly forecast is the recent level of volatility times the
usual share of it at each future hour (volatility.py). This compares the
profile the server used before (UTC hour of day over the last 1,500 bars) with
New York time (whose daylight saving the market sessions and the 17:00 roll
follow) and New York weekday-and-hour over about a year of bars, each pulled
toward its hour. Every forecast is made exactly as the server does
(``hourly_variance_path``) from the bars up to its origin, every fifth hour of
the history for the 7 pairs.

Scores for 1, 4 and 24 hours ahead: QLIKE (log of the forecast variance plus
the squared move over it; lower is better) with one scale per profile fitted
on the first 60 % of the history (the live learner calibrates the scale the
same way), and on the last 40 %: the share of moves inside the 80 % band built
with the tune-period 80 % quantile, and the band's average width.

    aifx research --vol     # writes research/volatility.md and research/volatility.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS
from .volatility import RANGE_WINDOW, hourly_variance_path, range_variance, scale_proxy

REPORT_DIR = Path("research")
NY = ZoneInfo("America/New_York")
VARIANTS = {
    "utc_hour": ("UTC の時間 (24区分)、直近1,500本 (以前の版)", dict(tail=3000, profile_window=1500)),
    "ny_hour": ("ニューヨーク時間の時間、直近6,000本", dict(tail=6500, profile_window=6000, profile_tz=NY)),
    "ny_week_hour": ("ニューヨーク時間の曜日×時間、直近6,000本 (採用)",
                     dict(tail=6500, profile_window=6000, profile_tz=NY, profile_weekday=True)),
}
H = (1, 4, 24)
EVERY = 5
TUNE_SHARE = 0.6


def evaluate(log=print) -> dict:
    rows = []
    for code in PAIRS:
        df = history.load_hourly(code)
        df = df[~df.index.duplicated()].sort_index()
        n, c, idx = len(df), df["close"].to_numpy(float), df.index
        split = int(n * TUNE_SHARE)
        for o in range(3000, n - max(H) - 1, EVERY):
            origin = (idx[o] + pd.Timedelta(hours=1)).to_pydatetime()
            row = {"test": o >= split, **{f"a{h}": math.log(c[o + h] / c[o]) * 1e4 for h in H}}
            for name, (_, cfg) in VARIANTS.items():
                cfg = dict(cfg)
                tail = df.iloc[max(0, o + 1 - cfg.pop("tail")): o + 1]
                yt = np.log(tail["close"].to_numpy(float))
                sq = scale_proxy(range_variance(tail), np.diff(yt), RANGE_WINDOW)
                var, _ = hourly_variance_path(list(tail.index.to_pydatetime()), yt, origin, max(H), None, sq=sq, **cfg)
                cum = np.cumsum(var) * 1e8
                for h in H:
                    row[f"{name}_{h}"] = float(cum[h - 1])
            rows.append(row)
        if log:
            log(f"{code}: {len(rows)} forecasts")
    R = pd.DataFrame(rows)
    out = {"every": EVERY, "n_tune": int((~R.test).sum()), "n_test": int(R.test.sum()), "h": {}}
    tu, te = ~R.test, R.test
    for h in H:
        a = R[f"a{h}"]
        for name in VARIANTS:
            v = R[f"{name}_{h}"]
            k = float(np.mean(a[tu] ** 2 / v[tu]))
            q = {p: float(np.mean(np.log(v[m] * k) + a[m] ** 2 / (v[m] * k))) for p, m in (("tune", tu), ("test", te))}
            z = float(np.quantile(np.abs(a[tu]) / np.sqrt(v[tu]), 0.8))
            out["h"].setdefault(str(h), {})[name] = {
                "qlike_tune": q["tune"], "qlike_test": q["test"],
                "cover80_test": float(np.mean(np.abs(a[te]) <= z * np.sqrt(v[te]))),
                "width80_test_bp": float(np.mean(2 * z * np.sqrt(v[te])))}
    return out


def report(res: dict) -> str:
    L = ["# 1時間足の予測レンジ: 時間帯ごとの値動きの荒さの測り方", "",
         "1時間足の予測レンジは「今の値動きの荒さ」に「その時間帯のふだんの荒さの割合」を掛けて作ります (aifx/volatility.py)。"
         "この割合をどの時計・どの区切りで測るのがよいかを、サーバーと同じ計算で比べました。予測はそれぞれの時点までの足だけから作り、"
         f"7ペアの履歴で5時間おきに {res['n_tune'] + res['n_test']:,} 回 (調整 {res['n_tune']:,}、検証 {res['n_test']:,})。", "",
         "- QLIKE: 予測した分散の対数 + 実際の値動きの2乗 ÷ 予測した分散 の平均。小さいほど、値動きの大きさを正しく予測できています。"
         "水準は調整期間 (最初の60%) で合わせています (本番でも実績で合わせています)。",
         "- 80%レンジ: 調整期間で80%が入るように決めた幅を検証期間に当てたときの、入った割合と平均の幅。", "",
         "| 何時間先 | 測り方 | QLIKE 調整 | QLIKE 検証 | 検証: 80%レンジに入った割合 | 平均の幅 (bp) |", "|---|---|---|---|---|---|"]
    for h, per in res["h"].items():
        for name, x in per.items():
            L.append(f"| {h} | {VARIANTS[name][0]} | {x['qlike_tune']:.4f} | {x['qlike_test']:.4f} | "
                     f"{x['cover80_test']:.1%} | {x['width80_test_bp']:.1f} |")
    L += ["", "ニューヨーク時間の曜日×時間 (各枠は時間ごとの値に20本分の重みで引き寄せる) は、1時間先・4時間先で調整期間と検証期間の"
          "どちらでも QLIKE が小さく、24時間先はほぼ同じでした。ロンドン・ニューヨークの取引時間やロールオーバーは夏時間で UTC の時刻が"
          "1時間ずれ、月曜のアジア時間や金曜の午後は他の曜日より静かなためです。改善の幅は小さいものの、両方の期間でそろって良いため"
          "本番に採用しました。", ""]
    return "\n".join(L)


def run(log=print) -> dict:
    res = evaluate(log=log)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "volatility.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORT_DIR / "volatility.md").write_text(report(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
