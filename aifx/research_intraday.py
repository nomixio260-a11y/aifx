"""Walk-forward research on 15-minute bars (Yahoo keeps about 60 days of 15m and 5m bars).

Two questions:

* 15-minute forecasts (15 min, 1 hour and 4 hours ahead): can the direction
  be predicted, and which volatility model gives the best ranges? Candidate
  variance measurements per bar are the squared close-to-close return, the
  high-low range and the realized variance from 5-minute bars; the
  time-of-day profile uses 15- or 60-minute slots.
* Hourly forecasts: does measuring each hour's variance from 15- or
  5-minute bars beat the high-low range now used live?

As in the long-history research, every setting is chosen on the first 60 %
of the period and scored on the last 40 %, which is never used for choosing.

    aifx research --intraday
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS
from .engine import BP, MODEL_KEYS, model_paths
from .learning import GAIN_ESS, GAIN_LAMBDA
from .research import (OUT_DIR, REPORT_DIR, _crps, _dm_by_date, _k80, _p, _pct, fitted_centre, learner_sim,
                       point_metrics, shape_metrics)
from .volatility import RANGE_WINDOW, intraday_variance_path, range_variance, scale_proxy

M15_H = (1, 4, 16)
FIT_15 = 2000          # bars handed to the models (~21 trading days)
STEP_15 = 2            # an origin every 30 minutes
PROFILE_WINDOW_15 = 1920   # 20 trading days of 15-minute bars
H1_H = (1, 4, 24)


def vol15_variants() -> dict[str, tuple]:
    out = {}
    for proxy in ("cc", "range", "rv5"):
        for pm in (15, 60):
            for lam in (0.97, 0.99):
                for rev in (0.99, 0.997):
                    out[f"{proxy}_p{pm}_{lam:g}_{rev:g}"] = (proxy, pm, lam, rev)
    return out


def intra_rv(coarse: pd.DataFrame, fine: pd.DataFrame, minutes: int, fine_minutes: int) -> np.ndarray:
    """Realized variance of each coarse bar from the finer bars inside it, aligned with
    diff(log coarse close); NaN where the fine bars do not cover the whole coarse bar."""
    r = np.diff(np.log(fine["close"].to_numpy(dtype=float)))
    bucket = fine.index[1:].floor(f"{minutes}min")
    g = pd.DataFrame({"sq": r * r, "b": bucket}).groupby("b")["sq"].agg(["sum", "count"])
    full = g["sum"].where(g["count"] >= minutes // fine_minutes)
    return full.reindex(coarse.index[1:]).to_numpy(dtype=float)


def _m15_pair(code: str) -> dict:
    df = history.load_intraday(code, "15m")
    df5 = history.load_intraday(code, "5m")
    y = np.log(df["close"].to_numpy())
    times = list(df.index.to_pydatetime())
    pk = range_variance(df)
    rv5 = intra_rv(df, df5, 15, 5)
    H = max(M15_H)
    variants = vol15_variants()
    rows = []
    for o in range(FIT_15, len(y) - 1 - H, STEP_15):
        lo = o + 1 - FIT_15
        hist = y[lo: o + 1]
        r = np.diff(hist)
        paths, _ = model_paths(hist, H)
        origin = times[o] + timedelta(minutes=15)
        proxies = {"cc": None,
                   "range": scale_proxy(pk[lo:o], r, RANGE_WINDOW),
                   "rv5": scale_proxy(rv5[lo:o], r, RANGE_WINDOW)}
        sig = {}
        for name, (proxy, pm, lam, rev) in variants.items():
            v, _ = intraday_variance_path(times[lo: o + 1], hist, origin, H, 15, None, lam, rev, True, proxies[proxy],
                                          PROFILE_WINDOW_15, 0.25, pm)
            cum = np.cumsum(v) * BP * BP
            sig[name] = [float(math.sqrt(cum[h - 1])) for h in M15_H]
        rows.append({
            "t": times[o].strftime("%Y-%m-%dT%H:%M"),
            "m": [[float(paths[k][h - 1]) for k in MODEL_KEYS] for h in M15_H],
            "a": [float((y[o + h] - y[o]) * BP) for h in M15_H],
            "s": sig,
            "r1": float(r[-1] * BP),
            "hour": times[o].hour,
        })
    return {"pair": code, "tf": "15m", "rows": rows}


def _h1rv_pair(code: str) -> dict:
    """Hourly forecasts over the ~60 days with 15m/5m bars, with intraday variance measurements."""
    df = history.load_intraday(code, "1h")
    df15 = history.load_intraday(code, "15m")
    df5 = history.load_intraday(code, "5m")
    y = np.log(df["close"].to_numpy())
    times = list(df.index.to_pydatetime())
    pk = range_variance(df)
    rv15 = intra_rv(df, df15, 60, 15)
    rv5 = intra_rv(df, df5, 60, 5)
    H = max(H1_H)
    start = int(np.searchsorted(df.index, df15.index[0] + pd.Timedelta(days=5)))
    rows = []
    for o in range(max(start, 3000), len(y) - 1 - H):
        lo = o + 1 - 3000
        hist = y[lo: o + 1]
        r = np.diff(hist)
        origin = times[o] + timedelta(hours=1)
        rng = scale_proxy(pk[lo:o], r, RANGE_WINDOW)
        a5 = scale_proxy(rv5[lo:o], r, RANGE_WINDOW)
        a15 = scale_proxy(rv15[lo:o], r, RANGE_WINDOW)
        variants = {"range": (rng, 0.97), "rv15": (a15, 0.97), "rv5": (a5, 0.97), "rv5_fast": (a5, 0.94),
                    "range_rv5": (None if rng is None or a5 is None else 0.5 * (rng + a5), 0.97)}
        sig = {}
        for name, (sq, lam) in variants.items():
            v, _ = intraday_variance_path(times[lo: o + 1], hist, origin, H, 60, None, lam, 0.985, True, sq)
            cum = np.cumsum(v) * BP * BP
            sig[name] = [float(math.sqrt(cum[h - 1])) for h in H1_H]
        rows.append({"t": times[o].strftime("%Y-%m-%dT%H:%M"), "a": [float((y[o + h] - y[o]) * BP) for h in H1_H],
                     "s": sig})
    return {"pair": code, "tf": "1h_rv", "rows": rows}


_JOBS = {"15m": _m15_pair, "1h_rv": _h1rv_pair}


def _run(job):
    code, tf = job
    path = OUT_DIR / f"{code}_{tf}.json"
    if path.exists():
        return str(path)
    path.write_text(json.dumps(_JOBS[tf](code)), encoding="utf-8")
    return str(path)


def compute(workers: int = 4, log=print) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(c, tf) for tf in _JOBS for c in PAIRS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path in ex.map(_run, jobs):
            log(f"done {path}")


class IPanel:
    """Walk-forward rows of all pairs for one intraday study, split 60/40 in time."""

    def __init__(self, tf: str):
        rows, pairs = [], []
        for code in PAIRS:
            data = json.loads((OUT_DIR / f"{code}_{tf}.json").read_text())
            rows.extend(data["rows"])
            pairs.extend([code] * len(data["rows"]))
        self.tf = tf
        self.h = M15_H if tf == "15m" else H1_H
        self.step = STEP_15 if tf == "15m" else 1
        self.pair = np.array(pairs)
        self.t = np.array([r["t"] for r in rows])
        self.time = self.t.astype("datetime64[m]")
        self.a = np.array([r["a"] for r in rows])
        self.s = {k: np.array([r["s"][k] for r in rows]) for k in rows[0]["s"]}
        if "m" in rows[0]:
            self.m = np.array([r["m"] for r in rows])
            self.r1 = np.array([r["r1"] for r in rows])
            self.hour = np.array([r["hour"] for r in rows])
        cut = np.quantile(self.time.astype("int64"), 0.6)
        self.tune = self.time.astype("int64") < cut
        self.test = ~self.tune


def _vol_table(P, base: str) -> dict:
    out = {}
    for key in P.s:
        res = {}
        for j, h in enumerate(P.h):
            tune, test = _crps(P, key, j, P.tune), _crps(P, key, j, P.test)
            btune, btest = _crps(P, base, j, P.tune), _crps(P, base, j, P.test)
            _, p = _dm_by_date(P.t[P.test], test, btest, max(0, math.ceil(h / P.step) - 1))
            z = np.abs(P.a[P.test, j]) / (P.s[key][P.test, j] * _k80(P, key, j))
            res[str(h)] = {"tune_rel": float(tune.mean() / btune.mean() - 1), "test_rel": float(test.mean() / btest.mean() - 1),
                           "p": None if key == base else p, "cover80": float(np.mean(z <= 1.2816)),
                           "tune_crps": float(tune.mean()), "test_crps": float(test.mean())}
        out[key] = res
    return out


def evaluate_15m() -> dict:
    P = IPanel("15m")
    zero = np.zeros(P.a.shape)
    ens = P.m.mean(axis=2)
    base = "cc_p15_0.97_0.99"
    vol = _vol_table(P, base)
    chosen = min(vol, key=lambda k: np.mean([vol[k][str(h)]["tune_rel"] for h in P.h]))
    cands = {"変化なし (ランダムウォーク)": zero, "6モデルの均等平均": ens}
    for i, k in enumerate(MODEL_KEYS[1:], start=1):
        cands[f"単独モデル: {k}"] = P.m[:, :, i]
    s1 = P.s[chosen]
    f = np.repeat((P.r1 / s1[:, 0])[:, None], len(P.h), axis=1)[:, :, None] * s1[:, :, None]
    c, co = fitted_centre(P, f, sig=chosen)
    cands["直前15分の値動き (反転/継続)"] = c
    c, co_g = fitted_centre(P, ens[:, :, None], sig=chosen)
    cands["均等平均 × ゲイン (調整期間で推定)"] = c
    cands[f"本番の学習ルール (λ={GAIN_LAMBDA:g}, 直近14日)"] = learner_sim(P, 14, GAIN_LAMBDA, GAIN_ESS, chosen)
    point = {name: {"tune": point_metrics(P, c, P.tune), "test": point_metrics(P, c, P.test)} for name, c in cands.items()}
    ranked = sorted(vol, key=lambda k: np.mean([vol[k][str(h)]["tune_rel"] for h in P.h]))
    full = np.ones(len(P.t), bool)
    return {
        "n": int(len(P.t)), "tune": [str(min(P.t[P.tune]))[:16], str(max(P.t[P.tune]))[:16]],
        "test": [str(min(P.t[P.test]))[:16], str(max(P.t[P.test]))[:16]],
        "base": base, "chosen": chosen, "ranked": ranked, "vol": vol, "point": point,
        "coefs": {"r1": co, "gain": co_g},
        "shape": {"before": shape_metrics(P, base), "after": shape_metrics(P, chosen)},
        "deployed_nu": {h: x["nu"] for h, x in shape_metrics(P, chosen, full).items()},
    }


def evaluate_h1rv() -> dict:
    P = IPanel("1h_rv")
    vol = _vol_table(P, "range")
    ranked = sorted((k for k in vol if k != "range"), key=lambda k: np.mean([vol[k][str(h)]["tune_rel"] for h in P.h]))
    return {"n": int(len(P.t)), "tune": [str(min(P.t[P.tune]))[:16], str(max(P.t[P.tune]))[:16]],
            "test": [str(min(P.t[P.test]))[:16], str(max(P.t[P.test]))[:16]], "vol": vol, "ranked": ranked}


VOL_LABEL = {"cc": "終値の変化", "range": "高値・安値の幅", "rv5": "5分足の変動"}


def _vol_name(key: str) -> str:
    if key in ("range", "rv15", "rv5", "rv5_fast", "range_rv5"):
        return {"range": "高値・安値の幅 (現在の方式)", "rv15": "15分足の変動", "rv5": "5分足の変動",
                "rv5_fast": "5分足の変動 (速く反応)", "range_rv5": "高値・安値の幅と5分足の平均"}[key]
    proxy, pm, lam, rev = key.split("_")
    return f"{VOL_LABEL[proxy]}・時間帯{pm[1:]}分刻み・λ{lam}・戻り{rev}"


def report(res: dict) -> str:
    m, h = res["15m"], res["1h_rv"]
    L = ["# 15分足の検証 (直近約60日)", "",
         "Yahoo Finance に残っている約60日分の15分足・5分足を使い、各時点までのデータだけで予測しました"
         f" (7ペア、30分ごと、{m['n']:,}件)。設定は前半6割 (調整期間 {m['tune'][0]}〜{m['tune'][1]}) で選び、"
         f"後半4割 (検証期間 {m['test'][0]}〜{m['test'][1]}) で確認しています。期間が短いので、日足・1時間足の検証より不確かさは大きめです。", "",
         "再実行: `aifx research --download && aifx research --intraday`", "",
         "## 方向 (15分足)", "", "各欄: 誤差改善率 / 方向的中率 / p値 (検証期間)", "",
         "| 方法 | 15分先 | 1時間先 | 4時間先 |", "|---|---|---|---|"]
    for name, x in m["point"].items():
        cells = []
        for hh in ("1", "4", "16"):
            v = x["test"][hh]
            hit = "—" if v["hit"] is None else f"{v['hit'] * 100:.1f}%"
            cells.append(f"{_pct(v['skill'])} / {hit} / p={_p(v['dm_p'])}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 値動きの大きさの推定 (15分足)", "",
          f"比較の基準は「{_vol_name(m['base'])}」。各欄: 調整期間の CRPS 差 / 検証期間の CRPS 差 (p値)。調整期間の成績順に上位8件。", "",
          "| 推定方法 | 15分先 | 1時間先 | 4時間先 |", "|---|---|---|---|"]
    for key in m["ranked"][:8]:
        v = m["vol"][key]
        L.append(f"| {_vol_name(key)} | " + " | ".join(
            f"{_pct(v[hh]['tune_rel'])} / {_pct(v[hh]['test_rel'])} (p={_p(v[hh]['p'])})" for hh in ("1", "4", "16")) + " |")
    L += ["", f"採用: 調整期間で1位の「{_vol_name(m['chosen'])}」。", "",
          "## レンジの形 (15分足)", "", "| 予測先 | 50% (前→後) | 80% (前→後) | 95% (前→後) |", "|---|---|---|---|"]
    for hh, lab in (("1", "15分先"), ("4", "1時間先"), ("16", "4時間先")):
        b, a = m["shape"]["before"][hh]["normal"]["cover"], m["shape"]["after"][hh]["t"]["cover"]
        L.append(f"| {lab} | {b['50'] * 100:.1f}% → {a['50'] * 100:.1f}% | {b['80'] * 100:.1f}% → {a['80'] * 100:.1f}% | "
                 f"{b['95'] * 100:.1f}% → {a['95'] * 100:.1f}% |")
    L += ["", "前 = 基準の変動推定 + 正規分布、後 = 採用した変動推定 + t 分布 (検証期間)。本番の t 分布の自由度は全期間で推定: "
          + ", ".join(f"{lab} ν={m['deployed_nu'][hh]}" for hh, lab in (("1", "15分先"), ("4", "1時間先"), ("16", "4時間先"))) + "。", "",
          "## 1時間足: 15分足・5分足で測った変動", "",
          f"1時間足の予測 ({h['n']:,}件) で、各1時間の変動を細かい足から測る方法を、現在の「高値・安値の幅」と比べました。"
          f" 調整期間 {h['tune'][0]}〜{h['tune'][1]}、検証期間 {h['test'][0]}〜{h['test'][1]}。", "",
          "| 推定方法 | 1時間先 | 4時間先 | 24時間先 |", "|---|---|---|---|"]
    for key in h["ranked"]:
        v = h["vol"][key]
        L.append(f"| {_vol_name(key)} | " + " | ".join(
            f"{_pct(v[hh]['tune_rel'])} / {_pct(v[hh]['test_rel'])} (p={_p(v[hh]['p'])})" for hh in ("1", "4", "24")) + " |")
    L.append("")
    return "\n".join(L)


def run(workers: int = 4, log=print) -> dict:
    compute(workers, log)
    res = {"15m": evaluate_15m(), "1h_rv": evaluate_h1rv()}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "intraday.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (REPORT_DIR / "intraday.md").write_text(report(res), encoding="utf-8")
    log(f"wrote {REPORT_DIR}/intraday.md")
    return res
