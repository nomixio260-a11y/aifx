"""Machine-learned direction forecasts, tested walk-forward: can they beat a coin flip?

Features known at each bar's close, pooled over the 7 pairs: returns over
several lookbacks (in units of recent volatility), RSI, distance from moving
averages, volatility regime, bar range, time of day and weekday, currency
strength built from all pairs (the pair's base minus quote), and the
interest-rate difference. Target: the sign of the move over the next H bars.

Models: gradient-boosted trees (LightGBM) and a regularised logistic
regression. Each is trained on data before the test period only, then
retrained every month of the test period on everything before that month
(as the live server would), with a gap of H bars so no training target
overlaps a test bar. Scores on the test period: hit rate, AUC, Brier skill
and the average signed move (pips per forecast), with a significance test
by blocks of days (targets of neighbouring bars and pairs overlap).

    aifx research --ml      # writes research/ml.md and research/ml.json (needs scikit-learn and lightgbm)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS

REPORT_DIR = Path("research")
CURS = ("USD", "JPY", "EUR", "GBP", "AUD")
CONFIG = {
    "1h": {"lags": (1, 2, 4, 8, 24, 72, 120), "str_lags": (1, 4, 24, 120), "horizons": (1, 4, 24), "sma": (24, 120),
           "tune_share": 0.6, "retrain": "MS"},
    "1d": {"lags": (1, 5, 20, 60, 120, 250), "str_lags": (1, 5, 20, 60), "horizons": (1, 5, 20), "sma": (20, 100),
           "split": "2017-01-01", "retrain": "YS"},
    "15m": {"lags": (1, 2, 4, 8, 16, 96), "str_lags": (1, 4, 16, 96), "horizons": (1, 4, 16), "sma": (16, 96),
            "tune_share": 0.6, "retrain": "W-MON"},
}
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 400,
              "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0,
              "verbose": -1, "seed": 7, "deterministic": True, "num_threads": 4}
ROUNDS = 300


def _load(tf: str, code: str) -> pd.DataFrame:
    if tf == "1h":
        df = history.load_hourly(code)
    elif tf == "15m":
        df = history.load_intraday(code, "15m")
    else:
        df = history.load_daily(code)
        df = df[df.index >= "2001-01-01"]
    idx = df.index.tz_convert(None) if getattr(df.index, "tz", None) is not None else df.index
    df = df.copy()
    df.index = pd.DatetimeIndex(idx).as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def _ewm_sd(r: pd.Series, lam: float) -> pd.Series:
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def build(tf: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per (bar, pair): features at the bar's close and future moves."""
    cfg = CONFIG[tf]
    frames = {code: _load(tf, code) for code in PAIRS}
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    closes = pd.DataFrame({c: frames[c]["close"].reindex(common) for c in PAIRS})
    logc = np.log(closes)
    r1 = logc.diff()
    sig = r1.apply(lambda s: _ewm_sd(s, 0.97 if tf != "1d" else 0.94))
    # currency strength per lag: average of +z (as base) / -z (as quote) over the pairs holding the currency
    strength = {}
    for k in cfg["str_lags"]:
        z = (logc - logc.shift(k)) / (sig * math.sqrt(k))
        st = {}
        for cur in CURS:
            parts = []
            for code, pair in PAIRS.items():
                if pair.base == cur:
                    parts.append(z[code])
                elif pair.quote == cur:
                    parts.append(-z[code])
            st[cur] = pd.concat(parts, axis=1).mean(axis=1)
        strength[k] = pd.DataFrame(st)
    days = pd.DatetimeIndex(common.normalize().unique())
    rates = history.rates_panel(days)
    vix = history.load_vix() if tf == "1d" else None
    rows = []
    for code, pair in PAIRS.items():
        df = frames[code].reindex(common)
        c, h, lo = df["close"], df["high"], df["low"]
        s = sig[code]
        f = pd.DataFrame(index=common)
        for k in cfg["lags"]:
            f[f"z{k}"] = (np.log(c) - np.log(c.shift(k))) / (s * math.sqrt(k))
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        f["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50) / 50
        for n in cfg["sma"]:
            f[f"sma{n}"] = np.log(c / c.rolling(n).mean()) / s
        f["volr"] = _ewm_sd(np.log(c).diff(), 0.9) / _ewm_sd(np.log(c).diff(), 0.995)
        f["range"] = np.log(h / lo) / s
        f["body"] = np.log(c / df["open"]) / s
        if tf != "1d":
            hr = common.hour + common.minute / 60
            f["hsin"], f["hcos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        f["dow"] = common.dayofweek
        for k in cfg["str_lags"]:
            f[f"str{k}"] = strength[k][pair.base] - strength[k][pair.quote]
            f[f"strb{k}"] = strength[k][pair.base]
            f[f"strq{k}"] = strength[k][pair.quote]
        day = common.normalize()
        f["carry"] = (rates[pair.base] - rates[pair.quote]).reindex(day).to_numpy()
        if vix is not None:
            f["vix"] = vix.reindex(day, method="ffill").to_numpy()
        f["pair"] = list(PAIRS).index(code)
        for H in cfg["horizons"]:
            f[f"y{H}"] = np.log(c.shift(-H) / c)
        f["pip"] = pair.pip
        f["price"] = c
        f["code"] = code
        rows.append(f)
    data = pd.concat(rows).replace([np.inf, -np.inf], np.nan)
    feats = [c for c in data.columns if not c.startswith("y") and c not in ("pip", "price", "code")]
    warm = max(cfg["lags"]) + max(cfg["sma"])
    data = data[data.index >= common[min(warm, len(common) - 1)]]
    return data, feats


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _block_t(values: pd.Series) -> float:
    """t statistic of the mean with day blocks (neighbouring bars and pairs are correlated)."""
    daily = values.groupby(values.index.normalize()).sum()
    counts = values.groupby(values.index.normalize()).size()
    m = daily.sum() / counts.sum()
    resid = (values - m).groupby(values.index.normalize()).sum()
    se = math.sqrt((resid ** 2).sum()) / counts.sum()
    return float(m / se) if se > 0 else float("nan")


def score(test: pd.DataFrame, H: int, p: np.ndarray) -> dict:
    y = test[f"y{H}"].to_numpy()
    ok = np.isfinite(y) & np.isfinite(p) & (y != 0)
    y, p, sub = y[ok], p[ok], test[ok]
    up = (y > 0).astype(int)
    pred = (p > 0.5).astype(int)
    signed_pips = pd.Series(np.where(pred == 1, 1, -1) * (np.exp(y) - 1) * sub["price"].to_numpy() / sub["pip"].to_numpy(),
                            index=sub.index)
    conf = np.abs(p - 0.5)
    top = conf >= np.quantile(conf, 0.9)
    hits = pd.Series((pred == up).astype(float) - 0.5, index=sub.index)
    return {"n": int(ok.sum()), "hit": float(np.mean(pred == up)), "hit_t": _block_t(hits),
            "auc": _auc(up, p), "brier_skill": float(1 - np.mean((p - up) ** 2) / 0.25),
            "pips": float(signed_pips.mean()), "pips_t": _block_t(signed_pips),
            "top10_hit": float(np.mean(pred[top] == up[top])), "top10_pips": float(signed_pips[top].mean()),
            "base_up": float(up.mean())}


def _fit_predict(kind: str, train: pd.DataFrame, test: pd.DataFrame, feats: list[str], H: int) -> np.ndarray:
    y = train[f"y{H}"]
    ok = y.notna() & (y != 0)
    X, t = train.loc[ok, feats], (y[ok] > 0).astype(int)
    if kind == "lgbm":
        import lightgbm as lgb
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X, t, categorical_feature=["pair"]), num_boost_round=ROUNDS)
        return booster.predict(test[feats])
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = [f for f in feats if f != "pair"]
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    model.fit(X[cols], t)
    return model.predict_proba(test[cols])[:, 1]


def evaluate(tf: str, log=print) -> dict:
    cfg = CONFIG[tf]
    data, feats = build(tf)
    times = data.index.unique().sort_values()
    split = pd.Timestamp(cfg["split"]) if "split" in cfg else times[int(len(times) * cfg["tune_share"])]
    out = {"tf": tf, "start": str(times[0]), "split": str(split), "end": str(times[-1]), "features": feats,
           "rows": int(len(data)), "h": {}}
    months = pd.date_range(split, times[-1] + pd.Timedelta(days=1), freq=cfg["retrain"])
    edges = [split] + [m for m in months if m > split] + [times[-1] + pd.Timedelta(days=1)]
    for H in cfg["horizons"]:
        res = {}
        for kind in ("lgbm", "logit"):
            preds = []
            for a, b in zip(edges[:-1], edges[1:]):
                test = data[(data.index >= a) & (data.index < b)]
                if not len(test):
                    continue
                # leave H + 1 bars (counted in bars, so across weekends too) between training targets and the test
                cut = times[max(0, int(times.searchsorted(a)) - (H + 1))]
                train = data[data.index < cut]
                p = _fit_predict(kind, train, test, feats, H)
                preds.append(pd.Series(p, index=test.index))
            test_all = data[data.index >= split]
            p_all = np.concatenate([s.to_numpy() for s in preds])
            res[kind] = score(test_all, H, p_all)
            if log:
                r = res[kind]
                log(f"{tf} H={H} {kind}: n={r['n']} hit={r['hit']:.4f} (t={r['hit_t']:.2f}) auc={r['auc']:.4f} "
                    f"brier_skill={r['brier_skill']:+.4f} pips={r['pips']:+.2f} (t={r['pips_t']:.2f}) top10={r['top10_hit']:.3f}")
        out["h"][str(H)] = res
    return out


def report(res: dict) -> str:
    L = ["# 機械学習による方向の予測 (ウォークフォワード検証)", "",
         "各足の終値の時点で分かる特徴 (複数の期間の値動き、RSI、移動平均からの乖離、値動きの荒さ、足の値幅、時間帯・曜日、"
         "7ペアから計算した通貨の強さ、金利差) から、その後 H 本の値動きが上か下かを予測しました。モデルは勾配ブースティング木 (LightGBM) と"
         "正則化したロジスティック回帰で、検証期間の前のデータだけで学習し、検証期間中も (本番と同じように) 定期的に、それまでのデータだけで学習し直しています。"
         "学習データの目的変数が検証期間にかからないよう、H 本分の間隔を空けています。", "",
         "的中率の t 値は、日ごとにまとめて計算しています (隣り合う足やペアの結果は独立ではないため)。「平均 pips」は予測した方向に持った場合の1回あたりの値動き (コスト抜き) です。", ""]
    for tf, r in res.items():
        name = {"1h": "1時間足", "1d": "日足", "15m": "15分足"}[tf]
        L += [f"## {name} (学習開始 {r['start'][:10]}、検証 {r['split'][:10]}〜{r['end'][:10]}、{r['rows']:,}行)", "",
              "| 予測先 | モデル | 件数 | 的中率 (t値) | AUC | Brierスキル | 平均 pips (t値) | 確信度上位10%の的中率 |", "|---|---|---|---|---|---|---|---|"]
        for H, m in r["h"].items():
            for kind, x in m.items():
                L.append(f"| {H}本後 | {'LightGBM' if kind == 'lgbm' else 'ロジスティック回帰'} | {x['n']:,} | {x['hit']:.1%} ({x['hit_t']:.2f}) | "
                         f"{x['auc']:.3f} | {x['brier_skill']:+.4f} | {x['pips']:+.2f} ({x['pips_t']:.2f}) | {x['top10_hit']:.1%} |")
        L.append("")
    return "\n".join(L)


def run(tfs=("1h", "1d", "15m"), log=print) -> dict:
    res = {tf: evaluate(tf, log=log) for tf in tfs}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "ml.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (REPORT_DIR / "ml.md").write_text(report(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    import sys
    run(tuple(sys.argv[1:]) or ("1h", "1d", "15m"))
