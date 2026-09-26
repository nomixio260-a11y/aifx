"""Technical indicators and textbook signals, as tested on 20 years of data (research/technical.md).

The same functions make the research's signals and the page's technical panel, so what the page
shows is exactly what was scored. Each signal is +1 (buy) / -1 (sell) / 0 at a bar's close, from that
bar and earlier ones only; ``signals_now`` gives every signal at the last bar with the indicator
values behind it and the consensus (buy calls minus sell calls over the group).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, bollinger, ichimoku, macd, rsi, sma
from .timeutil import LONDON

ICHI_SHIFT = 26
OHLC = ["open", "high", "low", "close"]

LEVELS = {2: "強い買い", 1: "買い", 0: "中立", -1: "売り", -2: "強い売り"}
LEVEL_KEYS = {2: "strong_buy", 1: "buy", 0: "neutral", -1: "sell", -2: "strong_sell"}

# key -> label, kind ("trend" 順張り / "reversal" 逆張り), buy and sell conditions, indicator values shown
SIGNALS: dict[str, dict] = {
    "sma20": {"label": "移動平均 20本と価格 (順張り)", "kind": "trend",
              "buy": "終値が20本移動平均より上", "sell": "終値が20本移動平均より下", "values": ("sma20",)},
    "sma75": {"label": "移動平均 75本と価格 (順張り)", "kind": "trend",
              "buy": "終値が75本移動平均より上", "sell": "終値が75本移動平均より下", "values": ("sma75",)},
    "sma200": {"label": "移動平均 200本と価格 (順張り)", "kind": "trend",
               "buy": "終値が200本移動平均より上", "sell": "終値が200本移動平均より下", "values": ("sma200",)},
    "ma_20_75": {"label": "移動平均 20/75 の上下 (順張り)", "kind": "trend",
                 "buy": "20本線が75本線より上", "sell": "20本線が75本線より下", "values": ("sma20", "sma75")},
    "ma_20_75_cross": {"label": "ゴールデンクロス / デッドクロス (20/75)", "kind": "trend",
                       "buy": "20本線が75本線を上抜けた足", "sell": "20本線が75本線を下抜けた足",
                       "values": ("sma20", "sma75")},
    "macd_state": {"label": "MACD とシグナルの上下 (順張り)", "kind": "trend",
                   "buy": "MACD (12,26,9) がシグナルより上 (ヒストグラムがプラス)", "sell": "MACD がシグナルより下",
                   "values": ("macd", "macd_signal", "macd_hist")},
    "macd_cross": {"label": "MACD のクロス (順張り)", "kind": "trend",
                   "buy": "MACD がシグナルを上抜けた足", "sell": "MACD がシグナルを下抜けた足",
                   "values": ("macd", "macd_signal")},
    "macd_zero": {"label": "MACD のゼロライン (順張り)", "kind": "trend",
                  "buy": "MACD が0より上", "sell": "MACD が0より下", "values": ("macd",)},
    "rsi_50_cross": {"label": "RSI 50 のクロス (順張り)", "kind": "trend",
                     "buy": "RSI (14) が50を上抜けた足", "sell": "RSI が50を下抜けた足", "values": ("rsi14",)},
    "rsi_30_70": {"label": "RSI 30/70 (逆張り)", "kind": "reversal",
                  "buy": "RSI (14) が30未満 (売られすぎ)", "sell": "RSI が70超 (買われすぎ)", "values": ("rsi14",)},
    "bb_pctb": {"label": "ボリンジャー %b (±2σの外で逆張り)", "kind": "reversal",
                "buy": "%b ≤ 0 (終値が-2σ以下)", "sell": "%b ≥ 1 (終値が+2σ以上)", "values": ("pctb",)},
    "bb_touch": {"label": "ボリンジャー ±2σタッチ (逆張り)", "kind": "reversal",
                 "buy": "安値が-2σに触れた", "sell": "高値が+2σに触れた", "values": ("bb_lower", "bb_upper")},
    "bb_break": {"label": "ボリンジャー ±2σブレイク (順張り)", "kind": "trend",
                 "buy": "終値が+2σを上抜けた足", "sell": "終値が-2σを下抜けた足", "values": ("bb_lower", "bb_upper")},
    "ichi_cloud": {"label": "一目均衡表 雲の上下 (順張り)", "kind": "trend",
                   "buy": "終値が雲の上", "sell": "終値が雲の下 (雲の中は中立)", "values": ("cloud_top", "cloud_bottom")},
    "ichi_tk": {"label": "一目 転換線と基準線の上下 (順張り)", "kind": "trend",
                "buy": "転換線が基準線より上", "sell": "転換線が基準線より下", "values": ("tenkan", "kijun")},
    "ichi_tk_cross": {"label": "一目 転換線と基準線のクロス (順張り)", "kind": "trend",
                      "buy": "転換線が基準線を上抜けた足", "sell": "転換線が基準線を下抜けた足", "values": ("tenkan", "kijun")},
    "ichi_sanyaku": {"label": "一目 三役好転 / 三役逆転 (順張り)", "kind": "trend",
                     "buy": "転換線>基準線、終値が雲の上、遅行スパンが26本前の価格より上",
                     "sell": "3条件がすべて逆", "values": ("tenkan", "kijun", "cloud_top", "cloud_bottom", "close_26")},
    "stoch_cross": {"label": "ストキャスティクス 20/80 圏のクロス (逆張り)", "kind": "reversal",
                    "buy": "%K が %D を上抜け、%D が20未満 (14,3,3)", "sell": "%K が %D を下抜け、%D が80超",
                    "values": ("stoch_k", "stoch_d")},
    "adx_di": {"label": "ADX>25 と ±DI (順張り)", "kind": "trend",
               "buy": "ADX (14) > 25 かつ +DI > -DI", "sell": "ADX > 25 かつ -DI > +DI",
               "values": ("adx", "plus_di", "minus_di")},
    "psar": {"label": "パラボリック SAR (順張り)", "kind": "trend",
             "buy": "SAR が価格の下 (上昇トレンド)", "sell": "SAR が価格の上 (下降トレンド)", "values": ("psar",)},
    "donchian20": {"label": "ドンチャン 20本ブレイク (順張り)", "kind": "trend",
                   "buy": "終値が直前20本の最高値を上抜け", "sell": "終値が直前20本の最安値を下抜け",
                   "values": ("don_high", "don_low")},
    "pivot": {"label": "ピボット (前日) の上下 (順張り)", "kind": "trend",
              "buy": "終値が前日のピボット P より上", "sell": "終値が P より下", "values": ("pivot", "r1", "s1", "r2", "s2")},
    "roc12": {"label": "ROC (12) のゼロライン (順張り)", "kind": "trend",
              "buy": "12本前より上", "sell": "12本前より下", "values": ("roc12",)},
    "williams_r": {"label": "ウィリアムズ %R (逆張り)", "kind": "reversal",
                   "buy": "%R (14) が-80未満", "sell": "%R が-20超", "values": ("williams_r",)},
    "cci_100": {"label": "CCI ±100 (逆張り)", "kind": "reversal",
                "buy": "CCI (20) が-100未満", "sell": "CCI が+100超", "values": ("cci",)},
}
KEYS = tuple(SIGNALS)
GROUPS = {"all": KEYS,
          "trend": tuple(k for k in KEYS if SIGNALS[k]["kind"] == "trend"),
          "reversal": tuple(k for k in KEYS if SIGNALS[k]["kind"] == "reversal")}
GROUP_LABEL = {"all": "全シグナル", "trend": "順張りのシグナル", "reversal": "逆張りのシグナル"}


# ------------------------------------------------------------------ indicators (not in indicators.py yet)

def stochastics(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    """Slow stochastics (14, 3, 3): %K is the close's place in the high-low range of the last ``k`` bars
    (0-100) averaged over ``smooth`` bars, %D the ``d``-bar average of %K. A flat range counts as 50."""
    hh = df["high"].rolling(k, min_periods=k).max()
    ll = df["low"].rolling(k, min_periods=k).min()
    rng = hh - ll
    raw = (100 * (df["close"] - ll) / rng.where(rng > 0)).where(rng > 0, 50.0).where(rng.notna())
    pk = raw.rolling(smooth, min_periods=smooth).mean()
    return pk, pk.rolling(d, min_periods=d).mean()


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Williams %R: -100 at the lowest low of the last ``n`` bars, 0 at the highest high."""
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    rng = hh - ll
    return (-100 * (hh - df["close"]) / rng.where(rng > 0)).where(rng > 0, -50.0).where(rng.notna())


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Commodity Channel Index (Lambert): (typical price - its n-bar mean) / (0.015 x mean absolute deviation)."""
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy(float)
    out = np.full(len(tp), np.nan)
    if len(tp) >= n:
        w = np.lib.stride_tricks.sliding_window_view(tp, n)
        m = w.mean(axis=1)
        md = np.abs(w - m[:, None]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = np.where(md > 0, (tp[n - 1:] - m) / (0.015 * md), np.nan)
    return pd.Series(out, index=df.index)


def roc(s: pd.Series, n: int = 12) -> pd.Series:
    """Rate of change over ``n`` bars, %."""
    return 100 * (s / s.shift(n) - 1)


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX with +DI and -DI (Wilder smoothing = EMA with alpha 1/n)."""
    h, lo, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -lo.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0).where(up.notna())
    mdm = dn.where((dn > up) & (dn > 0), 0.0).where(dn.notna())
    prev = c.shift()
    tr = pd.concat([h - lo, (h - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1).where(prev.notna())

    def wilder(x):
        return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    a = wilder(tr)
    pdi = 100 * wilder(pdm) / a.where(a > 0)
    mdi = 100 * wilder(mdm) / a.where(a > 0)
    s = pdi + mdi
    dx = 100 * (pdi - mdi).abs() / s.where(s > 0)
    return wilder(dx), pdi, mdi


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> tuple[pd.Series, pd.Series]:
    """Wilder's Parabolic SAR and its trend (+1 up, -1 down) after each bar.

    The first trend follows the first bar-to-bar change of the close. Each bar, the stop moves
    ``af`` of the way to the extreme point (af starts at ``step``, grows by ``step`` with each new
    extreme up to ``max_step``) without entering the previous two bars' range; a bar trading through
    the stop reverses the trend and restarts the stop at the old extreme point, moved out of this and the
    previous bar's range (as TA-Lib does)."""
    h = df["high"].to_numpy(float).tolist()
    lo = df["low"].to_numpy(float).tolist()
    c = df["close"].to_numpy(float)
    n = len(h)
    sar = np.full(n, np.nan)
    trend = np.zeros(n)
    if n < 2:
        return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)
    up = bool(c[1] >= c[0])
    s = min(lo[0], lo[1]) if up else max(h[0], h[1])
    ep = max(h[0], h[1]) if up else min(lo[0], lo[1])
    af = step
    sar[1], trend[1] = s, 1 if up else -1
    for i in range(2, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, lo[i - 1], lo[i - 2])
            if lo[i] < s:
                up, s, ep, af = False, max(ep, h[i], h[i - 1]), lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            s = max(s, h[i - 1], h[i - 2])
            if h[i] > s:
                up, s, ep, af = True, min(ep, lo[i], lo[i - 1]), h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_step)
        sar[i], trend[i] = s, 1 if up else -1
    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)


def donchian(df: pd.DataFrame, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """Highest high and lowest low of the ``n`` bars before each bar (the bar itself left out)."""
    return (df["high"].rolling(n, min_periods=n).max().shift(1),
            df["low"].rolling(n, min_periods=n).min().shift(1))


def london_trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """London trading day of each intraday bar (UTC bar-start index); the Sunday evening counts towards Monday."""
    loc = pd.DatetimeIndex(index.tz_convert(LONDON).tz_localize(None)).normalize()
    wd = loc.dayofweek.to_numpy()
    return loc + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")


def is_daily(df: pd.DataFrame) -> bool:
    """Daily bars carry a date index without time zone and time of day (Yahoo, London days)."""
    idx = pd.DatetimeIndex(df.index)
    return idx.tz is None and bool((idx == idx.normalize()).all())


def pivot_points(df: pd.DataFrame, daily: bool | None = None) -> pd.DataFrame:
    """Classic (floor trader) pivots from the previous day's high H, low L and close C:
    P = (H + L + C) / 3, R1 = 2P - L, S1 = 2P - H, R2 = P + (H - L), S2 = P - (H - L).
    Intraday bars are grouped into London trading days (only complete earlier days are used);
    daily bars use the previous bar."""
    daily = is_daily(df) if daily is None else daily
    if daily:
        prev = df[["high", "low", "close"]].shift(1)
    else:
        day = london_trading_day(pd.DatetimeIndex(df.index))
        g = df[["high", "low", "close"]].groupby(day.to_numpy())
        days = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()}).shift(1)
        prev = days.reindex(day.to_numpy())
        prev.index = df.index
    p = (prev["high"] + prev["low"] + prev["close"]) / 3
    rng = prev["high"] - prev["low"]
    return pd.DataFrame({"pivot": p, "r1": 2 * p - prev["low"], "s1": 2 * p - prev["high"],
                         "r2": p + rng, "s2": p - rng}, index=df.index)


def indicator_frame(df: pd.DataFrame, daily: bool | None = None) -> pd.DataFrame:
    """Every indicator the signals read, at each bar's close (bars up to and including that one)."""
    c = df["close"]
    lo_b, mid_b, up_b = bollinger(c, 20, 2.0)
    m, ms = macd(c)
    ichi = ichimoku(df)
    sa, sb = ichi["span_a"].shift(ICHI_SHIFT), ichi["span_b"].shift(ICHI_SHIFT)
    k, d = stochastics(df)
    a, pdi, mdi = adx(df)
    sar, trend = parabolic_sar(df)
    dh, dl = donchian(df)
    width = up_b - lo_b
    out = pd.DataFrame({
        "close": c, "sma20": sma(c, 20), "sma75": sma(c, 75), "sma200": sma(c, 200), "rsi14": rsi(c, 14),
        "macd": m, "macd_signal": ms, "macd_hist": m - ms,
        "bb_lower": lo_b, "bb_mid": mid_b, "bb_upper": up_b, "pctb": (c - lo_b) / width.where(width > 0),
        "tenkan": ichi["tenkan"], "kijun": ichi["kijun"],
        "cloud_top": np.fmax(sa, sb).where(sa.notna() & sb.notna()),
        "cloud_bottom": np.fmin(sa, sb).where(sa.notna() & sb.notna()), "close_26": c.shift(ICHI_SHIFT),
        "stoch_k": k, "stoch_d": d, "adx": a, "plus_di": pdi, "minus_di": mdi, "psar": sar, "psar_trend": trend,
        "don_high": dh, "don_low": dl, "williams_r": williams_r(df), "cci": cci(df), "roc12": roc(c, 12),
        "atr14": atr(df, 14)}, index=df.index)
    return out.join(pivot_points(df, daily))


# ------------------------------------------------------------------ signals

def _prev(a):
    if np.isscalar(a):
        return a
    out = np.empty_like(a)
    out[0] = np.nan
    out[1:] = a[:-1]
    return out


def _i8(mask) -> np.ndarray:
    return np.asarray(mask, dtype=np.int8)


def _state(a, b) -> np.ndarray:
    """+1 where a > b, -1 where a < b, 0 where equal or unknown."""
    with np.errstate(invalid="ignore"):
        return _i8(a > b) - _i8(a < b)


def _cross(a, b) -> np.ndarray:
    """+1 on the bar a crosses above b, -1 on the bar it crosses below."""
    pa, pb = _prev(a), _prev(b)
    with np.errstate(invalid="ignore"):
        return _i8((a > b) & (pa <= pb)) - _i8((a < b) & (pa >= pb))


def _zone(x, lo, hi) -> np.ndarray:
    """+1 below ``lo`` (oversold), -1 above ``hi`` (overbought)."""
    with np.errstate(invalid="ignore"):
        return _i8(x < lo) - _i8(x > hi)


def signal_frame(df: pd.DataFrame, daily: bool | None = None, ind: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every signal (+1 buy, -1 sell, 0 none) at each bar's close, as int8 columns in SIGNALS order."""
    ind = indicator_frame(df, daily) if ind is None else ind
    g = {k: ind[k].to_numpy(float) for k in ind.columns}
    c, hi, lo = g["close"], df["high"].to_numpy(float), df["low"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        tk = _state(g["tenkan"], g["kijun"])
        cloud = _i8(c > g["cloud_top"]) - _i8(c < g["cloud_bottom"])
        lag = _state(c, g["close_26"])
        touch_lo, touch_hi = lo <= g["bb_lower"], hi >= g["bb_upper"]
        k_cross = _cross(g["stoch_k"], g["stoch_d"])
        s = {
            "sma20": _state(c, g["sma20"]), "sma75": _state(c, g["sma75"]), "sma200": _state(c, g["sma200"]),
            "ma_20_75": _state(g["sma20"], g["sma75"]), "ma_20_75_cross": _cross(g["sma20"], g["sma75"]),
            "macd_state": _state(g["macd"], g["macd_signal"]), "macd_cross": _cross(g["macd"], g["macd_signal"]),
            "macd_zero": _state(g["macd"], 0.0), "rsi_50_cross": _cross(g["rsi14"], 50.0),
            "rsi_30_70": _zone(g["rsi14"], 30, 70),
            "bb_pctb": _i8(g["pctb"] <= 0) - _i8(g["pctb"] >= 1),
            "bb_touch": _i8(touch_lo & ~touch_hi) - _i8(touch_hi & ~touch_lo),
            "bb_break": _i8((c > g["bb_upper"]) & (_prev(c) <= _prev(g["bb_upper"])))
            - _i8((c < g["bb_lower"]) & (_prev(c) >= _prev(g["bb_lower"]))),
            "ichi_cloud": cloud, "ichi_tk": tk, "ichi_tk_cross": _cross(g["tenkan"], g["kijun"]),
            "ichi_sanyaku": _i8((tk == 1) & (cloud == 1) & (lag == 1)) - _i8((tk == -1) & (cloud == -1) & (lag == -1)),
            "stoch_cross": _i8((k_cross == 1) & (g["stoch_d"] < 20)) - _i8((k_cross == -1) & (g["stoch_d"] > 80)),
            "adx_di": _i8(g["adx"] > 25) * _state(g["plus_di"], g["minus_di"]),
            "psar": np.nan_to_num(g["psar_trend"]).astype(np.int8),
            "donchian20": _i8(c > g["don_high"]) - _i8(c < g["don_low"]),
            "pivot": _state(c, g["pivot"]), "roc12": _state(g["roc12"], 0.0),
            "williams_r": _zone(g["williams_r"], -80, -20), "cci_100": _zone(g["cci"], -100, 100),
        }
    return pd.DataFrame({k: s[k].astype(np.int8) for k in KEYS}, index=df.index)


def consensus_score(sig, keys=KEYS) -> np.ndarray | float:
    """(buy calls - sell calls) / number of signals in the group, in [-1, 1]."""
    if isinstance(sig, pd.DataFrame):
        return sig[list(keys)].to_numpy(float).sum(axis=1) / len(keys)
    return float(sum(int(sig[k]) for k in keys)) / len(keys)


def consensus_level(score, neutral: float, strong: float):
    """5 levels: 0 中立 when |score| <= neutral, ±2 (強い) when |score| >= strong, ±1 between."""
    a = np.abs(score)
    lvl = np.sign(score) * ((a > neutral + 1e-9).astype(int) + (a >= strong - 1e-9).astype(int))
    return lvl.astype(int) if isinstance(lvl, np.ndarray) else int(lvl)


def _num(x, d: int = 6):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def signals_now(df: pd.DataFrame, daily: bool | None = None, thresholds: dict | None = None) -> dict:
    """Every signal at the last bar of ``df`` (OHLC bars in time order: intraday bars with a UTC
    bar-start index, or daily bars with a date index), computed from that bar and earlier ones only.

    The EMA-type indicators (RSI, MACD, ADX, ATR) and the SAR depend a little on where the history
    starts, so pass at least ~500 bars (the research used the whole history). ``thresholds``
    ({group: {"neutral": x, "strong": y}}, the "consensus" thresholds in research/technical.json for
    the timeframe) adds the 5-level consensus."""
    daily = is_daily(df) if daily is None else daily
    ind = indicator_frame(df, daily)
    sig = signal_frame(df, daily, ind)
    last, vals = sig.iloc[-1], ind.iloc[-1]
    out = {"time": str(df.index[-1]), "daily": daily, "close": _num(vals["close"], 8), "signals": {}, "consensus": {}}
    for k, spec in SIGNALS.items():
        out["signals"][k] = {"label": spec["label"], "kind": spec["kind"], "signal": int(last[k]),
                             "values": {v: _num(vals[v], 8) for v in spec["values"]}}
    out["values"] = {k: _num(vals[k], 8) for k in ind.columns}
    for name, keys in GROUPS.items():
        calls = [int(last[k]) for k in keys]
        sc = consensus_score(last, keys)
        c = {"score": round(sc, 6), "buy": calls.count(1), "sell": calls.count(-1), "none": calls.count(0)}
        th = (thresholds or {}).get(name)
        if th:
            lvl = consensus_level(sc, th["neutral"], th["strong"])
            c["level"], c["label"] = lvl, LEVELS[lvl]
        out["consensus"][name] = c
    return out
