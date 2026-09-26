"""Technical indicators computed on daily closes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    line = ema(s, fast) - ema(s, slow)
    return line, line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, span_b: int = 52) -> dict[str, pd.Series]:
    """Ichimoku lines, unshifted: the leading spans are drawn ``kijun`` bars ahead, the lagging span
    ``kijun`` bars behind."""
    def mid(n):
        return (df["high"].rolling(n, min_periods=n).max() + df["low"].rolling(n, min_periods=n).min()) / 2
    t, k = mid(tenkan), mid(kijun)
    return {"tenkan": t, "kijun": k, "span_a": (t + k) / 2, "span_b": mid(span_b)}


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    lo, mid, up = bollinger(c)
    m, sig = macd(c)
    return pd.DataFrame(
        {
            "sma20": sma(c, 20),
            "sma75": sma(c, 75),
            "sma200": sma(c, 200),
            "bb_lower": lo,
            "bb_mid": mid,
            "bb_upper": up,
            "rsi14": rsi(c, 14),
            "macd": m,
            "macd_signal": sig,
            "atr14": atr(df, 14),
        },
        index=df.index,
    )


def technical_summary(df: pd.DataFrame, ind: pd.DataFrame) -> list[dict]:
    """Plain-language readings of the latest indicator values.

    Each item has a `bias` of +1 (bullish), -1 (bearish) or 0 (neutral).
    """
    last = ind.iloc[-1]
    prev = ind.iloc[-2]
    close = float(df["close"].iloc[-1])
    out: list[dict] = []

    r = float(last["rsi14"])
    if r >= 70:
        out.append({"name": "RSI(14)", "value": f"{r:.1f}", "bias": -1, "text": "買われすぎ圏 (70以上)"})
    elif r <= 30:
        out.append({"name": "RSI(14)", "value": f"{r:.1f}", "bias": 1, "text": "売られすぎ圏 (30以下)"})
    else:
        out.append({"name": "RSI(14)", "value": f"{r:.1f}", "bias": 0, "text": "中立圏 (30〜70)"})

    s20, s75 = float(last["sma20"]), float(last["sma75"])
    p20, p75 = float(prev["sma20"]), float(prev["sma75"])
    if p20 <= p75 and s20 > s75:
        out.append({"name": "移動平均 20/75", "value": "GC", "bias": 1, "text": "ゴールデンクロス発生"})
    elif p20 >= p75 and s20 < s75:
        out.append({"name": "移動平均 20/75", "value": "DC", "bias": -1, "text": "デッドクロス発生"})
    elif s20 > s75:
        out.append({"name": "移動平均 20/75", "value": "上", "bias": 1, "text": "短期線が長期線の上 (上昇基調)"})
    else:
        out.append({"name": "移動平均 20/75", "value": "下", "bias": -1, "text": "短期線が長期線の下 (下落基調)"})

    s200 = last["sma200"]
    if pd.notna(s200):
        above = close > float(s200)
        out.append({
            "name": "200日線",
            "value": "上" if above else "下",
            "bias": 1 if above else -1,
            "text": "価格が200日線より上 (長期上昇トレンド)" if above else "価格が200日線より下 (長期下降トレンド)",
        })

    m, sig = float(last["macd"]), float(last["macd_signal"])
    pm, psig = float(prev["macd"]), float(prev["macd_signal"])
    if pm <= psig and m > sig:
        txt, b = "シグナルを上抜け (買いシグナル)", 1
    elif pm >= psig and m < sig:
        txt, b = "シグナルを下抜け (売りシグナル)", -1
    elif m > sig:
        txt, b = "シグナルより上", 1
    else:
        txt, b = "シグナルより下", -1
    out.append({"name": "MACD", "value": f"{m - sig:+.4f}", "bias": b, "text": txt})

    lo, up = float(last["bb_lower"]), float(last["bb_upper"])
    pct_b = (close - lo) / (up - lo) if up > lo else 0.5
    if pct_b >= 1:
        txt, b = "+2σを上回る (過熱)", -1
    elif pct_b <= 0:
        txt, b = "-2σを下回る (売られすぎ)", 1
    else:
        txt, b = "バンド内", 0
    out.append({"name": "ボリンジャー %B", "value": f"{pct_b:.2f}", "bias": b, "text": txt})
    return out
