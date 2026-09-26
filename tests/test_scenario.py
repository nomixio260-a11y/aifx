from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from aifx import scenario
from aifx.timeutil import add_trading_minutes

UTC = timezone.utc


def _hourly(n=3000, seed=1, busy=9, quiet=22):
    """Hourly bars on the trading calendar whose ranges are 3x larger at ``busy`` than at ``quiet`` o'clock
    New York time (the market's clock, across its daylight-saving change)."""
    rng = np.random.default_rng(seed)
    t = datetime(2026, 1, 5, tzinfo=UTC)
    idx = []
    while len(idx) < n:
        t = add_trading_minutes(t, 1, 60)
        idx.append(t - pd.Timedelta(hours=1))
    idx = pd.DatetimeIndex(idx)
    ny = idx.tz_convert(scenario.NEW_YORK).hour
    scale = np.where(ny == busy, 3.0, np.where(ny == quiet, 1.0, 1.5)) * 4e-4
    r = rng.normal(0, scale)
    c = 150 * np.exp(np.cumsum(r))
    o = np.concatenate([[150.0], c[:-1]])
    wick = np.abs(rng.normal(0, scale)) * c
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick, "close": c}, index=idx)


def test_forecast_candles_are_real_candles_that_join_up_and_end_at_the_target():
    bars = _hourly()
    p0 = float(bars["close"].iloc[-1])
    target = p0 * 1.002
    cs, info = scenario.candles("1h", bars, 24, target, 60)
    assert len(cs) == 24 and info["k"] == scenario.K
    assert cs[0][0] == pytest.approx(p0)
    for a, b in zip(cs, cs[1:]):
        assert b[0] == pytest.approx(a[3])                      # each opens at the previous close
    for o, h, lo, c in cs:
        assert lo <= min(o, c) <= max(o, c) <= h
    assert cs[-1][3] == pytest.approx(target)
    # not dojis: the bodies are a real part of the candles
    body = np.mean([abs(c - o) / (h - lo) for o, h, lo, c in cs])
    assert body > 0.15
    assert scenario.candles("1h", bars, 24, target, 60)[0] == cs    # the same bars give the same candles


def test_candle_sizes_follow_the_time_of_day():
    bars = _hourly()
    size = scenario.sizes("1h", bars, 24, 60)
    hours = scenario.future_times(bars.index, 24, 60).tz_convert(scenario.NEW_YORK).hour
    busy = size[list(hours).index(9)]
    quiet = size[list(hours).index(22)]
    assert busy > 2 * quiet
    cs, info = scenario.candles("1h", bars, 24, None, 60)
    for (o, h, lo, c), s in zip(cs, info["size"]):
        assert np.log(h / lo) == pytest.approx(s, rel=1e-6)    # each candle has the predicted size


def test_future_bars_skip_the_weekend():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-09-25 20:00", tz="UTC")])   # Friday's last hour
    nxt = scenario.future_times(idx, 3, 60)
    assert list(nxt.dayofweek) == [6, 6, 6] and list(nxt.hour) == [21, 22, 23]   # Sunday evening
    days = scenario.future_times(pd.DatetimeIndex([pd.Timestamp("2026-09-25")]), 3, 0)
    assert list(days.dayofweek) == [0, 1, 2]                                    # Monday, Tuesday, Wednesday


def test_api_draws_forecast_candles_with_their_record(session):
    from aifx.api import build_api
    root, _ = session
    out = build_api(root)
    for tf in ("15m", "1h", "1d"):
        blk = out["pair/USDJPY.json"]["tf"][tf]
        C = blk["candles"]
        assert len(C["items"]) == len(C["x"]) == len(blk["path"]["steps"])
        assert C["x"] == [s["x"] for s in blk["path"]["steps"]]
        assert C["items"][0][0] == pytest.approx(blk["prediction"]["p0"], rel=1e-4)
        acc = C["accuracy"]
        assert acc["all"]["n"] > 0 and 0 <= acc["all"]["dir_hit"] <= 1 and acc["all"]["size_vs_atr"] is not None


def test_ichimoku_lines_are_the_midpoints_of_their_windows():
    from aifx import indicators
    bars = _hourly(n=200)
    ichi = indicators.ichimoku(bars)
    i = 150
    mid = lambda n: (bars["high"].iloc[i - n + 1:i + 1].max() + bars["low"].iloc[i - n + 1:i + 1].min()) / 2
    assert ichi["tenkan"].iloc[i] == pytest.approx(mid(9))
    assert ichi["kijun"].iloc[i] == pytest.approx(mid(26))
    assert ichi["span_a"].iloc[i] == pytest.approx((mid(9) + mid(26)) / 2)
    assert ichi["span_b"].iloc[i] == pytest.approx(mid(52))
    assert np.isnan(ichi["span_b"].iloc[50])                         # not enough bars yet


def test_chart_indicators_line_up_with_the_bars(session):
    from aifx.api import ICHI_SHIFT, build_api
    root, _ = session
    out = build_api(root)
    for tf in ("15m", "1h", "1d"):
        blk = out["pair/USDJPY.json"]["tf"][tf]
        n, ind, steps = len(blk["bars"]["t"]), blk["ind"], len(blk["path"]["steps"])
        for k in ("sma20", "sma75", "bb_upper", "bb_mid", "bb_lower", "rsi14", "macd", "macd_signal",
                  "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b", "ichi_lag"):
            assert len(ind[k]) == n, (tf, k)
        # the leading spans reach into the forecast bars, the lagging span stops 26 bars before the end
        assert len(ind["ichi_span_a_ahead"]) == len(ind["ichi_span_b_ahead"]) == min(ICHI_SHIFT, steps)
        closes = [c[1] for c in blk["bars"]["ohlc"]]
        assert ind["ichi_lag"][-ICHI_SHIFT - 1] == pytest.approx(closes[-1], abs=1e-3)
        assert all(v is None for v in ind["ichi_lag"][-ICHI_SHIFT:])
        assert all(lo <= mid <= hi for lo, mid, hi in zip(ind["bb_lower"], ind["bb_mid"], ind["bb_upper"]) if lo is not None)
