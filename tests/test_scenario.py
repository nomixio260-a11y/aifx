from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from aifx import scenario
from aifx.timeutil import add_trading_minutes

UTC = timezone.utc


def _hourly(n=3000, seed=1, busy=13, quiet=3):
    """Hourly bars on the trading calendar whose ranges are 3x larger at ``busy`` than at ``quiet`` o'clock."""
    rng = np.random.default_rng(seed)
    t = datetime(2026, 1, 5, tzinfo=UTC)
    idx = []
    while len(idx) < n:
        t = add_trading_minutes(t, 1, 60)
        idx.append(t - pd.Timedelta(hours=1))
    idx = pd.DatetimeIndex(idx)
    scale = np.where(idx.hour == busy, 3.0, np.where(idx.hour == quiet, 1.0, 1.5)) * 4e-4
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
    hours = scenario.future_times(bars.index, 24, 60).hour
    busy = size[list(hours).index(13)]
    quiet = size[list(hours).index(3)]
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
