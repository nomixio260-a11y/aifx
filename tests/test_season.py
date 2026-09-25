from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aifx import learning, scenario, season
from aifx.audit import audit, verify
from aifx.ledger import Ledger
from aifx.timeutil import add_trading_minutes, parse_iso

from .conftest import run_session

UTC = timezone.utc


def _hourly(n=4000, seed=2, drift_hour=21, drift_bp=4.0, weekend_jump_bp=0.0):
    """Hourly bars on the trading calendar, 5 bp noise, a steady move at ``drift_hour`` UTC and an
    optional jump at the first bar after each weekend."""
    rng = np.random.default_rng(seed)
    t = datetime(2025, 6, 2, tzinfo=UTC)
    starts = []
    while len(starts) < n:
        t = add_trading_minutes(t, 1, 60)
        starts.append(t - timedelta(hours=1))
    idx = pd.DatetimeIndex(starts)
    r = rng.normal(0, 5.0, n) + np.where(idx.hour == drift_hour, drift_bp, 0.0)
    after_pause = np.concatenate([[False], np.diff(idx.as_unit("ns").asi8) > 3600e9])
    r = r + np.where(after_pause, weekend_jump_bp, 0.0)
    c = 150 * np.exp(np.cumsum(r) / 1e4)
    o = np.concatenate([[150.0], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0003, "low": np.minimum(o, c) * 0.9997, "close": c}, index=idx)


def test_the_drift_finds_a_steady_hour_and_uses_only_earlier_days():
    bars = _hourly()
    origin = datetime(2026, 1, 14, 20, tzinfo=UTC)               # a Wednesday, 20:00 UTC
    d = season.step_drift(60, bars, origin, 6)
    assert 2.0 < d[1] < 6.5                                      # the bar starting at 21:00 (a Wednesday slot: ~35 bars)
    assert np.count_nonzero(d) <= 2
    # bars from the origin's day on do not matter
    later = bars.copy()
    later.loc[later.index >= pd.Timestamp("2026-01-14", tz="UTC"), "close"] *= 1.05
    assert np.array_equal(season.step_drift(60, later, origin, 6), d)
    # 15-minute bars get a quarter of the hour's move
    d15 = season.step_drift(15, bars, datetime(2026, 1, 14, 21, tzinfo=UTC), 4)
    assert np.allclose(d15, d[1] / 4)
    assert not season.step_drift(0, bars, origin, 5).any()        # daily bars: none


def test_the_centre_keeps_the_first_hour_only():
    assert list(season.centre_drift(np.array([1.0, 2.0, 3.0]), 60)) == [1.0, 0.0, 0.0]
    assert list(season.centre_drift(np.array([1.0, 1.0, 1.0, 1.0, 5.0]), 15)) == [1.0, 2.0, 3.0, 4.0, 0.0]
    assert not season.centre_drift(np.array([1.0, 2.0]), 0).any()


def test_weekend_gaps_do_not_make_a_drift():
    bars = _hourly(drift_bp=0.0, weekend_jump_bp=40.0)
    st = season.slot_stats(bars, datetime(2026, 1, 14, tzinfo=UTC))
    mu, t = st["slot"]
    assert np.abs(mu).max() < 5.0                               # the 40 bp jumps are left out


def test_learning_leaves_the_drift_out_of_the_gain_and_news_tilt():
    n = 400
    rng = np.random.default_rng(1)
    m = rng.normal(0, 3, (n, 6))
    d = rng.choice([-3.0, 0.0, 3.0], n)
    s = np.full(n, 10.0)
    a = d + rng.normal(0, 10, n)
    c0 = m.mean(axis=1) + d                                     # a model blend that happens to track the drift
    kw = dict(k=np.ones(n), g=np.zeros(n), c0=c0, c=d, x=np.zeros(n), half_life=200.0)
    with_d = learning.learn_arrays(None, m, a, s, d=d, **kw)
    without = learning.learn_arrays(None, m, a, s, **kw)
    assert abs(with_d.gain) < abs(without.gain)


def test_called_candles_point_the_called_way():
    bars = _hourly(n=3000)
    drift = np.zeros(24)
    drift[[0, 5, 9]] = [2.0, -2.0, 3.0]
    p0 = float(bars["close"].iloc[-1])
    cs, info = scenario.candles("1h", bars, 24, p0, 60, drift)
    assert info["call"][:10] == [1, 0, 0, 0, 0, -1, 0, 0, 0, 1]
    for j, want in ((0, 1), (5, -1), (9, 1)):
        o, h, lo, c = cs[j]
        assert np.sign(c - o) == want and lo <= min(o, c) <= max(o, c) <= h
    assert cs[-1][3] == pytest.approx(p0)


def test_forecasts_carry_the_drift_and_audit_rebuilds_it(tmp_path, monkeypatch):
    monkeypatch.setattr(season, "T_MIN", 0.0)                  # every slot makes a call, so the drift is never 0
    monkeypatch.setattr(season, "MIN_N", 3)
    root = tmp_path / "state"
    run_session(root, cycles=4)
    ledger = Ledger(root).load()
    preds = ledger.of_type("prediction")
    intraday = [f for p in preds if p["tf"] in ("15m", "1h") for f in p["fc"]]
    assert intraday and any(f["d"] != 0 for f in intraday)
    assert all(f["d"] == 0 for p in preds if p["tf"] == "1d" for f in p["fc"])
    # the centre uses the first hour only: 1 hour ahead on 15-minute bars, the next hour on hourly bars
    assert all(f["d"] == 0 for p in preds for f in p["fc"] if (p["tf"], f["h"]) in (("15m", 16), ("1h", 4), ("1h", 24)))
    for p in preds:
        for f in p["fc"]:
            sig = f["s"] * f["k"]
            assert f["c"] == pytest.approx(f["g"] * f["c0"] + f["b"] * p["news"]["x"] * sig + f["d"], abs=2e-3)
    assert verify(root)["ok"]
    rep = audit(root, sample=8)
    assert rep["results"] and all(r["ok"] for r in rep["results"])


def test_a_changed_or_misplaced_drift_is_caught(session):
    root, _ = session
    ledger = Ledger(root).load()
    at = parse_iso(ledger.records[-1]["at"])
    src_h = next(p for p in reversed(ledger.of_type("prediction")) if p["tf"] == "1h")
    src_d = next(p for p in reversed(ledger.of_type("prediction")) if p["tf"] == "1d")
    for n, (src, bump_c) in enumerate(((src_h, False), (src_d, True))):
        p = {k: v for k, v in src.items() if k not in ("seq", "prev", "hash", "at")}
        fc = [dict(f) for f in p["fc"]]
        fc[0]["d"] = fc[0]["d"] + 5.0                              # a drift that the centre does not include ...
        if bump_c:
            fc[0]["c"] = fc[0]["c"] + 5.0                          # ... or a consistent one on a daily forecast
        p["fc"] = fc
        ledger.append(p, at + timedelta(minutes=1 + n))
    msgs = " ".join(x["msg"] for x in verify(root)["problems"])
    assert "inconsistent" in msgs and "daily forecast" in msgs
