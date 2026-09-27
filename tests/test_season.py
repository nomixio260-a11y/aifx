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


def _bars(n=4000, minutes=60, seed=2, drift_ny=(16, 0), drift_bp=4.0, weekend_jump_bp=0.0, start=datetime(2025, 6, 2)):
    """Bars of ``minutes`` on the trading calendar, 5 bp noise, a steady move in the bar starting at
    ``drift_ny`` (hour, minute) New York time, and an optional jump at the first bar after each weekend."""
    rng = np.random.default_rng(seed)
    t = start.replace(tzinfo=UTC)
    starts = []
    while len(starts) < n:
        t = add_trading_minutes(t, 1, minutes)
        starts.append(t - timedelta(minutes=minutes))
    idx = pd.DatetimeIndex(starts)
    ny = idx.tz_convert(season.NEW_YORK)
    at = (ny.hour == drift_ny[0]) & (ny.minute == drift_ny[1])
    r = rng.normal(0, 5.0, n) + np.where(at, drift_bp, 0.0)
    after_pause = np.concatenate([[False], np.diff(idx.as_unit("ns").asi8) > minutes * 60e9])
    r = r + np.where(after_pause, weekend_jump_bp, 0.0)
    c = 150 * np.exp(np.cumsum(r) / 1e4)
    o = np.concatenate([[150.0], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0003, "low": np.minimum(o, c) * 0.9997, "close": c}, index=idx)


def test_the_drift_finds_a_steady_new_york_hour_and_uses_only_earlier_days():
    bars = _bars()                                               # June to January: across the change of US daylight time
    origin = datetime(2026, 1, 14, 20, tzinfo=UTC)               # a Wednesday, 15:00 in New York
    d, t = season.step_drift(60, bars, origin, 6)
    assert 2.0 < d[1] < 6.5 and abs(t[1]) >= season.T_MIN       # the bar starting 16:00 New York (21:00 UTC in winter)
    assert np.count_nonzero(d) <= 2
    # bars from the origin's day on do not matter
    later = bars.copy()
    later.loc[later.index >= pd.Timestamp("2026-01-14", tz="UTC"), "close"] *= 1.05
    d2, t2 = season.step_drift(60, later, origin, 6)
    assert np.array_equal(d2, d) and np.array_equal(t2, t)
    assert not season.step_drift(0, bars, origin, 5)[0].any()    # daily bars: none
    assert season.tier(t[1]) in ("mid", "high") and season.tier(0.5) is None and season.tier(-4.5) == "high"


def test_15_minute_bars_use_their_own_quarter_hours():
    bars = _bars(n=5000, minutes=15, drift_ny=(17, 0), drift_bp=6.0, start=datetime(2026, 7, 20))
    origin = datetime(2026, 9, 16, 20, 45, tzinfo=UTC)          # Wednesday 16:45 New York (daylight time)
    d, t = season.step_drift(15, bars, origin, 4)
    assert d[1] > 1.0 and not d[0] and not d[2]                  # only the quarter starting 17:00 New York
    assert season.centre_t(d, t, 15)[3] == pytest.approx(season.combined_t(d, t))


def test_the_centre_keeps_the_first_hour_only():
    assert list(season.centre_drift(np.array([1.0, 2.0, 3.0]), 60)) == [1.0, 0.0, 0.0]
    assert list(season.centre_drift(np.array([1.0, 1.0, 1.0, 1.0, 5.0]), 15)) == [1.0, 2.0, 3.0, 4.0, 0.0]
    assert not season.centre_drift(np.array([1.0, 2.0]), 0).any()
    assert season.combined_t(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == pytest.approx(4 / np.sqrt(2))


def test_weekend_gaps_do_not_make_a_drift():
    bars = _bars(drift_bp=0.0, weekend_jump_bp=40.0)
    st = season.slot_stats(bars, datetime(2026, 1, 14, tzinfo=UTC), 60)
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
    bars = _bars(n=3000)
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
    assert all((f["d"] == 0) == (f["dt"] == 0) for f in intraday)
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


def test_the_roll_moves_the_value_date_by_calendar_days():
    from datetime import date

    from aifx import fxcalendar
    assert fxcalendar.roll_days(date(2026, 9, 28), "USD", "JPY") == 1          # Monday
    assert fxcalendar.roll_days(date(2026, 9, 30), "USD", "JPY") == 3          # Wednesday: over the weekend
    # Thanksgiving (Thursday 27 November 2025, a USD holiday): Monday's roll moves spot two days, Tuesday's none
    assert fxcalendar.roll_days(date(2025, 11, 24), "EUR", "USD") == 2
    assert fxcalendar.roll_days(date(2025, 11, 25), "EUR", "USD") == 0
    # Japan's Silver Week (21-23 September 2026): spot stays on the 25th
    assert fxcalendar.roll_days(date(2026, 9, 21), "USD", "JPY") == 0
    hol = {c: fxcalendar.holidays(c, 2025, 2027) for c in ("USD", "JPY")}
    assert fxcalendar.spot_date(date(2026, 9, 21), "USD", "JPY", hol) == date(2026, 9, 25)


def test_roll_bar_calls_that_agree_with_the_rate_difference_are_high_confidence():
    from aifx.data import PAIRS
    idx = pd.DatetimeIndex([pd.Timestamp("2026-09-29 21:00", tz="UTC"),     # Tuesday 17:00 New York
                            pd.Timestamp("2026-09-29 22:00", tz="UTC")])
    e = season.roll_shift(idx, "AUD", "JPY", 3.5)                           # AUD pays 3.5% more: AUDJPY dips
    assert e[0] == pytest.approx(-3.5 / 360 * 100) and e[1] == 0
    d, t = np.array([-1.0, -1.0, 1.0]), np.array([-2.5, -2.5, 2.5])
    e3 = np.array([-3.0, -1.0, -3.0])
    assert list(season.promote(d, t, e3)) == [-season.T_HIGH, -2.5, 2.5]   # agreeing and large / too small / against
    # through step_drift: a call at the roll that agrees is promoted, the move itself is unchanged
    bars = _bars(n=3000, drift_ny=(17, 0), drift_bp=-1.2)
    origin = datetime(2026, 1, 14, 21, tzinfo=UTC)                          # Wednesday 16:00 New York (winter)
    d0, t0 = season.step_drift(60, bars, origin, 3)
    d1, t1 = season.step_drift(60, bars, origin, 3, PAIRS["AUDJPY"], 3.5)
    assert np.array_equal(d0, d1)
    if season.T_MIN <= abs(t0[0]) < season.T_HIGH and d0[0] < 0:
        assert t1[0] == -season.T_HIGH
    assert np.array_equal(season.step_drift(60, bars, origin, 3, PAIRS["AUDJPY"], None)[1], t0)


def test_no_call_for_the_first_bar_after_the_weekend_and_the_grounds_of_a_call():
    bars = _bars(n=3000, drift_ny=(17, 0), drift_bp=4.0)
    origin = datetime(2026, 1, 9, 22, tzinfo=UTC)                # Friday 17:00 New York: the close
    st = season.slot_stats(bars, origin, 60)
    sunday_open = pd.Timestamp("2026-01-11 22:00", tz="UTC")     # Sunday 17:00 New York (winter)
    d, t = season.bar_drift(st, [sunday_open], 60)
    assert d[0] == 0 and t[0] == 0                               # its move is the weekend's gap
    ex = season.explain(st, sunday_open, 60)
    assert ex["reopen"] and ex["used"] is None
    tuesday = pd.Timestamp("2026-01-13 22:00", tz="UTC")         # Tuesday 17:00 New York
    ex = season.explain(st, tuesday, 60)
    d, t = season.bar_drift(st, [tuesday], 60)
    assert not ex["reopen"] and ex["weekday"] == 1 and ex["minute"] == 17 * 60
    assert ex["slot"]["n"] >= season.MIN_N and ex["slot"]["t"] == pytest.approx(t[0])
    assert ex["used"] == "slot" and ex["slot"]["mean_bp"] == pytest.approx(d[0])


def test_api_gives_the_grounds_of_the_next_bar(session):
    from aifx.api import build_api
    root, _ = session
    out = build_api(root)
    for tf in ("15m", "1h"):
        B = out["pair/USDJPY.json"]["tf"][tf]["basis"]
        assert B and set(B) >= {"start", "weekday", "minute", "used", "reopen", "slot", "tod", "roll"}
        assert B["used"] in (None, "slot", "tod") and B["slot"]["n"] >= 0
    assert out["pair/USDJPY.json"]["tf"]["1d"]["basis"] is None
