from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aifx import pipeline as pl
from aifx import rates, research_trade, trade
from aifx.audit import audit, verify
from aifx.data import PAIRS
from aifx.engine import TIMEFRAMES
from aifx.ledger import Ledger
from aifx.pipeline import State
from aifx.timeutil import iso, parse_iso

from .conftest import fake_rates, run_session

UTC = timezone.utc


def _bars(rows, start="2026-09-21 00:00"):
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def test_settle_checks_the_stop_first_and_fills_gaps_at_the_open():
    pair = PAIRS["USDJPY"]
    origin = "2026-09-21T00:00:00Z"
    until = "2026-09-21T10:00:00Z"
    # the second bar reaches both levels: counted as a loss
    both = _bars([[150.0, 150.0, 150.0, 150.0], [150.0, 150.6, 149.4, 150.0], [150, 151, 150, 151]])
    r = trade.settle(1, 150.0, 149.5, 150.5, until, origin, both, 60, pair.pip, 0.5, None)
    assert r["how"] == "sl" and r["exit"] == 149.5 and r["pips"] == pytest.approx(-50.5) and r["R"] == pytest.approx(-1.01)
    # a bar that opens beyond the target fills at its open (a sell's target is below)
    gap = _bars([[150.0, 150.0, 150.0, 150.0], [150.1, 150.2, 150.0, 150.1], [149.2, 149.3, 149.0, 149.1]])
    r = trade.settle(-1, 150.0, 150.5, 149.5, until, origin, gap, 60, pair.pip, 0.0, None)
    assert r["how"] == "tp" and r["exit"] == 149.2
    # neither level before the time limit: closed at the limit
    flat = _bars([[150.0, 150.1, 149.9, 150.0]] * 14)
    r = trade.settle(1, 150.0, 149.5, 150.5, until, origin, flat, 60, pair.pip, 0.0, 2.0)
    assert r["how"] == "time" and r["t"] == until
    assert r["pips"] > 0                               # only the swap: +2 % less the markup
    assert trade.settle(1, 150.0, 149.5, 150.5, until, origin, flat.iloc[:5], 60, pair.pip, 0.0, None) is None


def test_rule_signals_follow_the_rate_difference_and_momentum():
    up = np.linspace(100, 110, 200)
    assert trade.rule_signal("1d", up, 2.5) == 1 and trade.rule_signal("1d", up, -2.5) == -1
    assert trade.rule_signal("1d", up, 1.0) == 0 and trade.rule_signal("1d", up, None) == 0
    assert trade.rule_signal("1h", up, 1.5) == 1          # carry and 120-hour momentum agree
    assert trade.rule_signal("1h", up, -1.5) == 0         # they disagree: no trade
    assert trade.rule_signal("15m", up, 3.0) == 0         # no tested rule


def test_stored_rates_match_the_research_panel():
    at = datetime(2026, 9, 24, 1, tzinfo=UTC)
    item, err = fake_rates(at)
    assert err is None
    assert rates.rate_diff(item, "USD", "JPY", "2026-09-20") == pytest.approx(3.5)
    assert rates.known_rate(item, "JPY", "2026-09-20") == pytest.approx(0.5)
    # a monthly average published this month is not known yet
    only_recent = {"series": {"IRSTCI01JPM156N": [["2026-09-01", 0.5]]}}
    assert rates.known_rate(only_recent, "JPY", "2026-09-20") is None
    assert rates.known_rate(only_recent, "JPY", "2026-11-02") == pytest.approx(0.5)


def test_live_backtest_takes_the_same_trades_as_the_research():
    rng = np.random.default_rng(3)
    n = 900
    idx = pd.date_range("2026-01-05", periods=n, freq="h", tz="UTC")
    c = 150 * np.exp(np.cumsum(rng.normal(0.0002, 0.001, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0008, "low": np.minimum(o, c) * 0.9992, "close": c}, index=idx)
    pair, tf = PAIRS["USDJPY"], TIMEFRAMES["1h"]
    item, _ = fake_rates(datetime(2026, 3, 1, tzinfo=UTC))
    live = trade.backtest(tf, pair, df, item)
    P = {"o": o, "h": df["high"].to_numpy(), "l": df["low"].to_numpy(), "c": c, "pip": pair.pip,
         "atr": trade.atr(df["high"].to_numpy(), df["low"].to_numpy(), c), "diff": np.full(n, 3.5),
         "acc": (np.zeros(n), np.zeros(n)), "start": 150}
    r = trade.RULES["1h"]
    sig = research_trade.signal("carry_mom", {"thr": r["thr"], "L": r["L"]}, P)
    ref = research_trade.simulate(P, sig, r["sl"], r["tp"], r["hold"], trade.COST_PIPS["USDJPY"])
    ends = [iso(t + timedelta(hours=1)) for t in idx]
    assert live and [(t["origin"], t["t"], t["dir"]) for t in live] == [(ends[x[0]], ends[x[1]], x[2]) for x in ref]


def test_a_trade_plan_that_does_not_follow_from_the_data_is_caught(tmp_path, monkeypatch):
    real = trade.plan

    def nudged(tf, pair, bars, origin, p0, item):
        out = real(tf, pair, bars, origin, p0, item)
        if out.get("atr"):
            out["atr"] = round(out["atr"] * 0.8, 8)      # a tighter stop than the rule gives
        return out

    monkeypatch.setattr(trade, "plan", nudged)
    root = tmp_path / "state"
    run_session(root, cycles=4)
    assert verify(root)["ok"]                            # consistent on its face ...
    monkeypatch.setattr(trade, "plan", real)
    rep = audit(root, sample=6)                          # ... but not what the committed data give
    assert rep["results"] and not any(r["ok"] for r in rep["results"])


def test_forged_trade_plans_are_caught(session):
    root, _ = session
    ledger = Ledger(root).load()
    src = next(r for r in reversed(ledger.of_type("prediction")) if r.get("trade", {}).get("atr"))
    at = parse_iso(ledger.records[-1]["at"])
    for n, change in enumerate([{"until": iso(parse_iso(src["trade"]["until"]) + timedelta(hours=3))},
                                {"dir": 1, "rule": "carry", "sl": src["p0"] + 1.0, "tp": src["p0"] + 2.0}]):
        p = {k: v for k, v in src.items() if k not in ("seq", "prev", "hash", "at")}
        p["trade"] = dict(p["trade"], **change)
        ledger.append(p, at + timedelta(minutes=1 + n))
    msgs = " ".join(x["msg"] for x in verify(root)["problems"])
    assert "time limit" in msgs and "wrong side" in msgs


def test_rates_are_fetched_once_a_day_and_used_only_after_they_were_stored(tmp_path):
    calls = []

    def counting(at):
        calls.append(at)
        return fake_rates(at)

    root = tmp_path / "state"
    run_session(root, cycles=4, rates_fetch=counting)
    assert len(calls) == 1
    state = State.open(root)
    fetched = state.rates.load()[0]["fetched_at"]
    for p in state.ledger.of_type("prediction"):
        if p["origin"] <= fetched:
            assert p["trade"]["diff"] is None             # not yet stored at the origin
    assert any(p["trade"]["diff"] is not None for p in state.ledger.of_type("prediction"))


def test_cycle_without_rates_still_forecasts(tmp_path):
    def failing(at):
        return None, "rates: offline"

    reports = run_session(tmp_path / "state", cycles=2, rates_fetch=failing)
    assert reports[-1].verify["ok"] and any("rates" in e for e in reports[-1].errors)
    assert pl.State.open(tmp_path / "state").ledger.of_type("prediction")


def test_api_shows_the_plan_levels_and_rule_record(session):
    from aifx.api import build_api
    root, _ = session
    out = build_api(root)
    blk = out["pair/USDJPY.json"]["tf"]["1h"]
    T = blk["trade"]
    assert T["rule"]["key"] == "carry_mom" and T["cost_pips"] == trade.COST_PIPS["USDJPY"]
    lv = T["plan"]["levels"]
    p0 = T["plan"]["p0"]
    assert lv["buy"]["sl"] < p0 < lv["buy"]["tp"] and lv["sell"]["tp"] < p0 < lv["sell"]["sl"]
    assert T["plan"]["x_until"] and "stats" in T["bt"] and "stats" in T["live"]
    assert out["pair/USDJPY.json"]["tf"]["15m"]["trade"]["rule"] is None
    assert any(p.get("signal") is not None for p in out["meta.json"]["pairs"])
