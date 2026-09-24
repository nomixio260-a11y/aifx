from datetime import timedelta

import numpy as np
import pytest

from aifx import backtest
from aifx.engine import MODEL_KEYS, TIMEFRAMES, bar_end, bars_until, model_paths
from aifx.learning import learn_horizon
from aifx.pipeline import State
from aifx.timeutil import iso, parse_iso

from .conftest import run_session


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setattr(backtest, "WINDOW", {"15m": timedelta(hours=8), "1h": timedelta(days=2), "1d": timedelta(days=15)})
    monkeypatch.setattr(backtest, "RECENT", {"15m": timedelta(hours=3), "1h": timedelta(hours=12), "1d": timedelta(days=5)})


def test_backtest_uses_only_data_up_to_each_origin_and_fills_outcomes(session, small_window):
    root, _ = session
    state = State.open(root)
    now = parse_iso(state.ledger.records[-1]["at"])
    tfs = [TIMEFRAMES["15m"], TIMEFRAMES["1h"], TIMEFRAMES["1d"]]
    rep = backtest.update(state, tfs, ["USDJPY"], now, budget=10_000)
    assert rep["added"] > 0 and rep["rows"]["USDJPY_15m"] > 20
    tf = TIMEFRAMES["15m"]
    cache = state.read_cache("backtest/15m_USDJPY.json")
    bars = state.prices.load("USDJPY", "15m")
    for o_iso, row in list(cache["rows"].items())[:: max(1, len(cache["rows"]) // 5)]:
        hist = bars_until(tf, bars, parse_iso(o_iso)).iloc[-tf.fit_bars:]
        paths, _ = model_paths(np.log(hist["close"].to_numpy()), max(tf.horizons))
        for h, f in row["h"].items():
            assert f["m"] == [round(float(paths[k][int(h) - 1]), 3) for k in MODEL_KEYS]
            assert (f["a"] is None) == (parse_iso(f["t"]) > now)          # outcomes only once the target has passed
    # a second update adds nothing new but keeps the rows; a code change rebuilds them
    assert backtest.update(state, tfs, ["USDJPY"], now, budget=10_000)["added"] == 0
    cache["v"] = "old-version"
    state.write_cache("backtest/15m_USDJPY.json", cache)
    assert backtest.update(state, [tf], ["USDJPY"], now, budget=10_000)["added"] > 0


def test_backtest_budget_is_shared_by_timeframes_and_pairs_newest_first(session, small_window):
    root, _ = session
    state = State.open(root)
    now = parse_iso(state.ledger.records[-1]["at"])
    tfs = list(TIMEFRAMES.values())
    rep = backtest.update(state, tfs, ["USDJPY", "EURUSD"], now, budget=12)
    assert rep["added"] == 12
    assert set(rep["rows"].values()) == {2}
    for tf in tfs:
        bars = bars_until(tf, state.prices.load("EURUSD", tf.key), now)
        newest = [bar_end(tf, ts) for ts in bars.index[-2:]]
        assert sorted(state.read_cache(f"backtest/{tf.key}_EURUSD.json")["rows"]) == [iso(t) for t in newest]
    # a cache written by other code is not reported (the next update rebuilds it)
    stale = state.read_cache("backtest/1h_EURUSD.json")
    stale["v"] = "old-version"
    state.write_cache("backtest/1h_EURUSD.json", stale)
    s = backtest.summary(state, [TIMEFRAMES["1h"]], ["EURUSD"], now, {"EURUSD": 5})
    assert s["tf"] == {} and s["pairs"] == {}


def test_backtest_replays_the_live_learning_rule(session, small_window):
    root, _ = session
    state = State.open(root)
    now = parse_iso(state.ledger.records[-1]["at"])
    tf = TIMEFRAMES["15m"]
    backtest.update(state, [tf], ["USDJPY", "EURUSD"], now, budget=10_000)
    samples = [{"pair": c, "origin": o, "hkey": "1", "p0": row["p0"], **row["h"]["1"]}
               for c in ("USDJPY", "EURUSD") for o, row in state.read_cache(f"backtest/15m_{c}.json")["rows"].items()]
    prior = next(r for r in reversed(state.ledger.records) if r["type"] == "prior" and r["tf"] == "15m")
    rows = backtest._replay(tf, samples, prior)
    for r in rows[-3:]:
        # what the live learner makes of the backtest forecasts scored before this origin
        known = sorted((k for k in rows if k["a"] is not None and k["t"] <= r["origin"]),
                       key=lambda k: (k["t"], k["origin"], k["pair"]))
        assert len(known) > 20
        st = learn_horizon(prior["h"]["1"], [{**{f: k[f] for f in ("m", "a", "s", "k", "g", "c0", "c")}, "x": 0.0}
                                             for k in known], tf.half_life)
        assert (r["k"], r["g"]) == (st.k, st.gain)
        assert r["c"] == pytest.approx(st.gain * float(np.dot(st.weights, r["m"])), abs=1e-12)
        assert r["sigma"] == pytest.approx(r["s"] * st.k)


def test_backtest_summary_reports_calibration_and_overlay(session, small_window):
    root, _ = session
    state = State.open(root)
    now = parse_iso(state.ledger.records[-1]["at"])
    tfs = list(TIMEFRAMES.values())
    backtest.update(state, tfs, ["USDJPY", "EURUSD"], now, budget=10_000)
    s = backtest.summary(state, tfs, ["USDJPY", "EURUSD"], now, {"USDJPY": 3, "EURUSD": 5})
    m = s["tf"]["15m"]["by_h"]["1"]["all"]
    assert m["n"] > 10 and all(0 <= m["cover"][lv] <= 1 for lv in ("50", "80", "95"))
    assert m["cover"]["50"] <= m["cover"]["80"] <= m["cover"]["95"]
    per_pair = s["tf"]["15m"]["by_pair"]
    assert per_pair["USDJPY"]["1"]["all"]["n"] + per_pair["EURUSD"]["1"]["all"]["n"] == m["n"]
    tl = s["tf"]["1h"]["timeline"]["1"]
    assert tl and all(0 <= q[1] <= 1 and q[3] > 0 for q in tl) and [q[0] for q in tl] == sorted(q[0] for q in tl)
    items = s["pairs"]["USDJPY"]["15m"]["1"]
    assert items and all(it["lo95"] <= it["lo80"] <= it["lo50"] <= it["c"] <= it["hi50"] <= it["hi80"] <= it["hi95"]
                         for it in items)


def test_cycle_runs_the_backtest_without_touching_the_ledger(tmp_path, small_window):
    reports = run_session(tmp_path / "state", cycles=2, backtest_budget=300)
    assert reports[-1].verify["ok"] and reports[-1].backtest["added"] > 0
    assert not list((tmp_path / "state").glob("ledger/*backtest*"))
    assert (tmp_path / "state" / "cache" / "backtest").is_dir()
